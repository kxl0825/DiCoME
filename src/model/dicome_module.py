import os
from typing import Dict, List, Literal, Optional, Tuple

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
import numpy as np
import pandas as pd
import sklearn.metrics as M
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import torch.optim as optim
from torch.utils.data import Dataset

from src.config import Config
from src.losses.aligned_vae_loss import aligned_vae_loss_func
from src.losses.evidential_loss import evidential_loss_dicome
from src.model.core_model import EvidentialFusionModel
from src.utils.metrics import Batch, OutputsForMetrics, compute_across_videos

try:
    from src.utils import logger
except ImportError:
    class PrintLogger:
        def print_info(self, msg, *_, **__): print(f"INFO: {msg}")
        def print_success(self, msg, *_, **__): print(f"SUCCESS: {msg}")
        def print_warning(self, msg, *_, **__): print(f"WARNING: {msg}")
        def print_error(self, msg, *_, **__): print(f"ERROR: {msg}")

    logger = PrintLogger()



class DiCoMEModule(pl.LightningModule):
    """Lightning trainer for Reliable Multi-View Evidential Learning.

    The module follows the paper's Conquer phase. It supervises fused
    Dirichlet evidence with the Uncertainty-Aware Evidential Loss and keeps the
    Semantic Manifold Projector aligned through the beta-VAE loss.
    """

    def __init__(self, config: Config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config.model_dump())

        self.model = EvidentialFusionModel(config, verbose=verbose)
        self.evidential_loss = evidential_loss_dicome
        self.vae_loss = aligned_vae_loss_func

        self.train_step_outputs = OutputsForMetrics()
        self.val_step_outputs = OutputsForMetrics()
        self.test_step_outputs = OutputsForMetrics()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Map legacy checkpoint keys to the public DiCoME module names."""
        state_dict = checkpoint.get("state_dict")
        if not state_dict:
            return

        key_mapping = {
            "model.beta_vae.": "model.semantic_manifold_projector.",
            "model.evidential_head_1.": "model.semantic_head.",
            "model.evidential_head_2.": "model.artifact_head.",
            "model.view1_norm.": "model.semantic_norm.",
            "model.view2_norm.": "model.artifact_norm.",
        }

        renamed_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix):]
                    break
            renamed_state_dict[new_key] = value

        checkpoint["state_dict"] = renamed_state_dict

    def forward(self, images: torch.Tensor):
        return self.model(images)

    def get_batch(self, batch: dict) -> Batch:
        """Convert a dataloader dictionary into the typed Batch container."""
        return Batch.from_dict(batch)

    # ------------------------------------------------------------------
    # Shared evidential helpers
    # ------------------------------------------------------------------

    def _predict_from_evidence(self, evidence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return Dirichlet expected probabilities and uncertainty K / S."""
        alpha = evidence + 1
        strength = torch.sum(alpha, dim=1, keepdim=True).clamp_min(1e-6)
        expected_probs = alpha / strength
        uncertainty = (self.config.num_classes / strength).squeeze(-1)
        return expected_probs, uncertainty

    def _update_outputs(
        self,
        outputs: OutputsForMetrics,
        labels: torch.Tensor,
        probs: torch.Tensor,
        idx: torch.Tensor,
        uncertainty: torch.Tensor,
        evidence: torch.Tensor,
    ) -> None:
        outputs.labels.update(labels.detach())
        outputs.probs.update(probs.detach())
        outputs.idx.update(idx.detach())
        outputs.uncertainty.update(uncertainty.detach())
        outputs.evidence.update(evidence.detach())

    def _compute_loss(
        self,
        fused_evidence: torch.Tensor,
        labels: torch.Tensor,
        semantic_features: torch.Tensor,
        reconstructed_features: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Combine evidential supervision with Semantic Manifold Loss."""
        loss_evidential = self.evidential_loss(
            evidence=fused_evidence,
            labels=labels,
            num_classes=self.config.num_classes,
            current_epoch=self.current_epoch,
            total_dicome_epochs=self.config.dicome_epochs,
        )
        loss_vae = self.vae_loss(
            z_original=semantic_features,
            z_reconstructed=reconstructed_features,
            mu=mu,
            log_var=log_var,
            beta_kld=self.config.beta_kld,
            lambda_align=self.config.lambda_align,
        )
        total_loss = loss_evidential + self.config.lambda_vae * loss_vae
        return total_loss, loss_evidential, loss_vae

    # ------------------------------------------------------------------
    # Training, validation, and testing
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        fused_evidence, _, _, features, reconstructed_features, _, mu, log_var = self(batch.images)

        loss, loss_evidential, loss_vae = self._compute_loss(
            fused_evidence=fused_evidence,
            labels=batch.labels,
            semantic_features=features,
            reconstructed_features=reconstructed_features,
            mu=mu,
            log_var=log_var,
        )

        batch_size = batch.images.size(0)
        self.log("dicome/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("dicome/loss_evidential", loss_evidential, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        self.log("dicome/loss_vae", loss_vae, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        with torch.no_grad():
            expected_probs, uncertainty = self._predict_from_evidence(fused_evidence)
        self._update_outputs(self.train_step_outputs, batch.labels, expected_probs, batch.idx, uncertainty, fused_evidence)
        return loss

    def on_train_epoch_end(self):
        if self.logger is None or self._get_log_dir() is None:
            return

        dataset = getattr(self.trainer.datamodule, "train_dataset", None)
        if dataset is not None:
            self.log_all_metrics(self.train_step_outputs, "train", dataset)
        self.train_step_outputs.reset()

    def validation_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        fused_evidence, _, _, features, reconstructed_features, _, mu, log_var = self(batch.images)

        _, loss_evidential, _ = self._compute_loss(
            fused_evidence=fused_evidence,
            labels=batch.labels,
            semantic_features=features,
            reconstructed_features=reconstructed_features,
            mu=mu,
            log_var=log_var,
        )
        self.log("val/loss_evidential", loss_evidential, on_step=False, on_epoch=True, logger=True, batch_size=batch.images.size(0))

        with torch.no_grad():
            expected_probs, uncertainty = self._predict_from_evidence(fused_evidence)
        self._update_outputs(self.val_step_outputs, batch.labels, expected_probs, batch.idx, uncertainty, fused_evidence)

    def on_validation_epoch_end(self):
        dataset = getattr(self.trainer.datamodule, "val_dataset", None)
        if self.logger is None or self._get_log_dir() is None or dataset is None:
            self.val_step_outputs.reset()
            return

        outputs = self._materialize_outputs(self.val_step_outputs, "val")
        if outputs is None:
            return

        labels, probs, idx, uncertainty, evidence = outputs
        files = self._files_from_indices(dataset, idx, "val")
        if files is None:
            self.val_step_outputs.reset()
            return

        self._save_frame_predictions("val", files, labels, probs)
        self._save_video_level_report("val", files, probs, labels, epoch=self.current_epoch)
        self.log_all_metrics(self.val_step_outputs, "val", dataset)
        self.val_step_outputs.reset()

    def test_step(self, batch, batch_idx):
        batch = self.get_batch(batch)
        fused_evidence, _, _, _, _, _, _, _ = self(batch.images)

        with torch.no_grad():
            expected_probs, uncertainty = self._predict_from_evidence(fused_evidence)
        self._update_outputs(self.test_step_outputs, batch.labels, expected_probs, batch.idx, uncertainty, fused_evidence)

    def on_test_epoch_end(self):
        dataset = getattr(self.trainer.datamodule, "test_dataset", None)
        if self.logger is None or self._get_log_dir() is None or dataset is None:
            self.test_step_outputs.reset()
            return

        outputs = self._materialize_outputs(self.test_step_outputs, "test")
        if outputs is None:
            return

        labels, probs, idx, uncertainty, evidence = outputs
        files = self._files_from_indices(dataset, idx, "test")
        if files is None:
            self.test_step_outputs.reset()
            return

        self._save_frame_predictions("test", files, labels, probs)
        self._save_video_level_report("test", files, probs, labels)
        self.log_all_metrics(self.test_step_outputs, "test", dataset)
        self.test_step_outputs.reset()

    # ------------------------------------------------------------------
    # Prediction export
    # ------------------------------------------------------------------

    def _materialize_outputs(
        self,
        outputs_for_metrics: OutputsForMetrics,
        stage: str,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        try:
            outputs = outputs_for_metrics.compute()
            labels = outputs["labels"].numpy()
            probs = outputs["probs"].numpy()
            idx = outputs["idx"].numpy()
            uncertainty = outputs["uncertainty"].numpy()
            evidence = outputs["evidence"].numpy()
        except Exception as exc:
            logger.print_error(f"[{stage}] Failed to collect evidential outputs: {exc}")
            outputs_for_metrics.reset()
            return None

        if len(labels) == 0:
            logger.print_warning(f"[{stage}] No predictions were collected for metric computation.")
            outputs_for_metrics.reset()
            return None

        return labels, probs, idx, uncertainty, evidence

    def _files_from_indices(self, dataset: Dataset, idx: np.ndarray, stage: str) -> Optional[List[str]]:
        files = getattr(dataset, "files", None)
        if files is None:
            logger.print_error(f"[{stage}] Dataset does not expose a files attribute.")
            return None
        if len(idx) == 0 or int(idx.min()) < 0 or int(idx.max()) >= len(files):
            logger.print_error(f"[{stage}] Prediction indices are outside the dataset file list.")
            return None
        return [files[int(i)] for i in idx]

    def _save_frame_predictions(
        self,
        stage: Literal["val", "test"],
        files: List[str],
        labels: np.ndarray,
        probs: np.ndarray,
    ) -> None:
        if probs.ndim != 2:
            logger.print_error(f"[{stage}] Expected a 2D probability array, got shape {probs.shape}.")
            return

        prob_columns = {f"prob_class_{i}": np.round(probs[:, i], 4) for i in range(probs.shape[1])}
        table = pd.DataFrame({
            "files": files,
            "labels": labels,
            "prediction": probs.argmax(axis=1),
            **prob_columns,
        })

        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"{stage}_frame_predictions.csv")
        table.to_csv(save_path, index=False, float_format="%.4f")
        logger.print_info(f"[{stage}] Saved frame-level evidential predictions to {save_path}.")

    def _save_video_level_report(
        self,
        stage: Literal["val", "test"],
        files: List[str],
        probs: np.ndarray,
        labels: np.ndarray,
        epoch: Optional[int] = None,
    ) -> None:
        """Aggregate frame-level probabilities into video-level predictions."""
        if probs.shape[1] != self.config.num_classes:
            logger.print_error(
                f"[{stage}] Probability shape {probs.shape} does not match num_classes={self.config.num_classes}."
            )
            return
        video_ids = [f.replace("\\", "/").split("/")[-2] for f in files]
        prob_cols = [f"prob_{i}" for i in range(self.config.num_classes)]

        frame_df = pd.concat(
            [
                pd.DataFrame({"video_id": video_ids, "label": labels}),
                pd.DataFrame(probs, columns=prob_cols),
            ],
            axis=1,
        )

        agg_funcs: Dict[str, str] = {
            "label": "first",
            **{col: "mean" for col in prob_cols},
        }
        video_df = frame_df.groupby("video_id").agg(agg_funcs).reset_index()


        video_df = video_df.rename(columns={col: f"prob_class_{i}" for i, col in enumerate(prob_cols)})
        prob_cols_renamed = [f"prob_class_{i}" for i in range(self.config.num_classes)]
        video_df["prediction"] = video_df[prob_cols_renamed].values.argmax(axis=1)
        video_df = video_df[["video_id", "label", "prediction", *prob_cols_renamed]]

        log_dir = self._get_log_dir()
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, f"{stage}_video_predictions.csv")
        video_df.to_csv(save_path, index=False, float_format="%.6f")
        logger.print_info(f"[{stage}] Saved video-level evidential predictions to {save_path}.")

    def _get_log_dir(self) -> str:
        """Resolve the CSV log directory used for prediction exports."""
        csv_logger = None
        trainer_loggers = getattr(self.trainer, "loggers", None)
        if trainer_loggers is not None:
            for logger_instance in trainer_loggers:
                if isinstance(logger_instance, pl_loggers.CSVLogger):
                    csv_logger = logger_instance
                    break
        elif isinstance(self.logger, pl_loggers.CSVLogger):
            csv_logger = self.logger

        if csv_logger and csv_logger.log_dir:
            log_dir = csv_logger.log_dir
        else:
            base_dir = os.path.join(self.config.run_dir, self.config.run_name, "csv_logs")
            current_logger = csv_logger if csv_logger else self.logger
            version = getattr(current_logger, "version", None)
            version_str = f"version_{version}" if version is not None else "version_default"
            log_dir = os.path.join(base_dir, version_str)

        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    # ------------------------------------------------------------------
    # Metric logging
    # ------------------------------------------------------------------

    def log_all_metrics(
        self,
        outputs_for_metrics: OutputsForMetrics,
        stage: Literal["train", "test", "val"],
        dataset: Dataset,
    ) -> None:
        outputs = self._materialize_outputs(outputs_for_metrics, stage)
        if outputs is None:
            return

        labels, probs, idx, _, _ = outputs
        files = self._files_from_indices(dataset, idx, stage)
        self.log_metrics(probs, labels, stage, stage, "frame", dataset)

        if files is not None:
            video_probs, video_labels = compute_across_videos(files, probs, labels)
            self.log_metrics(video_probs, video_labels, stage, stage, "video", dataset)
            self._log_dataset_specific_metrics(stage, dataset, files, probs, labels)

    def _log_dataset_specific_metrics(
        self,
        stage: Literal["train", "test", "val"],
        dataset: Dataset,
        files: List[str],
        probs: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        dataset2files = getattr(dataset, "dataset2files", None)
        if not dataset2files:
            return

        file2index = {path: idx for idx, path in enumerate(files)}
        for dataset_name, dataset_files in dataset2files.items():
            selected_files = np.intersect1d(files, dataset_files)
            if len(selected_files) == 0:
                continue

            selected_indices = [file2index[path] for path in selected_files]
            dataset_probs = probs[selected_indices]
            dataset_labels = labels[selected_indices]
            dataset_file_list = [files[i] for i in selected_indices]
            prefix = f"{stage}_dataset_{dataset_name}"

            self.log_metrics(dataset_probs, dataset_labels, stage, prefix, "frame", dataset)
            video_probs, video_labels = compute_across_videos(dataset_file_list, dataset_probs, dataset_labels)
            self.log_metrics(video_probs, video_labels, stage, prefix, "video", dataset)

    def log_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        stage: Literal["train", "test", "val"],
        prefix: str,
        level: Literal["frame", "video"],
        dataset: Dataset,
    ) -> None:
        """Log frame-level or video-level detection metrics."""
        if probs is None or labels is None or len(labels) == 0 or probs.shape[0] != len(labels):
            logger.print_warning(f"[{prefix}/{level}] Metric inputs are empty or inconsistent.")
            return

        probs = self._ensure_probability_matrix(probs, prefix, level)
        if probs is None:
            return

        scores = probs[:, 1]
        preds = probs.argmax(axis=1)
        batch_size = len(labels)

        if len(np.unique(labels)) < 2:
            auroc, mean_ap, eer = 0.0, 0.0, 0.0
            logger.print_warning(f"[{prefix}/{level}] Only one class is present; AUROC, mAP, and EER are set to 0.")
        else:
            auroc = M.roc_auc_score(labels, scores)
            mean_ap = M.average_precision_score(labels, scores)
            eer = self._calculate_eer(labels, scores, prefix, level)

        acc = M.accuracy_score(labels, preds)
        balanced_acc = M.balanced_accuracy_score(labels, preds)
        f1 = M.f1_score(
            labels,
            preds,
            average="binary" if self.config.num_classes == 2 else "macro",
            zero_division=0,
        )

        self.log(f"{prefix}_auroc_{level}", auroc, batch_size=batch_size)
        self.log(f"{prefix}_mAP_{level}", mean_ap, batch_size=batch_size)
        self.log(f"{prefix}_acc_{level}", acc, batch_size=batch_size)
        self.log(f"{prefix}_balanced_acc_{level}", balanced_acc, batch_size=batch_size)
        self.log(f"{prefix}_f1_score_{level}", f1, batch_size=batch_size)
        self.log(f"{prefix}_eer_{level}", eer, batch_size=batch_size)

    def _ensure_probability_matrix(
        self,
        probs: np.ndarray,
        prefix: str,
        level: str,
    ) -> Optional[np.ndarray]:
        if probs.ndim == 1 and self.config.num_classes == 2:
            return np.stack([1.0 - probs, probs], axis=-1)
        if probs.ndim != 2 or probs.shape[1] != self.config.num_classes:
            logger.print_error(
                f"[{prefix}/{level}] Probability shape {probs.shape} does not match "
                f"num_classes={self.config.num_classes}."
            )
            return None
        return probs

    def _calculate_eer(self, labels: np.ndarray, scores: np.ndarray, prefix: str, level: str) -> float:
        """Compute equal error rate from binary labels and positive-class scores."""
        try:
            fpr, tpr, _ = M.roc_curve(labels, scores, pos_label=1)
            return brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        except Exception as exc:
            logger.print_warning(f"[{prefix}/{level}] Failed to compute EER: {exc}")
            return 0.5

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "bn" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        optimizer = optim.AdamW(
            [
                {"params": decay_params, "weight_decay": self.config.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.config.dicome_learning_rate,
            betas=self.config.betas,
        )
        optimizer_config = {"optimizer": optimizer}

        if self.config.lr_scheduler == "cosine":
            total_steps = self._estimate_total_steps()
            if total_steps > 0:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=total_steps,
                    eta_min=self.config.min_lr,
                )
                optimizer_config["lr_scheduler"] = {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            else:
                logger.print_warning("Cosine scheduler is disabled because total training steps could not be estimated.")

        return optimizer_config

    def _estimate_total_steps(self) -> int:
        estimated_batches = getattr(self.trainer, "estimated_stepping_batches", 0)
        if estimated_batches and estimated_batches > 0:
            return int(estimated_batches)

        try:
            train_loader = self.trainer.datamodule.train_dataloader()
            steps_per_epoch = len(train_loader)
            max_epochs = self.trainer.max_epochs or self.config.max_epochs
            accumulate = self.trainer.accumulate_grad_batches or 1
            return int(max_epochs * steps_per_epoch // accumulate)
        except Exception as exc:
            logger.print_warning(f"Failed to estimate scheduler steps from the train dataloader: {exc}")
            return 0
