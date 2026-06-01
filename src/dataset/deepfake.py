import os
from typing import Callable, Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from src.config import Config
from src.utils import logger

from .base import BaseDataModule, BaseDataset, init_augmentations

try:
    from transformers import AutoProcessor
except ImportError as exc:
    raise ImportError("Please install transformers: pip install transformers") from exc

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------------------------
# H5 dataset
# ---------------------------------------------------------------------------


class H5DeepfakeDataset(BaseDataset):
    """Read image frames from an H5 file using keys listed in text split files.

    Expected H5 key format:
        <dataset>/<source>/<video>/<frame>

    Binary labels are inferred from the source segment: sources containing
    "real" map to 0, and all other sources map to 1.
    """

    def __init__(
        self,
        h5_path: str,
        txt_files: list[str],
        preprocess: Optional[Callable] = None,
        augmentations: Optional[Callable] = None,
        shuffle: bool = False,
        binary: bool = True,
        limit_files: Optional[int] = None,
    ):
        if not binary:
            raise NotImplementedError("Only binary labels are currently supported.")

        self.h5_path = h5_path
        self.h5_file = None
        self.vfm_preprocess = preprocess
        self.label2name = {0: "real", 1: "fake"}

        keys = self._load_keys(txt_files, shuffle=shuffle, limit_files=limit_files)
        keys, labels = self._build_labels(keys)

        super().__init__(
            files=keys,
            labels=labels,
            preprocess=None,
            augmentations=augmentations,
            shuffle=False,
        )

        self._open_h5()
        self.source2uid_map = self._build_source2uid(self.files)

        if self.vfm_preprocess is None:
            logger.print_warning("No foundation-model processor was provided.")

    # ------------------------------------------------------------------
    # Split loading and metadata parsing
    # ------------------------------------------------------------------

    def _load_keys(
        self,
        txt_files: list[str],
        shuffle: bool,
        limit_files: Optional[int],
    ) -> list[str]:
        logger.print_info(f"Loading H5 keys from {len(txt_files)} split files for {self.h5_path}")

        keys: list[str] = []
        skipped = 0
        for file_path in sorted(set(txt_files)):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    keys.extend(line.strip().replace("datasets/", "") for line in f if line.strip())
            except FileNotFoundError:
                logger.print_warning(f"Split file not found; skipping: {file_path}")
                skipped += 1
            except OSError as exc:
                logger.print_error(f"Failed to read split file {file_path}: {exc}")
                skipped += 1

        if skipped:
            logger.print_warning(f"Skipped {skipped} unreadable split files.")
        if not keys:
            raise ValueError(f"No valid H5 keys found in split files: {txt_files}")

        keys = sorted(set(keys))
        if limit_files is not None and limit_files > 0:
            if shuffle:
                np.random.RandomState(42).shuffle(keys)
            keys = keys[:limit_files]

        logger.print_info(f"Loaded {len(keys)} unique H5 keys.")
        return keys

    def _build_labels(self, keys: list[str]) -> tuple[list[str], list[int]]:
        valid_keys: list[str] = []
        labels: list[int] = []
        skipped = 0

        for key in tqdm(keys, desc="Parsing labels from H5 keys"):
            parts = key.split("/")
            if len(parts) < 3:
                logger.print_warning(f"Skipping invalid H5 key: {key}")
                skipped += 1
                continue

            source = parts[-3]
            labels.append(0 if "real" in source.lower() else 1)
            valid_keys.append(key)

        if skipped:
            logger.print_warning(f"Skipped {skipped} invalid H5 keys.")
        if not valid_keys:
            raise ValueError("No valid H5 keys remained after label parsing.")

        return valid_keys, labels

    def _build_source2uid(self, keys: list[str]) -> dict[str, int]:
        sources = sorted({key.split("/")[-3] for key in keys if len(key.split("/")) >= 3})
        if not sources:
            return {"unknown": -1}

        real_sources = [source for source in sources if "real" in source.lower()]
        fake_sources = [source for source in sources if "real" not in source.lower()]

        source2uid = {source: 0 for source in real_sources}
        source2uid.update({source: idx for idx, source in enumerate(fake_sources, start=1)})
        logger.print_info(f"Built source-to-id map: {source2uid}")
        return source2uid

    # ------------------------------------------------------------------
    # H5 file lifecycle
    # ------------------------------------------------------------------

    def _open_h5(self) -> None:
        if not os.path.exists(self.h5_path):
            raise FileNotFoundError(f"H5 file not found: {self.h5_path}")

        try:
            self.h5_file = h5py.File(self.h5_path, "r", swmr=True)
            logger.print_info(f"Opened H5 file: {self.h5_path}")
        except OSError as exc:
            self.h5_file = None
            raise OSError(f"Failed to open H5 file {self.h5_path}: {exc}") from exc

    def _ensure_h5_open(self) -> None:
        if self.h5_file is None:
            self._open_h5()

    def close_h5(self) -> None:
        if self.h5_file is not None:
            self.h5_file.close()
            self.h5_file = None
            logger.print_info(f"Closed H5 file: {self.h5_path}")

    def __del__(self):
        try:
            self.close_h5()
        except Exception:
            pass

    def __getstate__(self):
        """Drop the live H5 handle before DataLoader worker serialization."""
        state = self.__dict__.copy()
        state["h5_file"] = None
        return state

    # ------------------------------------------------------------------
    # Sample loading
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Dict:
        self._ensure_h5_open()
        key = self.files[idx]
        label = self.labels[idx]

        try:
            image = self._read_image(key)
            if self.augmentations is not None:
                image = self.augmentations(image)
            image_tensor = self._preprocess_image(image)
            video, identity, source_uid, frame = self._parse_sample_metadata(key)
        except Exception as exc:
            logger.print_error(f"Failed to load sample {key}: {exc}")
            return self._error_placeholder(idx, key, type(exc).__name__)

        return {
            "idx": idx,
            "image": image_tensor,
            "label": label,
            "path": f"{self.h5_path}::{key}",
            "video": video,
            "identity": identity,
            "source": source_uid,
            "frame": frame,
        }

    def _read_image(self, key: str) -> Image.Image:
        image_array = self.h5_file[key][()]
        if image_array.ndim == 2:
            return Image.fromarray(image_array).convert("RGB")
        if image_array.ndim == 3 and image_array.shape[2] == 3:
            return Image.fromarray(image_array).convert("RGB")
        raise ValueError(f"Unsupported image array shape: {image_array.shape}")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        if self.vfm_preprocess is None:
            raise RuntimeError("No foundation-model processor is available.")

        output = self.vfm_preprocess(images=image, return_tensors="pt")
        image_tensor = output["pixel_values"].squeeze(0)
        if image_tensor.dim() != 3:
            raise ValueError(f"Processor returned an invalid tensor shape: {image_tensor.shape}")
        return image_tensor

    def _parse_sample_metadata(self, key: str) -> tuple[str, str, int, str]:
        parts = key.split("/")
        source = parts[-3]
        video_part = parts[-2]
        frame = parts[-1]
        video = video_part.split("_")[0]
        identity = video_part.split("_")[1] if "_" in video_part else video
        source_uid = self.source2uid_map.get(source, -1)
        return video, identity, source_uid, frame

    def _error_placeholder(self, idx: int, key: str, error_type: str) -> Dict:
        size = self._processor_image_size(default=224)
        return {
            "idx": idx,
            "image": torch.zeros(3, size, size),
            "label": -1,
            "path": f"{self.h5_path}::{key}::{error_type}",
            "video": "error",
            "identity": "error",
            "source": -1,
            "frame": "error",
        }

    def _processor_image_size(self, default: int = 224) -> int:
        processor_size = getattr(self.vfm_preprocess, "size", None)
        if isinstance(processor_size, dict):
            for key in ("shortest_edge", "height", "crop_size"):
                if key in processor_size:
                    return int(processor_size[key])
        if isinstance(processor_size, (int, float)):
            return int(processor_size)
        return default

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------

    def get_class_names(self) -> dict[int, str]:
        return self.label2name

    def get_dataset_from_file(self, key: str) -> str:
        parts = key.split("/")
        return parts[-4] if len(parts) >= 4 else "unknown_dataset"

    def get_video_path(self, key: str) -> str:
        return key.rsplit("/", 1)[0] if "/" in key else "unknown_video_path"

    def print_statistics(self) -> None:
        super().print_statistics()
        if not self.files:
            print("No valid H5 keys are available.")
            return

        video_paths = [self.get_video_path(key) for key in self.files]
        dataset_names = [self.get_dataset_from_file(key) for key in self.files]
        frame_count = len(self.files)
        video_count = len(set(video_paths))

        print("\n--- H5 Dataset Statistics ---")
        print(f"H5 path: {self.h5_path}")
        print(f"Frames: {frame_count}")
        print(f"Estimated unique videos: {video_count}")

        df = pd.DataFrame({"dataset": dataset_names, "video": video_paths, "label": self.labels})
        print("\n--- Statistics per dataset ---")
        for dataset_name in sorted(df["dataset"].unique()):
            subset = df[df["dataset"] == dataset_name]
            real_count = int((subset["label"] == 0).sum())
            fake_count = int((subset["label"] == 1).sum())
            print(
                f"  - Dataset: {dataset_name}, "
                f"Videos: {subset['video'].nunique()}, "
                f"Frames: {len(subset)} "
                f"(Real: {real_count}, Fake: {fake_count})"
            )
        print("----------------------------\n")


# ---------------------------------------------------------------------------
# Lightning DataModule
# ---------------------------------------------------------------------------


class H5DeepfakeDataModule(BaseDataModule):
    """LightningDataModule for H5-based deepfake datasets."""

    def __init__(self, config: Config, vfm_model_name: str):
        super().__init__(config)
        self.vfm_model_name = vfm_model_name
        self.processor = AutoProcessor.from_pretrained(vfm_model_name)
        self.image_size = self._infer_image_size(self.processor)

    def _infer_image_size(self, processor, default: int = 224) -> int:
        image_processor = getattr(processor, "image_processor", None)
        feature_extractor = getattr(processor, "feature_extractor", None)
        size_config = getattr(image_processor or feature_extractor, "size", None)

        if isinstance(size_config, dict):
            for key in ("shortest_edge", "height", "crop_size"):
                if key in size_config:
                    return int(size_config[key])
        if isinstance(size_config, (int, float)):
            return int(size_config)
        return default

    def setup(self, stage: Optional[str] = None) -> None:
        train_aug = init_augmentations(self.image_size, train=True)
        eval_aug = init_augmentations(self.image_size, train=False)

        if stage in ("fit", "validate", None):
            if stage in ("fit", None):
                logger.print("\n[blue]Setting up training dataset (H5)...")
                self.train_dataset = H5DeepfakeDataset(
                    h5_path=self.config.trn_h5_path,
                    txt_files=self.config.trn_files,
                    preprocess=self.processor,
                    augmentations=train_aug,
                    shuffle=True,
                    binary=self.config.binary_labels,
                    limit_files=self.config.limit_trn_files,
                )
                self.train_dataset.print_statistics()

            logger.print("\n[blue]Setting up validation dataset (H5)...")
            self.val_dataset = H5DeepfakeDataset(
                h5_path=self.config.val_h5_path,
                txt_files=self.config.val_files,
                preprocess=self.processor,
                augmentations=eval_aug,
                shuffle=False,
                binary=self.config.binary_labels,
                limit_files=self.config.limit_val_files,
            )
            self.val_dataset.print_statistics()

        if stage in ("test", None):
            logger.print("\n[blue]Setting up test dataset (H5)...")
            self.test_dataset = self._build_test_dataset(eval_aug)

    def _build_test_dataset(self, eval_aug):
        if isinstance(self.config.tst_h5_path, dict):
            datasets = []
            for dataset_name, h5_path in sorted(self.config.tst_h5_path.items()):
                split_files = self.config.tst_files.get(dataset_name)
                if not split_files:
                    raise ValueError(f"No test split files configured for dataset: {dataset_name}")

                logger.print(f"\n[blue]Setting up test dataset: {dataset_name}")
                dataset = H5DeepfakeDataset(
                    h5_path=h5_path,
                    txt_files=split_files,
                    preprocess=self.processor,
                    augmentations=eval_aug,
                    shuffle=False,
                    binary=self.config.binary_labels,
                    limit_files=self.config.limit_tst_files,
                )
                dataset.print_statistics()
                datasets.append(dataset)

            if not datasets:
                raise ValueError("No test datasets were configured.")
            return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

        txt_files = self._flatten_test_files(self.config.tst_files)
        dataset = H5DeepfakeDataset(
            h5_path=self.config.tst_h5_path,
            txt_files=txt_files,
            preprocess=self.processor,
            augmentations=eval_aug,
            shuffle=False,
            binary=self.config.binary_labels,
            limit_files=self.config.limit_tst_files,
        )
        dataset.print_statistics()
        return dataset

    def _flatten_test_files(self, tst_files) -> List[str]:
        if isinstance(tst_files, dict):
            files: list[str] = []
            for _, split_files in tst_files.items():
                if not isinstance(split_files, list):
                    raise TypeError("Each test split entry must be a list of text files.")
                files.extend(split_files)
            return sorted(set(files))
        if isinstance(tst_files, list):
            return sorted(set(tst_files))
        raise TypeError("tst_files must be either a list or a dictionary of lists.")

    # ------------------------------------------------------------------
    # Dataloaders
    # ------------------------------------------------------------------

    def train_dataloader(self):
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._build_loader(self.test_dataset, shuffle=False)

    def _build_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.mini_batch_size,
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            persistent_workers=self.config.num_workers > 0,
            prefetch_factor=2 if self.config.num_workers > 0 else None,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        logger.print_info(f"Cleaning up datamodule resources for stage={stage}.")
        for name in ("train_dataset", "val_dataset", "test_dataset"):
            dataset = getattr(self, name, None)
            self._close_dataset(dataset)

    def _close_dataset(self, dataset) -> None:
        if dataset is None:
            return
        if hasattr(dataset, "close_h5"):
            dataset.close_h5()
            return
        if hasattr(dataset, "datasets"):
            for child_dataset in dataset.datasets:
                self._close_dataset(child_dataset)
