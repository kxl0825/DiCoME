import os
import sys
from pathlib import Path
from typing import Any

import fire
import lightning.pytorch as pl
from rich.console import Console
from rich.traceback import install as rich_traceback_install

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, load_config
from src.dataset.deepfake import H5DeepfakeDataModule
from src.model.dicome_module import DiCoMEModule
from src.utils.callbacks import get_callbacks
from src.utils.logger import get_loggers
from src.utils.setup import setup_environment

rich_traceback_install()
console = Console()


def _coerce_config_value(value: Any) -> Any:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "none":
            return None
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return value


class TrainingPipeline:
    """Command-line entry point for supervised training and testing."""

    def fit(self, config_path: str = "src/config/dicome_default.yaml", **overrides: Any) -> None:
        config = load_config(config_path)
        if overrides:
            clean_overrides = {k: _coerce_config_value(v) for k, v in overrides.items()}
            console.print(f"Applying CLI overrides: {clean_overrides}", style="yellow")
            config = Config(**config.model_copy(update=clean_overrides).model_dump())

        setup_environment(config)
        datamodule = H5DeepfakeDataModule(config=config, vfm_model_name=config.backbone)
        model = DiCoMEModule(config=config, verbose=True)

        trainer = pl.Trainer(
            devices=config.devices,
            max_epochs=config.max_epochs,
            precision=config.precision,
            accumulate_grad_batches=max(1, config.batch_size // config.mini_batch_size),
            fast_dev_run=config.fast_dev_run,
            log_every_n_steps=100,
            overfit_batches=config.overfit_batches,
            limit_train_batches=config.limit_train_batches,
            limit_val_batches=config.limit_val_batches,
            limit_test_batches=config.limit_test_batches,
            deterministic=config.deterministic,
            detect_anomaly=config.detect_anomaly,
            logger=get_loggers(config),
            callbacks=get_callbacks(config),
            default_root_dir=config.run_dir,
        )

        ckpt_path = config.resume_from_checkpoint
        if ckpt_path and not os.path.exists(ckpt_path):
            console.print(
                f"resume_from_checkpoint not found: {ckpt_path}; training from scratch.",
                style="yellow",
            )
            ckpt_path = None

        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
        if not config.fast_dev_run and config.limit_test_batches:
            trainer.test(model, datamodule=datamodule, ckpt_path="best")

    def test(
        self,
        checkpoint_path: str,
        config_path: str = "src/config/dicome_default.yaml",
        **overrides: Any,
    ) -> None:
        config = load_config(config_path)
        if overrides:
            clean_overrides = {k: _coerce_config_value(v) for k, v in overrides.items()}
            config = Config(**config.model_copy(update=clean_overrides).model_dump())

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(checkpoint_path)

        datamodule = H5DeepfakeDataModule(config=config, vfm_model_name=config.backbone)
        model = DiCoMEModule.load_from_checkpoint(checkpoint_path, config=config)
        trainer = pl.Trainer(
            devices=config.devices,
            precision=config.precision,
            logger=get_loggers(config),
            callbacks=get_callbacks(config),
            default_root_dir=config.run_dir,
            log_every_n_steps=1,
        )
        trainer.test(model, datamodule=datamodule)


def main() -> None:
    fire.Fire(TrainingPipeline)


if __name__ == "__main__":
    main()
