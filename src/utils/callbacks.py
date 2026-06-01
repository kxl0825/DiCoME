import os
from typing import List

from lightning.pytorch import callbacks as pl_callbacks
from rich.console import Console

from src.config import Config

console = Console()


def get_callbacks(config: Config, stage: str = "dicome") -> List[pl_callbacks.Callback]:
    """Initialize callbacks for supervised training."""
    checkpoint_dir = os.path.join(config.run_dir, config.run_name, "checkpoints")
    monitor_metric = "val_auroc_video"
    mode = "max"

    callbacks: List[pl_callbacks.Callback] = [
        pl_callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"dicome-best-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
            monitor=monitor_metric,
            mode=mode,
            save_top_k=3,
            save_last=True,
        ),
        pl_callbacks.LearningRateMonitor(logging_interval="step"),
        pl_callbacks.RichProgressBar(),
    ]

    console.print(
        f"--- [Callbacks] ModelCheckpoint enabled: {monitor_metric} ({mode}) ---",
        style="dim",
    )
    return callbacks
