import os
import shutil

import lightning.pytorch as pl
from rich.console import Console

from src.config import Config

console = Console()


def setup_environment(config: Config):
    """Set random seeds and prepare the run directory."""
    pl.seed_everything(config.seed, workers=True)
    console.print(f"--- [Setup] Global seed: {config.seed} ---", style="green")

    run_path = os.path.join(config.run_dir, config.run_name)
    if os.path.exists(run_path):
        if config.throw_exception_if_run_exists:
            raise FileExistsError(f"Run directory already exists: {run_path}")
        if run_path in {"/", "."} or len(run_path) < 5:
            raise ValueError(f"Refusing to remove unsafe run directory: {run_path}")
        console.print(f"--- [Setup] Removing existing run directory: {run_path} ---", style="yellow")
        shutil.rmtree(run_path)

    os.makedirs(os.path.join(run_path, "checkpoints"), exist_ok=True)
    console.print(f"--- [Setup] Run directory ready: {run_path} ---", style="green")
