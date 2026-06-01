import os
from typing import List

from lightning.pytorch import loggers as pl_loggers
from rich import print as rprint
from rich.console import Console

from src.config import Config
from .constants import IS_GLOBAL_ZERO

__all__ = [
    "print_error",
    "print_info",
    "print_warning",
    "print_warning_once",
    "print_header",
    "print_success",
    "print",
    "get_loggers",
]

printed_warnings = set()
console = Console()


def print_error(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[red bold]ERROR: [/red bold]{text}")


def print_warning(text="", only_zero_rank=False):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[yellow bold]WARNING: [/yellow bold]{text}")


def print_warning_once(text="", only_zero_rank=False):
    if text in printed_warnings:
        return
    printed_warnings.add(text)
    print_warning(text, only_zero_rank)


def print_info(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(f"[blue bold]INFO: [/blue bold]{text}")


def print(text="", only_zero_rank=True):
    if only_zero_rank and not IS_GLOBAL_ZERO:
        return
    rprint(text)


def print_header(text: str):
    rprint(f"\n[bold cyan]=== {text} ===[/bold cyan]\n")


def print_success(text: str):
    rprint(f"[bold green]SUCCESS: {text}[/bold green]")


def get_loggers(config: Config) -> List[pl_loggers.Logger]:
    """Create TensorBoard and CSV loggers for a training run."""
    run_loggers: List[pl_loggers.Logger] = []

    tb_log_dir = os.path.join(config.run_dir, config.run_name, "tb_logs")
    run_loggers.append(pl_loggers.TensorBoardLogger(save_dir=tb_log_dir, name=""))
    console.print(f"--- [Loggers] TensorBoard logs: {tb_log_dir} ---", style="dim")

    csv_log_dir = os.path.join(config.run_dir, config.run_name, "csv_logs")
    run_loggers.append(pl_loggers.CSVLogger(save_dir=csv_log_dir, name=""))
    console.print(f"--- [Loggers] CSV logs: {csv_log_dir} ---", style="dim")

    return run_loggers