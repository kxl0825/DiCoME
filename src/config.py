import os
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, ValidationError

Scheduler = Literal["cosine"]
Precision = Literal[
    16,
    32,
    64,
    "16",
    "16-true",
    "16-mixed",
    "bf16-true",
    "bf16-mixed",
    "32",
    "32-true",
    "64",
    "64-true",
]


class Head(str, Enum):
    Linear = "linear"
    LinearNorm = "LinearNorm"


class Backbone(str, Enum):
    CLIP_B_16 = "openai/clip-vit-base-patch16"
    CLIP_B_32 = "openai/clip-vit-base-patch32"
    CLIP_L_14 = "openai/clip-vit-large-patch14"
    CLIP_L_14_336 = "openai/clip-vit-large-patch14-336"
    BEIT_V2 = "beitv2_base_patch16_224.in1k_ft_in22k"
    DINO_V2 = "facebook/dinov2-base"


class LoraConfig(BaseModel):
    enabled: bool = True
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"


class PeftConfig(BaseModel):
    enabled: bool = True
    lora: Optional[LoraConfig] = Field(default_factory=LoraConfig)


class Config(BaseModel, validate_assignment=True):
    run_name: str = "dicome_example"
    run_dir: str = "runs/dicome"
    seed: int = 42
    throw_exception_if_run_exists: bool = False
    num_classes: int = 2
    binary_labels: bool = True

    backbone: str = Backbone.CLIP_L_14
    feature_dim: int = 64
    peft: PeftConfig = Field(default_factory=PeftConfig)

    vae_enabled: bool = True
    vae_latent_dim: int = 32
    beta_kld: float = 4.0
    lambda_align: float = 1.0
    lambda_vae: float = 0.1

    max_epochs: int = 100
    dicome_epochs: int = 100
    dicome_learning_rate: float = 1e-4
    lr_scheduler: Optional[Scheduler] = "cosine"
    min_lr: float = 1e-6
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    trn_h5_path: str = "data/h5/train.h5"
    val_h5_path: str = "data/h5/val.h5"
    tst_h5_path: Union[str, Dict[str, str]] = "data/h5/test.h5"
    trn_files: List[str] = Field(
        default_factory=lambda: [
            "data/splits/train/DF.txt",
            "data/splits/train/F2F.txt",
            "data/splits/train/FS.txt",
            "data/splits/train/NT.txt",
            "data/splits/train/real.txt",
        ]
    )
    val_files: List[str] = Field(
        default_factory=lambda: [
            "data/splits/val/DF.txt",
            "data/splits/val/F2F.txt",
            "data/splits/val/FS.txt",
            "data/splits/val/NT.txt",
            "data/splits/val/real.txt",
        ]
    )
    tst_files: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "test": [
                "data/splits/test/DF.txt",
                "data/splits/test/F2F.txt",
                "data/splits/test/FS.txt",
                "data/splits/test/NT.txt",
                "data/splits/test/real.txt",
            ]
        }
    )
    limit_trn_files: Optional[int] = None
    limit_val_files: Optional[int] = None
    limit_tst_files: Optional[int] = None

    batch_size: int = 128
    mini_batch_size: int = 128
    num_workers: int = 2
    devices: Union[List[int], str, int] = "auto"
    precision: Precision = "bf16-mixed"
    fast_dev_run: Union[int, bool] = False
    overfit_batches: Union[int, float] = 0.0
    limit_train_batches: Optional[Union[int, float]] = 1.0
    limit_val_batches: Optional[Union[int, float]] = 1.0
    limit_test_batches: Optional[Union[int, float]] = 1.0
    deterministic: Optional[bool] = None
    detect_anomaly: bool = False
    checkpoint_for_testing: str = "best"
    resume_from_checkpoint: Optional[str] = None


def load_config(path: str) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f) or {}
    try:
        return Config(**config_dict)
    except ValidationError:
        raise


def save_config(config: Config, path: str) -> None:
    config_dir = os.path.dirname(path)
    if config_dir:
        os.makedirs(config_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    save_config(Config(), "src/config/dicome_default.yaml")
