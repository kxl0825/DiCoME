from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.config import Config


class LoRA_CLIPEncoder(nn.Module):
    """CLIP vision encoder with optional LoRA adapters and output projection."""

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        config: Optional[Config] = None,
        feature_dim: int = 128,
    ):
        super().__init__()
        self.model_name = model_name
        self.feature_dim = feature_dim

        # ------------------------------------------------------------------
        # Load CLIP assets
        # ------------------------------------------------------------------
        self._preprocess = CLIPProcessor.from_pretrained(model_name)
        clip_model = CLIPModel.from_pretrained(model_name)
        clip_vision_model = clip_model.vision_model
        self.vfm_output_dim = clip_vision_model.config.hidden_size

        # ------------------------------------------------------------------
        # Optional LoRA adaptation
        # ------------------------------------------------------------------
        if self._lora_enabled(config):
            lora_config = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                target_modules=config.peft.lora.target_modules,
                r=config.peft.lora.rank,
                lora_alpha=config.peft.lora.alpha,
                lora_dropout=config.peft.lora.dropout,
                bias=config.peft.lora.bias,
                inference_mode=False,
            )
            self.vision_model = get_peft_model(clip_vision_model, lora_config)
        else:
            self.vision_model = clip_vision_model

        # ------------------------------------------------------------------
        # Feature projection
        # ------------------------------------------------------------------
        if self.vfm_output_dim == self.feature_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(self.vfm_output_dim, self.feature_dim)

    def _lora_enabled(self, config: Optional[Config]) -> bool:
        return (
            config is not None
            and config.peft.enabled
            and config.peft.lora is not None
            and config.peft.lora.enabled
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert a PIL image into a CLIP pixel tensor."""
        return self._preprocess(images=image, return_tensors="pt")["pixel_values"][0]

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return projected features and raw CLIP pooled features."""
        if hasattr(self.vision_model, "base_model"):
            outputs = self.vision_model.base_model(pixel_values=pixel_values)
        else:
            outputs = self.vision_model(pixel_values=pixel_values)

        raw_features = outputs.pooler_output
        projected_features = self.projection(raw_features)
        return projected_features, raw_features

    def get_features_dim(self) -> int:
        return self.feature_dim

    def get_vfm_output_dim(self) -> int:
        return self.vfm_output_dim
