from typing import Union

import torch
import torch.nn as nn

from src.config import Config
from src.encoders.clip_encoder import LoRA_CLIPEncoder
from src.modules.beta_vae_aligned import BetaVAEWithAlignment
from src.modules.evidential_head import EvidentialHead


class EvidentialFusionModel(nn.Module):
    """Reliable Multi-View Evidential Learning backbone.

    This module implements the paper's Divide-and-Conquer pipeline:

    1. Multi-view prior extraction: a LoRA-adapted CLIP encoder produces the
       Semantic View feature `f_s`.
    2. Semantic manifold construction: a beta-VAE reconstructs the
       manifold-consistent feature `f_c`.
    3. Geometric View Purification: the residual `f_r = f_s - f_c` is projected
       onto the orthogonal complement of `f_s` to obtain the Artifact View
       feature `f_a`.
    4. Uncertainty-Aware Evidential Learning: each view produces Dirichlet
       evidence, and Dempster-Shafer fusion combines the corresponding
       subjective opinions.
    """

    def __init__(self, config: Config, verbose: bool = False):
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes

        self.primary_encoder = LoRA_CLIPEncoder(
            model_name=config.backbone,
            config=config,
            feature_dim=config.feature_dim,
        )
        self.vfm_output_dim = self.primary_encoder.get_vfm_output_dim()
        self.semantic_manifold_projector = BetaVAEWithAlignment(
            feature_dim=config.feature_dim,
            latent_dim=config.vae_latent_dim,
        )

        self.semantic_head = EvidentialHead(config.feature_dim, config.num_classes)
        self.artifact_head = EvidentialHead(config.feature_dim, config.num_classes)
        self.semantic_norm = nn.LayerNorm(config.feature_dim)
        self.artifact_norm = nn.LayerNorm(config.feature_dim)

        if verbose:
            self.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Dempster-Shafer evidence fusion
    # ------------------------------------------------------------------

    def DS_Combin(self, alpha: Union[dict, list, tuple]) -> torch.Tensor:
        """Fuse two Dirichlet opinions with Dempster-Shafer orthogonal sum."""
        if isinstance(alpha, dict) and len(alpha) == 2:
            alpha_1, alpha_2 = alpha[0], alpha[1]
        elif isinstance(alpha, (list, tuple)) and len(alpha) == 2:
            alpha_1, alpha_2 = alpha[0], alpha[1]
        else:
            raise ValueError("DS_Combin expects exactly two Dirichlet alpha tensors.")

        return self._combine_two_opinions(alpha_1, alpha_2)

    def _combine_two_opinions(self, alpha_1: torch.Tensor, alpha_2: torch.Tensor) -> torch.Tensor:
        alpha_dict = {0: alpha_1, 1: alpha_2}
        belief, strength, evidence, uncertainty = {}, {}, {}, {}

        for view_idx in range(2):
            strength[view_idx] = torch.sum(alpha_dict[view_idx], dim=1, keepdim=True)
            evidence[view_idx] = alpha_dict[view_idx] - 1
            belief[view_idx] = evidence[view_idx] / strength[view_idx].expand(
                evidence[view_idx].shape
            )
            uncertainty[view_idx] = self.num_classes / strength[view_idx]

        belief_outer = torch.bmm(
            belief[0].view(-1, self.num_classes, 1),
            belief[1].view(-1, 1, self.num_classes),
        )
        conflict = torch.sum(belief_outer, dim=(1, 2)) - torch.diagonal(
            belief_outer, dim1=-2, dim2=-1
        ).sum(-1)

        belief_fused = (
            torch.mul(belief[0], belief[1])
            + torch.mul(belief[0], uncertainty[1].expand(belief[0].shape))
            + torch.mul(belief[1], uncertainty[0].expand(belief[0].shape))
        ) / ((1 - conflict).view(-1, 1).expand(belief[0].shape))

        uncertainty_fused = torch.mul(uncertainty[0], uncertainty[1]) / (
            (1 - conflict).view(-1, 1).expand(uncertainty[0].shape)
        )

        strength_fused = self.num_classes / uncertainty_fused
        evidence_fused = torch.mul(belief_fused, strength_fused.expand(belief_fused.shape))
        return evidence_fused + 1

    # ------------------------------------------------------------------
    # Divide-and-Conquer forward pass
    # ------------------------------------------------------------------

    def forward(self, images: torch.Tensor):
        """Return fused evidence and intermediate features used by the losses."""
        encoder_output = self.primary_encoder(images)
        semantic_feature = encoder_output[0] if isinstance(encoder_output, (list, tuple)) else encoder_output

        z, mu, log_var, manifold_feature = self.semantic_manifold_projector(semantic_feature)
        artifact_feature = self._geometric_view_purification(semantic_feature, manifold_feature)

        semantic_feature_norm = self.semantic_norm(semantic_feature)
        artifact_feature_norm = self.artifact_norm(artifact_feature)

        semantic_evidence = self.semantic_head(semantic_feature_norm)
        artifact_evidence = self.artifact_head(artifact_feature_norm)

        semantic_alpha = semantic_evidence + 1
        artifact_alpha = artifact_evidence + 1
        fused_alpha = self.DS_Combin({0: semantic_alpha, 1: artifact_alpha})
        fused_evidence = fused_alpha - 1

        return (
            fused_evidence,
            semantic_evidence,
            artifact_evidence,
            semantic_feature,
            manifold_feature,
            z,
            mu,
            log_var,
        )

    def _geometric_view_purification(
        self,
        semantic_feature: torch.Tensor,
        manifold_feature: torch.Tensor,
    ) -> torch.Tensor:
        """Project the residual onto the semantic orthogonal complement.

        Paper notation:
            f_r = f_s - f_c
            f_parallel = proj_{f_s}(f_r)
            f_a = f_r - f_parallel
        """
        residual_feature = semantic_feature - manifold_feature
        semantic_norm_sq = torch.sum(semantic_feature.pow(2), dim=1, keepdim=True) + 1e-8
        semantic_projection = (
            torch.sum(residual_feature * semantic_feature, dim=1, keepdim=True)
            / semantic_norm_sq
        ) * semantic_feature
        return residual_feature - semantic_projection

    def print_trainable_parameters(self) -> None:
        if hasattr(self.primary_encoder.vision_model, "print_trainable_parameters"):
            self.primary_encoder.vision_model.print_trainable_parameters()
