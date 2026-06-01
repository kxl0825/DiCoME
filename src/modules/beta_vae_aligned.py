from typing import Tuple

import torch
import torch.nn as nn


class BetaVAEWithAlignment(nn.Module):
    """Semantic manifold projector used in Geometric View Purification.

    The paper describes the CLIP semantic feature as `f_s` and reconstructs a
    manifold-consistent feature `f_c` with a beta-VAE. The residual
    `f_r = f_s - f_c` is later projected onto the orthogonal complement of
    `f_s` to form the purified Artifact View.
    """

    def __init__(self, feature_dim: int, latent_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        hidden_dim = 32

        # Encoder: semantic feature f_s -> latent distribution q(z | f_s).
        self.encoder_mlp = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, self.latent_dim)

        # Decoder: latent sample z -> manifold-consistent feature f_c.
        self.decoder_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.feature_dim),
        )

        self.init_weights()

    def init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        semantic_feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.encoder_mlp(semantic_feature)
        mu = self.fc_mu(hidden)
        log_var = self.fc_log_var(hidden)
        z = self.reparameterize(mu, log_var)
        reconstructed_feature = self.decoder_mlp(z)
        return z, mu, log_var, reconstructed_feature
