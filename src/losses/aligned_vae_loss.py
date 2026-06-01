import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Feature alignment losses
# ---------------------------------------------------------------------------


def cosine_similarity_loss(a: torch.Tensor, b: torch.Tensor, dim: int = 1, eps: float = 1e-8):

    a_norm = F.normalize(a, p=2, dim=dim, eps=eps)
    b_norm = F.normalize(b, p=2, dim=dim, eps=eps)
    cosine_sim = torch.sum(a_norm * b_norm, dim=dim)
    return 1.0 - cosine_sim


# ---------------------------------------------------------------------------
# Aligned VAE objective
# ---------------------------------------------------------------------------


def aligned_vae_loss_func(
    z_original: torch.Tensor,
    z_reconstructed: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta_kld: float,
    lambda_align: float,
) -> torch.Tensor:
    """Compute the aligned VAE loss.

    The objective combines semantic alignment and latent KL regularization. The
    original feature is detached for the alignment term so this auxiliary loss
    does not directly update the primary encoder through that path.
    """
    align_loss = cosine_similarity_loss(z_original.detach(), z_reconstructed).mean()
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    return lambda_align * align_loss + beta_kld * kld_loss
