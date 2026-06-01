import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dirichlet regularization
# ---------------------------------------------------------------------------


def _kl_dirichlet_to_uniform(alpha: torch.Tensor, num_classes: int) -> torch.Tensor:
    """KL divergence from Dirichlet(alpha) to a uniform Dirichlet prior."""
    beta = torch.ones((1, num_classes), device=alpha.device)

    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    sum_beta = torch.sum(beta, dim=1, keepdim=True)

    log_b_alpha = torch.lgamma(sum_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    log_b_beta = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(sum_beta)

    digamma_sum_alpha = torch.digamma(sum_alpha)
    digamma_alpha = torch.digamma(alpha)

    return (
        torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim=1, keepdim=True)
        + log_b_alpha
        + log_b_beta
    )


# ---------------------------------------------------------------------------
# Evidential supervised objective
# ---------------------------------------------------------------------------


def evidential_loss_dicome(
    evidence: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    current_epoch: int,
    total_dicome_epochs: int,
) -> torch.Tensor:
    """Compute the supervised evidential classification loss.

    The loss combines a digamma classification term with an annealed KL
    regularizer that discourages unsupported evidence for non-target classes.
    """
    alpha = evidence + 1
    alpha_sum = torch.sum(alpha, dim=1, keepdim=True)
    labels_one_hot = F.one_hot(labels, num_classes=num_classes).float()

    classification_loss = torch.sum(
        labels_one_hot * (torch.digamma(alpha_sum) - torch.digamma(alpha)),
        dim=1,
        keepdim=True,
    )

    if total_dicome_epochs <= 0:
        annealing_coef = torch.tensor(1.0, device=evidence.device)
    else:
        annealing_coef = torch.tensor(
            min(1.0, current_epoch / total_dicome_epochs),
            device=evidence.device,
        )

    non_target_alpha = (alpha - 1) * (1 - labels_one_hot) + 1
    regularization_loss = annealing_coef * _kl_dirichlet_to_uniform(
        non_target_alpha,
        num_classes,
    )

    return torch.mean(classification_loss + regularization_loss)
