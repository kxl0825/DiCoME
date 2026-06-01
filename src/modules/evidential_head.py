import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidentialHead(nn.Module):
    """Map a view feature into non-negative evidence for each class."""

    def __init__(self, feature_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc1(x))
        return F.softplus(self.fc2(hidden))
