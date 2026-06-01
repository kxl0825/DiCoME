from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import CatMetric


class OutputsForMetrics(nn.Module):
    """Collect tensors across Lightning steps before metric computation."""

    def __init__(self):
        super().__init__()
        self.probs = CatMetric()
        self.labels = CatMetric()
        self.idx = CatMetric()
        self.uncertainty = CatMetric()
        self.evidence = CatMetric()

    def reset(self):
        self.probs.reset()
        self.labels.reset()
        self.idx.reset()
        self.uncertainty.reset()
        self.evidence.reset()

    def compute(self) -> dict:
        return {
            "probs": self.probs.compute().cpu(),
            "labels": self.labels.compute().cpu().int(),
            "idx": self.idx.compute().cpu().int(),
            "uncertainty": self.uncertainty.compute().cpu(),
            "evidence": self.evidence.compute().cpu(),
        }


@dataclass
class Batch:
    """Typed wrapper for dataloader dictionaries."""

    images: None | torch.Tensor
    labels: None | torch.Tensor
    identity: None | torch.Tensor
    source: None | torch.Tensor
    idx: None | torch.Tensor
    paths: None | list[str]

    def __getitem__(self, key):
        return getattr(self, key)

    @staticmethod
    def from_dict(batch: dict):
        return Batch(
            images=batch.get("image"),
            labels=batch.get("label"),
            identity=batch.get("identity"),
            source=batch.get("source"),
            idx=batch.get("idx"),
            paths=batch.get("path"),
        )


def compute_across_videos(files: list, probs: np.ndarray, labels: np.ndarray):
    """Average frame-level probabilities into video-level predictions."""
    videos = [f.replace("\\", "/").split("/")[-2] for f in files]
    video2idx = {video: [] for video in set(videos)}
    for idx, video in enumerate(videos):
        video2idx[video].append(idx)

    video_probs = []
    video_labels = []
    for indices in video2idx.values():
        video_probs.append(np.mean(probs[indices], axis=0))
        video_labels.append(int(labels[indices[0]]))

    return np.array(video_probs), np.array(video_labels)
