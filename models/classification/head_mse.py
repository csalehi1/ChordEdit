"""MSE regression head: loss, decoding, and the head module."""

from __future__ import annotations

import torch
import torch.nn as nn


def regression_loss(pred: torch.Tensor, target_val: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """MSE loss between predicted scalar and true bucket value."""
    loss = (pred - target_val) ** 2  # (batch,)
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def decode_regression(pred: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
    """Snap continuous predictions to the nearest bucket index."""
    dists = (pred.unsqueeze(1) - buckets.unsqueeze(0)).abs()  # (N, K)
    return dists.argmin(dim=1)


def bucket_scores_regression(pred: torch.Tensor, buckets: torch.Tensor) -> torch.Tensor:
    """Per-bucket preference scores: negative squared distance to each bucket.

    Not log probabilities, but monotone in the head's own decoding rule and
    additive across the two heads, which is all the joint cell decode needs.
    """
    return -((pred.unsqueeze(1) - buckets.unsqueeze(0)) ** 2)


class RegressionHead(nn.Module):
    """
    Single linear head with sigmoid activation for bounded scalar regression.

    Outputs one value per sample in (0, 1), matching the [0, 1] range of
    the t_start / t_end bucket values.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x)).squeeze(-1)  # (N,)
