"""CORAL ordinal head: loss, decoding, and the head module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def ordinal_loss(logits: torch.Tensor, target_idx: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """
    Ordinal binary cross-entropy loss over K-1 cumulative thresholds.

    Each class index c is encoded as a binary vector where the first c
    entries are 1 and the rest are 0 (shown here for K=5 buckets):

        class 0 -> [0, 0, 0, 0]
        class 1 -> [1, 0, 0, 0]
        class 2 -> [1, 1, 0, 0]
        class 3 -> [1, 1, 1, 0]
        class 4 -> [1, 1, 1, 1]

    Because off-by-one errors flip fewer thresholds than large misses, the
    loss naturally penalises large errors more — consistent with an ordinal
    scale.
    """
    k_minus_1 = logits.size(-1)
    if k_minus_1 == 0:
        # Return 0.0 if N_BUCKETS_* for that tensor is 1, no learning possible
        return torch.tensor(0.0, device=logits.device, requires_grad=False)
    thresholds = torch.arange(k_minus_1, device=logits.device).unsqueeze(0)   # (1, K-1)
    targets = (thresholds < target_idx.unsqueeze(1)).float()                   # (batch, K-1)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")  # (batch, K-1)
    if weights is not None:
        loss = loss * weights.unsqueeze(1)
    return loss.mean()


def decode_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert raw head logits to bucket indices by counting exceeded thresholds.

    A threshold is considered exceeded when sigmoid(logit) > 0.5, which is
    equivalent to logit > 0.
    """
    return (logits > 0).long().sum(dim=-1)


class CoralHead(nn.Module):
    """
    Ordinal output head with shared weights across all K-1 thresholds (CORAL).

    Every threshold computes σ(w·x + b_k) with the same weight vector w and
    a per-threshold scalar bias b_k. Because the K-1 outputs differ only in
    their bias, the activation values are a rigid shift of a single dot product:
    exceeding threshold k forces all lower thresholds to be at least as likely,
    which is exactly the rank-consistency guarantee.

    Biases are initialised in decreasing order so the implied class probabilities
    are spread out from the first training step.
    """

    def __init__(self, in_features: int, num_thresholds: int):
        super().__init__()
        self.weight = nn.Linear(in_features, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(2.0, -2.0, num_thresholds))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight(x) + self.bias  # (N, 1) + (K-1,) → (N, K-1)
