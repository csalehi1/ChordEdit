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


def bucket_scores_ordinal(logits: torch.Tensor) -> torch.Tensor:
    """Per-bucket log probabilities from the K-1 cumulative threshold logits.

    P(y > k) = sigmoid(logit_k), so P(y = c) = P(y > c-1) - P(y > c) with the
    endpoints pinned at 1 and 0. The differences are not guaranteed positive
    when the thresholds are not monotone, so they are clamped before the log.
    """
    n, k_minus_1 = logits.shape
    ones = logits.new_ones(n, 1)
    zeros = logits.new_zeros(n, 1)
    surv = torch.cat([ones, torch.sigmoid(logits), zeros], dim=-1)  # (N, K+1)
    probs = (surv[:, :-1] - surv[:, 1:]).clamp(min=1e-12)           # (N, K)
    return probs.log() - probs.sum(dim=-1, keepdim=True).log()


class CoralHead(nn.Module):
    """
    Ordinal output head with per-threshold classifiers.

    Each of the K-1 thresholds has its own full weight vector (in_features → 1),
    giving the head enough capacity to learn independent decision boundaries when
    the optimal separating hyperplane differs across thresholds. The ordinal
    structure is enforced entirely by the loss (ordinal_loss), not by weight sharing.

    Biases are initialised in decreasing order so the model starts with a spread
    prediction rather than collapsing to the center class from step one.
    """

    def __init__(self, in_features: int, num_thresholds: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_thresholds)
        if num_thresholds > 0:
            self.fc.bias.data = torch.linspace(2.0, -2.0, num_thresholds)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (N, K-1)
