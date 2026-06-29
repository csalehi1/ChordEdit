"""Multiclass head with one-hot targets and cost-sensitive cross-entropy."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def index_to_one_hot(target_idx: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Map integer bucket indices to one-hot class vectors."""
    if n_classes == 0:
        raise ValueError("num_classes must be positive")
    return F.one_hot(target_idx.long(), num_classes=n_classes).float()


def ordinal_cost_matrix(
    n_classes: int,
    *,
    power: float = 1.0,
    normalize: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Build a (K, K) cost matrix where entry (i, j) is |i - j|^power.

    Row i is the cost of predicting each class j when the true class is i.
    When normalize=True, costs are scaled to [0, 1] by the maximum pairwise
    distance so the loss magnitude stays comparable across different K.
    """
    if n_classes <= 1:
        return torch.zeros(n_classes, n_classes, device=device)

    idx = torch.arange(n_classes, device=device, dtype=torch.float32)
    costs = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().pow(power)
    if normalize and n_classes > 1:
        costs = costs / costs.max().clamp(min=1e-9)
    return costs


def cost_sensitive_ce_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    cost_matrix: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cost-sensitive cross-entropy via expected misclassification cost.

    For each sample the loss is:

        sum_j  p(j | x) * cost[y, j]

    where y is the true class, p = softmax(logits), and cost[y, j] penalizes
    assigning probability mass to distant buckets. This keeps training
    differentiable while encoding ordinal nearness, nearby wrong classes cost
    less than far ones.

    When cost_matrix is None, an ordinal |i - j| matrix is built from the
    number of logits. Diagonal entries are zero so confident correct
    predictions are not penalized.

    class_weights is a length-K vector indexed by the true label (same effect as
    passing class_weights[y] into ordinal_loss / regression_loss in classify.py).
    sample_weights is an optional length-N vector for extra per-example scaling
    applied after class reweighting.
    """
    n_classes = logits.size(-1)
    if n_classes == 0:
        # Return 0.0 if N_BUCKETS_* for that tensor is *0*, no classes to predict.
        return torch.tensor(0.0, device=logits.device, requires_grad=False)
    if n_classes == 1:
        # Return 0.0 if N_BUCKETS_* for that tensor is *1*, no learning possible.
        return logits.sum() * 0.0

    if cost_matrix is None:
        cost_matrix = ordinal_cost_matrix(n_classes, device=logits.device)
    else:
        cost_matrix = cost_matrix.to(device=logits.device, dtype=logits.dtype)

    probs = F.softmax(logits, dim=-1)                              # (N, K)
    per_class_cost = cost_matrix[target_idx.long()]                # (N, K)
    loss = (probs * per_class_cost).sum(dim=-1)                    # (N,)

    if class_weights is not None:
        loss = loss * class_weights[target_idx.long()]
    if sample_weights is not None:
        loss = loss * sample_weights

    return loss.mean()


def one_hot_ce_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
    sample_weights: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Standard cross-entropy against one-hot targets (no ordinal cost matrix).

    Useful as an ablation against cost_sensitive_ce_loss.

    class_weights is passed to F.cross_entropy as its weight argument (length-K
    vector, per-class reweighting). sample_weights is an optional length-N vector
    multiplied onto per-sample losses after CE.
    """
    n_classes = logits.size(-1)
    if n_classes == 0:
        # Return 0.0 if N_BUCKETS_* for that tensor is *0*, no classes to predict.
        return torch.tensor(0.0, device=logits.device, requires_grad=False)
    if n_classes == 1:
        # Return 0.0 if N_BUCKETS_* for that tensor is *1*, no learning possible.
        return logits.sum() * 0.0

    loss = F.cross_entropy(
        logits,
        target_idx.long(),
        weight=class_weights,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    if sample_weights is not None:
        loss = loss * sample_weights

    return loss.mean()


def decode_classification(logits: torch.Tensor) -> torch.Tensor:
    """Pick the highest-logit bucket index."""
    return logits.argmax(dim=-1)


class ClassificationHead(nn.Module):
    """
    Linear multiclass head producing K logits per target.

    Outputs raw logits (no softmax); decoding and loss apply softmax downstream.
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)  # (N, K)
