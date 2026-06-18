"""MAE evaluation metric over bucket indices."""

from __future__ import annotations

import torch


def mae_buckets(pred_idx: torch.Tensor, true_idx: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error over bucket indices.

    Preferred over accuracy because it rewards near-misses. An MAE of 1.0
    means the model is off by one bucket on average.
    """
    return (pred_idx - true_idx).abs().float().mean()
