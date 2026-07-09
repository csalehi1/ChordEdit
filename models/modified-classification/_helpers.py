"""Shared helpers for the metric-predictor model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from settings import CLIP_COL, PSNR_COL, T_TARGET_FUNC


@dataclass(frozen=True)
class CombinedScoreBounds:
    psnr_min: float
    psnr_max: float
    clip_min: float
    clip_max: float


def combined_score_bounds_from_df(
    df: pd.DataFrame,
    psnr_col: str = PSNR_COL,
    clip_col: str = CLIP_COL,
) -> CombinedScoreBounds:
    return CombinedScoreBounds(
        float(df[psnr_col].min()),
        float(df[psnr_col].max()),
        float(df[clip_col].min()),
        float(df[clip_col].max()),
    )


def combined_score_bounds_from_arrays(
    psnr: np.ndarray,
    clip: np.ndarray,
) -> CombinedScoreBounds:
    return CombinedScoreBounds(
        float(np.nanmin(psnr)),
        float(np.nanmax(psnr)),
        float(np.nanmin(clip)),
        float(np.nanmax(clip)),
    )


def _scalarize_with_bounds(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: CombinedScoreBounds,
) -> np.ndarray:
    """Fixed-bounds scalarization matching T_TARGET_FUNC default weights (0.5/0.5)."""
    eps = 1e-8
    psnr_n = (psnr - bounds.psnr_min) / (bounds.psnr_max - bounds.psnr_min + eps)
    clip_n = (clip - bounds.clip_min) / (bounds.clip_max - bounds.clip_min + eps)
    return (psnr_n + clip_n) / 2.0


def target_metric_arrays(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: CombinedScoreBounds | None = None,
) -> np.ndarray:
    """Apply settings.T_TARGET_FUNC to numpy PSNR/CLIP grids."""
    if bounds is not None:
        return _scalarize_with_bounds(psnr, clip, bounds)
    shape = psnr.shape
    df = pd.DataFrame({PSNR_COL: psnr.ravel(), CLIP_COL: clip.ravel()})
    return T_TARGET_FUNC(df).to_numpy().reshape(shape)


def target_metric_torch(
    pred: torch.Tensor,
    bounds: CombinedScoreBounds,
) -> torch.Tensor:
    """Combined score for (N, 2) PSNR/CLIP predictions using fixed train bounds."""
    psnr = pred[:, 0].detach()
    clip = pred[:, 1].detach()
    psnr_n = (psnr - bounds.psnr_min) / (bounds.psnr_max - bounds.psnr_min + 1e-8)
    clip_n = (clip - bounds.clip_min) / (bounds.clip_max - bounds.clip_min + 1e-8)
    return (psnr_n + clip_n) / 2.0


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the token dimension."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def combine_text_embeddings(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair."""
    return torch.cat([a, b, a - b, a * b], dim=-1)


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """Pick torch device for inference.

    gpu: None -> cuda:0 if available else cpu; int -> cuda:N; "cpu" -> cpu.
    """
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def split_data(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test with an 80/10/10 ratio (row-level)."""
    train = df.sample(frac=0.8, random_state=seed)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=0.5, random_state=seed)
    test = remaining.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def split_data_by_sample(
    df: pd.DataFrame,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    sample_col: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sample_id so each edit triple stays wholly in one split."""
    if "labeled" in df.columns:
        df = df[df["labeled"]].copy()
    sample_ids = sorted(df[sample_col].unique())
    n = len(sample_ids)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sample_ids))
    n_train = max(1, round(train_frac * n))
    n_val = max(0, round(val_frac * n))
    if n_train + n_val >= n:
        n_val = max(0, min(n_val, n - n_train - 1))
    train_ids = set(perm[:n_train])
    val_ids = set(perm[n_train : n_train + n_val])
    test_ids = set(perm[n_train + n_val :])
    if not test_ids and n > 1:
        moved = perm[n_train - 1]
        train_ids.remove(moved)
        test_ids.add(moved)
    train_df = df[df[sample_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[sample_col].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[sample_col].isin(test_ids)].reset_index(drop=True)
    return train_df, val_df, test_df
