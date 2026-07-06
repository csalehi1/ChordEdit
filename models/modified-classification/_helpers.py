"""Shared helpers for the metric-predictor model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


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
