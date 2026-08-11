# scores.py

"""
Scalarization of the score vector s = (s_1, s_2) into a single objective phi,
following the paper's notation: s_1 = PSNR-Unedited, s_2 = CLIP-Edited.

Because metrics live on different scales, all scoring functions take
per-sample-normalized score deltas (see calc_normalized_deltas)

    Delta_i = (s_i - s_i^0) / (max_T s_i - min_T s_i)

where s_i^0 is the sample's default edit and min/max are over the same
sample's candidate edits on the timestep grid T.

Paper correspondence:
    naive_score -> phi_nai(Delta) = sum_i w_i * Delta_i
    cara_score  -> phi_CARA(Delta) = (1/alpha) * sum_i w_i * (1 - exp(-alpha * Delta_i))
    linex_score -> phi_linex(Delta) = (phi_nai(Delta) + phi_CARA(Delta)) / 2

with weights w_i and alpha the Arrow-Pratt coefficient of the underlying
utility.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch


def naive_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
) -> torch.Tensor:
    """
    Naive Score. Weighted sum of the metric deltas. Simple and interpretable,
    with each weight independently scaling its metric's importance. However, it
    is indifferent to balance: it cannot distinguish a candidate that improves
    both metrics moderately from one that maximizes one metric while tanking
    the other, so long as the weighted sums match.

        phi_nai(Delta) = sum_i w_i * Delta_i

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/50byxqmq3q
    """
    if weights is None:
        return deltas.sum(dim=-1)
    return (deltas * weights).sum(dim=-1)


def cara_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    CARA Score. Sum of deltas passed through the exponential (CARA) utility
    u(x) = (1 - e^{-alpha x}) / alpha, normalized so u(0) = 0 and u'(0) = 1.
    Strict concavity biases toward balance: regressions incur an exponentially
    growing penalty, so a severe regression cannot be offset by gains
    elsewhere. The cost is that gains saturate at w_i / alpha per metric, so
    the score cannot distinguish among candidates that improve a metric
    strongly versus very strongly. Recovers the Naive Score as alpha -> 0+.

        phi_CARA(Delta) = (1/alpha) * sum_i w_i * (1 - exp(-alpha * Delta_i))

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)
    alpha: float > 0

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/wbahputeel
    """
    # u(x) = (1 - e^{-alpha x}) / alpha
    # See https://en.wikipedia.org/wiki/Exponential_utility.
    u = -torch.expm1(-alpha * deltas) / alpha

    if weights is None:
        return u.sum(dim=-1)
    return (u * weights).sum(dim=-1)


def linex_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    LINEX Score. Average of the Naive and CARA Scores, equivalent to the
    linear-exponential (LINEX) utility u(x) = (x + (1 - e^{-alpha x}) / alpha) / 2,
    normalized so u(0) = 0 and u'(0) = 1. Keeps CARA's superlinear regression
    penalty while removing its reward cap: gains accrue at an asymptotic rate
    of 1/2 per unit, so the score is unbounded both above and below and can
    distinguish among candidates that improve both metrics. The trade-off is a
    weaker balance bias: at matched alpha its curvature is half of CARA's
    (u''(0) = -alpha/2 vs -alpha), and a large gain on one metric can
    partially offset weakness on the other. Recovers the Naive Score as
    alpha -> 0+.

        phi_LINEX(Delta) = (phi_nai(Delta) + phi_CARA(Delta)) / 2
                         = (1/2) * sum_i w_i * [Delta_i + (1 - exp(-alpha * Delta_i)) / alpha]

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)
    alpha: float > 0

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/7ambm2crdv
    """
    # Fuse naive + cara into one expm1 pass.
    u = -torch.expm1(-alpha * deltas) / alpha
    combined = deltas + u
    if weights is None:
        return 0.5 * combined.sum(dim=-1)
    return 0.5 * (combined * weights).sum(dim=-1)


def calc_norm_deltas(
    values: torch.Tensor,
    baseline_idx: int | torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample-normalized score deltas relative to a default edit.
    Because metrics live on different scales, scoring uses deltas

        Delta_i = (s_i - s_i^0) / (max_T s_i - min_T s_i)

    where min/max are over the same sample's candidate edits T and s_i^0 is
    the baseline (default) edit. When the per-metric range is positive,
    Delta_i is in [-1, 1]; Delta_i > 0 is an improvement over the default and
    Delta_i < 0 a regression.

    values: (..., N, C)
    baseline_idx: int or LongTensor matching values.shape[:-2]

    Returns: shape (..., N, C) with baseline rows at 0.
    """
    if values.ndim < 2:
        raise ValueError(f"values must be (..., N, C), got shape {tuple(values.shape)}")

    # Min-max normalize each sample independently over its N candidates.
    vmin = values.nan_to_num(nan=math.inf).amin(dim=-2, keepdim=True)
    vmax = values.nan_to_num(nan=-math.inf).amax(dim=-2, keepdim=True)
    normalized = (values - vmin) / (vmax - vmin + 1e-8)

    # Subtract the baseline (default) edit along the candidate axis.
    if isinstance(baseline_idx, torch.Tensor):
        leading = values.shape[:-2]
        if tuple(baseline_idx.shape) != tuple(leading):
            raise ValueError(f"baseline_idx shape {tuple(baseline_idx.shape)} must match {tuple(leading)}")
        c = values.shape[-1]
        idx = baseline_idx.to(dtype=torch.long, device=values.device)
        idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*leading, 1, c)
        baseline = torch.gather(normalized, dim=-2, index=idx)
    else:
        i = int(baseline_idx)
        baseline = normalized[..., i : i + 1, :]
    if torch.isnan(baseline).any():
        raise ValueError("Invalid NaN baseline cell")
    return normalized - baseline


def _group_norm_deltas(
    df: pd.DataFrame,
    metric_cols: list[str],
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """Pack a metrics DataFrame into per-sample normalized deltas.

    Groups by sample_id, packs equal-sized grids to (B, N, C), and applies
    calc_normalized_deltas in one batched call. Returns the (B, N, C) deltas
    and the df row indices per sample, aligned with the B axis.
    """
    from settings import DEFAULT_T_END, DEFAULT_T_START, SAMPLE_ID_COL, T_END_COL, T_START_COL

    if not metric_cols:
        raise ValueError("expected one or more metric column names")

    index_lists: list[np.ndarray] = []
    values_list: list[np.ndarray] = []
    baseline_list: list[int] = []

    for _, group in df.groupby(SAMPLE_ID_COL, sort=True):
        index_lists.append(group.index.to_numpy())
        values_list.append(group.loc[:, metric_cols].to_numpy(dtype=np.float64, copy=True))
        base_mask_t_start = np.isclose(group[T_START_COL].to_numpy(dtype=float), DEFAULT_T_START)
        base_mask_t_end = np.isclose(group[T_END_COL].to_numpy(dtype=float), DEFAULT_T_END)
        base_mask = base_mask_t_start & base_mask_t_end
        if int(base_mask.sum()) != 1:
            raise ValueError(f"Expected exactly one base row, found {int(base_mask.sum())}")
        # Append the index of the base row to both lists.
        baseline_list.append(int(np.flatnonzero(np.asarray(base_mask))[0]))

    values = torch.as_tensor(np.stack(values_list), dtype=torch.float64)
    baseline_idx = torch.as_tensor(baseline_list, dtype=torch.long)
    return calc_norm_deltas(values, baseline_idx), index_lists


def compute_delta_df(df: pd.DataFrame, *cols: str) -> pd.DataFrame:
    """Compute the delta target space for M: per-sample normalized deltas."""
    metric_cols = list(cols)
    deltas, index_lists = _group_norm_deltas(df, metric_cols)
    out = pd.DataFrame(np.nan, index=df.index, columns=list(cols), dtype=np.float64)
    deltas_np = deltas.detach().cpu().numpy()
    for k, idxs in enumerate(index_lists):
        out.loc[idxs, :] = deltas_np[k]
    return out


def score_df(
    df: pd.DataFrame,
    *cols: str,
    score_fn: Callable[..., torch.Tensor],
    **kwargs: Any,
) -> pd.Series:
    """
    Score each row of a metrics DataFrame via per-sample normalized deltas.

    Groups by sample_id, packs equal-sized grids to (B, N, C), applies
    calc_normalized_deltas then score_fn in one batched call, and returns a
    Series aligned to df.index.
    """
    deltas, index_lists = _group_norm_deltas(df, list(cols))

    # Process the score kwargs.
    score_kwargs = dict(kwargs)
    weights = score_kwargs.get("weights")
    if weights is not None and not isinstance(weights, torch.Tensor):
        score_kwargs["weights"] = torch.as_tensor(weights, dtype=deltas.dtype)

    scores = score_fn(deltas, **score_kwargs)
    scores_np = scores.detach().cpu().numpy()

    out = pd.Series(np.nan, index=df.index, dtype=np.float64, name=getattr(score_fn, "__name__", "score"))
    for idxs, row_scores in zip(index_lists, scores_np):
        out.loc[idxs] = row_scores
    return out
