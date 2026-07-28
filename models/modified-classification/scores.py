from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch


_EPS = 1e-8


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


def calc_normalized(
    values: torch.Tensor,
    *,
    per_sample: bool = True,
) -> torch.Tensor:
    """
    Min-max normalize metric columns. If per_sample is True (default),
    each leading sample is scaled independently over its N candidates.
    When False, each metric uses a single min/max over all non-metric
    dimensions.

    NaN marks an unlabeled candidate (e.g. a sparse timestep grid). NaN cells
    are excluded from the min/max — one missing cell cannot poison the rest of
    the sample — and stay NaN in the output. NaN support is for scoring/eval
    only: backprop through a NaN-sparse tensor yields NaN gradients (0 * NaN
    in elementwise backwards), so training must use dense labeled-only
    batches, as train_m's ranking loss does.

    values: (..., N, C)
    per_sample: bool = True

    Returns: (..., N, C)
    """
    if values.ndim < 2:
        raise ValueError(f"values must be (..., N, C), got shape {tuple(values.shape)}")

    if per_sample:
        # Independent range per sample along candidate axis N. Torch has no
        # nanmin/nanmax, so mask NaN with +/-inf sentinels that can never win
        # the reduction.
        vmin = values.nan_to_num(nan=math.inf).amin(dim=-2, keepdim=True)
        vmax = values.nan_to_num(nan=-math.inf).amax(dim=-2, keepdim=True)
    else:
        # One global range per metric across every leading / candidate dim.
        reduce_dims = tuple(range(values.ndim - 1))
        vmin = values.nan_to_num(nan=math.inf).amin(dim=reduce_dims, keepdim=True)
        vmax = values.nan_to_num(nan=-math.inf).amax(dim=reduce_dims, keepdim=True)

    return (values - vmin) / (vmax - vmin + _EPS)


def calc_deltas(
    values: torch.Tensor,
    baseline_idx: int | torch.Tensor,
) -> torch.Tensor:
    """
    Subtract the baseline (default) edit along the candidate axis.

    Every delta is relative to the baseline, so a NaN (unlabeled) baseline
    cell would silently invalidate the whole sample; raise instead.

    values: (..., N, C)
    baseline_idx: int or LongTensor matching values.shape[:-2]

    Returns: (..., N, C) with baseline rows at 0.
    """
    if values.ndim < 2:
        raise ValueError(f"values must be (..., N, C), got shape {tuple(values.shape)}")

    def _gather_baseline(values: torch.Tensor, baseline_idx: int | torch.Tensor) -> torch.Tensor:
        if isinstance(baseline_idx, torch.Tensor):
            leading = values.shape[:-2]
            if tuple(baseline_idx.shape) != tuple(leading):
                raise ValueError(f"baseline_idx shape {tuple(baseline_idx.shape)} must match {tuple(leading)}")
            c = values.shape[-1]
            idx = baseline_idx.to(dtype=torch.long, device=values.device)
            idx = idx.unsqueeze(-1).unsqueeze(-1).expand(*leading, 1, c)
            return torch.gather(values, dim=-2, index=idx)
        i = int(baseline_idx)
        return values[..., i : i + 1, :]

    baseline = _gather_baseline(values, baseline_idx)
    if torch.isnan(baseline).any():
        raise ValueError("baseline (default) cell is NaN/unlabeled for at least one sample")
    return values - baseline


def calc_normalized_deltas(
    values: torch.Tensor,
    baseline_idx: int | torch.Tensor,
    *,
    per_sample: bool = True,
) -> torch.Tensor:
    """
    Per-sample-normalized score deltas relative to a default edit.
    Because metrics live on different scales, scoring uses deltas

        Delta_i = (s_i - s_i^0) / (max_T s_i - min_T s_i)

    where min/max are over the same sample's candidate edits T when
    per_sample=True, and s_i^0 is the baseline (default) edit. When the
    per-metric range is positive, Delta_i is in [-1, 1]; Delta_i > 0 is an
    improvement over the default and Delta_i < 0 a regression.

    Equivalent to calc_deltas(calc_normalized(values, per_sample=...), ...).
    NaN (unlabeled) cells stay NaN without affecting labeled cells; the
    baseline cell itself must be labeled (calc_deltas raises otherwise).

    values: (..., N, C)
    baseline_idx: int or LongTensor matching values.shape[:-2]
    per_sample: bool = True

    Returns: (..., N, C)
    """
    return calc_deltas(calc_normalized(values, per_sample=per_sample), baseline_idx)


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
    from settings import DEFAULT_T_END, DEFAULT_T_START, SAMPLE_ID_COL, T_END_COL, T_START_COL

    if not cols:
        raise ValueError("expected one or more metric column names")

    metric_cols = list(cols)
    index_lists: list[np.ndarray] = []
    values_list: list[np.ndarray] = []
    baseline_list: list[int] = []

    for _, group in df.groupby(SAMPLE_ID_COL, sort=True):
        # Get the index of the group.
        index_lists.append(group.index.to_numpy())
        # Get the values of the group.
        values_list.append(group.loc[:, metric_cols].to_numpy(dtype=np.float64, copy=True))
        # Get the mask for the base row.
        base_mask_t_start = np.isclose(group[T_START_COL].to_numpy(dtype=float), DEFAULT_T_START)
        base_mask_t_end = np.isclose(group[T_END_COL].to_numpy(dtype=float), DEFAULT_T_END)
        base_mask = base_mask_t_start & base_mask_t_end
        # Verify that there is exactly one base row.
        if int(base_mask.sum()) != 1:
            raise ValueError(f"Expected exactly one base row, found {int(base_mask.sum())}")
        # Append the index of the base row to both lists.
        baseline_list.append(int(np.flatnonzero(np.asarray(base_mask))[0]))

    n_rows = {v.shape[0] for v in values_list}
    if len(n_rows) != 1:
        raise ValueError(f"ragged candidate counts across samples: {sorted(n_rows)}")

    values = torch.as_tensor(np.stack(values_list), dtype=torch.float64)
    baseline_idx = torch.as_tensor(baseline_list, dtype=torch.long)
    deltas = calc_normalized_deltas(values, baseline_idx)

    # Process the score kwargs.
    score_kwargs = dict(kwargs)
    weights = score_kwargs.get("weights")
    if weights is not None and not isinstance(weights, torch.Tensor):
        score_kwargs["weights"] = torch.as_tensor(weights, dtype=values.dtype)

    scores = score_fn(deltas, **score_kwargs)
    scores_np = scores.detach().cpu().numpy()

    out = pd.Series(np.nan, index=df.index, dtype=np.float64, name=getattr(score_fn, "__name__", "score"))
    for idxs, row_scores in zip(index_lists, scores_np):
        out.loc[idxs] = row_scores
    return out
