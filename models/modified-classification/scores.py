"""Scalar quality scores for T selection and M ranking loss.

Torch cores are the source of truth for ranking-relevant scores so the same
function can score true labels and differentiable predictions. DataFrame
wrappers match the scores.py.new call style (*cols, weights, ...).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

_EPS = 1e-8


def weighted_combined_score(
    values: torch.Tensor,
    *,
    weights: torch.Tensor | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Weight-blend metric columns. values shape (N, C) -> scores (N,).

    When normalize is True, each row is min-max scaled across columns before
    blending (same as scores.py.new).
    """
    n_weights = weights.shape[-1] if weights is not None else values.shape[-1]
    weights = weights if weights is not None else torch.ones(n_weights, device=values.device, dtype=values.dtype)

    # Normalize values within each sample_id group.
    def _normalize_values(values: torch.Tensor) -> torch.Tensor:
        min_values = values.amin(dim=0, keepdim=True)
        max_values = values.amax(dim=0, keepdim=True)
        return (values - min_values) / (max_values - min_values + _EPS)
    
    values = _normalize_values(values) if normalize else values
    return (values * weights).sum(dim=-1) / (weights.sum() + _EPS)


def agreement_score(
    values: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Agreement in [0, 1] from per-row metric spread. values (N, C) -> (N,)."""
    raise NotImplementedError()


def naive_pareto_score(
    values: torch.Tensor,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    raise NotImplementedError()


def softplus_score(
    values: torch.Tensor,
    *,
    baseline_idx: int,
    alpha: float = 1.0,
    beta: float = 2.0,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Return a smooth score for each row relative to the baseline row in its
    sample_id group. When normalize is True, each metric is min-max scaled
    within each sample_id group before deltas are taken. With softplus
    transforms s_i = sp(delta_i) of the (possibly normalized) metric deltas:

        sp(x) = (1/beta)*(ln(1+e^{beta*x})-ln(2))
        m     = sum_i s_i + alpha * prod_i s_i

    The baseline scores 0. Rows that also score 0 but differ from the baseline
    on every metric are shifted down by epsilon. Improvements are rewarded
    smoothly and regressions are penalized.

    Plot of the score surface (2 metrics):
    https://www.desmos.com/3d/9slzoluqbd
    """

    # Normalize values within each sample_id group.
    def _normalize_values(values: torch.Tensor) -> torch.Tensor:
        min_values = values.amin(dim=0, keepdim=True)
        max_values = values.amax(dim=0, keepdim=True)
        return (values - min_values) / (max_values - min_values + _EPS)

    # Calculate the shifted softplus score.
    def _shifted_softplus(values: torch.Tensor, beta: float) -> torch.Tensor:
        if beta == 0:
            return values / 2.0
        zero = torch.zeros((), device=values.device, dtype=values.dtype)
        return torch.logaddexp(zero, beta * values) / beta - math.log(2) / beta

    # Penalize rows where the score is 0 but it is not the baseline.
    def _penalize_zeros(values: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
        mask = (values == 0) & (deltas != 0).all(dim=-1)
        return values - _EPS * mask.to(dtype=values.dtype)

    values = _normalize_values(values) if normalize else values
    deltas = values - values[baseline_idx]
    s = _shifted_softplus(deltas, beta)
    scores = s.sum(dim=-1) + alpha * s.prod(dim=-1)
    return _penalize_zeros(scores, deltas)


def weighted_combined_score_df(
    df: pd.DataFrame,
    *cols: str,
    weights: np.ndarray | list[float] | torch.Tensor | None = None,
    normalize: bool = True,
) -> pd.Series:
    """Wrapper for weighted_combined_score that takes a DataFrame and returns a Series."""
    values = torch.as_tensor(
        df.loc[:, list(cols)].to_numpy(dtype=np.float64, copy=True),
        dtype=torch.float64,
    )
    weights = None if weights is None else torch.as_tensor(weights, dtype=values.dtype)
    out = weighted_combined_score(values, weights=weights, normalize=normalize)
    weights_tag = "-".join(f"{float(v):g}" for v in weights.tolist()) if weights is not None else "-".join("1" for _ in cols)
    return pd.Series(out.detach().cpu().numpy(), index=df.index, name=f"weighted_score_{weights_tag}")


def agreement_score_df(
    df: pd.DataFrame,
    *cols: str,
    normalize: bool = True,
) -> pd.Series:
    """Wrapper for agreement_score that takes a DataFrame and returns a Series."""
    raise NotImplementedError()


def naive_pareto_score_df(
    df: pd.DataFrame,
    *cols: str,
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
    normalize: bool = False,
) -> pd.Series:
    """Wrapper for naive_pareto_score that takes a DataFrame and returns a Series."""
    raise NotImplementedError()


def softplus_score_df(
    df: pd.DataFrame,
    *cols: str,
    alpha: float = 1.0,
    beta: float = 2.0,
    normalize: bool = True,
) -> pd.Series:
    """Wrapper for softplus_score that takes a DataFrame and returns a Series."""
    from settings import DEFAULT_T_END, DEFAULT_T_START, SAMPLE_ID_COL

    scores = pd.Series(0.0, index=df.index, name="softplus_score")
    for _, group in df.groupby(SAMPLE_ID_COL):
        values = torch.as_tensor(group.loc[:, list(cols)].to_numpy(dtype=np.float64, copy=True), dtype=torch.float64)
        base_mask = np.isclose(group["t_start"], DEFAULT_T_START) & np.isclose(group["t_end"], DEFAULT_T_END)
        if int(base_mask.sum()) != 1:
            raise ValueError(f"Expected exactly one base row, found {int(base_mask.sum())}")
        baseline_idx = int(np.flatnonzero(np.asarray(base_mask))[0])
        out = softplus_score(values, baseline_idx=baseline_idx, alpha=alpha, beta=beta, normalize=normalize)
        scores.loc[group.index] = out.detach().cpu().numpy()
    return scores
