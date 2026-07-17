import pandas as pd
import numpy as np

_EPS = 1e-8


def compute_weighted_combined_score(
    df: pd.DataFrame,
    *cols: str,
    weights: np.ndarray | None = None,
    normalize: bool = True,
) -> pd.Series:
    """
    Return a combined quality score that weights an arbitrary set of metrics.

    When normalize is True (default), each metric is min-max scaled to [0, 1]
    before blending; the result is also in [0, 1]. When normalize is False,
    raw metric values are blended directly. The weights need not sum to 1.
    """
    n_values = len(cols)
    values = df.loc[:, list(cols)].to_numpy(dtype=float)
    n_weights = len(weights) if weights is not None else n_values
    weights = weights if weights is not None else np.ones(n_values, dtype=float)
    if n_weights != n_values:
        raise ValueError(f"weights must have shape ({n_values},), got {weights.shape}")

    if normalize:
        min_values = np.nanmin(values, axis=1, keepdims=True)
        max_values = np.nanmax(values, axis=1, keepdims=True)
        values = (values - min_values) / (max_values - min_values + _EPS)

    combined = (values * weights).sum(axis=1) / (weights.sum() + _EPS)
    weight_tag = "-".join(f"{w:g}" for w in weights)
    return pd.Series(combined, index=df.index, name=f"weighted_score_{weight_tag}")


def compute_agreement_score(
    df: pd.DataFrame,
    *cols: str,
    normalize: bool = True,
) -> pd.Series:
    """
    Return a score in [0, 1] measuring how closely the given metrics agree.

    Agreement is 1 when all metrics are equal on a row, and 0 at the point of
    maximum per-row spread (max - min) in the population.
    """
    values = df.loc[:, list(cols)].to_numpy(dtype=float)

    if normalize:
        min_values = np.nanmin(values, axis=1, keepdims=True)
        max_values = np.nanmax(values, axis=1, keepdims=True)
        values = (values - min_values) / (max_values - min_values + _EPS)

    spread = np.nanmax(values, axis=1) - np.nanmin(values, axis=1)
    return pd.Series(1.0 - spread / (np.nanmax(spread) + _EPS), index=df.index, name="agreement_score")


def compute_naive_pareto_score(
    df: pd.DataFrame,
    *cols: str,
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
    normalize: bool = False,
) -> pd.Series:
    """
    Return a Pareto improvement score for each row relative to the
    baseline row in its sample_id group.

    The baseline is the row at (DEFAULT_T_START, DEFAULT_T_END). When
    normalize is True, each metric is min-max scaled within each sample_id
    group before deltas are taken. Each row's score is the product of
    max(0, delta_i) over all metrics relative to that baseline (the
    baseline itself scores 0).
    """
    # TODO: Currently not implemented
    raise NotImplementedError()


def compute_softplus_score(
    df: pd.DataFrame,
    *cols: str,
    alpha: float = 1.0,
    beta: float = 2.0,
    normalize: bool = True,
) -> pd.Series:
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
    from .settings import SAMPLE_ID_COL, DEFAULT_T_START, DEFAULT_T_END

    # Normalize values within each sample_id group.
    def _normalize_values(values: np.ndarray) -> np.ndarray:
        min_values = np.nanmin(values, axis=0, keepdims=True)
        max_values = np.nanmax(values, axis=0, keepdims=True)
        return (values - min_values) / (max_values - min_values + _EPS)

    # Calculate the shifted softplus score.
    def _shifted_softplus(x: np.ndarray, beta: float) -> np.ndarray:
        """Calculate (1/beta)*(ln(1+e^{beta*x})-ln(2))"""
        if beta == 0: return x / 2.0
        return np.logaddexp(0, beta * x) / beta - np.log(2) / beta

    # Penalize rows where the score is 0 but it is not the baseline.
    def _penalize_zeros(scores: np.ndarray, deltas: np.ndarray) -> np.ndarray:
        """Shift down rows where the score is 0 but every metric differs from baseline."""
        mask = (scores == 0) & np.all(deltas != 0, axis=1)
        return scores - _EPS * mask

    scores = pd.Series(0.0, index=df.index, name="softplus_score")
    for sample_id, group in df.groupby(SAMPLE_ID_COL):
        values = group.loc[:, list(cols)].to_numpy(dtype=float)
        values = _normalize_values(values) if normalize else values
        base_mask = (np.isclose(group["t_start"], DEFAULT_T_START) & np.isclose(group["t_end"], DEFAULT_T_END))
        if base_mask.sum() != 1:
            raise ValueError(f"Expected exactly one base row, found {base_mask.sum()}")
        base_pos = int(np.flatnonzero(np.asarray(base_mask))[0])
        deltas = values - values[base_pos]
        s = _shifted_softplus(deltas, beta)
        group_scores = s.sum(axis=1) + alpha * s.prod(axis=1)
        group_scores = _penalize_zeros(group_scores, deltas)
        scores.loc[group.index] = group_scores

    return scores
