import pandas as pd
import numpy as np


def compute_weighted_combined_score(
    df: pd.DataFrame,
    lambda_psnr: float = 0.5,
    lambda_clip: float = 0.5,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
    psnr_min: float | None = None,
    psnr_max: float | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> pd.Series:
    """
    Return a combined quality score in [0, 1] that *may weight* PSNR and
    CLIP similarity metrics.
    
    Each metric is min-max normalized then blended by lambda_psnr and
    lambda_clip. The weights need not sum to 1.
    """
    psnr = df[psnr_col].to_numpy(dtype=float)
    clip = df[clip_col].to_numpy(dtype=float)

    p_min = np.nanmin(psnr) if psnr_min is None else psnr_min
    p_max = np.nanmax(psnr) if psnr_max is None else psnr_max
    c_min = np.nanmin(clip) if clip_min is None else clip_min
    c_max = np.nanmax(clip) if clip_max is None else clip_max

    eps = 1e-8
    psnr_norm = (psnr - p_min) / (p_max - p_min + eps)
    clip_norm = (clip - c_min) / (c_max - c_min + eps)

    weight_sum = lambda_psnr + lambda_clip
    combined = (lambda_psnr * psnr_norm + lambda_clip * clip_norm) / (weight_sum + eps)
    return pd.Series(combined, index=df.index, name=f"weighted_score_p{lambda_psnr}-c{lambda_clip}")


def compute_agreement_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
) -> pd.Series:
    """
    Return a score in [0, 1] measuring how closely PSNR and CLIP similarity
    agree on raw values.

    Returns a score of 1 when the two metrics are equal, 0 at the point of 
    maximum disagreement in the population.
    """
    psnr = df[psnr_col].to_numpy(dtype=float)
    clip = df[clip_col].to_numpy(dtype=float)

    diff = np.abs(psnr - clip)
    return pd.Series(1.0 - diff / (diff.max() + 1e-8), index=df.index, name="agreement_score")


def _find_baseline_idx(
    group: pd.DataFrame,
    base_t_start: float,
    base_t_end: float,
    sample_id,
) -> int:
    """Return the index of the baseline row for the given sample_id."""
    base_mask = (np.isclose(group["t_start"], base_t_start) & np.isclose(group["t_end"], base_t_end))
    if base_mask.sum() == 0:
        # No exact match, find the closest row
        dist = np.abs(group["t_start"] - base_t_start) + np.abs(group["t_end"] - base_t_end)
        base_idx = dist.idxmin()
        base_row = group.loc[base_idx]
        print(
            f"Baseline ({base_t_start}, {base_t_end}) not found for sample_id={sample_id!r}; "
            f"using ({base_row['t_start']}, {base_row['t_end']})."
        )
        return base_idx
    if base_mask.sum() != 1:
        raise ValueError(f"Expected exactly one base row, found {base_mask.sum()}")
    return group.index[base_mask][0]


def _minmax_normalize(series: pd.Series) -> pd.Series:
    """Min-max scale a series to [0, 1]. Constant columns map to 0."""
    lo, hi = series.min(), series.max()
    return (series - lo) / (hi - lo + 1e-6)


def _group_deltas(
    group: pd.DataFrame,
    psnr_col: str,
    clip_col: str,
    base_idx,
    normalized: bool,
) -> tuple[np.ndarray, np.ndarray]:
    psnr = _minmax_normalize(group[psnr_col]) if normalized else group[psnr_col].astype(float)
    clip = _minmax_normalize(group[clip_col]) if normalized else group[clip_col].astype(float)
    delta_psnr = (psnr - psnr.loc[base_idx]).to_numpy(dtype=float)
    delta_clip = (clip - clip.loc[base_idx]).to_numpy(dtype=float)
    return delta_psnr, delta_clip


def compute_naive_pareto_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
    normalize: bool = False,
) -> pd.Series:
    """
    Return a Pareto improvement score for each row relative to the
    baseline row in its sample_id group.

    The baseline is the row at (DEFAULT_T_START, DEFAULT_T_END). When
    normalized is True, PSNR and CLIP are min-max scaled within each
    sample_id group before deltas are taken. Each row's score is
    max(0, delta_psnr) * max(0, delta_clip) relative to that baseline
    (the baseline itself scores 0).
    """
    from models.classification.settings import DEFAULT_T_START, DEFAULT_T_END

    base_t_start = DEFAULT_T_START if base_t_start is None else base_t_start
    base_t_end = DEFAULT_T_END if base_t_end is None else base_t_end

    scores = pd.Series(0.0, index=df.index, name="naive_pareto_score")

    for sample_id, group in df.groupby(sample_id_col):
        base_idx = _find_baseline_idx(group, base_t_start, base_t_end, sample_id)
        delta_psnr, delta_clip = _group_deltas(group, psnr_col, clip_col, base_idx, normalize)
        row_scores = np.maximum(0, delta_psnr) * np.maximum(0, delta_clip)
        scores.loc[group.index] = row_scores

    return scores


def compute_softplus_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
    alpha: float = 1.0,
    beta: float = 2.0,
    epsilon: float = 1e-6,
    normalize: bool = True,
) -> pd.Series:
    """
    Return a smooth score for each row relative to the baseline row in its
    sample_id group. When normalized is True, PSNR and CLIP are min-max
    scaled within each sample_id group before deltas are taken. With a, b
    the (possibly normalized) metrics and A, B the baseline values:

        sp(x)   = (1/beta)*(ln(1+e^{beta*x})-ln(2))
        m(a, b) = sp(a-A) + sp(b-B)
                
                Optional (alpha > 0): Bias towards Pareto improvement.
                + alpha*sp(a-A)*sp(b-B)

    The baseline scores 0. Rows that also score 0 but differ from the baseline
    on both metrics are shifted down by epsilon. Improvements are rewarded
    smoothly and regressions are penalized.

    Plot of the score surface:
    https://www.desmos.com/3d/9slzoluqbd
    """
    from models.classification.settings import DEFAULT_T_START, DEFAULT_T_END

    # Calculate the shifted softplus score.
    def _shifted_softplus(x: np.ndarray, beta: float) -> np.ndarray:
        """Calculate (1/beta)*(ln(1+e^{beta*x})-ln(2))"""
        return np.logaddexp(0, beta * x) / beta - np.log(2) / beta

    # Penalize rows where the score is 0 but it is not the baseline.
    def _penalize_zeros(scores, delta_psnr, delta_clip):
        """Zero out rows where the score is 0 but it is not the baseline."""
        mask = (scores == 0) & (delta_psnr != 0) & (delta_clip != 0)
        return scores - epsilon * mask

    base_t_start = DEFAULT_T_START if base_t_start is None else base_t_start
    base_t_end = DEFAULT_T_END if base_t_end is None else base_t_end

    scores = pd.Series(0.0, index=df.index, name="softplus_score")
    for sample_id, group in df.groupby(sample_id_col):
        base_idx = _find_baseline_idx(group, base_t_start, base_t_end, sample_id)
        delta_psnr, delta_clip = _group_deltas(group, psnr_col, clip_col, base_idx, normalize)
        s_psnr = _shifted_softplus(delta_psnr, beta)
        s_clip = _shifted_softplus(delta_clip, beta)
        row_scores = s_psnr + s_clip + alpha * s_psnr * s_clip
        row_scores = _penalize_zeros(row_scores, delta_psnr, delta_clip)
        scores.loc[group.index] = row_scores

    return scores