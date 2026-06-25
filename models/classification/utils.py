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
    return pd.Series(combined, index=df.index, name=f"weighted_score_p{lambda_psnr}_c{lambda_clip}")


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


def compute_naive_pareto_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
) -> pd.Series:
    """
    Return a Pareto improvement score for each row relative to the
    baseline row in its sample_id group.

    The baseline is the row where t_start == PAPER_T_START - T_DELTA_TARGET and
    t_end == PAPER_T_END. Each row's score is max(0, delta_psnr) * max(0,
    delta_clip) relative to that baseline (the baseline itself scores 0).
    """
    from models.classification.settings import PAPER_T_START, PAPER_T_END, T_DELTA_TARGET

    base_t_start = PAPER_T_START - T_DELTA_TARGET if base_t_start is None else base_t_start
    base_t_end = PAPER_T_END if base_t_end is None else base_t_end

    scores = pd.Series(0.0, index=df.index, name="naive_pareto_score")

    for sample_id, group in df.groupby(sample_id_col):
        base_mask = (
            np.isclose(group["t_start"], base_t_start)
            & np.isclose(group["t_end"], base_t_end)
        )
        # If the baseline row is not found, use the closest row (e.g.,
        # t_delta = 0.15 and data is listed by 0.1, will be 0.8).
        if base_mask.sum() == 0:
            dist = (np.abs(group["t_start"] - base_t_start) + np.abs(group["t_end"] - base_t_end))
            base_idx = dist.idxmin()
            base_row = group.loc[base_idx]
            print(
                f"Baseline ({base_t_start}, {base_t_end}) not found for sample_id={sample_id!r}; "
                f"using ({base_row['t_start']}, {base_row['t_end']})."
            )
        # If more than one baseline row is found, raise an error.
        elif base_mask.sum() != 1:
            raise ValueError(f"Expected exactly one base row, found {base_mask.sum()}")
        # If exactly one baseline row is found, use it.
        else:
            base_idx = group.index[base_mask][0]
        base_psnr = group.loc[base_idx, psnr_col]
        base_clip = group.loc[base_idx, clip_col]
        delta_psnr = group[psnr_col] - base_psnr
        delta_clip = group[clip_col] - base_clip
        row_scores = np.maximum(0, delta_psnr) * np.maximum(0, delta_clip)
        scores.loc[group.index] = row_scores

    return scores


def compute_pareto_biased_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_edited",
    sample_id_col: str = "sample_id",
    base_t_start: float | None = None,
    base_t_end: float | None = None,
    alpha: float = 2.0,
) -> pd.Series:
    """
    Return a smooth Pareto-improvement-inclined score for each row
    relative to the baseline row in its sample_id group.

    The baseline is the row where t_start == PAPER_T_START -
    T_DELTA_TARGET and t_end == PAPER_T_END. With a = PSNR, b = CLIP,
    and A, B the baseline values, each row receives:

        m(a, b) = s(a - A) + s(b - B) + alpha * s(a - A) * s(b - B)

    where s(t) = softplus(t) - log(2) and softplus(t) = log(1 +
    exp(t)). At the baseline, s(0) = 0 so m(A, B) = 0.
    """
    from models.classification.settings import PAPER_T_START, PAPER_T_END, T_DELTA_TARGET

    def shifted_softplus(t: np.ndarray) -> np.ndarray:
        softplus = np.log1p(np.exp(-np.abs(t))) + np.maximum(t, 0)
        return softplus - np.log(2)

    base_t_start = PAPER_T_START - T_DELTA_TARGET if base_t_start is None else base_t_start
    base_t_end = PAPER_T_END if base_t_end is None else base_t_end

    scores = pd.Series(0.0, index=df.index, name="pareto_biased_score")

    for sample_id, group in df.groupby(sample_id_col):
        base_mask = (
            np.isclose(group["t_start"], base_t_start)
            & np.isclose(group["t_end"], base_t_end)
        )
        # If the baseline row is not found, use the closest row (e.g.,
        # t_delta = 0.15 and data is listed by 0.1, will be 0.8).
        if base_mask.sum() == 0:
            dist = (np.abs(group["t_start"] - base_t_start) + np.abs(group["t_end"] - base_t_end))
            base_idx = dist.idxmin()
            base_row = group.loc[base_idx]
            print(
                f"Baseline ({base_t_start}, {base_t_end}) not found for sample_id={sample_id!r}; "
                f"using ({base_row['t_start']}, {base_row['t_end']})."
            )
        # If more than one baseline row is found, raise an error.
        elif base_mask.sum() != 1:
            raise ValueError(f"Expected exactly one base row, found {base_mask.sum()}")
        # If exactly one baseline row is found, use it.
        else:
            base_idx = group.index[base_mask][0]
        base_psnr = group.loc[base_idx, psnr_col]
        base_clip = group.loc[base_idx, clip_col]
        s_psnr = shifted_softplus((group[psnr_col] - base_psnr).to_numpy(dtype=float))
        s_clip = shifted_softplus((group[clip_col] - base_clip).to_numpy(dtype=float))
        row_scores = s_psnr + s_clip + alpha * s_psnr * s_clip
        scores.loc[group.index] = row_scores

    return scores