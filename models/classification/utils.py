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

    The baseline is the row where t_start == DEFAULT_T_START -
    DEFAULT_T_DELTA and t_end == DEFAULT_T_STOP.  If both PSNR and
    CLIP strictly exceed the baseline, the score is 1 + delta_psnr +
    delta_clip; otherwise it is 0. Rows whose group has no unambiguous
    baseline also score 0. Rows with t_start == base_t_start and t_end
    == base_t_end will score 1.
    """
    from models.classification.settings import PAPER_T_START, PAPER_T_END

    base_t_start = PAPER_T_START if base_t_start is None else base_t_start
    base_t_end = PAPER_T_END if base_t_end is None else base_t_end

    scores = pd.Series(0.0, index=df.index, name="naive_pareto_score")

    for _, group in df.groupby(sample_id_col):
        base_mask = (
            np.isclose(group["t_start"], base_t_start)
            & np.isclose(group["t_end"], base_t_end)
        )
        if base_mask.sum() != 1:
            # Exactly one such base row must exist
            raise ValueError(
                f"Expected exactly one base row, found {base_mask.sum()} "
                f"for t_start={base_t_start}, t_end={base_t_end}."
            )
        base_idx = group.index[base_mask][0]
        base_psnr = group.loc[base_idx, psnr_col]
        base_clip = group.loc[base_idx, clip_col]
        delta_psnr = group[psnr_col] - base_psnr
        delta_clip = group[clip_col] - base_clip
        improving = (delta_psnr > 0) & (delta_clip > 0)
        row_scores = np.zeros(len(group), dtype=float)
        row_scores[:] = 0.0
        row_scores[np.where(group.index == base_idx)[0][0]] = 1.0
        row_scores[improving.to_numpy()] = 1 + delta_psnr[improving] + delta_clip[improving]
        scores.loc[group.index] = row_scores

    return scores