import pandas as pd
import numpy as np


def compute_combined_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_similarity_target_image",
    psnr_min: float | None = None,
    psnr_max: float | None = None,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> pd.Series:
    """Return a combined quality score in [0, 1] that equally weights PSNR and
    CLIP similarity after min-max normalizing each to account for their
    differing value ranges.

    If min/max bounds are not provided, they are derived from the passed
    DataFrame, making the normalization relative to the observed population.
    Pass explicit bounds (e.g. from training data) to keep the scale fixed
    across splits or future data.
    """
    psnr = df[psnr_col].to_numpy(dtype=float)
    clip = df[clip_col].to_numpy(dtype=float)

    p_min = psnr.min() if psnr_min is None else psnr_min
    p_max = psnr.max() if psnr_max is None else psnr_max
    c_min = clip.min() if clip_min is None else clip_min
    c_max = clip.max() if clip_max is None else clip_max

    eps = 1e-8
    psnr_norm = (psnr - p_min) / (p_max - p_min + eps)
    clip_norm = (clip - c_min) / (c_max - c_min + eps)

    combined = (psnr_norm + clip_norm) / 2.0
    return pd.Series(combined, index=df.index, name="combined_score")


def compute_agreement_score(
    df: pd.DataFrame,
    psnr_col: str = "psnr",
    clip_col: str = "clip_similarity_target_image",
) -> pd.Series:
    """Return a score in [0, 1] measuring how closely PSNR and CLIP similarity
    agree on raw values.

    Returns a score of 1 when the two metrics are equal, 0 at the point of 
    maximum disagreement in the population.
    """
    psnr = df[psnr_col].to_numpy(dtype=float)
    clip = df[clip_col].to_numpy(dtype=float)

    diff = np.abs(psnr - clip)
    return pd.Series(1.0 - diff / (diff.max() + 1e-8), index=df.index, name="agreement_score")
