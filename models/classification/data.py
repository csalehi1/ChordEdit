import pandas as pd
import numpy as np


ID_TO_METRICS_PATH = "models/classifier/id_to_metrics.csv"


def compute_combined_quality_score(
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
    differing value ranges (~17-41 vs ~7-34 in the current dataset).

    If min/max bounds are not provided they are derived from the passed
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
    return pd.Series(combined, index=df.index, name="combined_quality_score")
