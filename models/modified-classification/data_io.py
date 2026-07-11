"""Shared data loading and embedding helpers for M and T training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model_m import MetricPredictor
from settings import (
    CELL_PATH_COL,
    IMAGE_PATH_COL,
    MASK_PATH_COL,
    METRICS_CSV,
    M_TARGET_COLS,
    PSNR_COL,
    SAMPLE_ID_COL,
    SOURCE_PROMPT_COL,
    INPUTS_CSV,
    T_DELTA_COL,
    T_TARGET_COL,
    T_TARGET_FUNC,
    TARGET_PROMPT_COL,
    TARGET_T_DELTA,
    DATASET_DIR,
    GENERATED_DIR,
)


def _normalize_sample_id(value) -> str:
    return f"{int(value):08d}"


def resolve_cell_path(cell_path: str) -> str:
    path = Path(cell_path)
    if path.is_absolute():
        return str(path)
    return str(GENERATED_DIR / cell_path.lstrip("/"))


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str(DATASET_DIR / image_path)


def resolve_mask_path(mask_path: str) -> str:
    path = Path(mask_path)
    if path.is_absolute():
        return str(path)
    return str(DATASET_DIR / mask_path)


def load_metrics(metrics_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    metrics_csv = metrics_csv or METRICS_CSV
    metrics = pd.read_csv(metrics_csv)
    metrics[SAMPLE_ID_COL] = metrics[SAMPLE_ID_COL].map(_normalize_sample_id)
    metrics[CELL_PATH_COL] = metrics[CELL_PATH_COL].map(resolve_cell_path)

    # Select a single t_delta slice if specified.
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics[T_DELTA_COL].values:
            raise ValueError(
                f"{TARGET_T_DELTA=} not found in {T_DELTA_COL} "
                f"(distinct: {sorted(metrics[T_DELTA_COL].unique())})."
            )
        metrics = metrics[metrics[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Merge prompt strings and source-image paths keyed by sample id.
    strings = pd.read_csv(INPUTS_CSV)
    strings[SAMPLE_ID_COL] = strings[SAMPLE_ID_COL].map(_normalize_sample_id)
    strings["source_path"] = strings[IMAGE_PATH_COL].map(resolve_image_path)
    strings["mask_path"] = strings[MASK_PATH_COL].map(resolve_mask_path)
    merge_cols = [
        SAMPLE_ID_COL,
        SOURCE_PROMPT_COL,
        TARGET_PROMPT_COL,
        "source_path",
        "mask_path",
    ]
    df = pd.merge(
        metrics,
        strings[merge_cols],
        on=SAMPLE_ID_COL,
        how="left",
    )
    if df[SOURCE_PROMPT_COL].isna().any():
        missing = df.loc[df[SOURCE_PROMPT_COL].isna(), SAMPLE_ID_COL].unique()
        raise ValueError(f"No prompt strings found for sample_ids: {missing.tolist()}")
    if df["mask_path"].isna().any():
        missing = df.loc[df["mask_path"].isna(), SAMPLE_ID_COL].unique()
        raise ValueError(f"No mask paths found for sample_ids: {missing.tolist()}")

    df["id"] = df[SAMPLE_ID_COL]

    # Partial-grid training: only rows with labeled=True are kept.
    if "labeled" not in df.columns:
        df["labeled"] = True
    df = df[df["labeled"]].copy()

    df[T_TARGET_COL] = T_TARGET_FUNC(df)

    cols = [
        SAMPLE_ID_COL,
        "id",
        "t_start",
        "t_end",
        T_DELTA_COL,
        *M_TARGET_COLS,
        T_TARGET_COL,
        "source_path",
        "mask_path",
        SOURCE_PROMPT_COL,
        TARGET_PROMPT_COL,
        "labeled",
    ]
    return df[cols].reset_index(drop=True)


def precompute_embeddings(
    df: pd.DataFrame, predictor: MetricPredictor, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode the source image, mask, and prompt pair once per sample_id."""
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    images = [Image.open(p).convert("RGB") for p in samples["source_path"]]
    masks = [Image.open(p).convert("RGB") for p in samples["mask_path"]]
    src_prompts = samples[SOURCE_PROMPT_COL].tolist()
    tar_prompts = samples[TARGET_PROMPT_COL].tolist()

    img_emb = predictor.image_encoder(images).to(device)
    mask_emb = predictor.image_encoder(masks).to(device)
    src_emb = predictor.text_encoder(src_prompts).to(device)
    tar_emb = predictor.text_encoder(tar_prompts).to(device)

    return {
        sid: {
            "img": img_emb[i],
            "mask": mask_emb[i],
            "src": src_emb[i],
            "tar": tar_emb[i],
        }
        for i, sid in enumerate(samples[SAMPLE_ID_COL].tolist())
    }


def build_tensors(
    df: pd.DataFrame, emb: dict[str, dict[str, torch.Tensor]]
) -> tuple[torch.Tensor, ...]:
    """Assemble per-row (img, mask, src, tar, t, y, sample_idx) tensors."""
    sample_ids = sorted(df[SAMPLE_ID_COL].unique())
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    img = torch.stack([emb[s]["img"] for s in df[SAMPLE_ID_COL]])
    mask = torch.stack([emb[s]["mask"] for s in df[SAMPLE_ID_COL]])
    src = torch.stack([emb[s]["src"] for s in df[SAMPLE_ID_COL]])
    tar = torch.stack([emb[s]["tar"] for s in df[SAMPLE_ID_COL]])
    t = torch.tensor(df[["t_start", "t_end"]].values, dtype=torch.float)
    y = torch.tensor(df[list(M_TARGET_COLS)].values, dtype=torch.float)
    # sample_idx groups grid rows for within-sample ranking loss.
    sample_idx = torch.tensor([sid_to_idx[s] for s in df[SAMPLE_ID_COL]], dtype=torch.long)
    return img, mask, src, tar, t, y, sample_idx


def df_to_metric_grids(
    df: pd.DataFrame,
    sample_ids: list,
    t_start_values: list[float] | tuple[float, ...],
    t_end_values: list[float] | tuple[float, ...],
    col: str,
) -> tuple[np.ndarray, dict]:
    """Build (N, n_start, n_end) ground-truth grid for one metric column."""
    t_start_values = list(t_start_values)
    t_end_values = list(t_end_values)
    n_img = len(sample_ids)
    n1, n2 = len(t_start_values), len(t_end_values)
    sid_to_k = {sid: k for k, sid in enumerate(sample_ids)}
    i_of = {v: i for i, v in enumerate(t_start_values)}
    j_of = {v: j for j, v in enumerate(t_end_values)}
    out = np.full((n_img, n1, n2), np.nan)
    for row in df.itertuples():
        out[sid_to_k[row.sample_id], i_of[row.t_start], j_of[row.t_end]] = getattr(row, col)
    return out, {"sid_to_k": sid_to_k, "i_of": i_of, "j_of": j_of}
