"""Shared data loading and embedding helpers for M and T training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from model_m import MetricPredictor
from settings import (
    CLIP_COL,
    IMAGE_PATH_COL,
    METRICS_CSV,
    PSNR_COL,
    SOURCE_IMAGE_NAME,
    SOURCE_IMAGE_PARENT_LEVEL,
    STRINGS_CSV,
    TARGET_COLS,
    TARGET_T_DELTA,
)


def source_path(image_path: str) -> str:
    # Map a cell-image path to its sample's source.png.
    return str(Path(image_path).parents[SOURCE_IMAGE_PARENT_LEVEL] / SOURCE_IMAGE_NAME)


def id_from_image_path(image_path: str) -> str:
    # Extract the 12-digit string-pair id from the sample folder name.
    sample_folder = Path(image_path).parents[SOURCE_IMAGE_PARENT_LEVEL].name
    return sample_folder.split("_")[-1]


def load_data(metrics_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    metrics_csv = metrics_csv or METRICS_CSV
    metrics = pd.read_csv(metrics_csv)
    metrics = metrics.rename(columns={PSNR_COL: "psnr", CLIP_COL: "clip"})

    # Select a single t_delta slice if specified.
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics["t_delta"].values:
            raise ValueError(
                f"{TARGET_T_DELTA=} not found in t_delta "
                f"(distinct: {sorted(metrics['t_delta'].unique())})."
            )
        metrics = metrics[metrics["t_delta"] == TARGET_T_DELTA].copy()

    metrics["source_path"] = metrics[IMAGE_PATH_COL].map(source_path)
    metrics["id"] = metrics[IMAGE_PATH_COL].map(id_from_image_path)

    # Merge prompt strings keyed by sample id.
    strings = pd.read_csv(STRINGS_CSV, dtype={"id": str})
    df = pd.merge(metrics, strings, on="id", how="left")
    if df["source_prompt"].isna().any():
        missing = df.loc[df["source_prompt"].isna(), "id"].unique()
        raise ValueError(f"No prompt strings found for ids: {missing.tolist()}")

    # Partial-grid training: only rows with labeled=True are kept.
    if "labeled" not in df.columns:
        df["labeled"] = True
    df = df[df["labeled"]].copy()

    cols = [
        "sample_id", "id", "t_start", "t_end", "t_delta",
        "psnr", "clip", "source_path", "source_prompt", "target_prompt", "labeled",
    ]
    return df[cols].reset_index(drop=True)


def precompute_embeddings(
    df: pd.DataFrame, predictor: MetricPredictor, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode the source image and prompt pair once per sample_id."""
    samples = df.drop_duplicates(subset="sample_id").sort_values("sample_id")
    images = [Image.open(p).convert("RGB") for p in samples["source_path"]]
    src_prompts = samples["source_prompt"].tolist()
    tar_prompts = samples["target_prompt"].tolist()

    img_emb = predictor.image_encoder(images).to(device)
    src_emb = predictor.text_encoder(src_prompts).to(device)
    tar_emb = predictor.text_encoder(tar_prompts).to(device)

    return {
        sid: {"img": img_emb[i], "src": src_emb[i], "tar": tar_emb[i]}
        for i, sid in enumerate(samples["sample_id"].tolist())
    }


def build_tensors(
    df: pd.DataFrame, emb: dict[str, dict[str, torch.Tensor]]
) -> tuple[torch.Tensor, ...]:
    """Assemble per-row (img, src, tar, t, y, sample_idx) tensors."""
    sample_ids = sorted(df["sample_id"].unique())
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}
    img = torch.stack([emb[s]["img"] for s in df["sample_id"]])
    src = torch.stack([emb[s]["src"] for s in df["sample_id"]])
    tar = torch.stack([emb[s]["tar"] for s in df["sample_id"]])
    t = torch.tensor(df[["t_start", "t_end"]].values, dtype=torch.float)
    y = torch.tensor(df[list(TARGET_COLS)].values, dtype=torch.float)
    # sample_idx groups grid rows for within-sample ranking loss.
    sample_idx = torch.tensor([sid_to_idx[s] for s in df["sample_id"]], dtype=torch.long)
    return img, src, tar, t, y, sample_idx


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
