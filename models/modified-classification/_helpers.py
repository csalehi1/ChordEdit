"""Shared helpers for model M training and inference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from settings import *

_EPS = 1e-8


def settings_hash(path: Path | None = None) -> str:
    path = path or Path(__file__).resolve().parent / "settings.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_settings_hash(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    out = run_dir / "settings_hash.txt"
    out.write_text(settings_hash() + "\n", encoding="utf-8")
    return out


def check_settings_hash(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    path = run_dir / "settings_hash.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run train_m.py to record settings hash for this run."
        )
    saved = path.read_text(encoding="utf-8").strip()
    live = settings_hash()
    if saved != live:
        raise RuntimeError(
            f"settings.py changed since this run (saved {saved[:12]}…, live {live[:12]}…). "
            "Re-run train_m.py or revert settings.py."
        )


def resolve_run_dir(outputs_dir: Path, run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(p for p in Path(outputs_dir).iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories in {outputs_dir}")
    return candidates[-1]


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the token dimension."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def combine_text_embeddings(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concat, difference, and Hadamard product of an embedding pair."""
    return torch.cat([a, b, a - b, a * b], dim=-1)


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """Pick torch device for inference.

    gpu: None -> cuda:0 if available else cpu; int -> cuda:N; "cpu" -> cpu.
    """
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def format_results(m: dict[str, float]) -> str:
    return f"loss={m['loss']:.4f}  " + "  ".join(
        f"{col}: MAE={m[f'mae_{col}']:.3f} R2={m[f'r2_{col}']:.3f}"
        for col in M_TARGET_COLS
    )


def normalize_target_columns(
    y: pd.DataFrame,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    y = y.copy()
    for col in M_TARGET_COLS:
        if bounds is None:
            col_min, col_max = np.nanmin(y[col]), np.nanmax(y[col])
        else:
            col_min, col_max = bounds[col]
        y[col] = (y[col] - col_min) / (col_max - col_min + _EPS)
    return y


def unnormalize_target_columns(
    y: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    y = y.copy()
    for col in M_TARGET_COLS:
        col_min, col_max = bounds[col]
        y[col] = y[col] * (col_max - col_min + _EPS) + col_min
    return y


def scalarize(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    *,
    already_normalized: bool = False,
) -> np.ndarray:
    """Combine PSNR and CLIP into scalar m using dataset min-max bounds."""
    if already_normalized:
        return (np.asarray(psnr, dtype=float) + np.asarray(clip, dtype=float)) / 2.0
    psnr_n = (np.asarray(psnr, dtype=float) - bounds[PSNR_COL][0]) / (
        bounds[PSNR_COL][1] - bounds[PSNR_COL][0] + _EPS
    )
    clip_n = (np.asarray(clip, dtype=float) - bounds[CLIP_COL][0]) / (
        bounds[CLIP_COL][1] - bounds[CLIP_COL][0] + _EPS
    )
    return (psnr_n + clip_n) / 2.0


def add_combined_score(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
    *,
    psnr_col: str = PSNR_COL,
    clip_col: str = CLIP_COL,
    out_col: str = T_TARGET_COL,
    already_normalized: bool = False,
) -> pd.Series:
    score = scalarize(
        df[psnr_col].to_numpy(dtype=float),
        df[clip_col].to_numpy(dtype=float),
        bounds,
        already_normalized=already_normalized,
    )
    return pd.Series(score, index=df.index, name=out_col)


def prep_sample_id(value) -> str:
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