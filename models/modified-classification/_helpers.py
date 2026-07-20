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
    """Fixed-width metric line so Train/Val columns stay aligned (incl. signed R²)."""
    return f"loss={m['loss']:7.4f}  " + "  ".join(
        f"{col}: MAE={m[f'mae_{col}']:6.3f} R2={m[f'r2_{col}']:7.3f}"
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


def unnormalize_metric_arrays(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    psnr_min, psnr_max = bounds[PSNR_COL]
    clip_min, clip_max = bounds[CLIP_COL]
    return (
        psnr * (psnr_max - psnr_min + _EPS) + psnr_min,
        clip * (clip_max - clip_min + _EPS) + clip_min,
    )


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


def combined_score_tensor(targets: torch.Tensor) -> torch.Tensor:
    """Scalar m from min-max normalized (psnr, clip), shape (N, 2) -> (N,)."""
    return (targets[:, 0] + targets[:, 1]) / 2.0


def t_target_scores(targets: torch.Tensor) -> torch.Tensor:
    """Apply T_TARGET_FUNC to (N, 2) min-max [psnr, clip] columns; shape (N,).

    Detaches from the autograd graph (T_TARGET_FUNC is pandas/numpy). Use this
    for true pairwise ranking preferences.
    """
    df = pd.DataFrame(
        {
            PSNR_COL: targets[:, 0].detach().cpu().numpy(),
            CLIP_COL: targets[:, 1].detach().cpu().numpy(),
        }
    )
    scores = np.asarray(T_TARGET_FUNC(df), dtype=np.float64).reshape(-1)
    return torch.as_tensor(scores, device=targets.device, dtype=targets.dtype)


def t_target_scores_torch(targets: torch.Tensor) -> torch.Tensor:
    """Differentiable stand-in for the default T_TARGET_FUNC (weighted PSNR/CLIP).

    Batch min-max then equal-weight blend — matches
    compute_weighted_combined_score(..., normalize=True) with default lambdas.
    """
    psnr, clip = targets[:, 0], targets[:, 1]
    psnr = (psnr - psnr.amin()) / (psnr.amax() - psnr.amin() + _EPS)
    clip = (clip - clip.amin()) / (clip.amax() - clip.amin() + _EPS)
    return 0.5 * (psnr + clip)


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Logistic pairwise loss: penalize pred ordering that disagrees with true."""
    if pred.shape[0] < 2:
        return pred.new_zeros(())
    diff_true = true.unsqueeze(1) - true.unsqueeze(0)
    diff_pred = pred.unsqueeze(1) - pred.unsqueeze(0)
    mask = diff_true > 0
    if not mask.any():
        return pred.new_zeros(())
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


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


def resolve_embedding_path(embedding_path: str) -> str:
    """Resolve a path from id_to_embeddings_*.csv to an absolute .pt path.

    Absolute CSV paths are returned as-is. Relative paths are under
    EMBEDDINGS_DIR/annotation_embeddings/ (matching grid_generate layout),
    whether written as `{id}/source.pt` or `annotation_embeddings/{id}/source.pt`.
    """
    path = Path(embedding_path)
    if path.is_absolute():
        return str(path)
    path = Path(str(embedding_path).lstrip("/"))
    if path.parts and path.parts[0] == EMBEDDINGS_SAMPLES_DIRNAME:
        return str(EMBEDDINGS_DIR / path)
    return str(EMBEDDINGS_DIR / EMBEDDINGS_SAMPLES_DIRNAME / path)