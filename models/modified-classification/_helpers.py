"""Shared helpers for model M training and inference.

Run-directory helpers (save/load settings, resolve run dir) do not bind settings
at import time. Call load_run_settings(RUN_DIR) before importing _data / models
(or using other helpers here) so those bind constants from the per-run snapshot.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SETTINGS_FILENAME = "settings.py"
_PACKAGE_DIR = Path(__file__).resolve().parent
_LIVE_SETTINGS = _PACKAGE_DIR / SETTINGS_FILENAME

_EPS = 1e-8


def _s():
    """Current settings module (live or per-run snapshot in sys.modules)."""
    import settings

    return settings


def save_run_settings(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / SETTINGS_FILENAME
    dst.write_bytes(_LIVE_SETTINGS.read_bytes())
    return dst


def load_run_settings(run_dir: Path):
    path = Path(run_dir) / SETTINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Re-run train_m.py to snapshot settings.py for this run.")
    already = [n for n in ("_data", "model_m", "model_t") if n in sys.modules]
    if already:
        raise RuntimeError(
            f"load_run_settings() must be called before importing modules that bind settings "
            f"(already imported: {', '.join(already)})."
        )
    spec = importlib.util.spec_from_file_location("settings", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load settings from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["settings"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_live_settings():
    """Load package settings.py under a private module name (for OUTPUTS_DIR bootstrap)."""
    spec = importlib.util.spec_from_file_location("_live_settings", _LIVE_SETTINGS)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load live settings from {_LIVE_SETTINGS}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        for col in _s().M_TARGET_COLS
    )


def normalize_target_columns(
    y: pd.DataFrame,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    y = y.copy()
    for col in _s().M_TARGET_COLS:
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
    for col in _s().M_TARGET_COLS:
        col_min, col_max = bounds[col]
        y[col] = y[col] * (col_max - col_min + _EPS) + col_min
    return y


def unnormalize_metric_arrays(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    s = _s()
    psnr_min, psnr_max = bounds[s.PSNR_COL]
    clip_min, clip_max = bounds[s.CLIP_COL]
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
    s = _s()
    psnr_n = (np.asarray(psnr, dtype=float) - bounds[s.PSNR_COL][0]) / (
        bounds[s.PSNR_COL][1] - bounds[s.PSNR_COL][0] + _EPS
    )
    clip_n = (np.asarray(clip, dtype=float) - bounds[s.CLIP_COL][0]) / (
        bounds[s.CLIP_COL][1] - bounds[s.CLIP_COL][0] + _EPS
    )
    return (psnr_n + clip_n) / 2.0


def combined_score_tensor(targets: torch.Tensor) -> torch.Tensor:
    """Scalar m from min-max normalized (psnr, clip), shape (N, 2) -> (N,)."""
    return (targets[:, 0] + targets[:, 1]) / 2.0


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
    psnr_col: str | None = None,
    clip_col: str | None = None,
    out_col: str | None = None,
    already_normalized: bool = False,
) -> pd.Series:
    s = _s()
    if psnr_col is None:
        psnr_col = s.PSNR_COL
    if clip_col is None:
        clip_col = s.CLIP_COL
    if out_col is None:
        out_col = s.T_TARGET_COL
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
    return str(_s().GENERATED_DIR / cell_path.lstrip("/"))


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    return str(_s().DATASET_DIR / image_path)


def resolve_mask_path(mask_path: str) -> str:
    path = Path(mask_path)
    if path.is_absolute():
        return str(path)
    return str(_s().DATASET_DIR / mask_path)


def resolve_embedding_path(embedding_path: str) -> str:
    """Resolve a path from id_to_embeddings_*.csv to an absolute .pt path.

    Absolute CSV paths are returned as-is. Relative paths are under
    EMBEDDINGS_DIR/annotation_embeddings/ (matching grid_generate layout),
    whether written as `{id}/source.pt` or `annotation_embeddings/{id}/source.pt`.
    """
    s = _s()
    path = Path(embedding_path)
    if path.is_absolute():
        return str(path)
    path = Path(str(embedding_path).lstrip("/"))
    if path.parts and path.parts[0] == s.EMBEDDINGS_SAMPLES_DIRNAME:
        return str(s.EMBEDDINGS_DIR / path)
    return str(s.EMBEDDINGS_DIR / s.EMBEDDINGS_SAMPLES_DIRNAME / path)
