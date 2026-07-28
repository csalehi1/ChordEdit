"""Run settings, model tensors, and training/score helpers.

Call load_run_settings(RUN_DIR) before importing _data / models so those modules
bind constants from the per-run settings snapshot.
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


# --- Run settings -----------------------------------------------------------------

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


# --- Model tensors ----------------------------------------------------------------

def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the token dimension."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


# --- Device / reporting / loss ----------------------------------------------------

def resolve_device(gpu: int | str | None = None) -> torch.device:
    """None -> cuda:0 if available else cpu; int -> cuda:N; \"cpu\" -> cpu."""
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


# --- Targets + T score ------------------------------------------------------------

def normalize_target_columns(
    y_df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    y_df = y_df.copy()
    for col in _s().M_TARGET_COLS:
        if bounds is None:
            col_min, col_max = np.nanmin(y_df[col]), np.nanmax(y_df[col])
        else:
            col_min, col_max = bounds[col]
        y_df[col] = (y_df[col] - col_min) / (col_max - col_min + _EPS)
    return y_df


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


def t_target_score_values(
    values: torch.Tensor,
    baseline_idx: int | torch.Tensor,
) -> torch.Tensor:
    """Per-sample normalized deltas then settings.T_TARGET_SCORE (phi on Delta).

    values: (..., N, C). Returns scores shaped (..., N).
    """
    from scores import calc_normalized_deltas
    deltas = calc_normalized_deltas(values, baseline_idx)
    return _s().T_TARGET_SCORE(deltas)


def scalarize(
    psnr: np.ndarray,
    clip: np.ndarray,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    already_normalized: bool = False,
    baseline_idx: int,
) -> np.ndarray:
    """Combine PSNR and CLIP via apply_t_score (preserves input shape)."""
    psnr = np.asarray(psnr, dtype=float)
    clip = np.asarray(clip, dtype=float)
    if psnr.shape != clip.shape:
        raise ValueError(f"psnr/clip shape mismatch: {psnr.shape} vs {clip.shape}")
    if not already_normalized and bounds is not None:
        s = _s()
        psnr = (psnr - bounds[s.PSNR_COL][0]) / (bounds[s.PSNR_COL][1] - bounds[s.PSNR_COL][0] + _EPS)
        clip = (clip - bounds[s.CLIP_COL][0]) / (bounds[s.CLIP_COL][1] - bounds[s.CLIP_COL][0] + _EPS)
    if psnr.ndim == 3:
        b, n1, n2 = psnr.shape
        values = torch.as_tensor(
            np.stack([psnr.reshape(b, n1 * n2), clip.reshape(b, n1 * n2)], axis=-1),
            dtype=torch.float64,
        )
        out = t_target_score_values(values, baseline_idx=baseline_idx)
        return out.detach().cpu().numpy().reshape(b, n1, n2)
    if psnr.ndim != 2:
        raise ValueError(f"psnr/clip must be 2D or 3D, got shape {psnr.shape}")
    values = torch.as_tensor(np.stack([psnr.ravel(), clip.ravel()], axis=-1), dtype=torch.float64)
    out = t_target_score_values(values, baseline_idx=baseline_idx)
    return out.detach().cpu().numpy().reshape(psnr.shape)
