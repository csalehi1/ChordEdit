"""Run settings, model tensors, and training/score helpers.

Call load_run_settings(RUN_DIR) before importing _data / models so those modules
bind constants from the per-run settings snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SETTINGS_FILENAME = "settings.json"
_PACKAGE_DIR = Path(__file__).resolve().parent
_SETTINGS_MODULE = _PACKAGE_DIR / "settings.py"


def _s():
    """Current settings module (live or per-run snapshot in sys.modules)."""
    import settings
    return settings


def current_commit_id() -> str:
    """Short id of the most recent commit, or "nogit" outside a repository."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_PACKAGE_DIR), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "nogit"
    except Exception:
        return "nogit"


def save_run_settings(run_dir: Path) -> Path:
    """Save the config this run used, so the run can be replayed exactly."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / SETTINGS_FILENAME
    dst.write_text(json.dumps(_s().CONFIG, indent=2, sort_keys=True) + "\n")
    return dst


def load_run_settings(run_dir: Path):
    """Bind the settings module to a run's saved config.

    Imports settings fresh under that name with the run's settings.json pinned,
    so evaluation sees exactly the training config regardless of any
    --settings-path on the current command line.
    """
    run_dir = Path(run_dir)
    already = [n for n in ("_data", "embeddings", "model_m", "model_t") if n in sys.modules]
    if already:
        raise RuntimeError(
            f"load_run_settings() must be called before importing modules that bind settings "
            f"(already imported: {', '.join(already)})."
        )

    path = run_dir / SETTINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run train_m.py to save the config for this run."
        )
    spec = importlib.util.spec_from_file_location("settings", _SETTINGS_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load settings from {_SETTINGS_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    # Seeded before the module body runs, where settings.py reads it in
    # preference to --settings-path (see settings.SETTINGS_JSON).
    mod.SETTINGS_PATH_OVERRIDE = str(path)
    sys.modules["settings"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_live_settings():
    """Load package settings.py under a private module name (for OUTPUTS_DIR bootstrap)."""
    spec = importlib.util.spec_from_file_location("_live_settings", _SETTINGS_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load live settings from {_SETTINGS_MODULE}")
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


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """None -> cuda:0 if available else cpu; int -> cuda:N; \"cpu\" -> cpu."""
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def grid_axes_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique (t_start, t_end) grid axes."""
    s = _s()
    return (
        np.sort(np.asarray(df[s.T_START_COL].unique())),
        np.sort(np.asarray(df[s.T_END_COL].unique())),
    )


def timestep_pairs_from_df(df: pd.DataFrame) -> np.ndarray:
    """Unique (t_start, t_end) pairs in df, shape (N, 2), sorted."""
    s = _s()
    return (
        df.loc[:, [s.T_START_COL, s.T_END_COL]]
        .drop_duplicates()
        .sort_values([s.T_START_COL, s.T_END_COL])
        .to_numpy(dtype=np.float64)
    )


def t_target_phi_values(
    values: torch.Tensor,
    baseline_idx: int | torch.Tensor,
) -> torch.Tensor:
    """Per-sample normalized deltas then settings.T_TARGET_PHI (phi on Delta).

    values: (..., N, C). Returns scores shaped (..., N).
    """
    from scores import calc_normalized_deltas
    deltas = calc_normalized_deltas(values, baseline_idx)
    return _s().T_TARGET_PHI(deltas)


def score_metric_grids(
    psnr: np.ndarray,
    clip: np.ndarray,
    baseline_idx: int | np.ndarray,
) -> np.ndarray:
    """Score a stack of (PSNR, CLIP) timestep grids into phi via per-sample deltas.

    psnr, clip: (n_samples, n_start, n_end) - always a stack, one grid per
    sample, each normalized over its own grid. Pass a single sample's grid as
    grid[None] so the sample axis is never ambiguous.
    baseline_idx: flat index of the default cell, i * n_end + j.

    Raw metric units are fine: the per-sample range normalization inside
    calc_normalized_deltas is invariant to any global affine rescaling, so no
    prior min-max is needed. NaN (unlabeled) cells stay NaN without affecting
    labeled cells; the baseline cell itself must be labeled.

    Returns: (n_samples, n_start, n_end)
    """
    psnr = np.asarray(psnr, dtype=float)
    clip = np.asarray(clip, dtype=float)
    if psnr.shape != clip.shape:
        raise ValueError(f"psnr/clip shape mismatch: {psnr.shape} vs {clip.shape}")
    if psnr.ndim != 3:
        raise ValueError(
            f"expected (n_samples, n_start, n_end), got {psnr.shape}; "
            f"pass grid[None] for a single sample"
        )

    b, n1, n2 = psnr.shape
    values = torch.as_tensor(
        np.stack([psnr.reshape(b, n1 * n2), clip.reshape(b, n1 * n2)], axis=-1),
        dtype=torch.float64,
    )
    idx = baseline_idx if isinstance(baseline_idx, int) else torch.as_tensor(baseline_idx, dtype=torch.long)
    out = t_target_phi_values(values, idx)
    return out.detach().cpu().numpy().reshape(b, n1, n2)
