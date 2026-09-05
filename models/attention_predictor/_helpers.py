# _helpers.py

"""
Run settings, model tensors, and training/score helpers.

Call load_run_settings(RUN_DIR) before importing _data / model so those modules
bind constants from the per-run settings snapshot.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
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
    already = [n for n in ("dataloader", "dataset", "embeddings", "model", "_wandb") if n in sys.modules]
    if already:
        raise RuntimeError(
            f"load_run_settings() must be called before importing modules that bind settings "
            f"(already imported: {', '.join(already)})."
        )

    path = run_dir / SETTINGS_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Re-run train.py to save the config for this run."
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
    """Load package settings.py under a private module name (for RUNS_DIR bootstrap)."""
    spec = importlib.util.spec_from_file_location("_live_settings", _SETTINGS_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load live settings from {_SETTINGS_MODULE}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def resolve_run_dir(runs_dir: Path, run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(p for p in Path(runs_dir).iterdir() if p.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories in {runs_dir}")
    return candidates[-1]


def prep_sample_id(value) -> str:
    """Zero-pad a raw id to the 8-digit sample_id used everywhere."""
    return f"{int(value):08d}"


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """Resolve the device to use for the model."""
    if gpu is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(gpu, str) and gpu.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{int(gpu)}")


def calc_phi(deltas: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """settings.SCORE_PHI on normalized deltas."""
    if weights is None:
        return _s().SCORE_PHI(deltas)
    return _s().SCORE_PHI(deltas, weights=weights)


def phi_from_delta_grids(
    delta_grids: np.ndarray,
    weights: np.ndarray | tuple[float, float] | None = None,
) -> np.ndarray:
    """settings.SCORE_PHI over (n_samples, n_start, n_end, 2) delta grids."""
    b, n1, n2, c = delta_grids.shape
    deltas = torch.as_tensor(delta_grids.reshape(b, n1 * n2, c), dtype=torch.float64)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    return calc_phi(deltas, weights=w).detach().cpu().numpy().reshape(b, n1, n2)


def format_metric_table(
    rows: list[tuple[str, dict[str, float]]],
) -> str:
    """Aligned split table: loss, per-target MAE/R2, and optional phi/regret."""

    def _mae_r2(m: dict[str, float], col: str) -> str:
        return f"MAE {m[f'mae_{col}']:.3f} R2 {m[f'r2_{col}']:.3f}"

    def _sel(m: dict[str, float], key: str, fmt: str) -> str:
        if key not in m:
            return ""
        return format(m[key], fmt)

    target_cols = list(_s().TARGET_COLS)
    split_w = max(5, *(len(name) for name, _ in rows))
    loss_w = max(4, *(len(f"{m['loss']:.4f}") for _, m in rows))
    target_ws = [
        max(len(col), *(len(_mae_r2(m, col)) for _, m in rows))
        for col in target_cols
    ]
    # Selection columns, appended after the per-target block. A key missing from
    # the metrics dict renders blank rather than raising.
    sel_cols = (
        ("phi rho", "phi_spearman"),
        ("regret", "regret_median"),
        ("gain", "gain_mean"),
        ("top1", "top1_accuracy"),
    )
    sel_vals = [[_sel(m, key, ".3f") for _, m in rows] for _, key in sel_cols]
    sel_ws = [max(len(head), *(len(v) for v in vals)) for (head, _), vals in zip(sel_cols, sel_vals)]

    header = (
        f"{'split':<{split_w}}  {'loss':<{loss_w}}  "
        + "  ".join(f"{col:<{w}}" for col, w in zip(target_cols, target_ws))
        + "  "
        + "  ".join(f"{head:<{w}}" for (head, _), w in zip(sel_cols, sel_ws))
    )
    lines = [f"    {header}"]
    for i, (name, metrics) in enumerate(rows):
        cells = "  ".join(
            f"{_mae_r2(metrics, col):<{w}}" for col, w in zip(target_cols, target_ws)
        )
        sel_cells = "  ".join(f"{vals[i]:<{w}}" for vals, w in zip(sel_vals, sel_ws))
        lines.append(
            f"    {name:<{split_w}}  {metrics['loss']:<{loss_w}.4f}  {cells}  {sel_cells}"
        )
    return "\n".join(lines)


def nearest_indices(values, targets) -> np.ndarray:
    """Index in `values` nearest each entry of `targets`, vectorized."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    return np.abs(values[None, :] - targets[:, None]).argmin(axis=1)


def get_default_cell(
    cell_t_pairs: np.ndarray,
    t_start_values: np.ndarray | None = None,
    t_end_values: np.ndarray | None = None,
) -> int:
    """Position of the default cell in the model's output cell order."""
    if isinstance(cell_t_pairs, torch.Tensor):
        cell_t_pairs = cell_t_pairs.detach().cpu().numpy()
    cell_t_pairs = np.asarray(cell_t_pairs, dtype=np.float64)
    if t_start_values is None:
        t_start_values = np.sort(np.unique(cell_t_pairs[:, 0]))
    if t_end_values is None:
        t_end_values = np.sort(np.unique(cell_t_pairs[:, 1]))
    cell_i = nearest_indices(t_start_values, cell_t_pairs[:, 0])
    cell_j = nearest_indices(t_end_values, cell_t_pairs[:, 1])
    default_i = int(nearest_indices(t_start_values, [_s().DEFAULT_T_START])[0])
    default_j = int(nearest_indices(t_end_values, [_s().DEFAULT_T_END])[0])
    found = np.flatnonzero((cell_i == default_i) & (cell_j == default_j))
    if found.size != 1:
        raise ValueError(f"Expected {found.size=} == 1")
    return int(found[0])
