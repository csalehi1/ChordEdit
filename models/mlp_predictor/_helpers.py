# _helpers.py

"""
Run settings, model tensors, and training/score helpers.

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

from scores import calc_norm_deltas

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
    already = [n for n in ("_data", "embeddings", "model", "_wandb") if n in sys.modules]
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


def resolve_device(gpu: int | str | None = None) -> torch.device:
    """Resolve the device to use for the model."""
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


def calc_phi(deltas: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    """settings.SCORE_PHI on normalized deltas."""
    if weights is None:
        return _s().SCORE_PHI(deltas)
    return _s().SCORE_PHI(deltas, weights=weights)


def delta_metric_grids(
    psnr: np.ndarray,
    clip: np.ndarray,
    baseline_idx: int | np.ndarray,
) -> np.ndarray:
    """Per-sample normalized deltas of a stack of (PSNR, CLIP) timestep grids."""
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
    deltas = calc_norm_deltas(values, idx)
    return deltas.detach().cpu().numpy().reshape(b, n1, n2, 2)


def phi_from_delta_grids(
    delta_grids: np.ndarray,
    weights: np.ndarray | tuple[float, float] | None = None,
) -> np.ndarray:
    """settings.SCORE_PHI over (n_samples, n_start, n_end, 2) delta grids."""
    b, n1, n2, c = delta_grids.shape
    deltas = torch.as_tensor(delta_grids.reshape(b, n1 * n2, c), dtype=torch.float64)
    w = None if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    return calc_phi(deltas, weights=w).detach().cpu().numpy().reshape(b, n1, n2)


def score_metric_grids(
    psnr: np.ndarray,
    clip: np.ndarray,
    baseline_idx: int | np.ndarray,
    weights: np.ndarray | tuple[float, float] | None = None,
) -> np.ndarray:
    """Score a stack of (PSNR, CLIP) timestep grids into phi via per-sample deltas.

    See delta_metric_grids for the grid contract. Inputs are raw metric values;
    grids that are already deltas must go through phi_from_delta_grids instead,
    since the min-max normalization here would rescale them a second time.

    Returns: (n_samples, n_start, n_end)
    """
    return phi_from_delta_grids(delta_metric_grids(psnr, clip, baseline_idx), weights=weights)


def format_metric_table(
    rows: list[tuple[str, dict[str, float], dict[str, float] | None]],
) -> str:
    """Aligned split table: loss, per-target MAE/R2, and optional phi/regret."""

    def _mae_r2(m: dict[str, float], col: str) -> str:
        return f"MAE {m[f'mae_{col}']:.3f} R2 {m[f'r2_{col}']:.3f}"

    def _sel(sel: dict[str, float] | None, key: str, fmt: str) -> str:
        if not sel or key not in sel:
            return ""
        return format(sel[key], fmt)

    target_cols = list(_s().TARGET_COLS)
    split_w = max(5, *(len(name) for name, _, _ in rows))
    # Prefer selection's train-objective loss when present (includes ranking).
    def _loss(m: dict[str, float], sel: dict[str, float] | None) -> float:
        if sel and "loss" in sel:
            return float(sel["loss"])
        return float(m["loss"])

    loss_w = max(4, *(len(f"{_loss(m, sel):.4f}") for _, m, sel in rows))
    target_ws = [
        max(len(col), *(len(_mae_r2(m, col)) for _, m, _ in rows))
        for col in target_cols
    ]
    # Selection columns, appended after the per-target block. A key missing from
    # the selection dict renders blank rather than raising.
    sel_cols = (
        ("phi rho", "phi_spearman"),
        ("regret", "regret_median"),
        ("gain", "gain_mean"),
        ("top1", "top1_accuracy"),
    )
    sel_vals = [[_sel(sel, key, ".3f") for _, _, sel in rows] for _, key in sel_cols]
    sel_ws = [max(len(head), *(len(v) for v in vals)) for (head, _), vals in zip(sel_cols, sel_vals)]

    header = (
        f"{'split':<{split_w}}  {'loss':<{loss_w}}  "
        + "  ".join(f"{col:<{w}}" for col, w in zip(target_cols, target_ws))
        + "  "
        + "  ".join(f"{head:<{w}}" for (head, _), w in zip(sel_cols, sel_ws))
    )
    lines = [f"    {header}"]
    for i, (name, metrics, sel) in enumerate(rows):
        cells = "  ".join(
            f"{_mae_r2(metrics, col):<{w}}" for col, w in zip(target_cols, target_ws)
        )
        sel_cells = "  ".join(f"{vals[i]:<{w}}" for vals, w in zip(sel_vals, sel_ws))
        lines.append(
            f"    {name:<{split_w}}  {_loss(metrics, sel):<{loss_w}.4f}  {cells}  {sel_cells}"
        )
    return "\n".join(lines)


"""
Grid indexing and the mean surface.
"""

MEAN_SURFACE_NAME = "mean_surface.pt"


def nearest_indices(values, targets) -> np.ndarray:
    """Index in `values` nearest each entry of `targets`, vectorized."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1)
    return np.abs(values[None, :] - targets[:, None]).argmin(axis=1)


def calc_mean_surface(cells) -> dict:
    """Mean true delta surface of one split's grids, keyed by the grid axes."""
    s = _s()
    mean_true = cells.y[cells.grid_rows].double().mean(dim=0).cpu()  # (n_cells, C)

    t_pairs = cells.t[cells.grid_rows[0]].detach().cpu().numpy()  # (n_cells, 2)
    t_start_values = np.sort(np.unique(t_pairs[:, 0]))
    t_end_values = np.sort(np.unique(t_pairs[:, 1]))
    i = nearest_indices(t_start_values, t_pairs[:, 0])
    j = nearest_indices(t_end_values, t_pairs[:, 1])
    grid = torch.full(
        (len(t_start_values), len(t_end_values), mean_true.shape[1]), float("nan"), dtype=torch.float64,
    )
    grid[i, j] = mean_true
    return {
        "t_start_values": torch.as_tensor(t_start_values, dtype=torch.float64),
        "t_end_values": torch.as_tensor(t_end_values, dtype=torch.float64),
        "mean_true_delta": grid,
        "prediction_space": str(s.PREDICTION_SPACE),
        "split": "train",
        "n_samples": cells.n_grids,
        "target_cols": list(s.TARGET_COLS),
    }


def mean_surface_from_dict(
    surface: dict,
    t_start_values: np.ndarray,
    t_end_values: np.ndarray,
) -> np.ndarray:
    """mean_surface dict -> (n_start, n_end, 2) mean target grid.

    Raises when the mean surface's grid axes do not match the selector's, so a
    stale mean surface can never be silently applied to the wrong grid.
    """
    cal_start = np.asarray(surface["t_start_values"], dtype=np.float64)
    cal_end = np.asarray(surface["t_end_values"], dtype=np.float64)
    if not (
        len(cal_start) == len(t_start_values)
        and len(cal_end) == len(t_end_values)
        and np.allclose(cal_start, np.asarray(t_start_values, dtype=np.float64))
        and np.allclose(cal_end, np.asarray(t_end_values, dtype=np.float64))
    ):
        raise ValueError(
            f"Expected ({cal_start.tolist()}, {cal_end.tolist()}) == "
            f"({np.asarray(t_start_values).tolist()}, {np.asarray(t_end_values).tolist()})"
        )
    return np.asarray(surface["mean_true_delta"], dtype=np.float64)


def gather_at_pairs(surface: dict, t_pairs: torch.Tensor) -> torch.Tensor:
    """The surface's mean true delta gathered at (N, 2) cell pairs, flat (N, C)."""
    pairs = t_pairs.detach().cpu().numpy()
    i = nearest_indices(surface["t_start_values"], pairs[:, 0])
    j = nearest_indices(surface["t_end_values"], pairs[:, 1])
    return surface["mean_true_delta"][i, j]


def save_mean_surface(run_dir: Path, surface: dict) -> Path:
    out = Path(run_dir) / MEAN_SURFACE_NAME
    torch.save(surface, out)
    return out


def load_mean_surface(run_dir: Path) -> dict | None:
    """The run's saved mean surface, or None when the run predates it."""
    path = Path(run_dir) / MEAN_SURFACE_NAME
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)
