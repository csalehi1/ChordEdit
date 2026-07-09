"""
Evaluate and calibrate timestep selector T on the test split.

T has no trainable weights; this script loads a trained M checkpoint, runs T
selection on held-out samples with full grids, and reports selection metrics
(regret, Spearman, deviate-gate precision/recall).

Run after m_train.py:

    python t_train.py
    python t_train.py --run-dir outputs/sdturbo_random10/20260101_120000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from data_io import df_to_metric_grids, precompute_embeddings
from _helpers import CombinedScoreBounds, combined_score_bounds_from_df, resolve_device
from model_m import MetricPredictor
from settings import (
    CLIP_COL,
    DEFAULT_T_END,
    DEFAULT_T_START,
    NOISE_FLOOR_M,
    OUTPUTS_DIR,
    PSNR_COL,
)
from model_t import TimestepPredictor, scalarize


def _resolve_run_dir(run_dir: Path | None) -> Path:
    """Use explicit run dir or default to the latest m_train.py output."""
    if run_dir is not None:
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    candidates = sorted(d for d in OUTPUTS_DIR.iterdir() if d.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No run directories in {OUTPUTS_DIR}")
    return candidates[-1]


def _bounds_from_ckpt(ckpt: dict, test_df: pd.DataFrame) -> CombinedScoreBounds:
    if "combined_score_bounds" in ckpt:
        pmn, pmx, cmn, cmx = ckpt["combined_score_bounds"]
        return CombinedScoreBounds(float(pmn), float(pmx), float(cmn), float(cmx))
    return combined_score_bounds_from_df(test_df)


def _per_image_spearman(true_grid: np.ndarray, pred_grid: np.ndarray) -> np.ndarray:
    # Rank correlation within each image's 11x11 grid.
    rhos = []
    for k in range(true_grid.shape[0]):
        rho, _ = spearmanr(true_grid[k].ravel(), pred_grid[k].ravel())
        rhos.append(np.nan if rho is None else float(rho))
    return np.array(rhos)


def _regret(true_m: np.ndarray, pred_m: np.ndarray) -> np.ndarray:
    # True m at argmax(pred_m) minus true m at argmax(true_m); lower is better.
    out = np.zeros(true_m.shape[0])
    for k in range(true_m.shape[0]):
        chosen = np.unravel_index(pred_m[k].argmax(), pred_m[k].shape)
        out[k] = true_m[k].max() - true_m[k][chosen]
    return out


def _gate_metrics(true_m: np.ndarray, pred_m: np.ndarray, default_i: int, default_j: int, nf: float):
    # Deviate-or-default gate: precision/recall for flagged improvable images.
    default_m = true_m[:, default_i, default_j]
    truly_improvable = (true_m.max(axis=(1, 2)) - default_m) > nf
    pred_gain = pred_m.max(axis=(1, 2)) - pred_m[:, default_i, default_j]
    flagged = pred_gain > nf
    tp = int(np.sum(flagged & truly_improvable))
    precision = tp / max(int(flagged.sum()), 1)
    recall = tp / max(int(truly_improvable.sum()), 1)
    return {
        "precision": precision,
        "recall": recall,
        "n_flagged": int(flagged.sum()),
        "n_improvable": int(truly_improvable.sum()),
        "noise_floor": nf,
    }


def train(
    run_dir: Path | None = None,
    noise_floor: float | None = None,
    gpu: int | str | None = None,
) -> dict:
    # Locate the M checkpoint and matching test split from m_train.py.
    run_dir = _resolve_run_dir(run_dir)
    weights_path = run_dir / "regressor_weights.pt"
    test_path = run_dir / "test.parquet.gz"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing M weights: {weights_path}")
    if not test_path.exists():
        raise FileNotFoundError(
            f"Missing test split at {test_path}; run m_train.py first."
        )

    test_df = pd.read_parquet(test_path)
    sample_ids = sorted(test_df["sample_id"].unique())
    t_start_values = sorted(test_df["t_start"].unique())
    t_end_values = sorted(test_df["t_end"].unique())
    n1, n2 = len(t_start_values), len(t_end_values)
    # Each test image must have a complete grid for selection eval.
    for sid, n in test_df.groupby("sample_id").size().items():
        if n != n1 * n2:
            raise ValueError(f"sample {sid} has {n} cells, expected {n1 * n2}")

    device = resolve_device(gpu)
    print(f"device: {device}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = MetricPredictor(freeze_encoders=True, device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_stats(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # Build T wrapper around the loaded M checkpoint.
    bounds = _bounds_from_ckpt(ckpt, test_df)
    t_predictor = TimestepPredictor(
        model,
        t_start_values=t_start_values,
        t_end_values=t_end_values,
        scalar_stats=bounds,
    )
    default_i = t_predictor._default_i
    default_j = t_predictor._default_j
    nf = NOISE_FLOOR_M if noise_floor is None else noise_floor

    emb = precompute_embeddings(test_df.drop_duplicates("sample_id"), model, device)
    true_psnr, _ = df_to_metric_grids(test_df, sample_ids, t_start_values, t_end_values, PSNR_COL)
    true_clip, _ = df_to_metric_grids(test_df, sample_ids, t_start_values, t_end_values, CLIP_COL)
    true_m = scalarize(true_psnr, true_clip, bounds)

    # Batched grid forward + gated selection per test image.
    pred_psnr = np.zeros_like(true_psnr)
    pred_clip = np.zeros_like(true_clip)
    pred_m = np.zeros_like(true_m)
    selections = []
    for sid in sample_ids:
        k = sample_ids.index(sid)
        e = emb[sid]
        grid = t_predictor.predict_grid_from_emb(e["img"], e["src"], e["tar"])
        pred_psnr[k], pred_clip[k], pred_m[k] = grid.psnr_grid, grid.clip_grid, grid.m_grid
        sel = t_predictor.select_from_grid(grid, noise_floor=nf)
        selections.append(
            {
                "sample_id": sid,
                "t_start": sel.t_start,
                "t_end": sel.t_end,
                "deviate": sel.deviate,
                "pred_gain": sel.pred_gain,
            }
        )

    reg = _regret(true_m, pred_m)
    rho_m = _per_image_spearman(true_m, pred_m)
    gate = _gate_metrics(true_m, pred_m, default_i, default_j, nf)

    metrics = {
        "run_dir": str(run_dir),
        "n_test_images": len(sample_ids),
        "grid": f"{n1}x{n2}",
        "regret_median": float(np.median(reg)),
        "regret_p90": float(np.percentile(reg, 90)),
        "spearman_m_median": float(np.nanmedian(rho_m)),
        "gate": gate,
        "combined_score_bounds": {
            "psnr_min": bounds.psnr_min,
            "psnr_max": bounds.psnr_max,
            "clip_min": bounds.clip_min,
            "clip_max": bounds.clip_max,
        },
        "default_t_start": DEFAULT_T_START,
        "default_t_end": DEFAULT_T_END,
    }

    out_metrics = run_dir / "t_train_metrics.json"
    out_selections = run_dir / "t_test_selections.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_selections, "w") as f:
        json.dump(selections, f, indent=2)

    print(f"T eval on {len(sample_ids)} test images  run={run_dir.name}")
    print(f"  regret median={metrics['regret_median']:.4f}  p90={metrics['regret_p90']:.4f}")
    print(f"  spearman m median={metrics['spearman_m_median']:.3f}")
    print(
        f"  gate precision={gate['precision']:.3f}  recall={gate['recall']:.3f}  "
        f"({gate['n_flagged']} flagged)"
    )
    print(f"Saved {out_metrics}")
    print(f"Saved {out_selections}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate timestep selector T on test split")
    parser.add_argument("--run-dir", type=Path, default=None, help="M training run directory")
    parser.add_argument("--noise-floor", type=float, default=None, help="Deviate gate noise floor")
    parser.add_argument(
        "--gpu",
        default=None,
        help='CUDA device index (e.g. 0, 1) or "cpu"; default cuda:0 if available',
    )
    args = parser.parse_args()
    gpu = args.gpu
    if gpu is not None and str(gpu).lower() != "cpu":
        gpu = int(gpu)
    train(run_dir=args.run_dir, noise_floor=args.noise_floor, gpu=gpu)


if __name__ == "__main__":
    main()
