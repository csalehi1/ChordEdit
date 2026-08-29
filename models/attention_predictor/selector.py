# selector.py

"""
Evaluate timestep selector.
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

from inspect import signature
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _helpers import *
from metrics import *


# Parse command line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--settings-path", default=None)
    return parser.parse_args()


# Bind the run's settings.json before importing modules that read settings at
# import time. Live settings are only used to locate the newest run.
_ARGS = parse_args()
RUN_DIR = resolve_run_dir(load_live_settings().RUNS_DIR if _ARGS.run_dir is None else None, _ARGS.run_dir)
load_run_settings(RUN_DIR)

from _data import ID_TO_SPLIT_NAME, df_to_metric_grids, load_split_df
from embeddings import get_embeddings_by_sample
from model import TimestepSelector, load_timestep_selector
from settings import *


def labeled_surfaces(
    true_phi: np.ndarray,   # (S, n_start, n_end), NaN outside the labeled set
    pred_phi: np.ndarray,
    default_i: int,
    default_j: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Flatten the phi grids to the cells labeled for every test image.

    Keeping one candidate set for all images is what makes their ranks
    comparable. Returns (true, pred, default_col) ready for metrics.py.
    """
    n = true_phi.shape[0]
    flat_true = true_phi.reshape(n, -1)
    flat_pred = pred_phi.reshape(n, -1)
    keep = np.isfinite(flat_true).all(axis=0) & np.isfinite(flat_pred).all(axis=0)
    default_flat = default_i * true_phi.shape[2] + default_j
    if not keep[default_flat]:
        raise ValueError("The default cell is not labeled for every test image")

    return (
        torch.as_tensor(flat_true[:, keep], dtype=torch.float64),
        torch.as_tensor(flat_pred[:, keep], dtype=torch.float64),
        int(np.cumsum(keep)[default_flat] - 1),
    )


def eval(run_dir: Path) -> dict:
    """Select (t_start, t_end) for every sample and score the test split."""
    print(f"Using settings from {run_dir / 'settings.json'}")

    # Confirm that the run artifacts exist.
    splits_path = run_dir / ID_TO_SPLIT_NAME
    weights_path = run_dir / "regressor_weights.pt"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path=}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path=}.")

    # Load all splits but metrics stay on test, the selections CSV covers every sample.
    splits = load_split_df(run_dir)
    all_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    test_df = splits["test"]
    all_sample_ids = sorted(all_df[SAMPLE_ID_COL].unique())
    test_sample_ids = sorted(test_df[SAMPLE_ID_COL].unique())
    test_pos = {sid: k for k, sid in enumerate(test_sample_ids)}
    t_start_values = sorted(all_df[T_START_COL].unique())
    t_end_values = sorted(all_df[T_END_COL].unique())

    # Load the predictor and wrap it in the selector. The checkpoint carries the
    # grid contract, and the mean surface is pulled in when the space needs it.
    device = resolve_device()
    print(f"Device: {device}.")
    t_selector = load_timestep_selector(weights_path, device=device)
    default_i, default_j = t_selector._default_i, t_selector._default_j
    n_labeled_pairs = len(t_selector._cell_i)

    # Load embeddings for every sample and true test metrics for evaluation.
    emb = get_embeddings_by_sample(all_df.drop_duplicates(SAMPLE_ID_COL), device)
    true_psnr, _ = df_to_metric_grids(test_df, test_sample_ids, t_start_values, t_end_values, PSNR_COL)
    true_clip, _ = df_to_metric_grids(test_df, test_sample_ids, t_start_values, t_end_values, CLIP_COL)
    true_phi = score_metric_grids(true_psnr, true_clip, default_i * len(t_end_values) + default_j)

    # Select (t_start, t_end) for every sample; fill pred grids only for test metrics.
    pred_psnr, pred_clip, pred_phi = np.zeros_like(true_psnr), np.zeros_like(true_clip), np.zeros_like(true_phi)
    all_selections: list[dict[str, float | str | bool]] = []
    for sid in all_sample_ids:
        e = emb[sid]
        grid = t_selector.pred_grid(e["img"], e["src"], e["tar"])
        sel = t_selector.select_grid(grid, noise_floor=NOISE_FLOOR_PHI)
        all_selections.append({
            "sample_id": sid,
            "t_start": sel.t_start,
            "t_end": sel.t_end,
            "deviate": sel.deviate,
            "pred_gain": sel.pred_gain,
        })
        if sid in test_pos:
            k = test_pos[sid]
            pred_psnr[k], pred_clip[k], pred_phi[k] = grid.psnr_grid, grid.clip_grid, grid.phi_grid

    test_selections = [s for s in all_selections if s["sample_id"] in test_pos]

    # Compute metrics on the test selections only.
    t_phi, p_phi, default_col = labeled_surfaces(true_phi, pred_phi, default_i, default_j)
    training = training_metrics(
        t_phi, p_phi, default_col,
        mse_weight=MSE_LOSS_WEIGHT,
        ranking_weight=RANKING_LOSS_WEIGHT,
        mse_top_k=MSE_LOSS_TOP_K,
        ranking_top_k=RANKING_LOSS_TOP_K,
    )
    selection = selection_metrics(t_phi, p_phi, default_col)
    metrics = {
        "run_dir": str(run_dir),
        "n_test_images": len(test_sample_ids),
        "grid": f"{len(t_start_values)}x{len(t_end_values)}",
        "n_labeled_pairs": int(n_labeled_pairs),
        "n_scored_cells": int(t_phi.shape[-1]),
        "prediction_space": str(PREDICTION_SPACE),
        "score_fn": str(SCORE_FN),
        # Kept under its historical name for consumers of this file.
        "spearman_phi_median": training["phi_spearman"],
        **training,
        **selection,
        "dataset_dir": str(DATASET_DIR),
        "default_t_start": DEFAULT_T_START,
        "default_t_end": DEFAULT_T_END,
    }

    # Save test metrics / selections; CSV covers every sample_id in the run.
    out_metrics = run_dir / "selection_metrics.json"
    out_selections = run_dir / "selections.json"
    scorer = SCORE_FN if "alpha" not in signature(SCORE_FNS[SCORE_FN][0]).parameters else f"{SCORE_FN}_a{int(PHI_ALPHA)}"
    start_col, end_col = f"{scorer}_t_start", f"{scorer}_t_end"
    out_predictions = run_dir / f"id_to_selections_{DIR_NAME.replace('_', '').lower()}.csv"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_selections, "w") as f:
        json.dump(test_selections, f, indent=2)
    pd.DataFrame(
        [{"sample_id": s["sample_id"], start_col: s["t_start"], end_col: s["t_end"]} for s in all_selections]
    ).to_csv(out_predictions, index=False)

    print(f"Selector eval on {len(test_sample_ids)} test images ({len(all_sample_ids)} selections)  run={run_dir.name}")
    print(f"  regret median={metrics['regret_median']:.4f}  p90={metrics['regret_p90']:.4f}")
    print(
        f"  rho_phi={metrics['spearman_phi_median']:.4f}  gain={metrics['gain_mean']:.4f}  "
        + "  ".join(f"top{k}={metrics[f'top{k}_accuracy']:.4f}" for k in TOP_K_VALUES)
    )
    print(f"  spearman phi median={metrics['spearman_phi_median']:.3f}")
    print(f"Saved {out_metrics}")
    print(f"Saved {out_selections}")
    print(f"Saved {out_predictions}")
    return metrics


def main() -> None:

    # Parse the command line arguments.
    args = parse_args()

    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Evaluate the timestep selector.
    eval(RUN_DIR)


if __name__ == "__main__":
    main()
