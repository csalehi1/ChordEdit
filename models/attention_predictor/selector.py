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
from scores import calc_norm_deltas

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

from dataset import ID_TO_SPLIT_NAME, get_dataset
from model import TimestepSelector, load_timestep_selector
from settings import *


def labeled_surfaces(
    true_phi: np.ndarray,   # (S, n_cells)
    pred_phi: np.ndarray,   # (S, n_cells)
    default_cell: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Pack true/pred phi for metrics.py. Every sample shares the same cells."""
    if true_phi.shape != pred_phi.shape:
        raise ValueError(f"Expected {true_phi.shape} == {pred_phi.shape}")
    if not (0 <= default_cell < true_phi.shape[-1]):
        raise ValueError(f"Expected default_cell in [0, {true_phi.shape[-1]}), got {default_cell}")
    if not np.isfinite(true_phi).all() or not np.isfinite(pred_phi).all():
        raise ValueError("Expected finite packed phi")
    return (
        torch.as_tensor(true_phi, dtype=torch.float64),
        torch.as_tensor(pred_phi, dtype=torch.float64),
        int(default_cell),
    )


def eval(run_dir: Path) -> dict:
    """Select (t_start, t_end) for every sample and score the test split."""
    print(f"Using settings from {run_dir / 'settings.json'}")

    # Confirm that the run artifacts exist. get_splits_df writes this file on a
    # new split; if it is missing here, do not draw a fresh split.
    splits_path = run_dir / ID_TO_SPLIT_NAME
    weights_path = run_dir / "regressor_weights.pt"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path=}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path=}.")

    # Load the predictor and wrap it in the selector. The checkpoint carries the
    # grid contract, and the mean surface is pulled in when the space needs it.
    device = resolve_device()
    print(f"Device: {device}.")
    t_selector = load_timestep_selector(weights_path, device=device)
    n_labeled_pairs = len(t_selector._cell_i)
    default_k = t_selector._default_k
    cell_i, cell_j = t_selector._cell_i, t_selector._cell_j

    # Replay the run's split membership; the selections CSV covers every
    # sample, metrics stay on test.
    bundle = get_dataset(device, run_dir)
    test_ds = bundle.test
    test_sample_ids = list(test_ds.sample_ids)
    test_pos = {sid: k for k, sid in enumerate(test_sample_ids)}

    # True test metrics: y_raw is (S, n_cells, 2) in the same canonical
    # (t_start, t_end)-sorted cell order the selector's cell_t_pairs use.
    true_raw = test_ds.y_raw.double().cpu()
    true_phi = calc_phi(calc_norm_deltas(true_raw, default_k)).numpy()

    # Select (t_start, t_end) for every sample; fill packed pred phi for test metrics.
    pred_phi = np.zeros_like(true_phi)
    all_selections: list[dict[str, float | str | bool]] = []
    for name in ("train", "val", "test"):
        split = bundle.splits[name]
        for sid in split.sample_ids:
            x = split[sid].x
            grid = t_selector.pred_grid(
                x.image_tokens, x.source_tokens, x.target_tokens, x.source_mask, x.target_mask,
            )
            sel = t_selector.select_grid(
                grid,
                noise_floor=NOISE_FLOOR_PHI,
                clip_floor=SELECT_CLIP_FLOOR,
                phi_weights=SELECT_PHI_WEIGHTS,
            )
            all_selections.append({
                "sample_id": sid,
                "t_start": sel.t_start,
                "t_end": sel.t_end,
                "deviate": sel.deviate,
                "pred_gain": sel.pred_gain,
            })
            if name == "test":
                pred_phi[test_pos[sid]] = grid.phi_grid[cell_i, cell_j]

    all_selections.sort(key=lambda s: s["sample_id"])
    test_selections = [s for s in all_selections if s["sample_id"] in test_pos]

    # Compute metrics on the test selections only.
    t_phi, p_phi, default_col = labeled_surfaces(true_phi, pred_phi, default_k)
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
        "grid": f"{t_selector.n_start}x{t_selector.n_end}",
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

    print(f"Selector eval on {len(test_sample_ids)} test images ({len(all_selections)} selections)  run={run_dir.name}")
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
