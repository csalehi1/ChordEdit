# train_t.py

"""
Evaluate timestep selector T on the test split.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)

Run after train_m.py:

    python train_t.py
    python train_t.py --run-dir runs/UltraEdit_Region_10000/20260101_120000
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
from model_m import SurrogateModel
from model_t import TimestepSelector, gate_metrics, per_image_spearman, regret
from settings import *


def eval(run_dir: Path) -> dict:
    print(f"Using settings from {run_dir / 'settings.json'}")

    # Confirm that the run artifacts exist.
    splits_path = run_dir / ID_TO_SPLIT_NAME
    surface_path = run_dir / "mean_surface.pt"
    weights_path = run_dir / "regressor_weights.pt"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path=}.")
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing {surface_path=}.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path=}.")

    # Load all splits but metrics stay on test, the selections CSV covers every sample.
    splits = load_split_df(run_dir)
    all_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    test_df = splits["test"]
    all_sample_ids = sorted(all_df["sample_id"].unique())
    test_sample_ids = sorted(test_df["sample_id"].unique())
    test_pos = {sid: k for k, sid in enumerate(test_sample_ids)}
    t_start_values = sorted(all_df["t_start"].unique())
    t_end_values = sorted(all_df["t_end"].unique())

    # Load the model and timestep predictor.
    device = resolve_device()
    print(f"Device: {device}.")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
    model = SurrogateModel(int(ckpt["img_dim"]), int(ckpt["text_dim"]), int(cell_t_pairs.shape[0]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # Initialize the timestep selector.
    surface = torch.load(surface_path, map_location="cpu", weights_only=True)
    mean_surface = np.asarray(surface["mean_true_delta"], dtype=np.float64) if M_TARGET_SPACE == "residual" else None
    t_selector = TimestepSelector(model, cell_t_pairs, t_start_values=t_start_values, t_end_values=t_end_values, mean_surface=mean_surface)
    default_i, default_j = t_selector._default_i, t_selector._default_j

    # Load embeddings for every sample and true test metrics for evaluation.
    emb = get_embeddings_by_sample(all_df.drop_duplicates("sample_id"), device)
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
    reg = regret(true_phi, pred_phi)
    rho_phi = per_image_spearman(true_phi, pred_phi)
    gate = gate_metrics(true_phi, pred_phi, default_i, default_j, NOISE_FLOOR_PHI)
    metrics = {
        "run_dir": str(run_dir),
        "n_test_images": len(test_sample_ids),
        "grid": f"{len(t_start_values)}x{len(t_end_values)}",
        "n_labeled_pairs": int(cell_t_pairs.shape[0]),
        "regret_median": float(np.median(reg)),
        "regret_p90": float(np.percentile(reg, 90)),
        "spearman_m_median": float(np.nanmedian(rho_phi)),  # JSON key kept for saved-run schema
        "gate": gate,
        "dataset_dir": str(DATASET_DIR),
        "default_t_start": DEFAULT_T_START,
        "default_t_end": DEFAULT_T_END,
    }

    # Save test metrics / selections; CSV covers every sample_id in the run.
    out_metrics = run_dir / "selection_metrics.json"
    out_selections = run_dir / "selections.json"
    scorer = T_TARGET_FN if "alpha" not in signature(T_TARGET_FNS[T_TARGET_FN][0]).parameters else f"{T_TARGET_FN}_a{int(PHI_ALPHA)}"
    start_col, end_col = f"{scorer}_t_start", f"{scorer}_t_end"
    out_predictions = run_dir / f"id_to_selections_{DIR_NAME.replace('_', '').lower()}.csv"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_selections, "w") as f:
        json.dump(test_selections, f, indent=2)
    pd.DataFrame(
        [{"sample_id": s["sample_id"], start_col: s["t_start"], end_col: s["t_end"]} for s in all_selections]
    ).to_csv(out_predictions, index=False)

    print(f"T eval on {len(test_sample_ids)} test images ({len(all_sample_ids)} selections)  run={run_dir.name}")
    print(f"  regret median={metrics['regret_median']:.4f}  p90={metrics['regret_p90']:.4f}")
    print(f"  spearman phi median={metrics['spearman_m_median']:.3f}")
    print(
        f"  gate precision={gate['precision']:.3f}  recall={gate['recall']:.3f}  "
        f"({gate['n_flagged']} flagged)"
    )
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
