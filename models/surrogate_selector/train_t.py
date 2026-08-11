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
import numpy as np
import os
import pandas as pd
import sys
import torch
from pathlib import Path

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from _helpers import current_commit_id, load_live_settings, load_run_settings, resolve_run_dir
from _helpers import resolve_device, score_metric_grids, timestep_pairs_from_df


# Parse command line arguments.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--noise-floor", type=float, default=None)
    parser.add_argument("--gpu", default=None)
    # Only affects which run is newest when --run-dir is absent: an explicit
    # run dir replays its own settings.json snapshot either way.
    parser.add_argument("--settings-path", default=None, help="config file to use instead of ./settings.json")
    return parser.parse_args()


# Resolve the run's settings snapshot before importing modules that bind
# settings at import time (load_run_settings raises otherwise; see _helpers).
_ARGS = parse_args()

# The live settings are only needed to find the newest run; an explicit
# --run-dir is self-contained (its snapshot pins the config), so skip loading
# them and the prompts they carry.
RUN_DIR = resolve_run_dir(load_live_settings().RUNS_DIR if _ARGS.run_dir is None else None, _ARGS.run_dir)
load_run_settings(RUN_DIR)

from _data import ID_TO_SPLIT_NAME, df_to_metric_grids, load_split_df
from embeddings import get_embeddings_by_sample
from model_m import SurrogateModel
from model_t import TimestepSelector, gate_metrics, per_image_spearman, regret
from settings import *


def evaluate(
    run_dir: Path,
    noise_floor: float | None = None,
    gpu: int | str | None = None,
) -> dict:
    print(f"Using settings from {run_dir / 'settings.py'}")

    weights_path = run_dir / "regressor_weights.pt"
    splits_path = run_dir / ID_TO_SPLIT_NAME
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing M weights: {weights_path}")
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits at {splits_path}; run train_m.py first.")

    test_df = load_split_df(run_dir)["test"]
    sample_ids = sorted(test_df["sample_id"].unique())
    t_start_values = sorted(test_df["t_start"].unique())
    t_end_values = sorted(test_df["t_end"].unique())
    n1, n2 = len(t_start_values), len(t_end_values)
    t_pairs = timestep_pairs_from_df(test_df)
    expected_pairs = {(float(a), float(b)) for a, b in t_pairs}
    for sid, group in test_df.groupby("sample_id"):
        pairs = set(zip(group["t_start"].astype(float), group["t_end"].astype(float)))
        if pairs != expected_pairs:
            raise ValueError(f"Unexpected shape mismatch: {len(pairs)} != {len(expected_pairs)}")

    # Load the model and timestep predictor.
    device = resolve_device(gpu)
    print(f"device: {device}")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = SurrogateModel(int(ckpt["img_dim"]), int(ckpt["text_dim"]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_stats(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # Load the timestep selector. T requires the run's mean_surface;
    # train_m.py writes it for new runs, calibrate_t.py retrofits older ones.
    # In the "residual" target space the surface is added back to predicted
    # residuals before phi; in "delta" space no offset is applied.
    from _helpers import MEAN_SURFACE_NAME, load_mean_surface
    from model_t import mean_surface_from_dict

    surface = load_mean_surface(run_dir)
    if surface is None:
        raise FileNotFoundError(f"Missing T mean surface: {run_dir / MEAN_SURFACE_NAME}")
    mean_surface = (
        mean_surface_from_dict(surface, np.asarray(t_start_values), np.asarray(t_end_values))
        if M_TARGET_SPACE == "residual" else None
    )
    print(
        f"Loaded mean_surface ({surface['split']} split, {surface['n_samples']} samples): "
        f"T selects {'residual + mean surface' if mean_surface is not None else 'predicted deltas'}."
    )
    t_selector = TimestepSelector(model, t_start_values=t_start_values, t_end_values=t_end_values, mean_surface=mean_surface)
    default_i = t_selector._default_i
    default_j = t_selector._default_j
    nf = NOISE_FLOOR_PHI if noise_floor is None else noise_floor

    emb = get_embeddings_by_sample(test_df.drop_duplicates("sample_id"), device)
    # model.release_encoders()
    true_psnr, _ = df_to_metric_grids(test_df, sample_ids, t_start_values, t_end_values, PSNR_COL)
    true_clip, _ = df_to_metric_grids(test_df, sample_ids, t_start_values, t_end_values, CLIP_COL)
    true_phi = score_metric_grids(true_psnr, true_clip, default_i * len(t_end_values) + default_j)

    pred_psnr = np.zeros_like(true_psnr)
    pred_clip = np.zeros_like(true_clip)
    pred_phi = np.zeros_like(true_phi)
    selections = []
    for k, sid in enumerate(sample_ids):
        e = emb[sid]
        # Predict only the labeled pairs so pred grids share true grids' NaN mask.
        grid = t_selector.predict_grid_from_emb(e["img"], e["mask"], e["src"], e["tar"], t_pairs=t_pairs)
        pred_psnr[k], pred_clip[k], pred_phi[k] = grid.psnr_grid, grid.clip_grid, grid.phi_grid
        sel = t_selector.select_from_grid(grid, noise_floor=nf)
        selections.append(
            {
                "sample_id": sid,
                "t_start": sel.t_start,
                "t_end": sel.t_end,
                "deviate": sel.deviate,
                "pred_gain": sel.pred_gain,
            }
        )

    reg = regret(true_phi, pred_phi)
    rho_phi = per_image_spearman(true_phi, pred_phi)
    gate = gate_metrics(true_phi, pred_phi, default_i, default_j, nf)

    metrics = {
        "run_dir": str(run_dir),
        "n_test_images": len(sample_ids),
        "grid": f"{n1}x{n2}",
        "n_labeled_pairs": len(t_pairs),
        "regret_median": float(np.median(reg)),
        "regret_p90": float(np.percentile(reg, 90)),
        "spearman_m_median": float(np.nanmedian(rho_phi)),  # JSON key kept for saved-run schema
        "gate": gate,
        "dataset_dir": str(DATASET_DIR),
        "default_t_start": DEFAULT_T_START,
        "default_t_end": DEFAULT_T_END,
    }

    # An explicit noise floor writes suffixed files so a gate sweep does not
    # clobber the run's own metrics (regret/Spearman do not depend on the gate).
    suffix = "" if noise_floor is None else f"_nf{str(nf).replace('.', 'p')}"
    out_metrics = run_dir / f"t_train_metrics{suffix}.json"
    out_selections = run_dir / f"t_test_selections{suffix}.json"
    with open(out_metrics, "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out_selections, "w") as f:
        json.dump(selections, f, indent=2)

    # The selected timestep pair per sample, tagged with the commit that
    # produced it so predictions from different code states never collide.
    out_predictions = run_dir / f"id_to_predictions_{current_commit_id()}{suffix}.csv"
    pd.DataFrame(
        [{
                SAMPLE_ID_COL: s["sample_id"],
                PRED_T_START_COL: s["t_start"],
                PRED_T_END_COL: s["t_end"],
        } for s in selections],
        columns=[SAMPLE_ID_COL, PRED_T_START_COL, PRED_T_END_COL],
    ).to_csv(out_predictions, index=False)

    print(f"T eval on {len(sample_ids)} test images  run={run_dir.name}")
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
    gpu = _ARGS.gpu
    if gpu is not None and str(gpu).lower() != "cpu":
        gpu = int(gpu)

    # Evaluate the timestep selector T on the test split.
    evaluate(run_dir=RUN_DIR, noise_floor=_ARGS.noise_floor, gpu=gpu)


if __name__ == "__main__":
    main()
