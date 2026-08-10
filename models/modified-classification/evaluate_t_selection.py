"""
T (selection step) for the trained surrogate M: for each TEST sample, run M
over all 121 grid cells to predict (psnr, clip), score each cell with LINEX
(alpha=LINEX_ALPHA, matching M's own training objective) using the same
per-sample-normalized-delta-relative-to-baseline formula as everywhere else
in this project, and take the argmax as M's final (t_start, t_end) pick.

Then compares REAL ground-truth psnr/clip at that pick against the fixed
ChordEdit default baseline (0.7, 0.3), in the same format used for the
other classifiers (compare_classifier_picks_to_baseline.py), so all results
are directly comparable in one table.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd
import torch

from _helpers import unnormalize_metric_arrays
from _data import prepare_df, target_bounds
from model_m import MetricRegressor
from settings import *
from train_m_tournament12k import build_surrogate_df, LINEX_ALPHA

REPO_ROOT = Path(_ROOT)
MERGED_METRICS_CSV = REPO_ROOT / "models" / "classification" / "data" / "id_to_metrics_sdxlturbo_tournament12k.csv"
RUN_DIR = REPO_ROOT / "models" / "modified-classification" / "outputs" / "sdxlturbo_tournament12k" / "20260802_223614"
EMBED_CACHE = REPO_ROOT / "models" / "modified-classification" / "outputs" / ".cache" / "embeddings"

BASELINE_T_START, BASELINE_T_END = 0.7, 0.3
T_VALUES = [round(0.1 * i, 1) for i in range(11)]


def linex_u(x: np.ndarray, alpha: float) -> np.ndarray:
    return 0.5 * (x + (1 - np.exp(-alpha * x)) / alpha)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Rebuilding surrogate df for target bounds...", flush=True)
    data_df = build_surrogate_df()
    bounds = target_bounds(data_df)
    print("bounds:", bounds, flush=True)

    ckpt = torch.load(RUN_DIR / "regressor_weights.pt", map_location=device, weights_only=False)
    regressor = MetricRegressor(img_dim=ckpt["img_dim"], text_dim=ckpt["text_dim"]).to(device)
    regressor.load_state_dict(ckpt["regressor_state_dict"])
    regressor.eval()

    cache_files = list(EMBED_CACHE.glob("*.pt"))
    if len(cache_files) != 1:
        raise RuntimeError(f"expected exactly one embedding cache file, found {cache_files}")
    emb_data = torch.load(cache_files[0], map_location="cpu", weights_only=False)
    sid_to_i = {sid: i for i, sid in enumerate(emb_data["sample_ids"])}

    splits_df = pd.read_csv(RUN_DIR / "id_to_split.csv", dtype={SAMPLE_ID_COL: str})
    test_ids = splits_df.loc[splits_df["split"] == "test", SAMPLE_ID_COL].tolist()
    print(f"{len(test_ids)} test samples", flush=True)

    real_metrics = pd.read_csv(MERGED_METRICS_CSV, dtype={"sample_id": str})
    real_lookup = {
        (r.sample_id, round(r.t_start, 1), round(r.t_end, 1)): (r.psnr, r.clip_edited)
        for r in real_metrics.itertuples()
    }

    n_grid = len(T_VALUES) * len(T_VALUES)
    t_grid = torch.tensor([(ts, te) for ts in T_VALUES for te in T_VALUES], dtype=torch.float, device=device)
    base_cell_idx = next(i for i, (ts, te) in enumerate(t_grid.tolist()) if round(ts, 1) == BASELINE_T_START and round(te, 1) == BASELINE_T_END)

    rows = []
    with torch.no_grad():
        for sid in test_ids:
            i = sid_to_i[sid]
            img = emb_data["img"][i].unsqueeze(0).repeat(n_grid, 1).to(device)
            mask = emb_data["mask"][i].unsqueeze(0).repeat(n_grid, 1).to(device)
            src = emb_data["src"][i].unsqueeze(0).repeat(n_grid, 1).to(device)
            tar = emb_data["tar"][i].unsqueeze(0).repeat(n_grid, 1).to(device)

            out = regressor(img, mask, src, tar, t_grid)
            pred_norm = regressor.denormalize(out).cpu().numpy()
            pred_psnr, pred_clip = unnormalize_metric_arrays(pred_norm[:, 0], pred_norm[:, 1], bounds)

            def minmax(col):
                lo, hi = col.min(), col.max()
                return (col - lo) / (hi - lo + 1e-6)

            psnr_n = minmax(pred_psnr)
            clip_n = minmax(pred_clip)
            delta_psnr = psnr_n - psnr_n[base_cell_idx]
            delta_clip = clip_n - clip_n[base_cell_idx]
            linex = linex_u(delta_psnr, LINEX_ALPHA) + linex_u(delta_clip, LINEX_ALPHA)

            best_idx = int(np.argmax(linex))
            pred_ts, pred_te = round(t_grid[best_idx, 0].item(), 1), round(t_grid[best_idx, 1].item(), 1)

            real_pred = real_lookup.get((sid, pred_ts, pred_te))
            real_base = real_lookup.get((sid, BASELINE_T_START, BASELINE_T_END))
            if real_pred is None or real_base is None:
                continue
            rows.append({
                "sample_id": sid, "pred_t_start": pred_ts, "pred_t_end": pred_te,
                "pred_psnr": real_pred[0], "pred_clip": real_pred[1],
                "base_psnr": real_base[0], "base_clip": real_base[1],
            })

    result_df = pd.DataFrame(rows)
    n = len(result_df)
    psnr_win = (result_df["pred_psnr"] > result_df["base_psnr"]).mean()
    clip_win = (result_df["pred_clip"] > result_df["base_clip"]).mean()
    both_win = ((result_df["pred_psnr"] > result_df["base_psnr"]) & (result_df["pred_clip"] > result_df["base_clip"])).mean()
    print(f"\n=== Surrogate M + T (LINEX alpha={LINEX_ALPHA}) ({n} test samples) ===")
    print(f"  mean PSNR-unedited:  baseline={result_df['base_psnr'].mean():.3f}   M+T-pick={result_df['pred_psnr'].mean():.3f}")
    print(f"  mean CLIP-edited:    baseline={result_df['base_clip'].mean():.3f}   M+T-pick={result_df['pred_clip'].mean():.3f}")
    print(f"  win rate:  PSNR better={psnr_win:.1%}   CLIP better={clip_win:.1%}   both better={both_win:.1%}")
    print(f"  mean pred_t_start={result_df.pred_t_start.mean():.3f}  mean pred_t_end={result_df.pred_t_end.mean():.3f}")

    out_csv = RUN_DIR / "baseline_vs_pick_comparison.csv"
    result_df.to_csv(out_csv, index=False)
    print(f"  saved -> {out_csv}")


if __name__ == "__main__":
    main()
