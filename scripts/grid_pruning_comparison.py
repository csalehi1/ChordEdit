"""
Standalone script: compares full-grid vs three grid-pruning search strategies
(diagonal restriction, coarse-to-fine, coordinate descent) against the real
compute_softplus_score metric from the ChordEdit Classification-Model branch,
on the real 700-image dense grid data.

Run with:
    python scripts/grid_pruning_comparison.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from models.classification.utils import compute_softplus_score
from models.classification.settings import DEFAULT_T_START, DEFAULT_T_END, _SOFTPLUS_ALPHA, _SOFTPLUS_BETA, _NORMALIZE

FULLGRID_CSV = "/shared/ssd_30T/salehi/id_to_metrics_sdturbo_fullgrid_unclamped_tdelta0_TRUE_paper_matching_format.csv"
SCORE_RANGE = 0.632  # avg per-image score spread, for context
FULL_GRID_CELLS = 121
N_IMAGES_PROJECTION = 10000
MS_PER_CELL = {"gpu_generation": 197.1, "clip_psnr_scoring": 177.0, "jpeg_save": 2.0}

COARSE = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def clip01(v):
    return max(0.0, min(1.0, round(v, 1)))


def load_scores():
    df = pd.read_csv(FULLGRID_CSV)
    df = df.rename(columns={"clip_similarity_target_image_edit_part": "clip_edited"})
    df["softplus_score"] = compute_softplus_score(
        df, alpha=_SOFTPLUS_ALPHA, beta=_SOFTPLUS_BETA, normalize=_NORMALIZE
    )
    scores = defaultdict(dict)
    raw = defaultdict(dict)
    for row in df.itertuples():
        scores[row.sample_id][(row.t_start, row.t_end)] = row.softplus_score
        raw[row.sample_id][(row.t_start, row.t_end)] = (row.psnr, row.clip_edited)
    return scores, raw


def strategy_predictions(scores, t_values):
    preds = {}

    best_t_start = max(t_values, key=lambda t: scores[(t, DEFAULT_T_END)])
    best_t_end = max(t_values, key=lambda t: scores[(best_t_start, t)])
    preds["coord_descent"] = (best_t_start, best_t_end)

    restricted = {k: v for k, v in scores.items() if k[1] <= k[0]}
    preds["diagonal"] = max(restricted, key=restricted.get)

    coarse_scores = {(t_s, t_e): scores[(t_s, t_e)] for t_s in COARSE for t_e in COARSE}
    coarse_best = max(coarse_scores, key=coarse_scores.get)
    cs, ce = coarse_best
    refine_candidates = {(clip01(cs + ds), clip01(ce + de)) for ds in (-0.1, 0.1) for de in (-0.1, 0.1)}
    refine_scores = {k: scores[k] for k in refine_candidates if k in scores}
    combined = {coarse_best: coarse_scores[coarse_best], **refine_scores}
    preds["coarse_to_fine"] = max(combined, key=combined.get)

    cells_used = {
        "coord_descent": len(t_values) * 2 - 1,
        "diagonal": len(restricted),
        "coarse_to_fine": len(coarse_scores) + len(refine_scores),
    }
    return preds, cells_used


def main():
    all_scores, all_raw = load_scores()
    t_values = sorted({k[0] for g in all_scores.values() for k in g})
    n = len(all_scores)

    names = ("coord_descent", "diagonal", "coarse_to_fine")
    labels = {"coord_descent": "Coordinate descent", "diagonal": "Diagonal restriction (t_e<=t_s)",
              "coarse_to_fine": "Coarse 6x6 -> refine 4"}

    stats = {name: {"exact": 0, "cells": 0, "pct_all": [], "gap_all": [], "pct_miss": [], "gap_miss": [],
                     "best_psnr": [], "found_psnr": [], "best_clip": [], "found_clip": []}
             for name in names}

    for sample_id, scores in all_scores.items():
        raw = all_raw[sample_id]
        sorted_vals = sorted(scores.values())
        true_best = max(scores, key=scores.get)
        true_score = scores[true_best]
        best_psnr, best_clip = raw[true_best]

        preds, cells_used = strategy_predictions(scores, t_values)
        for name, cell in preds.items():
            stats[name]["cells"] += cells_used[name]
            pct = 100 * sorted_vals.index(scores[cell]) / (len(sorted_vals) - 1)
            gap = true_score - scores[cell]
            stats[name]["pct_all"].append(pct)
            stats[name]["gap_all"].append(gap)
            found_psnr, found_clip = raw[cell]
            stats[name]["best_psnr"].append(best_psnr)
            stats[name]["found_psnr"].append(found_psnr)
            stats[name]["best_clip"].append(best_clip)
            stats[name]["found_clip"].append(found_clip)
            if cell == true_best:
                stats[name]["exact"] += 1
            else:
                stats[name]["pct_miss"].append(pct)
                stats[name]["gap_miss"].append(gap)

    header = f"{'Strategy':<32}{'Cells':<8}{'% of full':<11}{'Exact match':<15}{'Avg %ile':<11}{'Avg gap':<10}{'Time @10k imgs':<16}{'Time saved'}"
    print(header)
    full_days = FULL_GRID_CELLS * (sum(MS_PER_CELL.values()) / 1000) * N_IMAGES_PROJECTION / 86400
    for name in names:
        s = stats[name]
        cells_avg = s["cells"] / n
        exact_pct = 100 * s["exact"] / n
        avg_pct = sum(s["pct_all"]) / n
        avg_gap = sum(s["gap_all"]) / n
        days = cells_avg * (sum(MS_PER_CELL.values()) / 1000) * N_IMAGES_PROJECTION / 86400
        saved = full_days - days
        print(f"{labels[name]:<32}{cells_avg:<8.0f}{100*cells_avg/FULL_GRID_CELLS:<11.0f}"
              f"{s['exact']}/{n} ({exact_pct:.1f}%)  {avg_pct:<11.1f}{avg_gap:<10.4f}{days:<16.2f}{saved:.2f} days")

    print()
    print(f"{'Strategy':<32}{'# misses':<10}{'Median gap':<13}{'Mean gap':<11}{'Worst gap':<12}{'Worst % of range':<18}{'Worst %ile'}")
    for name in names:
        gm = stats[name]["gap_miss"]
        pm = stats[name]["pct_miss"]
        print(f"{labels[name]:<32}{len(gm):<10}{np.median(gm):<13.4f}{np.mean(gm):<11.4f}"
              f"{max(gm):<12.4f}{100*max(gm)/SCORE_RANGE:<18.1f}{min(pm):.1f}%")

    print()
    print(f"{'Strategy':<32}{'Mean PSNR (best)':<18}{'Mean PSNR (found)':<19}{'PSNR gap (dB)':<15}"
          f"{'Mean CLIP (best)':<18}{'Mean CLIP (found)':<19}{'CLIP gap'}")
    for name in names:
        s = stats[name]
        mean_best_psnr = np.mean(s["best_psnr"])
        mean_found_psnr = np.mean(s["found_psnr"])
        mean_best_clip = np.mean(s["best_clip"])
        mean_found_clip = np.mean(s["found_clip"])
        print(f"{labels[name]:<32}{mean_best_psnr:<18.3f}{mean_found_psnr:<19.3f}"
              f"{mean_best_psnr - mean_found_psnr:<15.3f}{mean_best_clip:<18.3f}{mean_found_clip:<19.3f}"
              f"{mean_best_clip - mean_found_clip:.3f}")

    print()
    for name in names:
        pm = np.array(stats[name]["pct_miss"])
        print(f"\n{labels[name]} ({len(pm)} misses):")
        bands = [
            ("still >=99th percentile (essentially tied)", (pm >= 99).sum()),
            ("95th-99th percentile (very close)", ((pm >= 95) & (pm < 99)).sum()),
            ("90th-95th percentile (close)", ((pm >= 90) & (pm < 95)).sum()),
            ("below 90th percentile (real miss)", (pm < 90).sum()),
        ]
        for label, count in bands:
            print(f"  {label:<45} {count:>4} ({100*count/len(pm):.1f}%)")


if __name__ == "__main__":
    main()
