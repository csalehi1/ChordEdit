# eval_rules.py

"""Post-hoc evaluation of ensembling and selection rules on trained runs.

Neither lever needs retraining: the ensemble averages predicted delta grids
across runs that share a split, and the selection rules re-rank an already
predicted grid. Both are therefore free at deployment time and are evaluated
here from saved checkpoints.

Usage:
    python eval_rules.py --runs runs/<dataset>/<run> [<run> ...] [--out rules.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _helpers import load_run_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


_ARGS = parse_args()
# Bind settings from the first run so every module below reads that run's config.
load_run_settings(_ARGS.runs[0])

from _data import df_to_metric_grids, load_split_df
from _helpers import phi_from_delta_grids, resolve_device
from embeddings import get_embeddings_by_sample
from model import SurrogateModel, TimestepSelector
from settings import *


def per_image_spearman(true_grid: np.ndarray, pred_grid: np.ndarray) -> np.ndarray:
    """Rank correlation within each image's timestep grid (over labeled cells).

    Lives here rather than in metrics.py: these grids are (S, n_start, n_end)
    numpy with NaN outside the labeled set, not metrics.py's flattened torch
    surfaces over one shared candidate set.
    """
    rhos = []
    for k in range(true_grid.shape[0]):
        t, p = true_grid[k].ravel(), pred_grid[k].ravel()
        labeled = np.isfinite(t) & np.isfinite(p)
        if labeled.sum() < 2:
            rhos.append(np.nan)
            continue
        rho, _ = spearmanr(t[labeled], p[labeled])
        rhos.append(np.nan if rho is None else float(rho))
    return np.array(rhos)


def load_selector(run_dir: Path, t_start_values, t_end_values, device) -> TimestepSelector:
    ckpt = torch.load(run_dir / "regressor_weights.pt", map_location=device, weights_only=False)
    cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
    img_shape = tuple(int(v) for v in ckpt["img_shape"])
    text_dim = int(tuple(ckpt["text_shape"])[-1])
    model = SurrogateModel(img_shape, text_dim, int(cell_t_pairs.shape[0]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()
    surface = torch.load(run_dir / "mean_surface.pt", map_location="cpu", weights_only=True)
    mean_surface = np.asarray(surface["mean_true_delta"], dtype=np.float64) if PREDICTION_SPACE == "residuals" else None
    return TimestepSelector(model, cell_t_pairs, t_start_values=t_start_values, t_end_values=t_end_values, mean_surface=mean_surface)


def predict_deltas(selector: TimestepSelector, emb: dict, sample_ids: list[str]) -> np.ndarray:
    """Predicted (n_samples, n_start, n_end, 2) delta grids."""
    out = None
    for k, sid in enumerate(sample_ids):
        e = emb[sid]
        grid = selector.pred_grid(e["img"], e["src"], e["tar"])
        if out is None:
            out = np.full((len(sample_ids), *grid.psnr_grid.shape, 2), np.nan)
        out[k, ..., 0] = grid.psnr_grid
        out[k, ..., 1] = grid.clip_grid
    return out


def evaluate_rule(
    pred_deltas: np.ndarray,
    true_deltas: np.ndarray,
    true_phi: np.ndarray,
    default_i: int,
    default_j: int,
    weights: tuple[float, float] | None = None,
    clip_floor: float | None = None,
    raw: tuple[np.ndarray, np.ndarray] | None = None,
) -> dict:
    """Score one selection rule against the true surface.

    raw carries the unnormalized (PSNR, CLIP) grids so the pick can also be
    reported in dB and CLIP points against the default cell, which the
    per-grid min-max normalization of the deltas otherwise hides.
    """
    rank_phi = phi_from_delta_grids(pred_deltas, weights=weights)
    if clip_floor is not None:
        # Restrict the argmax to cells clearing the floor, keeping the argmax
        # fallback when no cell qualifies.
        eligible = np.where(pred_deltas[..., 1] >= clip_floor, rank_phi, np.nan)
        has_any = np.isfinite(eligible).any(axis=(1, 2))
        rank_phi = np.where(has_any[:, None, None], eligible, rank_phi)

    n, n1, n2 = rank_phi.shape
    flat = rank_phi.reshape(n, -1)
    pick = np.nanargmax(flat, axis=1)
    pi, pj = np.unravel_index(pick, (n1, n2))
    rows = np.arange(n)

    gain = true_phi[rows, pi, pj]  # phi(default) == 0
    reg = np.nanmax(true_phi.reshape(n, -1), axis=1) - gain
    deviate = (pi != default_i) | (pj != default_j)
    out = {
        "regret_p50": float(np.median(reg)),
        "regret_p90": float(np.percentile(reg, 90)),
        "gain_mean": float(np.mean(gain)),
        "win_rate": float(np.mean(gain > 0)),
        "deviate_rate": float(np.mean(deviate)),
        "d_psnr_at_pick": float(np.mean(true_deltas[rows, pi, pj, 0])),
        "d_clip_at_pick": float(np.mean(true_deltas[rows, pi, pj, 1])),
    }
    if raw is not None:
        raw_psnr, raw_clip = raw
        out["db_at_pick"] = float(np.nanmean(raw_psnr[rows, pi, pj] - raw_psnr[:, default_i, default_j]))
        out["clip_pts_at_pick"] = float(np.nanmean(raw_clip[rows, pi, pj] - raw_clip[:, default_i, default_j]))
    return out


def main() -> None:
    run_dirs = list(_ARGS.runs)
    device = resolve_device()
    print(f"Device: {device}. Runs: {[r.name for r in run_dirs]}")

    # Every run must share a split for the ensemble to be meaningful.
    splits = load_split_df(run_dirs[0])
    for rd in run_dirs[1:]:
        if (rd / "id_to_split.csv").read_text() != (run_dirs[0] / "id_to_split.csv").read_text():
            raise ValueError(f"{rd.name} does not share {run_dirs[0].name}'s split")

    all_df = pd.concat([splits["train"], splits["val"], splits["test"]], ignore_index=True)
    test_df = splits["test"]
    test_sample_ids = sorted(test_df[SAMPLE_ID_COL].unique())
    t_start_values = sorted(all_df[T_START_COL].unique())
    t_end_values = sorted(all_df[T_END_COL].unique())

    emb = get_embeddings_by_sample(all_df.drop_duplicates(SAMPLE_ID_COL), device)

    # True surface, in the same delta space the predictions live in.
    true_psnr, _ = df_to_metric_grids(test_df, test_sample_ids, t_start_values, t_end_values, PSNR_COL)
    true_clip, _ = df_to_metric_grids(test_df, test_sample_ids, t_start_values, t_end_values, CLIP_COL)
    selector0 = load_selector(run_dirs[0], t_start_values, t_end_values, device)
    default_i, default_j = selector0._default_i, selector0._default_j
    from _helpers import delta_metric_grids

    baseline_idx = default_i * len(t_end_values) + default_j
    true_deltas = delta_metric_grids(true_psnr, true_clip, baseline_idx)
    true_phi = phi_from_delta_grids(true_deltas)

    # Per-run predictions, then the ensemble mean in delta space.
    per_run: dict[str, np.ndarray] = {}
    for rd in run_dirs:
        sel = selector0 if rd == run_dirs[0] else load_selector(rd, t_start_values, t_end_values, device)
        per_run[rd.name] = predict_deltas(sel, emb, test_sample_ids)
        print(f"Predicted {rd.name}")
    ensemble = np.mean(np.stack(list(per_run.values())), axis=0)

    results: dict[str, dict] = {}
    for name, pred in list(per_run.items()) + [("ensemble", ensemble)]:
        results[f"{name}|argmax"] = evaluate_rule(pred, true_deltas, true_phi, default_i, default_j, raw=(true_psnr, true_clip))
        results[f"{name}|argmax"]["rho_phi"] = float(
            np.nanmedian(per_image_spearman(true_phi, phi_from_delta_grids(pred)))
        )

    # CLIP-weight frontier on the ensemble. These are tuning knobs for a later
    # step: the shipped configuration leaves them neutral (weights (1,1), no floor),
    # which is the plain argmax row above.
    for w in (2.0, 4.0, 8.0):
        results[f"ensemble|w={w}"] = evaluate_rule(
            ensemble, true_deltas, true_phi, default_i, default_j, weights=(1.0, w), raw=(true_psnr, true_clip)
        )
    results["ensemble|w=4,tau=0.2"] = evaluate_rule(
        ensemble, true_deltas, true_phi, default_i, default_j, weights=(1.0, 4.0), clip_floor=0.2, raw=(true_psnr, true_clip)
    )

    # Model-free references. The default cell is the paper baseline; the best
    # fixed cell is the argmax of the train prior's own phi surface, so it is
    # chosen without touching test and is the bar a surrogate has to clear.
    n_test = true_phi.shape[0]
    rr_ = np.arange(n_test)

    def _fixed_cell(i: int, j: int) -> dict:
        g = true_phi[rr_, i, j]
        r = np.nanmax(true_phi.reshape(n_test, -1), axis=1) - g
        return {
            "regret_p50": float(np.median(r)), "regret_p90": float(np.percentile(r, 90)),
            "gain_mean": float(np.mean(g)), "win_rate": float(np.mean(g > 0)),
            "deviate_rate": float((i != default_i) or (j != default_j)),
            "d_psnr_at_pick": float(np.mean(true_deltas[rr_, i, j, 0])),
            "d_clip_at_pick": float(np.mean(true_deltas[rr_, i, j, 1])),
            "db_at_pick": float(np.nanmean(true_psnr[:, i, j] - true_psnr[:, default_i, default_j])),
            "clip_pts_at_pick": float(np.nanmean(true_clip[:, i, j] - true_clip[:, default_i, default_j])),
        }

    results["ref|default_cell"] = _fixed_cell(default_i, default_j)
    if selector0.mean_surface is not None:
        prior_phi = phi_from_delta_grids(selector0.mean_surface[None])[0]
        bi, bj = np.unravel_index(np.nanargmax(prior_phi), prior_phi.shape)
        results[f"ref|best_fixed({t_start_values[bi]:.1f},{t_end_values[bj]:.1f})"] = _fixed_cell(int(bi), int(bj))

    orc = np.nanmax(true_phi.reshape(n_test, -1), axis=1)
    oi, oj = np.unravel_index(np.nanargmax(true_phi.reshape(n_test, -1), axis=1), true_phi.shape[1:])
    rr = np.arange(n_test)
    results["oracle"] = {
        "regret_p50": 0.0, "regret_p90": 0.0,
        "gain_mean": float(np.mean(orc)), "win_rate": float(np.mean(orc > 0)),
        "deviate_rate": float(np.mean((oi != default_i) | (oj != default_j))),
        "d_psnr_at_pick": float(np.mean(true_deltas[rr, oi, oj, 0])),
        "d_clip_at_pick": float(np.mean(true_deltas[rr, oi, oj, 1])),
        "db_at_pick": float(np.nanmean(true_psnr[rr, oi, oj] - true_psnr[:, default_i, default_j])),
        "clip_pts_at_pick": float(np.nanmean(true_clip[rr, oi, oj] - true_clip[:, default_i, default_j])),
    }

    cols = ["rho_phi", "regret_p50", "regret_p90", "gain_mean", "win_rate", "deviate_rate", "d_psnr_at_pick", "d_clip_at_pick", "db_at_pick", "clip_pts_at_pick"]
    width = max(len(k) for k in results)
    print(f"\n{'rule'.ljust(width)} | " + " | ".join(c.rjust(9) for c in cols))
    for k, v in results.items():
        print(f"{k.ljust(width)} | " + " | ".join(f"{v.get(c, float('nan')):9.4f}" for c in cols))

    if _ARGS.out:
        _ARGS.out.write_text(json.dumps({"runs": [r.name for r in run_dirs], "results": results}, indent=2))
        print(f"\nSaved {_ARGS.out}")


if __name__ == "__main__":
    main()
