# .claude/eval_metrics.py

"""
Full evaluation-metric table for a trained attention_predictor run.

Recomputes every metric in the paper's "Evaluation Metrics" subsection from a
saved checkpoint, including the two that train.py does not track:

    regression_loss   MSE on z-scored deltas (train-split mean/std per metric)
    ranking_loss      logistic pairwise loss on phi over all within-grid pairs
    top<N>_accuracy   pick lands in the true top-N cells by phi, N = 1..5

Everything else (rho_phi, regret, gain, deviate/improvement rate, R2_i, rho_i,
per-metric gain) is recomputed here independently of train.py so the two paths
can be cross-checked.

Usage:
    python .claude/eval_metrics.py --run-dir runs/<DIR_NAME>/<run> [--out metrics.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.abspath(os.path.join(_DIR, ".."))
_ROOT = os.path.abspath(os.path.join(_PKG, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _PKG)

from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from _helpers import (
    calc_phi,
    gather_at_pairs,
    load_mean_surface,
    load_run_settings,
    resolve_device,
    resolve_run_dir,
    load_live_settings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full metric table for one run")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON output path (default: <run>/full_metrics.json)")
    parser.add_argument("--splits", default="train,val,test")
    return parser.parse_args()


_ARGS = parse_args()
RUN_DIR = resolve_run_dir(load_live_settings().RUNS_DIR if _ARGS.run_dir is None else None, _ARGS.run_dir)
load_run_settings(RUN_DIR)

from _data import build_splits_df, create_cell_tensors  # noqa: E402
from model import AttentionModel, deltas_from_preds  # noqa: E402
from settings import *  # noqa: E402,F403


"""
Metric primitives.
"""

def median_spearman(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Median and mean per-grid Spearman rho of two (S, N) score matrices.

    Uses scipy so tied values get average ranks, unlike train.py's ordinal
    ranking. With continuous phi the two agree to several decimals.
    """
    rhos = []
    for k in range(true.shape[0]):
        t, p = true[k], pred[k]
        ok = np.isfinite(t) & np.isfinite(p)
        if ok.sum() < 2:
            continue
        rho, _ = spearmanr(t[ok], p[ok])
        if rho is not None and np.isfinite(rho):
            rhos.append(float(rho))
    if not rhos:
        return float("nan"), float("nan")
    return float(np.median(rhos)), float(np.mean(rhos))


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor) -> float:
    """Mean log(1 + exp(-(pred_u - pred_v))) over pairs with true_u > true_v.

    Chunked over grids: the (N, N) pair tensor is the memory driver, not the
    grid count.
    """
    total, count = 0.0, 0
    for k in range(0, pred.shape[0], 64):
        p, t = pred[k : k + 64], true[k : k + 64]
        diff_true = t.unsqueeze(-1) - t.unsqueeze(-2)
        diff_pred = p.unsqueeze(-1) - p.unsqueeze(-2)
        mask = diff_true > 0
        n = int(mask.sum().item())
        if n == 0:
            continue
        total += float(torch.nn.functional.softplus(-diff_pred[mask]).sum().item())
        count += n
    return total / count if count else float("nan")


def top_n_accuracy(true_phi: torch.Tensor, pred_phi: torch.Tensor, n: int) -> float:
    """Fraction of grids whose argmax(pred phi) is in the true top-n cells by phi."""
    n = min(n, true_phi.shape[-1])
    top = true_phi.topk(n, dim=-1).indices               # (S, n)
    chosen = pred_phi.argmax(dim=-1, keepdim=True)       # (S, 1)
    return float((top == chosen).any(dim=-1).double().mean().item())


"""
Evaluation.
"""

@torch.no_grad()
def eval_split(
    model: AttentionModel,
    cells,
    z_mean: torch.Tensor,   # (C,) train-split delta mean, for the z-scored MSE
    z_std: torch.Tensor,    # (C,)
    mean_surface: torch.Tensor | None,
    chunk_grids: int = 128,
) -> dict[str, float]:
    """Every reported metric for one split, from whole-grid predictions."""
    model.regressor.eval()
    device = cells.t.device

    true_parts, pred_parts, base_parts = [], [], []
    for k in range(0, cells.n_grids, chunk_grids):
        sel = torch.arange(k, min(k + chunk_grids, cells.n_grids), device=device)
        img, src, tar, y = cells.gather_grids(sel)
        out = model.pred_cells(img, src, tar).double()
        base_parts.append(cells.grid_baseline[sel])
        pred_parts.append(deltas_from_preds(out, cells.grid_baseline[sel], mean_surface))
        true_parts.append(deltas_from_preds(y.double(), cells.grid_baseline[sel], mean_surface))

    true_d = torch.cat(true_parts)   # (S, n_cells, 2) deltas
    pred_d = torch.cat(pred_parts)
    baseline = torch.cat(base_parts)

    true_phi = calc_phi(true_d)
    pred_phi = calc_phi(pred_d)

    # Regression loss: MSE on deltas z-scored by the train split's per-metric
    # mean/std, so one unit of error weighs the same for PSNR and CLIP.
    zm = z_mean.to(true_d)
    zs = z_std.to(true_d).clamp_min(1e-12)
    mse_z = float(((((pred_d - zm) / zs) - ((true_d - zm) / zs)) ** 2).mean().item())

    # Selection.
    chosen = pred_phi.argmax(dim=-1, keepdim=True)
    gain = true_phi.gather(-1, chosen).squeeze(-1)   # true phi(default) == 0
    reg = true_phi.max(dim=-1).values - gain

    rho_phi_med, rho_phi_mean = median_spearman(true_phi.cpu().numpy(), pred_phi.cpu().numpy())

    out: dict[str, float] = {
        "n_samples": int(true_d.shape[0]),
        "n_cells": int(true_d.shape[1]),
        "regression_loss_zscored": mse_z,
        "regression_loss_phi_mse": float(((pred_phi - true_phi) ** 2).mean().item()),
        "ranking_loss": pairwise_ranking_loss(pred_phi, true_phi),
        "rho_phi_median": rho_phi_med,
        "rho_phi_mean": rho_phi_mean,
        "regret_median": float(reg.quantile(0.5).item()),
        "regret_p90": float(reg.quantile(0.9).item()),
        "gain_mean": float(gain.mean().item()),
        "deviate_rate": float((chosen.squeeze(-1) != baseline).double().mean().item()),
        "improvement_rate": float((gain > 0).double().mean().item()),
    }
    for n in (1, 2, 3, 4, 5):
        out[f"top{n}_accuracy"] = top_n_accuracy(true_phi, pred_phi, n)

    # Per-metric: R2 against the split's own mean, rank correlation, and the
    # true delta the picks land on.
    for i, name in enumerate(("psnr", "clip")):
        t_i, p_i = true_d[..., i], pred_d[..., i]
        ss_res = ((p_i - t_i) ** 2).sum()
        ss_tot = ((t_i - t_i.mean()) ** 2).sum().clamp_min(1e-12)
        med, _ = median_spearman(t_i.cpu().numpy(), p_i.cpu().numpy())
        # Removing each cell's population mean leaves only image-specific signal.
        med_img, _ = median_spearman(
            (t_i - t_i.mean(dim=0, keepdim=True)).cpu().numpy(),
            (p_i - p_i.mean(dim=0, keepdim=True)).cpu().numpy(),
        )
        out[f"r2_{name}"] = float((1 - ss_res / ss_tot).item())
        out[f"rho_{name}"] = med
        out[f"rho_{name}_image"] = med_img
        out[f"gain_{name}"] = float(t_i.gather(-1, chosen).squeeze(-1).mean().item())
        # How much the predictions move from sample to sample, per cell, relative
        # to how much the truth moves. A ratio near 0 means the model emits
        # essentially one surface for every input regardless of the image.
        out[f"spread_ratio_{name}"] = float(
            (p_i.std(dim=0).mean() / t_i.std(dim=0).mean().clamp_min(1e-12)).item()
        )

    return out


def main() -> None:
    args = _ARGS
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    weights_path = RUN_DIR / "regressor_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing {weights_path}")

    device = resolve_device()
    print(f"Run: {RUN_DIR}\nDevice: {device}")

    # Rebuild the exact splits train.py used: one shared entry point, so the
    # run's own SPLIT_SEED and PIE_BENCH snapshot drive the split here too.
    splits_df = build_splits_df()
    splits_cells = create_cell_tensors(splits_df, device)

    # Cross-check the split against the run's own id_to_split.csv.
    import pandas as pd
    saved = pd.read_csv(RUN_DIR / "id_to_split.csv", dtype={SAMPLE_ID_COL: str, "split": str})
    for name, (X, _) in splits_df.items():
        want = set(saved.loc[saved["split"] == name, SAMPLE_ID_COL])
        got = set(X[SAMPLE_ID_COL].astype(str))
        if want != got:
            raise ValueError(f"Split {name!r} does not match the run's id_to_split.csv "
                             f"({len(got)} rebuilt vs {len(want)} saved)")
    print("Splits match the run's id_to_split.csv.")

    # Load the checkpoint into a model sized from the token tables.
    train_cells = splits_cells["train"]
    img_shape = tuple(int(v) for v in train_cells.emb_table.img.shape[1:])
    text_dim = int(train_cells.emb_table.src.shape[-1])
    model = AttentionModel(img_shape, text_dim, train_cells.n_cells, device=device)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    if str(ckpt.get("prediction_space")) != str(PREDICTION_SPACE):
        raise ValueError(f"Checkpoint space {ckpt.get('prediction_space')!r} != {PREDICTION_SPACE!r}")

    # "residuals" re-anchors the targets on the train mean surface, exactly as
    # train.py does, before anything is scored.
    mean_surfaces: dict[str, torch.Tensor | None] = {name: None for name in splits_cells}
    if PREDICTION_SPACE == "residuals":
        surface = load_mean_surface(RUN_DIR)
        if surface is None:
            raise FileNotFoundError(f"Missing mean_surface.pt in {RUN_DIR}")
        for name, cells in splits_cells.items():
            mean_surfaces[name] = gather_at_pairs(surface, cells.t[cells.grid_rows[0]]).to(device=device, dtype=torch.float)
            cells.y.sub_(gather_at_pairs(surface, cells.t).to(cells.y))

    # z-scoring stats for the regression loss come from the train split only.
    y_train = train_cells.y[train_cells.grid_rows].double().reshape(-1, len(TARGET_COLS))
    z_mean, z_std = y_train.mean(0), y_train.std(0)
    print(f"Train delta z-scoring: mean {[round(v, 4) for v in z_mean.tolist()]} "
          f"std {[round(v, 4) for v in z_std.tolist()]}")

    results = {
        "run_dir": str(RUN_DIR),
        "prediction_space": str(PREDICTION_SPACE),
        "score_fn": str(SCORE_FN),
        "phi_alpha": float(PHI_ALPHA),
        "default_cell": [float(DEFAULT_T_START), float(DEFAULT_T_END)],
        "z_mean": z_mean.tolist(),
        "z_std": z_std.tolist(),
        "splits": {},
    }
    for name in [s.strip() for s in args.splits.split(",") if s.strip()]:
        m = eval_split(model, splits_cells[name], z_mean, z_std, mean_surfaces[name])
        results["splits"][name] = m
        print(f"\n[{name}] n={m['n_samples']}")
        for k, v in m.items():
            if k not in ("n_samples", "n_cells"):
                print(f"  {k:<28} {v:.4f}")

    out_path = args.out or (RUN_DIR / "full_metrics.json")
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
