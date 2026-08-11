# train_m.py

"""
Train the surrogate model M^ (paper: M^(x_src, c_src, c_tar, t*, t**) -> s):

    M(img_emb, mask_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)

Run from this directory:

    python train_m.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _data import (
    CellTensors,
    create_cell_tensors,
    load_df,
    prepare_df,
    save_split_df,
    split_df,
)
from _helpers import (
    calc_mean_surface,
    calc_phi,
    format_metric_table,
    gather_at_pairs,
    save_mean_surface,
    save_run_settings,
)
from model_m import SurrogateModel, pairwise_ranking_loss
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train metric surrogate M")
    # Read off argv by settings.py at import time, before this parser runs;
    # declared here so it shows up in --help and is not rejected as unknown.
    parser.add_argument("--settings-path", default=None, help="config file to use instead of ./settings.json")
    return parser.parse_args()


@torch.no_grad()
def eval_regression(
    model: SurrogateModel,
    cells: CellTensors,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Per-target MAE/RMSE/R^2 in target units, plus z-scored MSE loss."""
    
    # Evaluate in chunks of 16384 cells.
    EVAL_CHUNK = 16384

    # Set the model to evaluation mode.
    model.regressor.eval()
    preds, trues = [], []
    loss_sum, n = 0.0, 0
    mean, std = model.regressor.target_mean, model.regressor.target_std

    # Evaluate the model over the split's cells.
    for img, mask, src, tar, t, y in cells.iter_flat(EVAL_CHUNK):
        out = model.regressor(img, mask, src, tar, t)
        y_std = (y - mean) / std
        loss_sum += torch.nn.functional.mse_loss(out, y_std, reduction="sum").item()
        n += y.numel()
        preds.append(model.regressor.denormalize(out))
        trues.append(y)
    pred = torch.cat(preds)
    true = torch.cat(trues)
    err = pred - true

    # Calculate the metrics for each target column.
    metrics: dict[str, float] = {}
    metrics["loss"] = loss_sum / n
    for i, col in enumerate(M_TARGET_COLS):
        e = err[:, i]
        ss_res = (e ** 2).sum()
        ss_tot = ((true[:, i] - true[:, i].mean()) ** 2).sum().clamp(min=1e-12)
        metrics[f"mae_{col}"] = e.abs().mean().item()
        metrics[f"rmse_{col}"] = (e ** 2).mean().sqrt().item()
        metrics[f"r2_{col}"] = (1 - ss_res / ss_tot).item()
    
    return metrics


@torch.no_grad()
def eval_selection(
    model: SurrogateModel,
    cells: CellTensors,
    offset: torch.Tensor,
    chunk_grids: int = 256,
) -> dict[str, float]:
    """Evaluate how well predictions work for the selection model."""

    def _row_ranks(x: torch.Tensor) -> torch.Tensor:
        """Ordinal ranks along the last axis (ties broken by position)."""
        order = x.argsort(dim=-1)
        ranks = torch.empty_like(order)
        positions = torch.arange(x.shape[-1], device=x.device).expand_as(order)
        ranks.scatter_(-1, order, positions)
        return ranks.to(x.dtype)


    def _row_spearman(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Per-row Spearman rho between two (S, N) score matrices."""
        ra, rb = _row_ranks(a), _row_ranks(b)
        ra = ra - ra.mean(dim=-1, keepdim=True)
        rb = rb - rb.mean(dim=-1, keepdim=True)
        num = (ra * rb).sum(dim=-1)
        den = ra.norm(dim=-1) * rb.norm(dim=-1)
        return num / den.clamp_min(1e-12)

    model.regressor.eval()
    offset = offset.double().to(cells.t.device)
    true_delta_parts, pred_delta_parts, base_parts = [], [], []
    for k in range(0, cells.n_grids, chunk_grids):
        sel = torch.arange(k, min(k + chunk_grids, cells.n_grids), device=cells.t.device)
        img, mask, src, tar, t, _ = cells.gather_grids(sel)
        out = model.regressor.denormalize(model.regressor.forward_grid(img, mask, src, tar, t)).double()
        pred_delta_parts.append(out + offset)
        true_delta_parts.append(cells.y[cells.grid_rows[sel]].double() + offset)
        base_parts.append(cells.grid_baseline[sel])

    true_deltas = torch.cat(true_delta_parts)
    pred_deltas = torch.cat(pred_delta_parts)
    baseline = torch.cat(base_parts)
    true_phi = calc_phi(true_deltas)
    pred_phi = calc_phi(pred_deltas)

    rho = _row_spearman(true_phi, pred_phi)
    rho = rho[~rho.isnan()]
    chosen = pred_phi.argmax(dim=-1, keepdim=True)
    gain = true_phi.gather(-1, chosen).squeeze(-1)  # phi(default) == 0, so this is gain over the default
    reg = true_phi.max(dim=-1).values - gain

    # Per-metric rank agreement, and the true delta the picks land on: a selector
    # that trades CLIP away for PSNR shows up in these and not in phi.
    per_metric: dict[str, float] = {}
    for i, col in enumerate(("psnr", "clip")):
        t_i, p_i = true_deltas[..., i], pred_deltas[..., i]
        med = lambda x: float(x[~x.isnan()].quantile(0.5).item()) if x[~x.isnan()].numel() else float("nan")
        per_metric[f"{col}_rho"] = med(_row_spearman(t_i, p_i))
        per_metric[f"{col}_rho_imgspec"] = med(_row_spearman(t_i - t_i.mean(dim=0, keepdim=True), p_i - p_i.mean(dim=0, keepdim=True),
        ))
        per_metric[f"{col}_delta_at_pick"] = float(t_i.gather(-1, chosen).squeeze(-1).mean().item())

    # Use quantile(0.5) and not median() because torch's median takes the
    # lower of the two middle values, while numpy (and so train_t.py) averages them.
    return {
        "phi_spearman": float(rho.quantile(0.5).item()) if rho.numel() else float("nan"),
        "regret_median": float(reg.quantile(0.5).item()),
        "regret_p90": float(reg.quantile(0.9).item()),
        "gain_mean": float(gain.mean().item()),
        "deviate_rate": float((chosen.squeeze(-1) != baseline).double().mean().item()),
        **per_metric,
    }


def train(
    splits_df: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> Path:
    """Train the metric surrogate model and save run artifacts."""

    # Create run directory to save information to.
    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_settings(run_dir)
    print(f"Saved settings")

    # Build device-resident cell tensors for the train, val, and test sets.
    train_X = splits_df["train"][0]
    splits_cells = create_cell_tensors(splits_df, device)
    train_cells, val_cells, test_cells = splits_cells["train"], splits_cells["val"], splits_cells["test"]

    # Size the regressor from precomputed embedding dims (no ChordEdit encoders).
    img_dim = int(train_cells.emb_table.img.shape[1])
    text_dim = int(train_cells.emb_table.src.shape[1])
    model = SurrogateModel(img_dim, text_dim, device=device)

    # Calculate and save the train split's true mean delta surface. 
    # "residual" space predicts deviations from the this surface.
    surface = calc_mean_surface(train_cells)
    surface_path = save_mean_surface(run_dir, surface)
    print(f"Saved {surface_path.stem}")

    # Per-split grid-ordered offsets for phi (0s in "delta" space).
    offsets: dict[str, torch.Tensor] = {}
    for name, cells in splits_cells.items():
        if M_TARGET_SPACE == "residual":
            # Anchor each cell on the train split's mean value, so the towers
            # only have to predict how an image deviates from the mean surface. 
            offsets[name] = gather_at_pairs(surface, cells.t[cells.grid_rows[0]])
            cells.y.sub_(gather_at_pairs(surface, cells.t).to(cells.y))
        elif M_TARGET_SPACE == "delta":
            offsets[name] = torch.zeros(cells.n_cells, cells.y.shape[1], dtype=torch.float64)
        else:
            raise ValueError(f"Unknown {M_TARGET_SPACE=}")

    y_train = train_cells.y.detach().float().cpu()

    if NORMALIZE_TARGETS:
        # Store train mean/std so MSE is computed in z-scored space.
        model.regressor.set_target_stats(y_train.mean(0), y_train.std(0))
    print(
        "Target columns (train):\n"
        f"  {'Target':<38} {'Mean':>8} {'Std':>8}\n"
        + "\n".join(
            f"  {c:<38} "
            f"{model.regressor.target_mean[i]:8.3f} "
            f"{model.regressor.target_std[i]:8.3f}"
            for i, c in enumerate(M_TARGET_COLS)
        )
    )

    # Initialize the optimizer and (optional) cosine LR schedule.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS)) if LR_SCHEDULER == "cosine" else None)

    y_mean, y_std = model.regressor.target_mean, model.regressor.target_std
    loss_weights = torch.tensor([PSNR_LOSS_WEIGHT, CLIP_LOSS_WEIGHT], dtype=torch.float, device=device)
    loss_weights = loss_weights / loss_weights.mean()
    train_offset = offsets["train"].to(device=device, dtype=torch.float)

    # Train the model.
    weights_out = run_dir / "regressor_weights.pt"
    best_score = -float("inf")
    best_epoch, since_improved = 0, 0
    best_val_loss = float("inf")
    history: list[dict] = []
    n_cells, n_samples = len(train_X), train_X[SAMPLE_ID_COL].nunique()
    grids_per_batch = max(1, int(GRIDS_PER_BATCH))
    ema_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()} if EMA_DECAY > 0 else None

    # Iterate over the epochs.
    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.perf_counter()
        model.regressor.train()

        for (img, mask, src, tar, t, y), _ in train_cells.iter_grids(grids_per_batch, shuffle=True):

            # Predict the metric values for the timestep grid.
            out = model.regressor.forward_grid(img, mask, src, tar, t)

            # Weighted MSE loss on normalized targets.
            # TODO: Do we need different behavior if NORMALIZE_TARGETS=True?
            se = (out - (y - y_mean) / y_std) ** 2
            loss = (se * loss_weights).mean()

            # If specified, use per-sample ranking loss.
            if RANKING_LOSS_WEIGHT > 0:
                y_hat = model.regressor.denormalize(out)
                # Adding the train offset (zeros in the "delta" target space)
                # turns (residual) targets and predictions into the full
                # deltas phi consumes; the predicted grid is never normalized.
                true_delta, pred_delta = y + train_offset, y_hat + train_offset
                true_phi, pred_phi = calc_phi(true_delta), calc_phi(pred_delta)
                loss = loss + RANKING_LOSS_WEIGHT * pairwise_ranking_loss(pred_phi, true_phi, top_k=RANKING_TOP_K)

            # Backpropagate the training loss.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update the EMA state to maintain a smooth/stable copy of parameters.
            if ema_state is not None:
                with torch.no_grad():
                    for key, value in model.regressor.state_dict().items():
                        if value.dtype.is_floating_point:
                            ema_state[key].mul_(EMA_DECAY).add_(value.detach(), alpha=1.0 - EMA_DECAY)
                        else:
                            ema_state[key].copy_(value)

        if scheduler is not None:
            scheduler.step()

        # Evaluate the averaged weights when EMA is on.
        live_state = None
        if ema_state is not None:
            live_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()}
            model.regressor.load_state_dict(ema_state)

        regression_res = eval_regression(model, val_cells, device)
        selection_res = eval_selection(model, val_cells, offsets["val"])

        if CKPT_METRIC == "val_phi_spearman":
            score = selection_res.get("phi_spearman", float("nan"))
        elif CKPT_METRIC == "val_regret":
            score = -selection_res.get("regret_median", float("nan"))
        elif CKPT_METRIC == "val_gain_mean":
            score = selection_res.get("gain_mean", float("nan"))
        else:
            # Fallback to a regression-based metric.
            score = -regression_res["loss"]

        # Save the best weights if the checkpoint metric is improved.
        improved = score > best_score
        if improved:
            best_score, best_epoch, since_improved = score, epoch, 0
            best_val_loss = regression_res["loss"]
            torch.save({
                "regressor_state_dict": model.regressor.state_dict(),
                "target_mean": model.regressor.target_mean.cpu(),
                "target_std": model.regressor.target_std.cpu(),
                "target_cols": list(M_TARGET_COLS),
                "img_dim": img_dim,
                "text_dim": text_dim,
            }, weights_out)
        else:
            since_improved += 1

        # Evaluate the model on the train and val sets.
        train_results = eval_regression(model, train_cells, device)
        if live_state is not None:
            model.regressor.load_state_dict(live_state)
        history.append({
            "epoch": epoch, 
            "train": train_results, 
            "val": regression_res, 
            "val_selection": selection_res
        })
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch [{epoch:03d}/{EPOCHS:03d}]: {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
            + ("  *" if improved else "")
            + "\n"
            + format_metric_table([
                ("train", train_results, None),
                ("val", regression_res, selection_res),
            ])
        )

        # Early stop if the checkpoint metric has stalled.
        if EARLY_STOP_PATIENCE > 0 and epoch >= 5 and since_improved >= EARLY_STOP_PATIENCE:
            print(f"Early stop at epoch {epoch}: no improvement in {since_improved} epochs.")
            break

    # Load the best weights and evaluate the model on the test set.
    checkpoint = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(checkpoint["regressor_state_dict"])

    results = eval_regression(model, test_cells, device)
    test_sel = eval_selection(model, test_cells, offsets["test"])
    print("\n" + format_metric_table([("test", results, test_sel)]))

    # Save the splits and metrics.
    save_split_df(train_X, splits_df["val"][0], splits_df["test"][0], run_dir)
    metrics_out = run_dir / "m_train_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "ckpt_metric": str(CKPT_METRIC),
            "val_best_loss": best_val_loss,
            "val_best_selection": history[best_epoch - 1]["val_selection"] if history else {},
            "test": results,
            "test_selection": test_sel,
            "history": history,
        }, f, indent=4)

    return run_dir


def main() -> None:

    # Parse the command line arguments.
    args = parse_args()
    
    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load, prepare, and split the data into train/val/test sets.
    data_df = load_df()
    X_df, y_df = prepare_df(data_df)
    train_X, val_X, test_X, train_y, val_y, test_y = split_df(X_df, y_df)
    splits_df = {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}
    print(
        f"Dataset splits:\n"
        f"  train: {len(train_X)} cells ({train_X[SAMPLE_ID_COL].nunique()} samples)\n"
        f"  val: {len(val_X)} cells ({val_X[SAMPLE_ID_COL].nunique()} samples)\n"
        f"  test: {len(test_X)} cells ({test_X[SAMPLE_ID_COL].nunique()} samples)"
    )

    # Initialize the model and train it.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = train(splits_df, device)
    print(f"\nSaved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
