# train.py

"""
Train the surrogate model M^ (paper: M^(x_src, c_src, c_tar) -> s):

    M(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)

Run from this directory:

    python train.py
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

from _data import *
from _helpers import *
from _wandb import finish_run, init_run, log_epoch, log_summary
from metrics import *
from model import SurrogateModel, pairwise_ranking_loss
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train metric surrogate M")
    # Read off argv by settings.py at import time, before this parser runs.
    parser.add_argument("--settings-path", default=None)
    return parser.parse_args()


@torch.no_grad()
def eval_regression(
    model: SurrogateModel,
    cells: CellTensors,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    How well the predictor reproduces the two metric surfaces.

    loss         z-scored MSE over the split's cells
    per-col      see metrics.per_component_metrics
    """

    # Evaluate in chunks of 256 grids.
    EVAL_CHUNK = 256

    # Set the model to evaluation mode.
    model.regressor.eval()
    preds, trues, bases = [], [], []
    loss_sum, n = 0.0, 0
    mean, std = model.regressor.target_mean, model.regressor.target_std

    # Evaluate the model over the split's grids.
    for (img, src, tar, y), baseline in cells.iter_grids(EVAL_CHUNK, shuffle=False):
        out = model.regressor(img, src, tar)
        y_std = (y - mean) / std
        loss_sum += torch.nn.functional.mse_loss(out, y_std, reduction="sum").item()
        n += y.numel()
        preds.append(model.regressor.destandardize(out))
        trues.append(y)
        bases.append(baseline)
    pred = torch.cat(preds)
    true = torch.cat(trues)
    baseline = torch.cat(bases)
    # No phi here: pick the cell with the best predicted primary column.
    chosen = pred[..., 0].argmax(dim=-1)

    return {
        "loss": loss_sum / n,
        **per_component_metrics(true, pred, TARGET_COLS, chosen, baseline),
    }


@torch.no_grad()
def eval_selection(
    model: SurrogateModel,
    cells: CellTensors,
    offset: torch.Tensor,
    chunk_grids: int = 256,
) -> dict[str, float]:
    """
    How well the predicted surfaces serve selection, not regression.

    Maps both sides into delta space and hands the phi surfaces to
    metrics.training_metrics and metrics.selection_metrics; each metric column
    is then scored by metrics.per_component_metrics.
    """

    model.regressor.eval()
    offset = offset.double().to(cells.t.device)
    true_delta_parts, pred_delta_parts, base_parts = [], [], []
    for k in range(0, cells.n_grids, chunk_grids):
        sel = torch.arange(k, min(k + chunk_grids, cells.n_grids), device=cells.t.device)
        img, src, tar, _ = cells.gather_grids(sel)
        out = model.regressor.destandardize(model.regressor(img, src, tar)).double()
        pred_delta_parts.append(out + offset)
        true_delta_parts.append(cells.y[cells.grid_rows[sel]].double() + offset)
        base_parts.append(cells.grid_baseline[sel])

    true_deltas = torch.cat(true_delta_parts)
    pred_deltas = torch.cat(pred_delta_parts)
    baseline = torch.cat(base_parts)
    true_phi = calc_phi(true_deltas)
    pred_phi = calc_phi(pred_deltas)
    chosen = pred_phi.argmax(dim=-1)
    true_raw = cells.y_raw[cells.grid_rows].double()

    return {
        **training_metrics(
            true_phi, pred_phi, baseline,
            mse_weight=1.0,
            ranking_weight=RANKING_LOSS_WEIGHT,
            ranking_top_k=(None if RANKING_TOP_K <= 0 else RANKING_TOP_K),
        ),
        **selection_metrics(true_phi, pred_phi, baseline),
        **per_component_metrics(true_deltas, pred_deltas, ("psnr", "clip"), chosen, baseline),
        **comparison_metrics(true_phi, true_raw, ("psnr", "clip"), chosen, baseline),
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

    # Size the regressor from precomputed embedding dims and the data's grid.
    # cell_t_pairs maps output cell k to its canonical sorted (t_start, t_end).
    img_shape = tuple(int(v) for v in train_cells.emb_table.img.shape[1:])
    text_shape = tuple(int(v) for v in train_cells.emb_table.src.shape[1:])
    cell_t_pairs = train_cells.t[train_cells.grid_rows[0]].detach().cpu()
    model = SurrogateModel(img_shape, int(text_shape[-1]), train_cells.n_cells, device=device)

    # Calculate and save the train split's true mean delta surface. 
    # "residuals" space predicts deviations from the this surface.
    surface = calc_mean_surface(train_cells)
    surface_path = save_mean_surface(run_dir, surface)
    print(f"Saved {surface_path.stem}")

    # Per-split grid-ordered offsets for phi (0s in "deltas" space).
    offsets: dict[str, torch.Tensor] = {}
    for name, cells in splits_cells.items():
        if PREDICTION_SPACE == "residuals":
            # Anchor each cell on the train split's mean value, so the towers
            # only have to predict how an image deviates from the mean surface. 
            offsets[name] = gather_at_pairs(surface, cells.t[cells.grid_rows[0]])
            cells.y.sub_(gather_at_pairs(surface, cells.t).to(cells.y))
        elif PREDICTION_SPACE == "deltas":
            offsets[name] = torch.zeros(cells.n_cells, cells.y.shape[1], dtype=torch.float64)
        else:
            raise ValueError(f"Unknown {PREDICTION_SPACE=}")

    y_train = train_cells.y.detach().float().cpu()

    if NORMALIZE_TARGETS:
        # Store train mean/std so MSE is computed in z-scored space.
        model.regressor.set_target_standardization(y_train.mean(0), y_train.std(0))
    print(
        "Target columns (train):\n"
        f"  {'Target':<38} {'Mean':>8} {'Std':>8}\n"
        + "\n".join(
            f"  {c:<38} "
            f"{model.regressor.target_mean[i]:8.3f} "
            f"{model.regressor.target_std[i]:8.3f}"
            for i, c in enumerate(TARGET_COLS)
        )
    )

    # Initialize the optimizer and (optional) cosine LR schedule.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS)) if LR_SCHEDULER == "cosine" else None)

    y_mean, y_std = model.regressor.target_mean, model.regressor.target_std
    loss_weights = torch.tensor([PSNR_LOSS_WEIGHT, CLIP_LOSS_WEIGHT], dtype=torch.float, device=device)
    loss_weights = loss_weights / loss_weights.mean()
    train_offset = offsets["train"].to(device=device, dtype=torch.float)

    n_params = sum(p.numel() for p in model.regressor.parameters())
    run = init_run(run_dir, {
        "n_params": n_params,
        "img_shape": list(img_shape),
        "text_shape": list(text_shape),
        "n_cells": int(train_cells.n_cells),
        "grid": f"{len(surface['t_start_values'])}x{len(surface['t_end_values'])}",
        "n_train_samples": int(train_cells.n_grids),
        "n_val_samples": int(val_cells.n_grids),
        "n_test_samples": int(test_cells.n_grids),
    })

    try:
        # Train the model.
        weights_out = run_dir / "regressor_weights.pt"
        best_score = -float("inf")
        best_epoch, since_improved = 0, 0
        best_val_loss = float("inf")
        history: list[dict] = []
        n_cells, n_samples = len(train_X), train_X[SAMPLE_ID_COL].nunique()
        grids_per_batch = max(1, int(SAMPLES_PER_BATCH))
        ema_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()} if EMA_DECAY > 0 else None

        # Iterate over the epochs.
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.regressor.train()

            # Iterate over the batches.
            for (img, src, tar, y), _ in train_cells.iter_grids(grids_per_batch, shuffle=True):

                # Forward pass.
                out = model.regressor(img, src, tar)

                # Compute the loss.
                se = (out - (y - y_mean) / y_std) ** 2
                loss = (se * loss_weights).mean()
                if RANKING_LOSS_WEIGHT > 0:
                    y_hat = model.regressor.destandardize(out)
                    # Add the train offset to convert to delta target space.
                    true_delta, pred_delta = y + train_offset, y_hat + train_offset
                    true_phi, pred_phi = calc_phi(true_delta), calc_phi(pred_delta)
                    loss = loss + RANKING_LOSS_WEIGHT * pairwise_ranking_loss(pred_phi, true_phi, top_k=RANKING_TOP_K)

                # Backpropagate the loss.
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

            # Evaluate regression and selection on train and val.
            train_regression = eval_regression(model, train_cells, device)
            train_selection = eval_selection(model, train_cells, offsets["train"])
            val_regression = eval_regression(model, val_cells, device)
            val_selection = eval_selection(model, val_cells, offsets["val"])

            log_epoch(
                run, epoch, train_regression, train_selection, val_regression, val_selection,
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - epoch_start,
            )

            # Choose a checkpoint metric to gauge improvement.
            if CKPT_METRIC == "val_phi_spearman":
                score = val_selection.get("phi_spearman", float("nan"))
            elif CKPT_METRIC == "val_regret":
                score = -val_selection.get("regret_median", float("nan"))
            elif CKPT_METRIC == "val_gain_mean":
                score = val_selection.get("gain_mean", float("nan"))
            else:
                # Fallback to a regression-based metric.
                score = -val_regression["loss"]

            # Save the best weights if the checkpoint metric is improved.
            improved = score > best_score
            if improved:
                best_score, best_epoch, since_improved = score, epoch, 0
                best_val_loss = val_regression["loss"]
                torch.save({
                    "regressor_state_dict": model.regressor.state_dict(),
                    "target_mean": model.regressor.target_mean.cpu(),
                    "target_std": model.regressor.target_std.cpu(),
                    "target_cols": list(TARGET_COLS),
                    "img_shape": list(img_shape),
                    "text_shape": list(text_shape),
                    "cell_t_pairs": cell_t_pairs,
                }, weights_out)
            else:
                since_improved += 1

            if live_state is not None:
                model.regressor.load_state_dict(live_state)
            history.append({
                "epoch": epoch,
                "train": train_regression,
                "train_selection": train_selection,
                "val": val_regression,
                "val_selection": val_selection,
            })
            elapsed = time.perf_counter() - epoch_start
            print(
                f"Epoch [{epoch:03d}/{EPOCHS:03d}]: {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
                + ("  *" if improved else "")
                + "\n"
                + format_metric_table([
                    ("train", train_regression, train_selection),
                    ("val", val_regression, val_selection),
                ])
            )

            # Early stop if the checkpoint metric has stalled.
            if EARLY_STOP_PATIENCE > 0 and epoch >= 5 and since_improved >= EARLY_STOP_PATIENCE:
                print(f"No improvement in {since_improved} epochs. Early stopping at epoch {epoch}.")
                break

        # Load the best weights and evaluate on the test set.
        checkpoint = torch.load(weights_out, map_location=device, weights_only=False)
        model.regressor.load_state_dict(checkpoint["regressor_state_dict"])

        results = eval_regression(model, test_cells, device)
        test_sel = eval_selection(model, test_cells, offsets["test"])
        print("\n" + format_metric_table([("test", results, test_sel)]))

        # Summary rather than log, so the runs table ranks on final quality
        # instead of whatever the last epoch happened to produce.
        log_summary(
            run, results, test_sel,
            history[best_epoch - 1]["val_selection"] if history else {},
            best_epoch, len(history),
        )

        # Save the splits and metrics.
        save_splits_df(train_X, splits_df["val"][0], splits_df["test"][0], run_dir)
        metrics_out = run_dir / "regression_metrics.json"
        with open(metrics_out, "w") as f:
            json.dump({
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "ckpt_metric": str(CKPT_METRIC),
                "prediction_space": str(PREDICTION_SPACE),
                "val_best_loss": best_val_loss,
                "val_best_selection": history[best_epoch - 1]["val_selection"] if history else {},
                "test": results,
                "test_selection": test_sel,
                "history": history,
            }, f, indent=4)

        return run_dir

    finally:
        # Always close the run: a crash mid-training should still leave a
        # finished (and correctly marked) run rather than a dangling one.
        finish_run(run)


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
