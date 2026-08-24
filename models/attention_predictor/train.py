# train.py

"""
Train the grid surface predictor:

    predictor(x_src, c_src, c_tar) -> (n_cells, 2) grid of (psnr, clip)
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
from model import AttentionModel, deltas_from_preds
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train the grid surface predictor")
    # Read off argv by settings.py at import time, before this parser runs.
    parser.add_argument("--settings-path", default=None)
    return parser.parse_args()


def calc_loss(
    pred: torch.Tensor,                        # (G, n_cells, 2)
    true: torch.Tensor,                        # (G, n_cells, 2)
    baseline_idx: torch.Tensor,                # (G,)
    mean_surface: torch.Tensor | None = None,  # (n_cells, 2)
) -> torch.Tensor:
    """Calculate MSE between pred and true phi scores over the timestep grid."""
    pred_deltas = deltas_from_preds(pred, baseline_idx, mean_surface)
    true_deltas = deltas_from_preds(true, baseline_idx, mean_surface)
    pred_phi = calc_phi(pred_deltas)
    true_phi = calc_phi(true_deltas)
    return torch.nn.functional.mse_loss(pred_phi, true_phi)


"""
Evaluation.
"""

@torch.no_grad()
def eval_regression(
    model: AttentionModel,
    cells: CellTensors,
    device: torch.device | None = None,
    mean_surface: torch.Tensor | None = None,  # (n_cells, 2)
) -> dict[str, float]:
    """
    How well the predictor reproduces the two metric surfaces.

    loss         calc_loss over the split, grid-count weighted
    mae_<col>    mean absolute error, one key per target column
    rmse_<col>   root mean squared error
    r2_<col>     coefficient of determination against the split's own mean
    """

    # Evaluate in chunks of 256 grids.
    EVAL_CHUNK = 256

    model.regressor.eval()
    mean, std = model.regressor.target_mean, model.regressor.target_std

    preds, trues = [], []
    loss_sum, n_grids = 0.0, 0
    for (img, src, tar, y), baseline in cells.iter_grids(EVAL_CHUNK, shuffle=False):
        out = model.regressor(img, src, tar)
        y_std = (y - mean) / std
        # Weight each chunk by its grid count, since the last chunk is short.
        loss_sum += calc_loss(out, y_std, baseline, mean_surface).item() * y.shape[0]
        n_grids += y.shape[0]
        preds.append(model.regressor.destandardize(out).reshape(-1, y.shape[-1]))
        trues.append(y.reshape(-1, y.shape[-1]))
    pred = torch.cat(preds)
    true = torch.cat(trues)
    err = pred - true

    # Calculate the metrics for each target column.
    metrics: dict[str, float] = {}
    metrics["loss"] = loss_sum / max(n_grids, 1)
    for i, col in enumerate(TARGET_COLS):
        e = err[:, i]
        ss_res = (e ** 2).sum()
        ss_tot = ((true[:, i] - true[:, i].mean()) ** 2).sum().clamp(min=1e-12)
        metrics[f"mae_{col}"] = e.abs().mean().item()
        metrics[f"rmse_{col}"] = (e ** 2).mean().sqrt().item()
        metrics[f"r2_{col}"] = (1 - ss_res / ss_tot).item()

    return metrics


@torch.no_grad()
def eval_selection(
    model: AttentionModel,
    cells: CellTensors,
    chunk_grids: int = 256,
    mean_surface: torch.Tensor | None = None,  # (n_cells, 2)
) -> dict[str, float]:
    """How well the predicted surfaces serve selection, not regression.

        phi_spearman         median per-grid rank correlation of predicted vs
                             true phi; how well the whole surface is ordered
        regret_median/_p90   true phi lost by picking argmax(predicted phi)
                             instead of the true best cell; lower is better
        gain_mean            mean true phi at the picked cell. True phi at the
                             default cell is 0, so this is gain over the default
        improvement_rate     fraction of grids whose pick beats the default
        deviate_rate         fraction of grids that pick a non-default cell

    Plus, per metric (psnr, clip), how the pick treats that metric alone, so a
    predictor that trades CLIP away for PSNR shows up here and not in phi:

        rho_<col>            median rank correlation of that metric's surface
        rho_<col>_image      the same after removing the per-cell population
                             mean, i.e. only the image-specific variation
        delta_at_pick_<col>  mean true delta of that metric at the picked cell

    mean_surface is required under PREDICTION_SPACE "residuals": the targets in
    cells were residualized in-place by train(), so mapping both sides back to
    deltas (deltas_from_preds) needs the surface passed in.
    """

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
    device = cells.t.device
    surface = None if mean_surface is None else mean_surface.double().to(device)
    true_delta_parts, pred_delta_parts, base_parts = [], [], []
    for k in range(0, cells.n_grids, chunk_grids):
        sel = torch.arange(k, min(k + chunk_grids, cells.n_grids), device=device)
        img, src, tar, y = cells.gather_grids(sel)
        baseline = cells.grid_baseline[sel]
        out = model.pred_cells(img, src, tar).double()
        # Both sides leave PREDICTION_SPACE here, so phi sees deltas either way.
        pred_delta_parts.append(deltas_from_preds(out, baseline, surface))
        true_delta_parts.append(deltas_from_preds(y.double(), baseline, surface))
        base_parts.append(baseline)

    true_deltas = torch.cat(true_delta_parts)
    pred_deltas = torch.cat(pred_delta_parts)
    baseline = torch.cat(base_parts)
    true_phi = calc_phi(true_deltas)
    pred_phi = calc_phi(pred_deltas)

    rho = _row_spearman(true_phi, pred_phi)
    rho = rho[~rho.isnan()]
    chosen = pred_phi.argmax(dim=-1, keepdim=True)
    gain = true_phi.gather(-1, chosen).squeeze(-1)  # true phi(default) == 0
    reg = true_phi.max(dim=-1).values - gain

    # Per-metric rank agreement, and the true delta the picks land on: a selector
    # that trades CLIP away for PSNR shows up in these and not in phi.
    per_metric: dict[str, float] = {}
    for i, col in enumerate(("psnr", "clip")):
        t_i, p_i = true_deltas[..., i], pred_deltas[..., i]
        med = lambda x: float(x[~x.isnan()].quantile(0.5).item()) if x[~x.isnan()].numel() else float("nan")
        per_metric[f"rho_{col}"] = med(_row_spearman(t_i, p_i))
        per_metric[f"rho_{col}_image"] = med(_row_spearman(t_i - t_i.mean(dim=0, keepdim=True), p_i - p_i.mean(dim=0, keepdim=True)))
        per_metric[f"delta_at_pick_{col}"] = float(t_i.gather(-1, chosen).squeeze(-1).mean().item())

    # Use quantile(0.5) and not median() because torch's median takes the lower
    # of the two middle values, while numpy (and so selector.py) averages them.
    return {
        "phi_spearman": float(rho.quantile(0.5).item()) if rho.numel() else float("nan"),
        "regret_median": float(reg.quantile(0.5).item()),
        "regret_p90": float(reg.quantile(0.9).item()),
        "gain_mean": float(gain.mean().item()),
        "improvement_rate": float((gain > 0).double().mean().item()),
        "deviate_rate": float((chosen.squeeze(-1) != baseline).double().mean().item()),
        **per_metric,
    }


"""
Training.
"""

def train(
    splits_df: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> Path:
    """
    Train the grid surface predictor and save run artifacts.

    Saves regressor_weights.pt {regressor_state_dict, target_mean, target_std,
    target_cols, prediction_space, img_shape (C, S, S), text_shape (L, D),
    cell_t_pairs, t_start_values, t_end_values}, mean_surface.pt,
    id_to_split.csv, and regression_metrics.json.

    The mean surface is computed on the train split and always saved; under
    PREDICTION_SPACE "residuals" it is also subtracted from every split's
    targets here, so the heads regress deviations from it.
    """

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

    # Size the predictor from the token tables and the data's grid. cell_t_pairs
    # maps output cell k to its canonical sorted (t_start, t_end).
    img_shape = tuple(int(v) for v in train_cells.emb_table.img.shape[1:])
    text_shape = tuple(int(v) for v in train_cells.emb_table.src.shape[1:])
    cell_t_pairs = train_cells.t[train_cells.grid_rows[0]].detach().cpu()
    t_start_values = torch.as_tensor(np.sort(np.unique(cell_t_pairs[:, 0].numpy())), dtype=torch.float64)
    t_end_values = torch.as_tensor(np.sort(np.unique(cell_t_pairs[:, 1].numpy())), dtype=torch.float64)
    model = AttentionModel(img_shape, text_shape[-1], train_cells.n_cells, device=device)
    n_params = sum(p.numel() for p in model.regressor.parameters())
    print(
        f"Predictor: {n_params / 1e6:.2f}M params, "
        f"img {img_shape}, text {text_shape}, {train_cells.n_cells} cells, space {PREDICTION_SPACE!r}"
    )

    grid_shape = (len(t_start_values), len(t_end_values))

    run = init_run(run_dir, {
        "n_params": n_params,
        "img_shape": list(img_shape),
        "text_shape": list(text_shape),
        "n_cells": int(train_cells.n_cells),
        "grid": f"{grid_shape[0]}x{grid_shape[1]}",
        "n_train_samples": int(train_cells.n_grids),
        "n_val_samples": int(val_cells.n_grids),
        "n_test_samples": int(test_cells.n_grids),
    })

    # The mean surface is a mean *delta* surface, so it is only meaningful while
    # the targets are still deltas (see _helpers.calc_mean_surface).
    mean_surfaces: dict[str, torch.Tensor | None] = {name: None for name in splits_cells}
    if PREDICTION_SPACE != "raws":
        surface = calc_mean_surface(train_cells)
        surface_path = save_mean_surface(run_dir, surface)
        print(f"Saved {surface_path.stem}")
        if PREDICTION_SPACE == "residuals":
            # Anchor each cell on the train split's mean, so the heads only have
            # to predict how a sample deviates from the population surface.
            for name, cells in splits_cells.items():
                mean_surfaces[name] = gather_at_pairs(surface, cells.t[cells.grid_rows[0]]).to(device=device, dtype=torch.float)
                cells.y.sub_(gather_at_pairs(surface, cells.t).to(cells.y))

    # Only "raws" needs standardization; the delta spaces are already
    # commensurate and keep the buffers at identity.
    if PREDICTION_SPACE == "raws":
        y_train = train_cells.y.detach().float().cpu()
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

    try:
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

            # Iterate over the batches.
            for (img, src, tar, y), baseline in train_cells.iter_grids(grids_per_batch, shuffle=True):

                # Forward pass. Targets are z-scored to match the head outputs.
                out = model.regressor(img, src, tar)
                loss = calc_loss(out, (y - y_mean) / y_std, baseline, mean_surfaces["train"])

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
            train_regression = eval_regression(model, train_cells, device, mean_surfaces["train"])
            train_selection = eval_selection(model, train_cells, mean_surface=mean_surfaces["train"])
            val_regression = eval_regression(model, val_cells, device, mean_surfaces["val"])
            val_selection = eval_selection(model, val_cells, mean_surface=mean_surfaces["val"])

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
                    "prediction_space": str(PREDICTION_SPACE),
                    "img_shape": img_shape,
                    "text_shape": text_shape,
                    "cell_t_pairs": cell_t_pairs,
                    "t_start_values": t_start_values,
                    "t_end_values": t_end_values,
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
                f"\nEpoch [{epoch:03d}/{EPOCHS:03d}]: {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
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

        results = eval_regression(model, test_cells, device, mean_surfaces["test"])
        test_sel = eval_selection(model, test_cells, mean_surface=mean_surfaces["test"])
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
