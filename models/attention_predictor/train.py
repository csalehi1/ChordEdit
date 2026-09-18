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

import numpy as np
import torch

from _helpers import *
from _wandb import finish_run, init_run, log_epoch, log_summary
from dataloader import SplitDatasetLoader, get_dataloader
from dataset import ID_TO_SPLIT_NAME, TRAIN_METADATA_NAME, get_dataset
from metrics import *
from model import AttentionModel, pin_default
from selector import SelectorModel
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train the grid surface predictor")
    # Read off argv by settings.py at import time, before this parser runs.
    parser.add_argument("--settings-path", default=None)
    parser.add_argument("--pie-bench", action="store_true")
    return parser.parse_args()


def mse_loss(
    pred_phi: torch.Tensor,
    true_phi: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """MSE over all cells, or only the true top-k when top_k is set."""
    if top_k is not None and top_k < true_phi.shape[-1]:
        idx = true_phi.topk(top_k, dim=-1).indices
        pred_phi = pred_phi.gather(-1, idx)
        true_phi = true_phi.gather(-1, idx)
    return torch.nn.functional.mse_loss(pred_phi, true_phi)


def ranking_loss(
    pred_phi: torch.Tensor,
    true_phi: torch.Tensor,
    top_k: int | None = None,
    chunk: int = 64,
) -> torch.Tensor:
    """
    Mean softplus of inverted pairwise margins over pairs with true_u > true_v.

    The mean is over every qualifying pair pooled across samples. Samples are
    processed `chunk` at a time so the (n, n_cells, n_cells) pair tensors never
    cover a whole split at once when eval() scores this on all of it.
    """
    n, n_cells = true_phi.shape
    if n_cells < 2:
        return pred_phi.new_zeros(())
    total = pred_phi.new_zeros(())
    n_pairs = 0
    for k in range(0, n, chunk):
        t, p = true_phi[k : k + chunk], pred_phi[k : k + chunk]
        diff_true = t.unsqueeze(-1) - t.unsqueeze(-2)
        diff_pred = p.unsqueeze(-1) - p.unsqueeze(-2)
        mask = diff_true > 0
        if top_k is not None and top_k < n_cells:
            idx = t.topk(top_k, dim=-1).indices
            is_top = torch.zeros_like(t, dtype=torch.bool).scatter_(-1, idx, True)
            mask = mask & is_top.unsqueeze(-1)
        if mask.any():
            n_pairs += int(mask.sum().item())
            total = total + torch.nn.functional.softplus(-diff_pred[mask]).sum()
    if n_pairs == 0:
        return pred_phi.new_zeros(())
    return total / n_pairs


def col_loss(
    pred_deltas: torch.Tensor,
    true_deltas: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """Per-column MSE on the delta surfaces, one weight per metric."""
    if top_k is not None and top_k < true_deltas.shape[-2]:
        idx = true_deltas.topk(top_k, dim=-2).indices
        pred_deltas = pred_deltas.gather(-2, idx)
        true_deltas = true_deltas.gather(-2, idx)
    weights = pred_deltas.new_tensor(COL_LOSS_WEIGHTS)
    return (((pred_deltas - true_deltas) ** 2) * weights).mean()


def dev_loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation of the predicted and true per-image deviations, per column.

    Each cell is centered across the samples of the batch, which removes the shared
    surface, so the term scores the direction of the per-image structure and not its
    amplitude, which the post-training calibration handles.
    """
    dp = pred - pred.mean(dim=0, keepdim=True)
    dt = true - true.mean(dim=0, keepdim=True)
    corr = (dp * dt).sum(dim=(0, 1)) / (dp.pow(2).sum(dim=(0, 1)).sqrt() * dt.pow(2).sum(dim=(0, 1)).sqrt()).clamp(min=1e-8)
    return (1 - corr).mean()


def calc_loss(
    pred: torch.Tensor,                        # (G, n_cells, 2) regression-space deltas
    y_raw: torch.Tensor,                       # (G, n_cells, 2) raw CLIP/PSNR
    selector: SelectorModel,
    cell_mask: torch.Tensor | None = None,     # (n_cells,) bool, or None for all
) -> torch.Tensor:
    """Weighted phi MSE, pairwise ranking, per-column MSE and deviation correlation, in the regression space."""

    true = selector.model.regressor.to_training(y_raw)
    weights = None if PHI_WEIGHTS is None else pred.new_tensor(PHI_WEIGHTS)
    pred_phi = selector.calc_phi(pred, weights=weights)
    true_phi = selector.calc_phi(true, weights=weights)
    if cell_mask is not None:
        pred, true, pred_phi, true_phi = pred[:, cell_mask], true[:, cell_mask], pred_phi[:, cell_mask], true_phi[:, cell_mask]

    loss = pred_phi.new_zeros(())
    if MSE_LOSS_WEIGHT > 0:
        loss = loss + MSE_LOSS_WEIGHT * mse_loss(pred_phi, true_phi)
    if RANKING_LOSS_WEIGHT > 0:
        loss = loss + RANKING_LOSS_WEIGHT * ranking_loss(pred_phi, true_phi, top_k=RANKING_LOSS_TOP_K)
    if any(w > 0 for w in COL_LOSS_WEIGHTS):
        loss = loss + col_loss(pred, true)
    if DEV_LOSS_WEIGHT > 0:
        loss = loss + DEV_LOSS_WEIGHT * dev_loss(pred, true)
    return loss


"""
Evaluation.
"""


@torch.no_grad()
def predict_split(model: AttentionModel, loader: SplitDatasetLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Regression-space predictions and raw labels for a whole split, in double precision."""
    model.regressor.eval()
    preds, ys_raw = [], []
    for batch in loader:
        preds.append(model.regressor(
            batch.image_tokens,
            batch.source_tokens,
            batch.target_tokens,
            batch.source_mask,
            batch.target_mask,
            batch.mask_features,
            batch.mask_tokens,
        ))
        ys_raw.append(batch.y_raw)
    return torch.cat(preds).double(), torch.cat(ys_raw).double()


@torch.no_grad()
def fit_deviation_calibration(model: AttentionModel, loader: SplitDatasetLoader) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean predicted raw surface and, per column, the slope of the true on the predicted deviation.

    Deviations are taken on deltas versus the default cell, the quantity the model
    predicts; the per-image level of the raw labels is not its to explain.
    """
    pred, y_raw = predict_split(model, loader)
    default_cell = loader.dataset.default_cell
    raw = model.to_raw(pred)
    mean_pred = raw.mean(dim=0)
    dev_pred = pin_default(raw, default_cell) - pin_default(mean_pred, default_cell)
    dev_true = pin_default(y_raw, default_cell) - pin_default(model.regressor.mean_surface.double(), default_cell)
    slope = (dev_true * dev_pred).sum(dim=(0, 1)) / dev_pred.pow(2).sum(dim=(0, 1)).clamp(min=1e-12)
    return mean_pred, slope


@torch.no_grad()
def eval(
    model: AttentionModel,
    loader: SplitDatasetLoader,
    selector: SelectorModel,
    calibrated: bool = SELECT_CALIBRATED,
) -> dict[str, float]:
    """Regression, deviation and selection metrics for one split, from a single forward pass."""

    default_cell = loader.dataset.default_cell
    regressor = model.regressor
    pred, y_raw = predict_split(model, loader)
    weights = None if PHI_WEIGHTS is None else pred.new_tensor(PHI_WEIGHTS)

    # Regression space, where the loss lives.
    true = regressor.to_training(y_raw)
    pred_phi_reg, true_phi_reg = selector.calc_phi(pred, weights=weights), selector.calc_phi(true, weights=weights)

    # Raw units, where the per-image deviation from the train-mean surface is measured.
    raw = regressor.to_raw(pred)
    if calibrated:
        raw = regressor.calibrate_raw(raw)
    mean_delta = pin_default(regressor.mean_surface.double(), default_cell)
    dev_true, dev_pred = pin_default(y_raw, default_cell) - mean_delta, pin_default(raw, default_cell) - mean_delta

    # Selection space: per-sample normalized deltas, phi, and the selected cells.
    pred_sel, true_sel = regressor.to_selector(raw), regressor.to_selector(y_raw)
    true_phi, pred_phi = selector.calc_phi(true_sel, weights=weights), selector.calc_phi(pred_sel, weights=weights)
    selected = selector.select_deltas(pred_sel)

    # Restrict the per-cell stats to t_start > t_end cells, if requested.
    cell_mask = None if selector.cell_mask is None else selector.cell_mask.to(device=pred.device)
    loss = calc_loss(pred, y_raw, selector, cell_mask=cell_mask)
    if cell_mask is not None:
        inv = selected.new_full((true_phi.shape[-1],), -1)
        inv[cell_mask] = torch.arange(int(cell_mask.sum()), device=selected.device, dtype=selected.dtype)
        selected, default_cell = inv[selected], int(inv[default_cell].item())
        pred, true, pred_phi_reg, true_phi_reg, y_raw, dev_true, dev_pred, pred_sel, true_sel, true_phi, pred_phi = (
            x[:, cell_mask] for x in (pred, true, pred_phi_reg, true_phi_reg, y_raw, dev_true, dev_pred, pred_sel, true_sel, true_phi, pred_phi)
        )

    return {
        # How well the predicted phi surface matches the true one, and the loss terms.
        **training_metrics(
            true_phi, pred_phi,
            lambda: loss,
            lambda: mse_loss(pred_phi_reg, true_phi_reg),
            lambda: ranking_loss(pred_phi_reg, true_phi_reg, top_k=RANKING_LOSS_TOP_K),
            lambda: col_loss(pred, true),
        ),
        # How well the per-image deviations from the shared surface are predicted.
        **deviation_metrics(dev_true, dev_pred, TARGET_COLS),
        # How well the selected cell compares to the true best cell.
        **selection_metrics(true_phi, selected, default_cell),
        # Per-column training and selection metrics.
        **per_col_metrics(true_sel, y_raw, pred_sel, selected, default_cell, TARGET_COLS),
    }


"""
Training.
"""

def train(device: torch.device) -> None:
    """Train the grid surface predictor and save run artifacts."""

    # Create run directory to save information to.
    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    if any((run_dir / name).exists() for name in (ID_TO_SPLIT_NAME, TRAIN_METADATA_NAME, "regressor_weights.pt")):
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = RUNS_DIR / run_name
        print(f"{run_dir} already exists. Falling back to {run_name}.")

    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_settings(run_dir)
    print("Saved settings")

    # Build the device-resident datasets.
    bundle = get_dataset(device, run_dir)
    train, val, test = bundle.train, bundle.val, bundle.test
    print(
        f"Dataset splits:\n"
        f"  train: {train.n_samples * train.n_cells} cells ({train.n_samples} samples)\n"
        f"  val: {val.n_samples * val.n_cells} cells ({val.n_samples} samples)\n"
        f"  test: {test.n_samples * test.y.shape[1]} cells ({test.n_samples} samples)"
    )

    # Size the predictor from train metadata only.
    metadata = train.metadata
    model = AttentionModel(
        train.image_shape, 
        train.source_shape[-1],
        metadata.n_cells, 
        device=device,
        default_cell=metadata.default_cell,
        feat_dim=train.feature_shape[-1],
        mask_dim=train.mask_shape[-1],
    )

    cell_labels = metadata.cell_labels.detach().cpu().numpy()
    t_start_values = torch.as_tensor(np.sort(np.unique(cell_labels[:, 0])), dtype=torch.float64)
    t_end_values = torch.as_tensor(np.sort(np.unique(cell_labels[:, 1])), dtype=torch.float64)
    selector = SelectorModel(model, cell_labels)
    cell_mask = None if selector.cell_mask is None else selector.cell_mask.to(device)

    run = init_run(run_dir, {
        "n_cells": int(train.n_cells),
        "n_train_samples": int(train.n_samples),
        "n_val_samples": int(val.n_samples),
        "n_test_samples": int(test.n_samples),
    })

    model.regressor.set_metadata(metadata)
    print(
        "Regression scale (raw units per delta unit):\n"
        + "\n".join(f"  {c:<38} {s:8.3f}" for c, s in zip(TARGET_COLS, metadata.loss_scale.tolist()))
    )

    def checkpoint() -> dict:
        """Weights plus the sizing contract selector.py rebuilds the model from."""
        return {
            "regressor_state_dict": model.regressor.state_dict(),
            "target_cols": list(TARGET_COLS),
            "image_shape": train.image_shape,
            "source_shape": train.source_shape,
            "feature_shape": train.feature_shape,
            "mask_shape": train.mask_shape,
            "img_emb_pool": bool(IMG_EMB_POOL),
            "img_fusion": IMG_FUSION,
            "mask_features": MASK_FEATURES,
            "cell_t_pairs": metadata.cell_labels,
            "t_start_values": t_start_values,
            "t_end_values": t_end_values,
        }

    # Initialize the optimizer and (optional) cosine LR schedule.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS)) if LR_SCHEDULER == "cosine" else None)

    train_loader = get_dataloader(train, shuffle=True)
    val_loader = get_dataloader(val, shuffle=False)
    test_loader = get_dataloader(test, shuffle=False)

    try:
        # Train the model.
        weights_out = run_dir / "regressor_weights.pt"
        best_score = -float("inf")
        best_epoch, since_improved = 0, 0
        best_val_loss = float("inf")
        history: list[dict] = []
        n_cells, n_samples = train.n_samples * train.n_cells, train.n_samples
        ema_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()} if EMA_DECAY > 0 else None

        # Iterate over the epochs.
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.regressor.train()

            # Iterate over the batches.
            for batch in train_loader:

                # Forward pass, scored in the regression space against the raw labels.
                out = model.regressor(
                    batch.image_tokens,
                    batch.source_tokens,
                    batch.target_tokens,
                    batch.source_mask,
                    batch.target_mask,
                    batch.mask_features,
                    batch.mask_tokens,
                )
                loss = calc_loss(out, batch.y_raw, selector, cell_mask=cell_mask)

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
            train_metrics = eval(model, train_loader, selector)
            val_metrics = eval(model, val_loader, selector)

            log_epoch(
                run, epoch, train_metrics, val_metrics,
                lr=optimizer.param_groups[0]["lr"],
                seconds=time.perf_counter() - epoch_start,
            )

            # Choose a checkpoint metric to gauge improvement.
            if CKPT_METRIC == "val_dev_corr":
                score = val_metrics.get("dev_corr", float("nan"))
            elif CKPT_METRIC == "val_phi_spearman":
                score = val_metrics.get("phi_spearman", float("nan"))
            elif CKPT_METRIC == "val_regret":
                score = -val_metrics.get("regret_median", float("nan"))
            elif CKPT_METRIC == "val_gain_mean":
                score = val_metrics.get("gain_mean", float("nan"))
            elif CKPT_METRIC == "val_top1_accuracy":
                score = val_metrics.get("top1_accuracy", float("nan"))
            elif CKPT_METRIC == "val_top5_accuracy":
                score = val_metrics.get("top5_accuracy", float("nan"))
            elif CKPT_METRIC == "val_rho_phi_image":
                score = val_metrics.get("rho_phi_image", float("nan"))
            else:
                # Fallback to a regression-based metric.
                score = -val_metrics["loss"]

            # Save the best weights if the checkpoint metric is improved.
            improved = score > best_score
            if improved:
                best_score, best_epoch, since_improved = score, epoch, 0
                best_val_loss = val_metrics["loss"]
                torch.save(checkpoint(), weights_out)
            else:
                since_improved += 1

            if live_state is not None:
                model.regressor.load_state_dict(live_state)
            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "val": val_metrics,
            })
            elapsed = time.perf_counter() - epoch_start
            print(
                f"\nEpoch [{epoch:03d}/{EPOCHS:03d}]: {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
                + ("  *" if improved else "")
                + "\n"
                + format_metric_table([
                    ("train", train_metrics),
                    ("val", val_metrics),
                ])
            )

            # Early stop if the checkpoint metric has stalled.
            if EARLY_STOP_PATIENCE > 0 and epoch >= 5 and since_improved >= EARLY_STOP_PATIENCE:
                print(f"No improvement in {since_improved} epochs. Early stopping at epoch {epoch}.")
                break

        # Load the best weights, calibrate the deviation amplitude on val (one slope per
        # column, kept in the checkpoint), and evaluate on the test set both ways.
        model.regressor.load_state_dict(torch.load(weights_out, map_location=device, weights_only=False)["regressor_state_dict"])
        model.regressor.set_calibration(*fit_deviation_calibration(model, val_loader))
        torch.save(checkpoint(), weights_out)

        test_metrics = eval(model, test_loader, selector)
        test_calibrated = eval(model, test_loader, selector, calibrated=True)
        print("\n" + format_metric_table([("test", test_metrics), ("test (cal)", test_calibrated)]))
        print("Deviation slopes (val): " + ", ".join(f"{c} {s:.3f}" for c, s in zip(TARGET_COLS, model.regressor.dev_scale.tolist())))

        # Summary rather than log, so the runs table ranks on final quality
        # instead of whatever the last epoch happened to produce.
        log_summary(
            run, test_metrics,
            history[best_epoch - 1]["val"] if history else {},
            best_epoch, len(history),
        )
        # Offline wandb keeps the summary in its binary log only, so the run dir
        # carries a plain JSON copy for the sweep and report tooling.
        (run_dir / "regression_metrics.json").write_text(json.dumps({
            "best_epoch": best_epoch,
            "epochs_ran": len(history),
            "train_metrics_seeds": TRAIN_METRICS_SEEDS,
            "eval_metrics_seeds": EVAL_METRICS_SEEDS,
            "dev_scale": model.regressor.dev_scale.tolist(),
            "test": test_metrics,
            "test_calibrated": test_calibrated,
            "val_best": history[best_epoch - 1]["val"] if history else {},
            "history": history,
        }, indent=2) + "\n")

        print(f"\nSaved to {run_dir.resolve()}")

    finally:
        # Always close the run.
        finish_run(run)


def main() -> None:

    # Parse the command line arguments (settings.py already read them off argv).
    parse_args()

    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Splits are built inside train().
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
    


if __name__ == "__main__":
    main()
