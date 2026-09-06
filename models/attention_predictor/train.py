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
import torch

from _helpers import *
from _wandb import finish_run, init_run, log_epoch, log_summary
from dataloader import SplitDatasetLoader, get_dataloader
from dataset import get_dataset
from metrics import *
from model import AttentionModel, preds_to_deltas
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
) -> torch.Tensor:
    """Mean softplus of inverted pairwise margins over pairs with true_u > true_v."""
    if pred_phi.shape[-1] < 2:
        return pred_phi.new_zeros(())
    diff_true = true_phi.unsqueeze(-1) - true_phi.unsqueeze(-2)
    diff_pred = pred_phi.unsqueeze(-1) - pred_phi.unsqueeze(-2)
    mask = diff_true > 0
    if top_k is not None and top_k < true_phi.shape[-1]:
        idx = true_phi.topk(top_k, dim=-1).indices
        is_top = torch.zeros_like(true_phi, dtype=torch.bool).scatter_(-1, idx, True)
        mask = mask & is_top.unsqueeze(-1)
    if not mask.any():
        return pred_phi.new_zeros(())
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


def col_loss(
    pred_deltas: torch.Tensor,
    true_deltas: torch.Tensor,
) -> torch.Tensor:
    """Per-column MSE on the delta surfaces, one weight per metric."""
    weights = pred_deltas.new_tensor([PSNR_LOSS_WEIGHT, CLIP_LOSS_WEIGHT])
    return (((pred_deltas - true_deltas) ** 2) * weights).mean()


def calc_loss(
    pred: torch.Tensor,                        # (G, n_cells, 2)
    true: torch.Tensor,                        # (G, n_cells, 2)
    default_cell: int,                         # shared index
    mean_surface: torch.Tensor,                # (n_cells, 2)
    selector: SelectorModel,
) -> torch.Tensor:
    """Weighted phi MSE, pairwise ranking, and per-column MSE."""
    
    # Convert the predictions and targets to deltas.
    pred_deltas = preds_to_deltas(pred, default_cell, mean_surface)
    true_deltas = preds_to_deltas(true, default_cell, mean_surface)
    weights = None if TRAIN_PHI_WEIGHTS is None else pred_deltas.new_tensor(TRAIN_PHI_WEIGHTS)
    pred_phi = selector.calc_phi(pred_deltas, weights=weights, phi_func=TRAIN_SCORE_PHI)
    true_phi = selector.calc_phi(true_deltas, weights=weights, phi_func=TRAIN_SCORE_PHI)
    
    loss = pred_phi.new_zeros(())
    if MSE_LOSS_WEIGHT > 0:
        loss = loss + MSE_LOSS_WEIGHT * mse_loss(pred_phi, true_phi, top_k=MSE_LOSS_TOP_K)
    if RANKING_LOSS_WEIGHT > 0:
        loss = loss + RANKING_LOSS_WEIGHT * ranking_loss(pred_phi, true_phi, top_k=RANKING_LOSS_TOP_K)
    if PSNR_LOSS_WEIGHT > 0 or CLIP_LOSS_WEIGHT > 0:
        loss = loss + col_loss(pred_deltas, true_deltas)
    return loss


"""
Evaluation.
"""


@torch.no_grad()
def eval(
    model: AttentionModel,
    loader: SplitDatasetLoader,
    selector: SelectorModel,
) -> dict[str, float]:
    """Regression and selection metrics for one split, from a single forward pass."""
    
    # Set the model to evaluation mode.
    model.regressor.eval()
    dataset = loader.dataset
    mean, std = model.regressor.target_mean, model.regressor.target_std
    default_cell = dataset.default_cell
    surface = dataset.mean_surface.double()

    # Iterate over the batches of the dataset.
    preds, ys, ys_raw = [], [], []
    for batch in loader:
        out = model.regressor(
            batch.image_tokens,
            batch.source_tokens,
            batch.target_tokens,
            batch.source_mask,
            batch.target_mask,
            batch.mask_features,
        )
        preds.append(model.regressor.destandardize(out))
        ys.append(batch.y)
        ys_raw.append(batch.y_raw)
    pred = torch.cat(preds)
    y = torch.cat(ys)
    y_raw = torch.cat(ys_raw)

    # Both sides leave PREDICTION_SPACE here, so phi sees deltas either way.
    pred_deltas = preds_to_deltas(pred.double(), default_cell, surface)
    true_deltas = preds_to_deltas(y.double(), default_cell, surface)
    true_phi = selector.calc_phi(true_deltas)
    pred_phi = selector.calc_phi(pred_deltas)
    selected = selector.select_deltas(pred_deltas)

    pred_std = (pred - mean) / std
    y_std = (y - mean) / std
    loss_pred_deltas = preds_to_deltas(pred_std, default_cell, dataset.mean_surface)
    loss_true_deltas = preds_to_deltas(y_std, default_cell, dataset.mean_surface)
    weights = None if TRAIN_PHI_WEIGHTS is None else loss_pred_deltas.new_tensor(TRAIN_PHI_WEIGHTS)
    loss_pred_phi = selector.calc_phi(loss_pred_deltas, weights=weights, phi_func=TRAIN_SCORE_PHI)
    loss_true_phi = selector.calc_phi(loss_true_deltas, weights=weights, phi_func=TRAIN_SCORE_PHI)

    return {
        # How well the predicted phi surface matches the true one.
        **training_metrics(
            true_phi, pred_phi,
            lambda: calc_loss(pred_std, y_std, default_cell, dataset.mean_surface, selector),
            lambda: mse_loss(loss_pred_phi, loss_true_phi, top_k=MSE_LOSS_TOP_K),
            lambda: ranking_loss(loss_pred_phi, loss_true_phi, top_k=RANKING_LOSS_TOP_K),
            lambda: col_loss(loss_pred_deltas, loss_true_deltas),
        ),
        # How well the selected cell compares to the true best cell.
        **selection_metrics(
            true_phi, selected, default_cell
        ),
        # Per-column training and selection metrics.
        **per_col_metrics(
            true_deltas, y_raw.double(), pred_deltas,
            selected, default_cell, TARGET_COLS,
        ),
    }


"""
Training.
"""

def train(device: torch.device) -> None:
    """Train the grid surface predictor and save run artifacts."""

    # Create run directory to save information to.
    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_settings(run_dir)
    print(f"Saved settings")

    # Build the device-resident datasets.
    bundle = get_dataset(device, run_dir)
    train, val, test = bundle.train, bundle.val, bundle.test
    metadata = bundle.metadata
    print(
        f"Dataset splits:\n"
        f"  train: {train.n_samples * metadata.n_cells} cells ({train.n_samples} samples)\n"
        f"  val: {val.n_samples * metadata.n_cells} cells ({val.n_samples} samples)\n"
        f"  test: {test.n_samples * test.y.shape[1]} cells ({test.n_samples} samples)"
    )

    # Size the predictor from the bundle's metadata
    model = AttentionModel(
        metadata.image_shape, 
        metadata.source_shape[-1],
        metadata.n_cells, 
        device=device,
        default_cell=metadata.default_cell,
        feat_dim=metadata.feature_shape[-1],
    )

    t_start_values = torch.as_tensor(np.sort(np.unique(metadata.cell_labels[:, 0].numpy())), dtype=torch.float64)
    t_end_values = torch.as_tensor(np.sort(np.unique(metadata.cell_labels[:, 1].numpy())), dtype=torch.float64)
    selector = SelectorModel(model, metadata.cell_labels.numpy(), train.mean_surface)

    run = init_run(run_dir, {
        "image_shape": list(metadata.image_shape),
        "source_shape": list(metadata.source_shape),
        "target_shape": list(metadata.target_shape),
        "n_cells": int(metadata.n_cells),
        "n_train_samples": int(train.n_samples),
        "n_val_samples": int(val.n_samples),
        "n_test_samples": int(test.n_samples),
    })

    # Only "raws" needs standardization as the delta spaces are already standardized.
    if PREDICTION_SPACE == "raws":
        y_train = train.y.detach().float().reshape(-1, train.y.shape[-1]).cpu()
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
        n_cells, n_samples = train.n_samples * metadata.n_cells, train.n_samples
        ema_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()} if EMA_DECAY > 0 else None

        # Iterate over the epochs.
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.regressor.train()

            # Iterate over the batches.
            for batch in train_loader:

                # Forward pass. Targets are z-scored to match the head outputs.
                out = model.regressor(
                    batch.image_tokens, 
                    batch.source_tokens, 
                    batch.target_tokens,
                    batch.source_mask, 
                    batch.target_mask,
                    batch.mask_features,
                )
                loss = calc_loss(out, (batch.y - y_mean) / y_std, train.default_cell, train.mean_surface, selector)

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
            if CKPT_METRIC == "val_phi_spearman":
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
                torch.save({
                    "regressor_state_dict": model.regressor.state_dict(),
                    "target_mean": model.regressor.target_mean.cpu(),
                    "target_std": model.regressor.target_std.cpu(),
                    "target_cols": list(TARGET_COLS),
                    "prediction_space": str(PREDICTION_SPACE),
                    "image_shape": metadata.image_shape,
                    "source_shape": metadata.source_shape,
                    "img_emb_pool": bool(IMG_EMB_POOL),
                    "feature_shape": metadata.feature_shape,
                    "use_zedit_mask": bool(USE_ZEDIT_MASK),
                    "cell_t_pairs": metadata.cell_labels,
                    "t_start_values": t_start_values,
                    "t_end_values": t_end_values,
                }, weights_out)
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

        # Load the best weights and evaluate on the test set.
        checkpoint = torch.load(weights_out, map_location=device, weights_only=False)
        model.regressor.load_state_dict(checkpoint["regressor_state_dict"])

        test_metrics = eval(model, test_loader, selector)
        print("\n" + format_metric_table([("test", test_metrics)]))

        # Summary rather than log, so the runs table ranks on final quality
        # instead of whatever the last epoch happened to produce.
        log_summary(
            run, test_metrics,
            history[best_epoch - 1]["val"] if history else {},
            best_epoch, len(history),
        )

        # Save metrics. Splits membership and mean surface were written by get_dataset.
        metrics_out = run_dir / "regression_metrics.json"
        with open(metrics_out, "w") as f:
            json.dump({
                "best_epoch": best_epoch,
                "epochs_ran": len(history),
                "ckpt_metric": str(CKPT_METRIC),
                "prediction_space": str(PREDICTION_SPACE),
                "val_best_loss": best_val_loss,
                "val_best": history[best_epoch - 1]["val"] if history else {},
                "test": test_metrics,
                "history": history,
            }, f, indent=4)

        print(f"\nSaved to {run_dir.resolve()}")

    finally:
        # Always close the run.
        finish_run(run)


def main() -> None:

    # Parse the command line arguments.
    args = parse_args()

    # Set the random seeds.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Splits are built inside train().
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)
    


if __name__ == "__main__":
    main()
