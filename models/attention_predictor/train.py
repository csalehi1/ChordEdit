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
from model import AttentionModel
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


def calc_loss(
    pred: torch.Tensor,                        # (G, n_cells, 2) TRAINING_* units
    y_raw: torch.Tensor,                       # (G, n_cells, 2) raw CLIP/PSNR
    selector: SelectorModel,
) -> torch.Tensor:
    """Weighted phi MSE, pairwise ranking, and per-column MSE on selection surfaces."""

    regressor = selector.model.regressor
    pred_sel = regressor.to_selector(regressor.to_raw(pred))
    true_sel = regressor.to_selector(y_raw)
    weights = None if PHI_WEIGHTS is None else pred_sel.new_tensor(PHI_WEIGHTS)
    pred_phi = selector.calc_phi(pred_sel, weights=weights)
    true_phi = selector.calc_phi(true_sel, weights=weights)
    
    loss = pred_phi.new_zeros(())
    if MSE_LOSS_WEIGHT > 0:
        loss = loss + MSE_LOSS_WEIGHT * mse_loss(pred_phi, true_phi, top_k=MSE_LOSS_TOP_K)
    if RANKING_LOSS_WEIGHT > 0:
        loss = loss + RANKING_LOSS_WEIGHT * ranking_loss(pred_phi, true_phi, top_k=RANKING_LOSS_TOP_K)
    if any(w > 0 for w in COL_LOSS_WEIGHTS):
        loss = loss + col_loss(pred_sel, true_sel, top_k=COL_LOSS_TOP_K)
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
    default_cell = dataset.default_cell

    # Iterate over the batches of the dataset.
    preds, ys_raw = [], []
    for batch in loader:
        pred = model.regressor(
            batch.image_tokens,
            batch.source_tokens,
            batch.target_tokens,
            batch.source_mask,
            batch.target_mask,
            batch.mask_features,
        )
        preds.append(pred)
        ys_raw.append(batch.y_raw)
    pred = torch.cat(preds)
    y_raw = torch.cat(ys_raw)

    regressor = model.regressor
    pred_sel = regressor.to_selector(regressor.to_raw(pred.double()))
    true_sel = regressor.to_selector(y_raw.double())
    true_phi = selector.calc_phi(true_sel)
    pred_phi = selector.calc_phi(pred_sel)
    selected = selector.select_deltas(
        pred_sel,
        delta_weights=TRAINING_DELTA_WEIGHTS,
        delta_floors=TRAINING_DELTA_FLOORS,
        phi_floor=TRAINING_PHI_FLOOR,
        temperature=TRAINING_TEMPERATURE,
    )

    return {
        # How well the predicted phi surface matches the true one.
        **training_metrics(
            true_phi, pred_phi,
            lambda: calc_loss(pred, y_raw, selector),
            lambda: mse_loss(pred_phi, true_phi, top_k=MSE_LOSS_TOP_K),
            lambda: ranking_loss(pred_phi, true_phi, top_k=RANKING_LOSS_TOP_K),
            lambda: col_loss(pred_sel, true_sel, top_k=COL_LOSS_TOP_K),
        ),
        # How well the selected cell compares to the true best cell.
        **selection_metrics(
            true_phi, selected, default_cell
        ),
        # Per-column training and selection metrics.
        **per_col_metrics(
            true_sel, y_raw.double(), pred_sel,
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
    )

    t_start_values = torch.as_tensor(np.sort(np.unique(metadata.cell_labels[:, 0].numpy())), dtype=torch.float64)
    t_end_values = torch.as_tensor(np.sort(np.unique(metadata.cell_labels[:, 1].numpy())), dtype=torch.float64)
    selector = SelectorModel(model, metadata.cell_labels.numpy())

    run = init_run(run_dir, {
        "n_cells": int(train.n_cells),
        "n_train_samples": int(train.n_samples),
        "n_val_samples": int(val.n_samples),
        "n_test_samples": int(test.n_samples),
    })

    model.regressor.set_metadata(metadata)
    print(
        "Target columns (train):\n"
        f"  {'Target':<38} {'Mean':>8} {'Std':>8}\n"
        + "\n".join(
            f"  {c:<38} "
            f"{model.regressor.zscore_mean[i]:8.3f} "
            f"{model.regressor.zscore_std[i]:8.3f}"
            for i, c in enumerate(TARGET_COLS)
        )
    )

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

                # Forward pass. Targets are z-scored to match the head outputs.
                out = model.regressor(
                    batch.image_tokens, 
                    batch.source_tokens, 
                    batch.target_tokens,
                    batch.source_mask, 
                    batch.target_mask,
                    batch.mask_features,
                )
                loss = calc_loss(
                    out, batch.y_raw, selector,
                )

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
                    "target_cols": list(TARGET_COLS),
                    "regressor_pred_space": str(TRAINING_PRED_SPACE),
                    "image_shape": train.image_shape,
                    "source_shape": train.source_shape,
                    "img_emb_pool": bool(IMG_EMB_POOL),
                    "feature_shape": train.feature_shape,
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
