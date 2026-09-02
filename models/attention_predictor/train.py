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
from dataloader import get_dataloader
from dataset import SplitDataset, get_dataset
from metrics import *
from model import AttentionModel, preds_to_deltas
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train the grid surface predictor")
    # Read off argv by settings.py at import time, before this parser runs.
    parser.add_argument("--settings-path", default=None)
    parser.add_argument(
        "--pie-bench",
        action="store_true",
        help="Replace the UltraEdit test split with PIE_Bench_v1 for the current CHORD_EDIT_MODEL",
    )
    return parser.parse_args()


def calc_train_phi(deltas: torch.Tensor) -> torch.Tensor:
    """Training-only phi, at TRAIN_PHI_ALPHA / TRAIN_PHI_WEIGHTS.

    Every reported metric keeps calling _helpers.calc_phi at the canonical
    PHI_ALPHA with equal weights, so reshaping the objective never moves the
    scoreboard. With both settings left null this is calc_phi exactly.
    """
    if TRAIN_PHI_WEIGHTS is None:
        return TRAIN_SCORE_PHI(deltas)
    weights = deltas.new_tensor(TRAIN_PHI_WEIGHTS)
    return TRAIN_SCORE_PHI(deltas, weights=weights)


def selected_cells(
    pred_deltas: torch.Tensor,  # (N, n_cells, 2)
    pred_phi: torch.Tensor,     # (N, n_cells)
    default_cell: int,
) -> torch.Tensor:              # (N,)
    """Cell each sample would be sent to, under the selection-time levers.

    Plain argmax of the predicted phi unless SELECTOR_DELTA_WEIGHTS reweights the
    ranking or SELECTOR_DELTA_FLOORS restricts it to cells clearing per-column
    delta floors. SELECTOR_PHI_FLOOR then keeps the default cell unless the
    chosen cell's phi gain clears that threshold, matching
    selector.SelectorModel.select_batch.
    """
    rank_phi = pred_phi
    if SELECTOR_DELTA_WEIGHTS is not None:
        rank_phi = calc_phi(pred_deltas, weights=pred_deltas.new_tensor(SELECTOR_DELTA_WEIGHTS))
    if SELECTOR_DELTA_FLOORS is not None:
        floors = pred_deltas.new_tensor([
            float("-inf") if f is None else f for f in SELECTOR_DELTA_FLOORS
        ])
        eligible = (pred_deltas >= floors).all(dim=-1)
        keep = eligible | ~eligible.any(dim=-1, keepdim=True)
        rank_phi = rank_phi.masked_fill(~keep, -float("inf"))
    chosen = rank_phi.argmax(dim=-1)
    if SELECTOR_PHI_FLOOR is not None:
        n = chosen.shape[0]
        rows = torch.arange(n, device=chosen.device)
        gain = rank_phi[rows, chosen] - rank_phi[:, default_cell]
        chosen = torch.where(
            gain > SELECTOR_PHI_FLOOR,
            chosen,
            chosen.new_full((n,), default_cell),
        )
    return chosen


def calc_loss(
    pred: torch.Tensor,                        # (G, n_cells, 2)
    true: torch.Tensor,                        # (G, n_cells, 2)
    default_cell: int,                         # shared index
    mean_surface: torch.Tensor,                # (n_cells, 2)
) -> torch.Tensor:
    """Weighted phi MSE, pairwise ranking, and per-column MSE.

    A weight of 0 drops that term. The first two live in phi space, where error
    trades freely between the two metric columns; column_loss is the only term
    that holds each head to its own column.
    """

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

    def column_loss(
        pred_deltas: torch.Tensor,
        true_deltas: torch.Tensor,
    ) -> torch.Tensor:
        """Per-column MSE on the delta surfaces, one weight per metric."""
        weights = pred_deltas.new_tensor([PSNR_LOSS_WEIGHT, CLIP_LOSS_WEIGHT])
        return (((pred_deltas - true_deltas) ** 2) * weights).mean()

    # Calculate the pred and true deltas from the pred and true surfaces.
    pred_deltas = preds_to_deltas(pred, default_cell, mean_surface)
    true_deltas = preds_to_deltas(true, default_cell, mean_surface)
    pred_phi = calc_train_phi(pred_deltas)
    true_phi = calc_train_phi(true_deltas)

    # Calculate the loss.
    loss = pred_phi.new_zeros(())
    if MSE_LOSS_WEIGHT > 0:
        loss = loss + MSE_LOSS_WEIGHT * mse_loss(pred_phi, true_phi, top_k=MSE_LOSS_TOP_K)
    if RANKING_LOSS_WEIGHT > 0:
        loss = loss + RANKING_LOSS_WEIGHT * ranking_loss(pred_phi, true_phi, top_k=RANKING_LOSS_TOP_K)
    if PSNR_LOSS_WEIGHT > 0 or CLIP_LOSS_WEIGHT > 0:
        loss = loss + column_loss(pred_deltas, true_deltas)
    
    return loss


"""
Evaluation.
"""

# Grids per forward pass during evaluation; larger than the train batch since
# no activations are kept.
EVAL_CHUNK = 256


@torch.no_grad()
def eval_regression(
    model: AttentionModel,
    dataset: SplitDataset,
) -> dict[str, float]:
    """
    How well the predictor reproduces the two metric surfaces.

    loss         calc_loss over the split, grid-count weighted
    per-col      see metrics.per_component_metrics
    """
    model.regressor.eval()
    mean, std = model.regressor.target_mean, model.regressor.target_std
    device = dataset.y.device

    preds = []
    loss_sum = 0.0
    for k in range(0, dataset.n_samples, EVAL_CHUNK):
        sel = torch.arange(k, min(k + EVAL_CHUNK, dataset.n_samples), device=device)
        image_tokens, source_tokens, target_tokens, source_mask, target_mask, y = dataset.gather(sel)
        out = model.regressor(image_tokens, source_tokens, target_tokens, source_mask, target_mask)
        # Weight each chunk by its grid count, since the last chunk is short.
        loss_sum += calc_loss(out, (y - mean) / std, dataset.default_cell, dataset.mean_surface).item() * len(sel)
        preds.append(model.regressor.destandardize(out))
    pred = torch.cat(preds)
    # No phi here: pick the cell with the best predicted primary column.
    chosen = pred[..., 0].argmax(dim=-1)

    return {
        "loss": loss_sum / max(dataset.n_samples, 1),
        **per_component_metrics(dataset.y, pred, TARGET_COLS, chosen, dataset.default_cell),
    }


@torch.no_grad()
def eval_selection(
    model: AttentionModel,
    dataset: SplitDataset,
) -> dict[str, float]:
    """
    How well the predicted surfaces serve selection, not regression.

    Predicts whole grids, maps both sides into delta space, and hands the phi
    surfaces to metrics.training_metrics and metrics.selection_metrics; each
    metric column is then scored by metrics.per_component_metrics.
    """
    model.regressor.eval()
    device = dataset.y.device
    surface = dataset.mean_surface.double()
    default_cell = dataset.default_cell

    pred_delta_parts = []
    for k in range(0, dataset.n_samples, EVAL_CHUNK):
        sel = torch.arange(k, min(k + EVAL_CHUNK, dataset.n_samples), device=device)
        image_tokens, source_tokens, target_tokens, source_mask, target_mask, _ = dataset.gather(sel)
        out = model.pred_cells(image_tokens, source_tokens, target_tokens, source_mask, target_mask).double()
        pred_delta_parts.append(preds_to_deltas(out, default_cell, surface))

    # Both sides leave PREDICTION_SPACE here, so phi sees deltas either way.
    pred_deltas = torch.cat(pred_delta_parts)
    true_deltas = preds_to_deltas(dataset.y.double(), default_cell, surface)
    true_phi = calc_phi(true_deltas)
    pred_phi = calc_phi(pred_deltas)
    chosen = selected_cells(pred_deltas, pred_phi, default_cell)

    return {
        **training_metrics(
            true_phi, pred_phi, default_cell,
            mse_weight=MSE_LOSS_WEIGHT,
            ranking_weight=RANKING_LOSS_WEIGHT,
            mse_top_k=MSE_LOSS_TOP_K,
            ranking_top_k=RANKING_LOSS_TOP_K,
        ),
        **selection_metrics(true_phi, pred_phi, default_cell),
        **per_component_metrics(true_deltas, pred_deltas, ("psnr", "clip"), chosen, default_cell),
        **comparison_metrics(true_phi, dataset.y_raw.double(), ("psnr", "clip"), chosen, default_cell),
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
    meta = bundle.meta
    train_X = bundle.splits_df["train"][0]
    test_note = f" (from {PIE_BENCH_DIR_NAME})" if PIE_BENCH else ""
    print(
        f"Dataset splits:\n"
        f"  train: {train.n_samples * meta.n_cells} cells ({train.n_samples} samples)\n"
        f"  val: {val.n_samples * meta.n_cells} cells ({val.n_samples} samples)\n"
        f"  test: {test.n_samples * test.y.shape[1]} cells ({test.n_samples} samples){test_note}"
    )

    # Size the predictor from the bundle's metadata
    model = AttentionModel(
        meta.img_shape, 
        meta.src_shape[-1],
        meta.n_cells, 
        device=device,
        default_cell=meta.default_cell
    )
    n_params = sum(p.numel() for p in model.regressor.parameters())
    print(
        f"Predictor: {n_params / 1e6:.2f}M params, "
        f"image {meta.img_shape}, text {meta.src_shape}, "
        f"{meta.n_cells} cells, space {PREDICTION_SPACE!r}, visual {IMG_EMB_TYPE!r}"
    )

    t_start_values = torch.as_tensor(np.sort(np.unique(meta.t[:, 0].numpy())), dtype=torch.float64)
    t_end_values = torch.as_tensor(np.sort(np.unique(meta.t[:, 1].numpy())), dtype=torch.float64)
    grid_shape = (len(t_start_values), len(t_end_values))

    run = init_run(run_dir, {
        "n_params": n_params,
        "image_shape": list(meta.img_shape),
        "source_shape": list(meta.src_shape),
        "n_cells": int(meta.n_cells),
        "grid": f"{grid_shape[0]}x{grid_shape[1]}",
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
    loader = get_dataloader(train, shuffle=True)

    try:
        # Train the model.
        weights_out = run_dir / "regressor_weights.pt"
        best_score = -float("inf")
        best_epoch, since_improved = 0, 0
        best_val_loss = float("inf")
        history: list[dict] = []
        n_cells, n_samples = len(train_X), train.n_samples
        ema_state = {k: v.detach().clone() for k, v in model.regressor.state_dict().items()} if EMA_DECAY > 0 else None

        # Iterate over the epochs.
        for epoch in range(1, EPOCHS + 1):
            epoch_start = time.perf_counter()
            model.regressor.train()

            # Iterate over the batches.
            for batch in loader:

                # Forward pass. Targets are z-scored to match the head outputs.
                out = model.regressor(
                    batch.image_tokens, 
                    batch.source_tokens, 
                    batch.target_tokens,
                    batch.source_mask, 
                    batch.target_mask,
                )
                loss = calc_loss(out, (batch.y - y_mean) / y_std, train.default_cell, train.mean_surface)

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
            train_regression = eval_regression(model, train)
            train_selection = eval_selection(model, train)
            val_regression = eval_regression(model, val)
            val_selection = eval_selection(model, val)

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
            elif CKPT_METRIC == "val_top1_accuracy":
                score = val_selection.get("top1_accuracy", float("nan"))
            elif CKPT_METRIC == "val_top5_accuracy":
                score = val_selection.get("top5_accuracy", float("nan"))
            elif CKPT_METRIC == "val_rho_phi_image":
                score = val_selection.get("rho_phi_image", float("nan"))
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
                    "image_shape": meta.img_shape,
                    "source_shape": meta.src_shape,
                    "cell_t_pairs": meta.t,
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

        results = eval_regression(model, test)
        test_sel = eval_selection(model, test)
        print("\n" + format_metric_table([("test", results, test_sel)]))

        # Summary rather than log, so the runs table ranks on final quality
        # instead of whatever the last epoch happened to produce.
        log_summary(
            run, results, test_sel,
            history[best_epoch - 1]["val_selection"] if history else {},
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
                "val_best_selection": history[best_epoch - 1]["val_selection"] if history else {},
                "test": results,
                "test_selection": test_sel,
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
