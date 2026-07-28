"""
Train the metric surrogate M:

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
    create_dataloaders,
    load_df,
    model_inputs,
    prepare_df,
    save_split_df,
    split_df,
)
from _helpers import save_run_settings
from scores import calc_normalized_deltas
from model_m import MetricPredictor, format_results, pairwise_ranking_loss
from settings import *


def parse_args() -> argparse.Namespace:
    # Argument parser for the command line.
    parser = argparse.ArgumentParser(description="Train metric surrogate M")
    return parser.parse_args()


@torch.no_grad()
def evaluate(
    model: MetricPredictor,
    loader,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Per-target MAE/RMSE/R^2 in raw metric units, plus z-scored MSE loss."""
    
    if device is None:
        device = next(model.regressor.parameters()).device
    
    # Set the model to evaluation mode.
    model.regressor.eval()
    preds, trues = [], []
    loss_sum, n = 0.0, 0
    mean, std = model.regressor.target_mean, model.regressor.target_std

    # Evaluate the model on the given loader.
    for batch in loader:
        img, mask, src, tar, t, y = model_inputs(batch, device)
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


def train(
    model: MetricPredictor,
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    test_X: pd.DataFrame,
    test_y: pd.DataFrame,
) -> Path:
    """Train the metric surrogate model and save run artifacts."""

    # Create run directory to save information to.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_settings(run_dir)

    # Create dataloaders for the train, val, and test sets.
    use_ranking = RANKING_LOSS_WEIGHT > 0
    train_loader, val_loader, test_loader = create_dataloaders(model, train_X, train_y, val_X, val_y, test_X, test_y, group_train_by_sample=use_ranking)

    # Record encoder dimensions, then free VAE/text pipeline GPU memory when frozen.
    img_dim, text_dim = model.encoder_img_dim, model.encoder_text_dim
    if FREEZE_ENCODERS:
        model.release_encoders()

    target_cols = list(M_TARGET_COLS)
    y_train = torch.tensor(train_y[target_cols].values, dtype=torch.float)

    # Normalize the targets if specified.
    if NORMALIZE_TARGETS:
        # Store train mean/std so MSE is computed in z-scored space.
        # denormalize maps predictions back to raw metric units for ranking/eval.
        model.regressor.set_target_stats(y_train.mean(0), y_train.std(0))
    print(
        "Target columns (train):  "
        + "  ".join(
            f"{c}: mean={model.regressor.target_mean[i]:.3f} "
            f"std={model.regressor.target_std[i]:.3f}"
            for i, c in enumerate(M_TARGET_COLS)
        )
    )

    # Initialize the optimizer.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    y_mean, y_std = model.regressor.target_mean, model.regressor.target_std

    # Train the model.
    weights_out = run_dir / "regressor_weights.pt"
    best_val = float("inf")
    device = next(model.regressor.parameters()).device
    n_cells, n_samples = len(train_X), train_X[SAMPLE_ID_COL].nunique()
    
    # Iterate over the epochs.
    print("\n")
    for epoch in range(1, EPOCHS + 1):

        epoch_start = time.perf_counter()
        # Set the model to training mode.
        model.regressor.train()

        # Iterate over the train loader.
        for batch in train_loader:

            img, mask, src, tar, t, y = model_inputs(batch, device)
            out = model.regressor(img, mask, src, tar, t)

            # Standardize targets in z-scored space for MSE loss.
            mse = torch.nn.functional.mse_loss(out, (y - y_mean) / y_std)
            loss = mse

            # If specified, use per-sample ranking loss.
            if use_ranking:

                # Same apply_t_score (Δ then T_TARGET_SCORE) for true and pred order.
                # SampleGridBatchSampler yields one sample's full grid per batch.
                y_hat = model.regressor.denormalize(out)
                base_t_start = torch.as_tensor(DEFAULT_T_START, device=t.device, dtype=t.dtype)
                base_t_end = torch.as_tensor(DEFAULT_T_END, device=t.device, dtype=t.dtype)
                base_mask = (torch.isclose(t[:, 0], base_t_start) & torch.isclose(t[:, 1], base_t_end)).nonzero(as_tuple=False)
                if base_mask.numel() != 1:
                    raise ValueError(f"Expected exactly one default-(t_start,t_end) row in batch, found {int(base_mask.numel())}")
                baseline_idx = int(base_mask[0])
                
                # Calculate the true and predicted metrics.
                true_delta = calc_normalized_deltas(y, baseline_idx)
                pred_delta = calc_normalized_deltas(y_hat, baseline_idx)
                true_phi = T_TARGET_SCORE(true_delta)
                pred_phi = T_TARGET_SCORE(pred_delta)
                loss = loss + RANKING_LOSS_WEIGHT * pairwise_ranking_loss(pred_phi, true_phi)
            
            # Backpropagate the training loss.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Find val metrics every epoch and save best weights.
        val_results = evaluate(model, val_loader, device)
        improved = val_results["loss"] < best_val
        if improved:
            best_val = val_results["loss"]
            torch.save(
                {
                    "regressor_state_dict": model.regressor.state_dict(),
                    "target_mean": model.regressor.target_mean.cpu(),
                    "target_std": model.regressor.target_std.cpu(),
                    "target_cols": list(M_TARGET_COLS),
                    "img_dim": img_dim,
                    "text_dim": text_dim,
                },
                weights_out,
            )
        
        elapsed = time.perf_counter() - epoch_start
        train_results = evaluate(model, train_loader, device)
        print(
            f"Epoch [{epoch:03d}/{EPOCHS:03d}] | {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
            f"\n    {'Train:':<6} {format_results(train_results)}"
            f"\n    {'Val:':<6} {format_results(val_results)}"
            + ("  *" if improved else "")
        )

    # Load the best weights and evaluate the model on the test set.
    checkpoint = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(checkpoint["regressor_state_dict"])
    results = evaluate(model, test_loader, device)
    print(f"\n    {'Test:':<6} {format_results(results)}")

    # Save the splits and metrics.
    save_split_df(train_X, val_X, test_X, run_dir)
    metrics_out = run_dir / "m_train_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({"val_best_loss": best_val, "test": results}, f, indent=4)
    
    return run_dir


def main() -> None:

    # Parse arguments. NOTE: Currently unused.
    args = parse_args()
    
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # For data: load, prepare, and split into train/val/test sets.
    data_df = load_df()
    X_df, y_df = prepare_df(data_df)
    train_X, val_X, test_X, train_y, val_y, test_y = split_df(X_df, y_df)
    print(
        f"Splits: train={len(train_X)} cells ({train_X[SAMPLE_ID_COL].nunique()} samples)  "
        f"val={len(val_X)} cells ({val_X[SAMPLE_ID_COL].nunique()} samples)  "
        f"test={len(test_X)} cells ({test_X[SAMPLE_ID_COL].nunique()} samples)"
    )

    # Initialize and train the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricPredictor(device=device).to(device)
    run_dir = train(model, train_X, train_y, val_X, val_y, test_X, test_y)
    print(f"\nSaved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
