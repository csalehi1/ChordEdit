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
    IX_Y,
    create_dataloaders,
    load_df,
    model_inputs,
    prepare_df,
    save_splits_df,
    split_df,
)
from _helpers import format_results, save_settings_hash
from model_m import MetricPredictor
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
    """Per-target MAE/RMSE/R² in normalized metric units, plus standardized loss."""
    if device is None:
        device = next(model.parameters()).device
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
    metrics: dict[str, float] = {"loss": loss_sum / n}
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

    # Create dataloaders for the train, val, and test sets.
    train_loader, val_loader, test_loader = create_dataloaders(
        model, train_X, train_y, val_X, val_y, test_X, test_y
    )
    y_train = train_loader.dataset.tensors[IX_Y]
    print(f"Dataset: train={len(train_loader.dataset)} cells val={len(val_loader.dataset)} cells")

    # Normalize the targets if specified.
    if NORMALIZE_TARGETS:
        model.regressor.set_target_stats(y_train.mean(0), y_train.std(0))
    print(
        "Target stats (train):  "
        + "  ".join(
            f"{M_TARGET_LABELS[c]}: mean={model.regressor.target_mean[i]:.3f} "
            f"std={model.regressor.target_std[i]:.3f}"
            for i, c in enumerate(M_TARGET_COLS)
        )
    )

    # Initialize the optimizer.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mean, std = model.regressor.target_mean, model.regressor.target_std

    # Train the model.
    weights_out = run_dir / "regressor_weights.pt"
    best_val = float("inf")
    device = next(model.parameters()).device
    print(f"\nTraining for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
        model.regressor.train()
        for batch in train_loader:
            img, mask, src, tar, t, y = model_inputs(batch, device)
            out = model.regressor(img, mask, src, tar, t)
            y_std = (y - mean) / std
            loss = torch.nn.functional.mse_loss(out, y_std)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Evaluate the model on the train and val sets, save the best weights.
        train_results = evaluate(model, train_loader, device)
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
                    "img_dim": model.image_encoder.hidden_dim,
                    "text_dim": model.text_encoder.hidden_dim,
                },
                weights_out,
            )
        print(
            f"Epoch {epoch:03d}  train: {format_results(train_results)}  | val: {format_results(val_results)}"
            + ("  *" if improved else "")
        )

    # Load the best weights and evaluate the model on the test set.
    ckpt = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    results = evaluate(model, test_loader, device)
    print(f"Test: {format_results(results)}")

    # Save the run directory, splits, and metrics.
    save_settings_hash(run_dir)
    save_splits_df(train_X, val_X, test_X, run_dir)
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
    X, y = prepare_df(data_df)
    train_X, val_X, test_X, train_y, val_y, test_y = split_df(X, y)
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
