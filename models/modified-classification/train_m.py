"""
Train the metric surrogate M:

    M(img_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)

Run from this directory:

    python m_train.py
"""

from __future__ import annotations

import json
import os
import sys

# Add the repo root and this package to the Python path.
_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import settings
from data_io import build_tensors, load_data, precompute_embeddings
from _helpers import combined_score_bounds_from_df, split_data_by_sample, target_metric_torch
from model_m import MetricPredictor
from settings import *


def save_splits(splits: dict[str, pd.DataFrame], run_dir: Path) -> None:
    """Save train/val/test splits to parquet for t_train.py to reuse."""
    for name, df in splits.items():
        out = run_dir / f"{name}.parquet.gz"
        df.to_parquet(out, compression="gzip", index=False)
        print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def _loader(tensors: tuple[torch.Tensor, ...], shuffle: bool) -> DataLoader:
    # Wrap prebuilt tensors in a batched DataLoader.
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle)


def _ranking_loss(
    out_std: torch.Tensor,
    y_std: torch.Tensor,
    sample_idx: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    bounds: tuple[float, float, float, float],
) -> torch.Tensor:
    """Pairwise hinge loss: predicted m order should match true m within each sample."""
    from _helpers import CombinedScoreBounds

    b = CombinedScoreBounds(*bounds)
    pred = out_std * std + mean
    true = y_std * std + mean
    m_pred = target_metric_torch(pred, b)
    m_true = target_metric_torch(true, b)
    losses = []
    # Group batch rows by sample_id for within-grid pairwise comparisons.
    for sid in sample_idx.unique():
        mask = sample_idx == sid
        if mask.sum() < 2:
            continue
        mp = m_pred[mask]
        mt = m_true[mask]
        for i in range(len(mp)):
            for j in range(i + 1, len(mp)):
                if mt[i] == mt[j]:
                    continue
                sign = 1.0 if mt[i] > mt[j] else -1.0
                losses.append(torch.relu(sign * (mp[j] - mp[i])))
    if not losses:
        return out_std.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Per-target MAE/RMSE/R² in raw metric units, plus standardized loss."""
    model.regressor.eval()
    preds, trues = [], []
    loss_sum, n = 0.0, 0
    mean, std = model.regressor.target_mean, model.regressor.target_std
    for batch in loader:
        img, src, tar, t, y = (x.to(device) for x in batch[:5])
        out = model.regressor(img, src, tar, t)
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


def _fmt(m: dict[str, float]) -> str:
    # Format evaluation metrics as a single log line.
    return f"loss={m['loss']:.4f}  " + "  ".join(
        f"{col}: MAE={m[f'mae_{col}']:.3f} R2={m[f'r2_{col}']:.3f}"
        for col in M_TARGET_COLS
    )


def train(run_dir: Path | None = None) -> tuple[MetricPredictor, Path]:
    # Set the random seed for reproducibility.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load grid-ablation cells and split by sample_id (full grids stay intact).
    df = load_data()
    train_df, val_df, test_df = split_data_by_sample(
        df, seed=SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC
    )

    # Create a timestamped run directory and persist splits for t_train.py.
    if run_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OUTPUTS_DIR / timestamp
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_splits({"train": train_df, "val": val_df, "test": test_df}, run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Dataset: {len(df)} cells from {df['sample_id'].nunique()} samples "
        f"(t_delta={TARGET_T_DELTA})  split: train={len(train_df)} / "
        f"val={len(val_df)} / test={len(test_df)}  device={device}"
    )

    # Create the model; only the regressor MLP is trainable.
    model = MetricPredictor(freeze_encoders=FREEZE_ENCODERS, device=device)
    model.regressor.to(device)

    # Encode each (image, prompt pair) once; grid rows reuse cached embeddings.
    emb = precompute_embeddings(df, model, device)
    train_t = build_tensors(train_df, emb)
    val_t = build_tensors(val_df, emb)
    test_t = build_tensors(test_df, emb)

    # Standardize targets with train-split stats; also save scalar stats for T.
    y_train = train_t[4]
    if NORMALIZE_TARGETS:
        model.regressor.set_target_stats(y_train.mean(0), y_train.std(0))
    combined_score_bounds = combined_score_bounds_from_df(train_df)
    bounds_tuple = (
        combined_score_bounds.psnr_min,
        combined_score_bounds.psnr_max,
        combined_score_bounds.clip_min,
        combined_score_bounds.clip_max,
    )
    print(
        "Target stats (train):  "
        + "  ".join(
            f"{M_TARGET_LABELS[c]}: mean={model.regressor.target_mean[i]:.3f} "
            f"std={model.regressor.target_std[i]:.3f}"
            for i, c in enumerate(M_TARGET_COLS)
        )
    )

    train_loader = _loader(train_t, shuffle=True)
    val_loader = _loader(val_t, shuffle=False)
    test_loader = _loader(test_t, shuffle=False)

    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mean, std = model.regressor.target_mean, model.regressor.target_std

    weights_out = run_dir / "regressor_weights.pt"
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # Train the regressor on labeled grid cells.
        model.regressor.train()
        for batch in train_loader:
            img, src, tar, t, y, sample_idx = (x.to(device) for x in batch)
            out = model.regressor(img, src, tar, t)
            y_std = (y - mean) / std
            loss = torch.nn.functional.mse_loss(out, y_std)
            # Optional ranking loss aligns M with T's argmax objective.
            if RANKING_LOSS_WEIGHT > 0:
                loss = loss + RANKING_LOSS_WEIGHT * _ranking_loss(
                    out, y_std, sample_idx, mean, std, bounds_tuple
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_m = evaluate(model, train_loader, device)
        val_m = evaluate(model, val_loader, device)
        improved = val_m["loss"] < best_val
        # Save the best checkpoint by validation loss.
        if improved:
            best_val = val_m["loss"]
            torch.save(
                {
                    "regressor_state_dict": model.regressor.state_dict(),
                    "target_mean": model.regressor.target_mean.cpu(),
                    "target_std": model.regressor.target_std.cpu(),
                    "target_cols": list(M_TARGET_COLS),
                    "img_dim": model.image_encoder.hidden_dim,
                    "text_dim": model.text_encoder.hidden_dim,
                    "combined_score_bounds": bounds_tuple,
                    "config": {
                        k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(settings).items()
                        if k.isupper() and not k.startswith("_")
                    },
                },
                weights_out,
            )
        rank_note = f"  rank_w={RANKING_LOSS_WEIGHT}" if RANKING_LOSS_WEIGHT > 0 else ""
        print(
            f"Epoch {epoch:03d}  train: {_fmt(train_m)}  | val: {_fmt(val_m)}"
            + ("  *" if improved else "")
            + rank_note
        )

    # Reload best weights and report held-out test metrics.
    print(f"\nSaved {weights_out}  (best val loss={best_val:.4f})")
    ckpt = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    test_m = evaluate(model, test_loader, device)
    print(f"Test: {_fmt(test_m)}")

    with open(run_dir / "m_train_metrics.json", "w") as f:
        json.dump({"val_best_loss": best_val, "test": test_m}, f, indent=2)
    return model, run_dir


if __name__ == "__main__":
    train()
