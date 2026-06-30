"""
Train the metric predictor:

    M(img_emb, src_emb, tar_emb, t_start, t_end) -> (psnr, clip)

Each grid-ablation cell is one training example. The source image, source
prompt, and target prompt are encoded once per sample (encoders are frozen),
then every (t_start, t_end) cell reuses those embeddings. The trainable MLP
regresses the two measured metrics. Targets are standardized with train-split
statistics; reported errors are in raw metric units.

Run from this directory:

    python train.py
"""

from __future__ import annotations

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
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

import settings
from model import MetricPredictor
from models.classification.classify import split_data
from settings import *


def _source_path(image_path: str) -> str:
    # Map a cell-image path to its sample's source.png.
    return str(Path(image_path).parents[SOURCE_IMAGE_PARENT_LEVEL] / SOURCE_IMAGE_NAME)


def _id_from_image_path(image_path: str) -> str:
    # Extract the 12-digit string-pair id from the sample folder name.
    sample_folder = Path(image_path).parents[SOURCE_IMAGE_PARENT_LEVEL].name
    return sample_folder.split("_")[-1]


def load_data() -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    metrics = pd.read_csv(METRICS_CSV)
    metrics = metrics.rename(columns={PSNR_COL: "psnr", CLIP_COL: "clip"})

    # Select a single t_delta slice if specified.
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics["t_delta"].values:
            raise ValueError(
                f"{TARGET_T_DELTA=} not found in t_delta "
                f"(distinct: {sorted(metrics['t_delta'].unique())})."
            )
        metrics = metrics[metrics["t_delta"] == TARGET_T_DELTA].copy()

    metrics["source_path"] = metrics[IMAGE_PATH_COL].map(_source_path)
    metrics["id"] = metrics[IMAGE_PATH_COL].map(_id_from_image_path)

    # Merge the metrics with the string pairs on the id.
    strings = pd.read_csv(STRINGS_CSV, dtype={"id": str})
    df = pd.merge(metrics, strings, on="id", how="left")
    if df["source_prompt"].isna().any():
        missing = df.loc[df["source_prompt"].isna(), "id"].unique()
        raise ValueError(f"No prompt strings found for ids: {missing.tolist()}")

    cols = [
        "sample_id", "id", "t_start", "t_end", "t_delta",
        "psnr", "clip", "source_path", "source_prompt", "target_prompt",
    ]
    return df[cols].reset_index(drop=True)


def precompute_embeddings(
    df: pd.DataFrame, predictor: MetricPredictor, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode the source image and prompt pair once per sample_id."""
    samples = df.drop_duplicates(subset="sample_id").sort_values("sample_id")
    images = [Image.open(p).convert("RGB") for p in samples["source_path"]]
    src_prompts = samples["source_prompt"].tolist()
    tar_prompts = samples["target_prompt"].tolist()

    img_emb = predictor.image_encoder(images).to(device)
    src_emb = predictor.text_encoder(src_prompts).to(device)
    tar_emb = predictor.text_encoder(tar_prompts).to(device)

    return {
        sid: {"img": img_emb[i], "src": src_emb[i], "tar": tar_emb[i]}
        for i, sid in enumerate(samples["sample_id"].tolist())
    }


def build_tensors(
    df: pd.DataFrame, emb: dict[str, dict[str, torch.Tensor]]
) -> tuple[torch.Tensor, ...]:
    """Assemble per-row (img, src, tar, t, y) tensors from cached embeddings."""
    img = torch.stack([emb[s]["img"] for s in df["sample_id"]])
    src = torch.stack([emb[s]["src"] for s in df["sample_id"]])
    tar = torch.stack([emb[s]["tar"] for s in df["sample_id"]])
    t = torch.tensor(df[["t_start", "t_end"]].values, dtype=torch.float)
    y = torch.tensor(df[list(TARGET_COLS)].values, dtype=torch.float)
    return img, src, tar, t, y


def save_splits(splits: dict[str, pd.DataFrame], run_dir: Path) -> None:
    """Save the train/val/test splits to parquet files"""
    for name, df in splits.items():
        out = run_dir / f"{name}.parquet.gz"
        df.to_parquet(out, compression="gzip", index=False)
        print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def _loader(tensors: tuple[torch.Tensor, ...], shuffle: bool) -> DataLoader:
    # Create a data loader for the batch size.
    return DataLoader(TensorDataset(*tensors), batch_size=BATCH_SIZE, shuffle=shuffle)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    """Per-target MAE/RMSE/R^2 in raw metric units, plus standardized loss."""
    model.regressor.eval()
    preds, trues = [], []
    loss_sum, n = 0.0, 0
    mean, std = model.regressor.target_mean, model.regressor.target_std
    for img, src, tar, t, y in loader:
        img, src, tar, t, y = (x.to(device) for x in (img, src, tar, t, y))
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
    for i, col in enumerate(TARGET_COLS):
        e = err[:, i]
        ss_res = (e ** 2).sum()
        ss_tot = ((true[:, i] - true[:, i].mean()) ** 2).sum().clamp(min=1e-12)
        metrics[f"mae_{col}"] = e.abs().mean().item()
        metrics[f"rmse_{col}"] = (e ** 2).mean().sqrt().item()
        metrics[f"r2_{col}"] = (1 - ss_res / ss_tot).item()
    return metrics


def _fmt(m: dict[str, float]) -> str:
    # Format the metrics as a string.
    return f"loss={m['loss']:.4f}  " + "  ".join(
        f"{col}: MAE={m[f'mae_{col}']:.3f} R2={m[f'r2_{col}']:.3f}"
        for col in TARGET_COLS
    )


def train() -> MetricPredictor:
    # Set the random seed for reproducibility.
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load the data and split it into train/val/test sets.
    df = load_data()
    train_df, val_df, test_df = split_data(df, seed=SEED)

    # Create a timestamped run directory and save the splits.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    save_splits({"train": train_df, "val": val_df, "test": test_df}, run_dir)

    # Get the device and print the dataset summary.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Dataset: {len(df)} cells from {df['sample_id'].nunique()} samples "
        f"(t_delta={TARGET_T_DELTA})  split: train={len(train_df)} / "
        f"val={len(val_df)} / test={len(test_df)}  device={device}"
    )

    # Create the model and move it to the device.
    model = MetricPredictor(freeze_encoders=FREEZE_ENCODERS, device=device)
    model.regressor.to(device)

    # Precompute the embeddings for all samples.
    emb = precompute_embeddings(df, model, device)
    # Build the tensors for the train/val/test sets.
    train_t = build_tensors(train_df, emb)
    val_t = build_tensors(val_df, emb)
    test_t = build_tensors(test_df, emb)

    # Standardize the targets with train-split statistics.
    if NORMALIZE_TARGETS:
        y_train = train_t[-1]
        model.regressor.set_target_stats(y_train.mean(0), y_train.std(0))
    print(
        "Target stats (train):  "
        + "  ".join(
            f"{TARGET_LABELS[c]}: mean={model.regressor.target_mean[i]:.3f} "
            f"std={model.regressor.target_std[i]:.3f}"
            for i, c in enumerate(TARGET_COLS)
        )
    )

    # Create the data loaders.
    train_loader = _loader(train_t, shuffle=True)
    val_loader = _loader(val_t, shuffle=False)
    test_loader = _loader(test_t, shuffle=False)

    # Create the optimizer.
    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mean, std = model.regressor.target_mean, model.regressor.target_std

    # Create the output file and train the model.
    weights_out = run_dir / "regressor_weights.pt"
    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # Train the regressor.
        model.regressor.train()
        for img, src, tar, t, y in train_loader:
            # Move the data to the device.
            img, src, tar, t, y = (x.to(device) for x in (img, src, tar, t, y))
            # Compute the predictions and loss.
            out = model.regressor(img, src, tar, t)
            loss = torch.nn.functional.mse_loss(out, (y - mean) / std)
            # Zero the gradients and step the optimizer.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate the model on the train/val sets.
        train_m = evaluate(model, train_loader, device)
        val_m = evaluate(model, val_loader, device)
        improved = val_m["loss"] < best_val
        # Save the best model weights so far.
        if improved:
            best_val = val_m["loss"]
            torch.save(
                {
                    "regressor_state_dict": model.regressor.state_dict(),
                    "target_mean": model.regressor.target_mean.cpu(),
                    "target_std": model.regressor.target_std.cpu(),
                    "target_cols": list(TARGET_COLS),
                    "img_dim": model.image_encoder.hidden_dim,
                    "text_dim": model.text_encoder.hidden_dim,
                    "config": {
                        k: (str(v) if isinstance(v, Path) else v)
                        for k, v in vars(settings).items()
                        if k.isupper() and not k.startswith("_")
                    },
                },
                weights_out,
            )
        print(
            f"Epoch {epoch:03d}  train: {_fmt(train_m)}  | val: {_fmt(val_m)}"
            + ("  *" if improved else "")
        )

    # Load the best model weights and evaluate on the test set.
    print(f"\nSaved {weights_out}  (best val loss={best_val:.4f})")
    ckpt = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    test_m = evaluate(model, test_loader, device)
    print(f"Test on *: {_fmt(test_m)}")
    return model


if __name__ == "__main__":
    train()
