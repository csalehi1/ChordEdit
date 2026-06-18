"""
Train OrdinalPairClassifier to predict t_start and t_end
from (source_prompt, target_prompt) pairs.

Rows are filtered to those matching DELTA_VALUE for t_delta, then for each
sample_id the row with the highest combined_score is selected. The resulting
t_start and t_end values are quantile-binned into N_BINS ordinal buckets
passed to OrdinalPairClassifier.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.classification.model import (
    OrdinalPairClassifier,
    ordinal_loss,
    decode_ordinal,
    mae_buckets,
)
import models.classification.settings as _settings
from models.classification.settings import (
    BATCH_SIZE,
    COMPUTED_METRIC_COL,
    COMPUTED_METRIC_FN,
    COMPUTED_METRIC_LABEL,
    DATA_DIR,
    DELTA_VALUE,
    EPOCHS,
    ENCODER_LR,
    FREEZE_ENCODER,
    BODY_LR,
    METRICS_CSV,
    N_BUCKETS_END,
    N_BUCKETS_START,
    OUTPUTS_DIR,
    SEED,
    STRINGS_CSV,
    TARGET_COLUMN,
)


class PairDataset(Dataset):
    """Wraps (source_prompt, target_prompt, t_start_idx, t_end_idx) rows."""

    def __init__(self, df: pd.DataFrame):
        self.src = df["source_prompt"].tolist()
        self.tgt = df["target_prompt"].tolist()
        self.y1 = torch.tensor(df["t_start_idx"].values, dtype=torch.long)
        self.y2 = torch.tensor(df["t_end_idx"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx], self.y1[idx], self.y2[idx]


def load_data() -> pd.DataFrame:
    """Filter metrics to DELTA_VALUE rows, pick the highest TARGET_COLUMN row per id,
    and join with prompt strings. Maps discrete t_start/t_end values to ordinal indices."""
    metrics = pd.read_csv(METRICS_CSV, dtype={"sample_id": str})
    strings = pd.read_csv(STRINGS_CSV, dtype={"id": str})

    # Validate bucket counts on the original data
    for col, expected in (("t_start", N_BUCKETS_START), ("t_end", N_BUCKETS_END)):
        sorted_levels = sorted(metrics[col].unique())
        if len(sorted_levels) != expected:
            raise ValueError(
                f"{col} has {len(sorted_levels)} distinct values {sorted_levels}, "
                f"but expected {expected} from settings.py."
            )

    filtered = metrics[metrics["t_delta"] == DELTA_VALUE]
    best_idx = filtered.groupby("sample_id")[TARGET_COLUMN].idxmax()
    best = filtered.loc[best_idx, ["sample_id", "t_start", "t_end"]].reset_index(drop=True)

    df = pd.merge(best, strings, left_on="sample_id", right_on="id")

    # Build ordinal mappings from the full dataset, not the filtered one
    for col in ("t_start", "t_end"):
        sorted_levels = sorted(metrics[col].unique())
        level_to_index = {v: i for i, v in enumerate(sorted_levels)}
        df[f"{col}_idx"] = df[col].map(level_to_index)

    return df


def split_data(
    df: pd.DataFrame, seed: int = SEED
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df into train/val/test with an 80/10/10 ratio."""
    train = df.sample(frac=0.8, random_state=seed)
    remaining = df.drop(train.index)
    val = remaining.sample(frac=0.5, random_state=seed)
    test = remaining.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    data_dir: Path = DATA_DIR,
) -> None:
    """Save train/val/test splits as gzip-compressed Parquet files in data_dir."""
    data_dir.mkdir(exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out = data_dir / f"{name}.parquet.gz"
        df.to_parquet(out, compression="gzip", index=False)
        print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KB)")


def collate(batch):
    """Collate a list of (src, tgt, y1, y2) tuples into batched tensors."""
    srcs, tgts, y1s, y2s = zip(*batch)
    return list(srcs), list(tgts), torch.stack(y1s), torch.stack(y2s)


def eval_loader(
    model: OrdinalPairClassifier, loader: DataLoader, device: torch.device
    ) -> dict[str, float]:
    """Run model in eval mode over loader and return metric dict."""
    model.eval()
    all_p1, all_p2 = [], []
    all_l1, all_l2 = [], []
    with torch.no_grad():
        for srcs, tgts, l1, l2 in loader:
            l1 = l1.to(device)
            l2 = l2.to(device)
            out1, out2 = model(srcs, tgts)
            all_p1.append(decode_ordinal(out1))
            all_p2.append(decode_ordinal(out2))
            all_l1.append(l1)
            all_l2.append(l2)
    p1 = torch.cat(all_p1)
    p2 = torch.cat(all_p2)
    l1_out = torch.cat(all_l1)
    l2_out = torch.cat(all_l2)
    return {
        "mae_t_start": mae_buckets(p1, l1_out).item(),
        "mae_t_end": mae_buckets(p2, l2_out).item(),
        "acc_t_start": (p1 == l1_out).float().mean().item(),
        "acc_t_end": (p2 == l2_out).float().mean().item(),
        "acc_both": ((p1 == l1_out) & (p2 == l2_out)).float().mean().item(),
    }


def train() -> OrdinalPairClassifier:
    """Train the classifier and save weights + splits to a timestamped run directory."""
    df = load_data()
    train_df, val_df, test_df = split_data(df, seed=SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    save_splits(train_df, val_df, test_df, data_dir=run_dir)
    print(
        f"Dataset: {len(df)} samples  (t_delta={DELTA_VALUE}, target={TARGET_COLUMN})"
        f"  split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
    )

    train_loader = DataLoader(PairDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # Distinct ordered float values for each target, may differ between t_start and t_end
    buckets_start = torch.tensor(sorted(df["t_start"].unique()), dtype=torch.float)
    buckets_end = torch.tensor(sorted(df["t_end"].unique()), dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalPairClassifier(buckets1=buckets_start, buckets2=buckets_end, freeze_encoder=FREEZE_ENCODER).to(device)
    if FREEZE_ENCODER:
        optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=BODY_LR,
        )
    else:
        optimizer = torch.optim.Adam([
            {"params": model.encoder.parameters(), "lr": ENCODER_LR},
            {"params": list(model.body.parameters()) + list(model.head1.parameters()) + list(model.head2.parameters()), "lr": BODY_LR},
        ])

    weights_out = run_dir / "classifier_weights.pt"
    best_val_mae = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for srcs, tgts, y1, y2 in train_loader:
            y1 = y1.to(device)
            y2 = y2.to(device)
            l1, l2 = model(srcs, tgts)
            loss = ordinal_loss(l1, y1) + ordinal_loss(l2, y2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(srcs)

        val_metrics = eval_loader(model, val_loader, device)
        val_mae = val_metrics["mae_t_start"] + val_metrics["mae_t_end"]
        improved = val_mae < best_val_mae
        if improved:
            best_val_mae = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "buckets1": buckets_start.tolist(),
                    "buckets2": buckets_end.tolist(),
                    "config": {
                        "run_dir": str(run_dir),
                        **{
                            k: str(v) if isinstance(v, Path) else (v.__name__ if callable(v) else v)
                            for k, v in vars(_settings).items()
                            if k.isupper()
                        },
                    },
                },
                weights_out,
            )
        print(
            f"Epoch {epoch:02d}  loss={epoch_loss / len(train_df):.4f}"
            f"  val: MAE_start={val_metrics['mae_t_start']:.3f}  MAE_end={val_metrics['mae_t_end']:.3f}"
            f"  acc_start={val_metrics['acc_t_start']:.3f}  acc_end={val_metrics['acc_t_end']:.3f}  acc_both={val_metrics['acc_both']:.3f}"
            + ("  *" if improved else "")
        )

    print(f"Saved {weights_out}  (best val MAE={best_val_mae:.3f})")
    model.load_state_dict(torch.load(weights_out, map_location=device, weights_only=False)["state_dict"])

    test_metrics = eval_loader(model, test_loader, device)
    print(
        f"\nTest: MAE_start={test_metrics['mae_t_start']:.3f}  MAE_end={test_metrics['mae_t_end']:.3f}"
        f"  acc_start={test_metrics['acc_t_start']:.3f}  acc_end={test_metrics['acc_t_end']:.3f}  acc_both={test_metrics['acc_both']:.3f}"
    )
    return model


if __name__ == "__main__":
    train()
