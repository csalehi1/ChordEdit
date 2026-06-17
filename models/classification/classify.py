"""
Train OrdinalPairClassifier to predict max t_start and max t_end
from (source_prompt, target_prompt) pairs.

Rows are filtered to those matching DELTA_VALUE for t_delta, then grouped by
sample_id to find the maximum t_start and t_end per id. These aggregated values
are quantile-binned into 5 ordinal buckets (indices 0-4) matching the
BUCKETS = [0.0, 0.3, 0.6, 0.9, 1.0] scheme in classifier.py.
"""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from classifier import (
    OrdinalPairClassifier,
    ordinal_loss,
    decode_ordinal,
    mae_buckets,
)

METRICS_CSV = "id_to_metrics.csv"
STRINGS_CSV = "id_to_string_pair.csv"
WEIGHTS_OUT = "classifier_weights.pt"

TARGET_COLUMN = "clip_similarity_target_image"
DELTA_VALUE = 0.0

EPOCHS = 20
BATCH_SIZE = 32
LR = 1e-3
NUM_BINS = 5  # must match len(BUCKETS) in classifier.py


def load_data() -> pd.DataFrame:
    """Filter metrics to DELTA_VALUE rows, aggregate max t_start/t_end per id,
    and join with prompt strings. Bins each aggregate into NUM_BINS ordinal indices."""
    metrics = pd.read_csv(METRICS_CSV, dtype={"sample_id": str})
    strings = pd.read_csv(STRINGS_CSV, dtype={"id": str})

    aggregated = (
        metrics[metrics["t_delta"] == DELTA_VALUE]
        .groupby("sample_id", as_index=False)
        .agg(t_start_max=("t_start", "max"), t_end_max=("t_end", "max"))
    )

    df = pd.merge(aggregated, strings, left_on="sample_id", right_on="id")

    for col in ("t_start_max", "t_end_max"):
        df[f"{col}_idx"] = pd.qcut(df[col], q=NUM_BINS, labels=False, duplicates="drop").astype(int)

    return df


class PairDataset(Dataset):
    """Wraps (source_prompt, target_prompt, t_start_max_idx, t_end_max_idx) rows."""

    def __init__(self, df: pd.DataFrame):
        self.src = df["source_prompt"].tolist()
        self.tgt = df["target_prompt"].tolist()
        self.y1 = torch.tensor(df["t_start_max_idx"].values, dtype=torch.long)
        self.y2 = torch.tensor(df["t_end_max_idx"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx], self.y1[idx], self.y2[idx]


def collate(batch):
    """Collate a list of (src, tgt, y1, y2) tuples into batched tensors."""
    srcs, tgts, y1s, y2s = zip(*batch)
    return list(srcs), list(tgts), torch.stack(y1s), torch.stack(y2s)


def train() -> OrdinalPairClassifier:
    """Train the classifier and save weights to WEIGHTS_OUT."""
    df = load_data()
    print(f"Dataset: {len(df)} samples  (t_delta={DELTA_VALUE}, target={TARGET_COLUMN})")
    print(f"  t_start_max bins: {df['t_start_max_idx'].value_counts().sort_index().to_dict()}")
    print(f"  t_end_max   bins: {df['t_end_max_idx'].value_counts().sort_index().to_dict()}")

    dataset = PairDataset(df)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalPairClassifier(freeze_encoder=True).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for srcs, tgts, y1, y2 in loader:
            y1, y2 = y1.to(device), y2.to(device)
            l1, l2 = model(srcs, tgts)
            loss = ordinal_loss(l1, y1) + ordinal_loss(l2, y2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(srcs)

        model.eval()
        all_p1, all_p2, all_y1, all_y2 = [], [], [], []
        with torch.no_grad():
            for srcs, tgts, y1, y2 in loader:
                y1, y2 = y1.to(device), y2.to(device)
                l1, l2 = model(srcs, tgts)
                all_p1.append(decode_ordinal(l1))
                all_p2.append(decode_ordinal(l2))
                all_y1.append(y1)
                all_y2.append(y2)

        mae1 = mae_buckets(torch.cat(all_p1), torch.cat(all_y1)).item()
        mae2 = mae_buckets(torch.cat(all_p2), torch.cat(all_y2)).item()
        print(
            f"Epoch {epoch:02d}  loss={epoch_loss / len(dataset):.4f}"
            f"  MAE_t_start={mae1:.3f}  MAE_t_end={mae2:.3f}"
        )

    torch.save(model.state_dict(), WEIGHTS_OUT)
    print(f"Saved {WEIGHTS_OUT}")
    return model


if __name__ == "__main__":
    train()
