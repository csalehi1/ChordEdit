"""
Train OrdinalPairClassifier to predict t_start and t_end
from (source_prompt, target_prompt) pairs.

Rows are filtered to those matching T_DELTA_TARGET for t_delta, then for each
sample_id the row with the highest TARGET_COLUMN is selected. The resulting
t_start and t_end values are quantile-binned into N_BINS ordinal buckets
passed to OrdinalPairClassifier.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from models.classification.model import OrdinalPairClassifier, mae_buckets
from models.classification.head_coral import ordinal_loss
from models.classification.head_mse import regression_loss
from models.classification.head_ce import one_hot_ce_loss
import models.classification.settings as _settings
from models.classification.settings import (
    BATCH_SIZE,
    COMPUTED_METRIC_COL,
    COMPUTED_METRIC_FN,
    COMPUTED_METRIC_LABEL,
    DATA_DIR,
    T_DELTA_TARGET,
    EPOCHS,
    ENCODER_LR,
    FREEZE_ENCODER,
    MLP_LR,
    HEAD_TYPE,
    USE_CLASS_WEIGHTS,
    LABEL_SMOOTHING,
    WEIGHT_DECAY,
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
    """Filter metrics to T_DELTA_TARGET rows, pick the highest TARGET_COLUMN row per id,
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

    # Validate that T_DELTA_TARGET exists in the data
    if T_DELTA_TARGET not in metrics["t_delta"].values:
        raise ValueError(
            f"{T_DELTA_TARGET=} not found in t_delta column "
            f"(distinct values: {sorted(metrics['t_delta'].unique())})."
        )

    filtered = metrics[metrics["t_delta"] == T_DELTA_TARGET]
    if TARGET_COLUMN not in filtered.columns:
        filtered[COMPUTED_METRIC_COL] = COMPUTED_METRIC_FN(filtered)
    # Rows with default t-values will score 1 on Pareto Score so that
    # a maximum value will always exist.
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


def _class_weights(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from training label indices, normalized so mean = 1."""
    counts = torch.bincount(indices, minlength=n_classes).float().clamp(min=1)
    w = 1.0 / counts
    return w / w.mean()


def _balanced_accuracy(pred: torch.Tensor, true: torch.Tensor, n_classes: int) -> float:
    """Mean per-class recall; insensitive to majority-class collapse."""
    recalls = []
    for c in range(n_classes):
        mask = true == c
        if mask.any():
            recalls.append((pred[mask] == c).float().mean().item())
    return sum(recalls) / len(recalls) if recalls else 0.0


def _compute_loss(
    model: OrdinalPairClassifier,
    l1: torch.Tensor,
    l2: torch.Tensor,
    y1: torch.Tensor,
    y2: torch.Tensor,
    *,
    class_w_start: torch.Tensor | None,
    class_w_end: torch.Tensor | None,
    n_buckets_end: int,
) -> torch.Tensor:
    """Training/eval loss for the active head type."""
    if HEAD_TYPE == "CORAL":
        w1 = class_w_start[y1] if class_w_start is not None else None
        w2 = class_w_end[y2] if class_w_end is not None else None
        loss = ordinal_loss(l1, y1, w1)
        if n_buckets_end > 1:
            loss = loss + ordinal_loss(l2, y2, w2)
    elif HEAD_TYPE == "CE":
        loss = one_hot_ce_loss(
            l1, y1,
            class_weights=class_w_start,
            label_smoothing=LABEL_SMOOTHING,
        )
        if n_buckets_end > 1:
            loss = loss + one_hot_ce_loss(
                l2, y2,
                class_weights=class_w_end,
                label_smoothing=LABEL_SMOOTHING,
            )
    elif HEAD_TYPE == "MSE":
        w1 = class_w_start[y1] if class_w_start is not None else None
        w2 = class_w_end[y2] if class_w_end is not None else None
        loss = regression_loss(l1, model.buckets1[y1], w1)
        if n_buckets_end > 1:
            loss = loss + regression_loss(l2, model.buckets2[y2], w2)
    else:
        raise ValueError(f"HEAD_TYPE must be 'CORAL', 'MSE', or 'CE', got {HEAD_TYPE!r}")
    return loss


def _fmt_split_metrics(m: dict[str, float]) -> str:
    """Compact loss + per-target accuracy for epoch logging (start/end/both)."""
    return (
        f"loss={m['loss']:.4f}"
        f"  acc={m['acc_t_start']:.3f}/{m['acc_t_end']:.3f} ({m['acc_both']:.3f})"
    )


def eval_loader(
    model: OrdinalPairClassifier,
    loader: DataLoader,
    device: torch.device,
    *,
    class_w_start: torch.Tensor | None = None,
    class_w_end: torch.Tensor | None = None,
    n_buckets_end: int = 1,
    n_buckets_start: int | None = None,
) -> dict[str, float]:
    """Run model in eval mode over loader and return metric dict."""
    model.eval()
    all_p1, all_p2 = [], []
    all_l1, all_l2 = [], []
    val_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for srcs, tgts, l1, l2 in loader:
            l1 = l1.to(device)
            l2 = l2.to(device)
            out1, out2 = model(srcs, tgts)
            batch_loss = _compute_loss(
                model, out1, out2, l1, l2,
                class_w_start=class_w_start,
                class_w_end=class_w_end,
                n_buckets_end=n_buckets_end,
            )
            val_loss += batch_loss.item() * len(srcs)
            n_samples += len(srcs)
            p1, p2 = model.decode_bucket_indices(out1, out2)
            all_p1.append(p1)
            all_p2.append(p2)
            all_l1.append(l1)
            all_l2.append(l2)
    p1 = torch.cat(all_p1)
    p2 = torch.cat(all_p2)
    l1_out = torch.cat(all_l1)
    l2_out = torch.cat(all_l2)
    metrics = {
        "mae_t_start": mae_buckets(p1, l1_out).item(),
        "mae_t_end": mae_buckets(p2, l2_out).item(),
        "acc_t_start": (p1 == l1_out).float().mean().item(),
        "acc_t_end": (p2 == l2_out).float().mean().item(),
        "acc_both": ((p1 == l1_out) & (p2 == l2_out)).float().mean().item(),
        "bal_acc_t_start": _balanced_accuracy(
            p1, l1_out, n_buckets_start or int(l1_out.max().item()) + 1
        ),
    }
    if n_samples > 0:
        metrics["loss"] = val_loss / n_samples
    return metrics


def train() -> OrdinalPairClassifier:
    """Train the classifier and save weights + splits to a timestamped run directory.

    Each epoch logs train and val loss plus bucket accuracy
    (t_start / t_end / both correct). Checkpoints are selected by val balanced
    accuracy on t_start.
    """
    df = load_data()
    train_df, val_df, test_df = split_data(df, seed=SEED)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    save_splits(train_df, val_df, test_df, data_dir=run_dir)
    print(
        f"Dataset: {len(df)} samples  (t_delta={T_DELTA_TARGET}, target={TARGET_COLUMN})"
        f"  split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
    )

    train_loader = DataLoader(PairDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    # Distinct ordered float values for each target, may differ between t_start and t_end
    buckets_start = torch.tensor(sorted(df["t_start"].unique()), dtype=torch.float)
    buckets_end = torch.tensor(sorted(df["t_end"].unique()), dtype=torch.float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalPairClassifier(buckets1=buckets_start, buckets2=buckets_end, freeze_encoder=FREEZE_ENCODER, head_type=HEAD_TYPE).to(device)
    if FREEZE_ENCODER:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=MLP_LR,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": list(model.body.parameters()) + list(model.head1.parameters()) + list(model.head2.parameters()), "lr": MLP_LR, "weight_decay": WEIGHT_DECAY},
        ])

    if USE_CLASS_WEIGHTS:
        class_w_start = _class_weights(
            torch.tensor(train_df["t_start_idx"].values), len(buckets_start)
        ).to(device)
        class_w_end = _class_weights(
            torch.tensor(train_df["t_end_idx"].values), len(buckets_end)
        ).to(device)
        print(f"Class weights  t_start: {class_w_start.tolist()}")
        print(f"Class weights  t_end:   {class_w_end.tolist()}")
    else:
        class_w_start = class_w_end = None

    weights_out = run_dir / "classifier_weights.pt"
    best_val_score = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for srcs, tgts, y1, y2 in train_loader:
            y1 = y1.to(device)
            y2 = y2.to(device)
            l1, l2 = model(srcs, tgts)
            loss = _compute_loss(
                model, l1, l2, y1, y2,
                class_w_start=class_w_start,
                class_w_end=class_w_end,
                n_buckets_end=len(buckets_end),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(srcs)

        train_metrics = eval_loader(
            model, train_loader, device,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=len(buckets_end),
            n_buckets_start=len(buckets_start),
        )
        train_metrics["loss"] = epoch_loss / len(train_df)
        val_metrics = eval_loader(
            model, val_loader, device,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=len(buckets_end),
            n_buckets_start=len(buckets_start),
        )
        val_score = val_metrics["bal_acc_t_start"]
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
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
                            if k.isupper() and not k.startswith("_") and not isinstance(v, type({}.keys()))
                        },
                    },
                },
                weights_out,
            )
        print(
            f"Epoch {epoch:02d}  train: {_fmt_split_metrics(train_metrics)}"
            f"  val: {_fmt_split_metrics(val_metrics)}"
            + ("  *" if improved else "")
        )

    print(f"Saved {weights_out}  (best val bal_acc_start={best_val_score:.3f})")
    model.load_state_dict(torch.load(weights_out, map_location=device, weights_only=False)["state_dict"])

    test_metrics = eval_loader(
        model, test_loader, device,
        class_w_start=class_w_start,
        class_w_end=class_w_end,
        n_buckets_end=len(buckets_end),
        n_buckets_start=len(buckets_start),
    )
    print(f"\nTest: {_fmt_split_metrics(test_metrics)}")
    return model


if __name__ == "__main__":
    train()
