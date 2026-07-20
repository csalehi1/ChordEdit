"""
Train OrdinalPairClassifier to predict t_start and t_end
from (source_prompt, target_prompt) pairs.

Rows are filtered to those matching TARGET_T_DELTA for t_delta, then for each
sample_id the row with the highest C_TARGET_COL is selected. The resulting
t_start and t_end values are mapped to ordinal buckets for OrdinalPairClassifier.
"""

from __future__ import annotations

import sys
from datetime import datetime
from functools import partial
from pathlib import Path

# Allow `python train.py` from this directory (or elsewhere) to resolve the package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import time
import torch
from torch.utils.data import Dataset, DataLoader

from models.classification.model import OrdinalPairClassifier, mae_buckets
from models.classification.head_coral import ordinal_loss
from models.classification.head_mse import regression_loss
from models.classification.head_ce import cost_sensitive_ce_loss, one_hot_ce_loss
from models.classification._helpers import load_inputs_df
import models.classification.settings as _settings
from models.classification.settings import *

def _serialize_setting(v):
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, partial):
        return v.func.__name__
    if callable(v):
        return getattr(v, "__name__", repr(v))
    return v


class PairDataset(Dataset):
    """Wraps (source_prompt, target_prompt, t_start_idx, t_end_idx) rows."""

    def __init__(self, df: pd.DataFrame):
        self.src = df[SOURCE_PROMPT_COL].tolist()
        self.tgt = df[TARGET_PROMPT_COL].tolist()
        self.y1 = torch.tensor(df["t_start_idx"].values, dtype=torch.long)
        self.y2 = torch.tensor(df["t_end_idx"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int):
        return self.src[idx], self.tgt[idx], self.y1[idx], self.y2[idx]


def _normalize_sample_id(series: pd.Series) -> pd.Series:
    """Canonical string IDs so zero-padded and integer forms merge reliably."""
    return series.map(lambda x: str(int(x)))


def _map_to_bucket_idx(series: pd.Series, buckets) -> pd.Series:
    """Map float timestep values to ordinal indices into buckets (float-safe)."""
    bucket_arr = np.asarray(buckets, dtype=float)

    def _index(v: float) -> int:
        matches = np.flatnonzero(np.isclose(float(v), bucket_arr))
        if len(matches) != 1:
            raise ValueError(f"Value {v} does not uniquely match buckets {bucket_arr.tolist()}.")
        return int(matches[0])

    return series.map(_index)


def load_data() -> pd.DataFrame:
    """Filter metrics to TARGET_T_DELTA rows, pick the highest C_TARGET_COL row per id,
    and join with prompt strings. Maps t_start/t_end onto indices in GRID_T_*.
    """
    metrics = pd.read_csv(METRICS_CSV, dtype={SAMPLE_ID_COL: str})
    metrics[SAMPLE_ID_COL] = _normalize_sample_id(metrics[SAMPLE_ID_COL])

    strings = load_inputs_df()
    strings["id"] = strings[SAMPLE_ID_COL].astype(str)

    # Validate that TARGET_T_DELTA exists in the data
    if TARGET_T_DELTA not in metrics[T_DELTA_COL].values:
        raise ValueError(
            f"{TARGET_T_DELTA=} not found in {T_DELTA_COL} column "
            f"(distinct values: {sorted(metrics[T_DELTA_COL].unique())})."
        )

    filtered = metrics[metrics[T_DELTA_COL] == TARGET_T_DELTA].copy()
    if C_TARGET_COL not in filtered.columns:
        filtered[C_TARGET_COL] = C_TARGET_FUNC(filtered)
    # Rows with default t-values will score 0 on softplus so that
    # a maximum value will always exist among improving rows.
    best_idx = filtered.groupby(SAMPLE_ID_COL)[C_TARGET_COL].idxmax()
    best = filtered.loc[best_idx, [SAMPLE_ID_COL, T_START_COL, T_END_COL]].reset_index(drop=True)

    df = pd.merge(best, strings, left_on=SAMPLE_ID_COL, right_on="id")
    df["t_start_idx"] = _map_to_bucket_idx(df[T_START_COL], GRID_T_START)
    df["t_end_idx"] = _map_to_bucket_idx(df[T_END_COL], GRID_T_END)
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
    data_dir: Path = OUTPUTS_DIR,
) -> None:
    """Save train/val/test splits as gzip-compressed Parquet files in data_dir."""
    data_dir.mkdir(exist_ok=True)
    for name, df in (("train", train_df), ("val", val_df), ("test", test_df)):
        out = data_dir / f"{name}.parquet.gz"
        df.to_parquet(out, compression="gzip", index=False)
    print(f"Saved Splits: {data_dir} ({data_dir.stat().st_size / 1024:.1f} KB)")


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


def _head_params(model: OrdinalPairClassifier) -> list[torch.nn.Parameter]:
    """Trainable parameters for the MLP body and any active prediction heads."""
    params = list(model.body.parameters())
    if model.predict_start:
        params += list(model.head1.parameters())
    if model.predict_end:
        params += list(model.head2.parameters())
    return params


def _ce_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    *,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    if CE_LOSS_TYPE == "cost_sensitive_ce_loss":
        return cost_sensitive_ce_loss(logits, target_idx, class_weights=class_weights)
    elif CE_LOSS_TYPE == "one_hot_ce_loss":
        return one_hot_ce_loss(logits, target_idx, class_weights=class_weights, label_smoothing=LABEL_SMOOTHING)
    else:
        raise ValueError(f"CE_LOSS_TYPE {CE_LOSS_TYPE!r} not recognized")


def _compute_loss(
    model: OrdinalPairClassifier,
    l1: torch.Tensor | None,
    l2: torch.Tensor | None,
    y1: torch.Tensor,
    y2: torch.Tensor,
    *,
    class_w_start: torch.Tensor | None,
    class_w_end: torch.Tensor | None,
    n_buckets_start: int,
    n_buckets_end: int,
) -> torch.Tensor:
    """Training/eval loss for active head(s) only."""
    loss_parts: list[torch.Tensor] = []

    if n_buckets_start > 1:
        if l1 is None:
            raise ValueError("t_start head output required when n_buckets_start > 1")
        if HEAD_TYPE == "CORAL":
            w1 = class_w_start[y1] if class_w_start is not None else None
            loss_parts.append(ordinal_loss(l1, y1, w1))
        elif HEAD_TYPE == "CE":
            loss_parts.append(_ce_loss(l1, y1, class_weights=class_w_start))
        elif HEAD_TYPE == "MSE":
            w1 = class_w_start[y1] if class_w_start is not None else None
            loss_parts.append(regression_loss(l1, model.buckets1[y1], w1))
        else:
            raise ValueError(f"HEAD_TYPE must be 'CORAL', 'MSE', or 'CE', got {HEAD_TYPE!r}")

    if n_buckets_end > 1:
        if l2 is None:
            raise ValueError("t_end head output required when n_buckets_end > 1")
        if HEAD_TYPE == "CORAL":
            w2 = class_w_end[y2] if class_w_end is not None else None
            loss_parts.append(ordinal_loss(l2, y2, w2))
        elif HEAD_TYPE == "CE":
            loss_parts.append(_ce_loss(l2, y2, class_weights=class_w_end))
        elif HEAD_TYPE == "MSE":
            w2 = class_w_end[y2] if class_w_end is not None else None
            loss_parts.append(regression_loss(l2, model.buckets2[y2], w2))
        else:
            raise ValueError(f"HEAD_TYPE must be 'CORAL', 'MSE', or 'CE', got {HEAD_TYPE!r}")

    if not loss_parts:
        raise ValueError("At least one of t_start or t_end must have >1 bucket to train.")
    loss = loss_parts[0]
    for part in loss_parts[1:]:
        loss = loss + part
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
                n_buckets_start=n_buckets_start or 1,
                n_buckets_end=n_buckets_end,
            )
            val_loss += batch_loss.item() * len(srcs)
            n_samples += len(srcs)
            p1, p2 = model.decode_bucket_indices(out1, out2, batch_size=len(srcs))
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
        "bal_acc_t_end": _balanced_accuracy(
            p2, l2_out, n_buckets_end or int(l2_out.max().item()) + 1
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
        f"Dataset: {len(df)} samples  (t_delta={TARGET_T_DELTA}, target={C_TARGET_COL})"
        f"  split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
    )

    train_loader = DataLoader(PairDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(PairDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(PairDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    buckets_start = torch.tensor(np.asarray(GRID_T_START, dtype=float), dtype=torch.float)
    buckets_end = torch.tensor(np.asarray(GRID_T_END, dtype=float), dtype=torch.float)
    n_buckets_start = len(buckets_start)
    n_buckets_end = len(buckets_end)
    if n_buckets_start <= 1 and n_buckets_end <= 1:
        raise ValueError("At least one of t_start or t_end must have >1 bucket to train.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OrdinalPairClassifier(
        buckets1=buckets_start,
        buckets2=buckets_end,
        freeze_encoder=FREEZE_ENCODER,
        head_type=HEAD_TYPE,
    )
    model = model.to(device)
    active_heads = [name for name, on in (("t_start", model.predict_start), ("t_end", model.predict_end)) if on]
    print(f"Training heads: {', '.join(active_heads)}")
    if FREEZE_ENCODER:
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=MLP_LR,
            weight_decay=WEIGHT_DECAY,
        )
    else:
        optimizer = torch.optim.AdamW([
            {"params": model.encoder.parameters(), "lr": ENCODER_LR, "weight_decay": WEIGHT_DECAY},
            {"params": _head_params(model), "lr": MLP_LR, "weight_decay": WEIGHT_DECAY},
        ])

    if USE_CLASS_WEIGHTS:
        class_w_start = None
        class_w_end = None
        if model.predict_start:
            class_w_start = _class_weights(
                torch.tensor(train_df["t_start_idx"].values), n_buckets_start
            ).to(device)
            print(f"Class weights  t_start: {[f'{w:.2f}' for w in class_w_start.tolist()]}")
        if model.predict_end:
            class_w_end = _class_weights(
                torch.tensor(train_df["t_end_idx"].values), n_buckets_end
            ).to(device)
            print(f"Class weights  t_end:   {[f'{w:.2f}' for w in class_w_end.tolist()]}")
    else:
        class_w_start = class_w_end = None

    weights_out = run_dir / "classifier_weights.pt"
    # -inf so the first epoch always writes a checkpoint even when the metric is 0.0
    best_val_score = float("-inf")
    checkpoint_metric = "bal_acc_t_start" if model.predict_start else "bal_acc_t_end"

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
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
                n_buckets_start=n_buckets_start,
                n_buckets_end=n_buckets_end,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(srcs)

        train_metrics = eval_loader(
            model, train_loader, device,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=n_buckets_end,
            n_buckets_start=n_buckets_start,
        )
        train_metrics["loss"] = epoch_loss / len(train_df)
        val_metrics = eval_loader(
            model, val_loader, device,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=n_buckets_end,
            n_buckets_start=n_buckets_start,
        )
        val_score = val_metrics[checkpoint_metric]
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
                            k: _serialize_setting(v)
                            for k, v in vars(_settings).items()
                            if k.isupper() and not k.startswith("_") and not isinstance(v, type({}.keys()))
                        },
                    },
                },
                weights_out,
            )
        elapsed = time.perf_counter() - epoch_start
        print(
            f"Epoch {epoch:02d} ({elapsed:.2f}s)  Train: {_fmt_split_metrics(train_metrics)}"
            f"  Val: {_fmt_split_metrics(val_metrics)}"
            + ("  *" if improved else "")
        )

    print(f"Saved {weights_out}  (best val {checkpoint_metric}={best_val_score:.3f})")
    if not weights_out.exists():
        raise RuntimeError(f"No checkpoint was written to {weights_out}")
    model.load_state_dict(
        torch.load(weights_out, map_location=device, weights_only=False)["state_dict"],
        strict=False,
    )

    test_metrics = eval_loader(
        model, test_loader, device,
        class_w_start=class_w_start,
        class_w_end=class_w_end,
        n_buckets_end=n_buckets_end,
        n_buckets_start=n_buckets_start,
    )
    print(f"\nTest: {_fmt_split_metrics(test_metrics)}")
    return model


if __name__ == "__main__":
    train()
