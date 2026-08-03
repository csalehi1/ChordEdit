"""
Train OrdinalPairClassifier to predict t_start and t_end from precomputed
ChordEdit embeddings (image, mask, source prompt, target prompt).

Rows are filtered to those matching TARGET_T_DELTA for t_delta, then for each
sample_id the row with the highest C_TARGET_COL is selected. The resulting
t_start and t_end values are mapped to ordinal buckets for OrdinalPairClassifier.
Embeddings come from the packed/scattered caches via embeddings.get_embeddings.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from _data import (
    SampleTensors,
    add_target_score,
    build_grid_table,
    create_sample_tensors,
    drop_missing_inputs,
    load_df,
    save_split_df,
    select_best_rows,
    split_df,
)
from _helpers import save_run_settings
from model import OrdinalPairClassifier, mae_buckets
from head_coral import ordinal_loss
from head_mse import regression_loss
from head_ce import cost_sensitive_ce_loss, one_hot_ce_loss
from selection import SelectionSplit, selection_metrics, slice_grid
import settings
from settings import *

TRAIN_METRICS_NAME = "train_metrics.json"

# Checkpoint criteria, mapped to (metric key, +1 if higher is better else -1).
CKPT_METRICS = {
    "bal_acc_t_start": ("bal_acc_t_start", 1.0),
    "bal_acc_t_end": ("bal_acc_t_end", 1.0),
    "acc_both": ("acc_both", 1.0),
    "loss": ("loss", -1.0),
    "regret_median": ("regret_median", -1.0),
    "top1_hit_rate": ("top1_hit_rate", 1.0),
}


def _make_scheduler(optimizer: torch.optim.Optimizer):
    """Per-epoch learning-rate schedule selected by LR_SCHEDULER.

    Every schedule decays from LR over the full EPOCHS budget, so the epoch at
    which the best checkpoint lands is comparable across them. "none" keeps the
    constant learning rate the classifier trained with before.
    """
    if LR_SCHEDULER == "none":
        return None
    if LR_SCHEDULER == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR * LR_MIN_FACTOR)
    if LR_SCHEDULER == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=LR_MIN_FACTOR, total_iters=EPOCHS)
    if LR_SCHEDULER == "step":
        milestones = [max(1, int(EPOCHS * f)) for f in (0.6, 0.85)]
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    if LR_SCHEDULER == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.3, patience=2, min_lr=LR * LR_MIN_FACTOR
        )
    raise ValueError(f"Unknown LR_SCHEDULER={LR_SCHEDULER!r}")


def _class_weights(indices: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights from training label indices."""
    counts = torch.bincount(indices, minlength=n_classes).float()
    present = counts > 0
    if not present.any():
        raise ValueError("no labels to weight")
    w = torch.zeros_like(counts)
    w[present] = 1.0 / counts[present]
    return w / w[present].mean()


def _balanced_accuracy(pred: torch.Tensor, true: torch.Tensor, n_classes: int) -> float:
    """Mean per-class recall; insensitive to majority-class collapse."""
    recalls = []
    for c in range(n_classes):
        mask = true == c
        if mask.any():
            recalls.append((pred[mask] == c).float().mean().item())
    return sum(recalls) / len(recalls) if recalls else 0.0


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
    """Compact loss, per-target accuracy (start/end/both), and median regret."""
    out = (
        f"loss={m['loss']:.4f}"
        f"  acc={m['acc_t_start']:.3f}/{m['acc_t_end']:.3f} ({m['acc_both']:.3f})"
    )
    if "regret_median" in m:
        out += f"  regret={m['regret_median']:.4f}"
    return out


def eval_split(
    model: OrdinalPairClassifier,
    cells: SampleTensors,
    *,
    class_w_start: torch.Tensor | None = None,
    class_w_end: torch.Tensor | None = None,
    n_buckets_end: int = 1,
    n_buckets_start: int | None = None,
    sel: SelectionSplit | None = None,
) -> dict[str, float]:
    """Run model in eval mode over one split's tensors and return metric dict.

    When sel is given, the two heads are also decoded jointly over that split's
    labeled cells and the resulting selection metrics (regret and friends) are
    merged into the returned dict.
    """
    model.eval()
    all_p1, all_p2 = [], []
    all_l1, all_l2 = [], []
    all_cells = []
    val_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for img, mask, src, tar, l1, l2 in cells.iter_batches(BATCH_SIZE):
            out1, out2 = model(img, mask, src, tar)
            batch_loss = _compute_loss(
                model, out1, out2, l1, l2,
                class_w_start=class_w_start,
                class_w_end=class_w_end,
                n_buckets_start=n_buckets_start or 1,
                n_buckets_end=n_buckets_end,
            )
            val_loss += batch_loss.item() * len(l1)
            n_samples += len(l1)
            p1, p2 = model.decode_bucket_indices(out1, out2, batch_size=len(l1))
            all_p1.append(p1)
            all_p2.append(p2)
            all_l1.append(l1)
            all_l2.append(l2)
            if sel is not None:
                all_cells.append(model.decode_cells(out1, out2, sel.cell_start_idx, sel.cell_end_idx))
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
    if sel is not None:
        # iter_batches without shuffle walks the split in order, so the
        # concatenated picks line up row-for-row with the split's samples.
        metrics.update(selection_metrics(torch.cat(all_cells).cpu().numpy(), sel))
    return metrics


def train() -> OrdinalPairClassifier:
    """Train the classifier and save weights + splits to a run directory.

    Each epoch logs train and val loss, bucket accuracy (t_start / t_end / both
    correct), and median selection regret. Checkpoints are selected by
    CKPT_METRIC on the validation split.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Score once: select_best_rows and build_grid_table would each otherwise
    # recompute the per-sample delta normalization over every grid cell.
    cells_df = add_target_score(drop_missing_inputs(load_df()))
    df = select_best_rows(cells_df)
    grid = build_grid_table(cells_df)
    train_df, val_df, test_df = split_df(df)

    run_name = RUN_NAME or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_settings(run_dir)

    print(
        f"Dataset: {len(df)} samples  (t_delta={TARGET_T_DELTA}, target={C_TARGET_COL})"
        f"  split: train={len(train_df)} / val={len(val_df)} / test={len(test_df)}"
        f"  grid: {grid.phi.shape[1]} cells"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensors = create_sample_tensors({"train": train_df, "val": val_df, "test": test_df}, device)
    train_cells, val_cells, test_cells = tensors["train"], tensors["val"], tensors["test"]
    sel = {
        name: slice_grid(grid, split[SAMPLE_ID_COL].to_numpy(), device)
        for name, split in (("train", train_df), ("val", val_df), ("test", test_df))
    }
    img_dim = int(train_cells.img.shape[1])
    text_dim = int(train_cells.src.shape[1])

    buckets_start = torch.tensor(np.asarray(GRID_T_START, dtype=float), dtype=torch.float)
    buckets_end = torch.tensor(np.asarray(GRID_T_END, dtype=float), dtype=torch.float)
    n_buckets_start = len(buckets_start)
    n_buckets_end = len(buckets_end)
    if n_buckets_start <= 1 and n_buckets_end <= 1:
        raise ValueError("At least one of t_start or t_end must have >1 bucket to train.")

    model = OrdinalPairClassifier(
        img_dim,
        text_dim,
        buckets1=buckets_start,
        buckets2=buckets_end,
        head_type=HEAD_TYPE,
    )
    model = model.to(device)
    active_heads = [name for name, on in (("t_start", model.predict_start), ("t_end", model.predict_end)) if on]
    print(f"Training heads: {', '.join(active_heads)}  (img_dim={img_dim}, text_dim={text_dim})")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = _make_scheduler(optimizer)

    if USE_CLASS_WEIGHTS:
        class_w_start = None
        class_w_end = None
        if model.predict_start:
            class_w_start = _class_weights(train_cells.y1, n_buckets_start).to(device)
            print(f"Class weights  t_start: {[f'{w:.2f}' for w in class_w_start.tolist()]}")
        if model.predict_end:
            class_w_end = _class_weights(train_cells.y2, n_buckets_end).to(device)
            print(f"Class weights  t_end:   {[f'{w:.2f}' for w in class_w_end.tolist()]}")
    else:
        class_w_start = class_w_end = None

    weights_out = run_dir / "classifier_weights.pt"
    # -inf so the first epoch always writes a checkpoint even when the metric is 0.0
    best_val_score = float("-inf")
    if CKPT_METRIC not in CKPT_METRICS:
        raise ValueError(f"Unknown CKPT_METRIC={CKPT_METRIC!r}; expected one of {sorted(CKPT_METRICS)}")
    checkpoint_metric, ckpt_sign = CKPT_METRICS[CKPT_METRIC]
    if checkpoint_metric == "bal_acc_t_start" and not model.predict_start:
        checkpoint_metric = "bal_acc_t_end"
    best_epoch = 0
    history: list[dict] = []

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.train()
        epoch_loss = 0.0
        for img, mask, src, tar, y1, y2 in train_cells.iter_batches(BATCH_SIZE, shuffle=True):
            l1, l2 = model(img, mask, src, tar)
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
            epoch_loss += loss.item() * len(y1)

        train_metrics = eval_split(
            model, train_cells,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=n_buckets_end,
            n_buckets_start=n_buckets_start,
        )
        train_metrics["loss"] = epoch_loss / len(train_cells)
        val_metrics = eval_split(
            model, val_cells,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=n_buckets_end,
            n_buckets_start=n_buckets_start,
            sel=sel["val"],
        )
        val_score = ckpt_sign * val_metrics[checkpoint_metric]
        improved = val_score > best_val_score
        if improved:
            best_val_score = val_score
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "buckets1": buckets_start.tolist(),
                    "buckets2": buckets_end.tolist(),
                    "img_dim": img_dim,
                    "text_dim": text_dim,
                    "config": {"run_dir": str(run_dir), **settings.CONFIG},
                },
                weights_out,
            )
        lr_now = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_score)
            else:
                scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        history.append({
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_metrics["loss"],
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({elapsed:.2f}s)  Train: {_fmt_split_metrics(train_metrics)}"
            f"  Val: {_fmt_split_metrics(val_metrics)}"
            + ("  *" if improved else "")
        )

    print(f"Saved {weights_out}  (best val {checkpoint_metric}={ckpt_sign * best_val_score:.4f} at epoch {best_epoch})")
    if not weights_out.exists():
        raise RuntimeError(f"No checkpoint was written to {weights_out}")
    model.load_state_dict(
        torch.load(weights_out, map_location=device, weights_only=False)["state_dict"],
        strict=False,
    )

    save_split_df(train_df, val_df, test_df, run_dir)

    best_metrics = {
        name: eval_split(
            model, split_cells,
            class_w_start=class_w_start,
            class_w_end=class_w_end,
            n_buckets_end=n_buckets_end,
            n_buckets_start=n_buckets_start,
            sel=sel[name],
        )
        for name, split_cells in (("val", val_cells), ("test", test_cells))
    }
    # Always-the-default-cell is the reference every selection metric is read
    # against; it costs nothing to record alongside the model's own numbers.
    default_metrics = {
        name: selection_metrics(
            np.full(len(sel[name].best_phi), sel[name].baseline_cell), sel[name]
        )
        for name in ("val", "test")
    }

    (run_dir / TRAIN_METRICS_NAME).write_text(
        json.dumps({
            "run_dir": str(run_dir),
            "ckpt_metric": CKPT_METRIC,
            "best_epoch": best_epoch,
            "epochs_ran": EPOCHS,
            "n_cells": int(grid.phi.shape[1]),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "val": best_metrics["val"],
            "test": best_metrics["test"],
            "val_default": default_metrics["val"],
            "test_default": default_metrics["test"],
            "history": history,
        }, indent=2) + "\n")

    print(f"\nTest: {_fmt_split_metrics(best_metrics['test'])}")
    print(
        f"Test selection: regret median={best_metrics['test']['regret_median']:.4f}"
        f" mean={best_metrics['test']['regret_mean']:.4f}"
        f"  vs default-cell median={default_metrics['test']['regret_median']:.4f}"
        f" mean={default_metrics['test']['regret_mean']:.4f}"
    )
    return model


if __name__ == "__main__":
    train()
