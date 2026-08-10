"""
Train the metric surrogate M on our merged 12k tournament dataset's
region + background samples (style excluded -- no edit mask exists for it,
same as classify.py's LINEX run), with LINEX as the ranking-loss
scalarization instead of train_m.py's hardcoded simple average.

Bypasses _data.py's file-based load_df()/INPUTS_CSV/METRICS_CSV convention
(single DATASET_DIR, plain-numeric sample_id) since our merged IDs are
prefixed (region_/background_) to disambiguate 3 merged source datasets.
Builds the same DataFrame shape directly instead, with absolute
image/mask paths pointing at each sample's *original* source dataset dir,
then hands it to _data.py's prepare_df/split_df/create_dataloaders
unchanged.

Run from this directory:
    cd models/modified-classification && python3 train_m_tournament12k.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import numpy as np
import pandas as pd
import torch

from _data import create_dataloaders, model_inputs, prepare_df, save_splits_df, split_df
from _helpers import format_results, pairwise_ranking_loss, save_settings_hash
from model_m import MetricPredictor
from settings import *

MAPPING_JSON = Path("/shared/ssd_30T/zarageddes/tournament_12k_sdxlturbo/mapping_file.json")
MERGED_METRICS_CSV = Path(_ROOT) / "models" / "classification" / "data" / "id_to_metrics_sdxlturbo_tournament12k.csv"

SOURCE_DATASET_DIRS = {
    "region": Path("/shared/ssd_30T/mirick/datasets/ultra_edit/UltraEdit_Region_10000"),
    "background": Path("/shared/ssd_30T/salehi/datasets/UltraEdit_Background_1000_v2"),
}

LINEX_ALPHA = 5.0


def build_surrogate_df() -> pd.DataFrame:
    """One row per (sample_id, t_start, t_end) cell, region+background only,
    with absolute image_path/mask_image_path resolved from each sample's
    original source dataset dir (bypassing the single-DATASET_DIR
    assumption in _helpers.py's resolve_image_path/resolve_mask_path,
    since these are already absolute)."""
    mapping = json.loads(MAPPING_JSON.read_text())

    orig_mappings = {
        prefix: json.loads((d / "mapping_file.json").read_text())
        for prefix, d in SOURCE_DATASET_DIRS.items()
    }

    inputs_rows = []
    n_no_mask = 0
    for new_id, item in mapping.items():
        prefix = item["dataset"]
        if prefix not in SOURCE_DATASET_DIRS:
            continue  # style: no mask, excluded
        orig_item = orig_mappings[prefix][item["orig_sid"]]
        if "mask_image_path" not in orig_item:
            n_no_mask += 1
            continue  # a handful of region/background samples also lack a mask
        dataset_dir = SOURCE_DATASET_DIRS[prefix]
        inputs_rows.append({
            SAMPLE_ID_COL: new_id,
            SOURCE_PROMPT_COL: item["original_prompt"],
            TARGET_PROMPT_COL: item["editing_prompt"],
            IMAGE_PATH_COL: str(dataset_dir / orig_item["image_path"]),
            MASK_PATH_COL: str(dataset_dir / orig_item["mask_image_path"]),
        })
    inputs_df = pd.DataFrame(inputs_rows)
    print(f"inputs: {len(inputs_df)} samples (region+background), {n_no_mask} skipped (no mask_image_path)", flush=True)

    metrics_df = pd.read_csv(MERGED_METRICS_CSV, dtype={"sample_id": str})
    metrics_df = metrics_df.rename(columns={"psnr": PSNR_COL, "clip_edited": CLIP_COL})
    metrics_df = metrics_df[metrics_df[SAMPLE_ID_COL].isin(inputs_df[SAMPLE_ID_COL])]
    if TARGET_T_DELTA is not None:
        metrics_df = metrics_df[metrics_df[T_DELTA_COL] == TARGET_T_DELTA]

    n_drop = int(metrics_df.isna().any(axis=1).sum())
    if n_drop:
        print(f"Dropping {n_drop} rows with missing values (e.g. unreliable-mask samples)", flush=True)
        metrics_df = metrics_df.dropna().reset_index(drop=True)

    metrics_part = metrics_df[[SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *M_TARGET_COLS]]
    df = pd.merge(metrics_part, inputs_df, on=SAMPLE_ID_COL, how="inner")
    print(f"merged: {len(df)} rows ({df[SAMPLE_ID_COL].nunique()} samples)", flush=True)
    return df


def linex_score_tensor(
    t: torch.Tensor, values: torch.Tensor, *, alpha: float, default_t_start: float, default_t_end: float,
) -> torch.Tensor:
    """
    LINEX scalarization for a ranking-loss batch (one sample's full grid).
    values: (N, 2) [psnr, clip] in the same units for both true/pred calls.
    Per-sample min-max normalizes each column, subtracts the baseline row's
    (nearest to default_t_start/end) value, then applies the LINEX utility
    u(x) = (1/2)[x + (1/alpha)(1 - e^{-alpha*x})] to each delta and sums.
    """
    dist = (t[:, 0] - default_t_start).abs() + (t[:, 1] - default_t_end).abs()
    base_idx = int(dist.argmin().item())

    def _minmax(col: torch.Tensor) -> torch.Tensor:
        lo, hi = col.min(), col.max()
        return (col - lo) / (hi - lo + 1e-6)

    def _u(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * (x + (1 - torch.exp(-alpha * x)) / alpha)

    psnr_n = _minmax(values[:, 0])
    clip_n = _minmax(values[:, 1])
    delta_psnr = psnr_n - psnr_n[base_idx]
    delta_clip = clip_n - clip_n[base_idx]
    return _u(delta_psnr) + _u(delta_clip)


def train(model, train_X, train_y, val_X, val_y, test_X, test_y):
    from train_m import evaluate  # unmodified: plain MSE/MAE/R2, no ranking loss

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    use_ranking = RANKING_LOSS_WEIGHT > 0
    train_loader, val_loader, test_loader = create_dataloaders(
        model, train_X, train_y, val_X, val_y, test_X, test_y, group_train_by_sample=use_ranking,
    )

    img_dim, text_dim = model.encoder_img_dim, model.encoder_text_dim
    model.release_encoders()

    y_train = torch.tensor(train_y[list(M_TARGET_COLS)].values, dtype=torch.float)
    print(f"Dataset: train={len(train_X)} cells val={len(val_X)} cells")

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

    optimizer = torch.optim.AdamW(model.regressor.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    mean, std = model.regressor.target_mean, model.regressor.target_std

    weights_out = run_dir / "regressor_weights.pt"
    best_val = float("inf")
    device = next(model.regressor.parameters()).device
    n_cells, n_samples = len(train_X), train_X[SAMPLE_ID_COL].nunique()
    print(f"\nTraining for {EPOCHS} epochs (LINEX alpha={LINEX_ALPHA}, ranking_weight={RANKING_LOSS_WEIGHT})...")
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.perf_counter()
        model.regressor.train()
        train_loss_sum, train_n = 0.0, 0
        for batch in train_loader:
            img, mask, src, tar, t, y = model_inputs(batch, device)
            out = model.regressor(img, mask, src, tar, t)
            y_std = (y - mean) / std
            mse = torch.nn.functional.mse_loss(out, y_std)
            loss = mse
            if use_ranking:
                pred_m = linex_score_tensor(
                    t, model.regressor.denormalize(out),
                    alpha=LINEX_ALPHA, default_t_start=DEFAULT_T_START, default_t_end=DEFAULT_T_END,
                )
                true_m = linex_score_tensor(
                    t, y, alpha=LINEX_ALPHA, default_t_start=DEFAULT_T_START, default_t_end=DEFAULT_T_END,
                )
                loss = loss + RANKING_LOSS_WEIGHT * pairwise_ranking_loss(pred_m, true_m)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += mse.detach().item() * y.numel()
            train_n += y.numel()

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
        if epoch == EPOCHS:
            train_results = evaluate(model, train_loader, device)
            train_line = format_results(train_results)
        else:
            train_line = f"loss={train_loss_sum / max(train_n, 1):7.4f}  (running MSE)"
        print(
            f"Epoch [{epoch:03d}/{EPOCHS:03d}] | {n_cells} cells ({n_samples} samples) in {elapsed:.2f}s"
            f"\n    {'Train:':<6} {train_line}"
            f"\n    {'Val:':<6} {format_results(val_results)}"
            + ("  *" if improved else "")
        )

    ckpt = torch.load(weights_out, map_location=device, weights_only=False)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    results = evaluate(model, test_loader, device)
    print(f"\n    {'Test:':<6} {format_results(results)}")

    save_settings_hash(run_dir)
    save_splits_df(train_X, val_X, test_X, run_dir)
    import json as _json
    with open(run_dir / "m_train_metrics.json", "w") as f:
        _json.dump({"val_best_loss": best_val, "test": results}, f, indent=4)
    return run_dir


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    data_df = build_surrogate_df()
    X, y = prepare_df(data_df)
    train_X, val_X, test_X, train_y, val_y, test_y = split_df(X, y)
    print(
        f"Splits: train={len(train_X)} cells ({train_X[SAMPLE_ID_COL].nunique()} samples)  "
        f"val={len(val_X)} cells ({val_X[SAMPLE_ID_COL].nunique()} samples)  "
        f"test={len(test_X)} cells ({test_X[SAMPLE_ID_COL].nunique()} samples)"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetricPredictor(device=device).to(device)
    run_dir = train(model, train_X, train_y, val_X, val_y, test_X, test_y)
    print(f"\nSaved to {run_dir.resolve()}")


if __name__ == "__main__":
    main()
