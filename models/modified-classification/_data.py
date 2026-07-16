"""Shared data loading and embedding helpers for M and T training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Sampler, TensorDataset

from model_m import MetricPredictor
from settings import (
    BATCH_SIZE,
    CELL_PATH_COL,
    CLIP_COL,
    DATASET_DIR,
    GENERATED_DIR,
    IMAGE_PATH_COL,
    INPUTS_CSV,
    MASK_PATH_COL,
    METRICS_CSV,
    M_TARGET_COLS,
    PSNR_COL,
    SAMPLE_ID_COL,
    SEED,
    SOURCE_PROMPT_COL,
    T_DELTA_COL,
    T_END_COL,
    T_START_COL,
    TARGET_PROMPT_COL,
    TARGET_T_DELTA,
    TRAIN_FRAC,
    VAL_FRAC,
)


from _helpers import (
    normalize_target_columns,
    prep_sample_id,
    resolve_cell_path,
    resolve_image_path,
    resolve_mask_path,
)


METRICS_COLS = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *M_TARGET_COLS]
INPUTS_COLS = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]
DATA_COLS = list(dict.fromkeys(METRICS_COLS + INPUTS_COLS))
ID_TO_SPLIT_NAME = "id_to_split.csv"

# build_tensors TensorDataset layout. sample_idx is metadata only — not model input.
IX_SAMPLE_IDX = 0
IX_IMG = 1
IX_MASK = 2
IX_SRC = 3
IX_TAR = 4
IX_T = 5
IX_Y = 6
MODEL_BATCH_SLICE = slice(IX_IMG, IX_Y + 1)


def model_inputs(batch, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return (img, mask, src, tar, t, y) on device."""
    return tuple(x.to(device) for x in batch[MODEL_BATCH_SLICE])


class SampleGridBatchSampler(Sampler[list[int]]):
    """Yield one sample's full timestep grid per batch (for within-image ranking loss)."""

    def __init__(self, sample_idx: torch.Tensor, shuffle: bool = True):
        self.shuffle = shuffle
        self.sample_to_indices: dict[int, list[int]] = {}
        for i, sid in enumerate(sample_idx.tolist()):
            self.sample_to_indices.setdefault(sid, []).append(i)
        self.sample_ids = list(self.sample_to_indices.keys())

    def __iter__(self):
        ids = self.sample_ids.copy()
        if self.shuffle:
            ids = [ids[i] for i in torch.randperm(len(ids)).tolist()]
        for sid in ids:
            yield self.sample_to_indices[sid]

    def __len__(self) -> int:
        return len(self.sample_ids)


def save_splits_df(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
    run_dir: Path,
) -> None:
    rows = []
    for name, X in ("train", train_X), ("val", val_X), ("test", test_X):
        ids = X[[SAMPLE_ID_COL]].drop_duplicates()
        ids[SAMPLE_ID_COL] = ids[SAMPLE_ID_COL].astype(str)
        ids["split"] = name
        rows.append(ids)
    out = run_dir / ID_TO_SPLIT_NAME
    pd.concat(rows, ignore_index=True).sort_values(SAMPLE_ID_COL).to_csv(out, index=False)


def load_run_splits(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load train/val/test splits from id_to_split.csv."""
    run_dir = Path(run_dir)
    splits_path = run_dir / ID_TO_SPLIT_NAME
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits at {splits_path}; run train_m.py first.")
    splits_df = pd.read_csv(splits_path, dtype={SAMPLE_ID_COL: str, "split": str})
    df = load_df()
    df[SAMPLE_ID_COL] = df[SAMPLE_ID_COL].astype(str)
    return {
        name: df[df[SAMPLE_ID_COL].isin(splits_df.loc[splits_df["split"] == name, SAMPLE_ID_COL])].reset_index(
            drop=True
        )
        for name in ("train", "val", "test")
    }


def target_bounds(df: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    """Per-target (min, max) from dataset metrics (via load_df / DATASET_DIR)."""
    if df is None:
        df = load_df()
    return {col: (float(df[col].min()), float(df[col].max())) for col in M_TARGET_COLS}


def load_df(metrics_csv: Path | None = None, inputs_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    
    # Load METRICS_CSV and prepare/resolve sample_id and cell_path columns.
    metrics_csv = metrics_csv or METRICS_CSV
    metrics_df = pd.read_csv(metrics_csv)
    n_drop = int(metrics_df.isna().any(axis=1).sum())
    if n_drop:
        print(f"Dropping {n_drop} rows with missing values from {metrics_csv}")
        metrics_df = metrics_df.dropna().reset_index(drop=True)
    metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(prep_sample_id)
    metrics_df[CELL_PATH_COL] = metrics_df[CELL_PATH_COL].map(resolve_cell_path)
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
            raise ValueError(f"{TARGET_T_DELTA=} not found in {T_DELTA_COL}")
        metrics_df = metrics_df[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Load INPUTS_CSV and prepare/resolve image paths and prompts.
    inputs_csv = inputs_csv or INPUTS_CSV
    inputs_df = pd.read_csv(inputs_csv)
    if inputs_df.isna().any().any():
        raise ValueError(f"Missing values found in {inputs_csv}")
    inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(prep_sample_id)
    inputs_df[IMAGE_PATH_COL] = inputs_df[IMAGE_PATH_COL].map(resolve_image_path)
    inputs_df[MASK_PATH_COL] = inputs_df[MASK_PATH_COL].map(resolve_mask_path)

    # Merge metrics and inputs on sample_id, keep only required columns.
    merged_df = pd.merge(metrics_df[METRICS_COLS], inputs_df[INPUTS_COLS], on=SAMPLE_ID_COL, how="left").reset_index(drop=True)
    return merged_df


def prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a loaded table into X and min-max normalized y."""
    bounds = target_bounds(data_df)
    X = data_df.drop(columns=list(M_TARGET_COLS)).copy()
    y = normalize_target_columns(data_df[list(M_TARGET_COLS)], bounds=bounds)
    return X, y


def split_df(
    X: pd.DataFrame,
    y: pd.DataFrame,
    *,
    seed: int = SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,

) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split X and y by sample_id into train/val/test sets."""
    combined = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    train, val, test = _split_df_by_sample(combined, seed=seed, train_frac=train_frac, val_frac=val_frac)
    target_cols = list(M_TARGET_COLS)
    return (
        train.drop(columns=target_cols),
        val.drop(columns=target_cols),
        test.drop(columns=target_cols),
        train[target_cols],
        val[target_cols],
        test[target_cols],
    )


def grid_axes_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique (t_start, t_end) grid axes from a metrics DataFrame."""
    return np.sort(df[T_START_COL].unique()), np.sort(df[T_END_COL].unique())


def precompute_embeddings(
    df: pd.DataFrame, predictor: MetricPredictor, device: torch.device
) -> dict[str, dict[str, torch.Tensor]]:
    """Encode the source image, mask, and prompt pair once per sample_id."""
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    images = [Image.open(i).convert("RGB") for i in samples[IMAGE_PATH_COL]]
    masks = [Image.open(m).convert("RGB") for m in samples[MASK_PATH_COL]]
    src_prompts = samples[SOURCE_PROMPT_COL].tolist()
    tar_prompts = samples[TARGET_PROMPT_COL].tolist()

    img_emb = predictor.image_encoder(images).to(device)
    mask_emb = predictor.image_encoder(masks).to(device)
    src_emb = predictor.text_encoder(src_prompts).to(device)
    tar_emb = predictor.text_encoder(tar_prompts).to(device)

    return {
        sid: {"img": img_emb[i], "mask": mask_emb[i], "src": src_emb[i], "tar": tar_emb[i]}
        for i, sid in enumerate(samples[SAMPLE_ID_COL].tolist())
    }


def build_tensors(
    X: pd.DataFrame,
    y: pd.DataFrame,
    embeddings: dict[str, dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, ...]:
    """
    Assemble per-row (sample_idx, img, mask, src, tar, t, y) tensors from X
    and y where t is (t_start, t_end) and y is the target columns. Use
    precomputed embeddings for image, mask, source, and target text.

    The model only trains on MODEL_BATCH_SLICE, so sample_idx is metadata only.
    """
    return (
        torch.tensor(pd.factorize(X[SAMPLE_ID_COL], sort=True)[0], dtype=torch.long),
        torch.stack([embeddings[i]["img"] for i in X[SAMPLE_ID_COL]]),
        torch.stack([embeddings[i]["mask"] for i in X[SAMPLE_ID_COL]]),
        torch.stack([embeddings[i]["src"] for i in X[SAMPLE_ID_COL]]),
        torch.stack([embeddings[i]["tar"] for i in X[SAMPLE_ID_COL]]),
        torch.tensor(X[[T_START_COL, T_END_COL]].values, dtype=torch.float),
        torch.tensor(y[list(M_TARGET_COLS)].values, dtype=torch.float),
    )


def create_dataloaders(
    predictor: MetricPredictor,
    train_X: pd.DataFrame,
    train_y: pd.DataFrame,
    val_X: pd.DataFrame,
    val_y: pd.DataFrame,
    test_X: pd.DataFrame,
    test_y: pd.DataFrame,
    *,
    group_train_by_sample: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from split feature and target tables."""
    
    def _dataloader(
        X: pd.DataFrame,
        y: pd.DataFrame,
        shuffle: bool,
        *,
        by_sample: bool = False,
    ) -> DataLoader:
        tensors = build_tensors(X, y, embeddings)
        dataset = TensorDataset(*tensors)
        if by_sample:
            return DataLoader(
                dataset,
                batch_sampler=SampleGridBatchSampler(tensors[IX_SAMPLE_IDX], shuffle=shuffle),
            )
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    device = next(predictor.parameters()).device
    unique_X = (
        pd.concat([train_X, val_X, test_X], ignore_index=True)
        .drop_duplicates(SAMPLE_ID_COL)
        .sort_values(SAMPLE_ID_COL)
    )
    embeddings = precompute_embeddings(unique_X, predictor, device)

    return (
        _dataloader(train_X, train_y, shuffle=True, by_sample=group_train_by_sample),
        _dataloader(val_X, val_y, shuffle=False),
        _dataloader(test_X, test_y, shuffle=False),
    )


def df_to_metric_grids(
    df: pd.DataFrame,
    sample_ids: list,
    t_start_values: list[float] | tuple[float, ...],
    t_end_values: list[float] | tuple[float, ...],
    col: str,
) -> tuple:
    """Build (N, n_start, n_end) ground-truth grid for one metric column."""
    t_start_values = list(t_start_values)
    t_end_values = list(t_end_values)
    n_img = len(sample_ids)
    n1, n2 = len(t_start_values), len(t_end_values)
    sid_to_k = {sid: k for k, sid in enumerate(sample_ids)}
    i_of = {v: i for i, v in enumerate(t_start_values)}
    j_of = {v: j for j, v in enumerate(t_end_values)}
    out = np.full((n_img, n1, n2), np.nan)
    for row in df.itertuples():
        out[sid_to_k[getattr(row, SAMPLE_ID_COL)], i_of[getattr(row, T_START_COL)], j_of[getattr(row, T_END_COL)]] = getattr(row, col)
    return out, {"sid_to_k": sid_to_k, "i_of": i_of, "j_of": j_of}

def _split_df_by_sample(
    df: pd.DataFrame,
    seed: int = SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    sample_col: str = SAMPLE_ID_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by sample_id so each edit triple stays wholly in one split."""
    sample_ids = sorted(df[sample_col].unique())
    n = len(sample_ids)
    rng = np.random.default_rng(seed)
    perm = list(rng.permutation(sample_ids))
    n_train = max(1, round(train_frac * n))
    n_val = max(0, round(val_frac * n))
    if n_train + n_val >= n:
        n_val = max(0, min(n_val, n - n_train - 1))
    train_ids = set(perm[:n_train])
    val_ids = set(perm[n_train : n_train + n_val])
    test_ids = set(perm[n_train + n_val :])
    if not test_ids and n > 1:
        moved = perm[n_train - 1]
        train_ids.remove(moved)
        test_ids.add(moved)
    train = df[df[sample_col].isin(train_ids)].reset_index(drop=True)
    val = df[df[sample_col].isin(val_ids)].reset_index(drop=True)
    test = df[df[sample_col].isin(test_ids)].reset_index(drop=True)
    return train, val, test
