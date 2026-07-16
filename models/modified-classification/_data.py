"""Shared data loading and embedding helpers for M and T training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import time
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler

from model_m import MetricPredictor
from settings import *


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

# CellEmbeddingDataset / batch layout. sample_idx is metadata only — not model input.
IX_SAMPLE_IDX = 0
IX_IMG = 1
IX_MASK = 2
IX_SRC = 3
IX_TAR = 4
IX_T = 5
IX_Y = 6
MODEL_BATCH_SLICE = slice(IX_IMG, IX_Y + 1)

# One dataset item (and collated batch).
CellItem = tuple[
    torch.Tensor,  # sample_idx
    torch.Tensor,  # img
    torch.Tensor,  # mask
    torch.Tensor,  # src
    torch.Tensor,  # tar
    torch.Tensor,  # t
    torch.Tensor,  # y
]


def model_inputs(batch: CellItem, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return (img, mask, src, tar, t, y) on device."""
    # Embeddings live on CPU in the dataset; move only this batch to GPU.
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


@dataclass(frozen=True)
class EmbeddingTables:
    """One embedding row per unique sample_id. Shared across train/val/test datasets."""

    img: torch.Tensor   # (n_samples, img_dim)
    mask: torch.Tensor  # (n_samples, img_dim)
    src: torch.Tensor   # (n_samples, text_dim)
    tar: torch.Tensor   # (n_samples, text_dim)


class CellEmbeddingDataset(Dataset[CellItem]):
    """
    Map each grid cell to embeddings via a shared EmbeddingTables.

    The dataset has one row per (sample_id, t_start, t_end). Image/mask/text
    embeddings depend only on sample_id, so train/val/test datasets all reference
    the same EmbeddingTables and only store per-cell sample_idx, t, and y.
    """

    def __init__(
        self,
        sample_idx: torch.Tensor,
        emb_tables: EmbeddingTables,
        t: torch.Tensor,
        y: torch.Tensor,
    ):
        self.sample_idx = sample_idx
        self.emb_tables = emb_tables
        self.t = t
        self.y = y

    def __len__(self) -> int:
        return self.sample_idx.shape[0]

    def __getitem__(self, i: int) -> CellItem:
        sid = int(self.sample_idx[i])
        return (
            self.sample_idx[i],
            self.emb_tables.img[sid],
            self.emb_tables.mask[sid],
            self.emb_tables.src[sid],
            self.emb_tables.tar[sid],
            self.t[i],
            self.y[i],
        )


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


def load_splits_df(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load train/val/test metric tables using `id_to_split.csv` sample membership."""
    run_dir = Path(run_dir)
    splits_path = run_dir / ID_TO_SPLIT_NAME
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing splits at {splits_path}. Run train_m.py first.")
    splits_df = pd.read_csv(splits_path, dtype={SAMPLE_ID_COL: str, "split": str})
    df = load_df()
    out: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        split_ids = splits_df.loc[splits_df["split"] == name, SAMPLE_ID_COL]
        out[name] = df.loc[df[SAMPLE_ID_COL].isin(split_ids)].reset_index(drop=True)
    return out


def target_bounds(df: pd.DataFrame | None = None) -> dict[str, tuple[float, float]]:
    """Per-target (min, max) from dataset metrics (via load_df / DATASET_DIR)."""
    if df is None:
        df = load_df()
    bounds: dict[str, tuple[float, float]] = {}
    for col in M_TARGET_COLS:
        values = np.asarray(df[col], dtype=float)
        bounds[col] = (float(values.min()), float(values.max()))
    return bounds


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
        metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Load INPUTS_CSV and prepare/resolve image paths and prompts.
    inputs_csv = inputs_csv or INPUTS_CSV
    inputs_df = pd.read_csv(inputs_csv)
    if inputs_df.isna().any().any():
        raise ValueError(f"Missing values found in {inputs_csv}")
    inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(prep_sample_id)
    inputs_df[IMAGE_PATH_COL] = inputs_df[IMAGE_PATH_COL].map(resolve_image_path)
    inputs_df[MASK_PATH_COL] = inputs_df[MASK_PATH_COL].map(resolve_mask_path)

    # Merge metrics and inputs on sample_id, keep only required columns.
    metrics_part: pd.DataFrame = metrics_df.loc[:, METRICS_COLS]
    inputs_part: pd.DataFrame = inputs_df.loc[:, INPUTS_COLS]
    return pd.merge(metrics_part, inputs_part, on=SAMPLE_ID_COL, how="left").reset_index(drop=True)


def prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a loaded table into X and min-max normalized y."""
    bounds = target_bounds(data_df)
    X = data_df.drop(columns=list(M_TARGET_COLS)).copy()
    y_raw: pd.DataFrame = data_df.loc[:, list(M_TARGET_COLS)]
    y = normalize_target_columns(y_raw, bounds=bounds)
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
        train.loc[:, target_cols],
        val.loc[:, target_cols],
        test.loc[:, target_cols],
    )


def grid_axes_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted unique (t_start, t_end) grid axes from a metrics DataFrame."""
    return (
        np.sort(np.asarray(df[T_START_COL].unique())),
        np.sort(np.asarray(df[T_END_COL].unique())),
    )


def _get_embeddings(
    samples: pd.DataFrame,
    predictor: MetricPredictor,
    *,
    use_cache: bool = True,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU embedding tables for unique samples, using disk cache when possible."""
    
    # Disk cache keyed by encoder and inputs CSV so re-runs skip VAE/text encode.
    # NOTE: May be a source of future issues if the inputs CSV changes or new models are added.
    cache_key = hashlib.sha256(f"{INPUTS_CSV}|{SD_TURBO_ROOT}".encode()).hexdigest()[:12]
    cache_path = Path(OUTPUTS_DIR.parent / ".cache" / "embeddings" / f"{cache_key}.pt").resolve()
    
    sample_ids = samples[SAMPLE_ID_COL].tolist()

    # Try to load embeddings from cache if it exists and is complete.
    if use_cache and cache_path.exists():
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        id_to_i = {sid: i for i, sid in enumerate(data["sample_ids"])}
        if all(sid in id_to_i for sid in sample_ids):
            idxs = [id_to_i[sid] for sid in sample_ids]
            print(f"Loaded embeddings from cache: {cache_path} ({len(sample_ids)} samples)")
            return (
                sample_ids,
                data["img"][idxs].contiguous(),
                data["mask"][idxs].contiguous(),
                data["src"][idxs].contiguous(),
                data["tar"][idxs].contiguous(),
            )
        print(f"Embedding cache miss (incomplete): {cache_path}")

    n_samples = len(sample_ids)
    image_paths = samples[IMAGE_PATH_COL].tolist()
    mask_paths = samples[MASK_PATH_COL].tolist()
    src_prompts = samples[SOURCE_PROMPT_COL].tolist()
    tar_prompts = samples[TARGET_PROMPT_COL].tolist()
    img_chunks: list[torch.Tensor] = []
    mask_chunks: list[torch.Tensor] = []
    src_chunks: list[torch.Tensor] = []
    tar_chunks: list[torch.Tensor] = []

    # Encode embeddings in chunks (bounded peak RAM); keep results on CPU.
    print(f"Encoding embeddings for {n_samples} samples (batch_size={batch_size})...")
    for start in range(0, n_samples, batch_size):
        batch_start = time.perf_counter()
        end = min(start + batch_size, n_samples)
        images = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
        masks = [Image.open(p).convert("RGB") for p in mask_paths[start:end]]
        with torch.no_grad():
            img_chunks.append(predictor.image_encoder(images).float().cpu())
            mask_chunks.append(predictor.image_encoder(masks).float().cpu())
            src_chunks.append(predictor.text_encoder(src_prompts[start:end]).float().cpu())
            tar_chunks.append(predictor.text_encoder(tar_prompts[start:end]).float().cpu())
        del images, masks
        elapsed = time.perf_counter() - batch_start
        print(f"    Encoded [{end}/{n_samples}] ({end - start} samples in {elapsed:.2f}s)", flush=True)

    # TODO: What is the size of the embeddings? Update this comment.
    img_emb = torch.cat(img_chunks, dim=0)
    mask_emb = torch.cat(mask_chunks, dim=0)
    src_emb = torch.cat(src_chunks, dim=0)
    tar_emb = torch.cat(tar_chunks, dim=0)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "sample_ids": list(sample_ids),
            "img": img_emb,
            "mask": mask_emb,
            "src": src_emb,
            "tar": tar_emb,
        }, cache_path)
        print(f"Saved embedding cache: {cache_path}")

    return sample_ids, img_emb, mask_emb, src_emb, tar_emb


def get_embeddings_by_sample(
    df: pd.DataFrame,
    predictor: MetricPredictor,
    device: torch.device,
    *,
    use_cache: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """
    Get the embeddings for all sample_id, returned as a sample-indexed
    dictionary of separate embedding tensors.
    """
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = _get_embeddings(samples, predictor, use_cache=use_cache)
    img_emb, mask_emb, src_emb, tar_emb = img_emb.to(device), mask_emb.to(device), src_emb.to(device), tar_emb.to(device)
    return {
        sid: {"img": img_emb[i], "mask": mask_emb[i], "src": src_emb[i], "tar": tar_emb[i]}
        for i, sid in enumerate(sample_ids)
    }


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
) -> tuple[DataLoader[CellItem], DataLoader[CellItem], DataLoader[CellItem]]:
    """Build train/val/test DataLoaders from split feature and target tables."""
    
    # One shared embedding table for all splits; each dataset only stores cell-level rows.
    samples = pd.concat([train_X, val_X, test_X], ignore_index=True).drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = _get_embeddings(samples, predictor, use_cache=True)
    emb_tables = EmbeddingTables(img=img_emb, mask=mask_emb, src=src_emb, tar=tar_emb)
    sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    def _dataloader(X: pd.DataFrame, y: pd.DataFrame, shuffle: bool, by_sample: bool = False) -> DataLoader[CellItem]:
        # Make the dataset, dataloader objects for one split.
        dataset = CellEmbeddingDataset(
            torch.tensor([sample_id_to_idx[sid] for sid in X[SAMPLE_ID_COL].tolist()], dtype=torch.long),
            emb_tables,
            torch.tensor(X[[T_START_COL, T_END_COL]].values, dtype=torch.float),
            torch.tensor(y[list(M_TARGET_COLS)].values, dtype=torch.float),
        )
        if by_sample:
            # When RANKING_LOSS_WEIGHT > 0, use one sample's full timestep grid
            # per batch so pairwise ranking can compare cells within the same image.
            return DataLoader(dataset, batch_sampler=SampleGridBatchSampler(dataset.sample_idx, shuffle=shuffle))
        # Otherwise, use a fixed batch size for MSE, which treats each cell independently.
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

    return (
        # Train dataloader.
        _dataloader(train_X, train_y, shuffle=True, by_sample=group_train_by_sample),
        # Val dataloader.
        _dataloader(val_X, val_y, shuffle=False),
        # Test dataloader
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
    train_ids = list(perm[:n_train])
    val_ids = list(perm[n_train : n_train + n_val])
    test_ids = list(perm[n_train + n_val :])
    if not test_ids and n > 1:
        moved = train_ids.pop()
        test_ids.append(moved)
    train = df.loc[df[sample_col].isin(train_ids)].reset_index(drop=True)
    val = df.loc[df[sample_col].isin(val_ids)].reset_index(drop=True)
    test = df.loc[df[sample_col].isin(test_ids)].reset_index(drop=True)
    return train, val, test
