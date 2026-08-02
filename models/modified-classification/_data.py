"""Data loading, splits, and dataloaders for M and T training (embeddings live in embeddings.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from model_m import SurrogateModel
from embeddings import _prep_sample_id, get_embeddings
from settings import *

METRICS_COLS = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *M_TARGET_COLS]
INPUTS_COLS = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]
DATA_COLS = list(dict.fromkeys(METRICS_COLS + INPUTS_COLS))
ID_TO_SPLIT_NAME = "id_to_split.csv"

# CellItem: (sample_idx, img, mask, src, tar, t, y); model uses indices 1..6.
MODEL_BATCH_SLICE = slice(1, 7)

CellItem = tuple[
    torch.Tensor,  # sample_idx
    torch.Tensor,  # img
    torch.Tensor,  # mask
    torch.Tensor,  # src
    torch.Tensor,  # tar
    torch.Tensor,  # t
    torch.Tensor,  # y
]


# --- Dataset and loaders ----------------------------------------------------------

def model_inputs(batch: CellItem, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return (img, mask, src, tar, t, y) on device."""
    return tuple(x.to(device) for x in batch[MODEL_BATCH_SLICE])


@dataclass(frozen=True)
class EmbeddingTable:
    """One embedding row per unique sample_id; shared across train/val/test."""

    img: torch.Tensor   # (n_samples, img_dim)
    mask: torch.Tensor  # (n_samples, img_dim)
    src: torch.Tensor   # (n_samples, text_dim)
    tar: torch.Tensor   # (n_samples, text_dim)


class CellDataset(Dataset[CellItem]):
    """One row per (sample_id, t_start, t_end); embeddings looked up by sample_idx."""

    def __init__(
        self,
        sample_idx: torch.Tensor,
        emb_tables: EmbeddingTable,
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


# Used for per-sample ranking loss.
# Create batches of cells that belong to the same sample_id.
class PerSampleBatchSampler(Sampler[list[int]]):
    """One sample_id's full timestep grid per batch for per-sample ranking loss."""

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


"""
Dataframes.
"""

def load_df(metrics_csv: Path | None = None, inputs_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""

    # Clean metrics CSV: drop rows that do not have target component metrics or t_delta.
    metrics_csv = metrics_csv or METRICS_CSV
    metrics_df = pd.read_csv(metrics_csv)
    n_before = len(metrics_df)
    metrics_df = metrics_df.dropna(subset=list(M_TARGET_COLS)).reset_index(drop=True)
    if len(metrics_df) < n_before:
        print(f"Dropped {n_before - len(metrics_df)} metric rows missing {list(M_TARGET_COLS)}.")
    metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(_prep_sample_id)
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
            raise ValueError(f"{TARGET_T_DELTA=} not found in {T_DELTA_COL}")
        metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Clean inputs CSV: drop maskless rows, but use downloaded_mask_image_path when mask_image_path is empty.
    inputs_csv = inputs_csv or INPUTS_CSV
    inputs_df = pd.read_csv(inputs_csv)
    mask = inputs_df[MASK_PATH_COL].replace("", pd.NA)
    if "downloaded_mask_image_path" in inputs_df.columns:
        mask = mask.fillna(inputs_df["downloaded_mask_image_path"].replace("", pd.NA))
    inputs_df[MASK_PATH_COL] = mask
    n_before = len(inputs_df)
    inputs_df = inputs_df.dropna(subset=[MASK_PATH_COL]).reset_index(drop=True)
    if len(inputs_df) < n_before:
        print(f"Dropped {n_before - len(inputs_df)} input rows with no mask path.")
    if inputs_df[INPUTS_COLS].isna().to_numpy().any():
        raise ValueError(f"Missing values found in {inputs_csv}.")
    inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(_prep_sample_id)
    for col in (IMAGE_PATH_COL, MASK_PATH_COL):
        inputs_df[col] = [
            str(p) if (p := Path(path)).is_absolute() else str(DATASET_DIR / path)
            for path in inputs_df[col]
        ]

    return pd.merge(
        metrics_df.loc[:, METRICS_COLS],
        inputs_df.loc[:, INPUTS_COLS],
        on=SAMPLE_ID_COL,
        how="left",
    ).reset_index(drop=True)


def prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a loaded table into X and raw-metric y (PSNR / CLIP units)."""
    X_df = data_df.drop(columns=list(M_TARGET_COLS)).copy()
    # Keep raw labels: no global min-max. Per-sample range norm happens later
    # at score time (calc_normalized_deltas / LINEX). MSE still z-scores in train_m.
    y_df: pd.DataFrame = data_df.loc[:, list(M_TARGET_COLS)].copy()
    return X_df, y_df


def split_df(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    *,
    seed: int = SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframes by SAMPLE_ID_COL into train/val/test and X/y dataframes."""

    def split_df_by_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        sample_ids = np.sort(np.asarray(df[SAMPLE_ID_COL].unique()))
        n_samples = len(sample_ids)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(sample_ids)
        n_train = max(1, round(train_frac * n_samples))
        n_val = max(0, min(round(val_frac * n_samples), n_samples - n_train - 1))
        train_ids = perm[:n_train]
        val_ids = perm[n_train : n_train + n_val]
        test_ids = perm[n_train + n_val :]
        if test_ids.size == 0 and n_samples > 1:
            test_ids = train_ids[-1:]
            train_ids = train_ids[:-1]
        sample_col = df[SAMPLE_ID_COL].to_numpy()
        train = df.loc[np.isin(sample_col, train_ids)].reset_index(drop=True)
        val = df.loc[np.isin(sample_col, val_ids)].reset_index(drop=True)
        test = df.loc[np.isin(sample_col, test_ids)].reset_index(drop=True)
        return train, val, test

    Xy_df = pd.concat([X_df.reset_index(drop=True), y_df.reset_index(drop=True)], axis=1)
    train, val, test = split_df_by_sample(Xy_df)
    target_cols = list(M_TARGET_COLS)
    return (
        train.drop(columns=target_cols),
        val.drop(columns=target_cols),
        test.drop(columns=target_cols),
        train.loc[:, target_cols],
        val.loc[:, target_cols],
        test.loc[:, target_cols],
    )


def save_split_df(
    train_X: pd.DataFrame,
    val_X: pd.DataFrame,
    test_X: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Save the splits dataframes as a sample-to-split mapping .csv file."""
    rows = []
    for name, X in ("train", train_X), ("val", val_X), ("test", test_X):
        ids = X[[SAMPLE_ID_COL]].drop_duplicates()
        ids[SAMPLE_ID_COL] = ids[SAMPLE_ID_COL].astype(str)
        ids["split"] = name
        rows.append(ids)
    out = run_dir / ID_TO_SPLIT_NAME
    pd.concat(rows, ignore_index=True).sort_values(SAMPLE_ID_COL).to_csv(out, index=False)


def load_split_df(run_dir: Path) -> dict[str, pd.DataFrame]:
    """Load the splits dataframes using `id_to_split.csv` sample membership."""
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


"""
Dataloader utilities.
"""

def create_dataloaders(
    predictor: SurrogateModel,
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
    samples = pd.concat([train_X, val_X, test_X], ignore_index=True).drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = get_embeddings(samples, predictor)
    emb_table = EmbeddingTable(img=img_emb, mask=mask_emb, src=src_emb, tar=tar_emb)
    sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    def _dataloader(X_df: pd.DataFrame, y_df: pd.DataFrame, shuffle: bool, by_sample: bool = False) -> DataLoader[CellItem]:
        dataset = CellDataset(
            torch.tensor([sample_id_to_idx[sid] for sid in X_df[SAMPLE_ID_COL].tolist()], dtype=torch.long),
            emb_table,
            torch.tensor(X_df[[T_START_COL, T_END_COL]].values, dtype=torch.float),
            torch.tensor(y_df[list(M_TARGET_COLS)].values, dtype=torch.float),
        )
        if by_sample:
            return DataLoader(dataset, batch_sampler=PerSampleBatchSampler(dataset.sample_idx, shuffle=shuffle))
        return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)

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
