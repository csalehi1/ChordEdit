"""Data loading, splits, and training tensors for the classifier (embeddings live in embeddings.py)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embeddings import _prep_sample_id, get_embeddings
from settings import *

METRICS_COLS = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *C_TARGET_COLS]
INPUTS_COLS = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]
DATA_COLS = list(dict.fromkeys(METRICS_COLS + INPUTS_COLS))
ID_TO_SPLIT_NAME = "id_to_split.csv"

# Bucket-index label columns added by select_best_rows.
T_START_IDX_COL = "t_start_idx"
T_END_IDX_COL = "t_end_idx"


"""
Dataframes.
"""

def load_df(metrics_csv: Path | None = None, inputs_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""

    # Clean metrics CSV: drop rows that do not have target component metrics or t_delta.
    metrics_csv = metrics_csv or METRICS_CSV
    metrics_df = pd.read_csv(metrics_csv)
    n_before = len(metrics_df)
    metrics_df = metrics_df.dropna(subset=list(C_TARGET_COLS)).reset_index(drop=True)
    if len(metrics_df) < n_before:
        print(f"Dropped {n_before - len(metrics_df)} metric rows missing {list(C_TARGET_COLS)}.")
    metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(_prep_sample_id)
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
            raise ValueError(f"{TARGET_T_DELTA=} not found in {T_DELTA_COL}")
        metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Restrict the candidate cell set before the completeness check, so samples
    # labeled only outside the subset drop out rather than counting as ragged.
    # Grid values are tenths, so compare bucket indices instead of the floats.
    if CELL_SUBSET == "lower":
        starts = np.rint(metrics_df[T_START_COL].to_numpy(dtype=float) * 10)
        ends = np.rint(metrics_df[T_END_COL].to_numpy(dtype=float) * 10)
        metrics_df = metrics_df.loc[ends < starts].reset_index(drop=True)

    # Drop samples whose grid is incomplete after the row cleaning above. The
    # per-sample delta normalization ranges over a sample's own cells, so a
    # sample missing cells is scored on a different range than the rest, and
    # scores.score_df requires one baseline row and a uniform candidate count.
    cells_per_sample = metrics_df.groupby(SAMPLE_ID_COL)[T_START_COL].transform("size")
    n_cells = int(cells_per_sample.max())
    if (cells_per_sample < n_cells).any():
        dropped = metrics_df.loc[cells_per_sample < n_cells, SAMPLE_ID_COL].nunique()
        print(f"Dropped {dropped} sample(s) with fewer than {n_cells} labeled cells.")
        metrics_df = metrics_df.loc[cells_per_sample == n_cells].reset_index(drop=True)

    # Keep a precomputed score column (e.g. written back by add_data.ipynb)
    # so select_best_rows does not have to recompute it.
    metrics_cols = METRICS_COLS + ([C_TARGET_COL] if C_TARGET_COL in metrics_df.columns else [])

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
        metrics_df.loc[:, metrics_cols],
        inputs_df.loc[:, INPUTS_COLS],
        on=SAMPLE_ID_COL,
        how="left",
    ).reset_index(drop=True)


def _map_to_bucket_idx(series: pd.Series, buckets) -> pd.Series:
    """Map float timestep values to ordinal indices into buckets (float-safe)."""
    bucket_arr = np.asarray(buckets, dtype=float)

    def _index(v: float) -> int:
        matches = np.flatnonzero(np.isclose(float(v), bucket_arr))
        if len(matches) != 1:
            raise ValueError(f"Value {v} does not uniquely match buckets {bucket_arr.tolist()}.")
        return int(matches[0])

    return series.map(_index)


def drop_missing_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """Drop samples that have no prompts, i.e. no model inputs to label.

    The left merge in load_df leaves NaN prompts for ids absent from the inputs
    CSV. Idempotent, so callers that build both the label table and the grid
    table from one cell table can apply it once up front.
    """
    has_inputs = df[[SOURCE_PROMPT_COL, TARGET_PROMPT_COL]].notna().all(axis=1)
    if not (~has_inputs).any():
        return df
    dropped = df.loc[~has_inputs, SAMPLE_ID_COL].nunique()
    print(f"Dropped {dropped} sample(s) missing from {INPUTS_CSV}.")
    return df.loc[has_inputs].reset_index(drop=True)


def add_target_score(df: pd.DataFrame) -> pd.DataFrame:
    """Attach C_TARGET_COL if the metrics CSV does not already carry it.

    The per-sample delta normalization needs every cell of a grid, so this must
    run on the full one-row-per-cell table, not on selected rows.
    """
    if C_TARGET_COL in df.columns:
        return df
    df = df.copy()
    df[C_TARGET_COL] = C_TARGET_FUNC(df)
    return df


def select_best_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Pick the highest-C_TARGET_COL cell per sample and map it to bucket indices.

    Takes the one-row-per-cell table from load_df; returns one row per sample
    with T_START_IDX_COL / T_END_IDX_COL label columns.
    """
    df = add_target_score(drop_missing_inputs(df))

    # The baseline (DEFAULT_T_START, DEFAULT_T_END) cell scores exactly 0 under
    # any of the delta scores, so idxmax picks the best improving cell, or the
    # baseline itself when no candidate improves on it.
    best_idx = df.groupby(SAMPLE_ID_COL)[C_TARGET_COL].idxmax()
    keep_cols = [SAMPLE_ID_COL, T_START_COL, T_END_COL, C_TARGET_COL, *INPUTS_COLS[1:]]
    best = df.loc[best_idx, keep_cols].reset_index(drop=True)
    best[T_START_IDX_COL] = _map_to_bucket_idx(best[T_START_COL], GRID_T_START)
    best[T_END_IDX_COL] = _map_to_bucket_idx(best[T_END_COL], GRID_T_END)
    return best


def split_df(
    df: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataframe by SAMPLE_ID_COL into train/val/test."""
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


def save_split_df(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    run_dir: Path,
) -> None:
    """Save the splits dataframes as a sample-to-split mapping .csv file."""
    rows = []
    for name, df in ("train", train_df), ("val", val_df), ("test", test_df):
        ids = df[[SAMPLE_ID_COL]].drop_duplicates()
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
        raise FileNotFoundError(f"Missing splits at {splits_path}. Run train.py first.")
    splits_df = pd.read_csv(splits_path, dtype={SAMPLE_ID_COL: str, "split": str})
    df = select_best_rows(load_df())
    out: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        split_ids = splits_df.loc[splits_df["split"] == name, SAMPLE_ID_COL]
        out[name] = df.loc[df[SAMPLE_ID_COL].isin(split_ids)].reset_index(drop=True)
    return out


"""
Selection grid.

The classifier is trained on one row per sample, but its deliverable is the
cell it picks out of the whole grid. Judging that needs every cell's true
score, so the grid table packs the one-row-per-cell table into dense
(n_samples, n_cells) arrays that the selection metrics index into.
"""

@dataclass(frozen=True)
class GridTable:
    """Dense true-score grid for every sample, shared across splits.

    Cells are the (t_start, t_end) pairs actually present in the data, in a
    canonical order; the classifier's two heads are scored jointly over exactly
    these cells so it can never pick an unlabeled one.
    """

    sample_ids: np.ndarray                  # (B,) sorted
    cell_start_idx: np.ndarray              # (N,) bucket index into GRID_T_START
    cell_end_idx: np.ndarray                # (N,) bucket index into GRID_T_END
    phi: np.ndarray                         # (B, N) true C_TARGET_COL
    metrics: dict[str, np.ndarray]          # each (B, N) raw metric values
    baseline_cell: int                      # column of (DEFAULT_T_START, DEFAULT_T_END)

    def rows_for(self, sample_ids) -> np.ndarray:
        """Row indices for a split's sample ids, in the order given."""
        wanted = np.asarray(sample_ids)
        rows = np.searchsorted(self.sample_ids, wanted)
        if rows.max(initial=-1) >= len(self.sample_ids) or not np.array_equal(self.sample_ids[rows], wanted):
            raise ValueError("sample ids missing from the grid table")
        return rows


def build_grid_table(df: pd.DataFrame) -> GridTable:
    """Pack the one-row-per-cell table from load_df into a GridTable."""
    df = add_target_score(drop_missing_inputs(df))

    n_end = len(GRID_T_END)
    start_idx = _map_to_bucket_idx(df[T_START_COL], GRID_T_START).to_numpy()
    end_idx = _map_to_bucket_idx(df[T_END_COL], GRID_T_END).to_numpy()
    flat = start_idx * n_end + end_idx

    cells = np.unique(flat)
    col = np.searchsorted(cells, flat)
    sample_ids = np.sort(df[SAMPLE_ID_COL].unique())
    row = np.searchsorted(sample_ids, df[SAMPLE_ID_COL].to_numpy())

    def _pack(values: np.ndarray) -> np.ndarray:
        out = np.full((len(sample_ids), len(cells)), np.nan)
        out[row, col] = values
        return out

    phi = _pack(df[C_TARGET_COL].to_numpy(dtype=float))
    if np.isnan(phi).any():
        n_missing = int(np.isnan(phi).sum())
        raise ValueError(f"ragged grid: {n_missing} of {phi.size} (sample, cell) scores are missing")

    cell_start_idx = cells // n_end
    cell_end_idx = cells % n_end
    baseline = np.flatnonzero(
        np.isclose(np.asarray(GRID_T_START, dtype=float)[cell_start_idx], DEFAULT_T_START)
        & np.isclose(np.asarray(GRID_T_END, dtype=float)[cell_end_idx], DEFAULT_T_END)
    )
    if len(baseline) != 1:
        raise ValueError(f"expected exactly one ({DEFAULT_T_START}, {DEFAULT_T_END}) cell, found {len(baseline)}")

    return GridTable(
        sample_ids=sample_ids,
        cell_start_idx=cell_start_idx,
        cell_end_idx=cell_end_idx,
        phi=phi,
        metrics={c: _pack(df[c].to_numpy(dtype=float)) for c in C_TARGET_COLS},
        baseline_cell=int(baseline[0]),
    )


"""
Training tensors.
"""

@dataclass(frozen=True)
class EmbeddingTable:
    """One embedding row per unique sample_id; shared across train/val/test."""

    img: torch.Tensor   # (n_samples, img_dim)
    mask: torch.Tensor  # (n_samples, img_dim)
    src: torch.Tensor   # (n_samples, text_dim)
    tar: torch.Tensor   # (n_samples, text_dim)


@dataclass(frozen=True)
class SampleTensors:
    """One split's samples, resident on one device.

    One row per sample (the best grid cell): the four embeddings plus the
    bucket-index labels y1 (t_start) and y2 (t_end).
    """

    img: torch.Tensor   # (N, img_dim)
    mask: torch.Tensor  # (N, img_dim)
    src: torch.Tensor   # (N, text_dim)
    tar: torch.Tensor   # (N, text_dim)
    y1: torch.Tensor    # (N,) t_start bucket index
    y2: torch.Tensor    # (N,) t_end bucket index

    def __len__(self) -> int:
        return int(self.y1.shape[0])

    def iter_batches(self, batch_size: int, shuffle: bool = False):
        """Yield (img, mask, src, tar, y1, y2) batches."""
        n = len(self)
        order = torch.randperm(n, device=self.y1.device) if shuffle else torch.arange(n, device=self.y1.device)
        for k in range(0, n, batch_size):
            rows = order[k : k + batch_size]
            yield (
                self.img[rows],
                self.mask[rows],
                self.src[rows],
                self.tar[rows],
                self.y1[rows],
                self.y2[rows],
            )


def create_sample_tensors(
    splits: dict[str, pd.DataFrame],
    device: torch.device,
) -> dict[str, SampleTensors]:
    """Build device-resident SampleTensors for each split, sharing one embedding load.

    Embeddings come from the packed or scattered caches (get_embeddings raises
    with instructions otherwise).
    """
    samples = pd.concat(list(splits.values()), ignore_index=True).drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = get_embeddings(samples)
    emb_table = EmbeddingTable(
        img=img_emb.to(device),
        mask=mask_emb.to(device),
        src=src_emb.to(device),
        tar=tar_emb.to(device),
    )
    sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    out: dict[str, SampleTensors] = {}
    for name, df in splits.items():
        rows = torch.tensor([sample_id_to_idx[sid] for sid in df[SAMPLE_ID_COL].tolist()], dtype=torch.long, device=device)
        out[name] = SampleTensors(
            img=emb_table.img[rows],
            mask=emb_table.mask[rows],
            src=emb_table.src[rows],
            tar=emb_table.tar[rows],
            y1=torch.tensor(df[T_START_IDX_COL].values, dtype=torch.long, device=device),
            y2=torch.tensor(df[T_END_IDX_COL].values, dtype=torch.long, device=device),
        )
    return out
