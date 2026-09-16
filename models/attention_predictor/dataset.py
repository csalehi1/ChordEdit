# dataset.py

"""
Dataset assembly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _helpers import get_default_cell, prep_sample_id
from embeddings import EmbeddingsTable, SampleEmbeddings, get_embeddings
from model import to_deltas
from settings import *

ID_TO_SPLIT_NAME = "id_to_split.csv"
TRAIN_METADATA_NAME = "train_metadata.pt"


"""
Sample and split classes.
"""


@dataclass(frozen=True)
class DatasetMetadata:
    """Train-global grid identity and regression stats, shared by every split."""

    n_cells: int                     # cells per grid
    cell_labels: torch.Tensor        # (n_cells, 2) being (t_start, t_end)
    default_cell: int                # shared default-cell index; raises if not unique
    mean_surface: torch.Tensor       # (n_cells, C) raw CLIP/PSNR
    loss_scale: torch.Tensor         # (C,) raw units per regression-space unit


def get_train_metadata(run_dir: Path, built: DatasetMetadata | None = None) -> DatasetMetadata:
    """Load saved train metadata, or write `built` and return it."""

    path = Path(run_dir) / TRAIN_METADATA_NAME
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["n_cells"] = int(payload["n_cells"])
        payload["default_cell"] = int(payload["default_cell"])
        return DatasetMetadata(**payload)
    if built is None:
        raise FileNotFoundError(f"Missing {path}")

    torch.save({
        "n_cells": built.n_cells,
        "cell_labels": built.cell_labels.detach().cpu().contiguous(),
        "default_cell": built.default_cell,
        "mean_surface": built.mean_surface.detach().cpu().contiguous(),
        "loss_scale": built.loss_scale.detach().cpu().contiguous(),
    }, path)
    print(f"Saved {path.name}.")
    return built


@dataclass(frozen=True)
class SampleData:
    """One sample's data, embeddings and targets."""

    sample_id: str
    x: SampleEmbeddings
    y: torch.Tensor                  # (n_cells, C) regression-space deltas
    y_raw: torch.Tensor              # (n_cells, C) raw CLIP/PSNR


@dataclass(frozen=True)
class DatasetSplit:
    """One split's grids over the shared embedding table."""

    split_name: str                  # "train" / "val" / "test"
    sample_ids: tuple[str, ...]      # (N,) one sample_id per grid
    x: EmbeddingsTable               # shared across splits; index via sample_ids
    y: torch.Tensor                  # (N, n_cells, C) regression-space deltas
    y_raw: torch.Tensor              # (N, n_cells, C) raw CLIP/PSNR
    image_shape: tuple[int, ...]     # per-sample image_tokens shape
    source_shape: tuple[int, ...]    # per-sample source_tokens shape
    target_shape: tuple[int, ...]    # per-sample target_tokens shape
    feature_shape: tuple[int, ...]   # per-sample mask_features shape
    mask_shape: tuple[int, ...]      # per-sample mask_tokens shape
    metadata: DatasetMetadata        # train-global grid + regression stats

    def __post_init__(self):
        # Translate sample ids to table rows once, so gather is pure tensor
        # indexing, and cache the id -> grid position map __getitem__ uses.
        object.__setattr__(self, "_table_idx", self.x.sample_idx(list(self.sample_ids)))
        object.__setattr__(self, "_sid_to_i", {sid: i for i, sid in enumerate(self.sample_ids)})

    @property
    def n_samples(self) -> int:
        """Number of samples in the split."""
        return len(self.sample_ids)

    @property
    def n_cells(self) -> int:
        return self.metadata.n_cells

    @property
    def default_cell(self) -> int:
        return self.metadata.default_cell

    @property
    def cell_labels(self) -> torch.Tensor:
        return self.metadata.cell_labels

    def __getitem__(self, sample_id: str) -> SampleData:
        """Get SampleData by sample_id."""
        i = self._sid_to_i[sample_id]
        return SampleData(
            sample_id=sample_id,
            x=self.x[sample_id],
            y=self.y[i],
            y_raw=self.y_raw[i],
        )

    def gather(self, sel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Tensors for a batch of grids at integer indices sel."""
        idx = self._table_idx[sel]
        table = self.x
        return (
            table.image_tokens[idx],
            table.source_tokens[idx],
            table.target_tokens[idx],
            table.source_mask[idx],
            table.target_mask[idx],
            table.mask_features[idx],
            table.mask_tokens[idx],
            self.y[sel],
            self.y_raw[sel],
        )


@dataclass(frozen=True)
class DatasetSplitBundle:
    """Separate splits of the dataset, train/val/test."""

    splits: dict[str, DatasetSplit]

    @property
    def train(self) -> DatasetSplit:
        return self.splits["train"]

    @property
    def val(self) -> DatasetSplit:
        return self.splits["val"]

    @property
    def test(self) -> DatasetSplit:
        return self.splits["test"]


"""
Dataframes and splits.
"""


def get_df() -> pd.DataFrame:
    """Load the metrics, attach source-image paths and prompts, one row per cell."""

    def _load_df(metrics_csv: Path, inputs_csv: Path) -> pd.DataFrame:
        """Load and merge one metrics/inputs CSV pair, one row per cell."""

        def _load_pie_bench_df() -> pd.DataFrame:
            "Swap test for PIE-Bench samples with --pie-bench (pie_-prefixed ids)."
            pie_df = _load_df(PIE_METRICS_CSV, PIE_INPUTS_CSV)
            pie_df[SAMPLE_ID_COL] = PIE_SAMPLE_ID_PREFIX + pie_df[SAMPLE_ID_COL]
            return pie_df

        def _reduce_label_seeds(metrics_df: pd.DataFrame) -> pd.DataFrame:
            """Average list-valued metric cells over generation seeds.

            The seed column holds the generation seeds in the same order as the
            PSNR/CLIP lists, e.g. seed '[42, 43, 44, 45]' and psnr '[a, b, c, d]'.
            `<col>` becomes the mean over LABEL_SEEDS (the training labels) and
            `<col>__eval` the mean over EVAL_LABEL_SEEDS (the val/test labels). A NaN
            anywhere in a list voids the cell, so the sample set does not depend on the
            seeds chosen. Scalar cells (no seed column, or a single value) pass through.
            """

            def _parse_list_cell(value) -> np.ndarray:
                if isinstance(value, str) and value.strip().startswith("["):
                    parts = [p.strip() for p in value.strip()[1:-1].split(",")]
                    return np.array([float(p) if p else np.nan for p in parts], dtype=np.float64)
                if pd.isna(value):
                    return np.array([], dtype=np.float64)
                return np.array([float(value)], dtype=np.float64)

            def _at_seeds(vals: np.ndarray, i: int, wanted: tuple[int, ...] | None) -> np.ndarray:
                """The entries of vals at the wanted seeds of row i, or all of them."""
                if wanted is None or row_seeds is None or vals.size == 1:
                    return vals
                cell_seeds = row_seeds[i]
                if cell_seeds.size != vals.size:
                    raise ValueError(f"{vals.size} values but {cell_seeds.size} seeds in row {i}")
                hits = [np.flatnonzero(cell_seeds == seed) for seed in wanted]
                if any(h.size == 0 for h in hits):
                    raise ValueError(f"Row {i} has seeds {cell_seeds.tolist()}, expected {list(wanted)}")
                return vals[[int(h[0]) for h in hits]]

            row_seeds = [_parse_list_cell(v) for v in metrics_df[SEED_COL]] if SEED_COL in metrics_df.columns else None
            for col in TARGET_COLS:
                values = [_parse_list_cell(v) for v in metrics_df[col]]
                out = np.full((len(values), 2), np.nan, dtype=np.float64)
                for i, vals in enumerate(values):
                    if vals.size and not np.isnan(vals).any():
                        out[i] = _at_seeds(vals, i, LABEL_SEEDS).mean(), _at_seeds(vals, i, EVAL_LABEL_SEEDS or LABEL_SEEDS).mean()
                metrics_df[col], metrics_df[f"{col}__eval"] = out[:, 0], out[:, 1]
            return metrics_df

        is_primary = metrics_csv == METRICS_CSV

        # Clean metrics CSV: drop rows that do not have target metrics or t_delta.
        metrics_df = pd.read_csv(metrics_csv)
        metrics_df = _reduce_label_seeds(metrics_df)
        n_before = len(metrics_df)
        metrics_df = metrics_df.dropna(subset=list(TARGET_COLS)).reset_index(drop=True)
        if len(metrics_df) < n_before:
            print(f"Dropped {n_before - len(metrics_df)} metric rows missing {list(TARGET_COLS)}.")
        metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(prep_sample_id)
        if TARGET_T_DELTA is not None:
            if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
                raise ValueError(f"Expected {TARGET_T_DELTA=} in {T_DELTA_COL}")
            metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

        # Keep only samples with a complete grid.
        cells_per_sample = metrics_df.groupby(SAMPLE_ID_COL)[SAMPLE_ID_COL].transform("size")
        n_cells = int(cells_per_sample.mode().iat[0])
        incomplete = cells_per_sample != n_cells
        if incomplete.any():
            n_dropped = metrics_df.loc[incomplete, SAMPLE_ID_COL].nunique()
            print(f"Dropped {n_dropped} sample(s) with fewer than {n_cells} labeled cells.")
            metrics_df = metrics_df.loc[~incomplete].copy()

        # Optional slice of a large dataset.
        if is_primary and MAX_SAMPLES is not None:
            keep = np.sort(metrics_df[SAMPLE_ID_COL].unique())[:MAX_SAMPLES]
            if len(keep) < MAX_SAMPLES:
                print(f"{MAX_SAMPLES=} exceeds the {len(keep)} complete samples available; using all of them.")
            metrics_df = metrics_df.loc[metrics_df[SAMPLE_ID_COL].isin(keep)].copy()
            print(f"Sliced to {len(keep)} sample(s) ({MAX_SAMPLES=}).")

        # Clean inputs CSV.
        inputs_cols = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]
        inputs_df = pd.read_csv(inputs_csv)
        mask = inputs_df[MASK_PATH_COL].replace("", pd.NA)
        if "downloaded_mask_image_path" in inputs_df.columns:
            mask = mask.fillna(inputs_df["downloaded_mask_image_path"].replace("", pd.NA))
        inputs_df[MASK_PATH_COL] = mask
        n_before = len(inputs_df)
        inputs_df = inputs_df.dropna(subset=[MASK_PATH_COL]).reset_index(drop=True)
        if len(inputs_df) < n_before:
            print(f"Dropped {n_before - len(inputs_df)} input rows with no mask path.")
        if inputs_df[inputs_cols].isna().to_numpy().any():
            raise ValueError(f"Expected no missing values in {inputs_csv}")
        inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(prep_sample_id)
        for col in (IMAGE_PATH_COL, MASK_PATH_COL):
            inputs_df[col] = [str(p) if (p := Path(path)).is_absolute() else str(DATASET_DIR / path) for path in inputs_df[col]]

        metrics_cols = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *TARGET_COLS, *(f"{c}__eval" for c in TARGET_COLS)]
        df = pd.merge(metrics_df.loc[:, metrics_cols], inputs_df.loc[:, inputs_cols], on=SAMPLE_ID_COL, how="left").reset_index(drop=True)

        if is_primary and PIE_BENCH:
            df = pd.concat([df, _load_pie_bench_df()], ignore_index=True)
        return df

    return _load_df(METRICS_CSV, INPUTS_CSV)


def get_splits_df(splits_df_path: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create train/val/test frames keyed by split name. y is raw CLIP/PSNR."""
    
    def _prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the loaded table into features X and raw metric targets y."""
        X_df = data_df.drop(columns=list(TARGET_COLS)).copy()
        for col in TARGET_COLS:
            X_df[f"{col}__raw"] = data_df[col].to_numpy()
        y_df = data_df.loc[:, list(TARGET_COLS)].copy()
        return X_df, y_df

    def _split_df(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
        """Split X/y by sample_id into train/val/test."""
        sids = X_df[SAMPLE_ID_COL]
        is_pie = sids.str.startswith(PIE_SAMPLE_ID_PREFIX)
        ue_ids = np.sort(sids[~is_pie].unique())
        n_samples = len(ue_ids)
        perm = np.random.default_rng(SPLIT_SEED).permutation(ue_ids)
        n_train = max(1, round(TRAIN_FRAC * n_samples))
        n_val = max(0, min(round(VAL_FRAC * n_samples), n_samples - n_train - 1))
        train_ids = perm[:n_train]
        val_ids = perm[n_train : n_train + n_val]
        test_ids = perm[n_train + n_val :]
        if test_ids.size == 0 and n_samples > 1:
            test_ids = train_ids[-1:]
            train_ids = train_ids[:-1]
        if PIE_BENCH:
            test_ids = np.sort(sids[is_pie].unique())

        def _select(ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
            keep = sids.isin(ids).to_numpy()
            return X_df.loc[keep].reset_index(drop=True), y_df.loc[keep].reset_index(drop=True)

        train_X, train_y = _select(train_ids)
        val_X, val_y = _select(val_ids)
        test_X, test_y = _select(test_ids)
        return train_X, val_X, test_X, train_y, val_y, test_y

    def _save_splits_df(train_X: pd.DataFrame, val_X: pd.DataFrame, test_X: pd.DataFrame) -> None:
        """Save the sample-to-split mapping as id_to_split.csv in the run directory."""
        rows = []
        for name, X in ("train", train_X), ("val", val_X), ("test", test_X):
            ids = X[[SAMPLE_ID_COL]].drop_duplicates()
            ids[SAMPLE_ID_COL] = ids[SAMPLE_ID_COL].astype(str)
            ids["split"] = name
            rows.append(ids)
        splits_df_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(rows, ignore_index=True).sort_values(SAMPLE_ID_COL).to_csv(splits_df_path, index=False)
        print(f"Saved {splits_df_path.name}")

    X_df, y_df = _prepare_df(get_df())

    if splits_df_path.exists():
        membership = pd.read_csv(splits_df_path, dtype={SAMPLE_ID_COL: str, "split": str})
        out = {}
        for name in ("train", "val", "test"):
            ids = membership.loc[membership["split"] == name, SAMPLE_ID_COL]
            keep = X_df[SAMPLE_ID_COL].isin(ids).to_numpy()
            out[name] = (X_df.loc[keep].reset_index(drop=True), y_df.loc[keep].reset_index(drop=True))
        print(f"Replayed splits from {splits_df_path.name}")
        return out

    train_X, val_X, test_X, train_y, val_y, test_y = _split_df(X_df, y_df)
    _save_splits_df(train_X, val_X, test_X)
    return {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}


def get_dataset(
    device: torch.device,
    run_dir: Path,
    splits_df: dict[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> DatasetSplitBundle:
    """Build the device-resident SplitDatasetBundle."""
    run_dir = Path(run_dir)

    def _get_splits_df() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the splits_df from get_splits_df()."""
        return splits_df if splits_df is not None else get_splits_df(run_dir / ID_TO_SPLIT_NAME)

    def _get_embeddings() -> EmbeddingsTable:
        """Get the embeddings from get_embeddings()."""
        samples = pd.concat([X for X, _ in splits.values()], ignore_index=True)
        return get_embeddings(samples, device)

    def _grid_arrange(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """One split's cells as grids: (sample_ids, t (n_cells, 2), y_raw, y_raw at the eval seeds)."""
        del y_df  # raw metrics live on X_df as *__raw and *__eval columns
        Xs = X_df.sort_values([SAMPLE_ID_COL, T_START_COL, T_END_COL], kind="stable")
        sample_ids = Xs[SAMPLE_ID_COL].to_numpy()
        unique_ids, counts = np.unique(sample_ids, return_counts=True)
        if len(set(counts)) != 1:
            raise ValueError(f"Expected complete grids, got cell counts {sorted(set(counts))}")
        n, n_cells = len(unique_ids), int(counts[0])

        t = Xs[[T_START_COL, T_END_COL]].to_numpy(dtype=np.float64).reshape(n, n_cells, 2)
        canonical = np.round(t, 6).reshape(n, -1)
        keys, key_counts = np.unique(canonical, axis=0, return_counts=True)
        keep = (canonical == keys[key_counts.argmax()]).all(axis=1)
        if not keep.all():
            print(f"Dropped {int((~keep).sum())} sample(s) whose grid does not match the split's canonical cells.")

        def _grid(suffix: str) -> torch.Tensor:
            cols = [f"{c}{suffix}" for c in TARGET_COLS]
            return torch.tensor(Xs[cols].to_numpy(), dtype=torch.float).reshape(n, n_cells, -1)[torch.as_tensor(keep)]

        t_pairs = torch.tensor(t[keep][0], dtype=torch.float64)
        return list(unique_ids[keep]), t_pairs, _grid("__raw"), _grid("__eval")

    def _metadata_to_device(meta: DatasetMetadata) -> DatasetMetadata:
        """Copy train metadata tensors onto the dataset device."""
        return DatasetMetadata(
            n_cells=int(meta.n_cells),
            cell_labels=meta.cell_labels.to(device=device),
            default_cell=int(meta.default_cell),
            mean_surface=meta.mean_surface.to(device=device, dtype=torch.float),
            loss_scale=meta.loss_scale.to(device=device, dtype=torch.float),
        )

    def _get_metadata() -> DatasetMetadata:
        """Train-split grid and CLIP/PSNR stats, shared by every split and the model."""
        if (Path(run_dir) / TRAIN_METADATA_NAME).exists():
            return _metadata_to_device(get_train_metadata(run_dir))
        train_pairs, train_raw = arranged["train"][1], arranged["train"][2]
        if LOSS_SCALE == "median_range":
            # A typical sample's grid range per column, so deltas mostly lie in [-1, 1].
            ranges = train_raw.amax(dim=1) - train_raw.amin(dim=1)
            loss_scale = ranges.double().median(dim=0).values.to(dtype=train_raw.dtype)
        else:
            loss_scale = torch.tensor(LOSS_SCALE_VALUES, dtype=train_raw.dtype)
        built = DatasetMetadata(
            n_cells=int(train_pairs.shape[0]),
            cell_labels=train_pairs,
            default_cell=get_default_cell(train_pairs),
            mean_surface=train_raw.double().mean(dim=0).to(dtype=train_raw.dtype),
            loss_scale=loss_scale,
        )
        return _metadata_to_device(get_train_metadata(run_dir, built))

    def _get_splits(metadata: DatasetMetadata) -> dict[str, DatasetSplit]:
        """Get the splits from the packed grids, with val/test labels at the eval seeds if set."""
        out: dict[str, DatasetSplit] = {}
        for name, (sample_ids, t_pairs, y_raw, y_eval) in arranged.items():
            if int(t_pairs.shape[0]) != metadata.n_cells:
                raise ValueError(f"{name} grid has {t_pairs.shape[0]} cells, train has {metadata.n_cells}")
            if name != "train" and EVAL_LABEL_SEEDS is not None:
                y_raw = y_eval
            y_raw = y_raw.to(device)
            out[name] = DatasetSplit(
                split_name=name,
                sample_ids=tuple(sample_ids),
                x=table,
                y=to_deltas(y_raw, metadata.default_cell, metadata.loss_scale),
                y_raw=y_raw,
                image_shape=table.image_shape,
                source_shape=table.source_shape,
                target_shape=table.target_shape,
                feature_shape=table.feature_shape,
                mask_shape=table.mask_shape,
                metadata=metadata,
            )
        return out

    splits = _get_splits_df()
    table = _get_embeddings()
    arranged = {name: _grid_arrange(X, y) for name, (X, y) in splits.items()}
    return DatasetSplitBundle(splits=_get_splits(_get_metadata()))
