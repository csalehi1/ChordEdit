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

from _helpers import MEAN_SURFACE_NAME, gather_at_pairs, prep_sample_id
from embeddings import EmbeddingsTable, SampleEmbeddings, get_embeddings
from settings import *

ID_TO_SPLIT_NAME = "id_to_split.csv"


"""
Sample and split classes.
"""


@dataclass(frozen=True)
class SampleData:
    """One sample's embeddings and metric surfaces."""

    sample_id: str
    x: SampleEmbeddings
    y: torch.Tensor                  # (n_cells, C) PREDICTION_SPACE targets
    y_raw: torch.Tensor              # (n_cells, C) "raw" targets


@dataclass(frozen=True)
class SampleMeta:
    """Model-sizing and checkpoint contract, derived once from the train split."""

    img_shape: tuple[int, ...]     # per-sample image_tokens shape
    src_shape: tuple[int, ...]    # per-sample source_tokens shape
    tgt_shape: tuple[int, ...]    # per-sample target_tokens shape
    n_cells: int                     # cells per grid
    t: torch.Tensor                  # (n_cells, 2) being (t_start, t_end), float64 CPU
    default_cell: int                # shared default-cell index; raises if not unique


@dataclass(frozen=True)
class SplitDataset:
    """One split's grids over the shared embedding table, resident on one device."""

    split_name: str                  # "train" / "val" / "test"
    sample_ids: tuple[str, ...]      # (N,) one sample_id per grid
    embs: EmbeddingsTable            # shared across splits; index via sample_ids
    y: torch.Tensor                  # (N, n_cells, C) PREDICTION_SPACE targets
    y_raw: torch.Tensor              # (N, n_cells, C) "raw" targets
    default_cell: int                # position of the default cell within each grid

    # (n_cells, C) used when PREDICTION_SPACE is "residuals"
    mean_surface: torch.Tensor | None = None

    def __post_init__(self):
        # Translate sample ids to table rows once, so gather is pure tensor
        # indexing, and cache the id -> grid position map __getitem__ uses.
        object.__setattr__(self, "_table_idx", self.embs.sample_idx(list(self.sample_ids)))
        object.__setattr__(self, "_sid_to_i", {sid: i for i, sid in enumerate(self.sample_ids)})

    @property
    def n_samples(self) -> int:
        """Number of samples in the split."""
        return len(self.sample_ids)

    def __getitem__(self, sample_id: str) -> SampleData:
        """Get SampleData by sample_id."""
        i = self._sid_to_i[sample_id]
        return SampleData(
            sample_id=sample_id,
            x=self.embs.get_sample_embeddings(sample_id),
            y=self.y[i],
            y_raw=self.y_raw[i],
        )

    def gather(self, sel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Tensors for a batch of grids at integer indices sel."""
        idx = self._table_idx[sel]
        table = self.embs
        return (
            table.image_tokens[idx],
            table.source_tokens[idx],
            table.target_tokens[idx],
            table.source_mask[idx],
            table.target_mask[idx],
            self.y[sel],
        )


@dataclass(frozen=True)
class SplitDatasetBundle:
    """Everything train.py needs from get_dataset."""

    splits: dict[str, SplitDataset]
    splits_df: dict[str, tuple[pd.DataFrame, pd.DataFrame]]
    meta: SampleMeta

    @property
    def train(self) -> SplitDataset:
        return self.splits["train"]

    @property
    def val(self) -> SplitDataset:
        return self.splits["val"]

    @property
    def test(self) -> SplitDataset:
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

        is_primary = metrics_csv == METRICS_CSV

        # Clean metrics CSV: drop rows that do not have target metrics or t_delta.
        metrics_df = pd.read_csv(metrics_csv)
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

        metrics_cols = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *TARGET_COLS]
        df = pd.merge(metrics_df.loc[:, metrics_cols], inputs_df.loc[:, inputs_cols], on=SAMPLE_ID_COL, how="left").reset_index(drop=True)

        if is_primary and PIE_BENCH:
            df = pd.concat([df, _load_pie_bench_df()], ignore_index=True)
        return df

    return _load_df(METRICS_CSV, INPUTS_CSV)


def get_splits_df(splits_df_path: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Create train/val/test frames keyed by split name in PREDICTION_SPACE."""
    
    def _prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the loaded table into features X and targets y in PREDICTION_SPACE."""
        from scores import compute_delta_df

        X_df = data_df.drop(columns=list(TARGET_COLS)).copy()
        for col in TARGET_COLS:
            X_df[f"{col}__raw"] = data_df[col].to_numpy()
        if PREDICTION_SPACE == "raws":
            y_df = data_df.loc[:, list(TARGET_COLS)].copy()
        else:
            y_df = compute_delta_df(data_df, *TARGET_COLS)
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
) -> SplitDatasetBundle:
    """Build the device-resident SplitDatasetBundle."""
    run_dir = Path(run_dir)
    mean_surface: dict | None = None

    def _get_splits_df() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
        """Get the splits_df from get_splits_df()."""
        return splits_df if splits_df is not None else get_splits_df(run_dir / ID_TO_SPLIT_NAME)

    def _get_embeddings() -> EmbeddingsTable:
        """Get the embeddings from get_embeddings()."""
        samples = pd.concat([X for X, _ in splits.values()], ignore_index=True)
        return get_embeddings(samples, device)

    def _grid_arrange(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
        """One split's cells as grids: (sample_ids, t (n_cells, 2), y, y_raw)."""
        Xs = X_df.sort_values([SAMPLE_ID_COL, T_START_COL, T_END_COL], kind="stable")
        order = Xs.index.to_numpy()
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

        raw_cols = [f"{c}__raw" for c in TARGET_COLS]
        y = torch.tensor(y_df.loc[order, list(TARGET_COLS)].to_numpy(), dtype=torch.float).reshape(n, n_cells, -1)
        y_raw = torch.tensor(Xs[raw_cols].to_numpy(), dtype=torch.float).reshape(n, n_cells, -1)
        t_pairs = torch.tensor(t[keep][0], dtype=torch.float64)
        return list(unique_ids[keep]), t_pairs, y[torch.as_tensor(keep)], y_raw[torch.as_tensor(keep)]

    def _default_cell(t_pairs: torch.Tensor) -> int:
        """Index of the default (t_start, t_end) cell in a canonical grid."""
        hits = (
            torch.isclose(t_pairs[:, 0], torch.tensor(DEFAULT_T_START, dtype=t_pairs.dtype))
            & torch.isclose(t_pairs[:, 1], torch.tensor(DEFAULT_T_END, dtype=t_pairs.dtype))
        ).nonzero(as_tuple=False).flatten()
        if hits.numel() != 1:
            raise ValueError(f"Expected exactly one default cell at ({DEFAULT_T_START}, {DEFAULT_T_END}), got {hits.numel()}")
        return int(hits[0])

    def _get_metadata() -> SampleMeta:
        """Get the metadata from the train split."""

        def _calc_mean_surface() -> dict | None:
            """Mean train target surface over the grid axes, saved to run_dir."""
            if PREDICTION_SPACE == "raws":
                return None
            _, t_pairs, y, _ = arranged["train"]
            mean_cells = y.double().mean(dim=0)  # (n_cells, C)
            t_start_values = np.sort(np.unique(t_pairs[:, 0].numpy()))
            t_end_values = np.sort(np.unique(t_pairs[:, 1].numpy()))
            i = np.searchsorted(t_start_values, t_pairs[:, 0].numpy())
            j = np.searchsorted(t_end_values, t_pairs[:, 1].numpy())
            grid = torch.full((len(t_start_values), len(t_end_values), mean_cells.shape[1]), float("nan"), dtype=torch.float64)
            grid[i, j] = mean_cells
            surface = {
                "t_start_values": torch.as_tensor(t_start_values, dtype=torch.float64),
                "t_end_values": torch.as_tensor(t_end_values, dtype=torch.float64),
                "mean_true_delta": grid,
                "prediction_space": str(PREDICTION_SPACE),
                "split": "train",
                "n_samples": int(y.shape[0]),
                "target_cols": list(TARGET_COLS),
            }
            torch.save(surface, run_dir / MEAN_SURFACE_NAME)
            print(f"Saved {MEAN_SURFACE_NAME}")
            return surface

        nonlocal mean_surface
        mean_surface = _calc_mean_surface()
        _, t_pairs, _, _ = arranged["train"]
        return SampleMeta(
            img_shape=table.image_shape,
            src_shape=table.source_shape,
            tgt_shape=table.target_shape,
            n_cells=int(t_pairs.shape[0]),
            t=t_pairs,
            default_cell=_default_cell(t_pairs),
        )

    def _get_splits() -> dict[str, SplitDataset]:
        """Get the splits from the splits_df."""
        out: dict[str, SplitDataset] = {}
        for name, (sample_ids, t_pairs, y, y_raw) in arranged.items():
            anchor = None
            if PREDICTION_SPACE == "residuals":
                # Anchor each cell on the train split's mean, so the heads only
                # have to predict how a sample deviates from the population
                # surface. Each split anchors at its own labeled pairs.
                anchor = gather_at_pairs(mean_surface, t_pairs)  # (n_cells, C)
                y = y - anchor.to(y)
                anchor = anchor.to(device=device, dtype=torch.float)
            out[name] = SplitDataset(
                split_name=name,
                sample_ids=tuple(sample_ids),
                embs=table,
                y=y.to(device),
                y_raw=y_raw.to(device),
                default_cell=_default_cell(t_pairs),
                mean_surface=anchor,
            )
        return out

    def _get_bundle() -> SplitDatasetBundle:
        """Get the bundle from the splits, metadata, and mean surface."""
        return SplitDatasetBundle(splits=split_datasets, splits_df=splits, meta=meta)

    splits = _get_splits_df()
    table = _get_embeddings()
    arranged = {name: _grid_arrange(X, y) for name, (X, y) in splits.items()}
    meta = _get_metadata()
    split_datasets = _get_splits()
    return _get_bundle()
