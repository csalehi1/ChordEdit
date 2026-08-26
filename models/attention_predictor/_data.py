# _data.py

"""
Data loading, splits, and grid batching for the predictor and selector.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from embeddings import get_embeddings_mixed
from settings import *

ID_TO_SPLIT_NAME = "id_to_split.csv"


@dataclass(frozen=True)
class EmbeddingTable:
    """One embedding-table row per unique sample_id."""

    img: torch.Tensor   # (n_samples, C, S, S)
    src: torch.Tensor   # (n_samples, D_txt)
    tar: torch.Tensor   # (n_samples, D_txt)


@dataclass(frozen=True)
class CellTensors:
    """One split's cells plus the shared embedding table, resident on one device."""

    sample_idx: torch.Tensor     # (N,)
    t: torch.Tensor              # (N, 2)
    y: torch.Tensor              # (N, C)
    emb_table: EmbeddingTable
    grid_rows: torch.Tensor      # (S, n_cells)
    grid_baseline: torch.Tensor  # (S,)

    def __len__(self) -> int:
        return int(self.sample_idx.shape[0])

    @property
    def n_grids(self) -> int:
        return int(self.grid_rows.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.grid_rows.shape[1])

    def gather_grids(self, sel: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """(img, src, tar, y) for whole grids: token tables (G, ...), y (G, n_cells, .)."""
        rows = self.grid_rows[sel]
        sid = self.sample_idx[rows[:, 0]]
        return (
            self.emb_table.img[sid],
            self.emb_table.src[sid],
            self.emb_table.tar[sid],
            self.y[rows],
        )

    def iter_grids(self, grids_per_batch: int = 1, shuffle: bool = True):
        """Yield (grid inputs, baseline_idx) with whole grids per batch."""
        s = self.n_grids
        order = torch.randperm(s, device=self.grid_rows.device) if shuffle else torch.arange(s, device=self.grid_rows.device)
        step = max(1, int(grids_per_batch))
        for k in range(0, s, step):
            sel = order[k : k + step]
            yield self.gather_grids(sel), self.grid_baseline[sel]


def _build_grid_index(
    sample_idx: torch.Tensor,
    t: torch.Tensor,
    default_t: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group rows into complete per-sample grids that contain the default cell."""
    groups: dict[int, list[int]] = {}
    for i, sid in enumerate(sample_idx.tolist()):
        groups.setdefault(sid, []).append(i)
    if not groups:
        raise ValueError("Expected samples in the split")

    sizes = [len(v) for v in groups.values()]
    n_cells = max(set(sizes), key=sizes.count)
    is_default = (
        torch.isclose(t[:, 0], torch.as_tensor(default_t[0], dtype=t.dtype, device=t.device))
        & torch.isclose(t[:, 1], torch.as_tensor(default_t[1], dtype=t.dtype, device=t.device))
    )

    # Sort each grid's rows by (t_start, t_end) so cell k means the same
    # timestep pair in every sample; the per-cell anchor relies on that.
    t_cpu = t.detach().cpu()
    rows, baselines, dropped = [], [], 0
    for sid, idx in groups.items():
        if len(idx) != n_cells:
            dropped += 1
            continue
        idx = sorted(idx, key=lambda i: (float(t_cpu[i, 0]), float(t_cpu[i, 1])))
        flags = is_default[torch.as_tensor(idx, device=t.device)]
        hits = flags.nonzero(as_tuple=False).flatten()
        if hits.numel() != 1:
            dropped += 1
            continue
        rows.append(idx)
        baselines.append(int(hits[0]))
    if not rows:
        raise ValueError(f"Expected a complete {n_cells}-cell grid")
    if dropped:
        print(f"Dropped {dropped} sample(s) without a complete {n_cells}-cell grid and default cell.")
    return (
        torch.as_tensor(rows, dtype=torch.long, device=t.device),
        torch.as_tensor(baselines, dtype=torch.long, device=t.device),
    )


def create_cell_tensors(
    splits_df: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    device: torch.device,
) -> dict[str, CellTensors]:
    """Build device-resident CellTensors for each split, sharing one embedding table."""
    frames = [X for X, _ in splits_df.values()]
    samples = pd.concat(frames, ignore_index=True).drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, src_emb, tar_emb = get_embeddings_mixed(samples)

    # Build the shared embedding table.
    emb_table = EmbeddingTable(
        img=img_emb.to(device),
        src=src_emb.to(device),
        tar=tar_emb.to(device),
    )
    sid_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    # Build the CellTensors for each split.
    out: dict[str, CellTensors] = {}
    for name, (X_df, y_df) in splits_df.items():
        sample_idx = torch.tensor([sid_to_idx[sid] for sid in X_df[SAMPLE_ID_COL].tolist()], dtype=torch.long, device=device)
        t = torch.tensor(X_df[[T_START_COL, T_END_COL]].values, dtype=torch.float, device=device)
        y = torch.tensor(y_df[list(TARGET_COLS)].values, dtype=torch.float, device=device)
        grid_rows, grid_baseline = _build_grid_index(sample_idx, t, (DEFAULT_T_START, DEFAULT_T_END))
        out[name] = CellTensors(
            sample_idx=sample_idx, t=t, y=y, emb_table=emb_table,
            grid_rows=grid_rows, grid_baseline=grid_baseline,
        )
    return out


"""
Dataframes.
"""

def _prep_sample_id(value) -> str:
    return f"{int(value):08d}"

def load_df(
    metrics_csv: Path | None = None,
    inputs_csv: Path | None = None,
    *,
    apply_max_samples: bool = True,
) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    metrics_cols = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *TARGET_COLS]
    inputs_cols = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]

    # Clean metrics CSV: drop rows that do not have target component metrics or t_delta.
    metrics_csv = metrics_csv or METRICS_CSV
    metrics_df = pd.read_csv(metrics_csv)
    n_before = len(metrics_df)
    metrics_df = metrics_df.dropna(subset=list(TARGET_COLS)).reset_index(drop=True)
    if len(metrics_df) < n_before:
        print(f"Dropped {n_before - len(metrics_df)} metric rows missing {list(TARGET_COLS)}.")
    metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(_prep_sample_id)
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
            raise ValueError(f"Expected {TARGET_T_DELTA=} in {T_DELTA_COL}")
        metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    # Keep only samples with a complete grid. A sample missing cells cannot be
    # scored or selected over: its per-sample deltas are undefined without every
    # candidate, and selection requires one shared set of labeled
    # (t_start, t_end) pairs across the split.
    cells_per_sample = metrics_df.groupby(SAMPLE_ID_COL)[SAMPLE_ID_COL].transform("size")
    n_cells = int(cells_per_sample.mode().iat[0])
    incomplete = cells_per_sample != n_cells
    if incomplete.any():
        n_dropped = metrics_df.loc[incomplete, SAMPLE_ID_COL].nunique()
        print(f"Dropped {n_dropped} sample(s) with fewer than {n_cells} labeled cells.")
        metrics_df = metrics_df.loc[~incomplete].copy()

    # Optional slice of a large dataset, taken after the completeness filter so
    # the count is samples that will actually train. Sorted, so the slice is the
    # same set on every run and across configurations. Skipped for PIE-Bench
    # (MAX_SAMPLES only applies to the UltraEdit pool).
    if apply_max_samples and MAX_SAMPLES is not None:
        keep = np.sort(metrics_df[SAMPLE_ID_COL].unique())[:MAX_SAMPLES]
        if len(keep) < MAX_SAMPLES:
            print(f"{MAX_SAMPLES=} exceeds the {len(keep)} complete samples available; using all of them.")
        metrics_df = metrics_df.loc[metrics_df[SAMPLE_ID_COL].isin(keep)].copy()
        print(f"Sliced to {len(keep)} sample(s) ({MAX_SAMPLES=}).")

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
    if inputs_df[inputs_cols].isna().to_numpy().any():
        raise ValueError(f"Expected no missing values in {inputs_csv}")
    inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(_prep_sample_id)
    for col in (IMAGE_PATH_COL, MASK_PATH_COL):
        inputs_df[col] = [
            str(p) if (p := Path(path)).is_absolute() else str(DATASET_DIR / path)
            for path in inputs_df[col]
        ]

    return pd.merge(
        metrics_df.loc[:, metrics_cols],
        inputs_df.loc[:, inputs_cols],
        on=SAMPLE_ID_COL,
        how="left",
    ).reset_index(drop=True)


def load_pie_bench_xy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """All labeled PIE-Bench samples as (X, y), with pie_-prefixed sample ids."""
    data_df = load_df(PIE_METRICS_CSV, PIE_INPUTS_CSV, apply_max_samples=False)
    X_df, y_df = prepare_df(data_df)
    X_df = X_df.copy()
    X_df[SAMPLE_ID_COL] = PIE_SAMPLE_ID_PREFIX + X_df[SAMPLE_ID_COL].astype(str)
    return X_df, y_df


def build_splits_df() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Train/val/test frames; with PIE_BENCH, UltraEdit test is replaced by PIE-Bench."""
    data_df = load_df()
    X_df, y_df = prepare_df(data_df)
    train_X, val_X, test_X, train_y, val_y, test_y = split_df(X_df, y_df)
    if PIE_BENCH:
        test_X, test_y = load_pie_bench_xy()
    return {"train": (train_X, train_y), "val": (val_X, val_y), "test": (test_X, test_y)}



def prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a loaded table into features X and targets y in PREDICTION_SPACE.

    "raws" keeps the metric values as measured. "deltas" and "residuals" both
    start from the per-sample normalized improvements; the mean surface that
    turns deltas into residuals is only known once the train split exists, so
    train.py subtracts it there.
    """
    from scores import compute_delta_df

    X_df = data_df.drop(columns=list(TARGET_COLS)).copy()
    if PREDICTION_SPACE == "raws":
        y_df = data_df.loc[:, list(TARGET_COLS)].copy()
    else:
        y_df = compute_delta_df(data_df, *TARGET_COLS)
    return X_df, y_df


def split_df(
    X_df: pd.DataFrame,
    y_df: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
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
    target_cols = list(TARGET_COLS)
    return (
        train.drop(columns=target_cols),
        val.drop(columns=target_cols),
        test.drop(columns=target_cols),
        train.loc[:, target_cols],
        val.loc[:, target_cols],
        test.loc[:, target_cols],
    )


def save_splits_df(
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
        raise FileNotFoundError(f"Missing splits at {splits_path}. Run train.py first.")
    splits_df = pd.read_csv(splits_path, dtype={SAMPLE_ID_COL: str, "split": str})
    ue_df = load_df()
    pie_df = None
    if PIE_BENCH:
        pie_df = load_df(PIE_METRICS_CSV, PIE_INPUTS_CSV, apply_max_samples=False)
        pie_df = pie_df.copy()
        pie_df[SAMPLE_ID_COL] = PIE_SAMPLE_ID_PREFIX + pie_df[SAMPLE_ID_COL].astype(str)
    out: dict[str, pd.DataFrame] = {}
    for name in ("train", "val", "test"):
        split_ids = splits_df.loc[splits_df["split"] == name, SAMPLE_ID_COL]
        source = pie_df if (PIE_BENCH and name == "test") else ue_df
        out[name] = source.loc[source[SAMPLE_ID_COL].isin(split_ids)].reset_index(drop=True)
    return out


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
