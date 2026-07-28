"""Data loading, splits, and embedding tables for M and T training."""

from __future__ import annotations

import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from model_m import MetricPredictor
from settings import *
from _helpers import mean_pool, normalize_target_columns


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


# --- Path / id helpers ------------------------------------------------------------

def _prep_sample_id(value) -> str:
    return f"{int(value):08d}"


def _resolve_under(root: Path, path: str) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str(root / path)


def _resolve_embedding_path(embedding_path: str) -> str:
    """Absolute CSV paths as-is; relative under EMBEDDINGS_DIR/annotation_embeddings/."""
    path = Path(embedding_path)
    if path.is_absolute():
        return str(path)
    path = Path(str(embedding_path).lstrip("/"))
    if path.parts and path.parts[0] == EMBEDDINGS_SAMPLES_DIRNAME:
        return str(EMBEDDINGS_DIR / path)
    return str(EMBEDDINGS_DIR / EMBEDDINGS_SAMPLES_DIRNAME / path)


# --- Dataset / loaders ------------------------------------------------------------

def model_inputs(batch: CellItem, device: torch.device) -> tuple[torch.Tensor, ...]:
    """Return (img, mask, src, tar, t, y) on device."""
    return tuple(x.to(device) for x in batch[MODEL_BATCH_SLICE])


class SampleGridBatchSampler(Sampler[list[int]]):
    """One sample's full timestep grid per batch (within-image ranking loss)."""

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
    """One embedding row per unique sample_id; shared across train/val/test."""

    img: torch.Tensor   # (n_samples, img_dim)
    mask: torch.Tensor  # (n_samples, img_dim)
    src: torch.Tensor   # (n_samples, text_dim)
    tar: torch.Tensor   # (n_samples, text_dim)


class CellEmbeddingDataset(Dataset[CellItem]):
    """One row per (sample_id, t_start, t_end); embeddings looked up by sample_idx."""

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


# --- CSV load / prepare / split ---------------------------------------------------

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
    """Per-target (min, max) from dataset metrics."""
    if df is None:
        df = load_df()
    bounds: dict[str, tuple[float, float]] = {}
    for col in M_TARGET_COLS:
        values = np.asarray(df[col], dtype=float)
        bounds[col] = (float(values.min()), float(values.max()))
    return bounds


def load_df(metrics_csv: Path | None = None, inputs_csv: Path | None = None) -> pd.DataFrame:
    """Load metrics, attach source-image paths and prompts, one row per cell."""
    metrics_csv = metrics_csv or METRICS_CSV
    metrics_df = pd.read_csv(metrics_csv)
    n_drop = int(metrics_df.isna().any(axis=1).sum())
    if n_drop:
        print(f"Dropping {n_drop} rows with missing values from {metrics_csv}")
        metrics_df = metrics_df.dropna().reset_index(drop=True)
    metrics_df[SAMPLE_ID_COL] = metrics_df[SAMPLE_ID_COL].map(_prep_sample_id)
    if TARGET_T_DELTA is not None:
        if TARGET_T_DELTA not in metrics_df[T_DELTA_COL].values:
            raise ValueError(f"{TARGET_T_DELTA=} not found in {T_DELTA_COL}")
        metrics_df = metrics_df.loc[metrics_df[T_DELTA_COL] == TARGET_T_DELTA].copy()

    inputs_csv = inputs_csv or INPUTS_CSV
    inputs_df = pd.read_csv(inputs_csv)
    if inputs_df.isna().any().any():
        raise ValueError(f"Missing values found in {inputs_csv}")
    inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].map(_prep_sample_id)
    inputs_df[IMAGE_PATH_COL] = inputs_df[IMAGE_PATH_COL].map(lambda p: _resolve_under(DATASET_DIR, p))
    inputs_df[MASK_PATH_COL] = inputs_df[MASK_PATH_COL].map(lambda p: _resolve_under(DATASET_DIR, p))

    metrics_part: pd.DataFrame = metrics_df.loc[:, METRICS_COLS]
    inputs_part: pd.DataFrame = inputs_df.loc[:, INPUTS_COLS]
    return pd.merge(metrics_part, inputs_part, on=SAMPLE_ID_COL, how="left").reset_index(drop=True)


def prepare_df(data_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a loaded table into X and min-max normalized y."""
    bounds = target_bounds(data_df)
    X_df = data_df.drop(columns=list(M_TARGET_COLS)).copy()
    y_raw: pd.DataFrame = data_df.loc[:, list(M_TARGET_COLS)]
    # Min-max so PSNR/CLIP share [0, 1] before ranking; MSE still z-scores later.
    y = normalize_target_columns(y_raw, bounds=bounds)
    return X_df, y


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


def grid_axes_from_df(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique (t_start, t_end) grid axes."""
    return (
        np.sort(np.asarray(df[T_START_COL].unique())),
        np.sort(np.asarray(df[T_END_COL].unique())),
    )


def timestep_pairs_from_df(df: pd.DataFrame) -> np.ndarray:
    """Unique (t_start, t_end) pairs in df, shape (N, 2), sorted."""
    return (
        df.loc[:, [T_START_COL, T_END_COL]]
        .drop_duplicates()
        .sort_values([T_START_COL, T_END_COL])
        .to_numpy(dtype=np.float64)
    )


# --- Embeddings -------------------------------------------------------------------

def _latent_to_vector(t: torch.Tensor) -> torch.Tensor:
    return t.detach().float().reshape(-1).contiguous()


def _text_seq_to_vector(
    t: torch.Tensor,
    prompt: str,
    tokenizer,
    attn_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean-pool saved CLIP hidden states to match TextEncoder output dims."""
    t = t.detach().float()
    if t.ndim == 1:
        return t.contiguous()
    if t.ndim == 2 and t.shape[0] == 1:
        return t.reshape(-1).contiguous()
    if t.ndim == 2:
        hidden = t.unsqueeze(0)
    elif t.ndim == 3:
        hidden = t
    else:
        raise ValueError(f"Unexpected text embedding shape: {tuple(t.shape)}")

    if attn_mask is None:
        inputs = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        )
        attn_mask = inputs.attention_mask
    if attn_mask.shape[-1] != hidden.shape[1]:
        pooled = hidden.mean(dim=1)
    else:
        pooled = mean_pool(hidden, attn_mask)
    return pooled.reshape(-1).contiguous()


def _slice_embedding_cache(
    cache_path: Path,
    sample_ids: list[str],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Load embedding tables from cache if it covers all sample_ids; else None."""
    if not cache_path.exists():
        return None
    data = torch.load(cache_path, map_location="cpu", weights_only=False)
    id_to_i = {sid: i for i, sid in enumerate(data["sample_ids"])}
    if not all(sid in id_to_i for sid in sample_ids):
        return None
    idxs = [id_to_i[sid] for sid in sample_ids]
    print(f"Loaded embeddings from {cache_path} ({len(sample_ids)} samples)")
    return (
        sample_ids,
        data["img"][idxs].contiguous(),
        data["mask"][idxs].contiguous(),
        data["src"][idxs].contiguous(),
        data["tar"][idxs].contiguous(),
    )


def _load_embeddings_from_csv(
    samples: pd.DataFrame,
    predictor: MetricPredictor,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load from M_EMBEDDINGS_PATH, or pack per-sample .pt files into that cache."""
    if EMBEDDINGS_CSV is None:
        raise ValueError("EMBEDDINGS_CSV is None; cannot load precomputed embeddings")
    if not Path(EMBEDDINGS_CSV).exists():
        raise FileNotFoundError(f"Embeddings CSV not found: {EMBEDDINGS_CSV}")
    if predictor.pipeline is None:
        raise RuntimeError("MetricPredictor.pipeline is required to tokenize prompts when loading embeddings")
    tokenizer = predictor.pipeline.tokenizer

    sample_ids = samples[SAMPLE_ID_COL].tolist()
    cached = _slice_embedding_cache(Path(M_EMBEDDINGS_PATH), sample_ids)
    if cached is not None:
        return cached
    if Path(M_EMBEDDINGS_PATH).exists():
        print(f"{M_EMBEDDINGS_PATH} incomplete for requested samples; rebuilding from per-sample files")

    emb_df = pd.read_csv(EMBEDDINGS_CSV)
    if emb_df.isna().any().any():
        raise ValueError(f"Missing values found in {EMBEDDINGS_CSV}")
    emb_df[SAMPLE_ID_COL] = emb_df[SAMPLE_ID_COL].map(_prep_sample_id)
    for col in (SOURCE_EMB_COL, TARGET_EMB_COL, IMAGE_EMB_COL, MASK_EMB_COL):
        if col not in emb_df.columns:
            raise ValueError(f"Missing column {col!r} in {EMBEDDINGS_CSV}")
        emb_df[col] = emb_df[col].map(_resolve_embedding_path)

    id_to_row = emb_df.set_index(SAMPLE_ID_COL)
    if id_to_row.index.has_duplicates:
        dupes = id_to_row.index[id_to_row.index.duplicated()].unique().tolist()
        preview = ", ".join(str(s) for s in dupes[:5])
        raise ValueError(f"Duplicate sample_id(s) in {EMBEDDINGS_CSV}: {preview}")

    missing = [sid for sid in sample_ids if sid not in id_to_row.index]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        raise KeyError(f"Missing embeddings for {len(missing)} sample_id(s): {preview}{more}")

    id_to_prompts = samples.set_index(SAMPLE_ID_COL).loc[:, [SOURCE_PROMPT_COL, TARGET_PROMPT_COL]]
    src_prompts = [str(id_to_prompts.loc[sid, SOURCE_PROMPT_COL]) for sid in sample_ids]
    tar_prompts = [str(id_to_prompts.loc[sid, TARGET_PROMPT_COL]) for sid in sample_ids]
    src_masks = tokenizer(
        src_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).attention_mask
    tar_masks = tokenizer(
        tar_prompts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).attention_mask

    path_rows = [
        (
            sid,
            str(id_to_row.loc[sid, IMAGE_EMB_COL]),
            str(id_to_row.loc[sid, MASK_EMB_COL]),
            str(id_to_row.loc[sid, SOURCE_EMB_COL]),
            str(id_to_row.loc[sid, TARGET_EMB_COL]),
        )
        for sid in sample_ids
    ]

    def _load_pt(path: str) -> torch.Tensor:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing embedding file: {path}")
        t = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
        return t

    def _load_sample(row: tuple[str, str, str, str, str]):
        sid, img_p, mask_p, src_p, tar_p = row
        return sid, _load_pt(img_p), _load_pt(mask_p), _load_pt(src_p), _load_pt(tar_p)

    print(f"Packing embeddings from {EMBEDDINGS_CSV} ({len(sample_ids)} samples) -> {M_EMBEDDINGS_PATH}")
    img_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    src_rows: list[torch.Tensor] = []
    tar_rows: list[torch.Tensor] = []
    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, (sid, img_t, mask_t, src_t, tar_t) in enumerate(
            tqdm(pool.map(_load_sample, path_rows), total=len(path_rows), desc="Loading embeddings", unit="sample")
        ):
            img_rows.append(_latent_to_vector(img_t))
            mask_rows.append(_latent_to_vector(mask_t))
            src_rows.append(_text_seq_to_vector(src_t, src_prompts[i], tokenizer, attn_mask=src_masks[i : i + 1]))
            tar_rows.append(_text_seq_to_vector(tar_t, tar_prompts[i], tokenizer, attn_mask=tar_masks[i : i + 1]))

    img_emb = torch.stack(img_rows, dim=0)
    mask_emb = torch.stack(mask_rows, dim=0)
    src_emb = torch.stack(src_rows, dim=0)
    tar_emb = torch.stack(tar_rows, dim=0)

    Path(M_EMBEDDINGS_PATH).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(M_EMBEDDINGS_PATH) + ".tmp")
    torch.save(
        {"sample_ids": list(sample_ids), "img": img_emb, "mask": mask_emb, "src": src_emb, "tar": tar_emb},
        tmp_path,
    )
    tmp_path.replace(M_EMBEDDINGS_PATH)
    print(f"Saved {M_EMBEDDINGS_PATH}")
    return sample_ids, img_emb, mask_emb, src_emb, tar_emb


def _encode_embeddings(
    samples: pd.DataFrame,
    predictor: MetricPredictor,
    *,
    use_cache: bool = True,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode with ChordEdit encoders; optional disk cache keyed by inputs + SD root."""
    cache_key = hashlib.sha256(f"{INPUTS_CSV}|{SD_TURBO_ROOT}".encode()).hexdigest()[:12]
    cache_path = Path(OUTPUTS_DIR.parent / ".cache" / "embeddings" / f"{cache_key}.pt").resolve()
    sample_ids = samples[SAMPLE_ID_COL].tolist()

    if use_cache:
        cached = _slice_embedding_cache(cache_path, sample_ids)
        if cached is not None:
            return cached
        if cache_path.exists():
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

    img_emb = torch.cat(img_chunks, dim=0)
    mask_emb = torch.cat(mask_chunks, dim=0)
    src_emb = torch.cat(src_chunks, dim=0)
    tar_emb = torch.cat(tar_chunks, dim=0)

    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"sample_ids": list(sample_ids), "img": img_emb, "mask": mask_emb, "src": src_emb, "tar": tar_emb},
            cache_path,
        )
        print(f"Saved embedding cache: {cache_path}")

    return sample_ids, img_emb, mask_emb, src_emb, tar_emb


def _get_embeddings(
    samples: pd.DataFrame,
    predictor: MetricPredictor,
    *,
    use_cache: bool = True,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU embedding tables: disk when FREEZE_ENCODERS, else encode."""
    if FREEZE_ENCODERS and EMBEDDINGS_CSV is not None:
        return _load_embeddings_from_csv(samples, predictor)
    return _encode_embeddings(samples, predictor, use_cache=use_cache, batch_size=batch_size)


def get_embeddings_by_sample(
    df: pd.DataFrame,
    predictor: MetricPredictor,
    device: torch.device,
    *,
    use_cache: bool = True,
) -> dict[str, dict[str, torch.Tensor]]:
    """Embeddings keyed by sample_id."""
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
    samples = pd.concat([train_X, val_X, test_X], ignore_index=True).drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = _get_embeddings(samples, predictor, use_cache=True)
    emb_tables = EmbeddingTables(img=img_emb, mask=mask_emb, src=src_emb, tar=tar_emb)
    sample_id_to_idx = {sid: i for i, sid in enumerate(sample_ids)}

    def _dataloader(X: pd.DataFrame, y: pd.DataFrame, shuffle: bool, by_sample: bool = False) -> DataLoader[CellItem]:
        dataset = CellEmbeddingDataset(
            torch.tensor([sample_id_to_idx[sid] for sid in X[SAMPLE_ID_COL].tolist()], dtype=torch.long),
            emb_tables,
            torch.tensor(X[[T_START_COL, T_END_COL]].values, dtype=torch.float),
            torch.tensor(y[list(M_TARGET_COLS)].values, dtype=torch.float),
        )
        if by_sample:
            return DataLoader(dataset, batch_sampler=SampleGridBatchSampler(dataset.sample_idx, shuffle=shuffle))
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
