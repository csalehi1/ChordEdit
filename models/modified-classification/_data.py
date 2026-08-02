"""Data loading, splits, and embedding tables for M and T training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from model_m import SurrogateModel, _get_text_pooling, make_text_extractor
from settings import *

METRICS_COLS = [SAMPLE_ID_COL, T_START_COL, T_END_COL, T_DELTA_COL, *M_TARGET_COLS]
INPUTS_COLS = [SAMPLE_ID_COL, SOURCE_PROMPT_COL, TARGET_PROMPT_COL, IMAGE_PATH_COL, MASK_PATH_COL]
DATA_COLS = list(dict.fromkeys(METRICS_COLS + INPUTS_COLS))
ID_TO_SPLIT_NAME = "id_to_split.csv"
_PACKED_EMBEDDINGS_DIR = Path(__file__).resolve().parent / ".cache" / "packed_embeddings"

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


def _prep_sample_id(value) -> str:
    return f"{int(value):08d}"


def _prep_embedding_path(embedding_path: str) -> str:
    path = Path(embedding_path)
    if path.is_absolute():
        return str(path)
    path = Path(str(embedding_path).lstrip("/"))
    if path.parts and path.parts[0] == EMBEDDINGS_SAMPLES_DIRNAME:
        return str(EMBEDDINGS_DIR / path)
    return str(EMBEDDINGS_DIR / EMBEDDINGS_SAMPLES_DIRNAME / path)


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
Embeddings.

Scattered: many per-sample .pt files indexed by EMBEDDINGS_CSV (slow to load).
Packed: one stacked table at .cache/packed_embeddings/<CHORD_EDIT_MODEL>-<t_delta>-<dir_slug>.pt
(fast to load), tagged with a "meta" dict recording the model type, text pooling,
and pack provenance. A cache whose meta is missing or does not match the current
settings is treated as a miss and repacked. Text extraction from scattered files
is dispatched on CHORD_EDIT_PIPELINE_TYPE (see model_m.make_text_extractor) so
packed caches always correspond to the CHORD_EDIT_MODEL.
"""

def _get_packed_path() -> Path:
    """Packed cache path for the current settings."""
    t_delta = f"{TARGET_T_DELTA}".replace(".", "p")
    slug = DIR_NAME.replace("_", "").lower()
    return _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-{t_delta}-{slug}.pt"


def _expected_packed_meta() -> dict:
    """Meta that a valid packed cache must carry (primitives only: survives weights_only)."""
    return {
        "model": CHORD_EDIT_MODEL,
        "pipeline_type": CHORD_EDIT_PIPELINE_TYPE,
        "text_pooling": _get_text_pooling(),
        "image_size": int(CHORD_EDIT_IMAGE_SIZE),
        "use_center_crop": bool(USE_CENTER_CROP),
        "dir_name": DIR_NAME,
        "target_t_delta": TARGET_T_DELTA,
    }


def _load_pt(path: str) -> torch.Tensor:
    """Load one scattered embedding .pt file to a float32 CPU tensor."""
    t = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
    return t.detach().float()


def _load_packed_cache(
    packed_path: Path,
    sample_ids: list[str],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Packed tables sliced to sample_ids, or None (missing / bad meta / coverage)."""
    if not packed_path.exists():
        return None
    try:
        data = torch.load(packed_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:
        print(f"Failed to load {packed_path} ({e}). Repacking...")
        return None
    meta = data.get("meta")
    if not isinstance(meta, dict):
        print(f"{packed_path} has no meta. Repacking...")
        return None
    for key, expected in _expected_packed_meta().items():
        if meta.get(key) != expected:
            print(f"{packed_path} meta mismatch: {key}={meta.get(key)!r} expected {expected!r}; repacking")
            return None
    if data["img"].shape[1] != meta.get("img_dim") or data["src"].shape[1] != meta.get("text_dim"):
        print(f"{packed_path} table dims do not match meta; repacking")
        return None
    id_to_i = {sid: i for i, sid in enumerate(data["sids"])}
    n_missing = sum(sid not in id_to_i for sid in sample_ids)
    if n_missing:
        print(f"{packed_path} missing {n_missing} of {len(sample_ids)} requested samples; repacking")
        return None
    idxs = [id_to_i[sid] for sid in sample_ids]
    print(f"Loaded embeddings from {packed_path} ({len(sample_ids)} samples)")
    return (
        sample_ids,
        data["img"][idxs].contiguous(),
        data["mask"][idxs].contiguous(),
        data["src"][idxs].contiguous(),
        data["tar"][idxs].contiguous(),
    )


def _save_packed_cache(
    packed_path: Path,
    sample_ids: list[str],
    tables: dict[str, torch.Tensor],
    *,
    source: str,
) -> None:
    """Atomically write the packed table dict with meta; prints diff vs any old file.

    image_size/use_center_crop in meta describe the settings the cache was packed
    under; for source='scattered' the tensor content comes from the annotation
    pipeline, for source='encoder' from the frozen ChordEdit encoders.
    """
    meta = _expected_packed_meta() | {
        "source": source,
        "img_dim": int(tables["img"].shape[1]),
        "text_dim": int(tables["src"].shape[1]),
    }
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(packed_path) + ".tmp")
    torch.save({"sids": list(sample_ids), **tables, "meta": meta}, tmp_path)
    tmp_path.replace(packed_path)
    print(f"Saved packed embeddings: {packed_path}")


def _pack_scattered_cache(
    samples: pd.DataFrame,
    packed_path: Path,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pack scattered per-sample .pt files into packed tables; None if CSV cannot cover.

    Scattered embeddings stay per-sample on disk to best support symlinks between
    dataset sizes, but loading them is slow, so the packed table is cached.
    """
    sample_ids = samples[SAMPLE_ID_COL].tolist()
    if not sample_ids:
        raise ValueError("No samples to pack")
    if EMBEDDINGS_CSV is None or not Path(EMBEDDINGS_CSV).exists():
        return None

    # Verify the CSV file is valid and matches the expected columns.
    emb_df = pd.read_csv(EMBEDDINGS_CSV)
    if emb_df.isna().any().any():
        raise ValueError(f"Missing values found in {EMBEDDINGS_CSV}")
    emb_df[SAMPLE_ID_COL] = emb_df[SAMPLE_ID_COL].map(_prep_sample_id)
    emb_cols = [IMAGE_EMB_COL, MASK_EMB_COL, SOURCE_EMB_COL, TARGET_EMB_COL]
    for col in emb_cols:
        if col not in emb_df.columns:
            raise ValueError(f"Missing column {col!r} in {EMBEDDINGS_CSV}")
        emb_df[col] = emb_df[col].map(_prep_embedding_path)

    # Vectorized per-sample path lookup; NaN rows mark ids the CSV does not cover.
    id_to_row = emb_df.drop_duplicates(SAMPLE_ID_COL).set_index(SAMPLE_ID_COL).reindex(sample_ids)
    missing = id_to_row.index[id_to_row[emb_cols].isna().any(axis=1)].tolist()
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"Scattered embeddings missing {len(missing)} sample_id(s): {preview}{more}")
        return None
    path_rows = list(id_to_row[emb_cols].itertuples(index=False, name=None))

    # Prompts feed the tokenizer for attention masks; the left merge in load_df
    # can leave NaN prompts for ids absent from the inputs CSV.
    prompt_na = samples[[SOURCE_PROMPT_COL, TARGET_PROMPT_COL]].isna().any(axis=1)
    if prompt_na.any():
        bad = samples.loc[prompt_na, SAMPLE_ID_COL].tolist()
        raise ValueError(
            f"NaN prompts for {len(bad)} sample_id(s) (missing from {INPUTS_CSV}?): {bad[:5]}"
        )
    extract_text = make_text_extractor(
        samples[SOURCE_PROMPT_COL].tolist(),
        samples[TARGET_PROMPT_COL].tolist(),
    )

    # Probe dims from the first sample, then preallocate the packed tables so
    # peak memory stays ~1x table size (no row lists + torch.stack copy).
    n = len(sample_ids)
    img_dim = _load_pt(path_rows[0][0]).numel()
    text_dim = extract_text.text_dim(_load_pt(path_rows[0][2]), path_rows[0][2])
    img_emb = torch.empty((n, img_dim), dtype=torch.float32)
    mask_emb = torch.empty((n, img_dim), dtype=torch.float32)
    src_emb = torch.empty((n, text_dim), dtype=torch.float32)
    tar_emb = torch.empty((n, text_dim), dtype=torch.float32)

    def _load_row(i: int):
        img_p, mask_p, src_p, tar_p = path_rows[i]
        return i, _load_pt(img_p), _load_pt(mask_p), _load_pt(src_p), _load_pt(tar_p)

    def _check_image(t: torch.Tensor, i: int, kind: str) -> torch.Tensor:
        if t.numel() != img_dim:
            raise ValueError(f"{kind} numel {t.numel()} != {img_dim} for sample {sample_ids[i]}")
        return t

    with ThreadPoolExecutor(max_workers=32) as pool:
        # Load the embeddings in parallel using a thread pool; rows fill in-place.
        for i, img_t, mask_t, src_t, tar_t in tqdm(pool.map(_load_row, range(n)), total=n, desc="Packing embeddings", unit="sample"):
            img_emb[i] = _check_image(img_t, i, "image").reshape(-1)
            mask_emb[i] = _check_image(mask_t, i, "mask").reshape(-1)
            src_emb[i] = extract_text(src_t, i, "src", text_dim)
            tar_emb[i] = extract_text(tar_t, i, "tar", text_dim)

    tables = {"img": img_emb, "mask": mask_emb, "src": src_emb, "tar": tar_emb}
    _save_packed_cache(packed_path, sample_ids, tables, source="scattered")
    return sample_ids, img_emb, mask_emb, src_emb, tar_emb


def _encode_embeddings(
    samples: pd.DataFrame,
    predictor: SurrogateModel,
    packed_path: Path,
    *,
    batch_size: int = EMBED_BATCH_SIZE,
    cache_scattered: bool = True,
    cache_packed: bool = True,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode with the frozen ChordEdit encoders; optionally write both caches."""
    sample_ids = samples[SAMPLE_ID_COL].tolist()
    n_samples = len(sample_ids)
    image_paths = samples[IMAGE_PATH_COL].tolist()
    mask_paths = samples[MASK_PATH_COL].tolist()
    src_prompts = samples[SOURCE_PROMPT_COL].tolist()
    tar_prompts = samples[TARGET_PROMPT_COL].tolist()
    img_batches: list[torch.Tensor] = []
    mask_batches: list[torch.Tensor] = []
    src_batches: list[torch.Tensor] = []
    tar_batches: list[torch.Tensor] = []

    print(f"Encoding embeddings for {n_samples} samples (batch_size={batch_size})...")
    # Encode the embeddings in batches.
    for start in tqdm(range(0, n_samples, batch_size), desc="Encoding embeddings", unit="batch"):
        end = min(start + batch_size, n_samples)
        images = [Image.open(p).convert("RGB") for p in image_paths[start:end]]
        masks = [Image.open(p).convert("RGB") for p in mask_paths[start:end]]
        with torch.no_grad():
            img_batches.append(predictor.image_encoder(images).float().cpu())
            mask_batches.append(predictor.image_encoder(masks).float().cpu())
            src_batches.append(predictor.text_encoder(src_prompts[start:end]).float().cpu())
            tar_batches.append(predictor.text_encoder(tar_prompts[start:end]).float().cpu())
        del images, masks

    img_emb = torch.cat(img_batches, dim=0)
    mask_emb = torch.cat(mask_batches, dim=0)
    src_emb = torch.cat(src_batches, dim=0)
    tar_emb = torch.cat(tar_batches, dim=0)

    if cache_scattered:
        # Write many per-sample .pt files indexed by EMBEDDINGS_CSV.
        samples_root = EMBEDDINGS_DIR / EMBEDDINGS_SAMPLES_DIRNAME
        print(f"Caching scattered embeddings ({n_samples} samples) -> {samples_root}")
        rows: list[dict[str, str]] = []
        for i, sid in enumerate(tqdm(sample_ids, desc="Caching scattered", unit="sample")):
            sample_dir = samples_root / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            img_path = sample_dir / "image.pt"
            mask_path = sample_dir / "mask.pt"
            src_path = sample_dir / "source.pt"
            tar_path = sample_dir / "target.pt"
            torch.save(img_emb[i].contiguous(), img_path)
            torch.save(mask_emb[i].contiguous(), mask_path)
            torch.save(src_emb[i].contiguous(), src_path)
            torch.save(tar_emb[i].contiguous(), tar_path)
            rows.append({
                SAMPLE_ID_COL: sid,
                SOURCE_EMB_COL: str(src_path),
                TARGET_EMB_COL: str(tar_path),
                IMAGE_EMB_COL: str(img_path),
                MASK_EMB_COL: str(mask_path),
            })
        if EMBEDDINGS_CSV is not None:
            csv_path = Path(EMBEDDINGS_CSV)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).sort_values(SAMPLE_ID_COL).to_csv(csv_path, index=False)
            print(f"Saved scattered embeddings CSV: {csv_path}")

    if cache_packed:
        # Pack encoder outputs into one packed training table.
        tables = {"img": img_emb, "mask": mask_emb, "src": src_emb, "tar": tar_emb}
        _save_packed_cache(packed_path, sample_ids, tables, source="encoder")

    return sample_ids, img_emb, mask_emb, src_emb, tar_emb


def get_embeddings(
    samples: pd.DataFrame,
    predictor: SurrogateModel | None,
    *,
    batch_size: int = EMBED_BATCH_SIZE,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU embedding tables (img, mask, src, tar), packed for training.

    Tries the packed cache, then packs scattered files, and only then encodes
    with the (always frozen) ChordEdit encoders. Encoding requires a predictor
    with live encoders; predictor=None is pack-only and raises instead.
    """
    # Validate the ChordEdit model type up front: flux raises NotImplementedError
    # here regardless of cache state, matching SurrogateModel.__init__.
    _get_text_pooling()
    packed_path = _get_packed_path()
    sample_ids = samples[SAMPLE_ID_COL].tolist()

    cached = _load_packed_cache(packed_path, sample_ids)
    if cached is not None:
        print("Loaded packed embeddings from cache.")
        return cached

    packed = _pack_scattered_cache(samples, packed_path)
    if packed is not None:
        print("Loaded scattered embeddings from cache.")
        return packed

    if getattr(predictor, "image_encoder", None) is None or predictor is None:
        raise RuntimeError(
            f"Embeddings unavailable: {packed_path} cannot cover {len(sample_ids)} requested "
            f"samples and scattered embeddings ({EMBEDDINGS_CSV}) are missing or incomplete. "
            f"No encoder-bearing predictor was provided (predictor=None or encoders released), "
            f"so on-the-fly encoding is not possible. Produce scattered embeddings with the "
            f"annotation pipeline, or pass a SurrogateModel with live encoders."
        )
    return _encode_embeddings(samples, predictor, packed_path, batch_size=batch_size)


def get_embeddings_by_sample(
    df: pd.DataFrame,
    predictor: SurrogateModel | None,
    device: torch.device | str | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Embeddings keyed by sample_id, on device (default: the predictor's device)."""
    if device is None:
        if predictor is None:
            raise ValueError("No predictor provided, no device to embed on")
        device = predictor.regressor.target_mean.device
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, mask_emb, src_emb, tar_emb = get_embeddings(samples, predictor)
    img_emb, mask_emb, src_emb, tar_emb = img_emb.to(device), mask_emb.to(device), src_emb.to(device), tar_emb.to(device)
    return {sid:{ 
        "img": img_emb[i], 
        "mask": mask_emb[i], 
        "src": src_emb[i], 
        "tar": tar_emb[i]
        } for i, sid in enumerate(sample_ids)
    }


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
