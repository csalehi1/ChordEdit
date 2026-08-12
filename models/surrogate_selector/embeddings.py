# embeddings.py

"""Load embeddings from scattered files or packed tables."""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import *

_PACKED_EMBEDDINGS_DIR = Path(__file__).resolve().parent / ".cache" / "packed_embeddings"
_SCATTERED_EMBEDDINGS_DIR = SCATTERED_DIR / "annotation_embeddings"
_EMB_FILES = {"img": "image.pt", "src": "source.pt", "tar": "target.pt"}


def _scattered_path(sample_id: str, kind: str) -> Path:
    return _SCATTERED_EMBEDDINGS_DIR / sample_id / _EMB_FILES[kind]


def _text_dim(probe: torch.Tensor, path) -> int:
    """Validate a stored packing-ready text vector and return its dim."""
    if probe.numel() != probe.shape[-1]:
        raise ValueError(f"Expected a pooled text vector (D,) or (1, D), got {tuple(probe.shape)} in {path}.")
    return int(probe.shape[-1])


"""
Packed/scattered caches.
"""

def _get_packed_path() -> Path:
    """Packed cache path for the current settings."""
    t_delta = f"{TARGET_T_DELTA}".replace(".", "p")
    slug = DIR_NAME.replace("_", "").lower()
    return _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-{t_delta}-{slug}.pt"


def _expected_packed_meta() -> dict:
    """Meta that a valid packed cache must carry."""
    return {
        "model": CHORD_EDIT_MODEL,
        "pipeline_type": CHORD_EDIT_PIPELINE_TYPE,
        "layout": "img_src_tar_v1",
        "image_size": int(CHORD_EDIT_IMAGE_SIZE),
        "dir_name": DIR_NAME,
        "target_t_delta": TARGET_T_DELTA,
    }


def _load_pt(path: str | Path) -> torch.Tensor:
    """Load one scattered embedding .pt file to a float32 CPU tensor."""
    t = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
    return t.detach().float()


def _load_packed_cache(
    packed_path: Path,
    sample_ids: list[str],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Packed tables sliced to sample_ids, or None (missing / bad meta / coverage)."""

    # Check if the packed cache exists.
    if not packed_path.exists():
        return None
    try:
        data = torch.load(packed_path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:
        print(f"Failed to load {packed_path} ({e}). Repacking...")
        return None

    # Get the meta data from the packed cache.
    meta = data.get("meta")

    # Verify that the meta data is consistent with the expected meta data.
    if not isinstance(meta, dict):
        print(f"{packed_path} has no meta. Repacking...")
        return None
    for key, expected in _expected_packed_meta().items():
        if meta.get(key) != expected:
            print(f"{packed_path} meta mismatch: {key}={meta.get(key)!r} expected {expected!r}. Repacking...")
            return None
    if data["img"].shape[1] != meta.get("img_dim") or data["src"].shape[1] != meta.get("text_dim"):
        print(f"{packed_path} table dims do not match meta. Repacking...")
        return None
    id_to_i = {sid: i for i, sid in enumerate(data["sids"])}
    n_missing = sum(sid not in id_to_i for sid in sample_ids)
    if n_missing:
        print(f"{packed_path} missing {n_missing} of {len(sample_ids)} requested samples. Repacking...")
        return None

    # Get the indices of the samples in the packed cache.
    idxs = [id_to_i[sid] for sid in sample_ids]

    return (
        sample_ids,
        data["img"][idxs].contiguous(),
        data["src"][idxs].contiguous(),
        data["tar"][idxs].contiguous(),
    )


def _save_packed_cache(
    packed_path: Path,
    sample_ids: list[str],
    tables: dict[str, torch.Tensor],
) -> None:
    """Atomically write the packed table dict with meta from scattered files."""
    meta = _expected_packed_meta() | {
        "source": "scattered",
        "img_dim": int(tables["img"].shape[1]),
        "text_dim": int(tables["src"].shape[1]),
    }
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(packed_path) + ".tmp")
    torch.save({"sids": list(sample_ids), **tables, "meta": meta}, tmp_path)
    tmp_path.replace(packed_path)
    print(f"Saved packed embeddings: {packed_path}")


def _pack_scattered_cache(
    samples_df: pd.DataFrame,
    packed_path: Path,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pack scattered per-sample .pt files into packed tables."""
    sample_ids = samples_df[SAMPLE_ID_COL].tolist()
    if not sample_ids:
        raise ValueError("No samples to pack")

    missing = [sid for sid in sample_ids if not _scattered_path(sid, "img").exists()]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"Scattered embeddings missing {len(missing)} sample_id(s): {preview}{more}")
        return None

    # Probe dims from the first sample, then preallocate the packed tables so
    # that peak memory stays ~1x table size (no row lists + torch.stack copy).
    n = len(sample_ids)
    img_dim = _load_pt(_scattered_path(sample_ids[0], "img")).numel()
    src_probe_path = _scattered_path(sample_ids[0], "src")
    text_dim = _text_dim(_load_pt(src_probe_path), src_probe_path)
    img_emb = torch.empty((n, img_dim), dtype=torch.float32)
    src_emb = torch.empty((n, text_dim), dtype=torch.float32)
    tar_emb = torch.empty((n, text_dim), dtype=torch.float32)

    def _load_row(i: int):
        sid = sample_ids[i]
        return (
            i,
            _load_pt(_scattered_path(sid, "img")),
            _load_pt(_scattered_path(sid, "src")),
            _load_pt(_scattered_path(sid, "tar")),
        )

    def _check(t: torch.Tensor, i: int, kind: str, dim: int) -> torch.Tensor:
        if t.numel() != dim:
            raise ValueError(f"{kind} numel {t.numel()} != {dim} for sample {sample_ids[i]}")
        return t.reshape(-1)

    with ThreadPoolExecutor(max_workers=32) as pool:
        # Load the embeddings in parallel using a thread pool.
        for i, img_t, src_t, tar_t in tqdm(pool.map(_load_row, range(n)), total=n, desc="Packing embeddings", unit="sample"):
            img_emb[i] = _check(img_t, i, "image", img_dim)
            src_emb[i] = _check(src_t, i, "source", text_dim)
            tar_emb[i] = _check(tar_t, i, "target", text_dim)

    tables = {"img": img_emb, "src": src_emb, "tar": tar_emb}
    _save_packed_cache(packed_path, sample_ids, tables)
    return sample_ids, img_emb, src_emb, tar_emb


def get_embeddings(
    samples: pd.DataFrame,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU embedding tables (img, src, tar), packed for training."""
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

    raise RuntimeError(f"Embeddings unavailable: {packed_path} cannot cover {len(sample_ids)} requested.")


def get_embeddings_by_sample(
    df: pd.DataFrame,
    device: torch.device | str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Embeddings keyed by sample_id, on device."""
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, src_emb, tar_emb = get_embeddings(samples)
    img_emb, src_emb, tar_emb = img_emb.to(device), src_emb.to(device), tar_emb.to(device)

    return {sid: {
        "img": img_emb[i],
        "src": src_emb[i],
        "tar": tar_emb[i],
    } for i, sid in enumerate(sample_ids)}
