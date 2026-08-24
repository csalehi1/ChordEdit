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
_EMB_FILES = {"img": "image_tokens.pt", "src": "source.pt", "tar": "target.pt"}


def _scattered_path(sample_id: str, kind: str) -> Path:
    return _SCATTERED_EMBEDDINGS_DIR / sample_id / _EMB_FILES[kind]


def _img_token_shape(probe: torch.Tensor, path) -> tuple[int, ...]:
    """Validate a stored latent token tensor (C, S, S) and return its shape."""
    if probe.ndim != 3 or probe.shape[-1] != probe.shape[-2]:
        raise ValueError(f"Expected {tuple(probe.shape)} == (C, S, S) in {path}")
    return tuple(probe.shape)


def _text_shape(probe: torch.Tensor, path) -> tuple[int, ...]:
    """Validate a stored pooled text vector (D,) or (1, D) and return (D,)."""
    if probe.numel() != probe.shape[-1]:
        raise ValueError(f"Expected a pooled text vector (D,) or (1, D), got {tuple(probe.shape)} in {path}.")
    return (int(probe.shape[-1]),)


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
        "layout": "img_tokens_src_tar_pooled_v1",
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
    img_shape = tuple(meta.get("img_shape") or ())
    text_shape = tuple(meta.get("text_shape") or ())
    if tuple(data["img"].shape[1:]) != img_shape or tuple(data["src"].shape[1:]) != text_shape:
        print(f"{packed_path} table shapes do not match meta. Repacking...")
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
        "img_shape": tuple(tables["img"].shape[1:]),
        "text_shape": tuple(tables["src"].shape[1:]),
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
        raise ValueError("Expected samples to pack")

    missing = [sid for sid in sample_ids if not _scattered_path(sid, "img").exists()]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"Scattered embeddings missing {len(missing)} sample_id(s): {preview}{more}")
        return None

    # Probe shapes from the first sample, then preallocate the packed tables so
    # that peak memory stays ~1x table size (no row lists + torch.stack copy).
    # Image token shapes are kept as saved; text vectors are already pooled.
    n = len(sample_ids)
    img_probe_path = _scattered_path(sample_ids[0], "img")
    img_shape = _img_token_shape(_load_pt(img_probe_path), img_probe_path)
    src_probe_path = _scattered_path(sample_ids[0], "src")
    text_shape = _text_shape(_load_pt(src_probe_path), src_probe_path)
    img_emb = torch.empty((n, *img_shape), dtype=torch.float32)
    src_emb = torch.empty((n, *text_shape), dtype=torch.float32)
    tar_emb = torch.empty((n, *text_shape), dtype=torch.float32)

    def _load_row(i: int):
        sid = sample_ids[i]
        return (
            i,
            _load_pt(_scattered_path(sid, "img")),
            _load_pt(_scattered_path(sid, "src")),
            _load_pt(_scattered_path(sid, "tar")),
        )

    def _check(t: torch.Tensor, i: int, kind: str, shape: tuple[int, ...]) -> torch.Tensor:
        if tuple(t.shape) != shape:
            raise ValueError(f"Expected {tuple(t.shape)} == {shape} for {kind} sample {sample_ids[i]}")
        return t

    with ThreadPoolExecutor(max_workers=32) as pool:
        # Load the embeddings in parallel using a thread pool.
        for i, img_t, src_t, tar_t in tqdm(pool.map(_load_row, range(n)), total=n, desc="Packing embeddings", unit="sample"):
            img_emb[i] = _check(img_t, i, "image", img_shape)
            # Text rows may be saved as (D,) or (1, D); flatten before checking.
            src_emb[i] = _check(src_t.reshape(-1), i, "source", text_shape)
            tar_emb[i] = _check(tar_t.reshape(-1), i, "target", text_shape)

    tables = {"img": img_emb, "src": src_emb, "tar": tar_emb}
    _save_packed_cache(packed_path, sample_ids, tables)
    return sample_ids, img_emb, src_emb, tar_emb


def get_embeddings(
    samples: pd.DataFrame,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU tables (img, src, tar), packed for training.

    img keeps the saved token shape (n, C, S, S) VAE latents; the
    cross-attention regressor consumes the token structure directly. src/tar
    are (n, D) masked-mean-pooled prompt vectors, the pipeline's own
    source.pt / target.pt, so no pooling happens in the model.
    """
    packed_path = _get_packed_path()
    sample_ids = samples[SAMPLE_ID_COL].tolist()

    cached = _load_packed_cache(packed_path, sample_ids)
    if cached is None:
        cached = _pack_scattered_cache(samples, packed_path)
        if cached is None:
            raise RuntimeError(f"Embeddings unavailable: {packed_path} cannot cover {len(sample_ids)} requested.")
        print("Loaded scattered embeddings from cache.")
    else:
        print("Loaded packed embeddings from cache.")

    return cached


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
