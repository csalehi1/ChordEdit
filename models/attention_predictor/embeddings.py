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
_EMB_FILES = {
    "img": "image_tokens.pt",
    "src": "source.pt",
    "tar": "target.pt",
    "src_tokens": "source_tokens.pt",
    "tar_tokens": "target_tokens.pt",
    "src_mask": "source_mask.pt",
    "tar_mask": "target_mask.pt",
}

# mlp_predictor already builds and caches the pooled CLIP-L/14 image table, and
# its cache is keyed by DIR_NAME alone, so importing the module rather than
# forking it means both packages read and write the same file. It does
# `from settings import *`, which resolves to whichever settings module is
# already bound -- this package's, or a run snapshot under selector.py.
_MLP_DIR = Path(__file__).resolve().parents[1] / "mlp_predictor"


def get_clip_image_table(samples: pd.DataFrame, device: torch.device | str = "cuda") -> torch.Tensor:
    """(n, 1, D_clip) pooled CLIP-L/14 image embeddings, one row per sample.

    Shaped as a single token so it can be concatenated onto the visual token
    sequence the cross-attention reads. Encodes on a cache miss, which is why
    the cache must be warm before a parallel sweep.
    """
    if str(_MLP_DIR) not in sys.path:
        sys.path.append(str(_MLP_DIR))
    import clip_image

    # clip_image puts its own package dir at sys.path[0] on import. Drop it once
    # the module is bound, so nothing imported later resolves to mlp_predictor's
    # copy of a module this package also has.
    while str(_MLP_DIR) in sys.path:
        sys.path.remove(str(_MLP_DIR))

    if PIE_BENCH:
        raise NotImplementedError(
            "IMG_EMB_SOURCE != 'vae' has no PIE-Bench path: clip_image caches by DIR_NAME, "
            "so a mixed UltraEdit/PIE split would need two tables stitched together."
        )
    emb = clip_image.get_clip_image_embeddings(samples, device=device)
    return emb.unsqueeze(-2).contiguous()


def _scattered_path(
    sample_id: str,
    kind: str,
    *,
    scattered_dir: Path | None = None,
) -> Path:
    root = (scattered_dir if scattered_dir is not None else SCATTERED_DIR) / "annotation_embeddings"
    return root / sample_id / _EMB_FILES[kind]


def _img_token_shape(probe: torch.Tensor, path) -> tuple[int, ...]:
    """Validate a stored latent token tensor (C, S, S) and return its shape."""
    if probe.ndim != 3 or probe.shape[-1] != probe.shape[-2]:
        raise ValueError(f"Expected {tuple(probe.shape)} == (C, S, S) in {path}")
    return tuple(probe.shape)


def _text_shape(probe: torch.Tensor, path) -> tuple[int, ...]:
    """Validate a stored pooled text vector (D,) or (1, D) and return (1, D)."""
    if probe.numel() != probe.shape[-1]:
        raise ValueError(f"Expected a pooled text vector (D,) or (1, D), got {tuple(probe.shape)} in {path}.")
    return (1, int(probe.shape[-1]))


"""
Packed/scattered caches.
"""

def _get_packed_path(*, dir_name: str | None = None) -> Path:
    """Packed cache path for the current settings (or an override dir_name)."""
    t_delta = f"{TARGET_T_DELTA}".replace(".", "p")
    slug = (dir_name if dir_name is not None else DIR_NAME).replace("_", "").lower()
    return _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-{t_delta}-{slug}.pt"


def _expected_packed_meta(*, dir_name: str | None = None) -> dict:
    """Meta that a valid packed cache must carry."""
    return {
        "model": CHORD_EDIT_MODEL,
        "pipeline_type": CHORD_EDIT_PIPELINE_TYPE,
        "layout": "img_tokens_src_tar_pooled_v2",
        "image_size": int(CHORD_EDIT_IMAGE_SIZE),
        "dir_name": dir_name if dir_name is not None else DIR_NAME,
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
    *,
    dir_name: str | None = None,
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
    for key, expected in _expected_packed_meta(dir_name=dir_name).items():
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
    *,
    dir_name: str | None = None,
) -> None:
    """Atomically write the packed table dict with meta from scattered files."""
    meta = _expected_packed_meta(dir_name=dir_name) | {
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
    *,
    scattered_dir: Path | None = None,
    dir_name: str | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Pack scattered per-sample .pt files into packed tables."""
    sample_ids = samples_df[SAMPLE_ID_COL].tolist()
    if not sample_ids:
        raise ValueError("Expected samples to pack")

    missing = [
        sid for sid in sample_ids
        if not _scattered_path(sid, "img", scattered_dir=scattered_dir).exists()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
        print(f"Scattered embeddings missing {len(missing)} sample_id(s): {preview}{more}")
        return None

    # Probe shapes from the first sample, then preallocate the packed tables so
    # that peak memory stays ~1x table size (no row lists + torch.stack copy).
    # Image token shapes are kept as saved; text vectors are already pooled.
    n = len(sample_ids)
    img_probe_path = _scattered_path(sample_ids[0], "img", scattered_dir=scattered_dir)
    img_shape = _img_token_shape(_load_pt(img_probe_path), img_probe_path)
    src_probe_path = _scattered_path(sample_ids[0], "src", scattered_dir=scattered_dir)
    text_shape = _text_shape(_load_pt(src_probe_path), src_probe_path)
    img_emb = torch.empty((n, *img_shape), dtype=torch.float32)
    src_emb = torch.empty((n, *text_shape), dtype=torch.float32)
    tar_emb = torch.empty((n, *text_shape), dtype=torch.float32)

    def _load_row(i: int):
        sid = sample_ids[i]
        return (
            i,
            _load_pt(_scattered_path(sid, "img", scattered_dir=scattered_dir)),
            _load_pt(_scattered_path(sid, "src", scattered_dir=scattered_dir)),
            _load_pt(_scattered_path(sid, "tar", scattered_dir=scattered_dir)),
        )

    def _check(t: torch.Tensor, i: int, kind: str, shape: tuple[int, ...]) -> torch.Tensor:
        if tuple(t.shape) != shape:
            raise ValueError(f"Expected {tuple(t.shape)} == {shape} for {kind} sample {sample_ids[i]}")
        return t

    with ThreadPoolExecutor(max_workers=32) as pool:
        # Load the embeddings in parallel using a thread pool.
        for i, img_t, src_t, tar_t in tqdm(pool.map(_load_row, range(n)), total=n, desc="Packing embeddings", unit="sample"):
            img_emb[i] = _check(img_t, i, "image", img_shape)
            # Text rows may be saved as (D,) or (1, D); the table keeps the
            # single-token layout (1, D) that the text featurizer consumes.
            src_emb[i] = _check(src_t.reshape(1, -1), i, "source", text_shape)
            tar_emb[i] = _check(tar_t.reshape(1, -1), i, "target", text_shape)

    tables = {"img": img_emb, "src": src_emb, "tar": tar_emb}
    _save_packed_cache(packed_path, sample_ids, tables, dir_name=dir_name)
    return sample_ids, img_emb, src_emb, tar_emb


def get_embeddings(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
    dir_name: str | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU tables (img, src, tar), packed for training.

    img keeps the saved token shape (n, C, S, S) VAE latents; the
    cross-attention regressor consumes the token structure directly. src/tar
    are (n, 1, D) masked-mean-pooled prompt vectors, one text token each, the
    pipeline's own source.pt / target.pt, so no pooling happens in the model.

    Optional scattered_dir / dir_name point at a non-default dataset root
    (used for PIE-Bench when --pie-bench mixes UltraEdit train with PIE test).
    """
    packed_path = _get_packed_path(dir_name=dir_name)
    sample_ids = samples[SAMPLE_ID_COL].tolist()

    cached = _load_packed_cache(packed_path, sample_ids, dir_name=dir_name)
    if cached is None:
        cached = _pack_scattered_cache(
            samples, packed_path, scattered_dir=scattered_dir, dir_name=dir_name,
        )
        if cached is None:
            raise RuntimeError(f"Embeddings unavailable: {packed_path} cannot cover {len(sample_ids)} requested.")
        print("Loaded scattered embeddings from cache.")
    else:
        print("Loaded packed embeddings from cache.")

    return cached


def get_embeddings_mixed(
    samples: pd.DataFrame,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load embeddings, routing pie_*-prefixed ids to PIE_SCATTERED_DIR when PIE_BENCH."""
    if not PIE_BENCH:
        return get_embeddings(samples)

    sids = samples[SAMPLE_ID_COL].astype(str)
    is_pie = sids.str.startswith(PIE_SAMPLE_ID_PREFIX)
    ue_samples = samples.loc[~is_pie].copy()
    pie_samples = samples.loc[is_pie].copy()

    parts: list[tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if len(ue_samples):
        parts.append(get_embeddings(ue_samples))
    if len(pie_samples):
        # Disk folders use unprefixed ids; remap back to pie_* for the shared table.
        pie_raw = pie_samples.copy()
        pie_raw[SAMPLE_ID_COL] = pie_raw[SAMPLE_ID_COL].str.removeprefix(PIE_SAMPLE_ID_PREFIX)
        raw_ids, img, src, tar = get_embeddings(
            pie_raw,
            scattered_dir=PIE_SCATTERED_DIR,
            dir_name=PIE_BENCH_DIR_NAME,
        )
        parts.append(([PIE_SAMPLE_ID_PREFIX + sid for sid in raw_ids], img, src, tar))
    if not parts:
        raise ValueError("Expected samples with embeddings")

    sample_ids = [sid for part in parts for sid in part[0]]
    img_emb = torch.cat([part[1] for part in parts], dim=0)
    src_emb = torch.cat([part[2] for part in parts], dim=0)
    tar_emb = torch.cat([part[3] for part in parts], dim=0)
    return sample_ids, img_emb, src_emb, tar_emb


def get_embeddings_by_sample(
    df: pd.DataFrame,
    device: torch.device | str,
) -> dict[str, dict[str, torch.Tensor]]:
    """Embeddings keyed by sample_id, on device."""
    samples = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL)
    sample_ids, img_emb, src_emb, tar_emb = get_embeddings_mixed(samples)
    img_emb, src_emb, tar_emb = img_emb.to(device), src_emb.to(device), tar_emb.to(device)

    # The CLIP table is keyed by the frame's row order, so reindex it onto
    # sample_ids the packed loader returned.
    clip_emb = None
    if IMG_EMB_SOURCE != "vae":
        order = {sid: i for i, sid in enumerate(samples[SAMPLE_ID_COL].tolist())}
        table = get_clip_image_table(samples, device=device).to(device)
        clip_emb = table[[order[sid] for sid in sample_ids]]

    return {sid: {
        "img": img_emb[i],
        "src": src_emb[i],
        "tar": tar_emb[i],
        "clip": None if clip_emb is None else clip_emb[i],
    } for i, sid in enumerate(sample_ids)}


"""
Text token tables: full (77, D) prompt sequences plus their padding masks.
"""

_TEXT_TOKEN_LAYOUT = "text_tokens_masks_v1"


def _text_token_path(*, dir_name: str | None = None) -> Path:
    slug = (dir_name if dir_name is not None else DIR_NAME).replace("_", "").lower()
    return _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-texttokens-{slug}.pt"


def _text_token_meta(*, dir_name: str | None = None) -> dict:
    return {
        "model": CHORD_EDIT_MODEL,
        "layout": _TEXT_TOKEN_LAYOUT,
        "dir_name": dir_name if dir_name is not None else DIR_NAME,
    }


def get_text_token_tables(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
    dir_name: str | None = None,
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sids, src_tokens, tar_tokens, src_mask, tar_mask) from the scattered tree.

    Tokens are (n, 77, D) float16 on disk, masks (n, 77) bool. The pipeline's
    pooled source.pt is the mask-weighted mean of source_tokens.pt, which is
    asserted on a sample of rows at pack time so a mismatched mask cannot pass
    silently.
    """
    sample_ids = samples[SAMPLE_ID_COL].tolist()
    path = _text_token_path(dir_name=dir_name)

    if path.exists():
        try:
            data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            meta = data.get("meta")
            ok = isinstance(meta, dict) and all(
                meta.get(k) == v for k, v in _text_token_meta(dir_name=dir_name).items()
            )
            id_to_i = {sid: i for i, sid in enumerate(data["sids"])} if ok else {}
            if ok and not any(sid not in id_to_i for sid in sample_ids):
                idxs = [id_to_i[sid] for sid in sample_ids]
                print("Loaded packed text tokens from cache.")
                return (
                    sample_ids,
                    data["src_tokens"][idxs].float(), data["tar_tokens"][idxs].float(),
                    data["src_mask"][idxs].contiguous(), data["tar_mask"][idxs].contiguous(),
                )
            print(f"{path} unusable for this request. Repacking...")
        except Exception as e:
            print(f"Failed to load {path} ({e}). Repacking...")

    n = len(sample_ids)
    probe = _load_pt(_scattered_path(sample_ids[0], "src_tokens", scattered_dir=scattered_dir))
    if probe.ndim != 2:
        raise ValueError(f"Expected (T, D) prompt tokens, got {tuple(probe.shape)}")
    shape = tuple(probe.shape)
    src_tok = torch.empty((n, *shape), dtype=torch.float16)
    tar_tok = torch.empty((n, *shape), dtype=torch.float16)
    src_msk = torch.empty((n, shape[0]), dtype=torch.bool)
    tar_msk = torch.empty((n, shape[0]), dtype=torch.bool)

    def _load_row(i: int):
        sid = sample_ids[i]
        out = [i]
        for kind in ("src_tokens", "tar_tokens", "src_mask", "tar_mask"):
            fp = _scattered_path(sid, kind, scattered_dir=scattered_dir)
            if not fp.exists():
                raise FileNotFoundError(f"Missing {fp}")
            out.append(torch.load(fp, map_location="cpu", weights_only=True))
        return tuple(out)

    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, st, tt, sm, tm in tqdm(
            pool.map(_load_row, range(n)), total=n, desc="Packing text tokens", unit="sample"
        ):
            for name, t in (("source", st), ("target", tt)):
                if tuple(t.shape) != shape:
                    raise ValueError(f"Expected {shape} for {name} tokens of {sample_ids[i]}, got {tuple(t.shape)}")
            src_tok[i], tar_tok[i] = st.half(), tt.half()
            src_msk[i], tar_msk[i] = sm.bool().reshape(-1), tm.bool().reshape(-1)

    _verify_pooling(sample_ids, src_tok, src_msk, "src", scattered_dir=scattered_dir)

    meta = _text_token_meta(dir_name=dir_name) | {"text_shape": shape}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    torch.save({
        "sids": list(sample_ids), "meta": meta,
        "src_tokens": src_tok.contiguous().clone(), "tar_tokens": tar_tok.contiguous().clone(),
        "src_mask": src_msk.contiguous().clone(), "tar_mask": tar_msk.contiguous().clone(),
    }, tmp)
    tmp.replace(path)
    print(f"Saved packed text tokens: {path}")
    return sample_ids, src_tok.float(), tar_tok.float(), src_msk, tar_msk


def _verify_pooling(
    sample_ids: list[str],
    tokens: torch.Tensor,
    masks: torch.Tensor,
    kind: str,
    *,
    scattered_dir: Path | None = None,
    n_check: int = 64,
) -> None:
    """Assert the masked mean of the packed tokens reproduces the pooled vector.

    This is what proves the stored padding masks are the ones the pipeline
    pooled with; a mask off by a token would still look plausible otherwise.
    """
    step = max(1, len(sample_ids) // n_check)
    checked = 0
    for i in range(0, len(sample_ids), step):
        pooled = _load_pt(_scattered_path(sample_ids[i], kind, scattered_dir=scattered_dir)).reshape(-1)
        m = masks[i].float().unsqueeze(-1)
        got = (tokens[i].float() * m).sum(0) / m.sum(0).clamp(min=1e-9)
        err = (got - pooled).abs().max().item()
        if err > 5e-3:
            raise ValueError(
                f"Masked mean of {kind}_tokens does not reproduce {kind}.pt for "
                f"sample {sample_ids[i]} (max abs err {err:.3e})"
            )
        checked += 1
    print(f"Verified masked-mean pooling on {checked} sampled rows (max err < 5e-3).")
