# embeddings.py

"""
Cache layer and the ONE embedding-assembly path.

Scattered per-sample .pt files are packed into cached tables (formats on disk
are frozen), and get_embeddings turns a samples frame into a device-resident
EmbeddingsTable keyed by sample_id. Consumers look rows up by sample_id; the
table translates ids to row indices internally.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import *

_PACKED_EMBEDDINGS_DIR = Path(__file__).resolve().parent / ".cache" / "packed_embeddings"

# The image and its masked counterpart, each encoded both ways. The CLIP files
# are read separately by get_clip_tokens, which pools them on load rather than
# packing 257 tokens per sample into the cached tables.
_CLIP_TOKENS_FILE = "image_clip_tokens.pt"
_EMB_FILES = {
    "img": "image_vae_tokens.pt",
    "src_tokens": "source_tokens.pt",
    "tar_tokens": "target_tokens.pt",
    "src_mask": "source_mask.pt",
    "tar_mask": "target_mask.pt",
}

# CLIP-L/14 embeddings in the joint image/text space, L2-normalized, plus the
# mask area fraction. Only read when USE_IMG_MASK. Their cosines are what
# CLIP-Edited is a CLIPScore of, which no other embedding in the tree expresses.
_MASK_FILES = {
    "masked_proj": "masked_clip_proj.pt",
    "img_proj": "image_clip_proj.pt",
    "src_proj": "source_clip_proj.pt",
    "tar_proj": "target_clip_proj.pt",
    "mask_area": "mask_area.pt",
}

# One projected embedding plus four cosines and the mask area.
CLIP_PROJ_DIM = 768
FEATURE_DIM = CLIP_PROJ_DIM + 5



"""
Embeddings classes.
"""


@dataclass(frozen=True)
class SampleEmbeddings:
    """One sample's visual and textual embeddings."""

    image_tokens: torch.Tensor          # (C, S, S) vae; (1, D_clip) clip; (N_v+1, D) vae_clip
    source_tokens: torch.Tensor         # (T, D_txt) tokens, or (1, D_txt) pooled
    target_tokens: torch.Tensor         # like source_tokens
    source_mask: torch.Tensor           # (N_t,) bool; ones when pooled
    target_mask: torch.Tensor           # like source_mask
    mask_features: torch.Tensor         # (D_feat,); ones when unused


@dataclass(frozen=True)
class EmbeddingsTable:
    """Group of samples' visual and textual embeddings, resident on one device."""

    _sample_ids: tuple[str, ...]
    _sid_to_idx: dict[str, int]

    image_tokens: torch.Tensor          # (N, C, S, S) vae; (N, 1, D_clip) clip; (N, N_v+1, D) vae_clip
    source_tokens: torch.Tensor         # (N, T, D_txt) tokens, or (N, 1, D_txt) pooled
    target_tokens: torch.Tensor         # like source_tokens
    source_mask: torch.Tensor           # (N, N_t) bool; ones when pooled
    target_mask: torch.Tensor           # like source_mask
    mask_features: torch.Tensor         # (N, D_feat); ones when unused

    @property
    def feature_shape(self) -> tuple[int, ...]:
        """Per-sample mask_features shape."""
        return tuple(self.mask_features.shape[1:])

    @property
    def image_shape(self) -> tuple[int, ...]:
        """Per-sample image_tokens shape."""
        return tuple(self.image_tokens.shape[1:])

    @property
    def source_shape(self) -> tuple[int, ...]:
        """Per-sample source_tokens shape."""
        return tuple(self.source_tokens.shape[1:])

    @property
    def target_shape(self) -> tuple[int, ...]:
        """Per-sample target_tokens shape."""
        return tuple(self.target_tokens.shape[1:])

    def sample_idx(self, sample_ids: list[str]) -> torch.Tensor:
        """Map sample ids to sample indices."""
        idx = [self._sid_to_idx[sid] for sid in sample_ids]
        return torch.tensor(idx, dtype=torch.long, device=self.image_tokens.device)

    def get_sample_embeddings(self, sample_id: str) -> SampleEmbeddings:
        """Get one sample's embeddings by sample_id."""
        i = self._sid_to_idx[sample_id]
        return SampleEmbeddings(
            image_tokens=self.image_tokens[i],
            source_tokens=self.source_tokens[i],
            target_tokens=self.target_tokens[i],
            source_mask=self.source_mask[i],
            target_mask=self.target_mask[i],
            mask_features=self.mask_features[i],
        )


"""
Scattered caches.
"""

def _concat_vae_clip_tokens(vae: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Flatten VAE to (N, N_v, D_vae) and concatenate CLIP (N, 1, D_clip) on dim=-2."""
    n, c, s, s2 = vae.shape
    if s != s2 or s % PATCH_SIZE != 0:
        raise ValueError(f"Expected square latents with side divisible by {PATCH_SIZE=}, got {tuple(vae.shape)}")
    g = s // PATCH_SIZE
    tokens = vae.reshape(n, c, g, PATCH_SIZE, g, PATCH_SIZE)
    tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(n, g * g, c * PATCH_SIZE ** 2)
    width = max(tokens.shape[-1], clip.shape[-1])
    tokens = torch.nn.functional.pad(tokens, (0, width - tokens.shape[-1]))
    clip = torch.nn.functional.pad(clip.to(tokens.dtype), (0, width - clip.shape[-1]))
    return torch.cat([tokens, clip], dim=-2)


def get_clip_tokens(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
    tokens_file: str = _CLIP_TOKENS_FILE,
    pool: bool | None = None,
) -> torch.Tensor:
    """(n, N_v, D_clip) CLIP-L/14 vision tokens, pooled to one token per sample when pooling."""
    root = (scattered_dir if scattered_dir is not None else SCATTERED_DIR) / "annotation_embeddings"
    sample_ids = samples[SAMPLE_ID_COL].astype(str).tolist()
    pool = IMG_EMB_POOL if pool is None else pool

    def _load_clip_tokens(sample_id: str) -> torch.Tensor:
        """Load one sample's tokens, pooling them here so the table stays small."""
        tokens = torch.load(root / sample_id / tokens_file, map_location="cpu", weights_only=True)
        return tokens.mean(dim=0, keepdim=True) if pool else tokens

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(tqdm(
            pool.map(_load_clip_tokens, sample_ids),
            total=len(sample_ids), desc=f"Loading {tokens_file}", unit="sample",
        ))
    return torch.stack(rows).float().contiguous()


def get_img_mask_features(tables: dict[str, torch.Tensor]) -> torch.Tensor:
    """(n, FEATURE_DIM) masked-image block: the projected masked embedding and five scalars.

    The cosines are computed here rather than cached. Every vector is already
    L2-normalized in CLIP's joint space, so each one is a plain dot product.
    """
    def _cos(a: str, b: str) -> torch.Tensor:
        """Row-wise cosine, as a column so the scalars concatenate."""
        return (tables[a] * tables[b]).sum(dim=-1, keepdim=True)

    return torch.cat([
        tables["masked_proj"],
        # Nearly the CLIP label itself at low t_start, where the edit has barely
        # moved the image, so it anchors one end of every sample's surface.
        _cos("masked_proj", "tar_proj"),
        _cos("masked_proj", "src_proj"),
        _cos("img_proj", "tar_proj"),
        # How far apart the prompts are, which sets how large an edit is asked for.
        _cos("src_proj", "tar_proj"),
        tables["mask_area"],
    ], dim=-1).contiguous()


def get_scattered_embeddings(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load per-sample .pt files from the scattered tree into CPU tables."""

    def _load_scattered_cache(sample_id: str, kind: str) -> torch.Tensor:
        """Load one scattered file to a CPU tensor."""
        root = (scattered_dir if scattered_dir is not None else SCATTERED_DIR) / "annotation_embeddings"
        path = root / sample_id / (_EMB_FILES | _MASK_FILES)[kind]
        t = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
        return t.detach()

    sample_ids = samples[SAMPLE_ID_COL].tolist()
    if not sample_ids:
        raise ValueError("Expected samples to load")
    n = len(sample_ids)
    # The prompt tokens are always read: the pooled vectors are their masked
    # mean, so nothing else supplies the "pooled" text path.
    kinds = ["img", "src_tokens", "tar_tokens", "src_mask", "tar_mask"]
    if USE_IMG_MASK:
        kinds += list(_MASK_FILES)

    # Probe shapes from the first sample, then preallocate the tables so peak
    # memory stays ~1x table size (no row lists + torch.stack copy).
    img_probe = _load_scattered_cache(sample_ids[0], "img")
    if img_probe.ndim != 3 or img_probe.shape[-1] != img_probe.shape[-2]:
        raise ValueError(f"Expected (C, S, S) latent tokens, got {tuple(img_probe.shape)}")
    img_shape = tuple(img_probe.shape)
    tok_probe = _load_scattered_cache(sample_ids[0], "src_tokens")
    if tok_probe.ndim != 2:
        raise ValueError(f"Expected (T, D) prompt tokens, got {tuple(tok_probe.shape)}")
    t_len, t_dim = tok_probe.shape

    tables: dict[str, torch.Tensor] = {
        "img": torch.empty((n, *img_shape), dtype=torch.float32),
        "src": torch.empty((n, 1, t_dim), dtype=torch.float32),
        "tar": torch.empty((n, 1, t_dim), dtype=torch.float32),
        "src_tokens": torch.empty((n, t_len, t_dim), dtype=torch.float16),
        "tar_tokens": torch.empty((n, t_len, t_dim), dtype=torch.float16),
        "src_mask": torch.empty((n, t_len), dtype=torch.bool),
        "tar_mask": torch.empty((n, t_len), dtype=torch.bool),
    }

    if USE_IMG_MASK:
        for kind in ("masked_proj", "img_proj", "src_proj", "tar_proj"):
            tables[kind] = torch.empty((n, CLIP_PROJ_DIM), dtype=torch.float32)
        tables["mask_area"] = torch.empty((n, 1), dtype=torch.float32)

    def _load_row(i: int) -> tuple[int, dict[str, torch.Tensor]]:
        return i, {kind: _load_scattered_cache(sample_ids[i], kind) for kind in kinds}

    def _pool_prompt(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Masked mean over the prompt's real tokens, which is what source.pt held."""
        keep = mask.bool().reshape(-1, 1)
        return (tokens.float() * keep).sum(dim=0) / keep.sum().clamp(min=1)

    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, row in tqdm(pool.map(_load_row, range(n)), total=n, desc="Loading scattered embeddings", unit="sample"):
            if tuple(row["img"].shape) != img_shape:
                raise ValueError(f"Expected {img_shape} image tokens for sample {sample_ids[i]}, got {tuple(row['img'].shape)}")
            tables["img"][i] = row["img"].float()
            for kind in ("src_tokens", "tar_tokens"):
                if tuple(row[kind].shape) != (t_len, t_dim):
                    raise ValueError(f"Expected {(t_len, t_dim)} for {kind} of {sample_ids[i]}, got {tuple(row[kind].shape)}")
                tables[kind][i] = row[kind].half()
            tables["src_mask"][i] = row["src_mask"].bool().reshape(-1)
            tables["tar_mask"][i] = row["tar_mask"].bool().reshape(-1)
            tables["src"][i] = _pool_prompt(row["src_tokens"], row["src_mask"])
            tables["tar"][i] = _pool_prompt(row["tar_tokens"], row["tar_mask"])
            if USE_IMG_MASK:
                for kind in ("masked_proj", "img_proj", "src_proj", "tar_proj"):
                    tables[kind][i] = row[kind].float().reshape(-1)
                tables["mask_area"][i] = row["mask_area"].float().reshape(1)

    return tables


"""
Packed caches.
"""

def get_packed_embeddings(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
    dir_name: str | None = None,
) -> dict[str, torch.Tensor]:
    """CPU tables for samples via the packed caches, repacking on any miss."""

    sample_ids = samples[SAMPLE_ID_COL].tolist()
    slug = (dir_name if dir_name is not None else DIR_NAME).replace("_", "").lower()
    t_delta = f"{TARGET_T_DELTA}".replace(".", "p")
    main_path = _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-{t_delta}-{slug}.pt"
    token_path = _PACKED_EMBEDDINGS_DIR / f"{CHORD_EDIT_MODEL}-texttokens-{slug}.pt"
    main_meta = {
        "model": CHORD_EDIT_MODEL,
        "pipeline_type": CHORD_EDIT_PIPELINE_TYPE,
        "layout": "vae_tokens_src_tar_pooled_v3",
        "image_size": int(CHORD_EDIT_IMAGE_SIZE),
        "dir_name": dir_name if dir_name is not None else DIR_NAME,
        "target_t_delta": TARGET_T_DELTA,
    }
    token_meta = {
        "model": CHORD_EDIT_MODEL,
        "layout": "text_tokens_masks_v1",
        "dir_name": dir_name if dir_name is not None else DIR_NAME,
    }

    def _load_packed_cache(path: Path, expected_meta: dict, keys: tuple[str, ...]) -> dict[str, torch.Tensor] | None:
        """Cached tables sliced to sample_ids, or None (missing / bad meta / coverage)."""
        if not path.exists():
            return None
        try:
            data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except Exception as e:
            print(f"Failed to load {path} ({e}). Repacking...")
            return None
        meta = data.get("meta")
        if not isinstance(meta, dict) or any(meta.get(k) != v for k, v in expected_meta.items()):
            print(f"{path} meta unusable for this request. Repacking...")
            return None
        id_to_i = {sid: i for i, sid in enumerate(data["sids"])}
        n_missing = sum(sid not in id_to_i for sid in sample_ids)
        if n_missing:
            print(f"{path} missing {n_missing} of {len(sample_ids)} requested samples. Repacking...")
            return None
        idxs = [id_to_i[sid] for sid in sample_ids]
        return {k: data[k][idxs].contiguous() for k in keys}

    def _save_packed_cache(path: Path, meta: dict, tables: dict[str, torch.Tensor]) -> None:
        """Atomically write the packed table dict with its meta."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        torch.save({"sids": list(sample_ids), "meta": meta, **{k: v.contiguous().clone() for k, v in tables.items()}}, tmp)
        tmp.replace(path)
        print(f"Saved packed embeddings: {path}")

    use_tokens = TEXT_EMB_TYPE == "tokens"
    main_keys = ("img", "src", "tar") + (tuple(_MASK_FILES) if USE_IMG_MASK else ())
    main_meta |= {"masked": bool(USE_IMG_MASK)}
    main = _load_packed_cache(main_path, main_meta, main_keys)
    token = _load_packed_cache(token_path, token_meta, ("src_tokens", "tar_tokens", "src_mask", "tar_mask")) if use_tokens else {}
    if main is None or token is None:
        scattered = get_scattered_embeddings(samples, scattered_dir=scattered_dir)
        main = {k: scattered[k] for k in main_keys}
        _save_packed_cache(main_path, main_meta | {
            "source": "scattered",
            "img_shape": tuple(main["img"].shape[1:]),
            "text_shape": tuple(main["src"].shape[1:]),
        }, main)
        if use_tokens:
            token = {k: scattered[k] for k in ("src_tokens", "tar_tokens", "src_mask", "tar_mask")}
            _save_packed_cache(token_path, token_meta | {"text_shape": tuple(token["src_tokens"].shape[1:])}, token)
    else:
        print("Loaded packed embeddings from cache.")
        
    return main | token


def get_embeddings(
    samples: pd.DataFrame,
    device: torch.device | str,
) -> EmbeddingsTable:
    """Pack scattered caches into one device-resident table keyed by sample_id."""

    samples = samples.drop_duplicates(SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL).reset_index(drop=True)
    if not len(samples):
        raise ValueError("Expected samples with embeddings")

    if PIE_BENCH:
        if TEXT_EMB_TYPE == "tokens":
            raise NotImplementedError("TEXT_EMB_TYPE='tokens' has no PIE-Bench path yet")
        sids = samples[SAMPLE_ID_COL].astype(str)
        is_pie = sids.str.startswith(PIE_SAMPLE_ID_PREFIX)
        parts, sample_ids = [], []
        ue_samples = samples.loc[~is_pie]
        if len(ue_samples):
            parts.append(get_packed_embeddings(ue_samples))
            sample_ids += ue_samples[SAMPLE_ID_COL].tolist()
        pie_samples = samples.loc[is_pie].copy()
        if len(pie_samples):
            # Disk folders use unprefixed ids; restore the prefix for table keys.
            pie_samples[SAMPLE_ID_COL] = pie_samples[SAMPLE_ID_COL].str.removeprefix(PIE_SAMPLE_ID_PREFIX)
            parts.append(get_packed_embeddings(
                pie_samples, scattered_dir=PIE_SCATTERED_DIR, dir_name=PIE_BENCH_DIR_NAME,
            ))
            sample_ids += (PIE_SAMPLE_ID_PREFIX + pie_samples[SAMPLE_ID_COL]).tolist()
        tables = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    else:
        tables = get_packed_embeddings(samples)
        sample_ids = samples[SAMPLE_ID_COL].tolist()

    use_tokens = TEXT_EMB_TYPE == "tokens"
    source_tokens = tables["src_tokens"].float() if use_tokens else tables["src"]
    target_tokens = tables["tar_tokens"].float() if use_tokens else tables["tar"]

    # Get the image tokens.
    if IMG_EMB_TYPE == "vae":
        image_tokens = tables["img"]
    elif IMG_EMB_TYPE == "clip":
        image_tokens = get_clip_tokens(samples)
    elif IMG_EMB_TYPE == "vae_clip":
        clip = get_clip_tokens(samples)
        image_tokens = _concat_vae_clip_tokens(tables["img"], clip)

    if use_tokens:
        source_mask = tables["src_mask"].to(device)
        target_mask = tables["tar_mask"].to(device)
    else:
        n, n_t = source_tokens.shape[:2]
        source_mask = torch.ones(n, n_t, dtype=torch.bool, device=device)
        target_mask = torch.ones(n, n_t, dtype=torch.bool, device=device)

    mask_features = (
        get_img_mask_features(tables).to(device) if USE_IMG_MASK
        else torch.ones(len(sample_ids), FEATURE_DIM, device=device)
    )

    return EmbeddingsTable(
        _sample_ids=tuple(sample_ids),
        _sid_to_idx={sid: i for i, sid in enumerate(sample_ids)},
        image_tokens=image_tokens.to(device),
        source_tokens=source_tokens.to(device),
        target_tokens=target_tokens.to(device),
        source_mask=source_mask,
        target_mask=target_mask,
        mask_features=mask_features,
    )
