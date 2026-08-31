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
# forking it means both packages read and write the same file.
_MLP_DIR = Path(__file__).resolve().parents[1] / "mlp_predictor"


"""
Embeddings classes.
"""


@dataclass(frozen=True)
class SampleEmbeddings:
    """One sample's visual and textual embeddings."""

    image_tokens: torch.Tensor          # (C, S, S) vae; (1, D_clip) clip; (N_v+1, D) vae+clip
    source_tokens: torch.Tensor         # (T, D_txt) tokens, or (1, D_txt) pooled
    target_tokens: torch.Tensor         # like source_tokens
    source_mask: torch.Tensor | None    # (T,) bool; None when pooled
    target_mask: torch.Tensor | None    # like source_mask


@dataclass(frozen=True)
class EmbeddingsTable:
    """Group of samples' visual and textual embeddings, resident on one device."""

    _sample_ids: tuple[str, ...]
    _sid_to_idx: dict[str, int]

    image_tokens: torch.Tensor          # (N, C, S, S) vae; (N, 1, D_clip) clip; (N, N_v+1, D) vae+clip
    source_tokens: torch.Tensor         # (N, T, D_txt) tokens, or (N, 1, D_txt) pooled
    target_tokens: torch.Tensor         # like source_tokens
    source_mask: torch.Tensor | None    # (N, T) bool; None when pooled
    target_mask: torch.Tensor | None    # like source_mask

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
            source_mask=None if self.source_mask is None else self.source_mask[i],
            target_mask=None if self.target_mask is None else self.target_mask[i],
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


def _get_clip_image_tokens(samples: pd.DataFrame, device: torch.device | str) -> torch.Tensor:
    """(n, 1, D_clip) pooled CLIP-L/14 image embeddings, one row per sample.

    Rows follow the samples frame's order: UltraEdit samples first, then
    pie_*-prefixed ones, matching get_embeddings' table order. UltraEdit rows
    use the frame's own image paths and mlp_predictor's shared DIR_NAME-keyed
    cache; PIE rows are encoded from PIE_INPUTS_CSV's image paths into this
    package's own cache. Both encode on a miss, so warm the caches before
    parallel sweeps.
    """
    if str(_MLP_DIR) not in sys.path:
        sys.path.append(str(_MLP_DIR))
    import clip_image

    # clip_image puts its own package dir at sys.path[0] on import. Drop it
    # once the module is bound, so nothing imported later resolves to
    # mlp_predictor's copy of a module this package also has.
    while str(_MLP_DIR) in sys.path:
        sys.path.remove(str(_MLP_DIR))

    def _get_pie_clip_image_embeddings(pie_ids: list[str]) -> torch.Tensor:
        """(n, D_clip) pooled CLIP embeddings for pie_*-prefixed sample ids.

        PIE-Bench reuses UltraEdit's sample_id range (both count up from
        00000000), so a bare id can never distinguish the two pools: this
        cache keeps the pie_ prefix in its stored sids, and the prefix is
        stripped only to look up image paths in PIE_INPUTS_CSV, whose rows use
        the unprefixed on-disk ids. mlp_predictor's clip_image cache is keyed
        by DIR_NAME, which names the UltraEdit pool, so the PIE table is
        cached here under this package's cache dir instead; encoding still
        goes through clip_image's own encoder, so the pooling stays identical
        to the UltraEdit table's.
        """
        path = _PACKED_EMBEDDINGS_DIR / "clipL14-pooled-piebenchv1.pt"
        meta = {
            "clip_model": clip_image.CLIP_MODEL_NAME,
            "layout": "vision_hidden_meanpool_v1",
            "dir_name": PIE_BENCH_DIR_NAME,
            "dim": clip_image.CLIP_IMG_DIM,
        }

        if path.exists():
            try:
                data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                cached_meta = data.get("meta")
                ok = isinstance(cached_meta, dict) and all(cached_meta.get(k) == v for k, v in meta.items())
                id_to_i = {sid: i for i, sid in enumerate(data["sids"])} if ok else {}
                if ok and not any(sid not in id_to_i for sid in pie_ids):
                    print("Loaded PIE CLIP image embeddings from cache.")
                    return data["emb"][[id_to_i[sid] for sid in pie_ids]].contiguous()
                print(f"{path} unusable for this request. Recomputing...")
            except Exception as e:
                print(f"Failed to load {path} ({e}). Recomputing...")

        raw_ids = [sid.removeprefix(PIE_SAMPLE_ID_PREFIX) for sid in pie_ids]
        inputs_df = pd.read_csv(PIE_INPUTS_CSV, dtype={SAMPLE_ID_COL: str})
        inputs_df[SAMPLE_ID_COL] = inputs_df[SAMPLE_ID_COL].str.zfill(8)
        inputs_df = inputs_df.drop_duplicates(SAMPLE_ID_COL).set_index(SAMPLE_ID_COL, drop=False)
        missing = [sid for sid in raw_ids if sid not in inputs_df.index]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValueError(f"{PIE_INPUTS_CSV} missing {len(missing)} sample_id(s): {preview}")
        emb = clip_image._compute(inputs_df.loc[raw_ids].reset_index(drop=True), device)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        torch.save({"sids": list(pie_ids), "emb": emb.contiguous().clone(), "meta": meta}, tmp)
        tmp.replace(path)
        print(f"Saved PIE CLIP image embeddings: {path}")
        return emb

    sids = samples[SAMPLE_ID_COL].astype(str)
    is_pie = sids.str.startswith(PIE_SAMPLE_ID_PREFIX)
    parts = []
    ue_samples = samples.loc[~is_pie]
    if len(ue_samples):
        parts.append(clip_image.get_clip_image_embeddings(ue_samples, device=device))
    if is_pie.any():
        parts.append(_get_pie_clip_image_embeddings(sids.loc[is_pie].tolist()))
    emb = torch.cat(parts, dim=0)
    return emb.unsqueeze(-2).contiguous().float().cpu()


def get_scattered_embeddings(
    samples: pd.DataFrame,
    *,
    scattered_dir: Path | None = None,
) -> dict[str, torch.Tensor]:
    """Load per-sample .pt files from the scattered tree into CPU tables."""

    def _load_scattered_cache(sample_id: str, kind: str) -> torch.Tensor:
        """Load one scattered file to a CPU tensor."""
        root = (scattered_dir if scattered_dir is not None else SCATTERED_DIR) / "annotation_embeddings"
        path = root / sample_id / _EMB_FILES[kind]
        t = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"Expected Tensor in {path}, got {type(t)}")
        return t.detach()

    sample_ids = samples[SAMPLE_ID_COL].tolist()
    if not sample_ids:
        raise ValueError("Expected samples to load")
    n = len(sample_ids)
    use_tokens = TEXT_EMB_SOURCE == "tokens"
    kinds = ["img", "src", "tar"] + (["src_tokens", "tar_tokens", "src_mask", "tar_mask"] if use_tokens else [])

    # Probe shapes from the first sample, then preallocate the tables so peak
    # memory stays ~1x table size (no row lists + torch.stack copy).
    img_probe = _load_scattered_cache(sample_ids[0], "img")
    if img_probe.ndim != 3 or img_probe.shape[-1] != img_probe.shape[-2]:
        raise ValueError(f"Expected (C, S, S) latent tokens, got {tuple(img_probe.shape)}")
    img_shape = tuple(img_probe.shape)
    src_probe = _load_scattered_cache(sample_ids[0], "src")
    if src_probe.numel() != src_probe.shape[-1]:
        raise ValueError(f"Expected a pooled text vector (D,) or (1, D), got {tuple(src_probe.shape)}")
    text_shape = (1, int(src_probe.shape[-1]))
    tables: dict[str, torch.Tensor] = {
        "img": torch.empty((n, *img_shape), dtype=torch.float32),
        "src": torch.empty((n, *text_shape), dtype=torch.float32),
        "tar": torch.empty((n, *text_shape), dtype=torch.float32),
    }
    if use_tokens:
        tok_probe = _load_scattered_cache(sample_ids[0], "src_tokens")
        if tok_probe.ndim != 2:
            raise ValueError(f"Expected (T, D) prompt tokens, got {tuple(tok_probe.shape)}")
        t_len, t_dim = tok_probe.shape
        tables |= {
            "src_tokens": torch.empty((n, t_len, t_dim), dtype=torch.float16),
            "tar_tokens": torch.empty((n, t_len, t_dim), dtype=torch.float16),
            "src_mask": torch.empty((n, t_len), dtype=torch.bool),
            "tar_mask": torch.empty((n, t_len), dtype=torch.bool),
        }

    def _load_row(i: int) -> tuple[int, dict[str, torch.Tensor]]:
        return i, {kind: _load_scattered_cache(sample_ids[i], kind) for kind in kinds}

    with ThreadPoolExecutor(max_workers=32) as pool:
        for i, row in tqdm(pool.map(_load_row, range(n)), total=n, desc="Loading scattered embeddings", unit="sample"):
            if tuple(row["img"].shape) != img_shape:
                raise ValueError(f"Expected {img_shape} image tokens for sample {sample_ids[i]}, got {tuple(row['img'].shape)}")
            tables["img"][i] = row["img"].float()
            # Text rows may be saved as (D,) or (1, D) but must be loaded as (1, D).
            tables["src"][i] = row["src"].float().reshape(1, -1)
            tables["tar"][i] = row["tar"].float().reshape(1, -1)
            if use_tokens:
                for kind in ("src_tokens", "tar_tokens"):
                    if tuple(row[kind].shape) != (t_len, t_dim):
                        raise ValueError(f"Expected {(t_len, t_dim)} for {kind} of {sample_ids[i]}, got {tuple(row[kind].shape)}")
                    tables[kind][i] = row[kind].half()
                tables["src_mask"][i] = row["src_mask"].bool().reshape(-1)
                tables["tar_mask"][i] = row["tar_mask"].bool().reshape(-1)

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
        "layout": "img_tokens_src_tar_pooled_v2",
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

    use_tokens = TEXT_EMB_SOURCE == "tokens"
    main = _load_packed_cache(main_path, main_meta, ("img", "src", "tar"))
    token = _load_packed_cache(token_path, token_meta, ("src_tokens", "tar_tokens", "src_mask", "tar_mask")) if use_tokens else {}
    if main is None or token is None:
        scattered = get_scattered_embeddings(samples, scattered_dir=scattered_dir)
        main = {k: scattered[k] for k in ("img", "src", "tar")}
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
        if TEXT_EMB_SOURCE == "tokens":
            raise NotImplementedError("TEXT_EMB_SOURCE='tokens' has no PIE-Bench path yet")
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

    use_tokens = TEXT_EMB_SOURCE == "tokens"
    source_tokens = tables["src_tokens"].float() if use_tokens else tables["src"]
    target_tokens = tables["tar_tokens"].float() if use_tokens else tables["tar"]

    image_tokens = tables["img"]
    if IMG_EMB_SOURCE != "vae":
        clip = _get_clip_image_tokens(samples, device)
        image_tokens = clip if IMG_EMB_SOURCE == "clip" else _concat_vae_clip_tokens(image_tokens, clip)

    return EmbeddingsTable(
        _sample_ids=tuple(sample_ids),
        _sid_to_idx={sid: i for i, sid in enumerate(sample_ids)},
        image_tokens=image_tokens.to(device),
        source_tokens=source_tokens.to(device),
        target_tokens=target_tokens.to(device),
        source_mask=tables["src_mask"].to(device) if use_tokens else None,
        target_mask=tables["tar_mask"].to(device) if use_tokens else None,
    )
