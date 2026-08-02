"""
Embedding caches: load packed tables, pack scattered files, or encode fresh.

Scattered: many per-sample .pt files indexed by EMBEDDINGS_CSV (slow to load).
Packed: one stacked table at .cache/packed_embeddings/<CHORD_EDIT_MODEL>-<t_delta>-<dir_slug>.pt
(fast to load), tagged with a "meta" dict recording the model type, text pooling,
and pack provenance. A cache whose meta is missing or does not match the current
settings is treated as a miss and repacked. Text extraction from scattered files
is dispatched on CHORD_EDIT_PIPELINE_TYPE (see make_text_extractor) so packed
caches always correspond to the CHORD_EDIT_MODEL.

This module depends only on settings.py so it can be reused by other model
designs. The predictor argument is duck-typed: encoding needs callables
predictor.image_encoder(images) and predictor.text_encoder(prompts) that return
(N, dim) tensors; get_embeddings_by_sample's default device additionally reads
predictor.regressor.target_mean.device.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import *

_PACKED_EMBEDDINGS_DIR = Path(__file__).resolve().parent / ".cache" / "packed_embeddings"
_SCATTERED_EMBEDDINGS_DIR = Path(f"/shared/ssd_30T/mirick/embeddings/{CHORD_EDIT_MODEL}/{DIR_NAME}/annotation_embeddings/")


def _prep_sample_id(value) -> str:
    return f"{int(value):08d}"


def _prep_embedding_path(embedding_path: str) -> str:
    path = Path(embedding_path)
    if path.is_absolute():
        return str(path)
    path = Path(str(embedding_path).lstrip("/"))
    if path.parts and path.parts[0] == EMBEDDINGS_SAMPLES_DIRNAME:
        path = path.relative_to(EMBEDDINGS_SAMPLES_DIRNAME)
    return str(_SCATTERED_EMBEDDINGS_DIR / path)


"""
Per-model text extraction for packing scattered embeddings.

Scattered text files store whatever the annotation pipeline emitted; collapsing
them to one vector per prompt must match model_m.encode_text_pooled for the
current CHORD_EDIT_MODEL so packed caches and on-the-fly encoding agree per
model type: sd stores full last_hidden_state sequences (pooled here with
mean_pool), sdxl must store text_encoder_2's pooled embeds directly
(text_embeds is a projection of the EOS token and cannot be reconstructed from
hidden-state sequences), flux is not implemented.
"""

def _get_text_pooling() -> str:
    """Text pooling name for CHORD_EDIT_PIPELINE_TYPE."""
    if CHORD_EDIT_PIPELINE_TYPE == "sd":
        return "masked_mean"
    if CHORD_EDIT_PIPELINE_TYPE == "sdxl":
        return "pooled_embeds"
    if CHORD_EDIT_PIPELINE_TYPE == "flux":
        raise NotImplementedError()
    raise ValueError(f"Unsupported CHORD_EDIT_PIPELINE_TYPE={CHORD_EDIT_PIPELINE_TYPE!r}")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mask-weighted mean over the token dimension."""
    mask = attention_mask.unsqueeze(-1).expand_as(last_hidden).float()
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


class SdMaskedMeanTextExtractor:
    """SD branch: mask-weighted mean over the stored last_hidden_state sequence.

    Attention masks come from re-tokenizing the prompts (tokenizer only, no
    encoder weights), since the scattered files do not store them. Reuses
    mean_pool so this cannot drift from model_m.encode_text_pooled.
    """

    name = "masked_mean"

    def __init__(self, src_prompts: list[str], tar_prompts: list[str]):
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(str(CHORD_EDIT_MODEL_ROOT / "tokenizer"))
        self.seq_len = int(tokenizer.model_max_length)

        def _attn(prompts: list[str]) -> torch.Tensor:
            enc = tokenizer(
                list(prompts),
                padding="max_length",
                truncation=True,
                max_length=self.seq_len,
                return_tensors="pt",
            )
            return enc.attention_mask

        self._attn = {"src": _attn(src_prompts), "tar": _attn(tar_prompts)}

    def text_dim(self, probe: torch.Tensor, path) -> int:
        """Validate a stored text tensor's shape and return the hidden dim."""
        if probe.ndim != 3 or probe.shape[0] != 1:
            raise ValueError(f"Expected text sequence (1, T, D), got {tuple(probe.shape)} in {path}")
        if int(probe.shape[1]) != self.seq_len:
            raise ValueError(
                f"Stored sequence length {int(probe.shape[1])} != tokenizer max length "
                f"{self.seq_len} in {path}; scattered embeddings do not match this model's tokenizer"
            )
        return int(probe.shape[2])

    def __call__(self, t: torch.Tensor, i: int, kind: str, dim: int) -> torch.Tensor:
        """Collapse sample i's stored sequence for kind in {'src', 'tar'} to (dim,)."""
        hidden = t.reshape(1, self.seq_len, dim)
        return mean_pool(hidden, self._attn[kind][i].unsqueeze(0))[0]


class SdxlPooledTextExtractor:
    """SDXL branch: scattered files must already store text_encoder_2's pooled
    embeds (1, D); pooling cannot be redone from sequences without weights."""

    name = "pooled_embeds"

    def __init__(self, src_prompts: list[str], tar_prompts: list[str]):
        pass

    def text_dim(self, probe: torch.Tensor, path) -> int:
        if probe.numel() != probe.shape[-1]:
            raise ValueError(
                f"Expected pooled text embeds (1, D) or (D,), got {tuple(probe.shape)} in {path}. "
                f"SDXL pooled embeds (text_encoder_2 text_embeds) cannot be reconstructed from "
                f"hidden-state sequences; re-run the annotation pipeline storing pooled embeds."
            )
        return int(probe.shape[-1])

    def __call__(self, t: torch.Tensor, i: int, kind: str, dim: int) -> torch.Tensor:
        if t.numel() != dim:
            raise ValueError(f"Pooled text embeds numel {t.numel()} != {dim}")
        return t.reshape(-1)


def make_text_extractor(src_prompts: list[str], tar_prompts: list[str]):
    """Text extractor matching CHORD_EDIT_PIPELINE_TYPE (see expected_text_pooling)."""
    pooling = _get_text_pooling()
    extractor_cls = {
        SdMaskedMeanTextExtractor.name: SdMaskedMeanTextExtractor,
        SdxlPooledTextExtractor.name: SdxlPooledTextExtractor,
    }[pooling]
    return extractor_cls(src_prompts, tar_prompts)


"""
Packed/scattered caches and encoding.
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
    predictor,
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
        print(f"Caching scattered embeddings ({n_samples} samples) -> {_SCATTERED_EMBEDDINGS_DIR}")
        rows: list[dict[str, str]] = []
        for i, sid in enumerate(tqdm(sample_ids, desc="Caching scattered", unit="sample")):
            sample_dir = _SCATTERED_EMBEDDINGS_DIR / sid
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
    predictor,
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

    if getattr(predictor, "image_encoder", None) is None:
        raise RuntimeError(
            f"Embeddings unavailable: {packed_path} cannot cover {len(sample_ids)} requested "
            f"samples and scattered embeddings ({EMBEDDINGS_CSV}) are missing or incomplete. "
            f"No encoder-bearing predictor was provided (predictor=None or encoders released), "
            f"so on-the-fly encoding is not possible. Produce scattered embeddings with the "
            f"annotation pipeline, or pass a predictor with live encoders (e.g. SurrogateModel)."
        )
    return _encode_embeddings(samples, predictor, packed_path, batch_size=batch_size)


def get_embeddings_by_sample(
    df: pd.DataFrame,
    predictor,
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
