# clip_image.py

"""CLIP-L/14 image embeddings for the surrogate's image input.

The surrogate's image input has always been a flattened SD VAE latent, which
carries the mask geometry the PSNR label is a statistic of but has no
representation in which the CLIP-Edited cosine is expressible. CLIP-Edited is
scored with openai/clip-vit-large-patch14, so an embedding from that encoder
puts the image in the space the label is measured in.

The pooling mirrors how the cached text embeddings were produced by the
annotation pipeline (daniel_grid_optimizations/_pipeline.py::_pool_text_embeds):
a mask-weighted mean over the token dimension of the encoder's last hidden
states, before any joint projection. Images carry no padding, so the
mask-weighted mean reduces to a plain mean over the patch tokens. The result is
1024-d, matching the text embeddings' width.

Cached per dataset under .cache/clip_image_embeddings/, with a meta dict and a
rebuild on any mismatch, following embeddings.py's convention.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from settings import *

CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# ViT-L/14's vision hidden width, and its text encoder's hidden width. The
# regressor needs these before any data is loaded, so they are constants here
# and checked against the cache on load.
CLIP_IMG_DIM = 1024
CLIP_TXT_DIM = 768

_CLIP_IMAGE_DIR = Path(__file__).resolve().parent / ".cache" / "clip_image_embeddings"


def _cache_path(kind: str = "image") -> Path:
    slug = DIR_NAME.replace("_", "").lower()
    suffix = "" if kind == "image" else f"-{kind}"
    return _CLIP_IMAGE_DIR / f"clipL14-pooled{suffix}-{slug}.pt"


def _expected_meta(kind: str = "image") -> dict:
    return {
        "clip_model": CLIP_MODEL_NAME,
        "layout": f"{'vision' if kind == 'image' else 'text'}_hidden_meanpool_v1",
        "dir_name": DIR_NAME,
        "dim": CLIP_IMG_DIM if kind == "image" else CLIP_TXT_DIM,
    }


def _load_cache(path: Path, sample_ids: list[str], kind: str = "image") -> torch.Tensor | None:
    """Cached embeddings sliced to sample_ids, or None on miss / mismatch / coverage."""
    if not path.exists():
        return None
    try:
        data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except Exception as e:
        print(f"Failed to load {path} ({e}). Recomputing...")
        return None
    meta = data.get("meta")
    if not isinstance(meta, dict):
        print(f"{path} has no meta. Recomputing...")
        return None
    for key, expected in _expected_meta(kind).items():
        if meta.get(key) != expected:
            print(f"{path} meta mismatch: {key}={meta.get(key)!r} expected {expected!r}. Recomputing...")
            return None
    id_to_i = {sid: i for i, sid in enumerate(data["sids"])}
    n_missing = sum(sid not in id_to_i for sid in sample_ids)
    if n_missing:
        print(f"{path} missing {n_missing} of {len(sample_ids)} requested samples. Recomputing...")
        return None
    return data["emb"][[id_to_i[sid] for sid in sample_ids]].contiguous()


@torch.no_grad()
def _compute(samples: pd.DataFrame, device: torch.device | str, batch_size: int = 64) -> torch.Tensor:
    """Mean-pooled CLIP vision hidden states for each row of samples, (n, CLIP_IMG_DIM).

    JPEG decode and resize dominate the wall clock and hold no GIL, so batches
    are prepared on a thread pool that runs ahead of the GPU.
    """
    from concurrent.futures import ThreadPoolExecutor

    from transformers import CLIPImageProcessor, CLIPVisionModel

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = CLIPVisionModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    processor = CLIPImageProcessor.from_pretrained(CLIP_MODEL_NAME, use_fast=True)

    paths = samples[IMAGE_PATH_COL].tolist()
    chunks = [paths[k : k + batch_size] for k in range(0, len(paths), batch_size)]

    def _prepare(chunk: list[str]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in chunk]
        return processor(images=images, return_tensors="pt")["pixel_values"]

    out = torch.empty((len(paths), CLIP_IMG_DIM), dtype=torch.float32)
    row = 0
    with ThreadPoolExecutor(max_workers=16) as pool:
        for pixels in tqdm(pool.map(_prepare, chunks), total=len(chunks), desc="CLIP image embeddings", unit="batch"):
            hidden = model(pixel_values=pixels.to(device)).last_hidden_state  # (B, n_tokens, D)
            pooled = hidden.mean(dim=1)
            if pooled.shape[-1] != CLIP_IMG_DIM:
                raise ValueError(f"Expected {CLIP_IMG_DIM}-d hidden states, got {pooled.shape[-1]}")
            out[row : row + pooled.shape[0]] = pooled.float().cpu()
            row += pooled.shape[0]
    if row != len(paths):
        raise ValueError(f"Filled {row} rows for {len(paths)} samples")
    return out


@torch.no_grad()
def _compute_text(samples: pd.DataFrame, device: torch.device | str, batch_size: int = 256) -> torch.Tensor:
    """Mean-pooled CLIP text hidden states for both prompts, (n, 2 * CLIP_TXT_DIM).

    Pooling matches the image side and the annotation pipeline's text pooling:
    a mask-weighted mean over the token dimension of the last hidden states,
    before the joint projection. Prompts are padded, so unlike images the
    attention mask genuinely matters here.
    """
    from transformers import CLIPTextModel, CLIPTokenizerFast

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model = CLIPTextModel.from_pretrained(CLIP_MODEL_NAME).to(device).eval()
    tokenizer = CLIPTokenizerFast.from_pretrained(CLIP_MODEL_NAME)

    def _pool(prompts: list[str]) -> torch.Tensor:
        inputs = tokenizer(
            prompts, padding="max_length", truncation=True,
            max_length=tokenizer.model_max_length, return_tensors="pt",
        ).to(device)
        hidden = model(**inputs).last_hidden_state  # (B, T, D)
        mask = inputs["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
        return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    src = samples[SOURCE_PROMPT_COL].astype(str).tolist()
    tar = samples[TARGET_PROMPT_COL].astype(str).tolist()
    out = torch.empty((len(src), 2 * CLIP_TXT_DIM), dtype=torch.float32)
    for k in tqdm(range(0, len(src), batch_size), desc="CLIP text embeddings", unit="batch"):
        s = _pool(src[k : k + batch_size])
        t = _pool(tar[k : k + batch_size])
        if s.shape[-1] != CLIP_TXT_DIM:
            raise ValueError(f"Expected {CLIP_TXT_DIM}-d text hidden states, got {s.shape[-1]}")
        out[k : k + s.shape[0]] = torch.cat([s, t], dim=-1).float().cpu()
    return out


def _save_cache(path: Path, sample_ids: list[str], emb: torch.Tensor, kind: str = "image") -> None:
    """Atomically write the embedding table. Clone so no view drags a larger storage along."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(str(path) + ".tmp")
    torch.save({"sids": list(sample_ids), "emb": emb.contiguous().clone(), "meta": _expected_meta(kind)}, tmp_path)
    tmp_path.replace(path)
    print(f"Saved CLIP {kind} embeddings: {path}")


def get_clip_image_embeddings(
    samples: pd.DataFrame,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """CPU table of mean-pooled CLIP image embeddings, one row per sample."""
    sample_ids = samples[SAMPLE_ID_COL].tolist()
    path = _cache_path("image")

    cached = _load_cache(path, sample_ids, "image")
    if cached is not None:
        print("Loaded CLIP image embeddings from cache.")
        return cached

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    emb = _compute(samples, device)
    _save_cache(path, sample_ids, emb, "image")
    return emb


def get_clip_text_embeddings(
    samples: pd.DataFrame,
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """CPU table of mean-pooled CLIP (source, target) prompt embeddings per sample."""
    sample_ids = samples[SAMPLE_ID_COL].tolist()
    path = _cache_path("text")

    cached = _load_cache(path, sample_ids, "text")
    if cached is not None:
        print("Loaded CLIP text embeddings from cache.")
        return cached

    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        device = "cpu"
    emb = _compute_text(samples, device)
    _save_cache(path, sample_ids, emb, "text")
    return emb


def main() -> None:
    """Build the caches for every sample in INPUTS_CSV."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("image", "text", "both"), default="both")
    parser.add_argument("--settings-path", default=None)
    args = parser.parse_args()

    df = pd.read_csv(INPUTS_CSV, dtype={SAMPLE_ID_COL: str})
    df[SAMPLE_ID_COL] = df[SAMPLE_ID_COL].astype(str).str.zfill(8)
    df = df.drop_duplicates(subset=SAMPLE_ID_COL).sort_values(SAMPLE_ID_COL).reset_index(drop=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Building CLIP embeddings for {len(df)} samples ({args.kind}).")
    if args.kind in ("image", "both"):
        print(f"Image table: {tuple(get_clip_image_embeddings(df, device=device).shape)}")
    if args.kind in ("text", "both"):
        print(f"Text table: {tuple(get_clip_text_embeddings(df, device=device).shape)}")


if __name__ == "__main__":
    main()
