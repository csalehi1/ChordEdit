"""
pipeline.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _helpers import SampleRecord

if TYPE_CHECKING:
    from PIL import Image
    from pipeline_chord import ChordEditPipeline, _PromptCondition

    _Embed = Union[torch.Tensor, _PromptCondition]


# Singleton pipeline instance for this process (one per GPU shard).
_pipeline: ChordEditPipeline | None = None


def bind_pipeline(pipeline: ChordEditPipeline) -> None:
    """Register the loaded ChordEditPipeline for this process (one per GPU shard)."""
    global _pipeline
    _pipeline = pipeline


def _get_pipeline() -> ChordEditPipeline:
    if _pipeline is None:
        raise RuntimeError("bind_pipeline() must be called before run_factorized_grid")
    return _pipeline


def _u_estimate_delta0(x_anchor, src_embed, edit_embed, noise, t_s: float):
    """
    Exact t_delta==0 shortcut of the 4-forward default estimator.

    The blend is (δ·dv_s + t_start·dv_s0) / (t_start + δ). When δ=0 the first
    term is multiplied by zero, both noise levels coincide so dv_s0 == dv_s, and
    the estimate collapses to dv_s — two UNet forwards instead of four.

    Based on paper's pipeline_chord.py:u_estimate_default.
    """
    pipeline = _get_pipeline()
    batch, device = x_anchor.shape[0], x_anchor.device
    t_idx_s = pipeline._time_to_index(batch, t_s, device=device)

    noises = noise if isinstance(noise, (list, tuple)) else [noise]
    num_noises = len(noises)
    noise_stack = torch.stack(noises, dim=0)

    alpha_s, sigma_s = pipeline._get_alpha_sigma(x_anchor, t_idx_s)
    x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
    alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
    sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)

    # Renoirse anchor once at t_start; src and edit share this latent.
    z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack
    samples = z_s.repeat(1, 2, 1, 1, 1).reshape(num_noises * 2 * batch, *x_anchor.shape[1:])
    conds = pipeline._repeat_condition(pipeline._cat_conditions([src_embed, edit_embed]), num_noises)
    timesteps = t_idx_s.repeat(2 * num_noises)

    noise_pred = pipeline._predict_noise(sample=samples, timesteps=timesteps, cond=conds)
    noise_pred = noise_pred.reshape(num_noises, 2, batch, *x_anchor.shape[1:])
    x0_all = (z_s.unsqueeze(1) - sigma_s_b.unsqueeze(1) * noise_pred) / alpha_s_b.unsqueeze(1)
    x_src_p_s, x_tar_p_s = x0_all.unbind(dim=1)
    return (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)


def _u_estimate(x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float):
    """Interrupt paper's pipeline_chord.py:_u_estimate with a delta==0 fast path."""
    pipeline = _get_pipeline()
    if delta == 0.0:
        return _u_estimate_delta0(x_anchor, src_embed, edit_embed, noise, t_s)
    return pipeline._u_estimate(x_anchor, src_embed, edit_embed, noise, t_s, delta)


def _cleanup_decode_row(x_transport, edit_embed, noise, t_end_values, cleanup: bool):
    """
    Batched cleanup + per-cell VAE decode for one t_start row.

    Cleanup (_pred_x0) is one UNet forward across all t_end; VAE decode stays
    batch-1 (fastest for SD VAE). Assumes a single source image.
    """
    pipeline = _get_pipeline()
    n = len(t_end_values)
    batch = x_transport.shape[0]
    device = x_transport.device
    x_rep = x_transport.repeat(n, *([1] * (x_transport.dim() - 1)))

    if cleanup:
        timesteps = torch.cat(
            [pipeline._time_to_index(batch, float(t_end), device=device) for t_end in t_end_values],
            dim=0,
        )
        cond = pipeline._repeat_condition(edit_embed, n)
        noise_rep = noise.repeat(n, *([1] * (noise.dim() - 1)))
        x0 = pipeline._pred_x0(x_rep, timesteps, cond, noise_rep)
    else:
        x0 = x_rep

    decoded = torch.cat(
        [pipeline._decode_latent_to_image(x0[i * batch : (i + 1) * batch]) for i in range(n)],
        dim=0,
    )
    decoded, _ = pipeline._apply_safety_checker(decoded)
    pil_images = pipeline._tensor_to_pil(decoded)
    return [pil_images[i * batch] for i in range(n)]


def _flat_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Flatten to a compact 1-D float32 CPU tensor safe for torch.save.

    torch.save serializes the tensor's entire underlying storage; views that
    slice a batch share that storage, so saving a view would embed every
    sample's data in each file. reshape(-1) + clone guarantees a private
    buffer holding exactly numel elements.
    """
    return tensor.detach().reshape(-1).float().cpu().clone()


def _token_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Like _flat_cpu but keeps the token shape (no reshape(-1)).

    The clone is still mandatory: batch slices are views over the whole
    batch storage, and torch.save would serialize all of it.
    """
    return tensor.detach().float().cpu().clone()


def _pool_text_embeds(hidden: Any, prompts: List[str]) -> torch.Tensor:
    """Masked-mean pool text-encoder hidden states to one vector per prompt, (N, dim).

    Matches the classification repo's mean_pool / encode_text_pooled exactly:
    attention masks come from re-tokenizing the prompts (the pipeline is built
    with use_attention_mask=False, so encode_prompt retains no mask), then a
    mask-weighted mean over the token dimension.

    SD hidden states arrive as a raw tensor; SDXL and FLUX arrive as a
    _PromptCondition whose hidden_states carry the sequence axis. The mask must
    come from the tokenizer that produced that axis: pipeline.tokenizer for SD
    and SDXL (both SDXL tokenizers emit identical masks for the same prompt),
    or pipeline.mask_tokenizer when set (FLUX: the T5 tokenizer). max_length is
    taken from the hidden states themselves — identical to
    tokenizer.model_max_length for CLIP (77), and the pipeline's
    max_sequence_length for T5.
    """
    if not torch.is_tensor(hidden):
        hidden = hidden.hidden_states
    pipeline = _get_pipeline()
    tokenizer = getattr(pipeline, "mask_tokenizer", None) or pipeline.tokenizer
    inputs = tokenizer(
        list(prompts),
        padding="max_length",
        truncation=True,
        max_length=hidden.shape[1],
        return_tensors="pt",
    )
    mask = inputs.attention_mask.to(hidden.device).unsqueeze(-1).expand_as(hidden).float()
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def _split_prompt_batch(embeds: Any, index: int) -> Any:
    """Split one row out of a batched encode_prompt result (enables a single text forward)."""
    if torch.is_tensor(embeds):
        return embeds[index : index + 1]
    from pipeline_chord import _PromptCondition

    return _PromptCondition(
        hidden_states=embeds.hidden_states[index : index + 1],
        pooled_embeds=(None if embeds.pooled_embeds is None else embeds.pooled_embeds[index : index + 1]),
        time_ids=None if embeds.time_ids is None else embeds.time_ids[index : index + 1],
    )


def _save_sample_embeddings(
    embeddings_dir: Path,
    *,
    src_cpu: torch.Tensor,
    tgt_cpu: torch.Tensor,
    image_tokens_cpu: torch.Tensor,
    image_cpu: torch.Tensor | None = None,
    src_tokens_cpu: torch.Tensor | None = None,
    tgt_tokens_cpu: torch.Tensor | None = None,
    mask_cpu: torch.Tensor | None = None,
) -> None:
    """Write per-sample .pt files (safe to call from a background thread).

    Flat/pooled set for the MLP/classification models: image.pt (and mask.pt,
    when --cache-masks provides one) is the flattened VAE latent (C*H*W, e.g.
    16384 for sd-turbo at 512px); source.pt / target.pt are masked-mean-pooled
    text vectors (hidden_dim: 1024 sd-turbo, 2048 sdxl-turbo, 4096 flux).

    Token set for the attention_predictor model: image_tokens.pt is the
    unflattened VAE latent (C, S, S) — (4, 64, 64) for sd/sdxl-turbo,
    (16, 64, 64) for flux at 512px; source_tokens.pt / target_tokens.pt are
    raw last_hidden_state rows (L, D), e.g. (77, 1024), unpooled and with no
    attention mask applied.

    With --minimal-embeddings only source.pt, target.pt, and image_tokens.pt
    are written; the optional tensors arrive as None and are skipped.
    """
    assert src_cpu.ndim == 1 and tgt_cpu.ndim == 1
    assert image_cpu is None or image_cpu.ndim == 1
    assert mask_cpu is None or mask_cpu.ndim == 1
    assert src_tokens_cpu is None or src_tokens_cpu.ndim == 2
    assert tgt_tokens_cpu is None or tgt_tokens_cpu.ndim == 2
    assert image_tokens_cpu.ndim == 3 and image_tokens_cpu.shape[-1] == image_tokens_cpu.shape[-2]
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    torch.save(src_cpu, embeddings_dir / "source.pt")
    torch.save(tgt_cpu, embeddings_dir / "target.pt")
    torch.save(image_tokens_cpu, embeddings_dir / "image_tokens.pt")
    if image_cpu is not None:
        torch.save(image_cpu, embeddings_dir / "image.pt")
    if src_tokens_cpu is not None:
        torch.save(src_tokens_cpu, embeddings_dir / "source_tokens.pt")
    if tgt_tokens_cpu is not None:
        torch.save(tgt_tokens_cpu, embeddings_dir / "target_tokens.pt")
    if mask_cpu is not None:
        torch.save(mask_cpu, embeddings_dir / "mask.pt")


def run_factorized_grid(
    *,
    source_image: Image.Image,
    record: SampleRecord,
    base_config: Dict[str, Any],
    cell_pairs: List[Tuple[float, float]],
    t_delta: float,
    seed: int,
    embeddings_dir: Path | None = None,
    mask_image: Image.Image | None = None,
    skip_generated: bool = False,
    minimal_embeddings: bool = False,
) -> Dict[Tuple[float, float], Image.Image]:
    """Generate the given (t_start, t_end) cells for one source image."""
    pipeline = _get_pipeline()
    save_executor: ThreadPoolExecutor | None = None
    save_future: Future[None] | None = None
    # Ensure that we wait for the background save to complete, even if UNet work fails.
    # Wait for (or surface errors from) that writer, then shut down the executor.
    # This ensures that we don't leak resources if UNet work fails.
    try:
        with torch.no_grad():
            shared_params = pipeline._prepare_edit_params({**base_config, "t_delta": t_delta})

            # Batch image and mask through one VAE forward, avoiding a second encode when saving mask.pt.
            # Batch source and target prompts through one text-encoder forward, avoiding a second CLIP pass.
            # Meanwhile, torch.save runs on a background thread so disk I/O overlaps UNet transport/decode instead of blocking the GPU beforehand.
            image_pixels = pipeline._prepare_image_tensor(source_image)
            if embeddings_dir is not None and mask_image is not None:
                # Perform a single VAE encode with batch=2, then split latents.
                mask_pixels = pipeline._prepare_image_tensor(mask_image)
                latents_batch = pipeline._encode_image_to_latent(
                    torch.cat([image_pixels, mask_pixels], dim=0)
                )
                latents = latents_batch[0:1]
                mask_latents = latents_batch[1:2]
            else:
                latents = pipeline._encode_image_to_latent(image_pixels)
                mask_latents = None

            # Perform a single text-encoder call for both prompts, then split batch dim.
            prompt_batch = pipeline.encode_prompt([record.source_prompt, record.target_prompt])
            src_embed = _split_prompt_batch(prompt_batch, 0)
            tgt_embed = _split_prompt_batch(prompt_batch, 1)

            if embeddings_dir is not None:
                # Saved embeddings derive from the same tensors that condition
                # generation: the UNet's full-sequence hidden states (saved raw
                # as *_tokens.pt and pooled to one vector per prompt), and the
                # VAE latent (saved unflattened as image_tokens.pt and flat).
                # Copy to CPU before save so the background writer does not touch GPU tensors.
                hidden = prompt_batch if torch.is_tensor(prompt_batch) else prompt_batch.hidden_states
                pooled = _pool_text_embeds(hidden, [record.source_prompt, record.target_prompt])
                save_kwargs = dict(
                    embeddings_dir=embeddings_dir,
                    src_cpu=_flat_cpu(pooled[0]),
                    tgt_cpu=_flat_cpu(pooled[1]),
                    image_tokens_cpu=_token_cpu(latents[0]),
                )
                if not minimal_embeddings:
                    save_kwargs.update(
                        image_cpu=_flat_cpu(latents),
                        src_tokens_cpu=_token_cpu(hidden[0]),
                        tgt_tokens_cpu=_token_cpu(hidden[1]),
                        mask_cpu=_flat_cpu(mask_latents) if mask_latents is not None else None,
                    )
                if skip_generated:
                    # Encode-only (--skip-generated): nothing to overlap with, write synchronously.
                    _save_sample_embeddings(**save_kwargs)
                else:
                    # Overlap .pt writes with UNet work below.
                    save_executor = ThreadPoolExecutor(max_workers=1)
                    save_future = save_executor.submit(_save_sample_embeddings, **save_kwargs)

            if skip_generated:
                return {}

            noise_list = pipeline._prepare_noise_list(
                latents=latents,
                seed_value=seed,
                num_noises=shared_params["noise_samples"],
            )

            # Group pairs into rows so transport stays one-per-t_start.
            rows: OrderedDict[float, List[float]] = OrderedDict()
            for t_start, t_end in cell_pairs:
                rows.setdefault(t_start, []).append(t_end)

            # One transport per t_start (the dominant cost).
            transport: Dict[float, torch.Tensor] = {}
            for t_start, row_t_end_values in rows.items():
                params = pipeline._prepare_edit_params({**base_config, "t_start": t_start, "t_delta": t_delta})
                u_hat = _u_estimate(latents, src_embed, tgt_embed, noise_list, params["t_start"], params["t_delta"])
                transport[t_start] = (latents + params["step_scale"] * u_hat).detach()

            # Cleanup/decode one t_start row at a time (batched across t_end).
            results: Dict[Tuple[float, float], "Image.Image"] = {}
            for t_start, row_t_end_values in rows.items():
                row_images = _cleanup_decode_row(
                    transport[t_start],
                    tgt_embed,
                    noise_list[0],
                    row_t_end_values,
                    bool(shared_params["cleanup"]),
                )
                for t_end, image in zip(row_t_end_values, row_images):
                    results[(t_start, t_end)] = image
    finally:
        # Wait for the background save to complete, then shut down the executor.
        if save_future is not None:
            save_future.result()
        if save_executor is not None:
            save_executor.shutdown(wait=False)

    return results
