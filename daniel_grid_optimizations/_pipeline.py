"""
pipeline.py
"""

from __future__ import annotations

import sys
from collections import OrderedDict
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


def _cpu_tensor(embed: Any) -> torch.Tensor:
    """Detach a prompt embed or latent to CPU for torch.save."""
    if torch.is_tensor(embed):
        return embed.detach().cpu()
    return embed.hidden_states.detach().cpu()


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
    skip_grids: bool = False,
) -> Dict[Tuple[float, float], Image.Image]:
    """Generate the given (t_start, t_end) cells for one source image."""
    pipeline = _get_pipeline()
    with torch.no_grad():
        shared_params = pipeline._prepare_edit_params({**base_config, "t_delta": t_delta})

        # Shared preamble: encode image + prompts + noise once.
        pixel_values = pipeline._prepare_image_tensor(source_image)
        latents = pipeline._encode_image_to_latent(pixel_values)
        src_embed = pipeline.encode_prompt([record.source_prompt])
        tgt_embed = pipeline.encode_prompt([record.target_prompt])
        if embeddings_dir is not None:
            embeddings_dir.mkdir(parents=True, exist_ok=True)
            torch.save(_cpu_tensor(src_embed), embeddings_dir / "source.pt")
            torch.save(_cpu_tensor(tgt_embed), embeddings_dir / "target.pt")
            torch.save(_cpu_tensor(latents), embeddings_dir / "image.pt")
            if mask_image is not None:
                image_tensor = pipeline._prepare_image_tensor(mask_image)
                mask_latents = pipeline._encode_image_to_latent(image_tensor)
                torch.save(_cpu_tensor(mask_latents), embeddings_dir / "mask.pt")
        if skip_grids:
            # If we are skipping grids, return an empty dictionary.
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

    return results
