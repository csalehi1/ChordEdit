"""Factorized ChordEdit t_start by t_end grid generation.

Copied from scripts/daniel_run_grid_ablation.py and scripts/daniel_pipeline_chord.py.
Reuses shared encode/transport work across grid cells (requires n_steps=1):

  * 1 VAE encode + 1 prompt encode + 1 noise draw per image
  * 1 transport per unique t_start (delta==0 uses a 2-forward fast path)
  * 1 batched cleanup + per-cell decode per t_start row

On an NxN grid this turns N^2 full pipelines into 1 encode + N transports +
N^2 cleanups/decodes. Output is bit-for-bit identical to pipeline.__call__.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root

import torch

from common import LocalRecord

if TYPE_CHECKING:
    from PIL import Image

    from pipeline_chord import ChordEditPipeline, _PromptCondition

    _Embed = Union[torch.Tensor, _PromptCondition]


def _u_estimate(pipeline, x_anchor, src_embed, edit_embed, noise, t_s: float, delta: float):
    """
    Dispatch to the cheapest exact ``_u_estimate`` variant for this config.

    For default mode with ``delta == 0`` the four-forward estimate collapses to
    two forwards (see ``_u_estimate_delta0``). ``sym`` mode keeps the reference path.
    """
    if pipeline._chord_edit_mode == "sym":
        # Use the symmetric estimate for sym mode.
        return pipeline._u_estimate_sym(x_anchor, src_embed, edit_embed, noise, t_s, delta)
    if delta == 0.0:
        # Use the delta 0 optimization for delta == 0.
        return _u_estimate_delta0(pipeline, x_anchor, src_embed, edit_embed, noise, t_s)
    # Use the default estimate for other cases.
    return pipeline._u_estimate_default(x_anchor, src_embed, edit_embed, noise, t_s, delta)


def _u_estimate_delta0(
    pipeline,
    x_anchor: torch.Tensor,
    src_embed: "_Embed",
    edit_embed: "_Embed",
    noise: "torch.Tensor | List[torch.Tensor]",
    t_s: float,
) -> torch.Tensor:
    """
    Exact ``t_delta == 0`` specialization of ``_u_estimate_default``.

    With ``delta == 0`` the renoised anchor equals ``z_s``, so the last two of the
    four forwards duplicate the first two and the blend collapses to ``dv_s``.
    Running only the two unique forwards is ~2x cheaper and bit-for-bit identical.
    """
    batch, device = x_anchor.shape[0], x_anchor.device
    t_idx_s = pipeline._time_to_index(batch, t_s, device=device)

    noises = noise if isinstance(noise, (list, tuple)) else [noise]
    num_noises = len(noises)
    noise_stack = torch.stack(noises, dim=0)

    alpha_s, sigma_s = pipeline._get_alpha_sigma(x_anchor, t_idx_s)

    x_anchor_b = x_anchor.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
    alpha_s_b = alpha_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)
    sigma_s_b = sigma_s.unsqueeze(0).expand(num_noises, -1, -1, -1, -1)

    z_s = alpha_s_b * x_anchor_b + sigma_s_b * noise_stack

    # src and edit share z_s and t_idx_s; only the conditioning differs.
    samples = z_s.repeat(1, 2, 1, 1, 1).reshape(num_noises * 2 * batch, *x_anchor.shape[1:])
    conds = pipeline._cat_conditions([src_embed, edit_embed])
    conds = pipeline._repeat_condition(conds, num_noises)
    timesteps = t_idx_s.repeat(2 * num_noises)

    noise_pred = pipeline._predict_noise(sample=samples, timesteps=timesteps, cond=conds)

    noise_pred = noise_pred.reshape(num_noises, 2, batch, *x_anchor.shape[1:])
    x0_all = (z_s.unsqueeze(1) - sigma_s_b.unsqueeze(1) * noise_pred) / alpha_s_b.unsqueeze(1)
    x_src_p_s, x_tar_p_s = x0_all.unbind(dim=1)

    dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)
    return dv_s


def _cleanup_decode_row(
    pipeline,
    x_transport: torch.Tensor,
    edit_embed: "_Embed",
    noise: torch.Tensor,
    t_end_values: List[float],
    cleanup: bool,
) -> List["Image.Image"]:
    """
    Batched cleanup + per-cell decode for one ``t_start`` row of the grid.

    Cleanup (``_pred_x0``) is batched into one UNet forward across ``t_end``;
    the VAE decode is kept per-cell (batch 1 is fastest for the SD VAE); the
    device->CPU / PIL conversion runs once on the concatenated batch. Assumes a
    single source image (``x_transport.shape[0] == 1``).
    """
    n = len(t_end_values)
    batch = x_transport.shape[0]
    device = x_transport.device

    # Repeat the transport tensor for each t_end value.
    x_rep = x_transport.repeat(n, *([1] * (x_transport.dim() - 1)))

    if cleanup:
        # Prepare the cleanup timesteps and condition.
        timesteps = torch.cat(
            [pipeline._time_to_index(batch, float(t_end), device=device) for t_end in t_end_values],
            dim=0,
        )
        # Repeat the edit embed for each t_end value.
        cond = pipeline._repeat_condition(edit_embed, n)
        # Repeat the noise for each t_end value.
        noise_rep = noise.repeat(n, *([1] * (noise.dim() - 1)))
        x0 = pipeline._pred_x0(x_rep, timesteps, cond, noise_rep)
    else:
        x0 = x_rep

    # Decode the latent to image.
    decoded = torch.cat(
        [pipeline._decode_latent_to_image(x0[i * batch : (i + 1) * batch]) for i in range(n)],
        dim=0,
    )
    decoded, _ = pipeline._apply_safety_checker(decoded)
    pil_images = pipeline._tensor_to_pil(decoded)
    return [pil_images[i * batch] for i in range(n)]


# A grid ablation sweeps two timesteps independently: the transport start time
# t_start and the cleanup end time t_end. With N values on each axis we want an
# N x N grid of edited images per source image (here N = 11 -> 121 cells).
#
# NAIVE APPROACH (scripts/run_grid_ablation.py): call the full pipeline once per
# (t_start, t_end) cell. Every one of the N^2 calls repeats the *same* preamble
# that does not depend on the cell -- VAE-encode the source, encode the source &
# target prompts, draw the edit noise -- and then repeats the transport, which is
# the single most expensive step (several batched UNet forwards). So the naive
# scheme pays ~N^2 encodes, ~N^2 prompt encodes, and ~N^2 transports even though
# almost all of that work is identical across cells.
#
# KEY INSIGHT: for a single-step edit (n_steps == 1) the computation factorizes
# cleanly along the two axes. Each cell is exactly:
#
#     x_transport = x_src + step_scale * u_hat(x_src, t_start, t_delta)   # transport
#     x0          = pred_x0(x_transport, t_end)                           # cleanup
#     image       = VAE_decode(x0)                                        # decode
#
# t_start enters ONLY through the transport; t_end enters ONLY through the cleanup.
# The preamble depends on neither. That separability lets us separate each piece of
# work to the coarsest axis it actually depends on:
#
#   * preamble (encode/prompts/noise)     -> run ONCE per image.
#   * transport u_hat                     -> run ONCE per unique t_start (N times,
#                                            not N^2). This is the dominant win,
#                                            an ~N-fold cut in the costliest step.
#   * cleanup and decode                  -> the only genuinely per-cell work,
#                                            and even the cleanup is batched one
#                                            t_start-row at a time (_cleanup_decode_row).
#
# So the N^2 full pipelines collapse to: 1 encode + N transports + N^2 cleanups +
# N^2 decodes, all bit-for-bit identical to calling the reference pipeline per cell.
# Two further exact speedups stack on top: the t_delta == 0 transport fast path
# (_u_estimate_delta0, 4 UNet forwards -> 2) and the batched cleanup row above.
#
# Requires n_steps == 1 (guarded by settings.require_factorizable_config); with
# more steps the two axes no longer separate and the factorization breaks.
def run_factorized_grid(
    *,
    pipeline: "ChordEditPipeline",
    source_image: "Image.Image",
    record: LocalRecord,
    base_config: Dict[str, Any],
    t_start_values: List[float],
    t_end_values: List[float],
    t_delta: float,
    seed: int,
) -> Dict[Tuple[float, float], "Image.Image"]:
    """Generate every (t_start, t_end) cell for one source image."""
    with torch.no_grad():
        cfg = dict(base_config)
        cfg["t_delta"] = t_delta
        shared_params = pipeline._prepare_edit_params(cfg)

        # Prepare the source image, encode to latent and encode prompts.
        pixel_values = pipeline._prepare_image_tensor(source_image)
        latents = pipeline._encode_image_to_latent(pixel_values)
        src_embed = pipeline.encode_prompt([record.source_prompt])
        tgt_embed = pipeline.encode_prompt([record.target_prompt])
        noise_list = pipeline._prepare_noise_list(
            latents=latents,
            seed_value=seed,
            num_noises=shared_params["noise_samples"],
        )

        # Transport per unique t_start, re-prepare params so t_delta clamping matches.
        transport: Dict[float, torch.Tensor] = {}
        for t_start in t_start_values:
            cell_cfg = dict(base_config)
            cell_cfg["t_start"] = t_start
            cell_cfg["t_delta"] = t_delta
            params = pipeline._prepare_edit_params(cell_cfg)
            u_hat = _u_estimate(
                pipeline,
                latents,
                src_embed,
                tgt_embed,
                noise_list,
                params["t_start"],
                params["t_delta"],
            )
            transport[t_start] = (latents + params["step_scale"] * u_hat).detach()

        # Cleanup and decode for each (t_start, t_end) cell, one batched row per t_start.
        results: Dict[Tuple[float, float], "Image.Image"] = {}
        for t_start in t_start_values:
            row_images = _cleanup_decode_row(
                pipeline,
                transport[t_start],
                tgt_embed,
                noise_list[0],
                t_end_values,
                bool(shared_params["cleanup"]),
            )
            for t_end, image in zip(t_end_values, row_images):
                results[(t_start, t_end)] = image

    return results
