"""Optimized ChordEdit helpers for the factorized grid ablation.

Kept separate from pipeline_chord.py so the reference pipeline stays untouched.
Everything here mirrors the shapes and variable names of the corresponding
ChordEditPipeline methods to make the specialization easy to audit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Union

import torch

if TYPE_CHECKING:
    from pipeline_chord import ChordEditPipeline, _PromptCondition
    _Embed = Union[torch.Tensor, _PromptCondition]


def _u_estimate(
    pipeline: ChordEditPipeline,
    x_anchor,
    src_embed,
    edit_embed,
    noise,
    t_s: float,
    delta: float,
):
    """
    Dispatch to the cheapest exact ``_u_estimate`` variant for this config.

    Mirrors ``ChordEditPipeline._u_estimate`` but adds the ``delta == 0`` fast
    path: for default (non-``sym``) mode with ``delta == 0`` the four-forward
    estimate collapses to two forwards (see ``_u_estimate_delta0``), so we route
    there. ``sym`` mode ignores ``delta`` and keeps the reference path.
    """
    if pipeline._chord_edit_mode == "sym":
        print("Using symmetric edit mode ...")
        return pipeline._u_estimate_sym(x_anchor, src_embed, edit_embed, noise, t_s, delta)
    if delta == 0.0:
        print("Using t_delta=0 fast path ...")
        return _u_estimate_delta0(pipeline, x_anchor, src_embed, edit_embed, noise, t_s)
    return pipeline._u_estimate_default(x_anchor, src_embed, edit_embed, noise, t_s, delta)


def _u_estimate_delta0(
    pipeline: ChordEditPipeline,
    x_anchor: torch.Tensor,
    src_embed: _Embed,
    edit_embed: _Embed,
    noise: torch.Tensor | List[torch.Tensor],
    t_s: float,
) -> torch.Tensor:
    """Exact ``t_delta == 0`` specialization of ``_u_estimate_default``.

    ``_u_estimate_default`` issues four batched UNet forwards -- ``(z_s, src)``,
    ``(z_s, edit)``, ``(z_prev, src)``, ``(z_prev, edit)`` -- where ``z_prev`` is
    the anchor renoised at ``t_idx_s0 = time_to_index(t_s - delta)``. With
    ``delta == 0`` we have ``t_idx_s0 == t_idx_s``, so ``z_prev == z_s`` and the
    last two forwards duplicate the first two; the blended return
    ``(delta * dv_s + t_s * dv_s0) / (t_s + delta)`` then collapses to ``dv_s``.

    Running only the two unique forwards is ~2x cheaper and bit-for-bit
    identical to ``_u_estimate_default`` at ``delta == 0``. Default (non-``sym``)
    mode only: ``_u_estimate_sym`` ignores ``delta`` (its second timestep is
    ``1 - t_s``), so it has no equivalent saving.
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

    # src and edit share z_s and t_idx_s; only the conditioning differs, so the
    # two-forward batch is [z_s(src), z_s(edit)] per noise sample.
    samples = z_s.repeat(1, 2, 1, 1, 1).reshape(num_noises * 2 * batch, *x_anchor.shape[1:])

    conds = pipeline._cat_conditions([src_embed, edit_embed])
    conds = pipeline._repeat_condition(conds, num_noises)

    # Repeat t_idx_s for both forwards
    timesteps = t_idx_s.repeat(2 * num_noises)

    noise_pred = pipeline._predict_noise(
        sample=samples,
        timesteps=timesteps,
        cond=conds,
    )

    # alpha/sigma are identical across both forwards, so broadcast over the
    # (src, edit) axis instead of materializing the duplicated alpha/sigma cats.
    noise_pred = noise_pred.reshape(num_noises, 2, batch, *x_anchor.shape[1:])
    x0_all = (z_s.unsqueeze(1) - sigma_s_b.unsqueeze(1) * noise_pred) / alpha_s_b.unsqueeze(1)
    x_src_p_s, x_tar_p_s = x0_all.unbind(dim=1)

    # Directly sample R, instead of applying a smoothing delta
    dv_s = (x_tar_p_s - x_src_p_s).sum(dim=0) / float(num_noises)

    return dv_s
