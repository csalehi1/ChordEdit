# model.py

"""
Difference-aware grid surface predictor and timestep selector.

    Model architecture:
    predictor(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)
    selector(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from _helpers import (
    MEAN_SURFACE_NAME,
    load_mean_surface,
    mean_surface_from_dict,
    nearest_indices,
    phi_from_delta_grids,
    resolve_device,
)
from scores import calc_norm_deltas
from settings import *


def deltas_from_preds(
    preds: torch.Tensor,                             # (N, n_cells, 2)
    baseline_idx: int | torch.Tensor,
    mean_surface: torch.Tensor | None = None,        # (n_cells, 2)
) -> torch.Tensor:                                   # (N, n_cells, 2)
    """Map PREDICTION_SPACE to "deltas" space."""
    if PREDICTION_SPACE == "deltas":
        return preds
    if PREDICTION_SPACE == "residuals":
        if mean_surface is None:
            raise ValueError(f"Expected a mean_surface for {PREDICTION_SPACE=}")
        return preds + mean_surface.to(device=preds.device, dtype=preds.dtype)
    if PREDICTION_SPACE == "raws":
        return calc_norm_deltas(preds, baseline_idx)
    raise ValueError(f"Unknown {PREDICTION_SPACE=}")


# The true Delta at the default cell is identically 0 by construction, so the
# heads' predicted surface is re-baselined there rather than left to learn it.
# That only holds where the targets are deltas: under "raws" the metric value at
# the default cell is not 0 and subtracting it would corrupt the target space.
PIN_DEFAULT_CELL = PREDICTION_SPACE in ("deltas", "residuals")


def default_cell_index(
    cell_t_pairs: np.ndarray,  # (n_cells, 2)
    t_start_values: np.ndarray,
    t_end_values: np.ndarray,
    default_t_start: float = DEFAULT_T_START,
    default_t_end: float = DEFAULT_T_END,
) -> int:
    """Position of the default cell in the model's output cell order."""
    cell_t_pairs = np.asarray(cell_t_pairs, dtype=np.float64)
    cell_i = nearest_indices(t_start_values, cell_t_pairs[:, 0])
    cell_j = nearest_indices(t_end_values, cell_t_pairs[:, 1])
    default_i = _nearest_index(t_start_values, default_t_start)
    default_j = _nearest_index(t_end_values, default_t_end)
    found = np.flatnonzero((cell_i == default_i) & (cell_j == default_j))
    if found.size != 1:
        raise ValueError(
            f"Expected exactly one cell at the default "
            f"({default_t_start}, {default_t_end}), got {found.size}"
        )
    return int(found[0])


def combine_edit_features(f_src: torch.Tensor, f_tar: torch.Tensor) -> torch.Tensor:
    """Create difference-aware edit representation, z_edit."""
    return torch.cat([f_src, f_tar, f_tar - f_src, f_src * f_tar], dim=-1)


"""
Predictor.
"""

"""
VisionFeaturizer: spatial flattening + VisionProjector (P_v) -> F_v.
"""

class VisionProjector(nn.Module):
    """Project flattened visual tokens to attn_dim, P_v.

    No positional embedding: the paper's F_v has none, and a learned one was
    net-negative at every patch size both before and after the residual fix
    (docs/RESULTS_CLAUDE.md section 13).
    """

    def __init__(
        self,
        token_dim: int,
        attn_dim: int,
    ):
        super().__init__()
        self.proj = nn.Linear(token_dim, attn_dim)

    def forward(
        self,
        tokens: torch.Tensor,  # (N, N_v, token_dim)
    ) -> torch.Tensor:         # (N, N_v, d)
        return self.proj(tokens)


class VisionFeaturizer(nn.Module):
    """Project the visual tokens to attn_dim -> F_v.

    image_shape (C, S, S) is the VAE latent grid, spatially flattened into
    patch tokens before projection. image_shape (N_tok, D) is an
    already-tokenized sequence -- the pooled CLIP token under IMG_EMB_SOURCE
    "clip", or the padded VAE+CLIP concatenation under "vae+clip" -- projected
    as one stream. CLIP-Edited is scored with CLIP-L/14, so that encoder's
    space is the one the label is expressible in, which the VAE latent's is not.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...],
        attn_dim: int = ATTN_DIM,
        patch_size: int = PATCH_SIZE,
    ):
        super().__init__()
        if len(image_shape) == 3:
            channels, height, width = image_shape
            if height != width:
                raise ValueError(f"Expected {image_shape} to be a square")
            if patch_size < 1 or height % patch_size != 0:
                raise ValueError(f"Expected {patch_size} to divide {height}")
            self.channels = channels
            self.side = height
            self.patch_size = patch_size
            self.n_grid = height // patch_size
            self.n_tokens = self.n_grid ** 2
            self.token_dim = channels * patch_size ** 2
        elif len(image_shape) == 2:
            self.token_dim = int(image_shape[-1])
        else:
            raise ValueError(f"Expected (C, S, S) latents or (N_tok, D) tokens, got {image_shape}")
        self.image_shape = tuple(int(v) for v in image_shape)
        self.projector = VisionProjector(self.token_dim, attn_dim)

    def spatial_flatten(
        self,
        image_tokens: torch.Tensor,  # (N, C, S, S)
    ) -> torch.Tensor:               # (N, N_v, token_dim)
        """Split the latent into patch tokens; patch_size 1 is one token per position."""
        n = image_tokens.shape[0]
        if tuple(image_tokens.shape[1:]) != (self.channels, self.side, self.side):
            raise ValueError(f"Expected {image_tokens.shape} == (N, {self.channels}, {self.side}, {self.side})")
        g, p = self.n_grid, self.patch_size
        tokens = image_tokens.reshape(n, self.channels, g, p, g, p)
        return tokens.permute(0, 2, 4, 1, 3, 5).reshape(n, self.n_tokens, self.token_dim)

    def forward(
        self,
        image_tokens: torch.Tensor,  # (N, C, S, S) latents or (N, N_tok, D) tokens
    ) -> torch.Tensor:               # (N, N_v, d)
        """Return the projected visual tokens F_v serving as keys and values."""
        if image_tokens.dim() == 4:
            image_tokens = self.spatial_flatten(image_tokens)
        return self.projector(image_tokens)


"""
TextFeaturizer: token-dim concatenation + TextProjector (P_t) -> F_t.
"""

class TextProjector(nn.Module):
    """Project each prompt token, P_t: R^{D} -> R^{d}, shared across the pair."""

    def __init__(
        self,
        text_dim: int,
        attn_dim: int = ATTN_DIM,
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.proj = nn.Linear(text_dim, attn_dim)

    def forward(
        self,
        pair: torch.Tensor,  # (N, 2, D_txt)
    ) -> torch.Tensor:       # (N, 2, d)
        return self.proj(pair)


class TextFeaturizer(nn.Module):
    """Concatenate pooled prompt tokens along the token dim, TextProjector -> F_t."""

    def __init__(
        self,
        text_dim: int,
        attn_dim: int = ATTN_DIM,
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.projector = TextProjector(text_dim, attn_dim=attn_dim)

    def forward(
        self,
        src_emb: torch.Tensor,  # (N, 1, D_txt)
        tar_emb: torch.Tensor,  # (N, 1, D_txt)
    ) -> torch.Tensor:          # (N, 2, d)
        """Return the prompt queries F_t, source first then target."""
        if src_emb.shape != tar_emb.shape:
            raise ValueError(f"Expected {src_emb.shape} == {tar_emb.shape}")
        if src_emb.dim() != 3:
            raise ValueError(f"Expected {tuple(src_emb.shape)} == (N, 1, D_txt)")
        pair = torch.cat([src_emb, tar_emb], dim=-2)
        return self.projector(pair)


"""
TokenTextFeaturizer / TokenGrounder: the (77, D) prompt sequences as queries.
"""

class TokenTextFeaturizer(nn.Module):
    """Project every prompt token, P_t applied per token of both sequences."""

    def __init__(self, text_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.proj = nn.Linear(text_dim, attn_dim)

    def forward(
        self,
        src_tokens: torch.Tensor,  # (N, T, D_txt)
        tar_tokens: torch.Tensor,  # (N, T, D_txt)
    ) -> torch.Tensor:             # (N, 2, T, d)
        if src_tokens.shape != tar_tokens.shape:
            raise ValueError(f"Expected {src_tokens.shape} == {tar_tokens.shape}")
        return self.proj(torch.stack([src_tokens, tar_tokens], dim=1))


def diff_saliency(
    q: torch.Tensor,     # (N, 2, T, d), projected but not yet grounded
    masks: torch.Tensor, # (N, 2, T) bool
) -> torch.Tensor:       # (N, 2, T)
    """Per-token novelty of each prompt against the other, 1 - max cosine.

    Source and target prompts are near-duplicates differing in a few words, so a
    masked mean is mostly shared scaffold. Matching is soft rather than
    positional because a changed word can retokenize to a different length, and
    the text encoder is causal, so every state after the first edit differs even
    where the word does not. Computed on the pre-grounding states, so it
    measures prompt difference and not agreement with the image, and detached,
    so pooling stays a fixed data-dependent operator.
    """
    x = torch.nn.functional.normalize(q.float(), dim=-1)
    sim = torch.einsum("nid,njd->nij", x[:, 0], x[:, 1])  # (N, T_src, T_tar)
    neg = torch.finfo(sim.dtype).min
    m_src, m_tar = masks[:, 0], masks[:, 1]
    # Each direction ignores the other prompt's padding as a possible match.
    w_src = 1.0 - sim.masked_fill(~m_tar.unsqueeze(1), neg).max(dim=2).values
    w_tar = 1.0 - sim.masked_fill(~m_src.unsqueeze(2), neg).max(dim=1).values
    return (torch.stack([w_src, w_tar], dim=1) * masks).clamp(min=0.0).detach()


def append_null_token(
    tokens: torch.Tensor,      # (N, N_v, d)
    null_token: torch.Tensor,  # (1, 1, d)
) -> torch.Tensor:             # (N, N_v + 1, d)
    """Append the learned null key/value token to the visual tokens.

    Unconditional. A query naming content that is not in the source image would
    otherwise have to spend its whole softmax mass on patches that do not match
    it, and under IMG_EMB_SOURCE "clip" the visual sequence is a single token, so
    without the null key the softmax is over one element -- weight identically 1
    -- and cross-attention degenerates into a query-independent linear map.
    """
    return torch.cat([tokens, null_token.expand(tokens.shape[0], -1, -1)], dim=-2)


class TokenGrounder(nn.Module):
    """Ground every prompt token in the visual tokens, then pool over real tokens.

    Queries never interact in cross-attention, so the two prompts are folded
    into the batch (shared weights, no cross-prompt leakage) and padding queries
    are left to attend freely; the padding mask is applied at the pool, which is
    the only place it changes an answer.
    """

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout_rate: float = ATTN_DROPOUT,
        n_layers: int = ATTN_LAYERS,
        ffn_mult: float = FFN_MULT,
        layerscale_init: float | None = LAYERSCALE_INIT,
    ):
        super().__init__()
        self.kv_norm = nn.LayerNorm(attn_dim)  # once: the visual tokens are static
        self.layers = nn.ModuleList([
            CrossAttentionBlock(
                attn_dim, n_heads, dropout_rate,
                ffn_mult=ffn_mult, layerscale_init=layerscale_init,
                kv_prenorm=False,
            )
            for _ in range(max(1, int(n_layers)))
        ])
        self.out_norm = nn.LayerNorm(attn_dim)
        # A query naming content that is not in the image rests here instead of
        # spending its whole softmax on patches that do not match it.
        self.null_token = nn.Parameter(torch.zeros(1, 1, attn_dim))

    @staticmethod
    def _pool(
        x: torch.Tensor,  # (N, 2, T, d)
        w: torch.Tensor,  # (N, 2, T)
    ) -> torch.Tensor:    # (N, 2, d)
        w = w.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)

    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, T, d)
        tokens: torch.Tensor,   # (N, N_v, d)
        masks: torch.Tensor,    # (N, 2, T) bool
        saliency: torch.Tensor | None = None,  # (N, 2, T)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        n, two, t, d = queries.shape
        kv = self.kv_norm(append_null_token(tokens, self.null_token)).repeat_interleave(two, dim=0)

        q = queries.reshape(n * two, t, d)
        for layer in self.layers:
            q = layer(q, kv, kv_is_normed=True)
        grounded = self.out_norm(q).reshape(n, two, t, d)

        f_mean = self._pool(grounded, masks)
        f_diff = None if saliency is None else self._pool(grounded, saliency)
        return f_mean, f_diff


"""
CrossAttentionPooler: CrossAttn(Q=F_t, K=F_v, V=F_v).
"""

class CrossAttentionBlock(nn.Module):
    """One grounding step, CrossAttn(Q=F_t, K=F_v, V=F_v), as a pre-LN block."""

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout_rate: float = ATTN_DROPOUT,
        ffn_mult: float = FFN_MULT,
        layerscale_init: float | None = LAYERSCALE_INIT,
        kv_prenorm: bool = True,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout_rate, batch_first=True)
        self.q_norm = nn.LayerNorm(attn_dim)
        # Hoisted out by TokenGrounder: the keys/values are the same tensor at
        # every layer, so one norm shared across the stack is the same map.
        self.kv_norm = nn.LayerNorm(attn_dim) if kv_prenorm else None
        # LayerScale: at a small init the block is near-identity, so the queries
        # reach the readout unchanged at step 0 and grounding is learned rather
        # than assumed.
        self.gamma_attn = nn.Parameter(float(layerscale_init) * torch.ones(attn_dim)) if layerscale_init is not None else None
        self.gamma_ffn = None
        self.ffn_norm, self.ffn = None, None
        if ffn_mult > 0:
            n_hidden = max(1, int(round(ffn_mult * attn_dim)))
            self.ffn_norm = nn.LayerNorm(attn_dim)
            self.ffn = nn.Sequential(
                nn.Linear(attn_dim, n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_hidden, attn_dim),
            )
            if layerscale_init is not None:
                self.gamma_ffn = nn.Parameter(float(layerscale_init) * torch.ones(attn_dim))

    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, d)
        tokens: torch.Tensor,   # (N, N_v, d)
        kv_is_normed: bool = False,
    ) -> torch.Tensor:          # (N, 2, d)
        kv = tokens if (kv_is_normed or self.kv_norm is None) else self.kv_norm(tokens)
        grounded, _ = self.attn(self.q_norm(queries), kv, kv, need_weights=False)
        queries = queries + (grounded if self.gamma_attn is None else self.gamma_attn * grounded)
        if self.ffn is not None:
            update = self.ffn(self.ffn_norm(queries))
            queries = queries + (update if self.gamma_ffn is None else self.gamma_ffn * update)
        return queries


class CrossAttentionPooler(nn.Module):
    """Ground the prompt queries in the visual tokens over ATTN_LAYERS blocks."""

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout_rate: float = ATTN_DROPOUT,
        n_layers: int = ATTN_LAYERS,
        ffn_mult: float = FFN_MULT,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttentionBlock(attn_dim, n_heads, dropout_rate, ffn_mult=ffn_mult)
            for _ in range(max(1, int(n_layers)))
        ])
        self.null_token = nn.Parameter(torch.zeros(1, 1, attn_dim))

    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, d)
        tokens: torch.Tensor,   # (N, N_v, d)
    ) -> torch.Tensor:          # (N, 2, d)
        """Return the image-grounded prompt features [f_src; f_tar]."""
        tokens = append_null_token(tokens, self.null_token)
        for layer in self.layers:
            queries = layer(queries, tokens)
        return queries


"""
TextCombiner: C_theta(z_edit) -> h.
"""

class TextCombiner(nn.Module):
    """Shallow MLP C_theta compressing z_edit to the edit descriptor h."""

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_hidden: int = COMBINER_HIDDEN,
        dropout_rate: float = COMBINER_DROPOUT,
        n_segments: int = 4,
    ):
        super().__init__()
        self.attn_dim = attn_dim
        self.n_segments = int(n_segments)
        width = self.n_segments * attn_dim
        # Per-segment, not one LayerNorm over the whole concatenation: a single
        # norm rescales the segments jointly, so a difference segment that is
        # small because the prompts are near-duplicates -- which they always are
        # -- stays small next to the two large concat segments. One norm per
        # segment gives each term unit scale on its own.
        self.seg_norms = nn.ModuleList(nn.LayerNorm(attn_dim) for _ in range(self.n_segments))
        self.body = nn.Sequential(
            nn.Linear(width, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, attn_dim),
        )

    def forward(
        self,
        z_edit: torch.Tensor,  # (N, n_segments * d)
    ) -> torch.Tensor:         # (N, d)
        parts = z_edit.split(self.attn_dim, dim=-1)
        z_edit = torch.cat([nrm(part) for nrm, part in zip(self.seg_norms, parts)], dim=-1)
        return self.body(z_edit)


def make_metric_head(
    attn_dim: int,
    n_cells: int,
    n_hidden: int = HEAD_HIDDEN,
    dropout_rate: float = COMBINER_DROPOUT,
) -> nn.Module:
    """Readout G from the edit descriptor h to one scalar per grid cell.

    n_hidden 0 is the paper's bare Linear(d, n_cells), which forces every cell of
    the surface to be a linear functional of the same h.
    """
    if n_hidden <= 0:
        return nn.Linear(attn_dim, n_cells)
    return nn.Sequential(
        nn.Linear(attn_dim, n_hidden),
        nn.GELU(),
        nn.Dropout(dropout_rate),
        nn.Linear(n_hidden, n_cells),
    )


class AttentionRegressor(nn.Module):
    """
    Cross-attention predictor of the PSNR-Unedited and CLIP-Edited surfaces.

    The two heads G_PSNR and G_CLIP are independently parameterized linear
    layers over the shared edit descriptor h, so the smooth content-preservation
    surface and the instruction-dependent edit-strength surface are read out
    separately.
    """

    def __init__(
        self,
        image_shape: tuple[int, ...],
        text_dim: int,
        n_cells: int,
        n_targets: int = len(TARGET_COLS),
        attn_dim: int = ATTN_DIM,
        default_cell: int | None = None,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError(f"Expected 2 targets (PSNR, CLIP), got {n_targets=}")

        self.n_cells = int(n_cells)
        self.n_targets = n_targets
        # Bounded outputs are only meaningful where the targets are bounded.
        self.bounded = PREDICTION_SPACE == "deltas"

        self.text_source = str(TEXT_EMB_SOURCE)
        if USE_DIFF_SALIENCY and self.text_source != "tokens":
            raise ValueError("USE_DIFF_SALIENCY needs TEXT_EMB_SOURCE='tokens'")
        self.use_saliency = bool(USE_DIFF_SALIENCY)
        # True Delta at the default cell is identically 0 by construction, so
        # predicting it is a degree of freedom the heads would otherwise spend
        # learning a constant. Off only under "raws", where the constraint is
        # false; see PIN_DEFAULT_CELL above.
        if PIN_DEFAULT_CELL and default_cell is None:
            raise ValueError(f"{PREDICTION_SPACE=} needs the default cell index")
        self.default_cell = int(default_cell) if PIN_DEFAULT_CELL else None

        self.vision_featurizer = VisionFeaturizer(image_shape, attn_dim=attn_dim)
        if self.text_source == "tokens":
            self.text_featurizer = TokenTextFeaturizer(text_dim, attn_dim=attn_dim)
            self.cross_attn = TokenGrounder(attn_dim=attn_dim)
        else:
            self.text_featurizer = TextFeaturizer(text_dim, attn_dim=attn_dim)
            self.cross_attn = CrossAttentionPooler(attn_dim=attn_dim)
        # [f_src; f_tar; f_tar - f_src; f_src * f_tar], plus the two
        # difference-saliency pools when they are on.
        n_seg = 6 if self.use_saliency else 4
        # A shared h lets the PSNR-dominated gradient set the representation both
        # heads read. SPLIT_COMBINER gives each metric its own C_theta over the
        # same z_edit so the two surfaces can be described differently.
        if SPLIT_COMBINER:
            self.psnr_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
            self.clip_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
        else:
            self.combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)

        # G_PSNR and G_CLIP are independently parameterized readouts of the
        # edit descriptor, one scalar per timestep-grid cell.
        self.psnr_head = make_metric_head(attn_dim, n_cells)
        self.clip_head = make_metric_head(attn_dim, n_cells)

        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def destandardize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Undo the target z-scoring; identity unless the space standardizes."""
        return standardized * self.target_std + self.target_mean

    def set_target_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """
        Store the train target mean/std that forward() predicts in.

        Only PREDICTION_SPACE "raws" needs this because PSNR-Unedited and CLIP-Edited
        live on unrelated scales, so the heads regress z-scored targets.
        """
        mean = torch.as_tensor(mean, dtype=self.target_mean.dtype, device=self.target_mean.device).reshape(-1)
        std = torch.as_tensor(std, dtype=self.target_std.dtype, device=self.target_std.device).reshape(-1)
        if mean.numel() != self.n_targets or std.numel() != self.n_targets:
            raise ValueError(f"Expected {self.n_targets} target stats, got {mean.numel()} / {std.numel()}")
        self.target_mean.copy_(mean)
        # A constant target column would otherwise divide by zero.
        self.target_std.copy_(std.clamp(min=1e-8))

    def forward(
        self,
        image_tokens: torch.Tensor,               # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,              # (N, 1, D_txt) or (N, T, D_txt)
        target_tokens: torch.Tensor,              # (N, 1, D_txt) or (N, T, D_txt)
        source_mask: torch.Tensor | None = None,  # (N, T) bool, "tokens" only
        target_mask: torch.Tensor | None = None,  # (N, T) bool, "tokens" only
    ) -> torch.Tensor:                            # (N, n_cells, 2)
        """Return per-cell (psnr, clip) predictions, standardized where active."""
        # Ground both prompts in the source image with one cross-attention pass.
        tokens = self.vision_featurizer(image_tokens)
        f_diff = None
        if self.text_source == "tokens":
            if source_mask is None or target_mask is None:
                raise ValueError("TEXT_EMB_SOURCE='tokens' needs source_mask and target_mask")
            text_masks = torch.stack([source_mask, target_mask], dim=1)
            queries = self.text_featurizer(source_tokens, target_tokens)
            saliency = diff_saliency(queries, text_masks) if self.use_saliency else None
            grounded, f_diff = self.cross_attn(queries, tokens, text_masks, saliency)
        else:
            queries = self.text_featurizer(source_tokens, target_tokens)
            grounded = self.cross_attn(queries, tokens)
        f_src, f_tar = grounded[:, 0], grounded[:, 1]

        # Difference-aware edit descriptor, read out by the two metric heads.
        z_edit = combine_edit_features(f_src, f_tar)
        if f_diff is not None:
            z_edit = torch.cat([z_edit, f_diff[:, 0], f_diff[:, 1]], dim=-1)
        if SPLIT_COMBINER:
            h_psnr = self.psnr_combiner(z_edit)
            h_clip = self.clip_combiner(z_edit)
        else:
            h_psnr = h_clip = self.combiner(z_edit)
        out = torch.stack([self.psnr_head(h_psnr), self.clip_head(h_clip)], dim=-1)
        if self.bounded:
            out = 2.0 * torch.sigmoid(out) - 1.0
        if self.default_cell is not None:
            # Pin the default cell to the 0 the true delta takes there.
            out = out - out[:, self.default_cell : self.default_cell + 1, :]
        return out


class AttentionModel(nn.Module):
    """Trainable AttentionRegressor sized from precomputed token-table shapes."""

    def __init__(
        self,
        image_shape: tuple[int, ...],
        text_dim: int,
        n_cells: int,
        device: torch.device | str | None = None,
        default_cell: int | None = None,
    ):
        super().__init__()
        self.regressor = AttentionRegressor(
            image_shape, text_dim, n_cells, default_cell=default_cell,
        )
        if device is not None:
            self.regressor.to(device)

    def pred_cells(
        self,
        image_tokens: torch.Tensor,               # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,              # (N, 1, D_txt) or (N, T, D_txt)
        target_tokens: torch.Tensor,              # like source_tokens
        source_mask: torch.Tensor | None = None,  # (N, T) bool
        target_mask: torch.Tensor | None = None,  # (N, T) bool
    ) -> torch.Tensor:                            # (N, n_cells, 2)
        """Predict per-cell (psnr, clip) in PREDICTION_SPACE units."""
        standardized = self.regressor(image_tokens, source_tokens, target_tokens, source_mask, target_mask)
        return self.regressor.destandardize(standardized)


"""
Selector.
"""

@dataclass
class TimestepGridResult:
    psnr_grid: np.ndarray
    clip_grid: np.ndarray
    phi_grid: np.ndarray
    t_start_values: np.ndarray
    t_end_values: np.ndarray


@dataclass
class TimestepSelection:
    t_start: float
    t_end: float
    deviate: bool
    pred_gain: float
    phi_grid: np.ndarray
    psnr_grid: np.ndarray
    clip_grid: np.ndarray


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(nearest_indices(values, [target])[0])


class TimestepSelector:
    """Select (t_start, t_end) from the predicted grid; no trainable weights."""

    def __init__(
        self,
        model: AttentionModel,
        cell_t_pairs: np.ndarray,
        t_start_values: tuple[float, ...] | list[float] | np.ndarray,
        t_end_values: tuple[float, ...] | list[float] | np.ndarray,
        default_t_start: float = DEFAULT_T_START,
        default_t_end: float = DEFAULT_T_END,
        mean_surface: np.ndarray | None = None,
    ):
        self.model = model
        self.t_start_values = np.asarray(t_start_values, dtype=np.float64)
        self.t_end_values = np.asarray(t_end_values, dtype=np.float64)
        self.default_t_start = default_t_start
        self.default_t_end = default_t_end
        self._default_i = _nearest_index(self.t_start_values, default_t_start)
        self._default_j = _nearest_index(self.t_end_values, default_t_end)

        # Grid positions of the model's output cells, in the training cell order.
        cell_t_pairs = np.asarray(cell_t_pairs, dtype=np.float64)
        self._cell_i = nearest_indices(self.t_start_values, cell_t_pairs[:, 0])
        self._cell_j = nearest_indices(self.t_end_values, cell_t_pairs[:, 1])

        # Index of the default cell within that same cell order, so predictions
        # can be re-baselined before they are scattered into the grid.
        self._default_k = default_cell_index(
            cell_t_pairs, self.t_start_values, self.t_end_values, default_t_start, default_t_end,
        )

        # The mean surface is only an offset for "residuals"; hold it in cell
        # order to match what the model emits.
        self.mean_surface = None if mean_surface is None else np.asarray(mean_surface, dtype=np.float64)
        self._mean_surface_cells = (
            None if self.mean_surface is None
            else torch.as_tensor(self.mean_surface[self._cell_i, self._cell_j], dtype=torch.float32)
        )
        if PREDICTION_SPACE == "residuals" and self._mean_surface_cells is None:
            raise ValueError(f"Expected a mean_surface for {PREDICTION_SPACE=}")

    @property
    def n_start(self) -> int:
        return len(self.t_start_values)

    @property
    def n_end(self) -> int:
        return len(self.t_end_values)

    @torch.no_grad()
    def pred_grid(
        self,
        image_tokens: torch.Tensor,               # (C, S, S) or (N_tok, D), one sample
        source_tokens: torch.Tensor,              # (1, D_txt) or (T, D_txt)
        target_tokens: torch.Tensor,              # like source_tokens
        source_mask: torch.Tensor | None = None,  # (T,) bool
        target_mask: torch.Tensor | None = None,  # (T,) bool
    ) -> TimestepGridResult:
        """
        Single predictor forward covering every labeled (t_start, t_end) cell.

        Takes ONE unbatched sample's embeddings (embeddings.SampleEmbeddings
        fields). Scatters the predictions into the grid, maps them to deltas
        (deltas_from_preds), and scores them with phi (settings.SCORE_PHI).
        Because the predictor estimates the two metric surfaces rather than the
        scalarized objective, re-scoring a saved run under a different SCORE_FN
        needs no retraining.
        """
        device = self.model.regressor.target_mean.device
        batched = lambda t: None if t is None else t.unsqueeze(0).to(device)
        preds = self.model.pred_cells(
            batched(image_tokens),
            batched(source_tokens),
            batched(target_tokens),
            batched(source_mask),
            batched(target_mask),
        )

        # Map out of PREDICTION_SPACE before anything is scored.
        deltas = deltas_from_preds(preds, self._default_k, self._mean_surface_cells)
        cells = deltas[0].detach().cpu().numpy()

        # Scatter the per-cell deltas into the grid.
        grid = np.full((self.n_start, self.n_end, 2), np.nan)
        grid[self._cell_i, self._cell_j] = cells
        psnr, clip = grid[..., 0], grid[..., 1]
        phi = phi_from_delta_grids(grid[None])[0]
        return TimestepGridResult(psnr, clip, phi, self.t_start_values, self.t_end_values)

    def select_grid(
        self,
        grid: TimestepGridResult,
        noise_floor: float = NOISE_FLOOR_PHI,
        clip_floor: float | None = None,
        phi_weights: tuple[float, float] | None = None,
    ) -> TimestepSelection:
        """Argmax over labeled cells with a deviate-or-default gate."""

        # Reported phi used for pred_gain and TimestepSelection.phi_grid.
        phi = grid.phi_grid
        if not np.isfinite(phi).any():
            raise ValueError("Expected finite cells in phi_grid")

        rank_phi = phi
        if phi_weights is not None:
            # Reweight phi for selection.
            deltas = np.stack([grid.psnr_grid, grid.clip_grid], axis=-1)[None]
            rank_phi = phi_from_delta_grids(deltas, weights=phi_weights)[0]
        ranked_default = rank_phi[self._default_i, self._default_j]
        if clip_floor is not None:
            # Restrict argmax to cells whose CLIP delta clears the floor.
            eligible = np.where(grid.clip_grid >= clip_floor, rank_phi, np.nan)
            if np.isfinite(eligible).any():
                rank_phi = eligible

        i, j = np.unravel_index(np.nanargmax(rank_phi), rank_phi.shape)
        # Gate compares in ranking space and reported gain stays on default phi.
        gate_gain = float(rank_phi[i, j] - ranked_default)
        pred_gain = float(phi[i, j] - phi[self._default_i, self._default_j])

        # Return the selected cell if the gate is triggered.
        if gate_gain > noise_floor:
            return TimestepSelection(
                t_start=float(self.t_start_values[i]),
                t_end=float(self.t_end_values[j]),
                deviate=True,
                pred_gain=pred_gain,
                phi_grid=phi,
                psnr_grid=grid.psnr_grid,
                clip_grid=grid.clip_grid,
            )

        # Fallback to the default cell if the gate is not triggered.
        return TimestepSelection(
            t_start=self.default_t_start,
            t_end=self.default_t_end,
            deviate=False,
            pred_gain=pred_gain,
            phi_grid=phi,
            psnr_grid=grid.psnr_grid,
            clip_grid=grid.clip_grid,
        )


def load_timestep_selector(
    weights_path: Path | str,
    device: torch.device | str | None = None,
    gpu: int | str | None = None,
) -> TimestepSelector:
    """Load the trained predictor and wrap it in a selector."""
    weights_path = Path(weights_path)
    if device is None:
        device = resolve_device(gpu)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # The run must be replayed in the space it was trained in: the heads, the
    # target stats, and the mean surface all mean different things per space.
    space = str(ckpt.get("prediction_space", PREDICTION_SPACE))
    if space != PREDICTION_SPACE:
        raise ValueError(f"Expected {space=} == {PREDICTION_SPACE=}")

    # cell_t_pairs gives the output cell count and each cell's grid position.
    cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
    image_shape = tuple(int(v) for v in ckpt["image_shape"])
    text_dim = int(tuple(ckpt["source_shape"])[-1])
    # map_location put these on the model's device; numpy needs them back on host.
    t_start_values = np.asarray(ckpt["t_start_values"].cpu(), dtype=np.float64)
    t_end_values = np.asarray(ckpt["t_end_values"].cpu(), dtype=np.float64)
    model = AttentionModel(
        image_shape, text_dim, int(cell_t_pairs.shape[0]), device=device,
        # The regressor re-baselines its surface here, so the index has to be
        # rebuilt from the checkpoint's own grid axes rather than left None.
        default_cell=default_cell_index(cell_t_pairs, t_start_values, t_end_values),
    )
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # The checkpoint carries the training grid's axes, so the selector's grid is
    # the data's by construction.
    mean_surface = None
    if PREDICTION_SPACE == "residuals":
        surface = load_mean_surface(weights_path.parent)
        if surface is None:
            raise FileNotFoundError(f"Missing mean surface: {weights_path.parent / MEAN_SURFACE_NAME}")
        mean_surface = mean_surface_from_dict(surface, t_start_values, t_end_values)
        print(f"Loaded mean_surface ({surface['split']} split, {surface['n_samples']} samples).")

    print(f"Selector predicts in {PREDICTION_SPACE!r} space over {cell_t_pairs.shape[0]} cells.")
    return TimestepSelector(
        model,
        cell_t_pairs,
        t_start_values=t_start_values,
        t_end_values=t_end_values,
        mean_surface=mean_surface,
    )
