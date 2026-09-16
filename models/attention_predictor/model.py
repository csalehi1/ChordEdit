# model.py

"""
Difference-aware grid surface predictor.

    Model architecture:
    predictor(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)
"""

from __future__ import annotations

import os
import sys

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_DIR, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _DIR)

import math

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from settings import *

if TYPE_CHECKING:
    from dataset import DatasetMetadata


"""
Target-space transforms on (..., n_cells, C) grids.

The regression space is deltas versus the default cell in shared units (raw / one scale
per column), which inverts exactly to raw PSNR/CLIP. The selection space re-normalizes
each raw grid per sample to [0, 1] per column before taking deltas, the paper's phi input.
"""

_EPS = 1e-8


def pin_default(values: torch.Tensor, default_cell: int) -> torch.Tensor:
    """Zero the default cell so the heads do not have to learn its value."""
    # (N, n_cells, C), int -> (N, n_cells, C)
    return values - values[..., default_cell, :].unsqueeze(-2)


def to_deltas(raw: torch.Tensor, default_cell: int, scale: torch.Tensor) -> torch.Tensor:
    """Regression space: (raw - raw at the default cell) / scale, one scale per column."""
    # (N, n_cells, C), int, (C,) -> (N, n_cells, C)
    return pin_default(raw, default_cell) / scale.to(raw)


def persample_deltas(raw: torch.Tensor, default_cell: int) -> torch.Tensor:
    """Selection space: per-sample min-max of each column, then deltas versus the default cell."""
    # (N, n_cells, C), int -> (N, n_cells, C)
    vmin = raw.nan_to_num(nan=math.inf).amin(dim=-2, keepdim=True)
    vmax = raw.nan_to_num(nan=-math.inf).amax(dim=-2, keepdim=True)
    return pin_default((raw - vmin) / (vmax - vmin + _EPS), default_cell)


def combine_edit_features(f_src: torch.Tensor, f_tar: torch.Tensor) -> torch.Tensor:
    """Create difference-aware edit representation, z_edit."""
    return torch.cat([f_src, f_tar, f_tar - f_src, f_src * f_tar], dim=-1)


"""
Text and Vision Projectors/Featurizers.
"""

class VisionProjector(nn.Module):
    """Project flattened visual tokens to attn_dim, P_v."""

    def __init__(self, img_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.proj = nn.Linear(img_dim, attn_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # Project from (N, N_tok, img_dim) to (N, N_tok, attn_dim).
        return self.proj(tokens)


class TextProjector(nn.Module):
    """Project each prompt token with a shared linear projector, P_t."""

    def __init__(self, text_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.attn_dim = attn_dim
        self.proj = nn.Linear(text_dim, attn_dim)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        # Project the last dim from text_dim to attn_dim.
        return self.proj(pair)


class VisionFeaturizer(nn.Module):
    """Project the visual tokens to attn_dim, F_v."""

    def __init__(self, img_shape: tuple[int, ...], attn_dim: int = ATTN_DIM, patch_size: int = PATCH_SIZE):
        super().__init__()
        
        if IMG_EMB_TYPE == "vae":
            # Image shape is (C, S, S).
            channels, height, width = img_shape
            self.channels = channels
            self.side = height
            self.patch_size = patch_size
            self.n_grid = height // patch_size
            self.n_tokens = self.n_grid ** 2
            self.img_dim = channels * patch_size ** 2
        
        elif IMG_EMB_TYPE in ("clip", "vae_clip"):
            # Image shape is (1, D) or (N_v+1, D).
            self.img_dim = int(img_shape[-1])

        self.projector = VisionProjector(self.img_dim, attn_dim)

    def spatial_flatten(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Split the visual latent into patch tokens."""
        n = image_tokens.shape[0]
        g, p = self.n_grid, self.patch_size
        tokens = image_tokens.reshape(n, self.channels, g, p, g, p)
        return tokens.permute(0, 2, 4, 1, 3, 5).reshape(n, self.n_tokens, self.img_dim)

    def forward(self, image_tokens: torch.Tensor) -> torch.Tensor:
        """Return the projected visual tokens F_v serving as keys and values."""
        if IMG_EMB_TYPE == "vae":
            # Flatten from (N, C, S, S) to (N, N_v, C).
            image_tokens = self.spatial_flatten(image_tokens)
            # Pooling here rather than in the loader: the latent is stored as a
            # grid, so its tokens only exist once spatial_flatten has made them.
            if IMG_EMB_POOL:
                image_tokens = image_tokens.mean(dim=-2, keepdim=True)
        # Project from (N, N_tok, img_dim) to (N, N_tok, attn_dim).
        return self.projector(image_tokens)


class TextFeaturizer(nn.Module):
    """Project the prompt pair to attn_dim, F_t."""

    def __init__(self, text_dim: int, attn_dim: int = ATTN_DIM):
        super().__init__()
        self.projector = TextProjector(text_dim, attn_dim=attn_dim)

    def forward(self, src: torch.Tensor, tar: torch.Tensor) -> torch.Tensor:
        """Return the prompt queries F_t, source first then target."""
        # Stack and project from (N, N_t, D_txt)^2 to (N, 2, N_t, d).
        return self.projector(torch.stack([src, tar], dim=1))


class TextTokenMLP(nn.Module):
    """Per-token residual MLP on the projected prompt tokens.

    Without it a linear projection followed by the masked mean is the pooled prompt
    embedding, so token-level prompts would add nothing to pooling.
    """

    def __init__(self, attn_dim: int = ATTN_DIM, dropout_rate: float = COMBINER_DROPOUT):
        super().__init__()
        self.norm = nn.LayerNorm(attn_dim)
        self.body = nn.Sequential(
            nn.Linear(attn_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attn_dim, attn_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # (N, 2, N_t, d) -> (N, 2, N_t, d)
        return tokens + self.body(self.norm(tokens))


class MaskedVisionDescriptorizer(nn.Module):
    """Project the masked-image block to one edit descriptor segment, F_m."""

    def __init__(self, feat_dim: int, attn_dim: int = ATTN_DIM, dropout_rate: float = COMBINER_DROPOUT):
        super().__init__()
        # The block mixes a CLIP embedding with cosine and area scalars, whose
        # scales differ by orders of magnitude.
        self.in_norm = nn.LayerNorm(feat_dim)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, attn_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(attn_dim, attn_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return F_m, from (N, D_feat) to (N, d)."""
        return self.proj(self.in_norm(features))


"""
Text grounder and pooler: CrossAttn(Q=F_t, K=F_v, V=F_v)
"""


def text_token_diff_saliency(
    q: torch.Tensor,     # (N, 2, N_t, d), projected but not yet grounded
    masks: torch.Tensor, # (N, 2, N_t) bool
) -> torch.Tensor:       # (N, 2, N_t)
    """Per-token novelty of each prompt against the other, 1 - max cosine."""
    x = torch.nn.functional.normalize(q.float(), dim=-1)
    sim = torch.einsum("nid,njd->nij", x[:, 0], x[:, 1])  # (N, N_t, N_t)
    neg = torch.finfo(sim.dtype).min
    m_src, m_tar = masks[:, 0], masks[:, 1]
    # Each direction ignores the other prompt's padding as a possible match.
    w_src = 1.0 - sim.masked_fill(~m_tar.unsqueeze(1), neg).max(dim=2).values
    w_tar = 1.0 - sim.masked_fill(~m_src.unsqueeze(2), neg).max(dim=1).values
    return (torch.stack([w_src, w_tar], dim=1) * masks).clamp(min=0.0).detach()


class CrossAttentionBlock(nn.Module):
    """One grounding step, CrossAttn(Q=F_t, K=F_v, V=F_v), as a pre-LN block."""

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout_rate: float = ATTN_DROPOUT,
        ffn_mult: float = FFN_MULT,
        layerscale_init: float | None = LAYERSCALE_INIT,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout_rate, batch_first=True)
        self.q_norm = nn.LayerNorm(attn_dim)

        # Setup the layer-scale.
        if layerscale_init is not None:
            self.gamma_attn = nn.Parameter(float(layerscale_init) * torch.ones(attn_dim))
            self.gamma_ffn = nn.Parameter(float(layerscale_init) * torch.ones(attn_dim))
        else:
            self.gamma_attn, self.gamma_ffn = None, None

        # Setup the feed-forward network.
        if ffn_mult > 0:
            n_hidden = max(1, int(round(ffn_mult * attn_dim)))
            self.ffn = nn.Sequential(
                nn.LayerNorm(attn_dim),
                nn.Linear(attn_dim, n_hidden),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(n_hidden, attn_dim),
            )
        else:
            self.ffn = None


    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, d)
        kv: torch.Tensor,       # (N, N_v, d), already prenormed
    ) -> torch.Tensor:          # (N, 2, d)
        
        # Apply the cross-attention.
        grounded, _ = self.attn(self.q_norm(queries), kv, kv, need_weights=False)
        
        # Apply the layer-scale to the grounded queries.
        grounded = self.gamma_attn * grounded if self.gamma_attn is not None else grounded
        queries = queries + grounded
        
        # Apply the feed-forward network.
        if self.ffn is not None:
            update = self.ffn(queries)
            # Apply the layer-scale to the update.
            update = self.gamma_ffn * update if self.gamma_ffn is not None else update
            # Add the update to the queries.
            queries = queries + update
        
        return queries


class TextGrounder(nn.Module):
    """Ground every prompt token in the visual tokens."""

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
            CrossAttentionBlock(attn_dim, n_heads, dropout_rate, ffn_mult, layerscale_init)
            for _ in range(max(1, int(n_layers)))
        ])
        self.out_norm = nn.LayerNorm(attn_dim)
        self.null_token = nn.Parameter(torch.zeros(1, 1, attn_dim))
        # Tells the image keys from the masked-image keys.
        self.key_type = nn.Parameter(torch.zeros(2, 1, attn_dim))

    def forward(
        self,
        queries: torch.Tensor,      # (N, 2, N_t, d)
        tokens: torch.Tensor,       # (N, N_v, d)
        mask_tokens: torch.Tensor,  # (N, N_m, d), N_m may be 0
    ) -> torch.Tensor:              # (N, 2, N_t, d)
        n, _, n_t, d = queries.shape

        # Append the masked-image tokens and the null token to the visual tokens.
        tokens = torch.cat([tokens + self.key_type[0], mask_tokens + self.key_type[1], self.null_token.expand(n, -1, -1)], dim=-2)

        # Repeat the visual tokens for each prompt.
        kv = self.kv_norm(tokens).repeat_interleave(2, dim=0)

        # Reshape the queries to (N * 2, N_t, d) for the cross-attention blocks.
        q = queries.reshape(n * 2, n_t, d)

        # Apply the cross-attention blocks.
        for layer in self.layers:
            q = layer(q, kv)
        return self.out_norm(q).reshape(n, 2, n_t, d)


class TextPooler(nn.Module):
    """Pool grounded prompt tokens over N_t."""

    @staticmethod
    def _pool(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor: 
        """Pool the grounded queries over N_t with weights w."""
        # (N, 2, N_t, d), (N, 2, N_t) to (N, 2, d)
        w = w.to(x.dtype).unsqueeze(-1)
        return (x * w).sum(dim=2) / w.sum(dim=2).clamp(min=1e-6)

    def forward(
        self,
        groundeds: torch.Tensor,                      # (N, 2, N_t, d)
        masks: torch.Tensor,                          # (N, 2, N_t)
        saliencies: torch.Tensor | None = None,       # (N, 2, N_t)
    ) -> tuple[torch.Tensor, torch.Tensor | None]:    # (N, 2, d)
        
        # Pool with masks to address text token padding.
        f_means = self._pool(groundeds, masks)
        # Pool with saliencies to address text token novelty.
        f_diffs = None if saliencies is None else self._pool(groundeds, saliencies)
        
        return f_means, f_diffs


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
        self.n_segments = n_segments
        width = self.n_segments * attn_dim
        # One LayerNorm per segment so concat/difference terms keep their own scale.
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
    """Readout G from the edit descriptor h to one scalar per grid cell."""
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
        img_shape: tuple[int, ...],
        text_dim: int,
        n_cells: int,
        default_cell: int,
        n_targets: int = len(TARGET_COLS),
        attn_dim: int = ATTN_DIM,
        feat_dim: int = 0,
        mask_dim: int = 0,
    ):
        super().__init__()

        self.n_cells = n_cells
        self.n_targets = n_targets
        self.default_cell = default_cell

        # Focus on the difference-aware saliencies between prompt tokens.
        self.use_saliency = bool(USE_DIFF_SALIENCY)

        if USE_DIFF_SALIENCY and TEXT_EMB_POOL:
            raise ValueError("USE_DIFF_SALIENCY needs TEXT_EMB_POOL=false")

        # The masked-image tokens are keys only under attention; otherwise their mean is a segment.
        self.use_mask_tokens = MASK_FEATURES in ("tokens", "both") and IMG_FUSION != "none"

        self.vision_featurizer = VisionFeaturizer(img_shape, attn_dim=attn_dim)
        self.text_featurizer = TextFeaturizer(text_dim, attn_dim=attn_dim)
        if TEXT_TOKEN_MLP:
            self.text_token_mlp = TextTokenMLP(attn_dim=attn_dim)
        if IMG_FUSION == "attn":
            self.text_grounder = TextGrounder(attn_dim=attn_dim)
        else:
            # No grounding: normalize the projected prompt tokens as the grounder would.
            self.text_out_norm = nn.LayerNorm(attn_dim)
        self.text_pooler = TextPooler()
        if self.use_mask_tokens:
            self.mask_featurizer = VisionProjector(mask_dim, attn_dim)

        # Routed here, the block is one more z_edit segment, so it reaches the
        # heads without passing through the attention values.
        if MASK_FEATURES != "none":
            self.masked_vision_descriptorizer = MaskedVisionDescriptorizer(feat_dim, attn_dim=attn_dim)

        # Number of segments in the edit descriptor z_edit.
        n_seg = 6 if self.use_saliency else 4
        n_seg += int(IMG_FUSION == "concat") + int(self.use_mask_tokens and IMG_FUSION == "concat")
        n_seg += int(MASK_FEATURES != "none")

        # Each metric may have its own combiner C_theta to describe it differently.
        if SPLIT_COMBINER:
            self.psnr_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
            self.clip_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
        else:
            self.combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)

        self.psnr_head = make_metric_head(attn_dim, n_cells)
        self.clip_head = make_metric_head(attn_dim, n_cells)

        # Train stats: raw mean surface and the per-column regression scale.
        self.register_buffer("mean_surface", torch.zeros(n_cells, n_targets))
        self.register_buffer("loss_scale", torch.ones(n_targets))
        # Deviation calibration fitted on val after training; identity until then.
        self.register_buffer("mean_pred_surface", torch.zeros(n_cells, n_targets))
        self.register_buffer("dev_scale", torch.ones(n_targets))

    def set_metadata(self, metadata: DatasetMetadata) -> None:
        """Copy train stats onto the module buffers."""
        if metadata.n_cells != self.n_cells:
            raise ValueError(f"Expected {self.n_cells} cells, got {metadata.n_cells}")
        if metadata.default_cell != self.default_cell:
            raise ValueError(f"Expected default_cell {self.default_cell}, got {metadata.default_cell}")
        for stat, buf in ((metadata.mean_surface, self.mean_surface), (metadata.loss_scale, self.loss_scale)):
            t = torch.as_tensor(stat, dtype=buf.dtype, device=buf.device)
            if tuple(t.shape) != tuple(buf.shape):
                raise ValueError(f"Expected stats of shape {tuple(buf.shape)}, got {tuple(t.shape)}")
            buf.copy_(t)

    def set_calibration(self, mean_pred_surface: torch.Tensor, dev_scale: torch.Tensor) -> None:
        """Store the mean predicted raw surface and the per-column deviation slopes."""
        self.mean_pred_surface.copy_(mean_pred_surface.to(self.mean_pred_surface))
        self.dev_scale.copy_(dev_scale.to(self.dev_scale))

    def to_training(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw CLIP/PSNR grid to regression-space deltas."""
        return to_deltas(raw, self.default_cell, self.loss_scale)

    def to_raw(self, values: torch.Tensor) -> torch.Tensor:
        """Regression-space deltas to raw CLIP/PSNR, adding the train-mean default cell."""
        return values * self.loss_scale.to(values) + self.mean_surface[self.default_cell].to(values)

    def to_selector(self, raw: torch.Tensor) -> torch.Tensor:
        """Raw CLIP/PSNR grid to the per-sample normalized deltas phi is scored on."""
        return persample_deltas(raw, self.default_cell)

    def calibrate_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Rescale the deviation from the mean predicted surface by the fitted slopes."""
        mean = self.mean_pred_surface.to(raw)
        return mean + self.dev_scale.to(raw) * (raw - mean)

    def forward(
        self,
        image_tokens: torch.Tensor,     # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,    # (N, N_t, D_txt)
        target_tokens: torch.Tensor,    # (N, N_t, D_txt)
        source_mask: torch.Tensor,      # (N, N_t)
        target_mask: torch.Tensor,      # (N, N_t)
        mask_features: torch.Tensor,    # (N, D_feat)
        mask_tokens: torch.Tensor,      # (N, N_m, D_clip), N_m may be 0
    ) -> torch.Tensor:                  # (N, n_cells, 2)
        """Return per-cell (psnr, clip) deltas in the regression space."""

        # Featurize the image and text tokens.
        fv_tokens = self.vision_featurizer(image_tokens)
        ft_tokens = self.text_featurizer(source_tokens, target_tokens)
        ft_masks = torch.stack([source_mask, target_mask], dim=1)
        if TEXT_TOKEN_MLP:
            ft_tokens = self.text_token_mlp(ft_tokens)
        fm_tokens = self.mask_featurizer(mask_tokens) if self.use_mask_tokens else fv_tokens[:, :0]

        # Compute the text token saliency, if enabled.
        # Measures the novelty of each text token against the other.
        ft_saliencies = text_token_diff_saliency(ft_tokens, ft_masks) if self.use_saliency else None

        # Ground the text tokens in the image and masked-image tokens.
        if IMG_FUSION == "attn":
            groundeds = self.text_grounder(ft_tokens, fv_tokens, fm_tokens)
        else:
            groundeds = self.text_out_norm(ft_tokens)

        # Pool the grounded text tokens with masks and saliencies seperately.
        f_means, f_diffs = self.text_pooler(groundeds, ft_masks, ft_saliencies)
        # Resplit the mean pooled text tokens into source and target.
        f_src, f_tar = f_means[:, 0], f_means[:, 1]

        # Combine the source and target text tokens into a single edit descriptor.
        z_edit = combine_edit_features(f_src, f_tar)    # (N, 4d)
        # If difference-aware saliencies are enabled, add them to the edit descriptor.
        if f_diffs is not None:
            z_edit = torch.cat([z_edit, f_diffs[:, 0], f_diffs[:, 1]], dim=-1)    # (N, 6d)

        # Without attention, the mean projected image (and masked-image) tokens are segments.
        if IMG_FUSION == "concat":
            z_edit = torch.cat([z_edit, fv_tokens.mean(dim=-2)], dim=-1)
            if self.use_mask_tokens:
                z_edit = torch.cat([z_edit, fm_tokens.mean(dim=-2)], dim=-1)

        # If the masked-image block is enabled, add it as a final segment.
        if MASK_FEATURES != "none":
            f_m = self.masked_vision_descriptorizer(mask_features)
            z_edit = torch.cat([z_edit, f_m], dim=-1)

        if SPLIT_COMBINER:
            # Map the difference-aware edit representation with two separate combiners for the heads.
            h_psnr = self.psnr_combiner(z_edit)    # (N, d)
            h_clip = self.clip_combiner(z_edit)    # (N, d)
        else:
            # Map the difference-aware edit representation with a single combiner for the heads.
            h_psnr = h_clip = self.combiner(z_edit)

        # Send the compact edit descriptor(s) into the PSNR and CLIP heads.
        z_psnr = self.psnr_head(h_psnr)    # (N, n_cells)
        z_clip = self.clip_head(h_clip)    # (N, n_cells)

        # Stack the PSNR and CLIP predictions into a single tensor.
        z = torch.stack([z_psnr, z_clip], dim=-1)    # (N, n_cells, 2)
        if LOSS_SCALE == "median_range":
            # Deltas in units of a typical grid range lie in [-1, 1] for almost every cell.
            z = torch.tanh(z / 2.0)
        if HEAD_PARAM == "deviation":
            z = z + self.to_training(self.mean_surface)
        if PIN_DEFAULT_CELL:
            z = pin_default(z, self.default_cell)
        return z


class AttentionModel(nn.Module):
    """Trainable AttentionRegressor sized from precomputed token-table shapes."""

    def __init__(
        self,
        image_shape: tuple[int, ...],
        text_dim: int,
        n_cells: int,
        default_cell: int,
        device: torch.device | str | None = None,
        feat_dim: int = 0,
        mask_dim: int = 0,
    ):
        super().__init__()
        self.regressor = AttentionRegressor(image_shape, text_dim, n_cells, default_cell, feat_dim=feat_dim, mask_dim=mask_dim)
        if device is not None:
            self.regressor.to(device)

    def pred_cells(
        self,
        image_tokens: torch.Tensor,     # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,    # (N, N_t, D_txt)
        target_tokens: torch.Tensor,    # (N, N_t, D_txt)
        source_mask: torch.Tensor,      # (N, N_t)
        target_mask: torch.Tensor,      # (N, N_t)
        mask_features: torch.Tensor,    # (N, D_feat)
        mask_tokens: torch.Tensor,      # (N, N_m, D_clip)
    ) -> torch.Tensor:
        """Predict per-cell (psnr, clip) deltas in the regression space."""
        return self.regressor(image_tokens, source_tokens, target_tokens, source_mask, target_mask, mask_features, mask_tokens)

    def to_raw(self, values: torch.Tensor) -> torch.Tensor:
        return self.regressor.to_raw(values)

    def to_selector(self, raw: torch.Tensor) -> torch.Tensor:
        return self.regressor.to_selector(raw)

    def pred_raw(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Predict per-cell (psnr, clip) in raw CLIP/PSNR; calibrated when SELECT_CALIBRATED."""
        raw = self.to_raw(self.pred_cells(*inputs))
        return self.regressor.calibrate_raw(raw) if SELECT_CALIBRATED else raw
