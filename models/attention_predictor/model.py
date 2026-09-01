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

    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, N_t, d)
        tokens: torch.Tensor,   # (N, N_v, d)
    ) -> torch.Tensor:          # (N, 2, N_t, d)
        n, _, n_t, d = queries.shape

        # Append the null token to the visual tokens.
        tokens = torch.cat([tokens, self.null_token.expand(n, -1, -1)], dim=-2)
        
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
    ):
        super().__init__()

        self.n_cells = n_cells
        self.n_targets = n_targets
        self.default_cell = default_cell

        # Focus on the difference-aware saliencies between prompt tokens.
        self.use_saliency = bool(USE_DIFF_SALIENCY)
        
        if USE_DIFF_SALIENCY and TEXT_EMB_TYPE != "tokens":
            raise ValueError("USE_DIFF_SALIENCY needs TEXT_EMB_TYPE='tokens'")

        self.vision_featurizer = VisionFeaturizer(img_shape, attn_dim=attn_dim)
        self.text_featurizer = TextFeaturizer(text_dim, attn_dim=attn_dim)
        self.text_grounder = TextGrounder(attn_dim=attn_dim)
        self.text_pooler = TextPooler()

        # Number of segments in the edit descriptor z_edit.
        n_seg = 6 if self.use_saliency else 4

        # Each metric may have its own combiner C_theta to describe it differently.
        if SPLIT_COMBINER:
            self.psnr_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
            self.clip_combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)
        else:
            self.combiner = TextCombiner(attn_dim=attn_dim, n_segments=n_seg)

        self.psnr_head = make_metric_head(attn_dim, n_cells)
        self.clip_head = make_metric_head(attn_dim, n_cells)

        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def destandardize(self, standardized: torch.Tensor) -> torch.Tensor:
        """Undo the target z-scoring, which is identity unless the space standardizes."""
        return standardized * self.target_std + self.target_mean

    def set_target_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Store the train target mean/std that forward() preds in for PREDICTION_SPACE "raws"."""
        mean = torch.as_tensor(mean, dtype=self.target_mean.dtype, device=self.target_mean.device).reshape(-1)
        std = torch.as_tensor(std, dtype=self.target_std.dtype, device=self.target_std.device).reshape(-1)
        if mean.numel() != self.n_targets or std.numel() != self.n_targets:
            raise ValueError(f"Expected {self.n_targets} target stats, got {mean.numel()} / {std.numel()}")
        self.target_mean.copy_(mean)
        # A constant target column would otherwise divide by zero.
        self.target_std.copy_(std.clamp(min=1e-8))

    def forward(
        self,
        image_tokens: torch.Tensor,     # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,    # (N, N_t, D_txt)
        target_tokens: torch.Tensor,    # (N, N_t, D_txt)
        source_mask: torch.Tensor,      # (N, N_t)
        target_mask: torch.Tensor,      # (N, N_t)
    ) -> torch.Tensor:                  # (N, n_cells, 2)
        """Return per-cell (psnr, clip) predictions, standardized where active."""
        
        # Featurize the image and text tokens.
        fv_tokens = self.vision_featurizer(image_tokens)
        ft_tokens = self.text_featurizer(source_tokens, target_tokens)
        ft_masks = torch.stack([source_mask, target_mask], dim=1)

        # Compute the text token saliency, if enabled.
        # Measures the novelty of each text token against the other.
        ft_saliencies = text_token_diff_saliency(ft_tokens, ft_masks) if self.use_saliency else None
        
        # Ground the text tokens in the image tokens.
        groundeds = self.text_grounder(ft_tokens, fv_tokens)

        # Pool the grounded text tokens with masks and saliencies seperately.
        f_means, f_diffs = self.text_pooler(groundeds, ft_masks, ft_saliencies)
        # Resplit the mean pooled text tokens into source and target.
        f_src, f_tar = f_means[:, 0], f_means[:, 1]

        # Combine the source and target text tokens into a single edit descriptor.
        z_edit = combine_edit_features(f_src, f_tar)    # (N, 4d)
        # If difference-aware saliencies are enabled, add them to the edit descriptor.
        if f_diffs is not None:
            z_edit = torch.cat([z_edit, f_diffs[:, 0], f_diffs[:, 1]], dim=-1)    # (N, 6d)
        
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
       
        # If the pred space is bounded, apply the sigmoid function to the preds.
        if PREDICTION_SPACE == "deltas":
            z = 2.0 * torch.sigmoid(z) - 1.0
            # Would be delta_hat to match the paper's notation.

        if PIN_DEFAULT_CELL:
            # Pin the default cell to Delta=0.
            z = z - z[:, self.default_cell : self.default_cell + 1, :]

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
    ):
        super().__init__()
        self.regressor = AttentionRegressor(image_shape, text_dim, n_cells, default_cell)
        if device is not None:
            self.regressor.to(device)

    def pred_cells(
        self,
        image_tokens: torch.Tensor,     # (N, C, S, S) or (N, N_tok, D)
        source_tokens: torch.Tensor,    # (N, N_t, D_txt)
        target_tokens: torch.Tensor,    # (N, N_t, D_txt)
        source_mask: torch.Tensor,      # (N, N_t)
        target_mask: torch.Tensor,      # (N, N_t)
    ) -> torch.Tensor:                  # (N, n_cells, 2)
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
        source_tokens: torch.Tensor,              # (1, D_txt) or (N_t, D_txt)
        target_tokens: torch.Tensor,              # like source_tokens
        source_mask: torch.Tensor,                # (N_t,) bool
        target_mask: torch.Tensor,                # (N_t,) bool
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
        batched = lambda t: t.unsqueeze(0).to(device)
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
        # The pin uses this index, so rebuild it from the checkpoint's grid axes.
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
