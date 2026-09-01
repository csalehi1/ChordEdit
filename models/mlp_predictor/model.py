# model.py

"""
Surrogate model M^ and the timestep selector T over its predicted grid.

    Model architecture:
    M(img_emb, src_emb, tar_emb) -> (n_cells, 2) grid of (psnr, clip)
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
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
from settings import *


def pairwise_ranking_loss(pred: torch.Tensor, true: torch.Tensor, top_k: int = 0) -> torch.Tensor:
    """Logistic pairwise loss to penalize predictions that disagree with the ground truth."""
    if pred.shape[-1] < 2:
        return pred.new_zeros(())
    diff_true = true.unsqueeze(-1) - true.unsqueeze(-2)
    diff_pred = pred.unsqueeze(-1) - pred.unsqueeze(-2)
    mask = diff_true > 0
    if top_k and top_k < true.shape[-1]:
        idx = true.topk(top_k, dim=-1).indices
        is_top = torch.zeros_like(true, dtype=torch.bool).scatter_(-1, idx, True)
        mask = mask & is_top.unsqueeze(-1)
    if not mask.any():
        return pred.new_zeros(())
    return torch.nn.functional.softplus(-diff_pred[mask]).mean()


"""
VisionFeaturizer: spatial flattening + VisionProjector (P_v) -> F_v.
"""

class VisionProjector(nn.Module):
    """Project flattened visual tokens to attn_dim, P_v."""

    def __init__(
        self,
        token_dim: int,
        attn_dim: int,
        n_tokens: int,
        use_pos_emb: bool = USE_POS_EMB,
    ):
        super().__init__()
        self.proj = nn.Linear(token_dim, attn_dim)
        self.pos_emb = nn.Parameter(torch.zeros(n_tokens, attn_dim)) if use_pos_emb else None

    def forward(
        self,
        tokens: torch.Tensor,  # (N, N_v, token_dim)
    ) -> torch.Tensor:         # (N, N_v, d)
        tokens = self.proj(tokens)
        if self.pos_emb is not None:
            tokens = tokens + self.pos_emb
        return tokens


class VisionFeaturizer(nn.Module):
    """Spatially flatten the VAE latent, then VisionProjector -> F_v."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        attn_dim: int,
        patch_size: int = PATCH_SIZE,
        use_pos_emb: bool = USE_POS_EMB,
    ):
        super().__init__()
        channels, height, width = img_shape

        if height != width:
            raise ValueError(f"Expected {img_shape} to be a square")
        if patch_size < 1 or height % patch_size != 0:
            raise ValueError(f"Expected {patch_size} to divide {height}")

        self.channels = channels
        self.side = height
        self.patch_size = patch_size
        self.n_grid = height // patch_size
        self.n_tokens = self.n_grid ** 2
        self.token_dim = channels * patch_size ** 2
        self.projector = VisionProjector(
            self.token_dim, attn_dim, self.n_tokens, use_pos_emb=use_pos_emb,
        )

    def spatial_flatten(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
    ) -> torch.Tensor:          # (N, N_v, token_dim)
        """Split the latent into patch tokens; patch_size 1 is one token per position."""
        n = img_emb.shape[0]
        if tuple(img_emb.shape[1:]) != (self.channels, self.side, self.side):
            raise ValueError(f"Expected {img_emb.shape} == (N, {self.channels}, {self.side}, {self.side})")
        g, p = self.n_grid, self.patch_size
        tokens = img_emb.reshape(n, self.channels, g, p, g, p)
        return tokens.permute(0, 2, 4, 1, 3, 5).reshape(n, self.n_tokens, self.token_dim)

    def forward(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
    ) -> torch.Tensor:          # (N, N_v, d)
        """Return the projected visual tokens F_v serving as keys and values."""
        return self.projector(self.spatial_flatten(img_emb))


"""
TextFeaturizer: token-dim concatenation + TextProjector (P_t) -> F_t.
"""

class TextProjector(nn.Module):
    """Project each prompt token, P_t: R^{D} -> R^{d}, shared across the pair."""

    def __init__(
        self,
        text_dim: int,
        attn_dim: int,
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
        attn_dim: int,
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
VisualProjector: VisionFeaturizer -> mean pool over the token dim.
"""

class VisualProjector(nn.Module):
    """Mean-pool the visual tokens F_v to one (1, d_v) vector.

    attention_predictor keeps all N_v tokens as cross-attention keys and values.
    These towers take a single feature vector, so the token dim is averaged away
    here rather than inside the copied featurizer, which stays identical to the
    attention model's.
    """

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        attn_dim: int,
        patch_size: int = PATCH_SIZE,
        use_pos_emb: bool = USE_POS_EMB,
    ):
        super().__init__()
        self.featurizer = VisionFeaturizer(
            img_shape, attn_dim, patch_size=patch_size, use_pos_emb=use_pos_emb,
        )
        self.attn_dim = attn_dim

    def forward(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
    ) -> torch.Tensor:          # (N, 1, d_v)
        """Return the pooled visual feature, keeping the token dim."""
        return self.featurizer(img_emb).mean(dim=-2, keepdim=True)



"""
MLP.
"""

class MLPBlock(nn.Module):

    def __init__(self, n_in: int, n_out: int, dropout_rate: float):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.norm = nn.LayerNorm(n_out)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.norm(self.linear(x)))
        return self.dropout(h)


class MLPBody(nn.Module):

    def __init__(self, n_in: int, n_wide: int, n_hidden: int, n_out: int, dropout_rate: float):
        super().__init__()
        self.blocks = nn.ModuleList([
            MLPBlock(n_in, n_wide, dropout_rate),
            MLPBlock(n_wide, n_hidden, dropout_rate),
            MLPBlock(n_hidden, n_out, dropout_rate),
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


"""
Surrogate Model.
"""

class SurrogateRegressor(nn.Module):
    """PSNR-Unedited and CLIP-Edited MLP towers, each predicting all grid cells at once."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        text_dim: int,
        n_cells: int,
        n_targets: int = len(TARGET_COLS),
        img_proj_dim: int = IMG_PROJ_DIM,
        text_proj_dim: int = TEXT_PROJ_DIM,
        n_wide: int = MLP_WIDE,
        n_hidden: int = MLP_HIDDEN,
        n_inner: int = MLP_INNER,
        psnr_dropout_rate: float = MLP_PSNR_DROPOUT,
        clip_dropout_rate: float = MLP_CLIP_DROPOUT,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError("SurrogateRegressor expects exactly two targets (PSNR, CLIP)")

        # The visual path is the attention model's VisionFeaturizer, so it needs
        # the (C, S, S) latent grid rather than a flat embedding.
        self.img_emb_source = str(IMG_EMB_TYPE)
        if self.img_emb_source != "vae":
            raise ValueError(
                f"{IMG_EMB_TYPE=} has no latent grid for VisionFeaturizer; expected 'vae'"
            )

        # F_v mean-pooled to (1, d_v) and F_t as (2, d_t), both from the
        # featurizers attention_predictor uses.
        self.img_proj = VisualProjector(img_shape, img_proj_dim)
        self.text_proj = TextFeaturizer(text_dim, attn_dim=text_proj_dim)

        # Build the PSNR and CLIP towers over [pooled F_v; F_t] flattened.
        n_features = img_proj_dim + 2 * text_proj_dim
        self.psnr_body = MLPBody(n_features, n_wide, n_hidden, n_inner, psnr_dropout_rate)
        self.psnr_head = nn.Linear(n_inner, n_cells)
        self.clip_body = MLPBody(n_features, n_wide, n_hidden, n_inner, clip_dropout_rate)
        self.clip_head = nn.Linear(n_inner, n_cells)

        # Register the target mean and std for denormalization.
        self.register_buffer("target_mean", torch.zeros(n_targets))
        self.register_buffer("target_std", torch.ones(n_targets))

    def destandardize(self, standardized: torch.Tensor) -> torch.Tensor:
        return standardized * self.target_std + self.target_mean

    def set_target_standardization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.target_mean.copy_(mean.to(self.target_mean))
        self.target_std.copy_(std.to(self.target_std).clamp(min=1e-8))

    def forward(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
        src_emb: torch.Tensor,  # (N, 1, D_txt)
        tar_emb: torch.Tensor,  # (N, 1, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Return standardized per-cell metric predictions.

        Featurizes both modalities the way the attention model does, then
        flattens the pooled visual token and the two prompt tokens into one
        (d_v + 2 * d_t) vector for the towers.
        """
        visual = self.img_proj(img_emb)              # (N, 1, d_v)
        text = self.text_proj(src_emb, tar_emb)      # (N, 2, d_t)
        x = torch.cat([visual.flatten(1), text.flatten(1)], dim=-1)
        psnr = self.psnr_head(self.psnr_body(x))
        clip = self.clip_head(self.clip_body(x))
        return torch.stack([psnr, clip], dim=-1)


class SurrogateModel(nn.Module):
    """Trainable SurrogateRegressor sized from precomputed embedding dims."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        text_dim: int,
        n_cells: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.regressor = SurrogateRegressor(
            img_shape=img_shape,
            text_dim=text_dim,
            n_cells=n_cells,
            n_targets=len(TARGET_COLS),
        )
        if device is not None:
            self.to(device)

    def pred_emb(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
        src_emb: torch.Tensor,  # (N, 1, D_txt)
        tar_emb: torch.Tensor,  # (N, 1, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Predict per-cell (psnr, clip) in raw metric units."""
        out = self.regressor(img_emb, src_emb, tar_emb)
        return self.regressor.destandardize(out)


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
    """Select (t_start, t_end) via M^ grid predictions; no extra trainable weights."""

    def __init__(
        self,
        surrogate_model: SurrogateModel,
        cell_t_pairs: np.ndarray,
        t_start_values: tuple[float, ...] | list[float] | np.ndarray,
        t_end_values: tuple[float, ...] | list[float] | np.ndarray,
        default_t_start: float = DEFAULT_T_START,
        default_t_end: float = DEFAULT_T_END,
        mean_surface: np.ndarray | None = None,
    ):
        self.surrogate = surrogate_model
        self.t_start_values = np.asarray(t_start_values, dtype=np.float64)
        self.t_end_values = np.asarray(t_end_values, dtype=np.float64)
        self.default_t_start = default_t_start
        self.default_t_end = default_t_end
        self.mean_surface = None if mean_surface is None else np.asarray(mean_surface, dtype=np.float64)
        self._default_i = _nearest_index(self.t_start_values, default_t_start)
        self._default_j = _nearest_index(self.t_end_values, default_t_end)
        # Grid positions of the model's output cells, in the training cell order.
        cell_t_pairs = np.asarray(cell_t_pairs, dtype=np.float64)
        self._cell_i = nearest_indices(self.t_start_values, cell_t_pairs[:, 0])
        self._cell_j = nearest_indices(self.t_end_values, cell_t_pairs[:, 1])

    @property
    def n_start(self) -> int:
        return len(self.t_start_values)

    @property
    def n_end(self) -> int:
        return len(self.t_end_values)

    @torch.no_grad()
    def pred_grid(
        self,
        img_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> TimestepGridResult:
        """Single M^ forward predicting every labeled (t_start, t_end) cell."""

        device = self.surrogate.regressor.target_mean.device
        # Accept one unbatched sample: (C, S, S) latents and (1, D) prompts.
        img = (img_emb.unsqueeze(0) if img_emb.dim() == 3 else img_emb).to(device)
        src = (src_emb.unsqueeze(0) if src_emb.dim() == 2 else src_emb).to(device)
        tar = (tar_emb.unsqueeze(0) if tar_emb.dim() == 2 else tar_emb).to(device)
        pred = self.surrogate.pred_emb(img, src, tar)[0].cpu().numpy()

        # Scatter the per-cell preds into the grid.
        deltas = np.full((self.n_start, self.n_end, 2), np.nan)
        deltas[self._cell_i, self._cell_j] = pred
        if self.mean_surface is not None:
            # For residual space, add the train mean surface back.
            deltas = deltas + self.mean_surface
        psnr, clip = deltas[..., 0], deltas[..., 1]
        phi = phi_from_delta_grids(deltas[None])[0]
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
            raise ValueError("phi_grid has no finite cells")

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
    """Load the surrogate model and wrap with the selector model."""
    
    weights_path = Path(weights_path)
    if device is None:
        device = resolve_device(gpu)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    # Load the surrogate model. The checkpoint's cell_t_pairs give the output
    # cell count and each cell's (t_start, t_end) grid position.
    cell_t_pairs = ckpt["cell_t_pairs"].cpu().numpy()
    img_shape = tuple(int(v) for v in ckpt["img_shape"])
    text_dim = int(tuple(ckpt["text_shape"])[-1])
    model = SurrogateModel(img_shape, text_dim, int(cell_t_pairs.shape[0]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # Load the mean surface.
    surface = load_mean_surface(weights_path.parent)
    if surface is None:
        raise FileNotFoundError(f"Missing mean surface: {weights_path.parent / MEAN_SURFACE_NAME}")
    if str(surface.get("prediction_space")) != PREDICTION_SPACE:
        raise ValueError(f"Expected {surface.get('prediction_space')!r} == {PREDICTION_SPACE!r}")
    
    # The mean surface carries the training grid's axes, so the selector's grid is the data's by construction.
    t_start_values = np.asarray(surface["t_start_values"], dtype=np.float64)
    t_end_values = np.asarray(surface["t_end_values"], dtype=np.float64)
    mean_surface = (
        mean_surface_from_dict(surface, t_start_values, t_end_values)
        if PREDICTION_SPACE == "residuals" else None
    )
    print(f"Loaded mean_surface ({surface['split']} split, {surface['n_samples']} samples): T selects {'residual + mean surface' if mean_surface is not None else 'predicted deltas'}.")
    return TimestepSelector(model, cell_t_pairs, t_start_values=t_start_values, t_end_values=t_end_values, mean_surface=mean_surface)
