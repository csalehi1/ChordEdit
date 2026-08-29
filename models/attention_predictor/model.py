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
from scipy.stats import spearmanr

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
        attn_dim: int = ATTN_DIM,
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
        # Prompts arrive pooled to one token each (pipeline masked mean), so the
        # pair concatenates along the token dim and P_t maps each token to d.
        pair = torch.cat([src_emb, tar_emb], dim=-2)
        return self.projector(pair)


"""
CrossAttentionPooler: CrossAttn(Q=F_t, K=F_v, V=F_v).
"""

class CrossAttentionPooler(nn.Module):
    """Ground the prompt queries in the visual tokens, CrossAttn(Q=F_t, K=F_v, V=F_v)."""

    def __init__(
        self,
        attn_dim: int = ATTN_DIM,
        n_heads: int = N_HEADS,
        dropout_rate: float = ATTN_DROPOUT,
    ):
        super().__init__()
        # A single nn.MultiheadAttention layer.
        self.attn = nn.MultiheadAttention(attn_dim, n_heads, dropout=dropout_rate, batch_first=True)

    def forward(
        self,
        queries: torch.Tensor,  # (N, 2, d)
        tokens: torch.Tensor,   # (N, N_v, d)
    ) -> torch.Tensor:          # (N, 2, d)
        """Return the image-grounded prompt features [f_src; f_tar]."""
        grounded, _ = self.attn(queries, tokens, tokens, need_weights=False)
        return grounded


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
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.LayerNorm(4 * attn_dim),
            nn.Linear(4 * attn_dim, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(n_hidden, attn_dim),
        )

    def forward(
        self,
        z_edit: torch.Tensor,  # (N, 4 * d)
    ) -> torch.Tensor:         # (N, d)
        return self.body(z_edit)


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
        img_shape: tuple[int, int, int],
        text_dim: int,
        n_cells: int,
        n_targets: int = len(TARGET_COLS),
        attn_dim: int = ATTN_DIM,
    ):
        super().__init__()
        if n_targets != 2:
            raise ValueError(f"Expected 2 targets (PSNR, CLIP), got {n_targets=}")

        self.n_cells = int(n_cells)
        self.n_targets = n_targets
        # Bounded outputs are only meaningful where the targets are bounded.
        self.bounded = PREDICTION_SPACE == "deltas"

        self.vision_featurizer = VisionFeaturizer(img_shape, attn_dim=attn_dim)
        self.text_featurizer = TextFeaturizer(text_dim, attn_dim=attn_dim)
        self.cross_attn = CrossAttentionPooler(attn_dim=attn_dim)
        self.combiner = TextCombiner(attn_dim=attn_dim)

        # G_PSNR and G_CLIP are independently parameterized readouts of the shared
        # edit descriptor, one scalar per timestep-grid cell.
        self.psnr_head = nn.Linear(attn_dim, n_cells)
        self.clip_head = nn.Linear(attn_dim, n_cells)

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
        img_emb: torch.Tensor,  # (N, C, S, S)
        src_emb: torch.Tensor,  # (N, 1, D_txt)
        tar_emb: torch.Tensor,  # (N, 1, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Return per-cell (psnr, clip) predictions, standardized where active."""
        # Ground both prompts in the source image with one cross-attention pass.
        tokens = self.vision_featurizer(img_emb)
        queries = self.text_featurizer(src_emb, tar_emb)
        grounded = self.cross_attn(queries, tokens)
        f_src, f_tar = grounded[:, 0], grounded[:, 1]

        # Difference-aware edit descriptor, read out by the two metric heads.
        h = self.combiner(combine_edit_features(f_src, f_tar))
        out = torch.stack([self.psnr_head(h), self.clip_head(h)], dim=-1)
        if self.bounded:
            return 2.0 * torch.sigmoid(out) - 1.0
        return out


class AttentionModel(nn.Module):
    """Trainable AttentionRegressor sized from precomputed token-table shapes."""

    def __init__(
        self,
        img_shape: tuple[int, int, int],
        text_dim: int,
        n_cells: int,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.regressor = AttentionRegressor(img_shape, text_dim, n_cells)
        if device is not None:
            self.regressor.to(device)

    def pred_cells(
        self,
        img_emb: torch.Tensor,  # (N, C, S, S)
        src_emb: torch.Tensor,  # (N, 1, D_txt)
        tar_emb: torch.Tensor,  # (N, 1, D_txt)
    ) -> torch.Tensor:          # (N, n_cells, 2)
        """Predict per-cell (psnr, clip) in PREDICTION_SPACE units."""
        return self.regressor.destandardize(self.regressor(img_emb, src_emb, tar_emb))


"""
Selector.
"""

def per_image_spearman(true_grid: np.ndarray, pred_grid: np.ndarray) -> np.ndarray:
    """Rank correlation within each image's timestep grid (over labeled cells)."""
    rhos = []
    for k in range(true_grid.shape[0]):
        t, p = true_grid[k].ravel(), pred_grid[k].ravel()
        labeled = np.isfinite(t) & np.isfinite(p)
        if labeled.sum() < 2:
            rhos.append(np.nan)
            continue
        rho, _ = spearmanr(t[labeled], p[labeled])
        rhos.append(np.nan if rho is None else float(rho))
    return np.array(rhos)


def regret(true_phi: np.ndarray, pred_phi: np.ndarray) -> np.ndarray:
    """True phi at argmax(pred_phi) minus true phi at argmax(true_phi); lower is better."""
    out = np.zeros(true_phi.shape[0])
    for k in range(true_phi.shape[0]):
        chosen = np.unravel_index(np.nanargmax(pred_phi[k]), pred_phi[k].shape)
        out[k] = np.nanmax(true_phi[k]) - true_phi[k][chosen]
    return out


def gate_metrics(
    true_phi: np.ndarray,
    pred_phi: np.ndarray,
    default_i: int,
    default_j: int,
    noise_floor: float,
) -> dict:
    """Deviate-or-default gate: precision/recall for flagged improvable images."""
    default_phi = true_phi[:, default_i, default_j]
    truly_improvable = (np.nanmax(true_phi, axis=(1, 2)) - default_phi) > noise_floor
    pred_gain = np.nanmax(pred_phi, axis=(1, 2)) - pred_phi[:, default_i, default_j]
    flagged = pred_gain > noise_floor
    tp = int(np.sum(flagged & truly_improvable))
    return {
        "precision": tp / max(int(flagged.sum()), 1),
        "recall": tp / max(int(truly_improvable.sum()), 1),
        "n_flagged": int(flagged.sum()),
        "n_improvable": int(truly_improvable.sum()),
        "noise_floor": noise_floor,
    }


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
        default_cells = np.flatnonzero((self._cell_i == self._default_i) & (self._cell_j == self._default_j))
        if default_cells.size != 1:
            raise ValueError(
                f"Expected exactly one cell at the default "
                f"({default_t_start}, {default_t_end}), got {default_cells.size}"
            )
        self._default_k = int(default_cells[0])

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
        img_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> TimestepGridResult:
        """
        Single predictor forward covering every labeled (t_start, t_end) cell.

        Scatters the predictions into the grid, maps them to deltas
        (deltas_from_preds), and scores them with phi (settings.SCORE_PHI).
        Because the predictor estimates the two metric surfaces rather than the
        scalarized objective, re-scoring a saved run under a different SCORE_FN
        needs no retraining.
        """
        device = self.model.regressor.target_mean.device
        # Accept one unbatched sample: (C, S, S) latents and (1, D) prompts.
        img = (img_emb.unsqueeze(0) if img_emb.dim() == 3 else img_emb).to(device)
        src = (src_emb.unsqueeze(0) if src_emb.dim() == 2 else src_emb).to(device)
        tar = (tar_emb.unsqueeze(0) if tar_emb.dim() == 2 else tar_emb).to(device)
        preds = self.model.pred_cells(img, src, tar)

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
    img_shape = tuple(int(v) for v in ckpt["img_shape"])
    text_dim = int(tuple(ckpt["text_shape"])[-1])
    model = AttentionModel(img_shape, text_dim, int(cell_t_pairs.shape[0]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_standardization(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # The checkpoint carries the training grid's axes, so the selector's grid is
    # the data's by construction.
    t_start_values = np.asarray(ckpt["t_start_values"], dtype=np.float64)
    t_end_values = np.asarray(ckpt["t_end_values"], dtype=np.float64)

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
