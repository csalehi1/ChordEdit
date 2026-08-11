# model_t.py

"""
Timestep selector T over the discrete (t_start, t_end) grid.

    Model architecture:
    T(img, src_prompt, tar_prompt) -> (t_start, t_end)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from model_m import SurrogateModel
from _helpers import (
    MEAN_SURFACE_NAME,
    load_mean_surface,
    nearest_indices,
    phi_from_delta_grids,
    resolve_device,
)
from settings import (
    DEFAULT_T_END,
    DEFAULT_T_START,
    M_TARGET_SPACE,
    NOISE_FLOOR_PHI,
)


def mean_surface_from_dict(
    surface: dict,
    t_start_values: np.ndarray,
    t_end_values: np.ndarray,
) -> np.ndarray:
    """mean_surface dict -> (n_start, n_end, 2) mean true delta grid for T.

    Raises when the mean surface's grid axes do not match the selector's, so a
    stale mean surface can never be silently applied to the wrong grid.
    """
    cal_start = np.asarray(surface["t_start_values"], dtype=np.float64)
    cal_end = np.asarray(surface["t_end_values"], dtype=np.float64)
    if not (
        len(cal_start) == len(t_start_values)
        and len(cal_end) == len(t_end_values)
        and np.allclose(cal_start, np.asarray(t_start_values, dtype=np.float64))
        and np.allclose(cal_end, np.asarray(t_end_values, dtype=np.float64))
    ):
        raise ValueError(
            f"mean_surface axes {cal_start.tolist()} x {cal_end.tolist()} do not match "
            f"the selector's grid; recompute it with calibrate_t.py"
        )
    return np.asarray(surface["mean_true_delta"], dtype=np.float64)


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
    """Select (t_start, t_end) via M^ grid predictions; no extra trainable weights."""

    def __init__(
        self,
        surrogate_model: SurrogateModel,
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

    @property
    def n_start(self) -> int:
        return len(self.t_start_values)

    @property
    def n_end(self) -> int:
        return len(self.t_end_values)

    @torch.no_grad()
    def predict_grid_from_emb(
        self,
        img_emb: torch.Tensor,
        mask_emb: torch.Tensor,
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
        t_pairs: np.ndarray | None = None,
    ) -> TimestepGridResult:
        """Batched M^ forward over the grid."""
        device = self.surrogate.regressor.target_mean.device
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
        if mask_emb.dim() == 1:
            mask_emb = mask_emb.unsqueeze(0)
        if src_emb.dim() == 1:
            src_emb = src_emb.unsqueeze(0)
        if tar_emb.dim() == 1:
            tar_emb = tar_emb.unsqueeze(0)

        n1, n2 = self.n_start, self.n_end
        if t_pairs is None:
            tt1, tt2 = np.meshgrid(self.t_start_values, self.t_end_values, indexing="ij")
            t1, t2 = tt1.reshape(-1), tt2.reshape(-1)
            scatter = False
        else:
            t_pairs = np.asarray(t_pairs, dtype=np.float64)
            if t_pairs.ndim != 2 or t_pairs.shape[1] != 2:
                raise ValueError(f"t_pairs must have shape (N, 2), got {t_pairs.shape}")
            t1, t2 = t_pairs[:, 0], t_pairs[:, 1]
            scatter = True

        # Expand the embeddings to the number of cells.
        n_cells = len(t1)
        img = img_emb.to(device).expand(n_cells, -1)
        mask = mask_emb.to(device).expand(n_cells, -1)
        src = src_emb.to(device).expand(n_cells, -1)
        tar = tar_emb.to(device).expand(n_cells, -1)
        t = torch.tensor(np.stack([t1, t2], axis=1), dtype=torch.float, device=device)
        pred = self.surrogate.predict_emb(img, mask, src, tar, t).cpu().numpy()

        # Reshape the predictions into a grid.
        if not scatter:
            psnr = pred[:, 0].reshape(n1, n2)
            clip = pred[:, 1].reshape(n1, n2)
        else:
            i = nearest_indices(self.t_start_values, t1)
            j = nearest_indices(self.t_end_values, t2)
            psnr = np.full((n1, n2), np.nan)
            clip = np.full((n1, n2), np.nan)
            psnr[i, j] = pred[:, 0]
            clip[i, j] = pred[:, 1]

        # The towers predict per-sample normalized deltas (minus the mean
        # surface in the "residual" target space); adding the surface back
        # gives the full deltas phi consumes, so the predicted grid is never
        # renormalized. Axis cells the surface leaves NaN (unlabeled during
        # training) stay NaN, so selection never picks a cell it cannot pin.
        if self.mean_surface is not None:
            psnr = psnr + self.mean_surface[..., 0]
            clip = clip + self.mean_surface[..., 1]
        deltas = np.stack([psnr, clip], axis=-1)[None]
        phi = phi_from_delta_grids(deltas)[0]
        return TimestepGridResult(psnr, clip, phi, self.t_start_values, self.t_end_values)

    # @torch.no_grad()
    # def predict_grid(self, image, mask, src_prompt: str, tar_prompt: str) -> TimestepGridResult:
    #     img_emb, mask_emb, src_emb, tar_emb = self.surrogate.encode([image], [mask], [src_prompt], [tar_prompt])
    #     return self.predict_grid_from_emb(img_emb, mask_emb, src_emb, tar_emb)

    def select_from_grid(
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

    # @torch.no_grad()
    # def predict(
    #     self,
    #     image,
    #     mask,
    #     src_prompt: str,
    #     tar_prompt: str,
    #     noise_floor: float = NOISE_FLOOR_PHI,
    # ) -> TimestepSelection:
    #     grid = self.predict_grid(image, mask, src_prompt, tar_prompt)
    #     return self.select_from_grid(grid, noise_floor=noise_floor)


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

    # Load the surrogate model.
    model = SurrogateModel(int(ckpt["img_dim"]), int(ckpt["text_dim"]), device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_stats(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()

    # Load the mean surface.
    surface = load_mean_surface(weights_path.parent)
    if surface is None:
        raise FileNotFoundError(f"Missing T mean surface: {weights_path.parent / MEAN_SURFACE_NAME}")
    if str(surface.get("target_space")) != M_TARGET_SPACE:
        raise ValueError(f"Mismatch for target space: {surface.get('target_space')!r}.")
    
    # The mean surface carries the training grid's axes, so the selector's grid is the data's by construction.
    t_start_values = np.asarray(surface["t_start_values"], dtype=np.float64)
    t_end_values = np.asarray(surface["t_end_values"], dtype=np.float64)
    mean_surface = (
        mean_surface_from_dict(surface, t_start_values, t_end_values)
        if M_TARGET_SPACE == "residual" else None
    )
    print(f"Loaded mean_surface ({surface['split']} split, {surface['n_samples']} samples): T selects {'residual + mean surface' if mean_surface is not None else 'predicted deltas'}.")
    return TimestepSelector(model, t_start_values=t_start_values, t_end_values=t_end_values, mean_surface=mean_surface)
