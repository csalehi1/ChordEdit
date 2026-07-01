"""
Timestep selector T over the discrete (t_start, t_end) grid.

Given (img, c_src, c_tar), T predicts the best (t_start, t_end) by batched
forward through metric surrogate M, scalarizing (PSNR, CLIP) into m, argmax,
and applying a deviate-or-default gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model_m import MetricPredictor
from settings import (
    DEFAULT_T_END,
    DEFAULT_T_START,
    GRID_T_END,
    GRID_T_START,
    NOISE_FLOOR_M,
    W_CLIP,
    W_PSNR,
)


@dataclass(frozen=True)
class ScalarStats:
    psnr_mean: float
    psnr_std: float
    clip_mean: float
    clip_std: float


@dataclass
class TimestepGridResult:
    psnr_grid: np.ndarray
    clip_grid: np.ndarray
    m_grid: np.ndarray
    t_start_values: np.ndarray
    t_end_values: np.ndarray


@dataclass
class TimestepSelection:
    t_start: float
    t_end: float
    deviate: bool
    pred_gain: float
    m_grid: np.ndarray
    psnr_grid: np.ndarray
    clip_grid: np.ndarray


def scalarize(
    psnr: np.ndarray,
    clip: np.ndarray,
    stats: ScalarStats | None = None,
    w_psnr: float = W_PSNR,
    w_clip: float = W_CLIP,
) -> np.ndarray:
    """Combine PSNR and CLIP into scalar m (z-scored, weighted sum)."""
    if stats is None:
        zp = (psnr - psnr.mean()) / psnr.std()
        zc = (clip - clip.mean()) / clip.std()
    else:
        zp = (psnr - stats.psnr_mean) / stats.psnr_std
        zc = (clip - stats.clip_mean) / stats.clip_std
    return w_psnr * zp + w_clip * zc


def _nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(values - target)))


class TimestepPredictor:
    """Select (t_start, t_end) via M grid predictions — no extra trainable weights."""

    def __init__(
        self,
        metric_predictor: MetricPredictor,
        t_start_values: tuple[float, ...] | list[float] | np.ndarray = GRID_T_START,
        t_end_values: tuple[float, ...] | list[float] | np.ndarray = GRID_T_END,
        scalar_stats: ScalarStats | None = None,
        default_t_start: float = DEFAULT_T_START,
        default_t_end: float = DEFAULT_T_END,
    ):
        self.m = metric_predictor
        self.t_start_values = np.asarray(t_start_values, dtype=np.float64)
        self.t_end_values = np.asarray(t_end_values, dtype=np.float64)
        self.scalar_stats = scalar_stats
        self.default_t_start = default_t_start
        self.default_t_end = default_t_end
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
        src_emb: torch.Tensor,
        tar_emb: torch.Tensor,
    ) -> TimestepGridResult:
        """Batched M forward over the full grid; embeddings shape (1, D) or (D,)."""
        device = self.m.regressor.target_mean.device
        if img_emb.dim() == 1:
            img_emb = img_emb.unsqueeze(0)
        if src_emb.dim() == 1:
            src_emb = src_emb.unsqueeze(0)
        if tar_emb.dim() == 1:
            tar_emb = tar_emb.unsqueeze(0)

        # One batched M forward pass over all grid cells.
        tt1, tt2 = np.meshgrid(self.t_start_values, self.t_end_values, indexing="ij")
        t1, t2 = tt1.reshape(-1), tt2.reshape(-1)
        n_cells = len(t1)
        img = img_emb.expand(n_cells, -1)
        src = src_emb.expand(n_cells, -1)
        tar = tar_emb.expand(n_cells, -1)
        t = torch.tensor(np.stack([t1, t2], axis=1), dtype=torch.float, device=device)
        pred = self.m.predict_metrics_from_emb(img, src, tar, t).cpu().numpy()
        n1, n2 = self.n_start, self.n_end
        psnr = pred[:, 0].reshape(n1, n2)
        clip = pred[:, 1].reshape(n1, n2)
        m = scalarize(psnr, clip, self.scalar_stats)
        return TimestepGridResult(psnr, clip, m, self.t_start_values, self.t_end_values)

    @torch.no_grad()
    def predict_grid(self, image, src_prompt: str, tar_prompt: str) -> TimestepGridResult:
        img_emb, src_emb, tar_emb = self.m.encode(image, src_prompt, tar_prompt)
        return self.predict_grid_from_emb(img_emb, src_emb, tar_emb)

    def select_from_grid(self, grid: TimestepGridResult, noise_floor: float = NOISE_FLOOR_M) -> TimestepSelection:
        """Argmax m_grid with deviate-or-default gate."""
        m = grid.m_grid
        i, j = np.unravel_index(m.argmax(), m.shape)
        pred_gain = float(m.max() - m[self._default_i, self._default_j])
        # Deviate only when predicted gain exceeds the label noise floor.
        if pred_gain <= noise_floor:
            return TimestepSelection(
                t_start=self.default_t_start,
                t_end=self.default_t_end,
                deviate=False,
                pred_gain=pred_gain,
                m_grid=m,
                psnr_grid=grid.psnr_grid,
                clip_grid=grid.clip_grid,
            )
        return TimestepSelection(
            t_start=float(self.t_start_values[i]),
            t_end=float(self.t_end_values[j]),
            deviate=True,
            pred_gain=pred_gain,
            m_grid=m,
            psnr_grid=grid.psnr_grid,
            clip_grid=grid.clip_grid,
        )

    @torch.no_grad()
    def predict(
        self,
        image,
        src_prompt: str,
        tar_prompt: str,
        noise_floor: float = NOISE_FLOOR_M,
    ) -> TimestepSelection:
        grid = self.predict_grid(image, src_prompt, tar_prompt)
        return self.select_from_grid(grid, noise_floor=noise_floor)


def load_timestep_predictor(
    weights_path: Path | str,
    device: torch.device | str | None = None,
    scalar_stats: ScalarStats | None = None,
) -> TimestepPredictor:
    """Load M checkpoint and wrap with T."""
    weights_path = Path(weights_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = MetricPredictor(freeze_encoders=True, device=device)
    model.regressor.load_state_dict(ckpt["regressor_state_dict"])
    model.regressor.set_target_stats(ckpt["target_mean"], ckpt["target_std"])
    model.regressor.to(device).eval()
    # Restore scalarization stats saved during m_train.py when available.
    if scalar_stats is None and "scalar_stats" in ckpt:
        pm, ps, cm, cs = ckpt["scalar_stats"]
        scalar_stats = ScalarStats(float(pm), float(ps), float(cm), float(cs))
    return TimestepPredictor(model, scalar_stats=scalar_stats)
