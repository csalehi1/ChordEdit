"""
Sparse grid labeling policies and active acquisition for sublinear render budgets.

Policies choose which (t_start, t_end) cells to label per edit instance.
Active acquisition scores unlabeled cells using M predictions and interpolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from settings import DEFAULT_T_END, DEFAULT_T_START, GRID_T_END, GRID_T_START


@dataclass(frozen=True)
class GridCell:
    t_start: float
    t_end: float


def _grid_arrays() -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(GRID_T_START, dtype=np.float64), np.asarray(GRID_T_END, dtype=np.float64)


def default_cell() -> GridCell:
    return GridCell(DEFAULT_T_START, DEFAULT_T_END)


def policy_default_only() -> list[GridCell]:
    """k=1: label only the paper-default cell."""
    return [default_cell()]


def policy_1d_sweeps(
    fixed_t_end: float = DEFAULT_T_END,
    fixed_t_start: float = DEFAULT_T_START,
) -> list[GridCell]:
    """~21 cells: default + t_start sweep at fixed t_end + t_end sweep at fixed t_start."""
    t_start_vals, t_end_vals = _grid_arrays()
    cells: dict[tuple[float, float], GridCell] = {}
    for ts in t_start_vals:
        cells[(float(ts), fixed_t_end)] = GridCell(float(ts), fixed_t_end)
    for te in t_end_vals:
        cells[(fixed_t_start, float(te))] = GridCell(fixed_t_start, float(te))
    return list(cells.values())


def policy_structured_5x5() -> list[GridCell]:
    """25 cells on a coarse 5×5 subgrid of the full 11×11 lattice."""
    t_start_vals, t_end_vals = _grid_arrays()
    idx = np.linspace(0, len(t_start_vals) - 1, 5, dtype=int)
    cells = []
    for i in idx:
        for j in idx:
            cells.append(GridCell(float(t_start_vals[i]), float(t_end_vals[j])))
    return cells


def cells_to_mask(
    cells: Iterable[GridCell],
    t_start_vals: np.ndarray | None = None,
    t_end_vals: np.ndarray | None = None,
) -> np.ndarray:
    """Boolean mask (n_start, n_end) for labeled cells."""
    if t_start_vals is None or t_end_vals is None:
        t_start_vals, t_end_vals = _grid_arrays()
    mask = np.zeros((len(t_start_vals), len(t_end_vals)), dtype=bool)
    i_of = {v: i for i, v in enumerate(t_start_vals)}
    j_of = {v: j for j, v in enumerate(t_end_vals)}
    for cell in cells:
        i = i_of.get(cell.t_start)
        j = j_of.get(cell.t_end)
        if i is not None and j is not None:
            mask[i, j] = True
    return mask


def separable_interpolate_grid(
    t_start_labeled: np.ndarray,
    t_end_labeled: np.ndarray,
    psnr_labeled: np.ndarray,
    t_start_vals: np.ndarray,
    t_end_vals: np.ndarray,
) -> np.ndarray:
    """Fill 11×11 PSNR (or CLIP) surface from 1D slice labels via separable interpolation."""
    grid = np.full((len(t_start_vals), len(t_end_vals)), np.nan)
    for ts, te, val in zip(t_start_labeled, t_end_labeled, psnr_labeled):
        i = int(np.argmin(np.abs(t_start_vals - ts)))
        j = int(np.argmin(np.abs(t_end_vals - te)))
        grid[i, j] = val
    for i in range(len(t_start_vals)):
        row_known = ~np.isnan(grid[i])
        if row_known.any():
            grid[i] = np.where(row_known, grid[i], np.nanmean(grid[i, row_known]))
    for j in range(len(t_end_vals)):
        col_known = ~np.isnan(grid[:, j])
        if col_known.any():
            grid[:, j] = np.where(col_known, grid[:, j], np.nanmean(grid[col_known, j]))
    if np.isnan(grid).any():
        fill = np.nanmean(grid)
        grid = np.where(np.isnan(grid), fill, grid)
    return grid


def acquisition_interpolation_residual(
    m_pred: np.ndarray,
    labeled_mask: np.ndarray,
    t_start_vals: np.ndarray,
    t_end_vals: np.ndarray,
    psnr_labeled: np.ndarray | None = None,
    clip_labeled: np.ndarray | None = None,
) -> np.ndarray:
    """
    Score unlabeled cells by |m_pred - m_interp| where m_interp comes from
    separable interpolation on labeled 1D slices (when available).
    """
    scores = np.zeros_like(m_pred)
    if psnr_labeled is not None and clip_labeled is not None:
        # Build coarse interp from labeled points only
        ts, te, pv, cv = [], [], [], []
        for i in range(len(t_start_vals)):
            for j in range(len(t_end_vals)):
                if labeled_mask[i, j]:
                    ts.append(t_start_vals[i])
                    te.append(t_end_vals[j])
                    pv.append(psnr_labeled[i, j])
                    cv.append(clip_labeled[i, j])
        if ts:
            psnr_i = separable_interpolate_grid(
                np.array(ts), np.array(te), np.array(pv), t_start_vals, t_end_vals
            )
            clip_i = separable_interpolate_grid(
                np.array(ts), np.array(te), np.array(cv), t_start_vals, t_end_vals
            )
            from model_t import scalarize

            m_interp = scalarize(psnr_i, clip_i)
            scores = np.abs(m_pred - m_interp)
    scores[~labeled_mask] += 1.0
    scores[labeled_mask] = -np.inf
    # boost cells near predicted argmax
    i, j = np.unravel_index(m_pred.argmax(), m_pred.shape)
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ni, nj = i + di, j + dj
            if 0 <= ni < m_pred.shape[0] and 0 <= nj < m_pred.shape[1]:
                if not labeled_mask[ni, nj]:
                    scores[ni, nj] += 0.5
    return scores


def select_next_cells(
    scores: np.ndarray,
    labeled_mask: np.ndarray,
    budget: int,
) -> list[GridCell]:
    """Pick top-scoring unlabeled cells up to budget."""
    t_start_vals, t_end_vals = _grid_arrays()
    flat = scores.ravel().copy()
    flat[labeled_mask.ravel()] = -np.inf
    order = np.argsort(-flat)
    cells = []
    for idx in order:
        if len(cells) >= budget:
            break
        if flat[idx] == -np.inf:
            break
        i, j = np.unravel_index(int(idx), scores.shape)
        cells.append(GridCell(float(t_start_vals[i]), float(t_end_vals[j])))
    return cells


@dataclass
class ActiveLabelState:
    labeled_mask: np.ndarray
    psnr_grid: np.ndarray
    clip_grid: np.ndarray
    t_start_vals: np.ndarray
    t_end_vals: np.ndarray

    @classmethod
    def from_seed(cls, seed_cells: list[GridCell]) -> ActiveLabelState:
        t_start_vals, t_end_vals = _grid_arrays()
        n1, n2 = len(t_start_vals), len(t_end_vals)
        mask = cells_to_mask(seed_cells, t_start_vals, t_end_vals)
        return cls(
            labeled_mask=mask,
            psnr_grid=np.full((n1, n2), np.nan),
            clip_grid=np.full((n1, n2), np.nan),
            t_start_vals=t_start_vals,
            t_end_vals=t_end_vals,
        )

    def n_labeled(self) -> int:
        return int(self.labeled_mask.sum())

    def record_label(self, cell: GridCell, psnr: float, clip: float) -> None:
        i = int(np.argmin(np.abs(self.t_start_vals - cell.t_start)))
        j = int(np.argmin(np.abs(self.t_end_vals - cell.t_end)))
        self.labeled_mask[i, j] = True
        self.psnr_grid[i, j] = psnr
        self.clip_grid[i, j] = clip

    def next_cells(
        self,
        m_pred: np.ndarray,
        budget: int,
    ) -> list[GridCell]:
        scores = acquisition_interpolation_residual(
            m_pred,
            self.labeled_mask,
            self.t_start_vals,
            self.t_end_vals,
            psnr_labeled=np.where(self.labeled_mask, self.psnr_grid, np.nan),
            clip_labeled=np.where(self.labeled_mask, self.clip_grid, np.nan),
        )
        return select_next_cells(scores, self.labeled_mask, budget)


def run_active_labeling_loop(
    seed_policy: str = "1d_sweeps",
    max_labels: int = 25,
    batch_size: int = 5,
    m_pred_fn=None,
) -> ActiveLabelState:
    """
    Simulate active labeling: seed cells -> iterative acquisition until max_labels.

    m_pred_fn(state) -> m_grid np.ndarray must be provided for real use; without it
    returns state after seed policy only.
    """
    if seed_policy == "default_only":
        seed = policy_default_only()
    elif seed_policy == "5x5":
        seed = policy_structured_5x5()
    else:
        seed = policy_1d_sweeps()
    state = ActiveLabelState.from_seed(seed)
    if m_pred_fn is None:
        return state
    while state.n_labeled() < max_labels:
        m_pred = m_pred_fn(state)
        remaining = max_labels - state.n_labeled()
        new_cells = state.next_cells(m_pred, min(batch_size, remaining))
        if not new_cells:
            break
        for cell in new_cells:
            if state.n_labeled() >= max_labels:
                break
            # Caller must call record_label after rendering in production.
            state.labeled_mask[
                int(np.argmin(np.abs(state.t_start_vals - cell.t_start))),
                int(np.argmin(np.abs(state.t_end_vals - cell.t_end))),
            ] = True
    return state
