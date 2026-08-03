"""
Selection metrics for the timestep selector T.

M_hat is trained to regress (PSNR-Unedited, CLIP-Edited) on every labeled cell,
but what the pipeline consumes is the cell T picks out of that grid. Those are
not the same thing: two checkpoints with equal per-cell error can pick cells of
very different quality, because the phi surface is flat near its peak and the
cost of a miss depends on the objective's curvature.

Everything here judges a picked cell against that sample's whole grid:

    regret = max_T phi(T) - phi(T_picked)

and, because Delta is measured from the baseline (DEFAULT_T_START,
DEFAULT_T_END) cell and phi(0) = 0 for every scalarization in scores.py, phi at
the picked cell is also the gain over always using the default cell.

Two builders pack the same table: split_from_df for the evaluation scripts,
which work off the one-row-per-cell tables, and split_from_cells for training,
where the per-sample grids are already device-resident.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from _helpers import t_target_phi_values, timestep_pairs_from_df
from settings import *

if TYPE_CHECKING:  # _data pulls in model_m (and the ChordEdit encoders) at import
    from _data import CellTensors


@dataclass(frozen=True)
class SelectionSplit:
    """One split's dense true-score grid over the labeled cells.

    Cells are the (t_start, t_end) pairs actually present in the data, ordered
    by (t_start, t_end) so column k means the same timestep pair in every
    sample and in both builders. The true-score arrays stay on the host
    (metrics are computed once per epoch over a few hundred samples); only
    cell_t, which a grid forward through M_hat is driven from, is
    device-resident.
    """

    sample_ids: np.ndarray          # (n,) row order
    phi: np.ndarray                 # (n, N) true objective per cell
    best_phi: np.ndarray            # (n,) oracle score
    best_cell: np.ndarray           # (n,) oracle cell
    metrics: dict[str, np.ndarray]  # each (n, N) raw metric values
    baseline_cell: int              # column of (DEFAULT_T_START, DEFAULT_T_END)
    cell_t: torch.Tensor            # (N, 2) float (t_start, t_end)

    @property
    def n_samples(self) -> int:
        return int(self.phi.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.phi.shape[1])

    def rows_for(self, sample_ids) -> np.ndarray:
        """Row indices for the given sample ids, in the order given."""
        wanted = np.asarray(sample_ids)
        order = np.argsort(self.sample_ids, kind="stable")
        sorted_ids = self.sample_ids[order]
        pos = np.searchsorted(sorted_ids, wanted)
        if pos.max(initial=-1) >= len(sorted_ids):
            raise ValueError("sample ids missing from the selection split")
        rows = order[pos]
        if not np.array_equal(self.sample_ids[rows], wanted):
            raise ValueError("sample ids missing from the selection split")
        return rows


def _cell_key(t_start, t_end) -> tuple[float, float]:
    """Hashable (t_start, t_end) key, tolerant of float representation noise."""
    return (round(float(t_start), 6), round(float(t_end), 6))


def _baseline_cell(cell_t: np.ndarray) -> int:
    """Column of the default cell, which every Delta (and so phi) is relative to."""
    hits = np.flatnonzero(
        np.isclose(cell_t[:, 0], DEFAULT_T_START) & np.isclose(cell_t[:, 1], DEFAULT_T_END)
    )
    if len(hits) != 1:
        raise ValueError(
            f"expected exactly one ({DEFAULT_T_START}, {DEFAULT_T_END}) cell, found {len(hits)}"
        )
    return int(hits[0])


def _split(
    sample_ids: np.ndarray,
    cell_t: np.ndarray,
    values: np.ndarray,
    baseline_cell: int,
    device: torch.device | None = None,
) -> SelectionSplit:
    """Score a dense (n, N, C) block of raw metric values into a SelectionSplit.

    values' last axis is ordered like M_TARGET_COLS. Scoring runs in float64:
    phi differences near the top of a grid are what regret measures.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[-1] != len(M_TARGET_COLS):
        raise ValueError(f"expected (n, N, {len(M_TARGET_COLS)}) values, got {values.shape}")
    if np.isnan(values).any():
        n_missing = int(np.isnan(values).any(axis=-1).sum())
        n_total = values.shape[0] * values.shape[1]
        raise ValueError(f"ragged grid: {n_missing} of {n_total} (sample, cell) rows are unlabeled")

    phi = t_target_phi_values(torch.as_tensor(values, dtype=torch.float64), int(baseline_cell))
    phi = phi.detach().cpu().numpy()
    return SelectionSplit(
        sample_ids=np.asarray(sample_ids),
        phi=phi,
        best_phi=phi.max(axis=1),
        best_cell=phi.argmax(axis=1),
        metrics={col: values[..., i] for i, col in enumerate(M_TARGET_COLS)},
        baseline_cell=int(baseline_cell),
        cell_t=torch.tensor(np.asarray(cell_t, dtype=float), dtype=torch.float, device=device),
    )


def split_from_df(
    df: pd.DataFrame,
    device: torch.device | None = None,
    sample_ids=None,
) -> SelectionSplit:
    """Pack a one-row-per-cell table (load_df / load_split_df) into a SelectionSplit.

    df must carry M_TARGET_COLS, so callers holding the split frames apart pass
    pd.concat([X_df, y_df], axis=1). Rows come out sorted by sample id unless
    sample_ids gives an explicit order to align with.
    """
    missing = [c for c in (SAMPLE_ID_COL, T_START_COL, T_END_COL, *M_TARGET_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f"df is missing columns {missing}")

    cell_t = timestep_pairs_from_df(df)
    col_of = {_cell_key(a, b): k for k, (a, b) in enumerate(cell_t)}
    col = np.array(
        [col_of[_cell_key(a, b)] for a, b in zip(df[T_START_COL], df[T_END_COL])],
        dtype=int,
    )
    ids = np.sort(df[SAMPLE_ID_COL].unique())
    row = np.searchsorted(ids, df[SAMPLE_ID_COL].to_numpy())

    # One cell per (sample, t_start, t_end): with TARGET_T_DELTA unset the table
    # still holds one row per t_delta, and packing would silently keep whichever
    # landed last.
    flat = row * len(cell_t) + col
    if len(np.unique(flat)) != len(flat):
        raise ValueError(
            f"{len(flat) - len(np.unique(flat))} duplicate (sample, cell) rows; "
            f"set TARGET_T_DELTA to pick one {T_DELTA_COL}"
        )

    def _pack(values: np.ndarray) -> np.ndarray:
        out = np.full((len(ids), len(cell_t)), np.nan)
        out[row, col] = values
        return out

    values = np.stack([_pack(df[c].to_numpy(dtype=float)) for c in M_TARGET_COLS], axis=-1)
    sel = _split(ids, cell_t, values, _baseline_cell(cell_t), device)
    return sel if sample_ids is None else slice_grid(sel, sample_ids)


def split_from_cells(cells: CellTensors, sample_ids=None) -> SelectionSplit:
    """Build from the device-resident training tensors' complete per-sample grids.

    Rows are keyed by the embedding-table index CellTensors carries
    (sample_idx); it does not keep sample ids. Use split_from_df when the
    metrics have to be joined back to sample ids.
    """
    rows = cells.grid_rows
    baseline = torch.unique(cells.grid_baseline)
    if baseline.numel() != 1:
        raise ValueError("grids disagree on which column holds the default cell")

    # _build_grid_index sorts every grid's rows by (t_start, t_end), so one
    # grid's pairs stand for all of them, but only if they really do match.
    t = cells.t[rows]
    cell_t = t[0]
    if not torch.allclose(t, cell_t):
        raise ValueError("grids do not share one (t_start, t_end) cell order")

    sel = _split(
        sample_ids=cells.sample_idx[rows[:, 0]].detach().cpu().numpy(),
        cell_t=cell_t.detach().cpu().numpy(),
        values=cells.y[rows].detach().cpu().numpy(),
        baseline_cell=int(baseline[0]),
        device=cells.t.device,
    )
    return sel if sample_ids is None else slice_grid(sel, sample_ids)


def slice_grid(sel: SelectionSplit, sample_ids) -> SelectionSplit:
    """Restrict a SelectionSplit to the given samples, in the order given."""
    rows = sel.rows_for(sample_ids)
    return SelectionSplit(
        sample_ids=sel.sample_ids[rows],
        phi=sel.phi[rows],
        best_phi=sel.best_phi[rows],
        best_cell=sel.best_cell[rows],
        metrics={k: v[rows] for k, v in sel.metrics.items()},
        baseline_cell=sel.baseline_cell,
        cell_t=sel.cell_t,
    )


def cell_labels(sel: SelectionSplit) -> list[tuple[float, float]]:
    """(t_start, t_end) value pair for each cell column."""
    pairs = sel.cell_t.detach().cpu().numpy()
    return [(round(float(a), 3), round(float(b), 3)) for a, b in pairs]


def selection_metrics(picked: np.ndarray, sel: SelectionSplit) -> dict[str, float]:
    """Score one cell choice per sample against the oracle and the default cell.

    picked: (n,) column indices into the cell list, aligned with sel's rows.
    """
    picked = np.asarray(picked, dtype=int)
    n = len(picked)
    if n != len(sel.best_phi):
        raise ValueError(f"picked has {n} rows, split has {len(sel.best_phi)}")
    rows = np.arange(n)

    phi_at = sel.phi[rows, picked]
    regret = sel.best_phi - phi_at
    # phi(default) == 0, so phi at the picked cell is the gain over the default.
    gain = phi_at
    # Rank of the picked cell within the sample's grid, 0 = the oracle cell.
    rank = (sel.phi > phi_at[:, None]).sum(axis=1)

    out = {
        "regret_median": float(np.median(regret)),
        "regret_mean": float(regret.mean()),
        "regret_p90": float(np.percentile(regret, 90)),
        "gain_median": float(np.median(gain)),
        "gain_mean": float(gain.mean()),
        "gain_p10": float(np.percentile(gain, 10)),
        "top1_hit_rate": float((rank == 0).mean()),
        "top3_hit_rate": float((rank < 3).mean()),
        "pick_rank_median": float(np.median(rank)),
        "deviate_rate": float((picked != sel.baseline_cell).mean()),
        "win_rate": float((gain > 0).mean()),
        "loss_rate": float((gain < 0).mean()),
        "n_distinct_cells": int(len(np.unique(picked))),
    }
    for name, values in sel.metrics.items():
        out[f"achieved_{name}"] = float(values[rows, picked].mean())
    return out
