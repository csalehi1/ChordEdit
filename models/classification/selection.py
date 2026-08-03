"""
Selection metrics for the classifier.

Training scores the classifier against one labeled bucket pair per sample, but
what the pipeline consumes is the grid cell it picks. Those are not the same
thing: two configurations with equal bucket accuracy can pick cells of very
different quality, because the phi surface is flat near its peak and the cost
of a miss depends on the objective's curvature.

Everything here judges a picked cell against that sample's whole grid:

    regret = max_T phi(T) - phi(T_picked)

and, because phi at the baseline (DEFAULT_T_START, DEFAULT_T_END) cell is
exactly 0 by construction, phi at the picked cell is also the gain over always
using the default cell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from _data import GridTable
from settings import *


@dataclass(frozen=True)
class SelectionSplit:
    """One split's slice of the grid table.

    The true-score arrays stay on the host (metrics are computed once per
    epoch over a few hundred samples); only the cell index vectors, which the
    joint decode gathers with on every batch, are device-resident.
    """

    phi: np.ndarray                 # (n, N) true objective per cell
    best_phi: np.ndarray            # (n,) oracle score
    best_cell: np.ndarray           # (n,) oracle cell
    metrics: dict[str, np.ndarray]  # each (n, N) raw metric values
    baseline_cell: int
    cell_start_idx: torch.Tensor    # (N,) long
    cell_end_idx: torch.Tensor      # (N,) long

    @property
    def n_cells(self) -> int:
        return int(self.phi.shape[1])


def slice_grid(grid: GridTable, sample_ids, device: torch.device) -> SelectionSplit:
    """Restrict a GridTable to one split's samples, in the order given."""
    rows = grid.rows_for(sample_ids)
    phi = grid.phi[rows]
    return SelectionSplit(
        phi=phi,
        best_phi=phi.max(axis=1),
        best_cell=phi.argmax(axis=1),
        metrics={k: v[rows] for k, v in grid.metrics.items()},
        baseline_cell=grid.baseline_cell,
        cell_start_idx=torch.as_tensor(grid.cell_start_idx, dtype=torch.long, device=device),
        cell_end_idx=torch.as_tensor(grid.cell_end_idx, dtype=torch.long, device=device),
    )


def cell_labels(sel: SelectionSplit, grid: GridTable) -> list[tuple[float, float]]:
    """(t_start, t_end) value pair for each cell column."""
    starts = np.asarray(GRID_T_START, dtype=float)[grid.cell_start_idx]
    ends = np.asarray(GRID_T_END, dtype=float)[grid.cell_end_idx]
    return [(round(float(a), 3), round(float(b), 3)) for a, b in zip(starts, ends)]


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
