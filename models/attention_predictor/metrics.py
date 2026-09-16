# metrics.py

"""
Evaluation metrics over score surfaces.

Every function here takes surfaces of shape (N, n_cells): N samples, n_cells candidate
cells, already flattened and already in the space being scored (phi for the
selection metrics, one delta column for the per-metric ones). The true surface
comes first and the predicted surface second, without exception.

Pure torch: no settings, no model, no data. Nothing in this module reads a
config, so it can be imported before the per-run settings snapshot is bound and
shared by train.py and selector.py without either of them being able to drift
apart.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


# Reported Top-K cut-offs. For the 11 x 11 grid these are the top 0.83%, 4.1%,
# and 8.3% of the 121 candidate timestep pairs.
TOP_K_VALUES = (1, 5, 10)


"""
Individual metrics.

The *_at variants take an already-made selection instead of a predicted surface, so
model-free baselines are scored by exactly the same code.
"""



@torch.no_grad()
def regression_loss_scores(
    true_phi: torch.Tensor,
    pred_phi: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """(N,) MSE of pred vs true phi over the cells of each sample."""
    if top_k is not None and top_k < true_phi.shape[-1]:
        idx = true_phi.topk(top_k, dim=-1).indices
        true_phi = true_phi.gather(-1, idx)
        pred_phi = pred_phi.gather(-1, idx)
    return ((pred_phi - true_phi) ** 2).mean(dim=-1)


@torch.no_grad()
def ranking_loss_scores(
    true_phi: torch.Tensor,
    pred_phi: torch.Tensor,
    top_k: int | None = None,
    chunk: int = 64,
) -> torch.Tensor:
    """(N,) mean per-sample logistic pairwise loss over pairs with true_u > true_v."""
    out = true_phi.new_full((true_phi.shape[0],), float("nan"))
    for k in range(0, true_phi.shape[0], chunk):
        t, p = true_phi[k : k + chunk], pred_phi[k : k + chunk]
        diff_true = t.unsqueeze(-1) - t.unsqueeze(-2)
        diff_pred = p.unsqueeze(-1) - p.unsqueeze(-2)
        mask = diff_true > 0
        if top_k is not None and top_k < t.shape[-1]:
            idx = t.topk(top_k, dim=-1).indices
            is_top = torch.zeros_like(t, dtype=torch.bool).scatter_(-1, idx, True)
            mask = mask & is_top.unsqueeze(-1)
        n_pairs = mask.sum(dim=(-1, -2))
        sums = (F.softplus(-diff_pred) * mask).sum(dim=(-1, -2))
        out[k : k + chunk] = torch.where(n_pairs > 0, sums / n_pairs, out.new_full((), float("nan")))
    return out


def _top_k_accuracy_at(true_phi: torch.Tensor, selected_cells: torch.Tensor, k: int) -> torch.Tensor:
    """(N,) whether the selected cell lies in the true top-k cells by phi."""
    k = min(k, true_phi.shape[-1])
    top = true_phi.topk(k, dim=-1).indices
    return (top == selected_cells.unsqueeze(-1)).any(dim=-1)


def top_k_accuracy_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor, k: int) -> torch.Tensor:
    """(N,) _top_k_accuracy_at for the argmax(pred phi) selection."""
    return _top_k_accuracy_at(true_phi, pred_phi.argmax(dim=-1), k)


def rank_correlation_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) per-sample Spearman rho of the predicted and true surfaces."""
    def _row_ranks(x: torch.Tensor) -> torch.Tensor:
        order = x.argsort(dim=-1)
        xs = x.gather(-1, order)
        pos = torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(xs)
        starts = F.pad(xs[..., 1:] != xs[..., :-1], (1, 0), value=True)
        gid = starts.long().cumsum(-1) - 1
        sums = torch.zeros_like(xs).scatter_add_(-1, gid, pos)
        counts = torch.zeros_like(xs).scatter_add_(-1, gid, torch.ones_like(pos))
        return torch.empty_like(xs).scatter_(-1, order, (sums / counts).gather(-1, gid))

    ra = _row_ranks(true_phi); ra = ra - ra.mean(-1, keepdim=True)
    rb = _row_ranks(pred_phi); rb = rb - rb.mean(-1, keepdim=True)
    den = ra.norm(dim=-1) * rb.norm(dim=-1)
    # Constant rows (no ordering) are read as 0, not nan.
    return torch.where(den > 0, (ra * rb).sum(-1) / den, torch.zeros_like(den))


def _gain_at(true_phi: torch.Tensor, selected_cells: torch.Tensor) -> torch.Tensor:
    """(N,) true phi at the selected cell."""
    return true_phi.gather(-1, selected_cells.unsqueeze(-1)).squeeze(-1)


def gain_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) gain of the argmax(pred phi) selection."""
    return _gain_at(true_phi, pred_phi.argmax(dim=-1))


def _regret_at(true_phi: torch.Tensor, selected_cells: torch.Tensor) -> torch.Tensor:
    """(N,) true phi lost by taking the selected cell over the true best one."""
    return true_phi.max(dim=-1).values - _gain_at(true_phi, selected_cells)


def regret_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) regret of the argmax(pred phi) selection."""
    return _regret_at(true_phi, pred_phi.argmax(dim=-1))


def deviate_rate(true_phi: torch.Tensor, pred_phi: torch.Tensor, default_cell: int) -> float:
    """Fraction of samples whose selection differs from the default cell."""
    assert true_phi.shape == pred_phi.shape
    assert isinstance(default_cell, int)
    selected = pred_phi.argmax(dim=-1)
    return float((selected != default_cell).double().mean().item())


def improvement_rate(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> float:
    """Fraction of samples with strictly positive gain."""
    return float((gain_scores(true_phi, pred_phi) > 0).double().mean().item())


def spread_ratio(true: torch.Tensor, pred: torch.Tensor) -> float:
    """Across-image spread of the pred surface relative to the true one."""
    assert true.shape == pred.shape and true.ndim == 2
    if true.shape[0] < 2:
        return float("nan")
    den = true.double().std(dim=0).mean()
    num = pred.double().std(dim=0).mean()
    return float((num / den).item()) if float(den.item()) > 0 else float("nan")


def cross_image_rho(true: torch.Tensor, pred: torch.Tensor) -> float:
    """Median over cells of the across-image Spearman rho at a fixed cell."""
    assert true.shape == pred.shape and true.ndim == 2
    if true.shape[0] < 2:
        return float("nan")
    rho = rank_correlation_scores(true.double().T.contiguous(), pred.double().T.contiguous())
    return float(rho.quantile(0.5).item()) if rho.numel() else float("nan")


"""
Aggregators.
"""

def training_metrics(
    true_phi: torch.Tensor,          # (N, n_cells)
    pred_phi: torch.Tensor,          # (N, n_cells)
    loss_func: Callable[[], torch.Tensor],
    mse_func: Callable[[], torch.Tensor],
    ranking_func: Callable[[], torch.Tensor],
    col_func: Callable[[], torch.Tensor],
) -> dict[str, float]:
    """
    How well the predicted phi surface matches the true one.

    * loss
    * loss_mse
    * loss_ranking
    * loss_col
    * phi_spearman
    * rho_phi_image
    * phi_spread_ratio
    """
    assert true_phi.shape == pred_phi.shape
    rho = rank_correlation_scores(true_phi, pred_phi)
    return {
        "loss": float(loss_func().detach().item()),
        "loss_mse": float(mse_func().detach().item()),
        "loss_ranking": float(ranking_func().detach().item()),
        "loss_col": float(col_func().detach().item()),
        "phi_spearman": float(rho.quantile(0.5).item()) if rho.numel() else float("nan"),
        "rho_phi_image": cross_image_rho(true_phi, pred_phi),
        "phi_spread_ratio": spread_ratio(true_phi, pred_phi),
    }


def selection_metrics(
    true_phi: torch.Tensor,          # (N, n_cells)
    selected_cells: torch.Tensor,    # (N,)
    default_cell: int,
) -> dict[str, float]:
    """
    How well the selected cell compares to the true best cell.

    * phi
    * delta_phi
    * top<K>_accuracy
    * regret_median/p90
    * gain_mean
    * improvement_rate
    * deviate_rate
    * modal_cell_frac
    * n_distinct_cells
    """
    assert selected_cells.shape == (true_phi.shape[0],)
    assert isinstance(default_cell, int)
    gain = _gain_at(true_phi, selected_cells)
    delta_phi = gain - true_phi[:, default_cell]
    reg = true_phi.max(dim=-1).values - gain
    # How concentrated the selections are. A constant selector puts modal_cell_frac at
    # 1.0 with n_distinct_cells at 1, which no other selection metric reveals.
    counts = torch.bincount(selected_cells, minlength=true_phi.shape[-1])
    return {
        "phi": float(gain.mean().item()),
        "delta_phi": float(delta_phi.mean().item()),
        "modal_cell_frac": float((counts.max().double() / counts.sum().double()).item()),
        "n_distinct_cells": float((counts > 0).sum().item()),
        "regret_median": float(reg.quantile(0.5).item()),
        "regret_p90": float(reg.quantile(0.9).item()),
        "gain_mean": float(gain.mean().item()),
        "improvement_rate": float((gain > 0).double().mean().item()),
        "deviate_rate": float((selected_cells != default_cell).double().mean().item()),
        **{f"top{k}_accuracy": float(_top_k_accuracy_at(true_phi, selected_cells, k).double().mean().item()) for k in TOP_K_VALUES},
    }


def deviation_metrics(
    true_dev: torch.Tensor,          # (N, n_cells, C) true deviation from the mean surface
    pred_dev: torch.Tensor,          # (N, n_cells, C) predicted deviation
    cols: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """
    How well the per-image deviations from the shared surface are predicted.

    * dev_corr_<col>: Pearson correlation pooled over samples and cells
    * dev_slope_<col>: slope of the true on the predicted deviation; 1 means calibrated amplitude
    * dev_corr: mean of the per-column correlations
    """
    assert true_dev.shape == pred_dev.shape and true_dev.shape[-1] == len(cols)
    out: dict[str, float] = {}
    for i, col in enumerate(cols):
        t, p = true_dev[..., i].reshape(-1), pred_dev[..., i].reshape(-1)
        t, p = t - t.mean(), p - p.mean()
        out[f"dev_corr_{col}"] = float(((t * p).sum() / (t.norm() * p.norm()).clamp(min=1e-12)).item())
        out[f"dev_slope_{col}"] = float(((t * p).sum() / (p ** 2).sum().clamp(min=1e-12)).item())
    out["dev_corr"] = sum(out[f"dev_corr_{col}"] for col in cols) / len(cols)
    return out


def per_col_metrics(
    true_cols: torch.Tensor,         # (N, n_cells, C) delta surfaces
    true_raw: torch.Tensor,          # (N, n_cells, C) measured PSNR/CLIP
    pred_cols: torch.Tensor,         # (N, n_cells, C)
    selected_cells: torch.Tensor,    # (N,)
    default_cell: int,
    cols: tuple[str, ...] | list[str],
) -> dict[str, float]:
    """
    Per-column training and selection metrics.

    * <col>
    * delta_<col>
    * mae_<col>
    * rmse_<col>
    * r2_<col>
    * rho_<col>
    * rho_<col>_image
    * spread_ratio_<col>
    * gain_<col>
    * regret_<col>
    """
    assert true_cols.shape == pred_cols.shape == true_raw.shape and true_cols.ndim == 3
    assert true_cols.shape[-1] == len(cols)
    n = true_cols.shape[0]
    assert selected_cells.shape == (n,)
    assert isinstance(default_cell, int)

    idx = selected_cells.unsqueeze(-1)
    out: dict[str, float] = {}
    for i, col in enumerate(cols):
        t, p = true_cols[..., i], pred_cols[..., i]
        t_selected = t.gather(-1, idx).squeeze(-1)
        t_base = t[:, default_cell]
        t_best = t.max(dim=-1).values
        delta = t_selected - t_base

        raw = true_raw[..., i]
        raw_selected = raw.gather(-1, idx).squeeze(-1)
        raw_base = raw[:, default_cell]

        e = (p - t).reshape(-1)
        ss_res = (e ** 2).sum()
        ss_tot = ((t.reshape(-1) - t.mean()) ** 2).sum().clamp(min=1e-12)
        rho = rank_correlation_scores(t, p)

        out[col] = float(raw_selected.mean().item())
        out[f"delta_{col}"] = float((raw_selected - raw_base).mean().item())
        out[f"mae_{col}"] = float(e.abs().mean().item())
        out[f"rmse_{col}"] = float((e ** 2).mean().sqrt().item())
        out[f"r2_{col}"] = float((1 - ss_res / ss_tot).item())
        out[f"rho_{col}"] = float(rho.quantile(0.5).item()) if rho.numel() else float("nan")
        out[f"rho_{col}_image"] = cross_image_rho(t, p)
        out[f"spread_ratio_{col}"] = spread_ratio(t, p)
        out[f"gain_{col}"] = float(delta.mean().item())
        out[f"regret_{col}"] = float((t_best - t_selected).mean().item())

    return out
