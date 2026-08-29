# metrics.py

"""
Evaluation metrics over score surfaces.

Every function here takes surfaces of shape (N, |T|): N samples, |T| candidate
cells, already flattened and already in the space being scored (phi for the
selection metrics, one delta column for the per-metric ones). The true surface
comes first and the predicted surface second, without exception.

Pure torch: no settings, no model, no data. Nothing in this module reads a
config, so it can be imported before the per-run settings snapshot is bound and
shared by train.py and selector.py without either of them being able to drift
apart.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# Reported Top-K cut-offs. For the 11 x 11 grid these are the top 0.83%, 4.1%,
# and 8.3% of the 121 candidate timestep pairs.
TOP_K_VALUES = (1, 5, 10)


"""
Individual metrics.

The *_at variants take an already-made pick instead of a predicted surface, so
model-free baselines are scored by exactly the same code.
"""



@torch.no_grad()
def regression_loss_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) MSE of pred vs true phi over the cells of each sample."""
    return ((pred_phi - true_phi) ** 2).mean(dim=-1)


@torch.no_grad()
def ranking_loss_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor, chunk: int = 64) -> torch.Tensor:
    """(N,) mean per-sample logistic pairwise loss over pairs with true_u > true_v."""
    out = true_phi.new_full((true_phi.shape[0],), float("nan"))
    for k in range(0, true_phi.shape[0], chunk):
        t, p = true_phi[k : k + chunk], pred_phi[k : k + chunk]
        diff_true = t.unsqueeze(-1) - t.unsqueeze(-2)
        diff_pred = p.unsqueeze(-1) - p.unsqueeze(-2)
        mask = diff_true > 0
        n_pairs = mask.sum(dim=(-1, -2))
        sums = (F.softplus(-diff_pred) * mask).sum(dim=(-1, -2))
        out[k : k + chunk] = torch.where(n_pairs > 0, sums / n_pairs, out.new_full((), float("nan")))
    return out


def _top_k_accuracy_at(true_phi: torch.Tensor, chosen: torch.Tensor, k: int) -> torch.Tensor:
    """(N,) whether the chosen cell lies in the true top-k cells by phi."""
    k = min(k, true_phi.shape[-1])
    top = true_phi.topk(k, dim=-1).indices              # (N, k)
    return (top == chosen.reshape(-1, 1)).any(dim=-1)


def top_k_accuracy_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor, k: int) -> torch.Tensor:
    """(N,) _top_k_accuracy_at for the argmax(pred phi) pick."""
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


def _gain_at(true_phi: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """(N,) true phi at the chosen cell."""
    return true_phi.gather(-1, chosen.reshape(-1, 1)).squeeze(-1)


def gain_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) gain of the argmax(pred phi) pick."""
    return _gain_at(true_phi, pred_phi.argmax(dim=-1))


def _regret_at(true_phi: torch.Tensor, chosen: torch.Tensor) -> torch.Tensor:
    """(N,) true phi lost by taking the chosen cell over the true best one."""
    return true_phi.max(dim=-1).values - _gain_at(true_phi, chosen)


def regret_scores(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> torch.Tensor:
    """(N,) regret of the argmax(pred phi) pick."""
    return _regret_at(true_phi, pred_phi.argmax(dim=-1))


def deviate_rate(true_phi: torch.Tensor, pred_phi: torch.Tensor, baseline_idx: int | torch.Tensor) -> float:
    """Fraction of samples whose selection differs from the default cell."""
    assert true_phi.shape == pred_phi.shape
    chosen = pred_phi.argmax(dim=-1)
    if not isinstance(baseline_idx, torch.Tensor):
        baseline_idx = torch.full_like(chosen, int(baseline_idx))
    return float((chosen != baseline_idx.reshape(-1)).double().mean().item())


def improvement_rate(true_phi: torch.Tensor, pred_phi: torch.Tensor) -> float:
    """Fraction of samples with strictly positive gain."""
    return float((gain_scores(true_phi, pred_phi) > 0).double().mean().item())

"""
Aggregators.
"""

def training_metrics(
    true_phi: torch.Tensor,                # (N, |T|)
    pred_phi: torch.Tensor,                # (N, |T|)
    baseline_idx: int | torch.Tensor,      # unused; kept for a uniform call site
) -> dict[str, float]:
    """
    How well the predicted phi surface matches the true one.

    regression_loss
    ranking_loss
    phi_spearman
    """
    assert true_phi.shape == pred_phi.shape
    mse = regression_loss_scores(true_phi, pred_phi)
    rank = ranking_loss_scores(true_phi, pred_phi)
    rank_finite = rank[~rank.isnan()]
    rho = rank_correlation_scores(true_phi, pred_phi)
    return {
        "regression_loss": float(mse.mean().item()) if mse.numel() else float("nan"),
        "ranking_loss": float(rank_finite.mean().item()) if rank_finite.numel() else float("nan"),
        "phi_spearman": float(rho.quantile(0.5).item()) if rho.numel() else float("nan"),
    }


def _selection_metrics_at(
    true_phi: torch.Tensor,                # (N, |T|)
    chosen: torch.Tensor,                  # (N,) or (N, 1) picked cell index
    baseline_idx: int | torch.Tensor,      # default cell, shared or per sample
) -> dict[str, float]:
    """
    How well an already-made pick serves selection.

    top<K>_accuracy
    regret_median/p90
    gain_mean
    improvement_rate
    deviate_rate
    """
    chosen = chosen.reshape(-1)
    if not isinstance(baseline_idx, torch.Tensor):
        baseline_idx = torch.full_like(chosen, int(baseline_idx))
    gain = _gain_at(true_phi, chosen)
    reg = true_phi.max(dim=-1).values - gain
    return {
        "regret_median": float(reg.quantile(0.5).item()),
        "regret_p90": float(reg.quantile(0.9).item()),
        "gain_mean": float(gain.mean().item()),
        "improvement_rate": float((gain > 0).double().mean().item()),
        "deviate_rate": float((chosen != baseline_idx.reshape(-1)).double().mean().item()),
        **{f"top{k}_accuracy": float(_top_k_accuracy_at(true_phi, chosen, k).double().mean().item()) for k in TOP_K_VALUES},
    }


def selection_metrics(
    true_phi: torch.Tensor,                # (N, |T|)
    pred_phi: torch.Tensor,                # (N, |T|)
    baseline_idx: int | torch.Tensor,
) -> dict[str, float]:
    """How well argmax(pred phi) serves as a selector. See _selection_metrics_at."""
    return _selection_metrics_at(true_phi, pred_phi.argmax(dim=-1), baseline_idx)


def per_component_metrics(
    true: torch.Tensor,                        # (N, |T|, C)
    pred: torch.Tensor,                        # (N, |T|, C)
    cols: tuple[str, ...] | list[str],
    chosen: torch.Tensor,                      # (N,) or (N, 1)
    baseline_idx: int | torch.Tensor,          # default cell, shared or per sample
) -> dict[str, float]:
    """
    Per-column surface fit and selection side-effects, keyed by column name.

    Pick is the shared chosen index (e.g. argmax(pred_phi)). Deltas are relative
    to baseline_idx.

    <col>
    delta_<col>
    mae_<col>
    rmse_<col>
    r2_<col>
    rho_<col>
    gain_<col>
    regret_<col>
    """
    assert true.shape == pred.shape and true.ndim == 3
    assert true.shape[-1] == len(cols)
    n = true.shape[0]
    pick = chosen.reshape(-1)
    assert pick.shape == (n,)
    if not isinstance(baseline_idx, torch.Tensor):
        baseline_idx = torch.full((n,), int(baseline_idx), device=true.device, dtype=torch.long)
    baseline_idx = baseline_idx.reshape(-1)
    assert baseline_idx.shape == (n,)

    out: dict[str, float] = {}
    for i, col in enumerate(cols):
        t, p = true[..., i], pred[..., i]
        t_pick = t.gather(-1, pick.reshape(-1, 1)).squeeze(-1)
        t_base = t.gather(-1, baseline_idx.reshape(-1, 1)).squeeze(-1)
        t_best = t.max(dim=-1).values
        delta = t_pick - t_base

        e = (p - t).reshape(-1)
        ss_res = (e ** 2).sum()
        ss_tot = ((t.reshape(-1) - t.mean()) ** 2).sum().clamp(min=1e-12)
        rho = rank_correlation_scores(t, p)

        out[col] = float(t_pick.mean().item())
        out[f"delta_{col}"] = float(delta.mean().item())
        out[f"mae_{col}"] = float(e.abs().mean().item())
        out[f"rmse_{col}"] = float((e ** 2).mean().sqrt().item())
        out[f"r2_{col}"] = float((1 - ss_res / ss_tot).item())
        out[f"rho_{col}"] = float(rho.quantile(0.5).item()) if rho.numel() else float("nan")
        out[f"gain_{col}"] = float(delta.mean().item())
        out[f"regret_{col}"] = float((t_best - t_pick).mean().item())
    
    return out
