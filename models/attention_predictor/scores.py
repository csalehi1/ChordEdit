# scores.py

"""
Scalarization of the score vector s = (s_1, s_2) into a single objective phi,
following the paper's notation: s_1 = PSNR-Unedited, s_2 = CLIP-Edited.

Scoring functions take the selector-space surface, which for the paper
recipe is per-sample-normalized default-relative deltas

    Delta_i = (s_i - s_i^0) / (max_T s_i - min_T s_i)
"""

from __future__ import annotations

import torch


def naive_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
) -> torch.Tensor:
    """
    Naive Score. Weighted sum of the metric deltas. Simple and interpretable,
    with each weight independently scaling its metric's importance. However, it
    is indifferent to balance: it cannot distinguish a candidate that improves
    both metrics moderately from one that maximizes one metric while tanking
    the other, so long as the weighted sums match.

        phi_nai(Delta) = sum_i w_i * Delta_i

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/50byxqmq3q
    """
    if weights is None:
        return deltas.sum(dim=-1)
    return (deltas * weights).sum(dim=-1)


def cara_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    CARA Score. Sum of deltas passed through the exponential (CARA) utility
    u(x) = (1 - e^{-alpha x}) / alpha, normalized so u(0) = 0 and u'(0) = 1.
    Strict concavity biases toward balance: regressions incur an exponentially
    growing penalty, so a severe regression cannot be offset by gains
    elsewhere. The cost is that gains saturate at w_i / alpha per metric, so
    the score cannot distinguish among candidates that improve a metric
    strongly versus very strongly. Recovers the Naive Score as alpha -> 0+.

        phi_CARA(Delta) = (1/alpha) * sum_i w_i * (1 - exp(-alpha * Delta_i))

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)
    alpha: float > 0

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/wbahputeel
    """
    # u(x) = (1 - e^{-alpha x}) / alpha
    # See https://en.wikipedia.org/wiki/Exponential_utility.
    u = -torch.expm1(-alpha * deltas) / alpha

    if weights is None:
        return u.sum(dim=-1)
    return (u * weights).sum(dim=-1)


def linex_score(
    deltas: torch.Tensor,  # (..., N, C)
    weights: torch.Tensor | None = None,  # (C,)
    alpha: float = 2.0,
) -> torch.Tensor:
    """
    LINEX Score. Average of the Naive and CARA Scores, equivalent to the
    linear-exponential (LINEX) utility u(x) = (x + (1 - e^{-alpha x}) / alpha) / 2,
    normalized so u(0) = 0 and u'(0) = 1. Keeps CARA's superlinear regression
    penalty while removing its reward cap: gains accrue at an asymptotic rate
    of 1/2 per unit, so the score is unbounded both above and below and can
    distinguish among candidates that improve both metrics. The trade-off is a
    weaker balance bias: at matched alpha its curvature is half of CARA's
    (u''(0) = -alpha/2 vs -alpha), and a large gain on one metric can
    partially offset weakness on the other. Recovers the Naive Score as
    alpha -> 0+.

        phi_LINEX(Delta) = (phi_nai(Delta) + phi_CARA(Delta)) / 2
                         = (1/2) * sum_i w_i * [Delta_i + (1 - exp(-alpha * Delta_i)) / alpha]

    deltas: (..., N, C)
    weights: (C,) or None for equal weights (w_i = 1)
    alpha: float > 0

    Returns: shape (..., N)

    See a plot of the score surface (2 metrics):
    https://www.desmos.com/3d/7ambm2crdv
    """
    # Fuse naive + cara into one expm1 pass.
    u = -torch.expm1(-alpha * deltas) / alpha
    combined = deltas + u
    if weights is None:
        return 0.5 * combined.sum(dim=-1)
    return 0.5 * (combined * weights).sum(dim=-1)
