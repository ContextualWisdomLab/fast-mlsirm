from __future__ import annotations

import numpy as np

from .types import MLSIRMParams


def link_fixed_item_parameters(
    source: MLSIRMParams,
    target: MLSIRMParams,
    anchor_items: np.ndarray,
    factor_id: np.ndarray | None = None,
) -> tuple[MLSIRMParams, dict[str, np.ndarray]]:
    """Put source parameters on the target metric using fixed anchor items."""
    anchors = np.asarray(anchor_items, dtype=np.int64)
    if anchors.ndim != 1 or anchors.size == 0:
        raise ValueError("anchor_items must be a non-empty 1D array")
    if np.any(anchors < 0) or np.any(anchors >= source.alpha.size):
        raise ValueError("anchor_items must reference existing items")
    if source.alpha.shape != target.alpha.shape or source.b.shape != target.b.shape:
        raise ValueError("source and target item parameters must have matching shapes")
    if source.theta.shape[1] != target.theta.shape[1]:
        raise ValueError("source and target theta must have the same dimensionality")

    n_items = source.alpha.size
    n_dims = source.theta.shape[1]
    factors = np.zeros(n_items, dtype=np.int64) if factor_id is None else np.asarray(factor_id, dtype=np.int64)
    if factors.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if np.any(factors < 0) or np.any(factors >= n_dims):
        raise ValueError("factor_id values must be in 0..n_dims-1")

    linked = source.copy()
    scale = np.ones(n_dims, dtype=np.float64)
    shift = np.zeros(n_dims, dtype=np.float64)

    for dim in range(n_dims):
        dim_anchors = anchors[factors[anchors] == dim]
        if dim_anchors.size == 0:
            continue
        target_a = target.a[dim_anchors]
        if np.any(target_a <= 0):
            raise ValueError("target anchor slopes must be positive")
        scale[dim] = float(np.exp(np.mean(np.log(source.a[dim_anchors] / target_a))))
        shift[dim] = float(np.mean((source.b[dim_anchors] - target.b[dim_anchors]) / target_a))

        items = factors == dim
        linked.theta[:, dim] = scale[dim] * source.theta[:, dim] + shift[dim]
        linked.alpha[items] = source.alpha[items] - np.log(scale[dim])
        linked.b[items] = source.b[items] - linked.a[items] * shift[dim]

    return linked, {"scale": scale, "shift": shift, "anchor_items": anchors.copy()}


# --------------------------------------------------------------------------
# Characteristic-curve / moment IRT scale linking for separately-calibrated
# common-item designs (Kolen & Brennan 2014; Haebara 1980; Stocking & Lord
# 1983). Rust core is the compute path.
# --------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class IrtLinkResult:
    """IRT linking coefficients (theta_old = slope*theta_new + intercept) with
    the characteristic-curve criterion, iteration count, and method name."""

    slope: float       # theta_old = slope * theta_new + intercept
    intercept: float
    criterion: float   # characteristic-curve loss (0 for moment methods)
    n_iter: int
    method: str


def irt_link(
    a_old: np.ndarray,
    b_old: np.ndarray,
    a_new: np.ndarray,
    b_new: np.ndarray,
    method: str = "stocking_lord",
    q_theta: int = 41,
) -> IrtLinkResult:
    """Link a separately-calibrated *new* form onto the *old* (reference) scale
    from common items, returning ``theta_old = slope * theta_new + intercept``.

    ``a_*`` are slopes (``exp(alpha)`` in the engine's parameterization) and
    ``b_*`` the intercepts of the common items in the ``eta = a*theta + b``
    form. ``method`` is one of ``mean_mean``, ``mean_sigma``, ``haebara``,
    ``stocking_lord``; the characteristic-curve methods integrate over a
    standard-normal Gauss-Hermite grid of ``q_theta`` nodes."""
    from .fitstats import _core_module
    from .estimators.marginal import _gh

    core = _core_module()
    if core is None:  # pragma: no cover
        raise RuntimeError("irt_link requires the compiled Rust core")
    nodes, weights = _gh(int(q_theta))
    res = core.irt_link(
        np.asarray(a_old, dtype=np.float64),
        np.asarray(b_old, dtype=np.float64),
        np.asarray(a_new, dtype=np.float64),
        np.asarray(b_new, dtype=np.float64),
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        method=str(method),
    )
    return IrtLinkResult(
        slope=float(res["slope"]), intercept=float(res["intercept"]),
        criterion=float(res["criterion"]), n_iter=int(res["n_iter"]),
        method=str(method),
    )
