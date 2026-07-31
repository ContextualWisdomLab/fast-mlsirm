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
