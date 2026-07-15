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
    anchors_raw = np.asarray(anchor_items)
    if anchors_raw.ndim != 1 or anchors_raw.size == 0:
        raise ValueError("anchor_items must be a non-empty 1D array")
    a_fl = anchors_raw.astype(np.float64)
    if not np.all(np.isfinite(a_fl)) or np.any(a_fl < 0) or np.any(a_fl != np.floor(a_fl)):
        raise ValueError("anchor_items must be finite non-negative integers")
    # Range-check on the float BEFORE narrowing: uint64 max casts to -1 and
    # would slip past an upper-bound-only int64 check as a valid last-item index.
    if np.any(a_fl >= source.alpha.size):
        raise ValueError("anchor_items must reference existing items")
    anchors = a_fl.astype(np.int64)
    if anchors.size != np.unique(anchors).size:
        raise ValueError("anchor_items must be unique")
    if source.alpha.shape != target.alpha.shape or source.b.shape != target.b.shape:
        raise ValueError("source and target item parameters must have matching shapes")
    if source.theta.ndim != 2 or target.theta.ndim != 2:
        raise ValueError("source and target theta must be 2-D (items x dimensions)")
    if source.theta.shape[1] != target.theta.shape[1]:
        raise ValueError("source and target theta must have the same dimensionality")
    for arr, nm in (
        (source.alpha, "source.alpha"),
        (source.b, "source.b"),
        (target.alpha, "target.alpha"),
        (target.b, "target.b"),
    ):
        if not np.all(np.isfinite(np.asarray(arr, dtype=float))):
            raise ValueError(f"{nm} must be finite")

    n_items = source.alpha.size
    n_dims = source.theta.shape[1]
    if factor_id is None:
        factors = np.zeros(n_items, dtype=np.int64)
    else:
        f_raw = np.asarray(factor_id)
        f_fl = f_raw.astype(np.float64)
        if (
            f_raw.ndim != 1
            or not np.all(np.isfinite(f_fl))
            or np.any(f_fl < 0)
            or np.any(f_fl != np.floor(f_fl))
            or np.any(f_fl >= n_items)
        ):
            raise ValueError("factor_id must be a 1-D array of finite non-negative integers")
        factors = f_fl.astype(np.int64)
    if factors.shape != (n_items,):
        raise ValueError("factor_id length must match number of items")
    if np.any(factors >= n_dims):
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
        if not (np.isfinite(scale[dim]) and scale[dim] > 0.0 and np.isfinite(shift[dim])):
            raise ValueError("non-finite or non-positive linking coefficients (check anchor parameters)")

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
    the characteristic-curve criterion and explicit termination evidence."""

    slope: float       # theta_old = slope * theta_new + intercept
    intercept: float
    criterion: float   # characteristic-curve loss (0 for moment methods)
    n_iter: int
    method: str
    converged: bool = True
    termination_reason: str = "closed_form"
    max_iter: int = 0
    final_objective_span: float = 0.0
    objective_tolerance: float = 0.0
    final_parameter_span: float = 0.0
    parameter_tolerance: float = 0.0


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
    standard-normal Gauss-Hermite grid of ``q_theta`` nodes. Mean/sigma
    linking requires non-zero common-item difficulty spread on both scales.
    Characteristic-curve results expose both the objective and parameter
    simplex stopping criteria; inspect ``converged`` before using a result.

    References
    ----------
    Haebara, T. (1980). Equating logistic ability scales by a weighted least
    squares method. *Japanese Psychological Research, 22*(3), 144–149.
    https://doi.org/10.4992/psycholres1954.22.144

    Kolen, M. J., & Brennan, R. L. (2014). *Test equating, scaling, and
    linking: Methods and practices* (3rd ed.). Springer.
    https://doi.org/10.1007/978-1-4939-0317-7

    Stocking, M. L., & Lord, F. M. (1983). Developing a common metric in item
    response theory. *Applied Psychological Measurement, 7*(2), 201–210.
    https://doi.org/10.1177/014662168300700208
    """
    from .fitstats import _core_module
    from .estimators.marginal import _gh

    core = _core_module()
    if core is None:  # pragma: no cover
        raise RuntimeError("irt_link requires the compiled Rust core")
    ao = np.asarray(a_old, dtype=np.float64)
    bo = np.asarray(b_old, dtype=np.float64)
    an = np.asarray(a_new, dtype=np.float64)
    bn = np.asarray(b_new, dtype=np.float64)
    for _arr, _nm in ((ao, 'a_old'), (bo, 'b_old'), (an, 'a_new'), (bn, 'b_new')):
        if _arr.ndim != 1 or not np.all(np.isfinite(_arr)):
            raise ValueError(f'{_nm} must be a 1-D array of finite numbers')
    if ao.shape != bo.shape or an.shape != bn.shape or ao.shape != an.shape:
        raise ValueError('slope/intercept arrays must have matching lengths')
    if np.any(ao <= 0) or np.any(an <= 0):
        raise ValueError('slopes (a_old/a_new) must be positive')
    if isinstance(q_theta, (bool, np.bool_)) or not isinstance(q_theta, (int, np.integer)):
        raise ValueError('q_theta must be an integer quadrature size')
    nodes, weights = _gh(int(q_theta))
    res = core.irt_link(
        ao,
        bo,
        an,
        bn,
        np.asarray(nodes, dtype=np.float64),
        np.asarray(weights, dtype=np.float64),
        method=str(method),
    )
    return IrtLinkResult(
        slope=float(res["slope"]), intercept=float(res["intercept"]),
        criterion=float(res["criterion"]), n_iter=int(res["n_iter"]),
        method=str(method),
        converged=bool(res["converged"]),
        termination_reason=str(res["termination_reason"]),
        max_iter=int(res["max_iter"]),
        final_objective_span=float(res["final_objective_span"]),
        objective_tolerance=float(res["objective_tolerance"]),
        final_parameter_span=float(res["final_parameter_span"]),
        parameter_tolerance=float(res["parameter_tolerance"]),
    )
