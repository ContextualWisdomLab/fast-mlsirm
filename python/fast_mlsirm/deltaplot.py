"""Angoff Delta plot DIF detection. All numeric work happens in the Rust
core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import MAX_MAX_ITER

_NUMPY_INTEGER_SCALAR_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.intp,
    np.longlong,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.uintp,
    np.ulonglong,
)
_NUMPY_FLOATING_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)


@dataclass
class DeltaPlotResult:
    """Delta plot DIF result (deltaPlotR port; 0-based item indices).

    ``props`` and ``adj_props`` are ``n_items x 2`` (reference, focal)
    proportions before/after extreme adjustment; ``deltas`` the
    ``4 * qnorm(1 - p) + 13`` transformed difficulties; ``dist`` the
    ``n_iter x n_items`` perpendicular distances (row 0 = unpurified
    pass); ``axis_par`` the per-iteration major-axis ``(a, b)``;
    ``thresholds`` the per-iteration detection threshold; ``dif_items``
    the final flagged items; ``converged`` is False when purification
    hit ``max_iter`` without a stable flag set."""

    props: np.ndarray
    adj_props: np.ndarray
    deltas: np.ndarray
    dist: np.ndarray
    axis_par: np.ndarray
    thresholds: np.ndarray
    dif_items: np.ndarray
    n_iter: int
    converged: bool


def _exact_choice(value: object, name: str, allowed: tuple[str, ...]) -> str:
    """Return one exact built-in string from ``allowed`` without callbacks."""
    if type(value) is not str or value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


def _exact_optional_choice(
    value: object, name: str, allowed: tuple[str, ...]
) -> str | None:
    """Return ``None`` or one exact built-in string from ``allowed``."""
    if value is None:
        return None
    return _exact_choice(value, name, allowed)


def _exact_integer(value: object, name: str) -> int:
    """Return one trusted integer without invoking caller conversion hooks."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _exact_real(value: object, name: str) -> float:
    """Return one trusted real scalar without invoking caller conversion hooks."""
    value_type = type(value)
    try:
        if value_type is int or value_type is float:
            normalized = float(value)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            normalized = float(value)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
            normalized = float(value)
        else:
            raise ValueError(f"{name} must be a real number")
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def _exact_const_range(value: object) -> tuple[float, float]:
    """Return one trusted ``(lo, hi)`` pair without caller sequence hooks."""
    if type(value) is not tuple or len(value) != 2:
        raise ValueError("const_range must be a pair of real numbers")
    lower = _exact_real(value[0], "const_range")
    upper = _exact_real(value[1], "const_range")
    if not 0.0 < lower < upper < 1.0:
        raise ValueError("const_range must satisfy 0 < lo < hi < 1")
    return lower, upper


def _normalized_controls(
    threshold: object,
    alpha: object,
    fixed_threshold: object,
    extreme: object,
    const_range: object,
    nr_add: object,
    purify: object,
    max_iter: object,
) -> tuple[str, float, float, str, tuple[float, float], int, str | None, int]:
    """Normalize and domain-check delta-plot controls before side effects."""
    threshold_value = _exact_choice(threshold, "threshold", ("norm", "fixed"))
    extreme_value = _exact_choice(extreme, "extreme", ("constraint", "add"))
    purify_value = _exact_optional_choice(purify, "purify", ("IPP1", "IPP2", "IPP3"))
    alpha_value = _exact_real(alpha, "alpha")
    fixed_threshold_value = _exact_real(fixed_threshold, "fixed_threshold")
    const_range_value = _exact_const_range(const_range)
    nr_add_value = _exact_integer(nr_add, "nr_add")
    max_iter_value = _exact_integer(max_iter, "max_iter")

    if not 0.0 < alpha_value < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if nr_add_value < 1:
        raise ValueError("nr_add must be a positive integer >= 1")
    if not 1 <= max_iter_value <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    return (
        threshold_value,
        alpha_value,
        fixed_threshold_value,
        extreme_value,
        const_range_value,
        nr_add_value,
        purify_value,
        max_iter_value,
    )


def delta_plot(
    responses,
    group,
    *,
    threshold: str = "norm",
    alpha: float = 0.05,
    fixed_threshold: float = 1.5,
    extreme: str = "constraint",
    const_range: tuple[float, float] = (0.001, 0.999),
    nr_add: int = 1,
    purify: str | None = None,
    max_iter: int = 10,
) -> DeltaPlotResult:
    """Angoff Delta plot DIF detection
    (``mlsirm_core::dif::delta_plot``), a computational port of the
    method as IMPLEMENTED by the deltaPlotR R package (READ:
    ``R/deltaPlot.R`` and ``R/adjustExtreme.R`` at cran/deltaPlotR
    commit e2aeeb6; the ``type="response"`` input path only). NOT READ
    (cited only as referenced by the deltaPlotR sources): Angoff & Ford
    (1973); Magis & Facon (2012, 2014). Printing, plotting, and the
    proportion/delta input paths of the R function are out of scope.

    Semantic controls are validated and normalized to exact built-in
    identities before ``responses`` or ``group`` are materialized, so pass
    built-in strings/numbers or supported NumPy scalars rather than
    subclasses or conversion-protocol objects.

    ``responses`` is a 2-D person-by-item matrix of 0s and 1s (NaN =
    missing, dropped per item per group); ``group`` has one entry per
    person, 0 = reference, 1 = focal. ``threshold`` is ``"norm"``
    (normal approximation at ``alpha``) or ``"fixed"`` (flag when
    ``|dist| > |fixed_threshold|``; deltaPlotR's classical default is
    1.5). ``extreme`` is ``"constraint"`` (clamp proportions into
    ``const_range``) or ``"add"`` (replace exact-0/1 proportions using
    ``nr_add`` artificial successes and failures). ``purify`` enables
    iterative item purification with rule ``"IPP1"``, ``"IPP2"``, or
    ``"IPP3"``; a fixed threshold forces IPP1 semantics, matching R.
    The axis slope uses R's ``max(b1, b2)`` root selection (kept even
    under negative delta covariance, source-faithfully). Divergence
    from R: responses other than 0/1/NaN raise ``ValueError`` instead
    of being silently averaged.

    In LLM-as-a-Judge quality management this flags evaluation items
    whose difficulty rank differs between two judge populations (e.g.
    model families) without fitting an IRT model.

    References
    ----------
    Magis, D., & Facon, B. (2014). deltaPlotR: An R package for
    differential item functioning analysis with Angoff's delta plot.
    *Journal of Statistical Software, 59*(Code Snippet 1), 1-19.
    https://doi.org/10.18637/jss.v059.c01 (package paper; the R sources
    listed above were READ and ported.)
    """
    (
        threshold,
        alpha,
        fixed_threshold,
        extreme,
        const_range,
        nr_add,
        purify,
        max_iter,
    ) = _normalized_controls(
        threshold,
        alpha,
        fixed_threshold,
        extreme,
        const_range,
        nr_add,
        purify,
        max_iter,
    )
    xa = np.asarray(responses)
    if xa.ndim != 2:
        raise ValueError("responses must be a 2-D person-by-item matrix")
    n, ni = xa.shape
    if n < 1 or ni < 2:
        raise ValueError("need at least 1 person and 2 items")
    if np.iscomplexobj(xa):
        raise ValueError("responses must be real-valued")
    if xa.dtype.kind == "b":
        xa = xa.astype(np.float64)
    if xa.dtype.kind not in ("i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    xf = np.ascontiguousarray(xa, dtype=np.float64)
    bad = ~(np.isnan(xf) | (xf == 0.0) | (xf == 1.0))
    if bad.any():
        p, i = np.argwhere(bad)[0]
        raise ValueError(
            f"responses[{p}, {i}] = {xf[p, i]}; responses must be 0, 1, or NaN"
        )

    ga = np.asarray(group)
    if ga.ndim != 1 or ga.shape[0] != n:
        raise ValueError("group must be 1-D with one entry per person")
    if np.iscomplexobj(ga):
        raise ValueError("group must be real-valued")
    gf = np.asarray(ga, dtype=np.float64)
    if not np.all((gf == 0.0) | (gf == 1.0)):
        raise ValueError("group entries must be 0 (reference) or 1 (focal)")
    gu = np.ascontiguousarray(gf, dtype=np.uint8)

    if extreme == "constraint":
        ea, eb = const_range
    else:
        ea, eb = float(nr_add), 0.0
    tv = alpha if threshold == "norm" else fixed_threshold

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "py_delta_plot"):
        raise RuntimeError("Rust core with py_delta_plot is required")
    res = core.py_delta_plot(
        xf.ravel(),
        gu,
        int(n),
        int(ni),
        extreme,
        ea,
        eb,
        threshold,
        tv,
        purify,
        max_iter,
    )
    n_iter = int(res["n_iter"])
    return DeltaPlotResult(
        props=np.asarray(res["props"], dtype=np.float64).reshape(ni, 2),
        adj_props=np.asarray(res["adj_props"], dtype=np.float64).reshape(ni, 2),
        deltas=np.asarray(res["deltas"], dtype=np.float64).reshape(ni, 2),
        dist=np.asarray(res["dist"], dtype=np.float64).reshape(n_iter, ni),
        axis_par=np.asarray(res["axis_par"], dtype=np.float64).reshape(n_iter, 2),
        thresholds=np.asarray(res["thresholds"], dtype=np.float64),
        dif_items=np.asarray(res["dif_items"], dtype=np.int64),
        n_iter=n_iter,
        converged=bool(res["converged"]),
    )
