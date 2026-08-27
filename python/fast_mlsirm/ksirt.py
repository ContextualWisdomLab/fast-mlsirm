"""Kernel-smoothing nonparametric IRT: option characteristic curves by
Nadaraya-Watson regression on rank-based ordinal ability estimates (Ramsay,
1991, as cited in Mazza et al., 2014). All numeric work happens in the Rust
core; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_TRUSTED_NUMPY_INTEGER_TYPES = (
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
_TRUSTED_REAL_SEQUENCE_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        complex,
        np.bool_,
        *_TRUSTED_NUMPY_INTEGER_TYPES,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
    }
)
_KSIRT_KERNELS = ("gaussian", "quadratic", "uniform")
_MAX_KSIRT_RESPONSE_CELLS = 20_000_000
_MAX_KSIRT_RESPONSE_STRUCTURAL_NODES = 40_000_000


@dataclass
class KsirtResult:
    """Kernel-smoothed option characteristic curves."""

    theta: np.ndarray
    grid: np.ndarray
    bandwidth: np.ndarray
    options: list[np.ndarray]
    occ: list[np.ndarray]
    expected: list[np.ndarray]
    expected_total: np.ndarray


def _kernel_control(value: object) -> str:
    """Return a trusted KSIRT kernel name without caller-owned callbacks."""
    value_type = type(value)
    if value_type is str:
        normalized = value
    elif value_type is np.str_:
        normalized = str(value)
    else:
        raise ValueError("kernel must be gaussian, quadratic, or uniform")
    if normalized not in _KSIRT_KERNELS:
        raise ValueError("kernel must be gaussian, quadratic, or uniform")
    return normalized


def _nevalpoints_control(value: object) -> int:
    """Return a bounded evaluation-grid size without implicit coercion."""
    value_type = type(value)
    if value_type is int:
        parsed = value
    elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES):
        parsed = int(value)
    else:
        raise ValueError("nevalpoints must be an integer")
    if parsed < 2:
        raise ValueError("nevalpoints must be at least 2")
    if parsed > 100_000:
        raise ValueError("nevalpoints must be at most 100000")
    return parsed


def _response_shape_before_materialization(value: object) -> tuple[int, int]:
    """Return a bounded rectangular response shape without NumPy conversion."""
    if type(value) is np.ndarray:
        if value.ndim != 2:
            raise ValueError("responses must be a 2-D persons x items array")
        n_persons, n_items = value.shape
        if value.size > _MAX_KSIRT_RESPONSE_CELLS:
            raise ValueError(
                f"responses exceed {_MAX_KSIRT_RESPONSE_CELLS} logical cells"
            )
        return int(n_persons), int(n_items)

    if type(value) not in (list, tuple) or not value:
        raise ValueError("responses must be a 2-D persons x items array")

    n_persons = len(value)
    n_items: int | None = None
    logical_cells = 0
    structural_nodes = 0
    for row in value:
        if type(row) in (list, tuple):
            row_items = len(row)
        elif type(row) is np.ndarray and row.ndim == 1:
            row_items = int(row.shape[0])
        else:
            raise ValueError("responses must be a 2-D persons x items array")

        if n_items is None:
            n_items = row_items
        elif row_items != n_items:
            raise ValueError("responses must be a numeric array")

        logical_cells += row_items
        if logical_cells > _MAX_KSIRT_RESPONSE_CELLS:
            raise ValueError(
                f"responses exceed {_MAX_KSIRT_RESPONSE_CELLS} logical cells"
            )

        structural_nodes += 1 + row_items
        if structural_nodes > _MAX_KSIRT_RESPONSE_STRUCTURAL_NODES:
            raise ValueError(
                "responses exceed "
                f"{_MAX_KSIRT_RESPONSE_STRUCTURAL_NODES} structural nodes"
            )

    return n_persons, 0 if n_items is None else n_items


def _scalar_preserves_float64_identity(value: object) -> bool:
    """Return whether one trusted scalar survives exact Rust-f64 normalization."""
    value_type = type(value)
    if value_type in (bool, np.bool_):
        return True
    if value_type is int or any(
        value_type is trusted for trusted in _TRUSTED_NUMPY_INTEGER_TYPES
    ):
        exact = int(value)
        try:
            narrowed = float(exact)
        except OverflowError:
            return False
        return np.isfinite(narrowed) and int(narrowed) == exact
    if value_type in (float, np.float16, np.float32, np.float64):
        return True
    if value_type is np.longdouble:
        if not np.isfinite(value):
            return True
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = np.float64(value)
        return bool(np.isfinite(narrowed) and np.longdouble(narrowed) == value)
    if value_type in (complex, np.complex64, np.complex128, np.clongdouble):
        return True
    return False


def _array_preserves_float64_identity(value: np.ndarray) -> bool:
    """Return whether finite real array values survive exact float64 normalization."""
    if value.dtype.kind in ("i", "u") and value.dtype.itemsize > 4:
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = value.astype(np.float64)
            roundtrip = narrowed.astype(value.dtype)
        return bool(np.array_equal(roundtrip, value))
    if value.dtype.kind == "f" and value.dtype.itemsize > np.dtype(np.float64).itemsize:
        finite = np.isfinite(value)
        with np.errstate(over="ignore", invalid="ignore"):
            narrowed = value.astype(np.float64)
            roundtrip = narrowed.astype(value.dtype)
        return bool(np.array_equal(roundtrip[finite], value[finite]))
    return True


def _trusted_numeric_storage(value: object, name: str) -> np.ndarray:
    """Materialize only inert numeric identities without losing finite values."""
    if type(value) is np.ndarray:
        raw = value
    elif type(value) in (list, tuple):
        stack: list[object] = [value]
        while stack:
            current = stack.pop()
            if type(current) in (list, tuple):
                stack.extend(current)
                continue
            if type(current) is np.ndarray:
                if current.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(f"{name} must be a numeric array")
                if not _array_preserves_float64_identity(current):
                    raise ValueError(
                        f"{name} entries must be exactly representable as float64"
                    )
                continue
            if type(current) not in _TRUSTED_REAL_SEQUENCE_SCALAR_TYPES:
                raise ValueError(f"{name} must be a numeric array")
            if not _scalar_preserves_float64_identity(current):
                raise ValueError(
                    f"{name} entries must be exactly representable as float64"
                )
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a numeric array") from None
    else:
        raise ValueError(f"{name} must be a numeric array")
    return raw


def _lossless_float64_array(raw: np.ndarray, name: str) -> np.ndarray:
    """Normalize trusted real evidence without changing any finite value."""
    if not _array_preserves_float64_identity(raw):
        raise ValueError(f"{name} entries must be exactly representable as float64")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be numeric and convertible to float64") from None


def _real_float_array(value: object, name: str) -> np.ndarray:
    """Materialize trusted real numeric storage without caller callbacks."""
    raw = _trusted_numeric_storage(value, name)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError(f"{name} must be a numeric array")
    return _lossless_float64_array(raw, name)


def ksirt_analysis(
    responses: np.ndarray,
    kernel: str = "gaussian",
    nevalpoints: int = 51,
    bandwidth: np.ndarray | None = None,
) -> KsirtResult:
    """Kernel smoothing of option characteristic curves (compute in Rust;
    Ramsay, 1991, as cited in Mazza et al., 2014).

    Estimates each item's option characteristic curves by Nadaraya-Watson
    kernel regression of the option indicators on ordinal ability estimates
    ``Phi^-1(rank(total score)/(n+1))`` (ties broken by subject order),
    evaluated on an equally-spaced grid from ``Phi^-1(1/(n+1))`` to
    ``Phi^-1(n/(n+1))``. The default bandwidth is Silverman's rule
    ``1.06 * n^(-1/5)`` on the standard-normal ability metric. Formulas
    follow Mazza et al. (2014, Sections 2-2.3) and the KernSmoothIRT R
    package source (both read); Ramsay (1991) itself is cited only through
    Mazza et al. (2014). Standard errors and cross-validation bandwidths
    are deliberately not implemented (see the Rust module docs).

    In LLM-as-a-Judge item-quality management, nonparametric OCCs reveal
    non-monotone or poorly discriminating evaluation items without assuming
    a parametric response model.

    ``responses`` is a complete ``persons x items`` array of pre-scored
    numeric responses; each column's distinct values form that item's
    options. ``kernel`` is ``"gaussian"``, ``"quadratic"``, or
    ``"uniform"``. ``bandwidth`` optionally gives one positive value per
    item. Semantic controls are normalized before caller array materialization
    or compiled-core discovery. Response shape, logical cell count, minimum
    dimensions, built-in structural work, and finite-value identity through
    Rust ``f64`` normalization are validated before dispatch. Evidence admission
    accepts exact NumPy arrays or plain built-in list/tuple trees of trusted
    concrete numeric scalars; callback-bearing providers, complex values, and
    object/text storage fail before real-valued marshalling.

    References (APA 7th ed.):
        Mazza, A., Punzo, A., & McGuire, B. (2014). KernSmoothIRT: An R
            package for kernel smoothing in item response theory. *Journal
            of Statistical Software, 58*(6), 1-34.
            https://doi.org/10.18637/jss.v058.i06
        Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric
            item characteristic curve estimation. *Psychometrika, 56*(4),
            611-630. https://doi.org/10.1007/BF02294494 (as cited in Mazza
            et al., 2014)
    """
    kernel_value = _kernel_control(kernel)
    nevalpoints_value = _nevalpoints_control(nevalpoints)

    n_persons, n_items = _response_shape_before_materialization(responses)
    if n_persons < 2 or n_items < 1:
        raise ValueError("responses needs at least 2 persons and 1 item")
    y = _real_float_array(responses, "responses")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    bw = None
    if bandwidth is not None:
        bw_arr = _real_float_array(bandwidth, "bandwidth").reshape(-1)
        if bw_arr.shape[0] != n_items:
            raise ValueError("bandwidth must supply one value per item")
        if not np.all(np.isfinite(bw_arr)) or np.any(bw_arr <= 0.0):
            raise ValueError("bandwidths must be finite and positive")
        bw = [float(v) for v in bw_arr]

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "ksirt_occ"):
        raise RuntimeError("ksirt_analysis requires the compiled Rust core")

    res = core.ksirt_occ(
        y.reshape(-1),
        int(n_persons),
        int(n_items),
        kernel_value,
        nevalpoints_value,
        bw,
    )
    grid = np.asarray(res["grid"], dtype=np.float64)
    q = grid.shape[0]
    options = [np.asarray(o, dtype=np.float64) for o in res["options"]]
    occ = [
        np.asarray(flat, dtype=np.float64).reshape(len(opts), q)
        for flat, opts in zip(res["occ"], options)
    ]
    return KsirtResult(
        theta=np.asarray(res["theta"], dtype=np.float64),
        grid=grid,
        bandwidth=np.asarray(res["bandwidth"], dtype=np.float64),
        options=options,
        occ=occ,
        expected=[np.asarray(e, dtype=np.float64) for e in res["expected"]],
        expected_total=np.asarray(res["expected_total"], dtype=np.float64),
    )
