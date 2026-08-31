"""Rating Scale Model (Andrich, 1978): a Rasch-family polytomous model whose
category thresholds are shared across items, estimated by marginal-ML EM in the
Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES
from .irt_contract import MIN_IRT_ITEMS, validate_irt_response_matrix


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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    np.bool_,
    int,
    float,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
)
_ALLOWED_Q_THETA = frozenset({7, 11, 15, 21, 31, 41})
_MAX_RSM_RESPONSE_CELLS = 20_000_000
_MAX_RSM_RESPONSE_STRUCTURAL_NODES = 2 * _MAX_RSM_RESPONSE_CELLS
_RSM_RESOURCE_ERROR = (
    f"responses exceed the {_MAX_RSM_RESPONSE_CELLS}-cell RSM evidence budget"
)
_RSM_MATRIX_SHAPE_ERROR = "responses must be a 2-D persons x items array"
_RSM_MINIMUM_SHAPE_ERROR = (
    "responses must contain at least one person and at least two item columns"
)
_RSM_RESULT_ERROR = "invalid RSM Rust result payload"
_RSM_RESULT_KEYS = frozenset(
    {
        "item_location",
        "thresholds",
        "theta",
        "loglik_trace",
        "n_iter",
        "converged",
        "n_parameters",
    }
)


def _rsm_structural_resource_error() -> str:
    """Return the current structural-work diagnostic, including test overrides."""

    return (
        "responses exceed the "
        f"{_MAX_RSM_RESPONSE_STRUCTURAL_NODES}-node RSM structural evidence budget"
    )


@dataclass
class RsmFit:
    """Fitted rating scale model (Andrich, 1978).

    ``item_location`` is the per-item location ``delta_i``; ``thresholds`` the
    ``n_cat-1`` common category thresholds ``tau_k`` (shared across all items,
    centered so they sum to 0); ``theta`` the per-person EAP trait. The
    adjacent-category log-odds are ``ln[P(k)/P(k-1)] = theta - delta_i - tau_k``."""

    item_location: np.ndarray
    thresholds: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int


def _trusted_optional_category_count(value: int | None) -> int | None:
    """Admit the established built-in category-count contract without callbacks."""
    if value is None:
        return None
    if type(value) is not int:
        raise TypeError("n_cat must be an integer >= 2")
    if not 2 <= value <= MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(f"n_cat must be an integer in 2..{MAX_POLYTOMOUS_CATEGORIES}")
    return value


def _trusted_quadrature_points(value: int) -> int:
    """Return an established RSM quadrature size without caller hash/coercion hooks."""
    value_type = type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        normalized = int(value)
    else:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    if normalized not in _ALLOWED_Q_THETA:
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    return normalized


def _trusted_iteration_cap(value: int) -> int:
    """Admit the established built-in iteration cap without caller callbacks."""
    if type(value) is not int or not 1 <= value <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    return value


def _trusted_positive_tolerance(value: float) -> float:
    """Return a finite positive value that survives the Rust ``f64`` boundary exactly."""
    error = "tol must be finite and > 0"
    value_type = type(value)
    try:
        if value_type is int:
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != value:
                raise ValueError(error)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or int(normalized) != int(value):
                raise ValueError(error)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
            normalized = float(value)
            if not np.isfinite(normalized) or value_type(normalized) != value:
                raise ValueError(error)
        else:
            raise ValueError(error)
    except (OverflowError, ValueError):
        raise ValueError(error) from None
    if not np.isfinite(normalized) or normalized <= 0:
        raise ValueError(error)
    return normalized


def _trusted_response_source(value: object) -> object:
    """Admit inert, bounded response containers before NumPy invokes protocols."""
    if type(value) is np.ndarray:
        if int(value.size) > _MAX_RSM_RESPONSE_CELLS:
            raise ValueError(_RSM_RESOURCE_ERROR)
        if value.ndim != 2:
            raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
        if int(value.shape[0]) < 1 or int(value.shape[1]) < MIN_IRT_ITEMS:
            raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)
        return value
    if type(value) is not list and type(value) is not tuple:
        raise ValueError("responses must be a real numeric array")

    logical_cells = 0
    structural_nodes = 0
    matrix_like = len(value) > 0
    matrix_width: int | None = None
    for row in value:
        structural_nodes += 1
        if structural_nodes > _MAX_RSM_RESPONSE_STRUCTURAL_NODES:
            raise ValueError(_rsm_structural_resource_error())
        row_type = type(row)
        if row_type is np.ndarray:
            logical_cells += int(row.size)
            if logical_cells > _MAX_RSM_RESPONSE_CELLS:
                raise ValueError(_RSM_RESOURCE_ERROR)
            if row.ndim != 1:
                matrix_like = False
            else:
                row_width = int(row.shape[0])
                if matrix_width is None:
                    matrix_width = row_width
                elif matrix_width != row_width:
                    matrix_like = False
            continue
        if row_type is list or row_type is tuple:
            row_length = len(row)
            logical_cells += row_length
            if logical_cells > _MAX_RSM_RESPONSE_CELLS:
                raise ValueError(_RSM_RESOURCE_ERROR)
            structural_nodes += row_length
            if structural_nodes > _MAX_RSM_RESPONSE_STRUCTURAL_NODES:
                raise ValueError(_rsm_structural_resource_error())
            if matrix_width is None:
                matrix_width = row_length
            elif matrix_width != row_length:
                matrix_like = False
            continue
        # Flat built-in scalar sequences are known not to satisfy the established
        # persons-by-items matrix contract, but still charge their bounded work
        # before the stable 2-D diagnostic is replayed below.
        matrix_like = False
        if row_type not in _TRUSTED_RESPONSE_SCALAR_TYPES:
            raise ValueError("responses must be a real numeric array")
        logical_cells += 1
        if logical_cells > _MAX_RSM_RESPONSE_CELLS:
            raise ValueError(_RSM_RESOURCE_ERROR)

    if not matrix_like:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
    if matrix_width is not None and matrix_width < MIN_IRT_ITEMS:
        raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)

    # Value-wise validation is intentionally deferred until resource, matrix-rank,
    # rectangularity, and structurally impossible item-count checks have completed.
    for row in value:
        if type(row) is list or type(row) is tuple:
            if any(type(cell) not in _TRUSTED_RESPONSE_SCALAR_TYPES for cell in row):
                raise ValueError("responses must be a real numeric array")
    return value


def _real_numeric_response_matrix(value: object) -> np.ndarray:
    """Admit bounded real numeric response storage before ``float64`` marshalling."""
    source = _trusted_response_source(value)
    if type(source) is np.ndarray:
        admitted_shape = tuple(int(axis) for axis in source.shape)
        admitted_size = int(source.size)
        admitted_dtype = source.dtype
        if admitted_dtype.kind not in {"b", "i", "u", "f"}:
            raise ValueError("responses must be a real numeric array")

        snapshot = np.array(source, copy=True, order="K", subok=False)
        if (
            type(snapshot) is not np.ndarray
            or snapshot.ndim != 2
            or tuple(int(axis) for axis in snapshot.shape) != admitted_shape
            or int(snapshot.size) != admitted_size
            or snapshot.dtype != admitted_dtype
            or snapshot.dtype.kind not in {"b", "i", "u", "f"}
            or not snapshot.flags.owndata
            or np.shares_memory(snapshot, source)
        ):
            raise ValueError("responses must be a real numeric array")
        return np.ascontiguousarray(snapshot, dtype=np.float64)

    source_len = len(source)
    if type(source) is list:
        rows = source.copy()
        if len(rows) != source_len or len(source) != source_len:
            raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
    else:
        rows = source
    if len(rows) != source_len or not rows:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)

    first_row = rows[0]
    if type(first_row) is np.ndarray:
        expected_width = int(first_row.size)
    else:
        expected_width = len(first_row)

    row_snapshots: list[tuple[object, ...] | np.ndarray] = []
    for row in rows:
        row_type = type(row)
        if row_type is np.ndarray:
            admitted_size = int(row.size)
            admitted_dtype = row.dtype
            if (
                row.ndim != 1
                or admitted_size != expected_width
                or admitted_dtype.kind not in {"b", "i", "u", "f"}
            ):
                if row.ndim != 1 or admitted_size != expected_width:
                    raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
                raise ValueError("responses must be a real numeric array")
            row_snapshot = np.array(row, copy=True, order="K", subok=False)
            if (
                type(row_snapshot) is not np.ndarray
                or row_snapshot.ndim != 1
                or int(row_snapshot.size) != admitted_size
                or int(row_snapshot.size) != expected_width
                or row_snapshot.dtype != admitted_dtype
                or row_snapshot.dtype.kind not in {"b", "i", "u", "f"}
                or not row_snapshot.flags.owndata
                or np.shares_memory(row_snapshot, row)
            ):
                raise ValueError("responses must be a real numeric array")
            row_snapshots.append(row_snapshot)
            continue

        if row_type is list:
            admitted_size = len(row)
            if admitted_size != expected_width:
                raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
            row_snapshot = tuple(row)
            if len(row_snapshot) != admitted_size or len(row) != admitted_size:
                raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
        elif row_type is tuple:
            row_snapshot = row
            if len(row_snapshot) != expected_width:
                raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
        else:
            raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
        if any(type(cell) not in _TRUSTED_RESPONSE_SCALAR_TYPES for cell in row_snapshot):
            raise ValueError("responses must be a real numeric array")
        row_snapshots.append(row_snapshot)

    if type(source) is list and len(source) != source_len:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)

    array = np.asarray(row_snapshots)
    if np.iscomplexobj(array) or array.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError("responses must be a real numeric array")
    return np.ascontiguousarray(array, dtype=np.float64)


def _sealed_native_vector(
    value: object,
    *,
    expected_len: int | None = None,
    require_nonempty: bool = False,
) -> np.ndarray:
    """Seal the exact built-in list emitted by the PyO3 ``Vec<f64>`` binding."""

    if type(value) is not list:
        raise RuntimeError(_RSM_RESULT_ERROR)
    source_len = len(value)
    if expected_len is not None and source_len != expected_len:
        raise RuntimeError(_RSM_RESULT_ERROR)
    if require_nonempty and source_len == 0:
        raise RuntimeError(_RSM_RESULT_ERROR)

    snapshot = value.copy()
    if type(snapshot) is not list or len(snapshot) != source_len:
        raise RuntimeError(_RSM_RESULT_ERROR)
    if expected_len is not None and len(snapshot) != expected_len:
        raise RuntimeError(_RSM_RESULT_ERROR)
    if require_nonempty and len(snapshot) == 0:
        raise RuntimeError(_RSM_RESULT_ERROR)
    if any(type(item) is not float or not np.isfinite(item) for item in snapshot):
        raise RuntimeError(_RSM_RESULT_ERROR)

    array = np.array(snapshot, dtype=np.float64, copy=True, order="C")
    if (
        type(array) is not np.ndarray
        or array.dtype != np.dtype(np.float64)
        or array.ndim != 1
        or int(array.shape[0]) != len(snapshot)
        or not array.flags.c_contiguous
        or not array.flags.owndata
        or not np.all(np.isfinite(array))
    ):
        raise RuntimeError(_RSM_RESULT_ERROR)
    return array


def _validated_native_result(
    value: object,
    *,
    n_persons: int,
    n_items: int,
    n_cat: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, int]:
    """Seal and validate the deterministic Rust RSM result envelope."""

    if type(value) is not dict or len(value) != len(_RSM_RESULT_KEYS):
        raise RuntimeError(_RSM_RESULT_ERROR)
    snapshot = dict.copy(value)
    if len(snapshot) != len(_RSM_RESULT_KEYS):
        raise RuntimeError(_RSM_RESULT_ERROR)
    if any(type(key) is not str for key in snapshot):
        raise RuntimeError(_RSM_RESULT_ERROR)
    if frozenset(snapshot) != _RSM_RESULT_KEYS:
        raise RuntimeError(_RSM_RESULT_ERROR)

    item_location = _sealed_native_vector(
        dict.__getitem__(snapshot, "item_location"), expected_len=n_items
    )
    thresholds = _sealed_native_vector(
        dict.__getitem__(snapshot, "thresholds"), expected_len=n_cat - 1
    )
    theta = _sealed_native_vector(
        dict.__getitem__(snapshot, "theta"), expected_len=n_persons
    )
    loglik_trace = _sealed_native_vector(
        dict.__getitem__(snapshot, "loglik_trace"), require_nonempty=True
    )

    n_iter = dict.__getitem__(snapshot, "n_iter")
    converged = dict.__getitem__(snapshot, "converged")
    n_parameters = dict.__getitem__(snapshot, "n_parameters")
    if (
        type(n_iter) is not int
        or not 1 <= n_iter <= max_iter
        or type(converged) is not bool
        or type(n_parameters) is not int
        or n_parameters != n_items + n_cat - 2
    ):
        raise RuntimeError(_RSM_RESULT_ERROR)
    expected_trace_len = n_iter if converged else n_iter + 1
    if int(loglik_trace.shape[0]) != expected_trace_len:
        raise RuntimeError(_RSM_RESULT_ERROR)

    return (
        item_location,
        thresholds,
        theta,
        loglik_trace,
        n_iter,
        converged,
        n_parameters,
    )


def fit_rsm(
    responses: np.ndarray,
    n_cat: int | None = None,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> RsmFit:
    """Fit the rating scale model (compute in Rust; Andrich, 1978).

    The RSM is the Rasch-family polytomous model for items on a common rating scale
    (e.g. Likert): every item has its own location ``delta_i``, but the ``K-1``
    category thresholds ``tau_k`` are *shared across all items*. The
    adjacent-category log-odds are ``ln[P(X=k)/P(X=k-1)] = theta - delta_i - tau_k``,
    ``theta ~ N(0,1)``. This is a constrained partial-credit model (the PCM has
    item-specific thresholds); at ``K=2`` it reduces to the Rasch model. Estimated by
    marginal-ML EM with a Gauss-Hermite trait grid; the item locations and the shared
    thresholds are updated by a monotone ECM step and the thresholds are centered to
    sum to zero.

    ``responses`` is a persons x items array of integer category indices
    ``0..n_cat-1`` with at least two item columns (``NaN`` marks a missing cell,
    dropped under a missing-at-random assumption). ``n_cat`` defaults to
    ``max(responses) + 1``.

    References (APA 7th ed.):
        Andrich, D. (1978). A rating formulation for ordered response categories.
            *Psychometrika, 43*(4), 561-573. https://doi.org/10.1007/BF02293814
    """
    n_cat = _trusted_optional_category_count(n_cat)
    q_theta = _trusted_quadrature_points(q_theta)
    max_iter = _trusted_iteration_cap(max_iter)
    tol = _trusted_positive_tolerance(tol)

    y = _real_numeric_response_matrix(responses)
    if y.ndim != 2:
        raise ValueError(_RSM_MATRIX_SHAPE_ERROR)
    n_persons, n_items = y.shape
    if n_persons < 1 or n_items < MIN_IRT_ITEMS:
        raise ValueError(_RSM_MINIMUM_SHAPE_ERROR)
    missing = np.isnan(y)
    if np.any(~missing & ~np.isfinite(y)):
        raise ValueError("observed responses must be finite integer categories")
    observed = ~missing
    obs_values = y[observed]
    if obs_values.size and (
        np.any(obs_values != np.floor(obs_values)) or np.any(obs_values < 0)
    ):
        raise ValueError("observed responses must be non-negative integer categories")
    if n_cat is None:
        if obs_values.size == 0:
            raise ValueError("responses has no observed values")
        n_cat = int(obs_values.max()) + 1
        if n_cat < 2:
            raise ValueError("responses must contain at least two categories")
        if n_cat > MAX_POLYTOMOUS_CATEGORIES:
            raise ValueError(
                f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
            )
    if obs_values.size and np.any(obs_values >= n_cat):
        raise ValueError(
            f"observed responses must be integer categories in 0..{n_cat - 1}"
        )
    missing_items = np.flatnonzero(~observed.any(axis=0))
    if missing_items.size:
        raise ValueError(f"item {int(missing_items[0])} has no observed responses")
    validate_irt_response_matrix(y, "polytomous", n_categories=n_cat)
    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        raise RuntimeError("fit_rsm requires the compiled Rust core")

    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)
    res = core.fit_rsm(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        n_cat,
        q_theta,
        max_iter,
        tol,
    )
    (
        item_location,
        thresholds,
        theta,
        loglik_trace,
        n_iter,
        converged,
        n_parameters,
    ) = _validated_native_result(
        res,
        n_persons=int(n_persons),
        n_items=int(n_items),
        n_cat=n_cat,
        max_iter=max_iter,
    )
    return RsmFit(
        item_location=item_location,
        thresholds=thresholds,
        theta=theta,
        loglik_trace=loglik_trace,
        n_iter=n_iter,
        converged=converged,
        n_parameters=n_parameters,
    )