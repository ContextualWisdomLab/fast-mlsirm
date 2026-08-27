"""Linear Logistic Test Model (Fischer, 1973): an explanatory Rasch model in which
item difficulties are a linear combination of basic cognitive-operation parameters
through a fixed design matrix, estimated by marginal-ML EM in the Rust core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


_NUMPY_INTEGER_TYPES = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_TYPES = (np.complex64, np.complex128, np.clongdouble)
_TRUSTED_REAL_SCALAR_TYPES = (
    bool,
    int,
    float,
    np.bool_,
    *_NUMPY_INTEGER_TYPES,
    *_NUMPY_FLOAT_TYPES,
)
_TRUSTED_COMPLEX_SCALAR_TYPES = (complex, *_NUMPY_COMPLEX_TYPES)


def _boolean(value: object, name: str) -> bool:
    """Normalize one trusted Boolean control without caller truth-value callbacks."""

    value_type = type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def _positive_integer(value: object, name: str) -> int:
    """Normalize one trusted positive integer without caller conversion protocols."""

    value_type = type(value)
    if value_type is int:
        normalized = value
    elif value_type in _NUMPY_INTEGER_TYPES:
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_finite_real(value: object, name: str) -> float:
    """Normalize one trusted finite non-negative real without caller coercion hooks."""

    value_type = type(value)
    if value_type not in (int, float, *_NUMPY_INTEGER_TYPES, *_NUMPY_FLOAT_TYPES):
        raise ValueError(f"{name} must be finite and non-negative")
    try:
        normalized = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and non-negative") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _matrix_shape_error(name: str) -> str:
    if name == "responses":
        return "responses must be a 2-D persons x items array"
    return "q_design must be a 2-D items x basic-operations array"


def _validate_ndarray_storage(value: np.ndarray, name: str, *, row: bool) -> int:
    """Validate one exact NumPy carrier without invoking caller conversion protocols."""

    expected_ndim = 1 if row else 2
    if value.ndim != expected_ndim:
        raise ValueError(_matrix_shape_error(name))
    kind = value.dtype.kind
    if kind == "c":
        raise ValueError(f"{name} must be real-valued")
    if kind not in "biuf":
        raise ValueError(f"{name} must contain real-valued numeric evidence")
    return int(value.shape[0] if row else value.shape[1])


def _materialize_real_matrix(value: object, name: str) -> np.ndarray:
    """Create the package-owned float64 matrix after callback-free preflight."""

    try:
        materialized = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real-valued numeric evidence") from exc
    if materialized.ndim != 2:
        raise ValueError(_matrix_shape_error(name))
    return materialized


def _trusted_real_matrix(value: object, name: str) -> np.ndarray:
    """Materialize one callback-free real matrix after exact carrier preflight."""

    value_type = type(value)
    if value_type is np.ndarray:
        _validate_ndarray_storage(value, name, row=False)
        return _materialize_real_matrix(value, name)

    if value_type not in (list, tuple):
        raise ValueError(f"{name} must be an exact NumPy array or built-in matrix")
    if not value:
        raise ValueError(_matrix_shape_error(name))

    width: int | None = None
    for row_value in value:
        row_type = type(row_value)
        if row_type is np.ndarray:
            row_width = _validate_ndarray_storage(row_value, name, row=True)
        elif row_type in (list, tuple):
            row_width = len(row_value)
            for scalar in row_value:
                scalar_type = type(scalar)
                if scalar_type in _TRUSTED_COMPLEX_SCALAR_TYPES:
                    raise ValueError(f"{name} must be real-valued")
                if scalar_type not in _TRUSTED_REAL_SCALAR_TYPES:
                    raise ValueError(f"{name} must contain real-valued numeric evidence")
        else:
            raise ValueError(_matrix_shape_error(name))

        if width is None:
            width = row_width
        elif row_width != width:
            raise ValueError(_matrix_shape_error(name))

    return _materialize_real_matrix(value, name)


@dataclass
class LltmFit:
    """Fitted LLTM (Fischer, 1973).

    ``eta`` are the basic-operation easiness parameters (Fischer difficulty =
    ``-eta``); ``intercept`` the grand-mean easiness ``c`` (``NaN`` if not fit);
    ``b`` the induced item easinesses ``c + Q @ eta``; ``theta`` the person EAP
    abilities. When the LR test is computed, ``lr_stat``/``lr_df``/``lr_p`` give the
    likelihood-ratio test of the LLTM restriction against the saturated Rasch model
    (a small ``lr_p`` means the cognitive-operation decomposition does NOT fully
    explain the item difficulties)."""

    eta: np.ndarray
    intercept: float
    b: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    n_parameters: int
    loglik_rasch: float
    lr_stat: float
    lr_df: int
    lr_p: float


def fit_lltm(
    responses: np.ndarray,
    q_design: np.ndarray,
    fit_intercept: bool = True,
    compute_lr: bool = True,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> LltmFit:
    """Fit the Linear Logistic Test Model (compute in Rust; Fischer, 1973).

    LLTM is an *explanatory* Rasch model: item ``i``'s easiness (the sign convention
    returned here) is not free but a linear image
    ``b_i = c + sum_k q_ik * eta_k`` of ``K`` basic cognitive-operation parameters
    through a fixed weight matrix ``q_design`` (``q_ik`` = how many times operation
    ``k`` is engaged by item ``i``). With ``K << J`` parameters it tests
    whether a small set of operations explains the item parameters; the returned
    likelihood-ratio test against the saturated Rasch model is its classic use.

    ``responses`` is a persons x items 0/1 array (``NaN`` = missing, dropped under
    MAR). ``q_design`` is an items x basic-operations real array. The design must have
    full column rank (with the intercept column when ``fit_intercept``) for ``eta`` to
    be identified — a rank-deficient design (e.g. rows summing to a constant while
    fitting an intercept) is rejected.

    Fischer's (1973, 1995) canonical LLTM uses conditional maximum likelihood. This
    function instead fixes the ability distribution to ``N(0,1)`` and uses a
    Bock-Aitkin-style marginal-ML EM algorithm. This is a repository-specific
    estimator choice; finite-sample equality with Fischer's conditional-ML item
    estimates is not assumed.

    References (APA 7th ed.):
        Fischer, G. H. (1973). The linear logistic test model as an instrument in
            educational research. *Acta Psychologica, 37*(6), 359–374.
            https://doi.org/10.1016/0001-6918(73)90003-6
        Fischer, G. H. (1995). The linear logistic test model. In G. H. Fischer & I.
            W. Molenaar (Eds.), *Rasch models: Foundations, recent developments, and
            applications* (pp. 131–155). Springer.
            https://doi.org/10.1007/978-1-4612-4230-7_8
        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
            item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
            443–459.
            https://doi.org/10.1007/BF02293801
    """
    normalized_fit_intercept = _boolean(fit_intercept, "fit_intercept")
    normalized_compute_lr = _boolean(compute_lr, "compute_lr")
    normalized_max_iter = _positive_integer(max_iter, "max_iter")
    normalized_tol = _nonnegative_finite_real(tol, "tol")

    y = _trusted_real_matrix(responses, "responses")
    q = _trusted_real_matrix(q_design, "q_design")
    if np.isinf(y).any():
        raise ValueError("responses must contain only finite values or NaN missingness")
    if not np.isfinite(q).all():
        raise ValueError("q_design entries must be finite")

    n_persons, n_items = y.shape
    if q.shape[0] != n_items:
        raise ValueError("q_design must have one row per item")
    n_basic = q.shape[1]

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_lltm"):
        raise RuntimeError("fit_lltm requires the compiled Rust core")

    observed = ~np.isnan(y)
    yy = np.where(observed, y, 0.0).reshape(-1)
    res = core.fit_lltm(
        yy,
        observed.reshape(-1),
        q.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_basic),
        normalized_fit_intercept,
        normalized_compute_lr,
        normalized_max_iter,
        normalized_tol,
    )
    return LltmFit(
        eta=np.asarray(res["eta"], dtype=np.float64),
        intercept=float(res["intercept"]),
        b=np.asarray(res["b"], dtype=np.float64),
        theta=np.asarray(res["theta"], dtype=np.float64),
        loglik_trace=np.asarray(res["loglik_trace"], dtype=np.float64),
        n_iter=int(res["n_iter"]),
        converged=bool(res["converged"]),
        n_parameters=int(res["n_parameters"]),
        loglik_rasch=float(res["loglik_rasch"]),
        lr_stat=float(res["lr_stat"]),
        lr_df=int(res["lr_df"]),
        lr_p=float(res["lr_p"]),
    )
