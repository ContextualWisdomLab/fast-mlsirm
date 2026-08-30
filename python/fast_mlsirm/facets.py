"""Many-Facet Rasch Model (Linacre, 1989): the rating-scale Rasch model with a
rater-severity facet, estimated by marginal-ML EM in the Rust core. All numeric
work happens in Rust; this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .config import MAX_MAX_ITER, MAX_POLYTOMOUS_CATEGORIES


_MAX_FACETS_RESPONSE_CELLS = 20_000_000
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


def _integer_control(value: object, name: str, *, allow_none: bool = False) -> int | None:
    """Normalize one trusted integer control without caller callbacks."""
    if value is None and allow_none:
        return None
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is trusted_type for trusted_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _real_control(value: object, name: str) -> float:
    """Normalize one trusted real control without caller callbacks."""
    value_type = type(value)
    if value_type is int or value_type is float:
        try:
            return float(value)
        except OverflowError:
            raise ValueError(f"{name} must be finite and > 0") from None
    if any(
        value_type is trusted_type
        for trusted_type in (*_NUMPY_INTEGER_SCALAR_TYPES, *_NUMPY_FLOAT_SCALAR_TYPES)
    ):
        return float(value)
    raise ValueError(f"{name} must be a real number")


def _response_array(value: object) -> np.ndarray:
    """Materialize inert real-numeric response evidence after bounded preflight."""

    resource_error = (
        f"responses must contain at most {_MAX_FACETS_RESPONSE_CELLS:,} logical cells"
    )
    dimension_error = "responses must be a 3-D persons x items x raters array"
    nonempty_error = (
        "responses must contain at least one person, one item and one rater"
    )
    value_type = type(value)
    if value_type is np.ndarray:
        if value.size > _MAX_FACETS_RESPONSE_CELLS:
            raise ValueError(resource_error)
        response_array = value
    elif value_type is list or value_type is tuple:
        if len(value) == 0:
            raise ValueError(nonempty_error)

        # A valid non-empty 3-D built-in tree with N scalar cells visits at most
        # N person nodes + N item nodes + N scalar nodes.  Bounding traversal at
        # three times the logical-cell ceiling therefore preserves every valid
        # rectangular input admitted by the public 20M-cell contract while
        # preventing malformed empty-container fan-out from consuming
        # unbounded Python work before NumPy materialization.
        structural_budget = 3 * _MAX_FACETS_RESPONSE_CELLS
        structural_nodes = 0

        # Indexed frames keep temporary traversal state proportional to nesting
        # depth instead of sibling width. The public evidence contract is
        # exactly persons -> items -> raters -> scalar leaves for built-in
        # sequences, so rectangularity can also be proven before NumPy sees the
        # evidence.
        frames: list[list[object]] = [[value, 0, 0]]
        active_container_ids: set[int] = {id(value)}
        logical_cells = 0
        expected_items: int | None = None
        expected_raters: int | None = None

        while frames:
            frame = frames[-1]
            current = frame[0]
            child_index = int(frame[1])
            depth = int(frame[2])

            if child_index >= len(current):
                active_container_ids.remove(id(current))
                frames.pop()
                continue

            frame[1] = child_index + 1
            structural_nodes += 1
            if structural_nodes > structural_budget:
                raise ValueError(
                    "responses exceeded structural traversal budget of "
                    f"{structural_budget:,} nodes"
                )

            child = current[child_index]
            child_type = type(child)
            next_depth = depth + 1

            if child_type is list or child_type is tuple:
                if next_depth >= 3:
                    raise ValueError(dimension_error)
                child_length = len(child)
                if next_depth == 1:
                    if expected_items is None:
                        expected_items = child_length
                    elif child_length != expected_items:
                        raise ValueError(dimension_error)
                else:
                    if expected_raters is None:
                        expected_raters = child_length
                    elif child_length != expected_raters:
                        raise ValueError(dimension_error)
                child_id = id(child)
                if child_id in active_container_ids:
                    raise ValueError(dimension_error)
                active_container_ids.add(child_id)
                frames.append([child, 0, next_depth])
                continue

            if depth != 2:
                raise ValueError(dimension_error)

            trusted_scalar = (
                child_type is bool
                or child_type is np.bool_
                or child_type is int
                or child_type is float
                or any(
                    child_type is trusted_type
                    for trusted_type in (
                        *_NUMPY_INTEGER_SCALAR_TYPES,
                        *_NUMPY_FLOAT_SCALAR_TYPES,
                    )
                )
            )
            if not trusted_scalar:
                raise ValueError("responses must be a numeric array")

            logical_cells += 1
            if logical_cells > _MAX_FACETS_RESPONSE_CELLS:
                raise ValueError(resource_error)

        if (
            expected_items is None
            or expected_items == 0
            or expected_raters is None
            or expected_raters == 0
        ):
            raise ValueError(nonempty_error)

        try:
            response_array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("responses must be a numeric array") from exc
    else:
        raise ValueError("responses must be a numeric array")

    if np.iscomplexobj(response_array):
        raise ValueError("responses must be real-valued")
    if response_array.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    return response_array


def _native_result_error(detail: str) -> ValueError:
    """Build one stable package-owned error for an invalid Rust result envelope."""

    return ValueError(f"native fit_facets result {detail}")


def _native_float_vector(
    result: dict[str, object],
    key: str,
    *,
    exact_length: int | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
) -> np.ndarray:
    """Validate one inert native real vector and return an owned float64 copy."""

    value = result[key]
    value_type = type(value)
    if value_type is np.ndarray:
        array = value
    elif value_type is list or value_type is tuple:
        admitted_length = len(value)
        if exact_length is not None and admitted_length != exact_length:
            raise _native_result_error(f"{key} must have length {exact_length}")
        if min_length is not None and admitted_length < min_length:
            raise _native_result_error(f"{key} must contain at least {min_length} value")
        if max_length is not None and admitted_length > max_length:
            raise _native_result_error(f"{key} exceeds length limit {max_length}")

        sequence = value
        if value_type is list:
            snapshot_limit = (
                exact_length
                if exact_length is not None
                else max_length
                if max_length is not None
                else admitted_length
            )
            sequence = tuple(value[: snapshot_limit + 1])
            if len(sequence) != admitted_length:
                raise _native_result_error(f"{key} changed during validation")

        for entry in sequence:
            entry_type = type(entry)
            trusted = (
                entry_type is int
                or entry_type is float
                or any(
                    entry_type is scalar_type
                    for scalar_type in (
                        *_NUMPY_INTEGER_SCALAR_TYPES,
                        *_NUMPY_FLOAT_SCALAR_TYPES,
                    )
                )
            )
            if not trusted:
                raise _native_result_error(f"{key} must be a real numeric vector")
        array = np.asarray(sequence)
    else:
        raise _native_result_error(f"{key} must be a real numeric vector")

    if array.ndim != 1 or np.iscomplexobj(array) or array.dtype.kind not in ("i", "u", "f"):
        raise _native_result_error(f"{key} must be a real numeric 1-D vector")
    length = int(array.size)
    if exact_length is not None and length != exact_length:
        raise _native_result_error(f"{key} must have length {exact_length}")
    if min_length is not None and length < min_length:
        raise _native_result_error(f"{key} must contain at least {min_length} value")
    if max_length is not None and length > max_length:
        raise _native_result_error(f"{key} exceeds length limit {max_length}")

    owned = np.array(array, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(owned)):
        raise _native_result_error(f"{key} must contain only finite values")
    return owned


def _validate_native_fit_result(
    value: object,
    *,
    n_persons: int,
    n_items: int,
    n_raters: int,
    n_cat: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool, bool, int]:
    """Validate the package-owned PyO3 result envelope without numerical work."""

    if type(value) is not dict:
        raise _native_result_error("must be a built-in dict")
    required_keys = (
        "item_difficulty",
        "rater_severity",
        "thresholds",
        "theta",
        "loglik_trace",
        "n_iter",
        "converged",
        "connected",
        "n_parameters",
    )
    if len(value) != len(required_keys):
        raise _native_result_error(f"must contain exactly {len(required_keys)} keys")
    if any(type(key) is not str for key in value):
        raise _native_result_error("keys must be exact strings")

    # Freeze the exact built-in root before any value traversal.  The PyO3/native
    # provider may still retain its returned dict; later root rebinding must not
    # mix evidence observed at different times into one public fit record.
    result = value.copy()
    if len(result) != len(required_keys):
        raise _native_result_error(f"must contain exactly {len(required_keys)} keys")
    if any(type(key) is not str for key in result):
        raise _native_result_error("keys must be exact strings")
    missing_keys = [key for key in required_keys if key not in result]
    if missing_keys:
        raise _native_result_error(f"is missing required key {missing_keys[0]!r}")

    item_difficulty = _native_float_vector(
        result, "item_difficulty", exact_length=n_items
    )
    rater_severity = _native_float_vector(
        result, "rater_severity", exact_length=n_raters
    )
    thresholds = _native_float_vector(
        result, "thresholds", exact_length=n_cat - 1
    )
    theta = _native_float_vector(result, "theta", exact_length=n_persons)
    loglik_trace = _native_float_vector(
        result,
        "loglik_trace",
        min_length=1,
        max_length=max_iter + 1,
    )

    n_iter = result["n_iter"]
    if type(n_iter) is not int or not (1 <= n_iter <= max_iter):
        raise _native_result_error(f"n_iter must be an integer in 1..{max_iter}")
    converged = result["converged"]
    if type(converged) is not bool:
        raise _native_result_error("converged must be a boolean")
    valid_trace_lengths = {n_iter}
    if not converged:
        valid_trace_lengths.add(n_iter + 1)
    if int(loglik_trace.size) not in valid_trace_lengths:
        raise _native_result_error(
            "loglik_trace length must equal n_iter, or n_iter + 1 for a "
            "nonconverged terminal evaluation"
        )
    connected = result["connected"]
    if type(connected) is not bool:
        raise _native_result_error("connected must be a boolean")
    expected_parameters = n_items + (n_raters - 1) + (n_cat - 2)
    n_parameters = result["n_parameters"]
    if type(n_parameters) is not int or n_parameters != expected_parameters:
        raise _native_result_error(f"n_parameters must equal {expected_parameters}")

    return (
        item_difficulty,
        rater_severity,
        thresholds,
        theta,
        loglik_trace,
        n_iter,
        converged,
        connected,
        n_parameters,
    )


@dataclass
class FacetsFit:
    """Fitted many-facet Rasch model (Linacre, 1989).

    ``item_difficulty`` is the per-item ``d_i``; ``rater_severity`` the per-rater
    ``c_j`` (centered to sum 0; higher = harsher); ``thresholds`` the ``n_cat-1``
    common category thresholds (centered to sum 0); ``theta`` the per-person EAP
    trait. The adjacent-category log-odds are
    ``ln[P(k)/P(k-1)] = theta - d_i - c_j - f_k``. ``connected`` is False when the
    item-rater co-observation design splits into disconnected components — then
    severity/difficulty comparisons across components rest solely on the shared
    ``theta ~ N(0,1)`` assumption rather than on the rating design (Linacre's
    connectedness requirement)."""

    item_difficulty: np.ndarray
    rater_severity: np.ndarray
    thresholds: np.ndarray
    theta: np.ndarray
    loglik_trace: np.ndarray
    n_iter: int
    converged: bool
    connected: bool
    n_parameters: int


def fit_facets(
    responses: np.ndarray,
    n_cat: int | None = None,
    q_theta: int = 41,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> FacetsFit:
    """Fit the many-facet Rasch model (compute in Rust; Linacre, 1989).

    The MFRM extends the rating scale model (Andrich, 1978) with a rater facet:
    the rating of person ``p`` on item ``i`` by rater ``j`` follows the
    adjacent-category log-odds
    ``ln[P(Y=k)/P(Y=k-1)] = theta_p - d_i - c_j - f_k``, where ``d_i`` is item
    difficulty, ``c_j`` rater severity, and ``f_k`` the category thresholds
    shared across items and raters. ``theta ~ N(0,1)`` fixes the scale;
    severities and thresholds are centered to sum zero. Estimation is
    marginal-ML EM (Bock & Aitkin, 1981) on a Gauss-Hermite trait grid — not
    Linacre's JMLE, so estimates match Facets output only up to the JMLE-vs-MMLE
    difference.

    In LLM-as-a-Judge calibration, raters are judges: ``rater_severity``
    estimates each judge's harshness on a common logit scale, adjusted for item
    difficulty and respondent ability.

    ``responses`` is a ``persons x items x raters`` array of integer category
    indices ``0..n_cat-1``; ``NaN`` marks unscored cells (sparse judging plans),
    dropped under a missing-at-random assumption. ``n_cat`` defaults to
    ``max(responses) + 1``. Every item and every rater needs at least one
    observed rating.

    References (APA 7th ed.):
        Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
        Eckes, T. (2015). *Introduction to many-facet Rasch measurement*
            (2nd ed.). Peter Lang. https://doi.org/10.3726/978-3-653-04844-5
        Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation
            of item parameters: Application of an EM algorithm. *Psychometrika,
            46*(4), 443-459. https://doi.org/10.1007/BF02293801
        Andrich, D. (1978). A rating formulation for ordered response
            categories. *Psychometrika, 43*(4), 561-573.
            https://doi.org/10.1007/BF02293814
    """
    n_cat = _integer_control(n_cat, "n_cat", allow_none=True)
    if n_cat is not None and not (2 <= n_cat <= MAX_POLYTOMOUS_CATEGORIES):
        raise ValueError(f"n_cat must be an integer in 2..{MAX_POLYTOMOUS_CATEGORIES}")
    q_theta = _integer_control(q_theta, "q_theta")
    if q_theta not in (7, 11, 15, 21, 31, 41):
        raise ValueError("q_theta must be one of 7, 11, 15, 21, 31, 41")
    max_iter = _integer_control(max_iter, "max_iter")
    if not (1 <= max_iter <= MAX_MAX_ITER):
        raise ValueError(f"max_iter must be an integer in 1..{MAX_MAX_ITER}")
    tol = _real_control(tol, "tol")
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and > 0")

    response_array = _response_array(responses)
    y = response_array.astype(np.float64, copy=False)
    if y.ndim != 3:
        raise ValueError("responses must be a 3-D persons x items x raters array")
    n_persons, n_items, n_raters = y.shape
    if n_persons < 1 or n_items < 1 or n_raters < 1:
        raise ValueError(
            "responses must contain at least one person, one item and one rater"
        )
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
    missing_items = np.flatnonzero(~observed.any(axis=(0, 2)))
    if missing_items.size:
        raise ValueError(f"item {int(missing_items[0])} has no observed responses")
    missing_raters = np.flatnonzero(~observed.any(axis=(0, 1)))
    if missing_raters.size:
        raise ValueError(f"rater {int(missing_raters[0])} has no observed responses")
    yy = np.where(observed, y, 0.0).astype(np.int64).reshape(-1)

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_facets"):
        raise RuntimeError("fit_facets requires the compiled Rust core")
    raw_result = core.fit_facets(
        yy,
        observed.reshape(-1),
        int(n_persons),
        int(n_items),
        int(n_raters),
        int(n_cat),
        int(q_theta),
        int(max_iter),
        float(tol),
    )
    (
        item_difficulty,
        rater_severity,
        thresholds,
        theta,
        loglik_trace,
        n_iter,
        converged,
        connected,
        n_parameters,
    ) = _validate_native_fit_result(
        raw_result,
        n_persons=n_persons,
        n_items=n_items,
        n_raters=n_raters,
        n_cat=n_cat,
        max_iter=max_iter,
    )
    return FacetsFit(
        item_difficulty=item_difficulty,
        rater_severity=rater_severity,
        thresholds=thresholds,
        theta=theta,
        loglik_trace=loglik_trace,
        n_iter=n_iter,
        converged=converged,
        connected=connected,
        n_parameters=n_parameters,
    )