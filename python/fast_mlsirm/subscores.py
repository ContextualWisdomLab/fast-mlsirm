"""Haberman subscore added-value analysis via proportional reduction in mean
squared error (PRMSE; Haberman, 2008, as cited in Sinharay, 2010). All numeric
work happens in the Rust core; this module only validates and marshals
arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NoReturn

import numpy as np


_MAX_SUBSCORE_EVIDENCE_CELLS = 20_000_000
_MAX_SUBSCORE_STRUCTURE_NODES = 3 * _MAX_SUBSCORE_EVIDENCE_CELLS
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
_TRUSTED_NUMERIC_SEQUENCE_SCALAR_TYPES = frozenset(
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
_SUBSCORE_RESULT_KEYS = frozenset(
    {
        "alpha",
        "alpha_total",
        "corr",
        "disattenuated_corr",
        "prmse_s",
        "prmse_x",
        "prmse_sx",
        "tau",
        "beta",
        "gamma",
        "added_value_s",
        "added_value_sx",
        "observed",
        "total",
        "subscore_s",
        "subscore_x",
        "subscore_sx",
    }
)
_SUBSCORE_RESULT_ERROR = "invalid subscore Rust result payload"


@dataclass
class SubscoreResult:
    """Haberman subscore added-value analysis for ``K`` subscales.

    ``alpha`` are the per-subscale Cronbach alphas ( = ``prmse_s``);
    ``alpha_total`` the total-test alpha. ``corr`` is the ``(K+1) x (K+1)``
    correlation matrix of the observed subscores with the total score last;
    ``disattenuated_corr`` the ``K x K`` disattenuated subscore correlations
    (NaN diagonal). ``prmse_s``/``prmse_x``/``prmse_sx`` are the PRMSEs of
    predicting the true subscore from the observed subscore, the observed
    total, and both; ``tau``/``beta``/``gamma`` the augmented-regression
    weights. ``added_value_s`` is Haberman's rule ``PRMSE_s > PRMSE_x``;
    ``added_value_sx`` uses Sinharay's (2010) ``+ 0.01`` margin.
    ``observed`` (``n x K``), ``total`` (``n``), and the three estimator
    matrices ``subscore_s``/``subscore_x``/``subscore_sx`` (each ``n x K``)
    give per-person scores."""

    alpha: np.ndarray
    alpha_total: float
    corr: np.ndarray
    disattenuated_corr: np.ndarray
    prmse_s: np.ndarray
    prmse_x: np.ndarray
    prmse_sx: np.ndarray
    tau: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    added_value_s: np.ndarray
    added_value_sx: np.ndarray
    observed: np.ndarray
    total: np.ndarray
    subscore_s: np.ndarray
    subscore_x: np.ndarray
    subscore_sx: np.ndarray


def _trusted_numeric_array(value: object, name: str) -> np.ndarray:
    """Materialize inert numeric evidence after bounded callback-free preflight."""

    resource_error = (
        f"{name} must contain at most {_MAX_SUBSCORE_EVIDENCE_CELLS:,} logical cells"
    )
    value_type = type(value)
    if value_type is np.ndarray:
        if value.size > _MAX_SUBSCORE_EVIDENCE_CELLS:
            raise ValueError(resource_error)
        raw = value
    elif value_type in (list, tuple):
        # Use indexed frames so traversal memory is proportional to nesting
        # depth rather than sibling width. Active-path identities reject true
        # cycles while permitting the same acyclic subtree to appear in more
        # than one sibling position.
        frames: list[list[object]] = [[value, 0]]
        active_container_ids: set[int] = {id(value)}
        logical_cells = 0
        structural_nodes = 0

        while frames:
            frame = frames[-1]
            current = frame[0]
            child_index = int(frame[1])

            if child_index >= len(current):
                active_container_ids.remove(id(current))
                frames.pop()
                continue

            frame[1] = child_index + 1
            structural_nodes += 1
            if structural_nodes > _MAX_SUBSCORE_STRUCTURE_NODES:
                raise ValueError(
                    f"{name} exceeded structural traversal budget of "
                    f"{_MAX_SUBSCORE_STRUCTURE_NODES:,} nodes"
                )

            child = current[child_index]
            child_type = type(child)
            if child_type in (list, tuple):
                child_id = id(child)
                if child_id in active_container_ids:
                    raise ValueError(f"{name} must be acyclic numeric evidence")
                active_container_ids.add(child_id)
                frames.append([child, 0])
                continue

            if child_type is np.ndarray:
                if child.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(f"{name} must be real-numeric evidence")
                logical_cells += int(child.size)
            elif child_type in _TRUSTED_NUMERIC_SEQUENCE_SCALAR_TYPES:
                logical_cells += 1
            else:
                raise ValueError(f"{name} must be real-numeric evidence")

            if logical_cells > _MAX_SUBSCORE_EVIDENCE_CELLS:
                raise ValueError(resource_error)

        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be real-numeric evidence") from None
    else:
        raise ValueError(f"{name} must be real-numeric evidence")

    if raw.dtype.kind not in ("b", "i", "u", "f", "c"):
        raise ValueError(f"{name} must be real-numeric evidence")
    return raw


def _invalid_subscore_result() -> NoReturn:
    """Raise the stable fail-closed error for a stale/foreign native result."""
    raise RuntimeError(_SUBSCORE_RESULT_ERROR)


def _native_float_vector(
    value: object,
    *,
    expected_length: int | None = None,
    allowed_nan_indices: frozenset[int] = frozenset(),
) -> list[float]:
    """Validate one flattened Rust f64 vector before NumPy materialization."""
    if type(value) is np.ndarray:
        if value.ndim != 1 or value.dtype != np.dtype(np.float64):
            _invalid_subscore_result()
        if expected_length is not None and value.shape[0] != expected_length:
            _invalid_subscore_result()
        scalars: list[object] = list(value)
    elif type(value) in (list, tuple):
        if expected_length is not None and len(value) != expected_length:
            _invalid_subscore_result()
        scalars = list(value)
    else:
        _invalid_subscore_result()

    validated: list[float] = []
    for index, scalar in enumerate(scalars):
        scalar_type = type(scalar)
        if scalar_type is float:
            normalized = scalar
        elif scalar_type is np.float64:
            normalized = float(scalar)
        else:
            _invalid_subscore_result()
        if np.isnan(normalized):
            if index not in allowed_nan_indices:
                _invalid_subscore_result()
        elif not np.isfinite(normalized):
            _invalid_subscore_result()
        validated.append(normalized)
    return validated


def _native_bool_vector(value: object, *, expected_length: int) -> list[bool]:
    """Validate one Rust Boolean vector without conversion callbacks."""
    if type(value) is np.ndarray:
        if (
            value.ndim != 1
            or value.dtype != np.dtype(np.bool_)
            or value.shape[0] != expected_length
        ):
            _invalid_subscore_result()
        scalars: list[object] = list(value)
    elif type(value) in (list, tuple):
        if len(value) != expected_length:
            _invalid_subscore_result()
        scalars = list(value)
    else:
        _invalid_subscore_result()

    validated: list[bool] = []
    for scalar in scalars:
        if type(scalar) is bool:
            validated.append(scalar)
        elif type(scalar) is np.bool_:
            validated.append(bool(scalar))
        else:
            _invalid_subscore_result()
    return validated


def _native_float_scalar(value: object) -> float:
    """Validate one Rust f64 scalar before public result construction."""
    if type(value) is float:
        normalized = value
    elif type(value) is np.float64:
        normalized = float(value)
    else:
        _invalid_subscore_result()
    if not np.isfinite(normalized):
        _invalid_subscore_result()
    return normalized


def _validate_reliability_prmse_domains(
    alpha: list[float],
    alpha_total: float,
    prmse_s: list[float],
    prmse_x: list[float],
    prmse_sx: list[float],
) -> None:
    """Replay the explicit Rust guards without recomputing any estimator."""
    if any(value <= 0.0 or value > 1.0 for value in alpha):
        _invalid_subscore_result()
    if alpha_total <= 0.0 or alpha_total > 1.0:
        _invalid_subscore_result()
    for values in (prmse_s, prmse_x, prmse_sx):
        if any(value < 0.0 or value > 1.0 + 1e-9 for value in values):
            _invalid_subscore_result()


def _validated_subscore_result(
    value: object,
    *,
    n_persons: int,
    expected_k: int,
) -> tuple[
    list[float],
    float,
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[bool],
    list[bool],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
]:
    """Replay the exact subscore result schema before any NumPy coercion."""
    if type(value) is not dict:
        _invalid_subscore_result()
    keys = list(value.keys())
    if any(type(key) is not str for key in keys) or set(keys) != _SUBSCORE_RESULT_KEYS:
        _invalid_subscore_result()
    if expected_k < 2:
        _invalid_subscore_result()

    alpha = _native_float_vector(value["alpha"], expected_length=expected_k)
    k = expected_k
    alpha_total = _native_float_scalar(value["alpha_total"])
    corr = _native_float_vector(value["corr"], expected_length=(k + 1) * (k + 1))

    diagonal = frozenset(index * k + index for index in range(k))
    disattenuated_corr = _native_float_vector(
        value["disattenuated_corr"],
        expected_length=k * k,
        allowed_nan_indices=diagonal,
    )
    if any(not np.isnan(disattenuated_corr[index]) for index in diagonal):
        _invalid_subscore_result()

    prmse_s = _native_float_vector(value["prmse_s"], expected_length=k)
    prmse_x = _native_float_vector(value["prmse_x"], expected_length=k)
    prmse_sx = _native_float_vector(value["prmse_sx"], expected_length=k)
    _validate_reliability_prmse_domains(
        alpha,
        alpha_total,
        prmse_s,
        prmse_x,
        prmse_sx,
    )
    tau = _native_float_vector(value["tau"], expected_length=k)
    beta = _native_float_vector(value["beta"], expected_length=k)
    gamma = _native_float_vector(value["gamma"], expected_length=k)
    added_value_s = _native_bool_vector(value["added_value_s"], expected_length=k)
    added_value_sx = _native_bool_vector(value["added_value_sx"], expected_length=k)

    person_subscore_length = n_persons * k
    observed = _native_float_vector(
        value["observed"], expected_length=person_subscore_length
    )
    total = _native_float_vector(value["total"], expected_length=n_persons)
    subscore_s = _native_float_vector(
        value["subscore_s"], expected_length=person_subscore_length
    )
    subscore_x = _native_float_vector(
        value["subscore_x"], expected_length=person_subscore_length
    )
    subscore_sx = _native_float_vector(
        value["subscore_sx"], expected_length=person_subscore_length
    )
    return (
        alpha,
        alpha_total,
        corr,
        disattenuated_corr,
        prmse_s,
        prmse_x,
        prmse_sx,
        tau,
        beta,
        gamma,
        added_value_s,
        added_value_sx,
        observed,
        total,
        subscore_s,
        subscore_x,
        subscore_sx,
    )


def subscore_analysis(
    responses: np.ndarray,
    groups: np.ndarray,
) -> SubscoreResult:
    """Haberman subscore added-value analysis (compute in Rust; Haberman,
    2008, as cited in Sinharay, 2010).

    Decides, for each subscale of a test, whether reporting its subscore adds
    value over reporting the total score alone, by comparing the PRMSEs of
    three classical-test-theory estimators of the true subscore (from the
    observed subscore, from the observed total, and from both jointly).
    Formulas follow the Appendix of Sinharay (2010) and the CRAN ``subscore``
    package R source (both read); Haberman (2008) and Wainer et al. (2001)
    are cited only through Sinharay (2010). Degenerate samples (any Cronbach
    alpha outside ``(0, 1]``, zero-variance scores, a subscore collinear with
    the total) are rejected with ``ValueError`` rather than propagating NaN.

    In LLM-as-a-Judge item-quality management this decides whether
    per-domain judge subscores carry diagnostic information beyond the
    overall score, or whether reporting them would be statistically
    misleading.

    ``responses`` is a complete ``persons x items`` array of scored
    responses (``n >= 3``). ``groups`` assigns each item an integer subscale
    index in ``0..K`` (``K >= 2``, every subscale with at least 2 items,
    partition exhaustive by construction). Evidence admission accepts exact
    NumPy numeric arrays or exact built-in list/tuple trees containing only
    package-trusted concrete Python/NumPy numeric scalars or exact NumPy
    numeric array leaves. Logical-cell and structural traversal budgets are
    enforced before NumPy materialization, and callback-bearing or cyclic
    providers fail closed. The Rust result is replayed against the exact
    package-owned field, scalar, finiteness, reliability/PRMSE-domain, and
    cardinality contract before NumPy result marshalling, including the
    subscale count implied by the admitted group evidence.

    References (APA 7th ed.):
        Haberman, S. J. (2008). When can subscores have value? *Journal of
            Educational and Behavioral Statistics, 33*(2), 204-229.
            https://doi.org/10.3102/1076998607302636 (as cited in Sinharay,
            2010)
        Sinharay, S. (2010). *When can subscores be expected to have added
            value? Results from operational and simulated data* (ETS
            Research Rep. No. RR-10-16). Educational Testing Service.
        Wainer, H., Vevea, J. L., Camacho, F., Reeve, B. B., Rosa, K., &
            Nelson, L. (2001). Augmented scores — "borrowing strength" to
            compute scores based on small numbers of items. In D. Thissen &
            H. Wainer (Eds.), *Test scoring* (pp. 343-387). Lawrence
            Erlbaum. (as cited in Sinharay, 2010)
    """
    y_raw = _trusted_numeric_array(responses, "responses")
    if np.iscomplexobj(y_raw):
        raise ValueError("responses must be real-valued")
    y = np.asarray(y_raw, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = y.shape
    if n_persons < 3 or n_items < 4:
        # K >= 2 subscales with >= 2 items each needs at least 4 items;
        # rejecting here keeps degenerate shapes (e.g. huge zero-column
        # arrays) from crossing the Rust boundary at all.
        raise ValueError("responses needs at least 3 persons and 4 items")
    if not np.all(np.isfinite(y)):
        raise ValueError("responses must be complete (no missing values)")

    g = _trusted_numeric_array(groups, "groups").reshape(-1)
    if g.shape[0] != n_items:
        raise ValueError("groups must assign one subscale index per item")
    if np.iscomplexobj(g):
        raise ValueError("groups must be real-valued")
    if not np.issubdtype(g.dtype, np.integer):
        gf = np.asarray(g, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(gf)) or np.any(gf != np.round(gf)):
            raise ValueError("groups must be integer subscale indices")
        g = gf.astype(np.int64)
    if np.any(g < 0):
        raise ValueError("groups must be nonnegative subscale indices")
    if np.any(g >= n_items):
        # trust boundary: the subscale count drives Rust-side allocations
        raise ValueError("groups indices must be < n_items")
    expected_k = int(np.max(g)) + 1

    from .fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "subscore_analysis"):
        raise RuntimeError("subscore_analysis requires the compiled Rust core")

    raw_result = core.subscore_analysis(
        y.reshape(-1), int(n_persons), int(n_items), [int(v) for v in g]
    )
    (
        alpha,
        alpha_total,
        corr,
        disattenuated_corr,
        prmse_s,
        prmse_x,
        prmse_sx,
        tau,
        beta,
        gamma,
        added_value_s,
        added_value_sx,
        observed,
        total,
        subscore_s,
        subscore_x,
        subscore_sx,
    ) = _validated_subscore_result(
        raw_result,
        n_persons=int(n_persons),
        expected_k=expected_k,
    )
    k = expected_k
    return SubscoreResult(
        alpha=np.asarray(alpha, dtype=np.float64),
        alpha_total=alpha_total,
        corr=np.asarray(corr, dtype=np.float64).reshape(k + 1, k + 1),
        disattenuated_corr=np.asarray(disattenuated_corr, dtype=np.float64).reshape(k, k),
        prmse_s=np.asarray(prmse_s, dtype=np.float64),
        prmse_x=np.asarray(prmse_x, dtype=np.float64),
        prmse_sx=np.asarray(prmse_sx, dtype=np.float64),
        tau=np.asarray(tau, dtype=np.float64),
        beta=np.asarray(beta, dtype=np.float64),
        gamma=np.asarray(gamma, dtype=np.float64),
        added_value_s=np.asarray(added_value_s, dtype=bool),
        added_value_sx=np.asarray(added_value_sx, dtype=bool),
        observed=np.asarray(observed, dtype=np.float64).reshape(n_persons, k),
        total=np.asarray(total, dtype=np.float64),
        subscore_s=np.asarray(subscore_s, dtype=np.float64).reshape(n_persons, k),
        subscore_x=np.asarray(subscore_x, dtype=np.float64).reshape(n_persons, k),
        subscore_sx=np.asarray(subscore_sx, dtype=np.float64).reshape(n_persons, k),
    )