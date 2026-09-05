"""Mokken scale analysis: Loevinger scalability coefficients and the automated
item selection procedure (AISP). All numeric work happens in the Rust core;
this module only validates and marshals arrays."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice

import numpy as np

from .config import MAX_POLYTOMOUS_CATEGORIES


_TRUSTED_NUMPY_REAL_TYPES = (
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
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_TRUSTED_RESPONSE_SCALAR_TYPES = (bool, int, float) + _TRUSTED_NUMPY_REAL_TYPES
_INT64_MAX = (1 << 63) - 1
_INT64_EXCLUSIVE_UPPER_FLOAT = float(1 << 63)
_UINT32_MAX = (1 << 32) - 1
_MAX_MOKKEN_RESPONSE_CELLS = 20_000_000
_MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES = 2 * _MAX_MOKKEN_RESPONSE_CELLS
_MAX_MOKKEN_MATRIX_CELLS = 4_000_000
_MAX_MOKKEN_NUMERIC_ITEM_BYTES = max(
    np.dtype(dtype).itemsize for dtype in _TRUSTED_NUMPY_REAL_TYPES
)
_MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES = (
    _MAX_MOKKEN_RESPONSE_CELLS * _MAX_MOKKEN_NUMERIC_ITEM_BYTES
)


@dataclass
class MokkenResult:
    """Mokken scalability coefficients and (optionally) AISP scale labels.

    ``hij`` is the ``items x items`` matrix of pairwise scalability
    coefficients (NaN diagonal), ``hi`` the per-item coefficients, ``h`` the
    total scale coefficient; ``zij``/``zi``/``z`` are the matching Mokken Z
    statistics for the null hypothesis of inter-item independence.
    ``scale`` holds per-item AISP labels: 0 = unscalable, 1, 2, ... in
    formation order. Sample statistics follow the mokken R package
    (van der Ark, 2007)."""

    hij: np.ndarray
    hi: np.ndarray
    h: float
    zij: np.ndarray
    zi: np.ndarray
    z: float
    scale: np.ndarray


@dataclass(frozen=True)
class _ValidatedMokkenCoefficients:
    """Package-owned validated projection of the Rust coefficient payload."""

    hij: np.ndarray
    hi: np.ndarray
    h: float
    zij: np.ndarray
    zi: np.ndarray
    z: float


def _real_control(name: str, value: object) -> float:
    """Normalize one trusted finite real scalar without caller callbacks."""
    value_type = type(value)
    extended_source: np.longdouble | None = None
    try:
        if value_type is int or value_type is float:
            parsed = float(value)
        elif any(value_type is trusted for trusted in _TRUSTED_NUMPY_REAL_TYPES):
            if value_type is np.longdouble:
                extended_source = value
            parsed = float(value)
        elif (
            value_type is np.ndarray
            and value.ndim == 0
            and value.dtype.kind in ("i", "u", "f")
        ):
            scalar = value.item()
            if (
                value.dtype.kind == "f"
                and value.dtype.itemsize > np.dtype(np.float64).itemsize
            ):
                extended_source = np.longdouble(scalar)
            parsed = float(scalar)
        else:
            raise ValueError(f"{name} must be a real number")
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not np.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    if extended_source is not None and np.longdouble(parsed) != extended_source:
        raise ValueError(f"{name} must be exactly representable as float64")
    return parsed


def _raise_response_resource_error() -> None:
    """Reject response evidence outside the package materialization envelope."""
    raise ValueError(
        f"responses exceed {_MAX_MOKKEN_RESPONSE_CELLS:,} logical cells"
    )


def _raise_response_structural_resource_error() -> None:
    """Reject built-in response traversal outside the package work envelope."""
    raise ValueError(
        "responses exceed "
        f"{_MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:,} structural nodes"
    )


def _raise_item_matrix_resource_error(n_items: int) -> None:
    """Reject quadratic Mokken item work outside the Rust matrix budget."""
    raise ValueError(
        f"responses imply {n_items * n_items:,} item-matrix cells; "
        f"the limit is {_MAX_MOKKEN_MATRIX_CELLS:,}"
    )


def _raise_native_result_error() -> None:
    """Reject a compiled-core result outside the current binding contract."""
    raise ValueError("invalid Mokken Rust result payload")


def _native_float_vector(value: object, expected_size: int) -> np.ndarray:
    """Marshal one exact Rust ``Vec<f64>`` carrier after identity replay."""
    if type(value) is not list or len(value) != expected_size:
        _raise_native_result_error()
    snapshot = value.copy()
    if len(snapshot) != expected_size:
        _raise_native_result_error()
    if any(type(element) is not float for element in snapshot):
        _raise_native_result_error()
    return np.asarray(snapshot, dtype=np.float64)


def _native_finite_float_vector(value: object, expected_size: int) -> np.ndarray:
    """Marshal a Rust ``Vec<f64>`` whose live result domain is fully finite."""
    vector = _native_float_vector(value, expected_size)
    if not np.all(np.isfinite(vector)):
        _raise_native_result_error()
    return vector


def _native_pairwise_matrix(value: object, n_items: int) -> np.ndarray:
    """Replay deterministic structure of one symmetric Rust pairwise matrix."""
    matrix_cells = n_items * n_items
    vector = _native_float_vector(value, matrix_cells)
    matrix = vector.reshape(n_items, n_items)
    diagonal = np.diag(matrix)
    if not np.all(np.isnan(diagonal)):
        _raise_native_result_error()
    if np.count_nonzero(np.isfinite(vector)) != matrix_cells - n_items:
        _raise_native_result_error()
    if not np.array_equal(matrix, matrix.T, equal_nan=True):
        _raise_native_result_error()
    return vector


def _native_scale_vector(value: object, expected_size: int) -> np.ndarray:
    """Marshal exact Rust AISP labels after identity/domain replay."""
    if type(value) is not list or len(value) != expected_size:
        _raise_native_result_error()
    snapshot = value.copy()
    if len(snapshot) != expected_size:
        _raise_native_result_error()
    if any(type(element) is not int for element in snapshot):
        _raise_native_result_error()
    if any(element < 0 or element > _UINT32_MAX for element in snapshot):
        _raise_native_result_error()
    max_formation_label = expected_size // 2
    if any(element > max_formation_label for element in snapshot):
        _raise_native_result_error()

    positive_label_counts: dict[int, int] = {}
    for element in snapshot:
        if element > 0:
            positive_label_counts[element] = positive_label_counts.get(element, 0) + 1
    if positive_label_counts:
        highest_label = max(positive_label_counts)
        if len(positive_label_counts) != highest_label:
            _raise_native_result_error()
        if any(count < 2 for count in positive_label_counts.values()):
            _raise_native_result_error()
    return np.asarray(snapshot, dtype=np.int64)


def _validated_native_coefficients(
    result: object,
    n_items: int,
) -> _ValidatedMokkenCoefficients:
    """Validate the coefficient/statistic envelope before further Rust work."""
    if type(result) is not dict:
        _raise_native_result_error()
    required_fields = ("hij", "hi", "h", "zij", "zi", "z")
    if len(result) != len(required_fields):
        _raise_native_result_error()
    if any(type(key) is not str for key in result):
        _raise_native_result_error()

    snapshot = result.copy()
    if len(snapshot) != len(required_fields):
        _raise_native_result_error()
    if any(type(key) is not str for key in snapshot):
        _raise_native_result_error()
    if any(field not in snapshot for field in required_fields):
        _raise_native_result_error()

    h = snapshot["h"]
    z = snapshot["z"]
    if type(h) is not float or type(z) is not float:
        _raise_native_result_error()
    if not np.isfinite(h) or not np.isfinite(z):
        _raise_native_result_error()

    return _ValidatedMokkenCoefficients(
        hij=_native_pairwise_matrix(snapshot["hij"], n_items),
        hi=_native_finite_float_vector(snapshot["hi"], n_items),
        h=h,
        zij=_native_pairwise_matrix(snapshot["zij"], n_items),
        zi=_native_finite_float_vector(snapshot["zi"], n_items),
        z=z,
    )


def _validated_native_result(
    coefficients: _ValidatedMokkenCoefficients,
    scale: object,
    n_items: int,
) -> MokkenResult:
    """Validate the AISP carrier and assemble the public Mokken result."""
    scale_array = _native_scale_vector(scale, n_items)
    return MokkenResult(
        hij=coefficients.hij.reshape(n_items, n_items),
        hi=coefficients.hi,
        h=coefficients.h,
        zij=coefficients.zij.reshape(n_items, n_items),
        zi=coefficients.zi,
        z=coefficients.z,
        scale=scale_array,
    )


def _trusted_score_source(responses: object) -> object:
    """Admit and bound inert response containers before NumPy protocols run."""
    if type(responses) is np.ndarray:
        if responses.size > _MAX_MOKKEN_RESPONSE_CELLS:
            _raise_response_resource_error()
        return responses
    if type(responses) is not list and type(responses) is not tuple:
        raise ValueError("responses must be a numeric array")

    logical_cells = 0
    structural_nodes = 0
    rectangular_width: int | None = None
    rectangular_rows = True
    nested_ndarray_storage_issue: str | None = None
    for row in responses:
        row_type = type(row)
        if row_type is np.ndarray:
            row_cells = int(row.size)
            logical_cells += row_cells
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            structural_nodes += 1 + row_cells
            if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
                _raise_response_structural_resource_error()
            if nested_ndarray_storage_issue is None:
                if row.dtype.kind == "c":
                    nested_ndarray_storage_issue = "complex"
                elif row.dtype.kind not in ("b", "i", "u", "f"):
                    nested_ndarray_storage_issue = "numeric"
                elif int(row.nbytes) > _MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES:
                    nested_ndarray_storage_issue = "resource"
            if row.ndim != 1:
                rectangular_rows = False
            else:
                row_width = int(row.shape[0])
                if rectangular_width is None:
                    rectangular_width = row_width
                elif row_width != rectangular_width:
                    rectangular_rows = False
            continue
        if row_type is list or row_type is tuple:
            row_cells = len(row)
            logical_cells += row_cells
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            structural_nodes += 1 + row_cells
            if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
                _raise_response_structural_resource_error()
            if rectangular_width is None:
                rectangular_width = row_cells
            elif row_cells != rectangular_width:
                rectangular_rows = False
            continue
        # Preserve the historical flat built-in-sequence path long enough for
        # the established 2-D dimensionality diagnostic, without accepting
        # caller-defined numeric/container subclasses.
        rectangular_rows = False
        logical_cells += 1
        if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
            _raise_response_resource_error()
        structural_nodes += 1
        if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
            _raise_response_structural_resource_error()
        if row_type not in _TRUSTED_RESPONSE_SCALAR_TYPES:
            raise ValueError("responses must be a numeric array")

    if not rectangular_rows:
        raise ValueError("responses must be a 2-D persons x items array")
    if rectangular_width is not None and rectangular_width < 2:
        raise ValueError("mokken requires at least 2 items")
    if (
        rectangular_width is not None
        and rectangular_width * rectangular_width > _MAX_MOKKEN_MATRIX_CELLS
    ):
        _raise_item_matrix_resource_error(rectangular_width)
    if nested_ndarray_storage_issue == "complex":
        raise ValueError("responses must be real-valued")
    if nested_ndarray_storage_issue == "numeric":
        raise ValueError("responses must be a numeric array")
    if nested_ndarray_storage_issue == "resource":
        _raise_response_resource_error()
    return responses


def _snapshot_builtin_score_source(
    source: list[object] | tuple[object, ...],
) -> tuple[object, ...]:
    """Seal exact built-in response evidence before dense NumPy materialization."""
    row_count = len(source)
    if row_count > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
        _raise_response_structural_resource_error()

    if type(source) is list:
        rows = tuple(islice(source, row_count + 1))
        if len(rows) != row_count or len(source) != row_count:
            raise ValueError("responses must be a 2-D persons x items array")
    else:
        rows = source

    logical_cells = 0
    structural_nodes = 0
    snapshots: list[object] = []
    for row in rows:
        row_type = type(row)
        if row_type is np.ndarray:
            row_cells = int(row.size)
            logical_cells += row_cells
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            structural_nodes += 1 + row_cells
            if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
                _raise_response_structural_resource_error()
            if row.dtype.kind == "c":
                raise ValueError("responses must be real-valued")
            if row.dtype.kind not in ("b", "i", "u", "f"):
                raise ValueError("responses must be a numeric array")
            if int(row.nbytes) > _MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES:
                _raise_response_resource_error()
            admitted_shape = row.shape
            admitted_size = row_cells
            row_snapshot = np.array(row, copy=True)
            if row_snapshot.shape != admitted_shape or int(row_snapshot.size) != admitted_size:
                raise ValueError("responses must be a 2-D persons x items array")
            if row_snapshot.dtype.kind == "c":
                raise ValueError("responses must be real-valued")
            if row_snapshot.dtype.kind not in ("b", "i", "u", "f"):
                raise ValueError("responses must be a numeric array")
            if int(row_snapshot.nbytes) > _MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES:
                _raise_response_resource_error()
            snapshots.append(row_snapshot)
            continue

        if row_type is list:
            row_cells = len(row)
            logical_cells += row_cells
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            structural_nodes += 1 + row_cells
            if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
                _raise_response_structural_resource_error()
            row_snapshot = tuple(islice(row, row_cells + 1))
            if len(row_snapshot) != row_cells or len(row) != row_cells:
                raise ValueError("responses must be a 2-D persons x items array")
            snapshots.append(row_snapshot)
            continue

        if row_type is tuple:
            row_cells = len(row)
            logical_cells += row_cells
            if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
                _raise_response_resource_error()
            structural_nodes += 1 + row_cells
            if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
                _raise_response_structural_resource_error()
            snapshots.append(row)
            continue

        logical_cells += 1
        if logical_cells > _MAX_MOKKEN_RESPONSE_CELLS:
            _raise_response_resource_error()
        structural_nodes += 1
        if structural_nodes > _MAX_MOKKEN_RESPONSE_STRUCTURAL_NODES:
            _raise_response_structural_resource_error()
        snapshots.append(row)

    snapshot = tuple(snapshots)
    _trusted_score_source(snapshot)
    for row in snapshot:
        if type(row) is tuple and any(
            type(cell) not in _TRUSTED_RESPONSE_SCALAR_TYPES for cell in row
        ):
            raise ValueError("responses must be a numeric array")
    return snapshot


def _validated_scores(responses: object) -> tuple[np.ndarray, int, int]:
    """Validate score storage losslessly before signed-int64 marshalling."""
    source = _trusted_score_source(responses)
    admitted_ndarray_shape: tuple[int, ...] | None = None
    if type(source) is np.ndarray:
        if source.ndim != 2:
            raise ValueError("responses must be a 2-D persons x items array")
        admitted_ndarray_shape = tuple(int(axis) for axis in source.shape)
        source_n_items = int(source.shape[1])
        if source_n_items < 2:
            raise ValueError("mokken requires at least 2 items")
        if source_n_items * source_n_items > _MAX_MOKKEN_MATRIX_CELLS:
            _raise_item_matrix_resource_error(source_n_items)
        if source.dtype.kind == "c":
            raise ValueError("responses must be real-valued")
        if source.dtype.kind not in ("b", "i", "u", "f"):
            raise ValueError("responses must be a numeric array")
        if int(source.nbytes) > _MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES:
            _raise_response_resource_error()
    else:
        source = _snapshot_builtin_score_source(source)
    try:
        if type(source) is np.ndarray:
            raw = np.array(source, copy=True)
        else:
            raw = np.asarray(source)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses must be a numeric array") from None
    if raw.size > _MAX_MOKKEN_RESPONSE_CELLS:
        _raise_response_resource_error()
    if (
        admitted_ndarray_shape is not None
        and tuple(int(axis) for axis in raw.shape) != admitted_ndarray_shape
    ):
        raise ValueError("responses must be a 2-D persons x items array")
    if raw.ndim != 2:
        raise ValueError("responses must be a 2-D persons x items array")
    n_persons, n_items = raw.shape
    if n_items * n_items > _MAX_MOKKEN_MATRIX_CELLS:
        _raise_item_matrix_resource_error(n_items)
    if np.iscomplexobj(raw):
        raise ValueError("responses must be real-valued")
    if raw.dtype.kind not in ("b", "i", "u", "f"):
        raise ValueError("responses must be a numeric array")
    if int(raw.nbytes) > _MAX_MOKKEN_RESPONSE_SNAPSHOT_BYTES:
        _raise_response_resource_error()
    if not np.all(np.isfinite(raw)):
        raise ValueError("responses must be complete (no missing values)")

    kind = raw.dtype.kind
    if kind == "f":
        if np.any(raw != np.floor(raw)) or np.any(raw < 0):
            raise ValueError("responses must be non-negative integer scores")
        # 2**63 is exactly representable in every floating format that can
        # reach this boundary. INT64_MAX is not exactly representable in
        # float64, so use the exclusive upper bound instead of a lossy
        # comparison against INT64_MAX.
        if np.any(raw >= _INT64_EXCLUSIVE_UPPER_FLOAT):
            raise ValueError("responses exceed signed int64 range")
    elif kind == "i":
        if np.any(raw < 0):
            raise ValueError("responses must be non-negative integer scores")
    elif kind == "u" and raw.size and np.any(raw > np.uint64(_INT64_MAX)):
        # Keep the boundary in an unsigned NumPy scalar so NumPy 1.x value-based
        # promotion cannot round INT64_MAX through float64 before comparison.
        raise ValueError("responses exceed signed int64 range")

    try:
        scores = raw.astype(np.int64, copy=False)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("responses exceed signed int64 range") from None

    if scores.size and int(scores.max()) + 1 > MAX_POLYTOMOUS_CATEGORIES:
        raise ValueError(
            f"responses imply more than {MAX_POLYTOMOUS_CATEGORIES} categories"
        )
    return np.ascontiguousarray(scores.reshape(-1)), int(n_persons), int(n_items)


def mokken_analysis(
    responses: np.ndarray,
    lower_bound: float | None = None,
    alpha: float | None = None,
) -> MokkenResult:
    """Run Mokken scale analysis with explicitly governed AISP controls.

    Computes the Loevinger scalability coefficients ``Hij``, ``Hi``, ``H``
    with their Mokken Z statistics, and partitions items with the automated
    item selection procedure (AISP), following the sample statistics and
    ``search normal`` algorithm of the mokken R package (van der Ark, 2007).
    ``Hij = S_ij / Smax_ij`` where ``S`` is the sample covariance matrix and
    ``Smax_ij`` is the maximum covariance given the two items' marginals.

    ``lower_bound`` (the AISP scalability lower bound ``c``) and ``alpha``
    (the nominal significance level used by the algorithm) are substantive
    measurement-policy inputs. The package intentionally supplies no default
    values for either control. Conventional values reported in the literature
    do not identify a universal threshold that is valid for every estimand,
    instrument, population, or deployment, so a caller must provide values
    justified by its governed analysis plan rather than receive a rule-of-thumb
    decision from this API.

    In LLM-as-a-Judge item-quality management, AISP can flag evaluation items
    that do not scale with the rest and can expose multidimensional item pools
    before parametric IRT calibration, but the scientific control values remain
    the responsibility of the measurement design.

    ``responses`` is a complete ``persons x items`` array of integer scores
    (dichotomous 0/1 or polytomous); missing values are not supported because
    these Mokken sample statistics assume complete data (van der Ark, 2007).
    Explicitly supplied controls are normalized and range-checked before response
    traversal. When a control is omitted, response evidence is still validated
    before the missing-control error so malformed or hostile response containers
    retain their existing fail-closed boundary. Complex/object response storage
    and values outside signed ``int64`` are rejected before Rust marshalling.

    References (APA 7th ed.):
        van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
            Statistical Software, 20*(11), 1-19.
            https://doi.org/10.18637/jss.v020.i11
        Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing
            optimization algorithms for item selection in Mokken scale
            analysis. *Journal of Classification, 30*(1), 75-99.
            https://doi.org/10.1007/s00357-013-9122-y
        Mokken, R. J. (1971). *A theory and procedure of scale analysis*.
            De Gruyter. (as cited in van der Ark, 2007)
    """
    lower_bound_value: float | None = None
    alpha_value: float | None = None

    if lower_bound is not None:
        lower_bound_value = _real_control("lower_bound", lower_bound)
        if not (0.0 <= lower_bound_value < 1.0):
            raise ValueError("lower_bound must be in [0, 1)")
    if alpha is not None:
        alpha_value = _real_control("alpha", alpha)
        if not (0.0 < alpha_value < 1.0):
            raise ValueError("alpha must be in (0, 1)")

    x, n_persons, n_items = _validated_scores(responses)

    if lower_bound_value is None:
        raise ValueError(
            "lower_bound must be explicitly provided; no rule-of-thumb AISP "
            "threshold is authoritative"
        )
    if alpha_value is None:
        raise ValueError(
            "alpha must be explicitly provided; no nominal significance "
            "default is authoritative"
        )

    from .fitstats import _core_module

    core = _core_module()
    if (
        core is None
        or not hasattr(core, "mokken_coef_h")
        or not hasattr(core, "mokken_aisp")
    ):
        raise RuntimeError("mokken_analysis requires the compiled Rust core")

    coefficient_result = core.mokken_coef_h(x, n_persons, n_items)
    coefficients = _validated_native_coefficients(coefficient_result, n_items)
    del coefficient_result
    scale = core.mokken_aisp(
        x,
        n_persons,
        n_items,
        lower_bound_value,
        alpha_value,
    )
    return _validated_native_result(coefficients, scale, n_items)
