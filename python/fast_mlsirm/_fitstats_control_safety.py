"""Callback-free admission for public fit-statistics controls and BH evidence.

This composition layer replaces only Python validation and marshalling. The
S-X²/G² statistics, quadrature, BH/FDR decisions, and all production numerical
work remain owned by the compiled Rust fit-statistics implementation.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_SUPPORTED_QUADRATURE = (7, 11, 15, 21, 31, 41)
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
_NUMPY_FLOAT_SCALAR_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_BH_HARDENED_ATTR = "__fast_mlsirm_bh_admission_hardened__"
_MAX_PROBABILITY_TREE_DEPTH = 64


def _trusted_integer(value: Any, name: str) -> int:
    """Return one exact trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be one of {_SUPPORTED_QUADRATURE}")


def _trusted_integer_real(value: Any, name: str) -> float:
    """Return an integer-valued real only when float64 preserves it exactly."""
    integer = int(value)
    try:
        converted = float(integer)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if int(converted) != integer:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def _trusted_numpy_real(value: Any, name: str) -> float:
    """Return a trusted NumPy real only when Rust f64 preserves its value."""
    value_type = type(value)
    try:
        converted = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not np.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    if value_type(converted) != value:
        raise ValueError(f"{name} must be exactly representable as float64")
    return converted


def _trusted_real(value: Any, name: str) -> float:
    """Return one exact trusted real scalar without caller conversion hooks."""
    value_type = type(value)
    if value_type is int:
        return _trusted_integer_real(value, name)
    if value_type is float:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return _trusted_integer_real(value, name)
    if any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
        return _trusted_numpy_real(value, name)
    raise ValueError(f"{name} must be a finite number")


def _validate_sx2_controls(
    q_theta: Any,
    q_xi: Any,
    min_expected: Any,
    fdr_q: Any,
    min_effect: Any,
) -> tuple[int, int, float, float, float]:
    """Validate S-X² scalar controls without executing caller callbacks."""
    quadrature = []
    for name, value in (("q_theta", q_theta), ("q_xi", q_xi)):
        normalized = _trusted_integer(value, name)
        if normalized not in _SUPPORTED_QUADRATURE:
            raise ValueError(f"{name} must be one of {_SUPPORTED_QUADRATURE}")
        quadrature.append(normalized)

    numeric = []
    for name, value in (
        ("min_expected", min_expected),
        ("fdr_q", fdr_q),
        ("min_effect", min_effect),
    ):
        converted = _trusted_real(value, name)
        if not np.isfinite(converted):
            raise ValueError(f"{name} must be a finite number")
        numeric.append(converted)

    min_expected_value, fdr_q_value, min_effect_value = numeric
    if min_expected_value <= 0.0:
        raise ValueError("min_expected must be positive")
    if not 0.0 < fdr_q_value <= 1.0:
        raise ValueError("fdr_q must be in (0, 1]")
    if min_effect_value < 0.0:
        raise ValueError("min_effect must be non-negative")
    return (
        quadrature[0],
        quadrature[1],
        min_expected_value,
        fdr_q_value,
        min_effect_value,
    )


def _trusted_probability_tree(value: Any) -> None:
    """Preflight inert probability evidence without arbitrary conversion hooks."""
    stack: list[tuple[Any, int, bool]] = [(value, 0, False)]
    active_container_ids: set[int] = set()
    while stack:
        current, depth, leaving = stack.pop()
        current_type = type(current)
        if current_type is list or current_type is tuple:
            identity = id(current)
            if leaving:
                active_container_ids.remove(identity)
                continue
            if identity in active_container_ids:
                raise ValueError("p_values must not contain cyclic containers")
            if depth >= _MAX_PROBABILITY_TREE_DEPTH:
                raise ValueError("p_values nesting exceeds the supported depth")
            active_container_ids.add(identity)
            stack.append((current, depth, True))
            stack.extend((child, depth + 1, False) for child in reversed(current))
            continue
        if current_type is np.ndarray:
            if np.iscomplexobj(current) or current.dtype.kind not in "biuf":
                raise ValueError("p_values must contain real numeric probabilities")
            continue
        if current_type is bool or current_type is int or current_type is float:
            continue
        if current_type is np.bool_:
            continue
        if any(current_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            continue
        if any(current_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
            continue
        raise ValueError("p_values must contain real numeric probabilities")


def _trusted_probability_array(value: Any) -> np.ndarray:
    """Return package-owned float64 p-values after callback-free validation."""
    value_type = type(value)
    if value_type is np.ndarray:
        array = value
    elif value_type is list or value_type is tuple:
        _trusted_probability_tree(value)
        try:
            array = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("p_values must contain real numeric probabilities") from error
    elif (
        value_type is bool
        or value_type is int
        or value_type is float
        or value_type is np.bool_
        or any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES)
        or any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES)
    ):
        array = np.asarray(value)
    else:
        raise ValueError("p_values must contain real numeric probabilities")

    if np.iscomplexobj(array) or array.dtype.kind not in "biuf":
        raise ValueError("p_values must contain real numeric probabilities")
    if np.any(~np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise ValueError("p_values must be finite probabilities in [0, 1]")
    try:
        normalized = np.ascontiguousarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("p_values must contain real numeric probabilities") from error
    if array.dtype.kind != "b":
        round_trip = normalized.astype(array.dtype, copy=False)
        if not np.array_equal(round_trip, array):
            raise ValueError("p_values must be exactly representable as float64")
    return normalized


def _bind_legacy_bh(function: Callable[..., Any]) -> None:
    """Rebind the historical package export to the hardened BH callable."""
    from . import _legacy_init

    if hasattr(_legacy_init, "benjamini_hochberg"):
        _legacy_init.benjamini_hochberg = function


def install(fitstats_module: ModuleType) -> None:
    """Install callback-free S-X² controls and BH evidence admission."""
    fitstats_module._validate_sx2_controls = _validate_sx2_controls

    current_bh: Callable[..., Any] = fitstats_module.benjamini_hochberg
    if getattr(current_bh, _BH_HARDENED_ATTR, False):
        _bind_legacy_bh(current_bh)
        return
    original_bh = current_bh

    @wraps(original_bh)
    def safe_benjamini_hochberg(p_values: Any, q: Any = 0.05) -> np.ndarray:
        """Validate BH threshold and probability evidence before Rust discovery."""
        q_value = _trusted_real(q, "q")
        if not np.isfinite(q_value) or not 0.0 < q_value <= 1.0:
            raise ValueError("q must be in (0, 1]")
        probabilities = _trusted_probability_array(p_values)
        output_shape = probabilities.shape
        flat_probabilities = np.ascontiguousarray(probabilities.ravel())
        result = original_bh(flat_probabilities, q_value)
        return np.asarray(result, dtype=bool).reshape(output_shape)

    setattr(safe_benjamini_hochberg, _BH_HARDENED_ATTR, True)
    fitstats_module.benjamini_hochberg = safe_benjamini_hochberg
    _bind_legacy_bh(safe_benjamini_hochberg)
