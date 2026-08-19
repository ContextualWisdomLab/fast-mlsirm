"""Callback-free scalar admission for public S-X² controls.

This composition layer replaces only Python validation and marshalling. The
S-X²/G² statistics, quadrature, BH/FDR decisions, and all production numerical
work remain owned by the compiled Rust fit-statistics implementation.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

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


def _trusted_integer(value: Any, name: str) -> int:
    """Return one exact trusted integer without caller-controlled coercion."""
    value_type = type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return int(value)
    raise ValueError(f"{name} must be one of {_SUPPORTED_QUADRATURE}")


def _trusted_real(value: Any, name: str) -> float:
    """Return one exact trusted real scalar without caller conversion hooks."""
    value_type = type(value)
    if value_type is int or value_type is float:
        return float(value)
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
        return float(value)
    if any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_SCALAR_TYPES):
        return float(value)
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


def install(fitstats_module: ModuleType) -> None:
    """Install inert S-X² scalar validation on the fit-statistics module."""
    fitstats_module._validate_sx2_controls = _validate_sx2_controls
