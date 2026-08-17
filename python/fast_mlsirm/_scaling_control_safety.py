"""Trusted scalar boundary for paired-comparison scaling controls.

This module validates and normalizes caller-owned semantic controls before the
historical :mod:`fast_mlsirm.scaling` wrapper materializes data or discovers
the compiled core. Bradley-Terry estimation and every result-affecting
calculation remain owned by the Rust implementation.
"""

from __future__ import annotations

from functools import wraps
import math
from types import ModuleType
from typing import Any, Callable

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
            return float(value)
        if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_SCALAR_TYPES):
            return float(value)
        if any(value_type is scalar_type for scalar_type in _NUMPY_FLOATING_SCALAR_TYPES):
            return float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    raise ValueError(f"{name} must be a real number")


def _normalized_controls(alpha: object, max_iter: object, tol: object) -> tuple[float, int, float]:
    """Normalize and domain-check Bradley-Terry controls before side effects."""
    normalized_alpha = _exact_real(alpha, "alpha")
    normalized_max_iter = _exact_integer(max_iter, "max_iter")
    normalized_tol = _exact_real(tol, "tol")

    if not math.isfinite(normalized_alpha) or normalized_alpha < 0.0:
        raise ValueError("alpha must be finite and nonnegative")
    if not 1 <= normalized_max_iter <= MAX_MAX_ITER:
        raise ValueError(f"max_iter must be between 1 and {MAX_MAX_ITER}")
    if not math.isfinite(normalized_tol) or normalized_tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    return normalized_alpha, normalized_max_iter, normalized_tol


def install(scaling_module: ModuleType) -> None:
    """Install the Bradley-Terry control wrapper exactly once on ``scaling_module``."""
    original: Callable[..., Any] = scaling_module.bradley_terry_mm
    if getattr(original, "__fast_mlsirm_control_hardened__", False):
        return

    @wraps(original)
    def hardened_bradley_terry_mm(
        wins: object,
        alpha: object = 0.0,
        max_iter: object = 10_000,
        tol: object = 1e-8,
    ) -> Any:
        """Validate trusted controls, then delegate unchanged arithmetic to Rust."""
        normalized_alpha, normalized_max_iter, normalized_tol = _normalized_controls(
            alpha, max_iter, tol
        )
        return original(
            wins,
            alpha=normalized_alpha,
            max_iter=normalized_max_iter,
            tol=normalized_tol,
        )

    hardened_bradley_terry_mm.__fast_mlsirm_control_hardened__ = True
    scaling_module.bradley_terry_mm = hardened_bradley_terry_mm
