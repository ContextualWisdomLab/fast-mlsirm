"""Callback-free semantic-control admission for factor rotation APIs."""

from __future__ import annotations

import math

import numpy as np

_NUMPY_INTEGER_TYPES = frozenset(
    {
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.int_,
        np.uint,
        np.intp,
        np.uintp,
        np.longlong,
        np.ulonglong,
    }
)
_NUMPY_FLOAT_TYPES = frozenset({np.float16, np.float32, np.float64, np.longdouble})


def boolean(value: object, *, name: str) -> bool:
    """Return an exact built-in or NumPy Boolean as a built-in ``bool``."""
    if type(value) is bool:
        return value
    if type(value) is np.bool_:
        return bool(value)
    raise ValueError(f"{name} must be a boolean")


def integer(value: object, *, name: str) -> int:
    """Return a package-trusted integer identity as a built-in ``int``."""
    value_type = type(value)
    if value_type is int:
        return value
    if value_type in _NUMPY_INTEGER_TYPES:
        return int(value)
    raise ValueError(f"{name} must be an integer")


def real(value: object, *, name: str) -> float:
    """Return a finite package-trusted real scalar as a built-in ``float``."""
    value_type = type(value)
    try:
        if value_type is float:
            normalized = value
        elif value_type is int:
            normalized = float(value)
        elif value_type in _NUMPY_FLOAT_TYPES or value_type in _NUMPY_INTEGER_TYPES:
            normalized = float(value)
        else:
            raise ValueError(f"{name} must be a finite real number")
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be a finite real number")
    return normalized


def optional_real(value: object | None, *, name: str) -> float | None:
    """Normalize an optional finite real without caller protocol dispatch."""
    if value is None:
        return None
    return real(value, name=name)


def optional_integer(value: object | None, *, name: str) -> int | None:
    """Normalize an optional integer without caller protocol dispatch."""
    if value is None:
        return None
    return integer(value, name=name)
