"""Callback-free serving-export identity admission.

The serving artifact is a scientific deployment record: item-to-dimension
identity must be established before NumPy integer narrowing or Rust-owned
EAP-sum generation can observe it.  Numerical scoring arithmetic remains in
the existing Rust-backed serving implementation.
"""

from __future__ import annotations

import builtins
import math
from functools import wraps
from typing import Any, Callable

import numpy as np

_MAX_SERVING_DIMS = 64
_NUMPY_INTEGER_TYPES = (
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
_NUMPY_FLOAT_TYPES = (np.float16, np.float32, np.float64, np.longdouble)


def _factor_scalar(value: Any) -> int:
    """Normalize one trusted integral factor identity without caller callbacks."""
    value_type = builtins.type(value)
    if value_type is bool:
        normalized = int(value)
    elif value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = int(value)
    elif value_type is float or any(
        value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES
    ):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("factor_id values must be finite integers")
        normalized = int(numeric)
    else:
        raise ValueError("factor_id must contain trusted real numeric values")

    if not 0 <= normalized < _MAX_SERVING_DIMS:
        raise ValueError(
            f"factor_id values must be integers in 0..{_MAX_SERVING_DIMS - 1}"
        )
    return normalized


def _factor_id_vector(value: Any, n_items: int) -> np.ndarray:
    """Return a callback-free contiguous signed-64 factor identity vector."""
    value_type = builtins.type(value)
    if value_type is list or value_type is tuple:
        if len(value) != n_items:
            raise ValueError("factor_id length must match the fitted item count")
        normalized = [_factor_scalar(cell) for cell in value]
        return np.ascontiguousarray(normalized, dtype=np.int64)

    if value_type is not np.ndarray:
        raise ValueError("factor_id must be a 1-D real numeric array or sequence")
    if value.shape != (n_items,):
        raise ValueError("factor_id length must match the fitted item count")
    if value.dtype.kind not in "biuf":
        raise ValueError("factor_id must contain trusted real numeric values")

    if value.dtype.kind == "f":
        if not np.all(np.isfinite(value)) or not np.all(value == np.floor(value)):
            raise ValueError("factor_id values must be finite integers")
    if not np.all(value >= 0) or not np.all(value < _MAX_SERVING_DIMS):
        raise ValueError(
            f"factor_id values must be integers in 0..{_MAX_SERVING_DIMS - 1}"
        )
    return np.ascontiguousarray(value, dtype=np.int64)


def install(serving_module: Any) -> None:
    """Install serving-export factor-identity validation before lossy narrowing."""
    original: Callable[..., Any] = serving_module.export_serving_bundle

    @wraps(original)
    def safe_export_serving_bundle(
        result: Any,
        item_codes: Any,
        factor_id: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        # Preserve the legacy error ordering for unfinished calibrations and
        # item-code length mismatches.  Only a result that has already crossed
        # those two original preconditions reaches the new factor boundary.
        status = getattr(result, "convergence_status", None)
        if builtins.type(status) is not str or status.strip().lower() != "converged":
            return original(result, item_codes, factor_id, *args, **kwargs)

        params = result.params
        n_items = len(params.b)
        if len(item_codes) != n_items:
            return original(result, item_codes, factor_id, *args, **kwargs)

        factor_value = _factor_id_vector(factor_id, n_items)
        return original(result, item_codes, factor_value, *args, **kwargs)

    serving_module.export_serving_bundle = safe_export_serving_bundle
