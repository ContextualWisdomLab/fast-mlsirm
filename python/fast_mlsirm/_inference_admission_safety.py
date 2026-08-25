"""Callback-free admission guards for Rust-owned inference diagnostics."""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

_REAL_KINDS = frozenset({"b", "i", "u", "f"})


def _is_numpy_real_scalar(value: object, *, allow_bool: bool) -> bool:
    """Return whether ``value`` is a concrete package-trusted NumPy real scalar."""
    value_type = type(value)
    if value_type.__module__ != "numpy" or not issubclass(value_type, np.generic):
        return False
    try:
        kind = np.dtype(value_type).kind
    except TypeError:
        return False
    if kind == "b":
        return allow_bool
    return kind in {"i", "u", "f"}


def _is_real_scalar(value: object, *, allow_bool: bool) -> bool:
    value_type = type(value)
    if value_type is bool:
        return allow_bool
    if value_type is int or value_type is float:
        return True
    return _is_numpy_real_scalar(value, allow_bool=allow_bool)


def _normalized_nonnegative_real(value: object, name: str) -> float:
    """Normalize a trusted finite non-negative Rust ``f64`` control."""
    if not _is_real_scalar(value, allow_bool=False):
        raise ValueError(f"{name} must be a finite non-negative float")
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative float") from exc
    if not np.isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be a finite non-negative float")

    if type(value) is int:
        if int(normalized) != value:
            raise ValueError(f"{name} must be losslessly representable as float64")
    elif _is_numpy_real_scalar(value, allow_bool=False):
        value_type = type(value)
        try:
            if value_type(normalized) != value:
                raise ValueError(f"{name} must be losslessly representable as float64")
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be losslessly representable as float64") from exc
    return normalized


def _validate_row(row: object, name: str, expected: int) -> None:
    """Validate one inert matrix row without invoking caller protocols."""
    row_type = type(row)
    if row_type is np.ndarray:
        if row.ndim != 1 or row.shape[0] != expected:
            raise ValueError(f"{name} must be a square matrix")
        if row.dtype.kind not in _REAL_KINDS:
            if row.dtype.kind == "c":
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")
        return
    if row_type is not list and row_type is not tuple:
        raise ValueError(f"{name} must be an exact NumPy array or built-in matrix")
    if len(row) != expected:
        raise ValueError(f"{name} must be a square matrix")
    for value in row:
        if not _is_real_scalar(value, allow_bool=True):
            if type(value) is complex or _is_numpy_complex_scalar(value):
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")


def _is_numpy_complex_scalar(value: object) -> bool:
    value_type = type(value)
    if value_type.__module__ != "numpy" or not issubclass(value_type, np.generic):
        return False
    try:
        return np.dtype(value_type).kind == "c"
    except TypeError:
        return False


def _real_square_matrix(value: object, name: str) -> np.ndarray:
    """Seal square-matrix identity before NumPy float64 materialization."""
    value_type = type(value)
    if value_type is np.ndarray:
        if value.dtype.kind not in _REAL_KINDS:
            if value.dtype.kind == "c":
                raise ValueError(f"{name} must be real-valued")
            raise ValueError(f"{name} must contain real numeric values")
        if value.ndim != 2 or value.shape[0] == 0 or value.shape[0] != value.shape[1]:
            raise ValueError(f"{name} must be a square matrix")
        return np.ascontiguousarray(value, dtype=np.float64)

    if value_type is not list and value_type is not tuple:
        raise ValueError(f"{name} must be an exact NumPy array or built-in matrix")
    size = len(value)
    if size == 0:
        raise ValueError(f"{name} must be a square matrix")
    for row in value:
        _validate_row(row, name, size)
    return np.ascontiguousarray(np.asarray(value, dtype=np.float64))


def install(inference_module: ModuleType) -> None:
    """Install idempotent inference admission wrappers on ``inference_module``."""
    if getattr(inference_module.second_order_test, "_inference_admission_safe", False):
        return

    raw_second_order = inference_module.second_order_test
    raw_vcov = inference_module.vcov_from_hessian
    raw_standard_errors = inference_module.standard_errors_from_vcov

    @wraps(raw_second_order)
    def safe_second_order_test(hessian: Any, tol: Any = 1e-8):
        normalized_tol = _normalized_nonnegative_real(tol, "tol")
        matrix = _real_square_matrix(hessian, "hessian")
        return raw_second_order(matrix, tol=normalized_tol)

    @wraps(raw_vcov)
    def safe_vcov_from_hessian(hessian: Any, rcond: Any = 1e-10):
        normalized_rcond = _normalized_nonnegative_real(rcond, "rcond")
        matrix = _real_square_matrix(hessian, "hessian")
        return raw_vcov(matrix, rcond=normalized_rcond)

    @wraps(raw_standard_errors)
    def safe_standard_errors_from_vcov(vcov: Any):
        matrix = _real_square_matrix(vcov, "vcov")
        return raw_standard_errors(matrix)

    safe_second_order_test._inference_admission_safe = True
    safe_vcov_from_hessian._inference_admission_safe = True
    safe_standard_errors_from_vcov._inference_admission_safe = True

    inference_module.second_order_test = safe_second_order_test
    inference_module.vcov_from_hessian = safe_vcov_from_hessian
    inference_module.standard_errors_from_vcov = safe_standard_errors_from_vcov
