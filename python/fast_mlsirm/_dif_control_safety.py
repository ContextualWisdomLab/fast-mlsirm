"""Callback-free semantic-control adapters for observed-score DIF APIs.

The adapters normalize only Python-side controls. Logistic/Mantel-Haenszel
statistics, purification, Benjamini-Hochberg adjustment, and every numerical
result remain owned by the existing Rust-backed implementations.
"""

from __future__ import annotations

import builtins
import math
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

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
_NUMPY_FLOAT_TYPES = (
    np.float16,
    np.float32,
    np.float64,
    np.longdouble,
)
_USIZE_MAX = int(np.iinfo(np.uintp).max)


def _boolean(value: Any, name: str) -> bool:
    """Normalize one concrete Python/NumPy Boolean without truth callbacks."""
    value_type = builtins.type(value)
    if value_type is bool:
        return value
    if value_type is np.bool_:
        return bool(value)
    raise TypeError(f"{name} must be a bool")


def _real(value: Any, name: str) -> float:
    """Normalize one concrete real scalar without caller conversion hooks."""
    value_type = builtins.type(value)
    try:
        if value_type is int:
            normalized = float(value)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
            normalized = float(value)
        elif any(value_type is scalar_type for scalar_type in _NUMPY_FLOAT_TYPES):
            normalized = float(value)
        else:
            raise TypeError(f"{name} must be a real number")
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def _usize(value: Any, name: str, *, minimum: int) -> int:
    """Normalize one concrete unsigned-size control without index callbacks."""
    value_type = builtins.type(value)
    if value_type is int:
        normalized = value
    elif any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        normalized = int(value)
    else:
        raise TypeError(f"{name} must be an integer")
    if normalized < minimum:
        comparator = ">= 1" if minimum == 1 else ">= 0"
        raise ValueError(f"{name} must be {comparator}")
    if normalized > _USIZE_MAX:
        raise ValueError(f"{name} exceeds the platform usize range")
    return normalized


def _common_controls(exclude_studied_item: Any, fdr_q: Any) -> tuple[bool, float]:
    """Normalize controls shared by observed-score DIF entry points."""
    exclude = _boolean(exclude_studied_item, "exclude_studied_item")
    q = _real(fdr_q, "fdr_q")
    if not 0.0 < q <= 1.0:
        raise ValueError("fdr_q must be finite and in (0, 1]")
    return exclude, q


def install(dif_module: ModuleType) -> None:
    """Install idempotent semantic-control adapters on public DIF functions."""
    current: Callable[..., Any] = dif_module.logistic_dif
    if getattr(current, "__fast_mlsirm_dif_control_hardened__", False):
        return

    original_logistic = current
    original_mh_purified: Callable[..., Any] = dif_module.mantel_haenszel_dif_purified
    original_logistic_purified: Callable[..., Any] = dif_module.logistic_dif_purified

    @wraps(original_logistic)
    def safe_logistic_dif(
        responses: Any,
        group: Any,
        exclude_studied_item: Any = False,
        fdr_q: Any = 0.05,
        max_iter: Any = 50,
    ) -> Any:
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        iterations = _usize(max_iter, "max_iter", minimum=0)
        return original_logistic(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_iter=iterations,
        )

    @wraps(original_mh_purified)
    def safe_mantel_haenszel_dif_purified(
        responses: Any,
        group: Any,
        exclude_studied_item: Any = False,
        fdr_q: Any = 0.05,
        max_rounds: Any = 3,
        min_anchor_items: Any = 4,
    ) -> Any:
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        rounds = _usize(max_rounds, "max_rounds", minimum=1)
        anchors = _usize(min_anchor_items, "min_anchor_items", minimum=0)
        return original_mh_purified(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_rounds=rounds,
            min_anchor_items=anchors,
        )

    @wraps(original_logistic_purified)
    def safe_logistic_dif_purified(
        responses: Any,
        group: Any,
        exclude_studied_item: Any = False,
        fdr_q: Any = 0.05,
        max_iter: Any = 50,
        max_rounds: Any = 3,
        min_anchor_items: Any = 4,
    ) -> Any:
        exclude, q = _common_controls(exclude_studied_item, fdr_q)
        iterations = _usize(max_iter, "max_iter", minimum=0)
        rounds = _usize(max_rounds, "max_rounds", minimum=1)
        anchors = _usize(min_anchor_items, "min_anchor_items", minimum=0)
        return original_logistic_purified(
            responses,
            group,
            exclude_studied_item=exclude,
            fdr_q=q,
            max_iter=iterations,
            max_rounds=rounds,
            min_anchor_items=anchors,
        )

    safe_logistic_dif.__fast_mlsirm_dif_control_hardened__ = True
    safe_mantel_haenszel_dif_purified.__fast_mlsirm_dif_control_hardened__ = True
    safe_logistic_dif_purified.__fast_mlsirm_dif_control_hardened__ = True
    dif_module.logistic_dif = safe_logistic_dif
    dif_module.mantel_haenszel_dif_purified = safe_mantel_haenszel_dif_purified
    dif_module.logistic_dif_purified = safe_logistic_dif_purified
