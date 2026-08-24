"""Inert semantic-control validation for the public ICC adapter.

The ICC formulas and all result-affecting arithmetic remain Rust-owned. This
module only establishes exact trusted scalar identities and parameter ranges
before the legacy adapter can discover the native core or materialize ratings.
"""

from __future__ import annotations

import builtins
from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_NUMPY_REAL_SCALAR_TYPES = (
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


def _choice(value: Any, name: str, allowed: tuple[str, ...]) -> str:
    """Return one exact built-in string from ``allowed`` without callbacks."""
    if builtins.type(value) is not str:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    if value not in allowed:
        raise ValueError(f"{name} must be one of {', '.join(allowed)}")
    return value


def _real(value: Any, name: str) -> float:
    """Normalize one exact trusted real scalar without caller-owned coercion."""
    value_type = builtins.type(value)
    try:
        if value_type is int:
            normalized = float(value)
        elif value_type is float:
            normalized = value
        elif any(value_type is scalar_type for scalar_type in _NUMPY_REAL_SCALAR_TYPES):
            normalized = float(value)
        else:
            raise ValueError(f"{name} must be a real number")
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    return normalized


def install(reliability_module: ModuleType) -> None:
    """Wrap ``reliability.icc`` with callback-free semantic validation."""
    original: Callable[..., Any] = reliability_module.icc

    @wraps(original)
    def safe_icc(
        ratings: Any,
        model: str = "oneway",
        type: str = "consistency",
        unit: str = "single",
        r0: float = 0.0,
        conf_level: float = 0.95,
    ) -> Any:
        """Validate ICC controls before native discovery or ratings access."""
        model_value = _choice(model, "model", ("oneway", "twoway"))
        type_value = _choice(type, "type", ("consistency", "agreement"))
        unit_value = _choice(unit, "unit", ("single", "average"))
        r0_value = _real(r0, "r0")
        conf_level_value = _real(conf_level, "conf_level")
        if not 0.0 <= r0_value < 1.0:
            raise ValueError("r0 must be in [0, 1)")
        if not 0.0 < conf_level_value < 1.0:
            raise ValueError("conf_level must be in (0, 1)")
        return original(
            ratings,
            model=model_value,
            type=type_value,
            unit=unit_value,
            r0=r0_value,
            conf_level=conf_level_value,
        )

    reliability_module.icc = safe_icc
