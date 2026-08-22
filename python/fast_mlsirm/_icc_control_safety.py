"""Callback-free validation adapters for public reliability APIs.

All reliability formulas and result-affecting arithmetic remain Rust-owned.
This module establishes exact trusted scalar/evidence identities before legacy
adapters can discover the native core or materialize caller-controlled data.
"""

from __future__ import annotations

import builtins
import sys
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


def _integer(value: Any, name: str) -> int:
    """Normalize one exact trusted integer scalar without caller callbacks."""
    value_type = builtins.type(value)
    if value_type is int:
        return value
    if any(
        value_type is scalar_type
        for scalar_type in _NUMPY_REAL_SCALAR_TYPES
        if np.dtype(scalar_type).kind in "iu"
    ):
        return int(value)
    raise ValueError(f"{name} must be an integer")


def _trusted_sequence_tree(value: Any) -> bool:
    """Return whether an exact list/tuple tree contains only trusted reals."""
    stack = [value]
    while stack:
        current = stack.pop()
        current_type = builtins.type(current)
        if current_type is list or current_type is tuple:
            stack.extend(current)
            continue
        if current_type is bool or current_type is int or current_type is float:
            continue
        if current_type is np.bool_:
            continue
        if any(current_type is scalar_type for scalar_type in _NUMPY_REAL_SCALAR_TYPES):
            continue
        return False
    return True


def _real_numeric_array(value: Any, name: str, ndim: int) -> np.ndarray:
    """Marshal inert real-numeric evidence without arbitrary array callbacks."""
    value_type = builtins.type(value)
    if value_type is np.ndarray:
        arr = value
    elif value_type is list or value_type is tuple:
        if not _trusted_sequence_tree(value):
            raise ValueError(f"{name} must be real numeric evidence")
        try:
            arr = np.asarray(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be real numeric evidence") from exc
    else:
        raise ValueError(f"{name} must be real numeric evidence")

    if np.iscomplexobj(arr) or arr.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be real numeric evidence")
    if arr.ndim != ndim:
        dimension = "2-D persons x items" if ndim == 2 else "1-D"
        raise ValueError(f"{name} must be a {dimension} array")
    try:
        return np.ascontiguousarray(arr, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be real numeric evidence") from exc


def install(reliability_module: ModuleType) -> None:
    """Install callback-free control and evidence validation adapters."""
    original_icc: Callable[..., Any] = reliability_module.icc
    original_guttman: Callable[..., Any] = reliability_module.guttman_lambdas
    original_tenberge: Callable[..., Any] = reliability_module.tenberge_mu
    original_alpha: Callable[..., Any] = reliability_module.cronbach_alpha
    original_separation: Callable[..., Any] = reliability_module.separation_reliability

    @wraps(original_icc)
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
        return original_icc(
            ratings,
            model=model_value,
            type=type_value,
            unit=unit_value,
            r0=r0_value,
            conf_level=conf_level_value,
        )

    @wraps(original_guttman)
    def safe_guttman_lambdas(
        data: Any,
        n_sample_splits: int = 15000,
        seed: int = 1,
    ) -> Any:
        """Validate Guttman controls and raw evidence before native discovery."""
        split_value = _integer(n_sample_splits, "n_sample_splits")
        seed_value = _integer(seed, "seed")
        if split_value < 1:
            raise ValueError("n_sample_splits must be >= 1")
        if seed_value < 0:
            raise ValueError("seed must be non-negative")
        data_value = _real_numeric_array(data, "data", 2)
        return original_guttman(data_value, n_sample_splits=split_value, seed=seed_value)

    @wraps(original_tenberge)
    def safe_tenberge_mu(data: Any) -> Any:
        """Validate ten Berge raw evidence before native discovery."""
        return original_tenberge(_real_numeric_array(data, "data", 2))

    @wraps(original_alpha)
    def safe_cronbach_alpha(data: Any) -> Any:
        """Validate alpha raw evidence before native discovery."""
        return original_alpha(_real_numeric_array(data, "data", 2))

    @wraps(original_separation)
    def safe_separation_reliability(measures: Any, se: Any) -> Any:
        """Validate person-measure evidence before native discovery."""
        measures_value = _real_numeric_array(measures, "measures", 1)
        se_value = _real_numeric_array(se, "se", 1)
        return original_separation(measures_value, se_value)

    reliability_module.icc = safe_icc
    reliability_module.guttman_lambdas = safe_guttman_lambdas
    reliability_module.tenberge_mu = safe_tenberge_mu
    reliability_module.cronbach_alpha = safe_cronbach_alpha
    reliability_module.separation_reliability = safe_separation_reliability

    # ``_legacy_init`` copies public symbols before this installer runs. Rebind
    # only the four evidence-hardened reliability entry points so the top-level
    # package surface cannot retain a pre-install function object.
    legacy_name = f"{reliability_module.__package__}._legacy_init"
    legacy_module = sys.modules.get(legacy_name)
    if legacy_module is not None:
        legacy_module.guttman_lambdas = safe_guttman_lambdas
        legacy_module.tenberge_mu = safe_tenberge_mu
        legacy_module.cronbach_alpha = safe_cronbach_alpha
        legacy_module.separation_reliability = safe_separation_reliability
