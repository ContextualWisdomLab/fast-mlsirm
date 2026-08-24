"""Callback-free serving trust-boundary admission.

Serving artifacts are scientific deployment records: their structural controls,
item identities, dimension identities, and item-to-dimension identities must be
established before caller callbacks, NumPy integer narrowing, or Rust-owned
scoring can observe them. Numerical scoring arithmetic remains in the existing
Rust-backed serving implementation.
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


def _item_code_list(value: Any, n_items: int) -> list[str]:
    """Return inert built-in item identities without caller container callbacks."""
    value_type = builtins.type(value)
    if value_type is not list and value_type is not tuple:
        raise ValueError("item_codes must be a list or tuple of built-in strings")
    if len(value) != n_items:
        raise ValueError("item_codes length must match the fitted item count")

    normalized: list[str] = []
    for code in value:
        if builtins.type(code) is not str:
            raise ValueError("item_codes must contain built-in strings")
        normalized.append(code)
    return normalized


def _dimension_name_list(value: Any, n_dims: int) -> list[str] | None:
    """Return inert built-in dimension labels without caller callbacks."""
    if value is None:
        return None
    if builtins.type(value) is not list:
        raise ValueError("dim_names must be null or a list of built-in strings")
    if len(value) != n_dims:
        raise ValueError("dim_names length must match the fitted dimension count")

    normalized: list[str] = []
    for name in value:
        if builtins.type(name) is not str:
            raise ValueError("dim_names must contain built-in strings")
        normalized.append(name)
    return normalized


def _quadrature_integer(value: Any, *, label: str) -> int:
    """Normalize one trusted integer quadrature control to a JSON-safe int."""
    value_type = builtins.type(value)
    if value_type is int:
        return value
    if any(value_type is scalar_type for scalar_type in _NUMPY_INTEGER_TYPES):
        return int(value)
    raise ValueError(f"{label} must be an integer")


def _exact_string_keys(mapping: dict[Any, Any]) -> bool:
    """Return whether an exact built-in mapping contains only inert text keys."""
    return all(builtins.type(key) is str for key in mapping)


def install(serving_module: Any) -> None:
    """Install callback-free serving validation and export identity boundaries."""
    original_validate: Callable[..., Any] = serving_module._validate_bundle
    original_export: Callable[..., Any] = serving_module.export_serving_bundle

    @wraps(original_validate)
    def safe_validate_bundle(bundle: Any) -> Any:
        # Mirror the original validation order far enough to inspect
        # population.kind only after the preceding top-level contracts are
        # already known to be inert and valid. Otherwise defer untouched to
        # the original package-owned error path.
        if builtins.type(bundle) is dict and _exact_string_keys(bundle):
            schema_version = bundle.get("schema_version")
            if (
                builtins.type(schema_version) is int
                and schema_version == serving_module.SCHEMA_VERSION
            ):
                population = bundle.get("population")
                if (
                    builtins.type(population) is dict
                    and _exact_string_keys(population)
                ):
                    kind = population.get("kind")
                    if kind is not None and builtins.type(kind) is not str:
                        raise ValueError("bundle population kind must be a string")

        # Preserve every existing package-owned validation/error precedence.
        # `dim_names` is not consumed by the historical validator, so replay its
        # identity contract only after the rest of the bundle has been proven
        # valid and before any public scorer can discover the compiled core.
        result = original_validate(bundle)
        _dimension_name_list(bundle.get("dim_names"), bundle["n_dims"])
        return result

    @wraps(original_export)
    def safe_export_serving_bundle(
        result: Any,
        item_codes: Any,
        factor_id: Any,
        path: Any = None,
        q_theta: Any = 21,
        q_xi: Any = 11,
        eps_distance: Any = 1e-8,
        screening_audit: Any = None,
        dim_names: Any = None,
    ) -> Any:
        # Preserve the legacy convergence-error ordering. Only a result that
        # already crossed that original precondition reaches the artifact
        # identity and numerical-control trust boundaries below.
        status = getattr(result, "convergence_status", None)
        if builtins.type(status) is not str or status.strip().lower() != "converged":
            return original_export(
                result,
                item_codes,
                factor_id,
                path,
                q_theta,
                q_xi,
                eps_distance,
                screening_audit,
                dim_names,
            )

        params = result.params
        n_items = len(params.b)
        item_codes_value = _item_code_list(item_codes, n_items)
        factor_value = _factor_id_vector(factor_id, n_items)
        n_dims = int(factor_value.max()) + 1
        dim_names_value = _dimension_name_list(dim_names, n_dims)
        q_theta_value = _quadrature_integer(q_theta, label="q_theta")
        q_xi_value = _quadrature_integer(q_xi, label="q_xi")
        return original_export(
            result,
            item_codes_value,
            factor_value,
            path,
            q_theta_value,
            q_xi_value,
            eps_distance,
            screening_audit,
            dim_names_value,
        )

    serving_module._validate_bundle = safe_validate_bundle
    serving_module.export_serving_bundle = safe_export_serving_bundle
