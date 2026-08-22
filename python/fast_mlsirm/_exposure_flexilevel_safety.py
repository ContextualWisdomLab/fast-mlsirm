"""Fail-closed flexilevel admission before Rust discovery or lossy coercion.

This module wraps only the Python validation/marshalling boundary. Lord routing,
red/blue self-scoring, forward recursion, score-lattice probabilities, moments,
and every result-affecting calculation remain owned by the existing Rust core.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_USIZE_MAX = int(np.iinfo(np.uintp).max)
_REAL_NUMERIC_KINDS = frozenset({"b", "i", "u", "f"})


def _binary_responses(
    value: object,
    *,
    n_persons: int,
    n_items: int,
) -> np.ndarray:
    """Admit exact flexilevel response storage before ``uint8`` marshalling."""

    expected = n_persons * n_items
    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError("responses must be a real numeric array")
    if array.ndim == 2:
        if array.shape != (n_persons, n_items):
            raise ValueError(
                f"responses has shape {array.shape}, expected ({n_persons}, {n_items})"
            )
        array = array.reshape(-1)
    elif array.ndim == 1:
        if array.size != expected:
            raise ValueError("responses size must equal n_persons * n_items")
    else:
        raise ValueError("responses must be a 1-D or 2-D array")
    if not np.isin(array, (0, 1)).all():
        raise ValueError("responses must contain only 0 and 1")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _probabilities(value: object) -> np.ndarray:
    """Admit one odd real probability vector before ``float64`` marshalling."""

    array = np.asarray(value)
    if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
        raise ValueError("p must be a real numeric array")
    if array.ndim != 1:
        raise ValueError("p must be a 1-D array")
    if array.size < 3 or array.size % 2 == 0:
        raise ValueError("p length must be odd and at least 3")
    normalized = np.ascontiguousarray(array, dtype=np.float64)
    if (
        not np.isfinite(normalized).all()
        or (normalized < 0.0).any()
        or (normalized > 1.0).any()
    ):
        raise ValueError("p must contain finite values in [0, 1]")
    return normalized


def install(exposure_module: ModuleType) -> None:
    """Install callback-safe wrappers on the historical exposure module."""

    original_administer: Callable[..., Any] = exposure_module.flexilevel_administer
    original_distribution: Callable[..., Any] = (
        exposure_module.flexilevel_score_distribution
    )

    @wraps(original_administer)
    def safe_flexilevel_administer(
        responses: object,
        *,
        n_persons: int,
        n_items: int,
    ) -> dict:
        """Validate controls and response evidence before the Rust boundary."""

        n_persons_value = exposure_module._as_int(
            "n_persons", n_persons, minimum=1, maximum=_USIZE_MAX
        )
        n_items_value = exposure_module._as_int(
            "n_items", n_items, minimum=3, maximum=_USIZE_MAX
        )
        if n_items_value % 2 == 0:
            raise ValueError("n_items must be odd")
        if n_persons_value > _USIZE_MAX // n_items_value:
            raise ValueError("n_persons * n_items exceeds platform size")
        normalized = _binary_responses(
            responses,
            n_persons=n_persons_value,
            n_items=n_items_value,
        )
        return original_administer(
            normalized,
            n_persons=n_persons_value,
            n_items=n_items_value,
        )

    @wraps(original_distribution)
    def safe_flexilevel_score_distribution(p: object) -> dict:
        """Validate probability evidence before the Rust recursion boundary."""

        return original_distribution(_probabilities(p))

    exposure_module.flexilevel_administer = safe_flexilevel_administer
    exposure_module.flexilevel_score_distribution = safe_flexilevel_score_distribution
