"""Inert NumPy-container admission for exposure APIs.

The exposure public surface is typed in terms of ``numpy.ndarray``. Establish
that package-owned container identity before any ``np.asarray`` call so a
caller-defined ``__array__`` provider cannot execute code during validation.
All numerical psychometric work remains in the existing Rust core.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any, Callable

import numpy as np

_REAL_NUMERIC_KINDS = frozenset({"b", "i", "u", "f"})


def exact_ndarray(value: object, *, message: str) -> np.ndarray:
    """Return an exact ndarray or fail before invoking array protocols."""

    if type(value) is not np.ndarray:
        raise ValueError(message)
    return value


def install(exposure_module: ModuleType) -> None:
    """Replace exposure array admission helpers and guard inline CCAT groups."""

    def real_numeric_array(name: str, value: object) -> np.ndarray:
        array = exact_ndarray(value, message=f"{name} must be a real numeric array")
        if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
            raise ValueError(f"{name} must be a real numeric array")
        return np.ascontiguousarray(array, dtype=np.float64)

    def boolean_array(name: str, value: object) -> np.ndarray:
        array = exact_ndarray(value, message=f"{name} must be a boolean array")
        if array.dtype != np.bool_:
            raise ValueError(f"{name} must be a boolean array")
        return np.ascontiguousarray(array, dtype=np.bool_)

    def binary_response_array(name: str, value: object) -> np.ndarray:
        array = exact_ndarray(value, message=f"{name} must be a real numeric array")
        if np.iscomplexobj(array) or array.dtype.kind not in _REAL_NUMERIC_KINDS:
            raise ValueError(f"{name} must be a real numeric array")
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1-D array")
        values = np.ascontiguousarray(array, dtype=np.float64)
        if not np.isin(values, (0.0, 1.0)).all():
            raise ValueError(f"{name} must contain only 0 or 1")
        return np.ascontiguousarray(values, dtype=np.uint8)

    exposure_module._as_real_numeric_array = real_numeric_array
    exposure_module._as_boolean_array = boolean_array
    exposure_module._as_binary_response_array = binary_response_array

    original_ccat: Callable[..., Any] = exposure_module.ccat_select

    @wraps(original_ccat)
    def safe_ccat_select(
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray | None = None,
        *,
        groups: object,
        targets: np.ndarray,
        administered: np.ndarray,
        theta0: float,
    ) -> dict:
        """Reject unsafe CCAT controls/providers before legacy validation."""

        # Preserve the public contract that semantic controls fail before any
        # caller-owned array work.  The wrapped implementation validates this
        # value again, but only after it has been normalized to an inert
        # package-owned float.
        theta0_value = exposure_module._as_real_scalar("theta0", theta0)
        groups_value = exact_ndarray(
            groups, message="groups must be a real numeric array"
        )
        return original_ccat(
            a,
            b,
            c,
            groups=groups_value,
            targets=targets,
            administered=administered,
            theta0=theta0_value,
        )

    exposure_module.ccat_select = safe_ccat_select
