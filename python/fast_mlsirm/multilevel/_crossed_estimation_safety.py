"""Callback-free evidence admission for crossed person-effect estimation.

The production estimator and all result-affecting arithmetic remain Rust-owned.
This adapter only establishes inert numeric evidence before NumPy marshalling.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType

import numpy as np


_TRUSTED_NUMPY_INTEGER_TYPES = (
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
_TRUSTED_NUMERIC_SCALAR_TYPES = frozenset(
    {
        bool,
        int,
        float,
        complex,
        np.bool_,
        *_TRUSTED_NUMPY_INTEGER_TYPES,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
        np.complex64,
        np.complex128,
        np.clongdouble,
    }
)
_INSTALL_MARKER = "__fast_mlsirm_crossed_evidence_safety__"


def _trusted_numeric_storage(value: object, name: str) -> np.ndarray:
    """Materialize only exact numeric arrays or inert built-in sequences."""
    if type(value) is np.ndarray:
        raw = value
    elif type(value) in (list, tuple):
        stack: list[tuple[object, bool]] = [(value, False)]
        active_container_ids: set[int] = set()
        while stack:
            current, leaving = stack.pop()
            if leaving:
                active_container_ids.remove(id(current))
                continue
            if type(current) in (list, tuple):
                identity = id(current)
                if identity in active_container_ids:
                    raise ValueError(f"{name} must be a numeric array")
                active_container_ids.add(identity)
                stack.append((current, True))
                for index in range(len(current) - 1, -1, -1):
                    stack.append((current[index], False))
                continue
            if type(current) is np.ndarray:
                if current.dtype.kind not in ("b", "i", "u", "f", "c"):
                    raise ValueError(f"{name} must be a numeric array")
                continue
            if type(current) not in _TRUSTED_NUMERIC_SCALAR_TYPES:
                raise ValueError(f"{name} must be a numeric array")
        try:
            raw = np.asarray(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{name} must be a numeric array") from None
    else:
        raise ValueError(f"{name} must be a numeric array")

    if raw.dtype.kind not in ("b", "i", "u", "f", "c"):
        raise ValueError(f"{name} must be a numeric array")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        return np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} could not be converted safely") from None


def install(module: ModuleType) -> None:
    """Install one idempotent scientific-evidence guard on the estimator."""
    current = module.estimate_crossed_person_effects
    if getattr(current, _INSTALL_MARKER, False):
        return

    @wraps(current)
    def guarded_estimate_crossed_person_effects(
        responses: object,
        design: object,
        *,
        item_intercepts: object,
        item_slopes: object | None = None,
        person_offsets: object | None = None,
        prior_scale: object = 1.0,
        max_iter: object = 50,
        tol: object = 1e-8,
        worker_count: object = 1,
        device: object = "auto",
    ):
        trusted_responses = _trusted_numeric_storage(responses, "responses")
        trusted_intercepts = _trusted_numeric_storage(
            item_intercepts, "item_intercepts"
        )
        trusted_slopes = (
            None
            if item_slopes is None
            else _trusted_numeric_storage(item_slopes, "item_slopes")
        )
        trusted_offsets = (
            None
            if person_offsets is None
            else _trusted_numeric_storage(person_offsets, "person_offsets")
        )
        return current(
            trusted_responses,
            design,
            item_intercepts=trusted_intercepts,
            item_slopes=trusted_slopes,
            person_offsets=trusted_offsets,
            prior_scale=prior_scale,
            max_iter=max_iter,
            tol=tol,
            worker_count=worker_count,
            device=device,
        )

    setattr(guarded_estimate_crossed_person_effects, _INSTALL_MARKER, True)
    module.estimate_crossed_person_effects = guarded_estimate_crossed_person_effects


__all__ = ["install"]
