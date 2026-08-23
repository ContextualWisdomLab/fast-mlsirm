"""Callback-safe response-container admission for cognitive diagnosis APIs."""

from __future__ import annotations

from types import ModuleType

import numpy as np


_NUMPY_INTEGER_SCALAR_TYPES = (
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
_NUMPY_FLOAT_SCALAR_TYPES = (np.float16, np.float32, np.float64, np.longdouble)
_NUMPY_COMPLEX_SCALAR_TYPES = (np.complex64, np.complex128, np.clongdouble)
_TRUSTED_RESPONSE_SCALAR_TYPES = (
    bool,
    int,
    float,
    complex,
    np.bool_,
    *_NUMPY_INTEGER_SCALAR_TYPES,
    *_NUMPY_FLOAT_SCALAR_TYPES,
    *_NUMPY_COMPLEX_SCALAR_TYPES,
)
_ERROR = "responses must be a trusted NumPy array or built-in sequence"


def _reject_untrusted_response_container(value: object) -> None:
    """Reject callback-bearing providers while preserving inert array rows/DAGs.

    The trusted transport vocabulary is deliberately explicit: one exact NumPy
    array, or an exact built-in list/tuple tree whose leaves are package-known
    numeric scalar identities or exact NumPy arrays.  Sequence traversal tracks
    only active ancestors, so true cycles fail closed while repeated/shared rows
    remain valid.  No caller-defined ``__array__``, numeric, or container protocol
    is invoked during this admission pass.
    """

    stack: list[tuple[object, bool]] = [(value, False)]
    active_container_ids: set[int] = set()

    while stack:
        item, leaving = stack.pop()
        item_type = type(item)

        if leaving:
            active_container_ids.remove(id(item))
            continue

        if item_type is np.ndarray:
            continue

        if item_type is list or item_type is tuple:
            item_id = id(item)
            if item_id in active_container_ids:
                raise ValueError(_ERROR)
            active_container_ids.add(item_id)
            stack.append((item, True))
            stack.extend((child, False) for child in reversed(item))
            continue

        if any(item_type is scalar_type for scalar_type in _TRUSTED_RESPONSE_SCALAR_TYPES):
            continue

        raise ValueError(_ERROR)


def install(module: ModuleType) -> None:
    """Install the hardened container guard on the existing CDM module."""

    module._reject_untrusted_response_container = _reject_untrusted_response_container
