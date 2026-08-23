"""Callback-safe evidence-container admission for cognitive diagnosis APIs."""

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
_RESPONSE_ERROR = "responses must be a trusted NumPy array or built-in sequence"
_Q_GUARD_MARKER = "__fast_mlsirm_q_matrix_container_guard__"


def _reject_untrusted_numeric_container(value: object, *, error: str) -> None:
    """Reject callback-bearing providers while preserving inert arrays/sequence DAGs."""

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
                raise ValueError(error)
            active_container_ids.add(item_id)
            stack.append((item, True))
            stack.extend((child, False) for child in reversed(item))
            continue

        if any(
            item_type is scalar_type
            for scalar_type in _TRUSTED_RESPONSE_SCALAR_TYPES
        ):
            continue

        raise ValueError(error)


def _install_q_matrix_guard(module: ModuleType) -> None:
    """Wrap the canonical Q-matrix validator with callback-free container admission."""

    current = module._validate_q_matrix_input
    if getattr(current, _Q_GUARD_MARKER, False):
        return

    def validate_q_matrix_input(value: object, name: str, n_items: int):
        _reject_untrusted_numeric_container(
            value,
            error=f"{name} must be a trusted NumPy array or built-in sequence",
        )
        return current(value, name, n_items)

    setattr(validate_q_matrix_input, _Q_GUARD_MARKER, True)
    module._validate_q_matrix_input = validate_q_matrix_input


def _reject_untrusted_response_container(value: object) -> None:
    """Reject callback-bearing response providers and repair reload-time Q guarding.

    The trusted transport vocabulary is deliberately explicit: one exact NumPy
    array, or an exact built-in list/tuple tree whose leaves are package-known
    numeric scalar identities or exact NumPy arrays. Sequence traversal tracks
    only active ancestors, so true cycles fail closed while repeated/shared rows
    remain valid. No caller-defined ``__array__``, numeric, or container protocol
    is invoked during this admission pass.

    Direct ``fast_mlsirm.cdm`` reloads replace module globals without rerunning
    package initialization. Because every public CDM calibration path admits
    responses before Q-matrix evidence, repairing the Q validator here preserves
    the same callback-safe boundary after such a reload.
    """

    from . import cdm as module

    _install_q_matrix_guard(module)
    _reject_untrusted_numeric_container(value, error=_RESPONSE_ERROR)


def install(module: ModuleType) -> None:
    """Install callback-safe response and Q-matrix evidence guards."""

    _install_q_matrix_guard(module)
    module._reject_untrusted_response_container = _reject_untrusted_response_container
