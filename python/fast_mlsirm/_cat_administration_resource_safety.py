"""Resource preflight for dichotomous CAT administration evidence.

The CAT numerical owner remains Rust.  This module only rejects structurally
impossible partial administrations from inert container metadata before the
existing CAT validator performs value-wise scans or dense dtype marshalling.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Callable

import numpy as np

_SHAPE_ERROR = "administered and responses must be 1D arrays of equal length"
_LENGTH_ERROR = "administration length cannot exceed item bank size"
_MARKER = "__fast_mlsirm_cat_administration_resource_safe__"


def _vector_length(value: object) -> int | None:
    """Return inert one-dimensional length, or ``None`` for unsupported carriers."""
    if type(value) is np.ndarray:
        if value.ndim != 1:
            raise ValueError(_SHAPE_ERROR)
        return int(value.shape[0])
    if type(value) in (list, tuple):
        return len(value)
    return None


def install(cat_module: ModuleType) -> None:
    """Install fail-fast administration shape/length preflight idempotently."""
    original = cat_module._validate_administration
    if getattr(original, _MARKER, False):
        return

    def safe_validate_administration(
        bank: Any,
        factor_id: np.ndarray,
        administered: object,
        responses: object,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Preserve the existing callback/type-error contract for unsupported
        # top-level providers by delegating unchanged unless both carriers expose
        # inert exact-container metadata.
        administered_length = _vector_length(administered)
        if administered_length is None:
            return original(bank, factor_id, administered, responses)

        responses_length = _vector_length(responses)
        if responses_length is None:
            return original(bank, factor_id, administered, responses)

        if administered_length != responses_length:
            raise ValueError(_SHAPE_ERROR)

        # A valid partial administration cannot contain more observations than
        # the bank has items because administered identities must be unique.
        # Reject that impossible shape before value-wise range/uniqueness scans
        # or dense int64/float64 conversion of the caller evidence.
        n_items = int(np.asarray(bank.b).shape[0])
        if administered_length > n_items:
            raise ValueError(_LENGTH_ERROR)

        return original(bank, factor_id, administered, responses)

    setattr(safe_validate_administration, _MARKER, True)
    cat_module._validate_administration = safe_validate_administration
