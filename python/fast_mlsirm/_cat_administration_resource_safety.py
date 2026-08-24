"""Resource preflight for dichotomous CAT administration evidence.

The CAT numerical owner remains Rust.  This module only rejects structurally
impossible partial administrations from inert container metadata before the
existing CAT validator performs value-wise scans or dense dtype marshalling.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

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


def _reject_over_bank_administration(bank: Any, administered: object) -> None:
    """Reject exact-carrier administrations that cannot be unique in the bank."""
    administered_length = _vector_length(administered)
    if administered_length is None:
        return
    n_items = int(np.asarray(bank.b).shape[0])
    if administered_length > n_items:
        raise ValueError(_LENGTH_ERROR)


def install(cat_module: ModuleType) -> None:
    """Install fail-fast administration resource preflight idempotently."""
    original_validate = cat_module._validate_administration
    if bool(getattr(original_validate, _MARKER, False)):
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
            return original_validate(bank, factor_id, administered, responses)

        responses_length = _vector_length(responses)
        if responses_length is None:
            return original_validate(bank, factor_id, administered, responses)

        if administered_length != responses_length:
            raise ValueError(_SHAPE_ERROR)

        # A validated CAT administration requires unique item identities, so more
        # observations than bank items is structurally impossible on this surface.
        # `ability_standard_error` is intentionally not wrapped here: its historical
        # contract treats `administered` as a set-valued mask and applies np.unique,
        # so duplicate-laden or multidimensional evidence can legitimately have a
        # raw logical size above the bank item count before deduplication.
        _reject_over_bank_administration(bank, administered)
        return original_validate(bank, factor_id, administered, responses)

    setattr(safe_validate_administration, _MARKER, True)
    cat_module._validate_administration = safe_validate_administration
