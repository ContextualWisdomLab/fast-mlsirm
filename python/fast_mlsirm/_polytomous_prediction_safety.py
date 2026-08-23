"""Resource guard for public polytomous prediction grids.

The public GRM/GPCM prediction surface returns dense persons x items x categories
probabilities plus persons x items expected scores.  This adapter rejects an
oversized logical output grid before compiled-core discovery/allocation while
leaving all probability arithmetic in the Rust implementation.
"""

from __future__ import annotations

from functools import wraps
from types import ModuleType
from typing import Any

import numpy as np

MAX_POLYTOMOUS_PREDICTION_CELLS = 20_000_000
_GUARD_MARKER = "_fast_mlsirm_polytomous_prediction_resource_guard"


def _trusted_length(value: object) -> int | None:
    """Return a callback-free logical vector length when the container is trusted."""

    value_type = type(value)
    if value_type is np.ndarray:
        return int(value.size) if value.ndim == 1 else None
    if value_type is list or value_type is tuple:
        return len(value)
    return None


def _prediction_cells(module: ModuleType, fit: object, theta: object) -> int | None:
    """Return the dense probability-cell count when shape evidence is inert."""

    if type(fit) is not module.PolytomousFit:
        return None
    theta_count = _trusted_length(theta)
    if theta_count is None or theta_count == 0:
        return None

    slope = fit.slope
    cat_params = fit.cat_params
    if type(slope) is not np.ndarray or type(cat_params) is not np.ndarray:
        return None
    if slope.ndim != 1 or slope.size == 0:
        return None
    if (
        cat_params.ndim != 2
        or cat_params.shape[0] != slope.size
        or cat_params.shape[1] < 1
    ):
        return None

    return theta_count * int(slope.size) * int(cat_params.shape[1] + 1)


def install(module: ModuleType) -> None:
    """Install an idempotent pre-dispatch resource guard on prediction calls."""

    original = module._polytomous_predictions
    if getattr(original, _GUARD_MARKER, False):
        return

    @wraps(original)
    def guarded_predictions(fit: Any, theta: Any):
        cells = _prediction_cells(module, fit, theta)
        if cells is not None and cells > MAX_POLYTOMOUS_PREDICTION_CELLS:
            raise ValueError(
                f"{MAX_POLYTOMOUS_PREDICTION_CELLS:,}-cell polytomous prediction limit exceeded"
            )
        return original(fit, theta)

    setattr(guarded_predictions, _GUARD_MARKER, True)
    module._polytomous_predictions = guarded_predictions
