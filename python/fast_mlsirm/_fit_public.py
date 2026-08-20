"""Public production fitting surface without reference-backend authority."""

from __future__ import annotations

import numpy as np

from .backend import _REFERENCE_BACKEND_ACTIVE, normalize_production_backend
from .config import FitConfig
from .fit import fit as _fit_internal
from .types import FitResult


def fit(
    responses: np.ndarray,
    factor_id: np.ndarray,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Fit a latent-space model through the production Rust-owned backend."""
    production_config = config or FitConfig()
    # Enforce the production backend contract before entering the internal fit
    # implementation, then explicitly clear any inherited reference authority.
    # A caller that imports _reference_backend_scope therefore cannot turn this
    # exported production surface into a NumPy fitting path.
    normalize_production_backend(production_config.backend)
    token = _REFERENCE_BACKEND_ACTIVE.set(False)
    try:
        return _fit_internal(
            responses,
            factor_id,
            production_config,
            mask=mask,
            group_id=group_id,
            cluster_id=cluster_id,
            anchors=anchors,
            covariate=covariate,
        )
    finally:
        _REFERENCE_BACKEND_ACTIVE.reset(token)


# Preserve the established detailed public documentation without exposing the
# internal reference-authority keyword through ``inspect.signature``.
fit.__doc__ = _fit_internal.__doc__
