"""Public production fitting surface without reference-backend authority."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .backend import _REFERENCE_BACKEND_ACTIVE, resolve_backend
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
    if config is not None and type(config) is not FitConfig:
        raise ValueError("config must be a FitConfig or None")
    requested_config = FitConfig() if config is None else config
    # Resolve the production numerical owner before entering the internal fit
    # implementation. Besides failing closed when Rust is unavailable, passing
    # the concrete backend downstream prevents successful ``auto`` runs from
    # leaking the unresolved selector into FitResult metadata.
    concrete_backend = resolve_backend(requested_config.backend)
    production_config = replace(requested_config, backend=concrete_backend)

    # Explicitly clear any inherited reference authority. A caller that imports
    # _reference_backend_scope therefore cannot turn this exported production
    # surface into a NumPy fitting path, even with a production backend name.
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
