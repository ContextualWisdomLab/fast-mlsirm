"""Public production fitting surface without reference-backend authority."""

from __future__ import annotations

import numpy as np

from .backend import normalize_production_backend
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
    # Enforce the production backend contract at the exported entry point.
    # The internal reference scope is deliberately irrelevant here: callers
    # that import it cannot turn the public fit surface into a NumPy path.
    normalize_production_backend(production_config.backend)
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


# Preserve the established detailed public documentation without exposing the
# internal reference-authority keyword through ``inspect.signature``.
fit.__doc__ = _fit_internal.__doc__
