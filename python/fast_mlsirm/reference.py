"""Explicit non-production reference fitting for Rust parity validation."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import FitConfig
from .fit import fit
from .types import FitResult


def fit_reference(
    responses: np.ndarray,
    factor_id: np.ndarray,
    config: FitConfig | None = None,
    mask: np.ndarray | None = None,
    group_id: np.ndarray | None = None,
    cluster_id: np.ndarray | None = None,
    anchors: dict | None = None,
    covariate: dict | None = None,
) -> FitResult:
    """Run the NumPy parity implementation behind an explicit reference API.

    This path is for tests, numerical parity, and research inspection. It is
    not selected by ``FitConfig`` defaults or by the production CLI backend
    selector. Production callers must use :func:`fast_mlsirm.fit` so missing
    Rust capability fails closed.
    """
    reference_config = replace(config or FitConfig(), backend="numpy")
    return fit(
        responses,
        factor_id,
        reference_config,
        mask=mask,
        group_id=group_id,
        cluster_id=cluster_id,
        anchors=anchors,
        covariate=covariate,
        _allow_reference_backend=True,
    )
