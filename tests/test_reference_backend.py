"""Contract tests for the explicit non-production fitting reference."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import FitConfig
from fast_mlsirm.fit import fit
from fast_mlsirm.reference import fit_reference


def test_reference_rejects_rust_only_plain_unidimensional_mmle() -> None:
    """The reference label must not disguise a Rust-only legacy MMLE path."""
    with pytest.raises(RuntimeError, match="NumPy reference is unavailable"):
        fit_reference(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0, 0], dtype=np.int64),
            FitConfig(model="ULS2PLM", estimator="mmle", max_iter=1),
        )


def test_direct_fit_cannot_activate_numpy_reference_with_hidden_flag() -> None:
    """Only the named reference API may activate the NumPy fitting authority."""
    with pytest.raises(RuntimeError, match="fit_reference"):
        fit(
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([0, 0], dtype=np.int64),
            FitConfig(backend="numpy", max_iter=1, n_restarts=1),
            _allow_reference_backend=True,
        )
