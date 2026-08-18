"""Concrete NumPy scalar compatibility for hardened rotation controls."""

from __future__ import annotations

import numpy as np

from fast_mlsirm.rotation import (
    rotate_factor_loadings,
    rotation_criterion_value_gradient,
)


def test_numpy_rotation_controls_preserve_native_execution() -> None:
    """Trusted NumPy scalars cross the same Rust-owned rotation path as built-ins."""

    loadings = np.asarray(
        [[0.8, 0.2], [0.7, 0.1], [0.1, 0.8], [0.2, 0.7]], dtype=np.float64
    )
    result = rotate_factor_loadings(
        loadings,
        "varimax",
        normalize=np.bool_(False),
        n_starts=np.int64(1),
        seed=np.uint64(1),
        max_iter=np.int64(20),
        tolerance=np.float64(1e-5),
        function_window=np.int64(2),
        max_line_search=np.int64(5),
        basin_tolerance=np.float64(1e-8),
        max_threads=np.int64(0),
    )
    assert result.criterion == "varimax"
    assert result.mode == "orthogonal"

    value, gradient = rotation_criterion_value_gradient(
        loadings,
        "cf",
        kappa=np.float64(0.3),
    )
    assert np.isfinite(value)
    assert gradient.shape == loadings.shape


def test_numpy_optional_integer_control_preserves_native_execution() -> None:
    """A genuine NumPy integer remains valid for the Rust simplimax criterion."""

    loadings = np.asarray(
        [[0.8, 0.2], [0.7, 0.1], [0.1, 0.8], [0.2, 0.7]], dtype=np.float64
    )
    value, gradient = rotation_criterion_value_gradient(
        loadings,
        "simplimax",
        simplimax_zeros=np.int64(2),
    )
    assert np.isfinite(value)
    assert gradient.shape == loadings.shape
