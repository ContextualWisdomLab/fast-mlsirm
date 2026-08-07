"""Parity and allocation contracts for NumPy fallback infit/outfit reductions."""

from __future__ import annotations

import inspect
import tracemalloc

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.types import MLSIRMParams


def _fallback_params(
    *,
    n_persons: int,
    n_items: int,
    latent_dim: int = 2,
    rng: np.random.Generator,
) -> MLSIRMParams:
    """Return deterministic finite parameters for a fallback diagnostic call."""
    return MLSIRMParams(
        theta=rng.normal(size=(n_persons, 1)),
        alpha=rng.normal(scale=0.1, size=n_items),
        b=rng.normal(scale=0.4, size=n_items),
        xi=rng.normal(size=(n_persons, latent_dim)),
        zeta=rng.normal(size=(n_items, latent_dim)),
        tau=-0.2,
    )


def test_fallback_source_reuses_one_residual_buffer_without_mask_copy() -> None:
    """The reviewed source permanently excludes all full residual temporaries."""
    source = inspect.getsource(fitstats.infit_outfit)

    assert "observed.astype" not in source
    assert "(y - p) ** 2 * observed" not in source
    assert "np.subtract(y, p)" in source
    assert "np.square(resid2, out=resid2)" in source
    assert "np.multiply(resid2, observed, out=resid2)" in source
    assert "np.divide(resid2, v, out=resid2)" in source
    assert "np.sum(v, axis=0, where=observed)" in source


def test_infit_outfit_matches_independent_equations_at_missing_and_boundary_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The in-place path preserves both mean-square equations exactly."""
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: None)
    rng = np.random.default_rng(570)
    n_persons, n_items = 12, 4
    y = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.float64)
    observed = rng.random(y.shape) > 0.25
    observed[:, 0] = False
    observed[0, 1:] = True
    params = _fallback_params(
        n_persons=n_persons,
        n_items=n_items,
        latent_dim=2,
        rng=rng,
    )
    params.b[1] = -50.0
    params.b[2] = 50.0
    factor_id = np.zeros(n_items, dtype=np.int64)

    actual = fitstats.infit_outfit(
        y,
        factor_id,
        params,
        "MIRT",
        mask=observed,
    )

    y_filled = np.where(observed, y, 0.0)
    discrimination = np.exp(params.alpha)
    eta = discrimination[None, :] * params.theta[:, factor_id] + params.b[None, :]
    probability = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        1e-12,
        1.0 - 1e-12,
    )
    variance = probability * (1.0 - probability)
    residual_square = (y_filled - probability) ** 2 * observed
    observation_count = np.maximum(observed.sum(axis=0), 1)
    expected_outfit = np.sum(residual_square / variance, axis=0) / observation_count
    expected_infit = np.sum(residual_square, axis=0) / np.maximum(
        np.sum(variance, axis=0, where=observed),
        1e-12,
    )

    np.testing.assert_allclose(actual["outfit"], expected_outfit, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(actual["infit"], expected_infit, rtol=1e-13, atol=1e-13)
    assert actual["outfit"][0] == 0.0
    assert actual["infit"][0] == 0.0


def test_infit_outfit_allocation_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A realistic sparse spatial call stays below eight person-item buffers."""
    monkeypatch.setattr("fast_mlsirm.fitstats._core_module", lambda: None)
    rng = np.random.default_rng(42)
    n_persons, n_items = 500, 100
    y = rng.integers(0, 2, size=(n_persons, n_items)).astype(np.float64)
    observed = rng.random(y.shape) > 0.1
    observed[:, 0] = False
    params = _fallback_params(
        n_persons=n_persons,
        n_items=n_items,
        latent_dim=2,
        rng=rng,
    )
    factor_id = np.zeros(n_items, dtype=np.int64)

    fitstats.infit_outfit(y, factor_id, params, "MLSIRM", mask=observed)
    tracemalloc.start()
    result = fitstats.infit_outfit(
        y,
        factor_id,
        params,
        "MLSIRM",
        mask=observed,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert peak < n_persons * n_items * np.dtype(np.float64).itemsize * 8
    assert result["outfit"][0] == 0.0
    assert result["infit"][0] == 0.0
    assert np.all(np.isfinite(result["outfit"][1:]))
    assert np.all(np.isfinite(result["infit"][1:]))
