"""Numerical and allocation contracts for NumPy infit/outfit fallback."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import fast_mlsirm.fitstats as fitstats
from fast_mlsirm.types import MLSIRMParams


def _mirt_fixture() -> tuple[np.ndarray, np.ndarray, MLSIRMParams]:
    """Return deterministic responses, factor IDs, and MIRT parameters."""
    rng = np.random.default_rng(20260807)
    n_persons, n_items = 32, 5
    theta = np.linspace(-2.0, 2.0, n_persons, dtype=np.float64)[:, None]
    alpha = np.log(np.array([0.7, 1.0, 1.3, 1.6, 0.9], dtype=np.float64))
    intercept = np.array([-1.5, -0.4, 0.0, 0.8, 1.7], dtype=np.float64)
    eta = np.exp(alpha)[None, :] * theta + intercept[None, :]
    probability = 1.0 / (1.0 + np.exp(-eta))
    responses = (rng.random(probability.shape) < probability).astype(np.float64)
    params = MLSIRMParams(
        theta=theta,
        alpha=alpha,
        b=intercept,
        xi=np.zeros((n_persons, 1), dtype=np.float64),
        zeta=np.zeros((n_items, 1), dtype=np.float64),
        tau=-30.0,
    )
    return responses, np.zeros(n_items, dtype=np.int64), params


def _reference_statistics(
    responses: np.ndarray,
    observed: np.ndarray,
    params: MLSIRMParams,
) -> dict[str, np.ndarray]:
    """Return the governed pre-change MIRT infit and outfit equations."""
    eta = np.exp(params.alpha)[None, :] * params.theta + params.b[None, :]
    probability = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        1e-12,
        1.0 - 1e-12,
    )
    variance = probability * (1.0 - probability)
    residual_squared = (responses - probability) ** 2 * observed
    observed_count = np.maximum(observed.sum(axis=0), 1)
    return {
        "outfit": (residual_squared / variance * observed).sum(axis=0)
        / observed_count,
        "infit": residual_squared.sum(axis=0)
        / np.maximum((variance * observed).sum(axis=0), 1e-12),
    }


def test_numpy_fallback_matches_prechange_equations_with_missingness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In-place reductions preserve governed values for sparse observations."""
    responses, factor_id, params = _mirt_fixture()
    observed = np.ones_like(responses, dtype=bool)
    observed[::3, 1] = False
    observed[1::4, 3] = False
    observed[:, 4] = False
    expected = _reference_statistics(responses, observed, params)

    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    actual = fitstats.infit_outfit(
        responses,
        factor_id,
        params,
        "MIRT",
        mask=observed,
    )

    np.testing.assert_array_equal(actual["infit"], expected["infit"])
    np.testing.assert_array_equal(actual["outfit"], expected["outfit"])
    assert np.all(np.isfinite(actual["infit"]))
    assert np.all(np.isfinite(actual["outfit"]))
    assert actual["infit"][4] == 0.0
    assert actual["outfit"][4] == 0.0


def test_numpy_fallback_remains_finite_for_clipped_extreme_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The reusable residual buffer remains finite at both probability clips."""
    responses, factor_id, params = _mirt_fixture()
    params.b = np.array([-900.0, -700.0, 0.0, 700.0, 900.0], dtype=np.float64)
    observed = np.ones_like(responses, dtype=bool)
    expected = _reference_statistics(responses, observed, params)

    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    actual = fitstats.infit_outfit(
        responses,
        factor_id,
        params,
        "MIRT",
        mask=observed,
    )

    np.testing.assert_allclose(actual["infit"], expected["infit"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(actual["outfit"], expected["outfit"], rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(actual["infit"]))
    assert np.all(np.isfinite(actual["outfit"]))


def test_fallback_source_reuses_residual_buffer_without_numeric_mask_copy() -> None:
    """The allocation correction cannot regress to full-size temporary arrays."""
    source = inspect.getsource(fitstats.infit_outfit)
    assert "observed.astype" not in source
    assert "np.sum(v, axis=0, where=observed)" in source
    assert "np.divide(resid2, v, out=resid2)" in source
    assert "np.subtract(y, p)" in source
    assert "np.square(resid2, out=resid2)" in source
    assert "np.multiply(resid2, observed, out=resid2)" in source
    assert "resid2 / v" not in source
    assert "v * observed" not in source
