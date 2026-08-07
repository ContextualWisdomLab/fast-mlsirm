"""Parity and allocation-path contracts for the NumPy fit-statistics fallback."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

from fast_mlsirm import fitstats
from fast_mlsirm.types import MLSIRMParams


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, MLSIRMParams]:
    """Return sparse responses with a missing item and clipped probabilities."""

    random_generator = np.random.default_rng(42)
    n_persons = 41
    n_items = 6
    responses = random_generator.integers(0, 2, size=(n_persons, n_items)).astype(
        np.float64
    )
    observed = random_generator.random((n_persons, n_items)) > 0.23
    observed[:, 0] = False
    observed[0, 1] = True
    observed[0, 2] = True
    responses[0, 1] = 1.0
    responses[0, 2] = 0.0
    factor_id = np.zeros(n_items, dtype=np.int64)
    params = MLSIRMParams(
        theta=random_generator.normal(size=(n_persons, 1)),
        alpha=np.zeros(n_items, dtype=np.float64),
        b=np.array([0.0, 900.0, -900.0, 0.35, -0.4, 0.1]),
        xi=random_generator.normal(size=(n_persons, 2)),
        zeta=random_generator.normal(size=(n_items, 2)),
        tau=-0.25,
    )
    return responses, observed, factor_id, params


def _former_equations(
    responses: np.ndarray,
    observed: np.ndarray,
    factor_id: np.ndarray,
    params: MLSIRMParams,
    *,
    eps_distance: float,
) -> dict[str, np.ndarray]:
    """Evaluate the pre-change equations as an independent numerical oracle."""

    dimensions = np.asarray(factor_id, dtype=np.int64)
    slopes = np.exp(np.asarray(params.alpha, dtype=np.float64))
    eta = (
        slopes[None, :] * np.asarray(params.theta, dtype=np.float64)[:, dimensions]
        + np.asarray(params.b, dtype=np.float64)[None, :]
    )
    differences = (
        np.asarray(params.xi, dtype=np.float64)[:, None, :]
        - np.asarray(params.zeta, dtype=np.float64)[None, :, :]
    )
    distances = np.sqrt(
        eps_distance + np.sum(differences * differences, axis=2)
    )
    eta = eta - np.exp(float(params.tau)) * distances
    probabilities = np.clip(
        1.0 / (1.0 + np.exp(-np.clip(eta, -700.0, 700.0))),
        1e-12,
        1.0 - 1e-12,
    )
    variances = probabilities * (1.0 - probabilities)
    squared_residuals = (responses - probabilities) ** 2 * observed
    observation_counts = np.maximum(observed.sum(axis=0), 1)
    return {
        "outfit": (
            squared_residuals / variances * observed
        ).sum(axis=0)
        / observation_counts,
        "infit": squared_residuals.sum(axis=0)
        / np.maximum((variances * observed).sum(axis=0), 1e-12),
    }


def test_fallback_source_reuses_residual_buffer_without_numeric_mask_copy() -> None:
    """The source pins the bounded operations rather than a flaky peak heuristic."""

    source = inspect.getsource(fitstats.infit_outfit)

    assert "observed.astype" not in source
    assert "resid2 / v" not in source
    assert "v * observed" not in source
    assert "resid2 = np.subtract(y, p)" in source
    assert "np.square(resid2, out=resid2)" in source
    assert "np.multiply(resid2, observed, out=resid2)" in source
    assert "np.divide(resid2, v, out=resid2)" in source
    assert "np.sum(v, axis=0, where=observed)" in source


def test_fallback_matches_former_equations_with_sparse_and_boundary_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffer reuse preserves missingness and probability-clipping semantics."""

    responses, observed, factor_id, params = _fixture()
    epsilon = 1e-8
    expected = _former_equations(
        responses,
        observed,
        factor_id,
        params,
        eps_distance=epsilon,
    )
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)

    result = fitstats.infit_outfit(
        responses,
        factor_id,
        params,
        "mlsirm",
        mask=observed,
        eps_distance=epsilon,
    )

    np.testing.assert_allclose(
        result["infit"], expected["infit"], rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(
        result["outfit"], expected["outfit"], rtol=1e-13, atol=1e-13
    )
    assert result["infit"][0] == 0.0
    assert result["outfit"][0] == 0.0


def test_fallback_uses_in_place_division_and_boolean_where_reduction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback creates neither a numeric mask copy nor quotient buffer."""

    responses, observed, factor_id, params = _fixture()
    monkeypatch.setattr(fitstats, "_core_module", lambda: None)
    original_divide = np.divide
    original_sum = np.sum
    divide_calls: list[tuple[np.ndarray, np.ndarray, object]] = []
    sum_calls: list[dict[str, Any]] = []

    def recording_divide(
        numerator: np.ndarray,
        denominator: np.ndarray,
        *args: Any,
        **kwargs: Any,
    ) -> np.ndarray:
        """Record the quotient output identity before delegating to NumPy."""

        divide_calls.append((numerator, denominator, kwargs.get("out")))
        return original_divide(numerator, denominator, *args, **kwargs)

    def recording_sum(array: np.ndarray, *args: Any, **kwargs: Any) -> np.ndarray:
        """Record the Boolean where reduction before delegating to NumPy."""

        sum_calls.append(
            {
                "array": array,
                "axis": kwargs.get("axis"),
                "where": kwargs.get("where"),
            }
        )
        return original_sum(array, *args, **kwargs)

    monkeypatch.setattr(fitstats.np, "divide", recording_divide)
    monkeypatch.setattr(fitstats.np, "sum", recording_sum)

    fitstats.infit_outfit(
        responses,
        factor_id,
        params,
        "mlsirm",
        mask=observed,
    )

    assert len(divide_calls) == 1
    numerator, denominator, output = divide_calls[0]
    assert output is numerator
    assert numerator.shape == denominator.shape == observed.shape
    assert len(sum_calls) == 1
    assert sum_calls[0]["axis"] == 0
    assert sum_calls[0]["where"] is observed
