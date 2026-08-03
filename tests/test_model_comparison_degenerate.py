"""Regression tests for decision-safe zero-variance Vuong comparisons."""

from __future__ import annotations

import math

import pytest

from fast_mlsirm.model_comparison import ComparisonStatus, compare_nonnested_models


def test_constant_casewise_difference_returns_degenerate_decision_result():
    """Exact observational equivalence is a reportable status, not an exception."""
    result = compare_nonnested_models(
        [1.0, 1.5, 2.0, 2.5],
        [0.75, 1.25, 1.75, 2.25],
        2,
        2,
        model_a="distance",
        model_b="bifactor",
    )

    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert result.preferred_model is None
    assert math.isnan(result.raw_mean_loglik_difference)
    assert result.omega == pytest.approx(0.0)
    assert not result.variance_positive
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)
