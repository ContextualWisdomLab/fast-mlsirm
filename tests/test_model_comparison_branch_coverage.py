"""Focused branch coverage for model-comparison normalization and guards."""

from __future__ import annotations

import math

import fast_mlsirm.model_comparison as comparison_module
from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    compare_nonnested_models,
)


def _kernel_result(*, omega: float = 0.4) -> dict[str, float]:
    """Return a valid low-level result with a configurable variance scale."""
    return {
        "z": 1.5,
        "p_two_sided": 0.2,
        "omega": omega,
        "mean_diff": 0.1,
    }


def test_valid_string_relation_is_normalized(monkeypatch):
    """String callers reach the successful enum-conversion branch."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation="strictly_non_nested",
    )
    assert result.relation is ModelRelation.STRICTLY_NON_NESTED
    assert result.status is ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE


def test_nonfinite_omega_fails_closed(monkeypatch):
    """A non-finite Rust variance scale cannot produce preference inference."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(omega=float("nan")),
    )
    result = compare_nonnested_models([0.0, 0.1], [0.2, 0.3], 1, 1)
    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert not result.variance_positive
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)
