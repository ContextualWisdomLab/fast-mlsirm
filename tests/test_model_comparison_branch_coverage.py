"""Focused branch coverage for model-comparison normalization and guards."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fast_mlsirm.model_comparison as comparison_module
from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    VuongVarianceDegenerateError,
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
    assert result.status is ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST


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


def test_numpy_integer_counts_are_retained_and_numpy_boolean_is_rejected(monkeypatch):
    """NumPy integers remain valid counts while NumPy booleans cannot become one."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _kernel_result(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        np.int64(2),
        np.int64(1),
    )
    assert result.k_a == 2
    assert result.k_b == 1

    with pytest.raises(ValueError, match="k_a"):
        compare_nonnested_models(
            [0.0, 0.1],
            [0.2, 0.3],
            np.bool_(True),
            1,
        )


def test_unrelated_kernel_validation_error_is_not_reclassified(monkeypatch):
    """Only the dedicated legacy zero-variance text becomes the typed signal."""
    def invalid_kernel(*_args, **_kwargs):
        raise ValueError("some other low-level validation error")

    monkeypatch.setattr(comparison_module, "vuong_nonnested", invalid_kernel)
    with pytest.raises(ValueError, match="some other"):
        comparison_module._run_vuong(
            (0.0, 0.1),
            (0.2, 0.3),
            1,
            1,
            bic_correction=True,
        )


def test_typed_variance_error_records_no_raw_statistic(monkeypatch):
    """A typed exact-zero signal takes the dedicated degenerate-result path."""
    def zero_variance(*_args, **_kwargs):
        raise VuongVarianceDegenerateError("zero variance")

    monkeypatch.setattr(comparison_module, "_run_vuong", zero_variance)
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation=ModelRelation.OVERLAPPING,
    )
    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert math.isnan(result.raw_z)
    assert math.isnan(result.raw_p_two_sided)
