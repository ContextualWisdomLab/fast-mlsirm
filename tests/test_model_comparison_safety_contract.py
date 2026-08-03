"""Safety contract for relation-aware Rust-backed Vuong selection summaries."""

from __future__ import annotations

import itertools
import math

import pytest

import fast_mlsirm.model_comparison as comparison_module
from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    VuongVarianceDegenerateError,
    compare_nonnested_models,
)


def _sentinel() -> dict[str, float]:
    """Return a stable Rust-kernel-shaped selection-statistic result."""
    return {
        "z": 2.5,
        "p_two_sided": 0.01,
        "omega": 0.4,
        "mean_diff": 0.3,
    }


def test_default_relation_is_unknown_and_never_selects(monkeypatch):
    """Omitting mathematical relation metadata must fail closed."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models([1.0, 2.0], [0.0, 1.0], 2, 2)

    assert result.relation is ModelRelation.UNKNOWN
    assert result.status is ComparisonStatus.UNKNOWN_RELATION
    assert result.preferred_model is None


def test_strict_nonnested_requires_formal_distinguishability(monkeypatch):
    """Positive sample variance cannot substitute for Vuong's formal first stage."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models(
        [1.0, 2.0],
        [0.0, 1.0],
        2,
        2,
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )

    assert result.status is ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)
    assert result.raw_z == pytest.approx(2.5)
    assert result.raw_p_two_sided == pytest.approx(0.01)


def test_casewise_iterables_are_bounded(monkeypatch):
    """Infinite and oversized iterables terminate at the documented work bound."""
    monkeypatch.setattr(comparison_module, "MAX_CASEWISE_VALUES", 3)
    with pytest.raises(ValueError, match="at most 3"):
        compare_nonnested_models(
            itertools.repeat(1.0),
            [0.0, 1.0],
            1,
            1,
        )


@pytest.mark.parametrize(
    "label",
    ["x" * 129, "safe\nforged", "safe\x00forged"],
)
def test_labels_are_bounded_and_control_free(monkeypatch, label):
    """Audit labels cannot create oversized records or control-character injection."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    with pytest.raises(ValueError, match="model_a"):
        compare_nonnested_models(
            [1.0, 2.0],
            [0.0, 1.0],
            1,
            1,
            model_a=label,
        )


@pytest.mark.parametrize("parameter_count", [True, 1.5])
def test_parameter_counts_reject_booleans_and_fractional_values(parameter_count):
    """Parameter-count coercion cannot silently change the statistical penalty."""
    with pytest.raises(ValueError, match="k_a"):
        compare_nonnested_models(
            [1.0, 2.0],
            [0.0, 1.0],
            parameter_count,
            1,
        )


def test_legacy_kernel_zero_variance_becomes_a_typed_boundary_signal(monkeypatch):
    """Only the adapter, not decision logic, knows the legacy Rust error text."""
    def legacy_kernel(*_args, **_kwargs):
        raise ValueError("vuong_nonnested: omega^2 = 0")

    monkeypatch.setattr(comparison_module, "vuong_nonnested", legacy_kernel)
    with pytest.raises(VuongVarianceDegenerateError):
        comparison_module._run_vuong(
            (1.0, 2.0),
            (0.0, 1.0),
            1,
            1,
            bic_correction=True,
        )


def test_typed_degenerate_signal_is_consumed(monkeypatch):
    """Exact zero variance returns a reportable status instead of string matching."""
    def fail(*_args, **_kwargs):
        raise VuongVarianceDegenerateError("zero variance")

    monkeypatch.setattr(comparison_module, "_run_vuong", fail)
    result = compare_nonnested_models(
        [1.0, 2.0],
        [0.0, 1.0],
        1,
        1,
        relation=ModelRelation.STRICTLY_NON_NESTED,
    )

    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert result.omega == 0.0
    assert result.preferred_model is None
