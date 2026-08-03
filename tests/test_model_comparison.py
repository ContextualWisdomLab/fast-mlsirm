"""Tests for the decision-safe Rust-backed model-comparison wrapper."""

from __future__ import annotations

import math

import pytest

import fast_mlsirm.model_comparison as comparison_module
from fast_mlsirm.fitstats import vuong_nonnested
from fast_mlsirm.model_comparison import (
    ComparisonStatus,
    ModelRelation,
    compare_nonnested_models,
)


def _sentinel(*, z: float = 2.5, p: float = 0.01, omega: float = 0.4):
    """Return a stable Rust-kernel-shaped result dictionary."""
    return {
        "z": z,
        "p_two_sided": p,
        "omega": omega,
        "mean_diff": 0.3,
    }


def test_delegates_all_statistics_to_rust_kernel(monkeypatch):
    """The orchestration layer must not reproduce the statistic in Python."""
    seen = {}

    def fake_kernel(a, b, k_a, k_b, *, bic_correction):
        seen.update(a=a, b=b, k_a=k_a, k_b=k_b, bic=bic_correction)
        return _sentinel()

    monkeypatch.setattr(comparison_module, "vuong_nonnested", fake_kernel)
    result = compare_nonnested_models(
        (1.0, 2.0, 3.0),
        (0.5, 1.5, 2.0),
        4,
        2,
        model_a="MLS2PLM",
        model_b="MIRT",
        bic_correction=False,
    )

    assert seen == {
        "a": (1.0, 2.0, 3.0),
        "b": (0.5, 1.5, 2.0),
        "k_a": 4,
        "k_b": 2,
        "bic": False,
    }
    assert result.status is ComparisonStatus.MODEL_A_PREFERRED
    assert result.preferred_model == "MLS2PLM"
    assert result.raw_mean_loglik_difference == pytest.approx(0.3)
    assert result.omega == pytest.approx(0.4)


def test_negative_sign_prefers_model_b(monkeypatch):
    """A significant negative Rust z value selects the supplied B label."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(z=-3.0, p=0.002),
    )
    result = compare_nonnested_models(
        [0.0, 0.1], [0.2, 0.3], 1, 1, model_a="spatial", model_b="bifactor"
    )
    assert result.status is ComparisonStatus.MODEL_B_PREFERRED
    assert result.preferred_model == "bifactor"


@pytest.mark.parametrize("z", [0.0, 1.0])
def test_no_preference_when_not_significant_or_directionless(monkeypatch, z):
    """Non-significance and a zero direction both remain indeterminate."""
    p_value = 0.001 if z == 0.0 else 0.2
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(z=z, p=p_value),
    )
    result = compare_nonnested_models([0.0, 0.1], [0.2, 0.3], 1, 1)
    assert result.status is ComparisonStatus.NO_SIGNIFICANT_DIFFERENCE
    assert result.preferred_model is None
    assert result.warning is None


@pytest.mark.parametrize(
    ("kernel_result", "omega_tol"),
    [
        (_sentinel(omega=0.0), 1e-12),
        (_sentinel(omega=1e-14), 1e-12),
        (_sentinel(z=float("nan")), 0.0),
        (_sentinel(p=float("nan")), 0.0),
    ],
)
def test_degenerate_variance_or_nonfinite_inference_fails_closed(
    monkeypatch, kernel_result, omega_tol
):
    """Undefined normal inference never produces a model preference."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: kernel_result,
    )
    result = compare_nonnested_models(
        [0.0, 0.1], [0.2, 0.3], 1, 1, omega_tol=omega_tol
    )
    assert result.status is ComparisonStatus.VARIANCE_DEGENERATE
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert math.isnan(result.p_two_sided)
    assert "undefined" in result.warning


@pytest.mark.parametrize(
    "relation",
    [ModelRelation.NESTED, ModelRelation.BOUNDARY_NESTED],
)
def test_nested_relations_require_likelihood_ratio(monkeypatch, relation):
    """Nested and boundary-nested declarations suppress Vuong preference."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1], [0.2, 0.3], 1, 1, relation=relation
    )
    assert result.status is ComparisonStatus.REQUIRES_LIKELIHOOD_RATIO
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert "likelihood-ratio" in result.warning


def test_overlapping_relation_requires_formal_distinguishability(monkeypatch):
    """A numerical omega floor is not mislabeled as Vuong's formal test."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1],
        [0.2, 0.3],
        1,
        1,
        relation=ModelRelation.OVERLAPPING,
    )
    assert result.status is ComparisonStatus.REQUIRES_DISTINGUISHABILITY_TEST
    assert result.preferred_model is None
    assert "weighted-chi-square" in result.warning


def test_unknown_relation_fails_closed(monkeypatch):
    """Unknown nestedness is preserved as an explicit unresolved state."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1], [0.2, 0.3], 1, 1, relation=ModelRelation.UNKNOWN
    )
    assert result.status is ComparisonStatus.UNKNOWN_RELATION
    assert result.preferred_model is None
    assert "establish nestedness" in result.warning


def test_model_labels_are_trimmed_and_auditable(monkeypatch):
    """Whitespace is removed without replacing the caller's model names."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: _sentinel(),
    )
    result = compare_nonnested_models(
        [0.0, 0.1], [0.2, 0.3], 1, 1, model_a="  A  ", model_b=" B "
    )
    assert result.model_a == "A"
    assert result.model_b == "B"
    assert result.n_cases == 2
    assert result.k_a == 1
    assert result.k_b == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model_a": "", "model_b": "B"}, "model_a"),
        ({"model_a": "A", "model_b": "   "}, "model_b"),
        ({"model_a": "same", "model_b": "same"}, "distinct"),
        ({"relation": "not-a-relation"}, "relation"),
        ({"bic_correction": 1}, "boolean"),
        ({"alpha": 0.0}, "alpha"),
        ({"alpha": 1.0}, "alpha"),
        ({"alpha": float("nan")}, "alpha"),
        ({"omega_tol": -1.0}, "omega_tol"),
        ({"omega_tol": float("inf")}, "omega_tol"),
    ],
)
def test_orchestration_validation_guards(monkeypatch, kwargs, message):
    """Metadata and decision thresholds are validated before kernel use."""
    monkeypatch.setattr(
        comparison_module,
        "vuong_nonnested",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid metadata reached kernel")
        ),
    )
    with pytest.raises(ValueError, match=message):
        compare_nonnested_models([0.0, 0.1], [0.2, 0.3], 1, 1, **kwargs)


@pytest.mark.parametrize(
    ("bad_a", "bad_b", "message"),
    [
        (1.0, [0.0, 0.1], "loglik_a must be an iterable"),
        ("1.0,2.0", [0.0, 0.1], "loglik_a must be an iterable"),
        ([0.0, 0.1], object(), "loglik_b must be an iterable"),
        ([0.0, 0.1], b"0,1", "loglik_b must be an iterable"),
    ],
)
def test_casewise_inputs_require_numeric_iterables(bad_a, bad_b, message):
    """Scalar and text inputs fail with a stable public validation error."""
    with pytest.raises(ValueError, match=message):
        compare_nonnested_models(bad_a, bad_b, 1, 1)


def test_low_level_input_validation_is_preserved():
    """Malformed statistical inputs still fail through the trusted wrapper."""
    with pytest.raises(ValueError, match="equal-length"):
        compare_nonnested_models([1.0], [1.0], 1, 1)
    with pytest.raises(ValueError, match="equal-length"):
        compare_nonnested_models([1.0, 2.0], [1.0, 2.0, 3.0], 1, 1)
    with pytest.raises(ValueError, match="non-negative integer"):
        compare_nonnested_models([1.0, 2.0], [0.9, 1.8], -1, 1)


def test_public_wrapper_matches_rust_backed_low_level_result():
    """The public decision object preserves every exposed Rust statistic."""
    a = [-1.0, -1.2, -0.8, -1.4, -0.9, -1.1]
    b = [-1.5, -1.0, -1.4, -1.1, -1.3, -1.2]
    low_level = vuong_nonnested(a, b, 3, 2, bic_correction=True)
    result = compare_nonnested_models(
        a,
        b,
        3,
        2,
        model_a="MLS2PLM",
        model_b="BIFAC2PLM",
        bic_correction=True,
    )
    assert result.raw_mean_loglik_difference == pytest.approx(low_level["mean_diff"])
    assert result.omega == pytest.approx(low_level["omega"])
    if result.variance_positive:
        assert result.z == pytest.approx(low_level["z"])
        assert result.p_two_sided == pytest.approx(low_level["p_two_sided"])
