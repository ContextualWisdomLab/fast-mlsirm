"""Tests for cluster-aware non-nested model comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm.model_comparison import (
    ModelRelation,
    compare_nonnested_models,
)


def _likelihoods():
    """Return paired contributions with a stable model-A advantage."""
    b = np.array([-1.3, -1.1, -1.5, -1.0, -1.4, -1.2], dtype=float)
    a = b + np.array([0.5, 0.3, 0.6, 0.2, 0.7, 0.4], dtype=float)
    return a, b


def test_prefers_model_a_and_bootstrap_is_deterministic():
    """A clear positive likelihood advantage selects model A reproducibly."""
    a, b = _likelihoods()
    first = compare_nonnested_models(
        a, b, 3, 3, correction="none", bootstrap=200, seed=17
    )
    second = compare_nonnested_models(
        a, b, 3, 3, correction="none", bootstrap=200, seed=17
    )
    assert first.distinguishable
    assert first.z > 0.0
    assert first.preferred_model == "a"
    assert first.bootstrap_ci == second.bootstrap_ci
    assert first.bootstrap_ci[0] > 0.0


def test_cluster_aggregation_uses_independent_units():
    """Repeated cells are summed within first-seen query clusters."""
    a, b = _likelihoods()
    result = compare_nonnested_models(
        a,
        b,
        2,
        2,
        cluster_id=np.array(["q2", "q2", "q1", "q1", "q3", "q3"]),
        correction="none",
        bootstrap=0,
    )
    assert result.n_cases == 6
    assert result.n_clusters == 3
    assert math.isnan(result.bootstrap_ci[0])


def test_aic_and_bic_penalize_larger_model():
    """AIC and BIC corrections reduce model A's corrected mean advantage."""
    a, b = _likelihoods()
    raw = compare_nonnested_models(a, b, 3, 1, correction="none", bootstrap=0)
    aic = compare_nonnested_models(a, b, 3, 1, correction="aic", bootstrap=0)
    bic = compare_nonnested_models(a, b, 3, 1, correction="bic", bootstrap=0)
    assert aic.mean_difference < raw.mean_difference
    assert bic.mean_difference < raw.mean_difference


def test_indistinguishable_models_return_no_preference():
    """Constant casewise differences fail the numerical distinguishability gate."""
    b = np.array([-1.0, -1.2, -1.4, -1.6])
    a = b + 0.2
    result = compare_nonnested_models(a, b, 2, 2)
    assert not result.distinguishable
    assert result.preferred_model is None
    assert math.isnan(result.z)
    assert "indistinguishable" in result.warning.lower()


def test_boundary_relation_emits_primary_procedure_warning():
    """Boundary-nested declarations are not silently treated as ordinary Vuong cases."""
    a, b = _likelihoods()
    result = compare_nonnested_models(
        a,
        b,
        3,
        2,
        relation=ModelRelation.BOUNDARY_NESTED,
        correction="none",
        bootstrap=0,
    )
    assert "parametric-bootstrap" in result.warning


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"loglik_a": [1.0], "loglik_b": [1.0]}, "at least two"),
        ({"loglik_a": [1.0, np.nan], "loglik_b": [1.0, 1.0]}, "finite"),
        ({"loglik_a": [1.0, 2.0], "loglik_b": [1.0, 2.0, 3.0]}, "same length"),
        ({"loglik_a": [1.0, 2.0], "loglik_b": [1.0, 2.1], "k_a": -1}, "non-negative"),
        ({"loglik_a": [1.0, 2.0], "loglik_b": [1.0, 2.1], "correction": "x"}, "correction"),
        ({"loglik_a": [1.0, 2.0], "loglik_b": [1.0, 2.1], "relation": "x"}, "relation"),
    ],
)
def test_validation_guards(kwargs, message):
    """Malformed inputs fail before reaching the Rust statistic kernel."""
    defaults = {"loglik_a": [1.0, 2.0], "loglik_b": [0.9, 1.7], "k_a": 1, "k_b": 1}
    defaults.update(kwargs)
    with pytest.raises(ValueError, match=message):
        compare_nonnested_models(**defaults)


def test_cluster_validation_guards():
    """Cluster labels must align and identify at least two independent units."""
    a, b = _likelihoods()
    with pytest.raises(ValueError, match="matching"):
        compare_nonnested_models(a, b, 1, 1, cluster_id=[1, 2])
    with pytest.raises(ValueError, match="at least two clusters"):
        compare_nonnested_models(a, b, 1, 1, cluster_id=[1] * len(a))
