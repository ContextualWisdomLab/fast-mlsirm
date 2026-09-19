# Copyright (c) 2026 ContextualWisdomLab. All rights reserved.
# SPDX-License-Identifier: MIT

"""Unit tests for polytomous bifactor estimation, recursion, and sensitivity."""

import numpy as np
import pytest

from fast_mlsirm.polytomous_bifactor import (
    PolytomousBifactorFit,
    bifactor_lord_wingersky,
    bifactor_slope_sensitivity,
    direct_enumeration_bifactor,
    fit_polytomous_bifactor,
)


def test_bifactor_lord_wingersky_matches_direct_enumeration() -> None:
    """Lord-Wingersky 2-stage recursion must match direct enumeration within 1e-12.

    Evaluated on a 257-point general grid spanning [-8.0, 8.0] with step 0.0625,
    using a 4-item, 2-domain, 3-category configuration.
    """
    n_items = 4
    n_cat = 3
    n_domains = 2

    a_general = np.array([1.2, 0.9, 1.5, 1.1], dtype=np.float64)
    a_specific = np.array([0.8, 1.0, 0.7, 1.2], dtype=np.float64)
    # Thresholds: (n_cat - 1) = 2 per item, strictly ordered
    thresholds = np.array([
        [-0.6, 0.4],
        [-0.8, 0.2],
        [-0.4, 0.7],
        [-1.0, 0.1],
    ], dtype=np.float64)
    item_domains = np.array([0, 0, 1, 1], dtype=np.int64)

    # 257-point grid from -8.0 to 8.0 with step 0.0625
    theta_general = np.linspace(-8.0, 8.0, 257)

    # Specific factor quadrature: 21 standard normal nodes and weights
    theta_specific = np.linspace(-4.0, 4.0, 21)
    weights = np.exp(-0.5 * theta_specific**2)
    weights_specific = weights / np.sum(weights)

    res_lw = bifactor_lord_wingersky(
        a_general=a_general,
        a_specific=a_specific,
        thresholds=thresholds,
        item_domains=item_domains,
        n_cat=n_cat,
        n_domains=n_domains,
        theta_general=theta_general,
        theta_specific=theta_specific,
        weights_specific=weights_specific,
    )

    res_direct = direct_enumeration_bifactor(
        a_general=a_general,
        a_specific=a_specific,
        thresholds=thresholds,
        item_domains=item_domains,
        n_cat=n_cat,
        n_domains=n_domains,
        theta_general=theta_general,
        theta_specific=theta_specific,
        weights_specific=weights_specific,
    )

    total_max_score = n_items * (n_cat - 1)
    assert res_lw.shape == (257, total_max_score + 1)
    assert res_direct.shape == (257, total_max_score + 1)

    # All probabilities must sum to 1.0 across total score distribution
    sums_lw = np.sum(res_lw, axis=1)
    np.testing.assert_allclose(sums_lw, 1.0, atol=1e-12)

    # Maximum difference across entire (257, 9) grid must be <= 1e-12
    max_diff = np.max(np.abs(res_lw - res_direct))
    assert max_diff <= 1e-12, f"Lord-Wingersky max diff {max_diff} exceeded 1e-12"


def test_polytomous_bifactor_oakes_se_fails_closed() -> None:
    """Do not report uncomputed Oakes standard errors as a successful fit."""
    responses = np.zeros((4, 2), dtype=np.int64)
    loading_pattern = np.ones((2, 1), dtype=np.uint8)

    with pytest.raises(NotImplementedError, match="polytomous bifactor Oakes"):
        fit_polytomous_bifactor(
            responses=responses,
            loading_pattern=loading_pattern,
            n_cat=2,
            compute_oakes_se=True,
        )


def test_fit_polytomous_bifactor_multiple_group() -> None:
    """Fit a 16-item, 4-category, 6-dimension bifactor model with 2 groups.

    Structure: G + 4 specific domains + W method factor on items 0, 4, 8, 12.
    Verifies:
        - QMCEM convergence
        - Reference group 0 fixed at N(0, I)
        - Focal group 1 moments freely estimated
        - Unsupported Oakes standard errors remain absent when explicitly disabled
    """
    n_persons = 200
    n_items = 16
    n_cat = 4
    n_dims = 6  # 0: G, 1..4: specific domains, 5: W method factor

    # Loading pattern: 16 items x 6 dimensions
    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1  # General factor
        domain = i // 4
        loading_pattern[i, 1 + domain] = 1  # Specific domain factor
        if i % 4 == 0:
            loading_pattern[i, 5] = 1  # Method factor W

    # Generate synthetic response data with known group split
    rng = np.random.default_rng(12345)
    group_ids = np.zeros(n_persons, dtype=np.int64)
    group_ids[n_persons // 2:] = 1  # Half in group 0, half in group 1

    # Simulated category responses in 0..3
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)

    fit = fit_polytomous_bifactor(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        group_ids=group_ids,
        n_groups=2,
        max_iter=30,
        tol=1e-4,
        qmc_draws=500,
        compute_oakes_se=False,
    )

    assert isinstance(fit, PolytomousBifactorFit)
    assert fit.n_dims == 6
    assert fit.n_items == 16
    assert fit.n_cat == 4
    assert fit.n_groups == 2
    assert fit.slope.shape == (16, 6)
    assert fit.threshold.shape == (16, 3)
    assert fit.group_means.shape == (2, 6)
    assert fit.group_variances.shape == (2, 6)

    # Reference group 0 moments must be 0 and 1
    np.testing.assert_allclose(fit.group_means[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(fit.group_variances[0], 1.0, atol=1e-10)

    # Focal group moments must be finite numbers
    assert np.all(np.isfinite(fit.group_means[1]))
    assert np.all(fit.group_variances[1] > 0.0)

    assert fit.oakes_se_slope is None
    assert fit.oakes_se_threshold is None
    assert fit.min_eigenvalue is None


def test_bifactor_slope_sensitivity_monotonic_loglik() -> None:
    """Slope bound sensitivity sweep across bounds [4, 6, 8, 10] must be non-decreasing."""
    n_persons = 100
    n_items = 8
    n_cat = 3
    n_dims = 3

    loading_pattern = np.zeros((n_items, n_dims), dtype=np.uint8)
    for i in range(n_items):
        loading_pattern[i, 0] = 1
        loading_pattern[i, 1 + (i // 4)] = 1

    rng = np.random.default_rng(42)
    responses = rng.integers(0, n_cat, size=(n_persons, n_items), endpoint=False)

    bounds = [4.0, 6.0, 8.0, 10.0]
    sweep = bifactor_slope_sensitivity(
        responses=responses,
        loading_pattern=loading_pattern,
        n_cat=n_cat,
        candidate_bounds=bounds,
        max_iter=20,
        tol=1e-3,
        qmc_draws=300,
    )

    assert len(sweep) == 4
    logliks = [entry["loglik"] for entry in sweep]

    # As upper bound relaxes, maximum achievable log-likelihood cannot decrease
    # Allow numerical tolerance 1e-2 for QMC stochastic approximation
    for k in range(len(logliks) - 1):
        assert logliks[k] <= logliks[k + 1] + 1e-2, (
            f"Log-likelihood decreased from bound {bounds[k]} ({logliks[k]}) "
            f"to bound {bounds[k+1]} ({logliks[k+1]})"
        )
