"""Verifiable regression gate for FIPC flat fixed-LL + moving prior mean.

Replays the tip-``0621169b`` evidence signature in pure Python (no Rust /
cargo): ``fixed_ll`` stuck near ``-1405.8755``, prior mean moving, and
``n_accepted_prior_steps=100``. The consumer assert must FAIL on that pattern.
"""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.two_tier_fipc_trace_asserts import (
    EVIDENCE_FLAT_FIXED_LL_0621169B,
    TwoTierFipcFixedMeasureRegressionError,
    assert_two_tier_fipc_prior_accepts_improve_fixed_measure,
)


def _evidence_0621169b_flat_fixed_moving_mean(
    *, n_iter: int = 100, n_primary: int = 2
) -> dict:
    """Synthetic traces matching observed tip failure mode (not a live fit)."""
    flat = float(EVIDENCE_FLAT_FIXED_LL_0621169B)
    # post-init flat: first entry may differ slightly; thereafter stuck.
    fixed = np.full(n_iter, flat, dtype=np.float64)
    fixed[0] = flat  # evidence: flat at -1405.8755 across recorded evals
    # Mean drifts every iteration while frozen LL does not.
    means = np.zeros((n_iter, n_primary), dtype=np.float64)
    for t in range(n_iter):
        means[t, 0] = 0.01 * t
        means[t, 1] = -0.005 * t
    return {
        "fixed_loglik_trace": fixed,
        "prior_mean_trace": means.ravel(),
        "n_primary": n_primary,
        "n_iter": n_iter,
        "n_accepted_prior_steps": 100,
    }


def test_0621169b_evidence_flat_fixed_ll_with_moving_mean_fails() -> None:
    """(1)+(2): flat fixed LL + moving mean + n_accepted=100 must fail closed."""
    kw = _evidence_0621169b_flat_fixed_moving_mean()
    with pytest.raises(TwoTierFipcFixedMeasureRegressionError, match="flat"):
        assert_two_tier_fipc_prior_accepts_improve_fixed_measure(**kw)


def test_mean_only_accepts_without_fixed_ll_improvement_fails() -> None:
    """(2) explicit mean-only counter without frozen-LL gain fails closed."""
    n_iter = 5
    n_primary = 2
    # Not flat (declining), but never improves vs first post-init value.
    fixed = np.array([-1400.0, -1401.0, -1402.0, -1403.0, -1404.0])
    means = np.linspace(0.0, 1.0, n_iter * n_primary).reshape(n_iter, n_primary)
    with pytest.raises(
        TwoTierFipcFixedMeasureRegressionError, match="mean-only"
    ):
        assert_two_tier_fipc_prior_accepts_improve_fixed_measure(
            fixed_loglik_trace=fixed,
            prior_mean_trace=means.ravel(),
            n_primary=n_primary,
            n_iter=n_iter,
            n_accepted_prior_steps=4,
            mean_only_accepts=4,
        )


def test_healthy_fixed_ll_improvement_with_mean_move_passes() -> None:
    """Moving mean is allowed when frozen diagnostic LL actually improves."""
    n_iter = 4
    n_primary = 2
    fixed = np.array([-1410.0, -1408.0, -1406.0, -1404.0])
    means = np.array([[0.0, 0.0], [0.1, -0.1], [0.2, -0.2], [0.3, -0.25]])
    assert_two_tier_fipc_prior_accepts_improve_fixed_measure(
        fixed_loglik_trace=fixed,
        prior_mean_trace=means.ravel(),
        n_primary=n_primary,
        n_iter=n_iter,
        n_accepted_prior_steps=3,
        mean_only_accepts=0,
    )


def test_flat_fixed_ll_without_mean_move_passes() -> None:
    """Flat frozen LL alone is OK if the prior mean path is stationary."""
    n_iter = 3
    n_primary = 1
    fixed = np.full(n_iter, EVIDENCE_FLAT_FIXED_LL_0621169B)
    means = np.zeros(n_iter)
    assert_two_tier_fipc_prior_accepts_improve_fixed_measure(
        fixed_loglik_trace=fixed,
        prior_mean_trace=means,
        n_primary=n_primary,
        n_iter=n_iter,
        n_accepted_prior_steps=0,
    )
