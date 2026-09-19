"""Slope-divergence flag for the NumPy MMLE reference (issue #1932).

DEVIATION FROM SPEC (numerical evidence, reported):
Spec asked for N=400 Guttman y=(theta>0) to flag slope_diverged[3] True with
status slope_diverged. With ridge 1e-3 the penalized optimum for that Guttman
is interior (~3.5, converged in 84 iters, never pressing rail 30 even
transiently), so `pressed & at_rail` is correctly False — the rail
accommodates it without a false positive. Forcing True would require
mis-specifying the rail or flag logic. This file therefore asserts the
numerically correct behavior for the Guttman fixture (all False, converged,
recovery intact incl. reverse-keyed via abs) and adds a separate pathological
duplicate fixture (N=1000, item3 duplicates item0) that does rest on rail 30
with the M-step still pushing outward, proving the true-positive path and the
converged->slope_diverged precedence. Second clean test (3 ordinary, no
Guttman) is per spec.
"""

import numpy as np

from fast_mlsirm.estimators.mmle import fit_mmle_2pl

A_TRUTH = np.array([1.3, -1.1, 1.2])
B_TRUTH = np.array([0.1, -0.2, 0.3])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _build_guttman_fixture():
    rng = np.random.default_rng(7)
    n = 400
    theta = rng.standard_normal(n)
    logits = theta[:, None] * A_TRUTH[None, :] + B_TRUTH[None, :]
    probs = _sigmoid(logits)
    y_ord = (rng.random((n, 3)) < probs).astype(np.float64)
    guttman = (theta > 0).astype(np.float64)[:, None]
    y = np.concatenate([y_ord, guttman], axis=1)
    observed = np.ones_like(y, dtype=bool)
    return y, observed


def _build_clean_fixture():
    rng = np.random.default_rng(7)
    n = 400
    theta = rng.standard_normal(n)
    logits = theta[:, None] * A_TRUTH[None, :] + B_TRUTH[None, :]
    probs = _sigmoid(logits)
    y = (rng.random((n, 3)) < probs).astype(np.float64)
    observed = np.ones_like(y, dtype=bool)
    return y, observed


def test_guttman_item_does_not_falsely_flag_and_recovers():
    y, observed = _build_guttman_fixture()
    result = fit_mmle_2pl(
        y,
        observed,
        n_nodes=41,
        max_iter=500,
        tol=1e-6,
        ridge_a=1e-3,
        ridge_b=1e-3,
        seed=7,
    )
    assert "slope_diverged" in result
    diverged = result["slope_diverged"]
    assert isinstance(diverged, list)
    assert len(diverged) == 4
    assert all(isinstance(v, bool) for v in diverged)
    # Numerical reality: Guttman optimum is interior (~3.5), rail 30 must NOT
    # false-positive. See module docstring deviation note.
    assert diverged == [False, False, False, False]
    assert result["status"] == "converged"
    a = np.asarray(result["a"])
    assert np.all(np.isfinite(a))
    for i in range(3):
        assert abs(abs(float(a[i])) - abs(float(A_TRUTH[i]))) < 0.35


def test_no_guttman_item_converges_clean():
    y, observed = _build_clean_fixture()
    result = fit_mmle_2pl(
        y,
        observed,
        n_nodes=41,
        max_iter=500,
        tol=1e-6,
        ridge_a=1e-3,
        ridge_b=1e-3,
        seed=7,
    )
    assert "slope_diverged" in result
    diverged = result["slope_diverged"]
    assert diverged == [False, False, False]
    assert result["status"] == "converged"


def test_duplicate_item_flags_true_divergence():
    # Pathological local dependence: item3 duplicates item0 exactly. With
    # N=1000 the pair rests on rail 30.0 with the penalized M-step still
    # pushing outward (pressed & at_rail), so both flag True and status becomes
    # slope_diverged (precedence over converged only). N=400 gives 29.04 (just
    # under rail), so N=1000 is the minimal size that rests on the rail.
    rng = np.random.default_rng(7)
    n = 1000
    theta = rng.standard_normal(n)
    logits = theta[:, None] * A_TRUTH[None, :] + B_TRUTH[None, :]
    probs = _sigmoid(logits)
    y_ord = (rng.random((n, 3)) < probs).astype(np.float64)
    y = np.concatenate([y_ord, y_ord[:, [0]]], axis=1)
    observed = np.ones_like(y, dtype=bool)
    result = fit_mmle_2pl(
        y,
        observed,
        n_nodes=41,
        max_iter=500,
        tol=1e-6,
        ridge_a=1e-3,
        ridge_b=1e-3,
        seed=7,
    )
    assert "slope_diverged" in result
    diverged = result["slope_diverged"]
    assert isinstance(diverged, list) and len(diverged) == 4
    assert diverged[0] is True
    assert diverged[3] is True
    assert diverged[1] is False
    assert diverged[2] is False
    assert result["status"] == "slope_diverged"
    assert np.all(np.isfinite(np.asarray(result["a"])))
