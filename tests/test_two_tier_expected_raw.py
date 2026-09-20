"""Tests for two-tier expected raw total E[T|theta_focal] (manuscript G+4+W path)."""
from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm.polytomous import (
    PolytomousFit,
    check_bifactor_expected_total_score_monotonicity,
    predict_expected_response_polytomous,
)
from fast_mlsirm.two_tier_grm import (
    TwoTierGrmFit,
    expected_total_score_two_tier_from_fit,
    expected_total_score_two_tier_given_primary,
)

_PRIOR = "independent_standardized"


def _call(ap, asp, th, smap, grid, *, q=21, focal=0, **kw):
    return expected_total_score_two_tier_given_primary(
        ap,
        asp,
        th,
        smap,
        grid,
        focal_primary=focal,
        q_nuisance=q,
        nuisance_prior=_PRIOR,
        **kw,
    )


def test_q_nuisance_and_prior_required() -> None:
    ap = np.array([[1.0, 0.0], [1.2, 0.0]])
    asp = np.array([0.5, 0.4])
    th = np.array([[1.0, 0.0, -1.0], [0.5, -0.2, -1.2]])
    smap = np.array([0, 0], dtype=np.int64)
    grid = np.linspace(-2.0, 2.0, 5)
    with pytest.raises(TypeError):
        expected_total_score_two_tier_given_primary(
            ap, asp, th, smap, grid, focal_primary=0  # type: ignore[call-arg]
        )
    with pytest.raises(TypeError):
        expected_total_score_two_tier_given_primary(
            ap, asp, th, smap, grid, focal_primary=0, q_nuisance=11  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="independent_standardized"):
        expected_total_score_two_tier_given_primary(
            ap,
            asp,
            th,
            smap,
            grid,
            focal_primary=0,
            q_nuisance=11,
            nuisance_prior="phi_conditional",
        )
    with pytest.raises(ValueError):
        _call(ap, asp, th, smap, grid, q=0)


def test_matches_bifactor_monotonicity_when_no_extra_primary() -> None:
    """One primary + specifics ⇒ same curve as bifactor E[T|G] helper."""
    rng = np.random.default_rng(0)
    n_items = 6
    a_g = rng.uniform(0.6, 1.4, size=n_items)
    a_s = rng.uniform(0.3, 0.9, size=n_items)
    th = np.sort(rng.normal(size=(n_items, 3)), axis=1)[:, ::-1]
    smap = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    ap = a_g.reshape(-1, 1)
    grid = np.linspace(-3.0, 3.0, 17)
    got = _call(ap, a_s, th, smap, grid, q=21)

    class _BF:
        a_general = a_g
        a_specific = a_s
        threshold = th

    ref = check_bifactor_expected_total_score_monotonicity(_BF(), grid, q_specific=21)
    assert np.allclose(got.expected_total, ref.expected_total, atol=1e-10, rtol=0.0)
    assert got.prior == _PRIOR
    assert got.nuisance_mean == 0.0
    assert got.nuisance_sd == 1.0


def test_wording_cross_loads_change_curve_vs_specific_only() -> None:
    """Crossed W loadings must alter E[T|G] vs integrating only specifics."""
    a_g = np.array([1.0, 1.0, 1.0, 1.0])
    a_w = np.array([0.0, 0.0, 0.8, 0.8])
    a_s = np.array([0.5, 0.5, 0.5, 0.5])
    th = np.array([[1.0, 0.0, -1.0]] * 4)
    smap = np.array([0, 1, 2, 3], dtype=np.int64)
    ap = np.column_stack([a_g, a_w])
    grid = np.linspace(-2.0, 2.0, 9)
    with_w = _call(ap, a_s, th, smap, grid, q=11)
    no_w = _call(np.column_stack([a_g, np.zeros(4)]), a_s, th, smap, grid, q=11)
    assert not np.allclose(with_w.expected_total, no_w.expected_total)


def test_person_eap_points_accepted_without_ascending_grid() -> None:
    """Person G EAP vectors need not be sorted (unlike monotonicity helpers)."""
    ap = np.array([[1.0, 0.4], [1.1, 0.0]])
    asp = np.array([0.5, 0.6])
    th = np.array([[0.5, -0.5], [0.2, -0.8]])
    smap = np.array([0, 0], dtype=np.int64)
    eap = np.array([0.3, -1.2, 0.3, 2.0])
    out = _call(ap, asp, th, smap, eap, q=9)
    assert out.expected_total.shape == eap.shape
    assert out.expected_total[0] == pytest.approx(out.expected_total[2])


def test_category_support_rejects_nondecreasing_thresholds() -> None:
    ap = np.array([[1.0]])
    asp = np.array([0.0])
    smap = np.array([-1], dtype=np.int64)
    grid = np.array([0.0])
    with pytest.raises(ValueError, match="strictly decreasing"):
        _call(ap, asp, np.array([[0.0, 0.0, -1.0]]), smap, grid, q=5)
    with pytest.raises(ValueError, match="strictly decreasing"):
        _call(ap, asp, np.array([[-1.0, 0.0, 1.0]]), smap, grid, q=5)


def test_reverse_keyed_negative_g_slope_reverses_total_trend() -> None:
    """Negative a_G (reverse key) must decrease E[T|G] as G increases."""
    th = np.array([[1.5, 0.0, -1.5], [1.5, 0.0, -1.5]])
    smap = np.array([-1, -1], dtype=np.int64)
    asp = np.zeros(2)
    grid = np.linspace(-2.0, 2.0, 9)
    forward = _call(np.array([[1.2], [1.2]]), asp, th, smap, grid, q=15)
    reverse = _call(np.array([[1.2], [-1.2]]), asp, th, smap, grid, q=15)
    assert np.all(np.diff(forward.expected_total) > 0.0)
    # Mixed: one reverse item pulls the total slope down vs all-forward.
    assert reverse.expected_total[-1] < forward.expected_total[-1]
    assert reverse.expected_total[0] > forward.expected_total[0]


def test_nuisance_reference_mean_sd_shift_curve() -> None:
    ap = np.array([[1.0], [1.0]])
    asp = np.array([0.8, 0.8])
    th = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    smap = np.array([0, 0], dtype=np.int64)
    grid = np.array([0.0])
    base = _call(ap, asp, th, smap, grid, q=21)
    shifted = _call(ap, asp, th, smap, grid, q=21, nuisance_mean=1.0, nuisance_sd=0.5)
    assert not np.allclose(base.expected_total, shifted.expected_total)
    assert shifted.nuisance_mean == 1.0
    assert shifted.nuisance_sd == 0.5
    with pytest.raises(ValueError):
        _call(ap, asp, th, smap, grid, q=5, nuisance_sd=0.0)


def _g4w_16_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """16-item emotionality-like G+4+W simple structure (4 cats, reverse keys).

    Domain blocks of 4 items; wording-crossed indices match late-life style
    {4,5,6,9,12,14,15}; reverse keys on items 2,7,11,13 (negative a_G).
    """
    n_items = 16
    wording = {4, 5, 6, 9, 12, 14, 15}
    reverse = {2, 7, 11, 13}
    a_g = np.full(n_items, 1.1)
    for i in reverse:
        a_g[i] = -1.1
    a_w = np.zeros(n_items)
    for i in wording:
        a_w[i] = 0.55
    a_s = np.full(n_items, 0.65)
    smap = np.array([i // 4 for i in range(n_items)], dtype=np.int64)
    th = np.tile(np.array([1.2, 0.0, -1.2]), (n_items, 1))
    ap = np.column_stack([a_g, a_w])
    return ap, a_s, th, smap


def test_g4w_16_item_quantitative_bounds_and_wording_effect() -> None:
    ap, a_s, th, smap = _g4w_16_fixture()
    grid = np.linspace(-3.0, 3.0, 13)
    out = _call(ap, a_s, th, smap, grid, q=15)
    # 16 items × categories {0,1,2,3} ⇒ total in [0, 48]
    assert out.n_items == 16
    assert np.all(out.expected_total >= 0.0 - 1e-9)
    assert np.all(out.expected_total <= 48.0 + 1e-9)
    # Midpoint at G=0 under symmetric thresholds + balanced reverse set.
    assert out.expected_total[grid == 0.0][0] == pytest.approx(24.0, abs=1e-8)
    # Wording cross must move the curve relative to W=0.
    no_w = _call(np.column_stack([ap[:, 0], np.zeros(16)]), a_s, th, smap, grid, q=15)
    assert float(np.max(np.abs(out.expected_total - no_w.expected_total))) > 0.05
    # Reverse keys (negative a_G) compress the high-G total vs all-forward signs.
    ap_fwd = ap.copy()
    ap_fwd[:, 0] = np.abs(ap_fwd[:, 0])
    fwd = _call(ap_fwd, a_s, th, smap, grid, q=15)
    assert out.expected_total[-1] < fwd.expected_total[-1]
    assert out.expected_total[0] > fwd.expected_total[0]


def test_monte_carlo_reference_two_nuisance_item() -> None:
    """Independent GH product agrees with MC under N(0,1)×N(0,1) nuisances."""
    # Single wording-crossed item: G + W + S.
    a_g, a_w, a_s = 1.0, 0.7, 0.5
    th = np.array([[1.0, 0.0, -1.0]])
    g0 = 0.5
    q = 31
    gh = _call(
        np.array([[a_g, a_w]]),
        np.array([a_s]),
        th,
        np.array([0], dtype=np.int64),
        np.array([g0]),
        q=q,
    )

    rng = np.random.default_rng(20260921)
    n_mc = 80_000
    w = rng.normal(size=n_mc)
    s = rng.normal(size=n_mc)
    base = a_g * g0 + a_w * w + a_s * s
    cell = PolytomousFit(
        model="grm",
        slope=np.ones(1),
        cat_params=th,
        loglik=float("nan"),
        n_iter=0,
        converged=True,
        termination_reason="mc",
    )
    mc = float(predict_expected_response_polytomous(cell, base).mean())
    assert gh.expected_total[0] == pytest.approx(mc, abs=0.02)


def _stub_fit(*, phi: np.ndarray) -> TwoTierGrmFit:
    ap = np.array([[1.0, 0.0], [1.0, 0.4]])
    asp = np.array([0.5, 0.5])
    th = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    return TwoTierGrmFit(
        a_primary=ap,
        a_specific=asp,
        threshold=th,
        phi=np.asarray(phi, dtype=np.float64),
        theta_p_eap=np.zeros((2, 2)),
        theta_p_sd=np.ones((2, 2)),
        category_counts=np.ones((2, 4), dtype=np.int64),
        n_cat=4,
        n_primary=2,
        n_specific=1,
        loglik_trace=np.array([0.0]),
        n_iter=1,
        converged=True,
        termination_reason="tolerance_met",
        final_loglik_change=0.0,
        best_start=0,
        n_parameters=10,
    )


def test_from_fit_fail_closed_when_phi_not_identity() -> None:
    smap = np.array([0, 0], dtype=np.int64)
    grid = np.array([0.0, 1.0])
    ok = expected_total_score_two_tier_from_fit(
        _stub_fit(phi=np.eye(2)),
        grid,
        focal_primary=0,
        q_nuisance=9,
        specific_map=smap,
    )
    assert ok.prior == _PRIOR

    phi_corr = np.array([[1.0, 0.2], [0.2, 1.0]])
    with pytest.raises(ValueError, match="Phi == I"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=phi_corr),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
        )
    # Near-identity must still fail closed (no silent Phi≈I substitute).
    phi_near = np.array([[1.0, 1e-8], [1e-8, 1.0]])
    with pytest.raises(ValueError, match="Phi == I"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=phi_near),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
        )
