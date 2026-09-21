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
    # Scalar N(0,1) broadcast: all specifics share the same reference — basis is
    # the bifactor helper's unit-normal specific prior (documented producer case).
    got = _call(ap, a_s, th, smap, grid, q=21)

    class _BF:
        a_general = a_g
        a_specific = a_s
        threshold = th

    ref = check_bifactor_expected_total_score_monotonicity(_BF(), grid, q_specific=21)
    assert np.allclose(got.expected_total, ref.expected_total, atol=1e-10, rtol=0.0)
    assert got.prior == _PRIOR
    assert np.allclose(got.primary_ref_mean, [0.0])
    assert np.allclose(got.primary_ref_sd, [1.0])
    assert np.allclose(got.specific_ref_mean, np.zeros(3))
    assert np.allclose(got.specific_ref_sd, np.ones(3))


def test_wording_cross_loads_change_curve_vs_specific_only() -> None:
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
    th = np.array([[1.5, 0.0, -1.5], [1.5, 0.0, -1.5]])
    smap = np.array([-1, -1], dtype=np.int64)
    asp = np.zeros(2)
    grid = np.linspace(-2.0, 2.0, 9)
    forward = _call(np.array([[1.2], [1.2]]), asp, th, smap, grid, q=15)
    reverse = _call(np.array([[1.2], [-1.2]]), asp, th, smap, grid, q=15)
    assert np.all(np.diff(forward.expected_total) > 0.0)
    assert reverse.expected_total[-1] < forward.expected_total[-1]
    assert reverse.expected_total[0] > forward.expected_total[0]


def test_per_dimension_ref_sd_changes_integral_vs_scalar_bundle() -> None:
    """Distinct W vs S reference variances must not be silently scalar-bundled."""
    ap = np.array([[1.0, 0.8]])  # G, W
    asp = np.array([0.6])
    # Asymmetric thresholds so scale changes move E[Y|G].
    th = np.array([[2.0, 0.5, -0.5]])
    smap = np.array([0], dtype=np.int64)
    grid = np.array([1.0])
    # Scalar broadcast: W and S both N(0,1) — only valid when producer fixes both to 1.
    scalar = _call(ap, asp, th, smap, grid, q=21)
    distinct = _call(
        ap,
        asp,
        th,
        smap,
        grid,
        q=21,
        primary_ref_mean=np.array([0.0, 0.0]),
        primary_ref_sd=np.array([1.0, 0.5]),
        specific_ref_mean=np.array([0.0]),
        specific_ref_sd=np.array([1.5]),
    )
    assert not np.allclose(scalar.expected_total, distinct.expected_total)
    assert distinct.primary_ref_sd[1] == pytest.approx(0.5)
    assert distinct.specific_ref_sd[0] == pytest.approx(1.5)


def test_ref_distribution_rejects_bad_shape_nan_nonpositive() -> None:
    ap = np.array([[1.0, 0.4], [1.0, 0.0]])
    asp = np.array([0.5, 0.5])
    th = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    smap = np.array([0, 1], dtype=np.int64)
    grid = np.array([0.0])
    with pytest.raises(ValueError, match="primary_ref_sd"):
        _call(ap, asp, th, smap, grid, q=5, primary_ref_sd=np.array([1.0]))  # want 2
    with pytest.raises(ValueError, match="specific_ref_mean"):
        _call(ap, asp, th, smap, grid, q=5, specific_ref_mean=np.array([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="finite"):
        _call(ap, asp, th, smap, grid, q=5, primary_ref_mean=np.array([0.0, np.nan]))
    with pytest.raises(ValueError, match="> 0"):
        _call(ap, asp, th, smap, grid, q=5, specific_ref_sd=np.array([1.0, -0.1]))
    with pytest.raises(ValueError, match="> 0"):
        _call(ap, asp, th, smap, grid, q=5, primary_ref_sd=0.0)


def _two_primary_four_specific_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic 12-item two-primary (G + method factor) + 4-specific fixture.

    Item-to-factor pattern is arbitrary and deliberately not any study's
    item map: 4 specifics of 3 items, method-factor loadings on items
    {1, 3, 8, 10}, reverse-keyed general slopes on items {0, 6, 11}.
    """
    n_items = 12
    method = {1, 3, 8, 10}
    reverse = {0, 6, 11}
    a_g = np.full(n_items, 1.1)
    for i in reverse:
        a_g[i] = -1.1
    a_w = np.zeros(n_items)
    for i in method:
        a_w[i] = 0.55
    a_s = np.full(n_items, 0.65)
    smap = np.array([i // 3 for i in range(n_items)], dtype=np.int64)
    th = np.tile(np.array([1.2, 0.0, -1.2]), (n_items, 1))
    ap = np.column_stack([a_g, a_w])
    return ap, a_s, th, smap


def test_two_primary_quantitative_bounds_and_method_factor_effect() -> None:
    ap, a_s, th, smap = _two_primary_four_specific_fixture()
    grid = np.linspace(-3.0, 3.0, 13)
    # Producer case: all nuisance refs fixed at N(0,1) — scalar broadcast documented.
    out = _call(ap, a_s, th, smap, grid, q=15)
    assert out.n_items == 12
    assert np.all(out.expected_total >= 0.0 - 1e-9)
    assert np.all(out.expected_total <= 36.0 + 1e-9)
    assert out.expected_total[grid == 0.0][0] == pytest.approx(18.0, abs=1e-8)
    no_w = _call(np.column_stack([ap[:, 0], np.zeros(12)]), a_s, th, smap, grid, q=15)
    assert float(np.max(np.abs(out.expected_total - no_w.expected_total))) > 0.05
    ap_fwd = ap.copy()
    ap_fwd[:, 0] = np.abs(ap_fwd[:, 0])
    fwd = _call(ap_fwd, a_s, th, smap, grid, q=15)
    assert out.expected_total[-1] < fwd.expected_total[-1]
    assert out.expected_total[0] > fwd.expected_total[0]


def test_two_primary_distinct_method_vs_specific_reference_variances() -> None:
    ap, a_s, th, smap = _two_primary_four_specific_fixture()
    grid = np.array([-1.0, 0.0, 1.0])
    unit = _call(ap, a_s, th, smap, grid, q=11)
    # W tighter than specifics — must differ from all-unit scalar broadcast.
    mixed = _call(
        ap,
        a_s,
        th,
        smap,
        grid,
        q=11,
        primary_ref_mean=np.zeros(2),
        primary_ref_sd=np.array([1.0, 0.4]),
        specific_ref_mean=np.zeros(4),
        specific_ref_sd=np.array([1.2, 0.9, 1.1, 1.3]),
    )
    assert not np.allclose(unit.expected_total, mixed.expected_total)


def test_monte_carlo_reference_two_nuisance_item() -> None:
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


def _stub_fit(*, phi: np.ndarray, n_specific: int = 1) -> TwoTierGrmFit:
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
        n_specific=n_specific,
        loglik_trace=np.array([0.0]),
        n_iter=1,
        converged=True,
        termination_reason="tolerance_met",
        final_loglik_change=0.0,
        best_start=0,
        n_parameters=10,
    )


def test_from_fit_dual_gates_phi_and_consumer_identification() -> None:
    smap = np.array([0, 0], dtype=np.int64)
    grid = np.array([0.0, 1.0])
    ok = expected_total_score_two_tier_from_fit(
        _stub_fit(phi=np.eye(2)),
        grid,
        focal_primary=0,
        q_nuisance=9,
        specific_map=smap,
        orthogonal_primary_identification=True,
    )
    assert ok.prior == _PRIOR

    with pytest.raises(ValueError, match="orthogonal_primary_identification"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=np.eye(2)),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
            orthogonal_primary_identification=False,
        )
    with pytest.raises(TypeError):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=np.eye(2)),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
        )  # type: ignore[call-arg]

    with pytest.raises(ValueError, match="Phi == I"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=np.array([[1.0, 0.2], [0.2, 1.0]])),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
            orthogonal_primary_identification=True,
        )
    with pytest.raises(ValueError, match="Phi == I"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=np.array([[1.0, 1e-8], [1e-8, 1.0]])),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
            orthogonal_primary_identification=True,
        )
    with pytest.raises(ValueError, match="fit.n_specific"):
        expected_total_score_two_tier_from_fit(
            _stub_fit(phi=np.eye(2), n_specific=3),
            grid,
            focal_primary=0,
            q_nuisance=9,
            specific_map=smap,
            orthogonal_primary_identification=True,
        )


def test_specific_map_rejects_noninteger_before_int64_cast() -> None:
    """Fractional / non-finite maps must fail closed (no silent truncation)."""
    ap = np.array([[1.0], [1.0]])
    asp = np.array([0.5, 0.5])
    th = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    grid = np.array([0.0])
    with pytest.raises(ValueError, match="integers"):
        _call(ap, asp, th, np.array([0.5, 0.0]), grid, q=5)
    with pytest.raises(ValueError, match="finite"):
        _call(ap, asp, th, np.array([0.0, np.nan]), grid, q=5)
    with pytest.raises(ValueError, match="specific-free|without wrapping"):
        _call(ap, asp, th, np.array([-2, 0]), grid, q=5)
    # Valid ints still accepted (including float dtype that is integral).
    out = _call(ap, asp, th, np.array([0.0, 0.0]), grid, q=5)
    assert out.n_items == 2


def test_specific_map_rejects_uint64_wraparound_boundaries() -> None:
    """Reject values that would wrap under int64 cast (not that INT64_MAX is allowed).

    Numpy ``uint64.astype(int64)`` wraps: ``uint64.max → -1`` and ``2**63 →
    INT64_MIN``. This test asserts the pre-cast gate rejects those inputs so
    they cannot be misread as the specific-free sentinel or a negative index.
    It does **not** claim ``INT64_MAX`` itself is a valid specific id.
    """
    from fast_mlsirm.two_tier_grm import (
        _as_specific_map_int64,
        expected_total_score_two_tier_from_fit,
    )

    u64_max = np.array([np.iinfo(np.uint64).max], dtype=np.uint64)
    two63 = np.array([np.uint64(2**63)], dtype=np.uint64)
    # Document the wrap the pre-cast gate must block (astype alone is unsafe).
    assert u64_max.astype(np.int64)[0] == np.int64(-1)
    assert two63.astype(np.int64)[0] == np.iinfo(np.int64).min

    with pytest.raises(ValueError, match="without wrapping|int64"):
        _as_specific_map_int64(u64_max, n_items=1)
    with pytest.raises(ValueError, match="without wrapping|int64"):
        _as_specific_map_int64(two63, n_items=1)

    ap = np.array([[1.0]])
    asp = np.array([0.0])
    th = np.array([[1.0, 0.0, -1.0]])
    grid = np.array([0.0])
    with pytest.raises(ValueError, match="without wrapping|int64"):
        _call(ap, asp, th, u64_max, grid, q=5)
    with pytest.raises(ValueError, match="without wrapping|int64"):
        _call(ap, asp, th, two63, grid, q=5)

    fit = _stub_fit(phi=np.eye(2), n_specific=1)
    u64_pair = np.array([0, np.iinfo(np.uint64).max], dtype=np.uint64)
    with pytest.raises(ValueError, match="without wrapping|int64"):
        expected_total_score_two_tier_from_fit(
            fit,
            grid,
            focal_primary=0,
            q_nuisance=5,
            specific_map=u64_pair,
            orthogonal_primary_identification=True,
        )
    ok_u = np.array([0], dtype=np.uint64)
    assert _as_specific_map_int64(ok_u, n_items=1, n_specific=1)[0] == 0


def test_specific_map_rejects_sparse_index_inflating_n_specific() -> None:
    """Omit-fit path must not allocate max(map)+1 when max >= n_items."""
    ap = np.array([[1.0], [1.0]])
    asp = np.array([0.5, 0.5])
    th = np.array([[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    grid = np.array([0.0])
    # Representable int64 index 100 with only 2 items ⇒ derived n_specific=101.
    with pytest.raises(ValueError, match="n_specific=101 > n_items=2"):
        _call(ap, asp, th, np.array([0, 100], dtype=np.int64), grid, q=5)


def test_collapsed_gh_matches_product_meshgrid_reference() -> None:
    """Adopted 1-D collapse agrees with explicit product mesh at q=21."""
    a_g, a_w, a_s = 1.1, 0.8, 0.6
    th = np.array([[1.5, 0.0, -1.2]])
    grid = np.linspace(-2.0, 2.0, 9)
    q = 21
    got = _call(
        np.array([[a_g, a_w]]),
        np.array([a_s]),
        th,
        np.array([0], dtype=np.int64),
        grid,
        q=q,
        primary_ref_mean=np.array([0.0, 0.1]),
        primary_ref_sd=np.array([1.0, 0.7]),
        specific_ref_mean=np.array([0.2]),
        specific_ref_sd=np.array([1.3]),
    )
    unit_nodes, unit_w = np.polynomial.hermite_e.hermegauss(q)
    unit_w = unit_w / unit_w.sum()
    w_nodes = 0.1 + 0.7 * unit_nodes
    s_nodes = 0.2 + 1.3 * unit_nodes
    Ww, Ss = np.meshgrid(w_nodes, s_nodes, indexing="ij")
    ww, ss = np.meshgrid(unit_w, unit_w, indexing="ij")
    wmesh = (ww * ss).ravel()
    wmesh = wmesh / wmesh.sum()
    offset = (a_w * Ww + a_s * Ss).ravel()
    base = a_g * grid[:, None] + offset[None, :]
    cell = PolytomousFit(
        model="grm",
        slope=np.ones(1),
        cat_params=th,
        loglik=float("nan"),
        n_iter=0,
        converged=True,
        termination_reason="mesh_ref",
    )
    expected = predict_expected_response_polytomous(cell, base.reshape(-1))
    mesh_total = (expected.reshape(base.shape) * wmesh[None, :]).sum(axis=1)
    assert np.allclose(got.expected_total, mesh_total, atol=1e-9, rtol=0.0)
