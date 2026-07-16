"""Tests for the paper-grounded additions: zero inflation, position covariate,
validation gates, IRTree expansion, DIF analysis, Vuong, Q3/GDDM, and ICs."""

from __future__ import annotations

import numpy as np
import pytest

from fast_mlsirm import (
    FitConfig,
    dif_analysis,
    dimensionality_residuals,
    fit,
    irtree_expand,
    validate_judge,
    vuong_nonnested,
)


def _sim_2pl(seed=0, P=600, I=12, shift_by_group=None, gid=None):
    rng = np.random.default_rng(seed)
    fid = np.zeros(I, dtype=np.int64)
    a = 0.8 + 0.6 * rng.random(I)
    b = np.linspace(-1.2, 1.2, I)
    theta = rng.standard_normal(P)
    if shift_by_group is not None:
        theta = theta + np.asarray(shift_by_group)[gid]
    eta = a[None, :] * theta[:, None] + b[None, :]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    return y, fid, a, b


def test_zero_inflation_via_public_api():
    y, fid, *_ = _sim_2pl(seed=1)
    y[:150] = 0.0  # structural zeros: 25%
    cfg = FitConfig(
        model="MLSRM", estimator="mmle", max_iter=60, latent_dim=1,
        q_theta=15, q_xi=7, zero_inflation=True, rust_device="cpu",
    )
    r = fit(y, fid, cfg)
    pop = r.population
    assert 0.1 < pop["pi_zero"] < 0.45
    assert pop["zero_responsibility"][:150].mean() > 0.6
    # ULSRM: (b + zeta) per item + tau + pi = 12*2 + 1 + 1
    assert r.ic is not None and r.ic["n_parameters"] == 26
    plain = fit(y, fid, FitConfig(
        model="MLSRM", estimator="mmle", max_iter=60, latent_dim=1,
        q_theta=15, q_xi=7, rust_device="cpu",
    ))
    assert r.loglik_trace[-1] > plain.loglik_trace[-1]
    # BIC prefers the mixture on mixture data (Kang-Cohen-Sung: BIC decides)
    assert r.ic["bic"] < plain.ic["bic"]


def test_position_covariate_via_public_api():
    rng = np.random.default_rng(3)
    P, I = 800, 10
    fid = np.zeros(I, dtype=np.int64)
    gid = np.arange(P) % 2
    w = np.zeros((2, I))
    w[0] = np.linspace(0, 1, I)
    w[1] = np.linspace(1, 0, I)
    theta = rng.standard_normal(P)
    b = np.linspace(-1, 1, I)
    delta_true = -0.9
    eta = theta[:, None] + b[None, :] + delta_true * w[gid]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    cfg = FitConfig(model="ULSRM", estimator="mmle", max_iter=80, latent_dim=1,
                    q_theta=15, q_xi=7, rust_device="cpu")
    r = fit(y, fid, cfg, group_id=gid, covariate={"w": w, "init_delta": 0.0})
    assert abs(r.population["delta"] - delta_true) < 0.4, r.population["delta"]
    with pytest.raises(ValueError, match="multilevel"):
        fit(y, fid, cfg, cluster_id=gid, covariate={"w": w})


def test_validation_gates():
    rng = np.random.default_rng(5)
    human = (rng.random(500) < 0.5).astype(np.uint32)
    good = human.copy()
    flip = rng.random(500) < 0.03
    good[flip] = 1 - good[flip]
    verdict = validate_judge(good, human, k=2)
    assert verdict.passed, verdict.failed_gates
    bad = human.copy()
    flip = rng.random(500) < 0.4
    bad[flip] = 1 - bad[flip]
    verdict_bad = validate_judge(bad, human, k=2)
    assert not verdict_bad.passed
    assert "qwk" in verdict_bad.failed_gates


def test_irtree_expand_linear_tree():
    # 3 categories, 2 nodes (linear tree): node0 = "beyond cat0",
    # node1 = "cat2 given beyond cat0" (off-path for cat0)
    mapping = np.array([[0.0, 1.0, 1.0], [np.nan, 0.0, 1.0]])
    y = np.array([[0, 2], [1, np.nan]])
    expanded, factor_id = irtree_expand(y, mapping)
    assert expanded.shape == (2, 4)
    # person 0: item0 cat0 -> node0=0, node1=NaN; item1 cat2 -> node0=1, node1=1
    np.testing.assert_array_equal(expanded[0], [0.0, 1.0, np.nan, 1.0])
    # person 1: item0 cat1 -> node0=1, node1=0; item1 missing -> NaN, NaN
    np.testing.assert_array_equal(expanded[1], [1.0, np.nan, 0.0, np.nan])
    np.testing.assert_array_equal(factor_id, [0, 0, 1, 1])
    with pytest.raises(ValueError, match="integer categories"):
        irtree_expand(np.array([[5.0]]), mapping)


def test_dif_analysis_detects_injected_shift():
    rng = np.random.default_rng(11)
    P, I = 900, 8
    fid = np.zeros(I, dtype=np.int64)
    gid = (np.arange(P) % 2).astype(np.int64)
    a = np.ones(I)
    b = np.linspace(-1, 1, I)
    theta = rng.standard_normal(P)
    eta = a[None, :] * theta[:, None] + b[None, :]
    eta[:, 3] += np.where(gid == 1, 1.2, 0.0)  # uniform DIF on item 3
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    cfg = FitConfig(model="ULSRM", estimator="mmle", max_iter=50, latent_dim=1,
                    q_theta=15, q_xi=7, rust_device="cpu")
    res = dif_analysis(y, fid, gid, config=cfg, studied_items=[2, 3])
    assert res.flagged_bh[3], f"item 3 must flag: p={res.p_value[3]}"
    assert res.effect_size[3] > 0.5
    assert not res.flagged_bh[2] or res.p_value[2] > res.p_value[3]


def test_dif_analysis_compacts_sparse_group_labels(monkeypatch):
    """Equivalent group partitions must have identical DIF bookkeeping."""
    import importlib
    from types import SimpleNamespace

    fit_module = importlib.import_module("fast_mlsirm.fit")
    seen_group_ids = []

    def fake_fit(y, factor_id, config, group_id=None, anchors=None, **_kwargs):
        seen_group_ids.append(np.asarray(group_id).copy())
        n_items = y.shape[1]
        return SimpleNamespace(
            params=SimpleNamespace(
                alpha=np.zeros(n_items),
                b=np.arange(n_items, dtype=float),
                zeta=np.zeros((n_items, 1)),
                tau=0.0,
            ),
            loglik_trace=[1.0 if anchors is not None else 0.0],
        )

    monkeypatch.setattr(fit_module, "fit", fake_fit)
    y = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = dif_analysis(
        y,
        np.zeros(2, dtype=np.int64),
        np.array([10, 20]),
        studied_items=[0],
    )

    assert result.df[0] == 2.0
    assert result.b_by_group.shape == (2, 2)
    assert result.effect_size[0] == 1.0
    assert all(np.array_equal(gid, [0, 1]) for gid in seen_group_ids)


def test_vuong_and_dimensionality_wrappers():
    y, fid, *_ = _sim_2pl(seed=13, P=400, I=10)
    cfg = FitConfig(model="ULSRM", estimator="mmle", max_iter=40, latent_dim=1,
                    q_theta=15, q_xi=7, rust_device="cpu", zero_inflation=False)
    r = fit(y, fid, cfg)
    # Vuong on synthetic casewise logliks
    la = -1.0 + 0.1 * np.random.default_rng(0).random(400)
    lb = la - 0.15 - 0.2 * (np.random.default_rng(1).random(400) - 0.5)
    v = vuong_nonnested(la, lb, 10, 10, bic_correction=False)
    assert v["z"] > 0 and 0 <= v["p_two_sided"] <= 1
    # residual diagnostics on a well-fitting model: modest Q3, small GDDM
    d = dimensionality_residuals(y, fid, r.params, r.model)
    assert d["q3"].shape[0] == 10 * 9 // 2
    assert d["q3_max_abs"] < 0.5
    assert d["gddm"] < 0.05


def test_bifactor_parity_and_recovery():
    rng = np.random.default_rng(21)
    P, I, D = 500, 10, 2
    fid = np.array([i % D for i in range(I)])
    lam = 0.6 + 0.8 * rng.random(I)
    b = np.linspace(-1, 1, I)
    g = rng.standard_normal(P)
    th = rng.standard_normal((P, D))
    eta = th[:, fid] + b[None, :] + lam[None, :] * g[:, None]
    y = (rng.random((P, I)) < 1 / (1 + np.exp(-eta))).astype(float)
    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model="BIFAC2PLM", estimator="mmle", max_iter=80, backend=backend,
            rust_device="cpu", latent_dim=1, q_theta=15, q_xi=15,
        )
        results[backend] = fit(y, fid, cfg)
    r, n = results["rust"], results["numpy"]
    np.testing.assert_allclose(r.params.b, n.params.b, atol=1e-9)
    np.testing.assert_allclose(r.params.zeta, n.params.zeta, atol=1e-9)
    np.testing.assert_allclose(r.loglik_trace[-1], n.loglik_trace[-1], atol=1e-9)
    # loadings track the truth
    c = np.corrcoef(r.params.zeta[:, 0], lam)[0, 1]
    assert abs(c) > 0.5, f"lambda recovery: {c}"
    # jmle guard
    import pytest as _pytest

    with _pytest.raises(NotImplementedError, match="marginal estimator"):
        fit(y, fid, FitConfig(model="BIFAC2PLM", estimator="jmle"))


def test_bifactor_covariate_parity_uses_inner_product_predictor():
    rng = np.random.default_rng(22)
    P, I, D = 400, 8, 2
    fid = np.arange(I) % D
    gid = np.arange(P) % 2
    w = np.empty((2, I))
    w[0] = np.linspace(0.0, 1.0, I)
    w[1] = w[0, ::-1]
    theta = rng.standard_normal((P, D))
    general = rng.standard_normal(P)
    loadings = np.linspace(0.7, 1.3, I)
    intercepts = np.linspace(-0.8, 0.8, I)
    delta_true = -1.0
    eta = (
        theta[:, fid]
        + intercepts[None, :]
        + loadings[None, :] * general[:, None]
        + delta_true * w[gid]
    )
    y = (rng.random((P, I)) < 1.0 / (1.0 + np.exp(-eta))).astype(float)

    results = {}
    for backend in ("rust", "numpy"):
        cfg = FitConfig(
            model="BIFAC2PLM",
            estimator="mmle",
            max_iter=30,
            backend=backend,
            rust_device="cpu",
            latent_dim=1,
            q_theta=7,
            q_xi=7,
        )
        results[backend] = fit(
            y,
            fid,
            cfg,
            group_id=gid,
            covariate={"w": w, "init_delta": 0.0},
        )

    rust_delta = results["rust"].population["delta"]
    numpy_delta = results["numpy"].population["delta"]
    assert rust_delta < -0.2
    np.testing.assert_allclose(rust_delta, numpy_delta, atol=1e-8)
    np.testing.assert_allclose(
        results["rust"].loglik_trace[-1],
        results["numpy"].loglik_trace[-1],
        atol=1e-8,
    )


def test_m2_rmsea2_parity_and_fit():
    # M2 limited-information GOF (Maydeu-Olivares & Joe): Rust core vs the
    # NumPy reference, plus a well-specified-vs-local-dependence contrast.
    from fast_mlsirm import fitstats

    y, fid, _a, _b = _sim_2pl(seed=5, P=1800, I=12)
    res = fit(y, fid, FitConfig(model="MIRT", estimator="mmle", max_iter=200,
                                backend="rust", rust_device="cpu"))

    core = fitstats.m2(y, fid, res.params, "MIRT", q_theta=21)
    ref = fitstats._m2_numpy(y, ~np.isnan(y), fid, res.params, "MIRT", 21, 11, 1e-8)

    # exact structural agreement
    assert core.n_moments == ref.n_moments == 78
    assert core.n_parameters == ref.n_parameters == 24
    assert core.df == ref.df == 54.0
    assert core.n_complete == ref.n_complete == 1800
    # numeric parity (hand Cholesky vs LAPACK solve): tight but not bit-exact
    np.testing.assert_allclose(core.m2, ref.m2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(core.rmsea2, ref.rmsea2, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.srmsr, ref.srmsr, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.null_m2, ref.null_m2, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(core.cfi, ref.cfi, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.tli, ref.tli, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(core.rmsea2_ci_lower, ref.rmsea2_ci_lower, atol=1e-6)
    np.testing.assert_allclose(core.rmsea2_ci_upper, ref.rmsea2_ci_upper, atol=1e-6)
    assert core.null_df == ref.null_df == 66.0
    assert core.rmsea == core.rmsea2
    assert core.srmr == core.srmsr
    expected_cfi = np.clip(
        1.0 - (core.m2 - core.df) / (core.null_m2 - core.null_df), 0.0, 1.0
    )
    expected_tli = (
        (core.null_m2 / core.null_df) - (core.m2 / core.df)
    ) / ((core.null_m2 / core.null_df) - 1.0)
    np.testing.assert_allclose(core.cfi, expected_cfi, atol=1e-12)
    np.testing.assert_allclose(core.tli, expected_tli, atol=1e-12)

    # The main diagnostics path exposes the global indices only when explicitly
    # requested, keeping the O(s^3) M2 solve out of ordinary JML diagnostics.
    from fast_mlsirm import fit_diagnostics

    diag = fit_diagnostics(
        y, res.params, fid, model="MIRT", include_m2=True, estimator="mmle"
    )
    for key in ("m2", "rmsea", "srmr", "cfi", "tli", "null_m2"):
        assert key in diag.model_fit and np.isfinite(diag.model_fit[key])
    np.testing.assert_allclose(diag.model_fit["rmsea"], core.rmsea2, atol=1e-12)
    np.testing.assert_allclose(diag.model_fit["srmr"], core.srmsr, atol=1e-12)

    # well specified: small RMSEA2, CI brackets the point estimate
    assert core.rmsea2 < 0.03
    assert core.rmsea2_ci_lower <= core.rmsea2 + 1e-9 <= core.rmsea2_ci_upper + 1e-9

    # inject local dependence (duplicate item) -> M2 and RMSEA2 inflate
    y_ld = y.copy()
    y_ld[:, 1] = y_ld[:, 0]
    res_ld = fit(y_ld, fid, FitConfig(model="MIRT", estimator="mmle", max_iter=200,
                                      backend="rust", rust_device="cpu"))
    ld = fitstats.m2(y_ld, fid, res_ld.params, "MIRT", q_theta=21)
    assert ld.m2 > core.m2
    assert ld.rmsea2 > 0.08
    assert ld.srmsr > core.srmsr
    assert ld.cfi < core.cfi
    assert ld.tli < core.tli

    # JMLE/CMLE estimates may still be inspected against an explicit marginal
    # evaluation population, but ordinary chi-square inference is not claimed.
    descriptive = fitstats.m2(
        y, fid, res.params, "MIRT", q_theta=21, estimator="jmle"
    )
    assert np.isfinite(descriptive.m2)
    assert not descriptive.inference_valid
    assert np.isnan(descriptive.p_value)
    assert np.isnan(descriptive.rmsea2_ci_lower)


def test_m2_singlefree_uses_only_estimated_calibration_columns():
    """FIPC M2 counts free-population columns and excludes anchored items."""
    from fast_mlsirm import fit_diagnostics
    from fast_mlsirm import fitstats
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(2718)
    n_persons, n_items = 1600, 8
    factor_id = np.zeros(n_items, dtype=np.int64)
    alpha = np.log(np.linspace(0.8, 1.4, n_items))
    b = np.linspace(-1.0, 1.0, n_items)
    population_mean = np.array([0.4])
    population_sd = np.array([1.2])
    theta = population_mean[0] + population_sd[0] * rng.standard_normal(n_persons)
    probability = 1.0 / (
        1.0 + np.exp(-(theta[:, None] * np.exp(alpha)[None, :] + b[None, :]))
    )
    responses = (rng.random(probability.shape) < probability).astype(float)
    params = MLSIRMParams(
        theta=theta[:, None],
        alpha=alpha,
        b=b,
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    fixed_items = np.arange(n_items) < 3

    ordinary = fitstats.m2(
        responses,
        factor_id,
        params,
        "MIRT",
        prior_mean=population_mean,
        prior_sd=population_sd,
    )
    singlefree = fitstats.m2(
        responses,
        factor_id,
        params,
        "MIRT",
        prior_mean=population_mean,
        prior_sd=population_sd,
        estimate_population=True,
        fixed_items=fixed_items,
    )

    # Ordinary 2PL estimates 2I item columns. FIPC fixes three item rows and
    # instead estimates the population mean and SD: 2*(8-3) + 2 = 12.
    assert ordinary.n_parameters == 16
    assert singlefree.n_parameters == 12
    assert singlefree.n_moments == 36
    assert singlefree.df == 24.0
    assert np.isfinite(singlefree.m2)
    assert np.isfinite(singlefree.p_value)
    assert "estimated mean/SD" in singlefree.inference_note

    diagnostics = fit_diagnostics(
        responses,
        params,
        factor_id,
        model="MIRT",
        include_m2=True,
        estimator="mmle",
        convergence_status="converged",
        population={
            "kind": "singlefree",
            "mu": population_mean[None, :],
            "sigma": population_sd[None, :],
            "fixed_items": fixed_items,
            "tau_fixed": False,
        },
    )
    assert diagnostics.model_fit["m2_df"] == 24.0
    assert diagnostics.model_fit["m2"] == singlefree.m2

    with pytest.raises(ValueError, match="fixed_items must have shape"):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MIRT",
            prior_mean=population_mean,
            prior_sd=population_sd,
            estimate_population=True,
            fixed_items=np.zeros(n_items - 1, dtype=bool),
        )
    with pytest.raises(ValueError, match="requires estimator='mmle'"):
        fitstats.m2(
            responses,
            factor_id,
            params,
            "MIRT",
            estimator="jmle",
            fixed_items=fixed_items,
        )


def test_m2_multigroup_and_multilevel_structures():
    """Population structure changes M2 moments/covariance, not just labels."""
    from fast_mlsirm import fit_diagnostics, m2_multigroup, m2_multilevel
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(123)
    n_items = 10
    n_per_group = 700
    factor_id = np.zeros(n_items, dtype=np.int64)
    alpha = np.log(np.linspace(0.8, 1.4, n_items))
    b = np.linspace(-1.0, 1.0, n_items)
    group_id = np.repeat(np.arange(2), n_per_group)
    means = np.array([[0.0], [0.6]])
    sds = np.array([[1.0], [1.2]])
    theta = means[group_id, 0] + sds[group_id, 0] * rng.standard_normal(group_id.size)
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * np.exp(alpha) + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((responses.shape[0], 1)),
        alpha=alpha,
        b=b,
        xi=np.zeros((responses.shape[0], 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    group_fit = m2_multigroup(
        responses, factor_id, params, "MIRT", group_id, means, sds
    )
    assert group_fit.n_groups == 2
    assert group_fit.n_parameters == 2 * n_items + 2
    assert group_fit.df == 88.0
    assert group_fit.null_df == 90.0
    assert np.isfinite(group_fit.m2)
    assert np.isfinite(group_fit.cfi)
    with pytest.raises(ValueError, match="non-negative integers"):
        m2_multigroup(
            responses,
            factor_id,
            params,
            "MIRT",
            group_id.astype(float) + 0.25,
            means,
            sds,
        )
    with pytest.raises(ValueError, match="mask must match"):
        m2_multigroup(
            responses,
            factor_id,
            params,
            "MIRT",
            group_id,
            means,
            sds,
            mask=np.ones((1, 1), dtype=bool),
        )
    group_diagnostics = fit_diagnostics(
        responses,
        params,
        factor_id,
        model="MIRT",
        group_id=group_id,
        include_m2=True,
        estimator="mmle",
        population={"kind": "multigroup", "mu": means, "sigma": sds},
    )
    np.testing.assert_allclose(group_diagnostics.model_fit["m2"], group_fit.m2)
    assert group_diagnostics.model_fit["m2_n_groups"] == 2.0

    locally_dependent = responses.copy()
    locally_dependent[:, 1] = locally_dependent[:, 0]
    group_ld = m2_multigroup(
        locally_dependent, factor_id, params, "MIRT", group_id, means, sds
    )
    assert group_ld.m2 > group_fit.m2
    assert group_ld.cfi < group_fit.cfi
    assert group_ld.tli < group_fit.tli

    # Random-intercept data: the effective covariance comes from independent
    # cluster totals. There must be more clusters than retained M2 moments.
    n_items = 8
    n_clusters = 70
    cluster_size = 12
    sigma_u = 0.7
    cluster_id = np.repeat(np.arange(n_clusters), cluster_size)
    alpha = np.log(np.linspace(0.9, 1.3, n_items))
    b = np.linspace(-0.8, 0.8, n_items)
    u = np.repeat(sigma_u * rng.standard_normal(n_clusters), cluster_size)
    theta = rng.standard_normal(cluster_id.size) + u
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] * np.exp(alpha) + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    params = MLSIRMParams(
        theta=np.zeros((responses.shape[0], 1)),
        alpha=alpha,
        b=b,
        xi=np.zeros((responses.shape[0], 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    cluster_fit = m2_multilevel(
        responses, np.zeros(n_items, dtype=np.int64), params, "MIRT",
        cluster_id, sigma_u,
    )
    assert cluster_fit.n_clusters == n_clusters
    assert cluster_fit.n_parameters == 2 * n_items + 1
    assert cluster_fit.df == 19.0
    assert np.isfinite(cluster_fit.m2)
    assert np.isfinite(cluster_fit.p_value)
    assert "cluster-robust" in cluster_fit.inference_note
    cluster_diagnostics = fit_diagnostics(
        responses,
        params,
        np.zeros(n_items, dtype=np.int64),
        model="MIRT",
        cluster_id=cluster_id,
        include_m2=True,
        estimator="mmle",
        population={"kind": "multilevel", "sigma_u": sigma_u},
    )
    np.testing.assert_allclose(cluster_diagnostics.model_fit["m2"], cluster_fit.m2)
    assert cluster_diagnostics.model_fit["m2_n_clusters"] == float(n_clusters)


def test_m2_multilevel_integrates_one_shared_intercept_across_dimensions():
    """The scalar cluster effect induces cross-factor covariance."""
    from fast_mlsirm.fitstats import _m2_group_components
    from fast_mlsirm.types import MLSIRMParams

    factor_id = np.array([0, 0, 1, 1], dtype=np.int64)
    responses = np.zeros((20, 4), dtype=float)
    params = MLSIRMParams(
        theta=np.zeros((20, 2)),
        alpha=np.zeros(4),
        b=np.zeros(4),
        xi=np.zeros((20, 1)),
        zeta=np.zeros((4, 1)),
        tau=0.0,
    )
    common = dict(
        y0=responses,
        observed0=np.ones_like(responses, dtype=bool),
        d_of_i=factor_id,
        params=params,
        model="MIRT",
        q_theta=15,
        q_xi=7,
        eps_distance=1e-8,
        prior_mean=np.zeros(2),
        prior_sd=np.ones(2),
    )
    independent = _m2_group_components(**common)
    shared = _m2_group_components(**common, shared_sigma_u=0.8, q_u=15)

    # Pair order after four univariate moments is (0,1), (0,2), ... .
    cross_factor_pair = 5
    assert abs(independent["mom"][cross_factor_pair] - 0.25) < 1e-12
    assert shared["mom"][cross_factor_pair] > independent["mom"][cross_factor_pair]
    assert np.any(np.abs(shared["delta_shared"]) > 1e-8)


def test_m2_cmle_rasch_conditions_out_person_ability():
    """CMLE M2 uses raw-score conditional moments, not a Gaussian prior."""
    from fast_mlsirm import m2
    from fast_mlsirm.types import MLSIRMParams

    rng = np.random.default_rng(9)
    n_persons, n_items = 3000, 8
    b = np.linspace(-1.3, 1.3, n_items)
    theta = rng.standard_normal(n_persons)
    prob = 1.0 / (1.0 + np.exp(-(theta[:, None] + b)))
    responses = (rng.random(prob.shape) < prob).astype(float)
    assert np.all(np.bincount(responses.sum(axis=1).astype(int), minlength=n_items + 1) > 0)
    params = MLSIRMParams(
        theta=theta[:, None],
        alpha=np.zeros(n_items),
        b=b,
        xi=np.zeros((n_persons, 1)),
        zeta=np.zeros((n_items, 1)),
        tau=0.0,
    )
    result = m2(
        responses,
        np.zeros(n_items, dtype=np.int64),
        params,
        "MIRT",
        estimator="cmle",
    )
    assert result.estimator == "cmle"
    assert result.inference_valid
    assert result.n_parameters == (n_items - 1) + n_items
    assert result.df == 21.0
    assert result.p_value > 0.05
    assert result.rmsea2 < 0.02

    non_rasch = MLSIRMParams(
        theta=params.theta,
        alpha=np.full(n_items, 0.1),
        b=b,
        xi=params.xi,
        zeta=params.zeta,
        tau=0.0,
    )
    with pytest.raises(ValueError, match="Rasch"):
        m2(
            responses,
            np.zeros(n_items, dtype=np.int64),
            non_rasch,
            "MIRT",
            estimator="cmle",
        )


def test_irt_link_recovers_known_transform():
    # separately-calibrated common-item linking (Kolen & Brennan; Haebara;
    # Stocking-Lord): recover a known theta_old = A*theta_new + B.
    from fast_mlsirm import irt_link

    a_old = np.array([1.2, 0.8, 1.5, 1.0, 0.9, 1.3, 1.1, 0.7])
    b_old = np.array([-0.5, 0.3, 1.0, -1.2, 0.0, 0.6, -0.8, 0.4])
    A0, B0 = 1.3, 0.4
    a_new = A0 * a_old
    b_new = b_old + a_old * B0
    for method in ("mean_mean", "mean_sigma", "haebara", "stocking_lord"):
        r = irt_link(a_old, b_old, a_new, b_new, method=method)
        assert abs(r.slope - A0) < 1e-3, f"{method}: slope {r.slope}"
        assert abs(r.intercept - B0) < 1e-3, f"{method}: intercept {r.intercept}"
        assert r.converged
        if method in {"haebara", "stocking_lord"}:
            assert r.termination_reason == "tolerance_met"
            assert r.n_iter < r.max_iter
            assert r.final_objective_span <= r.objective_tolerance
            assert r.final_parameter_span <= r.parameter_tolerance
        else:
            assert r.termination_reason == "closed_form"
            assert r.n_iter == r.max_iter == 0
    import pytest as _pytest

    with _pytest.raises(Exception):
        irt_link(a_old, b_old, a_new, b_new, method="not_a_method")
    with _pytest.raises(ValueError, match="non-zero difficulty spread"):
        irt_link(
            np.ones(3),
            np.array([-0.5, 0.0, 0.5]),
            np.ones(3),
            np.zeros(3),
            method="mean_sigma",
        )
    with _pytest.raises(ValueError, match="integer quadrature size"):
        irt_link(a_old, b_old, a_new, b_new, q_theta=21.5)


def test_category_logprobs_binary_parity_and_gpcm_monotone():
    """The unified GPCM/nominal cell nests binary 2PL (bit-parity check) and is
    a proper log-softmax; GPCM scores make higher `base` favor higher categories.
    Parity reference for the Rust polytomous kernel (design spec)."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import category_logprobs

    rng = np.random.default_rng(0)
    base = rng.normal(size=32)
    b = 0.3

    # binary 2PL: logP_1 == log_sigmoid(base + b), logP_0 == log_sigmoid(-(base+b))
    lp = category_logprobs(base, [0.0, 1.0], [0.0, b])
    assert np.allclose(np.exp(lp).sum(axis=-1), 1.0, atol=1e-12)
    eta = base + b
    assert np.allclose(lp[:, 1], -np.logaddexp(0.0, -eta), atol=1e-12)
    assert np.allclose(lp[:, 0], -np.logaddexp(0.0, eta), atol=1e-12)

    # GPCM (scores 0,1,2): larger base shifts mass to the top category
    lp3 = category_logprobs(np.array([-2.0, 2.0]), [0.0, 1.0, 2.0], [0.0, 0.0, 0.0])
    assert np.allclose(np.exp(lp3).sum(axis=-1), 1.0, atol=1e-12)
    p_top = np.exp(lp3[:, 2])
    assert p_top[1] > p_top[0]

    # baseline must be pinned
    import pytest
    with pytest.raises(ValueError):
        category_logprobs(base, [0.5, 1.0], [0.0, b])


def test_gpcm_node_gradient_matches_finite_difference():
    """The analytic M-step gradient of the GPCM/nominal cell (category residual,
    score-weighted base residual, and nominal-score gradient) matches central
    finite differences — de-risks the Rust M-step before it is written."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import category_logprobs, gpcm_node_gradient

    scores = np.array([0.0, 1.0, 2.0, 3.0])
    intercepts = np.array([0.0, 0.2, -0.1, 0.3])
    counts = np.array([3.0, 5.0, 2.0, 4.0])
    base = 0.4

    def q(b, ic, sc):
        return float(np.dot(counts, category_logprobs(b, sc, ic)))

    g_ic, g_base, g_sc = gpcm_node_gradient(base, scores, intercepts, counts)
    h = 1e-6
    for m in range(1, 4):
        ic_p, ic_m = intercepts.copy(), intercepts.copy()
        ic_p[m] += h
        ic_m[m] -= h
        assert abs((q(base, ic_p, scores) - q(base, ic_m, scores)) / (2 * h) - g_ic[m - 1]) < 1e-5
        sc_p, sc_m = scores.copy(), scores.copy()
        sc_p[m] += h
        sc_m[m] -= h
        assert abs((q(base, intercepts, sc_p) - q(base, intercepts, sc_m)) / (2 * h) - g_sc[m - 1]) < 1e-5
    assert abs((q(base + h, intercepts, scores) - q(base - h, intercepts, scores)) / (2 * h) - g_base) < 1e-5

    # residual closure
    p = np.exp(category_logprobs(base, scores, intercepts))
    assert abs((counts - counts.sum() * p).sum()) < 1e-12


def test_fit_gpcm_numpy_recovers_known_parameters():
    """Unidimensional GPCM MMLE-EM (the polytomous parity reference) recovers
    known slopes and category intercepts from simulated data."""
    import numpy as np
    from fast_mlsirm.estimators.marginal import category_logprobs, fit_gpcm_numpy

    rng = np.random.default_rng(7)
    n_persons, n_items, k_cat = 4000, 6, 3
    a_true = rng.uniform(0.8, 1.8, n_items)
    c_true = np.zeros((n_items, k_cat))
    c_true[:, 1:] = rng.normal(0.0, 1.0, (n_items, k_cat - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k_cat, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k_cat, p=p[pp])

    res = fit_gpcm_numpy(y, k_cat, max_iter=80)
    assert np.isfinite(res["loglik"])
    assert res["converged"]
    assert res["termination_reason"] == "tolerance"
    assert res["n_iter"] < 80
    assert res["loglik_trace"].shape == (res["n_iter"] + 1,)
    assert np.all(np.isfinite(res["loglik_trace"]))
    assert np.all(np.diff(res["loglik_trace"]) >= -1e-10)
    assert res["final_delta"] <= res["stopping_tolerance"]
    assert res["loglik"] == res["loglik_trace"][-1]
    assert np.corrcoef(a_true, res["a"])[0, 1] > 0.9
    assert np.max(np.abs(a_true - res["a"])) < 0.35
    assert np.mean(np.abs(c_true[:, 1:] - res["intercepts"][:, 1:])) < 0.2


def test_fit_gpcm_numpy_reports_likelihood_at_returned_parameters():
    """The reference EM result and trace end at the returned parameter state."""
    import numpy as np

    from fast_mlsirm.estimators.marginal import _gh, category_logprobs, fit_gpcm_numpy

    y = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 1],
            [1, 2, 1],
            [2, 2, 2],
            [2, 1, 2],
            [1, 0, 1],
            [2, 2, 1],
        ],
        dtype=np.int64,
    )
    res = fit_gpcm_numpy(y, 3, q_theta=7, max_iter=1, tol=1e-6)

    nodes, weights = _gh(7)
    scores = np.arange(3, dtype=np.float64)
    log_node = np.zeros((y.shape[0], nodes.size), dtype=np.float64)
    for item in range(y.shape[1]):
        item_lp = category_logprobs(
            res["a"][item] * nodes, scores, res["intercepts"][item]
        )
        log_node += item_lp[:, y[:, item]].T
    log_node += np.log(weights)[None, :]
    maximum = log_node.max(axis=1, keepdims=True)
    reevaluated = float(
        np.sum(maximum[:, 0] + np.log(np.exp(log_node - maximum).sum(axis=1)))
    )

    assert res["n_iter"] == 1
    assert not res["converged"]
    assert res["termination_reason"] == "max_iter_reached"
    assert res["loglik_trace"].shape == (2,)
    assert np.all(np.isfinite(res["loglik_trace"]))
    assert res["final_delta"] > res["stopping_tolerance"]
    assert np.allclose(res["loglik"], res["loglik_trace"][-1], atol=1e-12)
    assert np.allclose(res["loglik"], reevaluated, atol=1e-12)


def test_fit_gpcm_numpy_rejects_malformed_controls_and_responses():
    import numpy as np
    import pytest

    from fast_mlsirm.estimators.marginal import fit_gpcm_numpy

    valid = np.array([[0, 1], [1, 2]], dtype=np.int64)
    for bad in (np.array([0, 1, 2]), np.empty((0, 2)), np.array([[0.5, 1.0]])):
        with pytest.raises(ValueError):
            fit_gpcm_numpy(bad, 3, q_theta=7, max_iter=1)
    for kwargs in (
        {"n_cat": 3.5},
        {"n_cat": 3, "max_iter": 0},
        {"n_cat": 3, "tol": 0.0},
    ):
        with pytest.raises(ValueError):
            fit_gpcm_numpy(valid, **kwargs)


def test_poly_cell_and_fitter_rust_numpy_parity():
    """The Rust polytomous cell matches the NumPy reference bit-for-bit, and the
    Rust unidimensional GPCM fitter agrees with the NumPy mirror on recovery."""
    import numpy as np
    import pytest
    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "fit_poly_unidim"):  # pragma: no cover
        pytest.skip("core built without polytomous functions")
    from fast_mlsirm.estimators.marginal import category_logprobs, fit_gpcm_numpy

    # cell parity: same softmax formula in both languages
    scores = np.array([0.0, 1.0, 2.0])
    intercepts = np.array([0.0, 0.3, -0.2])
    for base in (-1.3, 0.0, 0.75):
        rust = np.array(_core.gpcm_cell_logprobs(float(base), scores, intercepts))
        npy = category_logprobs(np.array([base]), scores, intercepts)[0]
        assert np.allclose(rust, npy, atol=1e-12), f"cell parity at base={base}"

    # fitter agreement: same EM/Newton, same GH grid -> same MLE
    rng = np.random.default_rng(11)
    n_persons, n_items, k = 2500, 5, 3
    a_true = rng.uniform(0.8, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.8, (n_items, k - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    y = np.zeros((n_persons, n_items), dtype=np.int64)
    for i in range(n_items):
        p = np.exp(category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    rust_fit = _core.fit_poly_unidim(y.ravel(), n_persons, n_items, k, None, "gpcm", 21, 80, 1e-6)
    npy_fit = fit_gpcm_numpy(y, k)
    assert np.allclose(np.array(rust_fit["slope"]), npy_fit["a"], atol=0.05)
    assert np.isfinite(rust_fit["loglik"])


def test_fit_polytomous_api_recovers_and_validates():
    """The public fit_polytomous wrapper (Rust compute) recovers GRM parameters
    and rejects malformed input."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous
    from fast_mlsirm.estimators.marginal import _gh

    try:
        from fast_mlsirm import _core  # noqa: F401
        if not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
            pytest.skip("core built without polytomous functions")
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")

    from fast_mlsirm.polytomous import _core_module
    if _core_module() is None:  # pragma: no cover
        pytest.skip("compiled core not available")

    # GRM recovery via the public API
    rng = np.random.default_rng(3)
    n_persons, n_items, k = 3000, 5, 4
    a_true = rng.uniform(0.9, 1.6, n_items)
    thr_true = np.array([1.3, 0.0, -1.3])
    nodes, _ = _gh(21)
    theta = rng.normal(0.0, 1.0, n_persons)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        for pp in range(n_persons):
            eta = a_true[i] * theta[pp] + thr_true  # cumulative logits P(Y>=k)
            cum = 1.0 / (1.0 + np.exp(-eta))
            p = np.empty(k)
            p[0] = 1 - cum[0]
            p[1:k - 1] = cum[:k - 2] - cum[1:k - 1]
            p[k - 1] = cum[k - 2]
            y[pp, i] = rng.choice(k, p=p / p.sum())
    fit = fit_polytomous(y, k, model="grm")
    assert fit.model == "grm" and np.isfinite(fit.loglik)
    assert fit.converged
    assert fit.termination_reason == "tolerance"
    assert fit.n_iter < 80
    assert fit.loglik_trace.shape == (fit.n_iter + 1,)
    assert fit.loglik == fit.loglik_trace[-1]
    assert fit.final_delta >= -32 * np.finfo(float).eps * (
        1 + abs(fit.loglik_trace[-2])
    )
    assert fit.final_delta <= fit.stopping_tolerance
    assert np.corrcoef(a_true, fit.slope)[0, 1] > 0.9

    # validation
    with pytest.raises(ValueError):
        fit_polytomous(y, k, model="nominal")       # unsupported model
    with pytest.raises(ValueError):
        fit_polytomous(y.astype(float) + 0.5, k)    # non-integer categories
    with pytest.raises(ValueError):
        fit_polytomous(y, 2)                          # category out of range
    with pytest.raises(ValueError):
        fit_polytomous(y, k, max_iter=0)
    with pytest.raises(ValueError):
        fit_polytomous(y, k, tol=np.nan)


def test_score_polytomous_recovers_theta():
    """fit_polytomous -> score_polytomous round-trip: EAP trait scores correlate
    with true theta (Rust compute end to end)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, score_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "score_poly_eap"):
        pytest.skip("compiled core without polytomous scoring")

    rng = np.random.default_rng(5)
    n_persons, n_items, k = 3000, 8, 3
    a_true = rng.uniform(0.9, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta_true = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(category_logprobs(a_true[i] * theta_true, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm")
    sc = score_polytomous(y, fit)
    assert sc["theta_eap"].shape == (n_persons,)
    assert np.all(sc["theta_sd"] > 0)
    assert np.corrcoef(theta_true, sc["theta_eap"])[0, 1] > 0.8


def test_score_polytomous_rejects_malformed_scoring_contract():
    """Scoring must not truncate quadrature controls or emit non-finite scores."""
    import numpy as np
    import pytest

    from fast_mlsirm import score_polytomous
    from fast_mlsirm.polytomous import PolytomousFit, _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "score_poly_eap"):
        pytest.skip("compiled core without polytomous scoring")

    fit = PolytomousFit(
        model="gpcm",
        slope=np.array([1.0]),
        cat_params=np.array([[0.2, -0.3]]),
        loglik=0.0,
        n_iter=0,
    )
    with pytest.raises(ValueError, match="q_theta must be one of"):
        score_polytomous(np.array([[0.0]]), fit, q_theta=21.5)

    fit.slope[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        score_polytomous(np.array([[1.0]]), fit)


def test_grm_cell_rust_numpy_parity():
    """The Rust GRM cumulative-logit cell matches the NumPy reference to 1e-12,
    and the NumPy GRM cell is a proper (normalized) log-distribution."""
    import numpy as np
    import pytest
    from fast_mlsirm.estimators.marginal import grm_category_logprobs

    # NumPy self-consistency: normalization + binary reduction
    for base in (-1.0, 0.3, 1.7):
        lp = grm_category_logprobs(np.array([base]), np.array([1.0, -1.0]))[0]
        assert abs(np.log(np.exp(lp).sum())) < 1e-12
    # binary GRM (K=2): P(Y=1) = sigmoid(base + beta)
    lp2 = grm_category_logprobs(np.array([0.4]), np.array([0.2]))[0]
    assert abs(lp2[1] - (-np.logaddexp(0.0, -(0.4 + 0.2)))) < 1e-12

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    if not hasattr(_core, "grm_cell_logprobs"):  # pragma: no cover
        pytest.skip("core built without grm cell")
    thr = np.array([1.3, 0.1, -1.2])
    for base in (-1.4, 0.0, 0.9):
        rust = np.array(_core.grm_cell_logprobs(float(base), thr))
        npy = grm_category_logprobs(np.array([base]), thr)[0]
        assert np.allclose(rust, npy, atol=1e-12), f"grm parity at base={base}"


def test_grm_cell_extreme_predictor_stays_finite():
    from fast_mlsirm.estimators.marginal import grm_category_logprobs

    thresholds = np.array([1.0, 0.0])
    expected_middle = -1000.0 + np.log1p(-np.exp(-1.0))
    npy = grm_category_logprobs(np.array([1000.0]), thresholds)[0]
    assert np.all(np.isfinite(npy)), npy
    np.testing.assert_allclose(npy[1], expected_middle, atol=1e-12)

    try:
        from fast_mlsirm import _core
    except Exception:  # pragma: no cover
        pytest.skip("compiled core not available")
    rust = np.asarray(_core.grm_cell_logprobs(1000.0, thresholds))
    assert np.all(np.isfinite(rust)), rust
    np.testing.assert_allclose(rust, npy, atol=1e-12)


def test_information_polytomous_api():
    """information_polytomous returns positive item/test information curves whose
    test info equals the item-info row sum (Rust compute)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, information_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_information_curves"):
        pytest.skip("compiled core without polytomous information")

    rng = np.random.default_rng(8)
    n_persons, n_items, k = 1500, 5, 3
    a_true = rng.uniform(1.0, 1.5, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.4, (n_items, k - 1))
    theta_p = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(category_logprobs(a_true[i] * theta_p, scores, c_true[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm")
    grid = np.linspace(-3, 3, 25)
    info = information_polytomous(fit, grid)
    assert info["item_info"].shape == (25, n_items)
    assert np.all(info["item_info"] >= 0) and np.all(info["test_info"] > 0)
    assert np.allclose(info["test_info"], info["item_info"].sum(axis=1))
    # information is highest in the interior for well-centered items
    assert info["test_info"].argmax() not in (0, 24)


def test_fit_polytomous_handles_missing_data():
    """fit_polytomous marginalizes NaN (missing) responses and still recovers
    slopes; score_polytomous accepts partially-missing rows."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, score_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
        pytest.skip("compiled core not available")

    rng = np.random.default_rng(21)
    n_persons, n_items, k = 5000, 6, 3
    a_true = rng.uniform(0.9, 1.6, n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta = rng.normal(0.0, 1.0, n_persons)
    scores = np.arange(k, dtype=float)
    y = np.full((n_persons, n_items), np.nan)
    for i in range(n_items):
        p = np.exp(category_logprobs(a_true[i] * theta, scores, c_true[i]))
        for pp in range(n_persons):
            if rng.random() < 0.25:            # ~25% MCAR missing -> stays NaN
                continue
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm")
    assert np.isfinite(fit.loglik)
    assert np.corrcoef(a_true, fit.slope)[0, 1] > 0.9
    sc = score_polytomous(y, fit)
    assert sc["theta_eap"].shape == (n_persons,) and np.all(np.isfinite(sc["theta_eap"]))


def test_fit_lsirm_polytomous_recovers_positions():
    """The latent-space polytomous LSIRM (Rust compute) recovers item positions
    (distance-matrix RMSE) and slopes (RMSE), and returns person scores."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_lsirm_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_lsirm"):
        pytest.skip("compiled core without polytomous LSIRM")

    rng = np.random.default_rng(4)
    n_persons, n_items, k, ld = 1000, 6, 3, 2
    # two separated item clusters
    zeta_true = np.zeros((n_items, ld))
    for i in range(n_items):
        zeta_true[i, 0] = (-1.2 if i < n_items // 2 else 1.2) + 0.3 * rng.standard_normal()
        zeta_true[i, 1] = 0.3 * rng.standard_normal()
    a_true = 1.0 + 0.1 * np.arange(n_items)
    c_true = np.zeros((n_items, k))
    c_true[:, 1:] = np.array([0.2, -0.2])
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for p in range(n_persons):
        theta = rng.standard_normal()
        xi = rng.standard_normal(ld)
        for i in range(n_items):
            base = a_true[i] * theta - np.sqrt(1e-8 + np.sum((xi - zeta_true[i]) ** 2))
            pr = np.exp(category_logprobs(np.array([base]), scores, c_true[i])[0])
            y[p, i] = rng.choice(k, p=pr / pr.sum())

    fit = fit_lsirm_polytomous(y, k, latent_dim=ld, model="gpcm", q_theta=7, q_xi=7, max_iter=30)
    assert fit.zeta.shape == (n_items, ld)
    assert fit.theta_eap.shape == (n_persons,) and fit.xi_eap.shape == (n_persons, ld)
    assert np.all(fit.theta_sd > 0) and np.isfinite(fit.loglik)

    def dmat(z):
        return np.array([np.linalg.norm(z[i] - z[j]) for i in range(n_items) for j in range(i + 1, n_items)])

    def rmse(u, v):
        return float(np.sqrt(np.mean((u - v) ** 2)))

    assert rmse(a_true, fit.slope) < 0.3, "slope RMSE"
    assert rmse(dmat(zeta_true), dmat(fit.zeta)) < 0.7, "position distance-matrix RMSE"


def test_polytomous_information_criteria():
    """Kang-Cohen-Sung (2009) model-selection indices for a polytomous fit:
    correct free-parameter count and finite, ordered indices."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, polytomous_information_criteria
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_poly_unidim"):
        pytest.skip("compiled core not available")

    rng = np.random.default_rng(6)
    n_persons, n_items, k = 1500, 5, 3
    a = rng.uniform(0.9, 1.5, n_items)
    c = np.zeros((n_items, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (n_items, k - 1))
    theta = rng.standard_normal(n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm")
    ic = polytomous_information_criteria(fit, n_persons)
    # slope (n_items) + intercepts (n_items*(K-1)) = n_items*K
    assert ic["n_parameters"] == n_items * k
    for key in ("aic", "bic", "caic", "aicc", "sabic"):
        assert np.isfinite(ic[key])
    # BIC/CAIC penalize free parameters more heavily than AIC at N=1500
    assert ic["aic"] < ic["bic"] < ic["caic"]
    with pytest.raises(ValueError):
        polytomous_information_criteria(fit, 1)


def test_item_fit_polytomous_sx2():
    """Generalized S-X² polytomous item fit (Kang & Chen, 2008, 2011) through the
    public API: well-formed per-item output, calibration at the fitted model
    (statistic tracks df, few false flags), and input validation."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, item_fit_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_item_fit_sx2"
    ):
        pytest.skip("compiled core built without poly_item_fit_sx2")

    rng = np.random.default_rng(11)
    n_persons, n_items, k = 1500, 8, 4
    a = rng.uniform(0.9, 1.5, n_items)
    c = np.zeros((n_items, k))
    c[:, 1:] = rng.normal(0.0, 0.6, (n_items, k - 1))
    theta = rng.standard_normal(n_persons)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n_persons, n_items), dtype=int)
    for i in range(n_items):
        p = np.exp(category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n_persons):
            y[pp, i] = rng.choice(k, p=p[pp])

    fit = fit_polytomous(y, k, model="gpcm")
    res = item_fit_polytomous(y, fit, q_theta=21)
    for key in ("statistic", "df", "p_value", "n_cells"):
        assert res[key].shape == (n_items,)
    assert np.all(np.isfinite(res["statistic"]))
    # df is the retained cell count minus m = n_cat item parameters
    assert np.array_equal(res["df"].astype(int), res["n_cells"] - k)
    finite_p = res["p_value"][np.isfinite(res["p_value"])]
    assert np.all((finite_p >= 0.0) & (finite_p <= 1.0))
    # a correctly fitted model is rarely flagged (contrast: G2 flags >30%)
    assert np.mean(finite_p < 0.05) < 0.30
    # statistic ~ df at the fitted parameters
    ratio = res["statistic"].sum() / res["df"].sum()
    assert 0.6 < ratio < 1.6, f"S-X2/df ratio off: {ratio}"

    # missing data (NaN) is marginalized: complete-case summed-score table
    y_miss = y.astype(float)
    y_miss[rng.random(y_miss.shape) < 0.05] = np.nan
    res_miss = item_fit_polytomous(y_miss, fit)
    assert np.all(np.isfinite(res_miss["statistic"]))

    # validation
    with pytest.raises(ValueError):
        item_fit_polytomous(y[:, :-1], fit)                 # wrong item count
    with pytest.raises(ValueError):
        item_fit_polytomous(y, fit, min_expected=0.0)       # non-positive floor


def test_m2_polytomous():
    """Polytomous M2 (Maydeu-Olivares & Joe, 2014) through the public API:
    correct moment/df bookkeeping, a good fit for correctly-specified data, and
    detection of a strongly misspecified (skewed-population) fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, m2_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_m2"):
        pytest.skip("compiled core built without poly_m2")

    def sim(theta, a, c, k):
        n, j = theta.size, a.size
        y = np.zeros((n, j), dtype=int)
        scores = np.arange(k, dtype=float)
        for i in range(j):
            p = np.exp(category_logprobs(a[i] * theta, scores, c[i]))
            for pp in range(n):
                y[pp, i] = np.random.default_rng(1000 + i * n + pp).choice(k, p=p[pp])
        return y

    rng = np.random.default_rng(7)
    n, j, k = 2000, 6, 3
    a = rng.uniform(0.9, 1.5, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))

    # correctly specified (normal ability) -> good fit
    theta = rng.standard_normal(n)
    y = sim(theta, a, c, k)
    fit = fit_polytomous(y, k, model="gpcm")
    res = m2_polytomous(y, fit)
    # Q = j*(k-1) + C(j,2)*(k-1)^2 ; P = j*k ; df = Q - P
    q = j * (k - 1) + (j * (j - 1) // 2) * (k - 1) ** 2
    assert res["n_moments"] == q
    assert res["n_parameters"] == j * k
    assert res["df"] == q - j * k
    assert res["null_df"] == q - j * (k - 1)
    assert np.isfinite(res["m2"]) and res["m2"] >= 0.0
    assert 0.0 <= res["p_value"] <= 1.0
    assert res["rmsea2_ci_lower"] <= res["rmsea2_ci_upper"] + 1e-9
    assert res["rmsea2"] < 0.05  # well-fitting
    assert res["null_m2"] > res["m2"]
    expected_cfi = np.clip(
        1.0 - (res["m2"] - res["df"]) / (res["null_m2"] - res["null_df"]),
        0.0,
        1.0,
    )
    expected_tli = (
        res["null_m2"] / res["null_df"] - res["m2"] / res["df"]
    ) / (res["null_m2"] / res["null_df"] - 1.0)
    np.testing.assert_allclose(res["cfi"], expected_cfi, atol=1e-12)
    np.testing.assert_allclose(res["tli"], expected_tli, atol=1e-12)

    # strongly misspecified: fit the wrong item parameters -> M2 must reject
    bad = fit
    bad.slope = a * 3.0  # inflate discriminations far from the truth
    res_bad = m2_polytomous(y, bad)
    assert res_bad["m2"] > res["m2"]
    assert res_bad["p_value"] < 0.05
    assert res_bad["cfi"] < res["cfi"]
    assert res_bad["tli"] < res["tli"]

    # validation: fewer than 3 items has non-positive df
    with pytest.raises((ValueError, RuntimeError)):
        fit2 = fit_polytomous(y[:, :2], k, model="gpcm")
        m2_polytomous(y[:, :2], fit2)


def test_m2_polytomous_rejects_nonconverged_calibration():
    """Finite parameters are not sufficient evidence for M2 inference."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, m2_polytomous
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_m2"
    ):
        pytest.skip("compiled core built without poly_m2")

    rng = np.random.default_rng(20260716)
    y = rng.integers(0, 3, size=(240, 6))
    fit = fit_polytomous(
        y,
        3,
        model="gpcm",
        q_theta=11,
        max_iter=1,
        tol=1e-12,
    )

    assert fit.converged is False
    assert fit.termination_reason == "max_iter"
    assert fit.n_iter == 1
    assert fit.final_delta > fit.stopping_tolerance
    assert np.all(np.isfinite(fit.slope))
    assert np.all(np.isfinite(fit.cat_params))
    with pytest.raises(RuntimeError, match="requires a converged fit"):
        m2_polytomous(y, fit, q_theta=11)


def test_local_dependence_polytomous():
    """Item-pair local dependence (Chen & Thissen, 1997) through the public API:
    correct per-pair bookkeeping, calibrated (few flags) for a locally
    independent fit, and detection of an injected testlet pair."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, local_dependence_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_local_dependence"
    ):
        pytest.skip("compiled core built without poly_local_dependence")

    def sim(theta, a, c, k, testlet=None):
        n, j = theta.size, a.size
        y = np.zeros((n, j), dtype=int)
        scores = np.arange(k, dtype=float)
        for i in range(j):
            base = a[i] * theta + (testlet if testlet is not None and i in (0, 1) else 0.0)
            p = np.exp(category_logprobs(base, scores, c[i]))
            for pp in range(n):
                y[pp, i] = np.random.default_rng(300 + i * n + pp).choice(k, p=p[pp])
        return y

    rng = np.random.default_rng(4)
    n, j, k = 1500, 5, 3
    a = rng.uniform(0.9, 1.4, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))

    # locally independent -> calibrated (the reference is conservative)
    theta = rng.standard_normal(n)
    y = sim(theta, a, c, k)
    fit = fit_polytomous(y, k, model="gpcm")
    ld = local_dependence_polytomous(y, fit)
    n_pairs = j * (j - 1) // 2
    for key in ("item_i", "item_j", "x2", "g2", "p_value", "cramers_v", "max_abs_std_resid"):
        assert ld[key].shape == (n_pairs,)
    assert ld["df"] == (k - 1) ** 2
    assert np.all(ld["item_i"] < ld["item_j"])
    assert np.all(np.isfinite(ld["x2"])) and np.all(ld["x2"] >= 0.0)
    finite_p = ld["p_value"][np.isfinite(ld["p_value"])]
    assert np.all((finite_p >= 0.0) & (finite_p <= 1.0))
    assert np.mean(finite_p < 0.05) < 0.35  # few flags under local independence

    # a strong shared testlet on items 0,1 -> that pair is strongly dependent
    dep = sim(theta, a, c, k, testlet=1.5 * rng.standard_normal(n))
    fit_d = fit_polytomous(dep, k, model="gpcm")
    ld_d = local_dependence_polytomous(dep, fit_d)
    pair01 = next(idx for idx in range(n_pairs)
                  if (ld_d["item_i"][idx], ld_d["item_j"][idx]) == (0, 1))
    assert ld_d["x2"][pair01] > np.median(ld_d["x2"])
    assert ld_d["p_value"][pair01] < 0.05

    with pytest.raises(ValueError):
        local_dependence_polytomous(y[:, :-1], fit)


def test_fit_nominal_polytomous():
    """Nominal categories model (Thissen, Cai & Bock, 2010) through the public
    API: correct shapes, GPCM nesting (loglik >= GPCM, linear recovered scores),
    and input validation."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_nominal_polytomous, fit_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "fit_nominal"):
        pytest.skip("compiled core built without fit_nominal")

    rng = np.random.default_rng(9)
    n, j, k = 2500, 5, 3
    a = rng.uniform(0.9, 1.5, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.4, (j, k - 1))
    theta = rng.standard_normal(n)
    scores = np.arange(k, dtype=float)
    y = np.zeros((n, j), dtype=int)
    for i in range(j):
        p = np.exp(category_logprobs(a[i] * theta, scores, c[i]))
        for pp in range(n):
            y[pp, i] = rng.choice(k, p=p[pp])

    nom = fit_nominal_polytomous(y, k)
    assert nom.scores.shape == (j, k - 1)
    assert nom.intercepts.shape == (j, k - 1)
    assert np.isfinite(nom.loglik)
    assert nom.converged
    assert nom.termination_reason == "tolerance"
    assert nom.loglik_trace.shape == (nom.n_iter + 1,)
    assert nom.loglik == nom.loglik_trace[-1]
    assert nom.final_delta <= nom.stopping_tolerance
    assert np.all(np.diff(nom.loglik_trace) >= -1e-10)

    # nests the GPCM: at least as high a loglik, and linear recovered scores
    gp = fit_polytomous(y, k, model="gpcm")
    assert nom.loglik >= gp.loglik - 0.5
    ratio = nom.scores[:, 1] / nom.scores[:, 0]
    assert np.all(np.abs(ratio - 2.0) < 0.4)  # a_k ~ a*k

    with pytest.raises(ValueError):
        fit_nominal_polytomous(y, 1)
    with pytest.raises(ValueError):
        fit_nominal_polytomous(y.astype(float) + 0.5, k)  # non-integer categories

    unfinished = fit_nominal_polytomous(y[:50], k, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.termination_reason == "max_iter"
    assert unfinished.n_iter == 1
    assert unfinished.loglik_trace.shape == (2,)
    assert unfinished.loglik == unfinished.loglik_trace[-1]
    assert np.isfinite(unfinished.final_delta)
    assert unfinished.final_delta > unfinished.stopping_tolerance

    malformed = (
        np.empty((0, j)),
        np.empty((n, 0)),
        np.column_stack((y[:, 0], np.full(n, np.nan))),
        np.where(np.arange(y.size).reshape(y.shape) == 0, np.inf, y),
    )
    for bad in malformed:
        with pytest.raises(ValueError):
            fit_nominal_polytomous(bad, k)
    for kwargs in ({"max_iter": 0}, {"tol": np.inf}, {"tol": -1.0}):
        with pytest.raises(ValueError):
            fit_nominal_polytomous(y, k, **kwargs)


def test_person_fit_polytomous():
    """Polytomous person fit l_z / l_z* (Drasgow-Levine-Williams, 1985; Snijders,
    2001) through the public API: calibrated on a clean fit and flagging
    inconsistent responders evaluated against that fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_polytomous, person_fit_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_person_fit"):
        pytest.skip("compiled core built without poly_person_fit")

    rng = np.random.default_rng(5)
    n, j, k = 1500, 20, 3
    a = 1.0 + 0.03 * np.arange(j)
    c = np.zeros((j, k))
    c[:, 1] = 0.6
    c[:, 2] = -0.6
    scores = np.arange(k, dtype=float)

    def sim_person(th):
        return [
            rng.choice(k, p=np.exp(category_logprobs(np.array([a[i] * th]), scores, c[i])[0]))
            for i in range(j)
        ]

    # clean sample -> fit -> calibrated person fit
    theta = rng.standard_normal(n)
    y = np.array([sim_person(theta[p]) for p in range(n)])
    fit = fit_polytomous(y, k, model="gpcm")
    pf = person_fit_polytomous(y, fit)
    for key in ("lz", "lz_star", "theta_eap", "flagged"):
        assert len(pf[key]) == n
    finite = np.isfinite(pf["lz_star"])
    assert np.mean(pf["flagged"]) < 0.15  # Type I near nominal
    assert abs(np.mean(pf["lz_star"][finite])) < 0.4
    assert 0.8 < np.std(pf["lz_star"][finite]) < 1.25

    # inconsistent responders (implied trait alternates +-1.6 across items) —
    # evaluated at the SAME clean fit — are flagged
    m = 40
    ab = y.copy()
    for p in range(m):
        for i in range(j):
            ti = 1.6 if i % 2 == 0 else -1.6
            pr = np.exp(category_logprobs(np.array([a[i] * ti]), scores, c[i])[0])
            ab[p, i] = rng.choice(k, p=pr)
    pf_ab = person_fit_polytomous(ab, fit)
    assert np.mean(pf_ab["flagged"][:m]) > 0.6

    with pytest.raises(ValueError):
        person_fit_polytomous(y[:, :-1], fit)


def test_cat_simulate_polytomous():
    """Polytomous CAT (Dodd, De Ayala & Koch, 1995) through the public API:
    recovers the trait efficiently and max-information beats random selection."""
    import numpy as np
    import pytest
    from fast_mlsirm import cat_simulate_polytomous, PolytomousFit
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_cat_simulate"):
        pytest.skip("compiled core built without poly_cat_simulate")

    j, k, z = 40, 4, 3
    slope = np.array([1.0 + 0.25 * (i % 3) for i in range(j)])
    cat = np.zeros((j, z))
    for i in range(j):
        b = -2.2 + 4.4 * i / (j - 1)
        cum = 0.0
        for m in range(z):
            cum += b + (m - (z - 1) / 2) * 0.9
            cat[i, m] = -slope[i] * cum
    fit = PolytomousFit(model="gpcm", slope=slope, cat_params=cat, loglik=0.0, n_iter=0)

    tt = np.random.default_rng(0).standard_normal(400)
    var = cat_simulate_polytomous(tt, fit, se_threshold=0.30, min_items=5, max_items=30, seed=1)
    for key in ("theta_eap", "theta_sd", "n_used"):
        assert var[key].shape == (tt.size,)
    rmse = np.sqrt(np.mean((var["theta_eap"] - tt) ** 2))
    assert rmse < 0.40
    assert var["n_used"].mean() < 0.75 * j  # far fewer than the bank
    assert np.all(var["n_used"] <= 30)

    adap = cat_simulate_polytomous(tt, fit, se_threshold=0.0, min_items=12, max_items=12, adaptive=True, seed=2)
    rand = cat_simulate_polytomous(tt, fit, se_threshold=0.0, min_items=12, max_items=12, adaptive=False, seed=3)
    r_a = np.sqrt(np.mean((adap["theta_eap"] - tt) ** 2))
    r_r = np.sqrt(np.mean((rand["theta_eap"] - tt) ** 2))
    assert r_a < r_r  # max-information more efficient than random

    with pytest.raises(ValueError):
        cat_simulate_polytomous(tt, fit, min_items=10, max_items=5)


def test_dif_polytomous():
    """Two-group IRT-LR DIF for polytomous items (Thissen, Steinberg & Wainer,
    1993) via the public API: correct bookkeeping, impact does not trigger DIF,
    and an injected uniform difficulty shift on one item is flagged while the
    anchor items stay clean."""
    import numpy as np
    import pytest
    from fast_mlsirm import dif_polytomous
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(
        __import__("fast_mlsirm")._core, "poly_dif"
    ):
        pytest.skip("compiled core built without poly_dif")

    k, j = 3, 8
    n_per = 900
    rng = np.random.default_rng(11)
    a = rng.uniform(0.9, 1.4, j)
    c = np.zeros((j, k))
    c[:, 1:] = rng.normal(0.0, 0.5, (j, k - 1))
    scores = np.arange(k, dtype=float)

    def gen(dif_on_item0):
        # group 0: theta ~ N(0,1); group 1 (focal): theta ~ N(0.5, 1.2^2) (impact)
        th0 = rng.standard_normal(n_per)
        th1 = 0.5 + 1.2 * rng.standard_normal(n_per)
        theta = np.concatenate([th0, th1])
        gid = np.concatenate([np.zeros(n_per, int), np.ones(n_per, int)])
        y = np.zeros((2 * n_per, j), dtype=float)
        for p in range(2 * n_per):
            focal = gid[p] == 1
            for i in range(j):
                ci = c[i].copy()
                if i == 0 and focal and dif_on_item0:
                    # uniform DIF: shift difficulty => intercept_m += m * a * delta
                    d = 0.7
                    ci = ci + a[i] * d * scores
                base = a[i] * theta[p]
                pr = np.exp(category_logprobs(base, scores, ci))
                y[p, i] = rng.choice(k, p=pr)
        return y, gid

    # impact but NO DIF: anchor items should be (mostly) clean, df = n_cat
    y0, gid0 = gen(dif_on_item0=False)
    res0 = dif_polytomous(y0, gid0, k, model="gpcm")
    assert res0["item"].shape == (j,)
    for key in ("lr", "df", "p_value", "flagged_bh", "effect_size"):
        assert res0[key].shape == (j,)
    assert np.all(res0["df"] == k)  # (G-1)*K = K
    assert np.all(res0["lr"] >= 0.0)
    p0 = res0["p_value"]
    assert np.all((p0 >= 0.0) & (p0 <= 1.0))
    assert res0["flagged_bh"].sum() <= 1  # impact alone must not manufacture DIF

    # inject uniform DIF on item 0: it should be flagged with the largest effect
    y1, gid1 = gen(dif_on_item0=True)
    res1 = dif_polytomous(y1, gid1, k, model="gpcm")
    assert res1["flagged_bh"][0]
    assert res1["p_value"][0] < 0.01
    assert res1["effect_size"][0] == res1["effect_size"].max()
    assert res1["flagged_bh"][1:].sum() <= 1  # anchors stay clean

    # studied_items subset restricts the sweep
    sub = dif_polytomous(y1, gid1, k, model="gpcm", studied_items=np.array([0, 3]))
    assert list(sub["item"]) == [0, 3]

    # non-contiguous labels {0,2} must densify to 2 groups (df == n_cat), giving
    # the SAME result as contiguous {0,1} -- not an inflated, conservative df.
    gid_gap = np.where(gid1 == 1, 2, 0)
    res_gap = dif_polytomous(y1, gid_gap, k, model="gpcm")
    assert np.all(res_gap["df"] == k)  # not (3-1)*k = 2k
    assert res_gap["flagged_bh"][0]  # strong DIF still detected despite label gap
    np.testing.assert_allclose(res_gap["lr"], res1["lr"], rtol=1e-6)

    with pytest.raises(ValueError):
        dif_polytomous(y1, gid1[:-5], k)  # group_id length mismatch


def test_mantel_haenszel_dif():
    """Observed-score Mantel-Haenszel DIF (Holland & Thayer, 1988) via the public API: a uniform
    (b-shift) DIF planted on one dichotomous item, no group impact. MH flags the planted item as
    practically large (ETS class B/C, BH-significant) with the delta sign matching the shift (harder for
    the focal group -> negative MH_D-DIF and STD-P-DIF), classifies the clean items as A, and returns the
    documented per-item arrays. Validation guards reject non-dichotomous responses and a single group."""
    import numpy as np
    import pytest
    from fast_mlsirm import mantel_haenszel_dif
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "mantel_haenszel_dif"):
        pytest.skip("compiled core built without mantel_haenszel_dif")

    n, n_items = 3000, 12
    dif_item = 6
    rng = np.random.default_rng(1988)
    a = np.full(n_items, 1.2)
    b = -0.8 + 0.14 * np.arange(n_items)
    group = np.tile([0, 1], n // 2).astype(np.int64)
    theta = rng.standard_normal(n)  # equal ability distribution -> no impact
    bmat = np.tile(b, (n, 1))
    bmat[group == 1, dif_item] += 0.7  # item harder for the focal group (uniform DIF)
    p = 1.0 / (1.0 + np.exp(-(a * (theta[:, None] - bmat))))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = mantel_haenszel_dif(y, group)
    for key in ("item", "alpha_mh", "chi2_mh", "p_value", "mh_d_dif", "se_d_dif", "std_p_dif",
                "ets_class", "flagged_bh"):
        assert res[key].shape == (n_items,), key
    assert list(res["item"]) == list(range(n_items))
    # planted item: BH-flagged, large negative delta, negative std-p-dif, class B or C
    assert res["flagged_bh"][dif_item]
    assert res["mh_d_dif"][dif_item] < -0.8
    assert res["std_p_dif"][dif_item] < 0.0
    assert res["ets_class"][dif_item] in ("B", "C")
    # clean items: negligible (class A) by the practical-significance classification
    clean = [i for i in range(n_items) if i != dif_item]
    assert all(res["ets_class"][i] == "A" for i in clean)
    assert np.all(np.abs(res["mh_d_dif"][clean]) < 1.0)

    # validation: non-dichotomous responses and a single-group sample are rejected
    ybad = y.copy()
    ybad[0, 0] = 2
    with pytest.raises(ValueError):
        mantel_haenszel_dif(ybad, group)
    with pytest.raises(ValueError):
        mantel_haenszel_dif(y, np.zeros(n, dtype=np.int64))


def test_dif_polytomous_grm_no_silent_false_negative():
    """A GRM studied item whose focal group never uses a middle category can
    disorder thresholds -> NaN loglik. The finiteness guard must surface that as
    NaN (unflagged) rather than let the `.max(0.0)` clamp report a strongly-DIF
    item as clean (lr=0, p=1)."""
    import numpy as np
    import pytest
    from fast_mlsirm import dif_polytomous
    from fast_mlsirm.estimators.marginal import grm_category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "poly_dif"):
        pytest.skip("compiled core built without poly_dif")

    k, j, n_per = 4, 6, 500
    rng = np.random.default_rng(7)
    a = rng.uniform(1.0, 1.4, j)
    # decreasing GRM thresholds (ordered)
    b = np.tile(np.array([1.2, 0.0, -1.2]), (j, 1))

    def draw(theta, ai, bi):
        p = np.exp(grm_category_logprobs(ai * theta, bi))
        return rng.choice(k, p=p)

    y = np.zeros((2 * n_per, j))
    gid = np.concatenate([np.zeros(n_per, int), np.ones(n_per, int)])
    theta = np.concatenate([rng.standard_normal(n_per), 0.6 + rng.standard_normal(n_per)])
    for p in range(2 * n_per):
        focal = gid[p] == 1
        for i in range(j):
            if i == 0 and focal:
                # strong DIF + squeeze a middle category out for the focal group
                # (category 1 ~ 0 counts) so the GRM per-group M-step disorders
                # thresholds and the fit goes non-finite -- the guard's trigger
                y[p, i] = draw(theta[p], a[i], np.array([4.2, 4.1, -4.2]))
            else:
                y[p, i] = draw(theta[p], a[i], b[i])

    res = dif_polytomous(y, gid, k, model="grm")
    # item 0 must NOT be a silent clean report: either surfaced NaN, or (if the
    # fit stayed finite) correctly flagged. A finite p>0.5 unflagged would be the
    # masked false-negative the guard exists to prevent.
    p0, flg0 = res["p_value"][0], res["flagged_bh"][0]
    assert np.isnan(p0) or flg0, f"item 0 silently reported clean: p={p0}, flagged={flg0}"
    # a NaN p-value must be reported unflagged (never counted as significant)
    assert not (np.isnan(p0) and flg0)


def test_u3_person_fit_polytomous():
    """Nonparametric polytomous U3poly (Emons, 2008) through the public API:
    correct bookkeeping, calibrated flag rate under a simulated cutoff, and
    detection of careless responders."""
    import numpy as np
    import pytest
    from fast_mlsirm import (
        fit_polytomous,
        u3_cutoff_polytomous,
        u3_person_fit_polytomous,
    )
    from fast_mlsirm.estimators.marginal import category_logprobs
    from fast_mlsirm.polytomous import _core_module

    if _core_module() is None or not hasattr(__import__("fast_mlsirm")._core, "u3_person_fit"):
        pytest.skip("compiled core built without u3_person_fit")

    k, j, n = 5, 20, 800
    rng = np.random.default_rng(3)
    a = rng.uniform(0.9, 1.5, j)
    # spread item difficulty across the bank so items differ in popularity -- the
    # regime where a popularity-based statistic like U3 has power (Emons, 2008)
    bdiff = np.linspace(-1.6, 1.6, j)
    c = np.zeros((j, k))
    for i in range(j):
        cum = 0.0
        for m in range(1, k):
            cum += bdiff[i] + (m - 1 - (k - 2) / 2) * 0.8
            c[i, m] = -a[i] * cum
    scores = np.arange(k, dtype=float)

    def gen(n_care):
        theta = rng.standard_normal(n)
        y = np.zeros((n, j))
        for p in range(n):
            for i in range(j):
                if p < n_care:  # careless: uniform-random category
                    y[p, i] = rng.integers(k)
                else:
                    pr = np.exp(category_logprobs(a[i] * theta[p], scores, c[i]))
                    y[p, i] = rng.choice(k, p=pr)
        return y

    # clean data -> fit a bank -> simulated cutoff -> near-nominal flag rate
    y0 = gen(0)
    res0 = u3_person_fit_polytomous(y0, k)
    for key in ("u3poly", "total_score", "flagged"):
        assert res0[key].shape == (n,)
    finite = res0["u3poly"][np.isfinite(res0["u3poly"])]
    assert np.all((finite >= 0.0) & (finite <= 1.0))
    assert not res0["flagged"].any()  # no cutoff -> nothing flagged

    fit = fit_polytomous(y0, k, model="gpcm")
    cutoff = u3_cutoff_polytomous(fit, n_persons=n, alpha=0.05, n_rep=60, seed=7)
    assert 0.0 < cutoff < 1.0
    flagged0 = u3_person_fit_polytomous(y0, k, cutoff=cutoff)["flagged"]
    assert flagged0.mean() < 0.15  # calibrated-ish on clean data

    # inject careless responders -> they are flagged far more than clean persons
    n_care = 120
    y1 = gen(n_care)
    res1 = u3_person_fit_polytomous(y1, k, cutoff=cutoff)
    care_rate = res1["flagged"][:n_care].mean()
    clean_rate = res1["flagged"][n_care:].mean()
    assert care_rate > 0.6, f"careless detection too weak: {care_rate}"
    assert care_rate > 3 * clean_rate

    # K=2 stays in range and total_score bookkeeping is correct
    yb = (rng.random((200, 10)) < 0.5).astype(float)
    rb = u3_person_fit_polytomous(yb, 2)
    assert np.array_equal(rb["total_score"], yb.sum(axis=1).astype(np.int64))

    # an all-missing respondent has no conditioning group -> undefined (NaN),
    # never a silent perfect fit that can't be flagged
    ymiss = y0.copy()
    ymiss[0, :] = np.nan
    rm = u3_person_fit_polytomous(ymiss, k, cutoff=cutoff)
    assert np.isnan(rm["u3poly"][0])
    assert not rm["flagged"][0]

    with pytest.raises(ValueError):
        u3_person_fit_polytomous(y0, k, cutoff=float("nan"))


def test_equate_observed_scores_and_neat():
    """Observed-score equating (Kolen & Brennan, 2014) through the public API:
    equipercentile self-equate is the identity, mean/linear recover a known
    transform, and NEAT chained/FE collapse to EG equipercentile under equal
    anchor distributions."""
    import numpy as np
    import pytest
    from fast_mlsirm import equate_neat, equate_observed_scores
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores"):
        pytest.skip("compiled core built without equating")

    rng = np.random.default_rng(5)
    k = 40
    x = np.clip(np.round(20 + 8 * rng.standard_normal(4000)), 0, k)

    # self-equate is the identity on populated scores
    r = equate_observed_scores(x, x, method="equipercentile", k_x=k, k_y=k)
    g = np.bincount(x.astype(int), minlength=k + 1)
    dev = np.abs(r.y_equivalents - r.x_scores)[g > 0]
    assert dev.max() < 1e-9
    assert r.design == "EG"

    # mean equating recovers Y = X + 5
    y = x + 5
    rm = equate_observed_scores(x, y, method="mean", k_x=k, k_y=k + 5)
    assert abs(rm.intercept - 5.0) < 1e-9 and abs(rm.slope - 1.0) < 1e-12

    # NEAT collapse under equal anchor distributions (identical anchor vectors)
    n = 6000
    kv, kx, ky = 15, 30, 40
    v = np.clip(np.round(7 + 3 * rng.standard_normal(n)), 0, kv)
    xt = np.clip(np.round(1.4 * v + 4 + 4 * rng.standard_normal(n)), 0, kx)
    yt = np.clip(np.round(2.0 * v + 6 + 5 * rng.standard_normal(n)), 0, ky)
    eg = equate_observed_scores(xt, yt, method="equipercentile", k_x=kx, k_y=ky)
    ch = equate_neat(xt, v, yt, v, method="chained", k_x=kx, k_y=ky, k_v=kv)
    fe = equate_neat(xt, v, yt, v, method="frequency_estimation", k_x=kx, k_y=ky, k_v=kv, w1=0.5)
    assert np.max(np.abs(ch.y_equivalents - eg.y_equivalents)) < 1e-9
    assert np.max(np.abs(fe.y_equivalents - eg.y_equivalents)) < 1e-9
    assert ch.design == "NEAT"

    with pytest.raises(ValueError):
        equate_observed_scores(x, y, method="bogus", k_x=k, k_y=k + 5)


def test_kernel_equating_and_presmoothing():
    """Kernel equating + log-linear presmoothing (von Davier et al., 2004;
    Holland & Thayer, 2000) through the public API: presmoothing preserves
    moments, uniform-kernel ext matches equipercentile, and a large bandwidth
    drives kernel equating to linear."""
    import numpy as np
    import pytest
    from fast_mlsirm import (
        equate_observed_scores,
        equate_observed_scores_kernel,
        loglinear_smooth,
    )
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_observed_scores_ext"):
        pytest.skip("compiled core built without kernel equating")

    rng = np.random.default_rng(7)
    k = 40
    x = np.clip(np.round(20 + 7 * rng.standard_normal(5000)), 0, k)
    counts = np.bincount(x.astype(int), minlength=k + 1).astype(float)

    # presmoothing preserves the first `degree` moments (on the u=x/k scale)
    fit = loglinear_smooth(counts, degree=4)
    assert fit["converged"] and abs(fit["probs"].sum() - 1.0) < 1e-12
    g = counts / counts.sum()
    for j, fm in enumerate(fit["moments"], start=1):
        sm = float(((np.arange(k + 1) / k) ** j * g).sum())
        assert abs(fm - sm) < 1e-8, f"moment {j}: {fm} vs {sm}"

    # uniform-kernel ext == equipercentile
    y = np.clip(np.round(22 + 8 * rng.standard_normal(5000)), 0, k)
    base = equate_observed_scores(x, y, method="equipercentile", k_x=k, k_y=k)
    uni = equate_observed_scores_kernel(x, y, continuization="uniform", k_x=k, k_y=k)
    assert np.max(np.abs(base.y_equivalents - uni.y_equivalents)) < 1e-12
    assert np.isnan(uni.h_x)

    # large-bandwidth Gaussian kernel -> linear equating
    lin = equate_observed_scores(x, y, method="linear", k_x=k, k_y=k)
    ker = equate_observed_scores_kernel(
        x, y, continuization="gaussian", k_x=k, k_y=k, bandwidth_x=1e6, bandwidth_y=1e6
    )
    assert np.max(np.abs(lin.y_equivalents - ker.y_equivalents)) < 1e-3
    assert ker.h_x == 1e6

    # penalty-selected bandwidth is finite and positive
    auto = equate_observed_scores_kernel(x, y, continuization="gaussian", k_x=k, k_y=k)
    assert np.isfinite(auto.h_x) and auto.h_x > 0

    # default degree clamps to k, so short forms (k < 6) do not error
    short = loglinear_smooth(np.array([10.0, 20.0, 30.0, 15.0, 8.0, 4.0]))  # k = 5
    assert short["converged"] and short["probs"].shape == (6,)

    # A successful function return is not a successful optimization.  This sparse
    # fixture stops before satisfying the log-linear score tolerance, so the
    # high-level equating path must not consume its unfinished density.
    sparse_counts = np.array([0, 1564, 426, 0, 1008, 0, 0])
    unfinished = loglinear_smooth(sparse_counts, degree=5)
    assert not unfinished["converged"] and unfinished["iters"] < 50
    assert unfinished["termination_reason"] == "line_search_stalled"
    assert unfinished["final_gradient_max"] > unfinished["gradient_tolerance"]
    sparse_scores = np.repeat(np.arange(7), sparse_counts).astype(float)
    with pytest.raises(ValueError, match="presmoothing did not converge"):
        equate_observed_scores_kernel(
            sparse_scores,
            sparse_scores,
            continuization="uniform",
            k_x=6,
            k_y=6,
            smooth_x=5,
            smooth_y=5,
        )

    with pytest.raises(ValueError):
        equate_observed_scores_kernel(x, y, continuization="bogus", k_x=k, k_y=k)
    with pytest.raises(ValueError):
        equate_observed_scores_kernel(x, y, continuization="gaussian", k_x=k, k_y=k, smooth_x=-1)


def test_equate_neat_linear_tucker_levine():
    """Tucker & Levine linear NEAT equating (Kolen & Brennan, 2014) through the
    public API: with equal anchor moments every variant collapses to EG linear,
    and the internal/external Levine gamma give distinct conversions."""
    import numpy as np
    import pytest
    from fast_mlsirm import equate_neat_linear, equate_observed_scores
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "equate_neat_linear"):
        pytest.skip("compiled core built without equate_neat_linear")

    rng = np.random.default_rng(5)
    n, kx, ky = 4000, 30, 40
    anchor = np.clip(np.round(7 + 3 * rng.standard_normal(n)), 0, 15)
    xt = np.clip(np.round(1.5 * anchor + 4 + 3 * rng.standard_normal(n)), 0, kx)
    yt = np.clip(np.round(1.8 * anchor + 6 + 4 * rng.standard_normal(n)), 0, ky)

    # collapse: shared anchor -> equal anchor moments -> EG linear
    eg = equate_observed_scores(xt, yt, method="linear", k_x=kx, k_y=ky)
    for method in ("tucker", "levine"):
        for ak in ("internal", "external"):
            r = equate_neat_linear(xt, anchor, yt, anchor, method=method, anchor_kind=ak, k_x=kx, k_y=ky)
            assert abs(r.slope - eg.slope) < 1e-9 and abs(r.intercept - eg.intercept) < 1e-9
            assert r.design == "NEAT"

    # internal vs external Levine differ on a genuinely non-equivalent anchor
    ancx = np.clip(np.round(6 + 2.5 * rng.standard_normal(n)), 0, 15)
    ancy = np.clip(np.round(9 + 2.5 * rng.standard_normal(n)), 0, 15)  # shifted
    xt2 = np.clip(np.round(1.2 * ancx + 5 + 3 * rng.standard_normal(n)), 0, kx)
    yt2 = np.clip(np.round(1.2 * ancy + 8 + 3 * rng.standard_normal(n)), 0, ky)
    li = equate_neat_linear(xt2, ancx, yt2, ancy, method="levine", anchor_kind="internal", k_x=kx, k_y=ky)
    le = equate_neat_linear(xt2, ancx, yt2, ancy, method="levine", anchor_kind="external", k_x=kx, k_y=ky)
    tk = equate_neat_linear(xt2, ancx, yt2, ancy, method="tucker", k_x=kx, k_y=ky)
    assert abs(li.slope - le.slope) > 1e-6 or abs(li.intercept - le.intercept) > 1e-6
    assert abs(tk.slope - li.slope) > 1e-6 or abs(tk.intercept - li.intercept) > 1e-6

    with pytest.raises(ValueError):
        equate_neat_linear(xt, anchor, yt, anchor, method="bogus", k_x=kx, k_y=ky)


def test_equating_standard_errors():
    """Standard errors of equating (Kolen & Brennan, 2014, ch. 7) through the
    public API: analytic and bootstrap linear SEE agree, Mean SEE is constant,
    the CI brackets the point estimate, and equipercentile SEE is bootstrap-only."""
    import numpy as np
    import pytest
    from fast_mlsirm import equating_standard_errors
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "bootstrap_see"):
        pytest.skip("compiled core built without SEE")

    rng = np.random.default_rng(7)
    k, n = 30, 3000
    x = np.clip(np.round(15 + 5 * rng.standard_normal(n)), 0, k)
    y = np.clip(np.round(16 + 5 * rng.standard_normal(n)), 0, k)

    a = equating_standard_errors(x, y, method="linear", route="analytic", k_x=k, k_y=k)
    b = equating_standard_errors(x, y, method="linear", route="bootstrap", k_x=k, k_y=k, n_boot=2000, seed=1)
    lo, hi = int(np.ceil(k * 0.1)), k - int(np.ceil(k * 0.1))
    rel = np.abs(b["se"][lo:hi] - a["se"][lo:hi]) / a["se"][lo:hi]
    assert rel.max() < 0.15, f"analytic vs bootstrap linear SEE: {rel.max()}"
    # CI brackets the point estimate
    assert np.all(b["ci_lo"][lo:hi] <= b["y_equivalents"][lo:hi] + 1e-9)
    assert np.all(b["y_equivalents"][lo:hi] <= b["ci_hi"][lo:hi] + 1e-9)
    assert b["n_boot"] == 2000 and abs(b["ci_level"] - 0.95) < 1e-12

    # Mean SEE constant in x
    m = equating_standard_errors(x, y, method="mean", route="analytic", k_x=k, k_y=k)
    assert np.allclose(m["se"], m["se"][0])

    # equipercentile has no analytic SEE; bootstrap works
    ep = equating_standard_errors(x, y, method="equipercentile", route="bootstrap", k_x=k, k_y=k, n_boot=300, seed=3)
    assert np.all(ep["se"][lo:hi] > 0) and ep["n_boot"] == 300
    with pytest.raises(ValueError):
        equating_standard_errors(x, y, method="equipercentile", route="analytic", k_x=k, k_y=k)
    with pytest.raises(ValueError):
        equating_standard_errors(x, y, method="linear", route="bogus", k_x=k, k_y=k)


def test_fit_response_times():
    """Lognormal response-time model (van der Linden, 2007) through the public
    API: recovers the item time parameters, the speed SD, and the speed EAP, and
    handles missing (NaN) response times."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rt_lognormal"):
        pytest.skip("compiled core built without fit_rt_lognormal")

    rng = np.random.default_rng(4)
    n, m = 800, 20
    beta = np.linspace(3.5, 4.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    sigma = 0.3
    tau = sigma * rng.standard_normal(n)
    y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
    times = np.exp(y)
    # inject ~20% missing as NaN
    times[rng.random((n, m)) < 0.2] = np.nan

    fit = fit_response_times(times)
    assert fit.converged
    assert fit.alpha.shape == (m,) and fit.tau_eap.shape == (n,)
    assert np.corrcoef(fit.beta, beta)[0, 1] > 0.95
    assert np.corrcoef(fit.alpha, alpha)[0, 1] > 0.85
    assert np.corrcoef(fit.tau_eap, tau)[0, 1] > 0.8
    assert abs(fit.sigma_tau - sigma) < 0.1
    assert fit.mu_tau == 0.0
    # loglik trace is non-decreasing (monotone EM)
    # (exposed via n_iter/converged; recompute a small fit to confirm determinism)
    fit2 = fit_response_times(times)
    assert np.allclose(fit.beta, fit2.beta)

    with pytest.raises(ValueError):
        fit_response_times(times.ravel())  # not 2-D


def test_fit_speed_accuracy():
    """Joint speed-accuracy model (van der Linden, 2007, Level 2) through the
    public API: recovers a positive ability-speed correlation with item banks
    fixed, and returns ~0 under true independence."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times, fit_speed_accuracy
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_speed_accuracy_covariance"):
        pytest.skip("compiled core built without fit_speed_accuracy_covariance")

    rng = np.random.default_rng(7)
    n, m = 1000, 20
    a = 0.9 + 0.6 * (np.arange(m) % 3) / 2.0
    b = np.linspace(-1.5, 1.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    beta = np.linspace(3.5, 4.5, m)

    def sim(rho, sig=0.3):
        za = rng.standard_normal(n)
        zb = rng.standard_normal(n)
        theta = za
        tau = rho * sig * za + sig * np.sqrt(1 - rho * rho) * zb
        pr = 1.0 / (1.0 + np.exp(-(a[None, :] * theta[:, None] + b[None, :])))
        resp = (rng.random((n, m)) < pr).astype(float)
        y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
        return resp, np.exp(y)

    resp, times = sim(0.5)
    res = fit_speed_accuracy(resp, times, a, b, alpha, beta)
    assert res["converged"]
    assert res["termination_reason"] == "converged"
    assert res["n_iter"] < 500
    assert res["final_loglik_change"] < 1e-6
    assert np.isfinite(res["loglik_trace"]).all()
    assert np.all(np.diff(res["loglik_trace"]) >= -1e-6 * np.maximum(np.abs(res["loglik_trace"][:-1]), 1))
    assert res["loglik"] == res["loglik_trace"][-1]
    assert abs(res["rho"] - 0.5) < 0.1, res["rho"]
    assert abs(res["sigma_tau"] - 0.3) < 0.05
    assert res["theta_eap"].shape == (n,) and res["tau_eap"].shape == (n,)

    # true independence -> rho ~ 0
    r0, t0 = sim(0.0)
    res0 = fit_speed_accuracy(r0, t0, a, b, alpha, beta)
    assert res0["converged"]
    assert res0["termination_reason"] == "converged"
    assert res0["final_loglik_change"] < 1e-6
    assert abs(res0["rho"]) < 0.08, res0["rho"]

    # works with a fitted RT model's alpha/beta
    rt = fit_response_times(times)
    res_rt = fit_speed_accuracy(resp, times, a, b, rt.alpha, rt.beta)
    assert res_rt["converged"]
    assert res_rt["termination_reason"] == "converged"
    assert res_rt["final_loglik_change"] < 1e-6
    assert abs(res_rt["rho"] - 0.5) < 0.15

    with pytest.warns(RuntimeWarning, match="max_iter_reached"):
        res_nc = fit_speed_accuracy(
            resp,
            times,
            a,
            b,
            alpha,
            beta,
            q=7,
            max_iter=1,
        )
    assert not res_nc["converged"]
    assert res_nc["termination_reason"] == "max_iter_reached"
    assert res_nc["n_iter"] == 1
    assert res_nc["final_loglik_change"] >= 1e-6

    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_speed_accuracy(
            resp,
            times,
            a,
            b,
            alpha,
            beta,
            q=7,
            max_iter=1,
            require_convergence=True,
        )

    with pytest.raises(ValueError):
        fit_speed_accuracy(resp.ravel(), times, a, b, alpha, beta)  # not 2-D


def test_rt_person_fit():
    """RT person fit (van der Linden & Guo, 2008): W ~ chi2(n-1) with l_t ~ N(0,1)
    on model-consistent data, and rapid-guessing responders flagged."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_response_times, rt_person_fit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "rt_person_fit"):
        pytest.skip("compiled core built without rt_person_fit")

    rng = np.random.default_rng(3)
    n, m = 2000, 20
    beta = np.linspace(3.5, 4.5, m)
    alpha = np.linspace(1.0, 3.0, m)
    tau = 0.3 * rng.standard_normal(n)
    y = beta[None, :] - tau[:, None] + rng.standard_normal((n, m)) / alpha[None, :]
    # first 10% rapid-guess on the last 7 items
    n_ab = n // 10
    for p in range(n_ab):
        y[p, -7:] = (beta[-7:] - tau[p]) - 2.5 + 0.3 * rng.standard_normal(7)
    times = np.exp(y)

    # exact calibration with the (uncontaminated) item parameters: W ~ chi2(n-1),
    # l_t ~ N(0,1) on the clean responders, ~.05 Type I, high power
    pf = rt_person_fit(times, alpha, beta)
    assert pf["w"].shape == (n,) and pf["z_resid"].shape == (n, m)
    clean = pf["l_t"][n_ab:]
    assert abs(clean.mean()) < 0.15 and 0.8 < clean.std() < 1.25
    assert pf["flagged"][n_ab:].mean() < 0.12
    assert pf["flagged"][:n_ab].mean() > 0.7
    # the tampered items are flagged too-fast (strongly negative residual)
    assert pf["item_flag"][:n_ab, -7:].mean() > 0.7
    assert np.all(pf["z_resid"][:n_ab, -7:] < 0)  # too-fast = negative

    # with a fitted RT model (production path) the aberrant are still detected;
    # fitting on the contaminated sample makes the clean responders conservative
    fit = fit_response_times(times)
    pf2 = rt_person_fit(times, fit.alpha, fit.beta)
    assert pf2["flagged"][:n_ab].mean() > 0.6

    with pytest.raises(ValueError):
        rt_person_fit(times.ravel(), alpha, beta)  # not 2-D


def _sim_cdm(rng, q, s, g, profiles, model="dina"):
    """Simulate DINA/DINO responses for the given bit-encoded true profiles."""
    n, (n_items, k) = len(profiles), q.shape
    y = np.empty((n, n_items))
    for j in range(n):
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            c = int(profiles[j])
            eta = (c & mask) == mask if model == "dina" else (c & mask) != 0
            p = 1.0 - s[i] if eta else g[i]
            y[j, i] = 1.0 if rng.random() < p else 0.0
    return y


def test_fit_cdm_dina_recovers_and_classifies():
    """DINA cognitive diagnosis (de la Torre, 2009): recover slip/guess and classify
    attribute mastery under a known Q-matrix; DINO reduces to DINA on single-attribute
    items."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_cdm, CdmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_cdm"):
        pytest.skip("compiled core built without fit_cdm")

    rng = np.random.default_rng(11)
    k, n_items, n = 3, 15, 1500
    # 6 single-attribute items (2 per attribute) + pairs + one triple.
    rows = []
    for a in range(k):
        rows += [[1 if t == a else 0 for t in range(k)]] * 2
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1], [1, 0, 1]]
    q = np.array(rows[:n_items], dtype=np.int64)
    s = np.full(n_items, 0.15)
    g = np.full(n_items, 0.15)
    profiles = rng.integers(0, 1 << k, size=n)
    y = _sim_cdm(rng, q, s, g, profiles)

    res = fit_cdm(y, q, model="dina")
    assert isinstance(res, CdmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert np.all(1.0 - res.slip > res.guess)  # identification
    assert np.sqrt(np.mean((res.slip - s) ** 2)) < 0.05
    assert np.sqrt(np.mean((res.guess - g) ** 2)) < 0.05
    # attribute classification agreement (marginal mastery vs truth)
    true_bits = ((profiles[:, None] >> np.arange(k)) & 1)
    attr_ok = (res.attribute_mastery() == true_bits).mean()
    assert attr_ok > 0.85, attr_ok
    # pattern-wise agreement (exact 3-bit profile)
    assert (res.map_profile == profiles).mean() > 0.75
    assert res.n_parameters == 2 * n_items + ((1 << k) - 1)
    assert res.profile_bits().shape == (n, k)

    # single-attribute Q => DINA and DINO share identical eta and thus identical fits.
    q1 = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.int64)
    prof1 = rng.integers(0, 4, size=800)
    y1 = _sim_cdm(rng, q1, np.full(4, 0.2), np.full(4, 0.2), prof1)
    a = fit_cdm(y1, q1, model="dina")
    b = fit_cdm(y1, q1, model="dino")
    assert np.allclose(a.slip, b.slip, atol=1e-9)
    assert np.allclose(a.guess, b.guess, atol=1e-9)

    # missing-at-random cells are dropped, not imputed.
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    resm = fit_cdm(ym, q, model="dina")
    assert resm.converged and np.all(1.0 - resm.slip > resm.guess)

    with pytest.raises(ValueError):
        fit_cdm(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_cdm(y, q, model="rasch")  # unknown gate
    with pytest.raises(ValueError):
        fit_cdm(y, np.zeros((n_items, k), dtype=np.int64))  # all-zero Q rows/cols


def test_fit_gdina_recovers_saturated_and_reduces_to_dina():
    """Saturated G-DINA (de la Torre, 2011): recover free reduced-class success
    probabilities, and confirm DINA-generated data yields the DINA identity-link
    pattern (only intercept + highest-order interaction nonzero)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_gdina, GdinaFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gdina"):
        pytest.skip("compiled core built without fit_gdina")

    def reduce_class(c, qmask, k):
        l, m = 0, 0
        for bit in range(k):
            if (qmask >> bit) & 1:
                l |= ((c >> bit) & 1) << m
                m += 1
        return l

    rng = np.random.default_rng(2011)
    k, n_items, n = 2, 12, 2500
    q = np.zeros((n_items, k), dtype=np.int64)
    for i in range(n_items):
        if i < 4:
            q[i, 0] = 1
        elif i < 8:
            q[i, 1] = 1
        else:
            q[i] = [1, 1]
    qmask = [int(np.dot(q[i], 1 << np.arange(k))) for i in range(n_items)]
    item_off = np.concatenate([[0], np.cumsum([1 << int(q[i].sum()) for i in range(n_items)])])

    # DINA truth in CSR layout: top reduced class = 1 - s, all others = g.
    s, g = 0.15, 0.2
    truth = np.empty(item_off[-1])
    for i in range(n_items):
        truth[item_off[i] : item_off[i + 1]] = g
        truth[item_off[i + 1] - 1] = 1.0 - s
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.empty((n, n_items))
    for j in range(n):
        for i in range(n_items):
            p = truth[item_off[i] + reduce_class(int(profiles[j]), qmask[i], k)]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_gdina(y, q)
    assert isinstance(res, GdinaFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.sqrt(np.mean((res.item_prob - truth) ** 2)) < 0.03
    assert res.n_parameters == int(item_off[-1]) + ((1 << k) - 1)
    # DINA identity-link pattern per item: delta_0 ~ g, delta_full ~ (1-s)-g, mids ~ 0.
    for i in range(n_items):
        d = res.item_delta_row(i)
        assert abs(d[0] - g) < 0.05
        assert abs(d[-1] - ((1.0 - s) - g)) < 0.05
        if len(d) > 2:
            assert np.all(np.abs(d[1:-1]) < 0.05)
    # all-mastered reduced class has the highest success probability.
    for i in range(n_items):
        row = res.item_prob_row(i)
        assert row[-1] >= row.max() - 1e-9

    # missing-at-random cells are dropped.
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    assert fit_gdina(ym, q).converged

    with pytest.raises(ValueError):
        fit_gdina(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_gdina(y, np.zeros((n_items, k), dtype=np.int64))  # all-zero Q rows/cols


def test_validate_q_matrix_corrects_misspecification():
    """PVAF Q-matrix validation (de la Torre & Chiu, 2016): the true Q validates to
    itself, and a Q with an over-specified and an under-specified item is corrected
    back to the truth while flagging exactly those items."""
    import numpy as np
    import pytest
    from fast_mlsirm import validate_q_matrix, QMatrixValidation
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "validate_q_matrix"):
        pytest.skip("compiled core built without validate_q_matrix")

    rng = np.random.default_rng(715)
    k, n_items, n = 3, 15, 3000
    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
            [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [1, 1, 1], [1, 1, 1], [1, 1, 1]]
    truth = np.array(rows, dtype=np.int64)
    s = np.full(n_items, 0.1)
    g = np.full(n_items, 0.1)
    profiles = rng.integers(0, 1 << k, size=n)
    y = _sim_cdm(rng, truth, s, g, profiles)

    # Anchor: the true Q validates to itself, nothing flagged.
    res = validate_q_matrix(y, truth, epsilon=0.95)
    assert isinstance(res, QMatrixValidation)
    assert np.array_equal(res.suggested_q, truth)
    assert not res.flagged.any()
    assert np.all(res.provisional_pvaf > 0.9)

    # Over-specify item 0 ({0} -> {0,1}) and under-specify item 6 ({0,1} -> {0}).
    prov = truth.copy()
    prov[0, 1] = 1
    prov[6, 1] = 0
    res2 = validate_q_matrix(y, prov, epsilon=0.95)
    assert np.array_equal(res2.suggested_q[0], truth[0])  # trimmed back
    assert np.array_equal(res2.suggested_q[6], truth[6])  # enlarged back
    assert res2.flagged[0] and res2.flagged[6]
    # the under-specified item's provisional q falls short of the cutoff
    assert res2.provisional_pvaf[6] < 0.95

    with pytest.raises(ValueError):
        validate_q_matrix(y.ravel(), truth)  # responses not 2-D
    with pytest.raises(ValueError):
        validate_q_matrix(y, truth, epsilon=1.5)  # epsilon out of range
    with pytest.raises(
        ValueError,
        match=r"G-DINA calibration did not converge after 1 of 1 M-steps",
    ):
        validate_q_matrix(y, truth, max_iter=1, tol=1e-12)


def test_gdina_wald_selection_classifies_items():
    """Item-level Wald model selection (de la Torre, 2011; de la Torre & Lee, 2013):
    a conjunctive (DINA), disjunctive (DINO), identity-additive (A-CDM), logit-additive
    (LLM), and log-additive (R-RUM) item are each classified as their reduced model,
    and an item with both main effects and an interaction keeps the saturated G-DINA.
    The LLM and R-RUM truths are additive ONLY on their own link (identity- and
    cross-link-nonadditive), so classifying them correctly exercises the link-transformed
    delta and its delta-method covariance, not just the identity link."""
    import numpy as np
    import pytest
    from fast_mlsirm import gdina_wald_selection, WaldModelSelection
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "gdina_wald_selection"):
        pytest.skip("compiled core built without gdina_wald_selection")

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))

    rng = np.random.default_rng(2011)
    k, n = 2, 8000
    # 5 single-attribute items per attribute (identification) + 6 pair items:
    # DINA, DINO, A-CDM, LLM, R-RUM, saturated.
    rows = [[1, 0]] * 5 + [[0, 1]] * 5 + [[1, 1]] * 6
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    # per reduced-class truth [none, a0, a1, both]
    truth_pair = {10: [0.15, 0.15, 0.15, 0.85],  # DINA (conjunctive)
                  11: [0.15, 0.85, 0.85, 0.85],  # DINO (disjunctive)
                  12: [0.10, 0.45, 0.45, 0.80],  # A-CDM (identity-additive)
                  # LLM: logit(P) = -3 + 2 a0 + 2 a1 (logit-additive, identity- &
                  # log-nonadditive).
                  13: [sig(-3.0), sig(-1.0), sig(-1.0), sig(1.0)],
                  # R-RUM: P = 0.92 * 0.3^(1-a0) * 0.4^(1-a1) (log-additive, identity- &
                  # logit-nonadditive).
                  14: [0.92 * 0.3 * 0.4, 0.92 * 0.4, 0.92 * 0.3, 0.92],
                  15: [0.10, 0.35, 0.35, 0.90]}  # saturated
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 10:
                a = i // 5  # attribute of this single item
                p = 0.85 if (c >> a) & 1 else 0.15
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)  # reduced class for a {0,1} item
                p = truth_pair[i][l]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = gdina_wald_selection(y, q, alpha=0.05)
    assert isinstance(res, WaldModelSelection)
    assert res.models == ["dina", "dino", "acdm", "llm", "rrum"]
    assert res.selected[10] == 0   # DINA
    assert res.selected[11] == 1   # DINO
    assert res.selected[12] == 2   # A-CDM
    assert res.selected[13] == 3   # LLM (logit link)
    assert res.selected[14] == 4   # R-RUM (log link)
    assert res.selected[15] == -1  # saturated G-DINA
    # single-attribute items carry no test (df 0), keep saturated
    assert np.all(res.selected[:10] == -1)
    assert np.all(res.wald_df[:10] == 0)
    # the tested pair items have the right degrees of freedom (K=2): DINA & DINO
    # df = 2^K-2 = 2; A-CDM, LLM, R-RUM df = 2^K-1-K = 1.
    assert res.wald_df[10, 0] == 2 and res.wald_df[10, 1] == 2
    assert res.wald_df[10, 2] == 1 and res.wald_df[10, 3] == 1 and res.wald_df[10, 4] == 1

    with pytest.raises(ValueError):
        gdina_wald_selection(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        gdina_wald_selection(y, q, alpha=0.0)  # alpha out of range
    with pytest.raises(
        ValueError,
        match=r"G-DINA calibration did not converge after 1 of 1 M-steps",
    ):
        gdina_wald_selection(y, q, max_iter=1, tol=1e-12)


def test_fit_ho_cdm_recovers_higher_order_structure():
    """Higher-order DINA (de la Torre & Douglas, 2004): a continuous trait structures
    attribute mastery; recover the attribute slopes/intercepts, slip/guess, and
    classification, and confirm the slope-zero reduction to independent attributes."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_ho_cdm, HoCdmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_cdm"):
        pytest.skip("compiled core built without fit_ho_cdm")

    rng = np.random.default_rng(2004)
    k, n = 3, 4000
    # 4 single-attribute items per attribute + 3 pair items (all attributes identified)
    rows = []
    for a in range(k):
        for _ in range(4):
            rows.append([1 if t == a else 0 for t in range(k)])
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    a_true = np.array([1.2, 1.5, 0.9])
    d_true = np.array([0.3, -0.5, 0.6])
    s, g = np.full(n_items, 0.12), np.full(n_items, 0.12)

    theta = rng.standard_normal(n)
    alpha = (rng.random((n, k)) < 1.0 / (1.0 + np.exp(-(theta[:, None] * a_true + d_true)))).astype(int)
    codes = (alpha * (1 << np.arange(k))).sum(1)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(codes[j])
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            eta = (c & mask) == mask
            p = 1.0 - s[i] if eta else g[i]
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_ho_cdm(y, q, model="dina")
    assert isinstance(res, HoCdmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == 2 * n_items + 2 * k
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    assert np.all(res.attr_slope > 0)  # anchored non-negative
    assert np.sqrt(np.mean((res.slip - s) ** 2)) < 0.05
    assert np.sqrt(np.mean((res.attr_slope - a_true) ** 2)) < 0.4  # identified at K=3
    assert np.sqrt(np.mean((res.attr_intercept - d_true) ** 2)) < 0.3
    # attribute classification agreement
    est = res.attribute_mastery()
    assert (est == alpha).mean() > 0.85

    with pytest.raises(ValueError):
        fit_ho_cdm(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_ho_cdm(y, q, model="rasch")  # unknown gate


def test_fit_ho_gdina_recovers_saturated_and_structure():
    """Higher-order G-DINA (de la Torre & Douglas, 2004; de la Torre, 2011): a free
    saturated item fit of DINA-patterned data recovers the DINA identity-link delta
    and the higher-order attribute parameters."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_ho_gdina, HoGdinaFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_ho_gdina"):
        pytest.skip("compiled core built without fit_ho_gdina")

    rng = np.random.default_rng(2011)
    k, n = 3, 3000
    rows = [[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 3  # 9 singles
    rows += [[1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 1, 1]]  # 5 pairs
    rows += [[1, 1, 1]]  # 1 triple
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    s, g = 0.15, 0.2
    a_true = np.array([1.2, 1.5, 0.9])
    d_true = np.array([0.3, -0.5, 0.6])

    theta = rng.standard_normal(n)
    alpha = (rng.random((n, k)) < 1.0 / (1.0 + np.exp(-(theta[:, None] * a_true + d_true)))).astype(int)
    codes = (alpha * (1 << np.arange(k))).sum(1)
    y = np.empty((n, n_items))
    for j in range(n):
        c = int(codes[j])
        for i in range(n_items):
            mask = int(np.dot(q[i], 1 << np.arange(k)))
            eta = (c & mask) == mask  # DINA gate
            p = (1.0 - s) if eta else g
            y[j, i] = 1.0 if rng.random() < p else 0.0

    res = fit_ho_gdina(y, q)
    assert isinstance(res, HoGdinaFit) and res.converged
    assert res.termination_reason == "tolerance_met"
    assert res.final_relative_loglik_change < res.stopping_tolerance
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.all(res.attr_slope > 0)  # anchored non-negative
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    # the triple item's identity-link delta shows the DINA pattern (intercept + top)
    triple = n_items - 1
    dl = res.item_delta[res.item_off[triple] : res.item_off[triple + 1]]
    assert abs(dl[0] - g) < 0.06 and abs(dl[-1] - ((1.0 - s) - g)) < 0.06
    assert np.all(np.abs(dl[1:-1]) < 0.06)
    # higher-order parameter recovery (identified at K=3)
    assert np.sqrt(np.mean((res.attr_slope - a_true) ** 2)) < 0.45
    # attribute classification
    est = (res.attr_prob >= 0.5).astype(int)
    assert (est == alpha).mean() > 0.9

    with pytest.raises(ValueError):
        fit_ho_gdina(y.ravel(), q)  # not 2-D
    with pytest.raises(ValueError, match="at least 3 attributes"):
        fit_ho_gdina(y[:, :2], np.eye(2, dtype=np.int64))
    for bad in (np.inf, -np.inf):
        malformed = y.copy()
        malformed[0, 0] = bad
        with pytest.raises(ValueError, match="only 0, 1, or NaN"):
            fit_ho_gdina(malformed, q)

    limited = y[:100].copy()
    limited[0, 0] = np.nan
    unfinished = fit_ho_gdina(limited, q, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert unfinished.termination_reason == "max_iter_reached"
    assert np.isfinite(unfinished.final_loglik_change)
    assert np.isfinite(unfinished.final_relative_loglik_change)


def test_fit_seq_gdina_recovers_polytomous_and_reduces_to_gdina():
    """Shared-Q sequential G-DINA (Ma & de la Torre, 2016): recover the ordered-category
    step and category probabilities of a polytomous item, and confirm that binary data
    (one step per item) reduces exactly to :func:`fit_gdina`. The M=2 truth is additive
    on neither trivial link — its two step tables are distinct and asymmetric, so the
    ordered structure (not a degenerate collapse) is what is recovered."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_seq_gdina, SeqGdinaFit, fit_gdina
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina"):
        pytest.skip("compiled core built without fit_seq_gdina")

    rng = np.random.default_rng(2016)
    k, n = 2, 6000
    # 4 single-attribute items per attribute (M=1 identification) + 4 pair M=2 items.
    rows = [[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 4
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    # per reduced-class step tables (class-major [00,10,01,11]); asymmetric, increasing.
    pair_s1 = {0: 0.25, 1: 0.55, 2: 0.50, 3: 0.85}
    pair_s2 = {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.70}
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.zeros((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 8:
                a = i // 4
                p = 0.85 if (c >> a) & 1 else 0.15
                y[j, i] = 1.0 if rng.random() < p else 0.0
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)
                cat = 0
                if rng.random() < pair_s1[l]:
                    cat = 1
                    if rng.random() < pair_s2[l]:
                        cat = 2
                y[j, i] = float(cat)

    res = fit_seq_gdina(y, q)
    assert isinstance(res, SeqGdinaFit) and res.converged
    assert res.termination_reason == "tolerance_met"
    assert abs(res.final_loglik_change) < res.stopping_tolerance
    assert np.isfinite(res.final_relative_loglik_change)
    assert res.max_cat.tolist() == [1] * 8 + [2] * 4  # M_i derived from data
    assert abs(res.profile_prob.sum() - 1.0) < 1e-9
    # category probabilities sum to 1 per (item, reduced class)
    for i in range(n_items):
        cp = res.item_cat_prob(i)
        assert cp.shape == (1 << int(res.k_required[i]), int(res.max_cat[i]) + 1)
        assert np.allclose(cp.sum(axis=1), 1.0, atol=1e-9)
    # recover the pair items' category probabilities (stable, model-predicted quantity)
    for i in range(8, n_items):
        cp = res.item_cat_prob(i)  # 4 classes x 3 categories
        for l in range(4):
            s1, s2 = pair_s1[l], pair_s2[l]
            truth = np.array([1 - s1, s1 * (1 - s2), s1 * s2])
            assert np.max(np.abs(cp[l] - truth)) < 0.04, f"item{i} class{l}: {cp[l]} vs {truth}"
    # attribute classification
    est = (res.attr_prob >= 0.5).astype(int)
    alpha = ((profiles[:, None] >> np.arange(k)) & 1)
    assert (est == alpha).mean() > 0.9

    # Binary data (M_i = 1 for all items) reduces to fit_gdina bit-for-bit.
    ybin = (y[:, :8] > 0).astype(float)
    qbin = q[:8]
    sq1 = fit_seq_gdina(ybin, qbin)
    g1 = fit_gdina(ybin, qbin)
    assert sq1.max_cat.tolist() == [1] * 8
    assert np.allclose(sq1.step_prob, g1.item_prob, atol=1e-12)
    assert len(sq1.loglik_trace) == len(g1.loglik_trace)
    assert np.allclose(sq1.loglik_trace, g1.loglik_trace, atol=1e-12)

    # Validation: an item stuck at category 0 (measures nothing) is rejected; a
    # non-integer category is rejected; missing (NaN) is allowed.
    with pytest.raises(ValueError):
        fit_seq_gdina(y.ravel(), q)  # not 2-D
    yzero = y.copy()
    yzero[:, 8] = 0.0
    with pytest.raises(ValueError, match="never leaves category 0"):
        fit_seq_gdina(yzero, q)
    ybad = y.copy()
    ybad[0, 8] = 1.5
    with pytest.raises(ValueError, match="non-negative integer category"):
        fit_seq_gdina(ybad, q)
    ymiss = y.copy()
    ymiss[0, 0] = np.nan
    assert fit_seq_gdina(ymiss, q).converged

    unfinished = fit_seq_gdina(y[:100], q, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert unfinished.termination_reason == "max_iter_reached"
    assert np.isfinite(unfinished.final_loglik_change)
    assert np.isfinite(unfinished.final_relative_loglik_change)
    assert unfinished.stopping_tolerance == 1e-12


def test_fit_seq_gdina_qr_per_step_q_reduces_and_recovers_structure():
    """Per-step-Q sequential G-DINA (Ma & de la Torre, 2016, restricted-Q full model):
    each ordered step of an item may require its OWN attributes. Three guards:

    (1) SHARED-Q REDUCTION -- expanding every step of an item to the item's Q reproduces
        :func:`fit_seq_gdina` BIT-EXACTLY (layout-aware step_prob compare: item-major
        ``s_off[i]+l*M_i+(k-1)`` vs step-row-major ``spo[step_off[i]+(k-1)]+l``; cat_prob
        and loglik_trace are class-major and compared directly). A dimension-map, layout,
        or union-collapse bug fails this exact-zero guard.
    (2) STRUCTURE -- item0 step1 q={A} (block width 2^1=2), step2 q={A,B} (width 2^2=4):
        the per-step widths and n_parameters must reflect the distinct step Qs, NOT a
        single union block. A large B-contrast in step 2 (s2(A1,B0)=0.20 vs s2(A1,B1)=0.80,
        gap 0.60) is recovered (gap >= 0.4) while the union stays lossless. Value recovery
        alone can't catch an over-collapse to the union; the width assertions can.
    (3) VALIDATION -- all-zero step row (a step measuring nothing), an attribute used by no
        step (all-zero union column), and n_steps != observed max are all rejected."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_seq_gdina, fit_seq_gdina_qr, SeqGdinaQrFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_seq_gdina_qr"):
        pytest.skip("compiled core built without fit_seq_gdina_qr")

    rng = np.random.default_rng(2016_3)
    k, n = 2, 6000
    # 4 single-attribute M=1 items per attribute (identification) + 4 shared-Q M=2 pairs.
    rows = [[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 4
    q = np.array(rows, dtype=np.int64)
    n_items = q.shape[0]
    pair_s1 = {0: 0.25, 1: 0.55, 2: 0.50, 3: 0.85}
    pair_s2 = {0: 0.15, 1: 0.30, 2: 0.25, 3: 0.70}
    profiles = rng.integers(0, 1 << k, size=n)
    y = np.zeros((n, n_items))
    for j in range(n):
        c = int(profiles[j])
        for i in range(n_items):
            if i < 8:
                a = i // 4
                p = 0.85 if (c >> a) & 1 else 0.15
                y[j, i] = 1.0 if rng.random() < p else 0.0
            else:
                l = (c & 1) + 2 * ((c >> 1) & 1)
                cat = 0
                if rng.random() < pair_s1[l]:
                    cat = 1
                    if rng.random() < pair_s2[l]:
                        cat = 2
                y[j, i] = float(cat)

    # (1) Shared-Q reduction: expand item Q into per-step rows sharing the item Q.
    n_steps = np.array([1] * 8 + [2] * 4, dtype=np.int64)
    step_rows = []
    for i in range(n_items):
        for _ in range(int(n_steps[i])):
            step_rows.append(q[i])
    step_q = np.vstack(step_rows).astype(np.int64)

    sh = fit_seq_gdina(y, q, tol=1e-8)
    qr = fit_seq_gdina_qr(y, step_q, n_steps, tol=1e-8)
    assert isinstance(qr, SeqGdinaQrFit) and qr.converged
    assert qr.termination_reason == "tolerance_met"
    assert qr.max_cat.tolist() == [1] * 8 + [2] * 4
    # cat_prob and loglik_trace are class-major: direct bit-exact compare.
    assert sh.cat_prob.shape == qr.cat_prob.shape
    assert np.array_equal(sh.cat_prob, qr.cat_prob)
    assert len(sh.loglik_trace) == len(qr.loglik_trace)
    assert np.array_equal(sh.loglik_trace, qr.loglik_trace)
    # step_prob layouts are transposed: cell-by-cell exact-zero difference.
    for i in range(n_items):
        Mi = int(n_steps[i])
        width = 1 << int(q[i].sum())
        for l in range(width):
            for kk in range(1, Mi + 1):
                sh_val = sh.step_prob[int(sh.s_off[i]) + l * Mi + (kk - 1)]
                qr_val = qr.item_step_prob(i, kk)[l]
                assert sh_val == qr_val, f"item{i} l{l} k{kk}: {sh_val} vs {qr_val}"

    # (2) Structure: distinct per-step Qs the shared-Q model cannot represent.
    #     item0 step1={A}, step2={A,B}; item1 M=1 {A}, item2 M=1 {B} pin both dims.
    step_q2 = np.array([[1, 0], [1, 1], [1, 0], [0, 1]], dtype=np.int64)
    n_steps2 = np.array([2, 1, 1], dtype=np.int64)
    s2_by_class = {0: 0.15, 1: 0.20, 2: 0.30, 3: 0.80}  # big B-contrast at A=1
    n2 = 8000
    al2 = rng.integers(0, 2, size=(n2, k))
    Y2 = np.zeros((n2, 3))
    for j in range(n2):
        a0, a1 = int(al2[j, 0]), int(al2[j, 1])
        # item0
        if rng.random() < (0.25 + 0.5 * a0):
            Y2[j, 0] = 1
            rcAB = a0 + 2 * a1
            if rng.random() < s2_by_class[rcAB]:
                Y2[j, 0] = 2
        Y2[j, 1] = 1.0 if rng.random() < (0.2 + 0.6 * a0) else 0.0
        Y2[j, 2] = 1.0 if rng.random() < (0.2 + 0.6 * a1) else 0.0
    qr2 = fit_seq_gdina_qr(Y2, step_q2, n_steps2, max_iter=1000, tol=1e-8)
    # per-step block widths reflect the distinct Qs (2 and 4), not a single union block.
    assert len(qr2.item_step_prob(0, 1)) == 2
    assert len(qr2.item_step_prob(0, 2)) == 4
    assert qr2.step_kq.tolist() == [1, 2, 1, 1]  # |q_ik| per step row
    # n_parameters = total step cells + (2^K - 1) free profile weights.
    assert qr2.n_parameters == (2 + 4 + 2 + 2) + ((1 << k) - 1)
    # large B-contrast recovered in step 2 (class A1B1 minus A1B0).
    s2 = qr2.item_step_prob(0, 2)
    assert s2[3] - s2[1] >= 0.4, f"B-gap too small: {s2}"
    # category space: P(X=2 | A1B1) >> P(X=2 | A1B0), P(X>=1) roughly equal.
    cp = qr2.cat_prob[int(qr2.cat_off[0]):int(qr2.cat_off[0]) + 4 * 3].reshape(4, 3)
    assert cp[3, 2] - cp[1, 2] >= 0.3
    assert abs((1 - cp[3, 0]) - (1 - cp[1, 0])) < 0.15  # P(X>=1) close across B
    # B is pinned by a single M=1 item (0.20/0.80 split), so its Bayes-optimal recovery is
    # ~0.8; well above the 0.5 chance rate, confirming both latent dims are identified.
    est = (qr2.attr_prob >= 0.5).astype(int)
    assert (est == al2).mean() > 0.75

    # (3) Validation.
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2.ravel(), step_q2, n_steps2)  # not 2-D
    zero_row = step_q2.copy()
    zero_row[1] = [0, 0]  # a step measuring nothing
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2, zero_row, n_steps2)
    dead_col = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.int64)  # attr B unused
    with pytest.raises(ValueError):
        fit_seq_gdina_qr(Y2, dead_col, n_steps2)
    with pytest.raises(ValueError, match="sum"):
        fit_seq_gdina_qr(Y2, step_q2, np.array([3, 1, 1], dtype=np.int64))  # wrapper: rows != sum(n_steps)
    # max observed category != declared n_steps -- reaches the Rust guard, NOT the wrapper's
    # row-count guard: keep sum(n_steps)=4 (matches step_q2's 4 rows) but let item0 declare 2
    # steps while the data never reaches category 2 (else x=y could index clp past item0's block).
    y_low = Y2.copy()
    y_low[y_low[:, 0] == 2, 0] = 1
    with pytest.raises(ValueError, match="max observed category"):
        fit_seq_gdina_qr(y_low, step_q2, n_steps2)
    ymiss = Y2.copy()
    ymiss[0, 0] = np.nan
    fm = fit_seq_gdina_qr(ymiss, step_q2, n_steps2, max_iter=1000)  # MAR: dropped, no crash
    assert np.isfinite(fm.loglik_trace[-1]) and abs(fm.profile_prob.sum() - 1.0) < 1e-9


def test_fit_crm_recovers_continuous_responses():
    """Continuous Response Model (Samejima, 1973): recover the item slope/intercept/
    residual-sd and the Samejima discrimination/difficulty from continuous bounded
    responses, plus the trait (continuous responses are information-rich)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_crm, CrmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_crm"):
        pytest.skip("compiled core built without fit_crm")

    rng = np.random.default_rng(1973)
    n_items, n = 15, 1500
    a_true = 0.8 + 0.05 * np.arange(n_items)
    d_true = -0.6 + 0.08 * np.arange(n_items)
    sigma_true = 0.6 + 0.02 * (np.arange(n_items) % 5)
    theta = rng.standard_normal(n)
    x = a_true * theta[:, None] + d_true + sigma_true * rng.standard_normal((n, n_items))
    z = 1.0 / (1.0 + np.exp(-x))  # in (0,1)

    res = fit_crm(z)
    assert isinstance(res, CrmFit) and res.converged
    assert res.termination_reason == "tolerance"
    assert res.final_delta <= res.stopping_tolerance
    assert res.n_iter + 1 == len(res.loglik_trace)
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == 3 * n_items
    assert np.all(res.slope > 0)  # reflection convention
    assert np.sqrt(np.mean((res.slope - a_true) ** 2)) < 0.15
    assert np.sqrt(np.mean((res.intercept - d_true) ** 2)) < 0.1
    assert np.sqrt(np.mean((res.resid_sd - sigma_true) ** 2)) < 0.1
    # Samejima re-parameterization
    assert np.sqrt(np.mean((res.discrimination - a_true / sigma_true) ** 2)) < 0.3
    assert np.sqrt(np.mean((res.difficulty - (-d_true / a_true)) ** 2)) < 0.2
    # trait recovery
    assert np.corrcoef(res.theta, theta)[0, 1] > 0.9

    limited = fit_crm(z, max_iter=1, tol=1e-12)
    assert not limited.converged
    assert limited.termination_reason == "max_iter"
    assert limited.n_iter == 1
    assert limited.final_delta > limited.stopping_tolerance

    # missing-at-random handling
    zm = z.copy()
    zm[rng.random(zm.shape) < 0.15] = np.nan
    assert fit_crm(zm).converged

    with pytest.raises(ValueError):
        fit_crm(z.ravel())  # responses not 2-D
    with pytest.raises(ValueError):
        fit_crm(np.full((4, 3), 1.5))  # outside (0,1)
    with pytest.raises(ValueError, match="at least one person"):
        fit_crm(np.empty((0, 3)))
    with pytest.raises(ValueError, match="at least one person"):
        fit_crm(np.empty((3, 0)))
    with pytest.raises(ValueError, match="max_iter"):
        fit_crm(z, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        fit_crm(z, tol=np.nan)
    with pytest.raises(ValueError, match="no observed responses"):
        missing_item = z.copy()
        missing_item[:, 0] = np.nan
        fit_crm(missing_item)


def test_fit_rsm_recovers_shared_thresholds():
    """Rating Scale Model (Andrich, 1978): recover item locations and the shared
    category thresholds (centered) plus the trait; K=2 reduces to Rasch."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_rsm, RsmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        pytest.skip("compiled core built without fit_rsm")

    rng = np.random.default_rng(1978)
    n_items, n_cat, n = 12, 5, 2500
    delta_true = -1.2 + 0.2 * np.arange(n_items)
    tau_true = np.array([0.9, 0.2, -0.3, -0.8])  # sums to 0
    theta = rng.standard_normal(n)

    def draw(th, d):
        # cumulative psi_k = k*th - k*d - sum_{m<=k} tau
        tk = np.concatenate([[0.0], np.cumsum(tau_true)])
        psi = np.arange(n_cat) * th - np.arange(n_cat) * d - tk
        p = np.exp(psi - psi.max())
        p /= p.sum()
        return rng.choice(n_cat, p=p)

    y = np.array([[draw(theta[j], delta_true[i]) for i in range(n_items)] for j in range(n)],
                 dtype=float)

    res = fit_rsm(y)
    assert isinstance(res, RsmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # monotone ascent
    assert res.n_parameters == n_items + n_cat - 2
    assert abs(res.thresholds.sum()) < 1e-6  # centered
    assert np.sqrt(np.mean((res.item_location - delta_true) ** 2)) < 0.15
    assert np.sqrt(np.mean((res.thresholds - tau_true) ** 2)) < 0.12
    assert np.corrcoef(res.theta, theta)[0, 1] > 0.85

    # missing-at-random
    ym = y.copy()
    ym[rng.random(ym.shape) < 0.15] = np.nan
    assert fit_rsm(ym).converged

    with pytest.raises(ValueError):
        fit_rsm(y.ravel())  # not 2-D


def test_fit_rsm_rejects_unidentified_or_malformed_inputs():
    """RSM must not report convergence for data that cannot identify a fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_rsm
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_rsm"):
        pytest.skip("compiled core built without fit_rsm")

    valid = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError, match="at least one person"):
        fit_rsm(np.empty((0, 2)), n_cat=2)
    with pytest.raises(ValueError, match="at least one person"):
        fit_rsm(np.empty((2, 0)), n_cat=2)
    with pytest.raises(ValueError, match="integer categories"):
        fit_rsm(np.array([[0.2, 1.0], [1.0, 0.0]]), n_cat=2)
    with pytest.raises(ValueError, match="finite integer categories"):
        fit_rsm(np.array([[0.0, np.inf], [1.0, 0.0]]), n_cat=2)
    with pytest.raises(ValueError, match="item 1 has no observed responses"):
        fit_rsm(np.array([[0.0, np.nan], [1.0, np.nan]]), n_cat=2)
    with pytest.raises(ValueError, match="max_iter"):
        fit_rsm(valid, n_cat=2, max_iter=0)
    with pytest.raises(ValueError, match="tol"):
        fit_rsm(valid, n_cat=2, tol=np.inf)

    unfinished = fit_rsm(valid, n_cat=2, max_iter=1)
    assert not unfinished.converged
    assert unfinished.n_iter == 1
    assert len(unfinished.loglik_trace) == 2
    assert np.all(np.isfinite(unfinished.loglik_trace))


def test_fit_2pl_recovers_confirmatory_loadings():
    """Compensatory MIRT (Reckase, 2009): recover a confirmatory 2-dimensional loading
    pattern (dim0-only, dim1-only, and BOTH-loading items) including a genuinely NEGATIVE
    loading, plus the per-dimension trait EAP; and reject a rotationally-degenerate
    (all-ones) pattern."""
    import numpy as np
    import pytest
    from fast_mlsirm import TwoPlFit, fit_2pl, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        pytest.skip("compiled core built without fit_2pl")

    rng = np.random.default_rng(2009)
    n, n_dims = 4000, 2
    pattern = np.array([[1, 0]] * 4 + [[0, 1]] * 4 + [[1, 1]] * 3, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    loading[:4, 0] = [1.2, 0.8, 1.5, -0.9]  # dim0-only, incl. a negative loading
    loading[4:8, 1] = [1.0, 1.3, 0.7, 1.1]  # dim1-only
    loading[8:] = [[0.9, 1.1], [1.2, -0.7], [0.8, 0.9]]  # both, incl. a negative cross-loading
    intercept = np.linspace(-0.8, 0.8, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_2pl(y, model=models.confirmatory(pattern), q=21)
    assert isinstance(res, TwoPlFit) and res.converged
    assert res.loading.shape == (n_items, n_dims) and res.n_dims == 2
    # off-pattern entries are exactly zero
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert np.sqrt(np.mean((res.loading - loading) ** 2)) < 0.13
    # negative loadings recovered with the correct sign (needs the symmetric clamp)
    assert res.loading[3, 0] < -0.5
    assert res.loading[9, 1] < -0.3
    # per-dimension trait recovery (positive correlation; a sign/swap bug -> near 0 or negative)
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.7, f"dim {d} theta corr {c}"
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # EM monotone
    assert res.termination_reason == "converged"
    assert res.n_iter < 500
    assert np.isfinite(res.final_loglik_change)
    assert res.final_loglik_change < 1e-6

    # a rotationally-degenerate all-ones pattern is rejected (no pure anchor per dimension)
    with pytest.raises(ValueError):
        fit_2pl(y, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)))
    fractional = pattern.astype(float)
    fractional[0, 1] = 0.5
    with pytest.raises(ValueError, match="exactly 0 or 1"):
        fit_2pl(y, model=models.confirmatory(fractional))
    with pytest.raises(ValueError, match="q must be a finite integer"):
        fit_2pl(y, model=models.confirmatory(pattern), q=15.5)
    with pytest.raises(ValueError, match="max_iter must be a finite integer"):
        fit_2pl(y, model=models.confirmatory(pattern), max_iter=1.5)
    # missing (MAR) handled
    ymiss = y.copy()
    ymiss[0, 0] = np.nan
    assert fit_2pl(ymiss, model=models.confirmatory(pattern), q=15).converged

    # A one-step run that has not met the documented tolerance is explicitly unfinished.
    unfinished = fit_2pl(y, model=models.confirmatory(pattern), q=7, max_iter=1, tol=1e-12)
    assert not unfinished.converged
    assert unfinished.termination_reason == "max_iter_reached"
    assert unfinished.n_iter == 1
    assert len(unfinished.loglik_trace) == 2
    assert unfinished.final_loglik_change >= 1e-12

    # estimate_corr=False reports Sigma = I; estimate_corr=True recovers a known correlation.
    ortho = fit_2pl(y, model=models.confirmatory(pattern), q=15, estimate_corr=False)
    assert np.allclose(ortho.corr, np.eye(n_dims))
    ncorr = np.linalg.cholesky(np.array([[1.0, 0.5], [0.5, 1.0]]))
    thc = (ncorr @ rng.standard_normal((n_dims, n))).T
    pc = 1.0 / (1.0 + np.exp(-(thc @ loading.T + intercept)))
    yc = (rng.random((n, n_items)) < pc).astype(float)
    rc = fit_2pl(yc, model=models.confirmatory(pattern), q=15, estimate_corr=True)
    assert rc.corr.shape == (n_dims, n_dims)
    assert np.allclose(np.diag(rc.corr), 1.0) and np.allclose(rc.corr, rc.corr.T)
    realized = np.corrcoef(thc.T)[0, 1]
    assert abs(rc.corr[0, 1] - realized) < 0.06, f"corr {rc.corr[0, 1]} vs realized {realized}"
    assert np.all(np.linalg.eigvalsh(rc.corr) > 0)  # positive-definite


def test_fit_2pl_qmc_high_dim():
    """QMC compensatory MIRT (Jank, 2005): the D>3 quasi-Monte-Carlo path the Gauss-Hermite
    product grid cannot reach. Recovers a D=4 confirmatory loading pattern (2 pure anchors per
    dimension + cross-loaders including a genuine NEGATIVE one) on Halton nodes; confirms the GH
    path still caps D<=3 while QMC/MC reach D<=6; and checks the wrapper plumbing is two-sided
    (a D<=3 QMC fit agrees with GH within QMC error but is NOT a silent bit-identical GH fallback)."""
    import numpy as np
    import pytest
    from fast_mlsirm import TwoPlFit, fit_2pl, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_2pl"):
        pytest.skip("compiled core built without fit_2pl")

    rng = np.random.default_rng(2005)
    n, n_dims = 2500, 4
    # 2 pure anchors per dim + 3 cross-loaders (one with a negative dim-1 loading).
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 2
    rows += [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        loading[2 * d, d] = 1.2 + 0.1 * d
        loading[2 * d + 1, d] = 0.9
    cross = 2 * n_dims
    loading[cross] = [1.0, -0.8, 0.0, 0.0]  # the negative cross-loader
    loading[cross + 1] = [0.0, 1.1, 0.7, 0.0]
    loading[cross + 2] = [0.0, 0.0, 0.8, 1.0]
    intercept = np.linspace(-0.5, 0.7, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    # GH cannot reach D=4; QMC (Halton) can.
    with pytest.raises(ValueError):
        fit_2pl(y, model=models.confirmatory(pattern), node_rule="gh")
    res = fit_2pl(y, model=models.confirmatory(pattern), node_rule="qmc", xi_points=4000, xi_seed=12345)
    assert isinstance(res, TwoPlFit) and res.n_dims == 4
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert np.sqrt(np.mean((res.loading - loading) ** 2)) < 0.18
    assert res.loading[cross, 1] < -0.3  # negative cross-loader recovered with sign
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.55, f"dim {d} theta corr {c}"
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)  # EM monotone

    # node_rule validation and D<=6 bounds.
    with pytest.raises(ValueError, match="node_rule"):
        fit_2pl(y, model=models.confirmatory(pattern), node_rule="nope")
    pat7 = np.eye(7, dtype=np.int64)
    y7 = (rng.random((200, 7)) < 0.5).astype(float)
    with pytest.raises(ValueError):
        fit_2pl(y7, model=models.confirmatory(pat7), node_rule="qmc", xi_points=200)  # D=7 > 6

    # Two-sided wrapper plumbing at D=2: GH and QMC agree within QMC error yet differ bit-wise
    # (a silent GH fallback on the QMC arm would make them identical).
    pat2 = np.array([[1, 0]] * 3 + [[0, 1]] * 3 + [[1, 1]], dtype=np.int64)
    ld2 = np.zeros((7, 2))
    for i in range(3):
        ld2[i, 0] = 1.0 + 0.1 * i
        ld2[3 + i, 1] = 1.0
    ld2[6] = [0.9, 0.8]
    ic2 = np.linspace(-0.4, 0.5, 7)
    th2 = rng.standard_normal((2000, 2))
    p2 = 1.0 / (1.0 + np.exp(-(th2 @ ld2.T + ic2)))
    y2 = (rng.random((2000, 7)) < p2).astype(float)
    gh2 = fit_2pl(y2, model=models.confirmatory(pat2), q=21, node_rule="gh")
    qmc2 = fit_2pl(y2, model=models.confirmatory(pat2), node_rule="qmc", xi_points=6000, xi_seed=0)
    max_abs = max(
        np.max(np.abs(gh2.loading - qmc2.loading)),
        np.max(np.abs(gh2.intercept - qmc2.intercept)),
    )
    assert max_abs < 0.10, f"QMC vs GH beyond QMC error: {max_abs}"
    assert max_abs > 1e-10, "QMC fit bit-identical to GH (silent fallback?)"


def test_fit_mhrm_recovers_high_dimensional_2pl():
    """MH-RM (Cai, 2010): high-dimensional confirmatory 2PL by Metropolis-Hastings Robbins-Monro
    stochastic approximation. Recovers a D=6 confirmatory loading pattern — the q**D Gauss-Hermite
    grid (21**6 ~ 8.6e7) and even the QMC E-step are infeasible at this dimensionality, which is the
    module's reason to exist — including a genuine NEGATIVE cross-loader, with reflection-
    canonicalized signs, per-dimension trait recovery, finite Louis observed-information SEs, and a
    tuned acceptance rate; and rejects rotationally-degenerate patterns and non-binary responses."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(2010)
    n, n_dims = 3000, 6
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 3  # 3 pure anchors per dim
    cross = [0] * n_dims
    cross[0] = 1
    cross[3] = 1
    rows.append(cross)  # one cross-loader on dims 0 and 3
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(3):
            loading[3 * d + a, d] = 0.9 + 0.1 * a
    xi = n_items - 1
    loading[xi, 0] = 1.0
    loading[xi, 3] = -0.7  # negative cross-loader
    intercept = np.linspace(-0.5, 0.6, n_items)
    theta = rng.standard_normal((n, n_dims))
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=1400, burn_in=280, mh_steps=8, seed=7)
    assert isinstance(res, MhrmFit) and res.n_dims == 6
    assert res.loading.shape == (n_items, n_dims)
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert res.n_parameters == int(pattern.sum()) + n_items
    onpat = pattern == 1
    assert np.sqrt(np.mean((res.loading[onpat] - loading[onpat]) ** 2)) < 0.22
    assert res.loading[xi, 3] < -0.3  # negative cross-loader recovered with sign
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.5, f"dim {d} theta corr {c}"
    # Louis SEs: right shape, finite on-pattern
    assert res.se_loading.shape == (n_items, n_dims)
    assert np.all(np.isfinite(res.se_loading[onpat]))
    assert res.se_intercept.shape == (n_items,)
    # acceptance auto-tuned into a sane band
    assert 0.1 < res.acceptance_rate < 0.7, res.acceptance_rate

    # rotationally-degenerate pattern (every item loads all dims -> no pure anchor) rejected
    with pytest.raises(ValueError):
        fit_mhrm(y, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)))
    # non-binary response rejected
    with pytest.raises(ValueError):
        ybad = y.copy()
        ybad[0, 0] = 2
        fit_mhrm(ybad, model=models.confirmatory(pattern))


def test_fit_mhrm_estimate_corr_recovers_factor_correlation():
    """MH-RM with estimate_corr (Cai, 2010b): recover a free latent factor CORRELATION at D=2 from
    theta ~ MVN(0, Phi), and confirm estimate_corr=False yields exactly the identity."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(2010)
    n, n_dims, rho = 3000, 2, 0.5
    per = 4
    n_items = per * n_dims
    pattern = np.zeros((n_items, n_dims), dtype=np.int64)
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(per):
            pattern[d * per + a, d] = 1
            loading[d * per + a, d] = 1.0 + 0.1 * a
    intercept = np.linspace(-0.4, 0.5, n_items)
    phi = np.array([[1.0, rho], [rho, 1.0]])
    theta = rng.multivariate_normal(np.zeros(n_dims), phi, size=n)
    p = 1.0 / (1.0 + np.exp(-(theta @ loading.T + intercept)))
    y = (rng.random((n, n_items)) < p).astype(float)

    res = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=1500, burn_in=320,
                   mh_steps=8, estimate_corr=True, seed=3)
    assert isinstance(res, MhrmFit)
    assert res.corr.shape == (n_dims, n_dims)
    assert np.allclose(np.diag(res.corr), 1.0)
    assert abs(res.corr[0, 1] - rho) < 0.12, res.corr[0, 1]
    assert res.n_parameters == n_items + n_items + n_dims * (n_dims - 1) // 2

    # estimate_corr=False -> exactly the identity, and fewer parameters
    res0 = fit_mhrm(y, model=models.confirmatory(pattern), max_cycles=400, burn_in=100,
                    estimate_corr=False, seed=3)
    assert np.array_equal(res0.corr, np.eye(n_dims))
    assert res0.n_parameters == n_items + n_items


def test_fit_mhrm_gpcm_recovers_high_dimensional_polytomous():
    """MH-RM GPCM (Muraki, 1992; Cai, 2010): high-dimensional confirmatory GENERALIZED PARTIAL CREDIT
    model by Metropolis-Hastings Robbins-Monro. Recovers a D=3 confirmatory pattern of loadings and
    UNORDERED step intercepts — the q**D Gauss-Hermite grid and the QMC E-step are infeasible for a
    polytomous item factor model at this dimensionality — including a genuinely NEGATIVE cross-loader,
    with reflection-canonicalized signs; exposes the family/n_cat/step result shape (intercept empty);
    and rejects out-of-range responses and a never-observed category (an unidentified step)."""
    import numpy as np
    import pytest
    from fast_mlsirm import MhrmFit, fit_mhrm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mhrm"):
        pytest.skip("compiled core built without fit_mhrm")

    rng = np.random.default_rng(1992)
    n, n_dims, n_cat = 3000, 3, 3
    rows = []
    for d in range(n_dims):
        rows += [[1 if k == d else 0 for k in range(n_dims)]] * 3  # 3 pure anchors per dim
    cross = [0] * n_dims
    cross[0] = 1
    cross[2] = 1
    rows.append(cross)  # one cross-loader on dims 0 and 2
    pattern = np.array(rows, dtype=np.int64)
    n_items = pattern.shape[0]
    loading = np.zeros((n_items, n_dims))
    for d in range(n_dims):
        for a in range(3):
            loading[3 * d + a, d] = 0.9 + 0.1 * a
    xi = n_items - 1
    loading[xi, 0] = 1.0
    loading[xi, 2] = -0.7  # negative cross-loader
    # non-monotone (unordered) steps per item
    step = np.column_stack([
        0.7 - 0.12 * (np.arange(n_items) % 3),
        -0.4 + 0.1 * (np.arange(n_items) % 4),
    ])
    theta = rng.standard_normal((n, n_dims))
    base = theta @ loading.T  # (n, n_items), no intercept
    # psi_k = k*base + step_k (step_0 = 0); sample categories from the softmax
    ks = np.arange(n_cat)
    full_step = np.hstack([np.zeros((n_items, 1)), step])  # (n_items, n_cat)
    psi = base[:, :, None] * ks[None, None, :] + full_step[None, :, :]  # (n, n_items, n_cat)
    psi -= psi.max(axis=2, keepdims=True)
    prob = np.exp(psi)
    prob /= prob.sum(axis=2, keepdims=True)
    u = rng.random((n, n_items))
    y = (u[:, :, None] > np.cumsum(prob, axis=2)).sum(axis=2).astype(float)  # inverse-CDF draw

    res = fit_mhrm(y, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat,
                   max_cycles=1200, burn_in=250, mh_steps=6, seed=11)
    assert isinstance(res, MhrmFit) and res.n_dims == n_dims
    assert res.family == "gpcm" and res.n_cat == n_cat
    assert res.loading.shape == (n_items, n_dims)
    assert res.step.shape == (n_items, n_cat - 1)
    assert res.intercept.size == 0  # 2PL intercept empty for GPCM
    assert res.se_step.shape == (n_items, n_cat - 1)
    assert np.all(res.loading[pattern == 0] == 0.0)
    assert res.n_parameters == int(pattern.sum()) + n_items * (n_cat - 1)
    onpat = pattern == 1
    assert np.sqrt(np.mean((res.loading[onpat] - loading[onpat]) ** 2)) < 0.25
    assert res.loading[xi, 2] < -0.3  # negative cross-loader recovered with sign
    assert np.sqrt(np.mean((res.step - step) ** 2)) < 0.25
    for d in range(n_dims):
        c = np.corrcoef(res.theta[:, d], theta[:, d])[0, 1]
        assert c > 0.5, f"dim {d} theta corr {c}"

    # out-of-range response (== n_cat) rejected
    with pytest.raises(ValueError):
        ybad = y.copy()
        ybad[0, 0] = n_cat
        fit_mhrm(ybad, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat)
    # a declared category never observed for an item (unidentified step) rejected
    with pytest.raises(ValueError):
        ycov = y.copy()
        ycov[ycov[:, 0] == 1, 0] = 0  # item 0 never shows category 1
        fit_mhrm(ycov, model=models.confirmatory(pattern), family="gpcm", n_cat=n_cat)


def test_fit_nominal_recovers_confirmatory_multidimensional_categories():
    """Confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen-Cai-Bock, 2010):
    recover a D=2 confirmatory pattern of CATEGORY-SPECIFIC multidimensional slopes (unordered
    categories) including a genuinely NEGATIVE cross-loader slope with an OPPOSITE-sign sibling
    category on the same dimension (the signature a per-item RMSE would average away), assessed up
    to per-dimension reflection; confirm the baseline/off-pattern slopes are exactly zero; and
    reject rotationally-degenerate patterns, out-of-range and unobserved categories, and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import NominalResponseFit, fit_nominal, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_nominal_model"):
        pytest.skip("compiled core built without fit_nominal_model")

    rng = np.random.default_rng(1972)
    n_dims, n_cat, n = 2, 3, 6000
    # items 0,1 pure dim0; items 2,3 pure dim1; item 4 cross-loader {0,1}.
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    anchor = [0, 2]  # pure anchor item per dim
    slope = np.zeros((n_items, n_cat, n_dims))
    slope[0, 1, 0], slope[0, 2, 0] = 1.4, 0.8
    slope[1, 1, 0], slope[1, 2, 0] = 1.0, 1.3
    slope[2, 1, 1], slope[2, 2, 1] = 1.2, 0.9
    slope[3, 1, 1], slope[3, 2, 1] = 1.1, 1.4
    slope[4, 1, 0], slope[4, 2, 0] = -1.1, 1.0   # negative + positive sibling on dim0
    slope[4, 1, 1], slope[4, 2, 1] = 0.9, 0.7
    intercept = np.zeros((n_items, n_cat))
    for i in range(n_items):
        for k in range(1, n_cat):
            intercept[i, k] = -0.2 + 0.15 * k - 0.05 * i
    theta = rng.standard_normal((n, n_dims))
    eta = np.zeros((n, n_items, n_cat))
    for k in range(1, n_cat):
        eta[:, :, k] = theta @ slope[:, k, :].T + intercept[:, k]
    ex = np.exp(eta - eta.max(axis=2, keepdims=True))
    probs = ex / ex.sum(axis=2, keepdims=True)
    u = rng.random((n, n_items))
    y = (probs.cumsum(axis=2) < u[:, :, None]).sum(axis=2)

    res = fit_nominal(y, n_cat, model=models.confirmatory(pattern), q=21)
    assert isinstance(res, NominalResponseFit) and res.converged
    assert res.slope.shape == (n_items, n_cat, n_dims) and res.n_dims == 2 and res.n_cat == 3
    # baseline category and off-pattern entries are EXACTLY zero
    assert np.all(res.slope[:, 0, :] == 0.0)
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert np.all(res.slope[i, :, d] == 0.0)
    assert np.all(res.intercept[:, 0] == 0.0)
    # free-parameter count = sum_i (n_cat-1)*(|S_i|+1)
    assert res.n_parameters == 2 * 2 + 2 * 2 + 2 * 2 + 2 * 2 + 2 * 3

    # per-dimension reflection alignment to truth (same rule applied to est), then compare
    est = res.slope.copy()
    for d in range(n_dims):
        if est[anchor[d], 1, d] * slope[anchor[d], 1, d] < 0:
            est[:, :, d] = -est[:, :, d]
    assert np.sqrt(np.mean((est - slope) ** 2)) < 0.16
    # the negative cross-loader slope and its opposite-sign sibling recovered with the right signs
    assert est[4, 1, 0] < -0.4
    assert est[4, 2, 0] > 0.4
    # per-dim trait EAP correlation (sign-aligned)
    for d in range(n_dims):
        th = res.theta[:, d].copy()
        if res.slope[anchor[d], 1, d] * slope[anchor[d], 1, d] < 0:
            th = -th
        assert np.corrcoef(th, theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH cannot reach D=4
        pat4 = np.eye(4, dtype=np.int64)
        fit_nominal(np.zeros((50, 4), dtype=np.int64), n_cat, model=models.confirmatory(pat4), node_rule="gh")
    with pytest.raises(ValueError):  # no pure anchor for either dim
        fit_nominal(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)))
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy()
        ybad[0, 0] = n_cat
        fit_nominal(ybad, n_cat, model=models.confirmatory(pattern))
    with pytest.raises(ValueError):  # an unobserved category for an item
        ygap = y.copy()
        ygap[ygap[:, 0] == 2, 0] = 1
        fit_nominal(ygap, n_cat, model=models.confirmatory(pattern))


def test_fit_grm_recovers_confirmatory_multidimensional_ordered_categories():
    """Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995):
    recover a D=2 confirmatory pattern of item discrimination vectors (ORDERED categories) including a
    genuinely NEGATIVE cross-loader on a positively-anchored dimension; confirm the recovered
    thresholds are strictly ordered and the baseline reflection is canonicalized (pure anchors
    positive); and reject rotationally-degenerate patterns, out-of-range and unobserved categories,
    and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import GrmFit, fit_grm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_grm"):
        pytest.skip("compiled core built without fit_grm")

    rng = np.random.default_rng(1969)
    n_dims, n_cat, n = 2, 3, 6000
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    slope = np.zeros((n_items, n_dims))
    slope[0, 0], slope[1, 0] = 1.4, 1.0
    slope[2, 1], slope[3, 1] = 1.2, 1.1
    slope[4, 0], slope[4, 1] = -1.0, 0.9   # negative cross-loader on dim0 (anchor item 0 positive)
    threshold = np.zeros((n_items, n_cat - 1))
    for i in range(n_items):
        threshold[i] = [1.1 + 0.05 * i, 0.05 * i - 0.9]  # strictly decreasing
    theta = rng.standard_normal((n, n_dims))
    # simulate via cumulative logits P(Y>=k)=sigmoid(base+beta_{k-1})
    y = np.zeros((n, n_items), dtype=np.int64)
    for i in range(n_items):
        base = theta @ slope[i]
        ge = 1.0 / (1.0 + np.exp(-(base[:, None] + threshold[i][None, :])))  # (n, n_cat-1)
        pk = np.zeros((n, n_cat))
        pk[:, 0] = 1.0 - ge[:, 0]
        for k in range(1, n_cat - 1):
            pk[:, k] = ge[:, k - 1] - ge[:, k]
        pk[:, n_cat - 1] = ge[:, n_cat - 2]
        pk = np.clip(pk, 1e-12, None)
        pk /= pk.sum(axis=1, keepdims=True)
        u = rng.random(n)
        y[:, i] = (pk.cumsum(axis=1) < u[:, None]).sum(axis=1)

    res = fit_grm(y, n_cat, model=models.confirmatory(pattern), q=21)
    assert isinstance(res, GrmFit) and res.converged
    assert res.slope.shape == (n_items, n_dims) and res.threshold.shape == (n_items, n_cat - 1)
    assert res.n_dims == 2 and res.n_cat == 3
    # off-pattern slopes exactly zero
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert res.slope[i, d] == 0.0
    # free-parameter count = sum_i (|S_i| + (n_cat-1))
    assert res.n_parameters == 4 * (1 + 2) + (2 + 2)
    # recovered thresholds strictly decreasing on every item
    assert np.all(res.threshold[:, 0] > res.threshold[:, 1])
    # canonical: pure anchors positive; negative cross-loader recovered negative
    assert res.slope[0, 0] > 0.5 and res.slope[2, 1] > 0.5
    assert res.slope[4, 0] < -0.4, f"neg cross-loader {res.slope[4, 0]}"
    assert np.sqrt(np.mean((res.slope - slope) ** 2)) < 0.16
    for d in range(n_dims):
        assert np.corrcoef(res.theta[:, d], theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH D=4
        fit_grm((np.arange(200).reshape(50, 4) % n_cat).astype(np.int64), n_cat,
                model=models.confirmatory(np.eye(4, dtype=np.int64)), node_rule="gh")
    with pytest.raises(ValueError):  # no pure anchor
        fit_grm(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)))
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy(); ybad[0, 0] = n_cat
        fit_grm(ybad, n_cat, model=models.confirmatory(pattern))
    with pytest.raises(ValueError):  # unobserved category
        ygap = y.copy(); ygap[ygap[:, 0] == 1, 0] = 0
        fit_grm(ygap, n_cat, model=models.confirmatory(pattern))


def test_fit_gpcm_recovers_confirmatory_multidimensional_adjacent_category():
    """Confirmatory MULTIDIMENSIONAL generalized partial credit model (Muraki, 1992): recover a D=2
    confirmatory pattern of item discrimination vectors (INTEGER-scored adjacent-category logits)
    including a genuinely NEGATIVE cross-loader on a positively-anchored dimension; recover the
    UNORDERED category step intercepts numerically (GPCM steps carry no ordering constraint, so a
    monotone canary would be vacuous — RMSE is the only guard); confirm the baseline reflection is
    canonicalized (pure anchors positive) while steps are left unflipped; and reject
    rotationally-degenerate patterns, out-of-range and unobserved categories, and GH D>3."""
    import numpy as np
    import pytest
    from fast_mlsirm import GpcmFit, fit_gpcm, models
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_gpcm"):
        pytest.skip("compiled core built without fit_gpcm")

    rng = np.random.default_rng(1992)
    n_dims, n_cat, n = 2, 4, 6000
    pattern = np.array([[1, 0], [1, 0], [0, 1], [0, 1], [1, 1]], dtype=np.int64)
    n_items = pattern.shape[0]
    slope = np.zeros((n_items, n_dims))
    slope[0, 0], slope[1, 0] = 1.4, 1.0
    slope[2, 1], slope[3, 1] = 1.2, 1.1
    slope[4, 0], slope[4, 1] = -1.0, 0.9   # negative cross-loader on dim0 (anchor item 0 positive)
    # UNORDERED step intercepts gamma_k (psi_k = k*base + gamma_k, gamma_0 = 0); deliberately
    # non-monotone across k to exercise the free-step estimator.
    step = np.array([
        [0.7, -0.4, 0.9],
        [-0.3, 0.6, 0.1],
        [0.5, 0.2, -0.6],
        [-0.2, 0.8, -0.3],
        [0.4, -0.5, 0.7],
    ])
    theta = rng.standard_normal((n, n_dims))
    # simulate via adjacent-category softmax P(Y=k) = softmax_k(k*base + gamma_k)
    y = np.zeros((n, n_items), dtype=np.int64)
    for i in range(n_items):
        base = theta @ slope[i]
        psi = np.zeros((n, n_cat))
        for k in range(1, n_cat):
            psi[:, k] = k * base + step[i, k - 1]
        psi -= psi.max(axis=1, keepdims=True)
        pk = np.exp(psi)
        pk /= pk.sum(axis=1, keepdims=True)
        u = rng.random(n)
        y[:, i] = (pk.cumsum(axis=1) < u[:, None]).sum(axis=1)

    res = fit_gpcm(y, n_cat, model=models.confirmatory(pattern), q=21)
    assert isinstance(res, GpcmFit) and res.converged
    assert res.slope.shape == (n_items, n_dims) and res.step.shape == (n_items, n_cat - 1)
    assert res.n_dims == 2 and res.n_cat == 4
    # off-pattern slopes exactly zero
    for i in range(n_items):
        for d in range(n_dims):
            if pattern[i, d] == 0:
                assert res.slope[i, d] == 0.0
    # free-parameter count = sum_i (|S_i| + (n_cat-1))
    assert res.n_parameters == 4 * (1 + 3) + (2 + 3)
    # canonical: pure anchors positive; negative cross-loader recovered negative
    assert res.slope[0, 0] > 0.5 and res.slope[2, 1] > 0.5
    assert res.slope[4, 0] < -0.4, f"neg cross-loader {res.slope[4, 0]}"
    assert np.sqrt(np.mean((res.slope - slope) ** 2)) < 0.16
    # UNORDERED steps recovered numerically (no ordering canary possible for GPCM)
    assert np.sqrt(np.mean((res.step - step) ** 2)) < 0.16, f"step RMSE {res.step}"
    for d in range(n_dims):
        assert np.corrcoef(res.theta[:, d], theta[:, d])[0, 1] > 0.6
    assert np.all(np.diff(res.loglik_trace) >= -1e-9)  # EM monotone

    # validation
    with pytest.raises(ValueError):  # GH D=4
        fit_gpcm((np.arange(200).reshape(50, 4) % n_cat).astype(np.int64), n_cat,
                 model=models.confirmatory(np.eye(4, dtype=np.int64)), node_rule="gh")
    with pytest.raises(ValueError):  # no pure anchor
        fit_gpcm(y, n_cat, model=models.confirmatory(np.ones((n_items, n_dims), dtype=np.int64)))
    with pytest.raises(ValueError):  # category out of range
        ybad = y.copy(); ybad[0, 0] = n_cat
        fit_gpcm(ybad, n_cat, model=models.confirmatory(pattern))
    with pytest.raises(ValueError):  # unobserved category
        ygap = y.copy(); ygap[ygap[:, 0] == 1, 0] = 0
        fit_gpcm(ygap, n_cat, model=models.confirmatory(pattern))


def test_fit_mixture_recovers_two_class_rasch():
    """Mixed Rasch / mixture IRT (Rost, 1990): recover two latent classes with a
    difficulty reversal (a single-class model cannot fit both orderings)."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_mixture, MixtureFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_mixture"):
        pytest.skip("compiled core built without fit_mixture")

    rng = np.random.default_rng(1990)
    n, j, pi_true = 1500, 15, 0.6
    b0 = np.linspace(-2.0, 2.0, j)
    # class 0: b0; class 1: -b0 (reversal). theta ~ N(0,1).
    cls = (rng.random(n) >= pi_true).astype(int)  # 0 w.p. pi_true
    theta = rng.standard_normal(n)
    y = np.empty((n, j))
    for p in range(n):
        b = b0 if cls[p] == 0 else -b0
        y[p] = (rng.random(j) < 1 / (1 + np.exp(-(theta[p] + b)))).astype(float)

    res = fit_mixture(y, n_classes=2, model="rasch", n_starts=8, seed=123)
    assert isinstance(res, MixtureFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert res.a.shape == (2, j) and np.allclose(res.a, 1.0)  # Rasch: a == 1
    assert res.n_parameters == 2 * j + 1  # 2 classes * J difficulties + (C-1)

    # permutation-match the two fitted classes to (b0, -b0) by difficulty SSE
    b_true = np.stack([b0, -b0])
    sse = lambda perm: float(np.sum((res.b[list(perm)] - b_true) ** 2))
    perm = (0, 1) if sse((0, 1)) <= sse((1, 0)) else (1, 0)
    brmse = np.sqrt(np.mean((res.b[list(perm)] - b_true) ** 2))
    assert brmse < 0.25, f"matched b RMSE {brmse}"
    assert abs(res.pi[perm[0]] - pi_true) < 0.06, f"pi {res.pi[perm[0]]}"

    # Adjusted Rand Index (label-invariant) between MAP class and truth
    def ari(a, b):
        from itertools import product
        ka, kb = a.max() + 1, b.max() + 1
        tab = np.zeros((ka, kb))
        for x, yv in zip(a, b):
            tab[x, yv] += 1
        c2 = lambda m: m * (m - 1) / 2
        idx = sum(c2(tab[i, k]) for i, k in product(range(ka), range(kb)))
        sa = sum(c2(tab[i].sum()) for i in range(ka))
        sb = sum(c2(tab[:, k].sum()) for k in range(kb))
        exp = sa * sb / c2(len(a))
        return (idx - exp) / (0.5 * (sa + sb) - exp)

    assert ari(res.map_class, cls) > 0.35, "class recovery (ARI) too low"

    with pytest.raises(ValueError):
        fit_mixture(y.ravel(), n_classes=2)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_mixture(y, n_classes=2, model="graded")  # unknown within-class model


def test_fit_lltm_recovers_basic_parameters():
    """LLTM (Fischer, 1973): recover the basic cognitive-operation parameters from the
    design matrix, and the LR test does not reject when the restriction holds."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_lltm, LltmFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_lltm"):
        pytest.skip("compiled core built without fit_lltm")

    rng = np.random.default_rng(1973)
    n, j, k = 2500, 20, 5
    # full-rank design with varying row sums (column k has period k+2)
    q = np.array([[(i + kk) % (kk + 2) for kk in range(k)] for i in range(j)], dtype=float)
    eta_true = np.array([0.6, -0.4, 0.9, -0.5, 0.3])
    c_true = -0.2
    b_true = c_true + q @ eta_true
    theta = rng.standard_normal(n)
    y = (rng.random((n, j)) < 1 / (1 + np.exp(-(theta[:, None] + b_true[None, :])))).astype(float)

    res = fit_lltm(y, q)
    assert isinstance(res, LltmFit) and res.converged
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert res.eta.shape == (k,) and res.b.shape == (j,)
    assert np.corrcoef(res.eta, eta_true)[0, 1] > 0.95
    assert np.sqrt(np.mean((res.b - b_true) ** 2)) < 0.15
    assert res.n_parameters == k + 1  # K basic + intercept
    # LR test: LLTM restriction HOLDS, so it should not reject
    assert res.lr_df == j - k - 1
    assert res.lr_p > 0.01, f"LR falsely rejected true LLTM: p={res.lr_p}"

    with pytest.raises(ValueError):
        fit_lltm(y.ravel(), q)  # responses not 2-D
    with pytest.raises(ValueError):
        # rows sum to a constant + intercept => rank-deficient design, rejected
        fit_lltm(y, np.ones((j, 1)))


def test_fit_testlet_recovers_local_dependence():
    """Testlet model (Bradlow, Wainer, & Wang, 1999): recover the per-testlet variance
    (local dependence), and confirm sigma^2=0 reduces to the ordinary Rasch/2PL fit."""
    import numpy as np
    import pytest
    from fast_mlsirm import fit_testlet, TestletFit
    from fast_mlsirm.fitstats import _core_module

    core = _core_module()
    if core is None or not hasattr(core, "fit_testlet"):
        pytest.skip("compiled core built without fit_testlet")

    rng = np.random.default_rng(1999)
    n, per, d = 800, 8, 2
    j = per * d
    tid = np.repeat(np.arange(d), per)  # contiguous testlets
    sig2 = np.array([0.6, 0.3])
    beta = np.tile(np.linspace(-1.5, 1.5, per), d)  # Rasch: b = -beta (a=1)
    theta = rng.standard_normal(n)
    gamma = rng.standard_normal((n, d)) * np.sqrt(sig2)[None, :]
    y = np.empty((n, j))
    for p in range(n):
        eta = theta[p] + beta - gamma[p, tid]
        y[p] = (rng.random(j) < 1 / (1 + np.exp(-eta))).astype(float)

    res = fit_testlet(y, tid, model="rasch")
    assert isinstance(res, TestletFit) and res.converged
    assert res.termination_reason == "converged"
    assert res.final_loglik_change < 1e-6
    assert np.all(np.diff(res.loglik_trace) >= -1e-6)
    assert np.all(res.a == 1.0)  # Rasch
    assert res.sigma2.shape == (d,)
    # the strong-LD testlet is recovered as clearly larger than the weak one
    assert res.sigma2[0] > 0.35 and res.sigma2[0] > res.sigma2[1]
    assert np.sqrt(np.mean((res.sigma2 - sig2) ** 2)) < 0.2

    # sigma^2 pinned to 0 => ordinary Rasch (no local dependence modeled)
    res0 = fit_testlet(y, tid, model="rasch", estimate_sigma=False, init_sigma2=0.0)
    assert np.all(res0.sigma2 == 0.0)
    assert res0.n_parameters == j  # fixed variances are not free parameters

    with pytest.raises(ValueError):
        fit_testlet(y.ravel(), tid)  # responses not 2-D
    with pytest.raises(ValueError):
        fit_testlet(y, tid, model="graded")  # unknown model

    with pytest.raises(RuntimeError, match="max_iter_reached"):
        fit_testlet(y[:40], tid, model="rasch", max_iter=1, require_convergence=True)
