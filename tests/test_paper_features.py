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
    np.testing.assert_allclose(core.rmsea2_ci_lower, ref.rmsea2_ci_lower, atol=1e-6)
    np.testing.assert_allclose(core.rmsea2_ci_upper, ref.rmsea2_ci_upper, atol=1e-6)

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
    import pytest as _pytest

    with _pytest.raises(Exception):
        irt_link(a_old, b_old, a_new, b_new, method="not_a_method")


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
    assert np.corrcoef(a_true, res["a"])[0, 1] > 0.9
    assert np.max(np.abs(a_true - res["a"])) < 0.35
    assert np.mean(np.abs(c_true[:, 1:] - res["intercepts"][:, 1:])) < 0.2


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
    assert np.corrcoef(a_true, fit.slope)[0, 1] > 0.9

    # validation
    with pytest.raises(ValueError):
        fit_polytomous(y, k, model="nominal")       # unsupported model
    with pytest.raises(ValueError):
        fit_polytomous(y.astype(float) + 0.5, k)    # non-integer categories
    with pytest.raises(ValueError):
        fit_polytomous(y, 2)                          # category out of range


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
