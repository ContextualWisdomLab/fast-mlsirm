"""Rust OLS+HC3 parity against the late-life fit_ols_hc3 / contrast formulas.

Reference algebra mirrors
``late-life-anxiety-reanalysis/analysis/library_regression_h1_h5.py``
(``fit_ols_hc3``, ``contrast``) without importing SciPy or calling Rscript.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm import chi2_sf_df1, contrast, fit_ols_hc
from fast_mlsirm.regression import f_sf, t_sf


def test_native_centered_products_match_manual_and_reject_invalid_controls():
    """NumPy 2.5 mean/std Notes parity through the actual Rust API.

    Tests declared term order, repeated-column powers, intercept and both
    population/sample denominators. Noninteger controls, degenerate inputs
    and nonfinite products are errors. Synthetic values are not study output.
    """
    from fast_mlsirm.regression import centered_product_design

    values = np.array([[1., 8.], [2., 3.], [4., 5.], [7., 2.], [9., 6.]])
    terms = [[], [1], [0, 1], [0], [1, 1]]
    centered = values - values.mean(axis=0)
    expected = np.column_stack([np.ones(5), centered[:, 1],
                                centered[:, 0] * centered[:, 1], centered[:, 0],
                                centered[:, 1] ** 2])
    for ddof in (0, 1):
        got = centered_product_design(values, terms, ddof=ddof)
        np.testing.assert_allclose(got["centers"], values.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(got["sds"], values.std(axis=0, ddof=ddof), atol=1e-14)
        np.testing.assert_allclose(got["design"], expected, atol=1e-14)
    for bad in ([[2]], [[.5]], [[True]], [[-1]], []):
        with pytest.raises(ValueError):
            centered_product_design(values, bad, ddof=1)
    for ddof in (True, .5, -1, 5):
        with pytest.raises(ValueError):
            centered_product_design(values, terms, ddof=ddof)
    with pytest.raises(ValueError, match="SD"):
        centered_product_design(np.ones((5, 2)), terms, ddof=1)
    with pytest.raises(ValueError, match="SD"):
        centered_product_design(np.full((1000, 2), .1), terms, ddof=1)
    with pytest.raises(ValueError, match="finite"):
        centered_product_design(values * np.nan, terms, ddof=1)
    with pytest.raises(ValueError, match="product"):
        centered_product_design(values * 1e50, [[0] * 10], ddof=1)


def _reference_fit_ols_hc3(x: np.ndarray, y: np.ndarray) -> dict:
    """NumPy reference matching late-life HC3 meat (sandwich::vcovHC HC3)."""
    n, k = x.shape
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    meat = x.T @ (((resid / (1.0 - hat)) ** 2)[:, None] * x)
    vcov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.diag(vcov))
    return {
        "beta": beta,
        "vcov": vcov,
        "se": se,
        "hat": hat,
        "resid": resid,
        "n": n,
        "k": k,
    }


def _reference_hc_vcov(x: np.ndarray, resid: np.ndarray, hat: np.ndarray, xtx_inv: np.ndarray, hc: str):
    n, k = x.shape
    if hc == "HC0":
        w = resid**2
    elif hc == "HC1":
        w = (n / (n - k)) * (resid**2)
    elif hc == "HC2":
        w = (resid**2) / (1.0 - hat)
    elif hc == "HC3":
        w = (resid / (1.0 - hat)) ** 2
    else:
        raise ValueError(hc)
    meat = x.T @ (w[:, None] * x)
    return xtx_inv @ meat @ xtx_inv


def _synthetic_design(n: int = 80, seed: int = 17) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    # Late-life style interaction design (centered-ish covariates).
    d = np.column_stack(
        [
            np.ones(n),
            x1,
            x2,
            x3,
            x1 * x2,
            x1 * x3,
            x2 * x3,
            x1 * x2 * x3,
        ]
    )
    # Heteroskedastic errors.
    eps = rng.normal(scale=0.5 + 0.4 * np.abs(x1), size=n)
    y = 1.2 + 0.7 * x1 - 0.4 * x2 + 0.3 * x3 + 0.2 * x1 * x2 + eps
    return d.astype(np.float64), y.astype(np.float64)


def test_fit_ols_hc3_parity_atol_1e6():
    x, y = _synthetic_design()
    ref = _reference_fit_ols_hc3(x, y)
    got = fit_ols_hc(x, y, hc="HC3")
    np.testing.assert_allclose(got["beta"], ref["beta"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(got["vcov"], ref["vcov"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(got["se"], ref["se"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(got["hat_diagonal"], ref["hat"], atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(got["residuals"], ref["resid"], atol=1e-6, rtol=0.0)


@pytest.mark.parametrize("hc", ["HC0", "HC1", "HC2", "HC3"])
def test_all_hc_types_match_numpy_reference(hc: str):
    x, y = _synthetic_design(n=60, seed=23)
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ x.T @ y
    resid = y - x @ beta
    hat = np.einsum("ij,jk,ik->i", x, xtx_inv, x)
    expected = _reference_hc_vcov(x, resid, hat, xtx_inv, hc)
    got = fit_ols_hc(x, y, hc=hc)
    np.testing.assert_allclose(got["beta"], beta, atol=1e-6, rtol=0.0)
    np.testing.assert_allclose(got["vcov"], expected, atol=1e-6, rtol=0.0)


def test_contrast_parity_with_late_life_formula():
    x, y = _synthetic_design()
    fit = fit_ols_hc(x, y, hc="HC3")
    vec = np.zeros(fit["k"], dtype=np.float64)
    vec[1] = 1.0
    # Late-life contrast: estimate, SE only (plus Wald from estimate^2/var).
    est = float(vec @ fit["beta"])
    var = float(vec @ fit["vcov"] @ vec)
    se = math.sqrt(var)
    got = contrast(fit["beta"], fit["vcov"], vec, df=fit["df"])
    assert abs(got["estimate"] - est) <= 1e-6
    assert abs(got["SE"] - se) <= 1e-6
    assert abs(got["wald_chi2"] - (est * est / var)) <= 1e-6
    assert abs(got["p_chi2"] - chi2_sf_df1(got["wald_chi2"])) <= 1e-6
    # Late-life chi2_sf_df1 = erfc(sqrt(q/2))
    q = got["wald_chi2"]
    assert abs(chi2_sf_df1(q) - math.erfc(math.sqrt(q / 2.0))) <= 1e-6


def test_distribution_tails_no_scipy():
    assert abs(chi2_sf_df1(3.841458820694124) - 0.05) < 1e-5
    assert abs(f_sf(3.841458820694124, 1.0, 1.0e8) - 0.05) < 5e-4
    assert abs(t_sf(1.6448536269514722, 1.0e8) - 0.05) < 5e-4


def test_regression_core_exports_without_scipy_rscript():
    import fast_mlsirm.regression as reg

    src = open(reg.__file__, encoding="utf-8").read()
    assert "scipy" not in src.lower()
    assert "rscript" not in src.lower()


def test_native_normal_wald_interval_matches_normal_reference_and_rejects_invalid_inputs():
    """Check actual native normal intervals, tail stability and invalid inputs.

    Source: statsmodels Developers (n.d.), ContrastResults.conf_int, actual
    source lines94-117. QuantLib Developers (n.d.), InverseCumulativeNormal
    source documents the rational approximation's relative error; tests use
    measured numerical bounds, not scientific acceptance thresholds.
    """
    from fast_mlsirm.regression import normal_wald_interval
    from scipy.stats import norm
    import statsmodels.api as sm
    import pytest

    rng = np.random.default_rng(20260928)
    x = np.column_stack([np.ones(64), rng.normal(size=(64, 2))])
    y = x @ np.array([0.2, -0.4, 0.7]) + rng.normal(size=64)
    fit = sm.OLS(y, x).fit(cov_type="HC3", use_t=False)
    vec = np.array([0.0, 1.0, -0.5])
    result = fit.t_test(vec)
    for alpha in (0.05, 0.10, 0.01):
        got = normal_wald_interval(float(result.effect[0]), float(result.sd[0]), alpha=alpha)
        np.testing.assert_allclose([got["lower"], got["upper"]],
                                   result.conf_int(alpha=alpha)[0], rtol=0, atol=1e-8)
        assert got["alpha"] == alpha
    for alpha in (0.95, 0.05, 1e-12, 1e-100):
        got = normal_wald_interval(0.0, 1.0, alpha=alpha)
        reference = norm.isf(alpha / 2)
        np.testing.assert_allclose(got["critical"], reference, rtol=1.2e-9, atol=1e-12)
        assert got["lower"] == -got["upper"]
        assert got["upper"] > 0
    for estimate, se, alpha in [(np.nan, 1, .05), (np.inf, 1, .05), (0, 0, .05),
                                (0, -1, .05), (0, np.inf, .05), (0, 1, 0),
                                (0, 1, 1), (0, 1, np.nan), (0, 1, 5e-324),
                                (1e308, 1e308, .05), (1e308, 1e-300, .05),
                                (True, 1, .05), (0, True, .05), (0, 1, True),
                                (1j, 1, .05), ([0], 1, .05)]:
        with pytest.raises(ValueError):
            normal_wald_interval(estimate, se, alpha=alpha)
