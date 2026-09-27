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


def test_actual_column_subset_comparison_and_rejection():
    """Check nested RSS/F against independent NumPy least squares.

    Source: statsmodels RegressionResults compare_f_test lines2813–2865;
    classical inference assumes homoscedastic uncorrelated errors.
    """
    from fast_mlsirm.regression import compare_ols_column_subset
    rng = np.random.default_rng(20260928)
    x = np.column_stack([np.ones(80), rng.normal(size=(80, 3))])
    y = x @ np.array([.2, -.5, .3, .7]) + rng.normal(size=80)
    keep = [0, 1, 2]
    got = compare_ols_column_subset(x, y, keep, intercept_column=0)
    full = y-x @ np.linalg.lstsq(x, y, rcond=None)[0]
    red = y-x[:, keep] @ np.linalg.lstsq(x[:, keep], y, rcond=None)[0]
    sse, reduced = full @ full, red @ red
    total = (y-y.mean()) @ (y-y.mean())
    f = (reduced-sse)/sse*76
    np.testing.assert_allclose([got[k] for k in ["full_r_squared", "adjusted_r_squared", "reduced_r_squared", "delta_r_squared", "classical_f", "df1", "df2"]],
        [1-sse/total, 1-79/76*sse/total, 1-reduced/total, (reduced-sse)/total, f, 1, 76], rtol=1e-12, atol=1e-12)
    assert 0 <= got["classical_p"] <= 1
    permuted = compare_ols_column_subset(x, y, [2, 0, 1], intercept_column=0)
    np.testing.assert_allclose(list(got.values()), list(permuted.values()), rtol=1e-12, atol=1e-12)
    for columns, intercept in [([], 0), ([0,1,2,3], 0), ([0,0], 0),
                                ([1,2], 0), ([0,4], 0), ([0,True], 0),
                                ([0,1], 1), ([0,1], True)]:
        with pytest.raises(ValueError):
            compare_ols_column_subset(x, y, columns, intercept_column=intercept)
    with pytest.raises(ValueError):
        compare_ols_column_subset(np.column_stack([np.zeros(80), x[:,1:]]), y, keep, intercept_column=0)
    with pytest.raises(ValueError):
        compare_ols_column_subset(np.column_stack([x, x[:,1]]), y, keep, intercept_column=0)


def test_native_residual_summary_intercept_and_degenerate_contract():
    """Validate source-defined summaries on actual OLS residuals and boundaries.

    statsmodels RegressionResults actual source lines2125–2237 supplies the
    centered/uncentered definitions. NumPy is an independent test oracle only.
    """
    from fast_mlsirm.regression import residual_summary, fit_ols_hc
    import pytest
    rng = np.random.default_rng(20260928)
    y = rng.normal(size=40).astype(np.float64) + 2
    for intercept in (True, False):
        x = rng.normal(size=(40, 2)).astype(np.float64)
        if intercept:
            x = np.column_stack([np.ones(40), x])
        fit = fit_ols_hc(x, y)
        got = residual_summary(y, fit["residuals"], rank=x.shape[1], has_intercept=intercept)
        oracle = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
        sse = oracle @ oracle
        centered = y - y.mean() if intercept else y
        total = centered @ centered
        r2 = 1 - sse / total
        np.testing.assert_allclose(list(got.values()), [sse, total, r2, 1-(40-int(intercept))/(40-x.shape[1])*(1-r2)], rtol=1e-12, atol=1e-12)
    assert residual_summary(y, y*10, rank=2, has_intercept=True)["r_squared"] < 0
    for a, b, rank, intercept in [(np.ones(4), np.ones(4), 1, True),
                                  (np.full(3, .1), np.ones(3), 1, True),
                                  (np.zeros(4), np.ones(4), 1, False),
                                  (y, y[:-1], 2, True), (y, y, 40, True),
                                  (y, y, True, True), (y, y, 2, 1),
                                  (y*np.inf, y, 2, True),
                                  (np.full(4, 1e308), np.ones(4), 1, False)]:
        with pytest.raises(ValueError):
            residual_summary(a, b, rank=rank, has_intercept=intercept)


def test_native_normal_wald_interval_matches_normal_reference_and_rejects_invalid_inputs():
    """Check native intervals against Python's independent normal quantile.

    Sources: statsmodels Developers (n.d.), ContrastResults.conf_int, actual
    source lines94-117; Python Software Foundation (n.d.), Python 3.12
    statistics.NormalDist.inv_cdf manual. QuantLib's actual source documents
    the existing rational approximation. Numerical bounds are regression
    tolerances, not scientific acceptance settings. No optional oracle package.
    """
    from fast_mlsirm.regression import normal_wald_interval
    from statistics import NormalDist
    import pytest

    for alpha in (0.05, 0.10, 0.01):
        got = normal_wald_interval(-0.4, 0.17, alpha=alpha)
        critical = -NormalDist().inv_cdf(alpha / 2)
        np.testing.assert_allclose([got["lower"], got["upper"]],
                                   [-0.4-critical*0.17, -0.4+critical*0.17],
                                   rtol=0, atol=1e-8)
        assert got["alpha"] == alpha
    for alpha in (0.95, 0.05, 1e-12, 1e-100):
        got = normal_wald_interval(0.0, 1.0, alpha=alpha)
        reference = -NormalDist().inv_cdf(alpha / 2)
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
