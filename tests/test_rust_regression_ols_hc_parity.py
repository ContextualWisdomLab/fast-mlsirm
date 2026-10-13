"""Rust OLS+HC3 parity against the late-life fit_ols_hc3 / contrast formulas.

Reference algebra mirrors
``late-life-anxiety-reanalysis/analysis/library_regression_h1_h5.py``
(``fit_ols_hc3``, ``contrast``) without importing SciPy or calling Rscript.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from fast_mlsirm import adjusted_r_squared, chi2_sf_df1, contrast, fit_ols_hc
from fast_mlsirm.regression import f_sf, t_sf


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


def test_normal_wald_tail_and_same_fit_contrast_vectors():
    # Independent fixed values: P(Chi2_1 >= z²) = erfc(|z| / sqrt(2)).
    for z in (0.0, 1.0, 1.959963984540054, 8.0, 12.0, 20.0):
        expected = math.erfc(abs(z) / math.sqrt(2.0))
        actual = chi2_sf_df1(z * z)
        assert math.isclose(actual, expected, rel_tol=1e-11, abs_tol=0.0)
        assert 0.0 < actual <= 1.0

    beta = np.array([0.2, -0.4, 0.3, 0.8, 0.1, -0.1, 0.25, -0.2, 0.05, 0.15])
    vcov = np.diag(np.arange(1.0, 11.0) / 100.0)
    vcov[6, 7] = vcov[7, 6] = 0.01
    df = 32.0
    h4 = np.array([0, 0, 0, 1, 0, 0, 2, -3, 0, -6], dtype=float)
    h5 = np.array([0, 0, 0, 1, 0, 0, -2, 3, 0, -6], dtype=float)
    difference = np.array([0, 0, 0, 0, 0, 0, 4, -6, 0, 0], dtype=float)
    for vec in [*np.eye(10), h4, h5, difference]:
        got = contrast(beta, vcov, vec, df=df)
        estimate = float(vec @ beta)
        se = math.sqrt(float(vec @ vcov @ vec))
        expected = math.erfc(abs(estimate / se) / math.sqrt(2.0))
        assert math.isclose(got["estimate"], estimate, rel_tol=1e-12, abs_tol=1e-12)
        assert math.isclose(got["SE"], se, rel_tol=1e-12)
        assert math.isclose(got["p_chi2"], expected, rel_tol=1e-11)
    h4_result = contrast(beta, vcov, h4, df=df)
    assert not math.isclose(h4_result["p_t"], h4_result["p_chi2"], rel_tol=1e-4)


def test_adjusted_r_squared_reduced_design_contract():
    # k counts the intercept; n=8, k=3 gives residual df=5.
    assert math.isclose(adjusted_r_squared(8, 3, 5.0, 20.0), 0.65)
    assert adjusted_r_squared(8, 3, 30.0, 20.0) < 0.0
    for args in ((3, 3, 1.0, 2.0), (8, 3, -1.0, 2.0), (8, 3, 1.0, 0.0)):
        with pytest.raises(ValueError):
            adjusted_r_squared(*args)
