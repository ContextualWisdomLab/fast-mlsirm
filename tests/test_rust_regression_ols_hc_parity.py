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


def test_nested_ols_f_one_two_matches_independent_synthetic():
    rng = np.random.default_rng(20260925)
    raw = rng.normal(size=(12, 4))
    x, w, z, e = (raw[:, j] - raw[:, j].mean() for j in range(4))
    full_x = np.column_stack(
        (np.ones(12), x, w, z, e, x * w, x * z, w * z, x * e, x * w * z)
    )
    reduced_x = full_x[:, :-1]
    y = 1.0 + 0.3 * x - 0.2 * w + 0.4 * z + 0.25 * full_x[:, -1]
    y += rng.normal(0.0, 0.5, size=12)
    assert np.linalg.matrix_rank(full_x) == 10
    assert np.linalg.matrix_rank(reduced_x) == 9

    full = fit_ols_hc(np.ascontiguousarray(full_x), y, hc="HC3")
    reduced = fit_ols_hc(np.ascontiguousarray(reduced_x), y, hc="HC3")
    sse_full = float(np.dot(full["residuals"], full["residuals"]))
    sse_reduced = float(np.dot(reduced["residuals"], reduced["residuals"]))
    ref_full = np.linalg.lstsq(full_x, y, rcond=None)[0]
    ref_reduced = np.linalg.lstsq(reduced_x, y, rcond=None)[0]
    ref_sse_full = float(np.sum((y - full_x @ ref_full) ** 2))
    ref_sse_reduced = float(np.sum((y - reduced_x @ ref_reduced) ** 2))
    np.testing.assert_allclose(
        [sse_full, sse_reduced], [ref_sse_full, ref_sse_reduced], rtol=1e-8
    )
    assert sse_reduced >= sse_full > 0.0

    f_stat = (sse_reduced - sse_full) / (sse_full / (len(y) - full_x.shape[1]))
    ref_f = (ref_sse_reduced - ref_sse_full) / (ref_sse_full / 2.0)
    assert math.isclose(f_stat, ref_f, rel_tol=1e-8)
    sst = float(np.sum((y - y.mean()) ** 2))
    delta_r2 = (sse_reduced - sse_full) / sst
    assert math.isclose(f_stat, 2.0 * delta_r2 / (sse_full / sst), rel_tol=1e-12)
    # The F(1, 2) survival function has an independent closed form.
    assert math.isclose(
        f_sf(f_stat, 1.0, 2.0), 1.0 - math.sqrt(f_stat / (f_stat + 2.0)), rel_tol=1e-11
    )


def test_regression_core_exports_without_scipy_rscript():
    import fast_mlsirm.regression as reg

    src = open(reg.__file__, encoding="utf-8").read()
    assert "scipy" not in src.lower()
    assert "rscript" not in src.lower()
