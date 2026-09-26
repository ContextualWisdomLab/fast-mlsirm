//! Rust-owned OLS + HC sandwich regression exposed through a modular PyO3 entrypoint.
//!
//! Numerical work lives in `mlsirm-core::regression`. This binding validates
//! NumPy layout, delegates to the core, and marshals results into Python dicts.

use mlsirm_core::regression::{
    chi2_sf_df1, conditional_slope, design_row_dot, f_sf, fit_ols_hc, linear_contrast,
    nested_ols_column_drop, normal_wald_interval, sample_mean_sd, slope_difference, t_sf,
    xwz_e_design_row, HcType, OlsFit, XWZ_E_K,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

fn parse_hc(hc: &str) -> PyResult<HcType> {
    HcType::parse(hc).map_err(PyValueError::new_err)
}

fn fit_dict(py: Python<'_>, fit: &OlsFit, vcov: &[f64], hc: &str) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("n", fit.n)?;
    out.set_item("k", fit.k)?;
    out.set_item("hc", hc)?;
    out.set_item("df", (fit.n - fit.k) as f64)?;
    out.set_item("sigma2", fit.sigma2)?;
    out.set_item("beta", PyArray1::from_slice(py, &fit.beta))?;
    out.set_item("residuals", PyArray1::from_slice(py, &fit.residuals))?;
    out.set_item("hat_diagonal", PyArray1::from_slice(py, &fit.hat_diagonal))?;
    out.set_item("vcov", PyArray1::from_slice(py, vcov))?;
    let se: Vec<f64> = (0..fit.k).map(|i| vcov[i * fit.k + i].sqrt()).collect();
    out.set_item("se", PyArray1::from_slice(py, &se))?;
    Ok(out.into())
}

fn contrast_dict(
    py: Python<'_>,
    result: mlsirm_core::regression::ContrastResult,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("estimate", result.estimate)?;
    out.set_item("SE", result.se)?;
    out.set_item("se", result.se)?;
    out.set_item("wald_chi2", result.wald_chi2)?;
    out.set_item("p_chi2", result.p_chi2)?;
    out.set_item("t_stat", result.t_stat)?;
    out.set_item("p_t", result.p_t)?;
    out.set_item("f_stat", result.f_stat)?;
    out.set_item("p_f", result.p_f)?;
    out.set_item("df", result.df)?;
    Ok(out.into())
}

#[pyfunction(name = "fit_ols_hc", signature = (x, y, hc="HC3"))]
fn py_fit_ols_hc(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    hc: &str,
) -> PyResult<Py<PyDict>> {
    if x.ndim() != 2 {
        return Err(PyValueError::new_err("x must be a 2-D array"));
    }
    if y.ndim() != 1 {
        return Err(PyValueError::new_err("y must be a 1-D array"));
    }
    let shape = x.shape();
    let n = shape[0];
    let k = shape[1];
    if y.len() != n {
        return Err(PyValueError::new_err("y length must equal x.shape[0]"));
    }
    let hc_ty = parse_hc(hc)?;
    // NumPy C-contiguous row-major matches our layout when we copy via to_vec.
    let x_vec: Vec<f64> = x.as_array().iter().copied().collect();
    let y_vec: Vec<f64> = y.as_slice()?.to_vec();
    let (fit, vcov) = fit_ols_hc(&x_vec, &y_vec, n, k, hc_ty).map_err(PyValueError::new_err)?;
    fit_dict(py, &fit, &vcov, &hc.to_ascii_uppercase())
}

#[pyfunction(name = "nested_ols_column_drop")]
fn py_nested_ols_column_drop(
    py: Python<'_>,
    x: PyReadonlyArray2<'_, f64>,
    y: PyReadonlyArray1<'_, f64>,
    drop_columns: Vec<usize>,
) -> PyResult<Py<PyDict>> {
    let shape = x.shape();
    if y.len() != shape[0] {
        return Err(PyValueError::new_err("y length must equal x.shape[0]"));
    }
    let x_vec: Vec<f64> = x.as_array().iter().copied().collect();
    let y_vec: Vec<f64> = y.as_slice()?.to_vec();
    let result = nested_ols_column_drop(&x_vec, &y_vec, shape[0], shape[1], &drop_columns)
        .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("n", result.n)?;
    out.set_item("k_full", result.k_full)?;
    out.set_item("k_reduced", result.k_reduced)?;
    out.set_item("sst", result.sst)?;
    out.set_item("sse_full", result.sse_full)?;
    out.set_item("sse_reduced", result.sse_reduced)?;
    out.set_item("r2_full", result.r2_full)?;
    out.set_item("adjusted_r2_full", result.adjusted_r2_full)?;
    out.set_item("r2_reduced", result.r2_reduced)?;
    out.set_item("adjusted_r2_reduced", result.adjusted_r2_reduced)?;
    out.set_item("delta_r2", result.delta_r2)?;
    out.set_item("f_stat", result.f_stat)?;
    out.set_item("p_f", result.p_f)?;
    out.set_item("df1", result.df1)?;
    out.set_item("df2", result.df2)?;
    Ok(out.into())
}

#[pyfunction(name = "linear_contrast")]
fn py_linear_contrast(
    py: Python<'_>,
    beta: PyReadonlyArray1<'_, f64>,
    vcov: PyReadonlyArray1<'_, f64>,
    contrast: PyReadonlyArray1<'_, f64>,
    df: f64,
) -> PyResult<Py<PyDict>> {
    let beta_s = beta.as_slice()?;
    let vcov_s = vcov.as_slice()?;
    let contrast_s = contrast.as_slice()?;
    let result = linear_contrast(beta_s, vcov_s, contrast_s, df).map_err(PyValueError::new_err)?;
    contrast_dict(py, result)
}

#[pyfunction(name = "xwz_e_design_row")]
fn py_xwz_e_design_row(py: Python<'_>, x: f64, w: f64, z: f64, e: f64) -> Py<PyArray1<f64>> {
    let row = xwz_e_design_row(x, w, z, e);
    PyArray1::from_slice(py, &row).into()
}

#[pyfunction(name = "design_row_dot")]
fn py_design_row_dot(row: PyReadonlyArray1<'_, f64>, beta: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    design_row_dot(row.as_slice()?, beta.as_slice()?).map_err(PyValueError::new_err)
}

#[pyfunction(
    name = "conditional_slope",
    signature = (beta, vcov, focal, x, w, z, e, df)
)]
fn py_conditional_slope(
    py: Python<'_>,
    beta: PyReadonlyArray1<'_, f64>,
    vcov: PyReadonlyArray1<'_, f64>,
    focal: &str,
    x: f64,
    w: f64,
    z: f64,
    e: f64,
    df: f64,
) -> PyResult<Py<PyDict>> {
    let result = conditional_slope(
        beta.as_slice()?,
        vcov.as_slice()?,
        focal,
        x,
        w,
        z,
        e,
        df,
    )
    .map_err(PyValueError::new_err)?;
    contrast_dict(py, result)
}

#[pyfunction(name = "slope_difference")]
fn py_slope_difference(
    py: Python<'_>,
    beta: PyReadonlyArray1<'_, f64>,
    vcov: PyReadonlyArray1<'_, f64>,
    focal: &str,
    probes_a: (f64, f64, f64, f64),
    probes_b: (f64, f64, f64, f64),
    df: f64,
) -> PyResult<Py<PyDict>> {
    let result = slope_difference(
        beta.as_slice()?,
        vcov.as_slice()?,
        focal,
        probes_a,
        probes_b,
        df,
    )
    .map_err(PyValueError::new_err)?;
    contrast_dict(py, result)
}

#[pyfunction(name = "chi2_sf_df1")]
fn py_chi2_sf_df1(q: f64) -> f64 {
    chi2_sf_df1(q)
}

#[pyfunction(name = "f_sf")]
fn py_f_sf(f: f64, df1: f64, df2: f64) -> f64 {
    f_sf(f, df1, df2)
}

#[pyfunction(name = "t_sf")]
fn py_t_sf(t: f64, df: f64) -> f64 {
    t_sf(t, df)
}

#[pyfunction(name = "sample_mean_sd")]
fn py_sample_mean_sd(values: PyReadonlyArray1<'_, f64>) -> PyResult<(f64, f64)> {
    sample_mean_sd(values.as_slice()?).map_err(PyValueError::new_err)
}

#[pyfunction(name = "normal_wald_interval")]
fn py_normal_wald_interval(
    estimate: f64,
    se: f64,
    confidence_level: f64,
) -> PyResult<(f64, f64)> {
    normal_wald_interval(estimate, se, confidence_level).map_err(PyValueError::new_err)
}

#[pymodule]
#[pyo3(name = "_regression_core")]
fn fast_mlsirm_regression_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("XWZ_E_K", XWZ_E_K)?;
    m.add_function(wrap_pyfunction!(py_fit_ols_hc, m)?)?;
    m.add_function(wrap_pyfunction!(py_nested_ols_column_drop, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_contrast, m)?)?;
    m.add_function(wrap_pyfunction!(py_xwz_e_design_row, m)?)?;
    m.add_function(wrap_pyfunction!(py_design_row_dot, m)?)?;
    m.add_function(wrap_pyfunction!(py_conditional_slope, m)?)?;
    m.add_function(wrap_pyfunction!(py_slope_difference, m)?)?;
    m.add_function(wrap_pyfunction!(py_chi2_sf_df1, m)?)?;
    m.add_function(wrap_pyfunction!(py_f_sf, m)?)?;
    m.add_function(wrap_pyfunction!(py_t_sf, m)?)?;
    m.add_function(wrap_pyfunction!(py_sample_mean_sd, m)?)?;
    m.add_function(wrap_pyfunction!(py_normal_wald_interval, m)?)?;
    Ok(())
}
