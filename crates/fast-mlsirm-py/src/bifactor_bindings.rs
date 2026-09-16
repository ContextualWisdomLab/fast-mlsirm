//! Python bindings for Rust-native bifactor scoreability diagnostics,
//! 2-stage Lord-Wingersky recursion, and QMCEM Bifactor GRM estimation.

use mlsirm_core::bifactor_indices::{
    bifactor_indices as core_bifactor_indices,
    bifactor_latent_response_indices_from_logit_slopes as core_bifactor_from_logit_slopes,
    BifactorIndicesConfig as CoreBifactorIndicesConfig,
    BifactorIndicesResult as CoreBifactorIndicesResult,
};
use mlsirm_core::bifactor_recursion::{
    bifactor_lord_wingersky as core_bifactor_lord_wingersky,
    direct_enumeration_bifactor as core_direct_enumeration_bifactor,
    BifactorItemParams,
};
use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm as core_fit_bifactor_grm,
    fit_bifactor_slope_sensitivity as core_fit_bifactor_slope_sensitivity,
    BifactorGrmConfig,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;

fn result_dict(
    py: Python<'_>,
    result: CoreBifactorIndicesResult,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("factor_item_counts", result.factor_item_counts)?;
    out.set_item("is_strict_bifactor", result.is_strict_bifactor)?;
    out.set_item("puc", result.puc)?;
    out.set_item("ecv_ss", result.ecv_ss)?;
    out.set_item("ecv_sg", result.ecv_sg)?;
    out.set_item("ecv_gs", result.ecv_gs)?;
    out.set_item("item_ecv", result.item_ecv)?;
    out.set_item("omega_total", result.omega_total)?;
    out.set_item("omega_hierarchical", result.omega_hierarchical)?;
    out.set_item("construct_replicability", result.construct_replicability)?;
    Ok(out.into())
}

fn config_from_shape(
    shape: &[usize],
    general_factor: usize,
    zero_tolerance: f64,
) -> CoreBifactorIndicesConfig {
    CoreBifactorIndicesConfig {
        n_items: shape[0],
        n_factors: shape[1],
        general_factor,
        zero_tolerance,
    }
}

#[pyfunction(name = "bifactor_indices")]
fn py_bifactor_indices(
    py: Python<'_>,
    loadings: PyReadonlyArray2<'_, f64>,
    uniquenesses: PyReadonlyArray1<'_, f64>,
    general_factor: usize,
    zero_tolerance: f64,
) -> PyResult<Py<PyDict>> {
    let shape = loadings.shape();
    let result = core_bifactor_indices(
        loadings.as_slice()?,
        uniquenesses.as_slice()?,
        config_from_shape(shape, general_factor, zero_tolerance),
    )
    .map_err(PyValueError::new_err)?;
    result_dict(py, result)
}

#[pyfunction(name = "bifactor_indices_from_logit_slopes")]
fn py_bifactor_indices_from_logit_slopes(
    py: Python<'_>,
    logit_slopes: PyReadonlyArray2<'_, f64>,
    general_factor: usize,
    zero_tolerance: f64,
) -> PyResult<Py<PyDict>> {
    let shape = logit_slopes.shape();
    let result = core_bifactor_from_logit_slopes(
        logit_slopes.as_slice()?,
        config_from_shape(shape, general_factor, zero_tolerance),
    )
    .map_err(PyValueError::new_err)?;
    result_dict(py, result)
}

#[pyfunction(name = "bifactor_lord_wingersky")]
#[allow(clippy::too_many_arguments)]
fn py_bifactor_lord_wingersky<'py>(
    py: Python<'py>,
    a_general: PyReadonlyArray1<'_, f64>,
    a_specific: PyReadonlyArray1<'_, f64>,
    thresholds: PyReadonlyArray1<'_, f64>,
    item_domains: PyReadonlyArray1<'_, i64>,
    n_cat: usize,
    n_domains: usize,
    theta_general: PyReadonlyArray1<'_, f64>,
    theta_specific: PyReadonlyArray1<'_, f64>,
    weights_specific: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let domains_usize: Vec<usize> = item_domains
        .as_slice()?
        .iter()
        .map(|&d| d as usize)
        .collect();

    let params = BifactorItemParams {
        a_general: a_general.as_slice()?.to_vec(),
        a_specific: a_specific.as_slice()?.to_vec(),
        thresholds: thresholds.as_slice()?.to_vec(),
        item_domains: domains_usize,
        n_cat,
        n_domains,
    };

    let res = core_bifactor_lord_wingersky(
        &params,
        theta_general.as_slice()?,
        theta_specific.as_slice()?,
        weights_specific.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;

    Ok(res.to_pyarray(py))
}

#[pyfunction(name = "direct_enumeration_bifactor")]
#[allow(clippy::too_many_arguments)]
fn py_direct_enumeration_bifactor<'py>(
    py: Python<'py>,
    a_general: PyReadonlyArray1<'_, f64>,
    a_specific: PyReadonlyArray1<'_, f64>,
    thresholds: PyReadonlyArray1<'_, f64>,
    item_domains: PyReadonlyArray1<'_, i64>,
    n_cat: usize,
    n_domains: usize,
    theta_general: PyReadonlyArray1<'_, f64>,
    theta_specific: PyReadonlyArray1<'_, f64>,
    weights_specific: PyReadonlyArray1<'_, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let domains_usize: Vec<usize> = item_domains
        .as_slice()?
        .iter()
        .map(|&d| d as usize)
        .collect();

    let params = BifactorItemParams {
        a_general: a_general.as_slice()?.to_vec(),
        a_specific: a_specific.as_slice()?.to_vec(),
        thresholds: thresholds.as_slice()?.to_vec(),
        item_domains: domains_usize,
        n_cat,
        n_domains,
    };

    let res = core_direct_enumeration_bifactor(
        &params,
        theta_general.as_slice()?,
        theta_specific.as_slice()?,
        weights_specific.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;

    Ok(res.to_pyarray(py))
}

#[pyfunction(name = "fit_bifactor_grm")]
#[allow(clippy::too_many_arguments)]
fn py_fit_bifactor_grm<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'_, i64>,
    observed: Option<PyReadonlyArray2<'_, bool>>,
    group_ids: Option<PyReadonlyArray1<'_, i64>>,
    n_groups: usize,
    loading_pattern: PyReadonlyArray2<'_, u8>,
    n_cat: usize,
    max_iter: usize,
    tol: f64,
    ridge: f64,
    newton_iter: usize,
    qmc_draws: usize,
    seed: u64,
    slope_bound: Option<f64>,
    compute_oakes_se: bool,
    device: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let y_shape = y.shape();
    let n_persons = y_shape[0];
    let n_items = y_shape[1];
    let n_dims = loading_pattern.shape()[1];

    let y_usize: Vec<usize> = y.as_slice()?.iter().map(|&val| val as usize).collect();
    let obs_slice = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let group_ids_vec: Option<Vec<usize>> = group_ids
        .map(|g| g.as_slice().map(|s| s.iter().map(|&v| v as usize).collect()))
        .transpose()?;

    let parsed_device = mlsirm_core::Device::parse(device)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let cfg = BifactorGrmConfig {
        max_iter,
        tol,
        ridge,
        newton_iter,
        qmc_draws,
        seed,
        slope_bound,
        compute_oakes_se,
        device: parsed_device,
    };

    let obs_vec = obs_slice.map(|s| s.to_vec());
    let lp_vec = loading_pattern.as_slice()?.to_vec();

    let res = py
        .detach(move || {
            core_fit_bifactor_grm(
                &y_usize,
                obs_vec.as_deref(),
                group_ids_vec.as_deref(),
                n_groups,
                &lp_vec,
                n_persons,
                n_items,
                n_dims,
                n_cat,
                &cfg,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("slope", res.slope.to_pyarray(py))?;
    out.set_item("threshold", res.threshold.to_pyarray(py))?;
    out.set_item("group_means", res.group_means.to_pyarray(py))?;
    out.set_item("group_variances", res.group_variances.to_pyarray(py))?;
    out.set_item("loglik", res.loglik)?;
    out.set_item("loglik_trace", res.loglik_trace.to_pyarray(py))?;
    out.set_item("n_iter", res.n_iter)?;
    out.set_item("converged", res.converged)?;

    if let Some(se_slope) = res.oakes_se_slope {
        out.set_item("oakes_se_slope", se_slope.to_pyarray(py))?;
    }
    if let Some(se_thr) = res.oakes_se_threshold {
        out.set_item("oakes_se_threshold", se_thr.to_pyarray(py))?;
    }
    if let Some(val) = res.min_eigenvalue {
        out.set_item("min_eigenvalue", val)?;
    }
    if let Some(val) = res.condition_number {
        out.set_item("condition_number", val)?;
    }

    Ok(out)
}

#[pyfunction(name = "fit_bifactor_slope_sensitivity")]
#[allow(clippy::too_many_arguments)]
fn py_fit_bifactor_slope_sensitivity(
    py: Python<'_>,
    y: PyReadonlyArray2<'_, i64>,
    observed: Option<PyReadonlyArray2<'_, bool>>,
    group_ids: Option<PyReadonlyArray1<'_, i64>>,
    n_groups: usize,
    loading_pattern: PyReadonlyArray2<'_, u8>,
    n_cat: usize,
    candidate_bounds: Vec<Option<f64>>,
    max_iter: usize,
    tol: f64,
    ridge: f64,
    newton_iter: usize,
    qmc_draws: usize,
    seed: u64,
    device: &str,
) -> PyResult<Py<PyList>> {
    let y_shape = y.shape();
    let n_persons = y_shape[0];
    let n_items = y_shape[1];
    let n_dims = loading_pattern.shape()[1];

    let y_usize: Vec<usize> = y.as_slice()?.iter().map(|&val| val as usize).collect();
    let obs_slice = observed.as_ref().map(|o| o.as_slice()).transpose()?;
    let group_ids_vec: Option<Vec<usize>> = group_ids
        .map(|g| g.as_slice().map(|s| s.iter().map(|&v| v as usize).collect()))
        .transpose()?;

    let parsed_device = mlsirm_core::Device::parse(device)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let cfg = BifactorGrmConfig {
        max_iter,
        tol,
        ridge,
        newton_iter,
        qmc_draws,
        seed,
        slope_bound: None,
        compute_oakes_se: false,
        device: parsed_device,
    };

    let obs_vec = obs_slice.map(|s| s.to_vec());
    let lp_vec = loading_pattern.as_slice()?.to_vec();

    let entries = py
        .detach(move || {
            core_fit_bifactor_slope_sensitivity(
                &y_usize,
                obs_vec.as_deref(),
                group_ids_vec.as_deref(),
                n_groups,
                &lp_vec,
                n_persons,
                n_items,
                n_dims,
                n_cat,
                &candidate_bounds,
                &cfg,
            )
        })
        .map_err(PyValueError::new_err)?;

    let out_list = PyList::empty(py);
    for e in entries {
        let d = PyDict::new(py);
        d.set_item("bound", e.bound)?;
        d.set_item("loglik", e.loglik)?;
        d.set_item("converged", e.converged)?;
        d.set_item("n_iter", e.n_iter)?;
        d.set_item("n_bounded_slopes", e.n_bounded_slopes)?;
        d.set_item("max_slope", e.max_slope)?;
        out_list.append(d)?;
    }

    Ok(out_list.into())
}

#[pymodule]
#[pyo3(name = "_bifactor_core")]
pub fn fast_mlsirm_bifactor_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_bifactor_indices, m)?)?;
    m.add_function(wrap_pyfunction!(py_bifactor_indices_from_logit_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(py_bifactor_lord_wingersky, m)?)?;
    m.add_function(wrap_pyfunction!(py_direct_enumeration_bifactor, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_bifactor_grm, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_bifactor_slope_sensitivity, m)?)?;
    Ok(())
}
