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
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
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

#[pymodule]
#[pyo3(name = "_bifactor_core")]
pub fn fast_mlsirm_bifactor_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_bifactor_indices, m)?)?;
    m.add_function(wrap_pyfunction!(py_bifactor_indices_from_logit_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(py_bifactor_lord_wingersky, m)?)?;
    m.add_function(wrap_pyfunction!(py_direct_enumeration_bifactor, m)?)?;
    Ok(())
}
