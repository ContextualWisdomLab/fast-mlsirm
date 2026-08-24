//! Python bindings for the sparse cross-classified multiple-membership
//! contextual-effects predictor and crossed `u_h` MAP estimator
//! (see `mlsirm_core::multilevel`).
//!
//! This module performs shape/type marshalling only. Weighted summation,
//! Newton/IRLS estimation of `u_h`, CPU worker partitioning, and the optional
//! wgpu person-score kernel are owned by `mlsirm_core::multilevel`.

use mlsirm_core::multilevel::{
    estimate_crossed_person_effects as core_estimate_crossed_person_effects,
    weighted_contextual_effect as core_weighted_contextual_effect, CrossedPersonEffectConfig,
};
use mlsirm_core::Device;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

// Keep the raw extension boundary aligned with the canonical Python design
// contract. The regression tests import the Python constant so drift fails CI.
const MAX_CONTEXT_MEMBERSHIPS: usize = 100_000;
const MAX_ROW_OFFSETS: usize = MAX_CONTEXT_MEMBERSHIPS + 1;

fn checked_usize_values(values: &[u64], name: &str) -> PyResult<Vec<usize>> {
    values
        .iter()
        .map(|&value| {
            usize::try_from(value).map_err(|_| {
                PyValueError::new_err(format!(
                    "{name} contains a value outside the platform index range"
                ))
            })
        })
        .collect()
}

/// Compute each observation's weighted contextual random-effect contribution
/// for a sparse CSR-style cross-classified multiple-membership design.
///
/// Parameters
/// ----------
/// row_offsets : numpy.ndarray[uint64]
///     CSR row pointer, length ``n_observations + 1``.
/// context_indices : numpy.ndarray[uint64]
///     Global (context_dimension, context) effect index per edge.
/// weights : numpy.ndarray[float64]
///     Membership weight per edge, same length as ``context_indices``.
/// effects : numpy.ndarray[float64]
///     Per-context random-effect value, indexed by ``context_indices``.
/// worker_count : int
///     Number of deterministic worker threads (``>= 1``); the result does
///     not depend on this value.
///
/// Returns
/// -------
/// numpy.ndarray[float64]
///     One weighted contextual effect per observation.
#[pyfunction(name = "weighted_contextual_effect")]
fn py_weighted_contextual_effect<'py>(
    py: Python<'py>,
    row_offsets: PyReadonlyArray1<'_, u64>,
    context_indices: PyReadonlyArray1<'_, u64>,
    weights: PyReadonlyArray1<'_, f64>,
    effects: PyReadonlyArray1<'_, f64>,
    worker_count: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let row_offsets = row_offsets.as_slice()?;
    let context_indices = context_indices.as_slice()?;
    if row_offsets.len() > MAX_ROW_OFFSETS {
        return Err(PyValueError::new_err(format!(
            "row_offsets exceeds maximum supported length of {MAX_ROW_OFFSETS}"
        )));
    }
    if context_indices.len() > MAX_CONTEXT_MEMBERSHIPS {
        return Err(PyValueError::new_err(format!(
            "context_indices exceeds maximum supported length of {MAX_CONTEXT_MEMBERSHIPS}"
        )));
    }

    let row_offsets = checked_usize_values(row_offsets, "row_offsets")?;
    let context_indices = checked_usize_values(context_indices, "context_indices")?;
    let result = core_weighted_contextual_effect(
        &row_offsets,
        &context_indices,
        weights.as_slice()?,
        effects.as_slice()?,
        worker_count,
    )
    .map_err(PyValueError::new_err)?;
    Ok(result.to_pyarray(py))
}

/// Estimate crossed / multiple-membership person effects ``u_h``.
///
/// Parameters
/// ----------
/// y : numpy.ndarray[float64]
///     Row-major ``n_persons * n_items`` binary responses. Non-finite or
///     negative cells are missing.
/// row_offsets : numpy.ndarray[uint64]
///     CSR row pointer, length ``n_persons + 1``.
/// context_indices : numpy.ndarray[uint64]
///     Flattened context-effect index per membership edge.
/// weights : numpy.ndarray[float64]
///     Membership weight per edge.
/// item_slopes : numpy.ndarray[float64]
///     Known item discriminations, length ``n_items``.
/// item_intercepts : numpy.ndarray[float64]
///     Known item intercepts, length ``n_items``.
/// person_offsets : numpy.ndarray[float64]
///     Optional person-level location offsets, length ``n_persons`` or empty.
/// classification_offsets : numpy.ndarray[uint64]
///     CSR pointer over flattened classifications.
/// n_persons, n_items, n_effects : int
///     Declared problem sizes.
/// prior_precision : float
///     Gaussian prior precision ``1 / sigma_u^2``.
/// max_iter : int
///     Newton iteration budget.
/// tol : float
///     Absolute effect-step tolerance.
/// worker_count : int
///     Deterministic CPU workers.
/// device : str
///     ``cpu``, ``gpu``, or ``auto``.
///
/// Returns
/// -------
/// dict
///     ``effects``, ``loglik``, ``n_iter``, ``converged``, ``used_gpu``,
///     and ``termination_reason``.
#[pyfunction(name = "estimate_crossed_person_effects")]
#[allow(clippy::too_many_arguments)]
fn py_estimate_crossed_person_effects<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'_, f64>,
    row_offsets: PyReadonlyArray1<'_, u64>,
    context_indices: PyReadonlyArray1<'_, u64>,
    weights: PyReadonlyArray1<'_, f64>,
    item_slopes: PyReadonlyArray1<'_, f64>,
    item_intercepts: PyReadonlyArray1<'_, f64>,
    person_offsets: PyReadonlyArray1<'_, f64>,
    classification_offsets: PyReadonlyArray1<'_, u64>,
    n_persons: usize,
    n_items: usize,
    n_effects: usize,
    prior_precision: f64,
    max_iter: usize,
    tol: f64,
    worker_count: usize,
    device: &str,
) -> PyResult<Py<PyDict>> {
    let row_offsets = row_offsets.as_slice()?;
    let context_indices = context_indices.as_slice()?;
    if row_offsets.len() > MAX_ROW_OFFSETS {
        return Err(PyValueError::new_err(format!(
            "row_offsets exceeds maximum supported length of {MAX_ROW_OFFSETS}"
        )));
    }
    if context_indices.len() > MAX_CONTEXT_MEMBERSHIPS {
        return Err(PyValueError::new_err(format!(
            "context_indices exceeds maximum supported length of {MAX_CONTEXT_MEMBERSHIPS}"
        )));
    }
    let device = Device::parse(device)
        .ok_or_else(|| PyValueError::new_err("device must be one of 'cpu', 'gpu', or 'auto'"))?;
    let row_offsets = checked_usize_values(row_offsets, "row_offsets")?;
    let context_indices = checked_usize_values(context_indices, "context_indices")?;
    let classification_offsets =
        checked_usize_values(classification_offsets.as_slice()?, "classification_offsets")?;
    let result = core_estimate_crossed_person_effects(
        y.as_slice()?,
        &row_offsets,
        &context_indices,
        weights.as_slice()?,
        item_slopes.as_slice()?,
        item_intercepts.as_slice()?,
        person_offsets.as_slice()?,
        &classification_offsets,
        n_persons,
        n_items,
        n_effects,
        CrossedPersonEffectConfig {
            prior_precision,
            max_iter,
            tol,
            worker_count,
            device,
        },
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("effects", result.effects.to_pyarray(py))?;
    out.set_item("loglik", result.loglik)?;
    out.set_item("n_iter", result.n_iter)?;
    out.set_item("converged", result.converged)?;
    out.set_item("used_gpu", result.used_gpu)?;
    out.set_item("termination_reason", result.termination_reason)?;
    Ok(out.into())
}

#[pymodule]
#[pyo3(name = "_multilevel_core")]
fn fast_mlsirm_multilevel_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_weighted_contextual_effect, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_crossed_person_effects, m)?)?;
    Ok(())
}
