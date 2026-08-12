//! Python bindings for the sparse cross-classified multiple-membership
//! contextual-effects predictor (see `mlsirm_core::multilevel`).
//!
//! This module performs shape/type marshalling only. The additive sum over
//! membership-weighted context effects, its determinism across worker
//! counts, and its input validation are owned by
//! `mlsirm_core::multilevel::weighted_contextual_effect`.

use mlsirm_core::multilevel::weighted_contextual_effect as core_weighted_contextual_effect;
use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

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
    let row_offsets: Vec<usize> = row_offsets
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
    let context_indices: Vec<usize> = context_indices
        .as_slice()?
        .iter()
        .map(|&v| v as usize)
        .collect();
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

#[pymodule]
#[pyo3(name = "_multilevel_core")]
fn fast_mlsirm_multilevel_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_weighted_contextual_effect, m)?)?;
    Ok(())
}
