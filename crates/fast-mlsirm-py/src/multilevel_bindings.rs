//! Python bindings for the sparse cross-classified multiple-membership
//! contextual-effects predictor (see `mlsirm_core::multilevel`).
//!
//! This module performs shape/type marshalling only. The additive sum over
//! membership-weighted context effects, its determinism across worker
//! counts, and its input validation are owned by
//! `mlsirm_core::multilevel::weighted_contextual_effect`.

use mlsirm_core::longitudinal::fit_longitudinal_state as core_fit_longitudinal_state;
use mlsirm_core::longitudinal_irt::{
    fit_hierarchical_ctar_rasch as core_fit_hierarchical_ctar_rasch,
    simulate_hierarchical_ctar_rasch as core_simulate_hierarchical_ctar_rasch,
    HierarchicalCtarRaschConfig,
};
use mlsirm_core::multilevel::{
    estimate_crossed_person_effects as core_estimate_crossed_person_effects,
    weighted_contextual_effect as core_weighted_contextual_effect, CrossedPersonEffectConfig,
};
use mlsirm_core::Device;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

// Keep the raw extension boundary aligned with the canonical Python design
// contract. The regression tests import the Python constant so drift fails CI.
const MAX_CONTEXT_MEMBERSHIPS: usize = 100_000;
const MAX_ROW_OFFSETS: usize = MAX_CONTEXT_MEMBERSHIPS + 1;
const MAX_HIERARCHICAL_OCCASIONS: usize = 100_000;
const MAX_HIERARCHICAL_ITEMS: usize = 4_096;

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

/// Fit the Rust-owned repeated-measurement state layer.
///
/// Parameters
/// ----------
/// row_offsets : numpy.ndarray[uint64]
///     CSR-style respondent pointer, length ``n_respondents + 1``.
/// sequence_indices : numpy.ndarray[uint64]
///     Discrete occasion indices aligned with ``values``.
/// time_offsets_milliseconds : numpy.ndarray[int64]
///     Exact millisecond offsets aligned with ``values``.
/// values : numpy.ndarray[float64]
///     Observed states; ``NaN`` marks a missing occasion.
/// state_kind : str
///     Compatibility wire label for the requested state predictor.
/// ar_coefficient : float or None
///     Caller-supplied discrete AR coefficient, or ``None`` for OLS trends.
/// worker_count : int
///     Number of deterministic worker threads (``>= 1``).
///
/// Returns
/// -------
/// dict
///     Predicted states, respondent intercepts/slopes, RMSE, and counts.
#[pyfunction(name = "fit_longitudinal_state")]
fn py_fit_longitudinal_state<'py>(
    py: Python<'py>,
    row_offsets: PyReadonlyArray1<'_, u64>,
    sequence_indices: PyReadonlyArray1<'_, u64>,
    time_offsets_milliseconds: PyReadonlyArray1<'_, i64>,
    values: PyReadonlyArray1<'_, f64>,
    state_kind: &str,
    ar_coefficient: Option<f64>,
    worker_count: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let row_offsets_view = row_offsets.as_slice()?;
    if row_offsets_view.len() > MAX_ROW_OFFSETS {
        return Err(PyValueError::new_err(format!(
            "row_offsets exceeds maximum supported length of {MAX_ROW_OFFSETS}"
        )));
    }
    let sequence_indices_view = sequence_indices.as_slice()?;
    if sequence_indices_view.len() > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "sequence_indices exceeds maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    let time_offsets_view = time_offsets_milliseconds.as_slice()?;
    if time_offsets_view.len() > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "time_offsets_milliseconds exceeds maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    let values_view = values.as_slice()?;
    if values_view.len() > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "values exceeds maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    let row_offsets = checked_usize_values(row_offsets_view, "row_offsets")?;
    let sequence_indices = checked_usize_values(sequence_indices_view, "sequence_indices")?;
    let time_offsets = time_offsets_view.to_vec();
    let values = values_view.to_vec();
    let state_kind = state_kind.to_owned();
    let fit = py
        .detach(move || {
            core_fit_longitudinal_state(
                &row_offsets,
                &sequence_indices,
                &time_offsets,
                &values,
                &state_kind,
                ar_coefficient,
                worker_count,
            )
        })
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("state", fit.state.to_pyarray(py))?;
    result.set_item("intercepts", fit.intercepts.to_pyarray(py))?;
    result.set_item("slopes", fit.slopes.to_pyarray(py))?;
    result.set_item("ar_coefficient", fit.ar_coefficient)?;
    result.set_item("rmse", fit.rmse)?;
    result.set_item("observed_count", fit.observed_count)?;
    result.set_item("transition_count", fit.transition_count)?;
    result.set_item("engine", "rust_cpu_multithreaded")?;
    Ok(result)
}

/// Fit the joint MAP hierarchical continuous-time AR(1) Rasch slice.
///
/// Parameters
/// ----------
/// row_offsets : numpy.ndarray[uint64]
///     CSR-style respondent pointer, length ``n_respondents + 1``.
/// time_offsets_milliseconds : numpy.ndarray[int64]
///     Exact millisecond offsets aligned with the occasion axis of
///     ``responses``.
/// responses : numpy.ndarray[float64]
///     Occasion-major binary matrix with shape
///     ``(n_occasions, n_items)``. ``NaN`` marks a missing response.
/// worker_count : int
///     Number of deterministic person-shard worker threads (``>= 1``).
/// max_iter : int
///     Maximum packed L-BFGS iterations (``>= 1``).
/// tolerance : float
///     Relative L-BFGS tolerance; must be finite and strictly positive.
/// hessian_step : float
///     Central-difference step for the hyperparameter Hessian.
///
/// Returns
/// -------
/// dict
///     Joint MAP states, Wald intervals, item intercepts, estimated
///     ``(mu, tau, lambda)``, and normative estimand metadata.
#[pyfunction(name = "fit_hierarchical_ctar_rasch")]
fn py_fit_hierarchical_ctar_rasch<'py>(
    py: Python<'py>,
    row_offsets: PyReadonlyArray1<'_, u64>,
    time_offsets_milliseconds: PyReadonlyArray1<'_, i64>,
    responses: PyReadonlyArray2<'_, f64>,
    worker_count: usize,
    max_iter: usize,
    tolerance: f64,
    hessian_step: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = responses.shape();
    if shape[0] > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "responses occasion axis exceeds maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    if shape[1] > MAX_HIERARCHICAL_ITEMS {
        return Err(PyValueError::new_err(format!(
            "responses item axis exceeds maximum supported length of {MAX_HIERARCHICAL_ITEMS}"
        )));
    }
    let row_offsets_view = row_offsets.as_slice()?;
    if row_offsets_view.len() > MAX_ROW_OFFSETS {
        return Err(PyValueError::new_err(format!(
            "row_offsets exceeds maximum supported length of {MAX_ROW_OFFSETS}"
        )));
    }
    let row_offsets = checked_usize_values(row_offsets_view, "row_offsets")?;
    let time_offsets_view = time_offsets_milliseconds.as_slice()?;
    if time_offsets_view.len() > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "time offsets exceed maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    let time_offsets = time_offsets_view.to_vec();
    let responses = responses.as_slice()?.to_vec();
    let n_items = shape[1];
    let config = HierarchicalCtarRaschConfig {
        worker_count,
        max_iter,
        tolerance,
        hessian_step,
    };
    let fit = py
        .detach(move || {
            core_fit_hierarchical_ctar_rasch(
                &row_offsets,
                &time_offsets,
                &responses,
                n_items,
                config,
            )
        })
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("state", fit.state.to_pyarray(py))?;
    result.set_item("state_se", fit.state_se.to_pyarray(py))?;
    result.set_item("state_lower", fit.state_lower.to_pyarray(py))?;
    result.set_item("state_upper", fit.state_upper.to_pyarray(py))?;
    result.set_item("item_intercepts", fit.item_intercepts.to_pyarray(py))?;
    result.set_item("population_mean", fit.population_mean)?;
    result.set_item("population_sd", fit.population_sd)?;
    result.set_item("decay_rate", fit.decay_rate)?;
    result.set_item("unit_time_ar_coefficient", fit.unit_time_ar_coefficient)?;
    result.set_item("hyperparameter_se", fit.hyperparameter_se.to_vec())?;
    result.set_item("hyperparameter_lower", fit.hyperparameter_lower.to_vec())?;
    result.set_item("hyperparameter_upper", fit.hyperparameter_upper.to_vec())?;
    result.set_item(
        "hyperparameter_intervals_identified",
        fit.hyperparameter_intervals_identified,
    )?;
    result.set_item("state_intervals_identified", fit.state_intervals_identified)?;
    result.set_item("observed_count", fit.observed_count)?;
    result.set_item("transition_count", fit.transition_count)?;
    result.set_item("status", fit.status)?;
    result.set_item("estimand_scope", fit.estimand_scope)?;
    result.set_item("transition_kind", fit.transition_kind)?;
    result.set_item("interval_kind", fit.interval_kind)?;
    result.set_item("engine", fit.engine)?;
    result.set_item("population_random_effects_estimated", true)?;
    result.set_item("ar_coefficient_estimated", true)?;
    result.set_item("ar_coefficient_source", "joint_map")?;
    result.set_item("multiple_membership_estimated", false)?;
    result.set_item("gpu_parity", false)?;
    Ok(result)
}

/// Simulate hierarchical continuous-time AR(1) Rasch responses.
///
/// Parameters
/// ----------
/// row_offsets : numpy.ndarray[uint64]
///     CSR-style respondent pointer, length ``n_respondents + 1``.
/// time_offsets_milliseconds : numpy.ndarray[int64]
///     Exact millisecond offsets aligned with the generated occasions.
/// item_intercepts : numpy.ndarray[float64]
///     Sum-to-zero Rasch item intercepts used as generating values.
/// population_mean : float
///     Generating population mean.
/// population_sd : float
///     Generating stationary standard deviation.
/// decay_rate : float
///     Generating continuous-time decay rate per day.
/// seed : int
///     Deterministic LCG seed.
///
/// Returns
/// -------
/// dict
///     Generating latent states and occasion-major binary responses.
#[pyfunction(name = "simulate_hierarchical_ctar_rasch")]
fn py_simulate_hierarchical_ctar_rasch<'py>(
    py: Python<'py>,
    row_offsets: PyReadonlyArray1<'_, u64>,
    time_offsets_milliseconds: PyReadonlyArray1<'_, i64>,
    item_intercepts: PyReadonlyArray1<'_, f64>,
    population_mean: f64,
    population_sd: f64,
    decay_rate: f64,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let item_intercepts_view = item_intercepts.as_slice()?;
    if item_intercepts_view.len() > MAX_HIERARCHICAL_ITEMS {
        return Err(PyValueError::new_err(format!(
            "item_intercepts exceeds maximum supported length of {MAX_HIERARCHICAL_ITEMS}"
        )));
    }
    let item_intercepts = item_intercepts_view.to_vec();
    let n_items = item_intercepts.len();
    let row_offsets_view = row_offsets.as_slice()?;
    if row_offsets_view.len() > MAX_ROW_OFFSETS {
        return Err(PyValueError::new_err(format!(
            "row_offsets exceeds maximum supported length of {MAX_ROW_OFFSETS}"
        )));
    }
    let row_offsets = checked_usize_values(row_offsets_view, "row_offsets")?;
    let time_offsets_view = time_offsets_milliseconds.as_slice()?;
    if time_offsets_view.len() > MAX_HIERARCHICAL_OCCASIONS {
        return Err(PyValueError::new_err(format!(
            "time offsets exceed maximum supported length of {MAX_HIERARCHICAL_OCCASIONS}"
        )));
    }
    let time_offsets = time_offsets_view.to_vec();
    let (state, responses) = py
        .detach(move || {
            core_simulate_hierarchical_ctar_rasch(
                &row_offsets,
                &time_offsets,
                n_items,
                population_mean,
                population_sd,
                decay_rate,
                &item_intercepts,
                seed,
            )
        })
        .map_err(PyValueError::new_err)?;
    let result = PyDict::new(py);
    result.set_item("state", state.to_pyarray(py))?;
    result.set_item("responses", responses.to_pyarray(py))?;
    result.set_item("n_items", n_items)?;
    Ok(result)
}

/// Estimate crossed / multiple-membership person effects ``u_h``.
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
    m.add_function(wrap_pyfunction!(py_fit_longitudinal_state, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit_hierarchical_ctar_rasch, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_hierarchical_ctar_rasch, m)?)?;
    m.add_function(wrap_pyfunction!(py_estimate_crossed_person_effects, m)?)?;
    Ok(())
}
