//! Provider-free Python bindings for the Rust rotation core.

use mlsirm_core::rotation::{
    rotate_factor_loadings as core_rotate_factor_loadings,
    rotation_criterion_from_name as core_rotation_criterion_from_name,
    select_rotation_criterion as core_select_rotation_criterion,
    RotationConfig as CoreRotationConfig, RotationMode as CoreRotationMode,
    RotationSelectionPolicy as CoreRotationSelectionPolicy, RotationSolution as CoreRotationSolution,
};
use numpy::{PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use pyo3::wrap_pyfunction;

fn optional_matrix_values(
    value: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Option<Vec<f64>>> {
    Ok(value
        .map(|array| array.as_slice().map(|slice| slice.to_vec()))
        .transpose()?)
}

fn rotation_solution_dict(
    py: Python<'_>,
    solution: &CoreRotationSolution,
) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("pattern_matrix", solution.pattern_matrix.clone())?;
    out.set_item("structure_matrix", solution.structure_matrix.clone())?;
    out.set_item("factor_correlation", solution.factor_correlation.clone())?;
    out.set_item("transform_matrix", solution.transform_matrix.clone())?;
    out.set_item("n_rows", solution.n_rows)?;
    out.set_item("n_factors", solution.n_factors)?;
    out.set_item("criterion", solution.criterion_name)?;
    out.set_item("mode", solution.mode.as_str())?;
    out.set_item("criterion_value", solution.criterion_value)?;
    out.set_item("gradient_norm", solution.gradient_norm)?;
    out.set_item("iterations", solution.iterations)?;
    out.set_item("converged", solution.converged)?;
    out.set_item("termination_reason", solution.termination_reason)?;
    out.set_item("best_start_index", solution.best_start_index)?;
    out.set_item("n_starts", solution.n_starts)?;
    out.set_item("converged_starts", solution.converged_starts)?;
    out.set_item("basin_support", solution.basin_support)?;
    out.set_item("distinct_minima", solution.distinct_minima)?;
    out.set_item("start_values", solution.start_values.clone())?;
    out.set_item("start_converged", solution.start_converged.clone())?;
    out.set_item("max_factor_correlation", solution.max_factor_correlation)?;
    out.set_item("normalized", solution.normalized)?;
    out.set_item("worker_count", solution.worker_count)?;
    out.set_item("backend", solution.backend)?;
    Ok(out.into())
}

#[pyfunction(name = "rotate_factor_loadings")]
#[allow(clippy::too_many_arguments)]
fn py_rotate_factor_loadings(
    py: Python<'_>,
    loadings: PyReadonlyArray2<'_, f64>,
    criterion: &str,
    mode: &str,
    normalize: bool,
    n_starts: usize,
    seed: u64,
    max_iter: usize,
    tolerance: f64,
    function_window: usize,
    max_line_search: usize,
    basin_tolerance: f64,
    max_threads: usize,
    kappa: Option<f64>,
    gamma: Option<f64>,
    delta: Option<f64>,
    simplimax_zeros: Option<usize>,
    target: Option<PyReadonlyArray2<'_, f64>>,
    weights: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Py<PyDict>> {
    let shape = loadings.shape();
    let rows = shape[0];
    let factors = shape[1];
    let core_criterion = core_rotation_criterion_from_name(
        criterion,
        rows,
        factors,
        kappa,
        gamma,
        delta,
        simplimax_zeros,
        optional_matrix_values(target)?,
        optional_matrix_values(weights)?,
    )
    .map_err(PyValueError::new_err)?;
    let core_mode = CoreRotationMode::parse(mode)
        .ok_or_else(|| PyValueError::new_err("mode must be orthogonal or oblique"))?;
    let config = CoreRotationConfig {
        mode: core_mode,
        normalize,
        n_starts,
        seed,
        max_iter,
        tolerance,
        function_window,
        max_line_search,
        basin_tolerance,
        max_threads,
    };
    let result = core_rotate_factor_loadings(
        loadings.as_slice()?,
        rows,
        factors,
        &core_criterion,
        &config,
    )
    .map_err(PyValueError::new_err)?;
    rotation_solution_dict(py, &result)
}

#[pyfunction(name = "rotation_criterion_value_gradient")]
#[allow(clippy::too_many_arguments)]
fn py_rotation_criterion_value_gradient(
    loadings: PyReadonlyArray2<'_, f64>,
    criterion: &str,
    kappa: Option<f64>,
    gamma: Option<f64>,
    delta: Option<f64>,
    simplimax_zeros: Option<usize>,
    target: Option<PyReadonlyArray2<'_, f64>>,
    weights: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<(f64, Vec<f64>)> {
    let shape = loadings.shape();
    let rows = shape[0];
    let factors = shape[1];
    let core_criterion = core_rotation_criterion_from_name(
        criterion,
        rows,
        factors,
        kappa,
        gamma,
        delta,
        simplimax_zeros,
        optional_matrix_values(target)?,
        optional_matrix_values(weights)?,
    )
    .map_err(PyValueError::new_err)?;
    let evaluation = core_criterion
        .evaluate(loadings.as_slice()?, rows, factors)
        .map_err(PyValueError::new_err)?;
    Ok((evaluation.value, evaluation.gradient))
}

#[pyfunction(name = "select_rotation_criterion")]
#[allow(clippy::too_many_arguments)]
fn py_select_rotation_criterion(
    py: Python<'_>,
    loadings: PyReadonlyArray2<'_, f64>,
    candidates: Vec<String>,
    mode: &str,
    policy: &str,
    normalize: bool,
    n_starts: usize,
    seed: u64,
    max_iter: usize,
    tolerance: f64,
    function_window: usize,
    max_line_search: usize,
    basin_tolerance: f64,
    max_threads: usize,
    kappa: Option<f64>,
    gamma: Option<f64>,
    delta: Option<f64>,
    simplimax_zeros: Option<usize>,
    target: Option<PyReadonlyArray2<'_, f64>>,
    weights: Option<PyReadonlyArray2<'_, f64>>,
    bootstrap_loadings: Option<PyReadonlyArray3<'_, f64>>,
    theory_target: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Py<PyDict>> {
    let shape = loadings.shape();
    let rows = shape[0];
    let factors = shape[1];
    let target_values = optional_matrix_values(target)?;
    let weight_values = optional_matrix_values(weights)?;
    let mut core_candidates = Vec::with_capacity(candidates.len());
    for name in &candidates {
        core_candidates.push(
            core_rotation_criterion_from_name(
                name,
                rows,
                factors,
                kappa,
                gamma,
                delta,
                simplimax_zeros,
                target_values.clone(),
                weight_values.clone(),
            )
            .map_err(PyValueError::new_err)?,
        );
    }
    let core_mode = CoreRotationMode::parse(mode)
        .ok_or_else(|| PyValueError::new_err("mode must be orthogonal or oblique"))?;
    let core_policy = CoreRotationSelectionPolicy::parse(policy)
        .ok_or_else(|| PyValueError::new_err("unknown rotation selection policy"))?;
    let config = CoreRotationConfig {
        mode: core_mode,
        normalize,
        n_starts,
        seed,
        max_iter,
        tolerance,
        function_window,
        max_line_search,
        basin_tolerance,
        max_threads,
    };
    let bootstrap_values = match bootstrap_loadings {
        Some(array) => {
            let bootstrap_shape = array.shape();
            if bootstrap_shape[1] != rows || bootstrap_shape[2] != factors {
                return Err(PyValueError::new_err(
                    "bootstrap loading matrices must match the reference shape",
                ));
            }
            let matrix_size = rows * factors;
            array
                .as_slice()?
                .chunks_exact(matrix_size)
                .map(|chunk| chunk.to_vec())
                .collect()
        }
        None => Vec::new(),
    };
    let theory_values = optional_matrix_values(theory_target)?;
    let result = core_select_rotation_criterion(
        loadings.as_slice()?,
        rows,
        factors,
        &core_candidates,
        &config,
        core_policy,
        &bootstrap_values,
        theory_values.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("selected_index", result.selected_index)?;
    out.set_item("selected_criterion", result.selected_criterion)?;
    out.set_item("policy", result.policy.as_str())?;
    out.set_item("bootstrap_replicates", result.bootstrap_replicates)?;
    out.set_item("evidence_grade", result.evidence_grade)?;
    out.set_item("warning", result.warning)?;
    let candidate_list = PyList::empty(py);
    for candidate in &result.candidates {
        let candidate_dict = PyDict::new(py);
        candidate_dict.set_item("criterion", candidate.criterion_name)?;
        candidate_dict.set_item("solution", rotation_solution_dict(py, &candidate.solution)?)?;
        candidate_dict.set_item("row_complexity", candidate.row_complexity)?;
        candidate_dict.set_item("factor_balance", candidate.factor_balance)?;
        candidate_dict.set_item(
            "max_factor_correlation",
            candidate.max_factor_correlation,
        )?;
        candidate_dict.set_item("convergence_rate", candidate.convergence_rate)?;
        candidate_dict.set_item("basin_support_rate", candidate.basin_support_rate)?;
        candidate_dict.set_item("bootstrap_congruence", candidate.bootstrap_congruence)?;
        candidate_dict.set_item(
            "bootstrap_min_congruence",
            candidate.bootstrap_min_congruence,
        )?;
        candidate_dict.set_item("target_rmse", candidate.target_rmse)?;
        candidate_dict.set_item("policy_score", candidate.policy_score)?;
        candidate_dict.set_item("pareto_optimal", candidate.pareto_optimal)?;
        candidate_list.append(candidate_dict)?;
    }
    out.set_item("candidates", candidate_list)?;
    Ok(out.into())
}

#[pymodule]
#[pyo3(name = "_rotation_core")]
fn fast_mlsirm_rotation_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_rotate_factor_loadings, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotation_criterion_value_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(py_select_rotation_criterion, m)?)?;
    Ok(())
}
