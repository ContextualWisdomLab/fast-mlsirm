#!/usr/bin/env python3
"""Apply the reviewed rotation integration to large repository entrypoints.

The GitHub contents API is inefficient for editing the repository's very large
PyO3 module. This deterministic one-shot patcher uses exact markers, refuses
ambiguous states, and is removed by its workflow after producing the reviewed
commit.
"""

from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    """Replace one exact marker or fail without modifying the file."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{path}: expected exactly one integration marker")
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, marker: str, content: str) -> None:
    """Append content only when its unique marker is absent."""
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if marker in text:
        raise SystemExit(f"{path}: integration marker already present")
    file_path.write_text(text.rstrip() + "\n\n" + content.rstrip() + "\n", encoding="utf-8")


def patch_core_module() -> None:
    """Expose the new Rust module and selector types."""
    replace_once(
        "crates/mlsirm-core/src/lib.rs",
        "pub mod reliability;\npub mod rsm;",
        "pub mod reliability;\npub mod rotation;\npub mod rsm;",
    )
    replace_once(
        "crates/mlsirm-core/src/rotation/mod.rs",
        "mod optimizer;\n\npub use criteria::{CriterionEvaluation, RotationCriterion};",
        "mod optimizer;\nmod selector;\n\npub use criteria::{CriterionEvaluation, RotationCriterion};\npub use selector::{\n    select_rotation_criterion, RotationCandidateEvidence, RotationSelectionPolicy,\n    RotationSelectionResult,\n};",
    )


def patch_pyo3_imports() -> None:
    """Add rotation core imports and 3-D NumPy support."""
    old = """use mlsirm_core::factor::{
    glb_fa_corr as core_glb_fa_corr, glb_fa_data as core_glb_fa_data,
    minres_fa_corr as core_minres_fa_corr, minres_fa_data as core_minres_fa_data,
    omega_total_1f_corr as core_omega_total_1f_corr,
    omega_total_1f_data as core_omega_total_1f_data, velicer_map_corr as core_velicer_map_corr,
    velicer_map_data as core_velicer_map_data, MinresFaResult,
};
"""
    new = old + """use mlsirm_core::rotation::{
    rotate_factor_loadings as core_rotate_factor_loadings,
    rotation_criterion_from_name as core_rotation_criterion_from_name,
    select_rotation_criterion as core_select_rotation_criterion,
    RotationConfig as CoreRotationConfig, RotationMode as CoreRotationMode,
    RotationSelectionPolicy as CoreRotationSelectionPolicy, RotationSolution as CoreRotationSolution,
};
"""
    replace_once("crates/fast-mlsirm-py/src/lib.rs", old, new)
    replace_once(
        "crates/fast-mlsirm-py/src/lib.rs",
        "use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};",
        "use numpy::{\n    PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,\n    PyUntypedArrayMethods,\n};",
    )


def binding_source() -> str:
    """Return the reviewed PyO3 binding block."""
    return r'''
fn optional_matrix_values(
    value: Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Option<Vec<f64>>> {
    value
        .map(|array| array.as_slice().map(|slice| slice.to_vec()))
        .transpose()
}

fn rotation_solution_dict(
    py: Python<'_>,
    solution: &CoreRotationSolution,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let out = pyo3::types::PyDict::new(py);
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

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn rotate_factor_loadings(
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
) -> PyResult<(
    HashMap<String, Vec<f64>>,
    HashMap<String, f64>,
    HashMap<String, usize>,
    HashMap<String, bool>,
    HashMap<String, String>,
)> {
    let shape = loadings.shape();
    let rows = shape[0];
    let factors = shape[1];
    let target_values = optional_matrix_values(target)?;
    let weight_values = optional_matrix_values(weights)?;
    let core_criterion = core_rotation_criterion_from_name(
        criterion,
        rows,
        factors,
        kappa,
        gamma,
        delta,
        simplimax_zeros,
        target_values,
        weight_values,
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
    let arrays = HashMap::from([
        ("pattern_matrix".to_string(), result.pattern_matrix),
        ("structure_matrix".to_string(), result.structure_matrix),
        ("factor_correlation".to_string(), result.factor_correlation),
        ("transform_matrix".to_string(), result.transform_matrix),
        ("start_values".to_string(), result.start_values),
        (
            "start_converged".to_string(),
            result
                .start_converged
                .into_iter()
                .map(|value| if value { 1.0 } else { 0.0 })
                .collect(),
        ),
    ]);
    let floats = HashMap::from([
        ("criterion_value".to_string(), result.criterion_value),
        ("gradient_norm".to_string(), result.gradient_norm),
        (
            "max_factor_correlation".to_string(),
            result.max_factor_correlation,
        ),
    ]);
    let integers = HashMap::from([
        ("n_rows".to_string(), result.n_rows),
        ("n_factors".to_string(), result.n_factors),
        ("iterations".to_string(), result.iterations),
        ("best_start_index".to_string(), result.best_start_index),
        ("n_starts".to_string(), result.n_starts),
        ("converged_starts".to_string(), result.converged_starts),
        ("basin_support".to_string(), result.basin_support),
        ("distinct_minima".to_string(), result.distinct_minima),
        ("worker_count".to_string(), result.worker_count),
    ]);
    let booleans = HashMap::from([
        ("converged".to_string(), result.converged),
        ("normalized".to_string(), result.normalized),
    ]);
    let strings = HashMap::from([
        ("criterion".to_string(), result.criterion_name.to_string()),
        ("mode".to_string(), result.mode.as_str().to_string()),
        (
            "termination_reason".to_string(),
            result.termination_reason.to_string(),
        ),
        ("backend".to_string(), result.backend.to_string()),
    ]);
    Ok((arrays, floats, integers, booleans, strings))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn rotation_criterion_value_gradient(
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn select_rotation_criterion(
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
) -> PyResult<Py<pyo3::types::PyDict>> {
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
            let one_size = rows * factors;
            array
                .as_slice()?
                .chunks_exact(one_size)
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
    let out = pyo3::types::PyDict::new(py);
    out.set_item("selected_index", result.selected_index)?;
    out.set_item("selected_criterion", result.selected_criterion)?;
    out.set_item("policy", result.policy.as_str())?;
    out.set_item("bootstrap_replicates", result.bootstrap_replicates)?;
    out.set_item("evidence_grade", result.evidence_grade)?;
    out.set_item("warning", result.warning)?;
    let candidate_list = pyo3::types::PyList::empty(py);
    for candidate in &result.candidates {
        let candidate_dict = pyo3::types::PyDict::new(py);
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
'''


def patch_pyo3_functions() -> None:
    """Insert binding functions and module registrations."""
    replace_once(
        "crates/fast-mlsirm-py/src/lib.rs",
        "\n#[pymodule]\n#[pyo3(name = \"_core\")]\nfn fast_mlsirm_core",
        binding_source() + "\n#[pymodule]\n#[pyo3(name = \"_core\")]\nfn fast_mlsirm_core",
    )
    replace_once(
        "crates/fast-mlsirm-py/src/lib.rs",
        "    m.add_function(wrap_pyfunction!(neg_loglik_and_grad, m)?)?;\n",
        "    m.add_function(wrap_pyfunction!(neg_loglik_and_grad, m)?)?;\n"
        "    m.add_function(wrap_pyfunction!(rotate_factor_loadings, m)?)?;\n"
        "    m.add_function(wrap_pyfunction!(rotation_criterion_value_gradient, m)?)?;\n"
        "    m.add_function(wrap_pyfunction!(select_rotation_criterion, m)?)?;\n",
    )


def patch_python_exports() -> None:
    """Expose the Python result types and functions from the package root."""
    import_block = """from .rotation import (
    RotationCriterionInfo as RotationCriterionInfo,
    RotationSolution as RotationSolution,
    available_rotation_criteria as available_rotation_criteria,
    rotate_factor_loadings as rotate_factor_loadings,
    rotation_criterion_value_gradient as rotation_criterion_value_gradient,
)
from .rotation_selection import (
    RotationCandidateEvidence as RotationCandidateEvidence,
    RotationSelectionResult as RotationSelectionResult,
    select_rotation_criterion as select_rotation_criterion,
)
"""
    replace_once(
        "python/fast_mlsirm/__init__.py",
        "from .types import DimensionalityDiagnostics",
        import_block + "from .types import DimensionalityDiagnostics",
    )
    replace_once(
        "python/fast_mlsirm/__init__.py",
        "    \"__version__\",\n",
        "    \"__version__\",\n"
        "    \"RotationCriterionInfo\",\n"
        "    \"RotationSolution\",\n"
        "    \"RotationCandidateEvidence\",\n"
        "    \"RotationSelectionResult\",\n"
        "    \"available_rotation_criteria\",\n"
        "    \"rotate_factor_loadings\",\n"
        "    \"rotation_criterion_value_gradient\",\n"
        "    \"select_rotation_criterion\",\n",
    )


def patch_factor_scope() -> None:
    """Remove the obsolete claim that the package has no rotation support."""
    replace_once(
        "python/fast_mlsirm/factor.py",
        "REDUCED SCOPE (spec decision): no rotation (loadings are unrotated), no\nSchmid-Leiman, no omega_hierarchical, no ML/WLS/GLS methods, no factor\nscores.",
        "REDUCED EXTRACTION SCOPE: factor extraction returns unrotated loadings;\nuse :mod:`fast_mlsirm.rotation` for Rust-backed analytic rotation. This\nextraction module still excludes Schmid-Leiman, omega_hierarchical, ML/WLS/GLS\nmethods, and factor scores.",
    )


def patch_docs() -> None:
    """Add stable user-facing documentation and changelog entry."""
    append_once(
        "README.md",
        "## Adaptive exploratory factor rotation",
        """## Adaptive exploratory factor rotation

`fast-mlsirm` now provides a Rust analytic-rotation registry, deterministic
orthogonal/oblique multi-start gradient projection, basin diagnostics, target
rotation, bifactor criteria, and a criterion-neutral empirical selector. The
selector uses explicit policies and optional bootstrap congruence; it never
compares incompatible criterion objective values or claims a universal optimum.
See [`docs/adaptive_factor_rotation.md`](docs/adaptive_factor_rotation.md).
""",
    )
    replace_once(
        "CHANGELOG.md",
        "## Unreleased\n\n### Changed\n",
        "## Unreleased\n\n### Added\n\n"
        "- Added Rust-backed adaptive exploratory factor rotation: a broad analytic criterion registry, orthogonal/oblique deterministic multi-start gradient projection, local-basin diagnostics, Kaiser normalization, target and bifactor rotations, and policy-conditional criterion-neutral selection with optional bootstrap Tucker congruence and Pareto evidence.\n\n"
        "### Changed\n",
    )


def main() -> None:
    """Apply every integration edit exactly once."""
    patch_core_module()
    patch_pyo3_imports()
    patch_pyo3_functions()
    patch_python_exports()
    patch_factor_scope()
    patch_docs()


if __name__ == "__main__":
    main()
