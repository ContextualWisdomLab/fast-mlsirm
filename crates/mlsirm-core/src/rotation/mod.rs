//! Adaptive exploratory factor rotation.
//!
//! This module separates three concerns that are often conflated in factor
//! software: a criterion registry, manifold optimization, and empirical
//! evidence about local minima.  It returns the best observed stationary basin
//! together with all start-level objective values; it never labels a finite
//! multi-start result as a mathematically proven global optimum.
//!
//! # References
//!
//! Bernaards, C. A., & Jennrich, R. I. (2005). Gradient projection algorithms
//! and software for arbitrary rotation criteria in factor analysis.
//! *Educational and Psychological Measurement, 65*(5), 676-696.
//! https://doi.org/10.1177/0013164404272507
//!
//! Browne, M. W. (2001). An overview of analytic rotation in exploratory factor
//! analysis. *Multivariate Behavioral Research, 36*(1), 111-150.
//! https://doi.org/10.1207/S15327906MBR3601_05

mod criteria;
mod matrix;
mod optimizer;

pub use criteria::{CriterionEvaluation, RotationCriterion};
use matrix::{factor_correlation, identity, matmul, random_oblique, random_orthogonal};
use optimizer::{optimize_start, OptimizerSettings, StartSolution};
use std::cmp::Ordering;

/// Rotation manifold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RotationMode {
    /// Preserve orthogonal factors.
    Orthogonal,
    /// Allow correlated factors with unit-diagonal factor correlation matrix.
    Oblique,
}

impl RotationMode {
    /// Parse a stable public mode name.
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "orthogonal" | "orth" | "t" => Some(Self::Orthogonal),
            "oblique" | "oblq" | "q" => Some(Self::Oblique),
            _ => None,
        }
    }

    /// Stable public identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Orthogonal => "orthogonal",
            Self::Oblique => "oblique",
        }
    }
}

/// Numerical and reproducibility settings for exploratory rotation.
#[derive(Clone, Debug)]
pub struct RotationConfig {
    /// Orthogonal or oblique manifold.
    pub mode: RotationMode,
    /// Apply Kaiser row normalization before optimization and undo it afterward.
    pub normalize: bool,
    /// Total deterministic starts, including the identity start.
    pub n_starts: usize,
    /// Seed used to derive every non-identity start.
    pub seed: u64,
    /// Maximum gradient-projection iterations per start.
    pub max_iter: usize,
    /// Projected-gradient convergence tolerance.
    pub tolerance: f64,
    /// Width of the non-monotone objective window.
    pub function_window: usize,
    /// Maximum step-halving attempts per iteration.
    pub max_line_search: usize,
    /// Relative objective tolerance used to group local-minimum basins.
    pub basin_tolerance: f64,
    /// Maximum coarse-grained CPU workers; zero uses available parallelism.
    pub max_threads: usize,
}

impl Default for RotationConfig {
    fn default() -> Self {
        Self {
            mode: RotationMode::Oblique,
            normalize: false,
            n_starts: 32,
            seed: 1,
            max_iter: 2_000,
            tolerance: 1e-5,
            function_window: 10,
            max_line_search: 20,
            basin_tolerance: 1e-8,
            max_threads: 0,
        }
    }
}

/// Metadata for one built-in criterion identifier.
#[derive(Clone, Copy, Debug)]
pub struct RotationCriterionInfo {
    /// Public method name.
    pub name: &'static str,
    /// Criterion family.
    pub family: &'static str,
    /// Whether orthogonal optimization is supported.
    pub orthogonal: bool,
    /// Whether oblique optimization is supported.
    pub oblique: bool,
    /// Whether a target matrix is required.
    pub requires_target: bool,
    /// Concise scientific description.
    pub description: &'static str,
}

/// Auditable result from deterministic global multi-start search.
#[derive(Clone, Debug)]
pub struct RotationSolution {
    /// Canonicalized pattern matrix, row major.
    pub pattern_matrix: Vec<f64>,
    /// Canonicalized structure matrix `pattern * factor_correlation`, row major.
    pub structure_matrix: Vec<f64>,
    /// Factor correlation matrix, row major.
    pub factor_correlation: Vec<f64>,
    /// Transformation matrix, row major.
    pub transform_matrix: Vec<f64>,
    /// Number of manifest variables.
    pub n_rows: usize,
    /// Number of factors.
    pub n_factors: usize,
    /// Criterion identifier.
    pub criterion_name: &'static str,
    /// Rotation manifold.
    pub mode: RotationMode,
    /// Best observed criterion value.
    pub criterion_value: f64,
    /// Final projected-gradient norm.
    pub gradient_norm: f64,
    /// Iterations used by the selected start.
    pub iterations: usize,
    /// Whether the selected start reached the gradient tolerance.
    pub converged: bool,
    /// Explicit selected-start termination reason.
    pub termination_reason: &'static str,
    /// Selected start index; zero is the identity start.
    pub best_start_index: usize,
    /// Total starts attempted.
    pub n_starts: usize,
    /// Starts that reached the gradient tolerance.
    pub converged_starts: usize,
    /// Starts in the best observed objective basin.
    pub basin_support: usize,
    /// Number of distinct observed objective basins.
    pub distinct_minima: usize,
    /// Start-level objective values in deterministic start order.
    pub start_values: Vec<f64>,
    /// Start-level convergence indicators.
    pub start_converged: Vec<bool>,
    /// Largest absolute off-diagonal factor correlation.
    pub max_factor_correlation: f64,
    /// Whether Kaiser normalization was applied.
    pub normalized: bool,
    /// Number of coarse CPU workers used.
    pub worker_count: usize,
    /// Numerical backend actually used.
    pub backend: &'static str,
}

/// Return the built-in criterion catalogue.
pub fn available_rotation_criteria() -> &'static [RotationCriterionInfo] {
    const CATALOGUE: &[RotationCriterionInfo] = &[
        RotationCriterionInfo { name: "quartimax", family: "orthomax", orthogonal: true, oblique: false, requires_target: false, description: "Orthogonal variable-complexity minimization." },
        RotationCriterionInfo { name: "varimax", family: "orthomax", orthogonal: true, oblique: false, requires_target: false, description: "Orthogonal factor-variance maximization." },
        RotationCriterionInfo { name: "orthomax", family: "orthomax", orthogonal: true, oblique: false, requires_target: false, description: "Continuous Orthomax gamma family." },
        RotationCriterionInfo { name: "crawford_ferguson", family: "crawford_ferguson", orthogonal: true, oblique: true, requires_target: false, description: "Continuous Crawford-Ferguson kappa family." },
        RotationCriterionInfo { name: "equamax", family: "crawford_ferguson", orthogonal: true, oblique: true, requires_target: false, description: "Crawford-Ferguson equamax special case." },
        RotationCriterionInfo { name: "parsimax", family: "crawford_ferguson", orthogonal: true, oblique: true, requires_target: false, description: "Crawford-Ferguson parsimax special case." },
        RotationCriterionInfo { name: "factor_parsimony", family: "crawford_ferguson", orthogonal: true, oblique: true, requires_target: false, description: "Crawford-Ferguson factor-parsimony endpoint." },
        RotationCriterionInfo { name: "oblimin", family: "oblimin", orthogonal: false, oblique: true, requires_target: false, description: "Continuous direct-oblimin gamma family." },
        RotationCriterionInfo { name: "quartimin", family: "oblimin", orthogonal: false, oblique: true, requires_target: false, description: "Direct-oblimin quartimin special case." },
        RotationCriterionInfo { name: "biquartimin", family: "oblimin", orthogonal: false, oblique: true, requires_target: false, description: "Direct-oblimin gamma=.5 special case." },
        RotationCriterionInfo { name: "covarimin", family: "oblimin", orthogonal: false, oblique: true, requires_target: false, description: "Direct-oblimin gamma=1 special case." },
        RotationCriterionInfo { name: "geomin", family: "geomin", orthogonal: true, oblique: true, requires_target: false, description: "Geometric-mean row-complexity criterion." },
        RotationCriterionInfo { name: "target", family: "target", orthogonal: true, oblique: true, requires_target: true, description: "Complete or NaN-partially specified target rotation." },
        RotationCriterionInfo { name: "pst", family: "target", orthogonal: true, oblique: true, requires_target: true, description: "Weighted partially specified target rotation." },
        RotationCriterionInfo { name: "entropy", family: "information", orthogonal: true, oblique: true, requires_target: false, description: "Minimum entropy criterion." },
        RotationCriterionInfo { name: "infomax", family: "information", orthogonal: true, oblique: true, requires_target: false, description: "Infomax information criterion." },
        RotationCriterionInfo { name: "mccammon", family: "information", orthogonal: true, oblique: false, requires_target: false, description: "McCammon minimum entropy-ratio criterion." },
        RotationCriterionInfo { name: "simplimax", family: "component_loss", orthogonal: false, oblique: true, requires_target: false, description: "Kiers simplimax component-loss criterion." },
        RotationCriterionInfo { name: "bifactor", family: "bifactor", orthogonal: true, oblique: true, requires_target: false, description: "Jennrich-Bentler biquartimin criterion." },
        RotationCriterionInfo { name: "bigeomin", family: "bifactor", orthogonal: true, oblique: true, requires_target: false, description: "Jennrich-Bentler bi-geomin criterion." },
        RotationCriterionInfo { name: "tandem_i", family: "tandem", orthogonal: true, oblique: false, requires_target: false, description: "Comrey tandem criterion I." },
        RotationCriterionInfo { name: "tandem_ii", family: "tandem", orthogonal: true, oblique: false, requires_target: false, description: "Comrey tandem criterion II." },
        RotationCriterionInfo { name: "oblimax", family: "oblimax", orthogonal: false, oblique: true, requires_target: false, description: "Scale-invariant oblimax criterion." },
        RotationCriterionInfo { name: "bentler", family: "invariant_simplicity", orthogonal: true, oblique: true, requires_target: false, description: "Bentler invariant pattern-simplicity criterion." },
        RotationCriterionInfo { name: "varimin", family: "anti_simple_structure", orthogonal: true, oblique: false, requires_target: false, description: "Orthogonal varimin complement of varimax." },
        RotationCriterionInfo { name: "lp_wls", family: "component_loss", orthogonal: true, oblique: true, requires_target: false, description: "Weighted L2 kernel for iterative Lp/FSS rotation." },
    ];
    CATALOGUE
}

/// Construct a built-in criterion from its public name and optional arguments.
#[allow(clippy::too_many_arguments)]
pub fn rotation_criterion_from_name(
    name: &str,
    rows: usize,
    factors: usize,
    kappa: Option<f64>,
    gamma: Option<f64>,
    delta: Option<f64>,
    simplimax_zeros: Option<usize>,
    target: Option<Vec<f64>>,
    weights: Option<Vec<f64>>,
) -> Result<RotationCriterion, String> {
    let normalized = name.trim().to_ascii_lowercase().replace('-', "_");
    let criterion = match normalized.as_str() {
        "quartimax" => RotationCriterion::Quartimax,
        "varimax" => RotationCriterion::Varimax,
        "varimin" => RotationCriterion::Varimin,
        "orthomax" => RotationCriterion::Orthomax { gamma: gamma.unwrap_or(1.0) },
        "crawford_ferguson" | "cf" => RotationCriterion::CrawfordFerguson { kappa: kappa.unwrap_or(0.0) },
        "equamax" => RotationCriterion::CrawfordFerguson { kappa: factors as f64 / (2.0 * rows as f64) },
        "parsimax" => RotationCriterion::CrawfordFerguson { kappa: (factors - 1) as f64 / (rows + factors - 2) as f64 },
        "factor_parsimony" => RotationCriterion::CrawfordFerguson { kappa: 1.0 },
        "oblimin" => RotationCriterion::Oblimin { gamma: gamma.unwrap_or(0.0) },
        "quartimin" => RotationCriterion::Oblimin { gamma: 0.0 },
        "biquartimin" => RotationCriterion::Oblimin { gamma: 0.5 },
        "covarimin" => RotationCriterion::Oblimin { gamma: 1.0 },
        "geomin" => RotationCriterion::Geomin { delta: delta.unwrap_or(0.01) },
        "target" => RotationCriterion::Target {
            target: target.ok_or_else(|| "target method requires target".to_string())?,
            weights: weights.unwrap_or_else(|| vec![1.0; rows * factors]),
        },
        "pst" | "partial_target" => RotationCriterion::Target {
            target: target.ok_or_else(|| "pst method requires target".to_string())?,
            weights: weights.ok_or_else(|| "pst method requires weights".to_string())?,
        },
        "entropy" => RotationCriterion::Entropy,
        "infomax" => RotationCriterion::Infomax,
        "mccammon" => RotationCriterion::McCammon,
        "simplimax" => RotationCriterion::Simplimax { zeros: simplimax_zeros.unwrap_or(rows) },
        "bifactor" | "biquartimin_bifactor" => RotationCriterion::Bifactor,
        "bigeomin" | "bi_geomin" => RotationCriterion::BiGeomin { delta: delta.unwrap_or(0.01) },
        "tandem_i" | "tandemi" => RotationCriterion::TandemI,
        "tandem_ii" | "tandemii" => RotationCriterion::TandemII,
        "oblimax" => RotationCriterion::Oblimax,
        "bentler" => RotationCriterion::Bentler,
        "lp_wls" => RotationCriterion::LpWls {
            weights: weights.ok_or_else(|| "lp_wls method requires weights".to_string())?,
        },
        _ => return Err(format!("unknown rotation criterion: {name}")),
    };
    criterion.validate(rows, factors)?;
    Ok(criterion)
}

/// Rotate an unrotated factor-loading matrix with deterministic multi-start GPA.
pub fn rotate_factor_loadings(
    loadings: &[f64],
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
    config: &RotationConfig,
) -> Result<RotationSolution, String> {
    validate_rotation_contract(loadings, rows, factors, criterion, config)?;
    let (working, row_scale) = normalize_rows(loadings, rows, factors, config.normalize)?;
    let starts = build_starts(factors, config.mode, config.n_starts, config.seed)?;
    let settings = OptimizerSettings {
        max_iter: config.max_iter,
        tolerance: config.tolerance,
        function_window: config.function_window,
        max_line_search: config.max_line_search,
    };
    let available = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1);
    let requested = if config.max_threads == 0 { available } else { config.max_threads };
    let workers = requested.max(1).min(starts.len());
    let chunk_size = starts.len().div_ceil(workers);
    let mut outcomes: Vec<Result<StartSolution, String>> = Vec::with_capacity(starts.len());
    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in starts.chunks(chunk_size) {
            let criterion_ref = criterion;
            let settings_ref = &settings;
            let working_ref = &working;
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|(index, transform)| {
                        optimize_start(
                            working_ref,
                            rows,
                            factors,
                            criterion_ref,
                            config.mode,
                            transform.clone(),
                            settings_ref,
                            *index,
                        )
                    })
                    .collect::<Vec<_>>()
            }));
        }
        for handle in handles {
            match handle.join() {
                Ok(mut result) => outcomes.append(&mut result),
                Err(_) => outcomes.push(Err("rotation worker panicked".into())),
            }
        }
    });
    let mut failures = Vec::new();
    let mut solutions = Vec::new();
    for outcome in outcomes {
        match outcome {
            Ok(solution) => solutions.push(solution),
            Err(error) => failures.push(error),
        }
    }
    if solutions.is_empty() {
        return Err(format!(
            "all rotation starts failed: {}",
            failures.first().cloned().unwrap_or_else(|| "unknown failure".into())
        ));
    }
    solutions.sort_by_key(|solution| solution.start_index);
    let any_converged = solutions.iter().any(|solution| solution.converged);
    let best_position = solutions
        .iter()
        .enumerate()
        .filter(|(_, solution)| !any_converged || solution.converged)
        .min_by(|(_, a), (_, b)| {
            a.criterion_value
                .partial_cmp(&b.criterion_value)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(index, _)| index)
        .ok_or_else(|| "no finite rotation solution was produced".to_string())?;
    let mut best = solutions[best_position].clone();
    canonicalize(&mut best, rows, factors, criterion);
    restore_rows(&mut best.pattern, &row_scale, rows, factors);
    let structure = matmul(
        &best.pattern,
        rows,
        factors,
        &best.factor_correlation,
        factors,
    );
    let start_values: Vec<f64> = solutions.iter().map(|s| s.criterion_value).collect();
    let start_converged: Vec<bool> = solutions.iter().map(|s| s.converged).collect();
    let best_value = best.criterion_value;
    let basin_support = start_values
        .iter()
        .filter(|value| objective_close(**value, best_value, config.basin_tolerance))
        .count();
    let distinct_minima = count_distinct_minima(&start_values, config.basin_tolerance);
    let max_factor_correlation = maximum_off_diagonal(&best.factor_correlation, factors);
    Ok(RotationSolution {
        pattern_matrix: best.pattern,
        structure_matrix: structure,
        factor_correlation: best.factor_correlation,
        transform_matrix: best.transform,
        n_rows: rows,
        n_factors: factors,
        criterion_name: criterion.name(),
        mode: config.mode,
        criterion_value: best.criterion_value,
        gradient_norm: best.gradient_norm,
        iterations: best.iterations,
        converged: best.converged,
        termination_reason: best.termination_reason,
        best_start_index: best.start_index,
        n_starts: config.n_starts,
        converged_starts: start_converged.iter().filter(|x| **x).count(),
        basin_support,
        distinct_minima,
        start_values,
        start_converged,
        max_factor_correlation,
        normalized: config.normalize,
        worker_count: workers,
        backend: "rust_cpu_coarse_multithreaded",
    })
}

fn validate_rotation_contract(
    loadings: &[f64],
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
    config: &RotationConfig,
) -> Result<(), String> {
    if rows < 2 || factors < 2 || rows < factors {
        return Err("rotation requires rows >= factors >= 2".into());
    }
    if loadings.len() != rows * factors || loadings.iter().any(|x| !x.is_finite()) {
        return Err("loadings must be a finite rows x factors matrix".into());
    }
    if !criterion.supports(config.mode) {
        return Err(format!(
            "{} does not support {} rotation",
            criterion.name(),
            config.mode.as_str()
        ));
    }
    criterion.validate(rows, factors)?;
    if config.n_starts == 0 || config.max_iter == 0 || config.function_window == 0 {
        return Err("n_starts, max_iter, and function_window must be positive".into());
    }
    if config.max_line_search == 0 {
        return Err("max_line_search must be positive".into());
    }
    if !config.tolerance.is_finite() || config.tolerance <= 0.0 {
        return Err("tolerance must be finite and positive".into());
    }
    if !config.basin_tolerance.is_finite() || config.basin_tolerance <= 0.0 {
        return Err("basin_tolerance must be finite and positive".into());
    }
    Ok(())
}

fn normalize_rows(
    loadings: &[f64],
    rows: usize,
    factors: usize,
    normalize: bool,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if !normalize {
        return Ok((loadings.to_vec(), vec![1.0; rows]));
    }
    let mut working = loadings.to_vec();
    let mut scale = vec![0.0; rows];
    for i in 0..rows {
        let ss: f64 = working[i * factors..(i + 1) * factors]
            .iter()
            .map(|x| x * x)
            .sum();
        if ss <= 1e-24 || !ss.is_finite() {
            return Err(format!("Kaiser normalization cannot scale zero row {i}"));
        }
        scale[i] = ss.sqrt();
        for j in 0..factors {
            working[i * factors + j] /= scale[i];
        }
    }
    Ok((working, scale))
}

fn restore_rows(pattern: &mut [f64], scale: &[f64], rows: usize, factors: usize) {
    for i in 0..rows {
        for j in 0..factors {
            pattern[i * factors + j] *= scale[i];
        }
    }
}

fn build_starts(
    factors: usize,
    mode: RotationMode,
    n_starts: usize,
    seed: u64,
) -> Result<Vec<(usize, Vec<f64>)>, String> {
    let mut starts = vec![(0, identity(factors))];
    for index in 1..n_starts {
        let derived = seed
            ^ (index as u64).wrapping_mul(0x9E3779B97F4A7C15)
            ^ 0xA0761D6478BD642F;
        let transform = match mode {
            RotationMode::Orthogonal => random_orthogonal(factors, derived)?,
            RotationMode::Oblique => random_oblique(factors, derived)?,
        };
        starts.push((index, transform));
    }
    Ok(starts)
}

fn canonicalize(
    solution: &mut StartSolution,
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
) {
    if criterion.has_labelled_columns() {
        return;
    }
    let first_permutable = usize::from(criterion.fixes_general_factor());
    let mut columns: Vec<usize> = (first_permutable..factors).collect();
    columns.sort_by(|a, b| {
        let key_a = column_key(&solution.pattern, rows, factors, *a);
        let key_b = column_key(&solution.pattern, rows, factors, *b);
        key_a
            .0
            .cmp(&key_b.0)
            .then_with(|| key_b.1.partial_cmp(&key_a.1).unwrap_or(Ordering::Equal))
            .then_with(|| a.cmp(b))
    });
    let mut order: Vec<usize> = (0..first_permutable).collect();
    order.extend(columns);
    let signs: Vec<f64> = order
        .iter()
        .map(|column| {
            let pivot = column_key(&solution.pattern, rows, factors, *column).0;
            if solution.pattern[pivot * factors + column] < 0.0 {
                -1.0
            } else {
                1.0
            }
        })
        .collect();
    let old_pattern = solution.pattern.clone();
    let old_transform = solution.transform.clone();
    let old_phi = solution.factor_correlation.clone();
    for i in 0..rows {
        for new_j in 0..factors {
            let old_j = order[new_j];
            solution.pattern[i * factors + new_j] =
                old_pattern[i * factors + old_j] * signs[new_j];
        }
    }
    for i in 0..factors {
        for new_j in 0..factors {
            let old_j = order[new_j];
            solution.transform[i * factors + new_j] =
                old_transform[i * factors + old_j] * signs[new_j];
        }
    }
    for new_i in 0..factors {
        for new_j in 0..factors {
            solution.factor_correlation[new_i * factors + new_j] = signs[new_i]
                * signs[new_j]
                * old_phi[order[new_i] * factors + order[new_j]];
        }
    }
}

fn column_key(pattern: &[f64], rows: usize, factors: usize, column: usize) -> (usize, f64) {
    let mut pivot = 0_usize;
    let mut largest = -1.0_f64;
    let mut sum_squares = 0.0;
    for i in 0..rows {
        let value = pattern[i * factors + column];
        let absolute = value.abs();
        if absolute > largest {
            largest = absolute;
            pivot = i;
        }
        sum_squares += value * value;
    }
    (pivot, sum_squares)
}

fn objective_close(value: f64, reference: f64, tolerance: f64) -> bool {
    (value - reference).abs() <= tolerance * (1.0 + reference.abs())
}

fn count_distinct_minima(values: &[f64], tolerance: f64) -> usize {
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mut count = 0_usize;
    let mut representative: Option<f64> = None;
    for value in ordered {
        if representative
            .map(|current| !objective_close(value, current, tolerance))
            .unwrap_or(true)
        {
            count += 1;
            representative = Some(value);
        }
    }
    count
}

fn maximum_off_diagonal(phi: &[f64], factors: usize) -> f64 {
    let mut maximum = 0.0_f64;
    for i in 0..factors {
        for j in 0..factors {
            if i != j {
                maximum = maximum.max(phi[i * factors + j].abs());
            }
        }
    }
    maximum
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixed() -> Vec<f64> {
        vec![0.72, 0.39, 0.65, 0.35, 0.60, 0.31, -0.31, 0.70, -0.28, 0.64, -0.25, 0.58]
    }

    #[test]
    fn catalogue_and_parser_cover_named_families() {
        assert!(available_rotation_criteria().len() >= 25);
        let names = [
            "quartimax", "varimax", "varimin", "orthomax", "cf", "equamax",
            "parsimax", "factor_parsimony", "oblimin", "quartimin", "biquartimin",
            "covarimin", "geomin", "entropy", "infomax", "mccammon", "simplimax",
            "bifactor", "bigeomin", "tandem_i", "tandem_ii", "oblimax", "bentler",
        ];
        for name in names {
            let factors = if matches!(name, "bifactor" | "bigeomin") { 3 } else { 2 };
            assert!(rotation_criterion_from_name(
                name, 6, factors, None, None, None, None, None, None
            )
            .is_ok(),
            "{name}");
        }
        assert!(rotation_criterion_from_name(
            "target", 4, 2, None, None, None, None, Some(vec![0.0; 8]), None
        )
        .is_ok());
        assert!(rotation_criterion_from_name(
            "pst", 4, 2, None, None, None, None, Some(vec![0.0; 8]), Some(vec![1.0; 8])
        )
        .is_ok());
        assert!(rotation_criterion_from_name(
            "lp_wls", 4, 2, None, None, None, None, None, Some(vec![1.0; 8])
        )
        .is_ok());
        assert!(rotation_criterion_from_name(
            "unknown", 4, 2, None, None, None, None, None, None
        )
        .is_err());
        assert!(rotation_criterion_from_name(
            "target", 4, 2, None, None, None, None, None, None
        )
        .is_err());
        assert!(rotation_criterion_from_name(
            "pst", 4, 2, None, None, None, None, Some(vec![0.0; 8]), None
        )
        .is_err());
    }

    #[test]
    fn multistart_rotation_is_deterministic_and_auditable() {
        let config = RotationConfig {
            mode: RotationMode::Orthogonal,
            normalize: true,
            n_starts: 8,
            seed: 42,
            max_iter: 500,
            tolerance: 1e-7,
            function_window: 10,
            max_line_search: 20,
            basin_tolerance: 1e-7,
            max_threads: 2,
        };
        let first = rotate_factor_loadings(&mixed(), 6, 2, &RotationCriterion::Varimax, &config)
            .unwrap();
        let second = rotate_factor_loadings(&mixed(), 6, 2, &RotationCriterion::Varimax, &config)
            .unwrap();
        assert_eq!(first.pattern_matrix, second.pattern_matrix);
        assert_eq!(first.start_values, second.start_values);
        assert_eq!(first.n_starts, 8);
        assert_eq!(first.worker_count, 2);
        assert_eq!(first.backend, "rust_cpu_coarse_multithreaded");
        assert_eq!(first.factor_correlation, identity(2));
        assert_eq!(first.structure_matrix, first.pattern_matrix);
        assert!(first.basin_support >= 1);
        assert!(first.distinct_minima >= 1);
        assert_eq!(first.mode.as_str(), "orthogonal");
    }

    #[test]
    fn oblique_and_labelled_target_outputs_preserve_contracts() {
        let config = RotationConfig {
            n_starts: 4,
            max_threads: 1,
            max_iter: 500,
            tolerance: 1e-7,
            ..RotationConfig::default()
        };
        let solution = rotate_factor_loadings(
            &mixed(),
            6,
            2,
            &RotationCriterion::Oblimin { gamma: 0.0 },
            &config,
        )
        .unwrap();
        assert_eq!(solution.pattern_matrix.len(), 12);
        assert_eq!(solution.structure_matrix.len(), 12);
        assert_eq!(solution.factor_correlation.len(), 4);
        assert!(solution.max_factor_correlation < 0.999);
        assert_eq!(solution.mode.as_str(), "oblique");

        let target = RotationCriterion::Target {
            target: vec![0.7, 0.0, 0.7, 0.0, 0.7, 0.0, 0.0, 0.7, 0.0, 0.7, 0.0, 0.7],
            weights: vec![1.0; 12],
        };
        let target_solution = rotate_factor_loadings(&mixed(), 6, 2, &target, &config).unwrap();
        assert_eq!(target_solution.criterion_name, "target");
    }

    #[test]
    fn invalid_rotation_contracts_fail_closed() {
        let criterion = RotationCriterion::Varimax;
        let mut config = RotationConfig { mode: RotationMode::Orthogonal, ..RotationConfig::default() };
        assert!(rotate_factor_loadings(&[0.0; 3], 2, 2, &criterion, &config).is_err());
        assert!(rotate_factor_loadings(&[0.0; 4], 1, 2, &criterion, &config).is_err());
        assert!(rotate_factor_loadings(&[f64::NAN; 4], 2, 2, &criterion, &config).is_err());
        config.mode = RotationMode::Oblique;
        assert!(rotate_factor_loadings(&[0.5, 0.2, 0.1, 0.6], 2, 2, &criterion, &config).is_err());
        config.mode = RotationMode::Orthogonal;
        config.n_starts = 0;
        assert!(rotate_factor_loadings(&[0.5, 0.2, 0.1, 0.6], 2, 2, &criterion, &config).is_err());
        config.n_starts = 1;
        config.max_line_search = 0;
        assert!(rotate_factor_loadings(&[0.5, 0.2, 0.1, 0.6], 2, 2, &criterion, &config).is_err());
        config.max_line_search = 1;
        config.tolerance = 0.0;
        assert!(rotate_factor_loadings(&[0.5, 0.2, 0.1, 0.6], 2, 2, &criterion, &config).is_err());
        config.tolerance = 1e-5;
        config.basin_tolerance = f64::NAN;
        assert!(rotate_factor_loadings(&[0.5, 0.2, 0.1, 0.6], 2, 2, &criterion, &config).is_err());
    }

    #[test]
    fn normalization_and_mode_parsing_cover_boundaries() {
        assert_eq!(RotationMode::parse("T"), Some(RotationMode::Orthogonal));
        assert_eq!(RotationMode::parse("Q"), Some(RotationMode::Oblique));
        assert_eq!(RotationMode::parse("bad"), None);
        let config = RotationConfig {
            mode: RotationMode::Orthogonal,
            normalize: true,
            n_starts: 1,
            ..RotationConfig::default()
        };
        assert!(rotate_factor_loadings(
            &[0.0, 0.0, 0.5, 0.2],
            2,
            2,
            &RotationCriterion::Varimax,
            &config
        )
        .is_err());
    }

    #[test]
    fn canonicalization_keeps_bifactor_general_column_first() {
        let mut solution = StartSolution {
            start_index: 0,
            pattern: vec![-0.8, 0.1, -0.7, -0.6, 0.8, 0.2, -0.7, 0.2, 0.9],
            transform: identity(3),
            factor_correlation: factor_correlation(&identity(3), 3),
            criterion_value: 0.0,
            gradient_norm: 0.0,
            iterations: 0,
            converged: true,
            termination_reason: "test",
        };
        canonicalize(&mut solution, 3, 3, &RotationCriterion::Bifactor);
        assert!(solution.pattern[0] > 0.0);
        assert!(objective_close(1.0, 1.0 + 1e-10, 1e-8));
        assert_eq!(count_distinct_minima(&[0.0, 0.0, 1.0], 1e-8), 2);
        assert_eq!(maximum_off_diagonal(&identity(3), 3), 0.0);
    }
}