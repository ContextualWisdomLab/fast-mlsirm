//! Orthogonal and oblique gradient-projection optimizers.

use super::criteria::{CriterionEvaluation, RotationCriterion};
use super::matrix::{
    cayley_step, crossprod, dot, factor_correlation, matmul, norm, normalize_columns,
    oblique_pattern, orthogonality_error, orthonormalize_columns, transpose,
};
use super::RotationMode;

/// Numerical settings shared by every deterministic start.
#[derive(Clone, Debug)]
pub(crate) struct OptimizerSettings {
    pub(crate) max_iter: usize,
    pub(crate) tolerance: f64,
    pub(crate) function_window: usize,
    pub(crate) max_line_search: usize,
}

/// Result from a single starting transform.
#[derive(Clone, Debug)]
pub(crate) struct StartSolution {
    pub(crate) start_index: usize,
    pub(crate) pattern: Vec<f64>,
    pub(crate) transform: Vec<f64>,
    pub(crate) factor_correlation: Vec<f64>,
    pub(crate) criterion_value: f64,
    pub(crate) gradient_norm: f64,
    pub(crate) iterations: usize,
    pub(crate) converged: bool,
    pub(crate) termination_reason: &'static str,
}

/// Optimize one start on the requested manifold.
pub(crate) fn optimize_start(
    unrotated: &[f64],
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
    mode: RotationMode,
    initial_transform: Vec<f64>,
    settings: &OptimizerSettings,
    start_index: usize,
) -> Result<StartSolution, String> {
    match mode {
        RotationMode::Orthogonal => optimize_orthogonal(
            unrotated,
            rows,
            factors,
            criterion,
            initial_transform,
            settings,
            start_index,
        ),
        RotationMode::Oblique => optimize_oblique(
            unrotated,
            rows,
            factors,
            criterion,
            initial_transform,
            settings,
            start_index,
        ),
    }
}

fn orthogonal_gradient(
    unrotated: &[f64],
    rows: usize,
    factors: usize,
    transform: &[f64],
    evaluation: &CriterionEvaluation,
) -> Vec<f64> {
    let gradient = crossprod(
        unrotated,
        &evaluation.gradient,
        rows,
        factors,
        factors,
    );
    let gram = crossprod(transform, &gradient, factors, factors, factors);
    let gram_t = transpose(&gram, factors, factors);
    let symmetric: Vec<f64> = gram
        .iter()
        .zip(gram_t)
        .map(|(a, b)| 0.5 * (a + b))
        .collect();
    let normal = matmul(transform, factors, factors, &symmetric, factors);
    gradient
        .iter()
        .zip(normal)
        .map(|(a, b)| a - b)
        .collect()
}

fn oblique_gradient(
    pattern: &[f64],
    inverse_transform: &[f64],
    rows: usize,
    factors: usize,
    transform: &[f64],
    evaluation: &CriterionEvaluation,
) -> Vec<f64> {
    let pattern_gradient = crossprod(
        pattern,
        &evaluation.gradient,
        rows,
        factors,
        factors,
    );
    let product = matmul(
        &pattern_gradient,
        factors,
        factors,
        inverse_transform,
        factors,
    );
    let mut gradient = transpose(&product, factors, factors);
    for value in &mut gradient {
        *value = -*value;
    }
    let mut column_inner = vec![0.0; factors];
    for i in 0..factors {
        for j in 0..factors {
            column_inner[j] += transform[i * factors + j] * gradient[i * factors + j];
        }
    }
    for i in 0..factors {
        for j in 0..factors {
            gradient[i * factors + j] -=
                transform[i * factors + j] * column_inner[j];
        }
    }
    gradient
}

fn bb_step(
    transform: &[f64],
    previous_transform: Option<&[f64]>,
    projected: &[f64],
    previous_projected: Option<&[f64]>,
    current_step: f64,
) -> f64 {
    match (previous_transform, previous_projected) {
        (Some(old_t), Some(old_g)) => {
            let delta_t: Vec<f64> = transform.iter().zip(old_t).map(|(a, b)| a - b).collect();
            let delta_g: Vec<f64> = projected.iter().zip(old_g).map(|(a, b)| a - b).collect();
            let denominator = dot(&delta_t, &delta_g).abs();
            if denominator > 1e-20 {
                (dot(&delta_t, &delta_t) / denominator).clamp(1e-10, 20.0)
            } else {
                (2.0 * current_step).min(20.0)
            }
        }
        _ => (2.0 * current_step).min(20.0),
    }
}

fn window_max(history: &[f64], width: usize) -> f64 {
    history[history.len().saturating_sub(width)..]
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

fn optimize_orthogonal(
    unrotated: &[f64],
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
    mut transform: Vec<f64>,
    settings: &OptimizerSettings,
    start_index: usize,
) -> Result<StartSolution, String> {
    if orthogonality_error(&transform, factors) > 1e-8 {
        orthonormalize_columns(&mut transform, factors, factors)?;
    }
    let mut pattern = matmul(unrotated, rows, factors, &transform, factors);
    let mut evaluation = criterion.evaluate(&pattern, rows, factors)?;
    let mut projected = orthogonal_gradient(
        unrotated,
        rows,
        factors,
        &transform,
        &evaluation,
    );
    let mut gradient_norm = norm(&projected);
    let mut step = 1.0;
    let mut history = vec![evaluation.value];
    let mut previous_transform: Option<Vec<f64>> = None;
    let mut previous_projected: Option<Vec<f64>> = None;
    let mut best = (
        evaluation.value,
        transform.clone(),
        pattern.clone(),
        gradient_norm,
        0_usize,
    );
    let mut termination = "maximum_iterations";
    let mut converged = false;
    let mut iterations = 0_usize;

    for iteration in 0..settings.max_iter {
        iterations = iteration;
        if gradient_norm < settings.tolerance {
            converged = true;
            termination = "projected_gradient_tolerance";
            break;
        }
        step = bb_step(
            &transform,
            previous_transform.as_deref(),
            &projected,
            previous_projected.as_deref(),
            step,
        );
        let target_value = window_max(&history, settings.function_window);
        let armijo = 0.5 * gradient_norm * gradient_norm;
        let mut trial_step = step;
        let mut accepted: Option<(Vec<f64>, Vec<f64>, CriterionEvaluation)> = None;
        for _ in 0..settings.max_line_search {
            if let Ok(mut candidate_transform) =
                cayley_step(&transform, &projected, factors, trial_step)
            {
                if orthogonality_error(&candidate_transform, factors) > 1e-10 {
                    orthonormalize_columns(&mut candidate_transform, factors, factors)?;
                }
                let candidate_pattern =
                    matmul(unrotated, rows, factors, &candidate_transform, factors);
                if let Ok(candidate_evaluation) =
                    criterion.evaluate(&candidate_pattern, rows, factors)
                {
                    if target_value - candidate_evaluation.value > armijo * trial_step {
                        accepted = Some((
                            candidate_transform,
                            candidate_pattern,
                            candidate_evaluation,
                        ));
                        break;
                    }
                }
            }
            trial_step *= 0.5;
        }
        let Some((candidate_transform, candidate_pattern, candidate_evaluation)) = accepted else {
            termination = "line_search_stalled";
            break;
        };
        previous_transform = Some(transform);
        previous_projected = Some(projected);
        transform = candidate_transform;
        pattern = candidate_pattern;
        evaluation = candidate_evaluation;
        projected = orthogonal_gradient(
            unrotated,
            rows,
            factors,
            &transform,
            &evaluation,
        );
        gradient_norm = norm(&projected);
        step = trial_step;
        history.push(evaluation.value);
        if evaluation.value < best.0 {
            best = (
                evaluation.value,
                transform.clone(),
                pattern.clone(),
                gradient_norm,
                iteration + 1,
            );
        }
        iterations = iteration + 1;
    }
    if !converged && gradient_norm < settings.tolerance {
        converged = true;
        termination = "projected_gradient_tolerance";
    }
    if !converged && best.0 < evaluation.value {
        evaluation = criterion.evaluate(&best.2, rows, factors)?;
        transform = best.1;
        pattern = best.2;
        projected = orthogonal_gradient(
            unrotated,
            rows,
            factors,
            &transform,
            &evaluation,
        );
        gradient_norm = norm(&projected);
        iterations = best.4;
    }
    Ok(StartSolution {
        start_index,
        pattern,
        transform,
        factor_correlation: super::matrix::identity(factors),
        criterion_value: evaluation.value,
        gradient_norm,
        iterations,
        converged,
        termination_reason: termination,
    })
}

fn optimize_oblique(
    unrotated: &[f64],
    rows: usize,
    factors: usize,
    criterion: &RotationCriterion,
    mut transform: Vec<f64>,
    settings: &OptimizerSettings,
    start_index: usize,
) -> Result<StartSolution, String> {
    normalize_columns(&mut transform, factors, factors)?;
    let (mut pattern, mut inverse_transform) =
        oblique_pattern(unrotated, rows, factors, &transform)?;
    let mut evaluation = criterion.evaluate(&pattern, rows, factors)?;
    let mut projected = oblique_gradient(
        &pattern,
        &inverse_transform,
        rows,
        factors,
        &transform,
        &evaluation,
    );
    let mut gradient_norm = norm(&projected);
    let mut step = 1.0;
    let mut history = vec![evaluation.value];
    let mut previous_transform: Option<Vec<f64>> = None;
    let mut previous_projected: Option<Vec<f64>> = None;
    let mut best = (
        evaluation.value,
        transform.clone(),
        pattern.clone(),
        gradient_norm,
        0_usize,
    );
    let mut termination = "maximum_iterations";
    let mut converged = false;
    let mut iterations = 0_usize;

    for iteration in 0..settings.max_iter {
        iterations = iteration;
        if gradient_norm < settings.tolerance {
            converged = true;
            termination = "projected_gradient_tolerance";
            break;
        }
        step = bb_step(
            &transform,
            previous_transform.as_deref(),
            &projected,
            previous_projected.as_deref(),
            step,
        );
        let target_value = window_max(&history, settings.function_window);
        let armijo = 0.5 * gradient_norm * gradient_norm;
        let mut trial_step = step;
        let mut accepted: Option<(Vec<f64>, Vec<f64>, Vec<f64>, CriterionEvaluation)> = None;
        for _ in 0..settings.max_line_search {
            let mut candidate_transform: Vec<f64> = transform
                .iter()
                .zip(&projected)
                .map(|(t, g)| t - trial_step * g)
                .collect();
            if normalize_columns(&mut candidate_transform, factors, factors).is_ok() {
                if let Ok((candidate_pattern, candidate_inverse)) =
                    oblique_pattern(unrotated, rows, factors, &candidate_transform)
                {
                    if let Ok(candidate_evaluation) =
                        criterion.evaluate(&candidate_pattern, rows, factors)
                    {
                        if target_value - candidate_evaluation.value > armijo * trial_step {
                            accepted = Some((
                                candidate_transform,
                                candidate_pattern,
                                candidate_inverse,
                                candidate_evaluation,
                            ));
                            break;
                        }
                    }
                }
            }
            trial_step *= 0.5;
        }
        let Some((candidate_transform, candidate_pattern, candidate_inverse, candidate_evaluation)) =
            accepted
        else {
            termination = "line_search_stalled";
            break;
        };
        previous_transform = Some(transform);
        previous_projected = Some(projected);
        transform = candidate_transform;
        pattern = candidate_pattern;
        inverse_transform = candidate_inverse;
        evaluation = candidate_evaluation;
        projected = oblique_gradient(
            &pattern,
            &inverse_transform,
            rows,
            factors,
            &transform,
            &evaluation,
        );
        gradient_norm = norm(&projected);
        step = trial_step;
        history.push(evaluation.value);
        if evaluation.value < best.0 {
            best = (
                evaluation.value,
                transform.clone(),
                pattern.clone(),
                gradient_norm,
                iteration + 1,
            );
        }
        iterations = iteration + 1;
    }
    if !converged && gradient_norm < settings.tolerance {
        converged = true;
        termination = "projected_gradient_tolerance";
    }
    if !converged && best.0 < evaluation.value {
        transform = best.1;
        pattern = best.2;
        let (_, inverse) = oblique_pattern(unrotated, rows, factors, &transform)?;
        evaluation = criterion.evaluate(&pattern, rows, factors)?;
        projected = oblique_gradient(
            &pattern,
            &inverse,
            rows,
            factors,
            &transform,
            &evaluation,
        );
        gradient_norm = norm(&projected);
        iterations = best.4;
    }
    Ok(StartSolution {
        start_index,
        pattern,
        factor_correlation: factor_correlation(&transform, factors),
        transform,
        criterion_value: evaluation.value,
        gradient_norm,
        iterations,
        converged,
        termination_reason: termination,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::matrix::{identity, random_oblique, random_orthogonal};

    fn mixed_loadings() -> Vec<f64> {
        let simple = vec![
            0.82, 0.05, 0.76, 0.08, 0.69, 0.12, 0.07, 0.80, 0.10, 0.72, 0.14, 0.66,
        ];
        let angle = 0.58_f64;
        let rotation = vec![angle.cos(), -angle.sin(), angle.sin(), angle.cos()];
        matmul(&simple, 6, 2, &rotation, 2)
    }

    fn settings() -> OptimizerSettings {
        OptimizerSettings {
            max_iter: 500,
            tolerance: 1e-7,
            function_window: 10,
            max_line_search: 20,
        }
    }

    #[test]
    fn orthogonal_solver_reduces_varimax_objective() {
        let a = mixed_loadings();
        let criterion = RotationCriterion::Varimax;
        let initial = criterion.evaluate(&a, 6, 2).unwrap().value;
        let solution = optimize_start(
            &a,
            6,
            2,
            &criterion,
            RotationMode::Orthogonal,
            identity(2),
            &settings(),
            0,
        )
        .unwrap();
        assert!(solution.criterion_value < initial);
        assert!(solution.gradient_norm < 1e-5);
        assert_eq!(solution.factor_correlation, identity(2));
        assert_eq!(solution.start_index, 0);
        assert!(solution.iterations > 0);
    }

    #[test]
    fn oblique_solver_reduces_quartimin_objective() {
        let a = mixed_loadings();
        let criterion = RotationCriterion::Oblimin { gamma: 0.0 };
        let initial = criterion.evaluate(&a, 6, 2).unwrap().value;
        let solution = optimize_start(
            &a,
            6,
            2,
            &criterion,
            RotationMode::Oblique,
            random_oblique(2, 17).unwrap(),
            &settings(),
            1,
        )
        .unwrap();
        assert!(solution.criterion_value < initial);
        assert!(solution.factor_correlation[1].abs() < 0.999);
        assert_eq!(solution.transform.len(), 4);
    }

    #[test]
    fn random_start_and_stalled_paths_are_represented() {
        let a = mixed_loadings();
        let criterion = RotationCriterion::Varimax;
        let strict = OptimizerSettings {
            max_iter: 1,
            tolerance: 1e-30,
            function_window: 1,
            max_line_search: 1,
        };
        let solution = optimize_start(
            &a,
            6,
            2,
            &criterion,
            RotationMode::Orthogonal,
            random_orthogonal(2, 99).unwrap(),
            &strict,
            2,
        )
        .unwrap();
        assert!(!solution.converged);
        assert!(matches!(
            solution.termination_reason,
            "maximum_iterations" | "line_search_stalled"
        ));
    }
}