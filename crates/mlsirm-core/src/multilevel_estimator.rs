//! MAP estimator for crossed / multiple-membership person effects `u_h`.
//!
//! This module owns the first buyer-visible random-effect *estimator* for the
//! contextual term already evaluated by [`crate::multilevel::weighted_contextual_effect`].
//! Persons may belong to several units of one classification at once (weighted
//! multiple membership) and to several classifications at once (crossed /
//! multiple-classification designs). Ordinary one-hot nesting is the singleton
//! special case of the same sparse design.
//!
//! # Linear predictor
//!
//! For binary response `Y_pi` from person `p` and item `i`,
//!
//! ```text
//! eta_pi = a_i * (theta_p + sum_h w_ph * u_h) + b_i
//! ```
//!
//! with known item slopes `a_i > 0`, known item intercepts `b_i`, known
//! non-negative membership weights that already sum to one within each
//! classification (Browne, Goldstein, & Rasbash, 2001, eq. 1), and an optional
//! person-level offset `theta_p`. The offset is the time-flow compatibility
//! hook: a longitudinal layer may supply already-estimated occasion states.
//! This slice does **not** estimate OLS trends or AR coefficients.
//!
//! # Estimand and identification
//!
//! Fox and Glas (2001) place a level-2 Gaussian prior on the group effects of
//! a multilevel IRT model. This kernel is the matching MAP / ridge point
//! estimator of the flattened effects `u_h`, not their Gibbs sampler:
//!
//! ```text
//! u_h ~ N(0, sigma_u^2)     (independent, prior_precision = 1 / sigma_u^2)
//! ```
//!
//! The reported estimate is re-centered to sum to zero inside each
//! classification so recovered effects are deviations, matching the usual
//! multilevel IRT location constraint. Item intercepts absorb the global
//! location. A classification with fewer than two levels is rejected because
//! centering would leave a non-identified singleton.
//!
//! The estimator is **not** a claim of Fox & Glas MCMC, Jeon & Rabe-Hesketh
//! adaptive quadrature, variance-component ML, or causal contextual effects.
//!
//! # Compute
//!
//! The `O(n_persons * n_items)` Bernoulli score / information reduction is
//! multithreaded on CPU and, when a wgpu adapter is present, offloaded to an
//! f32 GPU kernel with an f64 CPU fallback. The sparse-design Newton system
//! (`n_effects <= 128`) stays on CPU. Results do not depend on `worker_count`.
//!
//! # References
//!
//! Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
//! multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
//! 103-124. <https://doi.org/10.1177/1471082X0100100202>
//!
//! Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
//! IRT model. *Psychometrika, 66*, 271-288.
//! <https://doi.org/10.1007/BF02294839>

use std::thread;

use crate::mmle::{log_sigmoid, sigmoid_stable};
use crate::multilevel::weighted_contextual_effect;
use crate::Device;

/// Hard cap on flattened context effects for the dense Newton system.
pub const MAX_CROSSED_EFFECTS: usize = 128;

/// Upper bound on Newton iterations accepted from a caller.
pub const MAX_CROSSED_ITER: usize = 10_000;

const MEMBERSHIP_WEIGHT_TOLERANCE: f64 = 1e-12;

type EstimatorResult<T> = Result<T, String>;

/// Configuration for the crossed / multiple-membership MAP estimator.
#[derive(Clone, Copy, Debug)]
pub struct CrossedPersonEffectConfig {
    /// Gaussian prior precision `1 / sigma_u^2` (Fox & Glas, 2001, level-2).
    pub prior_precision: f64,
    /// Maximum Newton / IRLS iterations.
    pub max_iter: usize,
    /// Absolute effect-step convergence tolerance.
    pub tol: f64,
    /// Deterministic CPU worker count (`>= 1`); does not change the result.
    pub worker_count: usize,
    /// CPU / GPU / auto device policy for the person-score reduction.
    pub device: Device,
}

impl Default for CrossedPersonEffectConfig {
    fn default() -> Self {
        Self {
            prior_precision: 1.0,
            max_iter: 50,
            tol: 1e-8,
            worker_count: 1,
            device: Device::Auto,
        }
    }
}

/// MAP estimate of crossed / multiple-membership person effects `u_h`.
#[derive(Clone, Debug, PartialEq)]
pub struct CrossedPersonEffectEstimate {
    /// Centered context effects, length `n_effects`.
    pub effects: Vec<f64>,
    /// Observed-data Bernoulli log-likelihood plus Gaussian prior penalty.
    pub loglik: f64,
    /// Newton iterations actually performed.
    pub n_iter: usize,
    /// Whether the last step satisfied `tol`.
    pub converged: bool,
    /// Whether the person-score reduction used the wgpu kernel.
    pub used_gpu: bool,
    /// Machine-readable termination status.
    pub termination_reason: String,
}

/// Estimate crossed / multiple-membership `u_h` by Gaussian-prior MAP.
///
/// `y` is row-major `n_persons * n_items`. Finite non-negative observed cells
/// must be exactly `0` or `1`; a cell is treated as missing when it is
/// non-finite or strictly negative (the established `NaN` / `-1` mask
/// contract). `item_slopes` and `item_intercepts` have length `n_items`.
/// `person_offsets` is either empty (treated as zeros) or length `n_persons`.
/// `classification_offsets` is a CSR pointer over the flattened effect table:
/// classification `d` occupies `classification_offsets[d]`
/// .. classification_offsets[d + 1]`.
#[allow(clippy::too_many_arguments)]
pub fn estimate_crossed_person_effects(
    y: &[f64],
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    person_offsets: &[f64],
    classification_offsets: &[usize],
    n_persons: usize,
    n_items: usize,
    n_effects: usize,
    config: CrossedPersonEffectConfig,
) -> EstimatorResult<CrossedPersonEffectEstimate> {
    validate_estimator_inputs(
        y,
        row_offsets,
        context_indices,
        weights,
        item_slopes,
        item_intercepts,
        person_offsets,
        classification_offsets,
        n_persons,
        n_items,
        n_effects,
        &config,
    )?;

    let mut effects = vec![0.0_f64; n_effects];
    let mut used_gpu = false;
    let mut last_step = f64::INFINITY;
    let mut n_iter = 0usize;

    for iteration in 1..=config.max_iter {
        n_iter = iteration;
        let locations = person_locations(
            row_offsets,
            context_indices,
            weights,
            &effects,
            person_offsets,
            n_persons,
            config.worker_count,
        )?;
        let (residual, information, gpu_hit) = person_scores(
            y,
            item_slopes,
            item_intercepts,
            &locations,
            n_persons,
            n_items,
            config.worker_count,
            config.device,
        )?;
        used_gpu = used_gpu || gpu_hit;
        let rhs = effect_score(
            row_offsets,
            context_indices,
            weights,
            &residual,
            &effects,
            config.prior_precision,
            n_effects,
        );
        let mut system = effect_system(
            row_offsets,
            context_indices,
            weights,
            &information,
            config.prior_precision,
            n_effects,
        );
        let delta = solve_dense_system(&mut system, &rhs, n_effects)?;
        last_step = delta.iter().fold(0.0_f64, |acc, &step| acc.max(step.abs()));
        for (effect, step) in effects.iter_mut().zip(delta.iter()) {
            *effect += *step;
        }
        if last_step < config.tol {
            break;
        }
    }
    center_classifications(&mut effects, classification_offsets);

    let locations = person_locations(
        row_offsets,
        context_indices,
        weights,
        &effects,
        person_offsets,
        n_persons,
        config.worker_count,
    )?;
    let loglik = bernoulli_map_loglik(
        y,
        item_slopes,
        item_intercepts,
        &locations,
        &effects,
        config.prior_precision,
        n_persons,
        n_items,
    );
    if !loglik.is_finite() {
        return Err("crossed person-effect log-likelihood must be finite".to_string());
    }
    let converged = last_step < config.tol;
    Ok(CrossedPersonEffectEstimate {
        effects,
        loglik,
        n_iter,
        converged,
        used_gpu,
        termination_reason: if converged {
            "converged".to_string()
        } else {
            "max_iter_reached".to_string()
        },
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_estimator_inputs(
    y: &[f64],
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    person_offsets: &[f64],
    classification_offsets: &[usize],
    n_persons: usize,
    n_items: usize,
    n_effects: usize,
    config: &CrossedPersonEffectConfig,
) -> EstimatorResult<()> {
    if n_persons < 1 || n_items < 1 || n_effects < 1 {
        return Err("n_persons, n_items, and n_effects must be at least one".to_string());
    }
    if n_effects > MAX_CROSSED_EFFECTS {
        return Err(format!(
            "n_effects exceeds the dense Newton cap of {MAX_CROSSED_EFFECTS}"
        ));
    }
    let expected = crate::checked_mul_usize(n_persons, n_items, "response matrix is too large")?;
    if y.len() != expected {
        return Err("y must have length n_persons * n_items".to_string());
    }
    for &response in y {
        if response.is_finite() && response >= 0.0 && response != 0.0 && response != 1.0 {
            return Err("binary responses must contain only 0 or 1 for observed cells".to_string());
        }
    }
    if item_slopes.len() != n_items || item_intercepts.len() != n_items {
        return Err("item_slopes and item_intercepts must have length n_items".to_string());
    }
    if !person_offsets.is_empty() && person_offsets.len() != n_persons {
        return Err("person_offsets must be empty or have length n_persons".to_string());
    }
    for &slope in item_slopes {
        if !slope.is_finite() || slope <= 0.0 {
            return Err("item_slopes must be finite and strictly positive".to_string());
        }
    }
    for &intercept in item_intercepts {
        if !intercept.is_finite() {
            return Err("item_intercepts must be finite".to_string());
        }
    }
    for &offset in person_offsets {
        if !offset.is_finite() {
            return Err("person_offsets must be finite".to_string());
        }
    }
    if classification_offsets.len() < 2 {
        return Err("classification_offsets must contain at least one classification".to_string());
    }
    if classification_offsets[0] != 0
        || *classification_offsets.last().expect("non-empty") != n_effects
        || classification_offsets
            .windows(2)
            .any(|window| window[1] <= window[0])
    {
        return Err(
            "classification_offsets must start at zero, increase strictly, and end at n_effects"
                .to_string(),
        );
    }
    for window in classification_offsets.windows(2) {
        if window[1] - window[0] < 2 {
            return Err("each classification must contain at least two context levels".to_string());
        }
    }
    if !config.prior_precision.is_finite() || config.prior_precision <= 0.0 {
        return Err("prior_precision must be finite and strictly positive".to_string());
    }
    if !(1..=MAX_CROSSED_ITER).contains(&config.max_iter) {
        return Err(format!("max_iter must be in 1..={MAX_CROSSED_ITER}"));
    }
    if !config.tol.is_finite() || config.tol <= 0.0 {
        return Err("tol must be finite and strictly positive".to_string());
    }
    if config.worker_count == 0 {
        return Err("worker_count must be at least one".to_string());
    }
    if row_offsets.len() != n_persons + 1 {
        return Err("row_offsets must have length n_persons + 1".to_string());
    }
    // Touch the public predictor boundary so malformed CSR / non-finite
    // weights / duplicate row indices fail with the established messages
    // before Newton iteration begins.
    let dummy = vec![0.0_f64; n_effects];
    weighted_contextual_effect(
        row_offsets,
        context_indices,
        weights,
        &dummy,
        config.worker_count,
    )?;
    validate_membership_weight_totals(
        row_offsets,
        context_indices,
        weights,
        classification_offsets,
        n_effects,
    )?;
    Ok(())
}

fn validate_membership_weight_totals(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    classification_offsets: &[usize],
    n_effects: usize,
) -> EstimatorResult<()> {
    let n_classifications = classification_offsets.len() - 1;
    let mut effect_classification = vec![0usize; n_effects];
    for (classification, window) in classification_offsets.windows(2).enumerate() {
        for slot in &mut effect_classification[window[0]..window[1]] {
            *slot = classification;
        }
    }

    let mut totals = vec![0.0_f64; n_classifications];
    let mut compensation = vec![0.0_f64; n_classifications];
    for window in row_offsets.windows(2) {
        totals.fill(0.0);
        compensation.fill(0.0);
        for edge in window[0]..window[1] {
            let classification = effect_classification[context_indices[edge]];
            let adjusted = weights[edge] - compensation[classification];
            let next = totals[classification] + adjusted;
            compensation[classification] = (next - totals[classification]) - adjusted;
            totals[classification] = next;
        }
        if totals
            .iter()
            .any(|total| (*total - 1.0).abs() > MEMBERSHIP_WEIGHT_TOLERANCE)
        {
            return Err(
                "membership weights must sum to one within every classification".to_string(),
            );
        }
    }
    Ok(())
}

fn person_locations(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    effects: &[f64],
    person_offsets: &[f64],
    n_persons: usize,
    worker_count: usize,
) -> EstimatorResult<Vec<f64>> {
    let mut locations =
        weighted_contextual_effect(row_offsets, context_indices, weights, effects, worker_count)?;
    if locations.len() != n_persons {
        return Err("contextual predictor length must match n_persons".to_string());
    }
    if !person_offsets.is_empty() {
        for (location, offset) in locations.iter_mut().zip(person_offsets.iter()) {
            *location += *offset;
        }
    }
    Ok(locations)
}

#[allow(clippy::too_many_arguments)]
fn person_scores(
    y: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    locations: &[f64],
    n_persons: usize,
    n_items: usize,
    worker_count: usize,
    device: Device,
) -> EstimatorResult<(Vec<f64>, Vec<f64>, bool)> {
    if device != Device::Cpu {
        if let Some((residual, information)) = try_person_scores_gpu(
            y,
            item_slopes,
            item_intercepts,
            locations,
            n_persons,
            n_items,
        ) {
            return Ok((residual, information, true));
        }
        if device == Device::Gpu {
            eprintln!(
                "fast-mlsirm: GPU crossed person-effect scores requested but no usable GPU adapter was found; falling back to the CPU implementation."
            );
        }
    }
    let (residual, information) = person_scores_cpu(
        y,
        item_slopes,
        item_intercepts,
        locations,
        n_persons,
        n_items,
        worker_count,
    );
    Ok((residual, information, false))
}

#[cfg(all(feature = "gpu", not(coverage)))]
fn try_person_scores_gpu(
    y: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    locations: &[f64],
    n_persons: usize,
    n_items: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    crate::gpu_multilevel::person_irt_scores_gpu(
        y,
        item_slopes,
        item_intercepts,
        locations,
        n_persons,
        n_items,
    )
}

#[cfg(any(not(feature = "gpu"), coverage))]
fn try_person_scores_gpu(
    _y: &[f64],
    _item_slopes: &[f64],
    _item_intercepts: &[f64],
    _locations: &[f64],
    _n_persons: usize,
    _n_items: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    None
}

fn person_scores_cpu(
    y: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    locations: &[f64],
    n_persons: usize,
    n_items: usize,
    worker_count: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut residual = vec![0.0_f64; n_persons];
    let mut information = vec![0.0_f64; n_persons];
    if n_persons == 0 {
        return (residual, information);
    }
    let worker_count = worker_count.min(n_persons);
    let chunk_size = n_persons.div_ceil(worker_count);
    thread::scope(|scope| {
        for (chunk_index, (residual_chunk, information_chunk)) in residual
            .chunks_mut(chunk_size)
            .zip(information.chunks_mut(chunk_size))
            .enumerate()
        {
            let start = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, (residual_slot, information_slot)) in residual_chunk
                    .iter_mut()
                    .zip(information_chunk.iter_mut())
                    .enumerate()
                {
                    let person = start + offset;
                    let (person_residual, person_information) = person_score_one(
                        &y[person * n_items..(person + 1) * n_items],
                        item_slopes,
                        item_intercepts,
                        locations[person],
                    );
                    *residual_slot = person_residual;
                    *information_slot = person_information;
                }
            });
        }
    });
    (residual, information)
}

fn person_score_one(
    row: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    location: f64,
) -> (f64, f64) {
    let mut residual = 0.0_f64;
    let mut information = 0.0_f64;
    for (item, &response) in row.iter().enumerate() {
        if !response_is_observed(response) {
            continue;
        }
        let slope = item_slopes[item];
        let probability = sigmoid_stable(slope * location + item_intercepts[item]);
        residual += slope * (response - probability);
        information += slope * slope * probability * (1.0 - probability);
    }
    (residual, information)
}

fn response_is_observed(response: f64) -> bool {
    response.is_finite() && response >= 0.0
}

fn effect_score(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    residual: &[f64],
    effects: &[f64],
    prior_precision: f64,
    n_effects: usize,
) -> Vec<f64> {
    let mut score = vec![0.0_f64; n_effects];
    for (person, window) in row_offsets.windows(2).enumerate() {
        for edge in window[0]..window[1] {
            score[context_indices[edge]] += weights[edge] * residual[person];
        }
    }
    for (slot, effect) in score.iter_mut().zip(effects.iter()) {
        *slot -= prior_precision * *effect;
    }
    score
}

fn effect_system(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    information: &[f64],
    prior_precision: f64,
    n_effects: usize,
) -> Vec<f64> {
    let mut system = vec![0.0_f64; n_effects * n_effects];
    for (person, window) in row_offsets.windows(2).enumerate() {
        let weight_p = information[person];
        if weight_p == 0.0 {
            continue;
        }
        for left in window[0]..window[1] {
            let left_index = context_indices[left];
            let left_weight = weights[left];
            for right in window[0]..window[1] {
                system[left_index * n_effects + context_indices[right]] +=
                    left_weight * weights[right] * weight_p;
            }
        }
    }
    for effect in 0..n_effects {
        system[effect * n_effects + effect] += prior_precision;
    }
    system
}

fn solve_dense_system(matrix: &mut [f64], rhs: &[f64], n: usize) -> EstimatorResult<Vec<f64>> {
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut augmented = vec![0.0_f64; n * (n + 1)];
    for row in 0..n {
        let dest = row * (n + 1);
        augmented[dest..dest + n].copy_from_slice(&matrix[row * n..(row + 1) * n]);
        augmented[dest + n] = rhs[row];
    }
    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = augmented[col * (n + 1) + col].abs();
        for row in (col + 1)..n {
            let candidate = augmented[row * (n + 1) + col].abs();
            if candidate > pivot_abs {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if pivot_abs < 1e-14 {
            return Err("crossed person-effect Newton system is singular".to_string());
        }
        if pivot_row != col {
            for slot in 0..=n {
                augmented.swap(col * (n + 1) + slot, pivot_row * (n + 1) + slot);
            }
        }
        let pivot = augmented[col * (n + 1) + col];
        for row in (col + 1)..n {
            let factor = augmented[row * (n + 1) + col] / pivot;
            for slot in col..=n {
                let value = augmented[col * (n + 1) + slot];
                augmented[row * (n + 1) + slot] -= factor * value;
            }
        }
    }
    let mut solution = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut value = augmented[row * (n + 1) + n];
        for col in (row + 1)..n {
            value -= augmented[row * (n + 1) + col] * solution[col];
        }
        let pivot = augmented[row * (n + 1) + row];
        if !pivot.is_finite() || pivot.abs() < 1e-14 {
            return Err("crossed person-effect Newton system is singular".to_string());
        }
        solution[row] = value / pivot;
        if !solution[row].is_finite() {
            return Err("crossed person-effect Newton step must be finite".to_string());
        }
    }
    Ok(solution)
}

fn center_classifications(effects: &mut [f64], classification_offsets: &[usize]) {
    for window in classification_offsets.windows(2) {
        let slice = &mut effects[window[0]..window[1]];
        let mean = slice.iter().sum::<f64>() / slice.len() as f64;
        for value in slice {
            *value -= mean;
        }
    }
}

fn bernoulli_map_loglik(
    y: &[f64],
    item_slopes: &[f64],
    item_intercepts: &[f64],
    locations: &[f64],
    effects: &[f64],
    prior_precision: f64,
    n_persons: usize,
    n_items: usize,
) -> f64 {
    let mut loglik = 0.0_f64;
    for person in 0..n_persons {
        let location = locations[person];
        for item in 0..n_items {
            let response = y[person * n_items + item];
            if !response_is_observed(response) {
                continue;
            }
            let eta = item_slopes[item] * location + item_intercepts[item];
            loglik += if response >= 0.5 {
                log_sigmoid(eta)
            } else {
                log_sigmoid(-eta)
            };
        }
    }
    loglik -= 0.5 * prior_precision * effects.iter().map(|value| value * value).sum::<f64>();
    loglik
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crossed_design() -> (Vec<usize>, Vec<usize>, Vec<f64>, Vec<usize>) {
        // Four persons, two classifications (school, neighborhood), each with
        // two levels. Person 1 has weighted multiple school membership.
        // Flattened effects: school_0, school_1, neigh_0, neigh_1.
        let row_offsets = vec![0, 2, 5, 7, 9];
        let context_indices = vec![0, 2, 0, 1, 3, 1, 2, 1, 3];
        let weights = vec![1.0, 1.0, 0.6, 0.4, 1.0, 1.0, 1.0, 1.0, 1.0];
        let classification_offsets = vec![0, 2, 4];
        (
            row_offsets,
            context_indices,
            weights,
            classification_offsets,
        )
    }

    fn simulate_responses(
        row_offsets: &[usize],
        context_indices: &[usize],
        weights: &[f64],
        true_effects: &[f64],
        intercepts: &[f64],
        n_persons: usize,
        n_items: usize,
        seed: u64,
    ) -> Vec<f64> {
        let locations =
            weighted_contextual_effect(row_offsets, context_indices, weights, true_effects, 1)
                .unwrap();
        let mut y = vec![0.0_f64; n_persons * n_items];
        let mut state = seed;
        for person in 0..n_persons {
            for item in 0..n_items {
                let probability = sigmoid_stable(locations[person] + intercepts[item]);
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let unit = ((state >> 11) as f64) / ((1u64 << 53) as f64);
                y[person * n_items + item] = if unit < probability { 1.0 } else { 0.0 };
            }
        }
        y
    }

    #[test]
    fn recovers_crossed_membership_effects_below_rmse_gate() {
        let (row_offsets, context_indices, weights, classification_offsets) = crossed_design();
        let true_effects = vec![-0.8, 0.8, -0.5, 0.5];
        let n_persons = 4;
        let n_items = 80;
        let intercepts: Vec<f64> = (0..n_items)
            .map(|item| -1.2 + 2.4 * (item as f64) / ((n_items - 1) as f64))
            .collect();
        let slopes = vec![1.0; n_items];
        // Replicate the four-person design many times so each context has a
        // large effective sample while preserving the crossed weights.
        let repeats = 24usize;
        let mut row_offsets_rep = vec![0usize];
        let mut context_indices_rep = Vec::new();
        let mut weights_rep = Vec::new();
        for _ in 0..repeats {
            for person in 0..n_persons {
                let start = row_offsets[person];
                let end = row_offsets[person + 1];
                context_indices_rep.extend_from_slice(&context_indices[start..end]);
                weights_rep.extend_from_slice(&weights[start..end]);
                row_offsets_rep.push(context_indices_rep.len());
            }
        }
        let n_rep = n_persons * repeats;
        let y = simulate_responses(
            &row_offsets_rep,
            &context_indices_rep,
            &weights_rep,
            &true_effects,
            &intercepts,
            n_rep,
            n_items,
            20260818,
        );
        let estimate = estimate_crossed_person_effects(
            &y,
            &row_offsets_rep,
            &context_indices_rep,
            &weights_rep,
            &slopes,
            &intercepts,
            &[],
            &classification_offsets,
            n_rep,
            n_items,
            4,
            CrossedPersonEffectConfig {
                prior_precision: 0.05,
                max_iter: 40,
                tol: 1e-8,
                worker_count: 4,
                device: Device::Cpu,
            },
        )
        .unwrap();
        assert!(estimate.converged, "{}", estimate.termination_reason);
        let mse: f64 = estimate
            .effects
            .iter()
            .zip(true_effects.iter())
            .map(|(hat, truth)| (hat - truth) * (hat - truth))
            .sum::<f64>()
            / 4.0;
        let rmse = mse.sqrt();
        assert!(
            rmse < 0.20,
            "RMSE {rmse} exceeded the crossed u_h recovery gate; hats={:?} truth={:?}",
            estimate.effects,
            true_effects
        );
    }

    #[test]
    fn worker_count_does_not_change_the_estimate() {
        let (row_offsets, context_indices, weights, classification_offsets) = crossed_design();
        let intercepts = vec![-0.5, 0.0, 0.5, 1.0];
        let slopes = vec![1.0; 4];
        let y = vec![
            1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0,
        ];
        let config = |worker_count| CrossedPersonEffectConfig {
            worker_count,
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        };
        let one = estimate_crossed_person_effects(
            &y,
            &row_offsets,
            &context_indices,
            &weights,
            &slopes,
            &intercepts,
            &[],
            &classification_offsets,
            4,
            4,
            4,
            config(1),
        )
        .unwrap();
        let four = estimate_crossed_person_effects(
            &y,
            &row_offsets,
            &context_indices,
            &weights,
            &slopes,
            &intercepts,
            &[],
            &classification_offsets,
            4,
            4,
            4,
            config(4),
        )
        .unwrap();
        assert_eq!(one.effects, four.effects);
        assert_eq!(one.loglik, four.loglik);
    }

    #[test]
    fn rejects_singleton_classification() {
        let error = estimate_crossed_person_effects(
            &[1.0],
            &[0, 1],
            &[0],
            &[1.0],
            &[1.0],
            &[0.0],
            &[],
            &[0, 1],
            1,
            1,
            1,
            CrossedPersonEffectConfig::default(),
        )
        .unwrap_err();
        assert!(error.contains("at least two context levels"));
    }

    #[test]
    fn rejects_nonbinary_observed_response() {
        let error = estimate_crossed_person_effects(
            &[0.5, 0.0],
            &[0, 1, 2],
            &[0, 1],
            &[1.0, 1.0],
            &[1.0],
            &[0.0],
            &[],
            &[0, 2],
            2,
            1,
            2,
            CrossedPersonEffectConfig::default(),
        )
        .unwrap_err();
        assert!(error.contains("binary responses"));
    }

    #[test]
    fn rejects_zero_worker_count() {
        let error = estimate_crossed_person_effects(
            &[1.0, 0.0, 1.0, 0.0],
            &[0, 1, 2],
            &[0, 1],
            &[1.0, 1.0],
            &[1.0, 1.0],
            &[0.0, 0.0],
            &[],
            &[0, 2],
            2,
            2,
            2,
            CrossedPersonEffectConfig {
                worker_count: 0,
                ..CrossedPersonEffectConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.contains("worker_count"));
    }

    #[test]
    fn person_offsets_shift_the_linear_predictor() {
        let (row_offsets, context_indices, weights, classification_offsets) = crossed_design();
        let intercepts = vec![0.0, 0.0];
        let slopes = vec![1.0, 1.0];
        let y = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let baseline = estimate_crossed_person_effects(
            &y,
            &row_offsets,
            &context_indices,
            &weights,
            &slopes,
            &intercepts,
            &[],
            &classification_offsets,
            4,
            2,
            4,
            CrossedPersonEffectConfig {
                device: Device::Cpu,
                ..CrossedPersonEffectConfig::default()
            },
        )
        .unwrap();
        let shifted = estimate_crossed_person_effects(
            &y,
            &row_offsets,
            &context_indices,
            &weights,
            &slopes,
            &intercepts,
            &[2.0, 2.0, 2.0, 2.0],
            &classification_offsets,
            4,
            2,
            4,
            CrossedPersonEffectConfig {
                device: Device::Cpu,
                ..CrossedPersonEffectConfig::default()
            },
        )
        .unwrap();
        assert_ne!(baseline.effects, shifted.effects);
    }
}
