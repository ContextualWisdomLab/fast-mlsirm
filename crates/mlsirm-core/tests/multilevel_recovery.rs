//! Rust-owned known-truth recovery evidence for crossed contextual effects.
//!
//! The design is crossed (school x neighbourhood) with weighted multiple
//! membership in the school classification. The metrics are scoped to the
//! centered MAP effects estimated by this crate; they are not uncertainty or
//! causal-effect evidence.
//!
//! Fox, J.-P., & Glas, C. A. W. (2001). Bayesian estimation of a multilevel
//! IRT model. *Psychometrika, 66*, 271-288.
//! https://doi.org/10.1007/BF02294839
//!
//! Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
//! multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
//! 103-124. https://doi.org/10.1177/1471082X0100100202

use mlsirm_core::multilevel::{estimate_crossed_person_effects, CrossedPersonEffectConfig};
use mlsirm_core::Device;

const N_ITEMS: usize = 28;
const N_SCHOOLS: usize = 4;
const N_NEIGHBOURHOODS: usize = 3;
const COPIES_PER_CELL: usize = 8;

struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / 9007199254740992.0)
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn recovery_fixture() -> (
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let schools = [-1.2, -0.4, 0.4, 1.2];
    let neighbourhoods = [-0.8, 0.0, 0.8];
    let mut responses = Vec::new();
    let mut row_offsets = vec![0];
    let mut context_indices = Vec::new();
    let mut weights = Vec::new();
    let mut rng = Lcg(0x5EED_2025);

    for (school, &school_effect) in schools.iter().enumerate() {
        let partner = (school + 1) % N_SCHOOLS;
        for (neighbourhood, &neighbourhood_effect) in neighbourhoods.iter().enumerate() {
            for copy in 0..COPIES_PER_CELL {
                if copy % 3 == 0 {
                    context_indices.extend([school, partner]);
                    weights.extend([0.70, 0.30]);
                } else {
                    context_indices.push(school);
                    weights.push(1.0);
                }
                context_indices.push(N_SCHOOLS + neighbourhood);
                weights.push(1.0);
                row_offsets.push(context_indices.len());

                let school_location = if copy % 3 == 0 {
                    0.70 * school_effect + 0.30 * schools[partner]
                } else {
                    school_effect
                };
                let location = school_location + neighbourhood_effect;
                for item in 0..N_ITEMS {
                    let intercept = -1.4 + 2.8 * item as f64 / (N_ITEMS - 1) as f64;
                    let probability = sigmoid(location + intercept);
                    responses.push(if rng.uniform() < probability {
                        1.0
                    } else {
                        0.0
                    });
                }
            }
        }
    }

    let intercepts = (0..N_ITEMS)
        .map(|item| -1.4 + 2.8 * item as f64 / (N_ITEMS - 1) as f64)
        .collect();
    let truth = vec![-1.2, -0.4, 0.4, 1.2, -0.8, 0.0, 0.8];
    (
        responses,
        row_offsets,
        context_indices,
        weights,
        intercepts,
        truth,
    )
}

fn recovery_metrics(estimated: &[f64], truth: &[f64]) -> (f64, f64, f64) {
    let errors: Vec<f64> = estimated
        .iter()
        .zip(truth)
        .map(|(estimate, expected)| estimate - expected)
        .collect();
    let bias = errors.iter().sum::<f64>() / errors.len() as f64;
    let mae = errors.iter().map(|error| error.abs()).sum::<f64>() / errors.len() as f64;
    let rmse = (errors.iter().map(|error| error * error).sum::<f64>() / errors.len() as f64).sqrt();
    (bias, mae, rmse)
}

#[test]
fn crossed_multiple_membership_map_recovers_centered_context_effects() {
    let (responses, row_offsets, context_indices, weights, intercepts, truth) = recovery_fixture();
    let estimate = estimate_crossed_person_effects(
        &responses,
        &row_offsets,
        &context_indices,
        &weights,
        &[1.0; N_ITEMS],
        &intercepts,
        &[],
        &[0, N_SCHOOLS, N_SCHOOLS + N_NEIGHBOURHOODS],
        N_SCHOOLS * N_NEIGHBOURHOODS * COPIES_PER_CELL,
        N_ITEMS,
        N_SCHOOLS + N_NEIGHBOURHOODS,
        CrossedPersonEffectConfig {
            prior_precision: 0.25,
            max_iter: 40,
            tol: 1e-8,
            worker_count: 4,
            device: Device::Cpu,
        },
    )
    .expect("known-truth recovery fixture must fit");

    assert!(
        estimate.converged,
        "termination: {}",
        estimate.termination_reason
    );
    let (bias, mae, rmse) = recovery_metrics(&estimate.effects, &truth);
    assert!(bias.abs() < 0.10, "centered-effect bias too high: {bias}");
    assert!(mae < 0.20, "centered-effect MAE too high: {mae}");
    assert!(rmse < 0.25, "centered-effect RMSE too high: {rmse}");
}

#[test]
fn crossed_estimator_rejects_nonunit_membership_total_within_classification() {
    let result = estimate_crossed_person_effects(
        &[1.0],
        &[0, 2],
        &[0, 2],
        &[0.5, 1.0],
        &[1.0],
        &[0.0],
        &[],
        &[0, 2, 4],
        1,
        1,
        4,
        CrossedPersonEffectConfig {
            prior_precision: 1.0,
            max_iter: 5,
            tol: 1e-8,
            worker_count: 1,
            device: Device::Cpu,
        },
    );

    assert_eq!(
        result.expect_err("non-unit per-classification membership must fail"),
        "membership weights must sum to one within every classification"
    );
}
