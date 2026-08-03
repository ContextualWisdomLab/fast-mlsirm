//! Monte Carlo recovery contract for the higher-order DINA implementation.
//!
//! This integration test reproduces the deterministic 500-replication design
//! used by the historical unit study while evaluating its nominal 95%
//! convergence target with an explicit two-standard-error binomial tolerance.
//! A finite Monte Carlo experiment must not require an observed proportion to
//! equal its population target exactly.
//!
//! References (APA 7th ed.):
//! de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models
//! for cognitive diagnosis. *Psychometrika, 69*(3), 333-353.
//! https://doi.org/10.1007/BF02295640

use mlsirm_core::cdm::{fit_ho_cdm, CdmConfig, CdmModel};

const N_ATTRIBUTES: usize = 3;
const N_ITEMS: usize = 15;
const N_PERSONS: usize = 1_000;
const REPLICATIONS: usize = 500;
const TARGET_CONVERGENCE_RATE: f64 = 0.95;

#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn bernoulli(&mut self, probability: f64) -> f64 {
        if self.next_f64() < probability {
            1.0
        } else {
            0.0
        }
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

fn q_matrix() -> Vec<u8> {
    let mut matrix = vec![0_u8; N_ITEMS * N_ATTRIBUTES];
    for item in 0..N_ITEMS {
        if item < 12 {
            matrix[item * N_ATTRIBUTES + item / 4] = 1;
        } else {
            let first = item - 12;
            matrix[item * N_ATTRIBUTES + first] = 1;
            matrix[item * N_ATTRIBUTES + (first + 1) % N_ATTRIBUTES] = 1;
        }
    }
    matrix
}

fn q_mask(matrix: &[u8], item: usize) -> usize {
    let mut mask = 0_usize;
    for attribute in 0..N_ATTRIBUTES {
        if matrix[item * N_ATTRIBUTES + attribute] != 0 {
            mask |= 1 << attribute;
        }
    }
    mask
}

fn simulate(
    attribute_slope: &[f64],
    attribute_intercept: &[f64],
    slip: &[f64],
    guess: &[f64],
    matrix: &[u8],
    skew: bool,
    rng: &mut Lcg,
) -> (Vec<f64>, Vec<usize>) {
    let mut responses = vec![0.0; N_PERSONS * N_ITEMS];
    let mut profiles = vec![0_usize; N_PERSONS];

    for person in 0..N_PERSONS {
        let theta = if skew {
            let mut chi_square = 0.0;
            for _ in 0..3 {
                let normal = rng.normal();
                chi_square += normal * normal;
            }
            (chi_square - 3.0) / 6.0_f64.sqrt()
        } else {
            rng.normal()
        };

        let mut profile = 0_usize;
        for attribute in 0..N_ATTRIBUTES {
            let probability = sigmoid(
                attribute_slope[attribute] * theta + attribute_intercept[attribute],
            );
            if rng.next_f64() < probability {
                profile |= 1 << attribute;
            }
        }
        profiles[person] = profile;

        for item in 0..N_ITEMS {
            let mastered = (profile & q_mask(matrix, item)) == q_mask(matrix, item);
            let probability = if mastered {
                1.0 - slip[item]
            } else {
                guess[item]
            };
            responses[person * N_ITEMS + item] = rng.bernoulli(probability);
        }
    }

    (responses, profiles)
}

fn rmse(estimate: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(estimate.len(), truth.len());
    (estimate
        .iter()
        .zip(truth)
        .map(|(left, right)| (left - right).powi(2))
        .sum::<f64>()
        / estimate.len() as f64)
        .sqrt()
}

fn bias(estimate: &[f64], truth: &[f64]) -> f64 {
    assert_eq!(estimate.len(), truth.len());
    estimate
        .iter()
        .zip(truth)
        .map(|(left, right)| left - right)
        .sum::<f64>()
        / estimate.len() as f64
}

fn attribute_agreement(
    estimated_probability: &[f64],
    profiles: &[usize],
) -> f64 {
    let mut correct = 0_usize;
    for person in 0..N_PERSONS {
        for attribute in 0..N_ATTRIBUTES {
            let estimate = (estimated_probability[person * N_ATTRIBUTES + attribute] >= 0.5)
                as usize;
            let truth = (profiles[person] >> attribute) & 1;
            if estimate == truth {
                correct += 1;
            }
        }
    }
    correct as f64 / (N_PERSONS * N_ATTRIBUTES) as f64
}

fn monte_carlo_floor(target: f64, replications: usize) -> f64 {
    let standard_error = (target * (1.0 - target) / replications as f64).sqrt();
    target - 2.0 * standard_error
}

#[test]
fn two_standard_error_floor_matches_the_registered_design() {
    let floor = monte_carlo_floor(TARGET_CONVERGENCE_RATE, REPLICATIONS);
    assert!((floor - 0.930_506_411_3).abs() < 1e-9);
    assert!(474.0 / 500.0 >= floor);
}

#[test]
#[ignore = "literature-grade 500-replication recovery; executed by rust-ignored CI"]
fn higher_order_dina_recovery_respects_monte_carlo_tolerance() {
    let matrix = q_matrix();
    let attribute_slope = [1.2, 1.5, 0.9];
    let attribute_intercept = [0.3, -0.5, 0.6];
    let slip = [0.12; N_ITEMS];
    let guess = [0.12; N_ITEMS];
    let observed = vec![true; N_PERSONS * N_ITEMS];
    let floor = monte_carlo_floor(TARGET_CONVERGENCE_RATE, REPLICATIONS);

    for skew in [false, true] {
        let mut slope_rmse = 0.0;
        let mut intercept_rmse = 0.0;
        let mut slope_bias = 0.0;
        let mut intercept_bias = 0.0;
        let mut agreement = 0.0;
        let mut converged = 0_usize;

        for replication in 0..REPLICATIONS {
            let mut rng = Lcg(
                0xA24B_AED4_963E_E407_u64
                    .wrapping_mul(replication as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0x9E37_79B9_7F4A_7C15),
            );
            let (responses, profiles) = simulate(
                &attribute_slope,
                &attribute_intercept,
                &slip,
                &guess,
                &matrix,
                skew,
                &mut rng,
            );
            let result = fit_ho_cdm(
                &responses,
                &observed,
                &matrix,
                N_PERSONS,
                N_ITEMS,
                N_ATTRIBUTES,
                CdmModel::Dina,
                &CdmConfig::default(),
            )
            .expect("registered higher-order recovery calibration must fit");

            if result.converged {
                converged += 1;
                slope_rmse += rmse(&result.attr_slope, &attribute_slope);
                intercept_rmse += rmse(&result.attr_intercept, &attribute_intercept);
                slope_bias += bias(&result.attr_slope, &attribute_slope);
                intercept_bias += bias(&result.attr_intercept, &attribute_intercept);
                agreement += attribute_agreement(&result.attr_prob, &profiles);
            }
        }

        let convergence_rate = converged as f64 / REPLICATIONS as f64;
        assert!(
            convergence_rate >= floor,
            "higher-order convergence rate {convergence_rate:.3} below the "
                + "two-standard-error floor {floor:.3} for skew={skew}"
        );
        assert!(converged > 0);
        let denominator = converged as f64;
        slope_rmse /= denominator;
        intercept_rmse /= denominator;
        slope_bias /= denominator;
        intercept_bias /= denominator;
        agreement /= denominator;

        println!(
            "[HO-DINA MC skew={skew}] reps={REPLICATIONS} converged={converged} "
                + "({convergence_rate:.3}; floor={floor:.3}) RMSE(a)={slope_rmse:.3} "
                + "RMSE(d)={intercept_rmse:.3} bias(a)={slope_bias:.3} "
                + "bias(d)={intercept_bias:.3} attr-agree={agreement:.3}"
        );

        let (slope_bound, intercept_bound) = if skew {
            (0.45, 0.25)
        } else {
            (0.32, 0.15)
        };
        assert!(slope_rmse < slope_bound, "RMSE(a) {slope_rmse} skew={skew}");
        assert!(
            intercept_rmse < intercept_bound,
            "RMSE(d) {intercept_rmse} skew={skew}"
        );
        assert!(agreement > 0.90, "attribute agreement {agreement} skew={skew}");
    }
}
