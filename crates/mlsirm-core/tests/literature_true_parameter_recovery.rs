//! Literature-traceable true-parameter recovery experiments executed entirely
//! in Rust.
//!
//! The generating equation is Kang and Jeon (2025, Eq. 3) under their
//! between-item simple-structure restriction:
//!
//! `eta_pi = exp(alpha_i) * theta_p,d(i) + b_i
//!           - exp(tau) * ||xi_p - zeta_i||`.
//!
//! The deterministic simulation cell uses `P=500`, `D=2`, `I_d=8`, `K=2`,
//! `rho=.30`, and `gamma=1.5`, all values from the paper's recovery design.
//! Molenaar and Jeon (2026) motivate the regularized point-estimation recovery
//! checks. See `docs/papers/true-parameter-recovery-study.md` for complete
//! equation, identification, metric, and citation traceability.

use mlsirm_core::marginal::{
    fit_marginal, MarginalConfig, MarginalResult, PopulationSpec,
};
use mlsirm_core::{Device, ModelConfig, ModelType, PenaltyConfig};

const N_DIMS: usize = 2;
const ITEMS_PER_DIM: usize = 8;
const N_ITEMS: usize = N_DIMS * ITEMS_PER_DIM;
const LATENT_DIM: usize = 2;
const RHO: f64 = 0.30;
const GAMMA: f64 = 1.50;
const EPS_DISTANCE: f64 = 1e-8;

/// Small deterministic generator used only by this experiment.
///
/// The LCG and Box-Muller transform avoid adding a random-number dependency to
/// the statistical core while keeping the generated cell byte-for-byte stable.
#[derive(Clone, Copy)]
struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1_u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for upper in (1..values.len()).rev() {
            let index = ((upper + 1) as f64 * self.next_f64()).floor() as usize;
            values.swap(upper, index.min(upper));
        }
    }
}

#[derive(Clone)]
struct StudyCell {
    n_persons: usize,
    y: Vec<f64>,
    observed: Vec<bool>,
    factor_id: Vec<usize>,
    a_true: Vec<f64>,
    b_true: Vec<f64>,
    theta_true: Vec<f64>,
    zeta_true: Vec<f64>,
}

#[derive(Clone, Copy, Debug)]
struct RecoveryMetrics {
    a_corr: f64,
    a_rmse: f64,
    adjusted_b_corr: f64,
    adjusted_b_rmse: f64,
    theta_corr: f64,
    zeta_distance_corr: f64,
    gamma: f64,
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    assert!(count >= 2);
    let step = (end - start) / (count - 1) as f64;
    (0..count).map(|index| start + step * index as f64).collect()
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp_value = value.exp();
        exp_value / (1.0 + exp_value)
    }
}

fn simulate_paper_cell(n_persons: usize, seed: u64) -> StudyCell {
    assert!(n_persons > 0);
    let mut rng = Lcg(seed);
    let factor_id: Vec<usize> = (0..N_ITEMS)
        .map(|item| item / ITEMS_PER_DIM)
        .collect();
    let a_true = linspace(0.5, 2.5, N_ITEMS);
    let mut b_true = linspace(0.0, 5.0, N_ITEMS);
    rng.shuffle(&mut b_true);
    let zeta_true: Vec<f64> = (0..N_ITEMS * LATENT_DIM)
        .map(|_| rng.normal())
        .collect();

    let mut theta_true = vec![0.0; n_persons * N_DIMS];
    let mut xi_true = vec![0.0; n_persons * LATENT_DIM];
    for person in 0..n_persons {
        let common = rng.normal();
        theta_true[person * N_DIMS] = common;
        theta_true[person * N_DIMS + 1] =
            RHO * common + (1.0 - RHO * RHO).sqrt() * rng.normal();
        for axis in 0..LATENT_DIM {
            xi_true[person * LATENT_DIM + axis] = rng.normal();
        }
    }

    let mut y = vec![0.0; n_persons * N_ITEMS];
    for person in 0..n_persons {
        for item in 0..N_ITEMS {
            let mut squared_distance = EPS_DISTANCE;
            for axis in 0..LATENT_DIM {
                let difference = xi_true[person * LATENT_DIM + axis]
                    - zeta_true[item * LATENT_DIM + axis];
                squared_distance += difference * difference;
            }
            let dimension = factor_id[item];
            let eta = a_true[item] * theta_true[person * N_DIMS + dimension]
                + b_true[item]
                - GAMMA * squared_distance.sqrt();
            y[person * N_ITEMS + item] =
                if rng.next_f64() < sigmoid(eta) { 1.0 } else { 0.0 };
        }
    }

    StudyCell {
        n_persons,
        y,
        observed: vec![true; n_persons * N_ITEMS],
        factor_id,
        a_true,
        b_true,
        theta_true,
        zeta_true,
    }
}

fn fit_cell(cell: &StudyCell, device: Device, max_iter: usize) -> MarginalResult {
    let model = ModelConfig {
        n_persons: cell.n_persons,
        n_items: N_ITEMS,
        n_dims: N_DIMS,
        latent_dim: LATENT_DIM,
        model_type: ModelType::Mls2plm,
        eps_distance: EPS_DISTANCE,
    };
    let estimator = MarginalConfig {
        q_theta: 15,
        q_xi: 7,
        q_u: 11,
        max_iter,
        tol: 1e-3,
        m_steps: 4,
        init_zeta_radius: 0.5,
        init_sigma_u: 0.3,
        ..MarginalConfig::default()
    };
    fit_marginal(
        &cell.y,
        &cell.observed,
        &cell.factor_id,
        &model,
        &PopulationSpec::Single,
        &estimator,
        &PenaltyConfig::lsirm_prior(),
        device,
    )
    .expect("paper-design MMLE fit should succeed")
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    assert!(x.len() >= 2);
    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut covariance = 0.0;
    let mut x_variance = 0.0;
    let mut y_variance = 0.0;
    for (&left, &right) in x.iter().zip(y) {
        let x_delta = left - x_mean;
        let y_delta = right - y_mean;
        covariance += x_delta * y_delta;
        x_variance += x_delta * x_delta;
        y_variance += y_delta * y_delta;
    }
    covariance / (x_variance.sqrt() * y_variance.sqrt())
}

fn rmse(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    (x.iter()
        .zip(y)
        .map(|(&left, &right)| (left - right).powi(2))
        .sum::<f64>()
        / x.len() as f64)
        .sqrt()
}

fn standardized(values: &[f64]) -> Vec<f64> {
    let center = mean(values);
    let variance = values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / values.len() as f64;
    let scale = variance.sqrt().max(1e-12);
    values.iter().map(|value| (value - center) / scale).collect()
}

fn interaction_adjusted_easiness(
    intercept: &[f64],
    zeta: &[f64],
    gamma: f64,
    seed: u64,
) -> Vec<f64> {
    const DRAWS: usize = 2_048;
    let mut rng = Lcg(seed);
    let xi_draws: Vec<f64> = (0..DRAWS * LATENT_DIM)
        .map(|_| rng.normal())
        .collect();
    (0..N_ITEMS)
        .map(|item| {
            let mut distance_sum = 0.0;
            for draw in 0..DRAWS {
                let mut squared_distance = EPS_DISTANCE;
                for axis in 0..LATENT_DIM {
                    let difference = xi_draws[draw * LATENT_DIM + axis]
                        - zeta[item * LATENT_DIM + axis];
                    squared_distance += difference * difference;
                }
                distance_sum += squared_distance.sqrt();
            }
            intercept[item] - gamma * distance_sum / DRAWS as f64
        })
        .collect()
}

fn pairwise_distances(points: &[f64], rows: usize) -> Vec<f64> {
    assert_eq!(points.len(), rows * LATENT_DIM);
    let mut distances = Vec::with_capacity(rows * rows.saturating_sub(1) / 2);
    for left in 0..rows {
        for right in left + 1..rows {
            let mut squared_distance = 0.0;
            for axis in 0..LATENT_DIM {
                let difference = points[left * LATENT_DIM + axis]
                    - points[right * LATENT_DIM + axis];
                squared_distance += difference * difference;
            }
            distances.push(squared_distance.sqrt());
        }
    }
    distances
}

fn theta_correlation(cell: &StudyCell, result: &MarginalResult) -> f64 {
    let per_dimension: Vec<f64> = (0..N_DIMS)
        .map(|dimension| {
            let truth: Vec<f64> = (0..cell.n_persons)
                .map(|person| cell.theta_true[person * N_DIMS + dimension])
                .collect();
            let estimate: Vec<f64> = (0..cell.n_persons)
                .map(|person| result.theta_eap[person * N_DIMS + dimension])
                .collect();
            pearson(&truth, &estimate)
        })
        .collect();
    mean(&per_dimension)
}

fn recovery_metrics(cell: &StudyCell, result: &MarginalResult) -> RecoveryMetrics {
    let a_estimated: Vec<f64> = result.alpha.iter().map(|value| value.exp()).collect();
    let adjusted_truth = interaction_adjusted_easiness(
        &cell.b_true,
        &cell.zeta_true,
        GAMMA,
        0xA11CE,
    );
    let adjusted_estimate = interaction_adjusted_easiness(
        &result.b,
        &result.zeta,
        result.tau.exp(),
        0xA11CE,
    );
    let zeta_truth_distance = pairwise_distances(&cell.zeta_true, N_ITEMS);
    let zeta_estimated_distance = pairwise_distances(&result.zeta, N_ITEMS);
    RecoveryMetrics {
        a_corr: pearson(&cell.a_true, &a_estimated),
        a_rmse: rmse(&standardized(&cell.a_true), &standardized(&a_estimated)),
        adjusted_b_corr: pearson(&adjusted_truth, &adjusted_estimate),
        adjusted_b_rmse: rmse(
            &standardized(&adjusted_truth),
            &standardized(&adjusted_estimate),
        ),
        theta_corr: theta_correlation(cell, result),
        zeta_distance_corr: pearson(&zeta_truth_distance, &zeta_estimated_distance),
        gamma: result.tau.exp(),
    }
}

fn assert_likelihood_trace(trace: &[f64]) {
    assert!(!trace.is_empty(), "fit must report a likelihood trace");
    assert!(trace.iter().all(|value| value.is_finite()));
    for pair in trace.windows(2) {
        assert!(
            pair[1] >= pair[0] - 1e-3,
            "marginal likelihood decreased materially: {} -> {}",
            pair[0],
            pair[1]
        );
    }
}

#[test]
fn simple_structure_equation_matches_general_dot_product() {
    let alpha = 1.25_f64.ln();
    let theta: [f64; 2] = [0.40, -0.70];
    let factor = 1;
    let intercept = 0.80;
    let tau = 1.50_f64.ln();
    let xi: [f64; 2] = [0.25, -0.10];
    let zeta: [f64; 2] = [-0.35, 0.65];
    let distance = ((xi[0] - zeta[0]).powi(2) + (xi[1] - zeta[1]).powi(2)).sqrt();
    let simple = alpha.exp() * theta[factor] + intercept - tau.exp() * distance;
    let discrimination = [0.0, alpha.exp()];
    let general = discrimination[0] * theta[0] + discrimination[1] * theta[1]
        + intercept
        - tau.exp() * distance;
    assert!((simple - general).abs() < 1e-15);
}

#[test]
#[ignore = "deterministic literature recovery study; executed by rust-statistical CI"]
fn kang_jeon_2025_minimum_cell_recovers_true_parameters() {
    let cell = simulate_paper_cell(500, 0x5EED_2025);
    let result = fit_cell(&cell, Device::Cpu, 150);
    let metrics = recovery_metrics(&cell, &result);
    assert_likelihood_trace(&result.loglik_trace);
    assert!(result.b.iter().all(|value| value.is_finite()));
    assert!(result.alpha.iter().all(|value| value.is_finite()));
    assert!(result.theta_eap.iter().all(|value| value.is_finite()));
    assert!(result.zeta.iter().all(|value| value.is_finite()));
    assert!(
        metrics.adjusted_b_corr > 0.60,
        "adjusted-easiness correlation too low: {metrics:?}"
    );
    assert!(
        metrics.adjusted_b_rmse < 1.00,
        "standardized adjusted-easiness RMSE too high: {metrics:?}"
    );
    assert!(
        metrics.a_corr > 0.20,
        "discrimination correlation too low: {metrics:?}"
    );
    assert!(
        metrics.a_rmse < 1.30,
        "standardized discrimination RMSE too high: {metrics:?}"
    );
    assert!(
        metrics.theta_corr > 0.30,
        "trait-score correlation too low: {metrics:?}"
    );
    assert!(
        metrics.zeta_distance_corr > 0.00,
        "item-map distance recovery is not positive: {metrics:?}"
    );
    assert!(
        metrics.gamma.is_finite() && metrics.gamma > 0.10 && metrics.gamma < 5.0,
        "distance weight left the admissible recovery region: {metrics:?}"
    );
    eprintln!("Kang-Jeon 2025 recovery metrics: {metrics:?}");
}

#[test]
#[ignore = "explicit wgpu recovery parity; executed on Mesa Lavapipe CI"]
fn gpu_recovery_matches_cpu_on_paper_design() {
    let cell = simulate_paper_cell(300, 0x6A50_2025);
    let cpu = fit_cell(&cell, Device::Cpu, 35);
    let gpu = fit_cell(&cell, Device::Gpu, 35);
    assert_likelihood_trace(&cpu.loglik_trace);
    assert_likelihood_trace(&gpu.loglik_trace);
    let cpu_final = *cpu.loglik_trace.last().expect("CPU likelihood");
    let gpu_final = *gpu.loglik_trace.last().expect("GPU likelihood");
    let relative = (cpu_final - gpu_final).abs() / cpu_final.abs().max(1.0);
    assert!(relative < 5e-3, "CPU/GPU likelihood mismatch: {relative}");
    assert!(pearson(&cpu.b, &gpu.b) > 0.995);
    assert!(pearson(&cpu.alpha, &gpu.alpha) > 0.995);
    assert!(pearson(&cpu.theta_eap, &gpu.theta_eap) > 0.995);
    assert!(pearson(&cpu.zeta, &gpu.zeta) > 0.990);
    assert!((cpu.tau.exp() - gpu.tau.exp()).abs() < 0.10);
}
