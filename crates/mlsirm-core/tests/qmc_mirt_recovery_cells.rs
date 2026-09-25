//! Independent QMC-MIRT Monte Carlo recovery cells for retained compiler-baseline evidence.
//!
//! The historical unit study exercised D=4/D=5 and normal/skew conditions in one ignored
//! target. Each cell here preserves the same deterministic simulation, 500-replication
//! denominator, Halton point count, sample size, estimator configuration, and acceptance
//! thresholds while giving the evidence workflow an exact Cargo test target per condition.

use mlsirm_core::marginal::XiRuleKind;
use mlsirm_core::twopl::{fit_2pl, TwoPlConfig};

const REPS: usize = 500;
const XI_SEED: u64 = 0x2545_F491_4F6C_DD1D;

struct Lcg(u64);

impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }

    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn bern(&mut self, p: f64) -> f64 {
        if self.next_f64() < p { 1.0 } else { 0.0 }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

fn simulate(
    loading: &[f64],
    intercept: &[f64],
    thetas: &[f64],
    n: usize,
    n_items: usize,
    n_dims: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let mut y = vec![0.0; n * n_items];
    for person in 0..n {
        for item in 0..n_items {
            let mut eta = intercept[item];
            for dim in 0..n_dims {
                eta += loading[item * n_dims + dim] * thetas[person * n_dims + dim];
            }
            y[person * n_items + item] = rng.bern(sigmoid(eta));
        }
    }
    y
}

fn run_qmc_mirt_recovery_cell(n_dims: usize, xi_points: usize, n: usize, skew: bool) {
    assert!(
        (n_dims, xi_points, n) == (4, 4000, 2000)
            || (n_dims, xi_points, n) == (5, 6000, 1500),
        "only the governed D=4/D=5 recovery cells are valid"
    );

    let mut pattern = Vec::new();
    for dim in 0..n_dims {
        for _ in 0..2 {
            let mut row = vec![0u8; n_dims];
            row[dim] = 1;
            pattern.extend_from_slice(&row);
        }
    }
    for dim in 0..n_dims {
        let mut row = vec![0u8; n_dims];
        row[dim] = 1;
        row[(dim + 1) % n_dims] = 1;
        pattern.extend_from_slice(&row);
    }

    let n_items = 3 * n_dims;
    let mut loading = vec![0.0; n_items * n_dims];
    for dim in 0..n_dims {
        loading[(2 * dim) * n_dims + dim] = 1.2;
        loading[(2 * dim + 1) * n_dims + dim] = 0.9;
    }
    for dim in 0..n_dims {
        let item = 2 * n_dims + dim;
        loading[item * n_dims + dim] = 1.0;
        loading[item * n_dims + (dim + 1) % n_dims] = if dim % 2 == 0 { 0.7 } else { -0.7 };
    }
    let intercept: Vec<f64> = (0..n_items).map(|item| -0.5 + 0.1 * item as f64).collect();

    let (mut loading_sq_error, mut loading_count, mut loading_bias) = (0.0, 0.0, 0.0);
    let (mut theta_corr_sum, mut theta_corr_count) = (0.0, 0.0);
    let mut converged = 0usize;

    for rep in 0..REPS {
        let mut rng = Lcg(
            0x9E3779B97F4A7C15u64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                .wrapping_add(n_dims as u64 * 0x100000001B3),
        );
        let mut thetas = vec![0.0; n * n_dims];
        for dim in 0..n_dims {
            let column: Vec<f64> = (0..n)
                .map(|_| {
                    if skew {
                        let mut chi_square = 0.0;
                        for _ in 0..3 {
                            let z = rng.normal();
                            chi_square += z * z;
                        }
                        (chi_square - 3.0) / 6f64.sqrt()
                    } else {
                        rng.normal()
                    }
                })
                .collect();
            let mean = column.iter().sum::<f64>() / n as f64;
            let variance = column
                .iter()
                .map(|value| (value - mean) * (value - mean))
                .sum::<f64>()
                / n as f64;
            let sd = variance.sqrt();
            for person in 0..n {
                thetas[person * n_dims + dim] = (column[person] - mean) / sd;
            }
        }

        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let result = fit_2pl(
            &y,
            &observed,
            &pattern,
            n,
            n_items,
            n_dims,
            &TwoPlConfig {
                xi_rule: XiRuleKind::Halton,
                xi_points,
                xi_seed: XI_SEED,
                ..TwoPlConfig::default()
            },
        )
        .unwrap();

        if result.converged {
            converged += 1;
        }
        assert!(result.loglik_trace.iter().all(|value| value.is_finite()), "finite loglik (rep {rep})");
        for window in result.loglik_trace.windows(2) {
            assert!(window[1] >= window[0] - 1e-6, "monotone loglik (rep {rep})");
        }
        for item in 0..n_items {
            for dim in 0..n_dims {
                let estimate = result.loading[item * n_dims + dim];
                if pattern[item * n_dims + dim] == 0 {
                    assert_eq!(estimate, 0.0, "unloaded exactly zero");
                } else {
                    assert!(estimate.is_finite() && estimate.abs() <= 10.0, "loading in bound");
                    let error = estimate - loading[item * n_dims + dim];
                    loading_sq_error += error * error;
                    loading_count += 1.0;
                    loading_bias += error;
                }
            }
        }
        assert!(result.theta.iter().all(|value| value.is_finite()), "finite theta (rep {rep})");
        for dim in 0..n_dims {
            let estimated: Vec<f64> = (0..n).map(|person| result.theta[person * n_dims + dim]).collect();
            let truth: Vec<f64> = (0..n).map(|person| thetas[person * n_dims + dim]).collect();
            theta_corr_sum += corr(&estimated, &truth);
            theta_corr_count += 1.0;
        }
    }

    let loading_rmse = (loading_sq_error / loading_count).sqrt();
    let loading_mean_bias = loading_bias / loading_count;
    let theta_correlation = theta_corr_sum / theta_corr_count;
    let convergence = converged as f64 / REPS as f64;
    println!(
        "[qmc-mirt MC D={n_dims} xi={xi_points} N={n} skew={skew}] reps={REPS} conv={convergence:.3} loadRMSE={loading_rmse:.4} loadBias={loading_mean_bias:.4} thetaCorr={theta_correlation:.3}"
    );

    assert!(convergence > 0.90, "convergence {convergence} (D={n_dims} skew={skew})");
    if skew {
        assert!(loading_rmse < 0.26, "skew loading RMSE {loading_rmse} (D={n_dims})");
        assert!(theta_correlation > 0.50, "skew theta corr {theta_correlation} (D={n_dims})");
    } else {
        assert!(loading_mean_bias.abs() < 0.06, "loading bias {loading_mean_bias} (D={n_dims})");
        assert!(loading_rmse < 0.19, "loading RMSE {loading_rmse} (D={n_dims})");
        assert!(theta_correlation > 0.55, "theta corr {theta_correlation} (D={n_dims})");
    }
}

#[test]
#[ignore = "literature-grade QMC-MIRT Monte Carlo recovery (500 reps); dedicated evidence job"]
fn mc_qmc_mirt_recovery_500_d4_normal() {
    run_qmc_mirt_recovery_cell(4, 4000, 2000, false);
}

#[test]
#[ignore = "literature-grade QMC-MIRT Monte Carlo recovery (500 reps); dedicated evidence job"]
fn mc_qmc_mirt_recovery_500_d4_skew() {
    run_qmc_mirt_recovery_cell(4, 4000, 2000, true);
}

#[test]
#[ignore = "literature-grade QMC-MIRT Monte Carlo recovery (500 reps); dedicated evidence job"]
fn mc_qmc_mirt_recovery_500_d5_normal() {
    run_qmc_mirt_recovery_cell(5, 6000, 1500, false);
}

#[test]
#[ignore = "literature-grade QMC-MIRT Monte Carlo recovery (500 reps); dedicated evidence job"]
fn mc_qmc_mirt_recovery_500_d5_skew() {
    run_qmc_mirt_recovery_cell(5, 6000, 1500, true);
}
