//! Published NORM generating model: Robitzsch (2024), Eq. 2, §3.1, Table A1.
//! Robitzsch, A. (2024). A note on equivalent and nonequivalent parametrizations
//! of the two-parameter logistic item response model. Information, 15(11), 668.
//! https://doi.org/10.3390/info15110668
//! This is parameter recovery, not replication of the paper's AIC experiment.

#[path = "support/published_norm.rs"]
mod published_norm;

#[test]
fn published_norm_simulator_and_scale() {
    let y = published_norm::simulate(2000, 15110668);
    assert_eq!(y.len(), 20_000);
    assert_eq!(y, published_norm::simulate(2000, 15110668));
    assert_ne!(y, published_norm::simulate(2000, 15110669));
    for i in 0..10 {
        let p = published_norm::probability(i, published_norm::B[i]);
        assert!((p - 0.5).abs() < 1e-14);
    }
}

#[test]
#[ignore = "bounded hardware GPU recovery; run under the canonical GPU lease"]
#[cfg(feature = "gpu")]
fn published_norm_gpu_parameter_recovery() {
    use mlsirm_core::gpu_mstep::NewtonControl;
    use mlsirm_core::grm::{fit_grm_gpu_strict, GrmConfig};
    let q: usize = std::env::var("RECOVERY_Q")
        .expect("explicit RECOVERY_Q")
        .parse()
        .unwrap();
    assert!(q >= 121);
    let ctrl = NewtonControl {
        iterations: 4,
        backtracks: 25,
        ridge: 1e-3,
        gradient_tolerance: 0.005,
        step_tolerance: 1e-4,
        pivot_tolerance: 1e-10,
        armijo: 1e-4,
    };
    let cfg = GrmConfig {
        max_iter: 500,
        tol: 1e-4,
        q,
        ridge: ctrl.ridge,
        newton_iter: ctrl.iterations,
        ..GrmConfig::default()
    };
    let y = published_norm::simulate(2000, 15110668);
    // Only responses/pattern/settings cross this boundary: no truth/start vector.
    // Canonical initialization is slope=1 and empirical marginal logits.
    let outcome = fit_grm_gpu_strict(&y, None, &[1; 10], 2000, 10, 1, 2, &cfg, &ctrl);
    if let Err(ref error) = outcome {
        eprintln!("RECOVERY_FAILURE {error}");
    }
    let (fit, receipt) = outcome.expect("published NORM strict GPU fitting");
    assert!(fit.converged);
    assert_eq!(receipt.cpu_numerical_fallbacks, 0);
    assert_eq!(receipt.e_step_gpu_dispatches, fit.n_iter + 1);
    assert!(receipt.parameter_update > 0 && receipt.m_step_gpu_dispatches > 0);
    let mut errors_a = Vec::new();
    let mut errors_b = Vec::new();
    for i in 0..10 {
        assert!(fit.slope[i].is_finite() && fit.slope[i] > 0.0);
        let b = -fit.threshold[i] / fit.slope[i];
        errors_a.push(fit.slope[i] - published_norm::A[i]);
        errors_b.push(b - published_norm::B[i]);
        println!(
            "RECOVERY_ITEM item={} a={} b={} error_a={} error_b={}",
            i + 1,
            fit.slope[i],
            b,
            errors_a[i],
            errors_b[i]
        );
    }
    let (mean_a, rmse_a) = published_norm::errors(&errors_a);
    let (mean_b, rmse_b) = published_norm::errors(&errors_b);
    println!("RECOVERY_RESULT mode=single_smoke replications=1 n=2000 q={q} converged={} iterations={} mean_error_a={mean_a} rmse_a={rmse_a} mean_error_b={mean_b} rmse_b={rmse_b} loglik={}",
        fit.converged, fit.n_iter, fit.loglik_trace.last().unwrap());
    println!(
        "RECOVERY_DISPATCH {}",
        serde_json::to_string(&receipt).unwrap()
    );
    // Prospective coarse engineering regression bounds, NOT paper bias limits
    // or an assertion of Monte Carlo/unconditional scientific validity.
    assert!(rmse_a < 1.0 && rmse_b < 0.35, "raw-scale recovery failed");
}
