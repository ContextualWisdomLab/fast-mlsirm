//! Standalone reproduction test for replicate 3 M-step line search failure.
//! Robitzsch (2024) seed: 202612290006, N=2000, I=10.
mod support {
    pub mod published_norm;
}

#[test]
#[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
fn replicate_03_mstep_accepts_approximate_wolfe() {
    use mlsirm_core::gpu_mstep::NewtonControl;
    use mlsirm_core::grm::{fit_grm_gpu_strict, GrmConfig};

    let q = 121;
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
    let seed: u64 = 202612290006; // Replicate 03 seed
    let y = support::published_norm::simulate(2000, seed);
    let outcome = fit_grm_gpu_strict(&y, None, &[1; 10], 2000, 10, 1, 2, &cfg, &ctrl);

    let (fit, receipt) =
        outcome.expect("approximate-Wolfe line search must not fail at f32 resolution");
    println!(
        "REP03 converged={} n_iter={} reason={} slopes={:?}",
        fit.converged, fit.n_iter, fit.termination_reason, fit.slope
    );
    assert!(fit
        .slope
        .iter()
        .chain(&fit.threshold)
        .all(|x| x.is_finite()));
    assert_eq!(receipt.cpu_numerical_fallbacks, 0);
}
