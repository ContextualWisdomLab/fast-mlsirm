//! Actual strict-fit consumer controls for bounded Newton completion.
#![cfg(feature = "gpu")]

use mlsirm_core::gpu_mstep::NewtonControl;
use mlsirm_core::grm::{fit_grm_gpu_strict, GrmConfig};
use mlsirm_core::marginal::XiRuleKind;

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn strict_fit_does_not_promote_iteration_limit_to_convergence() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let persons = 27;
    let items = 4;
    let y: Vec<usize> = (0..persons)
        .flat_map(|p| (0..items).map(move |i| (p + i) % 3))
        .collect();
    let control = NewtonControl {
        iterations: 1,
        backtracks: 25,
        ridge: 1e12,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    };
    let cfg = GrmConfig {
        max_iter: 1,
        tol: 1e-12,
        q: 121,
        ridge: control.ridge,
        newton_iter: control.iterations,
        xi_rule: XiRuleKind::GaussHermite,
        xi_points: 121,
        xi_seed: 20261026,
    };
    let result = fit_grm_gpu_strict(
        &y,
        None,
        &vec![1; items],
        persons,
        items,
        1,
        3,
        &cfg,
        &control,
    );
    eprintln!(
        "STRICT_LIMIT_RESULT {:?}",
        result.as_ref().map(|(fit, receipt)| (
            fit.converged,
            fit.n_iter,
            &fit.termination_reason,
            &fit.loglik_trace,
            receipt,
        ))
    );
    let error = result.expect_err(
        "an iteration-limited tiny Newton sweep must not certify strict-fit convergence",
    );
    assert!(error.contains("nonconvergence"), "{error}");
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn multigroup_does_not_promote_iteration_limit_to_convergence() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig};
    let persons = 54;
    let items = 7;
    let y: Vec<usize> = (0..persons)
        .flat_map(|p| (0..items).map(move |i| (p + i) % 3))
        .collect();
    let groups: Vec<usize> = (0..persons).map(|p| p / 27).collect();
    let observed: Vec<bool> = (0..persons)
        .flat_map(|p| (0..items).map(move |_| p < 27))
        .collect();
    let cfg = BifactorMultigroupConfig {
        q_general: 3,
        q_specific: 3,
        max_iter: 1,
        tol: 1e-12,
        n_starts: 1,
        seed: 20261026,
        newton_iter: 1,
        ridge: 1e12,
        estimate_specific_means: false,
        estimate_specific_vars: false,
        device: mlsirm_core::Device::Gpu,
        ..Default::default()
    };
    // The focal group is entirely missing, so its posterior moments remain
    // at the prior. This is a finite dispatch fixture, not a fitted study.
    let result = fit_bifactor_grm_multigroup(
        &y,
        Some(&observed),
        &groups,
        2,
        &[0, 0, 1, 1, 2, 2, -1],
        persons,
        items,
        3,
        3,
        None,
        &cfg,
    );
    eprintln!(
        "MULTIGROUP_LIMIT_RESULT {:?}",
        result.as_ref().map(|fit| (
            fit.converged,
            fit.n_iter,
            &fit.termination_reason,
            &fit.loglik_trace,
        ))
    );
    let fit = result.expect("bounded update remains public Ok");
    assert!(
        !fit.converged,
        "iteration-limited item updates are not convergence evidence"
    );
    assert_eq!(fit.termination_reason, "max_iter_reached");
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn bounded_item_ok_preserves_iteration_limit_and_stationarity_receipts() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    use mlsirm_core::bifactor_grm::SlopePrior;
    use mlsirm_core::gpu_mstep::update_grm;
    let control = NewtonControl {
        iterations: 1,
        backtracks: 25,
        ridge: 1e12,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    };
    let (_, limited) = update_grm(
        &[1., 0.],
        1,
        &[0.],
        &[vec![2., 1.]],
        SlopePrior::None,
        &control,
    )
    .expect("bounded status5 remains Ok");
    assert_eq!(limited.iteration_limit_updates, 1);
    assert_eq!(limited.parameter_update, 1);
    let (_, stationary) = update_grm(
        &[1., 0.],
        1,
        &[0.],
        &[vec![1., 1.]],
        SlopePrior::None,
        &control,
    )
    .expect("stationary status1 remains Ok");
    assert_eq!(stationary.iteration_limit_updates, 0);
    assert_eq!(stationary.parameter_update, 0);
    assert!(stationary.gradient > 0);
}

#[test]
#[ignore = "requires actual GPU under the common host lease"]
fn strict_fit_can_recover_after_an_earlier_limited_sweep() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let persons = 180;
    let items = 4;
    let mut seed = 72915u64;
    let mut y = Vec::new();
    for p in 0..persons {
        let theta = (p as f64 - (persons - 1) as f64 / 2.0) / (persons as f64 / 5.0);
        for i in 0..items {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u = ((seed >> 11) as f64) / ((1u64 << 53) as f64);
            let a = 0.7 + 0.2 * i as f64;
            let ge1 = 1.0 / (1.0 + (-a * theta - 0.8).exp());
            let ge2 = 1.0 / (1.0 + (-a * theta + 0.8).exp());
            y.push(if u < 1.0 - ge1 {
                0
            } else if u < 1.0 - ge2 {
                1
            } else {
                2
            });
        }
    }
    let control = NewtonControl {
        iterations: 2,
        backtracks: 25,
        ridge: 1e-8,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    };
    let cfg = GrmConfig {
        max_iter: 60,
        tol: 1e-8,
        q: 3,
        ridge: control.ridge,
        newton_iter: control.iterations,
        xi_rule: XiRuleKind::GaussHermite,
        xi_points: 3,
        xi_seed: 20261026,
    };
    let (fit, receipt) = fit_grm_gpu_strict(
        &y,
        None,
        &vec![1; items],
        persons,
        items,
        1,
        3,
        &cfg,
        &control,
    )
    .expect("later stationary sweep can satisfy unchanged EM tolerance");
    eprintln!(
        "STRICT_RECOVERY_RESULT converged={} iterations={} limited={}",
        fit.converged, fit.n_iter, receipt.iteration_limit_updates
    );
    assert!(fit.converged);
    assert!(
        receipt.iteration_limit_updates > 0,
        "must exercise an earlier limited sweep"
    );
}
