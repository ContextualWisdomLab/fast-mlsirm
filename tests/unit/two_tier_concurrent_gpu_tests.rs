//! Synthetic control-flow fixtures, not study-grid precision evidence.
use super::*;

fn synthetic_config() -> TwoTierGrmConfig {
    TwoTierGrmConfig {
        estimate_primary_correlation: false,
        q_primary: 3,
        q_specific: 3,
        max_iter: 2,
        tol: 1e-8,
        n_starts: 1,
        seed: 9178,
        newton_iter: 20,
        ridge: 1e-8,
    }
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn smallest_real_two_tier_fit_dispatches_gpu_items() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let cfg = synthetic_config();
    // A generic crossed two-primary + one-specific pattern, not G-only.
    let primary = [true, false, true, true, true, false, true, true];
    let specific = [0, 0, 0, 0];
    let y: Vec<usize> = (0..16)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    let v = validate(&y, None, &primary, &specific, 16, 4, 2, 1, 2, &cfg).unwrap();
    let (z, w) = gh_rule(cfg.q_primary).unwrap();
    let (ts, ws) = gh_rule(cfg.q_specific).unwrap();
    let (coords, log_w) = build_primary_grid(z, w, 2, v.grid_size);
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    CPU_ITEM_FIT_CALLS.with(|c| c.set(0));
    let result = run_single_start_with_item_device(
        &v,
        &y,
        None,
        &cfg,
        &coords,
        &log_w,
        ts,
        &log_ws,
        v.grid_size,
        ts.len(),
        0,
        crate::Device::Gpu,
    )
    .unwrap();
    assert_eq!(
        CPU_ITEM_FIT_CALLS.with(|c| c.get()),
        0,
        "explicit GPU fitting still reaches the CPU item optimizer"
    );
    assert!(result.n_iter > 0);
    assert!(result.loglik_trace.iter().all(|x| x.is_finite()));
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn concurrent_mixed_common_free_and_explicit_moments() {
    use super::concurrent::*;
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let mut em = synthetic_config();
    em.max_iter = 1;
    let primary = [true, false, true, true, true, false, true, true];
    let specific = [0, 0, 0, 0];
    let mut y: Vec<usize> = (0..16)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    y.extend((0..24).flat_map(|p| (0..4).map(move |i| usize::from((p + 3 * i) % (5 + i) > 2))));
    let groups: Vec<usize> = (0..40).map(|p| usize::from(p >= 16)).collect();
    let mut moments = vec![MomentConstraint::fixed_standard(); 6];
    moments[3] = MomentConstraint {
        mean: 0.2,
        sd: 1.1,
        free_mean: true,
        free_sd: true,
    };
    // Focal W and specific are explicit fixed nonstandard controls.
    moments[4] = MomentConstraint {
        mean: -0.15,
        sd: 0.9,
        free_mean: false,
        free_sd: false,
    };
    moments[5] = MomentConstraint {
        mean: 0.3,
        sd: 1.2,
        free_mean: false,
        free_sd: false,
    };
    let cfg = ConcurrentConfig {
        em,
        newton: strict_control(20, 1e-8),
    };
    CPU_ITEM_FIT_CALLS.with(|c| c.set(0));
    let fit = fit_concurrent_gpu(
        &y,
        None,
        &groups,
        2,
        &primary,
        &specific,
        40,
        4,
        2,
        1,
        2,
        &[true, true, false, true],
        &moments,
        &cfg,
    )
    .unwrap();
    assert_eq!(CPU_ITEM_FIT_CALLS.with(|c| c.get()), 0);
    assert_eq!(fit.n_iter, 1);
    assert!(!fit.converged);
    for i in [0, 1, 3] {
        assert_eq!(
            fit.a_primary[0][i * 2..(i + 1) * 2],
            fit.a_primary[1][i * 2..(i + 1) * 2]
        );
        assert_eq!(fit.a_specific[0][i], fit.a_specific[1][i]);
        assert_eq!(fit.threshold[0][i], fit.threshold[1][i]);
    }
    assert_ne!(fit.threshold[0][2], fit.threshold[1][2]);
    assert_ne!(
        fit.threshold[0][0], 0.0,
        "common bank is estimated, not FIPC"
    );
    assert_eq!(&fit.moments[..3], &moments[..3]);
    assert_eq!(&fit.moments[4..], &moments[4..]);
    assert_ne!(fit.moments[3].mean, moments[3].mean);
    assert_ne!(fit.moments[3].sd, moments[3].sd);
    assert_eq!(fit.phi, vec![1., 0., 0., 1.]);
    assert_eq!(fit.starts.len(), 1);
    assert_eq!(fit.starts[0].sweeps.len(), 2); // initial + terminal
    let m = fit.starts[0].sweeps[0].m_step.as_ref().unwrap();
    assert_eq!((m.common_updates, m.free_updates), (3, 2));
    assert!(m.stages.gradient > 0);
    assert!(fit.starts[0].sweeps.iter().all(|s| s.groups.len() == 2
        && s.groups
            .iter()
            .all(|g| g.gpu_used && g.fallback_reason.is_none())));
    eprintln!(
        "CONCURRENT_SYNTHETIC {{iter:{},converged:{},starts:{:?}}}",
        fit.n_iter, fit.converged, fit.starts
    );
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn selected_g4w_all_free_focal_moments_uses_full_model() {
    // Retained polytomous integration NONPASS on the current Metal optimizer.
    selected_g4w_synthetic(4);
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn selected_g4w_binary_control_uses_six_factors() {
    // Control only, never a substitute for the four-category acceptance above.
    selected_g4w_synthetic(2);
}

fn selected_g4w_synthetic(n_cat: usize) {
    use super::concurrent::*;
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let mut em = synthetic_config();
    em.max_iter = 1;
    let wording = [4, 5, 6, 9, 12, 14, 15];
    let primary: Vec<bool> = (0..16).flat_map(|i| [true, wording.contains(&i)]).collect();
    let specific: Vec<i32> = (0..16).map(|i| i % 4).collect();
    let mut rng = SplitMix64(803729);
    let mut y = Vec::new();
    // Generic synthetic categories exercise the selected design, never private input.
    for person in 0..96 {
        for item in 0..16 {
            let general = rng.standard_normal();
            let shift = if person >= 48 { 0.25 } else { -0.1 };
            let x = general + shift + 0.02 * (item as f64);
            y.push(if n_cat == 2 {
                usize::from(x > 0.)
            } else if x < -0.7 {
                0
            } else if x < 0. {
                1
            } else if x < 0.7 {
                2
            } else {
                3
            });
        }
    }
    let groups: Vec<usize> = (0..96).map(|p| usize::from(p >= 48)).collect();
    let mut moments = vec![MomentConstraint::fixed_standard(); 12];
    for m in &mut moments[6..] {
        m.free_mean = true;
        m.free_sd = true;
    }
    let cfg = ConcurrentConfig {
        em,
        newton: strict_control(20, 1e-8),
    };
    let fit = fit_concurrent_gpu(
        &y,
        None,
        &groups,
        2,
        &primary,
        &specific,
        96,
        16,
        2,
        4,
        n_cat,
        &vec![true; 16],
        &moments,
        &cfg,
    )
    .unwrap();
    assert_eq!(fit.a_primary[0].len(), 32);
    assert_eq!(fit.a_specific[0].len(), 16);
    assert_eq!(fit.threshold[0].len(), 16 * (n_cat - 1));
    assert!(fit.threshold.iter().all(|bank| bank
        .chunks_exact(n_cat - 1)
        .all(|d| d.windows(2).all(|w| w[0] > w[1]))));
    assert_eq!(fit.moments.len(), 12);
    assert_eq!(fit.phi, vec![1., 0., 0., 1.]);
    assert_eq!(&fit.moments[..6], &moments[..6]);
    for d in 6..12 {
        assert_ne!(fit.moments[d].mean, moments[d].mean);
        assert_ne!(fit.moments[d].sd, moments[d].sd);
    }
    assert_eq!(fit.a_primary[0], fit.a_primary[1]);
    assert_eq!(fit.a_specific[0], fit.a_specific[1]);
    assert_eq!(fit.threshold[0], fit.threshold[1]);
    for i in 0..16 {
        if !wording.contains(&i) {
            assert_eq!(fit.a_primary[0][2 * i + 1], 0.);
        }
    }
    let m = fit.starts[0].sweeps[0].m_step.as_ref().unwrap();
    assert_eq!((m.common_updates, m.free_updates), (16, 0));
    assert_eq!(fit.starts[0].sweeps[0].cpu_gaussian_parameter_updates, 12);
    assert!(!fit.converged); // one sweep is not a study fit
    eprintln!(
        "SELECTED_G4W_SYNTHETIC n_cat={} updates={} mstep={:?}",
        n_cat, m.common_updates, m.stages
    );
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn concurrent_bounded_updates_cannot_certify_convergence_and_can_recover() {
    use super::concurrent::*;
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    // One-node integration isolates consumer convergence; no information or
    // study-precision claim. Both the common and free item branches execute.
    let primary = [true, false, true, true, true, false, true, true];
    let specific = [0, 0, 0, 0];
    let y: Vec<usize> = (0..32)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    let groups: Vec<usize> = (0..32).map(|p| p / 16).collect();
    let moments = vec![MomentConstraint::fixed_standard(); 6];
    let mut em = synthetic_config();
    em.q_primary = 1;
    em.q_specific = 1;
    em.max_iter = 1;
    em.tol = 1e-8;
    em.ridge = 1e12;
    em.newton_iter = 1;
    let limited = fit_concurrent_gpu(
        &y,
        None,
        &groups,
        2,
        &primary,
        &specific,
        32,
        4,
        2,
        1,
        2,
        &[true, false, true, true],
        &moments,
        &ConcurrentConfig {
            em,
            newton: strict_control(1, 1e12),
        },
    )
    .unwrap();
    assert!(!limited.converged);
    assert!(
        limited.final_param_change < em.tol,
        "tiny update is the counterexample"
    );
    assert!(
        limited.starts[0].sweeps[0]
            .m_step
            .as_ref()
            .unwrap()
            .stages
            .iteration_limit_updates
            > 0
    );
    assert_eq!(limited.termination_reason, "max_iter_reached");
    em.ridge = 1e-8;
    em.newton_iter = 2;
    em.max_iter = 12;
    em.n_starts = 2;
    let recovered = fit_concurrent_gpu(
        &y,
        None,
        &groups,
        2,
        &primary,
        &specific,
        32,
        4,
        2,
        1,
        2,
        &[true, false, true, true],
        &moments,
        &ConcurrentConfig {
            em,
            newton: strict_control(2, 1e-8),
        },
    )
    .unwrap();
    assert!(
        recovered.converged,
        "bounded sweep must not permanently prevent recovery"
    );
    assert_eq!(recovered.starts.len(), 2);
    let best = &recovered.starts[recovered.best_start];
    assert!(best
        .sweeps
        .iter()
        .filter_map(|s| s.m_step.as_ref())
        .any(|m| m.stages.iteration_limit_updates > 0));
    assert_eq!(
        best.sweeps[best.sweeps.len() - 2]
            .m_step
            .as_ref()
            .unwrap()
            .stages
            .iteration_limit_updates,
        0
    );
    assert!(best.sweeps.last().unwrap().m_step.is_none());
    assert!(recovered.final_param_change <= em.tol);
    eprintln!(
        "BOUNDED_RECOVERY starts={} iterations={} final_change={:e}",
        recovered.starts.len(),
        recovered.n_iter,
        recovered.final_param_change
    );
}

#[test]
fn concurrent_rejects_invalid_constraints_before_dispatch() {
    use super::concurrent::*;
    let y: Vec<usize> = (0..32)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    let primary = [true, false, true, true, true, false, true, true];
    let specific = [0, 0, 0, 0];
    let groups: Vec<usize> = (0..32).map(|p| p / 16).collect();
    let moments = vec![MomentConstraint::fixed_standard(); 6];
    let cfg = ConcurrentConfig {
        em: synthetic_config(),
        newton: strict_control(20, 1e-8),
    };
    let call = |mm: &[MomentConstraint], cc: &ConcurrentConfig, gg: &[usize], common: &[bool]| {
        fit_concurrent_gpu(
            &y, None, gg, 2, &primary, &specific, 32, 4, 2, 1, 2, common, mm, cc,
        )
        .unwrap_err()
    };
    let mut bad = moments.clone();
    bad[0].free_mean = true;
    assert!(call(&bad, &cfg, &groups, &[true; 4])
        .message
        .contains("reference group"));
    bad = moments.clone();
    bad[3].sd = f64::NAN;
    assert!(call(&bad, &cfg, &groups, &[true; 4])
        .message
        .contains("finite"));
    bad = moments.clone();
    bad[3].free_mean = true;
    assert!(call(&bad, &cfg, &groups, &[false; 4])
        .message
        .contains("directly common"));
    let mut wrong = cfg;
    wrong.newton.gradient_tolerance = 1e-6;
    assert!(call(&moments, &wrong, &groups, &[true; 4])
        .message
        .contains("1e-9"));
    wrong = cfg;
    wrong.em.estimate_primary_correlation = true;
    assert!(call(&moments, &wrong, &groups, &[true; 4])
        .message
        .contains("identity Phi"));
    assert!(call(&moments, &cfg, &groups[..31], &[true; 4])
        .message
        .contains("geometry"));
    assert!(call(&moments, &cfg, &vec![0; 32], &[true; 4])
        .message
        .contains("no persons"));
}

#[test]
#[ignore = "actual GPU required under the shared host lease"]
fn concurrent_group_sweep_matches_native_moments_with_missing_items() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    super::concurrent::synthetic_group_sweep_oracle();
}

fn strict_control(iterations: usize, ridge: f64) -> crate::gpu_mstep::NewtonControl {
    crate::gpu_mstep::NewtonControl {
        iterations,
        ridge,
        backtracks: 25,
        gradient_tolerance: 1e-9,
        step_tolerance: 1e-9,
        pivot_tolerance: 1e-14,
        armijo: 1e-4,
    }
}
