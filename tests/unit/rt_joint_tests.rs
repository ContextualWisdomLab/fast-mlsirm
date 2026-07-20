use super::*;

fn lcg(seed: u64) -> impl FnMut() -> f64 {
    let mut st = seed.max(1);
    move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    }
}
fn normal(u: &mut impl FnMut() -> f64) -> f64 {
    let u1 = u().max(1e-12);
    let u2 = u();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
#[test]
fn fixed_variance_m_step_maximizes_conditional_q() {
    let (s11, s12, s22, sigma_tau2) = (0.8_f64, 0.1_f64, 0.2_f64, 0.09_f64);
    let got = maximize_fixed_variance_covariance(s11, s12, s22, sigma_tau2, 0.999);
    let naive = s12 / s11;
    let got_q = covariance_q(got, s11, s12, s22, sigma_tau2);
    let naive_q = covariance_q(naive, s11, s12, s22, sigma_tau2);
    assert!(
        got_q > naive_q + 1e-6,
        "fixed-variance optimum {got_q} must beat S12/S11 {naive_q}"
    );

    let h = 1e-6;
    let numeric_score = (covariance_q(got + h, s11, s12, s22, sigma_tau2)
        - covariance_q(got - h, s11, s12, s22, sigma_tau2))
        / (2.0 * h);
    assert!(
        numeric_score.abs() < 1e-6,
        "fixed-variance score {numeric_score}"
    );
}

#[test]
fn covariance_helpers_cover_invalid_and_all_cubic_root_shapes() {
    assert_eq!(covariance_q(1.0, 1.0, 0.0, 1.0, 1.0), f64::NEG_INFINITY);
    let one = cubic_real_roots(0.0, 1.0, 1.0);
    let repeated = cubic_real_roots(0.0, -3.0, 2.0);
    let three = cubic_real_roots(0.0, -3.0, 0.0);
    assert_eq!(one.len(), 1);
    assert_eq!(repeated.len(), 2);
    assert_eq!(three.len(), 3);
    for (qa, qb, qc, roots) in [
        (0.0, 1.0, 1.0, &one),
        (0.0, -3.0, 2.0, &repeated),
        (0.0, -3.0, 0.0, &three),
    ] {
        assert!(roots
            .iter()
            .all(|&root| (root * root * root + qa * root * root + qb * root + qc).abs() < 1e-10));
    }
    assert_eq!(
        maximize_fixed_variance_covariance(1.0, 0.0, 1.0, -1.0, 0.9),
        0.0
    );
    assert!(joint_summary_is_finite(0.0, 1.0, &[0.0], &[0.0]));
    assert!(!joint_summary_is_finite(f64::NAN, 1.0, &[0.0], &[0.0]));
    assert!(!joint_summary_is_finite(0.0, f64::INFINITY, &[0.0], &[0.0]));
    assert!(!joint_summary_is_finite(0.0, 1.0, &[f64::NAN], &[0.0]));
    assert!(ensure_joint_summary_is_finite(0.0, 1.0, &[0.0], &[0.0]).is_ok());
    assert!(ensure_joint_summary_is_finite(f64::NAN, 1.0, &[0.0], &[0.0]).is_err());
}

#[test]
fn rejects_every_shape_data_and_control_boundary() {
    let response = [1.0];
    let time = [2.0];
    let observed = [true];
    let one = [1.0];
    let zero = [0.0];
    let base = SpeedAccuracyConfig::default();
    let call = |responses: &[f64],
                times: &[f64],
                mask: Option<&[bool]>,
                a: &[f64],
                b: &[f64],
                alpha: &[f64],
                beta: &[f64],
                n_persons: usize,
                n_items: usize,
                config: SpeedAccuracyConfig| {
        fit_speed_accuracy_covariance(
            responses, times, mask, a, b, alpha, beta, n_persons, n_items, config,
        )
    };

    assert!(call(&[], &[], None, &[], &[], &[], &[], 0, 1, base).is_err());
    assert!(call(&[], &[], None, &[], &[], &[], &[], usize::MAX, 2, base).is_err());
    assert!(call(&[], &time, None, &one, &zero, &one, &zero, 1, 1, base).is_err());
    assert!(call(&response, &time, None, &[], &zero, &one, &zero, 1, 1, base).is_err());
    assert!(call(
        &response,
        &time,
        Some(&[]),
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        base
    )
    .is_err());
    assert!(call(
        &response,
        &time,
        Some(&observed),
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        SpeedAccuracyConfig {
            max_iter: 0,
            ..base
        },
    )
    .is_err());
    assert!(call(
        &response,
        &time,
        None,
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        SpeedAccuracyConfig {
            rho_floor: 1.0,
            ..base
        },
    )
    .is_err());
    assert!(call(
        &response,
        &time,
        None,
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        SpeedAccuracyConfig {
            sigma_floor: 0.0,
            ..base
        },
    )
    .is_err());
    assert!(call(
        &response,
        &time,
        None,
        &[f64::NAN],
        &zero,
        &one,
        &zero,
        1,
        1,
        base
    )
    .is_err());
    assert!(call(
        &response,
        &time,
        None,
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        SpeedAccuracyConfig { q: 9, ..base }
    )
    .is_err());
    assert!(call(&[2.0], &time, None, &one, &zero, &one, &zero, 1, 1, base).is_err());
    assert!(call(
        &response,
        &[0.0],
        None,
        &one,
        &zero,
        &one,
        &zero,
        1,
        1,
        base
    )
    .is_err());
    assert!(call(
        &response,
        &[f64::MIN_POSITIVE],
        None,
        &one,
        &zero,
        &[1e154],
        &zero,
        1,
        1,
        SpeedAccuracyConfig {
            q: 7,
            max_iter: 1,
            ..base
        },
    )
    .is_err());
}

#[test]
fn fixed_sigma_full_fit_executes_the_constrained_m_step() {
    let fit = fit_speed_accuracy_covariance(
        &[1.0, 0.0, 0.0, 1.0],
        &[1.2, 0.8, 1.1, 0.9],
        None,
        &[1.0, 0.8],
        &[0.0, 0.1],
        &[1.0, 1.2],
        &[0.0, 0.0],
        2,
        2,
        SpeedAccuracyConfig {
            q: 7,
            max_iter: 2,
            fix_sigma_tau: Some(0.3),
            ..SpeedAccuracyConfig::default()
        },
    )
    .unwrap();
    assert_eq!(fit.sigma_tau, 0.3);
    assert_eq!(fit.n_iter, 2);
    assert!(fit.converged);
    assert_eq!(fit.termination_reason, "converged");
    assert!(fit.final_loglik_change <= SpeedAccuracyConfig::default().tol);
}

#[test]
fn rejects_invalid_item_parameters_and_controls() {
    let responses = [1.0];
    let times = [2.0];
    let a = [1.0];
    let b = [0.0];
    let beta = [1.0];
    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &a,
        &b,
        &[0.0],
        &beta,
        1,
        1,
        SpeedAccuracyConfig::default(),
    )
    .unwrap_err();
    assert!(err.contains("alpha"));

    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &a,
        &b,
        &[1.0],
        &beta,
        1,
        1,
        SpeedAccuracyConfig {
            tol: f64::NAN,
            ..SpeedAccuracyConfig::default()
        },
    )
    .unwrap_err();
    assert!(err.contains("tol"));

    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &a,
        &b,
        &[1.0],
        &beta,
        1,
        1,
        SpeedAccuracyConfig {
            fix_sigma_tau: Some(1e308),
            ..SpeedAccuracyConfig::default()
        },
    )
    .unwrap_err();
    assert!(err.contains("fix_sigma_tau"));
}

#[test]
fn rejects_unidentified_or_nonfinite_joint_calibrations() {
    let responses = [1.0];
    let times = [2.0];
    let observed = [false];
    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        Some(&observed),
        &[1.0],
        &[0.0],
        &[1.0],
        &[1.0],
        1,
        1,
        SpeedAccuracyConfig::default(),
    )
    .unwrap_err();
    assert!(err.contains("observed"));

    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &[0.0],
        &[0.0],
        &[1.0],
        &[1.0],
        1,
        1,
        SpeedAccuracyConfig::default(),
    )
    .unwrap_err();
    assert!(err.contains("discrimination"));

    let err = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &[1.0],
        &[0.0],
        &[1e308],
        &[1.0],
        1,
        1,
        SpeedAccuracyConfig::default(),
    )
    .unwrap_err();
    assert!(err.contains("non-finite"));
}

// Anchor A: at rho=0 the 2-D grid log-likelihood factorizes into the sum of the
// two 1-D grid log-likelihoods (certifies the Cholesky map, tensor weights, and
// logsumexp wiring exactly).
#[test]
fn joint_rho0_factorizes() {
    let (nodes, weights) = gh_rule(21).unwrap();
    let q = nodes.len();
    let lnw: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    // one 3-item person: accuracy la[a], and RT stats
    let a = [1.0_f64, 1.3, 0.8];
    let b = [0.2_f64, -0.4, 0.1];
    let alpha = [1.5_f64, 2.0, 1.1];
    let beta = [4.0_f64, 3.6, 4.2];
    let u = [1.0_f64, 0.0, 1.0];
    let y = [3.8_f64, 3.9, 4.5];
    let sig = 0.35_f64;
    let mut la = vec![0.0_f64; q];
    let (mut aj, mut bj, mut cj, mut kj) = (0.0, 0.0, 0.0, 0.0);
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    for i in 0..3 {
        for (ai, &z) in nodes.iter().enumerate() {
            let eta = a[i] * z + b[i];
            la[ai] += if u[i] > 0.5 {
                log_sigmoid(eta)
            } else {
                log_sigmoid(-eta)
            };
        }
        let a2 = alpha[i] * alpha[i];
        let d = y[i] - beta[i];
        aj += a2;
        bj += a2 * d;
        cj += a2 * d * d;
        kj += alpha[i].ln() - 0.5 * ln2pi;
    }
    // 2-D logsumexp at rho=0 (c=0, l22=sigma_tau)
    let mut mx = f64::NEG_INFINITY;
    let mut grid = vec![0.0_f64; q * q];
    for ai in 0..q {
        for bi in 0..q {
            let tau = sig * nodes[bi];
            let lt = kj - 0.5 * (aj * tau * tau + 2.0 * bj * tau + cj);
            let v = lnw[ai] + la[ai] + lnw[bi] + lt;
            grid[ai * q + bi] = v;
            if v > mx {
                mx = v;
            }
        }
    }
    let joint = mx + grid.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
    // two 1-D logsumexps
    let mxa = (0..q)
        .map(|ai| lnw[ai] + la[ai])
        .fold(f64::NEG_INFINITY, f64::max);
    let la1 = mxa
        + (0..q)
            .map(|ai| (lnw[ai] + la[ai] - mxa).exp())
            .sum::<f64>()
            .ln();
    let ltv: Vec<f64> = (0..q)
        .map(|bi| {
            let tau = sig * nodes[bi];
            lnw[bi] + kj - 0.5 * (aj * tau * tau + 2.0 * bj * tau + cj)
        })
        .collect();
    let mxb = ltv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lt1 = mxb + ltv.iter().map(|&v| (v - mxb).exp()).sum::<f64>().ln();
    assert!(
        (joint - (la1 + lt1)).abs() < 1e-10,
        "rho=0 factorization: {joint} vs {}",
        la1 + lt1
    );
}

// Anchor B/D + recovery: simulate under a known Sigma_P and recover (rho,
// sigma_tau) with the item banks frozen.
fn sim_and_fit(seed: u64, n: usize, rho_true: f64, sig_true: f64) -> SpeedAccuracyFit {
    let ni = 20usize;
    let a: Vec<f64> = (0..ni).map(|i| 0.9 + 0.6 * (i % 3) as f64 / 2.0).collect();
    let b: Vec<f64> = (0..ni)
        .map(|i| -1.5 + 3.0 * i as f64 / (ni - 1) as f64)
        .collect();
    let alpha: Vec<f64> = (0..ni)
        .map(|i| 1.0 + 2.0 * i as f64 / (ni - 1) as f64)
        .collect();
    let beta: Vec<f64> = (0..ni)
        .map(|i| 3.5 + 1.0 * i as f64 / (ni - 1) as f64)
        .collect();
    let mut u = lcg(seed);
    let mut resp = vec![0.0_f64; n * ni];
    let mut times = vec![0.0_f64; n * ni];
    let l22 = sig_true * (1.0 - rho_true * rho_true).sqrt();
    for p in 0..n {
        let za = normal(&mut u);
        let zb = normal(&mut u);
        let theta = za;
        let tau = rho_true * sig_true * za + l22 * zb;
        for i in 0..ni {
            let pr = 1.0 / (1.0 + (-(a[i] * theta + b[i])).exp());
            resp[p * ni + i] = if u() < pr { 1.0 } else { 0.0 };
            let ylog = beta[i] - tau + (1.0 / alpha[i]) * normal(&mut u);
            times[p * ni + i] = ylog.exp();
        }
    }
    fit_speed_accuracy_covariance(
        &resp,
        &times,
        None,
        &a,
        &b,
        &alpha,
        &beta,
        n,
        ni,
        SpeedAccuracyConfig::default(),
    )
    .unwrap()
}

#[test]
fn joint_recovers_rho_and_reduces_at_zero() {
    // Anchor D: recovery at rho=0.5
    let fit = sim_and_fit(11, 1000, 0.5, 0.3);
    assert!(fit.converged);
    assert_eq!(fit.termination_reason, "converged");
    let max_drop = fit
        .loglik_trace
        .windows(2)
        .map(|w| w[0] - w[1])
        .fold(f64::NEG_INFINITY, f64::max);
    let final_delta = fit.final_loglik_change;
    eprintln!(
        "[joint] converged={} n_iter={} trace len={} first={:.4} last={:.4} final_delta={:.12e} tol={:.12e} max_drop={:.3e}",
        fit.converged,
        fit.n_iter,
        fit.loglik_trace.len(),
        fit.loglik_trace[0],
        fit.loglik_trace.last().unwrap(),
        final_delta,
        SpeedAccuracyConfig::default().tol,
        max_drop
    );
    assert!(
        final_delta < SpeedAccuracyConfig::default().tol,
        "converged fit final delta {final_delta} exceeds tolerance"
    );
    assert!(
        fit.loglik_trace
            .windows(2)
            .all(|w| w[1] >= w[0] - 1e-6 * w[0].abs().max(1.0)),
        "loglik must be monotone (max drop {max_drop:.3e})"
    );
    assert!((fit.rho - 0.5).abs() < 0.1, "rho {}", fit.rho);
    assert!(
        (fit.sigma_tau - 0.3).abs() < 0.05,
        "sigma_tau {}",
        fit.sigma_tau
    );
    // Anchor B: true independence -> rho ~= 0
    let fit0 = sim_and_fit(12, 1000, 0.0, 0.3);
    assert!(fit0.converged);
    assert_eq!(fit0.termination_reason, "converged");
    assert!(fit0.final_loglik_change < SpeedAccuracyConfig::default().tol);
    assert!(
        fit0.rho.abs() < 0.08,
        "rho at independence should be ~0: {}",
        fit0.rho
    );
}

#[test]
fn joint_reports_max_iter_nonconvergence() {
    let ni = 4usize;
    let n = 20usize;
    let responses: Vec<f64> = (0..n * ni)
        .map(|idx| ((idx + idx / ni) % 2) as f64)
        .collect();
    let times: Vec<f64> = (0..n * ni)
        .map(|idx| 2.0 + (idx % ni) as f64 * 0.1)
        .collect();
    let fit = fit_speed_accuracy_covariance(
        &responses,
        &times,
        None,
        &vec![1.0; ni],
        &vec![0.0; ni],
        &vec![1.5; ni],
        &vec![1.0; ni],
        n,
        ni,
        SpeedAccuracyConfig {
            q: 7,
            max_iter: 1,
            ..SpeedAccuracyConfig::default()
        },
    )
    .unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.termination_reason, "max_iter_reached");
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.loglik_trace.len(), 2);
    assert!(fit.final_loglik_change.is_finite());
    assert!(fit.final_loglik_change >= SpeedAccuracyConfig::default().tol);
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn joint_monte_carlo_500() {
    let reps = 500usize;
    for &rho_true in &[0.0_f64, 0.5, -0.5] {
        let (mut sr, mut br, mut ss, mut bs, mut absr) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for r in 0..reps {
            let fit = sim_and_fit(200 + r as u64, 800, rho_true, 0.3);
            assert!(
                fit.converged,
                "replication {r} at rho={rho_true} exhausted {} iterations; final delta={}",
                fit.n_iter, fit.final_loglik_change
            );
            assert_eq!(fit.termination_reason, "converged");
            assert!(fit.final_loglik_change < SpeedAccuracyConfig::default().tol);
            sr += (fit.rho - rho_true).powi(2);
            br += fit.rho - rho_true;
            ss += (fit.sigma_tau - 0.3).powi(2);
            bs += fit.sigma_tau - 0.3;
            absr += fit.rho.abs();
        }
        let f = reps as f64;
        println!(
            "[joint 500] rho={rho_true}: RMSE(rho)={:.4} bias(rho)={:.4} RMSE(sigma)={:.4} \
             bias(sigma)={:.4} mean|rho|={:.4}",
            (sr / f).sqrt(),
            br / f,
            (ss / f).sqrt(),
            bs / f,
            absr / f
        );
        // provisional thresholds (retune after the first 500-rep run; with ~20
        // items the person-parameter measurement error inflates SD(rho_hat))
        assert!(
            (sr / f).sqrt() < 0.06,
            "rho RMSE too high: {}",
            (sr / f).sqrt()
        );
        assert!((br / f).abs() < 0.02, "rho bias too high: {}", br / f);
        assert!((bs / f).abs() < 0.05, "sigma_tau bias too high: {}", bs / f);
        if rho_true == 0.0 {
            // mean|rho_hat| ~ RMSE*sqrt(2/pi) ~ 0.033 for an unbiased estimator
            // (a dispersion sanity, not a bias check; bias(rho) above is the
            // real "recovers independence" anchor)
            assert!(absr / f < 0.05, "mean|rho| at rho=0: {}", absr / f);
        }
    }
}
