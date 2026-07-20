use super::*;

struct Lcg(u64);
impl Lcg {
    fn f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.f64().max(1e-12);
        let u2 = self.f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}
fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Simulate CRM data: X = a*theta + d + sigma*eps, Z = logistic(X).
#[allow(clippy::too_many_arguments)]
fn simulate_crm(
    a: &[f64],
    d: &[f64],
    sigma: &[f64],
    n: usize,
    n_items: usize,
    skew: bool,
    rng: &mut Lcg,
) -> (Vec<f64>, Vec<f64>) {
    let mut z = vec![0.0f64; n * n_items];
    let mut thetas = vec![0.0f64; n];
    for j in 0..n {
        let theta = if skew {
            let mut c = 0.0;
            for _ in 0..3 {
                let g = rng.normal();
                c += g * g;
            }
            (c - 3.0) / (6.0_f64).sqrt()
        } else {
            rng.normal()
        };
        thetas[j] = theta;
        for i in 0..n_items {
            let xij = a[i] * theta + d[i] + sigma[i] * rng.normal();
            z[j * n_items + i] = 1.0 / (1.0 + (-xij).exp());
        }
    }
    (z, thetas)
}

/// Unit test of the closed-form WLS + residual formula against a hand solve.
#[test]
fn crm_wls_matches_direct_solve() {
    // Three (theta, X) points with unit posterior weight -> ordinary least squares.
    let th = [-1.0f64, 0.0, 1.0];
    let xv = [0.2f64, 0.5, 1.4];
    let (mut s1, mut sth, mut sthth, mut sx, mut sxth, mut sxx) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for k in 0..3 {
        s1 += 1.0;
        sth += th[k];
        sthth += th[k] * th[k];
        sx += xv[k];
        sxth += xv[k] * th[k];
        sxx += xv[k] * xv[k];
    }
    let det = sthth * s1 - sth * sth;
    let a = (sxth * s1 - sth * sx) / det;
    let dd = (sthth * sx - sth * sxth) / det;
    // OLS slope = cov(theta,X)/var(theta); with theta mean 0: a = sxth/sthth
    assert!((a - sxth / sthth).abs() < 1e-12);
    // intercept = mean(X) - a*mean(theta) = mean(X) (theta mean 0)
    assert!((dd - sx / s1).abs() < 1e-12);
    let resid = (sxx - a * sxth - dd * sx) / s1;
    // residual = mean((X - a*theta - d)^2)
    let direct: f64 = (0..3)
        .map(|k| (xv[k] - a * th[k] - dd).powi(2))
        .sum::<f64>()
        / 3.0;
    assert!((resid - direct).abs() < 1e-12, "{resid} vs {direct}");
}

#[test]
fn crm_private_numeric_contracts_cover_all_defensive_outcomes() {
    assert_eq!(contextualize_crm_update(Ok(None), 3).unwrap(), None);
    assert_eq!(
        contextualize_crm_update(Err("singular update".to_owned()), 3).unwrap_err(),
        "singular update for item 3"
    );
    assert!(checked_crm_delta(f64::NAN, None, 1e-6).is_err());
    assert_eq!(checked_crm_delta(-10.0, None, 1e-6).unwrap(), None);
    assert!(checked_crm_delta(-11.0, Some(-10.0), 1e-6).is_err());
    let (_, tolerance, converged) = checked_crm_delta(-9.999_999, Some(-10.0), 1e-6)
        .unwrap()
        .unwrap();
    assert_eq!(tolerance, 11e-6);
    assert!(converged);
    assert!(
        !checked_crm_delta(-9.0, Some(-10.0), 1e-6)
            .unwrap()
            .unwrap()
            .2
    );

    let degenerate = CrmWlsStats {
        s1: 1.0,
        sth: 1.0,
        sthth: 1.0,
        sx: 1.0,
        sxth: 1.0,
        sxx: 1.0,
    };
    assert!(crm_wls_update(degenerate, 1e-6).unwrap().is_none());
    let nonfinite = CrmWlsStats {
        sthth: 2.0,
        sxx: f64::INFINITY,
        ..degenerate
    };
    assert!(crm_wls_update(nonfinite, 1e-6).is_err());

    let mut loadings = [-1.0, -2.0];
    reflect_crm_loadings(&mut loadings);
    assert_eq!(loadings, [1.0, 2.0]);
    reflect_crm_loadings(&mut loadings);
    assert_eq!(loadings, [1.0, 2.0]);
    assert!(crm_difficulty(0.0, 2.0).is_nan());
    assert_eq!(crm_difficulty(2.0, -4.0), 2.0);
}

/// Continuous responses are highly informative, so the model recovers the item
/// parameters, the Samejima re-parameterization, and the trait well.
#[test]
fn crm_recovers_params() {
    let (n_items, n) = (15usize, 1500usize);
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.05 * i as f64).collect();
    let d_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.08 * i as f64).collect();
    let sigma_true: Vec<f64> = (0..n_items).map(|i| 0.6 + 0.02 * (i % 5) as f64).collect();
    let mut rng = Lcg(73);
    let (z, thetas) = simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, false, &mut rng);
    let observed = vec![true; n * n_items];
    let res = fit_crm(&z, &observed, n, n_items, 41, 500, 1e-7).unwrap();
    assert!(res.converged);
    assert_eq!(res.termination_reason, "tolerance");
    assert!(res.final_delta <= res.stopping_tolerance);
    assert_eq!(res.n_iter + 1, res.loglik_trace.len());
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "loglik decreased {} -> {}", w[0], w[1]);
    }
    assert_eq!(res.n_parameters, 3 * n_items);
    assert!(
        rmse(&res.slope, &a_true) < 0.15,
        "a RMSE {}",
        rmse(&res.slope, &a_true)
    );
    assert!(
        rmse(&res.intercept, &d_true) < 0.1,
        "d RMSE {}",
        rmse(&res.intercept, &d_true)
    );
    assert!(
        rmse(&res.resid_sd, &sigma_true) < 0.1,
        "sigma RMSE {}",
        rmse(&res.resid_sd, &sigma_true)
    );
    assert!(res.slope.iter().all(|&x| x > 0.0)); // reflection convention
                                                 // Samejima re-parameterization recovers the generating discrimination/difficulty.
    let alpha_true: Vec<f64> = (0..n_items).map(|i| a_true[i] / sigma_true[i]).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -d_true[i] / a_true[i]).collect();
    assert!(rmse(&res.discrimination, &alpha_true) < 0.3, "alpha RMSE");
    assert!(rmse(&res.difficulty, &b_true) < 0.2, "b RMSE");
    // trait recovery (continuous responses are information-rich)
    assert!(
        corr(&res.theta, &thetas) > 0.9,
        "theta corr {}",
        corr(&res.theta, &thetas)
    );
}

#[test]
fn crm_handles_missing_data() {
    let (n_items, n) = (8usize, 600usize);
    let a_true = vec![1.0f64; n_items];
    let d_true = vec![0.0f64; n_items];
    let sigma_true = vec![0.7f64; n_items];
    let mut rng = Lcg(9);
    let (z, _t) = simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, false, &mut rng);
    let mut observed = vec![true; n * n_items];
    for o in observed.iter_mut() {
        if rng.f64() < 0.2 {
            *o = false;
        }
    }
    let res = fit_crm(&z, &observed, n, n_items, 21, 400, 1e-6).unwrap();
    assert!(
        res.converged,
        "{} after {} iterations",
        res.termination_reason, res.n_iter
    );
    assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
    assert!(res.resid_sd.iter().all(|&s| s > 0.0));
}

#[test]
fn crm_validate_rejects_malformed() {
    assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 3, 21, 10, 1e-6).is_err()); // wrong len
    assert!(fit_crm(&[0.5, 1.5], &[true, true], 1, 2, 21, 10, 1e-6).is_err()); // out of (0,1)
    assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 2, 99, 10, 1e-6).is_err()); // bad q
    assert!(fit_crm(&[], &[], 0, 2, 21, 10, 1e-6).is_err()); // no persons
    assert!(fit_crm(&[], &[], 2, 0, 21, 10, 1e-6).is_err()); // no items
    assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 2, 21, 0, 1e-6).is_err()); // no iterations
    assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 2, 21, 10, f64::NAN).is_err());
    assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 2, 21, 10, 0.0).is_err());
    assert!(fit_crm(&[0.5, 0.5], &[true, false], 1, 2, 21, 10, 1e-6).is_err());
    assert!(fit_crm(&[], &[], usize::MAX, 2, 21, 10, 1e-6).is_err());
    assert!(fit_crm(&[0.5, 0.5], &[true], 1, 2, 21, 10, 1e-6).is_err());
    assert!(fit_crm(&[], &[], 1, usize::MAX, 21, 10, 1e-6).is_err());
}

#[test]
fn crm_reports_iteration_limit_without_false_success() {
    let z = [0.2, 0.7, 0.4, 0.8, 0.6, 0.3, 0.9, 0.5];
    let observed = [true; 8];
    let res = fit_crm(&z, &observed, 4, 2, 21, 1, 1e-12).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination_reason, "max_iter");
    assert_eq!(res.n_iter, 1);
    assert_eq!(res.loglik_trace.len(), 2);
    assert!(res.final_delta > res.stopping_tolerance);
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_crm_recovery_500() {
    let (n_items, n, reps) = (15usize, 500usize, 500usize);
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.05 * i as f64).collect();
    let d_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.08 * i as f64).collect();
    let sigma_true: Vec<f64> = (0..n_items).map(|i| 0.6 + 0.02 * (i % 5) as f64).collect();
    for &skew in [false, true].iter() {
        let (mut ra, mut rd, mut rs, mut ba, mut nconv, mut tcorr) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0.0f64);
        for rep in 0..reps {
            let mut rng = Lcg(0x5DEECE66Du64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15));
            let (z, thetas) =
                simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, skew, &mut rng);
            let observed = vec![true; n * n_items];
            let res = fit_crm(&z, &observed, n, n_items, 41, 500, 1e-6).unwrap();
            assert!(
                res.converged,
                "CRM did not converge: skew={skew} rep={rep} reason={} n_iter={} final_delta={} tol={}",
                res.termination_reason,
                res.n_iter,
                res.final_delta,
                res.stopping_tolerance
            );
            if res.converged {
                nconv += 1;
            }
            ra += rmse(&res.slope, &a_true) / reps as f64;
            rd += rmse(&res.intercept, &d_true) / reps as f64;
            rs += rmse(&res.resid_sd, &sigma_true) / reps as f64;
            ba += (res.slope.iter().sum::<f64>() - a_true.iter().sum::<f64>())
                / n_items as f64
                / reps as f64;
            tcorr += corr(&res.theta, &thetas) / reps as f64;
        }
        println!(
            "[CRM MC skew={skew}] reps={reps} conv={:.2} RMSE(a)={:.3} RMSE(d)={:.3} \
             RMSE(sigma)={:.3} bias(a)={:.3} theta-corr={:.3}",
            nconv as f64 / reps as f64,
            ra,
            rd,
            rs,
            ba,
            tcorr
        );
        assert!(ra < 0.15, "RMSE(a) {ra} skew={skew}");
        assert!(rd < 0.12, "RMSE(d) {rd} skew={skew}");
        assert!(rs < 0.1, "RMSE(sigma) {rs} skew={skew}");
        assert!(tcorr > 0.9, "theta corr {tcorr} skew={skew}");
    }
}
