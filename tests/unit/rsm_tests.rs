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

fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Draw an RSM category for ability `theta`, location `delta`, thresholds `tau`.
fn draw_rsm(theta: f64, delta: f64, tau: &[f64], u: f64) -> usize {
    let lp = rsm_logprobs(theta, delta, tau);
    let mut cum = 0.0;
    for (k, l) in lp.iter().enumerate() {
        cum += l.exp();
        if u < cum {
            return k;
        }
    }
    lp.len() - 1
}

#[test]
fn rsm_k2_reduces_to_rasch() {
    // K=2: single threshold, centered to 0, so P(X=1) = sigmoid(theta - delta).
    let tau = [0.0f64];
    for ti in -20..=20 {
        for di in -10..=10 {
            let theta = ti as f64 * 0.3;
            let delta = di as f64 * 0.4;
            let lp = rsm_logprobs(theta, delta, &tau);
            assert!((lp[0] - log_sigmoid(-(theta - delta))).abs() < 1e-12);
            assert!((lp[1] - log_sigmoid(theta - delta)).abs() < 1e-12);
        }
    }
}

#[test]
fn rsm_probs_sum_to_one() {
    let tau = [0.7f64, -0.2, -0.5]; // K=4
    for ti in -20..=20 {
        let theta = ti as f64 * 0.3;
        let s: f64 = rsm_logprobs(theta, 0.3, &tau).iter().map(|l| l.exp()).sum();
        assert!((s - 1.0).abs() < 1e-12, "sum {s}");
    }
}

#[test]
fn rsm_recovers_params() {
    let (n_items, n_cat, n) = (12usize, 5usize, 2500usize);
    let delta_true: Vec<f64> = (0..n_items).map(|i| -1.2 + 0.2 * i as f64).collect();
    let tau_true = vec![0.9f64, 0.2, -0.3, -0.8]; // sum = 0
    let mut rng = Lcg(1978);
    let mut y = vec![0usize; n * n_items];
    let mut thetas = vec![0.0f64; n];
    for p in 0..n {
        let theta = rng.normal();
        thetas[p] = theta;
        for i in 0..n_items {
            y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
        }
    }
    let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-7).unwrap();
    assert!(res.converged);
    // ECM ascends the marginal loglik monotonically (backtracked M-steps).
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "loglik decreased {} -> {}", w[0], w[1]);
    }
    assert_eq!(res.n_parameters, n_items + n_cat - 2);
    assert!(
        (res.thresholds.iter().sum::<f64>()).abs() < 1e-6,
        "tau not centered"
    );
    assert!(
        rmse(&res.item_location, &delta_true) < 0.15,
        "delta RMSE {}",
        rmse(&res.item_location, &delta_true)
    );
    assert!(
        rmse(&res.thresholds, &tau_true) < 0.12,
        "tau RMSE {}",
        rmse(&res.thresholds, &tau_true)
    );
    assert!(
        corr(&res.theta, &thetas) > 0.85,
        "theta corr {}",
        corr(&res.theta, &thetas)
    );
}

/// Data generated with NON-centered thresholds must be recovered as the centered
/// equivalent (tau - mean, delta + mean). This exercises the re-centering sign:
/// a wrong sign shifts the model and breaks recovery.
#[test]
fn rsm_centers_noncentered_truth() {
    let (n_items, n_cat, n) = (10usize, 4usize, 2500usize);
    let delta_gen: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.15 * i as f64).collect();
    let tau_gen = vec![1.0f64, 0.5, -0.3]; // sum = 1.2, NOT centered
    let shift = tau_gen.iter().sum::<f64>() / (n_cat - 1) as f64; // 0.4
    let tau_expect: Vec<f64> = tau_gen.iter().map(|t| t - shift).collect();
    let delta_expect: Vec<f64> = delta_gen.iter().map(|d| d + shift).collect();
    let mut rng = Lcg(4242);
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        let theta = rng.normal();
        for i in 0..n_items {
            y[p * n_items + i] = draw_rsm(theta, delta_gen[i], &tau_gen, rng.f64());
        }
    }
    let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-7).unwrap();
    assert!(res.converged);
    assert!((res.thresholds.iter().sum::<f64>()).abs() < 1e-6);
    assert!(
        rmse(&res.thresholds, &tau_expect) < 0.12,
        "tau RMSE {}",
        rmse(&res.thresholds, &tau_expect)
    );
    assert!(
        rmse(&res.item_location, &delta_expect) < 0.15,
        "delta RMSE {}",
        rmse(&res.item_location, &delta_expect)
    );
}

#[test]
fn rsm_handles_missing_data() {
    let (n_items, n_cat, n) = (8usize, 4usize, 800usize);
    let delta_true = vec![-0.5f64, 0.0, 0.5, -0.3, 0.3, -0.6, 0.6, 0.1];
    let tau_true = vec![0.5f64, 0.0, -0.5];
    let mut rng = Lcg(55);
    let mut y = vec![0usize; n * n_items];
    let mut observed = vec![true; n * n_items];
    for p in 0..n {
        let theta = rng.normal();
        for i in 0..n_items {
            y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
            if rng.f64() < 0.15 {
                observed[p * n_items + i] = false;
            }
        }
    }
    let res = fit_rsm(&y, Some(&observed), n, n_items, n_cat, 21, 400, 1e-6).unwrap();
    assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
}

#[test]
fn rsm_validate_rejects_malformed() {
    assert!(fit_rsm(&[0, 1], None, 1, 2, 1, 21, 10, 1e-6).is_err()); // n_cat<2
    assert!(fit_rsm(&[0, 1, 2], None, 1, 2, 3, 21, 10, 1e-6).is_err()); // wrong len
    assert!(fit_rsm(&[0, 9], None, 1, 2, 3, 21, 10, 1e-6).is_err()); // category out of range
    assert!(fit_rsm(&[0, 1, 0, 1], None, 2, 2, 2, 99, 10, 1e-6).is_err()); // bad q
    assert!(fit_rsm(&[], None, 0, 1, 2, 21, 10, 1e-6).is_err()); // no persons
    assert!(fit_rsm(&[], None, 1, 0, 2, 21, 10, 1e-6).is_err()); // no items
    assert!(fit_rsm(&[0, 1], None, 1, 2, 2, 21, 0, 1e-6).is_err()); // no iterations
    assert!(fit_rsm(&[0, 1], None, 1, 2, RSM_MAX_CAT + 1, 21, 10, 1e-6).is_err());
    assert!(fit_rsm(&[0, 1], None, 1, 2, 2, 21, RSM_MAX_ITER + 1, 1e-6).is_err());
    assert!(fit_rsm(&[0, 1], None, 1, 2, 2, 21, 10, f64::INFINITY).is_err());
    let observed = [true, false, true, false];
    assert!(fit_rsm(&[0, 0, 1, 0], Some(&observed), 2, 2, 2, 21, 10, 1e-6).is_err());
    assert!(fit_rsm(&[0, 1], Some(&[true]), 1, 2, 2, 21, 10, 1e-6).is_err());

    let limited = fit_rsm(&[0, 1, 1, 0], None, 2, 2, 2, 21, 1, 1e-14).unwrap();
    assert!(!limited.converged);
    assert_eq!(limited.n_iter, 1);
    assert_eq!(limited.loglik_trace.len(), 2);

    let zero_gradient = tau_gradient(&[0.0], &[0.0], &[vec![0.0; 2]], &[0.0], 1, 2);
    assert_eq!(zero_gradient, vec![0.0]);
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_rsm_recovery_500() {
    let (n_items, n_cat, n, reps) = (12usize, 5usize, 1000usize, 500usize);
    let delta_true: Vec<f64> = (0..n_items).map(|i| -1.1 + 0.2 * i as f64).collect();
    let tau_true = vec![0.9f64, 0.2, -0.3, -0.8];
    for &skew in [false, true].iter() {
        let (mut rd, mut rt, mut bd, mut bt, mut nconv, mut tcorr) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0.0f64);
        let (mut max_n_iter, mut worst_stop_ratio, mut worst_delta, mut worst_tolerance) =
            (0usize, 0.0f64, 0.0f64, 0.0f64);
        for rep in 0..reps {
            let mut rng = Lcg(0xB5297A4Du64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15));
            let mut y = vec![0usize; n * n_items];
            let mut thetas = vec![0.0f64; n];
            for p in 0..n {
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
                thetas[p] = theta;
                for i in 0..n_items {
                    y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
                }
            }
            let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-6).unwrap();
            assert!(
                res.loglik_trace.iter().all(|value| value.is_finite()),
                "non-finite likelihood trace at rep={rep} skew={skew}"
            );
            for window in res.loglik_trace.windows(2) {
                assert!(
                    window[1] >= window[0] - 1e-6,
                    "likelihood decreased {} -> {} at rep={rep} skew={skew}",
                    window[0],
                    window[1]
                );
            }
            assert!(
                res.converged,
                "RSM did not converge at rep={rep} skew={skew}: n_iter={}",
                res.n_iter
            );
            assert!(res.n_iter <= 500);
            let trace_len = res.loglik_trace.len();
            let final_delta =
                (res.loglik_trace[trace_len - 1] - res.loglik_trace[trace_len - 2]).abs();
            let stopping_tolerance = 1e-6 * (1.0 + res.loglik_trace[trace_len - 2].abs());
            assert!(
                final_delta < stopping_tolerance,
                "stopping metric {final_delta} did not meet tolerance {stopping_tolerance} at rep={rep} skew={skew}"
            );
            let stop_ratio = final_delta / stopping_tolerance;
            if stop_ratio > worst_stop_ratio {
                worst_stop_ratio = stop_ratio;
                worst_delta = final_delta;
                worst_tolerance = stopping_tolerance;
            }
            max_n_iter = max_n_iter.max(res.n_iter);
            if res.converged {
                nconv += 1;
            }
            rd += rmse(&res.item_location, &delta_true) / reps as f64;
            rt += rmse(&res.thresholds, &tau_true) / reps as f64;
            bd += (res.item_location.iter().sum::<f64>() - delta_true.iter().sum::<f64>())
                / n_items as f64
                / reps as f64;
            bt += (res.thresholds.iter().sum::<f64>()) / reps as f64;
            tcorr += corr(&res.theta, &thetas) / reps as f64;
        }
        println!(
            "[RSM MC skew={skew}] reps={reps} conv={:.2} max_iter={max_n_iter}/500 \
             worst_stop={worst_delta:.6}/{worst_tolerance:.6} ratio={worst_stop_ratio:.3} \
             RMSE(delta)={:.3} RMSE(tau)={:.3} bias(delta)={:.3} sum(tau)={:.4} theta-corr={:.3}",
            nconv as f64 / reps as f64,
            rd,
            rt,
            bd,
            bt,
            tcorr
        );
        assert_eq!(nconv, reps, "not every RSM fit converged for skew={skew}");
        assert!(rd < 0.12, "RMSE(delta) {rd} skew={skew}");
        assert!(rt < 0.1, "RMSE(tau) {rt} skew={skew}");
        assert!(tcorr > 0.85, "theta corr {tcorr} skew={skew}");
    }
}
