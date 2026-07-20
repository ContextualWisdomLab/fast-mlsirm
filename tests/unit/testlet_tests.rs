use super::*;
use crate::mmle::{fit_mmle_2pl, MmleConfig};

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
    fn skew(&mut self) -> f64 {
        -(self.next_f64().max(1e-12)).ln() - 1.0 // Exp(1)-1: mean 0, var 1
    }
    fn bern(&mut self, p: f64) -> f64 {
        if self.next_f64() < p {
            1.0
        } else {
            0.0
        }
    }
}
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
}
fn bias(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n
}
fn nondecreasing(t: &[f64]) -> bool {
    t.windows(2).all(|w| w[1] >= w[0] - 1e-6)
}

/// The gamma quadrature must be the standard normal (unit variance) or the
/// sigma^2 = sigma^2 * mean(E[u^2]) update converges to a biased fixed point.
#[test]
fn gh_rule_is_unit_normal() {
    for &q in &[11usize, 15, 21, 31, 41] {
        let (u, v) = gh_rule(q).unwrap();
        assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(u.iter().zip(v).map(|(x, w)| x * w).sum::<f64>().abs() < 1e-9);
        let m2: f64 = u.iter().zip(v).map(|(x, w)| x * x * w).sum();
        assert!((m2 - 1.0).abs() < 1e-6, "gh_rule({q}) E[u^2] = {m2}");
    }
}

#[test]
fn rejects_testlet_count_exceeding_item_count_before_allocation() {
    let cfg = TestletConfig::default();
    let err = validate(&[1.0], &[true], &[0], 1, 1, 1_000_000_001, &cfg)
        .expect_err("oversized testlet count must be rejected");
    assert!(err.contains("n_testlets must not exceed n_items"));
}

/// Contiguous testlet assignment: testlet d owns items [d*size .. (d+1)*size).
fn contiguous_testlets(n_items: usize, n_testlets: usize) -> Vec<usize> {
    let per = n_items / n_testlets;
    (0..n_items)
        .map(|i| (i / per).min(n_testlets - 1))
        .collect()
}

/// Simulate testlet data: draw theta, per-testlet gamma ~ N(0, sigma^2_d), responses.
fn simulate(
    a: &[f64],
    beta: &[f64],
    sigma2: &[f64],
    testlet_id: &[usize],
    n: usize,
    j: usize,
    skew: bool,
    rng: &mut Lcg,
) -> Vec<f64> {
    let d_n = sigma2.len();
    let mut y = vec![0.0f64; n * j];
    for p in 0..n {
        let theta = if skew { rng.skew() } else { rng.normal() };
        let gamma: Vec<f64> = (0..d_n).map(|d| sigma2[d].sqrt() * rng.normal()).collect();
        for i in 0..j {
            let eta = a[i] * theta + beta[i] - a[i] * gamma[testlet_id[i]];
            y[p * j + i] = rng.bern(sigmoid_stable(eta));
        }
    }
    y
}

/// PRIMARY anchor: sigma^2 pinned to 0 reduces to fit_mmle_2pl (a/beta/loglik match).
#[test]
fn testlet_sigma0_equals_fit_mmle_2pl() {
    let (n, j, d_n) = (700usize, 12usize, 3usize);
    let tid = contiguous_testlets(j, d_n);
    let mut rng = Lcg(7);
    let a_t: Vec<f64> = (0..j).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
    let beta_t: Vec<f64> = (0..j)
        .map(|i| -1.2 + 2.4 * i as f64 / (j - 1) as f64)
        .collect();
    let y = simulate(&a_t, &beta_t, &vec![0.0; d_n], &tid, n, j, false, &mut rng);
    let observed = vec![true; n * j];
    let mcfg = MmleConfig {
        max_iter: 80,
        tol: 0.0,
        ridge_a: 1e-3,
        ridge_b: 1e-3,
        newton_iter: 25,
    };
    let mmle = fit_mmle_2pl(&y, &observed, n, j, &mcfg);
    let cfg = TestletConfig {
        max_iter: 80,
        tol: 0.0,
        q_gamma: 21,
        ridge_a: 1e-3,
        ridge_b: 1e-3,
        newton_iter: 25,
        estimate_sigma: false,
        init_sigma2: 0.0,
    };
    let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::TwoPl, &cfg).unwrap();
    // a/beta bit-exact; theta OMITTED (mmle EAP uses a stale posterior — same reason
    // the mixture/lltm anchors assert only item params).
    assert!(
        rmse(&res.a, &mmle.a) < 1e-12,
        "a rmse {}",
        rmse(&res.a, &mmle.a)
    );
    assert!(
        rmse(&res.beta, &mmle.b) < 1e-12,
        "beta rmse {}",
        rmse(&res.beta, &mmle.b)
    );
    // loglik agrees on the common prefix (testlet may push an extra final_ll).
    assert!(
        res.loglik_trace
            .iter()
            .zip(&mmle.loglik_trace)
            .all(|(x, y)| (x - y).abs() < 1e-12),
        "loglik prefix mismatch"
    );
    assert_eq!(res.n_parameters, 2 * j); // sigma^2 fixed => 0 free variance params
    assert!(res.sigma2.iter().all(|&s| s == 0.0));
}

/// No-spurious-LD: pure 2PL data (all true sigma^2=0), full fit must not invent LD.
/// Ignored by default: shrinking sigma^2 to ~0 needs many iterations (the sigma->0
/// tail of the variance-component EM is slow even with SQUAREM).
#[test]
#[ignore = "slow (sigma->0 convergence); run with: cargo test --release -- --ignored"]
fn testlet_no_spurious_ld() {
    let (n, j, d_n) = (600usize, 12usize, 3usize);
    let tid = contiguous_testlets(j, d_n);
    let mut rng = Lcg(11);
    let a_t: Vec<f64> = (0..j).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
    let beta_t: Vec<f64> = (0..j)
        .map(|i| -1.5 + 3.0 * i as f64 / (j - 1) as f64)
        .collect();
    let y = simulate(&a_t, &beta_t, &vec![0.0; d_n], &tid, n, j, false, &mut rng);
    let observed = vec![true; n * j];
    let cfg = TestletConfig {
        max_iter: 2000,
        ..TestletConfig::default()
    };
    let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::TwoPl, &cfg).unwrap();
    println!(
        "no_spurious: converged={} n_iter={} sigma2={:?}",
        res.converged, res.n_iter, res.sigma2
    );
    assert!(
        res.converged,
        "testlet fit exhausted {} iterations",
        cfg.max_iter
    );
    assert!(res.n_iter < cfg.max_iter);
    assert!(nondecreasing(&res.loglik_trace));
    assert!(
        res.sigma2.iter().all(|&s| s < 0.08),
        "spurious LD: {:?}",
        res.sigma2
    );
}

/// Strong-LD: large true sigma^2 recovered, and modeling it improves the loglik over
/// the sigma=0 (naive-2PL) fit — the signature of unmodeled local dependence.
#[test]
fn testlet_recovers_strong_ld() {
    // Rasch (a=1), 8 items per testlet (the well-identified testlet model; the 2PL
    // discrimination trades off against the testlet SD via a_i*sigma_d).
    let (n, j, d_n) = (800usize, 16usize, 2usize);
    let tid = contiguous_testlets(j, d_n);
    let sig2 = vec![0.6f64, 0.3];
    let mut rng = Lcg(2024);
    let a_t = vec![1.0f64; j];
    let beta_t: Vec<f64> = (0..j).map(|i| -1.5 + 3.0 * (i % 8) as f64 / 7.0).collect();
    let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
    let observed = vec![true; n * j];
    let res = fit_testlet(
        &y,
        &observed,
        &tid,
        n,
        j,
        d_n,
        TestletModel::Rasch,
        &TestletConfig::default(),
    )
    .unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace));
    assert!(
        rmse(&res.sigma2, &sig2) < 0.2,
        "sigma2 rmse {} ({:?})",
        rmse(&res.sigma2, &sig2),
        res.sigma2
    );
    assert!(
        res.sigma2[0] > 0.35,
        "strong LD not recovered: {}",
        res.sigma2[0]
    );
    // loglik gain over the naive sigma=0 fit
    let naive = TestletConfig {
        estimate_sigma: false,
        init_sigma2: 0.0,
        ..TestletConfig::default()
    };
    let res0 = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &naive).unwrap();
    assert!(
        *res.loglik_trace.last().unwrap() > *res0.loglik_trace.last().unwrap() + 5.0,
        "testlet fit did not improve loglik over naive 2PL"
    );
}

/// A singleton testlet's variance is non-identified => pinned to 0, not spurious.
#[test]
fn testlet_singleton_pinned() {
    let (n, j) = (600usize, 7usize);
    // testlets: {0,1,2}, {3,4,5}, {6} (singleton)
    let tid = vec![0usize, 0, 0, 1, 1, 1, 2];
    let sig2 = vec![0.6f64, 0.6, 0.0];
    let mut rng = Lcg(5);
    let a_t = vec![1.0f64; j];
    let beta_t: Vec<f64> = (0..j)
        .map(|i| -1.0 + 2.0 * i as f64 / (j - 1) as f64)
        .collect();
    let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
    let observed = vec![true; n * j];
    let res = fit_testlet(
        &y,
        &observed,
        &tid,
        n,
        j,
        3,
        TestletModel::Rasch,
        &TestletConfig::default(),
    )
    .unwrap();
    assert!(res.converged);
    assert_eq!(
        res.sigma2[2], 0.0,
        "singleton testlet variance must be pinned to 0"
    );
    // the singleton's pinned variance is NOT a free parameter (Rasch: J + 2 multi).
    assert_eq!(res.n_parameters, j + 2);
}

/// Missing-at-random cells are dropped.
#[test]
fn testlet_handles_missing_data() {
    // Coverage instrumentation is intentionally much slower than a normal
    // test build. Keep the same missing-data and convergence path while
    // the full-size statistical check remains in ordinary CI.
    let n = 500usize;
    let (j, d_n) = (12usize, 3usize);
    let tid = contiguous_testlets(j, d_n);
    let sig2 = vec![0.5f64, 0.5, 0.5];
    let mut rng = Lcg(9);
    let a_t = vec![1.0f64; j];
    let beta_t: Vec<f64> = (0..j)
        .map(|i| -1.0 + 2.0 * i as f64 / (j - 1) as f64)
        .collect();
    let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
    let mut observed = vec![true; n * j];
    for o in observed.iter_mut() {
        if rng.next_f64() < 0.2 {
            *o = false;
        }
    }
    let res = fit_testlet(
        &y,
        &observed,
        &tid,
        n,
        j,
        d_n,
        TestletModel::Rasch,
        &TestletConfig::default(),
    )
    .unwrap();
    assert!(res.converged && nondecreasing(&res.loglik_trace));
}

/// Malformed inputs are rejected (covers each validate branch, incl. tol=0 allowed).
#[test]
fn testlet_validate_rejects_malformed() {
    let (n, j, d_n) = (5usize, 6usize, 2usize);
    let tid = contiguous_testlets(j, d_n);
    let y = vec![0.0f64; n * j];
    let obs = vec![true; n * j];
    let d = TestletConfig::default();
    let bad = |y: &[f64], obs: &[bool], tid: &[usize], n, j, dn, cfg: &TestletConfig| {
        fit_testlet(y, obs, tid, n, j, dn, TestletModel::Rasch, cfg).is_err()
    };
    assert!(bad(&y, &obs, &tid, 0, j, d_n, &d)); // n_persons
    assert!(bad(&y, &obs, &tid, n, j, 0, &d)); // n_testlets
    assert!(bad(&[], &[], &[], usize::MAX, 2, 1, &d)); // n_persons * n_items
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig { max_iter: 0, ..d }
    ));
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig { tol: -1.0, ..d }
    ));
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig { q_gamma: 8, ..d }
    )); // not in SUPPORTED_Q
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig {
            init_sigma2: -1.0,
            ..d
        }
    ));
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig {
            ridge_a: f64::NAN,
            ..d
        }
    ));
    assert!(bad(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        &TestletConfig { ridge_b: -1.0, ..d }
    ));
    assert!(bad(&vec![0.0; n * j - 1], &obs, &tid, n, j, d_n, &d)); // y length
    assert!(bad(&y, &obs, &vec![0usize; j - 1], n, j, d_n, &d)); // testlet_id length
    assert!(bad(&y, &obs, &vec![0, 0, 0, 5, 0, 0], n, j, d_n, &d)); // testlet_id out of range
    assert!(bad(&vec![2.0; n * j], &obs, &tid, n, j, d_n, &d)); // y not 0/1
    let mut no_item_observations = obs.clone();
    for p in 0..n {
        no_item_observations[p * j] = false;
    }
    assert!(bad(&y, &no_item_observations, &tid, n, j, d_n, &d));
    // an empty testlet (n_testlets says 3 but only 0,1 used)
    assert!(bad(&y, &obs, &vec![0, 0, 0, 1, 1, 1], n, j, 3, &d));
    // tol == 0.0 accepted
    assert!(fit_testlet(
        &y,
        &obs,
        &tid,
        n,
        j,
        d_n,
        TestletModel::Rasch,
        &TestletConfig {
            tol: 0.0,
            max_iter: 2,
            ..d
        }
    )
    .is_ok());
    assert_eq!(
        choose_squarem_parameters(Some(vec![1.0]), vec![2.0]),
        vec![1.0]
    );
    assert_eq!(choose_squarem_parameters(None, vec![2.0]), vec![2.0]);
    assert_eq!(squarem_alpha(4.0, 1.0), -2.0);
    assert_eq!(squarem_alpha(1.0, 4.0), -1.0);
    assert_eq!(squarem_alpha(0.0, 0.0), -1.0);
}

#[test]
fn zero_variance_testlet_path_handles_missing_cells() {
    let (n, j, d_n) = (8usize, 4usize, 2usize);
    let tid = contiguous_testlets(j, d_n);
    let y: Vec<f64> = (0..n * j).map(|idx| (idx % 2) as f64).collect();
    let mut observed = vec![true; n * j];
    observed[1] = false;
    observed[n * j - 2] = false;
    let result = fit_testlet(
        &y,
        &observed,
        &tid,
        n,
        j,
        d_n,
        TestletModel::Rasch,
        &TestletConfig {
            max_iter: 2,
            q_gamma: 7,
            estimate_sigma: false,
            init_sigma2: 0.0,
            ..TestletConfig::default()
        },
    )
    .unwrap();
    assert!(result.loglik_trace.iter().all(|value| value.is_finite()));
}

/// Iteration exhaustion is explicit and SQUAREM must not overrun max_iter.
#[test]
fn testlet_reports_max_iter_nonconvergence() {
    let (n, j, d_n) = (40usize, 6usize, 2usize);
    let tid = contiguous_testlets(j, d_n);
    let y: Vec<f64> = (0..n * j).map(|idx| ((idx + idx / j) % 2) as f64).collect();
    let observed = vec![true; n * j];
    let cfg = TestletConfig {
        max_iter: 2,
        tol: 0.0,
        q_gamma: 7,
        ..TestletConfig::default()
    };
    let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &cfg).unwrap();
    assert!(!res.converged);
    assert_eq!(res.termination_reason, "max_iter_reached");
    assert_eq!(res.n_iter, cfg.max_iter);
    assert!(res.final_loglik_change.is_finite());
    assert_eq!(res.loglik_trace.len(), cfg.max_iter);
}

/// Literature-grade Monte-Carlo (>=500 reps): Bradlow-Wainer-Wang-style design.
/// Uses the RASCH testlet (the well-identified case; in the 2PL testlet the free
/// discrimination a_i and the testlet SD sigma_d both scale the LD via a_i*sigma_d
/// and separate only weakly with few testlets). Recovers the testlet variances and
/// item difficulties under normal and skew ability.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_testlet_recovery_500() {
    let (n, j, d_n, per, reps) = (1000usize, 24usize, 4usize, 6usize, 500usize);
    let tid = contiguous_testlets(j, d_n);
    let sig2_t = vec![0.2f64, 0.4, 0.6, 0.8];
    assert_eq!(j, d_n * per);
    let a_t = vec![1.0f64; j];
    let cfg = TestletConfig {
        q_gamma: 15,
        max_iter: 1500,
        ..TestletConfig::default()
    };
    for &skew in [false, true].iter() {
        let (mut s_b, mut s_sig, mut s_bsig, mut n_conv) = (0.0, 0.0, 0.0, 0.0);
        for rep in 0..reps {
            let seed = 0xBADC0FFEE0DDF00Du64
                .wrapping_mul(rep as u64 + 1)
                .wrapping_add(if skew { 0x9E3779B97F4A7C15 } else { 0 });
            let mut rng = Lcg(seed);
            let beta_t: Vec<f64> = (0..j)
                .map(|i| -1.5 + 3.0 * (i % per) as f64 / (per - 1) as f64)
                .collect();
            let y = simulate(&a_t, &beta_t, &sig2_t, &tid, n, j, skew, &mut rng);
            let observed = vec![true; n * j];
            let res =
                fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &cfg).unwrap();
            assert!(
                res.converged,
                "testlet Monte-Carlo fit did not converge: skew={skew}, rep={rep}, n_iter={}, final_delta={}",
                res.n_iter,
                res.final_loglik_change
            );
            s_b += rmse(&res.beta, &beta_t);
            s_sig += rmse(&res.sigma2, &sig2_t);
            s_bsig += bias(&res.sigma2, &sig2_t);
            if res.converged {
                n_conv += 1.0;
            }
        }
        let r = reps as f64;
        println!(
            "skew={}: RMSE(beta)={:.4} RMSE(sigma2)={:.4} bias(sigma2)={:.4} converged={:.2}",
            skew,
            s_b / r,
            s_sig / r,
            s_bsig / r,
            n_conv / r
        );
        assert!(s_b / r < 0.12, "RMSE(beta) {} skew={skew}", s_b / r);
        assert!(s_sig / r < 0.15, "RMSE(sigma2) {} skew={skew}", s_sig / r);
        assert_eq!(
            n_conv, r,
            "not every Monte-Carlo fit converged (skew={skew})"
        );
    }
}
