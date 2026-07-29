use super::*;

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
}

/// Brute-force elementary symmetric function of order `r` (sum over all size-`r` subsets).
fn esf_brute(eps: &[f64], r: usize) -> f64 {
    let k = eps.len();
    let mut total = 0.0;
    for mask in 0u64..(1u64 << k) {
        if (mask.count_ones() as usize) == r {
            let mut prod = 1.0;
            for i in 0..k {
                if mask & (1 << i) != 0 {
                    prod *= eps[i];
                }
            }
            total += prod;
        }
    }
    total
}

/// The summation-algorithm ESF (and the leave-one-out / leave-two-out passes) match the brute-force
/// subset sums exactly.
#[test]
fn esf_matches_brute_force() {
    let eps = [0.4, 1.1, 2.3, 0.7, 1.6];
    let k = eps.len();
    let g = esf(&eps);
    for r in 0..=k {
        assert!((g[r] - esf_brute(&eps, r)).abs() < 1e-10, "gamma_{r}");
    }
    // leave-one-out
    for omit in 0..k {
        let gi = esf_omit(&eps, omit);
        let sub: Vec<f64> = (0..k).filter(|&j| j != omit).map(|j| eps[j]).collect();
        for r in 0..k {
            assert!(
                (gi[r] - esf_brute(&sub, r)).abs() < 1e-10,
                "gamma^({omit})_{r}"
            );
        }
    }
    // leave-two-out
    let gij = esf_omit2(&eps, 1, 3);
    let sub: Vec<f64> = (0..k)
        .filter(|&j| j != 1 && j != 3)
        .map(|j| eps[j])
        .collect();
    for r in 0..k - 1 {
        assert!(
            (gij[r] - esf_brute(&sub, r)).abs() < 1e-10,
            "gamma^(1,3)_{r}"
        );
    }
}

/// Deterministic anchor: the analytic CML gradient and Hessian match finite differences of the
/// conditional log-likelihood (pins the sign of `d eps/d beta = -eps` and the ESF derivative
/// recursions — a sign error would flip the whole Newton direction).
#[test]
fn cml_gradient_hessian_match_finite_difference() {
    let beta = [-0.8, 0.3, 1.1, -0.2, 0.6];
    let k = beta.len();
    let s = [40.0, 55.0, 62.0, 48.0, 58.0];
    let nr = [0.0, 20.0, 30.0, 25.0, 15.0, 0.0]; // r = 0..=5, r=0,5 uninformative
    let (_ll, grad, hess) = cml_eval(&beta, &s, &nr);
    let eps = 1e-6;
    for i in 0..k {
        let mut bp = beta;
        bp[i] += eps;
        let mut bm = beta;
        bm[i] -= eps;
        let fd = (cml_eval(&bp, &s, &nr).0 - cml_eval(&bm, &s, &nr).0) / (2.0 * eps);
        assert!(
            (grad[i] - fd).abs() < 1e-4,
            "grad[{i}] {} vs FD {fd}",
            grad[i]
        );
    }
    let hh = 1e-4;
    for a in 0..k {
        for b in 0..k {
            let mut pp = beta;
            pp[a] += hh;
            pp[b] += hh;
            let mut pm = beta;
            pm[a] += hh;
            pm[b] -= hh;
            let mut mp = beta;
            mp[a] -= hh;
            mp[b] += hh;
            let mut mm = beta;
            mm[a] -= hh;
            mm[b] -= hh;
            let d2 =
                (cml_eval(&pp, &s, &nr).0 - cml_eval(&pm, &s, &nr).0 - cml_eval(&mp, &s, &nr).0
                    + cml_eval(&mm, &s, &nr).0)
                    / (4.0 * hh * hh);
            assert!(
                (hess[a * k + b] - d2).abs() < 1e-2,
                "hess[{a}][{b}] {} vs FD {d2}",
                hess[a * k + b]
            );
        }
    }
}

fn simulate(beta: &[f64], theta: &[f64], rng: &mut Lcg) -> Vec<u8> {
    let k = beta.len();
    let n = theta.len();
    let mut y = vec![0u8; n * k];
    for p in 0..n {
        for i in 0..k {
            let pr = 1.0 / (1.0 + (-(theta[p] - beta[i])).exp());
            y[p * k + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    y
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
}

/// THE DEFINING CML PROPERTY (person-distribution-free): the same beta_hat (up to the sum-zero
/// constant) is recovered whether the simulating theta is N(0,1) or strongly right-skewed. A plain
/// value-recovery test is INSUFFICIENT — JML also recovers beta at large k — so the discriminating
/// assertion is the AGREEMENT between the two distributions' estimates, not merely closeness to
/// truth.
#[test]
fn cml_is_person_distribution_free() {
    let mut beta = vec![-1.6, -0.9, -0.3, 0.2, 0.7, 1.2, 1.7, 0.0];
    center(&mut beta);
    let k = beta.len();
    let n = 4000usize;
    let mut rng = Lcg(918273);
    // (a) theta ~ N(0,1)
    let th_norm: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    // (b) theta strongly right-skew (standardized Exp - shifted), a very different distribution
    let th_skew: Vec<f64> = (0..n)
        .map(|_| 1.5 * (-(rng.next_f64().max(1e-12)).ln()) - 1.0)
        .collect();
    let ya = simulate(&beta, &th_norm, &mut rng);
    let yb = simulate(&beta, &th_skew, &mut rng);
    let fa = fit_rasch_cml(&ya, n, k, 100, 1e-9).unwrap();
    let fb = fit_rasch_cml(&yb, n, k, 100, 1e-9).unwrap();
    assert!(fa.converged && fb.converged);
    // both recover the truth within MC tolerance
    assert!(
        rmse(&fa.beta, &beta) < 0.15,
        "N(0,1) beta RMSE {}",
        rmse(&fa.beta, &beta)
    );
    assert!(
        rmse(&fb.beta, &beta) < 0.15,
        "skew beta RMSE {}",
        rmse(&fb.beta, &beta)
    );
    // and — the CML signature — the two estimates AGREE despite the very different ability
    // distributions (a distribution-DEPENDENT estimator would diverge here)
    assert!(
        rmse(&fa.beta, &fb.beta) < 0.15,
        "distribution-free property violated: N(0,1) vs skew beta RMSE {}",
        rmse(&fa.beta, &fb.beta)
    );
    // SEs finite and positive on-support
    assert!(fa.se.iter().all(|s| s.is_finite() && *s > 0.0));
}

/// Andersen (1973) LR: on Rasch-generated data an arbitrary (ability-independent) group split does
/// NOT reject (statistic near its df), while data with a group-specific difficulty shift (Rasch
/// misfit / DIF) is rejected with a large statistic. Pins the df and the upper-tail direction.
#[test]
fn andersen_lr_detects_group_difficulty_shift() {
    let mut beta = vec![-1.2, -0.6, 0.0, 0.6, 1.2, -0.3, 0.3, 0.9];
    center(&mut beta);
    let k = beta.len();
    let n = 3000usize;
    let mut rng = Lcg(0xA9D5);
    let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let group: Vec<u8> = (0..n).map(|p| (p % 2) as u8).collect();
    // (1) true Rasch, split by an ARBITRARY label (independent of ability): should NOT reject
    let y_fit = simulate(&beta, &theta, &mut rng);
    let t1 = andersen_lr_test(&y_fit, &group, 2, n, k, 100, 1e-9).unwrap();
    assert_eq!(t1.df, (2 - 1) * (k - 1));
    assert!(
        t1.lr / (t1.df as f64) < 3.0,
        "Rasch data over-rejected: LR {} df {}",
        t1.lr,
        t1.df
    );
    assert!(t1.p_value > 0.01, "Rasch data p too small: {}", t1.p_value);
    // (2) group 1 gets a difficulty shift on item 0 (violates Rasch invariance): should reject
    let mut y_dif = vec![0u8; n * k];
    for p in 0..n {
        for i in 0..k {
            let mut bi = beta[i];
            if i == 0 && group[p] == 1 {
                bi += 1.5;
            }
            let pr = 1.0 / (1.0 + (-(theta[p] - bi)).exp());
            y_dif[p * k + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    let t2 = andersen_lr_test(&y_dif, &group, 2, n, k, 100, 1e-9).unwrap();
    assert!(
        t2.lr > t1.lr + 15.0,
        "DIF not detected: LR {} vs baseline {}",
        t2.lr,
        t1.lr
    );
    assert!(t2.p_value < 0.01, "DIF p not significant: {}", t2.p_value);
    assert!(
        t1.converged && t2.converged,
        "converged flag not set on a converging fit"
    );
    // a starved max_iter surfaces non-convergence rather than a silently clamped lr=0
    let t_bad = andersen_lr_test(&y_dif, &group, 2, n, k, 1, 1e-9).unwrap();
    assert!(
        !t_bad.converged,
        "non-convergence must be surfaced, not masked"
    );
}

/// Validation guards.
#[test]
fn cml_validates() {
    let y = vec![0u8, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1]; // 3 persons x 4 items
    assert!(fit_rasch_cml(&y, 3, 4, 100, 1e-9).is_ok());
    assert!(fit_rasch_cml(&y, 3, 4, 0, 1e-9).is_err()); // max_iter 0
    let mut ybad = y.clone();
    ybad[0] = 2;
    assert!(fit_rasch_cml(&ybad, 3, 4, 100, 1e-9).is_err()); // non-binary
    assert!(fit_rasch_cml(&y, 3, 1, 100, 1e-9).is_err()); // n_items < 2 (length also wrong)
                                                          // all-perfect / all-zero -> no informative persons
    let yflat = vec![1u8; 3 * 4];
    assert!(fit_rasch_cml(&yflat, 3, 4, 100, 1e-9).is_err());
}

#[test]
fn cml_covers_every_validation_and_degenerate_numeric_exit() {
    let y = [1u8, 0, 0, 1, 1, 0];
    assert!(fit_rasch_cml(&y, 3, 2, 10, f64::NAN).is_err());
    assert!(fit_rasch_cml(&y, 3, 2, 10, 0.0).is_err());
    assert!(fit_rasch_cml(&y[..5], 3, 2, 10, 1e-8).is_err());
    assert!(fit_rasch_cml(&[], 1, CML_MAX_ITEMS + 1, 10, 1e-8).is_err());
    assert!(fit_rasch_cml(&[], usize::MAX, 2, 10, 1e-8).is_err());

    assert!(andersen_lr_test(&y, &[0, 1], 2, 3, 2, 10, 1e-8).is_err());
    assert!(andersen_lr_test(&y, &[0, 0, 0], 1, 3, 2, 10, 1e-8).is_err());
    assert!(andersen_lr_test(&y, &[0, 1, 2], 2, 3, 2, 10, 1e-8).is_err());
    assert!(andersen_lr_test(&y, &[0, 0, 0], 2, 3, 2, 10, 1e-8).is_err());

    let no_information_in_group_one = [1u8, 0, 0, 1, 1, 1, 1, 1];
    assert!(andersen_lr_test(
        &no_information_in_group_one,
        &[0, 0, 1, 1],
        2,
        4,
        2,
        10,
        1e-8,
    )
    .is_err());

    let stalled = fit_from_stats(&[f64::NAN, 0.0], &[0.0, 1.0, 0.0], 1, 1, 1e-8).unwrap();
    assert!(!stalled.converged);
    assert_eq!(stalled.n_iter, 1);
    let singular = fit_from_stats(&[0.0, 0.0], &[0.0, 0.0, 0.0], 0, 1, 1e-8).unwrap();
    assert!(singular.se.iter().all(|v| v.is_nan()));
}
