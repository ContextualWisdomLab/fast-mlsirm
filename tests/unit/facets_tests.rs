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

/// Draw an MFRM category for ability `theta`, item `d`, rater `c`, thresholds `f`.
fn draw_facets(theta: f64, d: f64, c: f64, f: &[f64], u: f64) -> usize {
    let lp = crate::rsm::rsm_logprobs(theta, d + c, f);
    let mut cum = 0.0;
    for (k, l) in lp.iter().enumerate() {
        cum += l.exp();
        if u < cum {
            return k;
        }
    }
    lp.len() - 1
}

/// Simulate a fully crossed design. Returns row-major `P*I*J` categories.
fn simulate(
    seed: u64,
    n_persons: usize,
    d: &[f64],
    c: &[f64],
    f: &[f64],
) -> Vec<usize> {
    let mut rng = Lcg(seed);
    let (ni, nj) = (d.len(), c.len());
    let mut y = vec![0usize; n_persons * ni * nj];
    for p in 0..n_persons {
        let theta = rng.normal();
        for i in 0..ni {
            for j in 0..nj {
                y[p * ni * nj + i * nj + j] = draw_facets(theta, d[i], c[j], f, rng.f64());
            }
        }
    }
    y
}

// ---------------------------------------------------------------------------
// FD anchors on the M-step objective. These asserts read the crate's
// `location_score_terms` / `f_gradient` outputs and compare them against
// central finite differences of the crate's `pair_ell`/`total_ell` at an
// ASYMMETRIC point. Mutations killed: sign flips in the gradients, d<->c index
// transposition (the location derivative would hit the wrong count block),
// suffix-sum off-by-one in `f_gradient` (shifts which residuals feed g_m).
// ---------------------------------------------------------------------------

#[test]
fn location_gradient_matches_fd() {
    // Asymmetric params and asymmetric fake counts (not a fitted state).
    let f = [0.9f64, -0.4, -0.5];
    let n_cat = 4usize;
    let nodes = [-1.3f64, 0.2, 1.7];
    let mut r = vec![0.0f64; nodes.len() * n_cat];
    let mut rng = Lcg(7);
    for v in r.iter_mut() {
        *v = 0.05 + rng.f64() * 2.0;
    }
    let loc0 = 0.37f64;
    let (mut g, mut h) = (0.0f64, 0.0f64);
    location_score_terms(loc0, &f, &r, &nodes, n_cat, &mut g, &mut h);
    let eps = 1e-6;
    let fd = (pair_ell(loc0 + eps, &f, &r, &nodes, n_cat)
        - pair_ell(loc0 - eps, &f, &r, &nodes, n_cat))
        / (2.0 * eps);
    // location_score_terms accumulates -d ell/d location... verify sign
    // convention explicitly: Newton uses step = g/h with update loc - step,
    // and the code defines g = -sum k (r - nP) = -d ell/d loc? No: the
    // derivative d ell/d loc = -sum_k k (r - n P) exactly, so g == fd.
    assert!(
        (g - fd).abs() < 1e-5,
        "analytic {g} vs FD {fd}"
    );
    assert!(h < 0.0, "location Hessian must be negative, got {h}");
    // Hessian FD check too (kills Var-of-score sign/formula mutations).
    let (mut gp, mut hp) = (0.0f64, 0.0f64);
    location_score_terms(loc0 + eps, &f, &r, &nodes, n_cat, &mut gp, &mut hp);
    let (mut gm, mut hm) = (0.0f64, 0.0f64);
    location_score_terms(loc0 - eps, &f, &r, &nodes, n_cat, &mut gm, &mut hm);
    let fd_h = (gp - gm) / (2.0 * eps);
    assert!((h - fd_h).abs() < 1e-4, "analytic H {h} vs FD {fd_h}");
}

#[test]
fn threshold_gradient_matches_fd() {
    let d = [0.3f64, -0.8];
    let c = [0.5f64, -0.1, -0.4];
    let f = [0.7f64, -0.2, -0.5];
    let n_cat = 4usize;
    let nodes = [-1.1f64, 0.4, 2.0];
    let n_pairs = d.len() * c.len();
    let mut rng = Lcg(11);
    let mut r = vec![vec![0.0f64; nodes.len() * n_cat]; n_pairs];
    for blk in r.iter_mut() {
        for v in blk.iter_mut() {
            *v = 0.05 + rng.f64();
        }
    }
    let g = f_gradient(&f, &d, &c, &r, &nodes, d.len(), c.len(), n_cat);
    let eps = 1e-6;
    for m in 0..f.len() {
        let mut fp = f.to_vec();
        fp[m] += eps;
        let mut fm = f.to_vec();
        fm[m] -= eps;
        let fd = (total_ell(&d, &c, &fp, &r, &nodes, d.len(), c.len(), n_cat)
            - total_ell(&d, &c, &fm, &r, &nodes, d.len(), c.len(), n_cat))
            / (2.0 * eps);
        assert!(
            (g[m] - fd).abs() < 1e-5,
            "f[{m}]: analytic {} vs FD {fd}",
            g[m]
        );
    }
}

// ---------------------------------------------------------------------------
// J=1 reduction anchor: with one rater the MFRM must reproduce fit_rsm.
// Asserts read fit_facets' item_difficulty/thresholds/loglik and fit_rsm's
// outputs. Mutations killed: wrong aggregation over the rater axis, pair
// indexing bugs (i*n_raters+j vs j*n_items+i), severity leaking into the fit.
// ---------------------------------------------------------------------------
#[test]
fn single_rater_reduces_to_rsm() {
    let d_true = [-1.2f64, -0.3, 0.4, 1.1];
    let f_true = [0.8f64, -0.8];
    let y = simulate(42, 400, &d_true, &[0.0], &f_true);
    let res = fit_facets(&y, None, 400, 4, 1, 3, 21, 300, 1e-8).unwrap();
    let rsm = crate::rsm::fit_rsm(&y, None, 400, 4, 3, 21, 300, 1e-8).unwrap();
    // sum(c)=0 with one rater forces c_1 = 0 exactly.
    assert!(res.rater_severity[0].abs() < 1e-12);
    for i in 0..4 {
        assert!(
            (res.item_difficulty[i] - rsm.item_location[i]).abs() < 1e-4,
            "item {i}: facets {} vs rsm {}",
            res.item_difficulty[i],
            rsm.item_location[i]
        );
    }
    for m in 0..2 {
        assert!((res.thresholds[m] - rsm.thresholds[m]).abs() < 1e-4);
    }
    let lf = *res.loglik_trace.last().unwrap();
    let lr = *rsm.loglik_trace.last().unwrap();
    assert!((lf - lr).abs() < 1e-4, "loglik facets {lf} vs rsm {lr}");
    assert!(res.connected);
    assert_eq!(res.n_parameters, 4 + 0 + 1);
}

// ---------------------------------------------------------------------------
// Severity recovery with an asymmetric severity vector. Asserts read
// res.rater_severity (crate output). Mutations killed: over-collapse (all
// severities shrink to ~0 -> corr undefined/rmse large), sign flip in the c
// update (corr ~ -1), rater/item dimension-map swap (J=5 != I=6 so shapes
// diverge and recovery fails).
// ---------------------------------------------------------------------------
#[test]
fn recovers_asymmetric_rater_severity() {
    let d_true = [-1.5f64, -0.9, -0.2, 0.3, 0.9, 1.6];
    let c_true = [-1.0f64, -0.3, 0.1, 0.4, 0.8]; // deliberately not centered
    let f_true = [1.0f64, 0.1, -1.1];
    let y = simulate(2024, 800, &d_true, &c_true, &f_true);
    let res = fit_facets(&y, None, 800, 6, 5, 4, 21, 500, 1e-8).unwrap();
    assert!(res.converged);
    assert!(res.connected);
    // Compare against the centered generating severities (model identifies c
    // only up to the sum-zero constraint; the mean shift moves into d).
    let mean_c = c_true.iter().sum::<f64>() / c_true.len() as f64;
    let c_centered: Vec<f64> = c_true.iter().map(|v| v - mean_c).collect();
    let r = corr(&res.rater_severity, &c_centered);
    let e = rmse(&res.rater_severity, &c_centered);
    assert!(r > 0.95, "severity corr {r}");
    assert!(e < 0.15, "severity rmse {e}");
    // Item difficulty absorbs the shift: d_hat ~ d_true + mean_c (+ f-centering
    // shift, which is 0 here up to sampling because f_true sums to 0).
    let d_shifted: Vec<f64> = d_true.iter().map(|v| v + mean_c).collect();
    let rd = corr(&res.item_difficulty, &d_shifted);
    assert!(rd > 0.95, "difficulty corr {rd}");
    // Structural invariants of the returned parameters (not test-local math):
    // both centerings hold on the crate output.
    let sum_c: f64 = res.rater_severity.iter().sum();
    let sum_f: f64 = res.thresholds.iter().sum();
    assert!(sum_c.abs() < 1e-9, "sum(c) = {sum_c}");
    assert!(sum_f.abs() < 1e-9, "sum(f) = {sum_f}");
    assert_eq!(res.n_parameters, 6 + 4 + 2);
    // Known limitation: a constant-shift mutation applied jointly to d and -c
    // is a model invariance and cannot be detected by any data-based test;
    // the discriminating anchors are the centering asserts above.
}

// ---------------------------------------------------------------------------
// Sparse judging plan: each person is scored by 2 of 5 raters on a rotating
// (non-contiguous) schedule. Asserts read crate outputs. Mutations killed:
// dense-only indexing (missing cells would feed category 0 counts), observed-
// mask offset bugs.
// ---------------------------------------------------------------------------
#[test]
fn sparse_design_recovers_severity_order() {
    let d_true = [-0.8f64, 0.0, 0.8];
    let c_true = [-0.9f64, -0.2, 0.0, 0.3, 0.8];
    let f_true = [0.6f64, -0.6];
    let (np, ni, nj) = (1500usize, 3usize, 5usize);
    let y = simulate(99, np, &d_true, &c_true, &f_true);
    // Rotating pairs (p, p+2 mod 5): non-contiguous rater unions, connected.
    let mut obs = vec![false; np * ni * nj];
    for p in 0..np {
        let (a, b) = (p % nj, (p + 2) % nj);
        for i in 0..ni {
            obs[p * ni * nj + i * nj + a] = true;
            obs[p * ni * nj + i * nj + b] = true;
        }
    }
    let res = fit_facets(&y, Some(&obs), np, ni, nj, 3, 21, 500, 1e-8).unwrap();
    assert!(res.connected);
    let mean_c = c_true.iter().sum::<f64>() / nj as f64;
    let c_centered: Vec<f64> = c_true.iter().map(|v| v - mean_c).collect();
    let r = corr(&res.rater_severity, &c_centered);
    assert!(r > 0.9, "sparse severity corr {r}");
    // The recovered severity ORDER must match (kills permutation/off-by-one
    // in the rater axis under a sparse mask).
    let mut idx: Vec<usize> = (0..nj).collect();
    idx.sort_by(|&a, &b| res.rater_severity[a].partial_cmp(&res.rater_severity[b]).unwrap());
    assert_eq!(idx, vec![0, 1, 2, 3, 4]);
}

// ---------------------------------------------------------------------------
// Connectivity flag. Asserts read res.connected. Mutations killed: joining
// only items to items (not raters), skipping the person-mediated union, or
// hardcoding true.
// ---------------------------------------------------------------------------
#[test]
fn disconnected_design_is_flagged() {
    // Two islands: persons 0..P/2 x item 0 x rater 0; persons P/2.. x item 1 x rater 1.
    let (np, ni, nj) = (60usize, 2usize, 2usize);
    let d = [0.0f64, 0.0];
    let c = [0.5f64, -0.5];
    let f = [0.0f64];
    let y = simulate(5, np, &d, &c, &f);
    let mut obs = vec![false; np * ni * nj];
    for p in 0..np {
        let island = usize::from(p >= np / 2);
        obs[p * ni * nj + island * nj + island] = true;
    }
    let res = fit_facets(&y, Some(&obs), np, ni, nj, 2, 7, 50, 1e-6).unwrap();
    assert!(!res.connected, "two islands must be flagged disconnected");

    // Bridging rater: rater 0 also scores item 1 for one person -> connected.
    let mut obs2 = obs.clone();
    obs2[0 * ni * nj + 1 * nj + 0] = true; // person 0, item 1, rater 0
    let res2 = fit_facets(&y, Some(&obs2), np, ni, nj, 2, 7, 50, 1e-6).unwrap();
    assert!(res2.connected, "bridge must connect the design");
}

// ---------------------------------------------------------------------------
// Validation errors.
// ---------------------------------------------------------------------------
#[test]
fn rejects_bad_inputs() {
    let y = vec![0usize; 4];
    assert!(fit_facets(&y, None, 2, 2, 1, 1, 7, 50, 1e-6).is_err()); // n_cat < 2
    assert!(fit_facets(&y, None, 2, 2, 1, 2, 8, 50, 1e-6).is_err()); // bad q
    assert!(fit_facets(&y, None, 2, 2, 1, 2, 7, 0, 1e-6).is_err()); // max_iter 0
    assert!(fit_facets(&y, None, 2, 2, 1, 2, 7, 50, f64::NAN).is_err());
    assert!(fit_facets(&y, None, 3, 2, 1, 2, 7, 50, 1e-6).is_err()); // len mismatch
    let y2 = vec![0usize, 5, 0, 0];
    assert!(fit_facets(&y2, None, 2, 2, 1, 3, 7, 50, 1e-6).is_err()); // cat >= n_cat
    // rater with no observations
    let y3 = vec![0usize; 2 * 1 * 2];
    let obs = vec![true, false, true, false];
    assert!(fit_facets(&y3, Some(&obs), 2, 1, 2, 2, 7, 50, 1e-6)
        .unwrap_err()
        .contains("rater 1"));
}

#[test]
fn loglik_trace_is_nondecreasing() {
    let y = simulate(3, 200, &[-0.5, 0.5], &[-0.4, 0.4], &[0.5, -0.5]);
    let res = fit_facets(&y, None, 200, 2, 2, 3, 21, 200, 1e-10).unwrap();
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-8, "EM must be monotone: {} -> {}", w[0], w[1]);
    }
}

// ---------------------------------------------------------------------------
// Monte-Carlo recovery, 500 replications (heavy; run with --ignored).
// Half the replications generate theta from a skewed distribution (mixture
// shift) to probe prior-misspecification robustness: bias is reported with a
// loose bound rather than asserted tightly.
// ---------------------------------------------------------------------------
#[test]
#[ignore = "500-replication Monte-Carlo; run with --ignored"]
fn monte_carlo_severity_bias_and_rmse() {
    let d_true = [-1.0f64, 0.0, 1.0];
    let c_true = [-0.7f64, 0.0, 0.7]; // centered
    let f_true = [0.9f64, -0.9];
    let (np, ni, nj) = (300usize, 3usize, 3usize);
    let reps = 500usize;
    let mut bias = vec![0.0f64; nj];
    let mut mse = vec![0.0f64; nj];
    for rep in 0..reps {
        let skewed = rep % 2 == 1;
        let mut rng = Lcg(10_000 + rep as u64);
        let mut y = vec![0usize; np * ni * nj];
        for p in 0..np {
            let theta = if skewed {
                // Standardized two-component location mixture (negatively
                // skewed), mean 0 / var ~1 by construction below.
                let z = rng.normal();
                let comp = if rng.f64() < 0.75 { 0.35 } else { -1.05 };
                (z * 0.8 + comp) / (0.8f64.powi(2) + 0.42f64).sqrt()
            } else {
                rng.normal()
            };
            for i in 0..ni {
                for j in 0..nj {
                    y[p * ni * nj + i * nj + j] =
                        draw_facets(theta, d_true[i], c_true[j], &f_true, rng.f64());
                }
            }
        }
        let res = fit_facets(&y, None, np, ni, nj, 3, 21, 500, 1e-8).unwrap();
        for j in 0..nj {
            let e = res.rater_severity[j] - c_true[j];
            bias[j] += e / reps as f64;
            mse[j] += e * e / reps as f64;
        }
    }
    for j in 0..nj {
        let rm = mse[j].sqrt();
        // Loose bounds: severity is a fixed effect over 300*3 ratings/rep.
        assert!(bias[j].abs() < 0.05, "rater {j} bias {}", bias[j]);
        assert!(rm < 0.2, "rater {j} rmse {rm}");
    }
}
