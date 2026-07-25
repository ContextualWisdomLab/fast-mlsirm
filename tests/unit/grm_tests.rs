use super::*;
use crate::poly::{fit_poly_unidim, PolyModel};

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
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}
fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for (a, b) in x.iter().zip(y) {
        sxy += (a - mx) * (b - my);
        sxx += (a - mx) * (a - mx);
        syy += (b - my) * (b - my);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

/// Simulate multidimensional GRM responses from slope (n_items*n_dims), thresholds
/// (n_items*(n_cat-1)), and traits (n_persons*n_dims).
fn simulate(
    slope: &[f64],
    threshold: &[f64],
    theta: &[f64],
    n: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    rng: &mut Lcg,
) -> Vec<usize> {
    let m1 = n_cat - 1;
    let mut y = vec![0usize; n * n_items];
    for p in 0..n {
        for i in 0..n_items {
            let mut base = 0.0f64;
            for d in 0..n_dims {
                base += slope[i * n_dims + d] * theta[p * n_dims + d];
            }
            let lp = grm_logprobs(base, &threshold[i * m1..(i + 1) * m1]);
            let probs: Vec<f64> = lp.iter().map(|l| l.exp()).collect();
            let u = rng.next_f64();
            let mut acc = 0.0;
            let mut cat = n_cat - 1;
            for (k, &pk) in probs.iter().enumerate() {
                acc += pk;
                if u < acc {
                    cat = k;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    y
}

/// D = 1 WITHIN-TOL reduction to fit_poly_unidim(GRM). True slopes are all POSITIVE (the domain
/// where fit_poly_unidim's log_a>0 is correctly specified); both fitters reach the same MLE up to
/// optimizer tolerance and the (positive) reflection, so recovered slope & thresholds & loglik
/// agree within a loose bound. NOT bit-exact (log_a vs unconstrained a differ in Newton path).
#[test]
fn grm_reduces_to_poly_grm_at_d1() {
    let (n, n_items, n_cat) = (2000usize, 6usize, 4usize);
    let m1 = n_cat - 1;
    let mut rng = Lcg(51169);
    let mut slope = vec![0.0f64; n_items * 1];
    let mut threshold = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        slope[i] = 0.8 + 0.25 * i as f64; // POSITIVE
                                          // strictly decreasing thresholds
        for j in 0..m1 {
            threshold[i * m1 + j] = 1.2 - 1.0 * j as f64 - 0.05 * i as f64;
        }
    }
    let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y = simulate(&slope, &threshold, &theta, n, n_items, 1, n_cat, &mut rng);
    let pattern = vec![1u8; n_items];
    let cfg = GrmConfig {
        q: 21,
        ..GrmConfig::default()
    };
    let mm = fit_grm(&y, None, &pattern, n, n_items, 1, n_cat, &cfg).unwrap();
    let pf = fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Grm, 21, 500, 1e-6).unwrap();
    // slopes agree (both positive), thresholds agree, within optimizer tolerance
    for i in 0..n_items {
        assert!(
            (mm.slope[i] - pf.slope[i]).abs() < 0.05,
            "slope[{i}] {} vs {}",
            mm.slope[i],
            pf.slope[i]
        );
        for j in 0..m1 {
            let d = (mm.threshold[i * m1 + j] - pf.cat_params[i][j]).abs();
            assert!(d < 0.06, "threshold[{i}][{j}] diff {d}");
        }
    }
    let mm_ll = *mm.loglik_trace.last().unwrap();
    assert!(
        (mm_ll - pf.loglik).abs() < 0.5,
        "loglik {mm_ll} vs {}",
        pf.loglik
    );
    assert_eq!(mm.n_parameters, n_items * (1 + m1));
}

/// Deterministic FD GRADIENT anchor at D=2 (GH) AND D=4 (Halton, NON-IDENTITY dims [0,2,3]) with
/// M=4 categories. The threshold block is STRICTLY DECREASING with gaps >> the FD eps (GRM NaNs on
/// inverted betas, unlike the finite-everywhere softmax); the slope block is distinct and the
/// per-category counts random+distinct, so a slope<->threshold slot transposition or a sign error
/// is detected. The M-step uses an FD Hessian, so pin the GRADIENT.
#[test]
fn grm_gradient_matches_finite_difference() {
    let n_cat = 4usize;
    for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter() {
        let l = dims.len();
        let (nodes, n_nodes) = if n_dims == 2 {
            let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: 15 }, n_dims).unwrap();
            (xn.grid, xn.logw.len())
        } else {
            let xn = build_xi_nodes(
                XiRule::Halton {
                    n: 200,
                    shift_seed: 0,
                },
                n_dims,
            )
            .unwrap();
            (xn.grid, xn.logw.len())
        };
        let mut rng = Lcg(2718 + n_dims as u64);
        let counts: Vec<Vec<f64>> = (0..n_nodes)
            .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 3.0).collect())
            .collect();
        // params: distinct slopes then STRICTLY DECREASING thresholds (gaps 0.7 >> eps).
        let mut params = vec![0.0f64; l + (n_cat - 1)];
        for t in 0..l {
            params[t] = 0.4 + 0.3 * t as f64 - if t == 1 { 0.9 } else { 0.0 };
        }
        for j in 0..(n_cat - 1) {
            params[l + j] = 1.0 - 0.7 * j as f64; // 1.0, 0.3, -0.4 (strictly decreasing)
        }
        let (_f0, grad) = grm_item_neg_ll_grad(&params, dims, &nodes, n_dims, &counts, n_cat);
        let eps = 1e-6;
        for j in 0..params.len() {
            let mut pp = params.clone();
            pp[j] += eps;
            let (fp, _) = grm_item_neg_ll_grad(&pp, dims, &nodes, n_dims, &counts, n_cat);
            let mut pm = params.clone();
            pm[j] -= eps;
            let (fm, _) = grm_item_neg_ll_grad(&pm, dims, &nodes, n_dims, &counts, n_cat);
            let fd = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[j] - fd).abs() < 1e-4,
                "grad[{j}] {} vs fd {fd} (D={n_dims})",
                grad[j]
            );
        }
    }
}

/// Deterministic OBJECTIVE-VALUE dims-map pin at D=4 (Halton, dims=[0,2,3]). The FD gradient anchor
/// is map-INVARIANT (a consistent wrong-node-column bug in base+gradient is invisible to a central
/// difference through the same buggy objective); and no D>=4 fit is exercised by the recovery/MC
/// tests. So compute the objective's per-node base and neg-loglik BY HAND with the CORRECT dim map
/// and assert the estimator's internal value equals it to < 1e-9 — pinning nodes[nd*n_dims + dims[t]].
#[test]
fn grm_objective_dims_map_pinned_at_d4() {
    let n_dims = 4usize;
    let dims = vec![0usize, 2, 3];
    let n_cat = 4usize;
    let l = dims.len();
    let xn = build_xi_nodes(
        XiRule::Halton {
            n: 64,
            shift_seed: 0,
        },
        n_dims,
    )
    .unwrap();
    let nodes = xn.grid;
    let n_nodes = xn.logw.len();
    let mut rng = Lcg(31337);
    let counts: Vec<Vec<f64>> = (0..n_nodes)
        .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 2.0).collect())
        .collect();
    let a = [0.9f64, -0.6, 0.7];
    let beta = [0.8f64, 0.0, -0.9]; // strictly decreasing
    let mut params = vec![0.0f64; l + (n_cat - 1)];
    params[..l].copy_from_slice(&a);
    params[l..].copy_from_slice(&beta);
    let (neg_ll, _g) = grm_item_neg_ll_grad(&params, &dims, &nodes, n_dims, &counts, n_cat);
    // hand computation with the CORRECT dim map [0,2,3]
    let mut hand = 0.0f64;
    for (nd, cnt) in counts.iter().enumerate() {
        let base = a[0] * nodes[nd * n_dims + 0]
            + a[1] * nodes[nd * n_dims + 2]
            + a[2] * nodes[nd * n_dims + 3];
        let lp = grm_logprobs(base, &beta);
        hand += cnt.iter().zip(&lp).map(|(r, l2)| r * l2).sum::<f64>();
    }
    assert!(
        (neg_ll - (-hand)).abs() < 1e-9,
        "objective dims-map mismatch: {neg_ll} vs {}",
        -hand
    );
}

// build a D=2 confirmatory GRM design (items 0,1 pure dim0; 2,3 pure dim1; item 4 cross-loader).
fn design_d2(n_cat: usize) -> (Vec<u8>, usize, Vec<f64>, Vec<f64>) {
    let n_dims = 2usize;
    let m1 = n_cat - 1;
    let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
    let n_items = 5usize;
    let mut slope = vec![0.0f64; n_items * n_dims];
    slope[0 * n_dims + 0] = 1.4;
    slope[1 * n_dims + 0] = 1.0;
    slope[2 * n_dims + 1] = 1.2;
    slope[3 * n_dims + 1] = 1.1;
    slope[4 * n_dims + 0] = -1.0; // NEGATIVE cross-loader on dim0 (anchor item 0 is positive)
    slope[4 * n_dims + 1] = 0.9;
    let mut threshold = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        for j in 0..m1 {
            threshold[i * m1 + j] = 1.1 - 1.0 * j as f64 + 0.05 * i as f64;
        }
    }
    (pattern, n_items, slope, threshold)
}

/// D = 2 recovery on GH nodes: pure anchors + a NEGATIVE cross-loader on dimension 0 (whose pure
/// anchor is positively keyed, so canonicalization preserves the cross-loader's sign). Recovered
/// thresholds must stay STRICTLY ordered on every item. Baseline structural checks + per-dim EAP.
#[test]
fn grm_recovers_d2_with_negative_cross_loader() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let m1 = n_cat - 1;
    let (pattern, n_items, slope, threshold) = design_d2(n_cat);
    let n = 6000usize;
    let mut rng = Lcg(4747);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(
        &slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng,
    );
    let cfg = GrmConfig {
        q: 21,
        ..GrmConfig::default()
    };
    let res = fit_grm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    assert!(res.converged);
    // off-pattern slopes EXACTLY zero
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.slope[i * n_dims + d], 0.0, "off-pattern zero");
            }
        }
    }
    // recovered thresholds strictly ordered-decreasing on EVERY item
    for i in 0..n_items {
        for j in 0..m1 - 1 {
            assert!(
                res.threshold[i * m1 + j] > res.threshold[i * m1 + j + 1],
                "ordered item {i}"
            );
        }
    }
    // canonical output: pure anchors positive; the negative cross-loader recovered NEGATIVE
    assert!(res.slope[0 * n_dims + 0] > 0.5, "anchor0 positive");
    assert!(res.slope[2 * n_dims + 1] > 0.5, "anchor2 positive");
    assert!(
        res.slope[4 * n_dims + 0] < -0.4,
        "neg cross-loader: {}",
        res.slope[4 * n_dims + 0]
    );
    assert!(
        rmse(&res.slope, &slope) < 0.16,
        "slope RMSE {}",
        rmse(&res.slope, &slope)
    );
    for d in 0..n_dims {
        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
        let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
        assert!(corr(&th, &tt) > 0.6, "theta{d} corr {}", corr(&th, &tt));
    }
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "EM monotone");
    }
}

/// The baked-in reflection canonicalization actually FIRES: a reverse-keyed LARGEST pure anchor on
/// dimension 0 (true slope strongly NEGATIVE) is flipped so it ends POSITIVE, a positively-keyed
/// co-loader on the same dimension ends NEGATIVE (whole-dimension flip), and the thresholds are
/// UNCHANGED and still ordered (the flip touches only slopes + theta, never betas).
#[test]
fn grm_reflection_fires_on_negative_anchor() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let m1 = n_cat - 1;
    // item0 pure dim0 (largest, NEGATIVE), item1 pure dim0 (positive), items 2,3 pure dim1.
    let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1];
    let n_items = 4usize;
    let mut slope = vec![0.0f64; n_items * n_dims];
    slope[0 * n_dims + 0] = -1.8; // reverse-keyed largest anchor on dim0
    slope[1 * n_dims + 0] = 1.0; // positively-keyed co-loader on dim0
    slope[2 * n_dims + 1] = 1.2;
    slope[3 * n_dims + 1] = 1.0;
    let mut threshold = vec![0.0f64; n_items * m1];
    for i in 0..n_items {
        for j in 0..m1 {
            threshold[i * m1 + j] = 0.9 - 1.0 * j as f64;
        }
    }
    let n = 4000usize;
    let mut rng = Lcg(8181);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(
        &slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng,
    );
    let cfg = GrmConfig {
        q: 21,
        ..GrmConfig::default()
    };
    let res = fit_grm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    // dim0's largest pure anchor (item 0) ends POSITIVE; co-loader (item 1) ends NEGATIVE.
    assert!(
        res.slope[0 * n_dims + 0] > 0.8,
        "reflected anchor positive: {}",
        res.slope[0 * n_dims + 0]
    );
    assert!(
        res.slope[1 * n_dims + 0] < -0.3,
        "co-loader flipped negative: {}",
        res.slope[1 * n_dims + 0]
    );
    // The reflection flips BOTH the slope column AND theta_d, keeping base = sum a_d theta_d
    // invariant. Since dim0 was flipped, the returned EAP theta_0 must correlate NEGATIVELY with
    // the true theta_0 (the data was generated with the negative anchor); dim1 (not flipped) stays
    // positive. Deleting the theta-negation half of the reflection inverts dim0's sign here.
    let th0: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 0]).collect();
    let tt0: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 0]).collect();
    let th1: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 1]).collect();
    let tt1: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 1]).collect();
    assert!(
        corr(&th0, &tt0) < -0.5,
        "flipped-dim theta corr must be negative: {}",
        corr(&th0, &tt0)
    );
    assert!(
        corr(&th1, &tt1) > 0.5,
        "unflipped-dim theta corr positive: {}",
        corr(&th1, &tt1)
    );
    // thresholds still strictly ordered (untouched by the reflection)
    for i in 0..n_items {
        for j in 0..m1 - 1 {
            assert!(
                res.threshold[i * m1 + j] > res.threshold[i * m1 + j + 1],
                "ordered item {i}"
            );
        }
    }
}

/// Structural invariants + validation guards.
#[test]
fn grm_validates_and_structural_invariants() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let (pattern, n_items, slope, threshold) = design_d2(n_cat);
    let n = 500usize;
    let mut rng = Lcg(99);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(
        &slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng,
    );
    let cfg = GrmConfig {
        q: 15,
        max_iter: 25,
        ..GrmConfig::default()
    };
    let res = fit_grm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    // free-parameter count = sum_i (|S_i| + (n_cat-1)): items 0-3 pure (1+2), item 4 cross (2+2).
    assert_eq!(res.n_parameters, 4 * (1 + 2) + (2 + 2));
    // grm_logprobs sum to 1 at a sample base
    let lp = grm_logprobs(0.4, &[0.8, -0.3]);
    let s: f64 = lp.iter().map(|l| l.exp()).sum();
    assert!((s - 1.0).abs() < 1e-12);
    // validation: GH D=4 rejected (y observes all categories so the D-bound is the sole reason);
    // no pure anchor rejected; category >= n_cat rejected; unobserved category rejected.
    let gh4 = GrmConfig::default();
    let pat4: Vec<u8> = (0..4)
        .flat_map(|d| (0..4).map(move |k| (k == d) as u8))
        .collect();
    let y4: Vec<usize> = (0..n * 4).map(|idx| idx % n_cat).collect();
    assert!(
        fit_grm(&y4, None, &pat4, n, 4, 4, n_cat, &gh4).is_err(),
        "GH D=4 rejected"
    );
    let no_anchor: Vec<u8> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
    assert!(
        fit_grm(&y, None, &no_anchor, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "no pure anchor rejected"
    );
    let mut ybad = y.clone();
    ybad[0] = n_cat;
    assert!(
        fit_grm(&ybad, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "bad category rejected"
    );
    let mut ygap = y.clone();
    for p in 0..n {
        if ygap[p * n_items + 0] == 1 {
            ygap[p * n_items + 0] = 0;
        }
    }
    assert!(
        fit_grm(&ygap, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "unobserved category rejected"
    );
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the multidimensional GRM at D=2 and D=3
/// under normal AND per-dim-standardized right-skew traits. The estimator canonicalizes reflection
/// (pure anchors positive), so truth is built positive-anchored and the estimate compares directly.
/// Per-rep monotone-EM + finiteness + threshold-ordering canaries.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_grm_recovery_500() {
    let reps = 500usize;
    let n_cat = 3usize;
    let m1 = n_cat - 1;
    for &(n_dims, q, n) in [(2usize, 15usize, 2500usize), (3usize, 11usize, 2000usize)].iter() {
        let mut pattern: Vec<u8> = Vec::new();
        for d in 0..n_dims {
            for _ in 0..2 {
                let mut r = vec![0u8; n_dims];
                r[d] = 1;
                pattern.extend_from_slice(&r);
            }
        }
        for d in 0..n_dims {
            let mut r = vec![0u8; n_dims];
            r[d] = 1;
            r[(d + 1) % n_dims] = 1;
            pattern.extend_from_slice(&r);
        }
        let n_items = 2 * n_dims + n_dims;
        let mut slope = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            slope[(2 * d) * n_dims + d] = 1.3; // pure anchors POSITIVE
            slope[(2 * d + 1) * n_dims + d] = 1.0;
        }
        for d in 0..n_dims {
            let ci = 2 * n_dims + d;
            slope[ci * n_dims + d] = 1.0;
            slope[ci * n_dims + (d + 1) % n_dims] = if d % 2 == 0 { 0.7 } else { -0.7 };
        }
        let mut threshold = vec![0.0f64; n_items * m1];
        for i in 0..n_items {
            for j in 0..m1 {
                threshold[i * m1 + j] = 1.0 - 1.2 * j as f64 + 0.04 * i as f64;
            }
        }
        for &skew in [false, true].iter() {
            let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
            let (mut tnum, mut tden) = (0.0f64, 0.0f64);
            let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
            let mut nconv = 0usize;
            for rep in 0..reps {
                let mut rng = Lcg(0x9E3779B97F4A7C15u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                    .wrapping_add(n_dims as u64 * 0x100000001B3));
                let mut theta = vec![0.0f64; n * n_dims];
                for d in 0..n_dims {
                    let col: Vec<f64> = (0..n)
                        .map(|_| {
                            if skew {
                                let mut cc = 0.0;
                                for _ in 0..3 {
                                    let z = rng.normal();
                                    cc += z * z;
                                }
                                (cc - 3.0) / 6f64.sqrt()
                            } else {
                                rng.normal()
                            }
                        })
                        .collect();
                    let m = col.iter().sum::<f64>() / n as f64;
                    let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                    let sd = v.sqrt();
                    for j in 0..n {
                        theta[j * n_dims + d] = (col[j] - m) / sd;
                    }
                }
                let y = simulate(
                    &slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng,
                );
                let cfg = GrmConfig {
                    q,
                    ..GrmConfig::default()
                };
                let res = fit_grm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
                if res.converged {
                    nconv += 1;
                }
                for w in res.loglik_trace.windows(2) {
                    assert!(w[1] >= w[0] - 1e-9, "monotone (rep {rep})");
                }
                assert!(
                    res.slope.iter().all(|v| v.is_finite()),
                    "finite slope (rep {rep})"
                );
                for i in 0..n_items {
                    for j in 0..m1 - 1 {
                        assert!(
                            res.threshold[i * m1 + j] > res.threshold[i * m1 + j + 1],
                            "ordered (rep {rep} item {i})"
                        );
                    }
                }
                for i in 0..n_items {
                    for d in 0..n_dims {
                        if pattern[i * n_dims + d] != 0 {
                            let e = res.slope[i * n_dims + d] - slope[i * n_dims + d];
                            lnum += e * e;
                            lden += 1.0;
                            lbias += e;
                        }
                    }
                }
                for i in 0..n_items {
                    for j in 0..m1 {
                        let e = res.threshold[i * m1 + j] - threshold[i * m1 + j];
                        tnum += e * e;
                        tden += 1.0;
                    }
                }
                for d in 0..n_dims {
                    let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let lrmse = (lnum / lden).sqrt();
            let trmse = (tnum / tden).sqrt();
            let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
            println!(
                "[grm MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                 loadRMSE={lrmse:.4} loadBias={lb:.4} threshRMSE={trmse:.4} thetaCorr={tc:.3}"
            );
            assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
            if skew {
                assert!(lrmse < 0.24, "skew load RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.55, "skew theta corr {tc} (D={n_dims})");
            } else {
                assert!(lb.abs() < 0.06, "load bias {lb} (D={n_dims})");
                assert!(lrmse < 0.16, "load RMSE {lrmse} (D={n_dims})");
                assert!(trmse < 0.16, "threshold RMSE {trmse} (D={n_dims})");
                assert!(tc > 0.6, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}
#[test]
fn grm_validation_sampling_rules_and_missing_paths() {
    let base = GrmConfig {
        q: 7,
        max_iter: 1,
        newton_iter: 1,
        ..GrmConfig::default()
    };
    let y = [0usize, 0, 1, 1, 0, 1, 1, 0];
    let observed = [true, false, true, true, true, true, true, true];
    let pattern = [1u8, 1];

    assert!(validate(&y, None, &pattern, 0, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &pattern, 4, 2, 1, 1, &base).is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GrmConfig {
            max_iter: 0,
            ..base
        }
    )
    .is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GrmConfig {
            tol: f64::NAN,
            ..base
        }
    )
    .is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GrmConfig { ridge: 0.0, ..base }
    )
    .is_err());
    assert!(validate(&y, None, &[], 4, 2, 0, 2, &base).is_err());
    assert!(validate(&y, None, &[1; 8], 4, 2, 4, 2, &base).is_err());
    assert!(validate(&y, None, &pattern, 4, 2, 1, 2, &GrmConfig { q: 3, ..base }).is_err());
    let halton = GrmConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 4,
        ..base
    };
    assert!(validate(&y, None, &[], 4, 2, 0, 2, &halton).is_err());
    assert!(validate(&y, None, &[1; 14], 4, 2, 7, 2, &halton).is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &GrmConfig {
            xi_points: 0,
            ..halton
        },
    )
    .is_err());
    assert!(validate(&y[..7], None, &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, Some(&[true]), &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[1], 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[2, 1], 4, 2, 1, 2, &base).is_err());
    let bad_y = [2usize, 0, 1, 1, 0, 1, 1, 0];
    assert!(validate(&bad_y, None, &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, None, &[0, 1], 4, 2, 1, 2, &base).is_err());
    assert!(validate(&y, Some(&[false; 8]), &pattern, 4, 2, 1, 2, &base).is_err());
    assert!(validate(&[0; 8], None, &pattern, 4, 2, 1, 2, &base).is_err());
    let cross = [1u8, 1, 1, 1];
    assert!(validate(&y, None, &cross, 4, 2, 2, 2, &base).is_err());
    for (n_items, xi_points, expected) in [
        (GM_MAX_NODES, GM_MAX_NODES, "count table"),
        (usize::MAX, GM_MAX_NODES, "overflows usize"),
    ] {
        assert!(validate(
            &[],
            None,
            &[],
            1,
            n_items,
            1,
            2,
            &GrmConfig {
                xi_rule: XiRuleKind::Halton,
                xi_points,
                ..base
            },
        )
        .unwrap_err()
        .contains(expected));
    }
    assert!(validate(&[], None, &[], usize::MAX, 2, 1, 2, &base)
        .unwrap_err()
        .contains("n_persons * n_items"));

    for xi_rule in [XiRuleKind::Halton, XiRuleKind::MonteCarlo] {
        let result = fit_grm(
            &y,
            Some(&observed),
            &pattern,
            4,
            2,
            1,
            2,
            &GrmConfig {
                xi_rule,
                xi_points: 16,
                xi_seed: 0,
                ..base
            },
        )
        .unwrap();
        assert_eq!(result.n_iter, 1);
        assert_eq!(result.termination_reason, "max_iter_reached");
        assert!(result.loglik_trace.iter().all(|value| value.is_finite()));
        assert!(result.theta.iter().all(|value| value.is_finite()));
    }
}

#[test]
fn grm_optimizer_and_em_diagnostics_cover_defensive_paths() {
    let dims = [0usize];
    let nodes = [-2.0, 0.0, 2.0];
    let zero_counts = vec![vec![0.0; 3]; 3];
    let initial = vec![1.0, 1.0, -1.0];
    assert_eq!(
        grm_m_step(initial.clone(), &dims, &nodes, 1, &zero_counts, 3, 0.1, 2),
        initial
    );
    let separated_counts = vec![
        vec![1000.0, 0.0, 0.0],
        vec![0.0, 1000.0, 0.0],
        vec![0.0, 0.0, 1000.0],
    ];
    let updated = grm_m_step(
        vec![0.0, 1.0, -1.0],
        &dims,
        &nodes,
        1,
        &separated_counts,
        3,
        -1.0e6,
        2,
    );
    assert!(updated.iter().all(|value| value.is_finite()));
    assert_eq!(checked_em_loglik_change(-10.0, None, 0).unwrap(), None);
    assert_eq!(
        checked_em_loglik_change(-9.5, Some(-10.0), 1).unwrap(),
        Some(0.5)
    );
    assert!(checked_em_loglik_change(f64::NAN, None, 2).is_err());
    assert!(checked_em_loglik_change(-10.5, Some(-10.0), 3).is_err());
}
