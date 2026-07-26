use super::*;
use crate::poly::fit_nominal as fit_nominal_unidim;

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
    fn cat(&mut self, probs: &[f64]) -> usize {
        let u = self.next_f64();
        let mut acc = 0.0;
        for (k, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return k;
            }
        }
        probs.len() - 1
    }
}

fn softmax(eta: &[f64]) -> Vec<f64> {
    let m = eta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let ex: Vec<f64> = eta.iter().map(|e| (e - m).exp()).collect();
    let s: f64 = ex.iter().sum();
    ex.iter().map(|e| e / s).collect()
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

/// Simulate multidimensional nominal responses from a dense slope tensor (n_items*n_cat*n_dims,
/// baseline cat 0 = 0), intercepts (n_items*n_cat), and traits (n_persons*n_dims).
fn simulate(
    slope: &[f64],
    intercept: &[f64],
    theta: &[f64],
    n: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    rng: &mut Lcg,
) -> Vec<usize> {
    let mut y = vec![0usize; n * n_items];
    let mut eta = vec![0.0f64; n_cat];
    for p in 0..n {
        for i in 0..n_items {
            eta[0] = 0.0;
            for k in 1..n_cat {
                let mut e = intercept[i * n_cat + k];
                for d in 0..n_dims {
                    e += slope[(i * n_cat + k) * n_dims + d] * theta[p * n_dims + d];
                }
                eta[k] = e;
            }
            let probs = softmax(&eta);
            y[p * n_items + i] = rng.cat(&probs);
        }
    }
    y
}

/// D = 1 REDUCTION: with D=1 and every item's S_i = {0}, fit_nominal reproduces
/// poly::fit_nominal BIT-EXACTLY (same init a_k=k / c_k=log(freq/freq0), same GH nodes+order,
/// same relative-tol + signed-monotone stopping, same nominal_m_step arithmetic generalized).
#[test]
fn nominal_reduces_to_fit_nominal_at_d1() {
    let (n, n_items, n_cat) = (1500usize, 6usize, 4usize);
    // truth: unidimensional nominal (a_k on the single dim, c_k intercepts)
    let mut rng = Lcg(202401);
    let mut slope = vec![0.0f64; n_items * n_cat * 1];
    let mut intercept = vec![0.0f64; n_items * n_cat];
    for i in 0..n_items {
        for k in 1..n_cat {
            slope[(i * n_cat + k) * 1] = 0.4 + 0.35 * k as f64 + 0.05 * i as f64;
            intercept[i * n_cat + k] = -0.3 + 0.2 * k as f64 - 0.1 * i as f64;
        }
    }
    let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y = simulate(&slope, &intercept, &theta, n, n_items, 1, n_cat, &mut rng);
    let pattern = vec![1u8; n_items]; // D=1, all load dim 0
    let cfg = NominalConfig {
        q: 21,
        ..NominalConfig::default()
    };
    let mm = fit_nominal(&y, None, &pattern, n, n_items, 1, n_cat, &cfg).unwrap();
    let fnom = fit_nominal_unidim(&y, None, n, n_items, n_cat, 21, 500, 1e-6).unwrap();
    // loglik traces bit-identical
    assert_eq!(
        mm.loglik_trace.len(),
        fnom.loglik_trace.len(),
        "trace length"
    );
    let dtrace = mm
        .loglik_trace
        .iter()
        .zip(&fnom.loglik_trace)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(dtrace < 1e-9, "loglik trace diff {dtrace}");
    // scores/intercepts bit-identical (fit_nominal stores z = n_cat-1 free per item; my slope
    // has baseline cat 0 = 0 then a_1..a_{K-1} on dim 0).
    let z = n_cat - 1;
    let mut dmax = 0.0f64;
    for i in 0..n_items {
        for k in 1..n_cat {
            let mine_a = mm.slope[(i * n_cat + k) * 1];
            let theirs_a = fnom.scores[i][k - 1];
            dmax = dmax.max((mine_a - theirs_a).abs());
            let mine_c = mm.intercept[i * n_cat + k];
            let theirs_c = fnom.intercepts[i][k - 1];
            dmax = dmax.max((mine_c - theirs_c).abs());
        }
    }
    let _ = z;
    assert!(dmax < 1e-9, "param diff {dmax}");
    assert_eq!(mm.n_parameters, n_items * 2 * (n_cat - 1));
}

/// Deterministic FD GRADIENT anchor on FIXED nodes at D=2 (GH, dims=[0,1]) AND D=4 (Halton,
/// NON-IDENTITY dims=[0,2,3]), with M=4 categories and RANDOM DISTINCT per-category counts so a
/// category<->dimension index transposition or a sign error produces a detectably wrong slot.
/// The M-step uses an FD Hessian, so the correctness-bearing map lives in the GRADIENT — pin
/// EVERY free slot (all a_kd and all c_k) against central differences of the objective.
#[test]
fn nominal_gradient_matches_finite_difference() {
    let n_cat = 4usize;
    for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter() {
        let l = dims.len();
        let nodes: Vec<f64>;
        let n_nodes: usize;
        if n_dims == 2 {
            let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: 15 }, n_dims).unwrap();
            n_nodes = xn.logw.len();
            nodes = xn.grid;
        } else {
            let xn = build_xi_nodes(
                XiRule::Halton {
                    n: 200,
                    shift_seed: 0,
                },
                n_dims,
            )
            .unwrap();
            n_nodes = xn.logw.len();
            nodes = xn.grid;
        }
        let mut rng = Lcg(2718 + n_dims as u64);
        // RANDOM DISTINCT expected counts per (node, category) — not equal across categories.
        let counts: Vec<Vec<f64>> = (0..n_nodes)
            .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 3.0).collect())
            .collect();
        // free param vector: [a_{1,d..}, a_{2,d..}, .., c_1, c_2, ..] with distinct values
        let z = n_cat - 1;
        let mut params = vec![0.0f64; z * l + z];
        for m in 0..(z * l) {
            params[m] = 0.3 + 0.17 * m as f64 - if m % 2 == 0 { 0.4 } else { 0.0 };
        }
        for k in 0..z {
            params[z * l + k] = -0.2 + 0.31 * k as f64;
        }
        let (_f0, grad) = nm_item_neg_ll_grad(&params, dims, &nodes, n_dims, &counts, n_cat);
        let eps = 1e-6;
        for j in 0..params.len() {
            let mut pp = params.clone();
            pp[j] += eps;
            let (fp, _) = nm_item_neg_ll_grad(&pp, dims, &nodes, n_dims, &counts, n_cat);
            let mut pm = params.clone();
            pm[j] -= eps;
            let (fm, _) = nm_item_neg_ll_grad(&pm, dims, &nodes, n_dims, &counts, n_cat);
            let fd = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[j] - fd).abs() < 1e-4,
                "grad[{j}] {} vs fd {fd} (D={n_dims})",
                grad[j]
            );
        }
    }
}

// Per-dimension reflection alignment: flip dim d of `est` (negate every category slope on d) so
// its pure-anchor item's category-1 slope matches the sign of `truth`'s. Deterministic; applied
// identically so a genuine sign/compensation bug in `est` survives as a mismatch elsewhere.
fn align_reflection(
    est: &mut [f64],
    truth: &[f64],
    anchor: &[usize],
    n_items: usize,
    n_cat: usize,
    n_dims: usize,
) {
    for d in 0..n_dims {
        let a = anchor[d];
        let ref_est = est[(a * n_cat + 1) * n_dims + d];
        let ref_tru = truth[(a * n_cat + 1) * n_dims + d];
        if ref_est * ref_tru < 0.0 {
            for i in 0..n_items {
                for k in 0..n_cat {
                    est[(i * n_cat + k) * n_dims + d] = -est[(i * n_cat + k) * n_dims + d];
                }
            }
        }
    }
}

/// D = 2 recovery on GH nodes: pure anchors per dim + a CROSS-loader carrying a genuinely
/// NEGATIVE category slope AND two OPPOSITE-sign sibling categories on the same loaded dim
/// (which catches a mutation collapsing the free per-category slopes to a shared scalar
/// discrimination). Assessed up to per-dimension reflection (aligned to truth).
#[test]
fn nominal_recovers_d2_with_signed_categories() {
    let (n_dims, n_cat) = (2usize, 3usize);
    // items 0,1 pure dim0; items 2,3 pure dim1; item 4 cross-loader {0,1}.
    let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
    let n_items = 5usize;
    let anchor = vec![0usize, 2]; // pure anchor per dim
    let mut slope = vec![0.0f64; n_items * n_cat * n_dims];
    let mut intercept = vec![0.0f64; n_items * n_cat];
    // pure dim0 anchors: positive, distinct per category
    slope[(0 * n_cat + 1) * n_dims + 0] = 1.4;
    slope[(0 * n_cat + 2) * n_dims + 0] = 0.8;
    slope[(1 * n_cat + 1) * n_dims + 0] = 1.0;
    slope[(1 * n_cat + 2) * n_dims + 0] = 1.3;
    // pure dim1 anchors
    slope[(2 * n_cat + 1) * n_dims + 1] = 1.2;
    slope[(2 * n_cat + 2) * n_dims + 1] = 0.9;
    slope[(3 * n_cat + 1) * n_dims + 1] = 1.1;
    slope[(3 * n_cat + 2) * n_dims + 1] = 1.4;
    // cross-loader (item 4): dim0 category-1 NEGATIVE, category-2 POSITIVE (opposite siblings);
    // dim1 positive.
    slope[(4 * n_cat + 1) * n_dims + 0] = -1.1; // negative sibling
    slope[(4 * n_cat + 2) * n_dims + 0] = 1.0; // positive sibling (same dim0)
    slope[(4 * n_cat + 1) * n_dims + 1] = 0.9;
    slope[(4 * n_cat + 2) * n_dims + 1] = 0.7;
    for i in 0..n_items {
        for k in 1..n_cat {
            intercept[i * n_cat + k] = -0.2 + 0.15 * k as f64 - 0.05 * i as f64;
        }
    }
    let n = 6000usize;
    let mut rng = Lcg(9090);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(
        &slope, &intercept, &theta, n, n_items, n_dims, n_cat, &mut rng,
    );
    let cfg = NominalConfig {
        q: 21,
        ..NominalConfig::default()
    };
    let res = fit_nominal(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    assert!(res.converged);
    // baseline + off-pattern EXACT zero
    for i in 0..n_items {
        for d in 0..n_dims {
            assert_eq!(
                res.slope[(i * n_cat + 0) * n_dims + d],
                0.0,
                "baseline slope zero"
            );
            if pattern[i * n_dims + d] == 0 {
                for k in 0..n_cat {
                    assert_eq!(
                        res.slope[(i * n_cat + k) * n_dims + d],
                        0.0,
                        "off-pattern zero"
                    );
                }
            }
        }
        assert_eq!(res.intercept[i * n_cat + 0], 0.0, "baseline intercept zero");
    }
    let mut est = res.slope.clone();
    align_reflection(&mut est, &slope, &anchor, n_items, n_cat, n_dims);
    assert!(
        rmse(&est, &slope) < 0.16,
        "slope RMSE {}",
        rmse(&est, &slope)
    );
    // the negative cross-loader category-1 slope on dim0 (sign pinned by anchor item 0), and its
    // opposite-sign sibling category-2 — both recovered with the right sign.
    assert!(
        est[(4 * n_cat + 1) * n_dims + 0] < -0.4,
        "neg sibling: {}",
        est[(4 * n_cat + 1) * n_dims + 0]
    );
    assert!(
        est[(4 * n_cat + 2) * n_dims + 0] > 0.4,
        "pos sibling: {}",
        est[(4 * n_cat + 2) * n_dims + 0]
    );
    // per-dim trait EAP correlation (sign-aligned)
    for d in 0..n_dims {
        let mut th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
        let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
        // align theta sign to truth via the same anchor reference
        let ref_est = res.slope[(anchor[d] * n_cat + 1) * n_dims + d];
        let ref_tru = slope[(anchor[d] * n_cat + 1) * n_dims + d];
        if ref_est * ref_tru < 0.0 {
            for v in th.iter_mut() {
                *v = -*v;
            }
        }
        assert!(corr(&th, &tt) > 0.6, "theta{d} corr {}", corr(&th, &tt));
    }
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-9, "EM monotone");
    }
}

/// Softmax-sum, structural zeros, parameter count, and validation guards.
#[test]
fn nominal_validates_and_structural_invariants() {
    let (n_dims, n_cat) = (2usize, 3usize);
    let pattern: Vec<u8> = vec![1, 0, 0, 1, 1, 1];
    let n_items = 3usize;
    let n = 400usize;
    let mut slope = vec![0.0f64; n_items * n_cat * n_dims];
    let mut intercept = vec![0.0f64; n_items * n_cat];
    slope[(0 * n_cat + 1) * n_dims + 0] = 1.2;
    slope[(0 * n_cat + 2) * n_dims + 0] = 1.0;
    slope[(1 * n_cat + 1) * n_dims + 1] = 1.1;
    slope[(1 * n_cat + 2) * n_dims + 1] = 0.9;
    slope[(2 * n_cat + 1) * n_dims + 0] = 0.8;
    slope[(2 * n_cat + 2) * n_dims + 0] = 0.7;
    slope[(2 * n_cat + 1) * n_dims + 1] = 0.9;
    slope[(2 * n_cat + 2) * n_dims + 1] = 0.6;
    for i in 0..n_items {
        for k in 1..n_cat {
            intercept[i * n_cat + k] = 0.1 * k as f64;
        }
    }
    let mut rng = Lcg(55);
    let mut theta = vec![0.0f64; n * n_dims];
    for v in theta.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(
        &slope, &intercept, &theta, n, n_items, n_dims, n_cat, &mut rng,
    );
    let cfg = NominalConfig {
        q: 15,
        max_iter: 30,
        ..NominalConfig::default()
    };
    let res = fit_nominal(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
    // parameter count invariant: sum_i (n_cat-1)*(|S_i|+1)  = 2*(1+1) [item0] + 2*(1+1) [item1] + 2*(2+1) [item2]
    assert_eq!(res.n_parameters, 2 * 2 + 2 * 2 + 2 * 3);
    // softmax probabilities sum to 1 at a few nodes (recompute a category dist for item 2)
    let eta = [
        0.0,
        slope[(2 * n_cat + 1) * n_dims + 0],
        slope[(2 * n_cat + 2) * n_dims + 0],
    ];
    let p = softmax(&eta);
    assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    // validation: GH D=4 rejected; no pure anchor rejected; category >= n_cat rejected;
    // unobserved category rejected.
    let gh4 = NominalConfig::default();
    let pat4: Vec<u8> = (0..4)
        .flat_map(|d| (0..4).map(move |k| (k == d) as u8))
        .collect();
    // y4 cycles through every category (so the unobserved-category guard does NOT fire): the
    // GH D>3 bound must be the SOLE rejection reason, else a NM_MAX_DIMS mutation survives (at
    // q=21, 21^4=194481 nodes sits under the node cap, so only the dim bound rejects it).
    let y4: Vec<usize> = (0..n * 4).map(|idx| idx % n_cat).collect();
    assert!(
        fit_nominal(&y4, None, &pat4, n, 4, 4, n_cat, &gh4).is_err(),
        "GH D=4 rejected"
    );
    // no pure anchor for either dim (all three items load BOTH dims). Uses the full 3-item y so
    // the y-length check passes and the pure-anchor identification guard is the failing branch.
    let no_anchor: Vec<u8> = vec![1, 1, 1, 1, 1, 1];
    assert!(
        fit_nominal(&y, None, &no_anchor, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "no pure anchor rejected"
    );
    // category >= n_cat
    let mut ybad = y.clone();
    ybad[0] = n_cat;
    assert!(
        fit_nominal(&ybad, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "bad category rejected"
    );
    // an item with an unobserved category (force item 0 to never show category 2)
    let mut ygap = y.clone();
    for p in 0..n {
        if ygap[p * n_items + 0] == 2 {
            ygap[p * n_items + 0] = 1;
        }
    }
    assert!(
        fit_nominal(&ygap, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
        "unobserved category rejected"
    );
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the multidimensional nominal at D=2 and
/// D=3 under normal AND per-dim-standardized right-skew traits, assessed up to per-dimension
/// reflection (aligned to truth) with label-invariant backstops (modal-category agreement,
/// per-dim trait EAP correlation). Per-rep monotone-EM + finiteness canaries.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_nominal_recovery_500() {
    let reps = 500usize;
    let n_cat = 3usize;
    for &(n_dims, q, n) in [(2usize, 15usize, 2500usize), (3usize, 11usize, 2000usize)].iter() {
        // 2 pure anchors per dim + one cross-loader per dim.
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
        let anchor: Vec<usize> = (0..n_dims).map(|d| 2 * d).collect();
        let mut slope = vec![0.0f64; n_items * n_cat * n_dims];
        let mut intercept = vec![0.0f64; n_items * n_cat];
        for d in 0..n_dims {
            slope[((2 * d) * n_cat + 1) * n_dims + d] = 1.3;
            slope[((2 * d) * n_cat + 2) * n_dims + d] = 0.8;
            slope[((2 * d + 1) * n_cat + 1) * n_dims + d] = 1.0;
            slope[((2 * d + 1) * n_cat + 2) * n_dims + d] = 1.2;
        }
        for d in 0..n_dims {
            let ci = 2 * n_dims + d;
            slope[(ci * n_cat + 1) * n_dims + d] = 1.0;
            slope[(ci * n_cat + 2) * n_dims + d] = 0.7;
            let d2 = (d + 1) % n_dims;
            slope[(ci * n_cat + 1) * n_dims + d2] = if d % 2 == 0 { 0.7 } else { -0.7 };
            slope[(ci * n_cat + 2) * n_dims + d2] = if d % 2 == 0 { -0.6 } else { 0.6 };
        }
        for i in 0..n_items {
            for k in 1..n_cat {
                intercept[i * n_cat + k] = -0.2 + 0.2 * k as f64 - 0.03 * i as f64;
            }
        }
        for &skew in [false, true].iter() {
            let (mut snum, mut sden, mut sbias) = (0.0f64, 0.0f64, 0.0f64);
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
                    &slope, &intercept, &theta, n, n_items, n_dims, n_cat, &mut rng,
                );
                let cfg = NominalConfig {
                    q,
                    ..NominalConfig::default()
                };
                let res = fit_nominal(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
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
                let mut est = res.slope.clone();
                align_reflection(&mut est, &slope, &anchor, n_items, n_cat, n_dims);
                for i in 0..n_items {
                    for k in 1..n_cat {
                        for d in 0..n_dims {
                            if pattern[i * n_dims + d] != 0 {
                                let e = est[(i * n_cat + k) * n_dims + d]
                                    - slope[(i * n_cat + k) * n_dims + d];
                                snum += e * e;
                                sden += 1.0;
                                sbias += e;
                            }
                        }
                    }
                }
                for d in 0..n_dims {
                    let mut th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| theta[j * n_dims + d]).collect();
                    let ref_est = res.slope[(anchor[d] * n_cat + 1) * n_dims + d];
                    let ref_tru = slope[(anchor[d] * n_cat + 1) * n_dims + d];
                    if ref_est * ref_tru < 0.0 {
                        for v in th.iter_mut() {
                            *v = -*v;
                        }
                    }
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let srmse = (snum / sden).sqrt();
            let (sb, tc, conv) = (sbias / sden, csum / ccnt, nconv as f64 / reps as f64);
            println!(
                "[nominal MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                 slopeRMSE={srmse:.4} slopeBias={sb:.4} thetaCorr={tc:.3}"
            );
            assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
            if skew {
                assert!(srmse < 0.30, "skew slope RMSE {srmse} (D={n_dims})");
                assert!(tc > 0.45, "skew theta corr {tc} (D={n_dims})");
            } else {
                assert!(sb.abs() < 0.08, "slope bias {sb} (D={n_dims})");
                assert!(srmse < 0.22, "slope RMSE {srmse} (D={n_dims})");
                assert!(tc > 0.5, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}
#[test]
fn nominal_validation_sampling_rules_and_missing_paths() {
    let base = NominalConfig {
        q: 7,
        max_iter: 1,
        newton_iter: 1,
        ..NominalConfig::default()
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
        &NominalConfig {
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
        &NominalConfig {
            max_iter: NM_MAX_ITER + 1,
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
        &NominalConfig {
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
        &NominalConfig { ridge: 0.0, ..base }
    )
    .is_err());
    assert!(validate(&y, None, &[], 4, 2, 0, 2, &base).is_err());
    assert!(validate(&y, None, &[1; 8], 4, 2, 4, 2, &base).is_err());
    assert!(validate(
        &y,
        None,
        &pattern,
        4,
        2,
        1,
        2,
        &NominalConfig { q: 3, ..base }
    )
    .is_err());
    let halton = NominalConfig {
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
        &NominalConfig {
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
        (NM_MAX_NODES, NM_MAX_NODES, "count table"),
        (usize::MAX, NM_MAX_NODES, "overflows usize"),
    ] {
        assert!(validate(
            &[],
            None,
            &[],
            1,
            n_items,
            1,
            2,
            &NominalConfig {
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
        let result = fit_nominal(
            &y,
            Some(&observed),
            &pattern,
            4,
            2,
            1,
            2,
            &NominalConfig {
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
fn nominal_optimizer_and_em_diagnostics_cover_defensive_paths() {
    let dims = [0usize];
    let nodes = [-2.0, 0.0, 2.0];
    let zero_counts = vec![vec![0.0; 3]; 3];
    let initial = vec![1.0, 2.0, 0.0, 0.0];
    assert_eq!(
        nm_m_step(initial.clone(), &dims, &nodes, 1, &zero_counts, 3, 0.1, 2),
        initial
    );
    let separated_counts = vec![
        vec![1000.0, 0.0, 0.0],
        vec![0.0, 1000.0, 0.0],
        vec![0.0, 0.0, 1000.0],
    ];
    let updated = nm_m_step(
        vec![0.0; 4],
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
