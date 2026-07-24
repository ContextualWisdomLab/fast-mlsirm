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
    fn bern(&mut self, p: f64) -> f64 {
        if self.next_f64() < p {
            1.0
        } else {
            0.0
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}
fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
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

/// Simulate compensatory M2PL responses from loadings (J*D), intercepts (J), and person
/// traits (N*D) via the same additive-logit model the estimator recovers.
fn simulate(
    loading: &[f64],
    intercept: &[f64],
    thetas: &[f64],
    n: usize,
    n_items: usize,
    n_dims: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let mut y = vec![0.0f64; n * n_items];
    for j in 0..n {
        for i in 0..n_items {
            let mut eta = intercept[i];
            for d in 0..n_dims {
                eta += loading[i * n_dims + d] * thetas[j * n_dims + d];
            }
            y[j * n_items + i] = rng.bern(sigmoid(eta));
        }
    }
    y
}

/// The orthogonal product GH grid reproduces the N(0, I) moments (sum w = 1, E[theta_d]=0,
/// Var=1, Cov=0) - catches a transposed nodes[g*D+d] or a bad Cartesian product.
#[test]
fn mirt_grid_moments() {
    let (nodes, logw) = build_grid(2, 15);
    let n = logw.len();
    let w: Vec<f64> = logw.iter().map(|l| l.exp()).collect();
    assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-10, "sum w");
    let (mut e0, mut e1, mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for g in 0..n {
        let (t0, t1) = (nodes[g * 2], nodes[g * 2 + 1]);
        e0 += w[g] * t0;
        e1 += w[g] * t1;
        v0 += w[g] * t0 * t0;
        v1 += w[g] * t1 * t1;
        c01 += w[g] * t0 * t1;
    }
    assert!(e0.abs() < 1e-9 && e1.abs() < 1e-9, "means");
    assert!(
        (v0 - 1.0).abs() < 1e-9 && (v1 - 1.0).abs() < 1e-9,
        "variances"
    );
    assert!(c01.abs() < 1e-9, "cross moment (orthogonality)");
}

/// Deterministic anchor: the analytic item gradient AND the full (n_i+1)x(n_i+1) Hessian
/// block - including the off-diagonal cross-Hessian H_{a0,a1} and the local->pattern-dim
/// map - match central finite differences of item_obj at D=2 for a BOTH-loading item, to
/// < 1e-4. A dims[k] indexing bug or a missing cross term fails this with no MC noise.
#[test]
fn mirt_item_grad_hess_matches_finite_difference() {
    // Two configs: identity map (dims=[0,1] on a D=2 grid) AND a NON-IDENTITY map
    // (dims=[0,2] on a D=3 grid, so nodes index dims[k]!=k) — the latter genuinely pins
    // the local-param -> pattern-dimension map that a k-vs-dims[k] bug would break.
    for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (3usize, vec![0usize, 2])].iter() {
        let (nodes, logw) = build_grid(n_dims, 15);
        let n_nodes = logw.len();
        let mut rng = Lcg(99);
        let (mut n_ig, mut r_ig) = (vec![0.0f64; n_nodes], vec![0.0f64; n_nodes]);
        for g in 0..n_nodes {
            n_ig[g] = 1.0 + rng.next_f64() * 3.0;
            r_ig[g] = n_ig[g] * rng.next_f64();
        }
        let (a, b) = (vec![0.8f64, -0.5], 0.3f64); // dims.len() == 2 for both configs
        let (ra, rb) = (1e-3, 1e-3);
        let np = dims.len() + 1;
        let (grad, amat) =
            item_grad_hess(dims, &a, b, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
        let obj = |aa: &[f64], bb: f64| {
            item_obj(dims, aa, bb, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb)
        };
        let eps = 1e-6;
        let perturb = |k: usize, s: f64| -> (Vec<f64>, f64) {
            let mut aa = a.clone();
            let mut bb = b;
            if k < dims.len() {
                aa[k] += s;
            } else {
                bb += s;
            }
            (aa, bb)
        };
        for k in 0..np {
            let (ap, bp) = perturb(k, eps);
            let (am, bm) = perturb(k, -eps);
            let fd = (obj(&ap, bp) - obj(&am, bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 1e-4,
                "grad[{k}] {} vs fd {fd} (D={n_dims})",
                grad[k]
            );
        }
        for jp in 0..np {
            let (ap, bp) = perturb(jp, eps);
            let (am, bm) = perturb(jp, -eps);
            let (gp, _) =
                item_grad_hess(dims, &ap, bp, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
            let (gm, _) =
                item_grad_hess(dims, &am, bm, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
            for k in 0..np {
                let dfd = (gp[k] - gm[k]) / (2.0 * eps);
                assert!((dfd + amat[k][jp]).abs() < 1e-4, "H[{k}][{jp}] D={n_dims}");
            }
        }
    }
}

/// D=1 (all items load the single dimension) recovers known 2PL parameters and matches
/// fit_mmle_2pl on the same data (gh_rule(41) is the same 41-node grid as mmle::GH_NODES).
#[test]
fn mirt_reduces_to_2pl_at_d1() {
    let (n, n_items) = (1500usize, 12usize);
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.7 + 0.1 * i as f64).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -1.0 + 0.18 * i as f64).collect();
    let mut rng = Lcg(2024);
    let thetas: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
    let y = simulate(&a_true, &b_true, &thetas, n, n_items, 1, &mut rng);
    let observed = vec![true; n * n_items];
    let pattern = vec![1u8; n_items];
    let cfg = TwoPlConfig {
        q: 41,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, 1, &cfg).unwrap();
    assert!(
        rmse(&res.loading, &a_true) < 0.12,
        "loading RMSE {}",
        rmse(&res.loading, &a_true)
    );
    assert!(rmse(&res.intercept, &b_true) < 0.12, "intercept RMSE");
    let m = fit_mmle_2pl(&y, &observed, n, n_items, &MmleConfig::default());
    assert!(
        rmse(&res.loading, &m.a) < 1e-2,
        "vs mmle a {}",
        rmse(&res.loading, &m.a)
    );
    assert!(
        rmse(&res.intercept, &m.b) < 1e-2,
        "vs mmle b {}",
        rmse(&res.intercept, &m.b)
    );
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "monotone");
    }
}

/// Non-trivial D=2 compensatory recovery: a confirmatory pattern (dim0-only, dim1-only,
/// AND both-loading items) with ASYMMETRIC, non-centered true loadings INCLUDING genuinely
/// NEGATIVE loadings. Recovers loadings with correct sign and per-dimension theta EAP
/// correlation. A dim-swap or a compensation-sign bug fails this.
#[test]
fn mirt_recovers_compensatory_d2() {
    let n_dims = 2usize;
    let mut pattern: Vec<u8> = Vec::new();
    for _ in 0..4 {
        pattern.extend_from_slice(&[1, 0]);
    }
    for _ in 0..4 {
        pattern.extend_from_slice(&[0, 1]);
    }
    for _ in 0..3 {
        pattern.extend_from_slice(&[1, 1]);
    }
    let n_items = 11usize;
    let a0 = [1.2, 0.8, 1.5, -0.9];
    let a1 = [1.0, 1.3, 0.7, 1.1];
    let both = [(0.9, 1.1), (1.2, -0.7), (0.8, 0.9)];
    let mut loading = vec![0.0f64; n_items * n_dims];
    for i in 0..4 {
        loading[i * 2] = a0[i];
        loading[(4 + i) * 2 + 1] = a1[i];
    }
    for i in 0..3 {
        loading[(8 + i) * 2] = both[i].0;
        loading[(8 + i) * 2 + 1] = both[i].1;
    }
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.16 * i as f64).collect();
    let n = 4000usize;
    let mut rng = Lcg(777);
    let mut thetas = vec![0.0f64; n * n_dims];
    for j in 0..n {
        thetas[j * 2] = rng.normal();
        thetas[j * 2 + 1] = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = TwoPlConfig {
        q: 21,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.loading[i * n_dims + d], 0.0, "unloaded exactly zero");
            }
        }
    }
    assert!(
        rmse(&res.loading, &loading) < 0.12,
        "loading RMSE {}",
        rmse(&res.loading, &loading)
    );
    assert!(
        res.loading[3 * 2] < -0.5,
        "negative dim0 loading recovered: {}",
        res.loading[3 * 2]
    );
    assert!(
        res.loading[9 * 2 + 1] < -0.3,
        "negative cross-loading: {}",
        res.loading[9 * 2 + 1]
    );
    let t0h: Vec<f64> = (0..n).map(|j| res.theta[j * 2]).collect();
    let t0t: Vec<f64> = (0..n).map(|j| thetas[j * 2]).collect();
    let t1h: Vec<f64> = (0..n).map(|j| res.theta[j * 2 + 1]).collect();
    let t1t: Vec<f64> = (0..n).map(|j| thetas[j * 2 + 1]).collect();
    // EAP shrinks toward the prior, so the true-vs-EAP correlation is bounded by test
    // information (not N); ~0.75-0.85 is the expected range. The POSITIVE sign is the key
    // faithfulness check (a dim-swap or sign bug would give a near-zero or negative corr).
    assert!(corr(&t0h, &t0t) > 0.70, "theta0 corr {}", corr(&t0h, &t0t));
    assert!(corr(&t1h, &t1t) > 0.70, "theta1 corr {}", corr(&t1h, &t1t));
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "monotone");
    }
}

/// Deterministic reflection tests. (b) `flip_corr_dim` negates EXACTLY the off-diagonals that
/// involve the flipped dimension (packed pairs (i,j), i<j) and leaves the rest untouched.
/// (a) A fit whose largest pure anchor on a dimension is reverse-keyed (true loading strongly
/// negative) is canonicalized by the reflection so that anchor ends POSITIVE and the dimension's
/// co-loaders flip sign — deleting the reflection block leaves the raw negative anchor and fails.
#[test]
fn mirt_reflection_flips_negative_anchor() {
    // (b) flip_corr_dim on D=4: pairs are m0=(0,1) m1=(0,2) m2=(0,3) m3=(1,2) m4=(1,3) m5=(2,3).
    // Flipping dim 1 must negate exactly m0,(0,1) m3,(1,2) m4,(1,3) and leave m1,m2,m5 alone.
    let mut off = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    flip_corr_dim(&mut off, 4, 1);
    assert_eq!(off, vec![-0.1, 0.2, 0.3, -0.4, -0.5, 0.6]);

    // (a) reverse-keyed largest anchor on dim 0. Items 0,1 pure dim0; 2,3 pure dim1; 4 cross.
    let n_dims = 2usize;
    let n_items = 5usize;
    let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
    let mut loading = vec![0.0f64; n_items * n_dims];
    loading[0 * 2] = -1.8; // reverse-keyed anchor, largest |loading| on dim 0
    loading[1 * 2] = 1.0;
    loading[2 * 2 + 1] = 1.2;
    loading[3 * 2 + 1] = 1.0;
    loading[4 * 2] = 0.9;
    loading[4 * 2 + 1] = 0.8;
    let intercept = vec![0.1, -0.2, 0.15, -0.1, 0.05];
    let n = 3000usize;
    let mut rng = Lcg(4242);
    let mut thetas = vec![0.0f64; n * n_dims];
    for j in 0..n {
        thetas[j * 2] = rng.normal();
        thetas[j * 2 + 1] = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = TwoPlConfig {
        q: 21,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
    // Canonical output: the largest pure anchor on dim 0 (item 0) ends POSITIVE; because the
    // whole dimension was reflected, the positively-keyed co-item (item 1) ends NEGATIVE.
    assert!(
        res.loading[0 * 2] > 0.8,
        "reflected anchor should be positive: {}",
        res.loading[0 * 2]
    );
    assert!(
        res.loading[1 * 2] < -0.3,
        "co-item flipped negative: {}",
        res.loading[1 * 2]
    );
}

/// Two-sided reduction anchor at D=2: the Halton QMC fit AGREES with the Gauss-Hermite fit
/// within QMC error AND DIFFERS from it bit-wise. The disagreement guard is essential — a
/// silent fallback to GH nodes on the Halton arm would make the two fits bit-identical and
/// pass a one-sided within-error check trivially.
#[test]
fn qmc_reduces_to_gh_within_error_d2() {
    let n_dims = 2usize;
    let mut pattern: Vec<u8> = Vec::new();
    for _ in 0..4 {
        pattern.extend_from_slice(&[1, 0]);
    }
    for _ in 0..4 {
        pattern.extend_from_slice(&[0, 1]);
    }
    pattern.extend_from_slice(&[1, 1]);
    let n_items = 9usize;
    let mut loading = vec![0.0f64; n_items * n_dims];
    for i in 0..4 {
        loading[i * 2] = 1.0 + 0.15 * i as f64;
        loading[(4 + i) * 2 + 1] = 1.1 - 0.1 * i as f64;
    }
    loading[8 * 2] = 0.9;
    loading[8 * 2 + 1] = 0.8;
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.15 * i as f64).collect();
    // A fixed moderate sample is enough to distinguish the QMC path from GH while keeping this
    // structural regression inside the repository's coverage-job budget.
    let (n, xi_points, max_error) = (750usize, 1500usize, 0.15);
    let mut rng = Lcg(1357);
    let mut thetas = vec![0.0f64; n * n_dims];
    for v in thetas.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let gh = fit_2pl(
        &y,
        &observed,
        &pattern,
        n,
        n_items,
        n_dims,
        &TwoPlConfig {
            q: 21,
            max_iter: 200,
            tol: 1e-5,
            ..TwoPlConfig::default()
        },
    )
    .unwrap();
    let qmc = fit_2pl(
        &y,
        &observed,
        &pattern,
        n,
        n_items,
        n_dims,
        &TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points,
            xi_seed: 0,
            max_iter: 200,
            tol: 1e-5,
            ..TwoPlConfig::default()
        },
    )
    .unwrap();
    let max_abs = gh
        .loading
        .iter()
        .zip(&qmc.loading)
        .chain(gh.intercept.iter().zip(&qmc.intercept))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    assert!(
        max_abs < max_error,
        "QMC and GH disagree beyond QMC error: {max_abs}"
    );
    assert!(
        max_abs > 1e-10,
        "QMC fit is bit-identical to GH (silent GH fallback?)"
    );
    for fit in [&gh, &qmc] {
        assert!(fit.converged, "fit did not converge: {fit:?}");
        assert!(fit.n_iter < 200);
        assert!(fit.final_loglik_change <= 1e-5);
    }
}

/// Deterministic FD anchor on a FIXED Halton node set at D=4 with a NON-IDENTITY dims map
/// [0,2,3] (so nodes are indexed dims[k] != k). Pins the analytic gradient and the full
/// (n_i+1)^2 Hessian — including the off-diagonal cross-Hessian and the local->pattern
/// dimension map — against central differences of item_obj to < 1e-4, on the SAME QMC nodes
/// the estimator uses. This is deterministic (fixed seed) and node-source specific, so a
/// cross-Hessian sign error or a dims[k] mis-map at D>3 fails here with no MC noise. (The
/// grid LAYOUT itself is pinned independently in nodes::halton_grid_layout_is_prime_per_axis.)
#[test]
fn qmc_item_grad_hess_matches_fd_on_halton_d4() {
    let n_dims = 4usize;
    let dims = vec![0usize, 2, 3];
    let xn = build_xi_nodes(
        XiRule::Halton {
            n: 240,
            shift_seed: 0,
        },
        n_dims,
    )
    .unwrap();
    let nodes = &xn.grid;
    let n_nodes = xn.logw.len();
    let mut rng = Lcg(2718);
    let (mut n_ig, mut r_ig) = (vec![0.0f64; n_nodes], vec![0.0f64; n_nodes]);
    for g in 0..n_nodes {
        n_ig[g] = 1.0 + rng.next_f64() * 3.0;
        r_ig[g] = n_ig[g] * rng.next_f64();
    }
    let (a, b) = (vec![0.7f64, -0.6, 0.9], 0.2f64);
    let (ra, rb) = (1e-3, 1e-3);
    let np = dims.len() + 1;
    let (grad, amat) = item_grad_hess(&dims, &a, b, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
    let obj =
        |aa: &[f64], bb: f64| item_obj(&dims, aa, bb, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
    let eps = 1e-6;
    let perturb = |k: usize, s: f64| -> (Vec<f64>, f64) {
        let mut aa = a.clone();
        let mut bb = b;
        if k < dims.len() {
            aa[k] += s;
        } else {
            bb += s;
        }
        (aa, bb)
    };
    for k in 0..np {
        let (ap, bp) = perturb(k, eps);
        let (am, bm) = perturb(k, -eps);
        let fd = (obj(&ap, bp) - obj(&am, bm)) / (2.0 * eps);
        assert!(
            (grad[k] - fd).abs() < 1e-4,
            "grad[{k}] {} vs fd {fd}",
            grad[k]
        );
    }
    for jp in 0..np {
        let (ap, bp) = perturb(jp, eps);
        let (am, bm) = perturb(jp, -eps);
        let (gp, _) = item_grad_hess(&dims, &ap, bp, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
        let (gm, _) = item_grad_hess(&dims, &am, bm, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
        for k in 0..np {
            let dfd = (gp[k] - gm[k]) / (2.0 * eps);
            assert!((dfd + amat[k][jp]).abs() < 1e-4, "H[{k}][{jp}]");
        }
    }
}

/// D=4 orthogonal recovery on Halton QMC nodes (the headline D>3 capability the GH grid cannot
/// reach). Confirmatory pattern: 2 pure anchors per dimension + cross-loaders INCLUDING a
/// genuine negative one, which is asserted recovered < 0 explicitly (a compensation-sign bug on
/// a shared dimension cannot be averaged away by an aggregate RMSE).
#[test]
fn qmc_recovers_compensatory_d4() {
    let n_dims = 4usize;
    let mut pattern: Vec<u8> = Vec::new();
    for d in 0..n_dims {
        for _ in 0..2 {
            let mut row = vec![0u8; n_dims];
            row[d] = 1;
            pattern.extend_from_slice(&row);
        }
    }
    // cross-loaders: (0,1) with a NEGATIVE dim-1 loading; (1,2); (2,3).
    pattern.extend_from_slice(&[1, 1, 0, 0]);
    pattern.extend_from_slice(&[0, 1, 1, 0]);
    pattern.extend_from_slice(&[0, 0, 1, 1]);
    let n_items = 2 * n_dims + 3; // 11
    let mut loading = vec![0.0f64; n_items * n_dims];
    for d in 0..n_dims {
        loading[(2 * d) * n_dims + d] = 1.2 + 0.1 * d as f64;
        loading[(2 * d + 1) * n_dims + d] = 0.9;
    }
    let cross = 2 * n_dims;
    loading[cross * n_dims + 0] = 1.0;
    loading[cross * n_dims + 1] = -0.8; // the negative cross-loader
    loading[(cross + 1) * n_dims + 1] = 1.1;
    loading[(cross + 1) * n_dims + 2] = 0.7;
    loading[(cross + 2) * n_dims + 2] = 0.8;
    loading[(cross + 2) * n_dims + 3] = 1.0;
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.12 * i as f64).collect();
    let (n, xi_points, loading_rmse_limit, negative_loading_limit, theta_corr_limit) =
        (800usize, 1600usize, 0.26, -0.15, 0.45);
    let mut rng = Lcg(9001);
    let mut thetas = vec![0.0f64; n * n_dims];
    for v in thetas.iter_mut() {
        *v = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points,
        xi_seed: 12345,
        max_iter: 200,
        tol: 1e-5,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert_eq!(res.n_dims, 4);
    for i in 0..n_items {
        for d in 0..n_dims {
            if pattern[i * n_dims + d] == 0 {
                assert_eq!(res.loading[i * n_dims + d], 0.0, "unloaded exactly zero");
            }
        }
    }
    assert!(
        rmse(&res.loading, &loading) < loading_rmse_limit,
        "loading RMSE {}",
        rmse(&res.loading, &loading)
    );
    // the negative cross-loader recovered negative (sign / compensation guard).
    assert!(
        res.loading[cross * n_dims + 1] < negative_loading_limit,
        "neg cross-loader: {}",
        res.loading[cross * n_dims + 1]
    );
    for d in 0..n_dims {
        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
        let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
        assert!(
            corr(&th, &tt) > theta_corr_limit,
            "theta{d} corr {}",
            corr(&th, &tt)
        );
    }
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "monotone");
    }
    assert!(res.converged, "fit did not converge: {res:?}");
    assert!(res.n_iter < 200);
    assert!(res.final_loglik_change <= 1e-5);
}

/// D=4 correlated WIRING on Halton QMC nodes: the correlated path runs at D>3 and returns a
/// valid positive-definite, unit-diagonal Sigma whose off-diagonals recover the POSITIVE
/// equicorrelation (truth rho=0.4) directionally, with monotone EM. This exercises the Cholesky
/// node map, Sigma M-step, and observed-objective backtracking at D>3. It is deliberately a
/// directional/structural check, NOT a
/// tight per-pair recovery: at an affordable point count the higher-prime Halton axes carry real
/// QMC error in individual Sigma off-diagonals (documented ceiling), so a broken M-step (Sigma=I,
/// non-PD, NaN, or sign-flipped) is what this catches. Tight per-pair Sigma recovery needs a much
/// larger point count (n>=8000 at N>=4000 brings the worst pair within ~0.14 of the realized
/// correlation) and is out of scope for a fast test.
#[test]
fn qmc_recovers_correlated_d4() {
    let n_dims = 4usize;
    // pure anchors: 2 per dim (identification under correlation needs pure indicators).
    let mut pattern: Vec<u8> = Vec::new();
    for d in 0..n_dims {
        for _ in 0..2 {
            let mut row = vec![0u8; n_dims];
            row[d] = 1;
            pattern.extend_from_slice(&row);
        }
    }
    let n_items = 2 * n_dims; // 8, all pure
    let mut loading = vec![0.0f64; n_items * n_dims];
    for d in 0..n_dims {
        loading[(2 * d) * n_dims + d] = 1.3;
        loading[(2 * d + 1) * n_dims + d] = 1.0;
    }
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * i as f64).collect();
    // Build an equicorrelation Sigma (all pairwise correlations = rho) and its Cholesky.
    let rho = 0.4f64;
    let mut sigma = vec![rho; n_dims * n_dims];
    for i in 0..n_dims {
        sigma[i * n_dims + i] = 1.0;
    }
    let lchol = chol_lower(&sigma, n_dims).unwrap();
    let (n, xi_points, min_correlation) = (700usize, 1400usize, 0.1);
    let mut rng = Lcg(20260716);
    let mut thetas = vec![0.0f64; n * n_dims];
    for j in 0..n {
        let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
        for k in 0..n_dims {
            let mut t = 0.0f64;
            for m in 0..=k {
                t += lchol[k * n_dims + m] * z[m];
            }
            thetas[j * n_dims + k] = t;
        }
    }
    // realized sample correlation of the drawn traits (the estimable target under finite N).
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points,
        xi_seed: 777,
        estimate_corr: true,
        max_iter: 200,
        tol: 1e-5,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert_eq!(res.corr.len(), n_dims * n_dims);
    for i in 0..n_dims {
        assert!(
            (res.corr[i * n_dims + i] - 1.0).abs() < 1e-9,
            "unit diagonal"
        );
    }
    // Structural: the returned Sigma is a valid positive-definite correlation matrix, and every
    // off-diagonal is a genuine (non-degenerate) correlation.
    assert!(chol_lower(&res.corr, n_dims).is_some(), "Sigma is PD");
    // Directional: the positive equicorrelation (truth rho=0.4) is recovered as a clearly
    // POSITIVE mean off-diagonal. A broken Sigma M-step returning I gives mean 0; a sign flip
    // gives a negative mean. We do NOT assert closeness to the realized ~0.41: at this
    // affordable point count the higher-prime Halton axes bias the recovered correlations UPWARD
    // by ~0.15 on the mean (the documented QMC ceiling), so tight closeness needs a much larger n.
    let mut rec_sum = 0.0f64;
    let mut cnt = 0.0f64;
    for i in 0..n_dims {
        for j in (i + 1)..n_dims {
            assert!(
                res.corr[i * n_dims + j].abs() < 0.999,
                "off-diagonal not degenerate"
            );
            rec_sum += res.corr[i * n_dims + j];
            cnt += 1.0;
        }
    }
    let rec_mean = rec_sum / cnt;
    assert!(
        rec_mean > min_correlation,
        "recovered mean correlation {rec_mean} not clearly positive"
    );
    assert!(
        rec_mean < 0.85,
        "recovered mean correlation {rec_mean} implausibly high"
    );
    // Observed-objective backtracking rejects a Sigma step that decreases the finite-QMC marginal
    // likelihood, so the converged trace must remain monotone up to roundoff.
    let trace = &res.loglik_trace;
    let max_dec = trace
        .windows(2)
        .map(|w| (w[0] - w[1]).max(0.0))
        .fold(0.0f64, f64::max);
    assert!(
        max_dec < 1e-6,
        "per-step decrease {max_dec} violates EM ascent"
    );
    assert!(*trace.last().unwrap() >= trace[0], "overall EM ascent");
    assert!(res.converged, "fit did not converge: {res:?}");
    assert!(res.n_iter < 200);
    assert!(res.final_loglik_change <= 1e-5);
    println!(
        "[QMC correlated D4] converged={} reason={} n_iter={}/{} final_change={} tolerance={} max_drop={}",
        res.converged,
        res.termination_reason,
        res.n_iter,
        200,
        res.final_loglik_change,
        1e-5,
        max_dec
    );
}

/// Rule-dependent validation: GH stays D<=3, QMC allows D<=6 and bounds xi_points; `q` is
/// unused on the QMC arms (an out-of-set q must NOT reject a Halton fit).
#[test]
fn mirt_qmc_validates() {
    let n = 200usize;
    // GH rejects D=4; Halton accepts it (needs a D=4 pattern with pure anchors).
    let gh4 = TwoPlConfig {
        estimate_corr: false,
        ..TwoPlConfig::default()
    };
    // build a minimal D=4 pattern (one pure anchor per dim) + data of the right shape.
    let n_dims4 = 4usize;
    let mut pat4: Vec<u8> = Vec::new();
    for d in 0..n_dims4 {
        let mut row = vec![0u8; n_dims4];
        row[d] = 1;
        pat4.extend_from_slice(&row);
    }
    let ni4 = n_dims4;
    let y4 = vec![1.0f64; n * ni4];
    let obs4 = vec![true; n * ni4];
    assert!(
        fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &gh4).is_err(),
        "GH D=4 rejected"
    );
    // Halton D=4 with an INVALID GH q (q ignored on the QMC arm) must SUCCEED.
    let ok = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 400,
        xi_seed: 1,
        q: 99,
        max_iter: 3,
        ..TwoPlConfig::default()
    };
    assert!(
        fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &ok).is_ok(),
        "Halton D=4 q=99 ok"
    );
    // Halton D=6 (the UPPER bound MIRT_MAX_DIMS_QMC = HALTON_PRIMES.len()) is ACCEPTED. Pins
    // the boundary so a shrink of the constant to 5 (silently rejecting valid D=6) is caught;
    // D=7 just below is REJECTED (beyond the prime axes).
    let mut pat6 = Vec::new();
    for d in 0..6 {
        let mut r = vec![0u8; 6];
        r[d] = 1;
        pat6.extend_from_slice(&r);
    }
    let y6 = vec![1.0f64; n * 6];
    let obs6 = vec![true; n * 6];
    let d6 = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 200,
        max_iter: 1,
        ..TwoPlConfig::default()
    };
    assert!(
        fit_2pl(&y6, &obs6, &pat6, n, 6, 6, &d6).is_ok(),
        "Halton D=6 accepted"
    );
    let d7 = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 100,
        ..TwoPlConfig::default()
    };
    let mut pat7 = Vec::new();
    for d in 0..7 {
        let mut r = vec![0u8; 7];
        r[d] = 1;
        pat7.extend_from_slice(&r);
    }
    let y7 = vec![1.0f64; n * 7];
    let obs7 = vec![true; n * 7];
    assert!(
        fit_2pl(&y7, &obs7, &pat7, n, 7, 7, &d7).is_err(),
        "Halton D=7 rejected"
    );
    // xi_points bounds: 0 rejected; MAX+1 rejected.
    let zero = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 0,
        ..TwoPlConfig::default()
    };
    assert!(
        fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &zero).is_err(),
        "xi_points=0 rejected"
    );
    let huge = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: MIRT_MAX_NODES + 1,
        ..TwoPlConfig::default()
    };
    assert!(
        fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &huge).is_err(),
        "xi_points>MAX rejected"
    );
    // MonteCarlo D=7 also rejected (its builder has no cap; validate is the sole guard).
    let mc7 = TwoPlConfig {
        xi_rule: XiRuleKind::MonteCarlo,
        xi_points: 100,
        ..TwoPlConfig::default()
    };
    assert!(
        fit_2pl(&y7, &obs7, &pat7, n, 7, 7, &mc7).is_err(),
        "MC D=7 rejected"
    );

    // Individually valid xi_points and item counts must not combine into an unbounded dense
    // E-step table. This input is tiny (one response per item), but without the aggregate guard
    // it attempts four 200_000 x 301 f64 tables before doing any statistical work.
    let table_items = MIRT_MAX_NODE_ITEM_CELLS / MIRT_MAX_NODES + 1;
    let table_y = vec![0.0; table_items];
    let table_obs = vec![true; table_items];
    let table_pattern = vec![1u8; table_items];
    let table_cfg = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: MIRT_MAX_NODES,
        max_iter: 1,
        ..TwoPlConfig::default()
    };
    let err = fit_2pl(
        &table_y,
        &table_obs,
        &table_pattern,
        1,
        table_items,
        1,
        &table_cfg,
    )
    .unwrap_err();
    assert!(err.contains("node * item table"), "{err}");
}

fn small_design() -> (Vec<u8>, Vec<f64>, Vec<f64>, usize) {
    let mut pattern: Vec<u8> = Vec::new();
    for _ in 0..3 {
        pattern.extend_from_slice(&[1, 0]);
    }
    for _ in 0..3 {
        pattern.extend_from_slice(&[0, 1]);
    }
    pattern.extend_from_slice(&[1, 1]);
    let n_items = 7usize;
    let mut loading = vec![0.0f64; n_items * 2];
    for i in 0..3 {
        loading[i * 2] = 1.0 + 0.2 * i as f64;
        loading[(3 + i) * 2 + 1] = 1.0 + 0.2 * i as f64;
    }
    loading[6 * 2] = 0.9;
    loading[6 * 2 + 1] = 0.8;
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.15 * i as f64).collect();
    (pattern, loading, intercept, n_items)
}

#[test]
fn mirt_validates_and_handles_missing() {
    let (pattern, loading, intercept, n_items) = small_design();
    let (n, n_dims) = (400usize, 2usize);
    let mut rng = Lcg(31);
    let mut thetas = vec![0.0f64; n * n_dims];
    for j in 0..n {
        thetas[j * 2] = rng.normal();
        thetas[j * 2 + 1] = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let cfg = TwoPlConfig::default();
    let mut observed = vec![true; n * n_items];
    observed[0] = false;
    observed[n_items + 3] = false;
    assert!(fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).is_ok());
    let obs = vec![true; n * n_items];
    let allones = vec![1u8; n_items * n_dims];
    assert!(fit_2pl(&y, &obs, &allones, n, n_items, n_dims, &cfg).is_err());
    let mut badrow = pattern.clone();
    badrow[0] = 0;
    badrow[1] = 0;
    assert!(fit_2pl(&y, &obs, &badrow, n, n_items, n_dims, &cfg).is_err());
    let mut nopure = pattern.clone();
    for i in 0..3 {
        nopure[i * 2 + 1] = 1; // items 0,1,2 now load both dims -> dim0 has no pure anchor
    }
    assert!(fit_2pl(&y, &obs, &nopure, n, n_items, n_dims, &cfg).is_err());
    assert!(fit_2pl(&y, &obs, &vec![1u8; n_items * 4], n, n_items, 4, &cfg).is_err());
    let badq = TwoPlConfig {
        q: 10,
        ..TwoPlConfig::default()
    };
    assert!(fit_2pl(&y, &obs, &pattern, n, n_items, n_dims, &badq).is_err());
    let mut ybad = y.clone();
    ybad[5] = 2.0;
    assert!(fit_2pl(&ybad, &obs, &pattern, n, n_items, n_dims, &cfg).is_err());
}

#[test]
fn mirt_validation_covers_every_scalar_shape_and_item_boundary() {
    let y = [0.0, 1.0];
    let observed = [true, true];
    let pattern = [1u8];
    let base = TwoPlConfig {
        q: 7,
        max_iter: 1,
        ..TwoPlConfig::default()
    };

    assert_eq!(checked_grid_nodes(1, 7), Ok(7));
    assert!(checked_grid_nodes(MIRT_MAX_NODES, 2).is_err());
    assert!(should_stop_item_newton(false, f64::INFINITY));
    assert!(should_stop_item_newton(true, 0.0));
    assert!(!should_stop_item_newton(true, 1.0));
    let mut mapped = [0.0, 0.0];
    assert!(map_corr_nodes(&[], &[1.0, -1.0], 1, &mut mapped));
    assert_eq!(mapped, [1.0, -1.0]);
    let mut invalid_mapped = [0.0; 3];
    assert!(!map_corr_nodes(
        &[0.9, 0.9, -0.9],
        &[0.0; 3],
        3,
        &mut invalid_mapped
    ));
    assert_eq!(
        marginal_loglik_on_nodes(
            &[1.0],
            &[false],
            &[1.0],
            &[0.0],
            &[vec![0]],
            1,
            1,
            1,
            &[0.0],
            &[0.0]
        ),
        0.0
    );

    assert!(validate(&[], &[], &[], 0, 1, 1, &base).is_err());
    let cfg = TwoPlConfig {
        max_iter: 0,
        ..base
    };
    assert!(validate(&y, &observed, &pattern, 2, 1, 1, &cfg).is_err());
    let cfg = TwoPlConfig {
        max_iter: MIRT_MAX_ITER + 1,
        ..base
    };
    assert!(validate(&y, &observed, &pattern, 2, 1, 1, &cfg).is_err());
    for tol in [0.0, f64::NAN, f64::INFINITY] {
        let cfg = TwoPlConfig { tol, ..base };
        assert!(validate(&y, &observed, &pattern, 2, 1, 1, &cfg).is_err());
    }
    for (ridge_a, ridge_b) in [
        (0.0, base.ridge_b),
        (f64::NAN, base.ridge_b),
        (base.ridge_a, 0.0),
        (base.ridge_a, f64::INFINITY),
    ] {
        let cfg = TwoPlConfig {
            ridge_a,
            ridge_b,
            ..base
        };
        assert!(validate(&y, &observed, &pattern, 2, 1, 1, &cfg).is_err());
    }

    assert!(validate(&y, &observed, &pattern, 2, 1, 0, &base).is_err());
    let bad_q = TwoPlConfig { q: 9, ..base };
    assert!(validate(&y, &observed, &pattern, 2, 1, 1, &bad_q).is_err());
    let qmc = TwoPlConfig {
        xi_rule: XiRuleKind::MonteCarlo,
        xi_points: 1,
        ..base
    };
    assert!(validate(&y, &observed, &pattern, 2, 1, 0, &qmc).is_err());
    let no_points = TwoPlConfig {
        xi_rule: XiRuleKind::Halton,
        xi_points: 0,
        ..base
    };
    assert!(validate(&y, &observed, &pattern, 2, 1, 1, &no_points).is_err());

    assert!(validate(&y[..1], &observed, &pattern, 2, 1, 1, &base).is_err());
    assert!(validate(&y, &observed[..1], &pattern, 2, 1, 1, &base).is_err());
    assert!(validate(&y, &observed, &[], 2, 1, 1, &base).is_err());
    assert!(validate(&[0.0, 2.0], &observed, &pattern, 2, 1, 1, &base).is_err());
    assert!(validate(&y, &observed, &[2], 2, 1, 1, &base).is_err());
    assert!(validate(&y, &observed, &[0], 2, 1, 1, &base).is_err());
    assert!(validate(&y, &[false, false], &pattern, 2, 1, 1, &base).is_err());

    let two_dim_y = [0.0, 1.0, 1.0, 0.0];
    let two_dim_observed = [true; 4];
    assert!(validate(&two_dim_y, &two_dim_observed, &[1, 1, 0, 1], 2, 2, 2, &base,).is_err());

    assert!(validate(&[], &[], &[], 1, usize::MAX, 1, &base).is_err());

    let (gradient, information) = item_grad_hess(
        &[0],
        &[1.0],
        0.0,
        &[0.0],
        &[0.0],
        &[0.0],
        1,
        1,
        base.ridge_a,
        base.ridge_b,
    );
    assert_eq!(gradient, vec![-base.ridge_a, 0.0]);
    assert_eq!(information[0][0], base.ridge_a);

    assert!(corr_line_search(&[0.0], &[f64::NAN], 0.0, &[1.0, 0.0, 0.0, 1.0], 2).is_none());
    let dims = vec![vec![0], vec![1]];
    let mut loading = vec![0.0; 4];
    let mut theta = vec![2.0, 3.0];
    let mut correlation = vec![0.5];
    reflect_mirt_dimensions(&mut loading, &mut theta, &mut correlation, &dims, 1, 2, 2);
    assert_eq!(theta, vec![2.0, 3.0]);
    loading[0] = -1.0;
    loading[3] = 1.0;
    reflect_mirt_dimensions(&mut loading, &mut theta, &mut correlation, &dims, 1, 2, 2);
    assert_eq!(loading[0], 1.0);
    assert_eq!(theta[0], -2.0);
    assert_eq!(correlation, vec![-0.5]);
}

#[test]
fn monte_carlo_fit_executes_the_seeded_node_path() {
    let y = [0.0, 1.0, 1.0, 0.0];
    let observed = [true; 4];
    let pattern = [1u8, 1u8];
    let cfg = TwoPlConfig {
        xi_rule: XiRuleKind::MonteCarlo,
        xi_points: 16,
        xi_seed: 17,
        max_iter: 1,
        newton_iter: 1,
        ..TwoPlConfig::default()
    };
    let fit = fit_2pl(&y, &observed, &pattern, 2, 2, 1, &cfg).unwrap();
    assert_eq!(fit.n_iter, 1);
    assert!(fit.loglik_trace.iter().all(|value| value.is_finite()));
}

/// The final E-step is a genuine evaluated stopping point: meeting tolerance there is
/// convergence even when it follows the last permitted M-step; otherwise exhaustion stays
/// explicit and reports the observed stopping metric.
#[test]
fn mirt_reports_final_stopping_evidence() {
    let pattern = vec![1u8, 0, 0, 1];
    let balanced = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let observed = vec![true; balanced.len()];
    let cfg = TwoPlConfig {
        q: 7,
        max_iter: 1,
        ..TwoPlConfig::default()
    };
    let stable = fit_2pl(&balanced, &observed, &pattern, 4, 2, 2, &cfg).unwrap();
    assert!(stable.converged);
    assert_eq!(stable.termination_reason, "converged");
    assert_eq!(stable.n_iter, cfg.max_iter);
    assert_eq!(stable.loglik_trace.len(), 2);
    assert!(stable.final_loglik_change <= cfg.tol);

    let mut y = vec![0.0f64; 20 * 4];
    for p in 0..20 {
        y[p * 4] = if p % 5 == 0 { 0.0 } else { 1.0 };
        y[p * 4 + 1] = if p % 3 == 0 { 1.0 } else { 0.0 };
        y[p * 4 + 2] = if p % 4 == 0 { 0.0 } else { 1.0 };
        y[p * 4 + 3] = if p % 6 == 0 { 1.0 } else { 0.0 };
    }
    let observed = vec![true; y.len()];
    let pattern4 = vec![1u8, 0, 1, 0, 0, 1, 0, 1];
    let strict = TwoPlConfig {
        q: 7,
        max_iter: 1,
        tol: 1e-12,
        ..TwoPlConfig::default()
    };
    let unfinished = fit_2pl(&y, &observed, &pattern4, 20, 4, 2, &strict).unwrap();
    assert!(!unfinished.converged);
    assert_eq!(unfinished.termination_reason, "max_iter_reached");
    assert_eq!(unfinished.n_iter, strict.max_iter);
    assert_eq!(unfinished.loglik_trace.len(), 2);
    assert!(unfinished.final_loglik_change >= strict.tol);
}

/// Literature-grade Monte-Carlo (>=500 reps): recover the compensatory loadings and traits
/// at D=2 and D=3 under BOTH a normal and a right-skew (per-dim z-standardized, so only the
/// SHAPE is misspecified) trait distribution. Loading RMSE is the primary target; the skew
/// arm uses a looser bound (recovery is genuinely harder under shape misspecification).
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_mirt_recovery_500() {
    let reps = 500usize;
    for &(n_dims, q, n) in [(2usize, 15usize, 3000usize), (3usize, 11usize, 2000usize)].iter() {
        let mut pattern: Vec<u8> = Vec::new();
        for d in 0..n_dims {
            for _ in 0..3 {
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
        let n_items = 3 * n_dims + n_dims;
        let mut loading = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            for k in 0..3 {
                loading[(d * 3 + k) * n_dims + d] = 0.9 + 0.3 * k as f64;
            }
        }
        for d in 0..n_dims {
            let base = 3 * n_dims + d;
            loading[base * n_dims + d] = 1.0;
            loading[base * n_dims + (d + 1) % n_dims] = 0.7;
        }
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.12 * i as f64).collect();

        for &skew in [false, true].iter() {
            let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
            let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
            let mut nconv = 0usize;
            for rep in 0..reps {
                let mut rng = Lcg(0x9E3779B97F4A7C15u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                    .wrapping_add(n_dims as u64 * 0x100000001B3));
                let mut thetas = vec![0.0f64; n * n_dims];
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
                        thetas[j * n_dims + d] = (col[j] - m) / sd;
                    }
                }
                let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                let observed = vec![true; n * n_items];
                let cfg = TwoPlConfig {
                    q,
                    ..TwoPlConfig::default()
                };
                let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    nconv += 1;
                }
                for w in res.loglik_trace.windows(2) {
                    assert!(w[1] >= w[0] - 1e-6, "monotone loglik (rep {rep})");
                }
                for i in 0..n_items {
                    for d in 0..n_dims {
                        let v = res.loading[i * n_dims + d];
                        if pattern[i * n_dims + d] == 0 {
                            assert_eq!(v, 0.0, "unloaded exactly zero");
                        } else {
                            assert!(v.is_finite() && v.abs() <= 10.0, "loading in bound");
                            let e = v - loading[i * n_dims + d];
                            lnum += e * e;
                            lden += 1.0;
                            lbias += e;
                        }
                    }
                }
                for d in 0..n_dims {
                    let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let lrmse = (lnum / lden).sqrt();
            let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
            println!(
                "[mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                 loadRMSE={lrmse:.4} loadBias={lb:.4} thetaCorr={tc:.3}"
            );
            // Thresholds calibrated from a 40-rep pilot (D2/D3 x normal/skew, N=3000/2000).
            assert!(conv > 0.95, "convergence {conv} (D={n_dims} skew={skew})");
            if skew {
                // Shape misspecification: loadings attenuate (bias ~ -0.06..-0.09, expected);
                // recovery is looser but the per-dim trait EAP stays clearly positive.
                assert!(lrmse < 0.20, "skew loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.62, "skew theta corr {tc} (D={n_dims})");
            } else {
                // Correctly-specified N(0,I): recovery is UNBIASED (the correctness signal).
                assert!(lb.abs() < 0.03, "loading bias {lb} (D={n_dims})");
                assert!(lrmse < 0.14, "loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.68, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}

/// Literature-grade Monte-Carlo (>=500 reps) for the HIGH-DIMENSIONAL QMC path (`D > 3`, which
/// the Gauss-Hermite product grid cannot reach): recover the compensatory loadings and traits
/// at D=4 and D=5 on Halton QMC nodes, under a normal AND a per-dim-standardized right-skew
/// trait. The QMC node set is FIXED across the EM run (so EM is monotone) and across reps (a
/// deterministic quadrature); the finite-node QMC bias is what the looser-than-GH thresholds
/// absorb, and averaging over reps is what pins the low-variance recovery the single fast test
/// cannot. Per-rep finiteness + monotone-EM canaries; non-convergence tracked separately.
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_qmc_mirt_recovery_500() {
    let reps = 500usize;
    for &(n_dims, xi_points, n) in [
        (4usize, 4000usize, 2000usize),
        (5usize, 6000usize, 1500usize),
    ]
    .iter()
    {
        // 2 pure anchors per dim (identification) + one cross-loader per dim.
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
        let mut loading = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            loading[(2 * d) * n_dims + d] = 1.2;
            loading[(2 * d + 1) * n_dims + d] = 0.9;
        }
        for d in 0..n_dims {
            let base = 2 * n_dims + d;
            loading[base * n_dims + d] = 1.0;
            // alternate the cross-loader sign so a compensation-sign bug cannot hide.
            loading[base * n_dims + (d + 1) % n_dims] = if d % 2 == 0 { 0.7 } else { -0.7 };
        }
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.1 * i as f64).collect();

        for &skew in [false, true].iter() {
            let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
            let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
            let mut nconv = 0usize;
            for rep in 0..reps {
                let mut rng = Lcg(0x9E3779B97F4A7C15u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                    .wrapping_add(n_dims as u64 * 0x100000001B3));
                let mut thetas = vec![0.0f64; n * n_dims];
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
                        thetas[j * n_dims + d] = (col[j] - m) / sd;
                    }
                }
                let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                let observed = vec![true; n * n_items];
                let cfg = TwoPlConfig {
                    xi_rule: XiRuleKind::Halton,
                    xi_points,
                    xi_seed: 0x2545_F491_4F6C_DD1D,
                    ..TwoPlConfig::default()
                };
                let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    nconv += 1;
                }
                assert!(
                    res.loglik_trace.iter().all(|v| v.is_finite()),
                    "finite loglik (rep {rep})"
                );
                for w in res.loglik_trace.windows(2) {
                    assert!(w[1] >= w[0] - 1e-6, "monotone loglik (rep {rep})");
                }
                for i in 0..n_items {
                    for d in 0..n_dims {
                        let v = res.loading[i * n_dims + d];
                        if pattern[i * n_dims + d] == 0 {
                            assert_eq!(v, 0.0, "unloaded exactly zero");
                        } else {
                            assert!(v.is_finite() && v.abs() <= 10.0, "loading in bound");
                            let e = v - loading[i * n_dims + d];
                            lnum += e * e;
                            lden += 1.0;
                            lbias += e;
                        }
                    }
                }
                assert!(
                    res.theta.iter().all(|v| v.is_finite()),
                    "finite theta (rep {rep})"
                );
                for d in 0..n_dims {
                    let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let lrmse = (lnum / lden).sqrt();
            let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
            println!(
                "[qmc-mirt MC D={n_dims} xi={xi_points} N={n} skew={skew}] reps={reps} \
                 conv={conv:.3} loadRMSE={lrmse:.4} loadBias={lb:.4} thetaCorr={tc:.3}"
            );
            // Looser than the GH MC: QMC carries an O(N^-1 (log N)^D) finite-node bias that
            // grows with D. Calibrated from a 50-rep pilot at D=4/5 x normal/skew (conv=1.000;
            // normal loadRMSE 0.13/0.17, bias ~0.01; skew loadRMSE 0.16/0.21, bias ~-0.07/-0.09;
            // thetaCorr 0.58-0.64) with margin for the 500-rep estimate.
            assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
            if skew {
                assert!(lrmse < 0.26, "skew loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.50, "skew theta corr {tc} (D={n_dims})");
            } else {
                assert!(lb.abs() < 0.06, "loading bias {lb} (D={n_dims})");
                assert!(lrmse < 0.19, "loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.55, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}

// ----- Correlated-Sigma extension (theta ~ MVN(0, Sigma)) -----

/// Draw N x D standard normals correlated through L = chol(Sigma): theta = L z.
fn draw_corr(l: &[f64], n: usize, d: usize, rng: &mut Lcg) -> Vec<f64> {
    let mut th = vec![0.0f64; n * d];
    for j in 0..n {
        let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
        for k in 0..d {
            let mut t = 0.0;
            for i in 0..=k {
                t += l[k * d + i] * z[i];
            }
            th[j * d + k] = t;
        }
    }
    th
}

/// Realized sample correlation off-diagonals (pairs i<j) of an N x D trait matrix.
fn sample_corr(th: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut mean = vec![0.0f64; d];
    for j in 0..n {
        for k in 0..d {
            mean[k] += th[j * d + k];
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut var = vec![0.0f64; d];
    let mut off = Vec::new();
    for i in 0..d {
        for j in 0..n {
            var[i] += (th[j * d + i] - mean[i]).powi(2);
        }
    }
    for i in 0..d {
        for k in i + 1..d {
            let mut cov = 0.0;
            for j in 0..n {
                cov += (th[j * d + i] - mean[i]) * (th[j * d + k] - mean[k]);
            }
            off.push(cov / (var[i] * var[k]).sqrt());
        }
    }
    off
}

/// estimate_corr = false reports Sigma = I exactly and keeps the orthogonal parameter count.
#[test]
fn mirt_estimate_corr_false_is_identity() {
    let (pattern, loading, intercept, n_items) = small_design();
    let (n, n_dims) = (300usize, 2usize);
    let mut rng = Lcg(5);
    let mut thetas = vec![0.0f64; n * n_dims];
    for t in thetas.iter_mut() {
        *t = rng.normal();
    }
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let res = fit_2pl(
        &y,
        &observed,
        &pattern,
        n,
        n_items,
        n_dims,
        &TwoPlConfig::default(),
    )
    .unwrap();
    assert_eq!(res.corr, vec![1.0, 0.0, 0.0, 1.0], "Sigma == I exactly");
    let nfree = pattern.iter().filter(|&&v| v == 1).count();
    assert_eq!(res.n_parameters, nfree + n_items, "no extra corr params");
}

/// flip_corr_dim negates exactly the correlations that involve the flipped dimension.
#[test]
fn mirt_flip_corr_dim_negates_involving_dim() {
    // D=3, off-diagonal order (0,1),(0,2),(1,2).
    let mut r = vec![0.3f64, -0.2, 0.5];
    flip_corr_dim(&mut r, 3, 0); // negate pairs touching dim 0: (0,1),(0,2); (1,2) unchanged
    assert_eq!(r, vec![-0.3, 0.2, 0.5]);
    flip_corr_dim(&mut r, 3, 1); // negate pairs touching dim 1: (0,1),(1,2); (0,2) unchanged
    assert_eq!(r, vec![0.3, 0.2, -0.5]);
}

/// Deterministic FD anchor: the analytic correlation gradient matches central finite
/// differences of Q_prior at a Sigma with NONZERO off-diagonals and a non-diagonal C.
#[test]
fn mirt_sigma_grad_matches_finite_difference() {
    for &(d, ref r0, ref c) in [
        (2usize, vec![0.35f64], vec![1.2f64, 0.5, 0.5, 0.9]),
        (
            3usize,
            vec![0.3f64, -0.15, 0.25],
            vec![1.1f64, 0.4, 0.2, 0.4, 0.95, -0.3, 0.2, -0.3, 1.05],
        ),
    ]
    .iter()
    {
        let sigma = build_corr(r0, d);
        let g = sigma_grad(&sigma, c, d).unwrap();
        let eps = 1e-6;
        for m in 0..r0.len() {
            let mut rp = r0.clone();
            let mut rm = r0.clone();
            rp[m] += eps;
            rm[m] -= eps;
            let qp = sigma_qprior(&build_corr(&rp, d), c, d).unwrap();
            let qm = sigma_qprior(&build_corr(&rm, d), c, d).unwrap();
            let fd = (qp - qm) / (2.0 * eps);
            assert!(
                (g[m] - fd).abs() < 1e-5,
                "D={d} grad[{m}] {} vs fd {fd}",
                g[m]
            );
        }
    }
}

/// Recover a KNOWN correlated Sigma (rho = 0.5) AND loadings at D=2, with the largest-|loading|
/// PURE anchor on dim 0 genuinely NEGATIVE so the reflection FIRES: the reported correlation
/// must then carry the flip-consistent sign (a missing Sigma sign-flip would report +rho).
#[test]
fn mirt_recovers_correlated_d2_with_reflection() {
    let n_dims = 2usize;
    let mut pattern: Vec<u8> = Vec::new();
    for _ in 0..4 {
        pattern.extend_from_slice(&[1, 0]);
    }
    for _ in 0..4 {
        pattern.extend_from_slice(&[0, 1]);
    }
    for _ in 0..2 {
        pattern.extend_from_slice(&[1, 1]);
    }
    let n_items = 10usize;
    let mut loading = vec![0.0f64; n_items * n_dims];
    // dim0 pure anchors: largest |.| is -1.6 (NEGATIVE) -> reflection flips dim 0.
    let a0 = [1.0, 0.8, -1.6, 1.1];
    let a1 = [1.2, 0.9, 1.4, 1.0];
    for i in 0..4 {
        loading[i * 2] = a0[i];
        loading[(4 + i) * 2 + 1] = a1[i];
    }
    loading[8 * 2] = 0.9;
    loading[8 * 2 + 1] = 0.8;
    loading[9 * 2] = 1.1;
    loading[9 * 2 + 1] = 0.7;
    let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.13 * i as f64).collect();
    let rho = 0.5;
    let lchol = chol_lower(&build_corr(&[rho], n_dims), n_dims).unwrap();
    let n = 5000usize;
    let mut rng = Lcg(4242);
    let thetas = draw_corr(&lchol, n, n_dims, &mut rng);
    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
    let observed = vec![true; n * n_items];
    let cfg = TwoPlConfig {
        q: 15,
        estimate_corr: true,
        ..TwoPlConfig::default()
    };
    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
    assert!(res.converged);
    // Sigma is a valid unit-diagonal correlation matrix.
    assert!((res.corr[0] - 1.0).abs() < 1e-12 && (res.corr[3] - 1.0).abs() < 1e-12);
    assert!((res.corr[1] - res.corr[2]).abs() < 1e-12, "symmetric");
    // The reflection fired on dim 0 (its true anchor was negative), so the reported theta_0
    // is negated -> the reported correlation is the flip-consistent -rho. The realized sample
    // correlation is the honest recovery target; after the flip its sign is negated.
    let r_true = sample_corr(&thetas, n, n_dims)[0];
    assert!(
        (res.corr[1] - (-r_true)).abs() < 0.06,
        "corr {} vs -R {}",
        res.corr[1],
        -r_true
    );
    assert!(
        res.corr[1] < -0.3,
        "flip-consistent NEGATIVE correlation, got {}",
        res.corr[1]
    );
    // Loadings recovered against the flip-adjusted truth (dim 0 negated by the reflection).
    let mut expected = loading.clone();
    for i in 0..n_items {
        expected[i * 2] = -expected[i * 2]; // dim 0 flipped
    }
    assert!(
        rmse(&res.loading, &expected) < 0.12,
        "loading RMSE {}",
        rmse(&res.loading, &expected)
    );
    assert!(
        res.loading[2 * 2] > 0.9,
        "flipped anchor now positive: {}",
        res.loading[2 * 2]
    );
    assert!(res.n_parameters == pattern.iter().filter(|&&v| v == 1).count() + n_items + 1);
    for w in res.loglik_trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-6, "EM monotone with the Sigma M-step");
    }
}

/// Literature-grade Monte-Carlo (>=500 reps): recover loadings AND the latent correlation at
/// D=2 (rho=0.5) and D=3 (exchangeable rho=0.4, verified PD) under a normal and a NORTA
/// right-skew marginal (single correlated normal -> monotone per-dim skew, so the copula
/// keeps the sign; corr is scored against the REALIZED sample correlation R_rep, not nominal).
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn mc_corr_mirt_recovery_500() {
    let reps = 500usize;
    for &(n_dims, q, n, ref true_off) in [
        (2usize, 15usize, 3000usize, vec![0.5f64]),
        (3usize, 11usize, 2000usize, vec![0.4f64, 0.4, 0.4]), // exchangeable, eig 1.8,0.6,0.6
    ]
    .iter()
    {
        let sigma_true = build_corr(true_off, n_dims);
        let lchol = chol_lower(&sigma_true, n_dims).expect("true Sigma must be PD");
        // pattern: 3 pure anchors per dim + one cross-loader per consecutive pair.
        let mut pattern: Vec<u8> = Vec::new();
        for dd in 0..n_dims {
            for _ in 0..3 {
                let mut r = vec![0u8; n_dims];
                r[dd] = 1;
                pattern.extend_from_slice(&r);
            }
        }
        for dd in 0..n_dims {
            let mut r = vec![0u8; n_dims];
            r[dd] = 1;
            r[(dd + 1) % n_dims] = 1;
            pattern.extend_from_slice(&r);
        }
        let n_items = 3 * n_dims + n_dims;
        let mut loading = vec![0.0f64; n_items * n_dims];
        for dd in 0..n_dims {
            for k in 0..3 {
                loading[(dd * 3 + k) * n_dims + dd] = 0.9 + 0.3 * k as f64; // positive anchors
            }
        }
        for dd in 0..n_dims {
            let base = 3 * n_dims + dd;
            loading[base * n_dims + dd] = 1.0;
            loading[base * n_dims + (dd + 1) % n_dims] = 0.7;
        }
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.12 * i as f64).collect();
        let n_off = n_dims * (n_dims - 1) / 2;

        for &skew in [false, true].iter() {
            let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
            let (mut cnum, mut cbias) = (0.0f64, 0.0f64);
            let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
            let (mut nconv, mut interior) = (0usize, 0usize);
            for rep in 0..reps {
                let mut rng = Lcg(0xD1B54A32D192ED03u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15)
                    .wrapping_add(n_dims as u64 * 0x100000001B3));
                // NORTA: correlated normals z = L u; per-dim monotone right-skew then
                // re-standardize (keeps the sign of the correlation, attenuated).
                let mut thetas = draw_corr(&lchol, n, n_dims, &mut rng);
                if skew {
                    for k in 0..n_dims {
                        for j in 0..n {
                            let z = thetas[j * n_dims + k];
                            thetas[j * n_dims + k] = (0.5 * z).exp(); // monotone lognormal skew
                        }
                        let col: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + k]).collect();
                        let m = col.iter().sum::<f64>() / n as f64;
                        let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                        let sd = v.sqrt();
                        for j in 0..n {
                            thetas[j * n_dims + k] = (thetas[j * n_dims + k] - m) / sd;
                        }
                    }
                }
                let r_rep = sample_corr(&thetas, n, n_dims); // honest recovery target
                let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                let observed = vec![true; n * n_items];
                let cfg = TwoPlConfig {
                    q,
                    estimate_corr: true,
                    ..TwoPlConfig::default()
                };
                let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    nconv += 1;
                }
                for w in res.loglik_trace.windows(2) {
                    assert!(w[1] >= w[0] - 1e-6, "EM monotone (rep {rep})");
                }
                // Sigma invariants: unit diagonal, symmetric, PD, |off|<1, all finite.
                for k in 0..n_dims {
                    assert!(
                        (res.corr[k * n_dims + k] - 1.0).abs() < 1e-9,
                        "unit diagonal"
                    );
                }
                assert!(chol_lower(&res.corr, n_dims).is_some(), "Sigma PD");
                let mut pinned = false;
                let off_est: Vec<f64> = {
                    let mut o = Vec::new();
                    for i in 0..n_dims {
                        for j in i + 1..n_dims {
                            let v = res.corr[i * n_dims + j];
                            assert!(v.is_finite() && v.abs() < 1.0, "corr in (-1,1)");
                            assert!((v - res.corr[j * n_dims + i]).abs() < 1e-12, "symmetric");
                            if v.abs() > 0.99 {
                                pinned = true;
                            }
                            o.push(v);
                        }
                    }
                    o
                };
                if !pinned {
                    interior += 1;
                }
                // Loadings: pure anchors positive -> reflection never fires -> no flip; score
                // vs truth directly.
                for i in 0..n_items {
                    for dd in 0..n_dims {
                        let v = res.loading[i * n_dims + dd];
                        if pattern[i * n_dims + dd] == 0 {
                            assert_eq!(v, 0.0);
                        } else {
                            assert!(v.is_finite() && v.abs() <= 10.0);
                            let e = v - loading[i * n_dims + dd];
                            lnum += e * e;
                            lden += 1.0;
                            lbias += e;
                        }
                    }
                }
                for m in 0..n_off {
                    let e = off_est[m] - r_rep[m]; // vs realized correlation
                    cnum += e * e;
                    cbias += e;
                    // correlation sign matches the (positive) truth
                    assert!(off_est[m] > 0.0, "corr sign matches truth (rep {rep})");
                }
                for dd in 0..n_dims {
                    let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + dd]).collect();
                    let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + dd]).collect();
                    csum += corr(&th, &tt);
                    ccnt += 1.0;
                }
            }
            let lrmse = (lnum / lden).sqrt();
            let crmse = (cnum / (reps * n_off) as f64).sqrt();
            let (lb, cb) = (lbias / lden, cbias / (reps * n_off) as f64);
            let (tc, conv) = (csum / ccnt, nconv as f64 / reps as f64);
            let int_frac = interior as f64 / reps as f64;
            println!(
                "[corr-mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                 loadRMSE={lrmse:.4} loadBias={lb:.4} corrRMSE={crmse:.4} corrBias={cb:.4} \
                 thetaCorr={tc:.3} interior={int_frac:.3}"
            );
            assert!(conv > 0.95, "convergence {conv} (D={n_dims} skew={skew})");
            assert!(
                int_frac > 0.95,
                "Sigma interior fraction {int_frac} (D={n_dims})"
            );
            assert!(
                crmse < 0.06,
                "correlation RMSE vs R_rep {crmse} (D={n_dims} skew={skew})"
            );
            if skew {
                assert!(lrmse < 0.20, "skew loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.62, "skew theta corr {tc} (D={n_dims})");
            } else {
                assert!(lb.abs() < 0.03, "loading bias {lb} (D={n_dims})");
                assert!(lrmse < 0.14, "loading RMSE {lrmse} (D={n_dims})");
                assert!(tc > 0.68, "theta corr {tc} (D={n_dims})");
            }
        }
    }
}
