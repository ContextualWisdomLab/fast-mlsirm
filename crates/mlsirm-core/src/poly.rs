//! Polytomous item-response cells and their expected-complete-data gradients,
//! the Rust compute path for the polytomous LSIRM extension
//! (see `docs/papers/gpcm-nominal-design-spec.md` and its literature
//! resolution). All numerical work lives here; the NumPy functions in
//! `fast_mlsirm.estimators.marginal` are parity references only.
//!
//! Two response families over a shared linear predictor `base = a*theta +
//! interaction(x)`:
//!
//! - **GRM** (Samejima 1968, cumulative logit) — the identification-clean
//!   default for the LSIRM family: the single latent-space interaction enters
//!   every cumulative logit as one shared shift inside `base`, so nothing
//!   cancels and no category scaling is forced.
//! - **GPCM** (Muraki 1992, adjacent-category softmax) — an option for
//!   partial-credit scoring; the category-constant term cancels in the softmax,
//!   so the space term enters category-score-scaled (a documented consequence).

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// GRM category log-probabilities for one node. `thresholds` holds the `K-1`
/// cumulative boundary intercepts `beta_k` (ordered *decreasing* for a valid
/// distribution); `base` is the shared person-item linear predictor. Returns
/// `log P(Y = k)` for `k = 0..K-1`. `P(Y>=k) = sigmoid(base + beta_k)`.
pub fn grm_logprobs(base: f64, thresholds: &[f64]) -> Vec<f64> {
    let kb = thresholds.len(); // number of boundaries = K-1
    let mut out = vec![0.0_f64; kb + 1];
    if kb == 0 {
        out[0] = 0.0;
        return out;
    }
    // A[j] = log sigmoid(base + beta_j) = log P(Y >= j+1)
    let a: Vec<f64> = thresholds.iter().map(|&b| log_sigmoid(base + b)).collect();
    // category 0: 1 - sigmoid(base + beta_0) = sigmoid(-(base + beta_0))
    out[0] = log_sigmoid(-(base + thresholds[0]));
    // middle categories 1..K-2: P = sigmoid(base+beta_{k-1}) - sigmoid(base+beta_k)
    for k in 1..kb {
        // log(e^{A[k-1]} - e^{A[k]}) = A[k-1] + log1p(-e^{A[k]-A[k-1]}), A[k-1] >= A[k]
        out[k] = a[k - 1] + (-((a[k] - a[k - 1]).exp())).ln_1p();
    }
    // top category K-1: sigmoid(base + beta_{K-2})
    out[kb] = a[kb - 1];
    out
}

/// Gradient of the expected complete-data log-likelihood `sum_k r_k log P(Y=k)`
/// at one node for the GRM cell. Returns `(g_base, g_thresholds)` where
/// `g_thresholds[j]` is the derivative wrt boundary intercept `beta_j`.
pub fn grm_node_gradient(base: f64, thresholds: &[f64], counts: &[f64]) -> (f64, Vec<f64>) {
    let kb = thresholds.len();
    let mut g_t = vec![0.0_f64; kb];
    let mut g_base = 0.0_f64;
    if kb == 0 {
        return (0.0, g_t);
    }
    let p: Vec<f64> = grm_logprobs(base, thresholds).iter().map(|&l| l.exp()).collect();
    // s[j] = sigmoid(base + beta_j) = P(Y >= j+1); v[j] = s[j](1-s[j])
    for j in 0..kb {
        let s = 1.0 / (1.0 + (-(base + thresholds[j])).exp());
        let v = s * (1.0 - s);
        // d q / d s_j = r_{j+1}/P_{j+1} - r_j/P_j  (boundary j sits between cats j and j+1)
        let dqds = counts[j + 1] / p[j + 1] - counts[j] / p[j];
        g_t[j] = v * dqds;
        g_base += v * dqds;
    }
    (g_base, g_t)
}

/// GPCM/nominal unified softmax cell. `scores[0] = intercepts[0] = 0` (baseline
/// category pinned). `psi_k = scores[k]*base + intercepts[k]`; returns the
/// stable `log softmax_k(psi)` for `k = 0..K-1`. Nests binary 2PL at `K=2`,
/// `scores=[0,1]`, `intercepts=[0,b]` (then `logP_1 = log_sigmoid(base+b)`).
pub fn gpcm_logprobs(base: f64, scores: &[f64], intercepts: &[f64]) -> Vec<f64> {
    let k = scores.len();
    let mut psi = vec![0.0_f64; k];
    let mut m = f64::NEG_INFINITY;
    for c in 0..k {
        psi[c] = scores[c] * base + intercepts[c];
        if psi[c] > m {
            m = psi[c];
        }
    }
    let mut sum = 0.0_f64;
    for c in 0..k {
        sum += (psi[c] - m).exp();
    }
    let log_z = m + sum.ln();
    psi.iter().map(|&p| p - log_z).collect()
}

/// Gradient of `sum_k r_k log P(Y=k)` at one node for the GPCM/nominal cell.
/// Returns `(g_intercepts, g_base, g_scores)` for the free coordinates
/// (`k = 1..K-1`); `g_intercepts[m-1] = resid_m`, `g_base = sum_k s_k*resid_k`,
/// `g_scores[m-1] = resid_m * base`, with `resid_k = r_k - n*P_k`.
pub fn gpcm_node_gradient(
    base: f64,
    scores: &[f64],
    intercepts: &[f64],
    counts: &[f64],
) -> (Vec<f64>, f64, Vec<f64>) {
    let k = scores.len();
    let p: Vec<f64> = gpcm_logprobs(base, scores, intercepts).iter().map(|&l| l.exp()).collect();
    let n: f64 = counts.iter().sum();
    let resid: Vec<f64> = (0..k).map(|c| counts[c] - n * p[c]).collect();
    let g_intercepts: Vec<f64> = resid[1..].to_vec();
    let g_base: f64 = (0..k).map(|c| scores[c] * resid[c]).sum();
    let g_scores: Vec<f64> = resid[1..].iter().map(|&r| r * base).collect();
    (g_intercepts, g_base, g_scores)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logsumexp0(v: &[f64]) -> f64 {
        let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
    }

    #[test]
    fn grm_logprobs_normalize_and_binary_parity() {
        // K=2, one threshold: P(Y=1)=sigmoid(base+beta), P(Y=0)=sigmoid(-(base+beta))
        let base = 0.4;
        let beta = -0.3;
        let lp = grm_logprobs(base, &[beta]);
        let z = logsumexp0(&lp);
        assert!(z.abs() < 1e-12, "not normalized: {z}");
        assert!((lp[1] - log_sigmoid(base + beta)).abs() < 1e-12);
        assert!((lp[0] - log_sigmoid(-(base + beta))).abs() < 1e-12);
        // K=4 normalization
        let lp4 = grm_logprobs(0.2, &[1.0, 0.0, -1.2]);
        assert!(logsumexp0(&lp4).abs() < 1e-10);
        assert!(lp4.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn grm_gradient_matches_finite_difference() {
        let base = 0.3;
        let thr = vec![1.1, 0.1, -0.9]; // decreasing => valid
        let counts = vec![4.0, 6.0, 3.0, 5.0];
        let q = |b: f64, t: &[f64]| -> f64 {
            grm_logprobs(b, t).iter().zip(&counts).map(|(l, r)| r * l).sum()
        };
        let (g_base, g_t) = grm_node_gradient(base, &thr, &counts);
        let h = 1e-6;
        assert!(((q(base + h, &thr) - q(base - h, &thr)) / (2.0 * h) - g_base).abs() < 1e-5);
        for j in 0..thr.len() {
            let mut tp = thr.clone();
            let mut tm = thr.clone();
            tp[j] += h;
            tm[j] -= h;
            let fd = (q(base, &tp) - q(base, &tm)) / (2.0 * h);
            assert!((fd - g_t[j]).abs() < 1e-5, "grm g_t[{j}]: {} vs {}", fd, g_t[j]);
        }
    }

    #[test]
    fn gpcm_logprobs_binary_parity_and_monotone() {
        let base = 0.5;
        let b = 0.2;
        let lp = gpcm_logprobs(base, &[0.0, 1.0], &[0.0, b]);
        assert!(logsumexp0(&lp).abs() < 1e-12);
        assert!((lp[1] - log_sigmoid(base + b)).abs() < 1e-12);
        // higher base -> more mass on top category (scores 0,1,2)
        let lo = gpcm_logprobs(-2.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]);
        let hi = gpcm_logprobs(2.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]);
        assert!(hi[2].exp() > lo[2].exp());
    }

    #[test]
    fn gpcm_gradient_matches_finite_difference() {
        let scores = vec![0.0, 1.0, 2.0, 3.0];
        let intercepts = vec![0.0, 0.2, -0.1, 0.3];
        let counts = vec![3.0, 5.0, 2.0, 4.0];
        let base = 0.4;
        let q = |b: f64, ic: &[f64], sc: &[f64]| -> f64 {
            gpcm_logprobs(b, sc, ic).iter().zip(&counts).map(|(l, r)| r * l).sum()
        };
        let (g_ic, g_base, g_sc) = gpcm_node_gradient(base, &scores, &intercepts, &counts);
        let h = 1e-6;
        assert!(((q(base + h, &intercepts, &scores) - q(base - h, &intercepts, &scores)) / (2.0 * h)
            - g_base)
            .abs()
            < 1e-5);
        for m in 1..scores.len() {
            let mut ip = intercepts.clone();
            let mut im = intercepts.clone();
            ip[m] += h;
            im[m] -= h;
            let fd = (q(base, &ip, &scores) - q(base, &im, &scores)) / (2.0 * h);
            assert!((fd - g_ic[m - 1]).abs() < 1e-5);
            let mut sp = scores.clone();
            let mut sm = scores.clone();
            sp[m] += h;
            sm[m] -= h;
            let fds = (q(base, &intercepts, &sp) - q(base, &intercepts, &sm)) / (2.0 * h);
            assert!((fds - g_sc[m - 1]).abs() < 1e-5);
        }
    }
}
