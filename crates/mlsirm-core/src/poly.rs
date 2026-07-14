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

/// Polytomous response family for the unidimensional fitter.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PolyModel {
    /// Cumulative-logit Graded Response Model (Samejima) — the LSIRM-family default.
    Grm,
    /// Adjacent-category softmax Generalized Partial Credit Model (Muraki).
    Gpcm,
}

/// Result of [`fit_poly_unidim`]. `slope[i]` is item `i`'s discrimination `a_i`;
/// `cat_params[i]` holds the `K-1` free category parameters (GPCM additive
/// intercepts, or GRM cumulative thresholds).
pub struct PolyFit {
    pub slope: Vec<f64>,
    pub cat_params: Vec<Vec<f64>>,
    pub loglik: f64,
    pub n_iter: usize,
}

/// Solve `H x = g` for small dense `H` (K x K) by Gauss elimination with partial
/// pivoting. Returns `g` unchanged if singular (degenerate M-step step).
fn solve_small(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Vec<f64> {
    let n = g.len();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if h[r][col].abs() > h[piv][col].abs() {
                piv = r;
            }
        }
        if h[piv][col].abs() < 1e-12 {
            return g; // singular: fall back to gradient step
        }
        h.swap(col, piv);
        g.swap(col, piv);
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = h[r][col] / h[col][col];
            for c in col..n {
                h[r][c] -= f * h[col][c];
            }
            g[r] -= f * g[col];
        }
    }
    (0..n).map(|i| g[i] / h[i][i]).collect()
}

/// Negative expected complete-data log-lik and its gradient for one item over
/// the quadrature nodes. `params = [log_a, cat_1..cat_{K-1}]`; `counts[node]` is
/// the length-`K` expected category-count vector at that node.
fn item_neg_ll_grad(
    params: &[f64],
    nodes: &[f64],
    counts: &[Vec<f64>],
    model: PolyModel,
) -> (f64, Vec<f64>) {
    let k = counts[0].len();
    let a = params[0].exp();
    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; params.len()];
    for (nd, &theta) in nodes.iter().enumerate() {
        let base = a * theta;
        match model {
            PolyModel::Gpcm => {
                let mut intercepts = vec![0.0_f64; k];
                intercepts[1..].copy_from_slice(&params[1..]);
                let lp = gpcm_logprobs(base, &scores, &intercepts);
                ll += counts[nd].iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
                let (g_ic, g_base, _g_sc) = gpcm_node_gradient(base, &scores, &intercepts, &counts[nd]);
                for m in 0..k - 1 {
                    grad[1 + m] += g_ic[m];
                }
                grad[0] += g_base * base; // d base / d log_a = a*theta = base
            }
            PolyModel::Grm => {
                let thr = &params[1..];
                let lp = grm_logprobs(base, thr);
                ll += counts[nd].iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
                let (g_base, g_t) = grm_node_gradient(base, thr, &counts[nd]);
                for m in 0..k - 1 {
                    grad[1 + m] += g_t[m];
                }
                grad[0] += g_base * base;
            }
        }
    }
    (-ll, grad.iter().map(|v| -v).collect())
}

/// Newton M-step for one item: a few steps with a finite-difference Hessian on
/// the analytic gradient (the parameter count `K` is small).
fn m_step_item(
    mut params: Vec<f64>,
    nodes: &[f64],
    counts: &[Vec<f64>],
    model: PolyModel,
    n_newton: usize,
) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (_f, g) = item_neg_ll_grad(&params, nodes, counts, model);
        let h = 1e-5;
        let mut hess = vec![vec![0.0_f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = item_neg_ll_grad(&pj, nodes, counts, model);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        // symmetrize + ridge for a well-posed solve
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
            }
            hess[r][r] += 1e-8;
        }
        let step = solve_small(hess, g);
        let mut max_step = 0.0_f64;
        for j in 0..np {
            params[j] -= step[j];
            max_step = max_step.max(step[j].abs());
        }
        if max_step < 1e-9 {
            break;
        }
    }
    params
}

/// Unidimensional polytomous marginal MLE via Bock-Aitkin EM (no latent-space
/// interaction) — the Rust compute path validating the [`PolyModel`] cells in a
/// full EM loop. `y` is `n_persons * n_items`, row-major, categories `0..n_cat-1`
/// (complete data). `theta ~ N(0,1)` on the `q_theta`-node Gauss-Hermite grid.
#[allow(clippy::too_many_arguments)]
pub fn fit_poly_unidim(
    y: &[usize],
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<PolyFit, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();

    // init: log_a = 0; category params from base rates (GPCM) / cumulative rates (GRM)
    let mut params = vec![vec![0.0_f64; n_cat]; n_items];
    for i in 0..n_items {
        let mut freq = vec![1e-3_f64; n_cat];
        for p in 0..n_persons {
            freq[y[p * n_items + i]] += 1.0;
        }
        let tot: f64 = freq.iter().sum();
        for f in freq.iter_mut() {
            *f /= tot;
        }
        match model {
            PolyModel::Gpcm => {
                for k in 1..n_cat {
                    params[i][k] = (freq[k] / freq[0]).ln();
                }
            }
            PolyModel::Grm => {
                // beta_k = logit(P(Y >= k)); cumulative from the top, ordered decreasing
                let mut cum = 0.0_f64;
                for k in (1..n_cat).rev() {
                    cum += freq[k];
                    let c = cum.clamp(1e-4, 1.0 - 1e-4);
                    params[i][k] = (c / (1.0 - c)).ln();
                }
            }
        }
    }

    let mut prev_ll = f64::NEG_INFINITY;
    let mut ll = f64::NEG_INFINITY;
    let mut it = 0;
    while it < max_iter {
        // per-item cell log-probs at each node: item_lp[i][node*n_cat + k]
        let mut item_lp = vec![vec![0.0_f64; qn * n_cat]; n_items];
        for i in 0..n_items {
            let a = params[i][0].exp();
            for (nd, &theta) in nodes.iter().enumerate() {
                let base = a * theta;
                let lp = match model {
                    PolyModel::Gpcm => {
                        let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                        let mut intercepts = vec![0.0_f64; n_cat];
                        intercepts[1..].copy_from_slice(&params[i][1..]);
                        gpcm_logprobs(base, &scores, &intercepts)
                    }
                    PolyModel::Grm => grm_logprobs(base, &params[i][1..]),
                };
                item_lp[i][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
        }
        // E-step: posteriors + expected counts r[i][node][k]
        let mut counts = vec![vec![vec![0.0_f64; n_cat]; qn]; n_items];
        ll = 0.0;
        let mut log_node = vec![0.0_f64; qn];
        for p in 0..n_persons {
            for nd in 0..qn {
                log_node[nd] = log_w[nd];
            }
            for i in 0..n_items {
                let yc = y[p * n_items + i];
                for nd in 0..qn {
                    log_node[nd] += item_lp[i][nd * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0_f64;
            for nd in 0..qn {
                denom += (log_node[nd] - mx).exp();
            }
            ll += mx + denom.ln();
            for i in 0..n_items {
                let yc = y[p * n_items + i];
                for nd in 0..qn {
                    let post = (log_node[nd] - mx).exp() / denom;
                    counts[i][nd][yc] += post;
                }
            }
        }
        // M-step per item
        for i in 0..n_items {
            params[i] = m_step_item(params[i].clone(), nodes, &counts[i], model, 10);
        }
        it += 1;
        if (ll - prev_ll).abs() < tol * (1.0 + prev_ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    let slope: Vec<f64> = (0..n_items).map(|i| params[i][0].exp()).collect();
    let cat_params: Vec<Vec<f64>> = params.iter().map(|p| p[1..].to_vec()).collect();
    Ok(PolyFit { slope, cat_params, loglik: ll, n_iter: it })
}

/// Fisher item information `I(theta) = sum_k (dP_k/dtheta)^2 / P_k` for one
/// polytomous item at trait value `theta`. GPCM reduces to `a^2 * Var_P(scores)`;
/// GRM to `a^2 * sum_k (v_k - v_{k+1})^2 / P_k` with `v_j = s_j(1-s_j)`,
/// `s_j = P(Y>=j)`. `cat_params` is this item's `K-1` category parameters.
pub fn poly_item_information(
    theta: f64,
    slope: f64,
    cat_params: &[f64],
    model: PolyModel,
) -> f64 {
    let a = slope;
    let base = a * theta;
    match model {
        PolyModel::Gpcm => {
            let k = cat_params.len() + 1;
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut intercepts = vec![0.0_f64; k];
            intercepts[1..].copy_from_slice(cat_params);
            let p: Vec<f64> =
                gpcm_logprobs(base, &scores, &intercepts).iter().map(|l| l.exp()).collect();
            let ebar: f64 = scores.iter().zip(&p).map(|(s, pp)| s * pp).sum();
            let var: f64 = scores.iter().zip(&p).map(|(s, pp)| pp * (s - ebar).powi(2)).sum();
            a * a * var
        }
        PolyModel::Grm => {
            let kk = cat_params.len() + 1; // K
            let p: Vec<f64> = grm_logprobs(base, cat_params).iter().map(|l| l.exp()).collect();
            let mut v = vec![0.0_f64; kk + 1]; // v[0]=v[K]=0
            for (j, item) in v.iter_mut().enumerate().take(kk).skip(1) {
                let s = 1.0 / (1.0 + (-(base + cat_params[j - 1])).exp());
                *item = s * (1.0 - s);
            }
            let mut info = 0.0_f64;
            for k in 0..kk {
                let d = v[k] - v[k + 1];
                info += d * d / p[k].max(1e-300);
            }
            a * a * info
        }
    }
}

/// Item information curves over a trait grid: returns a flattened
/// `n_theta * n_items` vector of `I_i(theta)` (row-major by theta). Test
/// information is the per-theta row sum.
#[allow(clippy::too_many_arguments)]
pub fn poly_information_curves(
    theta: &[f64],
    slope: &[f64],
    cat_params: &[f64],
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
) -> Result<Vec<f64>, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if slope.len() != n_items || cat_params.len() != n_items * (n_cat - 1) {
        return Err("slope/cat_params sizes inconsistent with n_items/n_cat".into());
    }
    let mut out = vec![0.0_f64; theta.len() * n_items];
    for (t, &th) in theta.iter().enumerate() {
        for i in 0..n_items {
            let cp = &cat_params[i * (n_cat - 1)..(i + 1) * (n_cat - 1)];
            out[t * n_items + i] = poly_item_information(th, slope[i], cp, model);
        }
    }
    Ok(out)
}

/// EAP trait scores from polytomous responses given fitted item parameters
/// (the Rust scoring companion to [`fit_poly_unidim`]). `slope[i]` is `a_i`;
/// `cat_params` is flattened `n_items * (n_cat-1)` (GPCM intercepts or GRM
/// thresholds). Returns `(theta_eap, theta_sd)` per person over a `theta~N(0,1)`
/// Gauss-Hermite grid.
#[allow(clippy::too_many_arguments)]
pub fn score_poly_eap(
    y: &[usize],
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: PolyModel,
    q_theta: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    if slope.len() != n_items || cat_params.len() != n_items * (n_cat - 1) {
        return Err("slope/cat_params sizes inconsistent with n_items/n_cat".into());
    }
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();
    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();

    // per-item cell log-probs at each node: item_lp[i][node*n_cat + k]
    let mut item_lp = vec![vec![0.0_f64; qn * n_cat]; n_items];
    for i in 0..n_items {
        let a = slope[i];
        let cp = &cat_params[i * (n_cat - 1)..(i + 1) * (n_cat - 1)];
        for (nd, &theta) in nodes.iter().enumerate() {
            let base = a * theta;
            let lp = match model {
                PolyModel::Gpcm => {
                    let mut intercepts = vec![0.0_f64; n_cat];
                    intercepts[1..].copy_from_slice(cp);
                    gpcm_logprobs(base, &scores, &intercepts)
                }
                PolyModel::Grm => grm_logprobs(base, cp),
            };
            item_lp[i][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
        }
    }

    let mut theta_eap = vec![0.0_f64; n_persons];
    let mut theta_sd = vec![0.0_f64; n_persons];
    let mut log_node = vec![0.0_f64; qn];
    for p in 0..n_persons {
        for nd in 0..qn {
            log_node[nd] = log_w[nd];
        }
        for i in 0..n_items {
            let yc = y[p * n_items + i];
            for nd in 0..qn {
                log_node[nd] += item_lp[i][nd * n_cat + yc];
            }
        }
        let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0_f64;
        for nd in 0..qn {
            denom += (log_node[nd] - mx).exp();
        }
        let (mut m1, mut m2) = (0.0_f64, 0.0_f64);
        for nd in 0..qn {
            let post = (log_node[nd] - mx).exp() / denom;
            m1 += post * nodes[nd];
            m2 += post * nodes[nd] * nodes[nd];
        }
        theta_eap[p] = m1;
        theta_sd[p] = (m2 - m1 * m1).max(0.0).sqrt();
    }
    Ok((theta_eap, theta_sd))
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
    fn fit_poly_unidim_recovers_gpcm() {
        let (n_persons, n_items, k) = (4000usize, 6usize, 3usize);
        let mut st = 20260714u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.16 * i as f64).collect();
        let c_true: Vec<Vec<f64>> = (0..n_items)
            .map(|i| vec![0.0, 0.3 - 0.1 * i as f64, -0.2 + 0.15 * i as f64])
            .collect();
        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let mut y = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let lp = gpcm_logprobs(a_true[i] * theta, &scores, &c_true[i]);
                let uu = u();
                let mut cum = 0.0_f64;
                let mut cat = k - 1;
                for (c, l) in lp.iter().enumerate() {
                    cum += l.exp();
                    if uu < cum {
                        cat = c;
                        break;
                    }
                }
                y[p * n_items + i] = cat;
            }
        }
        let fit = fit_poly_unidim(&y, n_persons, n_items, k, PolyModel::Gpcm, 21, 80, 1e-6).unwrap();
        assert!(fit.loglik.is_finite());
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mh) = (mean(&a_true), mean(&fit.slope));
        let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
        for i in 0..n_items {
            num += (a_true[i] - ma) * (fit.slope[i] - mh);
            da += (a_true[i] - ma).powi(2);
            dh += (fit.slope[i] - mh).powi(2);
        }
        let corr = num / (da.sqrt() * dh.sqrt());
        assert!(corr > 0.9, "slope corr {corr}; hat={:?}", fit.slope);
    }

    #[test]
    fn poly_item_information_matches_finite_difference() {
        // I(theta) = sum_k (dP_k/dtheta)^2 / P_k, checked against a central FD of the cell.
        let h = 1e-6;
        let cases: [(PolyModel, &[f64]); 2] =
            [(PolyModel::Gpcm, &[0.2, -0.3]), (PolyModel::Grm, &[1.1, -0.9])];
        for (model, cat) in cases.iter().copied() {
            let (a, theta) = (1.3_f64, 0.4_f64);
            let cell = |t: f64| -> Vec<f64> {
                let base = a * t;
                match model {
                    PolyModel::Gpcm => {
                        let k = cat.len() + 1;
                        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                        let mut ic = vec![0.0; k];
                        ic[1..].copy_from_slice(cat);
                        gpcm_logprobs(base, &scores, &ic).iter().map(|l| l.exp()).collect()
                    }
                    PolyModel::Grm => grm_logprobs(base, cat).iter().map(|l| l.exp()).collect(),
                }
            };
            let (pp, pm, p0) = (cell(theta + h), cell(theta - h), cell(theta));
            let mut fd_info = 0.0_f64;
            for k in 0..p0.len() {
                let dp = (pp[k] - pm[k]) / (2.0 * h);
                fd_info += dp * dp / p0[k];
            }
            let ana = poly_item_information(theta, a, cat, model);
            assert!((ana - fd_info).abs() < 1e-4, "{model:?}: analytic {ana} vs fd {fd_info}");
        }
    }

    #[test]
    fn score_poly_eap_recovers_true_theta() {
        let (n_persons, n_items, k) = (3000usize, 8usize, 3usize);
        let mut st = 424242u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.1 * i as f64).collect();
        let c_true: Vec<Vec<f64>> =
            (0..n_items).map(|i| vec![0.0, 0.2 - 0.05 * i as f64, -0.3 + 0.08 * i as f64]).collect();
        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let mut theta_true = vec![0.0_f64; n_persons];
        let mut y = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            theta_true[p] = theta;
            for i in 0..n_items {
                let lp = gpcm_logprobs(a_true[i] * theta, &scores, &c_true[i]);
                let uu = u();
                let mut cum = 0.0_f64;
                let mut cat = k - 1;
                for (c, l) in lp.iter().enumerate() {
                    cum += l.exp();
                    if uu < cum {
                        cat = c;
                        break;
                    }
                }
                y[p * n_items + i] = cat;
            }
        }
        // score with the TRUE item params (isolates the scorer from fit error)
        let cat_flat: Vec<f64> = c_true.iter().flat_map(|c| c[1..].iter().copied()).collect();
        let (eap, sd) =
            score_poly_eap(&y, n_persons, n_items, k, &a_true, &cat_flat, PolyModel::Gpcm, 41)
                .unwrap();
        assert!(sd.iter().all(|s| s.is_finite() && *s > 0.0));
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (mt, me) = (mean(&theta_true), mean(&eap));
        let (mut num, mut dt, mut de) = (0.0, 0.0, 0.0);
        for p in 0..n_persons {
            num += (theta_true[p] - mt) * (eap[p] - me);
            dt += (theta_true[p] - mt).powi(2);
            de += (eap[p] - me).powi(2);
        }
        let corr = num / (dt.sqrt() * de.sqrt());
        assert!(corr > 0.8, "theta EAP corr {corr}");
    }

    #[test]
    fn fit_poly_unidim_recovers_grm() {
        let (n_persons, n_items, k) = (4000usize, 6usize, 4usize);
        let mut st = 99887766u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
        // ordered-decreasing thresholds (valid GRM)
        let thr_true: Vec<Vec<f64>> = (0..n_items).map(|_| vec![1.4, 0.1, -1.2]).collect();
        let mut y = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let lp = grm_logprobs(a_true[i] * theta, &thr_true[i]);
                let uu = u();
                let mut cum = 0.0_f64;
                let mut cat = k - 1;
                for (c, l) in lp.iter().enumerate() {
                    cum += l.exp();
                    if uu < cum {
                        cat = c;
                        break;
                    }
                }
                y[p * n_items + i] = cat;
            }
        }
        let fit = fit_poly_unidim(&y, n_persons, n_items, k, PolyModel::Grm, 21, 80, 1e-6).unwrap();
        assert!(fit.loglik.is_finite());
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mh) = (mean(&a_true), mean(&fit.slope));
        let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
        for i in 0..n_items {
            num += (a_true[i] - ma) * (fit.slope[i] - mh);
            da += (a_true[i] - ma).powi(2);
            dh += (fit.slope[i] - mh).powi(2);
        }
        let corr = num / (da.sqrt() * dh.sqrt());
        assert!(corr > 0.9, "grm slope corr {corr}; hat={:?}", fit.slope);
        // thresholds recovered near truth (pooled mean abs error, item 0)
        let mae: f64 = (0..3).map(|j| (fit.cat_params[0][j] - thr_true[0][j]).abs()).sum::<f64>() / 3.0;
        assert!(mae < 0.25, "grm threshold MAE {mae}: {:?}", fit.cat_params[0]);
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
