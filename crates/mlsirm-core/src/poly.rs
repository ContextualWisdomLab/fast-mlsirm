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
pub(crate) fn solve_small(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Vec<f64> {
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
    observed: Option<&[bool]>,
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
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();

    // init: log_a = 0; category params from base rates (GPCM) / cumulative rates (GRM)
    let mut params = vec![vec![0.0_f64; n_cat]; n_items];
    for i in 0..n_items {
        let mut freq = vec![1e-3_f64; n_cat];
        for p in 0..n_persons {
            if is_obs(p, i) {
                freq[y[p * n_items + i]] += 1.0;
            }
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
                if !is_obs(p, i) {
                    continue;
                }
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
                if !is_obs(p, i) {
                    continue;
                }
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

/// Result of [`fit_nominal`]. Per item, `scores[i]` holds the `K-1` free
/// category scoring values `a_1..a_{K-1}` and `intercepts[i]` the `K-1` free
/// intercepts `c_1..c_{K-1}` (the baseline category is pinned `a_0 = c_0 = 0`).
pub struct NominalFit {
    pub scores: Vec<Vec<f64>>,
    pub intercepts: Vec<Vec<f64>>,
    pub loglik: f64,
    pub n_iter: usize,
}

/// Negative expected complete-data log-lik and gradient for one item of the
/// nominal model. `params = [a_1..a_{K-1}, c_1..c_{K-1}]` (2(K-1) free values);
/// the cell is `softmax_k(a_k·θ + c_k)` with `a_0 = c_0 = 0`. The gradient reuses
/// the softmax residual `r_k − n·P_k`: `∂/∂c_k = residual_k`,
/// `∂/∂a_k = residual_k·θ`, summed over the quadrature nodes.
fn nominal_item_neg_ll_grad(
    params: &[f64],
    nodes: &[f64],
    counts: &[Vec<f64>],
    n_cat: usize,
) -> (f64, Vec<f64>) {
    let z = n_cat - 1;
    let mut scores = vec![0.0_f64; n_cat];
    let mut intercepts = vec![0.0_f64; n_cat];
    for m in 0..z {
        scores[m + 1] = params[m];
        intercepts[m + 1] = params[z + m];
    }
    let mut ll = 0.0_f64;
    let mut grad = vec![0.0_f64; 2 * z];
    for (nd, &theta) in nodes.iter().enumerate() {
        let lp = gpcm_logprobs(theta, &scores, &intercepts);
        ll += counts[nd].iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
        let (g_int, _g_base, g_sc) = gpcm_node_gradient(theta, &scores, &intercepts, &counts[nd]);
        for m in 0..z {
            grad[m] += g_sc[m]; // d / d a_{m+1}
            grad[z + m] += g_int[m]; // d / d c_{m+1}
        }
    }
    (-ll, grad.iter().map(|v| -v).collect())
}

/// Newton M-step for one nominal item (finite-difference Hessian on the analytic
/// gradient), mirroring [`m_step_item`].
fn nominal_m_step(
    mut params: Vec<f64>,
    nodes: &[f64],
    counts: &[Vec<f64>],
    n_cat: usize,
    n_newton: usize,
) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (_f, g) = nominal_item_neg_ll_grad(&params, nodes, counts, n_cat);
        let h = 1e-5;
        let mut hess = vec![vec![0.0_f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = nominal_item_neg_ll_grad(&pj, nodes, counts, n_cat);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
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

/// Unidimensional nominal categories model (Bock, 1972; Thissen, Cai & Bock,
/// 2010) by Bock-Aitkin marginal MLE. Each item has a free scoring function
/// `a_k` and intercept `c_k` per category, `P(Y=k|θ) = softmax_k(a_k·θ + c_k)`,
/// identified by the baseline constraint `a_0 = c_0 = 0` with `θ ~ N(0,1)`. The
/// GPCM is the special case `a_k = a·k` (integer-linear scoring), so the nominal
/// fit nests it; parameters are identified up to a joint reflection
/// `(a_k, θ) → (−a_k, −θ)`. `y` is `n_persons * n_items` row-major, categories
/// `0..n_cat-1`; reuses the softmax cell and residual gradient of the GPCM path.
///
/// # References (APA 7th ed.)
///
/// Bock, R. D. (1972). Estimating item parameters and latent ability when
///   responses are scored in two or more nominal categories. *Psychometrika,
///   37*(1), 29–51. https://doi.org/10.1007/BF02291411
///
/// Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item
///   response model. In M. L. Nering & R. Ostini (Eds.), *Handbook of
///   polytomous item response theory models* (pp. 43–75). Routledge.
#[allow(clippy::too_many_arguments)]
pub fn fit_nominal(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<NominalFit, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }
    let z = n_cat - 1;
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();

    // init: integer (GPCM-like) scores a_k = k; intercepts from base rates
    let mut params = vec![vec![0.0_f64; 2 * z]; n_items];
    for i in 0..n_items {
        let mut freq = vec![1e-3_f64; n_cat];
        for p in 0..n_persons {
            if is_obs(p, i) {
                freq[y[p * n_items + i]] += 1.0;
            }
        }
        let tot: f64 = freq.iter().sum();
        for f in freq.iter_mut() {
            *f /= tot;
        }
        for m in 0..z {
            params[i][m] = (m + 1) as f64; // a_{m+1}
            params[i][z + m] = (freq[m + 1] / freq[0]).ln(); // c_{m+1}
        }
    }

    let mut prev_ll = f64::NEG_INFINITY;
    let mut ll = f64::NEG_INFINITY;
    let mut it = 0;
    while it < max_iter {
        let mut item_lp = vec![vec![0.0_f64; qn * n_cat]; n_items];
        for i in 0..n_items {
            let mut scores = vec![0.0_f64; n_cat];
            let mut intercepts = vec![0.0_f64; n_cat];
            for m in 0..z {
                scores[m + 1] = params[i][m];
                intercepts[m + 1] = params[i][z + m];
            }
            for (nd, &theta) in nodes.iter().enumerate() {
                let lp = gpcm_logprobs(theta, &scores, &intercepts);
                item_lp[i][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
        }
        let mut counts = vec![vec![vec![0.0_f64; n_cat]; qn]; n_items];
        ll = 0.0;
        let mut log_node = vec![0.0_f64; qn];
        for p in 0..n_persons {
            for nd in 0..qn {
                log_node[nd] = log_w[nd];
            }
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
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
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                for nd in 0..qn {
                    let post = (log_node[nd] - mx).exp() / denom;
                    counts[i][nd][yc] += post;
                }
            }
        }
        for i in 0..n_items {
            params[i] = nominal_m_step(params[i].clone(), nodes, &counts[i], n_cat, 10);
        }
        it += 1;
        if (ll - prev_ll).abs() < tol * (1.0 + prev_ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    let scores: Vec<Vec<f64>> = params.iter().map(|p| p[0..z].to_vec()).collect();
    let intercepts: Vec<Vec<f64>> = params.iter().map(|p| p[z..2 * z].to_vec()).collect();
    Ok(NominalFit { scores, intercepts, loglik: ll, n_iter: it })
}

/// Per-person polytomous person-fit result.
pub struct PolyPersonFit {
    /// Standardized log-likelihood `l_z` per person.
    pub lz: Vec<f64>,
    /// Snijders (2001) `l_z*` corrected for the estimated trait.
    pub lz_star: Vec<f64>,
    /// EAP trait estimate used (per person).
    pub theta_eap: Vec<f64>,
    /// `l_z* < flag_threshold` (aberrant / misfitting response pattern).
    pub flagged: Vec<bool>,
}

/// Person-fit statistics for polytomous responses under a fitted GRM/GPCM: the
/// standardized log-likelihood `l_z` (Drasgow, Levine & Williams, 1985) and its
/// estimated-trait correction `l_z*` (Snijders, 2001), evaluated at the EAP
/// trait. With `l_0 = Σ_i log P_i(y_i|θ̂)`, `E = Σ_i Σ_k P_ik log P_ik`, and
/// `V = Σ_i (Σ_k P_ik (log P_ik)² − (Σ_k P_ik log P_ik)²)`,
/// `l_z = (l_0 − E) / √V`; `l_z*` subtracts the covariance of the log-likelihood
/// with the trait score (`c = ΣCov / ΣI`, `τ² = V − (ΣCov)²/ΣI`) and adds the
/// MAP prior score `r_0 = −(θ̂ − μ)/σ²`. The score derivative `∂/∂θ log P_ik` is
/// taken by central difference, so the routine is model-agnostic. This reduces
/// exactly to the binary [`crate::fitstats::person_fit`] `l_z` at `n_cat = 2`.
/// Low (negative) values flag aberrant patterns.
///
/// # References (APA 7th ed.)
///
/// Drasgow, F., Levine, M. V., & Williams, E. A. (1985). Appropriateness
///   measurement with polychotomous item response models and standardized
///   indices. *British Journal of Mathematical and Statistical Psychology,
///   38*(1), 67–86. https://doi.org/10.1111/j.2044-8317.1985.tb00817.x
///
/// Snijders, T. A. B. (2001). Asymptotic null distribution of person fit
///   statistics with estimated person parameter. *Psychometrika, 66*(3),
///   331–342. https://doi.org/10.1007/BF02294437
#[allow(clippy::too_many_arguments)]
pub fn poly_person_fit(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: PolyModel,
    q_theta: usize,
    prior_mean: f64,
    prior_sd: f64,
    flag_threshold: f64,
) -> Result<PolyPersonFit, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if slope.len() != n_items {
        return Err("slope must have length n_items".into());
    }
    if cat_params.len() != n_items * (n_cat - 1) {
        return Err("cat_params must have length n_items*(n_cat-1)".into());
    }
    if !(prior_sd > 0.0) {
        return Err("prior_sd must be positive".into());
    }
    let z = n_cat - 1;
    let (theta_eap, _sd) =
        score_poly_eap(y, observed, n_persons, n_items, n_cat, slope, cat_params, model, q_theta)?;
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let cell = |i: usize, theta: f64| -> Vec<f64> {
        let a = slope[i];
        let cp = &cat_params[i * z..(i + 1) * z];
        let base = a * theta;
        match model {
            PolyModel::Gpcm => {
                let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; n_cat];
                ic[1..].copy_from_slice(cp);
                gpcm_logprobs(base, &scores, &ic)
            }
            PolyModel::Grm => grm_logprobs(base, cp),
        }
    };
    let h = 1e-4;
    let mut lz = vec![f64::NAN; n_persons];
    let mut lz_star = vec![f64::NAN; n_persons];
    let mut flagged = vec![false; n_persons];
    for p in 0..n_persons {
        let th = theta_eap[p];
        let (mut w, mut sv, mut sc, mut si) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut n_obs = 0usize;
        for i in 0..n_items {
            if !is_obs(p, i) {
                continue;
            }
            let lp = cell(i, th);
            let lpp = cell(i, th + h);
            let lpm = cell(i, th - h);
            let (mut mu, mut e2, mut cov, mut info) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for k in 0..n_cat {
                let pk = lp[k].exp();
                let lgk = lp[k];
                let dk = (lpp[k] - lpm[k]) / (2.0 * h); // d/dtheta log P_ik
                mu += pk * lgk;
                e2 += pk * lgk * lgk;
                cov += pk * lgk * dk;
                info += pk * dk * dk;
            }
            w += lp[y[p * n_items + i]] - mu;
            sv += e2 - mu * mu;
            sc += cov;
            si += info;
            n_obs += 1;
        }
        if n_obs < 2 || sv <= 0.0 {
            continue;
        }
        lz[p] = w / sv.sqrt();
        let c = if si > 1e-12 { sc / si } else { 0.0 };
        let r0 = -(th - prior_mean) / (prior_sd * prior_sd);
        let tau2 = sv - if si > 1e-12 { sc * sc / si } else { 0.0 };
        if tau2 > 1e-12 {
            lz_star[p] = (w + c * r0) / tau2.sqrt();
            flagged[p] = lz_star[p] < flag_threshold;
        }
    }
    Ok(PolyPersonFit { lz, lz_star, theta_eap, flagged })
}

/// Maximum-Fisher-information next-item selection for a polytomous CAT: returns
/// the index of the not-yet-`administered` bank item with the largest item
/// information at the current trait estimate `theta`, or `None` if all items are
/// administered. Reuses [`poly_item_information`].
pub fn poly_cat_next_item(
    theta: f64,
    administered: &[bool],
    slope: &[f64],
    cat_params: &[f64],
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
) -> Option<usize> {
    let z = n_cat - 1;
    let mut best: Option<usize> = None;
    let mut best_val = f64::NEG_INFINITY;
    for i in 0..n_items {
        if administered[i] {
            continue;
        }
        let info = poly_item_information(theta, slope[i], &cat_params[i * z..(i + 1) * z], model);
        if info > best_val {
            best_val = info;
            best = Some(i);
        }
    }
    best
}

/// Result of [`poly_cat_simulate`] — per simulee, the final EAP trait, its
/// posterior SD (the CAT standard error), and the number of items administered.
pub struct PolyCatResult {
    pub theta_eap: Vec<f64>,
    pub theta_sd: Vec<f64>,
    pub n_used: Vec<usize>,
}

/// Simulate a polytomous computerized adaptive test (Dodd, De Ayala & Koch,
/// 1995) for each true trait in `true_theta`, over a fixed GRM/GPCM item bank.
/// Items are picked by maximum Fisher information at the running EAP estimate
/// (or at random when `adaptive = false`, a baseline), responses are generated
/// at the simulee's true trait, and the trait + posterior SD are re-estimated
/// after each item via [`score_poly_eap`]. Administration stops once at least
/// `min_items` are given and the posterior SD falls below `se_threshold`, or at
/// `max_items`. Set `se_threshold = 0` with `min_items = max_items` for a
/// fixed-length CAT.
///
/// # References (APA 7th ed.)
///
/// Dodd, B. G., De Ayala, R. J., & Koch, W. R. (1995). Computerized adaptive
///   testing with polytomous items. *Applied Psychological Measurement, 19*(1),
///   5–22. https://doi.org/10.1177/014662169501900103
#[allow(clippy::too_many_arguments)]
pub fn poly_cat_simulate(
    true_theta: &[f64],
    slope: &[f64],
    cat_params: &[f64],
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
    q_theta: usize,
    se_threshold: f64,
    min_items: usize,
    max_items: usize,
    adaptive: bool,
    seed: u64,
) -> Result<PolyCatResult, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if slope.len() != n_items || cat_params.len() != n_items * (n_cat - 1) {
        return Err("slope/cat_params must match n_items and n_cat".into());
    }
    if n_items < 2 {
        return Err("CAT needs a bank of at least 2 items".into());
    }
    let z = n_cat - 1;
    let n_sim = true_theta.len();
    let max_it = max_items.min(n_items);
    let mut st = seed.max(1);
    let mut u = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let cell = |i: usize, theta: f64| -> Vec<f64> {
        let base = slope[i] * theta;
        let cp = &cat_params[i * z..(i + 1) * z];
        match model {
            PolyModel::Gpcm => {
                let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; n_cat];
                ic[1..].copy_from_slice(cp);
                gpcm_logprobs(base, &scores, &ic)
            }
            PolyModel::Grm => grm_logprobs(base, cp),
        }
    };
    let mut theta_eap = vec![0.0_f64; n_sim];
    let mut theta_sd = vec![0.0_f64; n_sim];
    let mut n_used = vec![0usize; n_sim];
    for s in 0..n_sim {
        let tt = true_theta[s];
        let mut administered = vec![false; n_items];
        let mut y = vec![0usize; n_items];
        let mut th = 0.0_f64;
        let mut se = f64::INFINITY;
        let mut count = 0usize;
        while count < max_it {
            if count >= min_items && se < se_threshold {
                break;
            }
            let pick = if adaptive {
                poly_cat_next_item(th, &administered, slope, cat_params, n_items, n_cat, model)
            } else {
                let remaining: Vec<usize> =
                    (0..n_items).filter(|&i| !administered[i]).collect();
                if remaining.is_empty() {
                    None
                } else {
                    Some(remaining[((u() * remaining.len() as f64) as usize).min(remaining.len() - 1)])
                }
            };
            let item = match pick {
                Some(i) => i,
                None => break,
            };
            // simulate the response at the true trait
            let lp = cell(item, tt);
            let draw = u();
            let (mut acc, mut cat) = (0.0_f64, n_cat - 1);
            for (c, l) in lp.iter().enumerate() {
                acc += l.exp();
                if draw <= acc {
                    cat = c;
                    break;
                }
            }
            administered[item] = true;
            y[item] = cat;
            count += 1;
            let (eap, sd) = score_poly_eap(
                &y, Some(&administered), 1, n_items, n_cat, slope, cat_params, model, q_theta,
            )?;
            th = eap[0];
            se = sd[0];
        }
        theta_eap[s] = th;
        theta_sd[s] = se;
        n_used[s] = count;
    }
    Ok(PolyCatResult { theta_eap, theta_sd, n_used })
}

/// Result of [`fit_poly_multigroup`]. `slope`/`cat_params` are the parameters
/// common across groups; when an item is studied, `studied_slope`/`studied_cat`
/// hold its per-group parameters (length `n_groups`). `mu`/`sigma` are the
/// per-group latent means/SDs (reference group 0 pinned to `N(0,1)`).
pub struct TwoGroupPolyFit {
    pub slope: Vec<f64>,
    pub cat_params: Vec<Vec<f64>>,
    pub studied_slope: Vec<f64>,
    pub studied_cat: Vec<Vec<f64>>,
    pub mu: Vec<f64>,
    pub sigma: Vec<f64>,
    pub loglik: f64,
    pub n_iter: usize,
}

/// Multi-group polytomous marginal MLE (Bock-Zimowski population), the estimator
/// behind the likelihood-ratio DIF test. Persons carry a `group_id` (group 0 is
/// the reference, pinned to `N(0,1)`); each other group's latent distribution
/// `N(mu_g, sigma_g^2)` is estimated. Items are common across groups (the
/// anchor) except `studied_item`, whose parameters are freed per group in the
/// augmented model (`studied_item = Some(j)`); with `studied_item = None` all
/// items are common (the compact model). The node-shift reparameterization
/// `theta_{g,t} = mu_g + sigma_g x_t` keeps the shared Gauss-Hermite weights, and
/// the per-item M-step reuses [`m_step_item`] by stacking each group's nodes and
/// expected counts (the concatenation is exactly the Bock-Zimowski pooling).
///
/// # References (APA 7th ed.)
///
/// Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van der
///   Linden & R. K. Hambleton (Eds.), *Handbook of modern item response theory*
///   (pp. 433–448). Springer. https://doi.org/10.1007/978-1-4757-2691-6_25
#[allow(clippy::too_many_arguments)]
pub fn fit_poly_multigroup(
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
    studied_item: Option<usize>,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<TwoGroupPolyFit, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if n_groups < 2 {
        return Err("n_groups must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    if group_id.len() != n_persons {
        return Err("group_id must have length n_persons".into());
    }
    if group_id.iter().any(|&g| g >= n_groups) {
        return Err("group_id labels must be < n_groups".into());
    }
    // Every declared group must be populated, otherwise the `df = (n_groups-1)*
    // n_cat` used by the LR test would count parameters no data can identify
    // (an empty group's item params stay at init and contribute nothing to the
    // likelihood), making the test miscalibrated. Callers with sparse labels
    // should compact them first (the Python wrapper does).
    let mut group_n = vec![0usize; n_groups];
    for &g in group_id {
        group_n[g] += 1;
    }
    if group_n.iter().any(|&c| c == 0) {
        return Err("every group 0..n_groups-1 must contain at least one person".into());
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if let Some(j) = studied_item {
        if j >= n_items {
            return Err("studied_item out of range".into());
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();

    // pooled init from all groups (same scheme as fit_poly_unidim)
    let mut params = vec![vec![0.0_f64; n_cat]; n_items];
    for i in 0..n_items {
        let mut freq = vec![1e-3_f64; n_cat];
        for p in 0..n_persons {
            if is_obs(p, i) {
                freq[y[p * n_items + i]] += 1.0;
            }
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
                let mut cum = 0.0_f64;
                for k in (1..n_cat).rev() {
                    cum += freq[k];
                    let c = cum.clamp(1e-4, 1.0 - 1e-4);
                    params[i][k] = (c / (1.0 - c)).ln();
                }
            }
        }
    }
    let mut studied_params: Vec<Vec<f64>> = match studied_item {
        Some(j) => vec![params[j].clone(); n_groups],
        None => Vec::new(),
    };
    let mut mu = vec![0.0_f64; n_groups];
    let mut sigma = vec![1.0_f64; n_groups];

    let mut prev_ll = f64::NEG_INFINITY;
    let mut ll = f64::NEG_INFINITY;
    let mut it = 0;
    while it < max_iter {
        // group-specific trait locations for the shared standard nodes
        let theta: Vec<Vec<f64>> = (0..n_groups)
            .map(|g| nodes.iter().map(|&x| mu[g] + sigma[g] * x).collect())
            .collect();
        // per-group cell log-probs
        let mut item_lp = vec![vec![vec![0.0_f64; qn * n_cat]; n_items]; n_groups];
        for g in 0..n_groups {
            for i in 0..n_items {
                let p_i = if Some(i) == studied_item { &studied_params[g] } else { &params[i] };
                let a = p_i[0].exp();
                for (t, &th) in theta[g].iter().enumerate() {
                    let base = a * th;
                    let lp = match model {
                        PolyModel::Gpcm => {
                            let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                            let mut ic = vec![0.0_f64; n_cat];
                            ic[1..].copy_from_slice(&p_i[1..]);
                            gpcm_logprobs(base, &scores, &ic)
                        }
                        PolyModel::Grm => grm_logprobs(base, &p_i[1..]),
                    };
                    item_lp[g][i][t * n_cat..(t + 1) * n_cat].copy_from_slice(&lp);
                }
            }
        }
        // E-step: per-group posteriors, expected counts, and trait moments
        let mut counts = vec![vec![vec![vec![0.0_f64; n_cat]; qn]; n_items]; n_groups];
        let mut w_acc = vec![0.0_f64; n_groups];
        let mut s1 = vec![0.0_f64; n_groups];
        let mut s2 = vec![0.0_f64; n_groups];
        ll = 0.0;
        let mut log_node = vec![0.0_f64; qn];
        for p in 0..n_persons {
            let g = group_id[p];
            for t in 0..qn {
                log_node[t] = log_w[t];
            }
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                for t in 0..qn {
                    log_node[t] += item_lp[g][i][t * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0_f64;
            for t in 0..qn {
                denom += (log_node[t] - mx).exp();
            }
            ll += mx + denom.ln();
            for t in 0..qn {
                let post = (log_node[t] - mx).exp() / denom;
                w_acc[g] += post;
                s1[g] += post * theta[g][t];
                s2[g] += post * theta[g][t] * theta[g][t];
                for i in 0..n_items {
                    if is_obs(p, i) {
                        counts[g][i][t][y[p * n_items + i]] += post;
                    }
                }
            }
        }
        // M-step, item parameters
        for i in 0..n_items {
            if Some(i) == studied_item {
                for g in 0..n_groups {
                    studied_params[g] =
                        m_step_item(studied_params[g].clone(), &theta[g], &counts[g][i], model, 10);
                }
            } else {
                let mut stacked_nodes = Vec::with_capacity(n_groups * qn);
                let mut stacked_counts = Vec::with_capacity(n_groups * qn);
                for g in 0..n_groups {
                    stacked_nodes.extend_from_slice(&theta[g]);
                    for t in 0..qn {
                        stacked_counts.push(counts[g][i][t].clone());
                    }
                }
                params[i] = m_step_item(params[i].clone(), &stacked_nodes, &stacked_counts, model, 10);
            }
        }
        // M-step, focal group latent distributions (reference g=0 pinned)
        for g in 1..n_groups {
            if w_acc[g] > 0.0 {
                let mean = s1[g] / w_acc[g];
                let var = (s2[g] / w_acc[g] - mean * mean).max(0.01);
                mu[g] = mean;
                sigma[g] = var.sqrt().clamp(0.1, 10.0);
            }
        }
        it += 1;
        if (ll - prev_ll).abs() < tol * (1.0 + prev_ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    let slope: Vec<f64> = (0..n_items).map(|i| params[i][0].exp()).collect();
    let cat_params: Vec<Vec<f64>> = params.iter().map(|p| p[1..].to_vec()).collect();
    let (studied_slope, studied_cat) = if studied_item.is_some() {
        (
            studied_params.iter().map(|p| p[0].exp()).collect(),
            studied_params.iter().map(|p| p[1..].to_vec()).collect(),
        )
    } else {
        (Vec::new(), Vec::new())
    };
    Ok(TwoGroupPolyFit { slope, cat_params, studied_slope, studied_cat, mu, sigma, loglik: ll, n_iter: it })
}

/// One studied item's likelihood-ratio DIF result.
pub struct PolyDifRow {
    pub item: usize,
    pub lr: f64,
    pub df: usize,
    pub p_value: f64,
    pub flagged_bh: bool,
    /// Unsigned across-group range (>= 0) of the item's mean category-location: a
    /// DIF magnitude, monotone in uniform DIF — a size, not a direction. `NaN` if
    /// the augmented fit for this item did not converge to finite parameters.
    pub effect_size: f64,
}

/// Likelihood-ratio DIF sweep for polytomous items (Thissen, Steinberg & Wainer,
/// 1993, framework; Woehr & Meriac, 2010, for GRM/GPCM). Fits the compact model
/// (all items common across groups) once, then, per studied item, the augmented
/// model (that item freed per group); `LR = 2(ll_aug - ll_compact)` is compared
/// to `chi²((n_groups-1) * n_cat)`. Because the focal latent distribution is
/// estimated in both models, genuine group ability differences (impact) are
/// absorbed and not misread as DIF. `studied_items = None` sweeps every item
/// against the all-others anchor; Benjamini-Hochberg controls the FDR at
/// `fdr_q`.
///
/// # References (APA 7th ed.)
///
/// Thissen, D., Steinberg, L., & Wainer, H. (1993). Detection of differential
///   item functioning using the parameters of item response models. In P. W.
///   Holland & H. Wainer (Eds.), *Differential item functioning* (pp. 67–113).
///   Erlbaum.
///
/// Woehr, D. J., & Meriac, J. P. (2010). Using polytomous item response theory
///   to examine differential item and test functioning: The case of work ethic.
///   In N. T. Tippins & S. Adler (Eds.), *Technology-enhanced assessment of
///   talent* (pp. 199–229). Jossey-Bass.
#[allow(clippy::too_many_arguments)]
pub fn poly_dif_sweep(
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    model: PolyModel,
    studied_items: Option<&[usize]>,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
    fdr_q: f64,
) -> Result<Vec<PolyDifRow>, String> {
    let con = fit_poly_multigroup(
        y, observed, group_id, n_groups, n_persons, n_items, n_cat, model, None, q_theta, max_iter,
        tol,
    )?;
    // A non-finite compact log-likelihood (e.g. GRM thresholds disordered on a
    // sparse category) would make every `2*(ll_aug - ll_con)` NaN, which the
    // `.max(0.0)` clamp below would silently turn into LR=0 / p=1 — reporting all
    // items as clean. Fail loudly instead.
    if !con.loglik.is_finite() {
        return Err("compact multi-group fit did not reach a finite log-likelihood \
                    (a group may have a rarely-used category; try model=\"gpcm\")"
            .into());
    }
    let items: Vec<usize> = match studied_items {
        Some(s) => s.to_vec(),
        None => (0..n_items).collect(),
    };
    let df = (n_groups - 1) * n_cat;
    let mut rows: Vec<PolyDifRow> = Vec::with_capacity(items.len());
    for &j in &items {
        if j >= n_items {
            return Err("studied item out of range".into());
        }
        let aug = fit_poly_multigroup(
            y, observed, group_id, n_groups, n_persons, n_items, n_cat, model, Some(j), q_theta,
            max_iter, tol,
        )?;
        // If this item's augmented fit diverged, surface it as NaN rather than let
        // `.max(0.0)` mask a failed fit as LR=0 (a silent "no DIF" false negative).
        let (lr, p_value) = if aug.loglik.is_finite() {
            let lr = (2.0 * (aug.loglik - con.loglik)).max(0.0);
            (lr, crate::fitstats::chi2_sf(lr, df as f64))
        } else {
            (f64::NAN, f64::NAN)
        };
        let bbar: Vec<f64> = aug
            .studied_cat
            .iter()
            .map(|c| c.iter().sum::<f64>() / c.len().max(1) as f64)
            .collect();
        let hi = bbar.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lo = bbar.iter().cloned().fold(f64::INFINITY, f64::min);
        rows.push(PolyDifRow {
            item: j,
            lr,
            df,
            p_value,
            flagged_bh: false,
            effect_size: hi - lo,
        });
    }
    let pvals: Vec<f64> = rows.iter().map(|r| r.p_value).collect();
    let bh = crate::fitstats::benjamini_hochberg(&pvals, fdr_q);
    for (r, &f) in rows.iter_mut().zip(&bh) {
        r.flagged_bh = f;
    }
    Ok(rows)
}

/// Per-person nonparametric polytomous person-fit result ([`u3_poly_person_fit`]).
pub struct U3PolyResult {
    /// Raw U3poly in `[0, 1]` (0 = popularity-consistent, 1 = maximally aberrant);
    /// `NaN` where undefined (an interior total score whose attainable weighted
    /// range collapses to zero).
    pub u3poly: Vec<f64>,
    /// `NC_p`, the person's summed ordinal score over observed items (the
    /// conditioning group the statistic is normalized within).
    pub total_score: Vec<usize>,
    /// `u3poly >= cutoff` (all `false` when `cutoff` is `None` or `u3poly` is
    /// `NaN`).
    pub flagged: Vec<bool>,
}

/// Min-plus and max-plus convolution of a set of per-item cumulative-weight
/// vectors `cw[i][0..=m]` (`cw[i][x] = sum_{s<=x} w_{i,s}`, `cw[i][0]=0`): returns
/// `(max_w, min_w)` over total step counts `0..=n_items*m`, where `max_w[t]` /
/// `min_w[t]` are the largest / smallest attainable weighted score for a response
/// pattern whose ordinal scores sum to `t`. Both are exact DPs; the max side is
/// **not** the flat "sum of the t largest step weights" shortcut, which
/// over-counts when a clamped (unused) category breaks within-item monotonicity.
fn u3_min_max_conv(cw: &[&[f64]], m: usize) -> (Vec<f64>, Vec<f64>) {
    let total = cw.len() * m;
    let mut max_dp = vec![f64::NEG_INFINITY; total + 1];
    let mut min_dp = vec![f64::INFINITY; total + 1];
    max_dp[0] = 0.0;
    min_dp[0] = 0.0;
    let mut reach = 0usize;
    for ci in cw {
        let mut nmax = vec![f64::NEG_INFINITY; total + 1];
        let mut nmin = vec![f64::INFINITY; total + 1];
        for t in 0..=reach {
            let (mv, nv) = (max_dp[t], min_dp[t]);
            if mv == f64::NEG_INFINITY {
                continue;
            }
            for x in 0..=m {
                let nt = t + x;
                let a = mv + ci[x];
                if a > nmax[nt] {
                    nmax[nt] = a;
                }
                let b = nv + ci[x];
                if b < nmin[nt] {
                    nmin[nt] = b;
                }
            }
        }
        max_dp = nmax;
        min_dp = nmin;
        reach += m;
    }
    (max_dp, min_dp)
}

/// van der Flier's (1980, 1982) `U3` person-fit statistic generalized to ordered
/// polytomous items (Emons, 2008): a *nonparametric* index that needs no fitted
/// IRT model. Each item-step response function `P(Y_i >= m)` is estimated by its
/// sample proportion, turned into a logit weight `w_{i,m} = ln(pi/(1-pi))`
/// (a degenerate step with `pi in {0,1}` contributes 0), and a person's observed
/// weighted score `W_p = sum_i sum_{m<=y_i} w_{i,m}` is compared to the largest
/// and smallest weighted scores attainable at that person's total score `NC_p`
/// (the conditioning group): `U3 = (maxW(NC_p) - W_p) / (maxW(NC_p) - minW(NC_p))`,
/// in `[0, 1]` with 1 = maximally popularity-inconsistent (misfit). The min/max
/// bounds are computed by exact DP ([`u3_min_max_conv`]). Perfect patterns
/// (`NC_p in {0, n_items*(n_cat-1)}`) take the reference `den = 1` (statistic 0),
/// matching the `PerFit` reference implementation; a `NaN` is returned only when
/// an *interior* score's attainable range collapses. Items must be keyed so a
/// higher category means more of the trait (recode reverse-keyed items first).
/// `cutoff` (see [`u3_poly_bootstrap_cutoff`]) flags `u3poly >= cutoff`; the raw
/// statistic has no reliable analytic reference distribution, so flagging uses a
/// simulated critical value rather than a normal approximation.
///
/// # References (APA 7th ed.)
///
/// Emons, W. H. M. (2008). Nonparametric person-fit analysis of polytomous item
///   scores. *Applied Psychological Measurement, 32*(3), 224–247.
///   https://doi.org/10.1177/0146621607302479
///
/// van der Flier, H. (1982). Deviant response patterns and comparability of test
///   scores. *Journal of Cross-Cultural Psychology, 13*(3), 267–298.
///   https://doi.org/10.1177/0022002182013003001
pub fn u3_poly_person_fit(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    cutoff: Option<f64>,
) -> Result<U3PolyResult, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if let Some(c) = cutoff {
        if !c.is_finite() {
            return Err("cutoff must be finite".into());
        }
    }
    let m = n_cat - 1;
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    // per-item cumulative logit weights cw[i][0..=m] from sample ISRF proportions
    let mut cw = vec![vec![0.0_f64; n_cat]; n_items];
    for i in 0..n_items {
        let mut freq = vec![0usize; n_cat];
        let mut nobs = 0usize;
        for p in 0..n_persons {
            if is_obs(p, i) {
                freq[y[p * n_items + i]] += 1;
                nobs += 1;
            }
        }
        // suffix counts: ge[s] = #(x >= s)
        let mut ge = vec![0usize; n_cat + 1];
        for c in (0..n_cat).rev() {
            ge[c] = ge[c + 1] + freq[c];
        }
        let mut cum = 0.0_f64;
        for step in 1..=m {
            let pi = if nobs > 0 { ge[step] as f64 / nobs as f64 } else { 0.0 };
            let w = if pi <= 0.0 || pi >= 1.0 { 0.0 } else { (pi / (1.0 - pi)).ln() };
            cum += w;
            cw[i][step] = cum;
        }
    }

    // shared bounds for the complete-data path (computed once)
    let all_cw: Vec<&[f64]> = cw.iter().map(|v| v.as_slice()).collect();
    let (gmax, gmin) = u3_min_max_conv(&all_cw, m);
    let full_complete = observed.is_none();

    let mut u3poly = vec![0.0_f64; n_persons];
    let mut total_score = vec![0usize; n_persons];
    for p in 0..n_persons {
        let complete = full_complete || (0..n_items).all(|i| is_obs(p, i));
        let (mx, mn, total_steps, wsum, nc) = if complete {
            let mut wsum = 0.0_f64;
            let mut nc = 0usize;
            for i in 0..n_items {
                let x = y[p * n_items + i];
                wsum += cw[i][x];
                nc += x;
            }
            (gmax[nc], gmin[nc], n_items * m, wsum, nc)
        } else {
            // ponytail: per-person DP only on the missing path; the person's
            // attainable range spans only their observed item-steps.
            let obs_cw: Vec<&[f64]> =
                (0..n_items).filter(|&i| is_obs(p, i)).map(|i| cw[i].as_slice()).collect();
            let (pmax, pmin) = u3_min_max_conv(&obs_cw, m);
            let mut wsum = 0.0_f64;
            let mut nc = 0usize;
            for &i in (0..n_items).filter(|&i| is_obs(p, i)).collect::<Vec<_>>().iter() {
                let x = y[p * n_items + i];
                wsum += cw[i][x];
                nc += x;
            }
            let ts = obs_cw.len() * m;
            (pmax[nc], pmin[nc], ts, wsum, nc)
        };
        total_score[p] = nc;
        u3poly[p] = if total_steps == 0 {
            // no observed item-steps => no conditioning group, statistic undefined
            // (distinct from a complete all-min/all-max pattern, which is a real 0)
            f64::NAN
        } else {
            // PerFit boundary: perfect patterns get reference den = 1 (statistic 0);
            // an interior score whose range collapses is genuinely undefined.
            let den = if nc == 0 || nc == total_steps { 1.0 } else { mx - mn };
            if den > 1e-9 { (mx - wsum) / den } else { f64::NAN }
        };
    }

    let flagged: Vec<bool> = match cutoff {
        Some(c) => u3poly.iter().map(|&v| v.is_finite() && v >= c).collect(),
        None => vec![false; n_persons],
    };
    Ok(U3PolyResult { u3poly, total_score, flagged })
}

/// Simulated critical value for [`u3_poly_person_fit`]: the empirical
/// `1 - alpha` quantile of the raw U3poly statistic under `n_rep` complete
/// datasets drawn from a fitted GRM/GPCM at `theta ~ N(0, 1)` (a parametric
/// bootstrap, following Emons, 2008, who used simulated critical values because
/// U3poly has no usable analytic null). `slope`/`cat_params` are the item bank
/// (`cat_params` flattened `n_items * (n_cat-1)`). Because the null distribution
/// depends on the latent distribution, a cutoff from this `N(0,1)` bootstrap is
/// only appropriate when that population assumption is reasonable. The
/// replications are complete (`n_items`-long) patterns, so the cutoff is
/// calibrated for complete responders; a person with substantial missing data has
/// a shorter, coarser null and should not be flagged against this cutoff.
#[allow(clippy::too_many_arguments)]
pub fn u3_poly_bootstrap_cutoff(
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: PolyModel,
    alpha: f64,
    n_rep: usize,
    seed: u64,
) -> Result<f64, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if slope.len() != n_items || cat_params.len() != n_items * (n_cat - 1) {
        return Err("slope/cat_params must match n_items and n_cat".into());
    }
    if n_persons < 1 || n_items < 1 {
        return Err("need at least one person and item".into());
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err("alpha must be in (0, 1)".into());
    }
    if n_rep < 1 {
        return Err("n_rep must be >= 1".into());
    }
    let z = n_cat - 1;
    let mut st = seed.max(1);
    let mut u = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let cell = |i: usize, theta: f64| -> Vec<f64> {
        let base = slope[i] * theta;
        let cp = &cat_params[i * z..(i + 1) * z];
        match model {
            PolyModel::Gpcm => {
                let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; n_cat];
                ic[1..].copy_from_slice(cp);
                gpcm_logprobs(base, &scores, &ic)
            }
            PolyModel::Grm => grm_logprobs(base, cp),
        }
    };
    let mut pool: Vec<f64> = Vec::with_capacity(n_rep * n_persons);
    let mut y = vec![0usize; n_persons * n_items];
    for _rep in 0..n_rep {
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let lp = cell(i, theta);
                let d = u();
                let (mut acc, mut cat) = (0.0_f64, n_cat - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if d <= acc {
                        cat = c;
                        break;
                    }
                }
                y[p * n_items + i] = cat;
            }
        }
        let res = u3_poly_person_fit(&y, None, n_persons, n_items, n_cat, None)?;
        pool.extend(res.u3poly.into_iter().filter(|v| v.is_finite()));
    }
    if pool.is_empty() {
        return Err("bootstrap produced no finite U3poly values".into());
    }
    pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let np = pool.len();
    let idx = (np as f64 - 1.0) * (1.0 - alpha);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let q = if lo == hi { pool[lo] } else { pool[lo] + (idx - lo as f64) * (pool[hi] - pool[lo]) };
    Ok(q)
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
    observed: Option<&[bool]>,
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
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if slope.len() != n_items || cat_params.len() != n_items * (n_cat - 1) {
        return Err("slope/cat_params sizes inconsistent with n_items/n_cat".into());
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
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
            if !is_obs(p, i) {
                continue;
            }
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

/// Per-item generalized S-X² polytomous item-fit result.
pub struct PolySX2Result {
    pub statistic: Vec<f64>,
    pub df: Vec<f64>,
    pub p_value: Vec<f64>,
    /// Retained independent cells `Σ_g (#supercats_g − 1)` before the `−m`
    /// parameter adjustment — the reference df when the statistic is evaluated
    /// at KNOWN (not estimated) item parameters.
    pub n_cells: Vec<usize>,
}

/// Generalized S-X² item fit for ordered polytomous IRT — Kang & Chen (2008)
/// for the GPCM, Kang & Chen (2011) for the GRM — the category extension of the
/// binary Orlando-Thissen statistic in [`crate::fitstats::s_x2`]. Persons are
/// grouped by the summed score `k = 0..F` (`F = n_items * (n_cat-1)`); the
/// model-expected category proportions
/// `E_ikz = ∫ P_i(z|θ) f*ᵢ(k-z|θ) φ(θ) dθ / ∫ f(k|θ) φ(θ) dθ`
/// use the generalized Lord-Wingersky summed-score recursion (Thissen,
/// Pommerich, Billeaud & Williams 1995), with `f*ᵢ` the leave-one-out
/// distribution. Groups `k < Z` collapse into the boundary group `Z`, groups
/// `k > F−Z` into `F−Z`, and `k = 0`, `k = F` are excluded (their off-boundary
/// cells are structurally zero); within a retained group adjacent *categories*
/// are collapsed left-to-right to hold every expected cell frequency at or above
/// `min_expected` (Kang & Chen's category-collapsing, feasible where score-group
/// collapsing would erase the table). `df = Σ_g (#cells_g − 1) − m`, with
/// `m = n_cat` estimated item parameters (slope + `K−1` category parameters).
///
/// At `n_cat = 2` this reduces exactly to [`crate::fitstats::s_x2`] on the same
/// grid; all items are assumed to share `n_cat` categories (the fitter's
/// setting). Only persons observed on every item enter the summed-score table.
///
/// # References (APA 7th ed.)
///
/// Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X² item fit
///   index for polytomous IRT models. *Journal of Educational Measurement,
///   45*(4), 391–406. https://doi.org/10.1111/j.1745-3984.2008.00070.x
///
/// Kang, T., & Chen, T. T. (2011). Performance of the generalized S-X² item fit
///   index for the graded response model. *Asia Pacific Education Review,
///   12*(1), 89–96. https://doi.org/10.1007/s12564-010-9082-4
///
/// Orlando, M., & Thissen, D. (2000). Likelihood-based item-fit indices for
///   dichotomous item response theory models. *Applied Psychological
///   Measurement, 24*(1), 50–64. https://doi.org/10.1177/01466216000241003
///
/// Thissen, D., Pommerich, M., Billeaud, K., & Williams, V. A. (1995). Item
///   response theory for scores on tests including polytomous items with ordered
///   responses. *Applied Psychological Measurement, 19*(1), 39–49.
///   https://doi.org/10.1177/014662169501900105
///
/// Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and
///   equipercentile observed-score "equatings." *Applied Psychological
///   Measurement, 8*(4), 453–461. https://doi.org/10.1177/014662168400800409
#[allow(clippy::too_many_arguments)]
pub fn poly_s_x2(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    slope: &[f64],
    cat_params: &[f64],
    model: PolyModel,
    q_theta: usize,
    min_expected: f64,
) -> Result<PolySX2Result, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if n_items < 2 {
        return Err("n_items must be >= 2".into());
    }
    if y.len() != n_persons * n_items {
        return Err("y must have length n_persons * n_items".into());
    }
    if slope.len() != n_items {
        return Err("slope must have length n_items".into());
    }
    if cat_params.len() != n_items * (n_cat - 1) {
        return Err("cat_params must have length n_items * (n_cat - 1)".into());
    }
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if y.iter().any(|&v| v >= n_cat) {
        return Err("response categories must be < n_cat".into());
    }

    let z = n_cat - 1; // highest category score Z
    let f_max = n_items * z; // perfect summed score F
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let qn = nodes.len();

    // per-item category probabilities at each node: probs[(i*qn + t)*n_cat + zc]
    let mut probs = vec![0.0_f64; n_items * qn * n_cat];
    for i in 0..n_items {
        let a = slope[i];
        let cp = &cat_params[i * z..(i + 1) * z];
        for (t, &theta) in nodes.iter().enumerate() {
            let base = a * theta;
            let lp = match model {
                PolyModel::Gpcm => {
                    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                    let mut intercepts = vec![0.0_f64; n_cat];
                    intercepts[1..].copy_from_slice(cp);
                    gpcm_logprobs(base, &scores, &intercepts)
                }
                PolyModel::Grm => grm_logprobs(base, cp),
            };
            let off = (i * qn + t) * n_cat;
            for zc in 0..n_cat {
                probs[off + zc] = lp[zc].exp();
            }
        }
    }

    // generalized Lord-Wingersky over an item subset: f[k*qn + t], k = 0..items*z
    let poly_lw = |items: &[usize]| -> Vec<f64> {
        let max_s = items.len() * z;
        let mut dist = vec![0.0_f64; (max_s + 1) * qn];
        for t in 0..qn {
            dist[t] = 1.0; // score 0 has probability 1 before adding any item
        }
        let mut cur = 0usize;
        for &i in items {
            let mut next = vec![0.0_f64; (max_s + 1) * qn];
            for s in 0..=cur {
                for zc in 0..n_cat {
                    let off = (i * qn) * n_cat + zc;
                    let (drow, srow) = ((s + zc) * qn, s * qn);
                    for t in 0..qn {
                        next[drow + t] += dist[srow + t] * probs[off + t * n_cat];
                    }
                }
            }
            cur += z;
            dist = next;
        }
        dist
    };

    let all: Vec<usize> = (0..n_items).collect();
    let f_all = poly_lw(&all);
    let denom: Vec<f64> = (0..=f_max)
        .map(|k| (0..qn).map(|t| f_all[k * qn + t] * weights[t]).sum())
        .collect();

    // observed counts by total score: nk[k] and obs[(i*(F+1)+k)*n_cat + zc]
    let complete = |p: usize| observed.map_or(true, |o| (0..n_items).all(|i| o[p * n_items + i]));
    let mut nk = vec![0.0_f64; f_max + 1];
    let mut obs = vec![0.0_f64; n_items * (f_max + 1) * n_cat];
    for p in 0..n_persons {
        if !complete(p) {
            continue;
        }
        let total: usize = (0..n_items).map(|i| y[p * n_items + i]).sum();
        nk[total] += 1.0;
        for i in 0..n_items {
            obs[(i * (f_max + 1) + total) * n_cat + y[p * n_items + i]] += 1.0;
        }
    }

    let m = n_cat as f64; // slope + (K-1) category parameters
    let n_buckets = f_max - 2 * z + 1; // groups k in [Z, F-Z] after boundary merge
    let mut out = PolySX2Result {
        statistic: vec![f64::NAN; n_items],
        df: vec![f64::NAN; n_items],
        p_value: vec![f64::NAN; n_items],
        n_cells: vec![0; n_items],
    };
    if n_buckets == 0 {
        return Ok(out);
    }

    for i in 0..n_items {
        let rest: Vec<usize> = (0..n_items).filter(|&j| j != i).collect();
        let f_rest = poly_lw(&rest); // f*ᵢ, scores 0..(F-z)
        let rest_max = f_max - z;
        // observed/expected counts per (bucket, category); k in [1, F-1]
        let mut bo = vec![0.0_f64; n_buckets * n_cat];
        let mut be = vec![0.0_f64; n_buckets * n_cat];
        let mut bn = vec![0.0_f64; n_buckets];
        for k in 1..f_max {
            if denom[k] <= 0.0 {
                continue;
            }
            let bucket = k.clamp(z, f_max - z) - z;
            bn[bucket] += nk[k];
            for zc in 0..n_cat {
                bo[bucket * n_cat + zc] += obs[(i * (f_max + 1) + k) * n_cat + zc];
                if k >= zc && k - zc <= rest_max {
                    let kr = k - zc;
                    let num: f64 = (0..qn)
                        .map(|t| probs[(i * qn + t) * n_cat + zc] * f_rest[kr * qn + t] * weights[t])
                        .sum();
                    be[bucket * n_cat + zc] += nk[k] * num / denom[k];
                }
            }
        }
        // per bucket: collapse adjacent categories to min_expected, accumulate chi-square
        let mut x2 = 0.0_f64;
        let mut cells = 0usize;
        for g in 0..n_buckets {
            if bn[g] <= 0.0 {
                continue;
            }
            let mut supers: Vec<(f64, f64)> = Vec::new();
            let (mut ao, mut ae) = (0.0_f64, 0.0_f64);
            for zc in 0..n_cat {
                ao += bo[g * n_cat + zc];
                ae += be[g * n_cat + zc];
                if ae >= min_expected {
                    supers.push((ao, ae));
                    ao = 0.0;
                    ae = 0.0;
                }
            }
            if ao > 0.0 || ae > 0.0 {
                if let Some(last) = supers.last_mut() {
                    last.0 += ao;
                    last.1 += ae;
                } else {
                    supers.push((ao, ae));
                }
            }
            for &(o, e) in &supers {
                if e > 0.0 {
                    x2 += (o - e) * (o - e) / e;
                }
            }
            cells += supers.len().saturating_sub(1);
        }
        out.statistic[i] = x2;
        out.n_cells[i] = cells;
        let df = cells as f64 - m;
        if df >= 1.0 {
            out.df[i] = df;
            out.p_value[i] = crate::fitstats::chi2_sf(x2, df);
        }
    }
    Ok(out)
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
    fn poly_k2_matches_trusted_binary_mmle() {
        // Cross-validation against an ALREADY-VALIDATED reference (not self-
        // recovery): at K=2 the GPCM cell is exactly the 2PL, P(Y=1) =
        // sigmoid(a*theta + c_1). The polytomous fitter must reproduce the
        // repo's binary MMLE-EM (mmle::fit_mmle_2pl, NumPy-parity + real-data
        // validated) item parameters on the same data, to a small RMSE.
        use crate::mmle::{fit_mmle_2pl, MmleConfig};
        let (n_persons, n_items) = (4000usize, 8usize);
        let mut st = 271828u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.12 * i as f64).collect();
        let b_true: Vec<f64> = (0..n_items).map(|i| -0.9 + 0.25 * i as f64).collect();
        let mut yf = vec![0.0_f64; n_persons * n_items];
        let mut yi = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let eta = a_true[i] * theta + b_true[i];
                let pr = 1.0 / (1.0 + (-eta).exp());
                let v = if u() < pr { 1.0 } else { 0.0 };
                yf[p * n_items + i] = v;
                yi[p * n_items + i] = v as usize;
            }
        }
        let observed = vec![true; n_persons * n_items];
        let bin = fit_mmle_2pl(
            &yf, &observed, n_persons, n_items,
            &MmleConfig { max_iter: 500, tol: 1e-7, ridge_a: 1e-4, ridge_b: 1e-4, newton_iter: 25 },
        );
        let rmse = |a: &[f64], b: &[f64]| {
            (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
        };
        // BOTH cells reduce to the 2PL at K=2 (GRM is the default): each must
        // match the trusted binary MMLE's item parameters on the same data.
        for model in [PolyModel::Gpcm, PolyModel::Grm] {
            let poly = fit_poly_unidim(&yi, None, n_persons, n_items, 2, model, 41, 300, 1e-7).unwrap();
            let c1: Vec<f64> = poly.cat_params.iter().map(|c| c[0]).collect();
            let ra = rmse(&poly.slope, &bin.a);
            let rb = rmse(&c1, &bin.b);
            assert!(ra < 0.1, "{model:?} slope RMSE vs trusted binary MMLE: {ra}");
            assert!(rb < 0.1, "{model:?} intercept RMSE vs trusted binary MMLE: {rb}");
        }
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
        let fit = fit_poly_unidim(&y, None, n_persons, n_items, k, PolyModel::Gpcm, 21, 80, 1e-6).unwrap();
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
    fn fit_poly_unidim_recovers_with_missing_data() {
        let (n_persons, n_items, k) = (5000usize, 6usize, 3usize);
        let mut st = 5150u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
        let c_true: Vec<Vec<f64>> =
            (0..n_items).map(|i| vec![0.0, 0.3 - 0.1 * i as f64, -0.2 + 0.1 * i as f64]).collect();
        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let mut y = vec![0usize; n_persons * n_items];
        let mut observed = vec![true; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                if u() < 0.25 {
                    observed[p * n_items + i] = false; // ~25% MCAR missing
                    continue;
                }
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
        let fit = fit_poly_unidim(
            &y,
            Some(&observed),
            n_persons,
            n_items,
            k,
            PolyModel::Gpcm,
            21,
            80,
            1e-6,
        )
        .unwrap();
        assert!(fit.loglik.is_finite());
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        let (ma, mh) = (mean(&a_true), mean(&fit.slope));
        let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
        for i in 0..n_items {
            num += (a_true[i] - ma) * (fit.slope[i] - mh);
            da += (a_true[i] - ma).powi(2);
            dh += (fit.slope[i] - mh).powi(2);
        }
        assert!(num / (da.sqrt() * dh.sqrt()) > 0.9, "slope corr under missingness");
        // absolute agreement, not just association
        let s_rmse = (a_true.iter().zip(&fit.slope).map(|(x, y)| (x - y).powi(2)).sum::<f64>()
            / n_items as f64)
            .sqrt();
        assert!(s_rmse < 0.2, "slope RMSE under missingness {s_rmse}");
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
            score_poly_eap(&y, None, n_persons, n_items, k, &a_true, &cat_flat, PolyModel::Gpcm, 41)
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
        let fit = fit_poly_unidim(&y, None, n_persons, n_items, k, PolyModel::Grm, 21, 80, 1e-6).unwrap();
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

    // deterministic uniform draws for the item-fit tests
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut st = seed;
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }

    #[test]
    fn poly_s_x2_reduces_to_binary_orlando_thissen() {
        // At K=2 the generalized S-X² must equal the trusted binary Orlando-
        // Thissen s_x2 (crate::fitstats) EXACTLY on the same quadrature grid:
        // both GRM and GPCM cells reduce to the 2PL P(Y=1)=sigmoid(a*theta+b),
        // and the summed-score recursion / expected proportions coincide. Large
        // N + few centered items keep either statistic out of its collapsing
        // regime, so the agreement is bit-for-bit (min_expected tiny on both).
        use crate::fitstats::{s_x2, SX2Config};
        use crate::nodes::XiRule;
        use crate::scoring::{ItemBank, PriorSpec};
        use crate::ModelType;
        let (n_persons, n_items, q_theta) = (4000usize, 6usize, 41usize);
        let mut u = rng(13579);
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.1 * i as f64).collect();
        let b_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.24 * i as f64).collect();
        let mut yi = vec![0usize; n_persons * n_items];
        for _p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let pr = 1.0 / (1.0 + (-(a_true[i] * theta + b_true[i])).exp());
                yi[_p * n_items + i] = if u() < pr { 1 } else { 0 };
            }
        }
        let yf: Vec<f64> = yi.iter().map(|&v| v as f64).collect();
        let observed_bool = vec![true; n_persons * n_items];
        let alpha: Vec<f64> = a_true.iter().map(|a| a.ln()).collect();
        let zeta = vec![0.0_f64; n_items];
        let fid = vec![0usize; n_items];
        let bank = ItemBank {
            alpha: &alpha, b: &b_true, zeta: &zeta, tau: -50.0, factor_id: &fid,
            model_type: ModelType::Mirt, n_dims: 1, latent_dim: 1, eps_distance: 1e-8,
        };
        let bin = s_x2(
            &bank, &yf, &observed_bool, n_persons, &PriorSpec::standard(1),
            &SX2Config { q_theta, xi_rule: XiRule::GaussHermite { q_xi: 1 }, min_expected: 1e-9, ..Default::default() },
            None,
        )
        .unwrap();
        for model in [PolyModel::Grm, PolyModel::Gpcm] {
            let poly = poly_s_x2(
                &yi, None, n_persons, n_items, 2, &a_true, &b_true, model, q_theta, 1e-9,
            )
            .unwrap();
            for i in 0..n_items {
                assert!(
                    (poly.statistic[i] - bin.statistic[i]).abs() < 1e-8,
                    "{model:?} item {i}: poly {} vs binary {}",
                    poly.statistic[i], bin.statistic[i]
                );
                assert_eq!(
                    poly.df[i], bin.df[i],
                    "{model:?} item {i} df: poly {:?} vs binary {:?}", poly.df[i], bin.df[i]
                );
            }
        }
    }

    #[test]
    fn poly_s_x2_is_calibrated_at_true_parameters() {
        // Kang & Chen (2008/2011) headline: under the true model the generalized
        // S-X² tracks its reference chi-square. Evaluated at the KNOWN generating
        // parameters the reference df is the retained cell count (no −m estimation
        // adjustment), so E[S-X²] ≈ Σ cells. We reproduce this — an ABSOLUTE
        // agreement of the sampling mean with its theoretical value, the analogue
        // of an RMSE recovery check for a fit statistic — for both GPCM (2008) and
        // GRM (2011), which is exactly what a mis-calibrated index (e.g. Yen's
        // Q1 / PARSCALE G², inflated to many times its df) would fail.
        let (n_persons, n_items, n_cat, reps) = (1500usize, 8usize, 4usize, 24usize);
        for model in [PolyModel::Gpcm, PolyModel::Grm] {
            let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * i as f64).collect();
            let cat_true: Vec<f64> = (0..n_items)
                .flat_map(|i| match model {
                    // GPCM additive intercepts (any reals)
                    PolyModel::Gpcm => vec![0.8 - 0.06 * i as f64, 0.0, -0.8 + 0.06 * i as f64],
                    // GRM thresholds must be strictly decreasing for a valid cdf
                    PolyModel::Grm => vec![1.1 + 0.04 * i as f64, 0.0, -1.1 - 0.04 * i as f64],
                })
                .collect();
            let z = n_cat - 1;
            let (mut stat_sum, mut cell_sum) = (0.0_f64, 0.0_f64);
            let mut n_flagged = 0usize;
            let mut n_tested = 0usize;
            for r in 0..reps {
                let mut u = rng(2024_0714 + r as u64 * 97);
                let mut yi = vec![0usize; n_persons * n_items];
                for p in 0..n_persons {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    for i in 0..n_items {
                        let base = a_true[i] * theta;
                        let cp = &cat_true[i * z..(i + 1) * z];
                        let lp = match model {
                            PolyModel::Gpcm => {
                                let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                                let mut ic = vec![0.0_f64; n_cat];
                                ic[1..].copy_from_slice(cp);
                                gpcm_logprobs(base, &scores, &ic)
                            }
                            PolyModel::Grm => grm_logprobs(base, cp),
                        };
                        let draw = u();
                        let mut acc = 0.0_f64;
                        let mut cat = n_cat - 1;
                        for (c, l) in lp.iter().enumerate() {
                            acc += l.exp();
                            if draw <= acc {
                                cat = c;
                                break;
                            }
                        }
                        yi[p * n_items + i] = cat;
                    }
                }
                let res =
                    poly_s_x2(&yi, None, n_persons, n_items, n_cat, &a_true, &cat_true, model, 21, 1.0)
                        .unwrap();
                for i in 0..n_items {
                    if res.n_cells[i] >= 1 && res.statistic[i].is_finite() {
                        stat_sum += res.statistic[i];
                        cell_sum += res.n_cells[i] as f64;
                        n_tested += 1;
                        if res.p_value[i].is_finite() && res.p_value[i] < 0.05 {
                            n_flagged += 1;
                        }
                    }
                }
            }
            let ratio = stat_sum / cell_sum;
            assert!(
                (0.85..=1.15).contains(&ratio),
                "{model:?}: mean S-X² / cells = {ratio} (stat {stat_sum}, cells {cell_sum})"
            );
            // df uses the −m adjustment, so p-values at true params are mildly
            // conservative; the flag rate stays far below the >30% seen for G².
            let flag_rate = n_flagged as f64 / n_tested as f64;
            assert!(flag_rate < 0.15, "{model:?}: flag rate {flag_rate} too high for the true model");
        }
    }

    /// One ability condition's aggregate recovery: absolute-agreement RMSE and
    /// mean |bias| for the slope and the category intercepts.
    struct McRecovery {
        cond: &'static str,
        a_rmse: f64,
        a_bias: f64,
        c_rmse: f64,
        c_bias: f64,
    }

    /// Monte-Carlo parameter-recovery study for the GPCM fitter, generating from
    /// the published item-parameter scheme of Kang & Chen (2008, p. 397): slopes
    /// `a_i ~ lognormal(0, 0.5²)` and four step difficulties `b_{i,c} ~
    /// N(means −1.5, −0.5, 0.5, 1.5; SD 0.5)`. Two ability conditions are run —
    /// NORMAL `θ ~ N(0, 1)` (the fitter's prior, so recovery is near-unbiased)
    /// and right-SKEWED `θ = Exp(1) − 1` (mean 0, var 1, skewness 2), a prior
    /// misspecification Kang & Chen flag as future work. Returns per-condition
    /// RMSE and mean |bias| (absolute agreement, not correlation) over `reps`
    /// replications on a fixed true item bank.
    ///
    /// # References (APA 7th ed.)
    ///
    /// Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X² item
    ///   fit index for polytomous IRT models. *Journal of Educational
    ///   Measurement, 45*(4), 391–406.
    ///   https://doi.org/10.1111/j.1745-3984.2008.00070.x
    /// Muraki, E. (1992). A generalized partial credit model: Application of an
    ///   EM algorithm. *Applied Psychological Measurement, 16*(2), 159–176.
    ///   https://doi.org/10.1177/014662169201600206
    fn mc_gpcm_recovery(reps: usize, n_persons: usize) -> Vec<McRecovery> {
        let (n_items, k) = (5usize, 5usize);
        let z_steps = k - 1; // 4 step difficulties
        let step_means = [-1.5_f64, -0.5, 0.5, 1.5];

        // fixed "true" item bank (drawn once) from the published scheme
        let mut bu = rng(96100);
        let mut bnorm = || {
            let u1 = bu().max(1e-12);
            let u2 = bu();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut a_true = vec![0.0_f64; n_items];
        let mut cat_true = vec![0.0_f64; n_items * z_steps]; // additive intercepts
        for i in 0..n_items {
            a_true[i] = (0.5 * bnorm()).exp(); // lognormal(0, 0.5²)
            let mut cum = 0.0_f64;
            for c in 0..z_steps {
                let b = step_means[c] + 0.5 * bnorm(); // step difficulty
                cum += b;
                cat_true[i * z_steps + c] = -a_true[i] * cum; // GPCM intercept
            }
        }

        let mut out = Vec::new();
        for (cond, skew) in [("normal", false), ("skew", true)] {
            // accumulate signed error and squared error per parameter over reps
            let mut a_err = vec![0.0_f64; n_items];
            let mut a_sq = vec![0.0_f64; n_items];
            let mut c_err = vec![0.0_f64; n_items * z_steps];
            let mut c_sq = vec![0.0_f64; n_items * z_steps];
            for rep in 0..reps {
                let mut u = rng(4242 + rep as u64 * 131 + if skew { 7 } else { 0 });
                let mut yi = vec![0usize; n_persons * n_items];
                for p in 0..n_persons {
                    let theta = if skew {
                        -(u().max(1e-12)).ln() - 1.0 // Exp(1) − 1: mean 0, var 1, skew 2
                    } else {
                        let u1 = u().max(1e-12);
                        let u2 = u();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    };
                    for i in 0..n_items {
                        let base = a_true[i] * theta;
                        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                        let mut ic = vec![0.0_f64; k];
                        ic[1..].copy_from_slice(&cat_true[i * z_steps..(i + 1) * z_steps]);
                        let lp = gpcm_logprobs(base, &scores, &ic);
                        let draw = u();
                        let (mut acc, mut cat) = (0.0_f64, k - 1);
                        for (c, l) in lp.iter().enumerate() {
                            acc += l.exp();
                            if draw <= acc {
                                cat = c;
                                break;
                            }
                        }
                        yi[p * n_items + i] = cat;
                    }
                }
                let fit = fit_poly_unidim(
                    &yi, None, n_persons, n_items, k, PolyModel::Gpcm, 21, 100, 1e-6,
                )
                .unwrap();
                for i in 0..n_items {
                    let ea = fit.slope[i] - a_true[i];
                    a_err[i] += ea;
                    a_sq[i] += ea * ea;
                    for c in 0..z_steps {
                        let ec = fit.cat_params[i][c] - cat_true[i * z_steps + c];
                        c_err[i * z_steps + c] += ec;
                        c_sq[i * z_steps + c] += ec * ec;
                    }
                }
            }
            let r = reps as f64;
            let rmse = |sq: &[f64]| (sq.iter().sum::<f64>() / (sq.len() as f64 * r)).sqrt();
            let mean_bias =
                |er: &[f64]| er.iter().map(|e| (e / r).abs()).sum::<f64>() / er.len() as f64;
            out.push(McRecovery {
                cond,
                a_rmse: rmse(&a_sq),
                a_bias: mean_bias(&a_err),
                c_rmse: rmse(&c_sq),
                c_bias: mean_bias(&c_err),
            });
        }
        out
    }

    fn assert_recovery(out: &[McRecovery], reps: usize, n_persons: usize) {
        for s in out {
            println!(
                "[MC recovery, θ={}] reps={reps} N={n_persons}  \
                 slope: RMSE={:.4} |bias|={:.4}  intercept: RMSE={:.4} |bias|={:.4}",
                s.cond, s.a_rmse, s.a_bias, s.c_rmse, s.c_bias
            );
            assert!(s.a_rmse.is_finite() && s.c_rmse.is_finite());
            if s.cond == "skew" {
                // prior misspecification: recovery holds but degrades (reported)
                assert!(s.a_rmse < 0.45, "skew slope RMSE too large: {}", s.a_rmse);
                assert!(s.c_rmse < 1.2, "skew intercept RMSE too large: {}", s.c_rmse);
            } else {
                // matched prior: tight, near-unbiased recovery
                assert!(s.a_rmse < 0.20, "normal slope RMSE too large: {}", s.a_rmse);
                assert!(s.c_rmse < 0.45, "normal intercept RMSE too large: {}", s.c_rmse);
                assert!(s.a_bias < 0.10, "normal slope bias too large: {}", s.a_bias);
            }
        }
    }

    #[test]
    fn fit_poly_unidim_recovery_ci_guard() {
        // Fast regression guard (few reps). The authoritative >=500-replication
        // study is `fit_poly_unidim_recovery_monte_carlo_500` (ignored below);
        // run it with: cargo test --release -- --ignored --nocapture
        let (reps, n_persons) = (20usize, 1500usize);
        assert_recovery(&mc_gpcm_recovery(reps, n_persons), reps, n_persons);
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn fit_poly_unidim_recovery_monte_carlo_500() {
        // 500-replication recovery study (the sample size common in the IRT
        // Monte-Carlo literature), N = 2000 per replication.
        let (reps, n_persons) = (500usize, 2000usize);
        assert_recovery(&mc_gpcm_recovery(reps, n_persons), reps, n_persons);
    }

    #[test]
    fn fit_nominal_nests_gpcm() {
        // The nominal model contains the GPCM (scores linear in k, a_k = a*k), so
        // fitting nominal to GPCM data must (a) reach a log-likelihood at least as
        // high as the GPCM fit and (b) recover linear scores: a_2/a_1 ≈ 2.
        let (n_persons, n_items, k) = (3000usize, 5usize, 3usize);
        let mut u = rng(778899);
        let a_gpcm: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
        let c_gpcm: Vec<Vec<f64>> = (0..n_items)
            .map(|i| vec![0.3 - 0.1 * i as f64, -0.4 + 0.1 * i as f64])
            .collect();
        let mut yi = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let base = a_gpcm[i] * theta;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&c_gpcm[i]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[p * n_items + i] = cat;
            }
        }
        let gpcm =
            fit_poly_unidim(&yi, None, n_persons, n_items, k, PolyModel::Gpcm, 41, 300, 1e-7).unwrap();
        let nom = fit_nominal(&yi, None, n_persons, n_items, k, 41, 300, 1e-7).unwrap();
        assert!(
            nom.loglik >= gpcm.loglik - 0.5,
            "nominal loglik {} should be >= GPCM {}", nom.loglik, gpcm.loglik
        );
        for i in 0..n_items {
            let (a1, a2) = (nom.scores[i][0], nom.scores[i][1]);
            assert!(
                (a2 / a1 - 2.0).abs() < 0.4,
                "item {i}: recovered scores not linear (a2/a1={})", a2 / a1
            );
        }
    }

    /// Aggregate nominal-model recovery (RMSE and mean |bias|) for the free
    /// scores and intercepts over `reps` datasets at fixed true parameters, with
    /// per-item sign alignment (the model is identified up to (a_k,θ)→(−a_k,−θ)).
    fn mc_nominal_recovery(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64, f64) {
        let (n_items, k) = (6usize, 4usize);
        let z = k - 1;
        let a_true: Vec<Vec<f64>> = (0..n_items)
            .map(|i| vec![0.9 + 0.04 * i as f64, 2.0 - 0.03 * i as f64, 2.7 + 0.05 * i as f64])
            .collect();
        let c_true: Vec<Vec<f64>> = (0..n_items)
            .map(|i| vec![0.5 - 0.05 * i as f64, 0.0, -0.6 + 0.05 * i as f64])
            .collect();
        let (mut a_err, mut a_sq, mut c_err, mut c_sq) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
        let mut cnt = 0.0_f64;
        for rep in 0..reps {
            let mut u = rng(31337 + rep as u64 * 131 + if skew { 9 } else { 0 });
            let mut yi = vec![0usize; n_persons * n_items];
            for p in 0..n_persons {
                let theta = if skew {
                    -(u().max(1e-12)).ln() - 1.0
                } else {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                for i in 0..n_items {
                    let mut scores = vec![0.0_f64; k];
                    let mut intercepts = vec![0.0_f64; k];
                    for m in 0..z {
                        scores[m + 1] = a_true[i][m];
                        intercepts[m + 1] = c_true[i][m];
                    }
                    let lp = gpcm_logprobs(theta, &scores, &intercepts);
                    let draw = u();
                    let (mut acc, mut cat) = (0.0_f64, k - 1);
                    for (c, l) in lp.iter().enumerate() {
                        acc += l.exp();
                        if draw <= acc {
                            cat = c;
                            break;
                        }
                    }
                    yi[p * n_items + i] = cat;
                }
            }
            let fit = fit_nominal(&yi, None, n_persons, n_items, k, 21, 200, 1e-6).unwrap();
            for i in 0..n_items {
                // align the reflection sign to the truth for this item
                let dot: f64 = (0..z).map(|m| fit.scores[i][m] * a_true[i][m]).sum();
                let s = if dot >= 0.0 { 1.0 } else { -1.0 };
                for m in 0..z {
                    let ea = s * fit.scores[i][m] - a_true[i][m];
                    a_err += ea;
                    a_sq += ea * ea;
                    let ec = fit.intercepts[i][m] - c_true[i][m];
                    c_err += ec;
                    c_sq += ec * ec;
                    cnt += 1.0;
                }
            }
        }
        (
            (a_sq / cnt).sqrt(),
            (a_err / cnt).abs(),
            (c_sq / cnt).sqrt(),
            (c_err / cnt).abs(),
        )
    }

    #[test]
    fn fit_nominal_recovery_ci_guard() {
        // Fast guard. Authoritative >=500-rep study is
        // fit_nominal_recovery_monte_carlo_500 (ignored).
        let (reps, n) = (12usize, 2000usize);
        let (ar, ab, cr, cb) = mc_nominal_recovery(reps, n, false);
        let (asr, _, csr, _) = mc_nominal_recovery(reps, n, true);
        println!(
            "[nominal recovery] reps={reps} N={n}  normal: score RMSE={ar:.4} |bias|={ab:.4} \
             intercept RMSE={cr:.4} |bias|={cb:.4}  skew: score RMSE={asr:.4} intercept RMSE={csr:.4}"
        );
        assert!(ar < 0.25 && cr < 0.30, "normal recovery too loose: a={ar}, c={cr}");
        assert!(ab < 0.12, "normal score bias too large: {ab}");
        assert!(asr > ar, "skew should degrade score recovery: {asr} vs {ar}");
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn fit_nominal_recovery_monte_carlo_500() {
        let (reps, n) = (500usize, 2000usize);
        let (ar, ab, cr, cb) = mc_nominal_recovery(reps, n, false);
        let (asr, asb, csr, _) = mc_nominal_recovery(reps, n, true);
        println!(
            "[nominal recovery 500] N={n}  normal: score RMSE={ar:.4} |bias|={ab:.4} \
             intercept RMSE={cr:.4} |bias|={cb:.4}  skew: score RMSE={asr:.4} |bias|={asb:.4} \
             intercept RMSE={csr:.4}"
        );
        assert!(ar < 0.15 && cr < 0.20, "normal recovery too loose: a={ar}, c={cr}");
        assert!(ab < 0.05, "normal score bias not near zero: {ab}");
        assert!(asr > ar + 0.03, "skew should measurably degrade recovery: {asr} vs {ar}");
    }

    #[test]
    fn poly_person_fit_matches_binary_lz_at_k2() {
        // At K=2 the polytomous l_z must equal the trusted binary person_fit l_z
        // on the same EAP trait (both cells reduce to the 2PL); l_z* matches to
        // finite-difference tolerance (poly uses a numerical trait derivative).
        use crate::fitstats::person_fit;
        use crate::scoring::ItemBank;
        use crate::ModelType;
        let (n_persons, n_items) = (1000usize, 12usize);
        let mut u = rng(56789);
        let a: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.06 * i as f64).collect();
        let b: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.14 * i as f64).collect();
        let mut yf = vec![0.0_f64; n_persons * n_items];
        let mut yi = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let u1 = u().max(1e-12);
            let u2 = u();
            let th = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let pr = 1.0 / (1.0 + (-(a[i] * th + b[i])).exp());
                let v = if u() < pr { 1.0 } else { 0.0 };
                yf[p * n_items + i] = v;
                yi[p * n_items + i] = v as usize;
            }
        }
        let obs = vec![true; n_persons * n_items];
        let poly =
            poly_person_fit(&yi, None, n_persons, n_items, 2, &a, &b, PolyModel::Gpcm, 41, 0.0, 1.0, -1.645)
                .unwrap();
        let alpha: Vec<f64> = a.iter().map(|x| x.ln()).collect();
        let zeta = vec![0.0_f64; n_items];
        let fid = vec![0usize; n_items];
        let bank = ItemBank {
            alpha: &alpha, b: &b, zeta: &zeta, tau: -50.0, factor_id: &fid,
            model_type: ModelType::Mirt, n_dims: 1, latent_dim: 1, eps_distance: 1e-8,
        };
        let xi = vec![0.0_f64; n_persons];
        let bin = person_fit(&bank, &yf, &obs, n_persons, &poly.theta_eap, &xi, &[], -1.645).unwrap();
        let (mut d_lz, mut d_lzs) = (0.0_f64, 0.0_f64);
        for p in 0..n_persons {
            if poly.lz[p].is_finite() && bin.lz[p].is_finite() {
                d_lz = d_lz.max((poly.lz[p] - bin.lz[p]).abs());
                d_lzs = d_lzs.max((poly.lz_star[p] - bin.lz_star[p]).abs());
            }
        }
        assert!(d_lz < 1e-6, "l_z max diff vs binary: {d_lz}");
        assert!(d_lzs < 5e-3, "l_z* max diff vs binary: {d_lzs}");
    }

    // GPCM person-fit Monte-Carlo: a fraction of respondents answer carelessly
    // (uniform random categories) and the rest come from the model; evaluated at
    // the true item parameters. Returns (Type I flag rate among model
    // respondents, power among careless respondents, mean l_z*, sd l_z*).
    fn mc_poly_person_fit(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64, f64) {
        let (n_items, k) = (20usize, 3usize);
        let z = k - 1;
        let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.03 * i as f64).collect();
        let cat_true: Vec<f64> = (0..n_items)
            .flat_map(|i| vec![0.6 - 0.01 * i as f64, -0.6 + 0.01 * i as f64])
            .collect();
        let n_care = n_persons / 10; // first 10% are careless
        let (mut n_norm, mut flag_norm, mut flag_care) = (0usize, 0usize, 0usize);
        let (mut sum, mut sum2) = (0.0_f64, 0.0_f64);
        for rep in 0..reps {
            let mut u = rng(7000 + rep as u64 * 131 + if skew { 3 } else { 0 });
            let mut yi = vec![0usize; n_persons * n_items];
            for p in 0..n_persons {
                let careless = p < n_care;
                let theta = if skew {
                    -(u().max(1e-12)).ln() - 1.0
                } else {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                for i in 0..n_items {
                    // careless / inconsistent responder: the implied trait alternates
                    // +-1.6 across items, so no single theta fits the pattern.
                    let theta_use = if careless {
                        if i % 2 == 0 { 1.6 } else { -1.6 }
                    } else {
                        theta
                    };
                    let base = a_true[i] * theta_use;
                    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                    let mut ic = vec![0.0_f64; k];
                    ic[1..].copy_from_slice(&cat_true[i * z..(i + 1) * z]);
                    let lp = gpcm_logprobs(base, &scores, &ic);
                    let draw = u();
                    let (mut acc, mut cat) = (0.0_f64, k - 1);
                    for (c, l) in lp.iter().enumerate() {
                        acc += l.exp();
                        if draw <= acc {
                            cat = c;
                            break;
                        }
                    }
                    yi[p * n_items + i] = cat;
                }
            }
            let pf = poly_person_fit(
                &yi, None, n_persons, n_items, k, &a_true, &cat_true, PolyModel::Gpcm, 21, 0.0, 1.0,
                -1.645,
            )
            .unwrap();
            for p in 0..n_persons {
                if p < n_care {
                    if pf.flagged[p] {
                        flag_care += 1;
                    }
                } else {
                    n_norm += 1;
                    if pf.flagged[p] {
                        flag_norm += 1;
                    }
                    if pf.lz_star[p].is_finite() {
                        sum += pf.lz_star[p];
                        sum2 += pf.lz_star[p] * pf.lz_star[p];
                    }
                }
            }
        }
        let mean = sum / n_norm as f64;
        let sd = (sum2 / n_norm as f64 - mean * mean).max(0.0).sqrt();
        (
            flag_norm as f64 / n_norm as f64,
            flag_care as f64 / (n_care * reps) as f64,
            mean,
            sd,
        )
    }

    #[test]
    fn poly_person_fit_type1_and_power() {
        // Fast guard. Authoritative >=500-rep study is
        // poly_person_fit_monte_carlo_500 (ignored).
        let (reps, n) = (8usize, 800usize);
        let (t1, power, mean, sd) = mc_poly_person_fit(reps, n, false);
        let (t1s, _, _, _) = mc_poly_person_fit(reps, n, true);
        println!(
            "[poly person-fit] normal: Type I(l_z*<-1.645)={t1:.3} power(careless)={power:.3} \
             mean(l_z*)={mean:.3} sd(l_z*)={sd:.3}  skew: Type I={t1s:.3}"
        );
        assert!((0.01..=0.12).contains(&t1), "Type I off nominal: {t1}");
        assert!(power > 0.5, "power to flag careless responders too low: {power}");
        assert!(mean.abs() < 0.4 && (0.75..=1.3).contains(&sd), "l_z* not ~N(0,1): mean={mean}, sd={sd}");
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn poly_person_fit_monte_carlo_500() {
        let (reps, n) = (500usize, 600usize);
        let (t1, power, mean, sd) = mc_poly_person_fit(reps, n, false);
        println!(
            "[poly person-fit 500] normal: Type I={t1:.4} power={power:.4} mean(l_z*)={mean:.4} \
             sd(l_z*)={sd:.4}"
        );
        // l_z* runs slightly high at a 20-item test (a documented finite-length
        // effect); it converges to nominal as the test lengthens.
        assert!((0.02..=0.11).contains(&t1), "Type I off nominal: {t1}");
        assert!(power > 0.7, "power too low: {power}");
        assert!(mean.abs() < 0.25 && (0.85..=1.2).contains(&sd), "l_z* not ~N(0,1): mean={mean}, sd={sd}");
    }

    /// A GPCM item bank for the CAT tests: `n_items` items with difficulties
    /// spread across the trait range so the adaptive selector has informative
    /// items at every ability level.
    fn cat_bank(n_items: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
        let z = k - 1;
        let mut slope = vec![0.0_f64; n_items];
        let mut cat = vec![0.0_f64; n_items * z];
        for i in 0..n_items {
            let a = 1.0 + 0.25 * (i % 3) as f64; // 1.0 / 1.25 / 1.5, cycling
            slope[i] = a;
            let b = -2.2 + 4.4 * i as f64 / (n_items - 1) as f64; // spread difficulty
            let mut cum = 0.0_f64;
            for m in 0..z {
                let step = b + (m as f64 - (z as f64 - 1.0) / 2.0) * 0.9;
                cum += step;
                cat[i * z + m] = -a * cum;
            }
        }
        (slope, cat)
    }

    fn cat_rmse(eap: &[f64], true_theta: &[f64]) -> f64 {
        let n = true_theta.len() as f64;
        (eap.iter().zip(true_theta).map(|(e, t)| (e - t).powi(2)).sum::<f64>() / n).sqrt()
    }

    #[test]
    fn poly_cat_recovers_and_beats_random() {
        // Fast guard. Authoritative >=500-simulee study is
        // poly_cat_monte_carlo_500 (ignored).
        let (n_items, k) = (40usize, 4usize);
        let (slope, cat) = cat_bank(n_items, k);
        let n_sim = 300usize;
        let mut u = rng(9001);
        let true_theta: Vec<f64> = (0..n_sim)
            .map(|_| {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            })
            .collect();
        // adaptive, variable length: stop at SE < 0.30
        let var = poly_cat_simulate(
            &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.30, 5, 30, true, 111,
        )
        .unwrap();
        let rmse_var = cat_rmse(&var.theta_eap, &true_theta);
        let mean_items = var.n_used.iter().sum::<usize>() as f64 / n_sim as f64;
        println!(
            "[poly CAT] var-len(SE<.30): RMSE={rmse_var:.3} mean_items={mean_items:.1}/{n_items}"
        );
        assert!(rmse_var < 0.40, "CAT theta RMSE too high: {rmse_var}");
        assert!(mean_items < 0.75 * n_items as f64, "CAT should use fewer than the bank: {mean_items}");
        // fixed length L=12: maximum-information beats random selection
        let adap = poly_cat_simulate(
            &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.0, 12, 12, true, 222,
        )
        .unwrap();
        let rand = poly_cat_simulate(
            &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.0, 12, 12, false, 333,
        )
        .unwrap();
        let (ra, rr) = (cat_rmse(&adap.theta_eap, &true_theta), cat_rmse(&rand.theta_eap, &true_theta));
        println!("[poly CAT] fixed L=12: adaptive RMSE={ra:.3} random RMSE={rr:.3}");
        assert!(ra < rr, "max-information CAT should beat random selection: {ra} vs {rr}");
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 simulees); run with: cargo test --release -- --ignored --nocapture"]
    fn poly_cat_monte_carlo_500() {
        let (n_items, k) = (40usize, 4usize);
        let (slope, cat) = cat_bank(n_items, k);
        let n_sim = 500usize;
        for (label, skew) in [("normal", false), ("skew", true)] {
            let mut u = rng(if skew { 7001 } else { 7000 });
            let true_theta: Vec<f64> = (0..n_sim)
                .map(|_| {
                    if skew {
                        -(u().max(1e-12)).ln() - 1.0
                    } else {
                        let u1 = u().max(1e-12);
                        let u2 = u();
                        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                    }
                })
                .collect();
            let var = poly_cat_simulate(
                &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.30, 5, 30, true, 4242,
            )
            .unwrap();
            let rmse = cat_rmse(&var.theta_eap, &true_theta);
            let mean_items = var.n_used.iter().sum::<usize>() as f64 / n_sim as f64;
            let adap = poly_cat_simulate(
                &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.0, 12, 12, true, 5,
            )
            .unwrap();
            let rand = poly_cat_simulate(
                &true_theta, &slope, &cat, n_items, k, PolyModel::Gpcm, 21, 0.0, 12, 12, false, 6,
            )
            .unwrap();
            let (ra, rr) =
                (cat_rmse(&adap.theta_eap, &true_theta), cat_rmse(&rand.theta_eap, &true_theta));
            println!(
                "[poly CAT 500 θ={label}] var-len RMSE={rmse:.4} mean_items={mean_items:.2}/{n_items}  \
                 fixed L=12: adaptive RMSE={ra:.4} random RMSE={rr:.4}"
            );
            assert!(rmse < 0.42, "{label} CAT RMSE too high: {rmse}");
            assert!(mean_items < 0.7 * n_items as f64, "{label} CAT not saving items: {mean_items}");
            assert!(ra < rr, "{label} adaptive should beat random: {ra} vs {rr}");
        }
    }

    // Two-group GPCM dataset generator for the DIF tests. group 0 = reference
    // theta~N(0,1); group 1 = focal theta~N(0.5, 1.2^2) (impact). `dif` on item 0
    // for the focal group: 0=none, 1=uniform (difficulty shift), 2=non-uniform
    // (slope 1.6x). `skew` draws the focal trait from Exp(1)-1 instead.
    fn gen_two_group_gpcm(
        n_per_group: usize, n_items: usize, k: usize, dif: u8, skew: bool, seed: u64,
    ) -> (Vec<usize>, Vec<usize>) {
        let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.05 * i as f64).collect();
        let int_true: Vec<Vec<f64>> = (0..n_items)
            .map(|i| vec![0.7 - 0.05 * i as f64, -0.7 + 0.05 * i as f64])
            .collect();
        let n_persons = 2 * n_per_group;
        let mut u = rng(seed);
        let mut yi = vec![0usize; n_persons * n_items];
        let mut gid = vec![0usize; n_persons];
        for p in 0..n_persons {
            let focal = p >= n_per_group;
            gid[p] = focal as usize;
            let theta = if !focal {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            } else if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                0.5 + 1.2 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            for i in 0..n_items {
                let (a, ints) = if i == 0 && focal && dif == 1 {
                    let d = 0.6; // uniform: shift difficulty => intercept_k += k*a*d
                    (
                        a_true[0],
                        vec![int_true[0][0] + a_true[0] * d, int_true[0][1] + 2.0 * a_true[0] * d],
                    )
                } else if i == 0 && focal && dif == 2 {
                    (a_true[0] * 1.6, int_true[0].clone())
                } else {
                    (a_true[i], int_true[i].clone())
                };
                let base = a * theta;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&ints);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[p * n_items + i] = cat;
            }
        }
        (yi, gid)
    }

    #[test]
    fn poly_dif_structural_recovers_impact_and_nesting() {
        // No DIF, but the focal group has impact N(0.5, 1.2^2): the estimator
        // must recover the focal distribution and keep the reference pinned; the
        // augmented (item-0-free) model must not fall below the compact one.
        let (n_items, k) = (10usize, 3usize);
        let (yi, gid) = gen_two_group_gpcm(1200, n_items, k, 0, false, 909);
        let np = gid.len();
        let con =
            fit_poly_multigroup(&yi, None, &gid, 2, np, n_items, k, PolyModel::Gpcm, None, 21, 200, 1e-6)
                .unwrap();
        assert_eq!(con.mu[0], 0.0);
        assert_eq!(con.sigma[0], 1.0);
        assert!((con.mu[1] - 0.5).abs() < 0.15, "focal mean not recovered: {}", con.mu[1]);
        assert!((con.sigma[1] - 1.2).abs() < 0.2, "focal sd not recovered: {}", con.sigma[1]);
        let aug = fit_poly_multigroup(
            &yi, None, &gid, 2, np, n_items, k, PolyModel::Gpcm, Some(0), 21, 200, 1e-6,
        )
        .unwrap();
        // nesting, with tolerance-scaled slack (EM loglik lags one M-step)
        let slack = 1e-6_f64.max(1e-6 * (1.0 + con.loglik.abs()));
        assert!(
            aug.loglik >= con.loglik - slack,
            "nesting violated: ll_aug={} ll_con={}", aug.loglik, con.loglik
        );
        assert_eq!(aug.studied_slope.len(), 2);
    }

    #[test]
    fn poly_dif_rejects_empty_declared_group() {
        // Declaring a group with no persons would make df = (n_groups-1)*n_cat
        // count parameters no data can identify (conservative, miscalibrated LR).
        // The data uses labels {0,1}; declaring n_groups=3 leaves group 2 empty.
        let (yi, gid) = gen_two_group_gpcm(300, 6, 3, 0, false, 4242);
        let np = gid.len();
        let err = fit_poly_multigroup(
            &yi, None, &gid, 3, np, 6, 3, PolyModel::Gpcm, None, 21, 50, 1e-4,
        );
        assert!(err.is_err(), "empty declared group should be rejected");
    }

    // (Type I over non-DIF items, power on item 0 when DIF is present, mean LR
    // among null items) over `reps` two-group datasets. df = (G-1)*K = K.
    fn mc_poly_dif(reps: usize, n_per_group: usize, n_items: usize, dif: u8, skew: bool) -> (f64, f64, f64) {
        let k = 3usize;
        let (mut t1_rej, mut t1_cnt) = (0usize, 0usize);
        let (mut pow_rej, mut lr_sum, mut lr_cnt) = (0usize, 0.0_f64, 0usize);
        for rep in 0..reps {
            let seed = 88_000 + rep as u64 * 131 + skew as u64 * 3 + dif as u64 * 7;
            let (yi, gid) = gen_two_group_gpcm(n_per_group, n_items, k, dif, skew, seed);
            let np = gid.len();
            let rows = poly_dif_sweep(
                &yi, None, &gid, 2, np, n_items, k, PolyModel::Gpcm, None, 21, 80, 1e-5, 0.05,
            )
            .unwrap();
            for r in &rows {
                let rej = r.p_value < 0.05;
                if r.item == 0 && dif != 0 {
                    if rej {
                        pow_rej += 1;
                    }
                } else {
                    // non-DIF items (and item 0 when dif==0) measure Type I
                    if rej {
                        t1_rej += 1;
                    }
                    t1_cnt += 1;
                    lr_sum += r.lr;
                    lr_cnt += 1;
                }
            }
        }
        let type1 = t1_rej as f64 / t1_cnt as f64;
        let power = if dif != 0 { pow_rej as f64 / reps as f64 } else { 0.0 };
        (type1, power, lr_sum / lr_cnt as f64)
    }

    #[test]
    fn poly_dif_type1_and_power() {
        // Fast guard (few reps => Type I lower bound is unmeasurable; mean(LR)~df
        // is the robust cheap calibration). Authoritative >=500-rep study with a
        // tight Type I band is poly_dif_monte_carlo_500.
        let df = 3.0; // (G-1)*K = K = 3
        let (t1, _, mean_lr) = mc_poly_dif(3, 400, 6, 0, false); // no DIF
        let (t1u, pow_u, _) = mc_poly_dif(3, 400, 6, 1, false); // uniform DIF on item 0
        println!(
            "[poly DIF] df={df}  no-DIF: Type I={t1:.3} mean(LR)={mean_lr:.2}  \
             uniform: Type I(others)={t1u:.3} power(item0)={pow_u:.3}"
        );
        assert!(t1 < 0.18, "Type I inflated: {t1}"); // lower bound needs the 500-rep test
        assert!((df - 1.2..=df + 1.4).contains(&mean_lr), "mean LR should ~ df={df}: {mean_lr}");
        assert!(pow_u > 0.6, "uniform DIF power too low: {pow_u}");
        assert!(t1u < 0.2, "non-DIF items over-flagged under DIF: {t1u}");
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn poly_dif_monte_carlo_500() {
        let reps = 500usize;
        let (t1, _, mean_lr) = mc_poly_dif(reps, 500, 8, 0, false);
        let (_, pow_u, _) = mc_poly_dif(reps, 500, 8, 1, false);
        let (_, pow_n, _) = mc_poly_dif(reps, 500, 8, 2, false);
        let (t1s, _, _) = mc_poly_dif(reps, 500, 8, 0, true);
        println!(
            "[poly DIF 500] df=3  no-DIF: Type I={t1:.4} mean(LR)={mean_lr:.3}  \
             power: uniform={pow_u:.3} non-uniform={pow_n:.3}  skew: Type I={t1s:.4}"
        );
        assert!((0.03..=0.075).contains(&t1), "Type I off nominal: {t1}");
        assert!((2.6..=3.4).contains(&mean_lr), "mean LR should ~ df=3: {mean_lr}");
        assert!(pow_u > 0.85 && pow_n > 0.7, "DIF power too low: uniform={pow_u} nonuniform={pow_n}");
    }

    // Hand-coded van der Flier dichotomous U3 (the trusted binary reference the
    // polytomous U3 must reduce to at n_cat=2), with the same den=1 boundary.
    fn u3_binary_vdf(y: &[usize], n_persons: usize, n_items: usize) -> Vec<f64> {
        let mut w = vec![0.0_f64; n_items];
        for i in 0..n_items {
            let s: usize = (0..n_persons).map(|p| y[p * n_items + i]).sum();
            let pi = s as f64 / n_persons as f64;
            w[i] = if pi <= 0.0 || pi >= 1.0 { 0.0 } else { (pi / (1.0 - pi)).ln() };
        }
        let mut sorted = w.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
        let mut topsum = vec![0.0_f64; n_items + 1];
        let mut botsum = vec![0.0_f64; n_items + 1];
        for s in 1..=n_items {
            topsum[s] = topsum[s - 1] + sorted[s - 1];
            botsum[s] = botsum[s - 1] + sorted[n_items - s];
        }
        let mut out = vec![0.0_f64; n_persons];
        for p in 0..n_persons {
            let (mut sc, mut wsum) = (0usize, 0.0_f64);
            for i in 0..n_items {
                if y[p * n_items + i] == 1 {
                    sc += 1;
                    wsum += w[i];
                }
            }
            let den = if sc == 0 || sc == n_items { 1.0 } else { topsum[sc] - botsum[sc] };
            out[p] = if den > 1e-9 { (topsum[sc] - wsum) / den } else { f64::NAN };
        }
        out
    }

    #[test]
    fn poly_u3_reduces_to_binary_vdf() {
        // At n_cat=2 the polytomous U3 must be identical to van der Flier's U3
        // (the "reduce to a trusted binary" correctness anchor).
        let mut u = rng(1234);
        let (n_persons, n_items) = (400usize, 12usize);
        let mut y = vec![0usize; n_persons * n_items];
        for v in y.iter_mut() {
            *v = if u() < 0.5 { 1 } else { 0 };
        }
        let res = u3_poly_person_fit(&y, None, n_persons, n_items, 2, None).unwrap();
        let vdf = u3_binary_vdf(&y, n_persons, n_items);
        let mut maxdev = 0.0_f64;
        for p in 0..n_persons {
            let (a, b) = (res.u3poly[p], vdf[p]);
            if a.is_nan() && b.is_nan() {
                continue;
            }
            maxdev = maxdev.max((a - b).abs());
        }
        assert!(maxdev < 1e-10, "U3poly(K=2) must equal vdF U3: maxdev={maxdev}");
        // orientation: a popularity-inconsistent person scores higher than a
        // consistent one. Build two persons on a fixed 4-item bank.
        let ni = 4;
        // popularities descending: item 0 easiest .. item 3 hardest
        let mut yy = vec![0usize; 40 * ni];
        let mut u2 = rng(99);
        for p in 0..40 {
            for i in 0..ni {
                let pi = 0.8 - 0.18 * i as f64; // 0.80,0.62,0.44,0.26
                yy[p * ni + i] = if u2() < pi { 1 } else { 0 };
            }
        }
        // consistent person (easy items 1, hard 0) vs reversed (hard 1, easy 0)
        yy[0 * ni..1 * ni].copy_from_slice(&[1, 1, 0, 0]);
        yy[1 * ni..2 * ni].copy_from_slice(&[0, 0, 1, 1]);
        let r2 = u3_poly_person_fit(&yy, None, 40, ni, 2, None).unwrap();
        assert!(r2.u3poly[1] > r2.u3poly[0], "reversed person must have larger U3");
        assert!(r2.u3poly[0] < 0.5 && r2.u3poly[1] > 0.5, "orientation off: {:?}", &r2.u3poly[..2]);
    }

    // GPCM data generator: first `n_care` persons are careless (uniform-random
    // categories, ignoring item popularity); the rest respond from the model.
    fn gen_u3_data(
        slope: &[f64], cat: &[f64], n_persons: usize, n_items: usize, k: usize,
        n_care: usize, skew: bool, seed: u64,
    ) -> Vec<usize> {
        let z = k - 1;
        let mut u = rng(seed);
        let mut y = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let careless = p < n_care;
            let theta = if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            for i in 0..n_items {
                if careless {
                    y[p * n_items + i] = ((u() * k as f64) as usize).min(k - 1);
                } else {
                    let base = slope[i] * theta;
                    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                    let mut ic = vec![0.0_f64; k];
                    ic[1..].copy_from_slice(&cat[i * z..(i + 1) * z]);
                    let lp = gpcm_logprobs(base, &scores, &ic);
                    let draw = u();
                    let (mut acc, mut c) = (0.0_f64, k - 1);
                    for (cc, l) in lp.iter().enumerate() {
                        acc += l.exp();
                        if draw <= acc {
                            c = cc;
                            break;
                        }
                    }
                    y[p * n_items + i] = c;
                }
            }
        }
        y
    }

    fn quantile_sorted(v: &mut Vec<f64>, q: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = v.len();
        let idx = (n as f64 - 1.0) * q;
        let (lo, hi) = (idx.floor() as usize, idx.ceil() as usize);
        if lo == hi { v[lo] } else { v[lo] + (idx - lo as f64) * (v[hi] - v[lo]) }
    }

    // Returns (marginal Type I, max |flag_rate - alpha| across total-score bins,
    // power on careless responders). The cutoff is the (1-alpha) quantile of null
    // U3poly estimated under the MATCHING latent shape from disjoint seeds.
    fn mc_u3poly(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64) {
        let (n_items, k) = (20usize, 5usize);
        let alpha = 0.05_f64;
        let (slope, cat) = cat_bank(n_items, k);
        let maxnc = n_items * (k - 1);
        let so = if skew { 7 } else { 0 };
        // cutoff from pooled null U3poly (seed base 900000, disjoint from eval)
        let mut pool = Vec::new();
        for b in 0..6u64 {
            let y = gen_u3_data(&slope, &cat, n_persons, n_items, k, 0, skew, 900_000 + b * 131 + so);
            let r = u3_poly_person_fit(&y, None, n_persons, n_items, k, None).unwrap();
            pool.extend(r.u3poly.into_iter().filter(|v| v.is_finite()));
        }
        let cutoff = quantile_sorted(&mut pool, 1.0 - alpha);
        let n_bins = 3usize;
        let (mut bin_flag, mut bin_tot) = (vec![0usize; n_bins], vec![0usize; n_bins]);
        let (mut t1_flag, mut t1_tot) = (0usize, 0usize);
        let (mut pw_flag, mut pw_tot) = (0usize, 0usize);
        let n_care = n_persons / 5; // 20% careless in the power datasets
        for rep in 0..reps as u64 {
            // null eval (disjoint seed base 100000)
            let yn = gen_u3_data(&slope, &cat, n_persons, n_items, k, 0, skew, 100_000 + rep * 131 + so);
            let rn = u3_poly_person_fit(&yn, None, n_persons, n_items, k, Some(cutoff)).unwrap();
            for p in 0..n_persons {
                if rn.u3poly[p].is_finite() {
                    t1_tot += 1;
                    if rn.flagged[p] {
                        t1_flag += 1;
                    }
                    let bin = (rn.total_score[p] * n_bins / (maxnc + 1)).min(n_bins - 1);
                    bin_tot[bin] += 1;
                    if rn.flagged[p] {
                        bin_flag[bin] += 1;
                    }
                }
            }
            // power eval (careless responders, seed base 200000)
            let ya = gen_u3_data(&slope, &cat, n_persons, n_items, k, n_care, skew, 200_000 + rep * 131 + so);
            let ra = u3_poly_person_fit(&ya, None, n_persons, n_items, k, Some(cutoff)).unwrap();
            for p in 0..n_care {
                if ra.u3poly[p].is_finite() {
                    pw_tot += 1;
                    if ra.flagged[p] {
                        pw_flag += 1;
                    }
                }
            }
        }
        let type1 = t1_flag as f64 / t1_tot.max(1) as f64;
        let bin_maxdev = (0..n_bins)
            .map(|b| (bin_flag[b] as f64 / bin_tot[b].max(1) as f64 - alpha).abs())
            .fold(0.0_f64, f64::max);
        let power = pw_flag as f64 / pw_tot.max(1) as f64;
        (type1, bin_maxdev, power)
    }

    #[test]
    fn poly_u3_type1_and_power() {
        // Fast guard. Authoritative >=500-rep study is poly_u3_monte_carlo_500.
        let (t1, _bindev, power) = mc_u3poly(6, 500, false);
        println!("[u3poly] normal: Type I={t1:.3} power(careless)={power:.3}");
        assert!((0.01..=0.12).contains(&t1), "Type I off nominal: {t1}");
        assert!(power > 0.5, "careless-detection power too low: {power}");
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn poly_u3_monte_carlo_500() {
        let reps = 500usize;
        let (t1n, bindev_n, pow_n) = mc_u3poly(reps, 600, false);
        let (t1s, bindev_s, pow_s) = mc_u3poly(reps, 600, true);
        println!(
            "[u3poly 500] normal: Type I={t1n:.4} bin-maxdev={bindev_n:.3} power={pow_n:.3}  \
             skew: Type I={t1s:.4} bin-maxdev={bindev_s:.3} power={pow_s:.3}"
        );
        // marginal Type I calibrated by the simulated cutoff; per-NC-bin deviation
        // reported (a single pooled cutoff cannot perfectly condition on the total
        // score — Emons 2008 uses simulated critical values for this reason).
        assert!((0.03..=0.08).contains(&t1n), "normal Type I off nominal: {t1n}");
        assert!(pow_n > 0.7, "normal careless power too low: {pow_n}");
        assert!(bindev_n < 0.10, "per-score-group miscalibration too large: {bindev_n}");
    }
}
