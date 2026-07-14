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
}
