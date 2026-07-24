//! Polytomous item-response cells and their expected-complete-data gradients,
//! the Rust compute path for the polytomous LSIRM extension
//! (see `docs/papers/gpcm-nominal-design-spec.md` and its literature
//! resolution). All numerical work lives here; the NumPy functions in
//! `fast_mlsirm.estimators.marginal` are parity references only.
//!
//! Two response families over a shared linear predictor `base = a*theta +
//! interaction(x)`:
//!
//! - **GRM** (Samejima 1969, cumulative logit) — the identification-clean
//!   default for the LSIRM family: the single latent-space interaction enters
//!   every cumulative logit as one shared shift inside `base`, so nothing
//!   cancels and no category scaling is forced.
//! - **GPCM** (Muraki 1992, adjacent-category softmax) — an option for
//!   partial-credit scoring; the category-constant term cancels in the softmax,
//!   so the space term enters category-score-scaled (a documented consequence).

pub(crate) const POLY_MAX_CAT: usize = 64;
pub(crate) const POLY_MAX_ITER: usize = 100_000;

fn validate_observed_categories(
    y: &[usize],
    observed: Option<&[bool]>,
    n_cat: usize,
) -> Result<(), String> {
    for (idx, &value) in y.iter().enumerate() {
        if observed.map_or(true, |o| o[idx]) && value >= n_cat {
            return Err("observed response categories must be < n_cat".into());
        }
    }
    Ok(())
}

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
/// Middle-category differences use an algebraically equivalent factorization
/// that remains finite when both cumulative logits are in the same extreme
/// tail.
///
/// # References
///
/// Samejima, F. (1969). Estimation of latent ability using a response pattern
/// of graded scores. *Psychometrika, 34*(S1), 1–97.
/// https://doi.org/10.1007/BF03372160
pub fn grm_logprobs(base: f64, thresholds: &[f64]) -> Vec<f64> {
    let kb = thresholds.len(); // number of boundaries = K-1
    let mut out = vec![0.0_f64; kb + 1];
    if kb == 0 {
        out[0] = 0.0;
        return out;
    }
    // category 0: 1 - sigmoid(base + beta_0) = sigmoid(-(base + beta_0))
    out[0] = log_sigmoid(-(base + thresholds[0]));
    // middle categories 1..K-2: P = sigmoid(base+beta_{k-1}) - sigmoid(base+beta_k)
    for k in 1..kb {
        let upper = base + thresholds[k - 1];
        let lower = base + thresholds[k];
        // sigmoid(upper) - sigmoid(lower)
        // = sigmoid(upper) * sigmoid(-lower) * (1 - exp(lower - upper)).
        // `-expm1` preserves a narrow category and avoids subtracting two
        // rounded log-sigmoids in the same extreme tail.
        out[k] = log_sigmoid(upper) + log_sigmoid(-lower) + (-(lower - upper).exp_m1()).ln();
    }
    // top category K-1: sigmoid(base + beta_{K-2})
    out[kb] = log_sigmoid(base + thresholds[kb - 1]);
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
    let log_p = grm_logprobs(base, thresholds);
    // Evaluate v/P in log space. Directly exponentiating a valid tail category
    // can underflow P to zero even though its score contribution is finite.
    for j in 0..kb {
        let eta = base + thresholds[j];
        let log_v = log_sigmoid(eta) + log_sigmoid(-eta);
        // d q / d s_j = r_{j+1}/P_{j+1} - r_j/P_j  (boundary j sits between cats j and j+1)
        let right = if counts[j + 1] == 0.0 {
            0.0
        } else {
            counts[j + 1] * (log_v - log_p[j + 1]).exp()
        };
        let left = if counts[j] == 0.0 {
            0.0
        } else {
            counts[j] * (log_v - log_p[j]).exp()
        };
        g_t[j] = right - left;
        g_base += right - left;
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
    let p: Vec<f64> = gpcm_logprobs(base, scores, intercepts)
        .iter()
        .map(|&l| l.exp())
        .collect();
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
/// intercepts, or GRM cumulative thresholds). `n_iter` counts completed M-steps;
/// therefore `loglik_trace` contains `n_iter + 1` observed-data likelihoods and
/// its endpoint is evaluated at the returned parameters.
pub struct PolyFit {
    pub slope: Vec<f64>,
    pub cat_params: Vec<Vec<f64>>,
    pub loglik: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub loglik_trace: Vec<f64>,
    pub final_delta: f64,
    pub stopping_tolerance: f64,
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

fn stabilized_newton_step(
    mut step: Vec<f64>,
    gradient: &[f64],
    gradient_norm: f64,
) -> (Vec<f64>, f64, f64) {
    let mut directional = gradient.iter().zip(&step).map(|(g, s)| g * s).sum::<f64>();
    if !step.iter().all(|value| value.is_finite()) || directional <= 0.0 {
        step = gradient.to_vec();
        directional = gradient_norm * gradient_norm;
    }
    let max_step = step.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if max_step <= 2.0 {
        return (step, directional, max_step);
    }
    for value in &mut step {
        *value *= 2.0 / max_step;
    }
    directional = gradient.iter().zip(&step).map(|(g, s)| g * s).sum();
    (step, directional, 2.0)
}

fn checked_em_delta(
    current: f64,
    previous: Option<f64>,
    tolerance: f64,
    iteration: usize,
) -> Result<Option<(f64, f64)>, String> {
    if !current.is_finite() {
        return Err(format!(
            "non-finite observed-data log-likelihood at iteration {iteration}"
        ));
    }
    let Some(previous) = previous else {
        return Ok(None);
    };
    let delta = current - previous;
    let stopping_tolerance = tolerance * (1.0 + previous.abs());
    let monotonic_tolerance = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    if delta < -monotonic_tolerance {
        return Err(format!(
            "EM observed-data log-likelihood decreased at iteration {iteration}: \
             delta={delta:.6e}, monotonic_tolerance={monotonic_tolerance:.6e}"
        ));
    }
    Ok(Some((delta, stopping_tolerance)))
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
                let (g_ic, g_base, _g_sc) =
                    gpcm_node_gradient(base, &scores, &intercepts, &counts[nd]);
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
        let (f0, g) = item_neg_ll_grad(&params, nodes, counts, model);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
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
        let (step, directional, max_step) =
            stabilized_newton_step(solve_small(hess, g.clone()), &g, grad_norm);
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _) = item_neg_ll_grad(&candidate, nodes, counts, model);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    params
}

/// Unidimensional polytomous marginal MLE via Bock-Aitkin EM (no latent-space
/// interaction) — the Rust compute path validating the [`PolyModel`] cells in a
/// full EM loop. `y` is `n_persons * n_items`, row-major, categories `0..n_cat-1`
/// (complete data). `theta ~ N(0,1)` on the `q_theta`-node Gauss-Hermite grid.
///
/// Convergence is checked on the observed-data log likelihood evaluated at the
/// same parameters that are returned. The M-step uses backtracking so that a
/// Newton proposal must improve its expected complete-data objective; a small
/// negative observed-data increment beyond floating-point roundoff is treated as
/// an algorithm error rather than as convergence (Dempster et al., 1977; Wu,
/// 1983).
///
/// # References
///
/// Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from
/// incomplete data via the EM algorithm. *Journal of the Royal Statistical
/// Society: Series B (Methodological), 39*(1), 1–22.
/// https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
///
/// Wu, C. F. J. (1983). On the convergence properties of the EM algorithm. *The
/// Annals of Statistics, 11*(1), 95–103.
/// https://doi.org/10.1214/aos/1176346060
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
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=POLY_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={POLY_MAX_CAT}"));
    }
    if !(1..=POLY_MAX_ITER).contains(&max_iter) {
        return Err(format!("max_iter must be in 1..={POLY_MAX_ITER}"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and > 0".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    validate_observed_categories(y, observed, n_cat)?;
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "q_theta")?;
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

    let mut it = 0;
    let mut converged = false;
    let mut termination_reason = "max_iter".to_owned();
    let mut final_delta = f64::INFINITY;
    let mut stopping_tolerance = f64::INFINITY;
    let mut loglik_trace = Vec::with_capacity(max_iter + 1);
    loop {
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
        let mut ll = 0.0;
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
        if let Some((delta, threshold)) =
            checked_em_delta(ll, loglik_trace.last().copied(), tol, it)?
        {
            final_delta = delta;
            stopping_tolerance = threshold;
            if final_delta <= stopping_tolerance {
                loglik_trace.push(ll);
                converged = true;
                termination_reason = "tolerance".to_owned();
                break;
            }
        }
        loglik_trace.push(ll);
        if it == max_iter {
            break;
        }
        // M-step per item. The next loop evaluates the observed likelihood at
        // these exact parameters before either convergence or max_iter return.
        for i in 0..n_items {
            params[i] = m_step_item(params[i].clone(), nodes, &counts[i], model, 10);
        }
        it += 1;
    }

    let ll = *loglik_trace.last().expect("EM trace is never empty");
    let slope: Vec<f64> = (0..n_items).map(|i| params[i][0].exp()).collect();
    let cat_params: Vec<Vec<f64>> = params.iter().map(|p| p[1..].to_vec()).collect();
    Ok(PolyFit {
        slope,
        cat_params,
        loglik: ll,
        n_iter: it,
        converged,
        termination_reason,
        loglik_trace,
        final_delta,
        stopping_tolerance,
    })
}

/// Result of [`fit_nominal`]. Per item, `scores[i]` holds the `K-1` free
/// category scoring values `a_1..a_{K-1}` and `intercepts[i]` the `K-1` free
/// intercepts `c_1..c_{K-1}` (the baseline category is pinned `a_0 = c_0 = 0`).
pub struct NominalFit {
    pub scores: Vec<Vec<f64>>,
    pub intercepts: Vec<Vec<f64>>,
    pub loglik: f64,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub loglik_trace: Vec<f64>,
    pub final_delta: f64,
    pub stopping_tolerance: f64,
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
        let (f0, g) = nominal_item_neg_ll_grad(&params, nodes, counts, n_cat);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
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
        let (step, directional, max_step) =
            stabilized_newton_step(solve_small(hess, g.clone()), &g, grad_norm);
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _) = nominal_item_neg_ll_grad(&candidate, nodes, counts, n_cat);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
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
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=POLY_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={POLY_MAX_CAT}"));
    }
    if !(1..=POLY_MAX_ITER).contains(&max_iter) {
        return Err(format!("max_iter must be in 1..={POLY_MAX_ITER}"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and > 0".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    let z = n_cat - 1;
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) && y[p * n_items + i] >= n_cat {
                return Err("observed response categories must be < n_cat".into());
            }
        }
    }
    for i in 0..n_items {
        if !(0..n_persons).any(|p| is_obs(p, i)) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "q_theta")?;
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

    let mut it = 0;
    let mut converged = false;
    let mut termination_reason = "max_iter".to_owned();
    let mut final_delta = f64::INFINITY;
    let mut stopping_tolerance = f64::INFINITY;
    let mut loglik_trace = Vec::with_capacity(max_iter + 1);
    loop {
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
        let mut ll = 0.0;
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
        if let Some((delta, threshold)) =
            checked_em_delta(ll, loglik_trace.last().copied(), tol, it)?
        {
            final_delta = delta;
            stopping_tolerance = threshold;
            if final_delta <= stopping_tolerance {
                loglik_trace.push(ll);
                converged = true;
                termination_reason = "tolerance".to_owned();
                break;
            }
        }
        loglik_trace.push(ll);
        if it == max_iter {
            break;
        }
        for i in 0..n_items {
            params[i] = nominal_m_step(params[i].clone(), nodes, &counts[i], n_cat, 10);
        }
        it += 1;
    }

    let ll = *loglik_trace.last().expect("EM trace is never empty");
    let scores: Vec<Vec<f64>> = params.iter().map(|p| p[0..z].to_vec()).collect();
    let intercepts: Vec<Vec<f64>> = params.iter().map(|p| p[z..2 * z].to_vec()).collect();
    Ok(NominalFit {
        scores,
        intercepts,
        loglik: ll,
        n_iter: it,
        converged,
        termination_reason,
        loglik_trace,
        final_delta,
        stopping_tolerance,
    })
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
    let (theta_eap, _sd) = score_poly_eap(
        y, observed, n_persons, n_items, n_cat, slope, cat_params, model, q_theta,
    )?;
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
    Ok(PolyPersonFit {
        lz,
        lz_star,
        theta_eap,
        flagged,
    })
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
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
            let item = if adaptive {
                poly_cat_next_item(th, &administered, slope, cat_params, n_items, n_cat, model)
                    .expect("count < max_items.min(n_items) leaves an unadministered item")
            } else {
                let remaining: Vec<usize> = (0..n_items).filter(|&i| !administered[i]).collect();
                remaining[((u() * remaining.len() as f64) as usize).min(remaining.len() - 1)]
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
                &y,
                Some(&administered),
                1,
                n_items,
                n_cat,
                slope,
                cat_params,
                model,
                q_theta,
            )
            .expect(
                "validated item parameters and generated in-range categories must remain scoreable",
            );
            th = eap[0];
            se = sd[0];
        }
        theta_eap[s] = th;
        theta_sd[s] = se;
        n_used[s] = count;
    }
    Ok(PolyCatResult {
        theta_eap,
        theta_sd,
        n_used,
    })
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
    pub converged: bool,
    pub termination_reason: String,
    pub loglik_trace: Vec<f64>,
    pub final_delta: f64,
    pub stopping_tolerance: f64,
}

#[derive(Clone, Copy)]
enum MultigroupEmStatus {
    First,
    Continue { delta: f64, tolerance: f64 },
    Converged { delta: f64, tolerance: f64 },
    NonFinite,
    NonMonotone,
}

fn multigroup_em_status(current: f64, previous: Option<f64>, tol: f64) -> MultigroupEmStatus {
    if !current.is_finite() {
        return MultigroupEmStatus::NonFinite;
    }
    let Some(previous) = previous else {
        return MultigroupEmStatus::First;
    };
    let delta = current - previous;
    let tolerance = tol * (1.0 + previous.abs());
    let monotonic_tolerance = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    if delta < -monotonic_tolerance {
        MultigroupEmStatus::NonMonotone
    } else if delta <= tolerance {
        MultigroupEmStatus::Converged { delta, tolerance }
    } else {
        MultigroupEmStatus::Continue { delta, tolerance }
    }
}

#[allow(clippy::too_many_arguments)]
fn record_multigroup_em_status(
    status: MultigroupEmStatus,
    current: f64,
    trace: &mut Vec<f64>,
    converged: &mut bool,
    termination_reason: &mut String,
    final_delta: &mut f64,
    stopping_tolerance: &mut f64,
) -> bool {
    match status {
        MultigroupEmStatus::NonFinite => {
            *termination_reason = "non_finite".to_owned();
            true
        }
        MultigroupEmStatus::NonMonotone => {
            trace.push(current);
            *termination_reason = "non_monotone".to_owned();
            true
        }
        MultigroupEmStatus::Converged { delta, tolerance } => {
            trace.push(current);
            *final_delta = delta;
            *stopping_tolerance = tolerance;
            *converged = true;
            *termination_reason = "tolerance".to_owned();
            true
        }
        MultigroupEmStatus::Continue { delta, tolerance } => {
            trace.push(current);
            *final_delta = delta;
            *stopping_tolerance = tolerance;
            false
        }
        MultigroupEmStatus::First => {
            trace.push(current);
            false
        }
    }
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
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=POLY_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={POLY_MAX_CAT}"));
    }
    if !(1..=POLY_MAX_ITER).contains(&max_iter) {
        return Err(format!("max_iter must be in 1..={POLY_MAX_ITER}"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and > 0".into());
    }
    if n_groups < 2 {
        return Err("n_groups must be >= 2".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
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
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    validate_observed_categories(y, observed, n_cat)?;
    if let Some(j) = studied_item {
        if j >= n_items {
            return Err("studied_item out of range".into());
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "q_theta")?;
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

    let mut ll: f64;
    let mut it = 0;
    let mut converged = false;
    let mut termination_reason = "max_iter".to_owned();
    let mut final_delta = f64::INFINITY;
    let mut stopping_tolerance = f64::INFINITY;
    let mut loglik_trace = Vec::with_capacity(max_iter + 1);
    loop {
        // group-specific trait locations for the shared standard nodes
        let theta: Vec<Vec<f64>> = (0..n_groups)
            .map(|g| nodes.iter().map(|&x| mu[g] + sigma[g] * x).collect())
            .collect();
        // per-group cell log-probs
        let mut item_lp = vec![vec![vec![0.0_f64; qn * n_cat]; n_items]; n_groups];
        for g in 0..n_groups {
            for i in 0..n_items {
                let p_i = if Some(i) == studied_item {
                    &studied_params[g]
                } else {
                    &params[i]
                };
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
        let status = multigroup_em_status(ll, loglik_trace.last().copied(), tol);
        if record_multigroup_em_status(
            status,
            ll,
            &mut loglik_trace,
            &mut converged,
            &mut termination_reason,
            &mut final_delta,
            &mut stopping_tolerance,
        ) {
            break;
        }
        if it == max_iter {
            break;
        }
        // M-step, item parameters
        for i in 0..n_items {
            if Some(i) == studied_item {
                for g in 0..n_groups {
                    studied_params[g] = m_step_item(
                        studied_params[g].clone(),
                        &theta[g],
                        &counts[g][i],
                        model,
                        10,
                    );
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
                params[i] = m_step_item(
                    params[i].clone(),
                    &stacked_nodes,
                    &stacked_counts,
                    model,
                    10,
                );
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
    Ok(TwoGroupPolyFit {
        slope,
        cat_params,
        studied_slope,
        studied_cat,
        mu,
        sigma,
        loglik: ll,
        n_iter: it,
        converged,
        termination_reason,
        loglik_trace,
        final_delta,
        stopping_tolerance,
    })
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

fn validate_poly_dif_compact(fit: &TwoGroupPolyFit, max_iter: usize) -> Result<(), String> {
    if !fit.loglik.is_finite() {
        return Err(
            "compact multi-group fit did not reach a finite log-likelihood \
             (a group may have a rarely-used category; try model=\"gpcm\")"
                .into(),
        );
    }
    if !fit.converged {
        return Err(format!(
            "compact multi-group fit did not converge: reason={}, iteration={}/{}, \
             final_delta={:.6e}, tolerance={:.6e}",
            fit.termination_reason, fit.n_iter, max_iter, fit.final_delta, fit.stopping_tolerance
        ));
    }
    Ok(())
}

fn poly_dif_metrics(
    augmented: &TwoGroupPolyFit,
    compact_loglik: f64,
    df: usize,
) -> (f64, f64, f64) {
    if augmented.converged && augmented.loglik.is_finite() {
        let lr = (2.0 * (augmented.loglik - compact_loglik)).max(0.0);
        let group_locations: Vec<f64> = augmented
            .studied_cat
            .iter()
            .map(|categories| categories.iter().sum::<f64>() / categories.len().max(1) as f64)
            .collect();
        let high = group_locations
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let low = group_locations
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        (lr, crate::fitstats::chi2_sf(lr, df as f64), high - low)
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    }
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
///   In J. A. Harkness, M. Braun, B. Edwards, T. P. Johnson, L. E. Lyberg,
///   P. P. Mohler, B.-E. Pennell, & T. W. Smith (Eds.), *Survey methods in
///   multinational, multiregional, and multicultural contexts* (pp. 419–433).
///   Wiley.
///   https://doi.org/10.1002/9780470609927.ch22
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
    validate_poly_dif_compact(&con, max_iter)?;
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
            y,
            observed,
            group_id,
            n_groups,
            n_persons,
            n_items,
            n_cat,
            model,
            Some(j),
            q_theta,
            max_iter,
            tol,
        )
        .expect("the compact fit validated the shared inputs and the item index is in range");
        // If this item's augmented fit diverged, surface it as NaN rather than let
        // `.max(0.0)` mask a failed fit as LR=0 (a silent "no DIF" false negative).
        let (lr, p_value, effect_size) = poly_dif_metrics(&aug, con.loglik, df);
        rows.push(PolyDifRow {
            item: j,
            lr,
            df,
            p_value,
            flagged_bh: false,
            effect_size,
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
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    validate_observed_categories(y, observed, n_cat)?;
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
            let pi = if nobs > 0 {
                ge[step] as f64 / nobs as f64
            } else {
                0.0
            };
            let w = if pi <= 0.0 || pi >= 1.0 {
                0.0
            } else {
                (pi / (1.0 - pi)).ln()
            };
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
            let obs_cw: Vec<&[f64]> = (0..n_items)
                .filter(|&i| is_obs(p, i))
                .map(|i| cw[i].as_slice())
                .collect();
            let (pmax, pmin) = u3_min_max_conv(&obs_cw, m);
            let mut wsum = 0.0_f64;
            let mut nc = 0usize;
            for &i in (0..n_items)
                .filter(|&i| is_obs(p, i))
                .collect::<Vec<_>>()
                .iter()
            {
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
            let den = if nc == 0 || nc == total_steps {
                1.0
            } else {
                mx - mn
            };
            if den > 1e-9 {
                (mx - wsum) / den
            } else {
                f64::NAN
            }
        };
    }

    let flagged: Vec<bool> = match cutoff {
        Some(c) => u3poly.iter().map(|&v| v.is_finite() && v >= c).collect(),
        None => vec![false; n_persons],
    };
    Ok(U3PolyResult {
        u3poly,
        total_score,
        flagged,
    })
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
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    debug_assert!(
        !pool.is_empty(),
        "validated complete bootstrap samples have finite boundary U3 values"
    );
    pool.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let np = pool.len();
    let idx = (np as f64 - 1.0) * (1.0 - alpha);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let q = if lo == hi {
        pool[lo]
    } else {
        pool[lo] + (idx - lo as f64) * (pool[hi] - pool[lo])
    };
    Ok(q)
}

/// Fisher item information `I(theta) = sum_k (dP_k/dtheta)^2 / P_k` for one
/// polytomous item at trait value `theta`. GPCM reduces to `a^2 * Var_P(scores)`;
/// GRM to `a^2 * sum_k (v_k - v_{k+1})^2 / P_k` with `v_j = s_j(1-s_j)`,
/// `s_j = P(Y>=j)`. `cat_params` is this item's `K-1` category parameters.
///
/// # References
///
/// Muraki, E. (1993). Information functions of the generalized partial credit
/// model. *Applied Psychological Measurement, 17*(4), 351–363.
/// <https://doi.org/10.1177/014662169301700403>
///
/// Samejima, F. (1969). Estimation of latent ability using a response pattern
/// of graded scores. *Psychometrika, 34*(S1), 1–97.
/// <https://doi.org/10.1007/BF03372160>
pub fn poly_item_information(theta: f64, slope: f64, cat_params: &[f64], model: PolyModel) -> f64 {
    let a = slope;
    let base = a * theta;
    match model {
        PolyModel::Gpcm => {
            let k = cat_params.len() + 1;
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut intercepts = vec![0.0_f64; k];
            intercepts[1..].copy_from_slice(cat_params);
            let p: Vec<f64> = gpcm_logprobs(base, &scores, &intercepts)
                .iter()
                .map(|l| l.exp())
                .collect();
            let ebar: f64 = scores.iter().zip(&p).map(|(s, pp)| s * pp).sum();
            let var: f64 = scores
                .iter()
                .zip(&p)
                .map(|(s, pp)| pp * (s - ebar).powi(2))
                .sum();
            a * a * var
        }
        PolyModel::Grm => {
            let kk = cat_params.len() + 1; // K
            let p: Vec<f64> = grm_logprobs(base, cat_params)
                .iter()
                .map(|l| l.exp())
                .collect();
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
    if n_items < 1 {
        return Err("need at least one item".into());
    }
    if theta.is_empty() {
        return Err("theta must be non-empty".into());
    }
    let expected_cat_params =
        crate::checked_mul_usize(n_items, n_cat - 1, "n_items * (n_cat - 1) overflows usize")?;
    if slope.len() != n_items || cat_params.len() != expected_cat_params {
        return Err("slope/cat_params sizes inconsistent with n_items/n_cat".into());
    }
    if theta.iter().any(|value| !value.is_finite())
        || slope.iter().any(|value| !value.is_finite())
        || cat_params.iter().any(|value| !value.is_finite())
    {
        return Err("theta, slope, and cat_params must be finite".into());
    }
    let output_len = crate::checked_mul_usize(theta.len(), n_items, "output size overflows")?;
    let mut out = vec![0.0_f64; output_len];
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
///
/// # References
///
/// Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
/// a microcomputer environment. *Applied Psychological Measurement, 6*(4),
/// 431–444. https://doi.org/10.1177/014662168200600405
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
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    let n_params =
        crate::checked_mul_usize(n_items, n_cat - 1, "n_items * (n_cat - 1) overflows usize")?;
    if slope.len() != n_items || cat_params.len() != n_params {
        return Err("slope/cat_params sizes inconsistent with n_items/n_cat".into());
    }
    if slope.iter().any(|v| !v.is_finite()) || cat_params.iter().any(|v| !v.is_finite()) {
        return Err("slope and cat_params must be finite".into());
    }
    for (idx, &yc) in y.iter().enumerate() {
        if observed.map_or(true, |o| o[idx]) && yc >= n_cat {
            return Err(format!(
                "observed responses must be categories in 0..{}; y[{idx}]={yc}",
                n_cat - 1
            ));
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "q_theta")?;
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
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if slope.len() != n_items {
        return Err("slope must have length n_items".into());
    }
    let n_item_steps = crate::checked_mul_usize(
        n_items,
        n_cat - 1,
        "n_items * (n_cat - 1) overflows usize",
    )?;
    if cat_params.len() != n_item_steps {
        return Err("cat_params must have length n_items * (n_cat - 1)".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    validate_observed_categories(y, observed, n_cat)?;

    let z = n_cat - 1; // highest category score Z
    let f_max = n_items * z; // perfect summed score F
    let (nodes, weights) = crate::quadrature::require_gh_rule(q_theta, "q_theta")?;
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
    for i in 0..n_items {
        let rest: Vec<usize> = (0..n_items).filter(|&j| j != i).collect();
        let f_rest = poly_lw(&rest); // f*ᵢ, scores 0..(F-z)
        let rest_max = f_max - z;
        // observed/expected counts per (bucket, category); k in [1, F-1]
        let mut bo = vec![0.0_f64; n_buckets * n_cat];
        let mut be = vec![0.0_f64; n_buckets * n_cat];
        let mut bn = vec![0.0_f64; n_buckets];
        for k in 1..f_max {
            debug_assert!(
                denom[k] > 0.0,
                "quadrature weights and item probabilities are positive"
            );
            let bucket = k.clamp(z, f_max - z) - z;
            bn[bucket] += nk[k];
            for zc in 0..n_cat {
                bo[bucket * n_cat + zc] += obs[(i * (f_max + 1) + k) * n_cat + zc];
                if k >= zc && k - zc <= rest_max {
                    let kr = k - zc;
                    let num: f64 = (0..qn)
                        .map(|t| {
                            probs[(i * qn + t) * n_cat + zc] * f_rest[kr * qn + t] * weights[t]
                        })
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
#[path = "../../../tests/unit/poly_tests.rs"]
mod tests;
