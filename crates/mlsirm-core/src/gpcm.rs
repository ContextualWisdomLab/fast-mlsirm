//! Confirmatory MULTIDIMENSIONAL generalized partial credit model (Muraki, 1992), the
//! ADJACENT-CATEGORY-LOGIT ordered polytomous model. Completes the polytomous multidimensional
//! trio alongside [`crate::nominal`] (unordered softmax) and [`crate::grm`]
//! (cumulative-logit).
//!
//! Each item `i` has `n_cat` ORDERED categories, a SINGLE multidimensional discrimination vector
//! `a_i` (free on the confirmatory 0/1 `loading_pattern`, items x D), and `n_cat - 1` free step
//! intercepts `step_i`. With INTEGER category scores `k = 0..n_cat-1` and `base_i = sum_{d in S_i}
//! a_id theta_d`, the linear predictor is `psi_ik = k * base_i + step_ik` (`step_i0 = 0` pinned) and
//! `P(Y_i = k | theta) = softmax_k(psi_ik)` — exactly `gpcm_logprobs(base_i, [0..n_cat-1],
//! [0, step_i1, .., step_i,{n_cat-1}])`. `theta ~ MVN(0, I_D)`.
//!
//! GPCM is the `a_ikd = k a_id` integer-scoring restriction of the multidimensional nominal (Bock's
//! free per-category slopes), but with a strictly smaller, single-slope-vector parametrization the
//! free-slope [`crate::nominal::fit_nominal`] cannot express as a mode. Unlike the GRM the
//! softmax is finite for ANY step values, so there is NO ordering constraint on the steps. Reduces
//! to `poly::fit_poly_unidim(PolyModel::Gpcm)` at `D = 1` within optimizer tolerance and up to
//! reflection (NOT bit-exact: `fit_poly_unidim` forces `a > 0` via a `log a` parametrization, while
//! the confirmatory model uses an UNCONSTRAINED slope so reverse-keyed / negative cross-loadings are
//! representable).
//!
//! **Estimation.** Bock-Aitkin marginal MLE (EM) over the `D`-dim latent grid, reusing the MIRT node
//! machinery (`nodes::build_xi_nodes`, `node_rule` gh/qmc/mc, so `D <= 3` uses Gauss-Hermite and
//! `D = 4..6` uses the Halton quasi-Monte-Carlo EM of Jank, 2005). Fixed node set before the EM loop
//! (monotone; `theta ~ MVN(0, I)`). The per-item M-step is an FD-Hessian Newton over
//! `[a_{d0}..a_{d,L-1}, step_1..step_{M-1}]` (`L = |S_i|`), byte-for-byte the ascent of
//! `poly::m_step_item` (ridge = Hessian conditioning only), with the GPCM node gradient chained to
//! the multidimensional slope: `d/da_id = sum_node g_base theta_d`, `d/dstep_j = sum_node
//! g_intercepts[j]` where `(g_intercepts, g_base, g_scores) = gpcm_node_gradient(base,
//! [0..M-1], [0,step..], counts_node)`. The integer scores are FIXED, so `g_scores` is DROPPED —
//! this is what makes the model the GPCM (fixed scoring) rather than the nominal (free scoring). EM
//! uses the SIGNED monotonic-decrease stopping guard (a decrease errors, not `.abs()`).
//!
//! **Identification.** Unit trait variances fix the per-dimension slope scale; `E[theta] = 0` the
//! step level; the integer scores fix the ordering/spacing. A PURE single-dimension anchor item per
//! dimension pins the rotation. The per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d,
//! -theta_d)` leaves `base` — hence every `psi_ik = k base + step_ik` and category probability —
//! INVARIANT, so it is CANONICALIZED (single slope per item makes the anchor sign unambiguous):
//! dimension `d` is flipped so its largest-magnitude pure anchor loads positively, negating that
//! dimension's slopes AND `theta_d` but NOT the steps.
//!
//! # References (APA 7th ed.)
//!
//! Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied
//! Psychological Measurement, 16*(2), 159-176. https://doi.org/10.1177/014662169201600206
//!
//! Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
//! https://doi.org/10.1007/978-0-387-89976-3
//!
//! Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM.
//! *Computational Statistics & Data Analysis, 48*(4), 685-701. https://doi.org/10.1016/j.csda.2004.03.019

use crate::marginal::XiRuleKind;
use crate::nodes::{build_xi_nodes, XiRule};
use crate::poly::{gpcm_logprobs, gpcm_node_gradient, solve_small};
use crate::quadrature::SUPPORTED_Q;

const GP_MAX_NODES: usize = 200_000;
const GP_MAX_COUNT_CELLS: usize = 60_000_000;
const GP_MAX_DIMS: usize = 3;
const GP_MAX_DIMS_QMC: usize = 6;
const GP_MAX_CAT: usize = 64;

/// Configuration for [`fit_gpcm`].
#[derive(Clone, Copy, Debug)]
pub struct GpcmConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// Gauss-Hermite nodes per dimension (used only for `xi_rule = GaussHermite`).
    pub q: usize,
    /// Newton (FD-Hessian) ridge — Hessian CONDITIONING only, NOT a parameter prior.
    pub ridge: f64,
    /// Inner Newton iterations per item M-step.
    pub newton_iter: usize,
    pub xi_rule: XiRuleKind,
    pub xi_points: usize,
    pub xi_seed: u64,
}

impl Default for GpcmConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            q: 21,
            ridge: 1e-8,
            newton_iter: 10,
            xi_rule: XiRuleKind::GaussHermite,
            xi_points: 4000,
            xi_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Result of [`fit_gpcm`].
#[derive(Clone, Debug)]
pub struct GpcmResult {
    pub n_dims: usize,
    pub n_cat: usize,
    /// Item discrimination slopes `a_id`, row-major `n_items * n_dims` (exactly `0.0` off-pattern).
    /// Per-dimension reflection-canonicalized so each dimension's largest pure anchor is positive.
    pub slope: Vec<f64>,
    /// Category step intercepts `step_ik`, row-major `n_items * (n_cat - 1)` (`k = 1..n_cat-1`;
    /// UNORDERED — the GPCM softmax is valid for any values).
    pub step: Vec<f64>,
    /// Per-person trait EAP `E[theta_jd | X_j]`, row-major `n_persons * n_dims`.
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// `sum_i (|S_i| + (n_cat - 1))` free item parameters.
    pub n_parameters: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &GpcmConfig,
) -> Result<usize, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=GP_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={GP_MAX_CAT}; got {n_cat}"));
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if !cfg.ridge.is_finite() || cfg.ridge <= 0.0 {
        return Err("ridge must be finite and positive".into());
    }
    let n_nodes = match cfg.xi_rule {
        XiRuleKind::GaussHermite => {
            if !(1..=GP_MAX_DIMS).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={GP_MAX_DIMS} for the Gauss-Hermite grid; use \
                     node_rule qmc/mc for D up to {GP_MAX_DIMS_QMC}"
                ));
            }
            if !SUPPORTED_Q.contains(&cfg.q) {
                return Err(format!("q must be one of {SUPPORTED_Q:?}; got {}", cfg.q));
            }
            let mut n = 1usize;
            for _ in 0..n_dims {
                n = n
                    .checked_mul(cfg.q)
                    .filter(|&v| v <= GP_MAX_NODES)
                    .ok_or_else(|| format!("q^n_dims exceeds the node cap {GP_MAX_NODES}"))?;
            }
            n
        }
        XiRuleKind::Halton | XiRuleKind::MonteCarlo => {
            if !(1..=GP_MAX_DIMS_QMC).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={GP_MAX_DIMS_QMC} for the Halton/MonteCarlo rules"
                ));
            }
            if !(1..=GP_MAX_NODES).contains(&cfg.xi_points) {
                return Err(format!(
                    "xi_points must be in 1..={GP_MAX_NODES}; got {}",
                    cfg.xi_points
                ));
            }
            cfg.xi_points
        }
    };
    let cells = n_nodes
        .checked_mul(n_items)
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "node * item * category count-table size overflows usize".to_string())?;
    if cells > GP_MAX_COUNT_CELLS {
        return Err(format!(
            "count table {cells} cells exceeds the cap {GP_MAX_COUNT_CELLS}; reduce nodes/items/categories"
        ));
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    let n_l = n_items
        .checked_mul(n_dims)
        .ok_or_else(|| "n_items * n_dims overflows usize".to_string())?;
    if loading_pattern.len() != n_l {
        return Err("loading_pattern must have length n_items * n_dims".into());
    }
    for (idx, &v) in loading_pattern.iter().enumerate() {
        if v != 0 && v != 1 {
            return Err(format!("loading_pattern[{idx}] must be 0 or 1; got {v}"));
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) && y[p * n_items + i] >= n_cat {
                return Err("observed response categories must be < n_cat".into());
            }
        }
    }
    for i in 0..n_items {
        if !(0..n_dims).any(|d| loading_pattern[i * n_dims + d] != 0) {
            return Err(format!(
                "item {i} loads no dimension (all-zero loading_pattern row)"
            ));
        }
        let mut seen = vec![false; n_cat];
        let mut any = false;
        for p in 0..n_persons {
            if is_obs(p, i) {
                any = true;
                seen[y[p * n_items + i]] = true;
            }
        }
        if !any {
            return Err(format!("item {i} has no observed responses"));
        }
        if let Some(k) = (0..n_cat).find(|&k| !seen[k]) {
            return Err(format!(
                "item {i} category {k} is never observed (unidentified GPCM step); every declared \
                 category must be observed"
            ));
        }
    }
    for d in 0..n_dims {
        let has_pure = (0..n_items).any(|i| {
            loading_pattern[i * n_dims + d] != 0
                && (0..n_dims)
                    .filter(|&d2| loading_pattern[i * n_dims + d2] != 0)
                    .count()
                    == 1
        });
        if !has_pure {
            return Err(format!(
                "dimension {d} has no pure single-loading anchor item (needed for identification)"
            ));
        }
    }
    Ok(n_nodes)
}

/// Negative expected complete-data log-lik and its gradient for ONE item of the multidimensional
/// GPCM. `params = [a_{d0}..a_{d,L-1}, step_1..step_{M-1}]` (`L = dims.len()`, `M = n_cat`). `base =
/// sum_t a_t * theta_{dims[t]}`; the softmax uses FIXED integer scores `[0..M-1]` and intercepts
/// `[0, step_1, .., step_{M-1}]`. `d/da_t = sum_node g_base * theta_{dims[t]}`, `d/dstep_j = sum_node
/// g_intercepts[j]`; `g_scores` is DROPPED because the scores are fixed (this is the GPCM, not the
/// nominal).
fn gpcm_item_neg_ll_grad(
    params: &[f64],
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
) -> (f64, Vec<f64>) {
    let l = dims.len();
    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
    let mut intercepts = vec![0.0f64; n_cat];
    intercepts[1..].copy_from_slice(&params[l..]); // step_1..step_{M-1}; intercepts[0] = 0 pinned
    let mut ll = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];
    for (nd, cnt) in counts.iter().enumerate() {
        let mut base = 0.0f64;
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * nodes[nd * n_dims + d];
        }
        let lp = gpcm_logprobs(base, &scores, &intercepts);
        ll += cnt.iter().zip(&lp).map(|(r, l2)| r * l2).sum::<f64>();
        let (g_ic, g_base, _g_sc) = gpcm_node_gradient(base, &scores, &intercepts, cnt);
        for (t, &d) in dims.iter().enumerate() {
            grad[t] += g_base * nodes[nd * n_dims + d];
        }
        for (j, gj) in g_ic.iter().enumerate() {
            grad[l + j] += gj;
        }
    }
    (-ll, grad.iter().map(|v| -v).collect())
}

/// Newton M-step for one item — mirrors `poly::m_step_item` (FD Hessian, ridge conditioning,
/// backtracking line search), generalized to the multidimensional slope. The GPCM softmax is finite
/// for any parameters, so (unlike the GRM) the line search needs no ordered-boundary safeguard.
#[allow(clippy::too_many_arguments)]
fn gpcm_m_step(
    mut params: Vec<f64>,
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
    ridge: f64,
    n_newton: usize,
) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g) = gpcm_item_neg_ll_grad(&params, dims, nodes, n_dims, counts, n_cat);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = gpcm_item_neg_ll_grad(&pj, dims, nodes, n_dims, counts, n_cat);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
            }
            hess[r][r] += ridge;
        }
        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }
        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
            max_step = 2.0;
        }
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _) =
                gpcm_item_neg_ll_grad(&candidate, dims, nodes, n_dims, counts, n_cat);
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

/// Fit the confirmatory MULTIDIMENSIONAL generalized partial credit model (Muraki, 1992) by
/// Bock-Aitkin marginal MLE. See the module docs for the model, estimation, and identification.
/// `y`/`observed` are row-major `n_persons * n_items` (`y` ordered categories `0..n_cat-1`, missing
/// cells dropped MAR); `loading_pattern` is row-major `n_items * n_dims` in `{0,1}`. Returns `Err`
/// on malformed / rotationally-underidentified / unobserved-category input.
#[allow(clippy::too_many_arguments)]
pub fn fit_gpcm(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &GpcmConfig,
) -> Result<GpcmResult, String> {
    let _n_nodes = validate(
        y,
        observed,
        loading_pattern,
        n_persons,
        n_items,
        n_dims,
        n_cat,
        cfg,
    )?;

    let (nodes, logw) = match cfg.xi_rule {
        XiRuleKind::GaussHermite => {
            let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: cfg.q }, n_dims)?;
            (xn.grid, xn.logw)
        }
        XiRuleKind::Halton => {
            let xn = build_xi_nodes(
                XiRule::Halton {
                    n: cfg.xi_points,
                    shift_seed: cfg.xi_seed,
                },
                n_dims,
            )?;
            (xn.grid, xn.logw)
        }
        XiRuleKind::MonteCarlo => {
            let xn = build_xi_nodes(
                XiRule::MonteCarlo {
                    n: cfg.xi_points,
                    seed: cfg.xi_seed.max(1),
                },
                n_dims,
            )?;
            (xn.grid, xn.logw)
        }
    };
    let qn = logw.len();
    let m1 = n_cat - 1; // step count
    let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();

    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] != 0)
                .collect()
        })
        .collect();
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    // Init: slope = 1.0 on the item's FIRST loaded dim (0 elsewhere); step_k = log(freq_k / freq_0)
    // (the GPCM baseline log-odds, NON-cumulative) — exactly fit_poly_unidim's GPCM init.
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let l = dims_of[i].len();
        let mut p = vec![0.0f64; l + m1];
        p[0] = 1.0;
        let mut freq = vec![1e-3f64; n_cat];
        for pp in 0..n_persons {
            if is_obs(pp, i) {
                freq[y[pp * n_items + i]] += 1.0;
            }
        }
        let tot: f64 = freq.iter().sum();
        for f in freq.iter_mut() {
            *f /= tot;
        }
        for k in 1..n_cat {
            p[l + (k - 1)] = (freq[k] / freq[0]).ln();
        }
        params.push(p);
    }

    let mut loglik_trace: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;
    let mut theta = vec![0.0f64; n_persons * n_dims];
    let mut log_node = vec![0.0f64; qn];

    let fill_lp = |params: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut all_lp: Vec<Vec<f64>> = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let l = dims_of[i].len();
            let mut intercepts = vec![0.0f64; n_cat];
            intercepts[1..].copy_from_slice(&params[i][l..]);
            let mut lp_i = vec![0.0f64; qn * n_cat];
            for nd in 0..qn {
                let mut base = 0.0f64;
                for (t, &d) in dims_of[i].iter().enumerate() {
                    base += params[i][t] * nodes[nd * n_dims + d];
                }
                let lp = gpcm_logprobs(base, &scores, &intercepts);
                lp_i[nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
            all_lp.push(lp_i);
        }
        all_lp
    };

    loop {
        let all_lp = fill_lp(&params);
        let mut counts = vec![vec![vec![0.0f64; n_cat]; qn]; n_items];
        let mut ll = 0.0f64;
        for p in 0..n_persons {
            log_node.copy_from_slice(&logw);
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                let lp = &all_lp[i];
                for nd in 0..qn {
                    log_node[nd] += lp[nd * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in log_node.iter() {
                denom += (v - mx).exp();
            }
            ll += mx + denom.ln();
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                for nd in 0..qn {
                    counts[i][nd][yc] += (log_node[nd] - mx).exp() / denom;
                }
            }
        }
        if !ll.is_finite() {
            return Err(format!(
                "non-finite observed-data log-likelihood at iteration {n_iter}"
            ));
        }
        loglik_trace.push(ll);

        if loglik_trace.len() >= 2 {
            let prev = loglik_trace[loglik_trace.len() - 2];
            final_loglik_change = ll - prev;
            let stop_tol = cfg.tol * (1.0 + prev.abs());
            let mono_tol = 32.0 * f64::EPSILON * (1.0 + prev.abs());
            if final_loglik_change < -mono_tol {
                return Err(format!(
                    "EM observed-data log-likelihood decreased at iteration {n_iter}: \
                     delta={final_loglik_change:.6e}"
                ));
            }
            if final_loglik_change <= stop_tol {
                converged = true;
                termination_reason = "tolerance_met".to_string();
                break;
            }
        }
        if n_iter == cfg.max_iter {
            break;
        }

        for i in 0..n_items {
            params[i] = gpcm_m_step(
                params[i].clone(),
                &dims_of[i],
                &nodes,
                n_dims,
                &counts[i],
                n_cat,
                cfg.ridge,
                cfg.newton_iter,
            );
        }
        n_iter += 1;
    }

    // Final EAP pass.
    {
        let all_lp = fill_lp(&params);
        for p in 0..n_persons {
            log_node.copy_from_slice(&logw);
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                let lp = &all_lp[i];
                for nd in 0..qn {
                    log_node[nd] += lp[nd * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in log_node.iter() {
                denom += (v - mx).exp();
            }
            for nd in 0..qn {
                let post = (log_node[nd] - mx).exp() / denom;
                for d in 0..n_dims {
                    theta[p * n_dims + d] += post * nodes[nd * n_dims + d];
                }
            }
        }
    }

    // Assemble dense slope (n_items * n_dims) + steps (n_items * (n_cat-1)).
    let mut slope = vec![0.0f64; n_items * n_dims];
    let mut step = vec![0.0f64; n_items * m1];
    let mut n_parameters = 0usize;
    for i in 0..n_items {
        let l = dims_of[i].len();
        n_parameters += l + m1;
        for (t, &d) in dims_of[i].iter().enumerate() {
            slope[i * n_dims + d] = params[i][t];
        }
        step[i * m1..(i + 1) * m1].copy_from_slice(&params[i][l..]);
    }

    // Per-dimension reflection canonicalization: flip dimension d (its slopes on every item AND
    // theta_d) so its largest-|slope| PURE anchor loads positively. base — hence psi and every step
    // — is invariant under the joint flip, so steps are NOT touched.
    for d in 0..n_dims {
        let mut anchor: Option<usize> = None;
        let mut best = 0.0f64;
        for i in 0..n_items {
            let is_pure = dims_of[i].len() == 1 && dims_of[i][0] == d;
            if is_pure && slope[i * n_dims + d].abs() > best {
                best = slope[i * n_dims + d].abs();
                anchor = Some(i);
            }
        }
        if let Some(ai) = anchor {
            if slope[ai * n_dims + d] < 0.0 {
                for i in 0..n_items {
                    slope[i * n_dims + d] = -slope[i * n_dims + d];
                }
                for p in 0..n_persons {
                    theta[p * n_dims + d] = -theta[p * n_dims + d];
                }
            }
        }
    }

    Ok(GpcmResult {
        n_dims,
        n_cat,
        slope,
        step,
        theta,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        n_parameters,
    })
}

#[cfg(test)]
mod tests {
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

    /// Simulate multidimensional GPCM responses: base = sum_d slope[i,d]*theta_d, then
    /// softmax_k(k*base + step_ik).
    fn simulate(
        slope: &[f64],
        step: &[f64],
        theta: &[f64],
        n: usize,
        n_items: usize,
        n_dims: usize,
        n_cat: usize,
        rng: &mut Lcg,
    ) -> Vec<usize> {
        let m1 = n_cat - 1;
        let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = 0.0f64;
                for d in 0..n_dims {
                    base += slope[i * n_dims + d] * theta[p * n_dims + d];
                }
                let mut intercepts = vec![0.0f64; n_cat];
                intercepts[1..].copy_from_slice(&step[i * m1..(i + 1) * m1]);
                let lp = gpcm_logprobs(base, &scores, &intercepts);
                let u = rng.next_f64();
                let mut acc = 0.0;
                let mut cat = n_cat - 1;
                for (k, l) in lp.iter().enumerate() {
                    acc += l.exp();
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

    /// D = 1 WITHIN-TOL reduction to fit_poly_unidim(GPCM). All-POSITIVE true slopes (fit_poly_unidim
    /// forces a>0 via log_a); both reach the same MLE up to optimizer tolerance and the positive
    /// reflection. NOT bit-exact.
    #[test]
    fn gpcm_reduces_to_poly_gpcm_at_d1() {
        let (n, n_items, n_cat) = (2000usize, 6usize, 4usize);
        let m1 = n_cat - 1;
        let mut rng = Lcg(717717);
        let mut slope = vec![0.0f64; n_items];
        let mut step = vec![0.0f64; n_items * m1];
        for i in 0..n_items {
            slope[i] = 0.8 + 0.2 * i as f64; // POSITIVE
                                             // UNORDERED steps (GPCM has no ordering constraint)
            step[i * m1] = 0.6 - 0.1 * i as f64;
            step[i * m1 + 1] = -0.4 + 0.05 * i as f64;
            step[i * m1 + 2] = 0.3 - 0.08 * i as f64;
        }
        let theta: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let y = simulate(&slope, &step, &theta, n, n_items, 1, n_cat, &mut rng);
        let pattern = vec![1u8; n_items];
        let cfg = GpcmConfig {
            q: 21,
            ..GpcmConfig::default()
        };
        let mm = fit_gpcm(&y, None, &pattern, n, n_items, 1, n_cat, &cfg).unwrap();
        let pf =
            fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Gpcm, 21, 500, 1e-6).unwrap();
        for i in 0..n_items {
            assert!(
                (mm.slope[i] - pf.slope[i]).abs() < 0.05,
                "slope[{i}] {} vs {}",
                mm.slope[i],
                pf.slope[i]
            );
            for j in 0..m1 {
                let d = (mm.step[i * m1 + j] - pf.cat_params[i][j]).abs();
                assert!(d < 0.06, "step[{i}][{j}] diff {d}");
            }
        }
        assert!(
            (*mm.loglik_trace.last().unwrap() - pf.loglik).abs() < 0.5,
            "loglik"
        );
        assert_eq!(mm.n_parameters, n_items * (1 + m1));
    }

    /// Deterministic FD GRADIENT anchor at D=2 (GH) AND D=4 (Halton, NON-IDENTITY dims [0,2,3]) with
    /// M=4 categories, NON-MONOTONE steps (locks in that the GPCM softmax is finite for any steps —
    /// no accidental ordering guard), and distinct random per-category counts (so a slope<->step slot
    /// transposition is detected). The M-step uses an FD Hessian, so pin the GRADIENT.
    #[test]
    fn gpcm_gradient_matches_finite_difference() {
        let n_cat = 4usize;
        for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter()
        {
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
            let mut rng = Lcg(1414 + n_dims as u64);
            let counts: Vec<Vec<f64>> = (0..n_nodes)
                .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 3.0).collect())
                .collect();
            let mut params = vec![0.0f64; l + (n_cat - 1)];
            for t in 0..l {
                params[t] = 0.4 + 0.3 * t as f64 - if t == 1 { 0.9 } else { 0.0 };
            }
            // NON-MONOTONE steps
            let steps = [0.8f64, -0.3, 1.1];
            for j in 0..(n_cat - 1) {
                params[l + j] = steps[j];
            }
            let (_f0, grad) = gpcm_item_neg_ll_grad(&params, dims, &nodes, n_dims, &counts, n_cat);
            let eps = 1e-6;
            for j in 0..params.len() {
                let mut pp = params.clone();
                pp[j] += eps;
                let (fp, _) = gpcm_item_neg_ll_grad(&pp, dims, &nodes, n_dims, &counts, n_cat);
                let mut pm = params.clone();
                pm[j] -= eps;
                let (fm, _) = gpcm_item_neg_ll_grad(&pm, dims, &nodes, n_dims, &counts, n_cat);
                let fd = (fp - fm) / (2.0 * eps);
                assert!(
                    (grad[j] - fd).abs() < 1e-4,
                    "grad[{j}] {} vs fd {fd} (D={n_dims})",
                    grad[j]
                );
            }
        }
    }

    /// Deterministic OBJECTIVE-VALUE dims-map pin at D=4 (Halton, dims=[0,2,3]). Computes base with the
    /// CORRECT dim map and gpcm_logprobs with LITERAL integer scores [0,1,2,3] and a literal 0.0
    /// baseline step, then matches the estimator's internal neg-loglik to < 1e-9. The FD anchor is
    /// map-invariant AND scores-invariant; this is the only guard against a wrong-node-column, a
    /// wrong-scores (e.g. [1,2,3,4]), or a dropped-baseline-step mutation on the QMC path.
    #[test]
    fn gpcm_objective_dims_map_pinned_at_d4() {
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
        let mut rng = Lcg(27182);
        let counts: Vec<Vec<f64>> = (0..n_nodes)
            .map(|_| (0..n_cat).map(|_| 0.1 + rng.next_f64() * 2.0).collect())
            .collect();
        let a = [0.9f64, -0.6, 0.7];
        let step = [0.5f64, -0.8, 0.2]; // non-monotone
        let mut params = vec![0.0f64; l + (n_cat - 1)];
        params[..l].copy_from_slice(&a);
        params[l..].copy_from_slice(&step);
        let (neg_ll, _g) = gpcm_item_neg_ll_grad(&params, &dims, &nodes, n_dims, &counts, n_cat);
        let mut hand = 0.0f64;
        for (nd, cnt) in counts.iter().enumerate() {
            let base = a[0] * nodes[nd * n_dims + 0]
                + a[1] * nodes[nd * n_dims + 2]
                + a[2] * nodes[nd * n_dims + 3];
            let lp = gpcm_logprobs(
                base,
                &[0.0, 1.0, 2.0, 3.0],
                &[0.0, step[0], step[1], step[2]],
            );
            hand += cnt.iter().zip(&lp).map(|(r, l2)| r * l2).sum::<f64>();
        }
        assert!(
            (neg_ll - (-hand)).abs() < 1e-9,
            "objective dims/scores map mismatch: {neg_ll} vs {}",
            -hand
        );
    }

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
        slope[4 * n_dims + 0] = -1.0; // negative cross-loader (dim0 anchor item 0 positive)
        slope[4 * n_dims + 1] = 0.9;
        let mut step = vec![0.0f64; n_items * m1];
        for i in 0..n_items {
            step[i * m1] = 0.5 + 0.05 * i as f64; // non-monotone across k
            if m1 > 1 {
                step[i * m1 + 1] = -0.4 + 0.03 * i as f64;
            }
        }
        (pattern, n_items, slope, step)
    }

    /// D = 2 recovery on GH nodes: pure anchors + a NEGATIVE cross-loader on dim0 (positively
    /// anchored). Asserts slope recovery, STEP recovery (numeric — GPCM steps are unordered, no
    /// ordering canary), per-dim EAP, finite steps, EM monotone.
    #[test]
    fn gpcm_recovers_d2_with_negative_cross_loader() {
        let (n_dims, n_cat) = (2usize, 3usize);
        let (pattern, n_items, slope, step) = design_d2(n_cat);
        let n = 6000usize;
        let mut rng = Lcg(3535);
        let mut theta = vec![0.0f64; n * n_dims];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GpcmConfig {
            q: 21,
            ..GpcmConfig::default()
        };
        let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
        assert!(res.converged);
        for i in 0..n_items {
            for d in 0..n_dims {
                if pattern[i * n_dims + d] == 0 {
                    assert_eq!(res.slope[i * n_dims + d], 0.0, "off-pattern zero");
                }
            }
        }
        assert!(res.step.iter().all(|v| v.is_finite()), "finite steps");
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
        assert!(
            rmse(&res.step, &step) < 0.16,
            "step RMSE {}",
            rmse(&res.step, &step)
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

    /// The reflection canonicalization FIRES — and is WITNESSED by the raw EM mode landing on the
    /// wrong side, so dropping the flip flips every assertion below (verified by mutation: disabling
    /// the canonicalization block makes this test fail on all three sign checks).
    ///
    /// The witness depends on which mirror mode raw EM converges to. Init is `+1.0` on each item's
    /// first loaded dim (see `fit_gpcm`), so the dim0 axis is oriented by its STRONGEST-|slope|
    /// loader. Here that is a positively-keyed CROSS-loader (`item1`, true `+1.7`), NOT the pure
    /// anchor: raw EM therefore orients theta_0 to the +item1 axis (its true orientation), and the
    /// WEAK reverse-keyed pure anchor (`item0`, true `-0.7`) converges NATIVELY NEGATIVE. Because the
    /// pure anchor is the sole pure dim0 item, canonicalization must FLIP dim0 to make it positive —
    /// negating item0 to `+0.7`, item1's dim0 slope to `-1.7`, and theta_0 to `-theta_0`. If the flip
    /// is removed, item0 stays `-0.7` (anchor check fails), item1 stays `+1.7` (co-loader check
    /// fails), and theta_0 stays positively correlated with truth (theta check fails). The STEPS are
    /// invariant under the joint (slope, theta) flip (GPCM steps are unordered — no ordering canary —
    /// so a reflection bug that also negated the steps could only be caught by this value check).
    #[test]
    fn gpcm_reflection_fires_on_negative_anchor() {
        let (n_dims, n_cat) = (2usize, 3usize);
        let m1 = n_cat - 1;
        // item0: WEAK reverse-keyed SOLE pure anchor on dim0 -> converges raw-NEGATIVE.
        // item1: STRONG positively-keyed cross-loader on dim0 -> dominates the dim0 orientation, so
        //        raw EM does NOT land the anchor in the canonical (positive) mode on its own.
        let pattern: Vec<u8> = vec![1, 0, 1, 1, 0, 1, 0, 1];
        let n_items = 4usize;
        let mut slope = vec![0.0f64; n_items * n_dims];
        slope[0 * n_dims + 0] = -0.7; // weak reverse-keyed SOLE pure anchor on dim0
        slope[1 * n_dims + 0] = 1.7; // strong cross-loader, positively keyed on dim0 (sets the axis)
        slope[1 * n_dims + 1] = 0.6;
        slope[2 * n_dims + 1] = 1.2; // pure anchor on dim1 (positively keyed -> dim1 not flipped)
        slope[3 * n_dims + 1] = 1.0;
        // non-monotone steps (unordered) so a step-negating reflection bug is caught by the RMSE check
        let mut step = vec![0.0f64; n_items * m1];
        for i in 0..n_items {
            step[i * m1] = 0.6;
            step[i * m1 + 1] = -0.5;
        }
        let n = 6000usize;
        let mut rng = Lcg(6262);
        let mut theta = vec![0.0f64; n * n_dims];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GpcmConfig {
            q: 21,
            ..GpcmConfig::default()
        };
        let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
        // canon FIRED: anchor flipped +, strong co-loader flipped -, theta_0 flipped (all three would
        // fail with the flip removed, because raw EM lands the anchor negative / co-loader positive).
        assert!(
            res.slope[0 * n_dims + 0] > 0.3,
            "reflected anchor positive: {}",
            res.slope[0 * n_dims + 0]
        );
        assert!(
            res.slope[1 * n_dims + 0] < -0.5,
            "co-loader flipped negative: {}",
            res.slope[1 * n_dims + 0]
        );
        // steps UNCHANGED by the reflection (recovered close to truth) — the unordered-step analogue
        // of the GRM's ordering canary: a step-negating reflection bug would blow this up.
        assert!(
            rmse(&res.step, &step) < 0.15,
            "steps preserved: RMSE {}",
            rmse(&res.step, &step)
        );
        // flipped dim0: EAP theta_0 correlates NEGATIVELY with truth; unflipped dim1 positive.
        let th0: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 0]).collect();
        let tt0: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 0]).collect();
        let th1: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 1]).collect();
        let tt1: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 1]).collect();
        assert!(
            corr(&th0, &tt0) < -0.5,
            "flipped-dim theta corr negative: {}",
            corr(&th0, &tt0)
        );
        assert!(
            corr(&th1, &tt1) > 0.5,
            "unflipped-dim theta corr positive: {}",
            corr(&th1, &tt1)
        );
    }

    /// Structural invariants + validation guards (constructed non-vacuously — the intended guard is
    /// the failing branch).
    #[test]
    fn gpcm_validates_and_structural_invariants() {
        let (n_dims, n_cat) = (2usize, 3usize);
        let (pattern, n_items, slope, step) = design_d2(n_cat);
        let n = 500usize;
        let mut rng = Lcg(88);
        let mut theta = vec![0.0f64; n * n_dims];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GpcmConfig {
            q: 15,
            max_iter: 25,
            ..GpcmConfig::default()
        };
        let res = fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
        assert_eq!(res.n_parameters, 4 * (1 + 2) + (2 + 2));
        let lp = gpcm_logprobs(0.4, &[0.0, 1.0, 2.0], &[0.0, 0.6, -0.4]);
        let s: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((s - 1.0).abs() < 1e-12);
        // GH D=4 rejected (y4 observes every category so the D-bound is the sole reason)
        let gh4 = GpcmConfig::default();
        let pat4: Vec<u8> = (0..4)
            .flat_map(|d| (0..4).map(move |k| (k == d) as u8))
            .collect();
        let y4: Vec<usize> = (0..n * 4).map(|idx| idx % n_cat).collect();
        assert!(
            fit_gpcm(&y4, None, &pat4, n, 4, 4, n_cat, &gh4).is_err(),
            "GH D=4 rejected"
        );
        // no pure anchor (3-item all-both pattern with the full 3-item y so the anchor guard fires)
        let no_anchor: Vec<u8> = vec![1, 1, 1, 1, 1, 1];
        assert!(
            fit_gpcm(&y, None, &no_anchor, n, n_items, n_dims, n_cat, &cfg).is_err(),
            "no pure anchor rejected"
        );
        let mut ybad = y.clone();
        ybad[0] = n_cat;
        assert!(
            fit_gpcm(&ybad, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
            "bad category rejected"
        );
        let mut ygap = y.clone();
        for p in 0..n {
            if ygap[p * n_items + 0] == 1 {
                ygap[p * n_items + 0] = 0;
            }
        }
        assert!(
            fit_gpcm(&ygap, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(),
            "unobserved category rejected"
        );
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recover the multidimensional GPCM at D=2 and D=3
    /// under normal AND per-dim-standardized right-skew traits. Per-rep monotone-EM + STEP finiteness
    /// canaries (a diverging step is GPCM's characteristic failure mode).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_gpcm_recovery_500() {
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
                slope[(2 * d) * n_dims + d] = 1.3;
                slope[(2 * d + 1) * n_dims + d] = 1.0;
            }
            for d in 0..n_dims {
                let ci = 2 * n_dims + d;
                slope[ci * n_dims + d] = 1.0;
                slope[ci * n_dims + (d + 1) % n_dims] = if d % 2 == 0 { 0.7 } else { -0.7 };
            }
            let mut step = vec![0.0f64; n_items * m1];
            for i in 0..n_items {
                step[i * m1] = 0.6 + 0.03 * i as f64;
                step[i * m1 + 1] = -0.5 + 0.02 * i as f64;
            }
            for &skew in [false, true].iter() {
                let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
                let (mut snum, mut sden) = (0.0f64, 0.0f64);
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
                    let y = simulate(&slope, &step, &theta, n, n_items, n_dims, n_cat, &mut rng);
                    let cfg = GpcmConfig {
                        q,
                        ..GpcmConfig::default()
                    };
                    let res =
                        fit_gpcm(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
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
                    assert!(
                        res.step.iter().all(|v| v.is_finite()),
                        "finite step (rep {rep})"
                    );
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
                            let e = res.step[i * m1 + j] - step[i * m1 + j];
                            snum += e * e;
                            sden += 1.0;
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
                let srmse = (snum / sden).sqrt();
                let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
                println!(
                    "[gpcm-mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                     loadRMSE={lrmse:.4} loadBias={lb:.4} stepRMSE={srmse:.4} thetaCorr={tc:.3}"
                );
                assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
                if skew {
                    assert!(lrmse < 0.24, "skew load RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.55, "skew theta corr {tc} (D={n_dims})");
                } else {
                    assert!(lb.abs() < 0.06, "load bias {lb} (D={n_dims})");
                    assert!(lrmse < 0.16, "load RMSE {lrmse} (D={n_dims})");
                    assert!(srmse < 0.16, "step RMSE {srmse} (D={n_dims})");
                    assert!(tc > 0.6, "theta corr {tc} (D={n_dims})");
                }
            }
        }
    }
}
