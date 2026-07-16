//! Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995),
//! the ORDERED-category counterpart of [`crate::nominal_mirt::fit_nominal_mirt`] and the polytomous
//! generalization of the compensatory MIRT ([`crate::mirt::fit_compensatory_mirt`]).
//!
//! Each item `i` has `n_cat` ORDERED categories, a SINGLE multidimensional discrimination vector
//! `a_i` (free on the confirmatory 0/1 `loading_pattern`, items x D), and `n_cat - 1` ORDERED
//! category boundary intercepts `beta_i`. The cumulative boundaries are
//! `P(Y_i >= k | theta) = sigmoid(sum_{d in S_i} a_id theta_d + beta_i,{k-1})` (`k = 1..n_cat-1`),
//! and the category probability is the adjacent difference — exactly
//! `grm_logprobs(base, beta_i)` with `base = sum_{d in S_i} a_id theta_d`. `theta ~ MVN(0, I_D)`.
//! Valid probabilities require the boundaries to be STRICTLY DECREASING
//! (`beta_i,0 > beta_i,1 > ... > beta_i,{M-2}`).
//!
//! At `D = 1` with `S_i = {0}` this is `poly::fit_poly_unidim(PolyModel::Grm)` — but WITHIN optimizer
//! tolerance and up to a reflection, not bit-exact: `fit_poly_unidim` forces `a > 0` via a `log a`
//! parametrization, whereas the confirmatory multidimensional model uses an UNCONSTRAINED slope so
//! that reverse-keyed / negative cross-loadings are representable (the compensatory-MIRT choice).
//!
//! **Estimation.** Bock-Aitkin marginal MLE (EM) over the `D`-dim latent grid, reusing the MIRT node
//! machinery (`nodes::build_xi_nodes`, `node_rule` gh/qmc/mc, so `D <= 3` uses Gauss-Hermite and
//! `D = 4..6` uses the Halton quasi-Monte-Carlo EM of Jank, 2005). The node set is built ONCE before
//! the EM loop; because `theta ~ MVN(0, I)` never reparametrizes the nodes, EM is monotone. The
//! per-item M-step is a finite-difference-Hessian Newton over `[a_{d0}..a_{d,L-1}, beta_1..beta_{M-1}]`
//! (`L = |S_i|`), byte-for-byte the ascent of `poly::m_step_item` (ridge = Hessian conditioning only,
//! NOT a prior), with the GRM node gradient chained to the multidimensional slope:
//! `d/da_id = sum_node g_base theta_d`, `d/dbeta_j = sum_node g_thr[j]` where
//! `(g_base, g_thr) = grm_node_gradient(base, beta, counts_node)`. The backtracking line search
//! REJECTS any step that makes the objective non-finite, which is exactly how the ordered-threshold
//! constraint is maintained WITHOUT an explicit reparametrization: every adjacent boundary pair
//! `(beta_{k-1}, beta_k)` is the middle category `k`'s only source, whose log-probability is `NaN`
//! (via `ln(-expm1(beta_k - beta_{k-1}))`) the instant `beta_{k-1} <= beta_k`, and `0 * NaN = NaN`
//! so a zero expected count cannot mask it — adjacency + transitivity therefore make a finite
//! objective imply a fully ordered `beta`.
//!
//! **Identification.** Unit trait variances fix the per-dimension slope scale; ordered thresholds fix
//! the category direction; a PURE single-dimension anchor item per dimension pins the rotation to the
//! coordinate axes (one slope per item, so identical to the compensatory MIRT). The per-dimension
//! reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves `base` — and therefore every threshold
//! and category probability — INVARIANT, so it is CANONICALIZED (unlike the nominal, whose
//! per-category slopes make the anchor sign ambiguous): dimension `d` is flipped so its
//! largest-magnitude pure anchor loads positively, negating that dimension's slopes AND the person
//! trait `theta_d`, but NOT the thresholds.
//!
//! # References (APA 7th ed.)
//!
//! Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores.
//! *Psychometrika Monograph Supplement, 34*(4, Pt. 2). https://doi.org/10.1007/BF03372160
//!
//! Muraki, E., & Carlson, J. E. (1995). Full-information factor analysis for polytomous item
//! responses. *Applied Psychological Measurement, 19*(1), 73-90.
//! https://doi.org/10.1177/014662169501900109
//!
//! Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
//! https://doi.org/10.1007/978-0-387-89976-3
//!
//! Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM.
//! *Computational Statistics & Data Analysis, 48*(4), 685-701. https://doi.org/10.1016/j.csda.2004.03.019

use crate::marginal::XiRuleKind;
use crate::nodes::{build_xi_nodes, XiRule};
use crate::poly::{grm_logprobs, grm_node_gradient, solve_small};
use crate::quadrature::SUPPORTED_Q;

const GM_MAX_NODES: usize = 200_000;
const GM_MAX_COUNT_CELLS: usize = 60_000_000;
const GM_MAX_DIMS: usize = 3;
const GM_MAX_DIMS_QMC: usize = 6;
const GM_MAX_CAT: usize = 64;

/// Configuration for [`fit_grm_mirt`].
#[derive(Clone, Copy, Debug)]
pub struct GrmMirtConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// Gauss-Hermite nodes per dimension (used only for `xi_rule = GaussHermite`).
    pub q: usize,
    /// Newton (FD-Hessian) ridge — Hessian CONDITIONING only, NOT a parameter prior (matches
    /// `poly::m_step_item`'s `1e-8`).
    pub ridge: f64,
    /// Inner Newton iterations per item M-step.
    pub newton_iter: usize,
    pub xi_rule: XiRuleKind,
    pub xi_points: usize,
    pub xi_seed: u64,
}

impl Default for GrmMirtConfig {
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

/// Result of [`fit_grm_mirt`].
#[derive(Clone, Debug)]
pub struct GrmMirtResult {
    pub n_dims: usize,
    pub n_cat: usize,
    /// Item discrimination slopes `a_id`, row-major `n_items * n_dims` (exactly `0.0` off-pattern).
    /// Per-dimension reflection-canonicalized so each dimension's largest pure anchor is positive.
    pub slope: Vec<f64>,
    /// Ordered boundary intercepts `beta_ik`, row-major `n_items * (n_cat - 1)` (strictly
    /// decreasing within each item).
    pub threshold: Vec<f64>,
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
    cfg: &GrmMirtConfig,
) -> Result<usize, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=GM_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={GM_MAX_CAT}; got {n_cat}"));
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
            if !(1..=GM_MAX_DIMS).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={GM_MAX_DIMS} for the Gauss-Hermite grid; use \
                     node_rule qmc/mc for D up to {GM_MAX_DIMS_QMC}"
                ));
            }
            if !SUPPORTED_Q.contains(&cfg.q) {
                return Err(format!("q must be one of {SUPPORTED_Q:?}; got {}", cfg.q));
            }
            let mut n = 1usize;
            for _ in 0..n_dims {
                n = n
                    .checked_mul(cfg.q)
                    .filter(|&v| v <= GM_MAX_NODES)
                    .ok_or_else(|| format!("q^n_dims exceeds the node cap {GM_MAX_NODES}"))?;
            }
            n
        }
        XiRuleKind::Halton | XiRuleKind::MonteCarlo => {
            if !(1..=GM_MAX_DIMS_QMC).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={GM_MAX_DIMS_QMC} for the Halton/MonteCarlo rules"
                ));
            }
            if !(1..=GM_MAX_NODES).contains(&cfg.xi_points) {
                return Err(format!("xi_points must be in 1..={GM_MAX_NODES}; got {}", cfg.xi_points));
            }
            cfg.xi_points
        }
    };
    let cells = n_nodes
        .checked_mul(n_items)
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "node * item * category count-table size overflows usize".to_string())?;
    if cells > GM_MAX_COUNT_CELLS {
        return Err(format!(
            "count table {cells} cells exceeds the cap {GM_MAX_COUNT_CELLS}; reduce nodes/items/categories"
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
            return Err(format!("item {i} loads no dimension (all-zero loading_pattern row)"));
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
                "item {i} category {k} is never observed (unidentified GRM boundary); every declared \
                 category must be observed"
            ));
        }
    }
    for d in 0..n_dims {
        let has_pure = (0..n_items).any(|i| {
            loading_pattern[i * n_dims + d] != 0
                && (0..n_dims).filter(|&d2| loading_pattern[i * n_dims + d2] != 0).count() == 1
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
/// GRM. `params = [a_{d0}..a_{d,L-1}, beta_1..beta_{M-1}]` (`L = dims.len()`, `M = n_cat`); the slope
/// block precedes the `M-1` ordered boundary intercepts. `base = sum_t a_t * theta_{dims[t]}`;
/// `d/da_t = sum_node g_base * theta_{dims[t]}`, `d/dbeta_j = sum_node g_thr[j]`, chaining the GRM
/// node gradient. At `D = 1` (`L = 1`) the slope is the single `a` (unconstrained, vs
/// `poly::item_neg_ll_grad`'s `log a`).
fn grm_item_neg_ll_grad(
    params: &[f64],
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
) -> (f64, Vec<f64>) {
    let l = dims.len();
    let beta = &params[l..]; // M-1 boundary intercepts
    debug_assert_eq!(beta.len(), n_cat - 1, "GRM param layout: L slopes + (n_cat-1) thresholds");
    let mut ll = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];
    for (nd, cnt) in counts.iter().enumerate() {
        let mut base = 0.0f64;
        for (t, &d) in dims.iter().enumerate() {
            base += params[t] * nodes[nd * n_dims + d];
        }
        let lp = grm_logprobs(base, beta);
        ll += cnt.iter().zip(&lp).map(|(r, l2)| r * l2).sum::<f64>();
        let (g_base, g_thr) = grm_node_gradient(base, beta, cnt);
        for (t, &d) in dims.iter().enumerate() {
            grad[t] += g_base * nodes[nd * n_dims + d];
        }
        for (j, gj) in g_thr.iter().enumerate() {
            grad[l + j] += gj;
        }
    }
    (-ll, grad.iter().map(|v| -v).collect())
}

/// Newton M-step for one item — mirrors `poly::m_step_item` (FD Hessian, ridge conditioning,
/// backtracking line search), generalized to the multidimensional slope. The line search rejects any
/// step whose objective is non-finite, which keeps `beta` strictly ordered (see the module docs).
#[allow(clippy::too_many_arguments)]
fn grm_m_step(
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
        let (f0, g) = grm_item_neg_ll_grad(&params, dims, nodes, n_dims, counts, n_cat);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = grm_item_neg_ll_grad(&pj, dims, nodes, n_dims, counts, n_cat);
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
        // A boundary-crossing FD-Hessian column can yield a non-finite step; fall back to the
        // (finite) gradient direction — this also protects the ordered-threshold constraint.
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
            let (candidate_f, _) = grm_item_neg_ll_grad(&candidate, dims, nodes, n_dims, counts, n_cat);
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

/// Fit the confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson,
/// 1995) by Bock-Aitkin marginal MLE. See the module docs for the model, estimation, and
/// identification. `y`/`observed` are row-major `n_persons * n_items` (`y` ordered categories
/// `0..n_cat-1`, missing cells dropped MAR); `loading_pattern` is row-major `n_items * n_dims` in
/// `{0,1}`. Returns `Err` on malformed / rotationally-underidentified / unobserved-category input.
#[allow(clippy::too_many_arguments)]
pub fn fit_grm_mirt(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &GrmMirtConfig,
) -> Result<GrmMirtResult, String> {
    let _n_nodes = validate(y, observed, loading_pattern, n_persons, n_items, n_dims, n_cat, cfg)?;

    let (nodes, logw) = match cfg.xi_rule {
        XiRuleKind::GaussHermite => {
            let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: cfg.q }, n_dims)?;
            (xn.grid, xn.logw)
        }
        XiRuleKind::Halton => {
            let xn = build_xi_nodes(XiRule::Halton { n: cfg.xi_points, shift_seed: cfg.xi_seed }, n_dims)?;
            (xn.grid, xn.logw)
        }
        XiRuleKind::MonteCarlo => {
            let xn = build_xi_nodes(XiRule::MonteCarlo { n: cfg.xi_points, seed: cfg.xi_seed.max(1) }, n_dims)?;
            (xn.grid, xn.logw)
        }
    };
    let qn = logw.len();
    let m1 = n_cat - 1; // boundary count

    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| (0..n_dims).filter(|&d| loading_pattern[i * n_dims + d] != 0).collect())
        .collect();
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    // Init: slope = 1.0 on the item's FIRST loaded dim (0 elsewhere); beta_k = logit(P(Y>=k))
    // cumulative-from-top, ordered DECREASING — exactly fit_poly_unidim's GRM init (base=theta at D=1).
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let l = dims_of[i].len();
        let mut p = vec![0.0f64; l + m1];
        p[0] = 1.0; // slope on the first loaded dim
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
        let mut cum = 0.0f64;
        for k in (1..n_cat).rev() {
            cum += freq[k];
            let c = cum.clamp(1e-4, 1.0 - 1e-4);
            p[l + (k - 1)] = (c / (1.0 - c)).ln();
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

    // Compute per-item node x category log-probs into `all_lp[i]` (reused by E-step and EAP pass).
    let fill_lp = |params: &[Vec<f64>]| -> Vec<Vec<f64>> {
        let mut all_lp: Vec<Vec<f64>> = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let l = dims_of[i].len();
            let beta = &params[i][l..];
            let mut lp_i = vec![0.0f64; qn * n_cat];
            for nd in 0..qn {
                let mut base = 0.0f64;
                for (t, &d) in dims_of[i].iter().enumerate() {
                    base += params[i][t] * nodes[nd * n_dims + d];
                }
                let lp = grm_logprobs(base, beta);
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
            return Err(format!("non-finite observed-data log-likelihood at iteration {n_iter}"));
        }
        loglik_trace.push(ll);

        // Stopping: relative tolerance + SIGNED monotonic-decrease guard (not the .abs() check,
        // which would accept a likelihood DECREASE as convergence).
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
            params[i] = grm_m_step(
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

    // Assemble dense slope (n_items * n_dims) + thresholds (n_items * (n_cat-1)).
    let mut slope = vec![0.0f64; n_items * n_dims];
    let mut threshold = vec![0.0f64; n_items * m1];
    let mut n_parameters = 0usize;
    for i in 0..n_items {
        let l = dims_of[i].len();
        n_parameters += l + m1;
        for (t, &d) in dims_of[i].iter().enumerate() {
            slope[i * n_dims + d] = params[i][t];
        }
        threshold[i * m1..(i + 1) * m1].copy_from_slice(&params[i][l..]);
    }

    // Per-dimension reflection canonicalization: flip dimension d (its slopes on every item AND
    // theta_d) so its largest-|slope| PURE anchor loads positively. `base` — hence every threshold —
    // is invariant under the joint flip, so thresholds are NOT touched (module docs).
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

    let ll = *loglik_trace.last().expect("EM trace is never empty");
    let _ = ll;
    Ok(GrmMirtResult {
        n_dims,
        n_cat,
        slope,
        threshold,
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
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
        slope: &[f64], threshold: &[f64], theta: &[f64],
        n: usize, n_items: usize, n_dims: usize, n_cat: usize, rng: &mut Lcg,
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
    fn grm_mirt_reduces_to_poly_grm_at_d1() {
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
        let cfg = GrmMirtConfig { q: 21, ..GrmMirtConfig::default() };
        let mm = fit_grm_mirt(&y, None, &pattern, n, n_items, 1, n_cat, &cfg).unwrap();
        let pf = fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Grm, 21, 500, 1e-6).unwrap();
        // slopes agree (both positive), thresholds agree, within optimizer tolerance
        for i in 0..n_items {
            assert!((mm.slope[i] - pf.slope[i]).abs() < 0.05, "slope[{i}] {} vs {}", mm.slope[i], pf.slope[i]);
            for j in 0..m1 {
                let d = (mm.threshold[i * m1 + j] - pf.cat_params[i][j]).abs();
                assert!(d < 0.06, "threshold[{i}][{j}] diff {d}");
            }
        }
        let mm_ll = *mm.loglik_trace.last().unwrap();
        assert!((mm_ll - pf.loglik).abs() < 0.5, "loglik {mm_ll} vs {}", pf.loglik);
        assert_eq!(mm.n_parameters, n_items * (1 + m1));
    }

    /// Deterministic FD GRADIENT anchor at D=2 (GH) AND D=4 (Halton, NON-IDENTITY dims [0,2,3]) with
    /// M=4 categories. The threshold block is STRICTLY DECREASING with gaps >> the FD eps (GRM NaNs on
    /// inverted betas, unlike the finite-everywhere softmax); the slope block is distinct and the
    /// per-category counts random+distinct, so a slope<->threshold slot transposition or a sign error
    /// is detected. The M-step uses an FD Hessian, so pin the GRADIENT.
    #[test]
    fn grm_mirt_gradient_matches_finite_difference() {
        let n_cat = 4usize;
        for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter() {
            let l = dims.len();
            let (nodes, n_nodes) = if n_dims == 2 {
                let xn = build_xi_nodes(XiRule::GaussHermite { q_xi: 15 }, n_dims).unwrap();
                (xn.grid, xn.logw.len())
            } else {
                let xn = build_xi_nodes(XiRule::Halton { n: 200, shift_seed: 0 }, n_dims).unwrap();
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
                assert!((grad[j] - fd).abs() < 1e-4, "grad[{j}] {} vs fd {fd} (D={n_dims})", grad[j]);
            }
        }
    }

    /// Deterministic OBJECTIVE-VALUE dims-map pin at D=4 (Halton, dims=[0,2,3]). The FD gradient anchor
    /// is map-INVARIANT (a consistent wrong-node-column bug in base+gradient is invisible to a central
    /// difference through the same buggy objective); and no D>=4 fit is exercised by the recovery/MC
    /// tests. So compute the objective's per-node base and neg-loglik BY HAND with the CORRECT dim map
    /// and assert the estimator's internal value equals it to < 1e-9 — pinning nodes[nd*n_dims + dims[t]].
    #[test]
    fn grm_mirt_objective_dims_map_pinned_at_d4() {
        let n_dims = 4usize;
        let dims = vec![0usize, 2, 3];
        let n_cat = 4usize;
        let l = dims.len();
        let xn = build_xi_nodes(XiRule::Halton { n: 64, shift_seed: 0 }, n_dims).unwrap();
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
        assert!((neg_ll - (-hand)).abs() < 1e-9, "objective dims-map mismatch: {neg_ll} vs {}", -hand);
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
    fn grm_mirt_recovers_d2_with_negative_cross_loader() {
        let (n_dims, n_cat) = (2usize, 3usize);
        let m1 = n_cat - 1;
        let (pattern, n_items, slope, threshold) = design_d2(n_cat);
        let n = 6000usize;
        let mut rng = Lcg(4747);
        let mut theta = vec![0.0f64; n * n_dims];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GrmMirtConfig { q: 21, ..GrmMirtConfig::default() };
        let res = fit_grm_mirt(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
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
                assert!(res.threshold[i * m1 + j] > res.threshold[i * m1 + j + 1], "ordered item {i}");
            }
        }
        // canonical output: pure anchors positive; the negative cross-loader recovered NEGATIVE
        assert!(res.slope[0 * n_dims + 0] > 0.5, "anchor0 positive");
        assert!(res.slope[2 * n_dims + 1] > 0.5, "anchor2 positive");
        assert!(res.slope[4 * n_dims + 0] < -0.4, "neg cross-loader: {}", res.slope[4 * n_dims + 0]);
        assert!(rmse(&res.slope, &slope) < 0.16, "slope RMSE {}", rmse(&res.slope, &slope));
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
    fn grm_mirt_reflection_fires_on_negative_anchor() {
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
        let y = simulate(&slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GrmMirtConfig { q: 21, ..GrmMirtConfig::default() };
        let res = fit_grm_mirt(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
        // dim0's largest pure anchor (item 0) ends POSITIVE; co-loader (item 1) ends NEGATIVE.
        assert!(res.slope[0 * n_dims + 0] > 0.8, "reflected anchor positive: {}", res.slope[0 * n_dims + 0]);
        assert!(res.slope[1 * n_dims + 0] < -0.3, "co-loader flipped negative: {}", res.slope[1 * n_dims + 0]);
        // The reflection flips BOTH the slope column AND theta_d, keeping base = sum a_d theta_d
        // invariant. Since dim0 was flipped, the returned EAP theta_0 must correlate NEGATIVELY with
        // the true theta_0 (the data was generated with the negative anchor); dim1 (not flipped) stays
        // positive. Deleting the theta-negation half of the reflection inverts dim0's sign here.
        let th0: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 0]).collect();
        let tt0: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 0]).collect();
        let th1: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + 1]).collect();
        let tt1: Vec<f64> = (0..n).map(|j| theta[j * n_dims + 1]).collect();
        assert!(corr(&th0, &tt0) < -0.5, "flipped-dim theta corr must be negative: {}", corr(&th0, &tt0));
        assert!(corr(&th1, &tt1) > 0.5, "unflipped-dim theta corr positive: {}", corr(&th1, &tt1));
        // thresholds still strictly ordered (untouched by the reflection)
        for i in 0..n_items {
            for j in 0..m1 - 1 {
                assert!(res.threshold[i * m1 + j] > res.threshold[i * m1 + j + 1], "ordered item {i}");
            }
        }
    }

    /// Structural invariants + validation guards.
    #[test]
    fn grm_mirt_validates_and_structural_invariants() {
        let (n_dims, n_cat) = (2usize, 3usize);
        let (pattern, n_items, slope, threshold) = design_d2(n_cat);
        let n = 500usize;
        let mut rng = Lcg(99);
        let mut theta = vec![0.0f64; n * n_dims];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng);
        let cfg = GrmMirtConfig { q: 15, max_iter: 25, ..GrmMirtConfig::default() };
        let res = fit_grm_mirt(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
        // free-parameter count = sum_i (|S_i| + (n_cat-1)): items 0-3 pure (1+2), item 4 cross (2+2).
        assert_eq!(res.n_parameters, 4 * (1 + 2) + (2 + 2));
        // grm_logprobs sum to 1 at a sample base
        let lp = grm_logprobs(0.4, &[0.8, -0.3]);
        let s: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!((s - 1.0).abs() < 1e-12);
        // validation: GH D=4 rejected (y observes all categories so the D-bound is the sole reason);
        // no pure anchor rejected; category >= n_cat rejected; unobserved category rejected.
        let gh4 = GrmMirtConfig::default();
        let pat4: Vec<u8> = (0..4).flat_map(|d| (0..4).map(move |k| (k == d) as u8)).collect();
        let y4: Vec<usize> = (0..n * 4).map(|idx| idx % n_cat).collect();
        assert!(fit_grm_mirt(&y4, None, &pat4, n, 4, 4, n_cat, &gh4).is_err(), "GH D=4 rejected");
        let no_anchor: Vec<u8> = vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        assert!(fit_grm_mirt(&y, None, &no_anchor, n, n_items, n_dims, n_cat, &cfg).is_err(), "no pure anchor rejected");
        let mut ybad = y.clone();
        ybad[0] = n_cat;
        assert!(fit_grm_mirt(&ybad, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(), "bad category rejected");
        let mut ygap = y.clone();
        for p in 0..n {
            if ygap[p * n_items + 0] == 1 {
                ygap[p * n_items + 0] = 0;
            }
        }
        assert!(fit_grm_mirt(&ygap, None, &pattern, n, n_items, n_dims, n_cat, &cfg).is_err(), "unobserved category rejected");
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recover the multidimensional GRM at D=2 and D=3
    /// under normal AND per-dim-standardized right-skew traits. The estimator canonicalizes reflection
    /// (pure anchors positive), so truth is built positive-anchored and the estimate compares directly.
    /// Per-rep monotone-EM + finiteness + threshold-ordering canaries.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_grm_mirt_recovery_500() {
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
                    let mut rng = Lcg(
                        0x9E3779B97F4A7C15u64
                            .wrapping_mul(rep as u64 + 1)
                            .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                            .wrapping_add(n_dims as u64 * 0x100000001B3),
                    );
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
                    let y = simulate(&slope, &threshold, &theta, n, n_items, n_dims, n_cat, &mut rng);
                    let cfg = GrmMirtConfig { q, ..GrmMirtConfig::default() };
                    let res = fit_grm_mirt(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
                    if res.converged {
                        nconv += 1;
                    }
                    for w in res.loglik_trace.windows(2) {
                        assert!(w[1] >= w[0] - 1e-9, "monotone (rep {rep})");
                    }
                    assert!(res.slope.iter().all(|v| v.is_finite()), "finite slope (rep {rep})");
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
                    "[grm-mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
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
}
