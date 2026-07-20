//! Confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson, 1995),
//! the ORDERED-category counterpart of [`crate::nominal::fit_nominal`] and the polytomous
//! generalization of the compensatory MIRT ([`crate::twopl::fit_2pl`]).
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
//! *Psychometrika, 34*(S1), 1-97. https://doi.org/10.1007/BF03372160
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
const GM_MAX_ITER: usize = 100_000;

/// Configuration for [`fit_grm`].
#[derive(Clone, Copy, Debug)]
pub struct GrmConfig {
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

impl Default for GrmConfig {
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

/// Result of [`fit_grm`].
#[derive(Clone, Debug)]
pub struct GrmResult {
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
    cfg: &GrmConfig,
) -> Result<usize, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=GM_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={GM_MAX_CAT}; got {n_cat}"));
    }
    if !(1..=GM_MAX_ITER).contains(&cfg.max_iter) {
        return Err(format!("max_iter must be in 1..={GM_MAX_ITER}"));
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
                // SUPPORTED_Q and the three-dimension bound cap this at 41^3 = 68,921.
                n *= cfg.q;
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
                return Err(format!(
                    "xi_points must be in 1..={GM_MAX_NODES}; got {}",
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
    // The count-table cap above bounds n_items, while validation bounds n_dims.
    let n_l = n_items * n_dims;
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
                "item {i} category {k} is never observed (unidentified GRM boundary); every declared \
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
    debug_assert_eq!(
        beta.len(),
        n_cat - 1,
        "GRM param layout: L slopes + (n_cat-1) thresholds"
    );
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
            let (candidate_f, _) =
                grm_item_neg_ll_grad(&candidate, dims, nodes, n_dims, counts, n_cat);
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

fn checked_em_loglik_change(
    current: f64,
    previous: Option<f64>,
    iteration: usize,
) -> Result<Option<f64>, String> {
    if !current.is_finite() {
        return Err(format!(
            "non-finite observed-data log-likelihood at iteration {iteration}"
        ));
    }
    let Some(previous) = previous else {
        return Ok(None);
    };
    let change = current - previous;
    let monotonicity_tolerance = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    if change < -monotonicity_tolerance {
        return Err(format!(
            "EM observed-data log-likelihood decreased at iteration {iteration}: delta={change:.6e}"
        ));
    }
    Ok(Some(change))
}

/// Fit the confirmatory MULTIDIMENSIONAL graded response model (Samejima, 1969; Muraki & Carlson,
/// 1995) by Bock-Aitkin marginal MLE. See the module docs for the model, estimation, and
/// identification. `y`/`observed` are row-major `n_persons * n_items` (`y` ordered categories
/// `0..n_cat-1`, missing cells dropped MAR); `loading_pattern` is row-major `n_items * n_dims` in
/// `{0,1}`. Returns `Err` on malformed / rotationally-underidentified / unobserved-category input.
#[allow(clippy::too_many_arguments)]
pub fn fit_grm(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &GrmConfig,
) -> Result<GrmResult, String> {
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

    let xi_rule = match cfg.xi_rule {
        XiRuleKind::GaussHermite => XiRule::GaussHermite { q_xi: cfg.q },
        XiRuleKind::Halton => XiRule::Halton {
            n: cfg.xi_points,
            shift_seed: cfg.xi_seed,
        },
        XiRuleKind::MonteCarlo => XiRule::MonteCarlo {
            n: cfg.xi_points,
            seed: cfg.xi_seed.max(1),
        },
    };
    let xn = build_xi_nodes(xi_rule, n_dims)?;
    let (nodes, logw) = (xn.grid, xn.logw);
    let qn = logw.len();
    let m1 = n_cat - 1; // boundary count

    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] != 0)
                .collect()
        })
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
        let previous = loglik_trace.last().copied();
        let change = checked_em_loglik_change(ll, previous, n_iter)?;
        loglik_trace.push(ll);

        // Stopping: relative tolerance + SIGNED monotonic-decrease guard (not the .abs() check,
        // which would accept a likelihood DECREASE as convergence).
        if let Some(change) = change {
            let prev = previous.expect("change requires a previous log-likelihood");
            final_loglik_change = change;
            let stop_tol = cfg.tol * (1.0 + prev.abs());
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
        let ai = anchor.expect("validation guarantees a pure anchor for every dimension");
        if slope[ai * n_dims + d] < 0.0 {
            for i in 0..n_items {
                slope[i * n_dims + d] = -slope[i * n_dims + d];
            }
            for p in 0..n_persons {
                theta[p * n_dims + d] = -theta[p * n_dims + d];
            }
        }
    }

    let ll = *loglik_trace.last().expect("EM trace is never empty");
    let _ = ll;
    Ok(GrmResult {
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
#[path = "../../../tests/unit/grm_tests.rs"]
mod tests;
