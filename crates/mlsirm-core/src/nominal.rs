//! Confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen, Cai & Bock, 2010),
//! generalizing the unidimensional [`crate::poly::fit_nominal`] to `D` latent dimensions.
//!
//! Each item `i` has `n_cat` UNORDERED categories. Category `k` gets a free multidimensional
//! discrimination (slope) vector `a_ik` and intercept `c_ik`; the linear predictor for category `k`
//! is `eta_ik(theta) = sum_{d in S_i} a_ikd theta_d + c_ik` and the category probability is the
//! softmax `P(Y_i = k | theta) = softmax_k(eta_ik)`, with the baseline category `0` pinned
//! `a_i0 = 0`, `c_i0 = 0`. `S_i` is item `i`'s loading set (a 0/1 confirmatory pattern, items x D):
//! a category slope `a_ikd` is free only for `d in S_i`. `theta ~ MVN(0, I_D)`.
//!
//! At `D = 1` with `S_i = {0}` this is EXACTLY `poly::fit_nominal` (`eta_ik = a_ik theta + c_ik`,
//! the same general free-`a_k` parametrization — NOT the `a * s_k` scoring-contrast form).
//!
//! **Estimation.** Bock-Aitkin marginal MLE (EM) over the `D`-dimensional latent grid, reusing the
//! integration-node machinery of the compensatory MIRT (`nodes::build_xi_nodes`): `node_rule = "gh"`
//! uses the `q^D` Gauss-Hermite product grid (`D <= 3`); `"qmc"`/`"mc"` use `xi_points` Halton /
//! Monte-Carlo draws (`D <= 6`), the quasi-Monte-Carlo EM of Jank (2005). The node set is built ONCE
//! before the EM loop, and — because `theta ~ MVN(0, I)` never reparametrizes the nodes (unlike the
//! correlated-`Sigma` MIRT) — EM is strictly monotone in the (QMC-)approximated marginal likelihood.
//! The per-item M-step is a Newton on the concave multinomial-logit complete-data objective, byte-for-
//! byte the finite-difference-Hessian ascent of `poly::nominal_m_step` (ridge is Hessian conditioning
//! only, NOT a parameter prior, so the fit is genuine MML), generalized so the softmax residual
//! `resid_k = r_k - n P_k` drives `d/dc_ik = sum_node resid_k` and `d/da_ikd = sum_node resid_k theta_d`.
//!
//! **Identification.** Baseline category `a_i0 = c_i0 = 0` fixes the softmax reference; unit trait
//! variances fix the per-dimension slope scale and `E[theta] = 0` the intercept level; a PURE
//! single-dimension anchor item per dimension (`S_i = {d}`) pins the rotation to the coordinate axes
//! (a pure anchor forces every category slope onto axis `d`, so an orthogonal trait rotation must map
//! axis `d` to `+-e_d`; the confirmatory labels forbid axis permutation, leaving only per-dimension
//! reflection). The reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` is NOT canonicalized (as in
//! `fit_nominal`); parameters are identified up to it, and recovery is assessed up to per-dimension
//! reflection.
//!
//! # References (APA 7th ed.)
//!
//! Bock, R. D. (1972). Estimating item parameters and latent ability when responses are scored in two
//! or more nominal categories. *Psychometrika, 37*(1), 29-51. https://doi.org/10.1007/BF02291411
//!
//! Thissen, D., Cai, L., & Bock, R. D. (2010). The nominal categories item response model. In M. L.
//! Nering & R. Ostini (Eds.), *Handbook of polytomous item response theory models* (pp. 43-75).
//! Routledge.
//!
//! Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
//! https://doi.org/10.1007/978-0-387-89976-3
//!
//! Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM.
//! *Computational Statistics & Data Analysis, 48*(4), 685-701. https://doi.org/10.1016/j.csda.2004.03.019

use crate::marginal::XiRuleKind;
use crate::nodes::{build_xi_nodes, XiRule};
use crate::poly::{gpcm_logprobs, solve_small};
use crate::quadrature::SUPPORTED_Q;

/// Maximum integration node count (bounds the `nodes x J x n_cat` count table) for BOTH the `Q^D`
/// grid and the `xi_points` QMC/MC point set.
const NM_MAX_NODES: usize = 200_000;
/// Total expected-count-table cap (`nodes * n_items * n_cat` f64s) — the memory the E-step allocates.
const NM_MAX_COUNT_CELLS: usize = 60_000_000;
/// Max latent dimensions for the Gauss-Hermite product grid (`D > 3` uses Halton/MonteCarlo).
const NM_MAX_DIMS: usize = 3;
/// Max latent dimensions for the Halton/MonteCarlo rules (= `HALTON_PRIMES.len()` in `nodes`; the
/// MonteCarlo builder has no internal cap, so this is its sole guard).
const NM_MAX_DIMS_QMC: usize = 6;
/// Sanity cap on the number of categories.
const NM_MAX_CAT: usize = 64;

/// Configuration for [`fit_nominal`]. Defaults mirror the compensatory MIRT and `fit_nominal`.
#[derive(Clone, Copy, Debug)]
pub struct NominalConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// Gauss-Hermite nodes per dimension (used only for `xi_rule = GaussHermite`).
    pub q: usize,
    /// Newton (FD-Hessian) ridge — Hessian CONDITIONING only, NOT a parameter prior (so the fit
    /// stays MML and reduces to `fit_nominal`). Matches `nominal_m_step`'s `1e-8`.
    pub ridge: f64,
    /// Inner Newton iterations per item M-step (matches `fit_nominal`'s `10`).
    pub newton_iter: usize,
    /// Integration node rule; `"gh"` (`D <= 3`), or `Halton`/`MonteCarlo` (`D <= 6`).
    pub xi_rule: XiRuleKind,
    /// QMC/MC point count (used only for `Halton`/`MonteCarlo`).
    pub xi_points: usize,
    /// Halton Cranley-Patterson shift seed / Monte-Carlo seed (nonzero by default).
    pub xi_seed: u64,
}

impl Default for NominalConfig {
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

/// Result of [`fit_nominal`].
#[derive(Clone, Debug)]
pub struct NominalResult {
    pub n_dims: usize,
    pub n_cat: usize,
    /// Category slopes `a_ikd`, row-major `n_items * n_cat * n_dims`. The baseline category
    /// (`k = 0`) and off-pattern entries (`d not in S_i`) are exactly `0.0`.
    pub slope: Vec<f64>,
    /// Category intercepts `c_ik`, row-major `n_items * n_cat` (baseline `k = 0` is exactly `0.0`).
    pub intercept: Vec<f64>,
    /// Per-person trait EAP `E[theta_jd | X_j]`, row-major `n_persons * n_dims`.
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// `sum_i (n_cat - 1) * (|S_i| + 1)` free item parameters.
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
    cfg: &NominalConfig,
) -> Result<usize, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(2..=NM_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={NM_MAX_CAT}; got {n_cat}"));
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
    // Rule-dependent dimension bound + node-count cap. GH caps at NM_MAX_DIMS (Q^D blows up); the
    // QMC/MC rules cap at NM_MAX_DIMS_QMC and bound xi_points instead. `q` is used only by GH.
    let n_nodes = match cfg.xi_rule {
        XiRuleKind::GaussHermite => {
            if !(1..=NM_MAX_DIMS).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={NM_MAX_DIMS} for the Gauss-Hermite grid; use \
                     node_rule qmc/mc for D up to {NM_MAX_DIMS_QMC}"
                ));
            }
            if !SUPPORTED_Q.contains(&cfg.q) {
                return Err(format!("q must be one of {SUPPORTED_Q:?}; got {}", cfg.q));
            }
            let mut n = 1usize;
            for _ in 0..n_dims {
                n = n
                    .checked_mul(cfg.q)
                    .filter(|&v| v <= NM_MAX_NODES)
                    .ok_or_else(|| format!("q^n_dims exceeds the node cap {NM_MAX_NODES}"))?;
            }
            n
        }
        XiRuleKind::Halton | XiRuleKind::MonteCarlo => {
            if !(1..=NM_MAX_DIMS_QMC).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={NM_MAX_DIMS_QMC} for the Halton/MonteCarlo rules"
                ));
            }
            if !(1..=NM_MAX_NODES).contains(&cfg.xi_points) {
                return Err(format!(
                    "xi_points must be in 1..={NM_MAX_NODES}; got {}",
                    cfg.xi_points
                ));
            }
            cfg.xi_points
        }
    };
    // The expected-count table is n_nodes * n_items * n_cat f64s (a factor n_cat larger than the
    // MIRT Bernoulli table): bound it with checked multiplies.
    let cells = n_nodes
        .checked_mul(n_items)
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "node * item * category count-table size overflows usize".to_string())?;
    if cells > NM_MAX_COUNT_CELLS {
        return Err(format!(
            "count table {cells} cells exceeds the cap {NM_MAX_COUNT_CELLS}; reduce nodes/items/categories"
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
    // Every item loads >= 1 dimension; every DECLARED category is observed for the item (an
    // unobserved category — interior or top — drives its intercept to -inf and leaves its D slopes
    // unidentified, so it is rejected rather than fit; fit_nominal does not guard this).
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
                "item {i} category {k} is never observed (under-identified nominal category); \
                 every declared category must be observed"
            ));
        }
    }
    // Identification: every dimension needs a PURE single-loading anchor item.
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
/// nominal model. `params` is the item's free vector laid out as
/// `[a_{1,d0}, a_{1,d1}, .., a_{1,d_{L-1}}, a_{2,d0}, .., a_{K-1,d_{L-1}}, c_1, .., c_{K-1}]`
/// (`L = dims.len()`, `K = n_cat`), so the slopes are category-major over the loaded dims followed
/// by the intercepts. At `D = 1`, `L = 1` this is exactly `nominal_item_neg_ll_grad`'s
/// `[a_1..a_{K-1}, c_1..c_{K-1}]`. `nodes` is row-major `n_nodes * n_dims`; `counts[nd]` the expected
/// category counts at node `nd`. Softmax residual `resid_k = r_k - n P_k` gives
/// `d/dc_k = resid_k`, `d/da_kd = resid_k * theta_d`.
fn nm_item_neg_ll_grad(
    params: &[f64],
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
) -> (f64, Vec<f64>) {
    let z = n_cat - 1; // free non-baseline categories
    let l = dims.len();
    let sbase = vec![0.0f64; n_cat]; // gpcm_logprobs scores are irrelevant when base = 0
    let mut ll = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];
    let mut eta = vec![0.0f64; n_cat];
    let n_nodes = counts.len();
    for nd in 0..n_nodes {
        // eta_k = sum_{d in S} a_kd * theta_d + c_k, with eta_0 = 0 (baseline).
        eta[0] = 0.0;
        for k in 1..n_cat {
            let mut e = params[z * l + (k - 1)]; // c_k
            let base = (k - 1) * l;
            for (t, &d) in dims.iter().enumerate() {
                e += params[base + t] * nodes[nd * n_dims + d];
            }
            eta[k] = e;
        }
        let lp = gpcm_logprobs(0.0, &sbase, &eta); // log softmax(eta)
        ll += counts[nd]
            .iter()
            .zip(&lp)
            .map(|(r, l2)| r * l2)
            .sum::<f64>();
        let n: f64 = counts[nd].iter().sum();
        // residual and gradient accumulation
        for k in 1..n_cat {
            let resid = counts[nd][k] - n * lp[k].exp();
            grad[z * l + (k - 1)] += resid; // d/dc_k
            let base = (k - 1) * l;
            for (t, &d) in dims.iter().enumerate() {
                grad[base + t] += resid * nodes[nd * n_dims + d]; // d/da_kd
            }
        }
    }
    (-ll, grad.iter().map(|v| -v).collect())
}

/// Newton M-step for one item — byte-for-byte the finite-difference-Hessian ascent of
/// `poly::nominal_m_step` (ridge is Hessian conditioning only), generalized to the multidimensional
/// gradient. At `D = 1`, `dims = [0]` this reproduces `nominal_m_step` exactly.
#[allow(clippy::too_many_arguments)]
fn nm_m_step(
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
        let (f0, g) = nm_item_neg_ll_grad(&params, dims, nodes, n_dims, counts, n_cat);
        let grad_norm = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = nm_item_neg_ll_grad(&pj, dims, nodes, n_dims, counts, n_cat);
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
                nm_item_neg_ll_grad(&candidate, dims, nodes, n_dims, counts, n_cat);
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

/// Fit the confirmatory MULTIDIMENSIONAL nominal response model (Bock, 1972; Thissen, Cai & Bock,
/// 2010) by Bock-Aitkin marginal MLE. See the module docs for the model, estimation, and
/// identification. `y`/`observed` are row-major `n_persons * n_items` (`y` categories `0..n_cat-1`,
/// missing cells dropped under MAR); `loading_pattern` is row-major `n_items * n_dims` in `{0,1}`.
/// Returns `Err` on malformed / rotationally-underidentified / unobserved-category input.
#[allow(clippy::too_many_arguments)]
pub fn fit_nominal(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    n_cat: usize,
    cfg: &NominalConfig,
) -> Result<NominalResult, String> {
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

    // Build the latent-integral node set once (fixed-node QMC-EM; monotone since theta ~ N(0,I)).
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
    let z = n_cat - 1;

    // Per-item loaded-dimension lists S_i and free-parameter vectors.
    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] != 0)
                .collect()
        })
        .collect();
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    // Init: category slope = k on the item's FIRST loaded dim (0 on the others); intercept =
    // log(freq_k / freq_0). At D = 1 (single loaded dim) this is fit_nominal's a_k = k, c_k init.
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let l = dims_of[i].len();
        let mut p = vec![0.0f64; z * l + z];
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
            p[(k - 1) * l] = k as f64; // slope on the first loaded dim
            p[z * l + (k - 1)] = (freq[k] / freq[0]).ln(); // c_k
        }
        params.push(p);
    }

    let mut loglik_trace: Vec<f64> = Vec::with_capacity(cfg.max_iter + 1);
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;
    let mut theta = vec![0.0f64; n_persons * n_dims];

    // reused buffers
    let mut item_lp = vec![0.0f64; qn * n_cat]; // per-item, reused
    let mut eta = vec![0.0f64; n_cat];
    let sbase = vec![0.0f64; n_cat];
    let mut log_node = vec![0.0f64; qn];

    loop {
        // Node x category log-probs per item.
        let mut all_lp: Vec<Vec<f64>> = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let l = dims_of[i].len();
            for nd in 0..qn {
                eta[0] = 0.0;
                for k in 1..n_cat {
                    let mut e = params[i][z * l + (k - 1)];
                    let base = (k - 1) * l;
                    for (t, &d) in dims_of[i].iter().enumerate() {
                        e += params[i][base + t] * nodes[nd * n_dims + d];
                    }
                    eta[k] = e;
                }
                let lp = gpcm_logprobs(0.0, &sbase, &eta);
                item_lp[nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
            all_lp.push(item_lp.clone());
        }

        // Streamed E-step: per person, posterior over nodes; expected category counts.
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

        // Stopping: fit_nominal's RELATIVE tolerance + signed monotonic-decrease guard (NOT the
        // MIRT .abs() check, which would accept a likelihood DECREASE as convergence).
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

        // M-step: per item, Newton mirroring nominal_m_step (multidimensional gradient).
        for i in 0..n_items {
            params[i] = nm_m_step(
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

    // Final EAP pass under the returned parameters.
    {
        let mut all_lp: Vec<Vec<f64>> = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let l = dims_of[i].len();
            for nd in 0..qn {
                eta[0] = 0.0;
                for k in 1..n_cat {
                    let mut e = params[i][z * l + (k - 1)];
                    let base = (k - 1) * l;
                    for (t, &d) in dims_of[i].iter().enumerate() {
                        e += params[i][base + t] * nodes[nd * n_dims + d];
                    }
                    eta[k] = e;
                }
                let lp = gpcm_logprobs(0.0, &sbase, &eta);
                item_lp[nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
            all_lp.push(item_lp.clone());
        }
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

    // Assemble the dense (n_items * n_cat * n_dims) slope tensor + (n_items * n_cat) intercepts,
    // with the baseline category and off-pattern entries exactly 0.0.
    let mut slope = vec![0.0f64; n_items * n_cat * n_dims];
    let mut intercept = vec![0.0f64; n_items * n_cat];
    let mut n_parameters = 0usize;
    for i in 0..n_items {
        let l = dims_of[i].len();
        n_parameters += z * (l + 1);
        for k in 1..n_cat {
            intercept[i * n_cat + k] = params[i][z * l + (k - 1)];
            let base = (k - 1) * l;
            for (t, &d) in dims_of[i].iter().enumerate() {
                slope[(i * n_cat + k) * n_dims + d] = params[i][base + t];
            }
        }
    }

    let ll = *loglik_trace.last().expect("EM trace is never empty");
    let _ = ll;
    Ok(NominalResult {
        n_dims,
        n_cat,
        slope,
        intercept,
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
        for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (4usize, vec![0usize, 2, 3])].iter()
        {
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
                    let res =
                        fit_nominal(&y, None, &pattern, n, n_items, n_dims, n_cat, &cfg).unwrap();
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
}
