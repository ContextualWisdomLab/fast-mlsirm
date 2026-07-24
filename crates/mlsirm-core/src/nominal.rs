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
/// Upper bound on caller-controlled EM iterations.
const NM_MAX_ITER: usize = 100_000;

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
    if !(1..=NM_MAX_ITER).contains(&cfg.max_iter) {
        return Err(format!("max_iter must be in 1..={NM_MAX_ITER}"));
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
                // SUPPORTED_Q and the three-dimension bound cap this at 41^3 = 68,921.
                n *= cfg.q;
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
        let previous = loglik_trace.last().copied();
        let change = checked_em_loglik_change(ll, previous, n_iter)?;
        loglik_trace.push(ll);

        // Stopping: fit_nominal's RELATIVE tolerance + signed monotonic-decrease guard (NOT the
        // MIRT .abs() check, which would accept a likelihood DECREASE as convergence).
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
#[path = "../../../tests/unit/nominal_tests.rs"]
mod tests;
