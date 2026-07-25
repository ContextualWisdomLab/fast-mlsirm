//! Many-Facet Rasch Model (MFRM; Linacre, 1989) by marginal-ML EM.
//!
//! The MFRM extends the rating scale model with a rater facet: each rating of
//! person `p` on item `i` by rater `j` follows the adjacent-category log-odds
//!
//! ```text
//! ln[ P(Y_pij = k | theta) / P(Y_pij = k-1 | theta) ]
//!     = theta_p - d_i - c_j - f_k,        k = 1..K-1,
//! ```
//!
//! with item difficulty `d_i`, rater severity `c_j`, and category thresholds
//! `f_k` shared across items and raters (the rating-scale form of Linacre's
//! model). The cumulative predictor is `psi_k = k*theta - k*(d_i + c_j) - T_k`
//! with `T_k = sum_{m<=k} f_m`, `psi_0 = 0`, `P(Y=k) = softmax_k(psi)` — exactly
//! the RSM cell ([`crate::rsm::rsm_logprobs`]) with location `d_i + c_j`, which
//! this module reuses. At `n_raters = 1` (severity centered to 0) the model
//! reduces to the RSM.
//!
//! Verified formulation: the adjacent-category identity
//! `psi_k - psi_{k-1} = theta - d_i - c_j - f_k` was re-derived here and
//! adversarially checked; it matches the published Linacre (1989) rating-scale
//! MFRM form as documented by Eckes (2015). We did **not** reproduce Linacre's
//! JMLE estimator: Facets-style JMLE is replaced by marginal ML (Bock & Aitkin,
//! 1981) with `theta ~ N(0,1)` on a Gauss-Hermite grid, matching this crate's
//! estimation contract (see `rsm.rs`, `mixed.rs`). Parameter estimates are
//! therefore comparable to Facets output only up to the JMLE-vs-MMLE difference.
//!
//! Identification: the probabilities are invariant under
//! `f_m -> f_m - c, d_i -> d_i + c` and under `c_j -> c_j - c, d_i -> d_i + c`;
//! the trait scale is fixed by `theta ~ N(0,1)`. Both shift redundancies are
//! removed by centering `sum_m f_m = 0` and `sum_j c_j = 0`, leaving
//! `n_items + (n_raters - 1) + (n_cat - 2)` free parameters.
//!
//! Connectedness: Linacre's connectedness requirement concerns *design*
//! linking. We report a `connected` flag from a union-find over facet elements
//! (items and raters), joining every element that appears in the same person's
//! observed cells. When `connected == false`, severity/difficulty comparisons
//! across components are anchored only by the shared `theta ~ N(0,1)`
//! assumption (model-prior linking), not by the rating design itself.
//!
//! # References (APA 7th ed.)
//! Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
//!   item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
//!   443-459. https://doi.org/10.1007/BF02293801
//! Eckes, T. (2015). *Introduction to many-facet Rasch measurement* (2nd ed.).
//!   Peter Lang. https://doi.org/10.3726/978-3-653-04844-5
//! Linacre, J. M. (1989). *Many-facet Rasch measurement*. MESA Press.
//! Andrich, D. (1978). A rating formulation for ordered response categories.
//!   *Psychometrika, 43*(4), 561-573. https://doi.org/10.1007/BF02293814

use crate::poly::solve_small;
use crate::rsm::rsm_logprobs;

const FACETS_MAX_CAT: usize = 64;
const FACETS_MAX_ITER: usize = 100_000;
const FACETS_MAX_CELLS: usize = 60_000_000;

/// Fitted many-facet Rasch model (Linacre, 1989). `item_difficulty` is `d_i`;
/// `rater_severity` the centered `c_j` (`sum = 0`, higher = harsher);
/// `thresholds` the `K-1` common category thresholds (centered, `sum = 0`);
/// `theta` the per-person EAP trait. `connected` is the design-linking flag
/// (see module docs).
#[derive(Clone, Debug)]
pub struct FacetsResult {
    pub item_difficulty: Vec<f64>,
    pub rater_severity: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub connected: bool,
    /// `n_items + (n_raters - 1) + (n_cat - 2)`.
    pub n_parameters: usize,
}

/// Fit the many-facet Rasch model (Linacre, 1989) by marginal-ML EM. `y` is
/// `n_persons * n_items * n_raters` row-major (rater fastest) categories
/// `0..n_cat-1`; `observed` marks scored cells (sparse judging plans allowed;
/// `None` = fully crossed). Ability `theta ~ N(0,1)` on the `q_theta`-node
/// Gauss-Hermite grid.
#[allow(clippy::too_many_arguments)]
pub fn fit_facets(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<FacetsResult, String> {
    if !(2..=FACETS_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={FACETS_MAX_CAT}"));
    }
    if n_persons < 1 || n_items < 1 || n_raters < 1 {
        return Err("n_persons, n_items and n_raters must be >= 1".into());
    }
    if !(1..=FACETS_MAX_ITER).contains(&max_iter) {
        return Err(format!("max_iter must be in 1..={FACETS_MAX_ITER}"));
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and > 0".into());
    }
    let n_pairs =
        crate::checked_mul_usize(n_items, n_raters, "n_items * n_raters overflows usize")?;
    let n_cells =
        crate::checked_mul_usize(n_persons, n_pairs, "n_persons * n_items * n_raters overflows")?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items * n_raters".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items * n_raters".into());
        }
    }
    for (idx, &cat) in y.iter().enumerate() {
        if observed.map_or(true, |o| o[idx]) && cat >= n_cat {
            return Err("response category out of range 0..n_cat-1".into());
        }
    }
    let is_obs = |p: usize, pair: usize| observed.map_or(true, |o| o[p * n_pairs + pair]);
    for i in 0..n_items {
        if !(0..n_persons)
            .any(|p| (0..n_raters).any(|j| is_obs(p, i * n_raters + j)))
        {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    for j in 0..n_raters {
        if !(0..n_persons).any(|p| (0..n_items).any(|i| is_obs(p, i * n_raters + j))) {
            return Err(format!("rater {j} has no observed responses"));
        }
    }
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let qn = nodes.len();
    let count_cells = n_pairs
        .checked_mul(qn)
        .and_then(|c| c.checked_mul(n_cat))
        .ok_or_else(|| "pair * node * category table size overflows usize".to_string())?;
    if count_cells > FACETS_MAX_CELLS {
        return Err(format!(
            "count table {count_cells} cells exceeds the cap {FACETS_MAX_CELLS}"
        ));
    }
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let kb = n_cat - 1;

    let connected = design_connected(n_persons, n_items, n_raters, &is_obs);

    // Init: item difficulty from the item mean category (as in rsm.rs), rater
    // severity and thresholds at 0.
    let mut d = vec![0.0f64; n_items];
    let mut c = vec![0.0f64; n_raters];
    let mut f = vec![0.0f64; kb];
    for i in 0..n_items {
        let (mut s, mut cnt) = (0.0f64, 0.0f64);
        for p in 0..n_persons {
            for j in 0..n_raters {
                if is_obs(p, i * n_raters + j) {
                    s += y[p * n_pairs + i * n_raters + j] as f64;
                    cnt += 1.0;
                }
            }
        }
        if cnt > 0.0 {
            let mean = s / cnt / kb as f64;
            d[i] = ((1.0 - mean).clamp(0.02, 0.98) / mean.clamp(0.02, 0.98)).ln();
        }
    }

    let mut it = 0usize;
    let mut converged = false;
    let mut loglik_trace: Vec<f64> = Vec::new();

    while it < max_iter {
        // Per-pair cell log-probs at each node (RSM cell, location d_i + c_j).
        let item_lp = pair_logprobs(&d, &c, &f, nodes, n_items, n_raters, n_cat);
        // E-step: posteriors -> expected counts r[pair][node][k].
        let mut r = vec![vec![0.0f64; qn * n_cat]; n_pairs];
        let mut ll = 0.0f64;
        let mut log_node = vec![0.0f64; qn];
        for p in 0..n_persons {
            log_node[..qn].copy_from_slice(&log_w[..qn]);
            for pair in 0..n_pairs {
                if !is_obs(p, pair) {
                    continue;
                }
                let yc = y[p * n_pairs + pair];
                for nd in 0..qn {
                    log_node[nd] += item_lp[pair][nd * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for nd in 0..qn {
                denom += (log_node[nd] - mx).exp();
            }
            ll += mx + denom.ln();
            for pair in 0..n_pairs {
                if !is_obs(p, pair) {
                    continue;
                }
                let yc = y[p * n_pairs + pair];
                for nd in 0..qn {
                    r[pair][nd * n_cat + yc] += (log_node[nd] - mx).exp() / denom;
                }
            }
        }

        loglik_trace.push(ll);
        it += 1;
        if loglik_trace.len() > 1 {
            let nn = loglik_trace.len();
            if (loglik_trace[nn - 1] - loglik_trace[nn - 2]).abs()
                < tol * (1.0 + loglik_trace[nn - 2].abs())
            {
                converged = true;
                break;
            }
        }

        // CM-1: per-item Newton on d_i (c, f fixed), aggregated over raters.
        // g = -sum_{j,nd,k} k*(r - n*P); h = -sum n*Var(score) < 0 (score = k).
        for i in 0..n_items {
            for _ in 0..25 {
                let (mut g, mut h) = (0.0f64, 0.0f64);
                for j in 0..n_raters {
                    let pair = i * n_raters + j;
                    location_score_terms(
                        d[i] + c[j],
                        &f,
                        &r[pair],
                        nodes,
                        n_cat,
                        &mut g,
                        &mut h,
                    );
                }
                if h >= 0.0 {
                    break;
                }
                let step = g / h;
                let cur = item_ell_d(i, &d, &c, &f, &r, nodes, n_raters, n_cat);
                let mut al = 1.0f64;
                let mut accepted = false;
                for _ in 0..24 {
                    let cand = d[i] - al * step;
                    let mut dc = d.clone();
                    dc[i] = cand;
                    if item_ell_d(i, &dc, &c, &f, &r, nodes, n_raters, n_cat) >= cur - 1e-12 {
                        d[i] = cand;
                        accepted = true;
                        break;
                    }
                    al *= 0.5;
                }
                if !accepted || (al * step).abs() < 1e-9 {
                    break;
                }
            }
        }

        // CM-2: per-rater Newton on c_j (d, f fixed), aggregated over items —
        // same algebra as CM-1 by the d<->c symmetry of the location d_i + c_j.
        for j in 0..n_raters {
            for _ in 0..25 {
                let (mut g, mut h) = (0.0f64, 0.0f64);
                for i in 0..n_items {
                    let pair = i * n_raters + j;
                    location_score_terms(
                        d[i] + c[j],
                        &f,
                        &r[pair],
                        nodes,
                        n_cat,
                        &mut g,
                        &mut h,
                    );
                }
                if h >= 0.0 {
                    break;
                }
                let step = g / h;
                let cur = rater_ell_c(j, &d, &c, &f, &r, nodes, n_items, n_raters, n_cat);
                let mut al = 1.0f64;
                let mut accepted = false;
                for _ in 0..24 {
                    let cand = c[j] - al * step;
                    let mut cc = c.clone();
                    cc[j] = cand;
                    if rater_ell_c(j, &d, &cc, &f, &r, nodes, n_items, n_raters, n_cat)
                        >= cur - 1e-12
                    {
                        c[j] = cand;
                        accepted = true;
                        break;
                    }
                    al *= 0.5;
                }
                if !accepted || (al * step).abs() < 1e-9 {
                    break;
                }
            }
        }

        // CM-3: joint Newton on the common thresholds f (d, c fixed),
        // aggregated over all (item, rater) pairs; FD Hessian of the gradient.
        for _ in 0..25 {
            let g = f_gradient(&f, &d, &c, &r, nodes, n_items, n_raters, n_cat);
            let mut hess = vec![vec![0.0f64; kb]; kb];
            let eps = 1e-5;
            for jj in 0..kb {
                let mut fp = f.clone();
                fp[jj] += eps;
                let gj = f_gradient(&fp, &d, &c, &r, nodes, n_items, n_raters, n_cat);
                for a in 0..kb {
                    hess[a][jj] = (gj[a] - g[a]) / eps;
                }
            }
            for a in 0..kb {
                for b in 0..kb {
                    hess[a][b] = 0.5 * (hess[a][b] + hess[b][a]);
                }
                hess[a][a] -= 1e-8;
            }
            let step = solve_small(hess, g.clone());
            let cur = total_ell(&d, &c, &f, &r, nodes, n_items, n_raters, n_cat);
            let mut al = 1.0f64;
            let mut accepted = false;
            let mut max_step = 0.0f64;
            for _ in 0..24 {
                let cand: Vec<f64> = (0..kb).map(|m| f[m] - al * step[m]).collect();
                if total_ell(&d, &c, &cand, &r, nodes, n_items, n_raters, n_cat) >= cur - 1e-12 {
                    max_step = (0..kb).map(|m| (al * step[m]).abs()).fold(0.0, f64::max);
                    f = cand;
                    accepted = true;
                    break;
                }
                al *= 0.5;
            }
            if !accepted || max_step < 1e-9 {
                break;
            }
        }

        // Re-center: f_m -> f_m - cf shifts T_k by -k*cf, compensated by
        // d_i -> d_i + cf (psi_k regains -k*cf through the k*(d+c) term).
        let cf = f.iter().sum::<f64>() / kb as f64;
        for fm in f.iter_mut() {
            *fm -= cf;
        }
        for di in d.iter_mut() {
            *di += cf;
        }
        // c_j -> c_j - cc, d_i -> d_i + cc leaves every location d_i + c_j fixed.
        let cc = c.iter().sum::<f64>() / n_raters as f64;
        for cj in c.iter_mut() {
            *cj -= cc;
        }
        for di in d.iter_mut() {
            *di += cc;
        }
    }

    // Final person EAP pass at the returned parameters.
    let item_lp = pair_logprobs(&d, &c, &f, nodes, n_items, n_raters, n_cat);
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0f64;
    let mut log_node = vec![0.0f64; qn];
    for p in 0..n_persons {
        log_node[..qn].copy_from_slice(&log_w[..qn]);
        for pair in 0..n_pairs {
            if !is_obs(p, pair) {
                continue;
            }
            let yc = y[p * n_pairs + pair];
            for nd in 0..qn {
                log_node[nd] += item_lp[pair][nd * n_cat + yc];
            }
        }
        let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for nd in 0..qn {
            denom += (log_node[nd] - mx).exp();
        }
        final_ll += mx + denom.ln();
        let mut m = 0.0f64;
        for (nd, &node) in nodes.iter().enumerate() {
            m += (log_node[nd] - mx).exp() / denom * node;
        }
        theta[p] = m;
    }
    if !converged {
        loglik_trace.push(final_ll);
    }

    Ok(FacetsResult {
        item_difficulty: d,
        rater_severity: c,
        thresholds: f,
        theta,
        loglik_trace,
        n_iter: it,
        converged,
        connected,
        n_parameters: n_items + (n_raters - 1) + (n_cat - 2),
    })
}

/// Per-pair RSM cell log-prob tables: `out[i*n_raters + j][nd*n_cat + k]`.
fn pair_logprobs(
    d: &[f64],
    c: &[f64],
    f: &[f64],
    nodes: &[f64],
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
) -> Vec<Vec<f64>> {
    let qn = nodes.len();
    let mut out = vec![vec![0.0f64; qn * n_cat]; n_items * n_raters];
    for i in 0..n_items {
        for j in 0..n_raters {
            let pair = i * n_raters + j;
            for (nd, &theta) in nodes.iter().enumerate() {
                let lp = rsm_logprobs(theta, d[i] + c[j], f);
                out[pair][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
        }
    }
    out
}

/// Accumulate the location gradient/Hessian terms of one (pair) count block:
/// `g += -sum_{nd,k} k*(r - n*P)`, `h += -sum_nd n*Var(score)`. Shared by the
/// `d_i` and `c_j` Newton steps (`d ln P_k / d location = -k + E[score]`).
fn location_score_terms(
    location: f64,
    f: &[f64],
    r_pair: &[f64],
    nodes: &[f64],
    n_cat: usize,
    g: &mut f64,
    h: &mut f64,
) {
    for (nd, &theta) in nodes.iter().enumerate() {
        let lp = rsm_logprobs(theta, location, f);
        let mut n = 0.0f64;
        for k in 0..n_cat {
            n += r_pair[nd * n_cat + k];
        }
        if n <= 0.0 {
            continue;
        }
        let (mut e1, mut e2) = (0.0f64, 0.0f64);
        for k in 0..n_cat {
            let pk = lp[k].exp();
            let kf = k as f64;
            e1 += kf * pk;
            e2 += kf * kf * pk;
            *g += -kf * (r_pair[nd * n_cat + k] - n * pk);
        }
        *h += -n * (e2 - e1 * e1);
    }
}

/// Expected complete-data log-lik of the cells involving item `i` (its row of
/// rater pairs) — the objective ascended by the `d_i` line search.
#[allow(clippy::too_many_arguments)]
fn item_ell_d(
    i: usize,
    d: &[f64],
    c: &[f64],
    f: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_raters: usize,
    n_cat: usize,
) -> f64 {
    (0..n_raters)
        .map(|j| pair_ell(d[i] + c[j], f, &r[i * n_raters + j], nodes, n_cat))
        .sum()
}

/// Expected complete-data log-lik of the cells involving rater `j` — the
/// objective ascended by the `c_j` line search.
#[allow(clippy::too_many_arguments)]
fn rater_ell_c(
    j: usize,
    d: &[f64],
    c: &[f64],
    f: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
) -> f64 {
    (0..n_items)
        .map(|i| pair_ell(d[i] + c[j], f, &r[i * n_raters + j], nodes, n_cat))
        .sum()
}

/// `sum_nd sum_k r[nd][k] * log P(k | theta_nd; location, f)` for one pair.
fn pair_ell(location: f64, f: &[f64], r_pair: &[f64], nodes: &[f64], n_cat: usize) -> f64 {
    let mut acc = 0.0f64;
    for (nd, &theta) in nodes.iter().enumerate() {
        let lp = rsm_logprobs(theta, location, f);
        for k in 0..n_cat {
            let rc = r_pair[nd * n_cat + k];
            if rc != 0.0 {
                acc += rc * lp[k];
            }
        }
    }
    acc
}

/// Total expected complete-data log-lik over all pairs (for the shared-`f`
/// line search).
#[allow(clippy::too_many_arguments)]
fn total_ell(
    d: &[f64],
    c: &[f64],
    f: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
) -> f64 {
    let mut acc = 0.0f64;
    for i in 0..n_items {
        for j in 0..n_raters {
            acc += pair_ell(d[i] + c[j], f, &r[i * n_raters + j], nodes, n_cat);
        }
    }
    acc
}

/// Gradient of the expected complete-data objective w.r.t. the common
/// thresholds: `g_m = -sum_{i,j,nd} sum_{k>=m} (r - n*P)` (0-indexed `m` for
/// `f_{m+1}`); suffix-residual form as in `rsm::tau_gradient`.
#[allow(clippy::too_many_arguments)]
fn f_gradient(
    f: &[f64],
    d: &[f64],
    c: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_items: usize,
    n_raters: usize,
    n_cat: usize,
) -> Vec<f64> {
    let kb = f.len();
    let mut g = vec![0.0f64; kb];
    for i in 0..n_items {
        for j in 0..n_raters {
            let pair = i * n_raters + j;
            for (nd, &theta) in nodes.iter().enumerate() {
                let lp = rsm_logprobs(theta, d[i] + c[j], f);
                let mut n = 0.0f64;
                for k in 0..n_cat {
                    n += r[pair][nd * n_cat + k];
                }
                if n <= 0.0 {
                    continue;
                }
                let mut suffix = 0.0f64;
                for k in (1..n_cat).rev() {
                    suffix += r[pair][nd * n_cat + k] - n * lp[k].exp();
                    g[k - 1] += -suffix;
                }
            }
        }
    }
    g
}

/// Design-linking flag: union-find over facet elements (`n_items` item nodes,
/// then `n_raters` rater nodes), joining every element observed for the same
/// person. `true` iff all items and raters form one component. Persons anchor
/// components to the trait scale only through `theta ~ N(0,1)` (module docs).
fn design_connected(
    n_persons: usize,
    n_items: usize,
    n_raters: usize,
    is_obs: &dyn Fn(usize, usize) -> bool,
) -> bool {
    let n = n_items + n_raters;
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for p in 0..n_persons {
        let mut first: Option<usize> = None;
        for i in 0..n_items {
            for j in 0..n_raters {
                if !is_obs(p, i * n_raters + j) {
                    continue;
                }
                for node in [i, n_items + j] {
                    match first {
                        None => first = Some(node),
                        Some(anchor) => {
                            let (ra, rb) = (find(&mut parent, anchor), find(&mut parent, node));
                            parent[rb] = ra;
                        }
                    }
                }
            }
        }
    }
    let root = find(&mut parent, 0);
    (1..n).all(|x| find(&mut parent, x) == root)
}

#[cfg(test)]
#[path = "../../../tests/unit/facets_tests.rs"]
mod tests;
