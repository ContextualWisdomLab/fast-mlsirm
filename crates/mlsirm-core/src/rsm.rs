//! Rating Scale Model (Andrich, 1978) by marginal-ML EM.
//!
//! The RSM is the Rasch-family polytomous model for items sharing a common rating
//! scale (e.g. Likert): every item has its own location `delta_i`, but the category
//! *threshold* structure `tau_1..tau_{K-1}` is **shared across all items**. The
//! adjacent-category log-odds are
//!
//! ```text
//! ln[ P(X_ij = k | theta) / P(X_ij = k-1 | theta) ] = theta_j - delta_i - tau_k,
//! ```
//!
//! for `k = 1..K-1`, so the cumulative predictor is
//! `psi_k(theta) = k*theta - k*delta_i - T_k` with `T_k = sum_{m<=k} tau_m` and
//! `psi_0 = 0`; `P(X=k|theta) = softmax_k(psi)`. This is exactly the GPCM cell
//! ([`crate::poly::gpcm_logprobs`]) with slope 1, scores `0..K-1`, and the structured
//! intercept `intercepts[k] = -k*delta_i - T_k`, which this module reuses.
//!
//! Distinct from the per-item PCM/GPCM (`poly.rs`, `mixed.rs`), whose category
//! thresholds are free per item; the RSM ties them to one common set. The trait
//! `theta ~ N(0,1)` fixes the scale; the remaining shift redundancy — the model is
//! invariant under `tau_m -> tau_m - c`, `delta_i -> delta_i + c` — is removed by
//! centering `sum_m tau_m = 0`. Fit by ECM: a per-item Newton for `delta`, then a
//! joint Newton for the common `tau` aggregated over items, then re-centering.
//!
//! # References (APA 7th ed.)
//! Andrich, D. (1978). A rating formulation for ordered response categories.
//!   *Psychometrika, 43*(4), 561-573. https://doi.org/10.1007/BF02293814

use crate::poly::{gpcm_logprobs, solve_small};

const RSM_MAX_CAT: usize = 64;
const RSM_MAX_ITER: usize = 100_000;
const RSM_MAX_COUNT_CELLS: usize = 60_000_000;

/// Fitted rating scale model (Andrich, 1978). `item_location` is the per-item
/// `delta_i`; `thresholds` the `K-1` common category thresholds `tau_k` (centered,
/// `sum = 0`); `theta` the per-person EAP trait.
#[derive(Clone, Debug)]
pub struct RsmResult {
    pub item_location: Vec<f64>,
    pub thresholds: Vec<f64>,
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `n_items + (n_cat - 1) - 1 = n_items + n_cat - 2`.
    pub n_parameters: usize,
}

/// RSM category log-probabilities at one node: `log P(X = k | theta)` for
/// `k = 0..K-1`, given item location `delta` and the `K-1` common thresholds `tau`.
/// Equals the GPCM cell with slope 1 and `intercepts[k] = -k*delta - T_k`.
pub fn rsm_logprobs(theta: f64, delta: f64, tau: &[f64]) -> Vec<f64> {
    let kb = tau.len(); // K-1
    let scores: Vec<f64> = (0..=kb).map(|c| c as f64).collect();
    let mut intercepts = vec![0.0f64; kb + 1];
    let mut t = 0.0f64;
    for k in 1..=kb {
        t += tau[k - 1]; // T_k
        intercepts[k] = -(k as f64) * delta - t;
    }
    gpcm_logprobs(theta, &scores, &intercepts)
}

/// Fit the rating scale model (Andrich, 1978) by marginal-ML EM. `y` is
/// `n_persons * n_items` row-major categories `0..n_cat-1`; `observed` marks
/// non-missing cells (dropped under MAR; `None` = all observed). Ability
/// `theta ~ N(0,1)` on the `q_theta`-node Gauss-Hermite grid.
#[allow(clippy::too_many_arguments)]
pub fn fit_rsm(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<RsmResult, String> {
    if !(2..=RSM_MAX_CAT).contains(&n_cat) {
        return Err(format!("n_cat must be in 2..={RSM_MAX_CAT}"));
    }
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(1..=RSM_MAX_ITER).contains(&max_iter) {
        return Err(format!("max_iter must be in 1..={RSM_MAX_ITER}"));
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
    for (idx, &c) in y.iter().enumerate() {
        if observed.map_or(true, |o| o[idx]) && c >= n_cat {
            return Err("response category out of range 0..n_cat-1".into());
        }
    }
    for i in 0..n_items {
        if !(0..n_persons).any(|p| observed.map_or(true, |o| o[p * n_items + i])) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let count_cells = nodes
        .len()
        .checked_mul(n_items)
        .and_then(|cells| cells.checked_mul(n_cat))
        .ok_or_else(|| "node * item * category count-table size overflows usize".to_string())?;
    if count_cells > RSM_MAX_COUNT_CELLS {
        return Err(format!(
            "count table {count_cells} cells exceeds the cap {RSM_MAX_COUNT_CELLS}"
        ));
    }
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let qn = nodes.len();
    let kb = n_cat - 1; // number of thresholds

    // Init: item locations from the item mean category, thresholds at 0.
    let mut delta = vec![0.0f64; n_items];
    let mut tau = vec![0.0f64; kb];
    for i in 0..n_items {
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for p in 0..n_persons {
            if is_obs(p, i) {
                s += y[p * n_items + i] as f64;
                c += 1.0;
            }
        }
        if c > 0.0 {
            // higher mean category -> easier item -> lower delta
            let mean = s / c / kb as f64; // in [0,1]
            delta[i] = ((1.0 - mean).clamp(0.02, 0.98) / mean.clamp(0.02, 0.98)).ln();
        }
    }

    let mut ll;
    let mut it = 0usize;
    let mut converged = false;
    let mut loglik_trace: Vec<f64> = Vec::new();
    // Expected category counts r[i][node][k].
    let cell = qn;

    while it < max_iter {
        // Per-item cell log-probs at each node.
        let mut item_lp = vec![vec![0.0f64; cell * n_cat]; n_items];
        for i in 0..n_items {
            for (nd, &theta) in nodes.iter().enumerate() {
                let lp = rsm_logprobs(theta, delta[i], &tau);
                item_lp[i][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
            }
        }
        // E-step: posteriors -> expected counts.
        let mut r = vec![vec![0.0f64; cell * n_cat]; n_items];
        ll = 0.0;
        let mut log_node = vec![0.0f64; qn];
        for p in 0..n_persons {
            log_node[..qn].copy_from_slice(&log_w[..qn]);
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
            let mut denom = 0.0f64;
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
                    r[i][nd * n_cat + yc] += (log_node[nd] - mx).exp() / denom;
                }
            }
        }

        loglik_trace.push(ll);
        // Converge check before the M-step so the returned params match the trace endpoint.
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

        // CM-1: per-item Newton on delta_i (tau fixed), with a backtracking line search
        // on the item objective so the step never lowers it (keeps ECM monotone).
        //   g = -sum_nd sum_k k*(r - n*P);  h = -sum_nd n*Var_nd(score) < 0.
        for i in 0..n_items {
            for _ in 0..25 {
                let (mut g, mut h) = (0.0f64, 0.0f64);
                for (nd, &theta) in nodes.iter().enumerate() {
                    let lp = rsm_logprobs(theta, delta[i], &tau);
                    let mut n = 0.0f64;
                    for k in 0..n_cat {
                        n += r[i][nd * n_cat + k];
                    }
                    // Validation guarantees an observed response for every item;
                    // posterior node weights are strictly positive, hence n > 0.
                    let (mut e1, mut e2) = (0.0f64, 0.0f64);
                    for k in 0..n_cat {
                        let pk = lp[k].exp();
                        let kf = k as f64;
                        e1 += kf * pk;
                        e2 += kf * kf * pk;
                        g += -(kf) * (r[i][nd * n_cat + k] - n * pk);
                    }
                    h += -n * (e2 - e1 * e1); // -Var(score)
                }
                // With at least two categories and positive expected count, the
                // score variance is positive and the Hessian is strictly negative.
                let step = g / h;
                let cur = item_ell(delta[i], &tau, &r[i], &nodes, n_cat);
                let mut al = 1.0f64;
                let mut accepted = false;
                for _ in 0..24 {
                    let cand = delta[i] - al * step;
                    if item_ell(cand, &tau, &r[i], &nodes, n_cat) >= cur - 1e-12 {
                        delta[i] = cand;
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

        // CM-2: joint Newton on the common tau (delta fixed), aggregated over items.
        //   g_m = -sum_i sum_nd sum_{k>=m} (r - n*P);  Hessian by finite differences of g.
        for _ in 0..25 {
            let g = tau_gradient(&tau, &delta, &r, &nodes, n_items, n_cat);
            let mut hess = vec![vec![0.0f64; kb]; kb];
            let eps = 1e-5;
            for j in 0..kb {
                let mut tp = tau.clone();
                tp[j] += eps;
                let gj = tau_gradient(&tp, &delta, &r, &nodes, n_items, n_cat);
                for a in 0..kb {
                    hess[a][j] = (gj[a] - g[a]) / eps;
                }
            }
            for a in 0..kb {
                for b in 0..kb {
                    hess[a][b] = 0.5 * (hess[a][b] + hess[b][a]);
                }
                hess[a][a] -= 1e-8; // keep the maximizer's Hessian negative definite
            }
            let step = solve_small(hess, g.clone());
            // Backtracking on the aggregate objective so the shared-tau step is monotone.
            let cur = total_ell(&delta, &tau, &r, &nodes, n_items, n_cat);
            let mut al = 1.0f64;
            let mut accepted = false;
            let mut max_step = 0.0f64;
            for _ in 0..24 {
                let cand: Vec<f64> = (0..kb).map(|j| tau[j] - al * step[j]).collect();
                if total_ell(&delta, &cand, &r, &nodes, n_items, n_cat) >= cur - 1e-12 {
                    max_step = (0..kb).map(|j| (al * step[j]).abs()).fold(0.0, f64::max);
                    tau = cand;
                    accepted = true;
                    break;
                }
                al *= 0.5;
            }
            if !accepted || max_step < 1e-9 {
                break;
            }
        }

        // Re-center tau (sum = 0), shifting the level into the item locations. The
        // model P(X=k|theta) is invariant under tau_m -> tau_m - c, delta_i -> delta_i + c
        // (then psi_k = k*theta - k*(delta+c) - (T_k - k*c) = k*theta - k*delta - T_k).
        let c = tau.iter().sum::<f64>() / kb as f64;
        for tm in tau.iter_mut() {
            *tm -= c;
        }
        for di in delta.iter_mut() {
            *di += c;
        }
    }

    // Final person EAP pass at the returned parameters; recompute the cell tables.
    let mut item_lp = vec![vec![0.0f64; cell * n_cat]; n_items];
    for i in 0..n_items {
        for (nd, &theta) in nodes.iter().enumerate() {
            let lp = rsm_logprobs(theta, delta[i], &tau);
            item_lp[i][nd * n_cat..(nd + 1) * n_cat].copy_from_slice(&lp);
        }
    }
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0f64;
    let mut log_node = vec![0.0f64; qn];
    for p in 0..n_persons {
        log_node[..qn].copy_from_slice(&log_w[..qn]);
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
    // On convergence the trace endpoint already holds the loglik at the returned
    // params (checked before the M-step); on a max-iter exit the last M-step moved
    // them, so record the loglik of the parameters actually returned.
    if !converged {
        loglik_trace.push(final_ll);
    }

    Ok(RsmResult {
        item_location: delta,
        thresholds: tau,
        theta,
        loglik_trace,
        n_iter: it,
        converged,
        n_parameters: n_items + n_cat - 2,
    })
}

/// Expected complete-data log-likelihood of one item over the nodes,
/// `sum_nd sum_k r[i][nd][k] * log P(k | theta_nd; delta, tau)` — the objective the
/// per-item `delta` and the common `tau` conditional-maximization steps ascend
/// (used by their backtracking line searches to guarantee EM monotonicity).
fn item_ell(delta: f64, tau: &[f64], r_i: &[f64], nodes: &[f64], n_cat: usize) -> f64 {
    let mut acc = 0.0f64;
    for (nd, &theta) in nodes.iter().enumerate() {
        let lp = rsm_logprobs(theta, delta, tau);
        for k in 0..n_cat {
            let rc = r_i[nd * n_cat + k];
            if rc != 0.0 {
                acc += rc * lp[k];
            }
        }
    }
    acc
}

/// Total expected complete-data item log-likelihood over all items (for the shared
/// `tau` line search).
fn total_ell(
    delta: &[f64],
    tau: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_items: usize,
    n_cat: usize,
) -> f64 {
    (0..n_items)
        .map(|i| item_ell(delta[i], tau, &r[i], nodes, n_cat))
        .sum()
}

/// Gradient of the expected complete-data objective w.r.t. the common thresholds:
/// `g_m = -sum_i sum_nd sum_{k>=m+1} (r - n*P)` (0-indexed `m` for `tau_{m+1}`).
fn tau_gradient(
    tau: &[f64],
    delta: &[f64],
    r: &[Vec<f64>],
    nodes: &[f64],
    n_items: usize,
    n_cat: usize,
) -> Vec<f64> {
    let kb = tau.len();
    let mut g = vec![0.0f64; kb];
    for i in 0..n_items {
        for (nd, &theta) in nodes.iter().enumerate() {
            let lp = rsm_logprobs(theta, delta[i], tau);
            let mut n = 0.0f64;
            for k in 0..n_cat {
                n += r[i][nd * n_cat + k];
            }
            if n <= 0.0 {
                continue;
            }
            // resid[k] = r[k] - n*P[k]; g_{tau_{m}} = -sum_{k>=m} resid[k], m = 1..K-1.
            // Accumulate the suffix sum of residuals.
            let mut suffix = 0.0f64;
            for k in (1..n_cat).rev() {
                suffix += r[i][nd * n_cat + k] - n * lp[k].exp();
                g[k - 1] += -suffix; // tau_k corresponds to g index k-1
            }
        }
    }
    g
}

#[cfg(test)]
#[path = "../../../tests/unit/rsm_tests.rs"]
mod tests;
