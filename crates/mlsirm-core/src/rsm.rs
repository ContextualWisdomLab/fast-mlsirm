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
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if max_iter < 1 {
        return Err("max_iter must be >= 1".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and > 0".into());
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
    let (nodes, weights) =
        crate::quadrature::gh_rule(q_theta).ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
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
                    if n <= 0.0 {
                        continue;
                    }
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
                if h.abs() < 1e-12 {
                    break;
                }
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
fn total_ell(delta: &[f64], tau: &[f64], r: &[Vec<f64>], nodes: &[f64], n_items: usize, n_cat: usize) -> f64 {
    (0..n_items).map(|i| item_ell(delta[i], tau, &r[i], nodes, n_cat)).sum()
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
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.f64().max(1e-12);
            let u2 = self.f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    fn log_sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            -(-x).exp().ln_1p()
        } else {
            x - x.exp().ln_1p()
        }
    }

    /// Draw an RSM category for ability `theta`, location `delta`, thresholds `tau`.
    fn draw_rsm(theta: f64, delta: f64, tau: &[f64], u: f64) -> usize {
        let lp = rsm_logprobs(theta, delta, tau);
        let mut cum = 0.0;
        for (k, l) in lp.iter().enumerate() {
            cum += l.exp();
            if u < cum {
                return k;
            }
        }
        lp.len() - 1
    }

    #[test]
    fn rsm_k2_reduces_to_rasch() {
        // K=2: single threshold, centered to 0, so P(X=1) = sigmoid(theta - delta).
        let tau = [0.0f64];
        for ti in -20..=20 {
            for di in -10..=10 {
                let theta = ti as f64 * 0.3;
                let delta = di as f64 * 0.4;
                let lp = rsm_logprobs(theta, delta, &tau);
                assert!((lp[0] - log_sigmoid(-(theta - delta))).abs() < 1e-12);
                assert!((lp[1] - log_sigmoid(theta - delta)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn rsm_probs_sum_to_one() {
        let tau = [0.7f64, -0.2, -0.5]; // K=4
        for ti in -20..=20 {
            let theta = ti as f64 * 0.3;
            let s: f64 = rsm_logprobs(theta, 0.3, &tau).iter().map(|l| l.exp()).sum();
            assert!((s - 1.0).abs() < 1e-12, "sum {s}");
        }
    }

    #[test]
    fn rsm_recovers_params() {
        let (n_items, n_cat, n) = (12usize, 5usize, 2500usize);
        let delta_true: Vec<f64> = (0..n_items).map(|i| -1.2 + 0.2 * i as f64).collect();
        let tau_true = vec![0.9f64, 0.2, -0.3, -0.8]; // sum = 0
        let mut rng = Lcg(1978);
        let mut y = vec![0usize; n * n_items];
        let mut thetas = vec![0.0f64; n];
        for p in 0..n {
            let theta = rng.normal();
            thetas[p] = theta;
            for i in 0..n_items {
                y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
            }
        }
        let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-7).unwrap();
        assert!(res.converged);
        // ECM ascends the marginal loglik monotonically (backtracked M-steps).
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "loglik decreased {} -> {}", w[0], w[1]);
        }
        assert_eq!(res.n_parameters, n_items + n_cat - 2);
        assert!((res.thresholds.iter().sum::<f64>()).abs() < 1e-6, "tau not centered");
        assert!(rmse(&res.item_location, &delta_true) < 0.15, "delta RMSE {}", rmse(&res.item_location, &delta_true));
        assert!(rmse(&res.thresholds, &tau_true) < 0.12, "tau RMSE {}", rmse(&res.thresholds, &tau_true));
        assert!(corr(&res.theta, &thetas) > 0.85, "theta corr {}", corr(&res.theta, &thetas));
    }

    /// Data generated with NON-centered thresholds must be recovered as the centered
    /// equivalent (tau - mean, delta + mean). This exercises the re-centering sign:
    /// a wrong sign shifts the model and breaks recovery.
    #[test]
    fn rsm_centers_noncentered_truth() {
        let (n_items, n_cat, n) = (10usize, 4usize, 2500usize);
        let delta_gen: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.15 * i as f64).collect();
        let tau_gen = vec![1.0f64, 0.5, -0.3]; // sum = 1.2, NOT centered
        let shift = tau_gen.iter().sum::<f64>() / (n_cat - 1) as f64; // 0.4
        let tau_expect: Vec<f64> = tau_gen.iter().map(|t| t - shift).collect();
        let delta_expect: Vec<f64> = delta_gen.iter().map(|d| d + shift).collect();
        let mut rng = Lcg(4242);
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            let theta = rng.normal();
            for i in 0..n_items {
                y[p * n_items + i] = draw_rsm(theta, delta_gen[i], &tau_gen, rng.f64());
            }
        }
        let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-7).unwrap();
        assert!(res.converged);
        assert!((res.thresholds.iter().sum::<f64>()).abs() < 1e-6);
        assert!(rmse(&res.thresholds, &tau_expect) < 0.12, "tau RMSE {}", rmse(&res.thresholds, &tau_expect));
        assert!(rmse(&res.item_location, &delta_expect) < 0.15, "delta RMSE {}", rmse(&res.item_location, &delta_expect));
    }

    #[test]
    fn rsm_handles_missing_data() {
        let (n_items, n_cat, n) = (8usize, 4usize, 800usize);
        let delta_true = vec![-0.5f64, 0.0, 0.5, -0.3, 0.3, -0.6, 0.6, 0.1];
        let tau_true = vec![0.5f64, 0.0, -0.5];
        let mut rng = Lcg(55);
        let mut y = vec![0usize; n * n_items];
        let mut observed = vec![true; n * n_items];
        for p in 0..n {
            let theta = rng.normal();
            for i in 0..n_items {
                y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
                if rng.f64() < 0.15 {
                    observed[p * n_items + i] = false;
                }
            }
        }
        let res = fit_rsm(&y, Some(&observed), n, n_items, n_cat, 21, 400, 1e-6).unwrap();
        assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rsm_validate_rejects_malformed() {
        assert!(fit_rsm(&[0, 1], None, 1, 2, 1, 21, 10, 1e-6).is_err()); // n_cat<2
        assert!(fit_rsm(&[0, 1, 2], None, 1, 2, 3, 21, 10, 1e-6).is_err()); // wrong len
        assert!(fit_rsm(&[0, 9], None, 1, 2, 3, 21, 10, 1e-6).is_err()); // category out of range
        assert!(fit_rsm(&[0, 1, 0, 1], None, 2, 2, 2, 99, 10, 1e-6).is_err()); // bad q
        assert!(fit_rsm(&[], None, 0, 1, 2, 21, 10, 1e-6).is_err()); // no persons
        assert!(fit_rsm(&[], None, 1, 0, 2, 21, 10, 1e-6).is_err()); // no items
        assert!(fit_rsm(&[0, 1], None, 1, 2, 2, 21, 0, 1e-6).is_err()); // no iterations
        assert!(fit_rsm(&[0, 1], None, 1, 2, 2, 21, 10, f64::INFINITY).is_err());
        let observed = [true, false, true, false];
        assert!(fit_rsm(&[0, 0, 1, 0], Some(&observed), 2, 2, 2, 21, 10, 1e-6).is_err());
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_rsm_recovery_500() {
        let (n_items, n_cat, n, reps) = (12usize, 5usize, 1000usize, 500usize);
        let delta_true: Vec<f64> = (0..n_items).map(|i| -1.1 + 0.2 * i as f64).collect();
        let tau_true = vec![0.9f64, 0.2, -0.3, -0.8];
        for &skew in [false, true].iter() {
            let (mut rd, mut rt, mut bd, mut bt, mut nconv, mut tcorr) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0.0f64);
            for rep in 0..reps {
                let mut rng = Lcg(
                    0xB5297A4Du64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15),
                );
                let mut y = vec![0usize; n * n_items];
                let mut thetas = vec![0.0f64; n];
                for p in 0..n {
                    let theta = if skew {
                        let mut c = 0.0;
                        for _ in 0..3 {
                            let g = rng.normal();
                            c += g * g;
                        }
                        (c - 3.0) / (6.0_f64).sqrt()
                    } else {
                        rng.normal()
                    };
                    thetas[p] = theta;
                    for i in 0..n_items {
                        y[p * n_items + i] = draw_rsm(theta, delta_true[i], &tau_true, rng.f64());
                    }
                }
                let res = fit_rsm(&y, None, n, n_items, n_cat, 41, 500, 1e-6).unwrap();
                if res.converged {
                    nconv += 1;
                }
                rd += rmse(&res.item_location, &delta_true) / reps as f64;
                rt += rmse(&res.thresholds, &tau_true) / reps as f64;
                bd += (res.item_location.iter().sum::<f64>() - delta_true.iter().sum::<f64>())
                    / n_items as f64
                    / reps as f64;
                bt += (res.thresholds.iter().sum::<f64>()) / reps as f64;
                tcorr += corr(&res.theta, &thetas) / reps as f64;
            }
            println!(
                "[RSM MC skew={skew}] reps={reps} conv={:.2} RMSE(delta)={:.3} RMSE(tau)={:.3} \
                 bias(delta)={:.3} sum(tau)={:.4} theta-corr={:.3}",
                nconv as f64 / reps as f64,
                rd,
                rt,
                bd,
                bt,
                tcorr
            );
            assert!(rd < 0.12, "RMSE(delta) {rd} skew={skew}");
            assert!(rt < 0.1, "RMSE(tau) {rt} skew={skew}");
            assert!(tcorr > 0.85, "theta corr {tcorr} skew={skew}");
        }
    }
}
