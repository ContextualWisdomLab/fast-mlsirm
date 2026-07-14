//! Latent-space polytomous item response model (polytomous LSIRM) by marginal
//! EM — the Rust compute path for the GRM/GPCM cell embedded in an interaction
//! map. Unidimensional trait `theta` with a `latent_dim`-dimensional latent
//! space; the person latent position `xi` is integrated over a tensor
//! Gauss-Hermite grid and the item position `zeta_i` is estimated. This is the
//! `fixed_gamma = 1` identification of Go et al. (2024) lsirm12pl (the distance
//! weight is fixed to standardize the map scale).
//!
//! Fully additive: reuses the [`crate::poly`] cells/gradients and the exact
//! `d eta / d zeta` distance derivative from the binary M-step
//! (`marginal.rs`), but touches neither the binary estimator nor the GPU.
//!
//! `base_i(theta, xi) = a_i * theta - ||xi - zeta_i||` (distance interaction,
//! gamma = 1); the polytomous cell turns `base` into category probabilities.
//! The item M-step reuses the binary chain rule: the per-node `g_base` (the
//! category-weighted residual from `poly::*_node_gradient`) multiplies
//! `d base / d a = a*theta` and `d base / d zeta_k = (xi_k - zeta_k)/dist`.

use crate::poly::{
    gpcm_logprobs, gpcm_node_gradient, grm_logprobs, grm_node_gradient, solve_small, PolyModel,
};

/// Result of [`fit_poly_lsirm`]. `zeta` is `n_items * latent_dim` item positions
/// (identified up to rotation/reflection/translation — compare via distances).
pub struct PolyLsirmFit {
    pub slope: Vec<f64>,
    pub cat_params: Vec<Vec<f64>>,
    pub zeta: Vec<f64>,
    pub loglik: f64,
    pub n_iter: usize,
}

/// Tensor Gauss-Hermite grid for a `latent_dim`-dimensional standard normal:
/// returns `(grid [n_xi * latent_dim], log_weights [n_xi])`.
fn xi_tensor_grid(q_xi: usize, latent_dim: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let (nodes, weights) =
        crate::quadrature::gh_rule(q_xi).ok_or_else(|| format!("unsupported q_xi {q_xi}"))?;
    let q = nodes.len();
    let n_xi = q.checked_pow(latent_dim as u32).ok_or("xi grid too large")?;
    if n_xi > 200_000 {
        return Err("q_xi ** latent_dim exceeds the tensor-grid limit".into());
    }
    let mut grid = vec![0.0_f64; n_xi * latent_dim];
    let mut logw = vec![0.0_f64; n_xi];
    for (idx, lw) in logw.iter_mut().enumerate() {
        let mut rem = idx;
        let mut acc = 0.0_f64;
        for k in 0..latent_dim {
            let j = rem % q;
            rem /= q;
            grid[idx * latent_dim + k] = nodes[j];
            acc += weights[j].ln();
        }
        *lw = acc;
    }
    Ok((grid, logw))
}

fn poly_cell(base: f64, model: PolyModel, cat: &[f64], n_cat: usize) -> Vec<f64> {
    match model {
        PolyModel::Grm => grm_logprobs(base, cat),
        PolyModel::Gpcm => {
            let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
            let mut ic = vec![0.0_f64; n_cat];
            ic[1..].copy_from_slice(cat);
            gpcm_logprobs(base, &scores, &ic)
        }
    }
}

/// Per-node `(category-parameter gradient, g_base)` for the chosen cell.
fn poly_cat_grad(base: f64, model: PolyModel, cat: &[f64], counts: &[f64]) -> (Vec<f64>, f64) {
    match model {
        PolyModel::Grm => {
            let (gb, gt) = grm_node_gradient(base, cat, counts);
            (gt, gb)
        }
        PolyModel::Gpcm => {
            let k = counts.len();
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut ic = vec![0.0_f64; k];
            ic[1..].copy_from_slice(cat);
            let (gi, gb, _gs) = gpcm_node_gradient(base, &scores, &ic, counts);
            (gi, gb)
        }
    }
}

#[allow(clippy::too_many_arguments)]
struct ItemCtx<'a> {
    model: PolyModel,
    n_cat: usize,
    latent_dim: usize,
    eps: f64,
    theta: &'a [f64],
    xi_grid: &'a [f64],
    n_xi: usize,
    rbar_i: &'a [f64], // [q_theta * n_xi * n_cat] expected counts
    lambda_alpha: f64,
    mu_alpha: f64,
    lambda_zeta: f64,
}

/// Negative penalized expected complete-data objective and its gradient for one
/// item. `params = [log_a, cat_1..cat_{K-1}, zeta_1..zeta_L]`.
fn item_neg_ll_grad(params: &[f64], c: &ItemCtx) -> (f64, Vec<f64>) {
    let a = params[0].exp();
    let cat = &params[1..c.n_cat];
    let zeta = &params[c.n_cat..c.n_cat + c.latent_dim];
    let mut ll = 0.0_f64;
    let mut g = vec![0.0_f64; params.len()];
    for (t, &theta_t) in c.theta.iter().enumerate() {
        for x in 0..c.n_xi {
            let node = t * c.n_xi + x;
            let counts = &c.rbar_i[node * c.n_cat..(node + 1) * c.n_cat];
            let ncount: f64 = counts.iter().sum();
            if ncount <= 0.0 {
                continue;
            }
            let xi = &c.xi_grid[x * c.latent_dim..(x + 1) * c.latent_dim];
            let mut dist2 = c.eps;
            for k in 0..c.latent_dim {
                let dd = xi[k] - zeta[k];
                dist2 += dd * dd;
            }
            let dist = dist2.sqrt();
            let base = a * theta_t - dist; // gamma = 1
            let lp = poly_cell(base, c.model, cat, c.n_cat);
            ll += counts.iter().zip(&lp).map(|(cc, l)| cc * l).sum::<f64>();
            let (g_cat, g_base) = poly_cat_grad(base, c.model, cat, counts);
            g[0] += g_base * (a * theta_t); // d base / d log_a
            for (m, gm) in g_cat.iter().enumerate() {
                g[1 + m] += gm;
            }
            for k in 0..c.latent_dim {
                let deta = (xi[k] - zeta[k]) / dist; // d base / d zeta_k
                g[1 + (c.n_cat - 1) + k] += g_base * deta;
            }
        }
    }
    // MAP penalties (Gaussian priors on alpha and the item positions)
    ll -= 0.5 * c.lambda_alpha * (params[0] - c.mu_alpha).powi(2);
    g[0] -= c.lambda_alpha * (params[0] - c.mu_alpha);
    for k in 0..c.latent_dim {
        let z = zeta[k];
        ll -= 0.5 * c.lambda_zeta * z * z;
        g[1 + (c.n_cat - 1) + k] -= c.lambda_zeta * z;
    }
    (-ll, g.iter().map(|v| -v).collect())
}

/// Backtracked numerical-Hessian Newton M-step for one item.
fn m_step_item(mut params: Vec<f64>, c: &ItemCtx, n_newton: usize) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g) = item_neg_ll_grad(&params, c);
        let h = 1e-5;
        let mut hess = vec![vec![0.0_f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f, gj) = item_neg_ll_grad(&pj, c);
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        for r in 0..np {
            for col in 0..np {
                hess[r][col] = 0.5 * (hess[r][col] + hess[col][r]);
            }
            hess[r][r] += 1e-6;
        }
        let step = solve_small(hess, g.clone());
        // backtracking: accept a decrease in the negative objective
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..25 {
            let cand: Vec<f64> = (0..np).map(|j| params[j] - alpha * step[j]).collect();
            let (fc, _) = item_neg_ll_grad(&cand, c);
            if fc < f0 - 1e-10 {
                params = cand;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
    }
    params
}

/// Fit a unidimensional-trait polytomous LSIRM by marginal EM (fixed gamma = 1,
/// distance interaction). `y` is `n_persons * n_items` row-major categories
/// `0..n_cat-1`; `observed` marks non-missing cells (None = all observed).
#[allow(clippy::too_many_arguments)]
pub fn fit_poly_lsirm(
    y: &[usize],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    n_cat: usize,
    latent_dim: usize,
    model: PolyModel,
    q_theta: usize,
    q_xi: usize,
    max_iter: usize,
    tol: f64,
) -> Result<PolyLsirmFit, String> {
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if latent_dim < 1 || latent_dim > 3 {
        return Err("latent_dim must be 1..3 for the tensor grid".into());
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
    let (theta, t_w) =
        crate::quadrature::gh_rule(q_theta).ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let t_logw: Vec<f64> = t_w.iter().map(|w| w.ln()).collect();
    let (xi_grid, x_logw) = xi_tensor_grid(q_xi, latent_dim)?;
    let n_xi = x_logw.len();
    let q_t = theta.len();
    let cell = q_t * n_xi;
    let eps = 1e-8_f64;
    let (lambda_alpha, mu_alpha, lambda_zeta) = (1.0_f64, 0.0_f64, 1.0_f64);

    // init: log_a = 0; category params from base rates; positions on a small ring
    let kp = n_cat - 1;
    let np = 1 + kp + latent_dim;
    let mut params = vec![vec![0.0_f64; np]; n_items];
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
                    let cc = cum.clamp(1e-4, 1.0 - 1e-4);
                    params[i][k] = (cc / (1.0 - cc)).ln();
                }
            }
        }
        // positive-manifold ring init for the positions
        let ang = 2.0 * std::f64::consts::PI * (i as f64) / (n_items as f64);
        params[i][1 + kp] = 0.5 * ang.cos();
        if latent_dim >= 2 {
            params[i][1 + kp + 1] = 0.5 * ang.sin();
        }
        if latent_dim >= 3 {
            params[i][1 + kp + 2] = 0.25 * (2.0 * ang).cos();
        }
    }

    let mut prev_ll = f64::NEG_INFINITY;
    let mut ll = f64::NEG_INFINITY;
    let mut it = 0;
    while it < max_iter {
        // per-item cell log-probs at each (theta, xi) node
        let mut item_lp = vec![vec![0.0_f64; cell * n_cat]; n_items];
        for i in 0..n_items {
            let a = params[i][0].exp();
            let cat = &params[i][1..n_cat];
            let zeta = &params[i][n_cat..n_cat + latent_dim];
            for (t, &theta_t) in theta.iter().enumerate() {
                for x in 0..n_xi {
                    let xi = &xi_grid[x * latent_dim..(x + 1) * latent_dim];
                    let mut dist2 = eps;
                    for k in 0..latent_dim {
                        let dd = xi[k] - zeta[k];
                        dist2 += dd * dd;
                    }
                    let base = a * theta_t - dist2.sqrt();
                    let lp = poly_cell(base, model, cat, n_cat);
                    let node = t * n_xi + x;
                    item_lp[i][node * n_cat..(node + 1) * n_cat].copy_from_slice(&lp);
                }
            }
        }
        // E-step: person posteriors -> expected category counts rbar[i][node][k]
        let mut rbar = vec![vec![0.0_f64; cell * n_cat]; n_items];
        ll = 0.0;
        let mut log_node = vec![0.0_f64; cell];
        for p in 0..n_persons {
            for t in 0..q_t {
                for x in 0..n_xi {
                    log_node[t * n_xi + x] = t_logw[t] + x_logw[x];
                }
            }
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                for node in 0..cell {
                    log_node[node] += item_lp[i][node * n_cat + yc];
                }
            }
            let mx = log_node.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0_f64;
            for node in 0..cell {
                denom += (log_node[node] - mx).exp();
            }
            ll += mx + denom.ln();
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let yc = y[p * n_items + i];
                for node in 0..cell {
                    rbar[i][node * n_cat + yc] += (log_node[node] - mx).exp() / denom;
                }
            }
        }
        // M-step: per-item Newton over [log_a, cat, zeta]
        for i in 0..n_items {
            let ctx = ItemCtx {
                model,
                n_cat,
                latent_dim,
                eps,
                theta,
                xi_grid: &xi_grid,
                n_xi,
                rbar_i: &rbar[i],
                lambda_alpha,
                mu_alpha,
                lambda_zeta,
            };
            params[i] = m_step_item(params[i].clone(), &ctx, 6);
        }
        it += 1;
        if (ll - prev_ll).abs() < tol * (1.0 + prev_ll.abs()) {
            break;
        }
        prev_ll = ll;
    }

    let slope = (0..n_items).map(|i| params[i][0].exp()).collect();
    let cat_params = (0..n_items).map(|i| params[i][1..n_cat].to_vec()).collect();
    let mut zeta = vec![0.0_f64; n_items * latent_dim];
    for i in 0..n_items {
        zeta[i * latent_dim..(i + 1) * latent_dim]
            .copy_from_slice(&params[i][n_cat..n_cat + latent_dim]);
    }
    Ok(PolyLsirmFit { slope, cat_params, zeta, loglik: ll, n_iter: it })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist_matrix(z: &[f64], n: usize, d: usize) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let mut s = 0.0;
                for k in 0..d {
                    let dd = z[i * d + k] - z[j * d + k];
                    s += dd * dd;
                }
                out.push(s.sqrt());
            }
        }
        out
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    #[test]
    fn fit_poly_lsirm_recovers_positions_and_slopes() {
        let (n_persons, n_items, k, ld) = (1500usize, 6usize, 3usize, 2usize);
        let mut st = 314159u64;
        let mut u = || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        macro_rules! nrm {
            () => {{
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            }};
        }
        // true item positions on two separated clusters, slopes, GPCM intercepts
        let mut zeta_true = vec![0.0_f64; n_items * ld];
        for i in 0..n_items {
            let cx = if i < n_items / 2 { -1.2 } else { 1.2 };
            zeta_true[i * ld] = cx + 0.3 * nrm!();
            zeta_true[i * ld + 1] = 0.3 * nrm!();
        }
        let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.08 * i as f64).collect();
        let c_true: Vec<Vec<f64>> =
            (0..n_items).map(|i| vec![0.0, 0.2 - 0.05 * i as f64, -0.2 + 0.05 * i as f64]).collect();
        let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let mut y = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let theta = nrm!();
            let xi: Vec<f64> = (0..ld).map(|_| nrm!()).collect();
            for i in 0..n_items {
                let mut dist2 = 1e-8;
                for kk in 0..ld {
                    let dd = xi[kk] - zeta_true[i * ld + kk];
                    dist2 += dd * dd;
                }
                let base = a_true[i] * theta - dist2.sqrt();
                let mut ic = vec![0.0; k];
                ic[1..].copy_from_slice(&c_true[i][1..]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let uu = u();
                let mut cum = 0.0;
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
        let fit = fit_poly_lsirm(&y, None, n_persons, n_items, k, ld, PolyModel::Gpcm, 7, 7, 40, 1e-5)
            .unwrap();
        assert!(fit.loglik.is_finite());
        // ABSOLUTE-agreement checks (correlation only shows association, not
        // identity): slope RMSE, and RMSE of the item-item distance matrix, which
        // is exactly invariant to the position rotation/reflection/translation
        // ambiguity while gamma = 1 fixes its absolute scale.
        let slope_rmse = rmse(&a_true, &fit.slope);
        assert!(slope_rmse < 0.25, "slope RMSE {slope_rmse}");
        let dm_true = dist_matrix(&zeta_true, n_items, ld);
        let dm_hat = dist_matrix(&fit.zeta, n_items, ld);
        let pos_rmse = rmse(&dm_true, &dm_hat);
        assert!(pos_rmse < 0.6, "position distance-matrix RMSE {pos_rmse}");
    }
}
