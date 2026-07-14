//! Respondent scoring with frozen item parameters: EAP, MAP, and summed-score
//! EAP (EAPsum) tables via the Lord-Wingersky recursion.
//!
//! Sources (see docs/papers/mmle-lsirm-formula-compilation.md): Bock & Mislevy
//! (1982) EAP; standard MAP scoring with the posterior Newton step; Thissen,
//! Pommerich, Billeaud & Williams (1995) summed-score EAP with the
//! Lord & Wingersky (1984) recursion.
//!
//! Population priors are per-dimension `N(mean_d, sd_d^2)`, which covers all
//! three population structures of the marginal estimator:
//! - single: `mean = 0, sd = 1`;
//! - multigroup: the group's `(mu_gd, sigma_gd)`;
//! - multilevel: `N(sigma_u * u_hat_c, 1)` conditional on a known cluster, or
//!   the marginal `N(0, sqrt(1 + sigma_u^2))` for an unknown cluster.

use crate::marginal::{build_tables, index_responses, person_pass, Contexts, Grids};
use crate::nodes::{build_xi_nodes, XiRule};
use crate::quadrature::gh_rule;
use crate::{model_exec_flags, ModelConfig, ModelType};

/// Frozen item parameters plus the model contract they were calibrated under.
pub struct ItemBank<'a> {
    pub alpha: &'a [f64],
    pub b: &'a [f64],
    /// Row-major `n_items x latent_dim`.
    pub zeta: &'a [f64],
    pub tau: f64,
    pub factor_id: &'a [usize],
    pub model_type: ModelType,
    pub n_dims: usize,
    pub latent_dim: usize,
    pub eps_distance: f64,
}

/// Per-dimension trait prior `N(mean_d, sd_d^2)`.
#[derive(Clone, Debug)]
pub struct PriorSpec {
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
}

impl PriorSpec {
    pub fn standard(n_dims: usize) -> Self {
        Self { mean: vec![0.0; n_dims], sd: vec![1.0; n_dims] }
    }
}

pub struct EapScores {
    pub theta_eap: Vec<f64>,
    pub theta_sd: Vec<f64>,
    pub xi_eap: Vec<f64>,
    pub loglik: Vec<f64>,
}

pub struct MapScores {
    pub theta_map: Vec<f64>,
    pub theta_se: Vec<f64>,
    pub xi_map: Vec<f64>,
    pub log_posterior: Vec<f64>,
    pub converged: Vec<bool>,
}

/// Summed-score EAP conversion table for one trait dimension.
pub struct EapSumTable {
    pub dim: usize,
    /// Item count of the dimension; scores run 0..=n_items_dim.
    pub n_items_dim: usize,
    /// `P(score = s)` under the prior (model-implied score distribution).
    pub score_prob: Vec<f64>,
    /// `E[theta_d | score = s]`.
    pub eap: Vec<f64>,
    /// `SD[theta_d | score = s]`.
    pub sd: Vec<f64>,
}

fn validate_bank(bank: &ItemBank<'_>) -> Result<usize, String> {
    let n_items = bank.b.len();
    if bank.alpha.len() != n_items
        || bank.factor_id.len() != n_items
        || bank.zeta.len() != n_items * bank.latent_dim
    {
        return Err("item bank arrays have inconsistent lengths".into());
    }
    if bank.factor_id.iter().any(|&d| d >= bank.n_dims) {
        return Err("factor_id values must be in 0..n_dims-1".into());
    }
    if bank.n_dims == 0 || bank.latent_dim == 0 {
        return Err("parameter dimensions must be positive".into());
    }
    if bank.eps_distance <= 0.0 {
        return Err("eps_distance must be positive".into());
    }
    Ok(n_items)
}

fn validate_prior(prior: &PriorSpec, n_dims: usize) -> Result<(), String> {
    if prior.mean.len() != n_dims || prior.sd.len() != n_dims {
        return Err("prior mean/sd must have one entry per trait dimension".into());
    }
    if prior.sd.iter().any(|&s| s <= 0.0) {
        return Err("prior sds must be positive".into());
    }
    Ok(())
}

fn scoring_grids(
    bank: &ItemBank<'_>,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<Grids, String> {
    let (_, uses_space) = model_exec_flags(bank.model_type);
    let (t_nodes, t_weights) =
        gh_rule(q_theta).ok_or_else(|| format!("unsupported quadrature size {q_theta}"))?;
    let (x_grid, x_logw) = if uses_space {
        let nodes = build_xi_nodes(xi_rule, bank.latent_dim)?;
        (nodes.grid, nodes.logw)
    } else {
        (vec![0.0; bank.latent_dim], vec![0.0])
    };
    Ok(Grids {
        t_nodes: t_nodes.to_vec(),
        t_logw: t_weights.iter().map(|w| w.ln()).collect(),
        n_x: x_logw.len(),
        x_grid,
        x_logw,
        q_t: q_theta,
    })
}

fn prior_contexts(prior: &PriorSpec) -> Contexts {
    Contexts {
        n_ctx: 1,
        shift: prior.mean.clone(),
        scale: prior.sd.clone(),
        u_nodes: Vec::new(),
        u_logw: Vec::new(),
    }
}

fn bank_model_config(bank: &ItemBank<'_>, n_persons: usize, n_items: usize) -> ModelConfig {
    ModelConfig {
        n_persons,
        n_items,
        n_dims: bank.n_dims,
        latent_dim: bank.latent_dim,
        model_type: bank.model_type,
        eps_distance: bank.eps_distance,
    }
}

/// EAP scoring (Bock & Mislevy 1982) of `n_persons` response vectors against
/// the frozen bank, under a shared per-dimension prior.
pub fn score_eap(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<EapScores, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    let grids = scoring_grids(bank, q_theta, xi_rule)?;
    let ctx = prior_contexts(prior);
    let config = bank_model_config(bank, n_persons, n_items);
    let tables =
        build_tables(bank.alpha, bank.b, bank.zeta, bank.tau, &config, bank.factor_id, &ctx, &grids);
    let resp = index_responses(y, observed, n_persons, n_items);
    let cell = grids.q_t * grids.n_x;
    let mut l_buf = vec![0.0_f64; bank.n_dims * cell];
    let mut log_zdx = vec![0.0_f64; bank.n_dims * grids.n_x];

    let mut out = EapScores {
        theta_eap: vec![0.0; n_persons * bank.n_dims],
        theta_sd: vec![0.0; n_persons * bank.n_dims],
        xi_eap: vec![0.0; n_persons * bank.latent_dim],
        loglik: vec![0.0; n_persons],
    };
    for p in 0..n_persons {
        let lp = person_pass(
            p, 0, &tables, &resp, bank.factor_id, bank.n_dims, n_items, &grids, &mut l_buf,
            &mut log_zdx,
        );
        out.loglik[p] = lp;
        let mut theta_m2 = vec![0.0_f64; bank.n_dims];
        for x in 0..grids.n_x {
            let mut lx = grids.x_logw[x] - lp;
            for d in 0..bank.n_dims {
                lx += log_zdx[d * grids.n_x + x];
            }
            let px = lx.exp();
            for k in 0..bank.latent_dim {
                out.xi_eap[p * bank.latent_dim + k] += px * grids.x_grid[x * bank.latent_dim + k];
            }
            for d in 0..bank.n_dims {
                for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                    let theta = prior.mean[d] + prior.sd[d] * node_t;
                    let pt = (grids.t_logw[t] + l_buf[d * cell + t * grids.n_x + x]
                        - log_zdx[d * grids.n_x + x])
                        .exp();
                    out.theta_eap[p * bank.n_dims + d] += px * pt * theta;
                    theta_m2[d] += px * pt * theta * theta;
                }
            }
        }
        for d in 0..bank.n_dims {
            let m = out.theta_eap[p * bank.n_dims + d];
            out.theta_sd[p * bank.n_dims + d] = (theta_m2[d] - m * m).max(0.0).sqrt();
        }
    }
    Ok(out)
}

#[inline]
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

/// Solve the symmetric linear system `H x = g` in place (Gauss-Jordan with
/// partial pivoting); `H` is `n x n` row-major. Returns None when singular.
fn solve_sym(mut h: Vec<f64>, mut g: Vec<f64>, n: usize) -> Option<Vec<f64>> {
    for col in 0..n {
        let mut piv = col;
        for r in (col + 1)..n {
            if h[r * n + col].abs() > h[piv * n + col].abs() {
                piv = r;
            }
        }
        if h[piv * n + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..n {
                h.swap(col * n + c, piv * n + c);
            }
            g.swap(col, piv);
        }
        let d = h[col * n + col];
        for c in 0..n {
            h[col * n + c] /= d;
        }
        g[col] /= d;
        for r in 0..n {
            if r != col {
                let f = h[r * n + col];
                if f != 0.0 {
                    for c in 0..n {
                        h[r * n + c] -= f * h[col * n + c];
                    }
                    g[r] -= f * g[col];
                }
            }
        }
    }
    Some(g)
}

/// MAP scoring: damped Newton ascent of the log posterior over
/// `(theta in R^D, xi in R^K)` per person, with standard errors from the
/// diagonal of the inverse observed information at the mode.
pub fn score_map(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    max_iter: usize,
    tol: f64,
) -> Result<MapScores, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    if y.len() != n_persons * n_items || observed.len() != y.len() {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let (n_dims, latent_dim) = (bank.n_dims, bank.latent_dim);
    let n_par = n_dims + if uses_space { latent_dim } else { 0 };
    let gamma = if uses_space { bank.tau.exp() } else { 0.0 };

    let mut out = MapScores {
        theta_map: vec![0.0; n_persons * n_dims],
        theta_se: vec![0.0; n_persons * n_dims],
        xi_map: vec![0.0; n_persons * latent_dim],
        log_posterior: vec![0.0; n_persons],
        converged: vec![false; n_persons],
    };

    // log posterior and its gradient / observed information at (theta, xi)
    let eval = |p: usize, par: &[f64], grad: Option<&mut Vec<f64>>, info: Option<&mut Vec<f64>>| -> f64 {
        let theta = &par[..n_dims];
        let xi = &par[n_dims..];
        let mut lp = 0.0;
        let mut g = vec![0.0_f64; n_par];
        let mut h = vec![0.0_f64; n_par * n_par];
        for i in 0..n_items {
            let idx = p * n_items + i;
            if !observed[idx] {
                continue;
            }
            let d = bank.factor_id[i];
            let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
            let mut eta = a * theta[d] + bank.b[i];
            let mut dist = 1.0;
            if uses_space {
                let mut dist2 = bank.eps_distance;
                for k in 0..latent_dim {
                    let diff = xi[k] - bank.zeta[i * latent_dim + k];
                    dist2 += diff * diff;
                }
                dist = dist2.sqrt();
                eta -= gamma * dist;
            }
            let yy = y[idx];
            lp += yy * log_sigmoid(eta) + (1.0 - yy) * log_sigmoid(-eta);
            let prob = sigmoid(eta);
            let resid = yy - prob;
            let w = prob * (1.0 - prob);
            // d eta / d theta_d = a ; d eta / d xi_k = -gamma (xi_k - zeta_ik)/dist
            g[d] += resid * a;
            h[d * n_par + d] += w * a * a;
            if uses_space {
                for k in 0..latent_dim {
                    let u_k = -gamma * (xi[k] - bank.zeta[i * latent_dim + k]) / dist;
                    g[n_dims + k] += resid * u_k;
                    h[d * n_par + n_dims + k] += w * a * u_k;
                    h[(n_dims + k) * n_par + d] += w * a * u_k;
                    for k2 in 0..latent_dim {
                        let u_k2 = -gamma * (xi[k2] - bank.zeta[i * latent_dim + k2]) / dist;
                        h[(n_dims + k) * n_par + n_dims + k2] += w * u_k * u_k2;
                    }
                }
            }
        }
        for d in 0..n_dims {
            let z = (theta[d] - prior.mean[d]) / prior.sd[d];
            lp -= 0.5 * z * z;
            g[d] -= z / prior.sd[d];
            h[d * n_par + d] += 1.0 / (prior.sd[d] * prior.sd[d]);
        }
        if uses_space {
            for k in 0..latent_dim {
                lp -= 0.5 * xi[k] * xi[k];
                g[n_dims + k] -= xi[k];
                h[(n_dims + k) * n_par + n_dims + k] += 1.0;
            }
        }
        if let Some(gr) = grad {
            *gr = g;
        }
        if let Some(inf) = info {
            *inf = h;
        }
        lp
    };

    for p in 0..n_persons {
        let mut par = vec![0.0_f64; n_par];
        let mut lp = eval(p, &par, None, None);
        let mut converged = false;
        for _ in 0..max_iter {
            let mut g = Vec::new();
            let mut h = Vec::new();
            eval(p, &par, Some(&mut g), Some(&mut h));
            let Some(step_dir) = solve_sym(h.clone(), g.clone(), n_par) else {
                break;
            };
            let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < tol {
                converged = true;
                break;
            }
            let mut step = 1.0_f64;
            let mut accepted = false;
            for _ in 0..25 {
                let cand: Vec<f64> =
                    par.iter().zip(&step_dir).map(|(v, s)| v + step * s).collect();
                let cand_lp = eval(p, &cand, None, None);
                if cand_lp > lp {
                    par = cand;
                    lp = cand_lp;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                converged = g_norm < tol.max(1e-4);
                break;
            }
        }
        // SEs from the observed information at the mode.
        let mut h = Vec::new();
        eval(p, &par, None, Some(&mut h));
        for d in 0..n_dims {
            let mut e = vec![0.0_f64; n_par];
            e[d] = 1.0;
            let se = solve_sym(h.clone(), e, n_par)
                .map(|col| col[d].max(0.0).sqrt())
                .unwrap_or(f64::NAN);
            out.theta_se[p * n_dims + d] = se;
        }
        out.theta_map[p * n_dims..(p + 1) * n_dims].copy_from_slice(&par[..n_dims]);
        if uses_space {
            out.xi_map[p * latent_dim..(p + 1) * latent_dim].copy_from_slice(&par[n_dims..]);
        }
        out.log_posterior[p] = lp;
        out.converged[p] = converged;
    }
    Ok(out)
}

/// Lord-Wingersky (1984) recursion: `probs` is `n_items x n_nodes` row-major
/// success probabilities; returns the `(n_items + 1) x n_nodes` summed-score
/// distribution.
pub fn lord_wingersky(probs: &[f64], n_items: usize, n_nodes: usize) -> Vec<f64> {
    assert_eq!(probs.len(), n_items * n_nodes);
    let mut f = vec![0.0_f64; (n_items + 1) * n_nodes];
    if n_items == 0 {
        for x in 0..n_nodes {
            f[x] = 1.0;
        }
        return f;
    }
    for x in 0..n_nodes {
        f[x] = 1.0 - probs[x];
        f[n_nodes + x] = probs[x];
    }
    let mut prev = vec![0.0_f64; (n_items + 1) * n_nodes];
    for n in 1..n_items {
        prev[..(n + 1) * n_nodes].copy_from_slice(&f[..(n + 1) * n_nodes]);
        for r in 0..=(n + 1) {
            for x in 0..n_nodes {
                let p = probs[n * n_nodes + x];
                let stay = if r <= n { prev[r * n_nodes + x] * (1.0 - p) } else { 0.0 };
                let up = if r >= 1 { prev[(r - 1) * n_nodes + x] * p } else { 0.0 };
                f[r * n_nodes + x] = stay + up;
            }
        }
    }
    f
}

/// Summed-score EAP tables (Thissen et al. 1995), one per trait dimension:
/// `E[theta_d | summed score over the dimension's items]`, with the item
/// success probabilities marginalized over the latent-space nodes.
pub fn eapsum_tables(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<Vec<EapSumTable>, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    let grids = scoring_grids(bank, q_theta, xi_rule)?;
    let ctx = prior_contexts(prior);
    let config = bank_model_config(bank, 1, n_items);
    let tables =
        build_tables(bank.alpha, bank.b, bank.zeta, bank.tau, &config, bank.factor_id, &ctx, &grids);
    let cell = grids.q_t * grids.n_x;

    let mut out = Vec::new();
    for d in 0..bank.n_dims {
        let items: Vec<usize> = (0..n_items).filter(|&i| bank.factor_id[i] == d).collect();
        let n_d = items.len();
        if n_d == 0 {
            out.push(EapSumTable {
                dim: d,
                n_items_dim: 0,
                score_prob: vec![1.0],
                eap: vec![prior.mean[d]],
                sd: vec![prior.sd[d]],
            });
            continue;
        }
        // success probabilities on the joint (t, x) node set
        let mut probs = vec![0.0_f64; n_d * cell];
        for (row, &i) in items.iter().enumerate() {
            for c in 0..cell {
                probs[row * cell + c] = tables.logp1[i * cell + c].exp();
            }
        }
        let score_dist = lord_wingersky(&probs, n_d, cell);
        // joint node weights and theta values
        let mut w = vec![0.0_f64; cell];
        let mut theta_val = vec![0.0_f64; cell];
        for (t, &node_t) in grids.t_nodes.iter().enumerate() {
            let theta = prior.mean[d] + prior.sd[d] * node_t;
            for x in 0..grids.n_x {
                let c = t * grids.n_x + x;
                w[c] = (grids.t_logw[t] + grids.x_logw[x]).exp();
                theta_val[c] = theta;
            }
        }
        let mut score_prob = vec![0.0_f64; n_d + 1];
        let mut eap = vec![0.0_f64; n_d + 1];
        let mut sd = vec![0.0_f64; n_d + 1];
        for s in 0..=n_d {
            let (mut p0, mut m1, mut m2) = (0.0_f64, 0.0_f64, 0.0_f64);
            for c in 0..cell {
                let v = w[c] * score_dist[s * cell + c];
                p0 += v;
                m1 += v * theta_val[c];
                m2 += v * theta_val[c] * theta_val[c];
            }
            score_prob[s] = p0;
            if p0 > 0.0 {
                eap[s] = m1 / p0;
                sd[s] = (m2 / p0 - eap[s] * eap[s]).max(0.0).sqrt();
            } else {
                eap[s] = prior.mean[d];
                sd[s] = prior.sd[d];
            }
        }
        out.push(EapSumTable { dim: d, n_items_dim: n_d, score_prob, eap, sd });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::XiRule;

    fn small_bank() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let alpha = vec![0.1, -0.1, 0.2, 0.0, 0.05, -0.05];
        let b = vec![0.4, -0.3, 0.1, -0.6, 0.2, 0.0];
        let zeta = vec![0.5, -0.4, -0.6, 0.3, 0.2, 0.7, -0.1, -0.5, 0.4, 0.4, -0.3, 0.1];
        let factor_id = vec![0, 1, 0, 1, 0, 1];
        (alpha, b, zeta, factor_id)
    }

    fn bank<'a>(
        alpha: &'a [f64],
        b: &'a [f64],
        zeta: &'a [f64],
        factor_id: &'a [usize],
    ) -> ItemBank<'a> {
        ItemBank {
            alpha,
            b,
            zeta,
            tau: 0.0,
            factor_id,
            model_type: ModelType::Mls2plm,
            n_dims: 2,
            latent_dim: 2,
            eps_distance: 1e-8,
        }
    }

    #[test]
    fn eap_map_agree_and_react_to_data() {
        let (alpha, b, zeta, fid) = small_bank();
        let bk = bank(&alpha, &b, &zeta, &fid);
        let prior = PriorSpec::standard(2);
        // all-pass vs all-fail on dim 0 items (0, 2, 4)
        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let observed = vec![true; 12];
        let eap = score_eap(
            &bk, &y, &observed, 2, &prior, 21, XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        assert!(eap.theta_eap[0] > eap.theta_eap[2], "dim-0 pass > dim-0 fail");
        let map = score_map(&bk, &y, &observed, 2, &prior, 50, 1e-8).unwrap();
        assert!(map.converged.iter().all(|&c| c));
        // EAP and MAP should agree loosely for these smooth posteriors
        for p in 0..2 {
            for d in 0..2 {
                let diff = (eap.theta_eap[p * 2 + d] - map.theta_map[p * 2 + d]).abs();
                assert!(diff < 0.6, "EAP/MAP disagree: {diff}");
            }
            assert!(map.theta_se[p * 2].is_finite() && map.theta_se[p * 2] > 0.0);
        }
    }

    #[test]
    fn prior_shift_moves_scores() {
        let (alpha, b, zeta, fid) = small_bank();
        let bk = bank(&alpha, &b, &zeta, &fid);
        let empty_y = vec![0.0; 6];
        let none_obs = vec![false; 6];
        let base = score_eap(
            &bk, &empty_y, &none_obs, 1, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        assert!(base.theta_eap[0].abs() < 1e-9, "no data -> prior mean");
        let shifted_prior = PriorSpec { mean: vec![0.7, -0.2], sd: vec![1.0, 1.0] };
        let shifted = score_eap(
            &bk, &empty_y, &none_obs, 1, &shifted_prior, 15,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        assert!((shifted.theta_eap[0] - 0.7).abs() < 1e-9);
        assert!((shifted.theta_eap[1] + 0.2).abs() < 1e-9);
    }

    #[test]
    fn lord_wingersky_sums_to_one_and_matches_enumeration() {
        let probs = vec![0.3, 0.6, 0.2, 0.8, 0.5, 0.5];
        let f = lord_wingersky(&probs, 3, 2);
        for x in 0..2 {
            let total: f64 = (0..4).map(|r| f[r * 2 + x]).sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
        // enumeration for node 0: p = (0.3, 0.2, 0.5)
        let (p1, p2, p3) = (0.3, 0.2, 0.5);
        let expect0 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3);
        assert!((f[0] - expect0).abs() < 1e-12);
        let expect3 = p1 * p2 * p3;
        assert!((f[3 * 2] - expect3).abs() < 1e-12);
    }

    #[test]
    fn eapsum_tables_are_monotone_in_score() {
        let (alpha, b, zeta, fid) = small_bank();
        let bk = bank(&alpha, &b, &zeta, &fid);
        let tables = eapsum_tables(
            &bk, &PriorSpec::standard(2), 21, XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        assert_eq!(tables.len(), 2);
        for tab in &tables {
            assert_eq!(tab.eap.len(), tab.n_items_dim + 1);
            let total: f64 = tab.score_prob.iter().sum();
            assert!((total - 1.0).abs() < 1e-9, "score probs must sum to 1");
            for s in 1..tab.eap.len() {
                assert!(
                    tab.eap[s] > tab.eap[s - 1] - 1e-9,
                    "EAPsum must be nondecreasing in the summed score"
                );
            }
        }
    }

    #[test]
    fn multilevel_marginal_prior_widens_sd() {
        let (alpha, b, zeta, fid) = small_bank();
        let bk = bank(&alpha, &b, &zeta, &fid);
        let sigma_u = 0.8_f64;
        let marginal_prior = PriorSpec {
            mean: vec![0.0; 2],
            sd: vec![(1.0 + sigma_u * sigma_u).sqrt(); 2],
        };
        let t1 = eapsum_tables(&bk, &PriorSpec::standard(2), 15, XiRule::GaussHermite { q_xi: 7 })
            .unwrap();
        let t2 = eapsum_tables(&bk, &marginal_prior, 15, XiRule::GaussHermite { q_xi: 7 })
            .unwrap();
        // wider prior -> more extreme conversion at the top score
        let top1 = *t1[0].eap.last().unwrap();
        let top2 = *t2[0].eap.last().unwrap();
        assert!(top2 > top1, "marginal multilevel prior should widen the scale");
    }

    #[test]
    fn rejects_bad_inputs() {
        let (alpha, b, zeta, fid) = small_bank();
        let bk = bank(&alpha, &b, &zeta, &fid);
        let prior = PriorSpec::standard(2);
        assert!(score_eap(
            &bk, &[0.0; 5], &[true; 5], 1, &prior, 21, XiRule::GaussHermite { q_xi: 7 }
        )
        .is_err());
        let bad_prior = PriorSpec { mean: vec![0.0], sd: vec![1.0] };
        assert!(score_eap(
            &bk, &[0.0; 6], &[true; 6], 1, &bad_prior, 21, XiRule::GaussHermite { q_xi: 7 }
        )
        .is_err());
        let neg_sd = PriorSpec { mean: vec![0.0; 2], sd: vec![1.0, -1.0] };
        assert!(eapsum_tables(&bk, &neg_sd, 21, XiRule::GaussHermite { q_xi: 7 }).is_err());
    }
}
