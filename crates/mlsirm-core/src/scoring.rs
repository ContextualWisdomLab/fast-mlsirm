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

pub(crate) fn validate_bank(bank: &ItemBank<'_>) -> Result<usize, String> {
    let n_items = bank.b.len();
    let expected_zeta = n_items
        .checked_mul(bank.latent_dim)
        .ok_or_else(|| "n_items * latent_dim overflows usize".to_string())?;
    if bank.alpha.len() != n_items
        || bank.factor_id.len() != n_items
        || bank.zeta.len() != expected_zeta
    {
        return Err("item bank arrays have inconsistent lengths".into());
    }
    if bank.factor_id.iter().any(|&d| d >= bank.n_dims) {
        return Err("factor_id values must be in 0..n_dims-1".into());
    }
    if bank.n_dims == 0 || bank.latent_dim == 0 {
        return Err("parameter dimensions must be positive".into());
    }
    if bank
        .alpha
        .iter()
        .chain(bank.b)
        .chain(bank.zeta)
        .any(|v| !v.is_finite())
        || !bank.tau.is_finite()
    {
        return Err("item bank parameters must be finite".into());
    }
    if !bank.eps_distance.is_finite() || bank.eps_distance <= 0.0 {
        return Err("eps_distance must be positive and finite".into());
    }
    Ok(n_items)
}

fn validate_dichotomous_responses(
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
) -> Result<(), String> {
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if y.iter()
        .zip(observed)
        .any(|(&value, &is_observed)| is_observed && value != 0.0 && value != 1.0)
    {
        return Err("observed responses must be 0 or 1".into());
    }
    Ok(())
}

pub(crate) fn validate_prior(prior: &PriorSpec, n_dims: usize) -> Result<(), String> {
    if prior.mean.len() != n_dims || prior.sd.len() != n_dims {
        return Err("prior mean/sd must have one entry per trait dimension".into());
    }
    if prior.sd.iter().any(|&s| !s.is_finite() || s <= 0.0) {
        return Err("prior sds must be positive and finite".into());
    }
    if prior.mean.iter().any(|&m| !m.is_finite()) {
        return Err("prior means must be finite".into());
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
    score_eap_device(bank, y, observed, n_persons, prior, q_theta, xi_rule, crate::Device::Cpu)
}

/// EAP scoring with an explicit compute device. `Device::Cpu` keeps the exact
/// f64 reduction (the default); `Device::Gpu`/`Auto` offloads to the wgpu
/// `score_pass` kernel (f32, ~1e-4) when an adapter is present, otherwise CPU.
#[allow(clippy::too_many_arguments)]
pub fn score_eap_device(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
    device: crate::Device,
) -> Result<EapScores, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;
    let grids = scoring_grids(bank, q_theta, xi_rule)?;
    let ctx = prior_contexts(prior);
    let config = bank_model_config(bank, n_persons, n_items);
    let tables =
        build_tables(bank.alpha, bank.b, bank.zeta, bank.tau, &config, bank.factor_id, &ctx, &grids);
    let resp = index_responses(y, observed, n_persons, n_items);
    // GPU EAP path (Bock-Mislevy on wgpu, f32) when a device is requested;
    // falls back to the exact CPU reduction when Cpu, no adapter, or the model
    // exceeds the kernel bounds (n_dims/latent_dim <= 8).
    if device != crate::Device::Cpu {
        #[cfg(all(feature = "gpu", not(coverage)))]
        {
            if let Some(gpu_out) =
                try_score_eap_gpu(bank, prior, &grids, &tables, &resp, n_persons, n_items)
            {
                return Ok(gpu_out);
            }
        }
    }
    Ok(score_eap_cpu_reduce(bank, prior, &grids, &tables, &resp, n_persons, n_items))
}

/// The scalar f64 CPU EAP reduction (the parity reference for `score_eap_gpu`).
#[allow(clippy::too_many_arguments)]
fn score_eap_cpu_reduce(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    grids: &crate::marginal::Grids,
    tables: &crate::marginal::Tables,
    resp: &crate::marginal::ResponseIndex,
    n_persons: usize,
    n_items: usize,
) -> EapScores {
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
            p, 0, tables, resp, bank.factor_id, bank.n_dims, n_items, grids, &mut l_buf,
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
    out
}


/// Build the GPU score inputs (CSR-flattened responses) and dispatch the
/// `score_pass` kernel; `None` on no-adapter or out-of-bounds models.
#[cfg(all(feature = "gpu", not(coverage)))]
#[allow(clippy::too_many_arguments)]
fn try_score_eap_gpu(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    grids: &crate::marginal::Grids,
    tables: &crate::marginal::Tables,
    resp: &crate::marginal::ResponseIndex,
    n_persons: usize,
    n_items: usize,
) -> Option<EapScores> {
    let mut pos_off: Vec<u32> = Vec::with_capacity(n_persons + 1);
    let mut pos_items: Vec<u32> = Vec::new();
    pos_off.push(0);
    for p in 0..n_persons {
        for &i in &resp.pos[p] {
            pos_items.push(i as u32);
        }
        pos_off.push(pos_items.len() as u32);
    }
    let mut miss_off: Vec<u32> = Vec::with_capacity(n_persons + 1);
    let mut miss_items: Vec<u32> = Vec::new();
    miss_off.push(0);
    for p in 0..n_persons {
        for &i in &resp.miss[p] {
            miss_items.push(i as u32);
        }
        miss_off.push(miss_items.len() as u32);
    }
    let inputs = crate::gpu_marginal::GpuScoreInputs {
        n_persons,
        n_items,
        n_dims: bank.n_dims,
        latent_dim: bank.latent_dim,
        q_t: grids.q_t,
        n_x: grids.n_x,
        logp0: &tables.logp0,
        logp1: &tables.logp1,
        c0: &tables.c0,
        t_logw: &grids.t_logw,
        x_logw: &grids.x_logw,
        t_nodes: &grids.t_nodes,
        x_grid: &grids.x_grid,
        prior_mean: &prior.mean,
        prior_sd: &prior.sd,
        factor_id: bank.factor_id,
        pos_off: &pos_off,
        pos_items: &pos_items,
        miss_off: &miss_off,
        miss_items: &miss_items,
    };
    let out = crate::gpu_marginal::score_eap_gpu(&inputs)?;
    Some(EapScores {
        theta_eap: out.theta_eap,
        theta_sd: out.theta_sd,
        xi_eap: out.xi_eap,
        loglik: out.loglik,
    })
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
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);
    let (n_dims, latent_dim) = (bank.n_dims, bank.latent_dim);
    let n_par = n_dims + if uses_space { latent_dim } else { 0 };
    let gamma = if kind == crate::InteractionKind::Distance { bank.tau.exp() } else { 0.0 };

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
            match kind {
                crate::InteractionKind::None => {}
                crate::InteractionKind::Distance => {
                    let mut dist2 = bank.eps_distance;
                    for k in 0..latent_dim {
                        let diff = xi[k] - bank.zeta[i * latent_dim + k];
                        dist2 += diff * diff;
                    }
                    dist = dist2.sqrt();
                    eta -= gamma * dist;
                }
                crate::InteractionKind::Inner => {
                    for k in 0..latent_dim {
                        eta += bank.zeta[i * latent_dim + k] * xi[k];
                    }
                }
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
                    let u_k = match kind {
                        crate::InteractionKind::Distance => {
                            -gamma * (xi[k] - bank.zeta[i * latent_dim + k]) / dist
                        }
                        crate::InteractionKind::Inner => bank.zeta[i * latent_dim + k],
                        crate::InteractionKind::None => 0.0,
                    };
                    g[n_dims + k] += resid * u_k;
                    h[d * n_par + n_dims + k] += w * a * u_k;
                    h[(n_dims + k) * n_par + d] += w * a * u_k;
                    for k2 in 0..latent_dim {
                        let u_k2 = match kind {
                            crate::InteractionKind::Distance => {
                                -gamma * (xi[k2] - bank.zeta[i * latent_dim + k2]) / dist
                            }
                            crate::InteractionKind::Inner => bank.zeta[i * latent_dim + k2],
                            crate::InteractionKind::None => 0.0,
                        };
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


/// Item information of the four-parameter logistic model (Magis 2013, APM,
/// "A note on the item information function of the four-parameter logistic
/// model"): with `P = c + (d - c) sigmoid(eta)` and slope `a`,
/// `I(theta) = a^2 (P - c)^2 (d - P)^2 / ((d - c)^2 P (1 - P))`.
/// `c = 0, d = 1` reduces to the 2PL `a^2 P (1 - P)`. For the latent-space
/// models the information is with respect to the trait direction at a fixed
/// latent-space position.
pub fn item_information_4pl(a: f64, p: f64, c: f64, d: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 || d <= c {
        return 0.0;
    }
    let num = a * a * (p - c) * (p - c) * (d - p) * (d - p);
    num / ((d - c) * (d - c) * p * (1.0 - p))
}

/// Per-item information at arbitrary `(theta_d, xi)` points for a frozen
/// bank; also the per-dimension test information (sum over the dimension's
/// items). `theta` is `n_points x n_dims`, `xi` is `n_points x latent_dim`.
pub fn bank_information(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_items = validate_bank(bank)?;
    if theta.len() != n_points * bank.n_dims || xi.len() != n_points * bank.latent_dim {
        return Err("theta/xi shapes must match n_points".into());
    }
    let (free_alpha, _uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance { bank.tau.exp() } else { 0.0 };
    let mut item_info = vec![0.0_f64; n_points * n_items];
    let mut test_info = vec![0.0_f64; n_points * bank.n_dims];
    for p in 0..n_points {
        for i in 0..n_items {
            let d = bank.factor_id[i];
            let a = if free_alpha { bank.alpha[i].exp() } else { 1.0 };
            let mut eta = a * theta[p * bank.n_dims + d] + bank.b[i];
            match kind {
                crate::InteractionKind::None => {}
                crate::InteractionKind::Distance => {
                    let mut dist2 = bank.eps_distance;
                    for k in 0..bank.latent_dim {
                        let diff =
                            xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
                        dist2 += diff * diff;
                    }
                    eta -= gamma * dist2.sqrt();
                }
                crate::InteractionKind::Inner => {
                    for k in 0..bank.latent_dim {
                        eta += bank.zeta[i * bank.latent_dim + k] * xi[p * bank.latent_dim + k];
                    }
                }
            }
            let prob = sigmoid(eta);
            let info = item_information_4pl(a, prob, 0.0, 1.0);
            item_info[p * n_items + i] = info;
            test_info[p * bank.n_dims + d] += info;
        }
    }
    Ok((item_info, test_info))
}

/// Warm's (1989) weighted-likelihood ability estimates for a UNIDIMENSIONAL dichotomous test.
///
/// The maximum-likelihood ability estimate carries an `O(1/n)` bias; Warm removes its leading term by
/// weighting the likelihood by a function `w(theta)` with `w'/w = J(theta) / (2 I(theta))`, giving the
/// estimating equation
///
/// ```text
///   dlnL/dtheta + J(theta) / (2 I(theta)) = 0,
/// ```
///
/// where, for the 4-parameter logistic `P_i = c_i + (d_i - c_i) sigmoid(a_i (theta - b_i))` (the 3PL is
/// `d_i = 1`, the 2PL is `c_i = 0, d_i = 1`), with `P_i' = a_i (d_i - c_i) s_i (1 - s_i)` and
/// `P_i'' = a_i^2 (d_i - c_i) s_i (1 - s_i)(1 - 2 s_i)` (`s_i = sigmoid(a_i(theta - b_i))`):
///
/// - `dlnL/dtheta = sum_i (y_i - P_i) P_i' / (P_i Q_i)` (score);
/// - `I(theta) = sum_i P_i'^2 / (P_i Q_i)` (test information, [`item_information_4pl`] per item);
/// - `J(theta) = sum_i P_i' P_i'' / (P_i Q_i)` (the Warm correction; computed DIRECTLY from `P' P''`).
///
/// `J` is **not** `I'(theta)/2`: they coincide only for the 2PL/Rasch (`c = 0, d = 1`), where the
/// weight is `sqrt(I)` (the Jeffreys prior); for the 3PL/4PL `J != I'` (the second derivative carries
/// `1 - 2 s` while `I'` carries `1 - 2 P`), so a `sqrt(I)`-weighted estimator applies the wrong 3PL/4PL
/// correction. Two properties Warm establishes: the estimator removes the leading MLE bias, and — unlike
/// the MLE, which is `+/-infinity` for the all-correct / all-incorrect pattern — it yields a FINITE
/// estimate there. The estimate is the GLOBAL maximizer of the weighted log-likelihood `Phi`
/// (`Phi' = g`), located by a grid scan of `g` (its trapezoidal cumulative integral recovers `Phi`) plus
/// a local root refinement; this is robust to the 3PL/4PL case where the weighted likelihood can be
/// multimodal (Samejima, 1973; Yen, Burket & Sykes, 1991), which a single bracketed root can get wrong.
/// The reported standard error is `1 / sqrt(I(theta_wle))` (asymptotic).
///
/// `a`/`b`/`c`/`d` are per-item NATURAL-scale parameters (length `n_items`; `a` is the slope, NOT
/// log-alpha) with `0 <= c_i < d_i <= 1`; `y`/`observed` are row-major `n_persons * n_items` (missing
/// items dropped per person). `theta_bound` bounds the search grid; when the finite Warm root lies
/// beyond it (very easy/hard items relative to the pattern) the estimate is clamped to the boundary and
/// `boundary` is set. A person with no observed items gets `NaN` theta/se with `boundary` set (ability
/// undefined).
///
/// # References (APA 7th ed.)
///
/// Warm, T. A. (1989). Weighted likelihood estimation of ability in item response theory.
/// *Psychometrika, 54*(3), 427-450. <https://doi.org/10.1007/BF02294627>
pub struct WleScores {
    /// Weighted-likelihood ability estimate per person.
    pub theta: Vec<f64>,
    /// Asymptotic standard error `1 / sqrt(I(theta_wle))` (`NaN` if the test information is ~0).
    pub se: Vec<f64>,
    /// `true` when the finite root fell outside `[-theta_bound, theta_bound]` and `theta` was clamped.
    pub boundary: Vec<bool>,
}

#[allow(clippy::too_many_arguments)]
pub fn score_wle(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    theta_bound: f64,
    tol: f64,
) -> Result<WleScores, String> {
    let n_items = a.len();
    if n_items == 0 {
        return Err("need at least one item".into());
    }
    if b.len() != n_items || c.len() != n_items || d.len() != n_items {
        return Err("a, b, c, d must have equal length".into());
    }
    for i in 0..n_items {
        if !a[i].is_finite() || !b[i].is_finite() || !c[i].is_finite() || !d[i].is_finite() {
            return Err("item parameters must be finite".into());
        }
        if !(0.0..1.0).contains(&c[i]) || c[i] >= d[i] || d[i] > 1.0 {
            return Err("require 0 <= c_i < d_i <= 1".into());
        }
    }
    if !theta_bound.is_finite() || theta_bound <= 0.0 {
        return Err("theta_bound must be finite and positive".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;

    // (g, I) at theta for person p, where g = score + J/(2I) is the Warm estimating function. The
    // clamp on s keeps P Q away from 0 (item_information_4pl-style saturation), and the I floor guards
    // the J/(2I) division when every observed item is saturated.
    let eval = |p: usize, theta: f64| -> (f64, f64) {
        let (mut score, mut info, mut jterm) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n_items {
            let idx = p * n_items + i;
            if !observed[idx] {
                continue;
            }
            let s = sigmoid(a[i] * (theta - b[i])).clamp(1e-12, 1.0 - 1e-12);
            let dc = d[i] - c[i];
            let pp = c[i] + dc * s;
            let pq = (pp * (1.0 - pp)).max(1e-300);
            let p1 = a[i] * dc * s * (1.0 - s); // P'
            let p2 = a[i] * a[i] * dc * s * (1.0 - s) * (1.0 - 2.0 * s); // P''
            score += (y[idx] - pp) * p1 / pq;
            info += p1 * p1 / pq;
            jterm += p1 * p2 / pq;
        }
        (score + jterm / (2.0 * info.max(1e-12)), info)
    };

    // The WLE is the GLOBAL maximizer of the weighted log-likelihood `Phi` with `Phi'(theta) = g`; for
    // the 2PL/Rasch `Phi` is unimodal, but for the 3PL/4PL the weighted likelihood can have SEVERAL
    // stationary points (Samejima, 1973; Yen, Burket & Sykes, 1991), so a single bracketed bisection can
    // converge to a non-dominant root. Recover `Phi` (up to a constant) as the trapezoidal cumulative
    // integral of `g` over a grid, take the global-max node, and refine the root of `g` around it.
    const GRID: usize = 512;
    let h = 2.0 * theta_bound / GRID as f64;
    let mut gvals = vec![0.0f64; GRID + 1];
    let mut out = WleScores {
        theta: vec![0.0; n_persons],
        se: vec![0.0; n_persons],
        boundary: vec![false; n_persons],
    };
    for p in 0..n_persons {
        // No observed items -> ability is undefined; do not report a spurious theta = 0.
        if (0..n_items).all(|i| !observed[p * n_items + i]) {
            out.theta[p] = f64::NAN;
            out.se[p] = f64::NAN;
            out.boundary[p] = true;
            continue;
        }
        for k in 0..=GRID {
            gvals[k] = eval(p, -theta_bound + h * k as f64).0;
        }
        // Phi_0 = 0 (reference); track the global argmax over the grid nodes.
        let (mut phi, mut best_phi, mut best_k) = (0.0f64, 0.0f64, 0usize);
        for k in 1..=GRID {
            phi += 0.5 * (gvals[k - 1] + gvals[k]) * h;
            if phi > best_phi {
                best_phi = phi;
                best_k = k;
            }
        }
        let theta_hat = if best_k == 0 || best_k == GRID {
            // Global max at a boundary node: the finite Warm root lies at/beyond the hard bound.
            out.boundary[p] = true;
            -theta_bound + h * best_k as f64
        } else {
            // Interior max: Phi' = g crosses + -> - in [node-1, node+1]; refine by bisection.
            let (mut a0, mut b0) = (
                -theta_bound + h * (best_k as f64 - 1.0),
                -theta_bound + h * (best_k as f64 + 1.0),
            );
            let mut ga = eval(p, a0).0;
            if ga * eval(p, b0).0 > 0.0 {
                -theta_bound + h * best_k as f64 // no clean sign change (narrow mode): use the node
            } else {
                for _ in 0..200 {
                    if b0 - a0 < tol {
                        break;
                    }
                    let mid = 0.5 * (a0 + b0);
                    let gm = eval(p, mid).0;
                    if gm == 0.0 {
                        a0 = mid;
                        b0 = mid;
                        break;
                    }
                    if (gm > 0.0) == (ga > 0.0) {
                        a0 = mid;
                        ga = gm;
                    } else {
                        b0 = mid;
                    }
                }
                0.5 * (a0 + b0)
            }
        };
        out.theta[p] = theta_hat;
        let info = eval(p, theta_hat).1;
        out.se[p] = if info > 1e-12 { (1.0 / info).sqrt() } else { f64::NAN };
    }
    Ok(out)
}

/// One step of adaptive EAP testing: score the responses so far by EAP, pick
/// the trait dimension with the largest posterior SD, and return the
/// unadministered items of that dimension ranked by information at the current
/// EAP point. Bock and Mislevy (1982) support the noniterative EAP scoring, and
/// Wang et al. (2010) describe multidimensional CAT with information-based item
/// selection. Choosing the largest-posterior-SD dimension is a repository
/// policy, not a procedure prescribed by either source.
///
/// # References
///
/// Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
/// a microcomputer environment. *Applied Psychological Measurement, 6*(4),
/// 431–444. <https://doi.org/10.1177/014662168200600405>
///
/// Wang, C.-S., Kuo, C.-L., & Chao, C.-Y. (2010). A multidimensional
/// computerized adaptive testing system for enhancing the Chinese as second
/// language proficiency test. In N. E. Mastorakis, V. Mladenov, Z. Bojkovic,
/// & S. Kartalopoulos (Eds.), *Selected topics in education and educational
/// technology* (pp. 245–252). WSEAS Press.
pub struct CatStep {
    pub theta_eap: Vec<f64>,
    pub theta_sd: Vec<f64>,
    pub xi_eap: Vec<f64>,
    pub target_dim: usize,
    /// Unadministered item indices, best first.
    pub ranked_items: Vec<usize>,
    /// Information of each ranked item at the current EAP point.
    pub ranked_info: Vec<f64>,
}

#[allow(clippy::too_many_arguments)]
pub fn cat_next_item(
    bank: &ItemBank<'_>,
    y: &[f64],
    administered: &[bool],
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<CatStep, String> {
    let n_items = validate_bank(bank)?;
    if y.len() != n_items || administered.len() != n_items {
        return Err("y and administered must have length n_items".into());
    }
    if y.iter()
        .zip(administered)
        .any(|(&value, &is_administered)| {
            is_administered && (!value.is_finite() || (value != 0.0 && value != 1.0))
        })
    {
        return Err("administered responses must be 0 or 1".into());
    }
    let scores = score_eap(bank, y, administered, 1, prior, q_theta, xi_rule)?;
    let target_dim = (0..bank.n_dims)
        .max_by(|&a, &b| {
            scores.theta_sd[a]
                .partial_cmp(&scores.theta_sd[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let (item_info, _) = bank_information(bank, &scores.theta_eap, &scores.xi_eap, 1)?;
    let mut candidates: Vec<usize> = (0..n_items)
        .filter(|&i| !administered[i] && bank.factor_id[i] == target_dim)
        .collect();
    if candidates.is_empty() {
        candidates = (0..n_items).filter(|&i| !administered[i]).collect();
    }
    candidates.sort_by(|&a, &b| {
        item_info[b].partial_cmp(&item_info[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let ranked_info: Vec<f64> = candidates.iter().map(|&i| item_info[i]).collect();
    Ok(CatStep {
        theta_eap: scores.theta_eap,
        theta_sd: scores.theta_sd,
        xi_eap: scores.xi_eap,
        target_dim,
        ranked_items: candidates,
        ranked_info,
    })
}

/// Plausible values (Marsman et al., 2016): seeded categorical draws of `theta`
/// from each person posterior over the scoring grid, for secondary analyses
/// that need the ability distribution rather than point EAPs. The fixed item
/// bank and discrete quadrature-grid sampler are repository implementation
/// choices; this routine does not propagate item-parameter uncertainty.
/// Returns row-major `n_persons x n_draws x n_dims`.
///
/// # References
///
/// Marsman, M., Maris, G., Bechger, T., & Glas, C. (2016). What can we learn
/// from plausible values? *Psychometrika, 81*(2), 274–289.
/// <https://doi.org/10.1007/s11336-016-9497-x>
#[allow(clippy::too_many_arguments)]
pub fn plausible_values(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
    n_draws: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let n_items = validate_bank(bank)?;
    validate_prior(prior, bank.n_dims)?;
    validate_dichotomous_responses(y, observed, n_persons, n_items)?;
    if n_draws == 0 {
        return Err("n_draws must be >= 1".into());
    }
    let grids = scoring_grids(bank, q_theta, xi_rule)?;
    let ctx = prior_contexts(prior);
    let config = bank_model_config(bank, n_persons, n_items);
    let tables = build_tables(
        bank.alpha, bank.b, bank.zeta, bank.tau, &config, bank.factor_id, &ctx, &grids,
    );
    let resp = index_responses(y, observed, n_persons, n_items);
    let cell = grids.q_t * grids.n_x;
    let mut l_buf = vec![0.0_f64; bank.n_dims * cell];
    let mut log_zdx = vec![0.0_f64; bank.n_dims * grids.n_x];
    let mut state = seed.max(1);
    let mut unif = move || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut out = vec![0.0_f64; n_persons * n_draws * bank.n_dims];
    for p in 0..n_persons {
        let lp = person_pass(
            p, 0, &tables, &resp, bank.factor_id, bank.n_dims, n_items, &grids, &mut l_buf,
            &mut log_zdx,
        );
        let mut px = vec![0.0_f64; grids.n_x];
        for x in 0..grids.n_x {
            let mut lx = grids.x_logw[x] - lp;
            for d in 0..bank.n_dims {
                lx += log_zdx[d * grids.n_x + x];
            }
            px[x] = lx.exp();
        }
        for draw in 0..n_draws {
            let ux = unif();
            let mut acc = 0.0;
            let mut x_sel = grids.n_x - 1;
            for (x, &w) in px.iter().enumerate() {
                acc += w;
                if ux <= acc {
                    x_sel = x;
                    break;
                }
            }
            for d in 0..bank.n_dims {
                let ut = unif();
                let mut acc_t = 0.0;
                let mut t_sel = grids.q_t - 1;
                for t in 0..grids.q_t {
                    let pt = (grids.t_logw[t] + l_buf[d * cell + t * grids.n_x + x_sel]
                        - log_zdx[d * grids.n_x + x_sel])
                        .exp();
                    acc_t += pt;
                    if ut <= acc_t {
                        t_sel = t;
                        break;
                    }
                }
                out[(p * n_draws + draw) * bank.n_dims + d] =
                    prior.mean[d] + prior.sd[d] * grids.t_nodes[t_sel];
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod cat_pv_tests {
    use super::*;
    use crate::nodes::XiRule;
    use crate::ModelType;

    fn bank_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
        let alpha = vec![0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.25];
        let b = vec![0.5, -0.5, 0.0, 1.0, -1.0, 0.3, -0.3, 0.8];
        let zeta = vec![0.0; 8];
        let factor_id = vec![0, 1, 0, 1, 0, 1, 0, 1];
        (alpha, b, zeta, factor_id)
    }

    #[test]
    fn information_reduces_to_2pl_and_peaks_at_b() {
        // 4PL formula with c=0, d=1 equals a^2 P (1-P)
        let i1 = item_information_4pl(1.5, 0.4, 0.0, 1.0);
        assert!((i1 - 1.5f64 * 1.5 * 0.4 * 0.6).abs() < 1e-12);
        // guessing shrinks information (Magis 2013)
        let i3pl = item_information_4pl(1.5, 0.4, 0.2, 1.0);
        assert!(i3pl < i1);
        assert_eq!(item_information_4pl(1.5, 0.0, 0.0, 1.0), 0.0);
    }

    #[test]
    fn cat_selects_informative_item_on_target_dim() {
        let (alpha, b, zeta, fid) = bank_fixture();
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: -30.0,
            factor_id: &fid,
            model_type: ModelType::Mirt,
            n_dims: 2,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        // dim 0 already has two answers; dim 1 has none -> target dim 1
        let y = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let administered = vec![true, false, true, false, false, false, false, false];
        let step = cat_next_item(
            &bank, &y, &administered, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .unwrap();
        assert_eq!(step.target_dim, 1, "unmeasured dimension must be targeted");
        assert!(step.ranked_items.iter().all(|&i| fid[i] == 1 && !administered[i]));
        // ranked by information: descending
        for w in step.ranked_info.windows(2) {
            assert!(w[0] >= w[1]);
        }
        let mut invalid_y = y.clone();
        invalid_y[0] = 2.0;
        assert!(cat_next_item(
            &bank, &invalid_y, &administered, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .is_err());
        invalid_y[0] = f64::NAN;
        assert!(cat_next_item(
            &bank, &invalid_y, &administered, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 },
        )
        .is_err());
    }

    #[test]
    fn plausible_values_track_the_posterior() {
        let (alpha, b, zeta, fid) = bank_fixture();
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: -30.0,
            factor_id: &fid,
            model_type: ModelType::Mirt,
            n_dims: 2,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        // person 0 passes everything on dim 0, person 1 fails everything
        let y = vec![
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let observed = vec![true; 16];
        let pv = plausible_values(
            &bank, &y, &observed, 2, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 }, 200, 7,
        )
        .unwrap();
        let mean_p0_d0: f64 =
            (0..200).map(|r| pv[(r) * 2]).sum::<f64>() / 200.0;
        let mean_p1_d0: f64 =
            (0..200).map(|r| pv[(200 + r) * 2]).sum::<f64>() / 200.0;
        assert!(
            mean_p0_d0 > mean_p1_d0 + 0.5,
            "PV means must separate pass-all from fail-all: {mean_p0_d0} vs {mean_p1_d0}"
        );
        // draws are reproducible
        let pv2 = plausible_values(
            &bank, &y, &observed, 2, &PriorSpec::standard(2), 15,
            XiRule::GaussHermite { q_xi: 7 }, 200, 7,
        )
        .unwrap();
        assert_eq!(pv, pv2);
    }
}


/// Empirical (marginal) reliability of the EAP scale scores per trait
/// dimension: `rho_d = Var(theta_hat_d) / (Var(theta_hat_d) + mean(SE_d^2))`.
/// This follows the posterior variance decomposition in Bechger et al. (2003).
/// Because reliability does not establish model fit (Stanley & Edwards, 2016),
/// report it alongside the fit statistics, never instead of them.
///
/// # References
///
/// Bechger, T. M., Maris, G., Verstralen, H. H. F. M., & Béguin, A. A. (2003).
/// Using classical test theory in combination with item response theory.
/// *Applied Psychological Measurement, 27*(5), 319–334.
/// <https://doi.org/10.1177/0146621603257518>
///
/// Stanley, L. M., & Edwards, M. C. (2016). Reliability and model fit.
/// *Educational and Psychological Measurement, 76*(6), 976–985.
/// <https://doi.org/10.1177/0013164416638900>
pub fn empirical_reliability(
    theta_eap: &[f64],
    theta_sd: &[f64],
    n_persons: usize,
    n_dims: usize,
) -> Result<Vec<f64>, String> {
    if theta_eap.len() != n_persons * n_dims || theta_sd.len() != theta_eap.len() {
        return Err("theta_eap/theta_sd must be n_persons x n_dims".into());
    }
    if n_persons < 2 {
        return Err("empirical reliability needs n_persons >= 2".into());
    }
    if n_dims == 0 {
        return Err("empirical reliability needs n_dims >= 1".into());
    }
    if theta_eap.iter().any(|value| !value.is_finite()) {
        return Err("theta_eap values must be finite".into());
    }
    if theta_sd.iter().any(|&value| !value.is_finite() || value < 0.0) {
        return Err("theta_sd values must be finite and non-negative".into());
    }
    let mut out = vec![f64::NAN; n_dims];
    for d in 0..n_dims {
        let n = n_persons as f64;
        let mean: f64 = (0..n_persons).map(|p| theta_eap[p * n_dims + d]).sum::<f64>() / n;
        let var: f64 = (0..n_persons)
            .map(|p| {
                let v = theta_eap[p * n_dims + d] - mean;
                v * v
            })
            .sum::<f64>()
            / n;
        let mse: f64 = (0..n_persons)
            .map(|p| theta_sd[p * n_dims + d] * theta_sd[p * n_dims + d])
            .sum::<f64>()
            / n;
        if var + mse > 0.0 {
            out[d] = var / (var + mse);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod reliability_tests {
    use super::*;

    #[test]
    fn empirical_reliability_tracks_signal_to_noise() {
        // wide score spread + small SEs -> high rho; flat scores -> low rho
        let n = 200usize;
        let eap: Vec<f64> = (0..n).map(|p| -2.0 + 4.0 * p as f64 / n as f64).collect();
        let sd_small = vec![0.3_f64; n];
        let sd_large = vec![1.5_f64; n];
        let hi = empirical_reliability(&eap, &sd_small, n, 1).unwrap()[0];
        let lo = empirical_reliability(&eap, &sd_large, n, 1).unwrap()[0];
        assert!(hi > 0.85, "high-information scale must be reliable: {hi}");
        assert!(lo < hi - 0.2, "noisier scale must be less reliable: {lo} vs {hi}");
        assert!(empirical_reliability(&eap, &sd_small, 3, 1).is_err());
        assert!(empirical_reliability(&[], &[], 2, 0).is_err());
        assert!(empirical_reliability(&[0.0, f64::NAN], &[0.3, 0.3], 2, 1).is_err());
        assert!(empirical_reliability(&[0.0, 1.0], &[-0.3, 0.3], 2, 1).is_err());
        assert!(empirical_reliability(&[0.0, 1.0], &[0.3, f64::INFINITY], 2, 1).is_err());
    }
}


#[cfg(test)]
mod validate_branch_tests {
    use super::*;
    use crate::nodes::XiRule;

    fn ok_bank<'a>(alpha: &'a [f64], b: &'a [f64], zeta: &'a [f64], fid: &'a [usize]) -> ItemBank<'a> {
        ItemBank {
            alpha, b, zeta, tau: -30.0, factor_id: fid,
            model_type: crate::ModelType::Mirt, n_dims: 1, latent_dim: 1, eps_distance: 1e-8,
        }
    }

    #[test]
    fn validate_bank_rejects_malformed_banks() {
        let y = vec![0.0; 3];
        let obs = vec![true; 3];
        let prior = PriorSpec::standard(1);
        let rule = XiRule::GaussHermite { q_xi: 7 };
        // inconsistent alpha length
        let (a, b, z, f) = (vec![0.0; 2], vec![0.0; 3], vec![0.0; 3], vec![0usize; 3]);
        assert!(score_eap(&ok_bank(&a, &b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
        // factor_id out of range (>= n_dims)
        let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![5usize, 0, 0]);
        assert!(score_eap(&ok_bank(&a, &b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
        // latent_dim zero
        let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 0], vec![0usize; 3]);
        let mut bk = ok_bank(&a, &b, &z, &f);
        bk.latent_dim = 0;
        assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
        // eps_distance non-positive
        let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0usize; 3]);
        let mut bk = ok_bank(&a, &b, &z, &f);
        bk.eps_distance = 0.0;
        assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
        // y/observed length mismatch
        let bk = ok_bank(&a, &b, &z, &f);
        assert!(score_eap(&bk, &vec![0.0; 6], &vec![true; 6], 1, &prior, 7, rule).is_err());

        // Public Rust scoring must reject non-finite calibrated parameters rather
        // than returning an apparently successful result filled with NaNs.
        let mut bad_b = b.clone();
        bad_b[0] = f64::NAN;
        assert!(score_eap(&ok_bank(&a, &bad_b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
        let mut bk = ok_bank(&a, &b, &z, &f);
        bk.tau = f64::INFINITY;
        assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
        let mut bk = ok_bank(&a, &b, &z, &f);
        bk.eps_distance = f64::NAN;
        assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());

        // Observed responses are dichotomous. NaN and other categories were
        // previously classified as zero by index_responses.
        for bad in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
            let mut bad_y = y.clone();
            bad_y[0] = bad;
            assert!(score_eap(&ok_bank(&a, &b, &z, &f), &bad_y, &obs, 1, &prior, 7, rule).is_err());
        }

        // Adversarial dimensions must return an error instead of overflowing
        // n_persons * n_items in a debug-build panic.
        assert!(score_eap(
            &ok_bank(&a, &b, &z, &f),
            &[],
            &[],
            usize::MAX,
            &prior,
            7,
            rule
        )
        .is_err());
    }
}


#[cfg(all(test, feature = "gpu", not(coverage)))]
mod gpu_score_tests {
    use super::*;
    use crate::nodes::XiRule;

    #[test]
    fn gpu_eap_matches_cpu_reduction() {
        let (n_items, n_persons, latent_dim) = (6usize, 40usize, 1usize);
        let alpha: Vec<f64> = (0..n_items).map(|i| 0.1 * i as f64 - 0.2).collect();
        let b: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.2 * i as f64).collect();
        let zeta: Vec<f64> =
            (0..n_items * latent_dim).map(|i| 0.3 * (i % 3) as f64 - 0.3).collect();
        let fid = vec![0usize; n_items];
        let mut st = 12345u64;
        let mut u = move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let mut y = vec![0.0_f64; n_persons * n_items];
        for v in y.iter_mut() {
            *v = if u() < 0.5 { 1.0 } else { 0.0 };
        }
        let observed = vec![true; n_persons * n_items];
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: -0.3,
            factor_id: &fid,
            model_type: crate::ModelType::Mls2plm,
            n_dims: 1,
            latent_dim,
            eps_distance: 1e-8,
        };
        let prior = PriorSpec::standard(1);
        let grids = scoring_grids(&bank, 21, XiRule::GaussHermite { q_xi: 11 }).unwrap();
        let ctx = prior_contexts(&prior);
        let config = bank_model_config(&bank, n_persons, n_items);
        let tables =
            build_tables(bank.alpha, bank.b, bank.zeta, bank.tau, &config, bank.factor_id, &ctx, &grids);
        let resp = index_responses(&y, &observed, n_persons, n_items);
        let cpu = score_eap_cpu_reduce(&bank, &prior, &grids, &tables, &resp, n_persons, n_items);
        match try_score_eap_gpu(&bank, &prior, &grids, &tables, &resp, n_persons, n_items) {
            None => eprintln!("no GPU adapter present; skipping GPU EAP parity check"),
            Some(gpu) => {
                for p in 0..n_persons {
                    assert!(
                        (gpu.loglik[p] - cpu.loglik[p]).abs() < 2e-3,
                        "loglik p={p}: gpu {} vs cpu {}",
                        gpu.loglik[p],
                        cpu.loglik[p]
                    );
                    assert!((gpu.theta_eap[p] - cpu.theta_eap[p]).abs() < 2e-3);
                    assert!((gpu.theta_sd[p] - cpu.theta_sd[p]).abs() < 2e-3);
                    assert!((gpu.xi_eap[p] - cpu.xi_eap[p]).abs() < 2e-3);
                }
            }
        }
    }
}

#[cfg(test)]
mod wle_tests {
    use super::{item_information_4pl, score_wle};

    fn sig(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

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

    /// The Warm estimating function `g = score + J/(2I)` recomputed INDEPENDENTLY from FINITE-DIFFERENCE
    /// derivatives of `P` (no analytic `P'`/`P''`), so a sign error in the implementation's `J = P' P''`
    /// term is not shared. Returns `g` at `theta`.
    fn g_fd(a: &[f64], b: &[f64], c: &[f64], d: &[f64], y: &[f64], theta: f64) -> f64 {
        let h = 1e-4;
        let pf = |i: usize, t: f64| c[i] + (d[i] - c[i]) * sig(a[i] * (t - b[i]));
        let (mut score, mut info, mut jterm) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let p0 = pf(i, theta);
            let p1 = (pf(i, theta + h) - pf(i, theta - h)) / (2.0 * h); // P' by FD
            let p2 = (pf(i, theta + h) - 2.0 * p0 + pf(i, theta - h)) / (h * h); // P'' by FD
            let pq = p0 * (1.0 - p0);
            score += (y[i] - p0) * p1 / pq;
            info += p1 * p1 / pq;
            jterm += p1 * p2 / pq;
        }
        score + jterm / (2.0 * info)
    }

    /// Root anchor across {2PL, 3PL, Rasch}: the returned `theta_hat` satisfies the Warm estimating
    /// equation, verified by the FD-derivative recomputation (independent of the analytic derivatives).
    #[test]
    fn wle_estimating_equation_root() {
        let j = 10usize;
        let a2: Vec<f64> = (0..j).map(|i| 0.8 + 0.09 * i as f64).collect();
        let b: Vec<f64> = (0..j).map(|i| -2.0 + 0.4 * i as f64).collect();
        let y: Vec<f64> = (0..j).map(|i| (i % 2) as f64).collect(); // mixed -> interior root
        let obs = vec![true; j];
        let one = vec![1.0f64; j];
        let zero = vec![0.0f64; j];
        let d = vec![1.0f64; j];
        let c02 = vec![0.2f64; j];
        for (label, a, c) in [
            ("2PL", &a2, &zero),
            ("3PL", &a2, &c02),
            ("Rasch", &one, &zero),
        ] {
            let res = score_wle(a, &b, c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
            assert!(!res.boundary[0], "{label}: unexpected boundary");
            let g = g_fd(a, &b, c, &d, &y, res.theta[0]);
            assert!(g.abs() < 1e-4, "{label}: WLE root residual {g} at theta {}", res.theta[0]);
            // SE matches 1/sqrt(I) recomputed from item_information_4pl at the estimate
            let info: f64 = (0..j)
                .map(|i| {
                    let p = c[i] + (d[i] - c[i]) * sig(a[i] * (res.theta[0] - b[i]));
                    item_information_4pl(a[i], p, c[i], d[i])
                })
                .sum();
            assert!((res.se[0] - (1.0 / info).sqrt()).abs() < 1e-9, "{label}: SE");
        }
    }

    /// Finiteness (scoped to the 2PL, `c=0, d=1`): the all-correct and all-incorrect patterns — where
    /// the MLE is `+/-infinity` — return FINITE, interior WLE estimates, with correct > incorrect.
    #[test]
    fn wle_finite_at_perfect_score_2pl() {
        let j = 6usize;
        let a: Vec<f64> = (0..j).map(|i| 1.0 + 0.1 * i as f64).collect();
        let b: Vec<f64> = (0..j).map(|i| -1.5 + 0.6 * i as f64).collect();
        let c = vec![0.0f64; j];
        let d = vec![1.0f64; j];
        let obs = vec![true; j];
        let all1 = vec![1.0f64; j];
        let all0 = vec![0.0f64; j];
        let hi = score_wle(&a, &b, &c, &d, &all1, &obs, 1, 20.0, 1e-9).unwrap();
        let lo = score_wle(&a, &b, &c, &d, &all0, &obs, 1, 20.0, 1e-9).unwrap();
        assert!(hi.theta[0].is_finite() && !hi.boundary[0], "all-correct theta {}", hi.theta[0]);
        assert!(lo.theta[0].is_finite() && !lo.boundary[0], "all-incorrect theta {}", lo.theta[0]);
        assert!(hi.theta[0] > lo.theta[0], "correct {} !> incorrect {}", hi.theta[0], lo.theta[0]);
        // the FD estimating equation is also ~0 at these finite roots
        assert!(g_fd(&a, &b, &c, &d, &all1, hi.theta[0]).abs() < 1e-4);
        assert!(g_fd(&a, &b, &c, &d, &all0, lo.theta[0]).abs() < 1e-4);
    }

    /// Monotonicity: for a fixed Rasch item set the WLE is nondecreasing in the number-correct score.
    #[test]
    fn wle_monotone_in_raw_score() {
        let j = 8usize;
        let a = vec![1.0f64; j];
        let b: Vec<f64> = (0..j).map(|i| -2.0 + 0.5 * i as f64).collect();
        let c = vec![0.0f64; j];
        let d = vec![1.0f64; j];
        let obs = vec![true; j];
        let mut prev = f64::NEG_INFINITY;
        for k in 0..=j {
            let y: Vec<f64> = (0..j).map(|i| if i < k { 1.0 } else { 0.0 }).collect();
            let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
            assert!(
                res.theta[0] >= prev - 1e-9,
                "raw score {k}: theta {} < previous {prev}",
                res.theta[0]
            );
            prev = res.theta[0];
        }
    }

    /// Validation guards trip non-vacuously.
    #[test]
    fn wle_validates() {
        let a = vec![1.0, 1.2];
        let b = vec![0.0, 0.5];
        let c = vec![0.0, 0.0];
        let d = vec![1.0, 1.0];
        let y = vec![1.0, 0.0];
        let obs = vec![true, true];
        assert!(score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).is_ok());
        // length mismatch
        assert!(score_wle(&a, &b[..1], &c, &d, &y, &obs, 1, 20.0, 1e-9).is_err());
        // c >= d
        let cbad = vec![1.0, 0.0];
        assert!(score_wle(&a, &b, &cbad, &d, &y, &obs, 1, 20.0, 1e-9).is_err());
        // response not 0/1
        let ybad = vec![2.0, 0.0];
        assert!(score_wle(&a, &b, &c, &d, &ybad, &obs, 1, 20.0, 1e-9).is_err());
        // theta_bound non-positive
        assert!(score_wle(&a, &b, &c, &d, &y, &obs, 1, 0.0, 1e-9).is_err());
    }

    /// The 3PL weighted likelihood is multimodal here; the WLE must return the GLOBAL mode, not merely
    /// a root of the estimating equation. Adversarial-review worst case: a single bracketed bisection
    /// returns `theta ~ +1.70`, but the dominant weighted-likelihood mode is `theta ~ -4.13` (~10x more
    /// probable). Pins the global-mode selection.
    #[test]
    fn wle_selects_global_mode_3pl_multimodal() {
        let a = [0.59, 1.38, 2.16, 3.45, 1.53, 2.58, 1.13, 1.02, 2.9, 2.07];
        let b = [-3.5, -3.78, -0.06, 2.82, 2.51, 2.73, -2.84, 3.48, 1.77, 0.07];
        let c = [0.37, 0.23, 0.26, 0.45, 0.28, 0.3, 0.4, 0.22, 0.22, 0.21];
        let d = [1.0f64; 10];
        let y = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let obs = [true; 10];
        let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
        assert!(
            res.theta[0] < -3.0,
            "did not select the global mode: theta {} (expected ~ -4.13, not the +1.70 root)",
            res.theta[0]
        );
    }

    /// A person with no observed items has undefined ability: `NaN` estimate and SE, flagged — not a
    /// spurious `theta = 0` (which the `g == 0` bisection shortcut would otherwise return).
    #[test]
    fn wle_all_missing_is_nan() {
        let a = [1.0, 1.2, 0.9];
        let b = [-0.5, 0.0, 0.7];
        let c = [0.0f64; 3];
        let d = [1.0f64; 3];
        let y = [0.0, 0.0, 0.0];
        let obs = [false, false, false];
        let res = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap();
        assert!(res.theta[0].is_nan() && res.se[0].is_nan() && res.boundary[0]);
    }

    /// Literature-grade bias comparison (>=500 reps): Warm's WLE has smaller mean bias than the MLE,
    /// especially at extreme abilities where perfect/near-perfect patterns bias the (boundary-clamped)
    /// MLE. Run with: `cargo test -p mlsirm-core --release wle_reduces_mle_bias_500 -- --ignored`.
    #[test]
    #[ignore]
    fn wle_reduces_mle_bias_500() {
        let reps = 500usize;
        let j = 15usize;
        let a: Vec<f64> = (0..j).map(|i| 0.9 + 0.05 * (i % 5) as f64).collect();
        let b: Vec<f64> = (0..j).map(|i| -2.0 + 4.0 * i as f64 / (j as f64 - 1.0)).collect();
        let c = vec![0.0f64; j];
        let d = vec![1.0f64; j];
        let obs = vec![true; j];
        // MLE by bisection on the score (clamped to +/-6 for separable patterns).
        let mle = |y: &[f64]| -> f64 {
            let score = |t: f64| -> f64 {
                (0..j)
                    .map(|i| {
                        let p = sig(a[i] * (t - b[i]));
                        a[i] * (y[i] - p)
                    })
                    .sum::<f64>()
            };
            let (mut loi, mut hii) = (-6.0f64, 6.0f64);
            let (glo, ghi) = (score(loi), score(hii));
            if glo * ghi > 0.0 {
                return if glo > 0.0 { hii } else { loi };
            }
            for _ in 0..100 {
                let mid = 0.5 * (loi + hii);
                if score(mid) > 0.0 {
                    loi = mid;
                } else {
                    hii = mid;
                }
            }
            0.5 * (loi + hii)
        };
        let grid = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let (mut wle_abs, mut mle_abs) = (0.0f64, 0.0f64);
        for &theta in &grid {
            let (mut wsum, mut msum, mut n) = (0.0f64, 0.0f64, 0usize);
            for rep in 0..reps {
                let mut rng = Lcg(0x9E1E_u64.wrapping_mul(rep as u64 + 1).wrapping_add((theta as i64 as u64).wrapping_mul(97)));
                let y: Vec<f64> = (0..j)
                    .map(|i| {
                        let p = sig(a[i] * (theta - b[i]));
                        if rng.next_f64() < p { 1.0 } else { 0.0 }
                    })
                    .collect();
                let w = score_wle(&a, &b, &c, &d, &y, &obs, 1, 20.0, 1e-9).unwrap().theta[0];
                wsum += w - theta;
                msum += mle(&y) - theta;
                n += 1;
            }
            let (wb, mb) = (wsum / n as f64, msum / n as f64);
            println!("[wle bias theta={theta}] WLE={wb:.4} MLE={mb:.4}");
            wle_abs += wb.abs();
            mle_abs += mb.abs();
        }
        println!("[wle] sum|bias| WLE={wle_abs:.4} MLE={mle_abs:.4}");
        assert!(wle_abs < mle_abs, "WLE did not reduce aggregate bias: {wle_abs} vs {mle_abs}");
    }
}
