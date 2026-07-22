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
        Self {
            mean: vec![0.0; n_dims],
            sd: vec![1.0; n_dims],
        }
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

pub(crate) fn validate_dichotomous_responses(
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

fn scoring_grids(bank: &ItemBank<'_>, q_theta: usize, xi_rule: XiRule) -> Result<Grids, String> {
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
///
/// The default execution policy is [`crate::Device::Auto`]: prefer the wgpu
/// kernel when the crate is built with GPU support and an adapter is usable,
/// otherwise fall back to the f64 CPU reduction. Use [`score_eap_device`] with
/// [`crate::Device::Cpu`] when a hardware-independent f64 reference is needed.
pub fn score_eap(
    bank: &ItemBank<'_>,
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
) -> Result<EapScores, String> {
    score_eap_device(
        bank,
        y,
        observed,
        n_persons,
        prior,
        q_theta,
        xi_rule,
        crate::Device::Auto,
    )
}

/// EAP scoring with an explicit compute device. `Device::Cpu` keeps the exact
/// f64 reduction; `Device::Gpu`/`Auto` offloads to the wgpu `score_pass` kernel
/// (f32, ~1e-4) when an adapter is present, otherwise CPU. An explicit
/// `Device::Gpu` request emits a warning when it falls back.
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
    let tables = build_tables(
        bank.alpha,
        bank.b,
        bank.zeta,
        bank.tau,
        &config,
        bank.factor_id,
        &ctx,
        &grids,
    );
    let resp = index_responses(y, observed, n_persons, n_items);
    Ok(dispatch_eap_device(
        bank, prior, &grids, &tables, &resp, n_persons, n_items, device,
    ))
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
            p,
            0,
            tables,
            resp,
            bank.factor_id,
            bank.n_dims,
            n_items,
            grids,
            &mut l_buf,
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

/// GPU EAP path (Bock-Mislevy on wgpu, f32) when a device is requested; falls back to the exact CPU
/// reduction when Cpu, no adapter, or the model exceeds the kernel bounds.
#[cfg(all(feature = "gpu", not(coverage)))]
#[allow(clippy::too_many_arguments)]
fn dispatch_eap_device(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    grids: &crate::marginal::Grids,
    tables: &crate::marginal::Tables,
    resp: &crate::marginal::ResponseIndex,
    n_persons: usize,
    n_items: usize,
    device: crate::Device,
) -> EapScores {
    if device != crate::Device::Cpu {
        if let Some(gpu_out) =
            try_score_eap_gpu(bank, prior, grids, tables, resp, n_persons, n_items)
        {
            return gpu_out;
        }
        if device == crate::Device::Gpu {
            eprintln!(
                "fast-mlsirm: GPU scoring requested but no usable GPU adapter was found or the model exceeds GPU kernel bounds; falling back to the CPU implementation."
            );
        }
    }
    score_eap_cpu_reduce(bank, prior, grids, tables, resp, n_persons, n_items)
}

#[cfg(any(not(feature = "gpu"), coverage))]
#[allow(clippy::too_many_arguments)]
fn dispatch_eap_device(
    bank: &ItemBank<'_>,
    prior: &PriorSpec,
    grids: &crate::marginal::Grids,
    tables: &crate::marginal::Tables,
    resp: &crate::marginal::ResponseIndex,
    n_persons: usize,
    n_items: usize,
    device: crate::Device,
) -> EapScores {
    if device == crate::Device::Gpu {
        eprintln!(
            "fast-mlsirm: GPU scoring requested but this build has no GPU support; falling back to the CPU implementation."
        );
    }
    score_eap_cpu_reduce(bank, prior, grids, tables, resp, n_persons, n_items)
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

/// MAP scoring: damped Fisher scoring of the log posterior over
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
    if max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be positive and finite".into());
    }
    let (free_alpha, uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);
    let (n_dims, latent_dim) = (bank.n_dims, bank.latent_dim);
    let n_par = n_dims + if uses_space { latent_dim } else { 0 };
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };

    let mut out = MapScores {
        theta_map: vec![0.0; n_persons * n_dims],
        theta_se: vec![0.0; n_persons * n_dims],
        xi_map: vec![0.0; n_persons * latent_dim],
        log_posterior: vec![0.0; n_persons],
        converged: vec![false; n_persons],
    };

    // log posterior and its gradient / observed information at (theta, xi)
    let eval = |p: usize,
                par: &[f64],
                grad: Option<&mut Vec<f64>>,
                info: Option<&mut Vec<f64>>,
                observed_curvature: bool|
     -> f64 {
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
                    // model_exec_flags guarantees that a spatial model is either distance or
                    // inner-product; InteractionKind::None always has uses_space=false.
                    let u_k = if kind == crate::InteractionKind::Distance {
                        -gamma * (xi[k] - bank.zeta[i * latent_dim + k]) / dist
                    } else {
                        bank.zeta[i * latent_dim + k]
                    };
                    g[n_dims + k] += resid * u_k;
                    h[d * n_par + n_dims + k] += w * a * u_k;
                    h[(n_dims + k) * n_par + d] += w * a * u_k;
                    for k2 in 0..latent_dim {
                        let u_k2 = if kind == crate::InteractionKind::Distance {
                            -gamma * (xi[k2] - bank.zeta[i * latent_dim + k2]) / dist
                        } else {
                            bank.zeta[i * latent_dim + k2]
                        };
                        let entry = (n_dims + k) * n_par + n_dims + k2;
                        h[entry] += w * u_k * u_k2;
                        if observed_curvature && kind == crate::InteractionKind::Distance {
                            let diff_k = xi[k] - bank.zeta[i * latent_dim + k];
                            let diff_k2 = xi[k2] - bank.zeta[i * latent_dim + k2];
                            let diagonal = if k == k2 { 1.0 } else { 0.0 };
                            let eta_second = -gamma
                                * (diagonal / dist - diff_k * diff_k2 / (dist * dist * dist));
                            // -d2 log p(y|eta) = w eta'eta' - (y-p) eta''.
                            h[entry] -= resid * eta_second;
                        }
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
        let mut lp = eval(p, &par, None, None, false);
        let mut converged = false;
        for _ in 0..max_iter {
            let mut g = Vec::new();
            let mut h = Vec::new();
            eval(p, &par, Some(&mut g), Some(&mut h), false);
            let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
            if g_norm < tol {
                converged = true;
                break;
            }
            // Fisher information is positive semidefinite and the proper Gaussian priors add a
            // strictly positive diagonal. Fail closed if finite-precision elimination nevertheless
            // cannot solve the system rather than unwinding a public scoring call.
            let Some(step_dir) = solve_sym(h, g, n_par) else {
                break;
            };
            let mut step = 1.0_f64;
            let mut accepted = false;
            for _ in 0..25 {
                let cand: Vec<f64> = par
                    .iter()
                    .zip(&step_dir)
                    .map(|(v, s)| v + step * s)
                    .collect();
                let cand_lp = eval(p, &cand, None, None, false);
                if cand_lp > lp {
                    par = cand;
                    lp = cand_lp;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        // SEs from the observed information at the mode.
        let mut h = Vec::new();
        eval(p, &par, None, Some(&mut h), true);
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
                let stay = if r <= n {
                    prev[r * n_nodes + x] * (1.0 - p)
                } else {
                    0.0
                };
                let up = if r >= 1 {
                    prev[(r - 1) * n_nodes + x] * p
                } else {
                    0.0
                };
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
    let tables = build_tables(
        bank.alpha,
        bank.b,
        bank.zeta,
        bank.tau,
        &config,
        bank.factor_id,
        &ctx,
        &grids,
    );
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
        out.push(EapSumTable {
            dim: d,
            n_items_dim: n_d,
            score_prob,
            eap,
            sd,
        });
    }
    Ok(out)
}

#[cfg(test)]
#[path = "../../../tests/unit/scoring_tests.rs"]
mod tests;

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
    bank_information_device(bank, theta, xi, n_points, crate::Device::Auto)
}

/// Fixed-bank information with an explicit compute device. `Device::Auto`
/// prefers the wgpu f32 kernel and falls back to the scalar Rust f64
/// implementation. `Device::Gpu` warns when no usable adapter is available.
pub fn bank_information_device(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
    device: crate::Device,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n_items = validate_bank(bank)?;
    let theta_len = crate::checked_mul_usize(n_points, bank.n_dims, "n_points * n_dims overflows")?;
    let xi_len =
        crate::checked_mul_usize(n_points, bank.latent_dim, "n_points * latent_dim overflows")?;
    if theta.len() != theta_len || xi.len() != xi_len {
        return Err("theta/xi shapes must match n_points".into());
    }
    if theta.iter().chain(xi).any(|value| !value.is_finite()) {
        return Err("theta/xi values must be finite".into());
    }
    Ok(dispatch_information_device(
        bank, theta, xi, n_points, n_items, device,
    ))
}

fn bank_information_cpu_reduce(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
    n_items: usize,
) -> (Vec<f64>, Vec<f64>) {
    let (free_alpha, _uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
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
                        let diff = xi[p * bank.latent_dim + k] - bank.zeta[i * bank.latent_dim + k];
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
    (item_info, test_info)
}

#[cfg(all(feature = "gpu", not(coverage)))]
fn try_bank_information_gpu(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
    n_items: usize,
) -> Option<(Vec<f64>, Vec<f64>)> {
    let (free_alpha, _uses_space) = model_exec_flags(bank.model_type);
    let kind = crate::interaction_kind(bank.model_type);
    let interaction_kind = match kind {
        crate::InteractionKind::None => 0,
        crate::InteractionKind::Distance => 1,
        crate::InteractionKind::Inner => 2,
    };
    let gamma = if kind == crate::InteractionKind::Distance {
        bank.tau.exp()
    } else {
        0.0
    };
    let output =
        crate::gpu_scoring::bank_information_gpu(&crate::gpu_scoring::GpuInformationInputs {
            n_points,
            n_items,
            n_dims: bank.n_dims,
            latent_dim: bank.latent_dim,
            free_alpha,
            interaction_kind,
            gamma,
            eps_distance: bank.eps_distance,
            alpha: bank.alpha,
            b: bank.b,
            zeta: bank.zeta,
            factor_id: bank.factor_id,
            theta,
            xi,
        })?;
    Some((output.item_info, output.test_info))
}

#[cfg(all(feature = "gpu", not(coverage)))]
fn dispatch_information_device(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
    n_items: usize,
    device: crate::Device,
) -> (Vec<f64>, Vec<f64>) {
    if device != crate::Device::Cpu {
        if let Some(output) = try_bank_information_gpu(bank, theta, xi, n_points, n_items) {
            return output;
        }
        if device == crate::Device::Gpu {
            eprintln!(
                "fast-mlsirm: GPU bank information requested but no usable GPU adapter was found, the output exceeds GPU indexing bounds, or f32 arithmetic produced invalid information; falling back to the CPU implementation."
            );
        }
    }
    bank_information_cpu_reduce(bank, theta, xi, n_points, n_items)
}

#[cfg(any(not(feature = "gpu"), coverage))]
fn dispatch_information_device(
    bank: &ItemBank<'_>,
    theta: &[f64],
    xi: &[f64],
    n_points: usize,
    n_items: usize,
    device: crate::Device,
) -> (Vec<f64>, Vec<f64>) {
    if device == crate::Device::Gpu {
        eprintln!(
            "fast-mlsirm: GPU bank information requested but this build has no GPU support; falling back to the CPU implementation."
        );
    }
    bank_information_cpu_reduce(bank, theta, xi, n_points, n_items)
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
/// The grid resolution scales with the steepest item discrimination; combinations that would require
/// more than 65,536 intervals are rejected rather than returned with an unresolved global mode. This
/// bounded adaptive search is a repository implementation choice, not a procedure specified by Warm.
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

fn finite_wle_value(value: f64, message: String) -> Result<f64, String> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(message)
    }
}

fn refine_wle_root(
    mut lower: f64,
    mut upper: f64,
    tol: f64,
    evaluate: &mut dyn FnMut(f64) -> f64,
) -> Result<f64, &'static str> {
    let mut lower_value = evaluate(lower);
    if lower_value * evaluate(upper) > 0.0 {
        return Err("failed to bracket the global WLE mode");
    }
    for _ in 0..200 {
        if upper - lower < tol {
            return Ok(lower + 0.5 * (upper - lower));
        }
        let mid = lower + 0.5 * (upper - lower);
        let mid_value = evaluate(mid);
        if mid_value == 0.0 {
            return Ok(mid);
        }
        if (mid_value > 0.0) == (lower_value > 0.0) {
            lower = mid;
            lower_value = mid_value;
        } else {
            upper = mid;
        }
    }
    Err("WLE root refinement did not converge")
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
    //
    // A fixed theta grid is not sufficient here: a logistic transition has width O(1 / |a_i|), and a
    // high-discrimination 3PL/4PL item can therefore create a dominant mode entirely between two fixed
    // nodes. Keep the historical 512-node floor for ordinary tests, but guarantee four intervals per
    // unit of the steepest item's logit scale. Refuse pathological controls that would exceed the
    // explicit work bound instead of silently returning the wrong mode.
    const MIN_GRID: usize = 512;
    const MAX_GRID: usize = 65_536;
    const INTERVALS_PER_LOGIT: f64 = 4.0;
    let max_abs_a = a.iter().fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    if max_abs_a == 0.0 {
        return Err("at least one item must have nonzero discrimination".into());
    }
    let required_grid = (2.0 * theta_bound * max_abs_a * INTERVALS_PER_LOGIT).ceil();
    if !required_grid.is_finite() || required_grid > MAX_GRID as f64 {
        return Err(format!(
            "theta_bound and item discrimination require more than {MAX_GRID} WLE grid intervals"
        ));
    }
    let grid = (required_grid as usize).max(MIN_GRID);
    let h = 2.0 * (theta_bound / grid as f64);
    let grid_theta = |k: usize| theta_bound * (2.0 * k as f64 / grid as f64 - 1.0);
    let mut gvals = vec![0.0f64; grid + 1];
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
        for (k, gval) in gvals.iter_mut().enumerate() {
            *gval = finite_wle_value(
                eval(p, grid_theta(k)).0,
                format!("non-finite WLE estimating function for person {p}"),
            )?;
        }
        // Phi_0 = 0 (reference); track the global argmax over the grid nodes.
        let (mut phi, mut best_phi, mut best_k) = (0.0f64, 0.0f64, 0usize);
        for k in 1..=grid {
            // The adaptive grid constrains `max(|a|) * theta_bound`, while the finite estimating
            // function scales with `a`; hence each trapezoid increment is finite after the check above.
            phi += 0.5 * (gvals[k - 1] + gvals[k]) * h;
            if phi > best_phi {
                best_phi = phi;
                best_k = k;
            }
        }
        let theta_hat = if best_k == 0 || best_k == grid {
            // Global max at a boundary node: the finite Warm root lies at/beyond the hard bound.
            out.boundary[p] = true;
            grid_theta(best_k)
        } else {
            // Interior max: Phi' = g crosses + -> - in [node-1, node+1]; refine by bisection.
            let mut evaluate = |theta| eval(p, theta).0;
            refine_wle_root(
                grid_theta(best_k - 1),
                grid_theta(best_k + 1),
                tol,
                &mut evaluate,
            )
            .map_err(|reason| format!("{reason} for person {p}"))?
        };
        out.theta[p] = theta_hat;
        let info = eval(p, theta_hat).1;
        out.se[p] = if info > 1e-12 {
            (1.0 / info).sqrt()
        } else {
            f64::NAN
        };
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
/// Wang, H.-P., Kuo, B.-C., & Chao, R.-C. (2010). A multidimensional
/// computerized adaptive testing system for enhancing the Chinese as second
/// language proficiency test. In H. Fujita & J. Sasaki (Eds.), *Selected topics
/// in education and educational technology* (pp. 245–252). WSEAS Press.
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
    cat_next_item_device(
        bank,
        y,
        administered,
        prior,
        q_theta,
        xi_rule,
        crate::Device::Auto,
    )
}

/// One CAT step with an explicit device for both EAP scoring and item
/// information. `Auto` is GPU-first; CPU remains the f64 fallback.
#[allow(clippy::too_many_arguments)]
pub fn cat_next_item_device(
    bank: &ItemBank<'_>,
    y: &[f64],
    administered: &[bool],
    prior: &PriorSpec,
    q_theta: usize,
    xi_rule: XiRule,
    device: crate::Device,
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
    let scores = score_eap_device(bank, y, administered, 1, prior, q_theta, xi_rule, device)?;
    let target_dim = (0..bank.n_dims)
        .max_by(|&a, &b| {
            scores.theta_sd[a]
                .partial_cmp(&scores.theta_sd[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let (item_info, _) =
        bank_information_device(bank, &scores.theta_eap, &scores.xi_eap, 1, device)?;
    let mut candidates: Vec<usize> = (0..n_items)
        .filter(|&i| !administered[i] && bank.factor_id[i] == target_dim)
        .collect();
    if candidates.is_empty() {
        candidates = (0..n_items).filter(|&i| !administered[i]).collect();
    }
    candidates.sort_by(|&a, &b| {
        item_info[b]
            .partial_cmp(&item_info[a])
            .unwrap_or(std::cmp::Ordering::Equal)
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
        bank.alpha,
        bank.b,
        bank.zeta,
        bank.tau,
        &config,
        bank.factor_id,
        &ctx,
        &grids,
    );
    let resp = index_responses(y, observed, n_persons, n_items);
    let cell = grids.q_t * grids.n_x;
    let mut l_buf = vec![0.0_f64; bank.n_dims * cell];
    let mut log_zdx = vec![0.0_f64; bank.n_dims * grids.n_x];
    let mut state = seed.max(1);
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut out = vec![0.0_f64; n_persons * n_draws * bank.n_dims];
    for p in 0..n_persons {
        let lp = person_pass(
            p,
            0,
            &tables,
            &resp,
            bank.factor_id,
            bank.n_dims,
            n_items,
            &grids,
            &mut l_buf,
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
#[path = "../../../tests/unit/scoring_cat_pv_tests.rs"]
mod cat_pv_tests;

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
    if theta_sd
        .iter()
        .any(|&value| !value.is_finite() || value < 0.0)
    {
        return Err("theta_sd values must be finite and non-negative".into());
    }
    let mut out = vec![f64::NAN; n_dims];
    for d in 0..n_dims {
        let n = n_persons as f64;
        let mean: f64 = (0..n_persons)
            .map(|p| theta_eap[p * n_dims + d])
            .sum::<f64>()
            / n;
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
#[path = "../../../tests/unit/scoring_reliability_tests.rs"]
mod reliability_tests;

#[cfg(test)]
#[path = "../../../tests/unit/scoring_validate_branch_tests.rs"]
mod validate_branch_tests;

#[cfg(all(test, feature = "gpu", not(coverage)))]
#[path = "../../../tests/unit/scoring_gpu_score_tests.rs"]
mod gpu_score_tests;

#[cfg(test)]
#[path = "../../../tests/unit/scoring_wle_tests.rs"]
mod wle_tests;
