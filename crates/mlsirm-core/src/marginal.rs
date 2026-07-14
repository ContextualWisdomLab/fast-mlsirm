//! Marginal maximum likelihood (MMLE) via EM for the latent-space model family,
//! with optional multigroup and multilevel population structures.
//!
//! Person latents `(theta_p in R^D, xi_p in R^K)` are random effects integrated
//! out by Gauss-Hermite quadrature; item quantities `(alpha, b, zeta, tau)` are
//! structural parameters (Bock & Aitkin 1981 extended to the simple-structure
//! latent-space contract; see docs/mmle_marginal_lsirm_design.md). Tractability
//! rests on the simple-structure factorization: conditional on `xi_p` the trait
//! dimensions are independent, so the per-person integral costs
//! `Q_xi^K * (Q_u) * sum_d Q_theta` instead of `Q^(1+D+K)`.
//!
//! Population structures (mutually exclusive):
//! - Single: `theta_pd ~ N(0,1)`.
//! - Multigroup (Bock & Zimowski 1997): `theta_pd ~ N(mu_gd, sigma_gd^2)`,
//!   common item parameters, reference group 0 pinned at `N(0,1)`.
//! - Multilevel (Fox & Glas 2001 random intercept):
//!   `theta_pd = sigma_u * u_c + e_pd`, `u_c ~ N(0,1)` shared within cluster
//!   `c`, `e_pd ~ N(0,1)`; `sigma_u` estimated.
//!
//! The M-step is generalized EM: a few Armijo-backtracked gradient ascent steps
//! per item on the expected complete-data log-likelihood (plus the L2 penalties
//! of `PenaltyConfig`, i.e. MAP-flavored MMLE that keeps sparse items finite),
//! then a backtracked Newton step for the global `tau`, then closed-form
//! population-moment updates. Every step is deterministic — the Rust<->NumPy
//! parity contract for this estimator is exact algorithm equality.

use crate::quadrature::gh_rule;
use crate::{model_exec_flags, Device, ModelConfig, ModelType, PenaltyConfig};

#[derive(Clone, Debug)]
pub enum PopulationSpec {
    Single,
    /// `group_id[p] in 0..n_groups`; group 0 is the fixed `N(0,1)` reference.
    Multigroup { group_id: Vec<usize>, n_groups: usize },
    /// `cluster_id[p] in 0..n_clusters`.
    Multilevel { cluster_id: Vec<usize>, n_clusters: usize },
}

#[derive(Clone, Copy, Debug)]
pub struct MarginalConfig {
    /// Gauss-Hermite nodes for each trait dimension (must be a supported rule).
    pub q_theta: usize,
    /// Gauss-Hermite nodes per latent-space axis (tensor grid of `q_xi^K`).
    pub q_xi: usize,
    /// Gauss-Hermite nodes for the multilevel random intercept.
    pub q_u: usize,
    pub max_iter: usize,
    /// Convergence: absolute change of the penalized marginal log-likelihood.
    pub tol: f64,
    /// Gradient-ascent steps per item per M-step.
    pub m_steps: usize,
    /// Initial radius of the deterministic item-position circle init.
    pub init_zeta_radius: f64,
    /// Initial `sigma_u` (multilevel only).
    pub init_sigma_u: f64,
}

impl Default for MarginalConfig {
    fn default() -> Self {
        Self {
            q_theta: 21,
            q_xi: 11,
            q_u: 15,
            max_iter: 200,
            tol: 1e-5,
            m_steps: 4,
            init_zeta_radius: 0.5,
            init_sigma_u: 0.3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MarginalResult {
    pub alpha: Vec<f64>,
    pub b: Vec<f64>,
    /// Item positions, row-major `n_items x latent_dim`, PCA-aligned.
    pub zeta: Vec<f64>,
    pub tau: f64,
    /// EAP trait scores, row-major `n_persons x n_dims`.
    pub theta_eap: Vec<f64>,
    /// Posterior SDs matching `theta_eap`.
    pub theta_sd: Vec<f64>,
    /// EAP person positions, row-major `n_persons x latent_dim`, PCA-aligned.
    pub xi_eap: Vec<f64>,
    /// Multigroup: `n_groups x n_dims` trait means (empty otherwise).
    pub mu: Vec<f64>,
    /// Multigroup: `n_groups x n_dims` trait SDs (empty otherwise).
    pub sigma: Vec<f64>,
    /// Multilevel: random-intercept SD (0 otherwise).
    pub sigma_u: f64,
    /// Multilevel: EAP cluster intercepts (empty otherwise).
    pub u_eap: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
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

/// Tensor-product latent-space grid: `q_xi^K` nodes with product weights.
fn xi_grid(q_xi: usize, latent_dim: usize) -> (Vec<f64>, Vec<f64>) {
    let (nodes, weights) = gh_rule(q_xi).expect("validated earlier");
    let n_grid = q_xi.pow(latent_dim as u32);
    let mut grid = vec![0.0_f64; n_grid * latent_dim];
    let mut logw = vec![0.0_f64; n_grid];
    for j in 0..n_grid {
        let mut rem = j;
        for k in 0..latent_dim {
            let idx = rem % q_xi;
            rem /= q_xi;
            grid[j * latent_dim + k] = nodes[idx];
            logw[j] += weights[idx].ln();
        }
    }
    (grid, logw)
}

/// Population contexts: the trait value plugged into `eta` is
/// `theta(t, s, d) = shift[s*D+d] + scale[s*D+d] * t`.
struct Contexts {
    n_ctx: usize,
    shift: Vec<f64>,
    scale: Vec<f64>,
    /// Multilevel: standard-normal u nodes and log-weights; empty otherwise.
    u_nodes: Vec<f64>,
    u_logw: Vec<f64>,
}

fn build_contexts(
    pop: &PopulationSpec,
    mu: &[f64],
    sigma: &[f64],
    sigma_u: f64,
    n_dims: usize,
    q_u: usize,
) -> Contexts {
    match pop {
        PopulationSpec::Single => Contexts {
            n_ctx: 1,
            shift: vec![0.0; n_dims],
            scale: vec![1.0; n_dims],
            u_nodes: Vec::new(),
            u_logw: Vec::new(),
        },
        PopulationSpec::Multigroup { n_groups, .. } => Contexts {
            n_ctx: *n_groups,
            shift: mu.to_vec(),
            scale: sigma.to_vec(),
            u_nodes: Vec::new(),
            u_logw: Vec::new(),
        },
        PopulationSpec::Multilevel { .. } => {
            let (nodes, weights) = gh_rule(q_u).expect("validated earlier");
            let mut shift = vec![0.0_f64; q_u * n_dims];
            let mut scale = vec![1.0_f64; q_u * n_dims];
            for (v, &node) in nodes.iter().enumerate() {
                for d in 0..n_dims {
                    shift[v * n_dims + d] = sigma_u * node;
                    scale[v * n_dims + d] = 1.0;
                }
            }
            Contexts {
                n_ctx: q_u,
                shift,
                scale,
                u_nodes: nodes.to_vec(),
                u_logw: weights.iter().map(|w| w.ln()).collect(),
            }
        }
    }
}

/// Item-response tables and their per-dimension all-zero baseline.
/// `logp1`/`logp0` are `[ctx][item][t][x]` flattened; `c0` is `[ctx][dim][t][x]`.
struct Tables {
    logp1: Vec<f64>,
    logp0: Vec<f64>,
    c0: Vec<f64>,
}

struct Grids {
    t_nodes: Vec<f64>,
    t_logw: Vec<f64>,
    x_grid: Vec<f64>,
    x_logw: Vec<f64>,
    q_t: usize,
    n_x: usize,
}

#[allow(clippy::too_many_arguments)]
fn eta_at(
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: f64,
    free_alpha: bool,
    uses_space: bool,
    latent_dim: usize,
    eps_distance: f64,
    i: usize,
    theta: f64,
    x_node: &[f64],
) -> f64 {
    let a = if free_alpha { alpha[i].exp() } else { 1.0 };
    let mut eta = a * theta + b[i];
    if uses_space {
        let mut dist2 = eps_distance;
        for k in 0..latent_dim {
            let diff = x_node[k] - zeta[i * latent_dim + k];
            dist2 += diff * diff;
        }
        eta -= tau.exp() * dist2.sqrt();
    }
    eta
}

#[allow(clippy::too_many_arguments)]
fn build_tables(
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: f64,
    config: &ModelConfig,
    factor_id: &[usize],
    ctx: &Contexts,
    grids: &Grids,
) -> Tables {
    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let mut logp1 = vec![0.0_f64; ctx.n_ctx * n_items * cell];
    let mut logp0 = vec![0.0_f64; ctx.n_ctx * n_items * cell];
    let mut c0 = vec![0.0_f64; ctx.n_ctx * n_dims * cell];
    for s in 0..ctx.n_ctx {
        for i in 0..n_items {
            let d = factor_id[i];
            let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
            for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                let theta = shift + scale * node_t;
                for x in 0..n_x {
                    let eta = eta_at(
                        alpha,
                        b,
                        zeta,
                        tau,
                        free_alpha,
                        uses_space,
                        latent_dim,
                        config.eps_distance,
                        i,
                        theta,
                        &grids.x_grid[x * latent_dim..(x + 1) * latent_dim],
                    );
                    let idx = (s * n_items + i) * cell + t * n_x + x;
                    logp1[idx] = log_sigmoid(eta);
                    logp0[idx] = log_sigmoid(-eta);
                    c0[(s * n_dims + d) * cell + t * n_x + x] += logp0[idx];
                }
            }
        }
    }
    Tables { logp1, logp0, c0 }
}

/// Per-person response index: positives and missing cells, item-major.
struct ResponseIndex {
    pos: Vec<Vec<usize>>,
    miss: Vec<Vec<usize>>,
}

fn index_responses(y: &[f64], observed: &[bool], n_persons: usize, n_items: usize) -> ResponseIndex {
    let mut pos = vec![Vec::new(); n_persons];
    let mut miss = vec![Vec::new(); n_persons];
    for p in 0..n_persons {
        for i in 0..n_items {
            let idx = p * n_items + i;
            if !observed[idx] {
                miss[p].push(i);
            } else if y[idx] == 1.0 {
                pos[p].push(i);
            }
        }
    }
    ResponseIndex { pos, miss }
}

/// Build the person work buffer `l[d][t][x]` for person `p` in context `s`
/// and reduce it: returns (per-(d,x) logsumexp over t into `log_zdx`, and the
/// person log-marginal for this context).
#[allow(clippy::too_many_arguments)]
fn person_pass(
    p: usize,
    s: usize,
    tables: &Tables,
    resp: &ResponseIndex,
    factor_id: &[usize],
    n_dims: usize,
    n_items: usize,
    grids: &Grids,
    l_buf: &mut [f64],
    log_zdx: &mut [f64],
) -> f64 {
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    l_buf[..n_dims * cell].copy_from_slice(&tables.c0[s * n_dims * cell..(s + 1) * n_dims * cell]);
    for &i in &resp.miss[p] {
        let d = factor_id[i];
        let src = (s * n_items + i) * cell;
        for c in 0..cell {
            l_buf[d * cell + c] -= tables.logp0[src + c];
        }
    }
    for &i in &resp.pos[p] {
        let d = factor_id[i];
        let src = (s * n_items + i) * cell;
        for c in 0..cell {
            l_buf[d * cell + c] += tables.logp1[src + c] - tables.logp0[src + c];
        }
    }
    // logsumexp over t for each (d, x)
    for d in 0..n_dims {
        for x in 0..n_x {
            let mut max = f64::NEG_INFINITY;
            for t in 0..q_t {
                let v = grids.t_logw[t] + l_buf[d * cell + t * n_x + x];
                if v > max {
                    max = v;
                }
            }
            let mut sum = 0.0;
            for t in 0..q_t {
                sum += (grids.t_logw[t] + l_buf[d * cell + t * n_x + x] - max).exp();
            }
            log_zdx[d * n_x + x] = max + sum.ln();
        }
    }
    // logsumexp over x of (log w_x + sum_d log_zdx)
    let mut max = f64::NEG_INFINITY;
    for x in 0..n_x {
        let mut v = grids.x_logw[x];
        for d in 0..n_dims {
            v += log_zdx[d * n_x + x];
        }
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0;
    for x in 0..n_x {
        let mut v = grids.x_logw[x];
        for d in 0..n_dims {
            v += log_zdx[d * n_x + x];
        }
        sum += (v - max).exp();
    }
    max + sum.ln()
}

/// E-step accumulators (per context, on the (t, x) grid).
struct EStep {
    /// `[ctx][dim][t][x]` expected person counts.
    nbar: Vec<f64>,
    /// `[ctx][item][t][x]` expected positive counts.
    rbar: Vec<f64>,
    /// `[ctx][item][t][x]` expected missing-cell corrections.
    mbar: Vec<f64>,
    /// Marginal (unpenalized) log-likelihood.
    loglik: f64,
    /// Multilevel: `E[u_c^2 | Y]` summed over clusters (in u-node units of the
    /// standard normal, i.e. before scaling by `sigma_u`).
    sum_e_v2: f64,
    /// Multilevel: cluster posteriors over u nodes, `[cluster][v]`.
    cluster_post: Vec<f64>,
}

/// Accumulate person `p`'s posterior (weighted by `w_outer`, the cluster
/// posterior weight for multilevel or 1.0 otherwise) into the E-step arrays.
#[allow(clippy::too_many_arguments)]
fn accumulate_person(
    p: usize,
    s: usize,
    w_outer: f64,
    resp: &ResponseIndex,
    factor_id: &[usize],
    n_dims: usize,
    n_items: usize,
    grids: &Grids,
    l_buf: &[f64],
    log_zdx: &[f64],
    log_lp: f64,
    estep: &mut EStep,
    post_buf: &mut [f64],
) {
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    // post_x(x) = exp(log w_x + sum_d log_zdx - log_lp)
    // post(d,t,x) = post_x(x) * exp(log w_t + l - log_zdx)
    for x in 0..n_x {
        let mut lx = grids.x_logw[x] - log_lp;
        for d in 0..n_dims {
            lx += log_zdx[d * n_x + x];
        }
        let px = lx.exp();
        for d in 0..n_dims {
            for t in 0..q_t {
                let pt = (grids.t_logw[t] + l_buf[d * cell + t * n_x + x]
                    - log_zdx[d * n_x + x])
                    .exp();
                post_buf[d * cell + t * n_x + x] = w_outer * px * pt;
            }
        }
    }
    let base = s * n_dims * cell;
    for c in 0..n_dims * cell {
        estep.nbar[base + c] += post_buf[c];
    }
    for &i in &resp.pos[p] {
        let d = factor_id[i];
        let dst = (s * n_items + i) * cell;
        for c in 0..cell {
            estep.rbar[dst + c] += post_buf[d * cell + c];
        }
    }
    for &i in &resp.miss[p] {
        let d = factor_id[i];
        let dst = (s * n_items + i) * cell;
        for c in 0..cell {
            estep.mbar[dst + c] += post_buf[d * cell + c];
        }
    }
}

/// Route one E-step through the requested device. `Cpu` is the deterministic
/// f64 reference; `Gpu`/`Auto` run the wgpu f32 kernels when an adapter is
/// present (`Gpu` warns on fallback, `Auto` is silent). Accumulation noise on
/// the GPU is ~1e-4 relative — the M-step and the final EAP pass always stay
/// on the CPU in f64.
#[allow(clippy::too_many_arguments)]
fn e_step_device(
    device: Device,
    tables: &Tables,
    resp: &ResponseIndex,
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    ctx: &Contexts,
    grids: &Grids,
) -> EStep {
    match device {
        Device::Cpu => e_step(tables, resp, factor_id, config, pop, ctx, grids),
        Device::Gpu | Device::Auto => {
            #[cfg(all(feature = "gpu", not(coverage)))]
            {
                match e_step_gpu_adapter(tables, resp, factor_id, config, pop, ctx, grids) {
                    Some(estep) => return estep,
                    None => {
                        if matches!(device, Device::Gpu) {
                            eprintln!(
                                "fast-mlsirm: GPU device requested but no usable GPU adapter \
                                 was found; falling back to the CPU implementation."
                            );
                        }
                    }
                }
            }
            e_step(tables, resp, factor_id, config, pop, ctx, grids)
        }
    }
}

/// Bridge the CPU-side E-step data model onto the wgpu kernels.
#[cfg(all(feature = "gpu", not(coverage)))]
#[allow(clippy::too_many_arguments)]
fn e_step_gpu_adapter(
    tables: &Tables,
    resp: &ResponseIndex,
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    ctx: &Contexts,
    grids: &Grids,
) -> Option<EStep> {
    let (n_persons, n_items, n_dims) = (config.n_persons, config.n_items, config.n_dims);
    let cell = grids.q_t * grids.n_x;
    // Person-major CSR lists.
    let build_csr = |lists: &[Vec<usize>]| {
        let mut off = Vec::with_capacity(lists.len() + 1);
        let mut items = Vec::new();
        off.push(0u32);
        for l in lists {
            items.extend(l.iter().map(|&i| i as u32));
            off.push(items.len() as u32);
        }
        (off, items)
    };
    let (pos_off, pos_items) = build_csr(&resp.pos);
    let (miss_off, miss_items) = build_csr(&resp.miss);
    // Item-major person lists.
    let invert = |lists: &[Vec<usize>]| {
        let mut per_item: Vec<Vec<u32>> = vec![Vec::new(); n_items];
        for (p, l) in lists.iter().enumerate() {
            for &i in l {
                per_item[i].push(p as u32);
            }
        }
        let mut off = Vec::with_capacity(n_items + 1);
        let mut persons = Vec::new();
        off.push(0u32);
        for l in &per_item {
            persons.extend_from_slice(l);
            off.push(persons.len() as u32);
        }
        (off, persons)
    };
    let (item_pos_off, item_pos_persons) = invert(&resp.pos);
    let (item_miss_off, item_miss_persons) = invert(&resp.miss);

    let (all_ctx, ctx_of_person): (bool, Vec<u32>) = match pop {
        PopulationSpec::Single => (false, vec![0u32; n_persons]),
        PopulationSpec::Multigroup { group_id, .. } => {
            (false, group_id.iter().map(|&g| g as u32).collect())
        }
        PopulationSpec::Multilevel { .. } => (true, vec![0u32; n_persons]),
    };

    let inputs = crate::gpu_marginal::GpuEStepInputs {
        logp0: &tables.logp0,
        logp1: &tables.logp1,
        c0: &tables.c0,
        t_logw: &grids.t_logw,
        x_logw: &grids.x_logw,
        factor_id,
        ctx_of_person: &ctx_of_person,
        all_ctx,
        n_ctx: ctx.n_ctx,
        pos_off: &pos_off,
        pos_items: &pos_items,
        miss_off: &miss_off,
        miss_items: &miss_items,
        item_pos_off: &item_pos_off,
        item_pos_persons: &item_pos_persons,
        item_miss_off: &item_miss_off,
        item_miss_persons: &item_miss_persons,
    };

    let mut loglik = 0.0_f64;
    let mut sum_e_v2 = 0.0_f64;
    let n_ctx = ctx.n_ctx;
    let mut w_outer_fn = |lp: &[f64]| -> Vec<f64> {
        let mut w = vec![0.0_f64; n_ctx * n_persons];
        match pop {
            PopulationSpec::Single => {
                for p in 0..n_persons {
                    loglik += lp[p * n_ctx];
                    w[p] = 1.0;
                }
            }
            PopulationSpec::Multigroup { group_id, .. } => {
                for p in 0..n_persons {
                    let s = group_id[p];
                    loglik += lp[p * n_ctx + s];
                    w[s * n_persons + p] = 1.0;
                }
            }
            PopulationSpec::Multilevel { cluster_id, n_clusters } => {
                let mut log_cluster = vec![0.0_f64; n_clusters * n_ctx];
                for c in 0..*n_clusters {
                    log_cluster[c * n_ctx..(c + 1) * n_ctx].copy_from_slice(&ctx.u_logw);
                }
                for p in 0..n_persons {
                    let c = cluster_id[p];
                    for v in 0..n_ctx {
                        log_cluster[c * n_ctx + v] += lp[p * n_ctx + v];
                    }
                }
                let mut post = vec![0.0_f64; n_clusters * n_ctx];
                for c in 0..*n_clusters {
                    let row = &log_cluster[c * n_ctx..(c + 1) * n_ctx];
                    let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                    loglik += max + sum.ln();
                    for v in 0..n_ctx {
                        let pw = (row[v] - max).exp() / sum;
                        post[c * n_ctx + v] = pw;
                        sum_e_v2 += pw * ctx.u_nodes[v] * ctx.u_nodes[v];
                    }
                }
                for p in 0..n_persons {
                    let c = cluster_id[p];
                    for v in 0..n_ctx {
                        w[v * n_persons + p] = post[c * n_ctx + v];
                    }
                }
            }
        }
        w
    };

    let out = crate::gpu_marginal::e_step_gpu(config, &inputs, &mut w_outer_fn)?;
    debug_assert_eq!(out.nbar.len(), n_ctx * n_dims * cell);
    Some(EStep {
        nbar: out.nbar,
        rbar: out.rbar,
        mbar: out.mbar,
        loglik,
        sum_e_v2,
        cluster_post: Vec::new(),
    })
}

/// Full deterministic E-step over all persons (CPU f64 reference).
#[allow(clippy::too_many_arguments)]
fn e_step(
    tables: &Tables,
    resp: &ResponseIndex,
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    ctx: &Contexts,
    grids: &Grids,
) -> EStep {
    let (n_persons, n_items, n_dims) = (config.n_persons, config.n_items, config.n_dims);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let mut estep = EStep {
        nbar: vec![0.0; ctx.n_ctx * n_dims * cell],
        rbar: vec![0.0; ctx.n_ctx * n_items * cell],
        mbar: vec![0.0; ctx.n_ctx * n_items * cell],
        loglik: 0.0,
        sum_e_v2: 0.0,
        cluster_post: Vec::new(),
    };
    let mut l_buf = vec![0.0_f64; n_dims * cell];
    let mut log_zdx = vec![0.0_f64; n_dims * n_x];
    let mut post_buf = vec![0.0_f64; n_dims * cell];

    match pop {
        PopulationSpec::Single => {
            for p in 0..n_persons {
                let lp = person_pass(
                    p, 0, tables, resp, factor_id, n_dims, n_items, grids, &mut l_buf,
                    &mut log_zdx,
                );
                estep.loglik += lp;
                accumulate_person(
                    p, 0, 1.0, resp, factor_id, n_dims, n_items, grids, &l_buf, &log_zdx,
                    lp, &mut estep, &mut post_buf,
                );
            }
        }
        PopulationSpec::Multigroup { group_id, .. } => {
            for p in 0..n_persons {
                let s = group_id[p];
                let lp = person_pass(
                    p, s, tables, resp, factor_id, n_dims, n_items, grids, &mut l_buf,
                    &mut log_zdx,
                );
                estep.loglik += lp;
                accumulate_person(
                    p, s, 1.0, resp, factor_id, n_dims, n_items, grids, &l_buf, &log_zdx,
                    lp, &mut estep, &mut post_buf,
                );
            }
        }
        PopulationSpec::Multilevel { cluster_id, n_clusters } => {
            let q_u = ctx.n_ctx;
            // Pass 1: per-person conditional marginals log L_p(v).
            let mut lp_v = vec![0.0_f64; n_persons * q_u];
            for p in 0..n_persons {
                for v in 0..q_u {
                    lp_v[p * q_u + v] = person_pass(
                        p, v, tables, resp, factor_id, n_dims, n_items, grids, &mut l_buf,
                        &mut log_zdx,
                    );
                }
            }
            // Cluster posteriors over u nodes.
            let mut log_cluster = vec![0.0_f64; n_clusters * q_u];
            for c in 0..*n_clusters {
                for v in 0..q_u {
                    log_cluster[c * q_u + v] = ctx.u_logw[v];
                }
            }
            for p in 0..n_persons {
                let c = cluster_id[p];
                for v in 0..q_u {
                    log_cluster[c * q_u + v] += lp_v[p * q_u + v];
                }
            }
            estep.cluster_post = vec![0.0_f64; n_clusters * q_u];
            for c in 0..*n_clusters {
                let row = &log_cluster[c * q_u..(c + 1) * q_u];
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                estep.loglik += max + sum.ln();
                for v in 0..q_u {
                    let post = ((row[v] - max).exp()) / sum;
                    estep.cluster_post[c * q_u + v] = post;
                    estep.sum_e_v2 += post * ctx.u_nodes[v] * ctx.u_nodes[v];
                }
            }
            // Pass 2: accumulate expected counts weighted by cluster posteriors.
            for p in 0..n_persons {
                let c = cluster_id[p];
                for v in 0..q_u {
                    let w_outer = estep.cluster_post[c * q_u + v];
                    if w_outer < 1e-14 {
                        continue;
                    }
                    let lp = person_pass(
                        p, v, tables, resp, factor_id, n_dims, n_items, grids, &mut l_buf,
                        &mut log_zdx,
                    );
                    accumulate_person(
                        p, v, w_outer, resp, factor_id, n_dims, n_items, grids, &l_buf,
                        &log_zdx, lp, &mut estep, &mut post_buf,
                    );
                }
            }
        }
    }
    estep
}

/// Expected complete-data log-likelihood contribution of one item (plus its L2
/// penalties), used by the M-step line searches.
#[allow(clippy::too_many_arguments)]
fn item_q(
    i: usize,
    alpha_i: f64,
    b_i: f64,
    zeta_i: &[f64],
    tau: f64,
    estep: &EStep,
    ctx: &Contexts,
    grids: &Grids,
    config: &ModelConfig,
    factor_id: &[usize],
    penalty: &PenaltyConfig,
) -> f64 {
    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let d = factor_id[i];
    let mut q = 0.0;
    for s in 0..ctx.n_ctx {
        let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
        for (t, &node_t) in grids.t_nodes.iter().enumerate() {
            let theta = shift + scale * node_t;
            for x in 0..n_x {
                let idx = t * n_x + x;
                let n = estep.nbar[(s * n_dims + d) * cell + idx]
                    - estep.mbar[(s * n_items + i) * cell + idx];
                let r = estep.rbar[(s * n_items + i) * cell + idx];
                if n <= 0.0 && r <= 0.0 {
                    continue;
                }
                let eta = eta_at(
                    &[alpha_i],
                    &[b_i],
                    zeta_i,
                    tau,
                    free_alpha,
                    uses_space,
                    latent_dim,
                    config.eps_distance,
                    0,
                    theta,
                    &grids.x_grid[x * latent_dim..(x + 1) * latent_dim],
                );
                q += r * log_sigmoid(eta) + (n - r) * log_sigmoid(-eta);
            }
        }
    }
    q -= 0.5 * penalty.lambda_b * b_i * b_i;
    if free_alpha {
        let da = alpha_i - penalty.mu_alpha;
        q -= 0.5 * penalty.lambda_alpha * da * da;
    }
    if uses_space {
        let z2: f64 = zeta_i.iter().map(|z| z * z).sum();
        q -= 0.5 * penalty.lambda_zeta * z2;
    }
    q
}

/// One M-step over items: a few Armijo-backtracked gradient ascent steps each
/// (generalized EM — each accepted step increases the expected complete-data
/// objective).
#[allow(clippy::too_many_arguments)]
fn m_step_items(
    alpha: &mut [f64],
    b: &mut [f64],
    zeta: &mut [f64],
    tau: f64,
    estep: &EStep,
    ctx: &Contexts,
    grids: &Grids,
    config: &ModelConfig,
    factor_id: &[usize],
    penalty: &PenaltyConfig,
    m_steps: usize,
) {
    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let gamma = tau.exp();
    for i in 0..n_items {
        let d = factor_id[i];
        let mut zeta_i: Vec<f64> = zeta[i * latent_dim..(i + 1) * latent_dim].to_vec();
        let mut cur_q = item_q(
            i, alpha[i], b[i], &zeta_i, tau, estep, ctx, grids, config, factor_id, penalty,
        );
        for _ in 0..m_steps {
            // Analytic gradient of the expected complete-data objective, plus
            // the diagonal expected (Fisher) information used as a
            // preconditioner — plain gradient steps scale poorly across the
            // mixed (alpha, b, zeta) curvature and stall the slope updates.
            let a = if free_alpha { alpha[i].exp() } else { 1.0 };
            let (mut g_alpha, mut g_b) = (0.0_f64, 0.0_f64);
            let mut g_zeta = vec![0.0_f64; latent_dim];
            let (mut i_alpha, mut i_b) = (0.0_f64, 0.0_f64);
            let mut i_zeta = vec![0.0_f64; latent_dim];
            for s in 0..ctx.n_ctx {
                let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
                for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                    let theta = shift + scale * node_t;
                    for x in 0..n_x {
                        let idx = t * n_x + x;
                        let n = estep.nbar[(s * n_dims + d) * cell + idx]
                            - estep.mbar[(s * n_items + i) * cell + idx];
                        let r = estep.rbar[(s * n_items + i) * cell + idx];
                        if n <= 0.0 && r <= 0.0 {
                            continue;
                        }
                        let x_node = &grids.x_grid[x * latent_dim..(x + 1) * latent_dim];
                        let mut dist = 0.0;
                        let eta = {
                            let mut e = a * theta + b[i];
                            if uses_space {
                                let mut dist2 = config.eps_distance;
                                for k in 0..latent_dim {
                                    let diff = x_node[k] - zeta_i[k];
                                    dist2 += diff * diff;
                                }
                                dist = dist2.sqrt();
                                e -= gamma * dist;
                            }
                            e
                        };
                        let prob = sigmoid(eta);
                        let resid = r - n * prob;
                        let info = (n * prob * (1.0 - prob)).max(0.0);
                        g_b += resid;
                        i_b += info;
                        if free_alpha {
                            let deta = a * theta;
                            g_alpha += resid * deta;
                            i_alpha += info * deta * deta;
                        }
                        if uses_space {
                            for k in 0..latent_dim {
                                let deta = gamma * (x_node[k] - zeta_i[k]) / dist;
                                g_zeta[k] += resid * deta;
                                i_zeta[k] += info * deta * deta;
                            }
                        }
                    }
                }
            }
            g_b -= penalty.lambda_b * b[i];
            if free_alpha {
                g_alpha -= penalty.lambda_alpha * (alpha[i] - penalty.mu_alpha);
            }
            if uses_space {
                for k in 0..latent_dim {
                    g_zeta[k] -= penalty.lambda_zeta * zeta_i[k];
                }
            }
            // Preconditioned ascent direction d = g / (I + lambda), a damped
            // Fisher-scoring step per coordinate.
            let d_b = g_b / (i_b + penalty.lambda_b + 1e-8);
            let d_alpha = g_alpha / (i_alpha + penalty.lambda_alpha + 1e-8);
            let d_zeta: Vec<f64> = (0..latent_dim)
                .map(|k| g_zeta[k] / (i_zeta[k] + penalty.lambda_zeta + 1e-8))
                .collect();
            let mut slope = g_b * d_b + g_alpha * d_alpha;
            for k in 0..latent_dim {
                slope += g_zeta[k] * d_zeta[k];
            }
            if slope < 1e-20 {
                break;
            }
            let mut step = 1.0_f64;
            let mut accepted = false;
            for _ in 0..30 {
                let cand_b = b[i] + step * d_b;
                let cand_alpha = if free_alpha { alpha[i] + step * d_alpha } else { alpha[i] };
                let cand_zeta: Vec<f64> = (0..latent_dim)
                    .map(|k| zeta_i[k] + step * d_zeta[k])
                    .collect();
                let cand_q = item_q(
                    i, cand_alpha, cand_b, &cand_zeta, tau, estep, ctx, grids, config,
                    factor_id, penalty,
                );
                if cand_q > cur_q + 1e-4 * step * slope {
                    b[i] = cand_b;
                    if free_alpha {
                        alpha[i] = cand_alpha.clamp(-6.0, 3.0);
                    }
                    zeta_i = cand_zeta;
                    cur_q = cand_q;
                    accepted = true;
                    break;
                }
                step *= 0.5;
            }
            if !accepted {
                break;
            }
        }
        zeta[i * latent_dim..(i + 1) * latent_dim].copy_from_slice(&zeta_i);
    }
}

/// Global `tau` (log gamma) update: backtracked Newton-like step on the summed
/// expected objective. Skipped for models without the latent space.
#[allow(clippy::too_many_arguments)]
fn m_step_tau(
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: &mut f64,
    estep: &EStep,
    ctx: &Contexts,
    grids: &Grids,
    config: &ModelConfig,
    factor_id: &[usize],
    penalty: &PenaltyConfig,
) {
    let (_, uses_space) = model_exec_flags(config.model_type);
    if !uses_space {
        return;
    }
    let total_q = |tau_c: f64| -> f64 {
        let mut q = 0.0;
        for i in 0..config.n_items {
            q += item_q(
                i,
                alpha[i],
                b[i],
                &zeta[i * config.latent_dim..(i + 1) * config.latent_dim],
                tau_c,
                estep,
                ctx,
                grids,
                config,
                factor_id,
                penalty,
            );
        }
        // item_q already contains per-item penalties; add the tau penalty once.
        let dt = tau_c - penalty.mu_tau;
        q - 0.5 * penalty.lambda_tau * dt * dt
    };
    // Analytic gradient and expected-information Hessian in tau.
    let (free_alpha, _) = model_exec_flags(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let gamma = tau.exp();
    let (mut grad, mut info) = (0.0_f64, 0.0_f64);
    for i in 0..n_items {
        let d = factor_id[i];
        let a = if free_alpha { alpha[i].exp() } else { 1.0 };
        for s in 0..ctx.n_ctx {
            let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
            for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                let theta = shift + scale * node_t;
                for x in 0..n_x {
                    let idx = t * n_x + x;
                    let n = estep.nbar[(s * n_dims + d) * cell + idx]
                        - estep.mbar[(s * n_items + i) * cell + idx];
                    let r = estep.rbar[(s * n_items + i) * cell + idx];
                    if n <= 0.0 && r <= 0.0 {
                        continue;
                    }
                    let x_node = &grids.x_grid[x * latent_dim..(x + 1) * latent_dim];
                    let mut dist2 = config.eps_distance;
                    for k in 0..latent_dim {
                        let diff = x_node[k] - zeta[i * latent_dim + k];
                        dist2 += diff * diff;
                    }
                    let dist = dist2.sqrt();
                    let eta = a * theta + b[i] - gamma * dist;
                    let prob = sigmoid(eta);
                    let resid = r - n * prob;
                    let deta = -gamma * dist;
                    grad += resid * deta;
                    info += n * prob * (1.0 - prob) * deta * deta;
                }
            }
        }
    }
    grad -= penalty.lambda_tau * (*tau - penalty.mu_tau);
    info += penalty.lambda_tau;
    if info <= 0.0 {
        return;
    }
    let dir = grad / info;
    let cur = total_q(*tau);
    let mut step = 1.0_f64;
    for _ in 0..20 {
        let cand = (*tau + step * dir).clamp(-10.0, 5.0);
        if total_q(cand) > cur {
            *tau = cand;
            return;
        }
        step *= 0.5;
    }
}

/// Rotate `zeta` (and `xi_eap`) so the principal axes of the item configuration
/// align with the coordinate axes (rotation/reflection identifiability; see
/// design doc §4). Deterministic: Jacobi eigen-decomposition of the uncentered
/// second-moment matrix, sign fixed so each axis's largest-|coordinate| item is
/// positive.
fn pca_align(zeta: &mut [f64], xi: &mut [f64], n_items: usize, n_persons: usize, k: usize) {
    if k < 2 {
        // Only the reflection is free: fix the sign convention.
        if k == 1 {
            let (mut max_abs, mut sign) = (0.0_f64, 1.0_f64);
            for i in 0..n_items {
                if zeta[i].abs() > max_abs {
                    max_abs = zeta[i].abs();
                    sign = if zeta[i] >= 0.0 { 1.0 } else { -1.0 };
                }
            }
            if sign < 0.0 {
                zeta.iter_mut().for_each(|z| *z = -*z);
                xi.iter_mut().for_each(|z| *z = -*z);
            }
        }
        return;
    }
    // Uncentered K x K second moment of zeta.
    let mut m = vec![0.0_f64; k * k];
    for i in 0..n_items {
        for r in 0..k {
            for c in 0..k {
                m[r * k + c] += zeta[i * k + r] * zeta[i * k + c];
            }
        }
    }
    // Jacobi rotations (K is small: 2 or 3).
    let mut rot = vec![0.0_f64; k * k];
    for r in 0..k {
        rot[r * k + r] = 1.0;
    }
    for _ in 0..50 {
        let (mut p, mut q, mut off) = (0, 1, 0.0_f64);
        for r in 0..k {
            for c in (r + 1)..k {
                if m[r * k + c].abs() > off {
                    off = m[r * k + c].abs();
                    p = r;
                    q = c;
                }
            }
        }
        if off < 1e-12 {
            break;
        }
        let theta_ang = 0.5 * (2.0 * m[p * k + q]).atan2(m[p * k + p] - m[q * k + q]);
        let (c, s) = (theta_ang.cos(), theta_ang.sin());
        for r in 0..k {
            let (mrp, mrq) = (m[r * k + p], m[r * k + q]);
            m[r * k + p] = c * mrp + s * mrq;
            m[r * k + q] = -s * mrp + c * mrq;
        }
        for col in 0..k {
            let (mpc, mqc) = (m[p * k + col], m[q * k + col]);
            m[p * k + col] = c * mpc + s * mqc;
            m[q * k + col] = -s * mpc + c * mqc;
        }
        for r in 0..k {
            let (rp, rq) = (rot[r * k + p], rot[r * k + q]);
            rot[r * k + p] = c * rp + s * rq;
            rot[r * k + q] = -s * rp + c * rq;
        }
    }
    // Order columns by descending eigenvalue (diagonal of m).
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a2, &b2| {
        m[b2 * k + b2].partial_cmp(&m[a2 * k + a2]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let apply = |data: &mut [f64], n_rows: usize| {
        for row in 0..n_rows {
            let mut new = vec![0.0_f64; k];
            for (out_c, &src_c) in order.iter().enumerate() {
                for r in 0..k {
                    new[out_c] += data[row * k + r] * rot[r * k + src_c];
                }
            }
            data[row * k..(row + 1) * k].copy_from_slice(&new);
        }
    };
    apply(zeta, n_items);
    apply(xi, n_persons);
    // Sign convention per axis.
    for c in 0..k {
        let (mut max_abs, mut sign) = (0.0_f64, 1.0_f64);
        for i in 0..n_items {
            if zeta[i * k + c].abs() > max_abs {
                max_abs = zeta[i * k + c].abs();
                sign = if zeta[i * k + c] >= 0.0 { 1.0 } else { -1.0 };
            }
        }
        if sign < 0.0 {
            for i in 0..n_items {
                zeta[i * k + c] = -zeta[i * k + c];
            }
            for p in 0..n_persons {
                xi[p * k + c] = -xi[p * k + c];
            }
        }
    }
}

fn validate(
    y: &[f64],
    observed: &[bool],
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    mcfg: &MarginalConfig,
) -> Result<(), String> {
    let n = config
        .n_persons
        .checked_mul(config.n_items)
        .ok_or("n_persons * n_items overflows")?;
    if y.len() != n || observed.len() != n {
        return Err("y and observed must both have length n_persons * n_items".into());
    }
    if factor_id.len() != config.n_items {
        return Err("factor_id length must match number of items".into());
    }
    if factor_id.iter().any(|&d| d >= config.n_dims) {
        return Err("factor_id values must be in 0..n_dims-1".into());
    }
    if matches!(config.model_type, ModelType::Uls2plm | ModelType::Ulsrm) && config.n_dims != 1 {
        return Err("unidimensional models require n_dims == 1".into());
    }
    if config.n_dims == 0 || config.latent_dim == 0 {
        return Err("parameter dimensions must be positive".into());
    }
    if config.latent_dim > 3 {
        return Err("marginal estimator supports latent_dim <= 3 (grid quadrature)".into());
    }
    if config.eps_distance <= 0.0 {
        return Err("eps_distance must be positive".into());
    }
    for q in [mcfg.q_theta, mcfg.q_xi, mcfg.q_u] {
        if gh_rule(q).is_none() {
            return Err(format!(
                "unsupported quadrature size {q}; supported: {:?}",
                crate::quadrature::SUPPORTED_Q
            ));
        }
    }
    if y.iter().zip(observed).any(|(&v, &o)| o && v != 0.0 && v != 1.0) {
        return Err("observed responses must be 0 or 1".into());
    }
    match pop {
        PopulationSpec::Single => {}
        PopulationSpec::Multigroup { group_id, n_groups } => {
            if group_id.len() != config.n_persons {
                return Err("group_id length must match n_persons".into());
            }
            if *n_groups == 0 || group_id.iter().any(|&g| g >= *n_groups) {
                return Err("group_id values must be in 0..n_groups-1".into());
            }
        }
        PopulationSpec::Multilevel { cluster_id, n_clusters } => {
            if cluster_id.len() != config.n_persons {
                return Err("cluster_id length must match n_persons".into());
            }
            if *n_clusters == 0 || cluster_id.iter().any(|&c| c >= *n_clusters) {
                return Err("cluster_id values must be in 0..n_clusters-1".into());
            }
        }
    }
    Ok(())
}

/// Marginal EM calibration for the latent-space model family.
///
/// `y`/`observed` are row-major `n_persons * n_items`; missing cells (where
/// `observed` is false) are excluded from every product — MAR-safe by
/// construction. The `device` routes the E-step hot path: `Cpu` runs the f64
/// scalar reference; `Gpu`/`Auto` use the wgpu f32 kernels when an adapter is
/// present and otherwise fall back to the CPU path.
#[allow(clippy::too_many_arguments)]
pub fn fit_marginal(
    y: &[f64],
    observed: &[bool],
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    mcfg: &MarginalConfig,
    penalty: &PenaltyConfig,
    device: Device,
) -> Result<MarginalResult, String> {
    validate(y, observed, factor_id, config, pop, mcfg)?;
    let (_, uses_space) = model_exec_flags(config.model_type);
    let (n_persons, n_items, n_dims, latent_dim) =
        (config.n_persons, config.n_items, config.n_dims, config.latent_dim);

    let (t_nodes, t_weights) = gh_rule(mcfg.q_theta).expect("validated");
    let (x_grid, x_logw) = if uses_space {
        xi_grid(mcfg.q_xi, latent_dim)
    } else {
        // MIRT: a single dummy latent-space node at the origin with weight 1.
        (vec![0.0; latent_dim], vec![0.0])
    };
    let grids = Grids {
        t_nodes: t_nodes.to_vec(),
        t_logw: t_weights.iter().map(|w| w.ln()).collect(),
        n_x: x_logw.len(),
        x_grid,
        x_logw,
        q_t: mcfg.q_theta,
    };

    // --- Initialization (deterministic) ---
    let mut alpha = vec![0.0_f64; n_items];
    let mut b = vec![0.0_f64; n_items];
    for i in 0..n_items {
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..n_persons {
            let idx = p * n_items + i;
            if observed[idx] {
                num += y[idx];
                den += 1.0;
            }
        }
        let prop: f64 = if den > 0.0 { (num / den).clamp(0.02, 0.98) } else { 0.5 };
        b[i] = (prop / (1.0 - prop)).ln();
    }
    let mut zeta = vec![0.0_f64; n_items * latent_dim];
    if uses_space {
        for i in 0..n_items {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (n_items.max(1) as f64);
            zeta[i * latent_dim] = mcfg.init_zeta_radius * angle.cos();
            if latent_dim >= 2 {
                zeta[i * latent_dim + 1] = mcfg.init_zeta_radius * angle.sin();
            }
            if latent_dim >= 3 {
                zeta[i * latent_dim + 2] =
                    mcfg.init_zeta_radius * (2.0 * angle).cos() * 0.5;
            }
        }
    }
    let mut tau = if uses_space { 0.0 } else { -30.0 };
    let (n_groups, n_clusters) = match pop {
        PopulationSpec::Multigroup { n_groups, .. } => (*n_groups, 0),
        PopulationSpec::Multilevel { n_clusters, .. } => (0, *n_clusters),
        PopulationSpec::Single => (0, 0),
    };
    let mut mu = vec![0.0_f64; n_groups * n_dims];
    let mut sigma = vec![1.0_f64; n_groups * n_dims];
    let mut sigma_u = if n_clusters > 0 { mcfg.init_sigma_u } else { 0.0 };

    let resp = index_responses(y, observed, n_persons, n_items);
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;

    for iteration in 0..mcfg.max_iter {
        let ctx = build_contexts(pop, &mu, &sigma, sigma_u, n_dims, mcfg.q_u);
        let tables = build_tables(&alpha, &b, &zeta, tau, config, factor_id, &ctx, &grids);
        let estep =
            e_step_device(device, &tables, &resp, factor_id, config, pop, &ctx, &grids);
        loglik_trace.push(estep.loglik);

        // M-step: items, then tau, then population parameters.
        m_step_items(
            &mut alpha, &mut b, &mut zeta, tau, &estep, &ctx, &grids, config, factor_id,
            penalty, mcfg.m_steps,
        );
        m_step_tau(
            &alpha, &b, &zeta, &mut tau, &estep, &ctx, &grids, config, factor_id, penalty,
        );
        match pop {
            PopulationSpec::Single => {}
            PopulationSpec::Multigroup { .. } => {
                let cell = grids.q_t * grids.n_x;
                for g in 1..n_groups {
                    for d in 0..n_dims {
                        let (shift, scale) = (mu[g * n_dims + d], sigma[g * n_dims + d]);
                        let (mut w_sum, mut m1, mut m2) = (0.0_f64, 0.0_f64, 0.0_f64);
                        for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                            let theta = shift + scale * node_t;
                            for x in 0..grids.n_x {
                                let w = estep.nbar[(g * n_dims + d) * cell + t * grids.n_x + x];
                                w_sum += w;
                                m1 += w * theta;
                                m2 += w * theta * theta;
                            }
                        }
                        if w_sum > 1e-10 {
                            let mean = m1 / w_sum;
                            let var = (m2 / w_sum - mean * mean).max(0.01);
                            mu[g * n_dims + d] = mean;
                            sigma[g * n_dims + d] = var.sqrt().clamp(0.1, 10.0);
                        }
                    }
                }
            }
            PopulationSpec::Multilevel { .. } => {
                if n_clusters > 0 {
                    let e_v2 = estep.sum_e_v2 / n_clusters as f64;
                    // theta = sigma_u * v + e; EM update of the intercept scale.
                    sigma_u = (sigma_u * sigma_u * e_v2).sqrt().clamp(0.0, 10.0);
                }
            }
        }
        if iteration > 0 {
            let delta = (loglik_trace[iteration] - loglik_trace[iteration - 1]).abs();
            if delta < mcfg.tol {
                converged = true;
                break;
            }
        }
    }

    // --- Final EAP pass with the converged parameters ---
    let ctx = build_contexts(pop, &mu, &sigma, sigma_u, n_dims, mcfg.q_u);
    let tables = build_tables(&alpha, &b, &zeta, tau, config, factor_id, &ctx, &grids);
    let cell = grids.q_t * grids.n_x;
    let mut l_buf = vec![0.0_f64; n_dims * cell];
    let mut log_zdx = vec![0.0_f64; n_dims * grids.n_x];
    let mut theta_eap = vec![0.0_f64; n_persons * n_dims];
    let mut theta_m2 = vec![0.0_f64; n_persons * n_dims];
    let mut xi_eap = vec![0.0_f64; n_persons * latent_dim];
    let mut u_eap = vec![0.0_f64; n_clusters];

    // Cluster posteriors for the final parameters (multilevel).
    let cluster_post: Vec<f64> = match pop {
        PopulationSpec::Multilevel { cluster_id, n_clusters } => {
            let q_u = ctx.n_ctx;
            let mut log_cluster = vec![0.0_f64; n_clusters * q_u];
            for c in 0..*n_clusters {
                for v in 0..q_u {
                    log_cluster[c * q_u + v] = ctx.u_logw[v];
                }
            }
            for p in 0..n_persons {
                let c = cluster_id[p];
                for v in 0..q_u {
                    log_cluster[c * q_u + v] += person_pass(
                        p, v, &tables, &resp, factor_id, n_dims, n_items, &grids, &mut l_buf,
                        &mut log_zdx,
                    );
                }
            }
            let mut post = vec![0.0_f64; n_clusters * q_u];
            for c in 0..*n_clusters {
                let row = &log_cluster[c * q_u..(c + 1) * q_u];
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = row.iter().map(|&v| (v - max).exp()).sum();
                for v in 0..q_u {
                    post[c * q_u + v] = (row[v] - max).exp() / sum;
                    u_eap[c] += post[c * q_u + v] * sigma_u * ctx.u_nodes[v];
                }
            }
            post
        }
        _ => Vec::new(),
    };

    for p in 0..n_persons {
        let (contexts, weights): (Vec<usize>, Vec<f64>) = match pop {
            PopulationSpec::Single => (vec![0], vec![1.0]),
            PopulationSpec::Multigroup { group_id, .. } => (vec![group_id[p]], vec![1.0]),
            PopulationSpec::Multilevel { cluster_id, .. } => {
                let c = cluster_id[p];
                let q_u = ctx.n_ctx;
                ((0..q_u).collect(), cluster_post[c * q_u..(c + 1) * q_u].to_vec())
            }
        };
        for (&s, &w_outer) in contexts.iter().zip(&weights) {
            if w_outer < 1e-14 {
                continue;
            }
            let lp = person_pass(
                p, s, &tables, &resp, factor_id, n_dims, n_items, &grids, &mut l_buf,
                &mut log_zdx,
            );
            for x in 0..grids.n_x {
                let mut lx = grids.x_logw[x] - lp;
                for d in 0..n_dims {
                    lx += log_zdx[d * grids.n_x + x];
                }
                let px = w_outer * lx.exp();
                for k in 0..latent_dim {
                    xi_eap[p * latent_dim + k] += px * grids.x_grid[x * latent_dim + k];
                }
                for d in 0..n_dims {
                    let (shift, scale) =
                        (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
                    for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                        let theta = shift + scale * node_t;
                        let pt = (grids.t_logw[t] + l_buf[d * cell + t * grids.n_x + x]
                            - log_zdx[d * grids.n_x + x])
                            .exp();
                        theta_eap[p * n_dims + d] += px * pt * theta;
                        theta_m2[p * n_dims + d] += px * pt * theta * theta;
                    }
                }
            }
        }
    }
    let theta_sd: Vec<f64> = theta_eap
        .iter()
        .zip(&theta_m2)
        .map(|(&m, &m2)| (m2 - m * m).max(0.0).sqrt())
        .collect();

    if uses_space {
        pca_align(&mut zeta, &mut xi_eap, n_items, n_persons, latent_dim);
    }

    let n_iter = loglik_trace.len();
    Ok(MarginalResult {
        alpha,
        b,
        zeta,
        tau,
        theta_eap,
        theta_sd,
        xi_eap,
        mu,
        sigma,
        sigma_u,
        u_eap,
        loglik_trace,
        n_iter,
        converged,
    })
}
