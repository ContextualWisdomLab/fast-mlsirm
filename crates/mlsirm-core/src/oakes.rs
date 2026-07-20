//! Item-parameter standard errors for the marginal estimator via Oakes'
//! identity (Oakes 1999), the method Pritikin (2017, Cogent Psychology,
//! "A comparison of parameter covariance estimation methods for item response
//! models in an expectation-maximization framework") recommends over the
//! supplemented-EM family: at the MML solution
//!
//! `d^2 l / d xi d xi' = d^2 Q(xi | xi0) / d xi d xi'
//!                     + d^2 Q(xi | xi0) / d xi d xi0'   at xi0 = xi`
//!
//! The first term (the M-step Hessian at a FIXED posterior) is differenced
//! from the analytic Q-gradient without re-running the E-step; the second
//! (cross) term needs one E-step per perturbed coordinate — `k + 1` E-steps
//! total, versus `2k` for a central difference of the marginal score.
//!
//! Scope: item-side parameters plus `tau` (per-item `alpha`/`b`/`zeta` as the
//! model frees them), conditional on the fitted population parameters; the
//! penalized (MAP) curvature is used, matching the estimator's objective.
//! Anchors, zero inflation and covariates are not supported here. E-steps run
//! on the CPU in f64 — finite differences would drown in the f32 GPU noise.
//!
//! # References
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479–482. https://doi.org/10.1111/1467-9868.00188
//!
//! Pritikin, J. N. (2017). A comparison of parameter covariance estimation
//! methods for item response models in an expectation-maximization framework.
//! *Cogent Psychology, 4*(1), Article 1279435.
//! https://doi.org/10.1080/23311908.2017.1279435

use crate::marginal::{
    build_contexts_pub as build_contexts, build_tables, e_step_pub as e_step, index_responses,
    Contexts, EStepCounts, Grids, MarginalConfig, PopulationSpec, XiRuleKind,
};
use crate::nodes::build_xi_nodes;
use crate::quadrature::gh_rule;
use crate::{model_exec_flags, ModelConfig, PenaltyConfig};

pub struct OakesResult {
    /// Parameter labels in vector order (`alpha:i`, `b:i`, `zeta:i:k`, `tau`).
    pub labels: Vec<String>,
    /// Standard errors (sqrt of the diagonal of the inverse information).
    pub se: Vec<f64>,
    /// Observed (penalized) information matrix, row-major `k x k`.
    pub information: Vec<f64>,
}

struct ParamVec {
    free_alpha: bool,
    uses_space: bool,
    /// tau occupies a slot only for the distance interaction kind.
    tau_free: bool,
    n_items: usize,
    latent_dim: usize,
}

impl ParamVec {
    fn len(&self) -> usize {
        let per_item =
            1 + usize::from(self.free_alpha) + if self.uses_space { self.latent_dim } else { 0 };
        self.n_items * per_item + usize::from(self.tau_free)
    }

    fn labels(&self) -> Vec<String> {
        let mut out = Vec::new();
        for i in 0..self.n_items {
            if self.free_alpha {
                out.push(format!("alpha:{i}"));
            }
            out.push(format!("b:{i}"));
            if self.uses_space {
                for k in 0..self.latent_dim {
                    out.push(format!("zeta:{i}:{k}"));
                }
            }
        }
        if self.tau_free {
            out.push("tau".into());
        }
        out
    }

    fn pack(&self, alpha: &[f64], b: &[f64], zeta: &[f64], tau: f64) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.len());
        for i in 0..self.n_items {
            if self.free_alpha {
                v.push(alpha[i]);
            }
            v.push(b[i]);
            if self.uses_space {
                for k in 0..self.latent_dim {
                    v.push(zeta[i * self.latent_dim + k]);
                }
            }
        }
        if self.tau_free {
            v.push(tau);
        }
        v
    }

    fn unpack(&self, v: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let mut alpha = vec![0.0_f64; self.n_items];
        let mut b = vec![0.0_f64; self.n_items];
        let mut zeta = vec![0.0_f64; self.n_items * self.latent_dim];
        let mut cursor = 0usize;
        for i in 0..self.n_items {
            if self.free_alpha {
                alpha[i] = v[cursor];
                cursor += 1;
            }
            b[i] = v[cursor];
            cursor += 1;
            if self.uses_space {
                for k in 0..self.latent_dim {
                    zeta[i * self.latent_dim + k] = v[cursor];
                    cursor += 1;
                }
            }
        }
        let tau = if self.tau_free { v[cursor] } else { -30.0 };
        (alpha, b, zeta, tau)
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

/// Analytic gradient of the penalized expected complete-data log-likelihood
/// `Q(xi | posterior counts)` with respect to the packed parameter vector.
#[allow(clippy::too_many_arguments)]
fn q_gradient(
    pv: &ParamVec,
    xi: &[f64],
    counts: &EStepCounts,
    ctx: &Contexts,
    grids: &Grids,
    config: &ModelConfig,
    factor_id: &[usize],
    penalty: &PenaltyConfig,
) -> Vec<f64> {
    let (alpha, b, zeta, tau) = pv.unpack(xi);
    let (free_alpha, uses_space) = (pv.free_alpha, pv.uses_space);
    let kind = crate::interaction_kind(config.model_type);
    let (n_items, n_dims, latent_dim) = (config.n_items, config.n_dims, config.latent_dim);
    let (q_t, n_x) = (grids.q_t, grids.n_x);
    let cell = q_t * n_x;
    let gamma = tau.exp();
    let mut g = vec![0.0_f64; pv.len()];
    let mut cursor = 0usize;
    let mut g_tau = 0.0_f64;
    for i in 0..n_items {
        let d = factor_id[i];
        let a = if free_alpha { alpha[i].exp() } else { 1.0 };
        let (mut g_alpha, mut g_b) = (0.0_f64, 0.0_f64);
        let mut g_zeta = vec![0.0_f64; latent_dim];
        for s in 0..ctx.n_ctx {
            let (shift, scale) = (ctx.shift[s * n_dims + d], ctx.scale[s * n_dims + d]);
            for (t, &node_t) in grids.t_nodes.iter().enumerate() {
                let theta = shift + scale * node_t;
                for x in 0..n_x {
                    let idx = t * n_x + x;
                    let n = counts.nbar[(s * n_dims + d) * cell + idx]
                        - counts.mbar[(s * n_items + i) * cell + idx];
                    let r = counts.rbar[(s * n_items + i) * cell + idx];
                    if n <= 0.0 && r <= 0.0 {
                        continue;
                    }
                    let mut eta = a * theta + b[i];
                    let mut dist = 1.0;
                    match kind {
                        crate::InteractionKind::None => {}
                        crate::InteractionKind::Distance => {
                            let mut dist2 = config.eps_distance;
                            for k in 0..latent_dim {
                                let diff =
                                    grids.x_grid[x * latent_dim + k] - zeta[i * latent_dim + k];
                                dist2 += diff * diff;
                            }
                            dist = dist2.sqrt();
                            eta -= gamma * dist;
                        }
                        crate::InteractionKind::Inner => {
                            for k in 0..latent_dim {
                                eta += zeta[i * latent_dim + k] * grids.x_grid[x * latent_dim + k];
                            }
                        }
                    }
                    let resid = r - n * sigmoid(eta);
                    g_b += resid;
                    if free_alpha {
                        g_alpha += resid * a * theta;
                    }
                    if uses_space {
                        for k in 0..latent_dim {
                            let deta = if kind == crate::InteractionKind::Distance {
                                gamma
                                    * (grids.x_grid[x * latent_dim + k] - zeta[i * latent_dim + k])
                                    / dist
                            } else {
                                debug_assert_eq!(kind, crate::InteractionKind::Inner);
                                grids.x_grid[x * latent_dim + k]
                            };
                            g_zeta[k] += resid * deta;
                        }
                        if kind == crate::InteractionKind::Distance {
                            g_tau += resid * (-gamma * dist);
                        }
                    }
                }
            }
        }
        g_b -= penalty.lambda_b * b[i];
        if free_alpha {
            g_alpha -= penalty.lambda_alpha * (alpha[i] - penalty.mu_alpha);
            g[cursor] = g_alpha;
            cursor += 1;
        }
        g[cursor] = g_b;
        cursor += 1;
        if uses_space {
            for k in 0..latent_dim {
                g[cursor] = g_zeta[k] - penalty.lambda_zeta * zeta[i * latent_dim + k];
                cursor += 1;
            }
        }
    }
    if pv.tau_free {
        g[cursor] = g_tau - penalty.lambda_tau * (tau - penalty.mu_tau);
    }
    g
}

/// Invert a symmetric positive-definite matrix in place (Gauss-Jordan with
/// partial pivoting). Returns None when (numerically) singular.
fn invert(mut m: Vec<f64>, k: usize) -> Option<Vec<f64>> {
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r * k + col].abs() > m[piv * k + col].abs() {
                piv = r;
            }
        }
        if m[piv * k + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(col * k + c, piv * k + c);
                inv.swap(col * k + c, piv * k + c);
            }
        }
        let d = m[col * k + col];
        for c in 0..k {
            m[col * k + c] /= d;
            inv[col * k + c] /= d;
        }
        for r in 0..k {
            if r != col {
                let f = m[r * k + col];
                if f != 0.0 {
                    for c in 0..k {
                        m[r * k + c] -= f * m[col * k + c];
                        inv[r * k + c] -= f * inv[col * k + c];
                    }
                }
            }
        }
    }
    Some(inv)
}

fn invert_information(information: Vec<f64>, k: usize) -> Result<Vec<f64>, String> {
    invert(information, k)
        .ok_or_else(|| "observed information is singular; SEs unavailable".to_string())
}

/// Observed-information standard errors via Oakes' identity at the fitted
/// parameters. `h` is the finite-difference step (default 1e-5 scaled).
#[allow(clippy::too_many_arguments)]
pub fn observed_information_oakes(
    y: &[f64],
    observed: &[bool],
    factor_id: &[usize],
    config: &ModelConfig,
    pop: &PopulationSpec,
    mcfg: &MarginalConfig,
    penalty: &PenaltyConfig,
    alpha: &[f64],
    b: &[f64],
    zeta: &[f64],
    tau: f64,
    mu: &[f64],
    sigma: &[f64],
    sigma_u: f64,
    h: f64,
) -> Result<OakesResult, String> {
    if mcfg.zero_inflation {
        return Err("Oakes SEs with the zero-inflated mixture are not supported yet".into());
    }
    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
    let pv = ParamVec {
        free_alpha,
        uses_space,
        tau_free: crate::interaction_kind(config.model_type) == crate::InteractionKind::Distance,
        n_items: config.n_items,
        latent_dim: config.latent_dim,
    };
    let k = pv.len();
    let (t_nodes, t_weights) =
        gh_rule(mcfg.q_theta).ok_or_else(|| "unsupported q_theta".to_string())?;
    let (x_grid, x_logw) = if uses_space {
        let rule = match mcfg.xi_rule {
            XiRuleKind::GaussHermite => crate::nodes::XiRule::GaussHermite { q_xi: mcfg.q_xi },
            XiRuleKind::Halton => crate::nodes::XiRule::Halton {
                n: mcfg.xi_points,
                shift_seed: mcfg.xi_seed,
            },
            XiRuleKind::MonteCarlo => crate::nodes::XiRule::MonteCarlo {
                n: mcfg.xi_points,
                seed: mcfg.xi_seed.max(1),
            },
        };
        let nodes = build_xi_nodes(rule, config.latent_dim)?;
        (nodes.grid, nodes.logw)
    } else {
        (vec![0.0; config.latent_dim], vec![0.0])
    };
    let grids = Grids {
        t_nodes: t_nodes.to_vec(),
        t_logw: t_weights.iter().map(|w| w.ln()).collect(),
        n_x: x_logw.len(),
        x_grid,
        x_logw,
        q_t: mcfg.q_theta,
    };
    let ctx = build_contexts(pop, mu, sigma, sigma_u, config.n_dims, mcfg.q_u);
    let resp = index_responses(y, observed, config.n_persons, config.n_items);

    let estep_at = |xi_vec: &[f64]| -> EStepCounts {
        let (a0, b0, z0, t0) = pv.unpack(xi_vec);
        let tables = build_tables(&a0, &b0, &z0, t0, config, factor_id, &ctx, &grids);
        e_step(&tables, &resp, factor_id, config, pop, &ctx, &grids)
    };

    let xi0 = pv.pack(alpha, b, zeta, tau);
    let counts0 = estep_at(&xi0);

    // Term A: M-step Hessian — central FD of the Q-gradient over xi at the
    // FIXED base posterior (no E-steps).
    let mut info = vec![0.0_f64; k * k];
    for j in 0..k {
        let hj = h * (1.0 + xi0[j].abs());
        let mut xp = xi0.clone();
        xp[j] += hj;
        let mut xm = xi0.clone();
        xm[j] -= hj;
        let gp = q_gradient(&pv, &xp, &counts0, &ctx, &grids, config, factor_id, penalty);
        let gm = q_gradient(&pv, &xm, &counts0, &ctx, &grids, config, factor_id, penalty);
        for c in 0..k {
            info[j * k + c] += (gp[c] - gm[c]) / (2.0 * hj);
        }
    }
    // Term B: cross derivative — forward FD over xi0 (one E-step per
    // coordinate), gradient evaluated at the base xi.
    let g0 = q_gradient(
        &pv, &xi0, &counts0, &ctx, &grids, config, factor_id, penalty,
    );
    for j in 0..k {
        let hj = h * (1.0 + xi0[j].abs());
        let mut x0p = xi0.clone();
        x0p[j] += hj;
        let counts_p = estep_at(&x0p);
        let gp = q_gradient(
            &pv, &xi0, &counts_p, &ctx, &grids, config, factor_id, penalty,
        );
        for c in 0..k {
            info[j * k + c] += (gp[c] - g0[c]) / hj;
        }
    }
    // observed information = -(A + B), symmetrized
    let mut sym = vec![0.0_f64; k * k];
    for r in 0..k {
        for c in 0..k {
            sym[r * k + c] = -0.5 * (info[r * k + c] + info[c * k + r]);
        }
    }
    let inv = invert_information(sym.clone(), k)?;
    let se: Vec<f64> = (0..k).map(|j| inv[j * k + j].max(0.0).sqrt()).collect();
    Ok(OakesResult {
        labels: pv.labels(),
        se,
        information: sym,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/oakes_tests.rs"]
mod tests;
