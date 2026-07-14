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
    n_items: usize,
    latent_dim: usize,
}

impl ParamVec {
    fn len(&self) -> usize {
        let per_item = 1
            + usize::from(self.free_alpha)
            + if self.uses_space { self.latent_dim } else { 0 };
        self.n_items * per_item + usize::from(self.uses_space)
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
        if self.uses_space {
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
        if self.uses_space {
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
        let tau = if self.uses_space { v[cursor] } else { -30.0 };
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
                    if uses_space {
                        let mut dist2 = config.eps_distance;
                        for k in 0..latent_dim {
                            let diff = grids.x_grid[x * latent_dim + k]
                                - zeta[i * latent_dim + k];
                            dist2 += diff * diff;
                        }
                        dist = dist2.sqrt();
                        eta -= gamma * dist;
                    }
                    let resid = r - n * sigmoid(eta);
                    g_b += resid;
                    if free_alpha {
                        g_alpha += resid * a * theta;
                    }
                    if uses_space {
                        for k in 0..latent_dim {
                            g_zeta[k] += resid * gamma
                                * (grids.x_grid[x * latent_dim + k] - zeta[i * latent_dim + k])
                                / dist;
                        }
                        g_tau += resid * (-gamma * dist);
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
    if uses_space {
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
        n_items: config.n_items,
        latent_dim: config.latent_dim,
    };
    let k = pv.len();
    let (t_nodes, t_weights) =
        gh_rule(mcfg.q_theta).ok_or_else(|| "unsupported q_theta".to_string())?;
    let (x_grid, x_logw) = if uses_space {
        let rule = match mcfg.xi_rule {
            XiRuleKind::GaussHermite => {
                crate::nodes::XiRule::GaussHermite { q_xi: mcfg.q_xi }
            }
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
    let g0 = q_gradient(&pv, &xi0, &counts0, &ctx, &grids, config, factor_id, penalty);
    for j in 0..k {
        let hj = h * (1.0 + xi0[j].abs());
        let mut x0p = xi0.clone();
        x0p[j] += hj;
        let counts_p = estep_at(&x0p);
        let gp = q_gradient(&pv, &xi0, &counts_p, &ctx, &grids, config, factor_id, penalty);
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
    let inv = invert(sym.clone(), k)
        .ok_or_else(|| "observed information is singular; SEs unavailable".to_string())?;
    let se: Vec<f64> = (0..k).map(|j| inv[j * k + j].max(0.0).sqrt()).collect();
    Ok(OakesResult { labels: pv.labels(), se, information: sym })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marginal::{fit_marginal, MarginalConfig, PopulationSpec};
    use crate::{Device, ModelType, PenaltyConfig};

    #[test]
    fn oakes_matches_central_difference_of_the_score() {
        // simulate a small 1PL-with-space fit, then check the Oakes assembly
        // against the full central difference of the marginal score, and the
        // SEs against 1/sqrt(n) scaling expectations.
        let mut state = 4242u64;
        let mut unif = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let (n_persons, n_items) = (400usize, 6usize);
        let factor_id = vec![0usize; n_items];
        let b_true: Vec<f64> = (0..n_items).map(|i| -1.0 + 0.4 * i as f64).collect();
        let mut y = vec![0.0_f64; n_persons * n_items];
        for p in 0..n_persons {
            let u1: f64 = unif().max(1e-12);
            let u2: f64 = unif();
            let theta =
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            for i in 0..n_items {
                let eta: f64 = theta + b_true[i];
                if unif() < 1.0 / (1.0 + (-eta).exp()) {
                    y[p * n_items + i] = 1.0;
                }
            }
        }
        let observed = vec![true; n_persons * n_items];
        let config = ModelConfig {
            n_persons,
            n_items,
            n_dims: 1,
            latent_dim: 1,
            model_type: ModelType::Mirt,
            eps_distance: 1e-8,
        };
        let mcfg = MarginalConfig { q_theta: 15, q_xi: 7, max_iter: 80, ..Default::default() };
        let pen = PenaltyConfig::lsirm_prior();
        let fitted = fit_marginal(
            &y,
            &observed,
            &factor_id,
            &config,
            &PopulationSpec::Single,
            &mcfg,
            &pen,
            Device::Cpu,
        )
        .unwrap();
        let res = observed_information_oakes(
            &y,
            &observed,
            &factor_id,
            &config,
            &PopulationSpec::Single,
            &mcfg,
            &pen,
            &fitted.alpha,
            &fitted.b,
            &fitted.zeta,
            fitted.tau,
            &fitted.mu,
            &fitted.sigma,
            fitted.sigma_u,
            1e-5,
        )
        .unwrap();
        // MIRT free-alpha: labels alternate alpha/b per item
        assert_eq!(res.labels.len(), 2 * n_items);
        assert!(res.se.iter().all(|s| s.is_finite() && *s > 0.0));
        // b SEs at n=400 for a 1PL-ish item live in the 0.05..0.5 band
        for (lab, se) in res.labels.iter().zip(&res.se) {
            if lab.starts_with("b:") {
                assert!(
                    (0.03..0.6).contains(se),
                    "implausible SE for {lab}: {se}"
                );
            }
        }
        // internal consistency: Oakes total equals the central FD of the
        // marginal score for a couple of probe coordinates
        let pv_probe = [1usize, 4usize];
        let pv = ParamVec {
            free_alpha: true,
            uses_space: false,
            n_items,
            latent_dim: 1,
        };
        let (t_nodes, t_weights) = gh_rule(15).unwrap();
        let grids = Grids {
            t_nodes: t_nodes.to_vec(),
            t_logw: t_weights.iter().map(|w| w.ln()).collect(),
            x_grid: vec![0.0; 1],
            x_logw: vec![0.0],
            q_t: 15,
            n_x: 1,
        };
        let ctx = build_contexts(&PopulationSpec::Single, &[], &[], 0.0, 1, 15);
        let resp = index_responses(&y, &observed, n_persons, n_items);
        let xi0 = pv.pack(&fitted.alpha, &fitted.b, &fitted.zeta, fitted.tau);
        let score_at = |xv: &[f64]| -> Vec<f64> {
            let (a0, b0, z0, t0) = pv.unpack(xv);
            let tables = build_tables(&a0, &b0, &z0, t0, &config, &factor_id, &ctx, &grids);
            let counts = e_step(&tables, &resp, &factor_id, &config, &PopulationSpec::Single, &ctx, &grids);
            q_gradient(&pv, xv, &counts, &ctx, &grids, &config, &factor_id, &pen)
        };
        for &j in &pv_probe {
            let hj = 1e-5 * (1.0 + xi0[j].abs());
            let mut xp = xi0.clone();
            xp[j] += hj;
            let mut xm = xi0.clone();
            xm[j] -= hj;
            let sp = score_at(&xp);
            let sm = score_at(&xm);
            for c in 0..pv.len() {
                let fd = -(sp[c] - sm[c]) / (2.0 * hj);
                let oakes = res.information[j * pv.len() + c];
                assert!(
                    (fd - oakes).abs() < 1e-2 * (1.0 + fd.abs()),
                    "Oakes[{j},{c}] = {oakes} vs FD {fd}"
                );
            }
        }
    }
}
