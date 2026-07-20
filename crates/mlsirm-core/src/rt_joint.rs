//! Joint speed-accuracy hierarchical model using the person-level covariance
//! structure of van der Linden (2007): a bivariate-normal distribution that ties
//! ability `theta` (from an accuracy 2PL model) to speed `tau` (from the lognormal
//! response-time model),
//!
//! ```text
//! (theta_j, tau_j) ~ Normal2( 0,  [[1, rho*sigma_tau], [rho*sigma_tau, sigma_tau^2]] )
//! ```
//!
//! with the accuracy responses and the log response times conditionally
//! independent given `(theta, tau)`. The headline quantity is `rho`, the
//! ability-speed correlation.
//!
//! The original article illustrates the framework with a normal-ogive response
//! model and Bayesian MCMC. This crate instead provides a repository-specific
//! *two-stage* (limited-information) marginal-ML adaptation: the item parameters
//! of both measurement models are held fixed after separate calibration, the
//! accuracy side is logistic, and only `(rho, sigma_tau)` is estimated over a 2-D
//! Gauss-Hermite grid. This estimator and its closed-form covariance M-step must
//! therefore not be attributed to the original article.
//!
//! Note the `rho` estimated here is *not* the attenuated correlation of the two
//! separately-scored EAPs — those are biased toward zero by EAP shrinkage — but
//! the consistent marginal-ML person-covariance.
//!
//! # References (APA 7th ed.)
//!
//! van der Linden, W. J. (2007). A hierarchical framework for modeling speed and
//!   accuracy on test items. *Psychometrika, 72*(3), 287–308.
//!   https://doi.org/10.1007/s11336-006-1478-z

use crate::quadrature::gh_rule;

#[inline]
fn log_sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        -(-x).exp().ln_1p()
    } else {
        x - x.exp().ln_1p()
    }
}

#[inline]
fn covariance_q(c: f64, s11: f64, s12: f64, s22: f64, sigma_tau2: f64) -> f64 {
    let det = sigma_tau2 - c * c;
    if !(det.is_finite() && det > 0.0) {
        return f64::NEG_INFINITY;
    }
    -0.5 * (det.ln() + (sigma_tau2 * s11 - 2.0 * c * s12 + s22) / det)
}

fn cubic_real_roots(qa: f64, qb: f64, qc: f64) -> Vec<f64> {
    let p = qb - qa * qa / 3.0;
    let q = 2.0 * qa * qa * qa / 27.0 - qa * qb / 3.0 + qc;
    let discriminant = (q * 0.5).powi(2) + (p / 3.0).powi(3);
    let shift = -qa / 3.0;
    let scale = (q * q).abs() + (p * p * p).abs() + 1.0;
    let disc_tol = 64.0 * f64::EPSILON * scale;

    if discriminant > disc_tol {
        vec![
            (-0.5 * q + discriminant.sqrt()).cbrt()
                + (-0.5 * q - discriminant.sqrt()).cbrt()
                + shift,
        ]
    } else if discriminant >= -disc_tol {
        let u = (-0.5 * q).cbrt();
        vec![2.0 * u + shift, -u + shift]
    } else {
        let radius = 2.0 * (-p / 3.0).sqrt();
        let cos_arg = (-0.5 * q / (-(p / 3.0).powi(3)).sqrt()).clamp(-1.0, 1.0);
        let phi = cos_arg.acos();
        (0..3)
            .map(|k| radius * ((phi + 2.0 * std::f64::consts::PI * k as f64) / 3.0).cos() + shift)
            .collect()
    }
}

/// Maximize the covariance part of the expected complete-data log-likelihood
/// when `Var(theta) = 1` and `Var(tau) = sigma_tau2` are fixed. The score
/// equation is cubic in `c = Cov(theta, tau)`:
///
/// `c^3 - s12*c^2 + (sigma_tau2*(s11 - 1) + s22)*c - s12*sigma_tau2 = 0`.
///
/// Evaluate every real stationary point plus the positive-definiteness bounds
/// so the fixed-variance branch remains a genuine EM M-step.
fn maximize_fixed_variance_covariance(
    s11: f64,
    s12: f64,
    s22: f64,
    sigma_tau2: f64,
    rho_limit: f64,
) -> f64 {
    let bound = rho_limit * sigma_tau2.sqrt();
    let qa = -s12;
    let qb = sigma_tau2 * (s11 - 1.0) + s22;
    let qc = -s12 * sigma_tau2;
    let mut candidates = vec![-bound, bound, 0.0];
    candidates.extend(cubic_real_roots(qa, qb, qc));

    let mut best_c = 0.0;
    let mut best_q = covariance_q(best_c, s11, s12, s22, sigma_tau2);
    for candidate in candidates {
        if candidate.is_finite() && candidate >= -bound && candidate <= bound {
            let value = covariance_q(candidate, s11, s12, s22, sigma_tau2);
            if value > best_q {
                best_q = value;
                best_c = candidate;
            }
        }
    }
    best_c
}

fn joint_summary_is_finite(final_ll: f64, acc11: f64, theta_eap: &[f64], tau_eap: &[f64]) -> bool {
    final_ll.is_finite()
        && acc11.is_finite()
        && theta_eap
            .iter()
            .chain(tau_eap)
            .all(|value| value.is_finite())
}

fn ensure_joint_summary_is_finite(
    final_ll: f64,
    acc11: f64,
    theta_eap: &[f64],
    tau_eap: &[f64],
) -> Result<(), String> {
    if joint_summary_is_finite(final_ll, acc11, theta_eap, tau_eap) {
        Ok(())
    } else {
        Err("joint speed-accuracy final likelihood or EAPs became non-finite".into())
    }
}

/// Controls for [`fit_speed_accuracy_covariance`].
#[derive(Clone, Copy, Debug)]
pub struct SpeedAccuracyConfig {
    /// Gauss-Hermite nodes per dimension (in `{7, 11, 15, 21, 31, 41}`).
    pub q: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// `|rho|` clamp (positive-definiteness guard on `Sigma_P`).
    pub rho_floor: f64,
    pub sigma_floor: f64,
    /// `Some(s)` holds `sigma_tau = s` fixed (e.g. at the stage-1 value), leaving
    /// only `rho` free.
    pub fix_sigma_tau: Option<f64>,
}

impl Default for SpeedAccuracyConfig {
    fn default() -> Self {
        Self {
            q: 21,
            max_iter: 500,
            tol: 1e-6,
            rho_floor: 0.999,
            sigma_floor: 1e-4,
            fix_sigma_tau: None,
        }
    }
}

/// Result of [`fit_speed_accuracy_covariance`].
#[derive(Clone, Debug)]
pub struct SpeedAccuracyFit {
    /// Ability-speed correlation (the headline output).
    pub rho: f64,
    pub sigma_tau: f64,
    /// Posterior second moment of `theta` (`S11`); a diagnostic — `~1` when the
    /// accuracy and RT calibrations share a metric. Reported, never re-estimated.
    pub s_theta2: f64,
    pub loglik: f64,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// Joint-posterior EAP ability / speed (borrow strength through `rho`).
    pub theta_eap: Vec<f64>,
    pub tau_eap: Vec<f64>,
}

/// Estimate a two-stage marginal-ML adaptation of the van der Linden (2007)
/// person covariance
/// `Sigma_P = [[1, rho*sigma_tau], [rho*sigma_tau, sigma_tau^2]]` by two-stage
/// marginal ML, holding the item parameters fixed. `responses` (0/1) and `times`
/// (`> 0` where observed) are `n_persons * n_items` row-major; `observed` masks
/// both (`None` = fully observed). `a`/`b` are the accuracy 2PL raw slope /
/// intercept (`eta = a_i*theta + b_i`); `alpha`/`beta` are the lognormal time
/// discrimination / intensity. At least one paired observation and one observed
/// item with non-zero accuracy discrimination are required to identify `rho`.
#[allow(clippy::too_many_arguments)]
pub fn fit_speed_accuracy_covariance(
    responses: &[f64],
    times: &[f64],
    observed: Option<&[bool]>,
    a: &[f64],
    b: &[f64],
    alpha: &[f64],
    beta: &[f64],
    n_persons: usize,
    n_items: usize,
    config: SpeedAccuracyConfig,
) -> Result<SpeedAccuracyFit, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be positive".into());
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows".to_string())?;
    if responses.len() != expected || times.len() != expected {
        return Err("responses and times must have length n_persons * n_items".into());
    }
    if a.len() != n_items || b.len() != n_items || alpha.len() != n_items || beta.len() != n_items {
        return Err("item-parameter vectors must have length n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != expected {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if config.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !(config.tol.is_finite() && config.tol > 0.0) {
        return Err("tol must be positive and finite".into());
    }
    if !(config.rho_floor.is_finite() && config.rho_floor > 0.0 && config.rho_floor < 1.0) {
        return Err("rho_floor must be finite and strictly between 0 and 1".into());
    }
    if !(config.sigma_floor.is_finite() && config.sigma_floor > 0.0) {
        return Err("sigma_floor must be positive and finite".into());
    }
    if let Some(s) = config.fix_sigma_tau {
        if !(s.is_finite() && s > 0.0 && (s * s).is_finite()) {
            return Err("fix_sigma_tau must be positive and finite".into());
        }
    }
    if a.iter().chain(b).chain(beta).any(|x| !x.is_finite()) {
        return Err("a, b, and beta must contain only finite values".into());
    }
    if alpha
        .iter()
        .any(|x| !x.is_finite() || *x <= 0.0 || !(*x * *x).is_finite())
    {
        return Err(
            "alpha must be positive with finite squares; otherwise the joint likelihood is non-finite".into(),
        );
    }
    let (nodes, weights) =
        gh_rule(config.q).ok_or_else(|| format!("unsupported q {}", config.q))?;
    let q = nodes.len();
    let lnw: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let ln2pi = (2.0 * std::f64::consts::PI).ln();

    // precompute per-person accuracy log-lik at each theta node (theta = z_a is
    // independent of Sigma_P, so this is one-time) and the RT sufficient stats.
    let mut la = vec![0.0_f64; n_persons * q];
    let mut aj = vec![0.0_f64; n_persons];
    let mut bj = vec![0.0_f64; n_persons];
    let mut cj = vec![0.0_f64; n_persons];
    let mut kj = vec![0.0_f64; n_persons];
    let mut n_observed = 0usize;
    let mut n_accuracy_informative = 0usize;
    for p in 0..n_persons {
        for i in 0..n_items {
            if !is_obs(p, i) {
                continue;
            }
            n_observed += 1;
            if a[i] != 0.0 {
                n_accuracy_informative += 1;
            }
            let u = responses[p * n_items + i];
            if u != 0.0 && u != 1.0 {
                return Err("responses must be 0 or 1 where observed".into());
            }
            for (ai, &z) in nodes.iter().enumerate() {
                let eta = a[i] * z + b[i];
                la[p * q + ai] += if u > 0.5 {
                    log_sigmoid(eta)
                } else {
                    log_sigmoid(-eta)
                };
            }
            let t = times[p * n_items + i];
            if !t.is_finite() || t <= 0.0 {
                return Err("response times must be finite and positive where observed".into());
            }
            let y = t.ln();
            let a2 = alpha[i] * alpha[i];
            let d = y - beta[i];
            aj[p] += a2;
            bj[p] += a2 * d;
            cj[p] += a2 * d * d;
            kj[p] += alpha[i].ln() - 0.5 * ln2pi;
        }
    }
    if n_observed == 0 {
        return Err("at least one response-time pair must be observed".into());
    }
    if n_accuracy_informative == 0 {
        return Err(
            "at least one observed response must have non-zero accuracy discrimination".into(),
        );
    }

    let mut sigma_tau2 = match config.fix_sigma_tau {
        Some(s) => s * s,
        None => 0.09, // 0.3^2 warm start
    };
    let mut c = 0.0_f64; // covariance rho*sigma_tau; warm start rho = 0
    let pd_eps = 1e-12_f64;
    let mut trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut lj = vec![0.0_f64; q * q];

    for it in 0..config.max_iter {
        n_iter = it + 1;
        let l22 = (sigma_tau2 - c * c).max(pd_eps).sqrt();
        let (mut acc11, mut acc12, mut acc22) = (0.0_f64, 0.0, 0.0);
        let mut loglik = 0.0_f64;
        for p in 0..n_persons {
            // grid log-joint, tracking the max for a stable logsumexp
            let mut mx = f64::NEG_INFINITY;
            for ai in 0..q {
                let base = lnw[ai] + la[p * q + ai];
                let za = nodes[ai];
                for bi in 0..q {
                    let tau = c * za + l22 * nodes[bi];
                    let lt = kj[p] - 0.5 * (aj[p] * tau * tau + 2.0 * bj[p] * tau + cj[p]);
                    let val = base + lnw[bi] + lt;
                    lj[ai * q + bi] = val;
                    if val > mx {
                        mx = val;
                    }
                }
            }
            let mut denom = 0.0_f64;
            for &val in lj.iter() {
                denom += (val - mx).exp();
            }
            let logl_j = mx + denom.ln();
            loglik += logl_j;
            for ai in 0..q {
                let za = nodes[ai];
                for bi in 0..q {
                    let tau = c * za + l22 * nodes[bi];
                    let w = (lj[ai * q + bi] - logl_j).exp();
                    acc11 += w * za * za;
                    acc12 += w * za * tau;
                    acc22 += w * tau * tau;
                }
            }
        }
        if !loglik.is_finite() || !acc11.is_finite() || !acc12.is_finite() || !acc22.is_finite() {
            return Err(
                "joint speed-accuracy likelihood or posterior moments became non-finite".into(),
            );
        }
        trace.push(loglik);
        if it > 0 && (trace[it] - trace[it - 1]).abs() < config.tol {
            converged = true;
            break;
        }
        // M-step (exact constrained maximizer, sigma_theta^2 == 1)
        let s11 = acc11 / n_persons as f64;
        let s12 = acc12 / n_persons as f64;
        let s22 = acc22 / n_persons as f64;
        if let Some(s) = config.fix_sigma_tau {
            sigma_tau2 = s * s;
            c = maximize_fixed_variance_covariance(s11, s12, s22, sigma_tau2, config.rho_floor);
        } else {
            let c_new = s12 / s11;
            let v_new = (s22 - s12 * s12 * (s11 - 1.0) / (s11 * s11)).max(config.sigma_floor);
            sigma_tau2 = v_new;
            let sig = v_new.sqrt();
            let rho = (c_new / sig).clamp(-config.rho_floor, config.rho_floor);
            c = rho * sig;
        }
        // re-clamp covariance for positive definiteness under fixed sigma_tau too
        let sig = sigma_tau2.sqrt();
        let rho = (c / sig).clamp(-config.rho_floor, config.rho_floor);
        c = rho * sig;
    }

    // final pass: EAPs + loglik at converged Sigma_P
    let l22 = (sigma_tau2 - c * c).max(pd_eps).sqrt();
    let mut theta_eap = vec![0.0_f64; n_persons];
    let mut tau_eap = vec![0.0_f64; n_persons];
    let mut final_ll = 0.0_f64;
    let mut acc11 = 0.0_f64;
    for p in 0..n_persons {
        let mut mx = f64::NEG_INFINITY;
        for ai in 0..q {
            let base = lnw[ai] + la[p * q + ai];
            let za = nodes[ai];
            for bi in 0..q {
                let tau = c * za + l22 * nodes[bi];
                let lt = kj[p] - 0.5 * (aj[p] * tau * tau + 2.0 * bj[p] * tau + cj[p]);
                let val = base + lnw[bi] + lt;
                lj[ai * q + bi] = val;
                if val > mx {
                    mx = val;
                }
            }
        }
        let mut denom = 0.0_f64;
        for &val in lj.iter() {
            denom += (val - mx).exp();
        }
        let logl_j = mx + denom.ln();
        final_ll += logl_j;
        let (mut te, mut ts) = (0.0_f64, 0.0_f64);
        for ai in 0..q {
            let za = nodes[ai];
            for bi in 0..q {
                let tau = c * za + l22 * nodes[bi];
                let w = (lj[ai * q + bi] - logl_j).exp();
                te += w * za;
                ts += w * tau;
                acc11 += w * za * za;
            }
        }
        theta_eap[p] = te;
        tau_eap[p] = ts;
    }
    ensure_joint_summary_is_finite(final_ll, acc11, &theta_eap, &tau_eap)?;
    if trace
        .last()
        .is_none_or(|last| last.to_bits() != final_ll.to_bits())
    {
        trace.push(final_ll);
    }
    let final_loglik_change = trace
        .windows(2)
        .last()
        .map_or(f64::INFINITY, |pair| (pair[1] - pair[0]).abs());
    let termination_reason = if converged {
        "converged"
    } else {
        "max_iter_reached"
    };
    let sigma_tau = sigma_tau2.sqrt();
    let rho = c / sigma_tau;
    Ok(SpeedAccuracyFit {
        rho,
        sigma_tau,
        s_theta2: acc11 / n_persons as f64,
        loglik: final_ll,
        loglik_trace: trace,
        n_iter,
        converged,
        termination_reason: termination_reason.to_string(),
        final_loglik_change,
        theta_eap,
        tau_eap,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/rt_joint_tests.rs"]
mod tests;
