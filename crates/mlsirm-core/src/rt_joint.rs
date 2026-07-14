//! Joint speed-accuracy hierarchical model (van der Linden, 2007, Level 2): a
//! person-level bivariate-normal distribution that ties ability `theta` (from an
//! accuracy 2PL model) to speed `tau` (from the lognormal response-time model),
//!
//! ```text
//! (theta_j, tau_j) ~ Normal2( 0,  [[1, rho*sigma_tau], [rho*sigma_tau, sigma_tau^2]] )
//! ```
//!
//! with the accuracy responses and the log response times conditionally
//! independent given `(theta, tau)`. The headline quantity is `rho`, the
//! ability-speed correlation.
//!
//! This is the *two-stage* (limited-information) estimator: the item parameters
//! of both measurement models are held fixed (from their separate calibrations)
//! and only the person covariance `(rho, sigma_tau)` is estimated, by marginal ML
//! over a 2-D Gauss-Hermite grid. Unlike the pure response-time model, the
//! accuracy side is logistic, so the joint marginal likelihood is not closed form
//! and requires quadrature.
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
        Self { q: 21, max_iter: 500, tol: 1e-6, rho_floor: 0.999, sigma_floor: 1e-4, fix_sigma_tau: None }
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
    /// Joint-posterior EAP ability / speed (borrow strength through `rho`).
    pub theta_eap: Vec<f64>,
    pub tau_eap: Vec<f64>,
}

/// Estimate the van der Linden (2007) Level-2 person covariance
/// `Sigma_P = [[1, rho*sigma_tau], [rho*sigma_tau, sigma_tau^2]]` by two-stage
/// marginal ML, holding the item parameters fixed. `responses` (0/1) and `times`
/// (`> 0` where observed) are `n_persons * n_items` row-major; `observed` masks
/// both (`None` = fully observed). `a`/`b` are the accuracy 2PL raw slope /
/// intercept (`eta = a_i*theta + b_i`); `alpha`/`beta` are the lognormal time
/// discrimination / intensity.
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
    if responses.len() != n_persons * n_items || times.len() != n_persons * n_items {
        return Err("responses and times must have length n_persons * n_items".into());
    }
    if a.len() != n_items || b.len() != n_items || alpha.len() != n_items || beta.len() != n_items {
        return Err("item-parameter vectors must have length n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_persons * n_items {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if let Some(s) = config.fix_sigma_tau {
        if !(s.is_finite() && s > 0.0) {
            return Err("fix_sigma_tau must be positive and finite".into());
        }
    }
    let (nodes, weights) = gh_rule(config.q).ok_or_else(|| format!("unsupported q {}", config.q))?;
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
    for p in 0..n_persons {
        for i in 0..n_items {
            if !is_obs(p, i) {
                continue;
            }
            let u = responses[p * n_items + i];
            if u != 0.0 && u != 1.0 {
                return Err("responses must be 0 or 1 where observed".into());
            }
            for (ai, &z) in nodes.iter().enumerate() {
                let eta = a[i] * z + b[i];
                la[p * q + ai] += if u > 0.5 { log_sigmoid(eta) } else { log_sigmoid(-eta) };
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
        trace.push(loglik);
        // M-step (exact constrained maximizer, sigma_theta^2 == 1)
        let s11 = acc11 / n_persons as f64;
        let s12 = acc12 / n_persons as f64;
        let s22 = acc22 / n_persons as f64;
        let c_new = s12 / s11;
        if let Some(s) = config.fix_sigma_tau {
            sigma_tau2 = s * s;
            c = c_new; // covariance; rho = c/s
        } else {
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

        if it > 0 && (trace[it] - trace[it - 1]).abs() < config.tol {
            converged = true;
            break;
        }
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
    trace.push(final_ll);
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
        theta_eap,
        tau_eap,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f64 {
        let mut st = seed.max(1);
        move || {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((st >> 11) as f64) / ((1u64 << 53) as f64)
        }
    }
    fn normal(u: &mut impl FnMut() -> f64) -> f64 {
        let u1 = u().max(1e-12);
        let u2 = u();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let mut sab = 0.0;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        for (&xi, &yi) in x.iter().zip(y) {
            sab += (xi - mx) * (yi - my);
            saa += (xi - mx).powi(2);
            sbb += (yi - my).powi(2);
        }
        sab / (saa.sqrt() * sbb.sqrt())
    }

    // Anchor A: at rho=0 the 2-D grid log-likelihood factorizes into the sum of the
    // two 1-D grid log-likelihoods (certifies the Cholesky map, tensor weights, and
    // logsumexp wiring exactly).
    #[test]
    fn joint_rho0_factorizes() {
        let (nodes, weights) = gh_rule(21).unwrap();
        let q = nodes.len();
        let lnw: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
        // one 3-item person: accuracy la[a], and RT stats
        let a = [1.0_f64, 1.3, 0.8];
        let b = [0.2_f64, -0.4, 0.1];
        let alpha = [1.5_f64, 2.0, 1.1];
        let beta = [4.0_f64, 3.6, 4.2];
        let u = [1.0_f64, 0.0, 1.0];
        let y = [3.8_f64, 3.9, 4.5];
        let sig = 0.35_f64;
        let mut la = vec![0.0_f64; q];
        let (mut aj, mut bj, mut cj, mut kj) = (0.0, 0.0, 0.0, 0.0);
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        for i in 0..3 {
            for (ai, &z) in nodes.iter().enumerate() {
                let eta = a[i] * z + b[i];
                la[ai] += if u[i] > 0.5 { log_sigmoid(eta) } else { log_sigmoid(-eta) };
            }
            let a2 = alpha[i] * alpha[i];
            let d = y[i] - beta[i];
            aj += a2;
            bj += a2 * d;
            cj += a2 * d * d;
            kj += alpha[i].ln() - 0.5 * ln2pi;
        }
        // 2-D logsumexp at rho=0 (c=0, l22=sigma_tau)
        let mut mx = f64::NEG_INFINITY;
        let mut grid = vec![0.0_f64; q * q];
        for ai in 0..q {
            for bi in 0..q {
                let tau = sig * nodes[bi];
                let lt = kj - 0.5 * (aj * tau * tau + 2.0 * bj * tau + cj);
                let v = lnw[ai] + la[ai] + lnw[bi] + lt;
                grid[ai * q + bi] = v;
                if v > mx {
                    mx = v;
                }
            }
        }
        let joint = mx + grid.iter().map(|&v| (v - mx).exp()).sum::<f64>().ln();
        // two 1-D logsumexps
        let mxa = (0..q).map(|ai| lnw[ai] + la[ai]).fold(f64::NEG_INFINITY, f64::max);
        let la1 = mxa + (0..q).map(|ai| (lnw[ai] + la[ai] - mxa).exp()).sum::<f64>().ln();
        let ltv: Vec<f64> = (0..q)
            .map(|bi| {
                let tau = sig * nodes[bi];
                lnw[bi] + kj - 0.5 * (aj * tau * tau + 2.0 * bj * tau + cj)
            })
            .collect();
        let mxb = ltv.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let lt1 = mxb + ltv.iter().map(|&v| (v - mxb).exp()).sum::<f64>().ln();
        assert!((joint - (la1 + lt1)).abs() < 1e-10, "rho=0 factorization: {joint} vs {}", la1 + lt1);
    }

    // Anchor B/D + recovery: simulate under a known Sigma_P and recover (rho,
    // sigma_tau) with the item banks frozen.
    fn sim_and_fit(seed: u64, n: usize, rho_true: f64, sig_true: f64) -> SpeedAccuracyFit {
        let ni = 20usize;
        let a: Vec<f64> = (0..ni).map(|i| 0.9 + 0.6 * (i % 3) as f64 / 2.0).collect();
        let b: Vec<f64> = (0..ni).map(|i| -1.5 + 3.0 * i as f64 / (ni - 1) as f64).collect();
        let alpha: Vec<f64> = (0..ni).map(|i| 1.0 + 2.0 * i as f64 / (ni - 1) as f64).collect();
        let beta: Vec<f64> = (0..ni).map(|i| 3.5 + 1.0 * i as f64 / (ni - 1) as f64).collect();
        let mut u = lcg(seed);
        let mut resp = vec![0.0_f64; n * ni];
        let mut times = vec![0.0_f64; n * ni];
        let l22 = sig_true * (1.0 - rho_true * rho_true).sqrt();
        for p in 0..n {
            let za = normal(&mut u);
            let zb = normal(&mut u);
            let theta = za;
            let tau = rho_true * sig_true * za + l22 * zb;
            for i in 0..ni {
                let pr = 1.0 / (1.0 + (-(a[i] * theta + b[i])).exp());
                resp[p * ni + i] = if u() < pr { 1.0 } else { 0.0 };
                let ylog = beta[i] - tau + (1.0 / alpha[i]) * normal(&mut u);
                times[p * ni + i] = ylog.exp();
            }
        }
        fit_speed_accuracy_covariance(
            &resp, &times, None, &a, &b, &alpha, &beta, n, ni, SpeedAccuracyConfig::default(),
        )
        .unwrap()
    }

    #[test]
    fn joint_recovers_rho_and_reduces_at_zero() {
        // Anchor D: recovery at rho=0.5
        let fit = sim_and_fit(11, 1000, 0.5, 0.3);
        assert!(fit.converged);
        let max_drop = fit.loglik_trace.windows(2).map(|w| w[0] - w[1]).fold(f64::NEG_INFINITY, f64::max);
        eprintln!("[joint] trace len={} first={:.4} last={:.4} max_drop={:.3e}", fit.loglik_trace.len(), fit.loglik_trace[0], fit.loglik_trace.last().unwrap(), max_drop);
        assert!(
            fit.loglik_trace.windows(2).all(|w| w[1] >= w[0] - 1e-6 * w[0].abs().max(1.0)),
            "loglik must be monotone (max drop {max_drop:.3e})"
        );
        assert!((fit.rho - 0.5).abs() < 0.1, "rho {}", fit.rho);
        assert!((fit.sigma_tau - 0.3).abs() < 0.05, "sigma_tau {}", fit.sigma_tau);
        // Anchor B: true independence -> rho ~= 0
        let fit0 = sim_and_fit(12, 1000, 0.0, 0.3);
        assert!(fit0.rho.abs() < 0.08, "rho at independence should be ~0: {}", fit0.rho);
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn joint_monte_carlo_500() {
        let reps = 500usize;
        for &rho_true in &[0.0_f64, 0.5, -0.5] {
            let (mut sr, mut br, mut ss, mut bs, mut absr) = (0.0, 0.0, 0.0, 0.0, 0.0);
            for r in 0..reps {
                let fit = sim_and_fit(200 + r as u64, 800, rho_true, 0.3);
                sr += (fit.rho - rho_true).powi(2);
                br += fit.rho - rho_true;
                ss += (fit.sigma_tau - 0.3).powi(2);
                bs += fit.sigma_tau - 0.3;
                absr += fit.rho.abs();
            }
            let f = reps as f64;
            println!(
                "[joint 500] rho={rho_true}: RMSE(rho)={:.4} bias(rho)={:.4} RMSE(sigma)={:.4} \
                 bias(sigma)={:.4} mean|rho|={:.4}",
                (sr / f).sqrt(), br / f, (ss / f).sqrt(), bs / f, absr / f
            );
            // provisional thresholds (retune after the first 500-rep run; with ~20
            // items the person-parameter measurement error inflates SD(rho_hat))
            assert!((sr / f).sqrt() < 0.06, "rho RMSE too high: {}", (sr / f).sqrt());
            assert!((br / f).abs() < 0.02, "rho bias too high: {}", br / f);
            assert!((bs / f).abs() < 0.05, "sigma_tau bias too high: {}", bs / f);
            if rho_true == 0.0 {
                // mean|rho_hat| ~ RMSE*sqrt(2/pi) ~ 0.033 for an unbiased estimator
                // (a dispersion sanity, not a bias check; bias(rho) above is the
                // real "recovers independence" anchor)
                assert!(absr / f < 0.05, "mean|rho| at rho=0: {}", absr / f);
            }
        }
    }
}
