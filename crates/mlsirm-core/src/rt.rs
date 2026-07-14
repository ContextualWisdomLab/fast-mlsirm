//! Lognormal response-time (RT) measurement model (van der Linden, 2007): the
//! speed-side analogue of the 2PL for item response *times*. For person `j` with
//! latent speed `tau_j` and item `i` with time intensity `beta_i` and time
//! discrimination `alpha_i > 0`,
//!
//! ```text
//! ln(T_ij) | tau_j  ~  Normal( beta_i - tau_j,  1 / alpha_i^2 )
//! ```
//!
//! i.e. the log response time is normal with mean `beta_i - tau_j` and standard
//! deviation `1/alpha_i` (higher speed => shorter time; higher `alpha` => sharper
//! timing). Item parameters and the speed distribution are estimated by marginal
//! maximum likelihood with `tau_j ~ Normal(mu_tau, sigma_tau^2)` marginalized out,
//! and speed is scored by EAP.
//!
//! Because the model is *conditionally Gaussian with a unit loading on `tau`*, the
//! speed posterior, the marginal likelihood, and the EAP are all available in
//! exact closed form (matrix-determinant / Sherman-Morrison), so the estimator
//! needs neither quadrature nor a line search — the EM is exact coordinate ascent.
//!
//! Identification: the log-time metric fixes the speed *scale* (`alpha_i`
//! multiplies the residual, not `beta_i - tau_j`, so there is no `alpha`↔`sigma_tau`
//! trade-off), leaving only the speed *location* free. The estimator pins the
//! population `mu_tau = 0` and estimates `sigma_tau` directly from the
//! between-person, same-person cross-item log-time covariance.
//!
//! # References (APA 7th ed.)
//!
//! van der Linden, W. J. (2007). A hierarchical framework for modeling speed and
//!   accuracy on test items. *Psychometrika, 72*(3), 287–308.
//!   https://doi.org/10.1007/s11336-006-1478-z

/// Estimation controls for [`fit_rt_lognormal`].
#[derive(Clone, Copy, Debug)]
pub struct RtConfig {
    pub max_iter: usize,
    pub tol: f64,
    /// Minimum residual variance `1/alpha_i^2` (bounds `alpha_i` away from `inf`).
    pub var_floor: f64,
    /// Minimum `sigma_tau^2`.
    pub sigma_floor: f64,
    /// `None` estimates `sigma_tau` (default, faithful identification with
    /// `mu_tau = 0`); `Some(s)` holds `sigma_tau = s` fixed (a genuine restriction —
    /// it forces every same-person inter-item log-time covariance to `s^2` — not a
    /// harmless normalization; use only for a deliberately standardized metric).
    pub fix_sigma_tau: Option<f64>,
}

impl Default for RtConfig {
    fn default() -> Self {
        Self { max_iter: 500, tol: 1e-6, var_floor: 1e-4, sigma_floor: 1e-4, fix_sigma_tau: None }
    }
}

/// Fitted lognormal RT model.
#[derive(Clone, Debug)]
pub struct RtFit {
    /// Time discriminations `alpha_i > 0` (length `n_items`).
    pub alpha: Vec<f64>,
    /// Time intensities `beta_i` (length `n_items`).
    pub beta: Vec<f64>,
    /// Pinned to 0 (the identification constraint).
    pub mu_tau: f64,
    /// Estimated speed SD.
    pub sigma_tau: f64,
    /// EAP speed `tau_hat_j` (length `n_persons`).
    pub tau_eap: Vec<f64>,
    /// Posterior SD of the speed EAP.
    pub tau_sd: Vec<f64>,
    pub loglik: f64,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
}

/// Fit the lognormal RT measurement model by marginal-ML EM (van der Linden,
/// 2007). `times` is `n_persons * n_items` row-major raw response times (`> 0`
/// where observed); `observed` is an optional missingness mask of the same length
/// (`None` = fully observed). Returns item `alpha`/`beta`, the estimated
/// `sigma_tau`, and per-person EAP speed.
pub fn fit_rt_lognormal(
    times: &[f64],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    config: RtConfig,
) -> Result<RtFit, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be positive".into());
    }
    if times.len() != n_persons * n_items {
        return Err("times must have length n_persons * n_items".into());
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
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    // log-times where observed
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) {
                let t = times[p * n_items + i];
                if !t.is_finite() || t <= 0.0 {
                    return Err("response times must be finite and positive where observed".into());
                }
                y[p * n_items + i] = t.ln();
            }
        }
    }
    let mut n_i = vec![0usize; n_items];
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) {
                n_i[i] += 1;
            }
        }
    }
    if n_i.iter().any(|&c| c == 0) {
        return Err("every item must be observed by at least one person".into());
    }

    // init: method-of-moments beta (E[y]=beta since mu_tau=0), flat alpha/sigma_tau
    let mut beta = vec![0.0_f64; n_items];
    for i in 0..n_items {
        let mut s = 0.0;
        for p in 0..n_persons {
            if is_obs(p, i) {
                s += y[p * n_items + i];
            }
        }
        beta[i] = s / n_i[i] as f64;
    }
    let mut alpha = vec![1.0_f64; n_items];
    let mut sigma_tau2 = match config.fix_sigma_tau {
        Some(s) => s * s,
        None => 0.09, // 0.3^2, a benign start
    };
    let mut tau_eap = vec![0.0_f64; n_persons];
    let mut tau_sd = vec![0.0_f64; n_persons];
    let mut v_all = vec![0.0_f64; n_persons];
    let mut s_all = vec![0.0_f64; n_persons];
    let mut trace: Vec<f64> = Vec::new();
    let ln2pi = (2.0 * std::f64::consts::PI).ln();
    let mut converged = false;
    let mut n_iter = 0usize;

    for it in 0..config.max_iter {
        n_iter = it + 1;
        let a: Vec<f64> = alpha.iter().map(|&al| al * al).collect();
        let lna: Vec<f64> = a.iter().map(|&ai| ai.ln()).collect();
        // E-step (exact Gaussian posterior) + marginal log-likelihood
        let mut loglik = 0.0_f64;
        for p in 0..n_persons {
            let (mut a_sum, mut num, mut ar2, mut ld, mut nj) = (0.0, 0.0, 0.0, 0.0, 0usize);
            for i in 0..n_items {
                if is_obs(p, i) {
                    let r = y[p * n_items + i] - beta[i];
                    a_sum += a[i];
                    num += a[i] * (-r);
                    ar2 += a[i] * r * r;
                    ld += lna[i];
                    nj += 1;
                }
            }
            let pj = 1.0 / sigma_tau2 + a_sum;
            let te = num / pj;
            let vj = 1.0 / pj;
            tau_eap[p] = te;
            v_all[p] = vj;
            s_all[p] = te * te + vj;
            loglik +=
                -0.5 * (nj as f64 * ln2pi - ld + sigma_tau2.ln() + pj.ln() + ar2 - pj * te * te);
        }
        trace.push(loglik);

        // M-step (closed form): beta, then alpha with fresh beta, then sigma_tau
        for i in 0..n_items {
            let mut s = 0.0;
            for p in 0..n_persons {
                if is_obs(p, i) {
                    s += y[p * n_items + i] + tau_eap[p];
                }
            }
            beta[i] = s / n_i[i] as f64;
        }
        for i in 0..n_items {
            let mut ss = 0.0;
            for p in 0..n_persons {
                if is_obs(p, i) {
                    let e = y[p * n_items + i] - beta[i] + tau_eap[p];
                    // + v_all[p] is the EM posterior-variance correction; dropping
                    // it biases alpha high
                    ss += e * e + v_all[p];
                }
            }
            let resvar = (ss / n_i[i] as f64).max(config.var_floor);
            alpha[i] = 1.0 / resvar.sqrt();
        }
        if config.fix_sigma_tau.is_none() {
            let mean_s: f64 = s_all.iter().sum::<f64>() / n_persons as f64;
            sigma_tau2 = mean_s.max(config.sigma_floor);
        }

        if it > 0 && (trace[it] - trace[it - 1]).abs() < config.tol {
            converged = true;
            break;
        }
    }

    // final EAP + log-likelihood at the converged parameters
    let a: Vec<f64> = alpha.iter().map(|&al| al * al).collect();
    let lna: Vec<f64> = a.iter().map(|&ai| ai.ln()).collect();
    let mut final_ll = 0.0_f64;
    for p in 0..n_persons {
        let (mut a_sum, mut num, mut ar2, mut ld, mut nj) = (0.0, 0.0, 0.0, 0.0, 0usize);
        for i in 0..n_items {
            if is_obs(p, i) {
                let r = y[p * n_items + i] - beta[i];
                a_sum += a[i];
                num += a[i] * (-r);
                ar2 += a[i] * r * r;
                ld += lna[i];
                nj += 1;
            }
        }
        let pj = 1.0 / sigma_tau2 + a_sum;
        let te = num / pj;
        tau_eap[p] = te;
        tau_sd[p] = (1.0 / pj).sqrt();
        final_ll += -0.5 * (nj as f64 * ln2pi - ld + sigma_tau2.ln() + pj.ln() + ar2 - pj * te * te);
    }
    trace.push(final_ll);

    Ok(RtFit {
        alpha,
        beta,
        mu_tau: 0.0,
        sigma_tau: sigma_tau2.sqrt(),
        tau_eap,
        tau_sd,
        loglik: final_ll,
        loglik_trace: trace,
        n_iter,
        converged,
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
    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let mut sab = 0.0;
        let mut saa = 0.0;
        let mut sbb = 0.0;
        for (&x, &yv) in a.iter().zip(b) {
            sab += (x - ma) * (yv - mb);
            saa += (x - ma).powi(2);
            sbb += (yv - mb).powi(2);
        }
        sab / (saa.sqrt() * sbb.sqrt())
    }

    // Anchor 1: the Woodbury/closed-form marginal log-likelihood equals a naive
    // dense multivariate-normal log-pdf (certifies ln|Sigma|, the quadratic form,
    // and every sign convention of the likelihood path).
    #[test]
    fn rt_marginal_loglik_matches_dense_mvn() {
        let alpha = [1.5_f64, 2.0, 0.8];
        let beta = [4.0_f64, 3.5, 4.2];
        let sig2 = 0.09_f64;
        let yv = [3.7_f64, 3.9, 4.5]; // one person's log-times
        let n = 3usize;
        // closed form (E-step block)
        let a: Vec<f64> = alpha.iter().map(|&al| al * al).collect();
        let (mut a_sum, mut num, mut ar2, mut ld) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let r = yv[i] - beta[i];
            a_sum += a[i];
            num += a[i] * (-r);
            ar2 += a[i] * r * r;
            ld += a[i].ln();
        }
        let pj = 1.0 / sig2 + a_sum;
        let te = num / pj;
        let ln2pi = (2.0 * std::f64::consts::PI).ln();
        let closed = -0.5 * (n as f64 * ln2pi - ld + sig2.ln() + pj.ln() + ar2 - pj * te * te);
        // dense: Sigma = sig2*ones + diag(1/a_i); log N(y; beta, Sigma)
        let mut sigma = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                sigma[i][j] = sig2 + if i == j { 1.0 / a[i] } else { 0.0 };
            }
        }
        // Cholesky L (SPD)
        let mut l = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..=i {
                let mut s = sigma[i][j];
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        let logdet = 2.0 * (0..n).map(|i| l[i][i].ln()).sum::<f64>();
        // solve Sigma x = r via L L^T x = r
        let r: Vec<f64> = (0..n).map(|i| yv[i] - beta[i]).collect();
        let mut z = vec![0.0_f64; n];
        for i in 0..n {
            let mut s = r[i];
            for k in 0..i {
                s -= l[i][k] * z[k];
            }
            z[i] = s / l[i][i];
        }
        let mut x = vec![0.0_f64; n];
        for i in (0..n).rev() {
            let mut s = z[i];
            for k in (i + 1)..n {
                s -= l[k][i] * x[k];
            }
            x[i] = s / l[i][i];
        }
        let quad: f64 = (0..n).map(|i| r[i] * x[i]).sum();
        let dense = -0.5 * (n as f64 * ln2pi + logdet + quad);
        assert!((closed - dense).abs() < 1e-9, "Woodbury {closed} vs dense {dense}");
    }

    // Anchor 2: with sigma_tau -> 0 the model collapses to the per-item lognormal
    // MLE (beta_i = mean log-time, 1/alpha_i^2 = var of log-time).
    #[test]
    fn rt_reduces_to_lognormal_mle_when_speed_degenerate() {
        let mut u = lcg(5);
        let (np, ni) = (600usize, 8usize);
        let beta_t: Vec<f64> = (0..ni).map(|i| 3.5 + 0.1 * i as f64).collect();
        let alpha_t: Vec<f64> = (0..ni).map(|i| 1.2 + 0.1 * i as f64).collect();
        let mut times = vec![0.0_f64; np * ni];
        for p in 0..np {
            for i in 0..ni {
                let y = beta_t[i] + (1.0 / alpha_t[i]) * normal(&mut u); // tau ~ 0
                times[p * ni + i] = y.exp();
            }
        }
        let cfg = RtConfig { fix_sigma_tau: Some(1e-6), ..Default::default() };
        let fit = fit_rt_lognormal(&times, None, np, ni, cfg).unwrap();
        for i in 0..ni {
            let col: Vec<f64> = (0..np).map(|p| (times[p * ni + i]).ln()).collect();
            let m = col.iter().sum::<f64>() / np as f64;
            let var = col.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / np as f64;
            assert!((fit.beta[i] - m).abs() < 1e-2, "beta {} vs mle {m}", fit.beta[i]);
            assert!((1.0 / (fit.alpha[i] * fit.alpha[i]) - var).abs() < 1e-2, "alpha resvar mismatch");
        }
    }

    // Tier-1 recovery guard + monotone loglik.
    #[test]
    fn rt_recovers_parameters() {
        let (recov, _bias) = mc_rt(1, 800, false);
        assert!(recov.converged);
        assert!(recov.mono, "loglik trace must be non-decreasing");
        assert!(recov.corr_alpha > 0.85, "alpha corr {}", recov.corr_alpha);
        assert!(recov.corr_beta > 0.95, "beta corr {}", recov.corr_beta);
        assert!(recov.corr_tau > 0.8, "tau corr {}", recov.corr_tau);
        assert!((recov.sigma_hat - 0.3).abs() < 0.1, "sigma_tau {}", recov.sigma_hat);
    }

    struct RtRecov {
        converged: bool,
        mono: bool,
        corr_alpha: f64,
        corr_beta: f64,
        corr_tau: f64,
        sigma_hat: f64,
    }

    // One replication (or the aggregate for reps>1) of the recovery study.
    // Returns per-item RMSE/bias via the `bias` out-struct for the MC.
    fn mc_rt(seed: u64, n_persons: usize, skew: bool) -> (RtRecov, RtBias) {
        let ni = 20usize;
        let beta_t: Vec<f64> = (0..ni).map(|i| 3.5 + 1.0 * i as f64 / (ni - 1) as f64).collect();
        let alpha_t: Vec<f64> = (0..ni).map(|i| 1.0 + 2.0 * i as f64 / (ni - 1) as f64).collect();
        let sigma_true = 0.3_f64;
        let mut u = lcg(6000 + seed);
        let mut times = vec![0.0_f64; n_persons * ni];
        let mut obs = vec![true; n_persons * ni];
        let mut tau_true = vec![0.0_f64; n_persons];
        for p in 0..n_persons {
            // speed: normal, or mean-0 standardized skew (shifted exponential)
            let tau = if skew {
                sigma_true * (-(u().max(1e-12)).ln() - 1.0) // Exp(1)-1 has mean 0, var 1
            } else {
                sigma_true * normal(&mut u)
            };
            tau_true[p] = tau;
            for i in 0..ni {
                if u() < 0.3 {
                    obs[p * ni + i] = false;
                    times[p * ni + i] = 1.0; // placeholder (masked)
                    continue;
                }
                let y = beta_t[i] - tau + (1.0 / alpha_t[i]) * normal(&mut u);
                times[p * ni + i] = y.exp();
            }
        }
        let fit = fit_rt_lognormal(&times, Some(&obs), n_persons, ni, RtConfig::default()).unwrap();
        let mono = fit.loglik_trace.windows(2).all(|w| w[1] >= w[0] - 1e-6);
        let recov = RtRecov {
            converged: fit.converged,
            mono,
            corr_alpha: corr(&fit.alpha, &alpha_t),
            corr_beta: corr(&fit.beta, &beta_t),
            corr_tau: corr(&fit.tau_eap, &tau_true),
            sigma_hat: fit.sigma_tau,
        };
        let rmse = |est: &[f64], tru: &[f64]| -> f64 {
            (est.iter().zip(tru).map(|(&e, &t)| (e - t).powi(2)).sum::<f64>() / est.len() as f64).sqrt()
        };
        let bias = |est: &[f64], tru: &[f64]| -> f64 {
            est.iter().zip(tru).map(|(&e, &t)| e - t).sum::<f64>() / est.len() as f64
        };
        let b = RtBias {
            rmse_alpha: rmse(&fit.alpha, &alpha_t),
            rmse_beta: rmse(&fit.beta, &beta_t),
            bias_alpha: bias(&fit.alpha, &alpha_t),
            bias_beta: bias(&fit.beta, &beta_t),
            sigma_bias: fit.sigma_tau - sigma_true,
            corr_tau: recov.corr_tau,
        };
        (recov, b)
    }

    struct RtBias {
        rmse_alpha: f64,
        rmse_beta: f64,
        bias_alpha: f64,
        bias_beta: f64,
        sigma_bias: f64,
        corr_tau: f64,
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn rt_monte_carlo_500() {
        let reps = 500usize;
        for skew in [false, true] {
            let (mut ra, mut rb, mut ba, mut bb, mut sb, mut ct) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            for r in 0..reps {
                let (_rec, b) = mc_rt(100 + r as u64, 800, skew);
                ra += b.rmse_alpha;
                rb += b.rmse_beta;
                ba += b.bias_alpha;
                bb += b.bias_beta;
                sb += b.sigma_bias;
                ct += b.corr_tau;
            }
            let f = reps as f64;
            let label = if skew { "skew" } else { "normal" };
            println!(
                "[rt 500] {label}: RMSE(alpha)={:.4} RMSE(beta)={:.4} bias(alpha)={:.4} \
                 bias(beta)={:.4} bias(sigma)={:.4} corr(tau)={:.3}",
                ra / f, rb / f, ba / f, bb / f, sb / f, ct / f
            );
            // beta is a per-item weighted normal regression given tau -> robust to
            // the speed-distribution shape in BOTH conditions:
            assert!(rb / f < 0.05, "{label} beta RMSE too high: {}", rb / f);
            assert!((bb / f).abs() < 0.02, "{label} beta bias too high: {}", bb / f);
            assert!(ra / f < 0.15, "{label} alpha RMSE too high: {}", ra / f);
            if !skew {
                // under a correctly-specified normal speed prior, everything is
                // unbiased and speed recovers well; under skew alpha may carry a
                // small posterior-variance-correction bias (reported, not asserted)
                assert!((ba / f).abs() < 0.05, "normal alpha bias: {}", ba / f);
                assert!((sb / f).abs() < 0.05, "normal sigma_tau bias: {}", sb / f);
                assert!(ct / f > 0.9, "normal tau corr: {}", ct / f);
            }
        }
    }
}
