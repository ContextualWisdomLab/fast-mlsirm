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
        Self {
            max_iter: 500,
            tol: 1e-6,
            var_floor: 1e-4,
            sigma_floor: 1e-4,
            fix_sigma_tau: None,
        }
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
    pub termination_reason: String,
    pub final_loglik_change: f64,
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
    if config.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !(config.tol.is_finite() && config.tol > 0.0) {
        return Err("tol must be positive and finite".into());
    }
    if !(config.var_floor.is_finite() && config.var_floor > 0.0) {
        return Err("var_floor must be positive and finite".into());
    }
    if !(config.sigma_floor.is_finite() && config.sigma_floor > 0.0) {
        return Err("sigma_floor must be positive and finite".into());
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
        // Validation bounds every observed log-time to the finite f64 log
        // domain; the variance floors keep this Gaussian likelihood finite.
        trace.push(loglik);

        // Stop at the likelihood state that is actually returned. Checking after
        // the M-step would return parameters one update beyond the state whose
        // likelihood change met `tol`.
        if it > 0 && (trace[it] - trace[it - 1]).abs() < config.tol {
            converged = true;
            break;
        }

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
        final_ll +=
            -0.5 * (nj as f64 * ln2pi - ld + sigma_tau2.ln() + pj.ln() + ar2 - pj * te * te);
    }
    if converged {
        // The loop broke before the M-step, so this recomputation is the same
        // parameter state. Replace the endpoint instead of duplicating it.
        *trace.last_mut().expect("a converged fit has a likelihood") = final_ll;
    } else {
        // At max_iter the final M-step has not yet been evaluated in the trace.
        trace.push(final_ll);
    }
    let final_loglik_change = trace
        .windows(2)
        .last()
        .map_or(f64::INFINITY, |w| (w[1] - w[0]).abs());
    let termination_reason = if converged {
        "converged"
    } else {
        "max_iter_reached"
    };

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
        termination_reason: termination_reason.to_string(),
        final_loglik_change,
    })
}

/// Per-person response-time person-fit result ([`rt_person_fit`]).
pub struct RtPersonFit {
    /// `W_j = sum_i z_hat_ij^2`, distributed `chi2(n_j - 1)` under the model
    /// (`NaN` for a person with fewer than 2 observed times).
    pub w: Vec<f64>,
    /// Degrees of freedom `n_j - 1`.
    pub df: Vec<usize>,
    /// Wilson-Hilferty standardization of `W` (`~ N(0,1)`; positive = aberrant).
    /// The field name is retained for API compatibility; it is not a separate
    /// literature statistic named `l_t`.
    pub l_t: Vec<f64>,
    /// Upper-tail p-value `P(chi2_{df} >= W)`.
    pub p_value: Vec<f64>,
    /// `p_value < alpha_level`.
    pub flagged: Vec<bool>,
    /// The per-person ML-profiled speed used (differs from an EAP speed).
    pub tau_ml: Vec<f64>,
    /// `n_persons * n_items` studentized log-time residuals (`~ N(0,1)` marginally;
    /// `NaN` where unobserved). A strongly negative value is a too-fast response.
    pub z_resid: Vec<f64>,
    /// `n_persons * n_items` one-sided too-fast flags (`z_resid < -z_fast`).
    pub item_flag: Vec<bool>,
}

/// Sinharay's (2018) frequentist response-time person-fit statistic under a fitted
/// lognormal RT model. For each person the speed is profiled by per-person ML, so
/// the sum of squared standardized log-time
/// residuals `W_j = sum_i [alpha_i (ln T_ij - (beta_i - tau_hat_j))]^2` is
/// *exactly* `chi2(n_j - 1)` under the model — an orthogonal-projection identity,
/// not an asymptotic approximation, so the estimated-speed correction is a clean
/// loss of one degree of freedom (the RT analogue of `l_z*`). Detects speed
/// *inconsistency across items* — rapid guessing (a cluster of implausibly fast
/// responses) or item preknowledge (fast responses concentrated on hard items) —
/// which appear as strongly negative residuals; a uniformly fast-but-consistent
/// responder is correctly *not* flagged because the profile absorbs the speed
/// level. `alpha`/`beta` come from a fitted [`RtFit`]; `alpha_level` flags the
/// aggregate `W`, `z_fast` the per-item one-sided too-fast residual.
///
/// The per-item studentized ML residuals are a fixed-bank diagnostic provided by
/// this crate. Van der Linden and Guo (2008) motivate the interpretation of
/// unusually fast item responses, but their Bayesian leave-one-out procedure is
/// not the statistic implemented here.
/// Inputs whose squared time discriminations or profiled residual arithmetic are
/// non-finite are rejected rather than returned as undefined diagnostics.
///
/// # References (APA 7th ed.)
///
/// van der Linden, W. J., & Guo, F. (2008). Bayesian procedures for identifying
///   aberrant response-time patterns in adaptive testing. *Psychometrika, 73*(3),
///   365–384. https://doi.org/10.1007/s11336-007-9046-8
///
/// Sinharay, S. (2018). A new person-fit statistic for the lognormal model for
///   response times. *Journal of Educational Measurement, 55*(4), 457–476.
///   https://doi.org/10.1111/jedm.12188
#[allow(clippy::too_many_arguments)]
pub fn rt_person_fit(
    times: &[f64],
    observed: Option<&[bool]>,
    n_persons: usize,
    n_items: usize,
    alpha: &[f64],
    beta: &[f64],
    alpha_level: f64,
    z_fast: f64,
) -> Result<RtPersonFit, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must be positive".into());
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if times.len() != n_cells {
        return Err("times must have length n_persons * n_items".into());
    }
    if alpha.len() != n_items || beta.len() != n_items {
        return Err("alpha and beta must have length n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if alpha.iter().any(|a| {
        let a2 = *a * *a;
        !a.is_finite() || *a <= 0.0 || !a2.is_finite() || a2 <= 0.0
    }) {
        return Err("alpha values must have finite positive squares".into());
    }
    if beta.iter().any(|b| !b.is_finite()) {
        return Err("beta values must be finite".into());
    }
    if !(0.0 < alpha_level && alpha_level < 1.0) {
        return Err("alpha_level must be in (0, 1)".into());
    }
    if !z_fast.is_finite() || z_fast < 0.0 {
        return Err("z_fast must be finite and non-negative".into());
    }
    let is_obs = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);

    let mut w = vec![f64::NAN; n_persons];
    let mut df = vec![0usize; n_persons];
    let mut l_t = vec![f64::NAN; n_persons];
    let mut p_value = vec![f64::NAN; n_persons];
    let mut flagged = vec![false; n_persons];
    let mut tau_ml = vec![f64::NAN; n_persons];
    let mut z_resid = vec![f64::NAN; n_persons * n_items];
    let mut item_flag = vec![false; n_persons * n_items];

    for p in 0..n_persons {
        // pass 1: profiled speed tau_hat = sum a_i(beta_i - y_i) / sum a_i
        let (mut num, mut s, mut nj) = (0.0_f64, 0.0_f64, 0usize);
        for i in 0..n_items {
            if !is_obs(p, i) {
                continue;
            }
            let t = times[p * n_items + i];
            if !t.is_finite() || t <= 0.0 {
                return Err("response times must be finite and positive where observed".into());
            }
            let a2 = alpha[i] * alpha[i];
            let contribution = a2 * (beta[i] - t.ln());
            if !contribution.is_finite() {
                return Err("non-finite response-time profile contribution".into());
            }
            num += contribution;
            s += a2;
            if !num.is_finite() || !s.is_finite() {
                return Err("non-finite response-time profile accumulation".into());
            }
            nj += 1;
        }
        if nj < 2 || s <= 0.0 {
            continue; // undefined; leave NaN/unflagged
        }
        let tau_hat = num / s;
        // `num` and the strictly positive `s` were checked finite above, so
        // their weighted-average ratio is finite as well.
        tau_ml[p] = tau_hat;
        // pass 2: residuals + statistics
        let mut wj = 0.0_f64;
        for i in 0..n_items {
            if !is_obs(p, i) {
                continue;
            }
            let y = times[p * n_items + i].ln();
            let zhat = alpha[i] * (y - beta[i] + tau_hat);
            let z2 = zhat * zhat;
            if !zhat.is_finite() || !z2.is_finite() {
                return Err("non-finite response-time residual".into());
            }
            wj += z2;
            if !wj.is_finite() {
                return Err("non-finite response-time person-fit statistic".into());
            }
            let h = alpha[i] * alpha[i] / s; // leverage
            let iz = zhat / (1.0 - h).max(1e-12).sqrt();
            // `h` is a finite squared-loading share in [0, 1], while finite
            // `z2` above bounds `zhat`; the floored denominator keeps `iz`
            // finite.
            z_resid[p * n_items + i] = iz;
            item_flag[p * n_items + i] = iz < -z_fast;
        }
        let dj = nj - 1;
        w[p] = wj;
        df[p] = dj;
        p_value[p] = crate::fitstats::chi2_sf(wj, dj as f64);
        flagged[p] = p_value[p] < alpha_level;
        // Wilson-Hilferty
        let d = 2.0 / (9.0 * dj as f64);
        l_t[p] = ((wj / dj as f64).cbrt() - (1.0 - d)) / d.sqrt();
    }

    Ok(RtPersonFit {
        w,
        df,
        l_t,
        p_value,
        flagged,
        tau_ml,
        z_resid,
        item_flag,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/rt_tests.rs"]
mod tests;
