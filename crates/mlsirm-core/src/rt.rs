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
        if !loglik.is_finite() {
            return Err("response-time log-likelihood became non-finite".into());
        }
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
        final_ll += -0.5 * (nj as f64 * ln2pi - ld + sigma_tau2.ln() + pj.ln() + ar2 - pj * te * te);
    }
    if !final_ll.is_finite() {
        return Err("response-time final log-likelihood became non-finite".into());
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
    let termination_reason = if converged { "converged" } else { "max_iter_reached" };

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
        if !tau_hat.is_finite() {
            return Err("non-finite profiled speed".into());
        }
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
            if !h.is_finite() || !iz.is_finite() {
                return Err("non-finite studentized response-time residual".into());
            }
            z_resid[p * n_items + i] = iz;
            item_flag[p * n_items + i] = iz < -z_fast;
        }
        let dj = nj - 1;
        w[p] = wj;
        df[p] = dj;
        p_value[p] = crate::fitstats::chi2_sf(wj, dj as f64);
        if !p_value[p].is_finite() {
            return Err("non-finite response-time person-fit p-value".into());
        }
        flagged[p] = p_value[p] < alpha_level;
        // Wilson-Hilferty
        let d = 2.0 / (9.0 * dj as f64);
        l_t[p] = ((wj / dj as f64).cbrt() - (1.0 - d)) / d.sqrt();
        if !l_t[p].is_finite() {
            return Err("non-finite response-time person-fit standardization".into());
        }
    }

    Ok(RtPersonFit { w, df, l_t, p_value, flagged, tau_ml, z_resid, item_flag })
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

    #[test]
    fn rt_reports_max_iter_nonconvergence() {
        let n_persons = 20usize;
        let n_items = 4usize;
        let times: Vec<f64> = (0..n_persons * n_items)
            .map(|idx| 2.0 + (idx % n_items) as f64 * 0.1)
            .collect();
        let fit = fit_rt_lognormal(
            &times,
            None,
            n_persons,
            n_items,
            RtConfig { max_iter: 1, ..RtConfig::default() },
        )
        .unwrap();
        assert!(!fit.converged);
        assert_eq!(fit.termination_reason, "max_iter_reached");
        assert_eq!(fit.n_iter, 1);
        assert_eq!(fit.loglik_trace.len(), 2);
        assert!(fit.final_loglik_change.is_finite());
        assert!(fit.final_loglik_change >= RtConfig::default().tol);
        assert_eq!(fit.loglik, *fit.loglik_trace.last().unwrap());
    }

    #[test]
    fn rt_rejects_invalid_controls() {
        let times = [2.0_f64];
        for config in [
            RtConfig { max_iter: 0, ..RtConfig::default() },
            RtConfig { tol: f64::NAN, ..RtConfig::default() },
            RtConfig { var_floor: f64::INFINITY, ..RtConfig::default() },
            RtConfig { sigma_floor: 0.0, ..RtConfig::default() },
        ] {
            assert!(fit_rt_lognormal(&times, None, 1, 1, config).is_err());
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

    // Anchor: at true item params the residuals are N(0,1) and W is exactly
    // chi-square — chi2(n) at known tau, chi2(n-1) once tau is profiled.
    #[test]
    fn rt_person_fit_chi2_at_true_params() {
        let mut u = lcg(31);
        let (np, ni) = (30000usize, 20usize);
        let beta: Vec<f64> = (0..ni).map(|i| 3.5 + i as f64 / (ni - 1) as f64).collect();
        let alpha: Vec<f64> = (0..ni).map(|i| 1.0 + 2.0 * i as f64 / (ni - 1) as f64).collect();
        let mut times = vec![0.0_f64; np * ni];
        let mut tau = vec![0.0_f64; np];
        for p in 0..np {
            let tj = 0.3 * normal(&mut u);
            tau[p] = tj;
            for i in 0..ni {
                times[p * ni + i] = (beta[i] - tj + normal(&mut u) / alpha[i]).exp();
            }
        }
        // (1) known tau: z ~ N(0,1), mean(sum z^2) ~ n
        let (mut sz, mut sz2, mut cnt, mut sw) = (0.0_f64, 0.0, 0.0, 0.0);
        for p in 0..np {
            let mut wk = 0.0;
            for i in 0..ni {
                let z = alpha[i] * (times[p * ni + i].ln() - beta[i] + tau[p]);
                sz += z;
                sz2 += z * z;
                cnt += 1.0;
                wk += z * z;
            }
            sw += wk;
        }
        let mz = sz / cnt;
        let sdz = (sz2 / cnt - mz * mz).sqrt();
        assert!(mz.abs() < 0.02 && (sdz - 1.0).abs() < 0.03, "known-tau z not N(0,1): {mz}, {sdz}");
        assert!((sw / np as f64 - ni as f64).abs() < 0.03 * ni as f64, "known-tau W not chi2(n)");
        // (2) profiled (production path): W ~ chi2(n-1), l_t ~ N(0,1), Type I ~ .05
        let pf = rt_person_fit(&times, None, np, ni, &alpha, &beta, 0.05, 1.645).unwrap();
        let mw = pf.w.iter().sum::<f64>() / np as f64;
        assert!((mw - (ni - 1) as f64).abs() < 0.03 * (ni - 1) as f64, "profiled W not chi2(n-1): {mw}");
        let mlt = pf.l_t.iter().sum::<f64>() / np as f64;
        let sdlt = (pf.l_t.iter().map(|&x| (x - mlt).powi(2)).sum::<f64>() / np as f64).sqrt();
        assert!(mlt.abs() < 0.05 && (sdlt - 1.0).abs() < 0.05, "l_t not N(0,1): {mlt}, {sdlt}");
        let t1 = pf.flagged.iter().filter(|&&f| f).count() as f64 / np as f64;
        assert!((0.03..=0.07).contains(&t1), "Type I: {t1}");
        // (3) per-item studentized residual ~ N(0,1)
        let iz: Vec<f64> = pf.z_resid.iter().cloned().filter(|v| v.is_finite()).collect();
        let miz = iz.iter().sum::<f64>() / iz.len() as f64;
        let sdiz = (iz.iter().map(|&x| (x - miz).powi(2)).sum::<f64>() / iz.len() as f64).sqrt();
        assert!(miz.abs() < 0.02 && (sdiz - 1.0).abs() < 0.03, "item_z not N(0,1): {miz}, {sdiz}");
    }

    // (Type I over consistent responders, power over aberrant, l_t mean/sd, and
    // per-item recall of tampered responses). mode 0 = rapid guessing on the last
    // items; mode 1 = preknowledge on the first items. fit_items uses MML-estimated
    // item params (production path) instead of the true ones.
    fn mc_rt_pf(reps: usize, n_persons: usize, skew: bool, mode: u8, fit_items: bool) -> (f64, f64, f64, f64, f64) {
        let ni = 20usize;
        let beta: Vec<f64> = (0..ni).map(|i| 3.5 + i as f64 / (ni - 1) as f64).collect();
        let alpha: Vec<f64> = (0..ni).map(|i| 1.0 + 2.0 * i as f64 / (ni - 1) as f64).collect();
        let n_ab = n_persons / 10;
        let (mut t1n, mut t1c, mut pwn, mut pwc) = (0usize, 0usize, 0usize, 0usize);
        let (mut lts, mut lt2, mut ltc) = (0.0_f64, 0.0, 0usize);
        let (mut recn, mut recc) = (0usize, 0usize);
        for rep in 0..reps as u64 {
            let mut u = lcg(70_000 + rep * 131 + skew as u64 * 3 + mode as u64 * 7 + fit_items as u64 * 11);
            let mut times = vec![0.0_f64; n_persons * ni];
            let mut tampered = vec![false; n_persons * ni];
            for p in 0..n_persons {
                let ab = p < n_ab;
                let tj = if skew { 0.3 * (-(u().max(1e-12)).ln() - 1.0) } else { 0.3 * normal(&mut u) };
                for i in 0..ni {
                    let short = ab
                        && match mode {
                            0 => i >= ni - ni * 35 / 100, // last 35%
                            _ => i < ni * 30 / 100,       // first 30%
                        };
                    let y = if short {
                        (beta[i] - tj) - 2.5 + 0.3 * normal(&mut u)
                    } else {
                        beta[i] - tj + normal(&mut u) / alpha[i]
                    };
                    times[p * ni + i] = y.exp();
                    tampered[p * ni + i] = short;
                }
            }
            let (ea, eb) = if fit_items {
                // calibrate on a FRESH CLEAN sample: isolates item-parameter
                // sampling uncertainty (the production regime) rather than the
                // separate contamination-by-aberrant-responders effect.
                let mut uc = lcg(80_000 + rep * 131 + skew as u64 * 3);
                let mut ct = vec![0.0_f64; n_persons * ni];
                for p in 0..n_persons {
                    let tj = if skew { 0.3 * (-(uc().max(1e-12)).ln() - 1.0) } else { 0.3 * normal(&mut uc) };
                    for i in 0..ni {
                        ct[p * ni + i] = (beta[i] - tj + normal(&mut uc) / alpha[i]).exp();
                    }
                }
                let fit = fit_rt_lognormal(&ct, None, n_persons, ni, RtConfig::default()).unwrap();
                (fit.alpha, fit.beta)
            } else {
                (alpha.clone(), beta.clone())
            };
            let pf = rt_person_fit(&times, None, n_persons, ni, &ea, &eb, 0.05, 1.645).unwrap();
            for p in 0..n_persons {
                if !pf.w[p].is_finite() {
                    continue;
                }
                if p < n_ab {
                    if pf.flagged[p] {
                        pwn += 1;
                    }
                    pwc += 1;
                    for i in 0..ni {
                        if tampered[p * ni + i] {
                            recc += 1;
                            if pf.item_flag[p * ni + i] {
                                recn += 1;
                            }
                        }
                    }
                } else {
                    if pf.flagged[p] {
                        t1n += 1;
                    }
                    t1c += 1;
                    lts += pf.l_t[p];
                    lt2 += pf.l_t[p] * pf.l_t[p];
                    ltc += 1;
                }
            }
        }
        let mlt = lts / ltc as f64;
        (
            t1n as f64 / t1c as f64,
            pwn as f64 / pwc as f64,
            mlt,
            (lt2 / ltc as f64 - mlt * mlt).sqrt(),
            recn as f64 / recc.max(1) as f64,
        )
    }

    #[test]
    fn rt_person_fit_type1_and_power() {
        let (t1, pw, mlt, sdlt, _) = mc_rt_pf(6, 800, false, 0, false);
        let (_, pw_pre, _, _, rec) = mc_rt_pf(6, 800, false, 1, false);
        let (t1s, _, _, _, _) = mc_rt_pf(6, 800, true, 0, false);
        let (t1f, pwf, _, _, _) = mc_rt_pf(4, 800, false, 0, true); // production path
        println!(
            "[rt-pf] Type I={t1:.3} power(guess)={pw:.3} power(preknow)={pw_pre:.3} \
             l_t=({mlt:.2},{sdlt:.2}) skew Type I={t1s:.3} fitted Type I={t1f:.3} recall={rec:.3}"
        );
        assert!((0.01..=0.12).contains(&t1), "Type I: {t1}");
        assert!(pw > 0.5 && pw_pre > 0.5, "power: {pw}/{pw_pre}");
        assert!(mlt.abs() < 0.4 && (0.75..=1.3).contains(&sdlt), "l_t: {mlt}/{sdlt}");
        assert!((0.01..=0.12).contains(&t1s), "skew Type I: {t1s}");
        assert!((0.01..=0.13).contains(&t1f) && pwf > 0.5, "fitted path: {t1f}/{pwf}");
    }

    #[test]
    fn rt_person_fit_rejects_invalid_parameters_and_controls() {
        let times = vec![1.0, 2.0, 1.5, 2.5];
        let alpha = vec![1.0, 1.5];
        let beta = vec![0.0, 0.5];
        let bad = |alpha: &[f64], beta: &[f64], alpha_level: f64, z_fast: f64| {
            rt_person_fit(&times, None, 2, 2, alpha, beta, alpha_level, z_fast).is_err()
        };
        assert!(bad(&[0.0, 1.5], &beta, 0.05, 1.645));
        assert!(bad(&[f64::NAN, 1.5], &beta, 0.05, 1.645));
        assert!(bad(&[1e308, 1.5], &beta, 0.05, 1.645));
        assert!(bad(&[1e-308, 1e-308], &beta, 0.05, 1.645));
        assert!(bad(&alpha, &[0.0, f64::INFINITY], 0.05, 1.645));
        assert!(bad(&alpha, &[1e308, 1e308], 0.05, 1.645));
        assert!(bad(&alpha, &beta, f64::NAN, 1.645));
        assert!(bad(&alpha, &beta, 0.05, -0.1));
        assert!(bad(&alpha, &beta, 0.05, f64::INFINITY));
        assert!(rt_person_fit(&[], None, usize::MAX, 2, &alpha, &beta, 0.05, 1.645).is_err());
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn rt_person_fit_monte_carlo_500() {
        for skew in [false, true] {
            for mode in [0u8, 1] {
                let (t1, pw, mlt, sdlt, rec) = mc_rt_pf(500, 600, skew, mode, false);
                println!(
                    "[rt-pf 500] skew={skew} mode={mode}: Type I={t1:.4} power={pw:.3} \
                     l_t=({mlt:.3},{sdlt:.3}) item-recall={rec:.3}"
                );
                assert!((0.03..=0.08).contains(&t1), "Type I off nominal: {t1}");
                assert!(pw > 0.7, "power too low: {pw}");
            }
        }
        // production path: fit item params by MML, then person-fit
        let (t1f, pwf, _, _, _) = mc_rt_pf(500, 600, false, 0, true);
        println!("[rt-pf 500] fitted-items: Type I={t1f:.4} power={pwf:.3}");
        assert!((0.03..=0.09).contains(&t1f), "fitted-item Type I off nominal: {t1f}");
        assert!(pwf > 0.7, "fitted-item power too low: {pwf}");
    }
}
