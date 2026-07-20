//! Continuous Response Model (Samejima, 1973) by marginal-ML EM.
//!
//! The CRM is the limit of the graded response model as the number of ordered
//! categories grows without bound, for an item scored on a *continuous* bounded
//! scale. Operationally (Wang & Zeng, 1998), the logit of the bounded response is
//! conditionally normal and linear in the latent trait: for a response
//! `Z_ij in (0,1)`, the transform `X_ij = ln(Z_ij / (1 - Z_ij))` satisfies
//!
//! ```text
//! X_ij | theta_j ~ Normal( a_i * theta_j + d_i , sigma_i^2 ),   theta_j ~ N(0,1)
//! ```
//!
//! with item slope `a_i` (loading of the transformed response on the trait),
//! intercept `d_i`, and residual standard deviation `sigma_i`. This is Samejima's
//! CRM in the logit metric; the classic operating-characteristic parameters map as
//! `a_i = gamma_i`, `d_i = -gamma_i b_i`, `sigma_i = gamma_i / alpha_i`, so the fit
//! reports the derived **discrimination** `alpha_i = a_i / sigma_i` and
//! **difficulty** `b_i = -d_i / a_i` alongside the working `(a, d, sigma)`.
//!
//! The `Z -> X` Jacobian `ln|dX/dZ| = -ln(Z(1-Z))` is constant in the item
//! parameters, so it is omitted from the EM (it only shifts the reported
//! transformed-space log-likelihood by a data-only constant). The item M-step is a
//! closed-form weighted least squares (regress `X` on `theta` under the posterior)
//! plus a residual-variance update — no Newton iteration.
//!
//! Only the continuous-response data type is new; the quadrature, EM bookkeeping,
//! and identification (`theta ~ N(0,1)` fixes the scale) mirror the crate's other
//! marginal-ML fits. Convergence requires a finite, non-decreasing observed-data
//! log-likelihood and a signed final increment no larger than
//! `tol * (1 + |previous log-likelihood|)` (Dempster et al., 1977; Wu, 1983).
//!
//! # References (APA 7th ed.)
//! Samejima, F. (1973). Homogeneous case of the continuous response model.
//!   *Psychometrika, 38*(2), 203-219. https://doi.org/10.1007/BF02291114
//! Wang, T., & Zeng, L. (1998). Item parameter estimation for a continuous response
//!   model using an EM algorithm. *Applied Psychological Measurement, 22*(4),
//!   333-344. https://doi.org/10.1177/014662169802200402
//! Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977). Maximum likelihood from
//!   incomplete data via the EM algorithm. *Journal of the Royal Statistical Society:
//!   Series B (Methodological), 39*(1), 1-22.
//!   https://doi.org/10.1111/j.2517-6161.1977.tb01600.x
//! Wu, C. F. J. (1983). On the convergence properties of the EM algorithm.
//!   *The Annals of Statistics, 11*(1), 95-103.
//!   https://doi.org/10.1214/aos/1176346060

/// Fitted continuous response model (Samejima, 1973). `slope`/`intercept`/`resid_sd`
/// are the working `(a_i, d_i, sigma_i)` of the logit-normal form; `discrimination`
/// and `difficulty` are the derived Samejima `(alpha_i, b_i)` (`b_i` is `NaN` for a
/// non-discriminating item whose slope is ~0, where difficulty is undefined).
/// `theta` is the per-person EAP trait.
#[derive(Clone, Debug)]
pub struct CrmResult {
    pub slope: Vec<f64>,
    pub intercept: Vec<f64>,
    pub resid_sd: Vec<f64>,
    pub discrimination: Vec<f64>,
    pub difficulty: Vec<f64>,
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Why fitting stopped: `"tolerance"` or `"max_iter"`.
    pub termination_reason: String,
    /// Signed final observed-data log-likelihood increment.
    pub final_delta: f64,
    /// Effective observed-data log-likelihood increment required for convergence.
    pub stopping_tolerance: f64,
    /// `3 * n_items` (slope, intercept, residual sd per item).
    pub n_parameters: usize,
}

#[derive(Clone, Copy)]
struct CrmWlsStats {
    s1: f64,
    sth: f64,
    sthth: f64,
    sx: f64,
    sxth: f64,
    sxx: f64,
}

fn contextualize_crm_update(
    update: Result<Option<(f64, f64, f64)>, String>,
    item: usize,
) -> Result<Option<(f64, f64, f64)>, String> {
    match update {
        Ok(value) => Ok(value),
        Err(message) => Err(format!("{message} for item {item}")),
    }
}

fn checked_crm_delta(
    current: f64,
    previous: Option<f64>,
    tol: f64,
) -> Result<Option<(f64, f64, bool)>, String> {
    if !current.is_finite() {
        return Err("CRM observed-data log-likelihood became non-finite".into());
    }
    let Some(previous) = previous else {
        return Ok(None);
    };
    let delta = current - previous;
    let stopping_tolerance = tol * (1.0 + previous.abs());
    let monotone_slack = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    if delta < -monotone_slack {
        return Err(format!(
            "CRM EM log-likelihood decreased by {delta:e}, beyond numerical slack {monotone_slack:e}"
        ));
    }
    Ok(Some((
        delta,
        stopping_tolerance,
        delta <= stopping_tolerance,
    )))
}

fn crm_wls_update(stats: CrmWlsStats, eps: f64) -> Result<Option<(f64, f64, f64)>, String> {
    let det = stats.sthth * stats.s1 - stats.sth * stats.sth;
    if det.abs() < 1e-12 {
        return Ok(None);
    }
    let slope = (stats.sxth * stats.s1 - stats.sth * stats.sx) / det;
    let intercept = (stats.sthth * stats.sx - stats.sth * stats.sxth) / det;
    let resid = (stats.sxx - slope * stats.sxth - intercept * stats.sx) / stats.s1;
    if !slope.is_finite() || !intercept.is_finite() || !resid.is_finite() {
        return Err("CRM M-step produced non-finite values".into());
    }
    Ok(Some((slope, intercept, resid.max(eps * eps).sqrt())))
}

fn reflect_crm_loadings(loadings: &mut [f64]) {
    if loadings.iter().sum::<f64>() < 0.0 {
        loadings.iter_mut().for_each(|loading| *loading = -*loading);
    }
}

fn crm_difficulty(slope: f64, intercept: f64) -> f64 {
    if slope.abs() > 1e-6 {
        -intercept / slope
    } else {
        f64::NAN
    }
}

/// Fit the continuous response model (Samejima, 1973) by marginal-ML EM.
/// `responses` is row-major `n_persons * n_items` with entries in `(0, 1)`
/// (values are clamped to `[eps, 1-eps]` before the logit transform); `observed`
/// marks non-missing cells (dropped under MAR). `theta ~ N(0,1)` on the
/// `q_theta`-node Gauss-Hermite grid. Returns `Err` on malformed input.
#[allow(clippy::too_many_arguments)]
pub fn fit_crm(
    responses: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    q_theta: usize,
    max_iter: usize,
    tol: f64,
) -> Result<CrmResult, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("n_persons and n_items must both be positive".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    let n_parameters = 3usize
        .checked_mul(n_items)
        .ok_or_else(|| "3 * n_items overflows usize".to_string())?;
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if responses.len() != expected {
        return Err("responses must have length n_persons * n_items".into());
    }
    if observed.len() != expected {
        return Err("observed must have length n_persons * n_items".into());
    }
    let mut item_observed = vec![0usize; n_items];
    for (idx, &z) in responses.iter().enumerate() {
        if observed[idx] && (!z.is_finite() || z <= 0.0 || z >= 1.0) {
            return Err("observed responses must lie in the open interval (0, 1)".into());
        }
        if observed[idx] {
            item_observed[idx % n_items] += 1;
        }
    }
    if let Some(item) = item_observed.iter().position(|&count| count == 0) {
        return Err(format!("item {item} has no observed responses"));
    }
    let (nodes, weights) = crate::quadrature::gh_rule(q_theta)
        .ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
    let q = nodes.len();
    let log_w: Vec<f64> = weights.iter().map(|w| w.ln()).collect();
    let eps = 1e-6;
    let ln_2pi = (2.0 * std::f64::consts::PI).ln();

    // Logit-transform the observed responses once.
    let mut x = vec![0.0f64; n_persons * n_items];
    for idx in 0..responses.len() {
        if observed[idx] {
            let z = responses[idx].clamp(eps, 1.0 - eps);
            x[idx] = (z / (1.0 - z)).ln();
        }
    }

    // Init: unit loading; intercept = item mean of X; residual sd = item sd of X with
    // the unit-trait variance removed (floored) so the loading has room to explain it.
    let mut a = vec![1.0f64; n_items];
    let mut d = vec![0.0f64; n_items];
    let mut sigma = vec![1.0f64; n_items];
    for i in 0..n_items {
        let (mut s1, mut sx, mut sxx) = (0.0f64, 0.0f64, 0.0f64);
        for j in 0..n_persons {
            let idx = j * n_items + i;
            if observed[idx] {
                s1 += 1.0;
                sx += x[idx];
                sxx += x[idx] * x[idx];
            }
        }
        if s1 > 0.0 {
            let mean = sx / s1;
            let var = (sxx / s1 - mean * mean).max(eps);
            d[i] = mean;
            sigma[i] = (var - 1.0).max(0.25 * var).sqrt();
        }
    }

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut post = vec![0.0f64; q];

    for _ in 0..max_iter {
        // Per-item expected sufficient statistics for the weighted regression.
        let mut s1 = vec![0.0f64; n_items];
        let mut sth = vec![0.0f64; n_items];
        let mut sthth = vec![0.0f64; n_items];
        let mut sx = vec![0.0f64; n_items];
        let mut sxth = vec![0.0f64; n_items];
        let mut sxx = vec![0.0f64; n_items];
        let mut total_ll = 0.0f64;
        let log_sigma: Vec<f64> = sigma.iter().map(|s| s.ln()).collect();

        for j in 0..n_persons {
            for (qi, &node) in nodes.iter().enumerate() {
                let mut acc = log_w[qi];
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let r = (x[idx] - a[i] * node - d[i]) / sigma[i];
                        acc += -0.5 * ln_2pi - log_sigma[i] - 0.5 * r * r;
                    }
                }
                post[qi] = acc;
            }
            let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in post.iter() {
                denom += (v - mx).exp();
            }
            total_ll += mx + denom.ln();
            for v in post.iter_mut() {
                *v = (*v - mx).exp() / denom;
            }
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let xij = x[idx];
                    for (qi, &node) in nodes.iter().enumerate() {
                        let p = post[qi];
                        s1[i] += p;
                        sth[i] += p * node;
                        sthth[i] += p * node * node;
                        sx[i] += p * xij;
                        sxth[i] += p * xij * node;
                        sxx[i] += p * xij * xij;
                    }
                }
            }
        }
        let convergence = checked_crm_delta(total_ll, loglik_trace.last().copied(), tol)?;
        loglik_trace.push(total_ll);

        // Converge check before the M-step so returned params match the trace endpoint.
        if let Some((_, _, reached_tolerance)) = convergence {
            if reached_tolerance {
                converged = true;
                break;
            }
        }

        // M-step: closed-form WLS of X on theta, then the residual variance.
        for i in 0..n_items {
            let stats = CrmWlsStats {
                s1: s1[i],
                sth: sth[i],
                sthth: sthth[i],
                sx: sx[i],
                sxth: sxth[i],
                sxx: sxx[i],
            };
            if let Some((ai, di, sigma_i)) =
                contextualize_crm_update(crm_wls_update(stats, eps), i)?
            {
                a[i] = ai;
                d[i] = di;
                sigma[i] = sigma_i;
            }
        }
        n_iter += 1;
    }

    // Reflection convention: make the average loading non-negative (theta -> -theta,
    // a -> -a leaves the model invariant), so recovery is comparable to a
    // positive-loading generating truth.
    reflect_crm_loadings(&mut a);

    // Final person EAP pass at the (possibly sign-flipped) converged parameters; the
    // flipped slopes yield the correspondingly reflected trait, keeping the fit
    // invariant.
    let log_sigma: Vec<f64> = sigma.iter().map(|s| s.ln()).collect();
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0f64;
    for j in 0..n_persons {
        for (qi, &node) in nodes.iter().enumerate() {
            let mut acc = log_w[qi];
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let r = (x[idx] - a[i] * node - d[i]) / sigma[i];
                    acc += -0.5 * ln_2pi - log_sigma[i] - 0.5 * r * r;
                }
            }
            post[qi] = acc;
        }
        let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for v in post.iter() {
            denom += (v - mx).exp();
        }
        final_ll += mx + denom.ln();
        let mut m = 0.0f64;
        for (qi, &node) in nodes.iter().enumerate() {
            m += (post[qi] - mx).exp() / denom * node;
        }
        theta[j] = m;
    }
    if !converged {
        let previous = *loglik_trace
            .last()
            .expect("positive max_iter always produces a CRM log-likelihood endpoint");
        let (_, _, reached_tolerance) = checked_crm_delta(final_ll, Some(previous), tol)?
            .expect("a previous endpoint always produces convergence evidence");
        loglik_trace.push(final_ll);
        converged = reached_tolerance;
    }

    let final_delta = loglik_trace[loglik_trace.len() - 1] - loglik_trace[loglik_trace.len() - 2];
    let stopping_tolerance = tol * (1.0 + loglik_trace[loglik_trace.len() - 2].abs());
    let termination_reason = if converged { "tolerance" } else { "max_iter" };

    let discrimination: Vec<f64> = (0..n_items).map(|i| a[i] / sigma[i]).collect();
    // Samejima difficulty b = -d/a is undefined for a non-discriminating item
    // (slope ~ 0); report NaN there rather than a misleading blow-up.
    let difficulty: Vec<f64> = (0..n_items).map(|i| crm_difficulty(a[i], d[i])).collect();

    Ok(CrmResult {
        slope: a,
        intercept: d,
        resid_sd: sigma,
        discrimination,
        difficulty,
        theta,
        loglik_trace,
        n_iter,
        converged,
        termination_reason: termination_reason.to_string(),
        final_delta,
        stopping_tolerance,
        n_parameters,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/crm_tests.rs"]
mod tests;
