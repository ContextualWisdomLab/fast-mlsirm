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
//! marginal-ML fits.
//!
//! # References (APA 7th ed.)
//! Samejima, F. (1973). Homogeneous case of the continuous response model.
//!   *Psychometrika, 38*(2), 203-219. https://doi.org/10.1007/BF02291114
//! Wang, T., & Zeng, L. (1998). Item parameter estimation for a continuous response
//!   model using an EM algorithm. *Applied Psychological Measurement, 22*(4),
//!   333-344. https://doi.org/10.1177/014662169802200402

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
    /// `3 * n_items` (slope, intercept, residual sd per item).
    pub n_parameters: usize,
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
    if responses.len() != n_persons * n_items {
        return Err("responses must have length n_persons * n_items".into());
    }
    if observed.len() != n_persons * n_items {
        return Err("observed must have length n_persons * n_items".into());
    }
    for (idx, &z) in responses.iter().enumerate() {
        if observed[idx] && (!z.is_finite() || z <= 0.0 || z >= 1.0) {
            return Err("observed responses must lie in the open interval (0, 1)".into());
        }
    }
    let (nodes, weights) =
        crate::quadrature::gh_rule(q_theta).ok_or_else(|| format!("unsupported q_theta {q_theta}"))?;
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
        loglik_trace.push(total_ll);

        // Converge check before the M-step so returned params match the trace endpoint.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < tol {
                converged = true;
                break;
            }
        }

        // M-step: closed-form WLS of X on theta, then the residual variance.
        for i in 0..n_items {
            let det = sthth[i] * s1[i] - sth[i] * sth[i];
            if det.abs() < 1e-12 {
                continue; // degenerate (all posterior mass at one node) -> keep previous
            }
            let ai = (sxth[i] * s1[i] - sth[i] * sx[i]) / det;
            let di = (sthth[i] * sx[i] - sth[i] * sxth[i]) / det;
            let resid = (sxx[i] - ai * sxth[i] - di * sx[i]) / s1[i];
            a[i] = ai;
            d[i] = di;
            sigma[i] = resid.max(eps * eps).sqrt();
        }
        n_iter += 1;
    }

    // Reflection convention: make the average loading non-negative (theta -> -theta,
    // a -> -a leaves the model invariant), so recovery is comparable to a
    // positive-loading generating truth.
    if a.iter().sum::<f64>() < 0.0 {
        for ai in a.iter_mut() {
            *ai = -*ai;
        }
    }

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
        loglik_trace.push(final_ll);
    }

    let discrimination: Vec<f64> = (0..n_items).map(|i| a[i] / sigma[i]).collect();
    // Samejima difficulty b = -d/a is undefined for a non-discriminating item
    // (slope ~ 0); report NaN there rather than a misleading blow-up.
    let difficulty: Vec<f64> =
        (0..n_items).map(|i| if a[i].abs() > 1e-6 { -d[i] / a[i] } else { f64::NAN }).collect();

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
        n_parameters: 3 * n_items,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.f64().max(1e-12);
            let u2 = self.f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    /// Simulate CRM data: X = a*theta + d + sigma*eps, Z = logistic(X).
    #[allow(clippy::too_many_arguments)]
    fn simulate_crm(
        a: &[f64],
        d: &[f64],
        sigma: &[f64],
        n: usize,
        n_items: usize,
        skew: bool,
        rng: &mut Lcg,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut z = vec![0.0f64; n * n_items];
        let mut thetas = vec![0.0f64; n];
        for j in 0..n {
            let theta = if skew {
                let mut c = 0.0;
                for _ in 0..3 {
                    let g = rng.normal();
                    c += g * g;
                }
                (c - 3.0) / (6.0_f64).sqrt()
            } else {
                rng.normal()
            };
            thetas[j] = theta;
            for i in 0..n_items {
                let xij = a[i] * theta + d[i] + sigma[i] * rng.normal();
                z[j * n_items + i] = 1.0 / (1.0 + (-xij).exp());
            }
        }
        (z, thetas)
    }

    /// Unit test of the closed-form WLS + residual formula against a hand solve.
    #[test]
    fn crm_wls_matches_direct_solve() {
        // Three (theta, X) points with unit posterior weight -> ordinary least squares.
        let th = [-1.0f64, 0.0, 1.0];
        let xv = [0.2f64, 0.5, 1.4];
        let (mut s1, mut sth, mut sthth, mut sx, mut sxth, mut sxx) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for k in 0..3 {
            s1 += 1.0;
            sth += th[k];
            sthth += th[k] * th[k];
            sx += xv[k];
            sxth += xv[k] * th[k];
            sxx += xv[k] * xv[k];
        }
        let det = sthth * s1 - sth * sth;
        let a = (sxth * s1 - sth * sx) / det;
        let dd = (sthth * sx - sth * sxth) / det;
        // OLS slope = cov(theta,X)/var(theta); with theta mean 0: a = sxth/sthth
        assert!((a - sxth / sthth).abs() < 1e-12);
        // intercept = mean(X) - a*mean(theta) = mean(X) (theta mean 0)
        assert!((dd - sx / s1).abs() < 1e-12);
        let resid = (sxx - a * sxth - dd * sx) / s1;
        // residual = mean((X - a*theta - d)^2)
        let direct: f64 = (0..3).map(|k| (xv[k] - a * th[k] - dd).powi(2)).sum::<f64>() / 3.0;
        assert!((resid - direct).abs() < 1e-12, "{resid} vs {direct}");
    }

    /// Continuous responses are highly informative, so the model recovers the item
    /// parameters, the Samejima re-parameterization, and the trait well.
    #[test]
    fn crm_recovers_params() {
        let (n_items, n) = (15usize, 1500usize);
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.05 * i as f64).collect();
        let d_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.08 * i as f64).collect();
        let sigma_true: Vec<f64> = (0..n_items).map(|i| 0.6 + 0.02 * (i % 5) as f64).collect();
        let mut rng = Lcg(73);
        let (z, thetas) = simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, false, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_crm(&z, &observed, n, n_items, 41, 500, 1e-7).unwrap();
        assert!(res.converged);
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "loglik decreased {} -> {}", w[0], w[1]);
        }
        assert_eq!(res.n_parameters, 3 * n_items);
        assert!(rmse(&res.slope, &a_true) < 0.15, "a RMSE {}", rmse(&res.slope, &a_true));
        assert!(rmse(&res.intercept, &d_true) < 0.1, "d RMSE {}", rmse(&res.intercept, &d_true));
        assert!(rmse(&res.resid_sd, &sigma_true) < 0.1, "sigma RMSE {}", rmse(&res.resid_sd, &sigma_true));
        assert!(res.slope.iter().all(|&x| x > 0.0)); // reflection convention
        // Samejima re-parameterization recovers the generating discrimination/difficulty.
        let alpha_true: Vec<f64> = (0..n_items).map(|i| a_true[i] / sigma_true[i]).collect();
        let b_true: Vec<f64> = (0..n_items).map(|i| -d_true[i] / a_true[i]).collect();
        assert!(rmse(&res.discrimination, &alpha_true) < 0.3, "alpha RMSE");
        assert!(rmse(&res.difficulty, &b_true) < 0.2, "b RMSE");
        // trait recovery (continuous responses are information-rich)
        assert!(corr(&res.theta, &thetas) > 0.9, "theta corr {}", corr(&res.theta, &thetas));
    }

    #[test]
    fn crm_handles_missing_data() {
        let (n_items, n) = (8usize, 600usize);
        let a_true = vec![1.0f64; n_items];
        let d_true = vec![0.0f64; n_items];
        let sigma_true = vec![0.7f64; n_items];
        let mut rng = Lcg(9);
        let (z, _t) = simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, false, &mut rng);
        let mut observed = vec![true; n * n_items];
        for o in observed.iter_mut() {
            if rng.f64() < 0.2 {
                *o = false;
            }
        }
        let res = fit_crm(&z, &observed, n, n_items, 21, 400, 1e-6).unwrap();
        assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
        assert!(res.resid_sd.iter().all(|&s| s > 0.0));
    }

    #[test]
    fn crm_validate_rejects_malformed() {
        assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 3, 21, 10, 1e-6).is_err()); // wrong len
        assert!(fit_crm(&[0.5, 1.5], &[true, true], 1, 2, 21, 10, 1e-6).is_err()); // out of (0,1)
        assert!(fit_crm(&[0.5, 0.5], &[true, true], 1, 2, 99, 10, 1e-6).is_err()); // bad q
    }

    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_crm_recovery_500() {
        let (n_items, n, reps) = (15usize, 500usize, 500usize);
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.05 * i as f64).collect();
        let d_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.08 * i as f64).collect();
        let sigma_true: Vec<f64> = (0..n_items).map(|i| 0.6 + 0.02 * (i % 5) as f64).collect();
        for &skew in [false, true].iter() {
            let (mut ra, mut rd, mut rs, mut ba, mut nconv, mut tcorr) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize, 0.0f64);
            for rep in 0..reps {
                let mut rng = Lcg(
                    0x5DEECE66Du64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15),
                );
                let (z, thetas) =
                    simulate_crm(&a_true, &d_true, &sigma_true, n, n_items, skew, &mut rng);
                let observed = vec![true; n * n_items];
                let res = fit_crm(&z, &observed, n, n_items, 41, 500, 1e-6).unwrap();
                if res.converged {
                    nconv += 1;
                }
                ra += rmse(&res.slope, &a_true) / reps as f64;
                rd += rmse(&res.intercept, &d_true) / reps as f64;
                rs += rmse(&res.resid_sd, &sigma_true) / reps as f64;
                ba += (res.slope.iter().sum::<f64>() - a_true.iter().sum::<f64>())
                    / n_items as f64
                    / reps as f64;
                tcorr += corr(&res.theta, &thetas) / reps as f64;
            }
            println!(
                "[CRM MC skew={skew}] reps={reps} conv={:.2} RMSE(a)={:.3} RMSE(d)={:.3} \
                 RMSE(sigma)={:.3} bias(a)={:.3} theta-corr={:.3}",
                nconv as f64 / reps as f64,
                ra,
                rd,
                rs,
                ba,
                tcorr
            );
            assert!(ra < 0.15, "RMSE(a) {ra} skew={skew}");
            assert!(rd < 0.12, "RMSE(d) {rd} skew={skew}");
            assert!(rs < 0.1, "RMSE(sigma) {rs} skew={skew}");
            assert!(tcorr > 0.9, "theta corr {tcorr} skew={skew}");
        }
    }
}
