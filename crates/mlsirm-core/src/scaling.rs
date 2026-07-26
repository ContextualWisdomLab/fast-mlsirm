//! Thurstone Case V paired-comparison scaling.
//!
//! Implements Thurstone's (1927) law of comparative judgment under the
//! Case V assumptions (equal and uncorrelated discriminal dispersions)
//! exactly as implemented by the CRAN `psych` package's `thurstone()`
//! main routine.
//!
//! Source status:
//! - **READ**: `psych` (Revelle) `R/thurstone.R` (main routine) and
//!   `R/matrix.addition.R` (the `%+%` operator used to form the model
//!   matrix), fetched from CRAN. Every formula below is traceable to
//!   those lines.
//! - **NOT READ (as-cited)**: Thurstone, L. L. (1927). A law of
//!   comparative judgment. *Psychological Review, 34*, 273-286.
//!   https://doi.org/10.1037/h0070288 — paywalled at verification time.
//!   Attribution is therefore "Thurstone (1927), Case V, as implemented
//!   in psych's `thurstone()`" (same governance precedent as the
//!   Guttman (1945) module in `reliability.rs`).
//!
//! Algorithm (thurstone.R lines 25-28), with `choice[i][j]` = proportion
//! of judgments preferring column object `j` over row object `i`
//! (psych convention, `man/thurstone.Rd`):
//!
//! ```text
//! z_ij     = qnorm(choice_ij)
//! S_j      = mean_i(z_ij) - min_k mean_i(z_ik)   # column means, min scale = 0
//! model_ij = pnorm(S_j - S_i)
//! error    = model - choice
//! GF       = 1 - sum(error^2) / sum(choice^2)    # FULL matrix incl. diagonal
//! ```
//!
//! DERIVED (verified by reading `matrix.addition.R` lines 11-17): in
//! `pnorm(-scale.values %+% t(scale.values))` the `%+%` operator coerces
//! `-scale.values` to an n-by-1 column whose row sums are `-S_i` and
//! `t(scale.values)` is 1-by-n with column sums `S_j`, so the argument is
//! `S_j - S_i`.
//!
//! DOC/CODE CONFLICT (documented): `man/thurstone.Rd` describes a
//! "lower off diagonal" goodness of fit, but the code returns a single GF
//! over the FULL matrix with no `lower.tri` masking. This implementation
//! pins CODE behavior.
//!
//! DELIBERATE SAFETY DIVERGENCE: psych's direct-matrix path passes 0/1
//! proportions straight into `qnorm`, producing infinities; this
//! implementation rejects entries outside the open interval (0, 1)
//! instead (psych itself clamps to [1/(4n), 1-1/(4n)] in its upstream
//! `choice.mat()` converter, which is out of scope here along with the
//! rank-order/item converters).
//!
//! Numerics: reuses the shared approximate kernels
//! [`crate::nodes::inv_normal_cdf`] (Acklam, relative error < 1.15e-9)
//! and [`crate::fitstats::erfc`] (Numerical Recipes rational,
//! |error| < 1.2e-7). Outputs inherit that approximation error; the test
//! pins are verified to agree with a 50-digit oracle within 1e-6 on the
//! tested fixtures (no universal accuracy claim, especially for
//! proportions near 0 or 1).
//!
//! # References
//!
//! Thurstone, L. L. (1927). A law of comparative judgment. *Psychological
//! Review, 34*, 273-286. https://doi.org/10.1037/h0070288
//!
//! Revelle, W. (2024). *psych: Procedures for psychological, psychometric,
//! and personality research* (R package). Northwestern University.
//! https://CRAN.R-project.org/package=psych

use crate::fitstats::erfc;
use crate::nodes::inv_normal_cdf;

/// Result of a Thurstone Case V scaling.
#[derive(Clone, Debug)]
pub struct ThurstoneResult {
    /// Scale values, length `n`, shifted so the minimum is exactly 0
    /// (thurstone.R line 25).
    pub scale: Vec<f64>,
    /// Goodness of fit `1 - sum(residual^2)/sum(choice^2)` over the full
    /// matrix including the diagonal (thurstone.R line 28).
    pub gof: f64,
    /// Model choice matrix `pnorm(S_j - S_i)`, row-major `n * n`.
    pub model: Vec<f64>,
    /// Residual matrix `model - choice`, row-major `n * n`.
    pub residual: Vec<f64>,
}

fn norm_cdf(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

/// Thurstone Case V scaling of a square choice matrix.
///
/// `choice` is row-major `n * n`; `choice[i*n + j]` is the proportion of
/// judgments preferring column object `j` over row object `i`. Entries
/// must lie strictly in (0, 1); the diagonal is used as-is (psych does
/// not special-case it).
pub fn thurstone_case_v(choice: &[f64], n: usize) -> Result<ThurstoneResult, String> {
    if n < 2 {
        return Err("thurstone_case_v needs at least 2 objects".into());
    }
    if choice.len() != n * n {
        return Err(format!(
            "choice must be a row-major {n}x{n} matrix ({} entries), got {}",
            n * n,
            choice.len()
        ));
    }
    for &c in choice {
        if !c.is_finite() {
            return Err("choice proportions must be finite".into());
        }
        if c <= 0.0 || c >= 1.0 {
            return Err("choice proportions must lie strictly in (0, 1)".into());
        }
    }

    // Column means of z = qnorm(choice), shifted so min scale value is 0
    // (thurstone.R line 25).
    let mut colmeans = vec![0.0f64; n];
    for i in 0..n {
        for j in 0..n {
            colmeans[j] += inv_normal_cdf(choice[i * n + j]);
        }
    }
    for m in colmeans.iter_mut() {
        *m /= n as f64;
    }
    let min = colmeans.iter().cloned().fold(f64::INFINITY, f64::min);
    let scale: Vec<f64> = colmeans.iter().map(|&m| m - min).collect();

    // model_ij = pnorm(S_j - S_i) (thurstone.R line 26 via %+%, DERIVED
    // above); residual and full-matrix GF (lines 27-28).
    let mut model = vec![0.0f64; n * n];
    let mut residual = vec![0.0f64; n * n];
    let mut sse = 0.0f64;
    let mut ssc = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let m = norm_cdf(scale[j] - scale[i]);
            let e = m - choice[i * n + j];
            model[i * n + j] = m;
            residual[i * n + j] = e;
            sse += e * e;
            ssc += choice[i * n + j] * choice[i * n + j];
        }
    }
    let gof = 1.0 - sse / ssc;

    Ok(ThurstoneResult {
        scale,
        gof,
        model,
        residual,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/scaling_tests.rs"]
mod tests;
