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

/// Result of a Bradley-Terry MM fit.
#[derive(Clone, Debug)]
pub struct BradleyTerryResult {
    /// Centered log-worth parameters (mean exactly 0; choix
    /// `log_transform` convention).
    pub params: Vec<f64>,
    /// Exp-scale worths rescaled to sum `n` (mean 1; choix
    /// `exp_transform` convention).
    pub weights: Vec<f64>,
    /// Number of MM updates performed when convergence fired.
    pub iterations: usize,
}

/// Bradley-Terry maximum-likelihood (or Dirichlet-MAP) estimation via the
/// MM algorithm.
///
/// Implements the minorization-maximization iteration of Hunter (2004)
/// exactly as implemented by the `choix` Python package (v0.4.1,
/// `mm.py`/`utils.py`/`convergence.py`).
///
/// Source status:
/// - **READ**: choix 0.4.1 source (`_mm`, `_mm_pairwise`,
///   `exp_transform`, `log_transform`, `NormOfDifferenceTest`); every
///   formula below is traceable to those lines.
/// - **NOT READ (as-cited)**: Hunter, D. R. (2004). MM algorithms for
///   generalized Bradley-Terry models. *Annals of Statistics, 32*(1),
///   384-406. https://doi.org/10.1214/aos/1079120141 — download blocked
///   at verification time; cited as the algorithm origin per choix's
///   docstring. Bradley, R. A., & Terry, M. E. (1952). Rank analysis of
///   incomplete block designs: I. The method of paired comparisons.
///   *Biometrika, 39*(3/4), 324-345. https://doi.org/10.2307/2334029 —
///   unacquired; cited as the model origin.
///
/// Model: `P(i beats j) = w_i / (w_i + w_j)`. `wins` is row-major
/// `n * n`; `wins[i*n + j]` = number of comparisons in which object `i`
/// beat object `j` (diagonal must be 0). Counts need not be integers
/// (deliberate divergence from choix's `(winner, loser)` pair lists; the
/// update uses counts only as nonnegative multiplicative weights, and
/// `c` identical pairs contribute exactly `c/(w_i + w_j)` — DERIVED).
///
/// Per iteration (choix `_mm` + `_mm_pairwise`):
///
/// ```text
/// w        = exp_transform(params)          # exp(params - mean), sum = n
/// wins_i   = sum_j W[i][j]
/// denoms_i = sum_j (W[i][j] + W[j][i]) / (w_i + w_j)
/// params'  = log((wins + alpha)/(denoms + alpha)) - mean(...)
/// ```
///
/// Convergence (choix `NormOfDifferenceTest`, order 1): fires when the
/// L1 distance between successive centered parameter vectors is
/// `<= tol * n`; the first update never fires (no previous vector), so at
/// least 2 updates occur. Non-convergence within `max_iter` updates is an
/// error, as is a non-finite update (e.g. an item with zero wins at
/// `alpha = 0`, whose MLE worth is 0 and has no finite log-worth).
///
/// All-zero `wins` is rejected regardless of `alpha` — INTENTIONAL
/// divergence from choix, which for `alpha > 0` returns uniform zero
/// parameters on empty data; no-data inputs are rejected here as a
/// safety contract. No strong-connectivity (Ford, 1957 — NOT READ)
/// pre-check is done: an unbeatable item simply fails to converge, which
/// the tests demonstrate numerically.
pub fn bradley_terry_mm(
    wins: &[f64],
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<BradleyTerryResult, String> {
    if n < 2 {
        return Err("bradley_terry_mm needs at least 2 objects".into());
    }
    if wins.len() != n * n {
        return Err(format!(
            "wins must be a row-major {n}x{n} matrix ({} entries), got {}",
            n * n,
            wins.len()
        ));
    }
    for (k, &c) in wins.iter().enumerate() {
        if !c.is_finite() {
            return Err("win counts must be finite".into());
        }
        if c < 0.0 {
            return Err("win counts must be nonnegative".into());
        }
        if k / n == k % n && c != 0.0 {
            return Err("diagonal of the wins matrix must be zero".into());
        }
    }
    if wins.iter().all(|&c| c == 0.0) {
        return Err("wins matrix has no comparisons".into());
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err("alpha must be finite and nonnegative".into());
    }
    if !tol.is_finite() || tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if max_iter == 0 {
        return Err("max_iter must be at least 1".into());
    }

    let nf = n as f64;
    let exp_transform = |params: &[f64]| -> Vec<f64> {
        let mean = params.iter().sum::<f64>() / nf;
        let mut w: Vec<f64> = params.iter().map(|p| (p - mean).exp()).collect();
        let s: f64 = w.iter().sum();
        for x in w.iter_mut() {
            *x *= nf / s;
        }
        w
    };

    let mut params = vec![0.0f64; n];
    let mut prev: Option<Vec<f64>> = None;
    for it in 1..=max_iter {
        let w = exp_transform(&params);
        let mut win_totals = vec![0.0f64; n];
        let mut denoms = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                let c = wins[i * n + j];
                if c > 0.0 {
                    win_totals[i] += c;
                    let val = c / (w[i] + w[j]);
                    denoms[i] += val;
                    denoms[j] += val;
                }
            }
        }
        let mut newp: Vec<f64> = (0..n)
            .map(|k| ((win_totals[k] + alpha) / (denoms[k] + alpha)).ln())
            .collect();
        if newp.iter().any(|p| !p.is_finite()) {
            return Err(
                "MM update produced non-finite parameters (an item with zero wins \
                 has no finite log-worth when alpha = 0)"
                    .into(),
            );
        }
        let mean = newp.iter().sum::<f64>() / nf;
        for p in newp.iter_mut() {
            *p -= mean;
        }
        if let Some(pr) = &prev {
            let dist: f64 = newp.iter().zip(pr.iter()).map(|(a, b)| (a - b).abs()).sum();
            if dist <= tol * nf {
                let weights = exp_transform(&newp);
                return Ok(BradleyTerryResult {
                    params: newp,
                    weights,
                    iterations: it,
                });
            }
        }
        prev = Some(newp.clone());
        params = newp;
    }
    Err(format!(
        "bradley_terry_mm did not converge after {max_iter} iterations \
         (the comparison graph may violate the strong-connectivity condition)"
    ))
}

#[cfg(test)]
#[path = "../../../tests/unit/scaling_tests.rs"]
mod tests;
