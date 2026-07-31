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

/// Result of a Luce Spectral Ranking (LSR / I-LSR) fit.
#[derive(Clone, Debug)]
pub struct LsrResult {
    /// Centered log-worth parameters (mean exactly 0; choix
    /// `log_transform` convention).
    pub params: Vec<f64>,
    /// Stationary distribution of the LSR Markov chain, scaled to sum
    /// `n` (choix `statdist` convention).
    pub weights: Vec<f64>,
    /// Number of LSR passes performed (1 for the one-shot spectral
    /// estimator; the pass count at convergence for I-LSR).
    pub iterations: usize,
}

/// Shared input validation for [`lsr_pairwise`] / [`ilsr_pairwise`].
fn lsr_validate(wins: &[f64], n: usize, alpha: f64) -> Result<(), String> {
    if n < 2 {
        return Err("lsr_pairwise needs at least 2 objects".into());
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
    Ok(())
}

/// One LSR pass: build the Markov-chain generator from `wins` under the
/// current worths `w`, solve for its stationary distribution scaled to
/// sum `n`, and return `(params, weights)` (choix `lsr_pairwise_dense` +
/// `statdist` + `log_transform`).
fn lsr_pass(wins: &[f64], n: usize, alpha: f64, w: &[f64]) -> Result<(Vec<f64>, Vec<f64>), String> {
    let nf = n as f64;
    // chain = alpha * ones(n, n); chain[loser][winner] += c / (w_win + w_lose).
    // choix seeds the diagonal with alpha too; that is behaviorally
    // equivalent to seeding off-diagonals only, because the subsequent
    // row-sum subtraction cancels any initial diagonal (VERIFIED
    // algebraically; do not treat the variants as distinct).
    let mut chain = vec![alpha; n * n];
    for i in 0..n {
        for j in 0..n {
            let c = wins[i * n + j];
            if c > 0.0 {
                chain[j * n + i] += c / (w[i] + w[j]);
            }
        }
    }
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| chain[i * n + j]).sum();
        chain[i * n + i] = -row_sum;
    }
    // Overflow guard: finite inputs can still overflow the transition
    // rates or row sums (e.g. counts or alpha near 1e308).
    if chain.iter().any(|x| !x.is_finite()) {
        return Err("LSR Markov chain has non-finite transition rates (overflow)".into());
    }
    statdist_params(chain, n)
}

/// Stationary distribution of the generator `chain` (row-major `n * n`,
/// zero row sums, finite entries), scaled to sum `n`, plus its centered
/// logs (choix `statdist` + `log_transform`). Shared by [`lsr_pairwise`]
/// / [`ilsr_pairwise`] (via `lsr_pass`) and [`rank_centrality`].
fn statdist_params(mut chain: Vec<f64>, n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let nf = n as f64;
    // Normalize the generator to unit max magnitude: the stationary
    // distribution is invariant under global rescaling of all transition
    // rates, and without this an O(count) generator dwarfs the O(1)
    // sum-constraint row, so the relative pivot threshold below would
    // falsely reject validly connected matrices with huge counts
    // (impl-review finding; regression-tested for 1e20/1e150 scalings).
    let cmax = chain.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    if cmax > 0.0 {
        for x in chain.iter_mut() {
            *x /= cmax;
        }
    }

    // Stationary distribution: solve pi . chain = 0 with sum(pi) = n.
    // The n stationarity equations (columns of `chain`) sum to zero, so
    // any one is redundant; drop the last and append the sum constraint.
    // Gaussian elimination with partial pivoting (equivalent to choix's
    // LU-based `statdist` for a rank-(n-1) irreducible generator).
    let mut m = vec![0.0f64; n * n];
    let mut rhs = vec![0.0f64; n];
    for col in 0..n - 1 {
        for j in 0..n {
            m[col * n + j] = chain[j * n + col];
        }
    }
    for j in 0..n {
        m[(n - 1) * n + j] = 1.0;
    }
    rhs[n - 1] = nf;
    let scale = m.iter().fold(0.0f64, |a, &x| a.max(x.abs()));
    for col in 0..n {
        let (piv_row, piv_abs) =
            (col..n)
                .map(|r| (r, m[r * n + col].abs()))
                .fold(
                    (col, -1.0),
                    |best, cur| if cur.1 > best.1 { cur } else { best },
                );
        if piv_abs <= 1e-12 * scale {
            return Err(
                "stationary distribution could not be computed (comparison graph \
                 may be disconnected; consider alpha > 0)"
                    .into(),
            );
        }
        if piv_row != col {
            for j in 0..n {
                m.swap(col * n + j, piv_row * n + j);
            }
            rhs.swap(col, piv_row);
        }
        for r in col + 1..n {
            let f = m[r * n + col] / m[col * n + col];
            if f != 0.0 {
                for j in col..n {
                    m[r * n + j] -= f * m[col * n + j];
                }
                rhs[r] -= f * rhs[col];
            }
        }
    }
    let mut pi = vec![0.0f64; n];
    for col in (0..n).rev() {
        let mut v = rhs[col];
        for j in col + 1..n {
            v -= m[col * n + j] * pi[j];
        }
        pi[col] = v / m[col * n + col];
    }

    // Post-solve guards: a custom elimination must PROVE stationarity,
    // not just return something positive (spec-review mandate).
    if pi.iter().any(|x| !x.is_finite() || *x <= 0.0) {
        return Err(
            "stationary distribution could not be computed (non-positive or \
             non-finite stationary mass; comparison graph may be disconnected)"
                .into(),
        );
    }
    let pi_sum: f64 = pi.iter().sum();
    if (pi_sum - nf).abs() > 1e-8 * nf {
        return Err("stationary distribution failed the sum constraint".into());
    }
    for col in 0..n {
        let mut res = 0.0f64;
        let mut mag = 0.0f64;
        for j in 0..n {
            let term = pi[j] * chain[j * n + col];
            res += term;
            mag += term.abs();
        }
        if res.abs() > 1e-8 * (mag + 1.0) {
            return Err("stationary distribution failed the residual check".into());
        }
    }

    let logs: Vec<f64> = pi.iter().map(|x| x.ln()).collect();
    let mean = logs.iter().sum::<f64>() / nf;
    let params: Vec<f64> = logs.iter().map(|l| l - mean).collect();
    Ok((params, pi))
}

/// Luce Spectral Ranking: one-shot spectral estimate of Bradley-Terry
/// log-worths from a dense pairwise win-count matrix.
///
/// Implements the LSR algorithm exactly as implemented by the `choix`
/// Python package (v0.4.1, `lsr.py` `lsr_pairwise_dense` with uniform
/// initial weights, `utils.py` `statdist`/`log_transform`).
///
/// Source status:
/// - **READ**: choix 0.4.1 source (`lsr_pairwise_dense`,
///   `ilsr_pairwise_dense`, `_ilsr`, `statdist`, `log_transform`,
///   `exp_transform`, `NormOfDifferenceTest`); every formula is
///   traceable to those lines.
/// - **NOT READ (as-cited)**: Maystre, L., & Grossglauser, M. (2015).
///   Fast and accurate inference of Plackett-Luce models. *Advances in
///   Neural Information Processing Systems, 28*, 172-180 — cited as the
///   algorithm origin per choix's docstrings ([MG15]).
///
/// Model: `P(i beats j) = w_i / (w_i + w_j)`. `wins` is row-major
/// `n * n`; `wins[i*n + j]` = number of comparisons in which object `i`
/// beat object `j` (diagonal must be 0; counts need not be integers —
/// same DERIVED generalization as [`bradley_terry_mm`]). The Markov
/// chain accrues rate `c / (w_i + w_j)` on the loser→winner edge (plus
/// `alpha` everywhere as a regularizer); its stationary distribution,
/// scaled to sum `n`, is the worth estimate, and `params` are its
/// centered logs. The stationary solve uses Gaussian elimination with
/// partial pivoting plus positivity, sum, and residual guards, and
/// errors on disconnected comparison graphs at `alpha = 0` (choix raises
/// `ValueError` there too).
///
/// NOTE: `alpha` here regularizes the *chain rates*; it is NOT the same
/// regularization path as the Dirichlet-MAP `alpha` of
/// [`bradley_terry_mm`], and the two estimators disagree for
/// `alpha > 0` (both follow their sources; verified numerically). At
/// `alpha = 0` the I-LSR fixed point is the Bradley-Terry MLE, so
/// [`ilsr_pairwise`] and [`bradley_terry_mm`] agree there.
///
/// choix's `initial_params` is not exposed: the one-shot estimator
/// always starts from uniform weights, so `exp_transform` stability on
/// shifted inputs is unobservable through this API (documented per spec
/// review).
pub fn lsr_pairwise(wins: &[f64], n: usize, alpha: f64) -> Result<LsrResult, String> {
    lsr_validate(wins, n, alpha)?;
    let w = vec![1.0f64; n];
    let (params, weights) = lsr_pass(wins, n, alpha, &w)?;
    Ok(LsrResult {
        params,
        weights,
        iterations: 1,
    })
}

/// Iterative Luce Spectral Ranking: maximum-likelihood Bradley-Terry
/// estimation by repeated spectral passes (choix `ilsr_pairwise_dense`).
///
/// See [`lsr_pairwise`] for the model, source status, and per-pass
/// algebra. Each pass rebuilds the chain with the current worths
/// `w = exp_transform(params)` (`exp(params - mean)` scaled to sum `n`)
/// and re-solves. Convergence (choix `NormOfDifferenceTest`, order 1):
/// fires when the L1 distance between successive centered parameter
/// vectors is `<= tol * n`; the first pass never fires, so at least 2
/// passes occur. `iterations` is the pass count at convergence.
/// Non-convergence within `max_iter` passes is an error. At `alpha = 0`
/// the fixed point is the Bradley-Terry MLE (equals
/// [`bradley_terry_mm`]; verified to 4e-19 on the test fixtures).
pub fn ilsr_pairwise(
    wins: &[f64],
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<LsrResult, String> {
    lsr_validate(wins, n, alpha)?;
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
        let (newp, weights) = lsr_pass(wins, n, alpha, &w)?;
        if let Some(pr) = &prev {
            let dist: f64 = newp.iter().zip(pr.iter()).map(|(a, b)| (a - b).abs()).sum();
            if dist <= tol * nf {
                return Ok(LsrResult {
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
        "ilsr_pairwise did not converge after {max_iter} iterations"
    ))
}

/// Rank Centrality: one-shot spectral estimate of log-worths from the
/// *ratios* of pairwise wins.
///
/// Implements the algorithm exactly as implemented by the `choix`
/// Python package (v0.4.1, `lsr.py` `rank_centrality`, `utils.py`
/// `statdist`/`log_transform`).
///
/// Source status:
/// - **READ**: choix 0.4.1 source (`rank_centrality`, `statdist`,
///   `log_transform`); every formula is traceable to those lines.
/// - **NOT READ (as-cited)**: Negahban, S., Oh, S., & Shah, D. (2017).
///   Rank Centrality: Ranking from pairwise comparisons. *Operations
///   Research, 65*(1), 266-287 — cited as the algorithm origin per
///   choix's docstring ([NOS12]). The paper formulates a
///   *discrete-time* random walk with a max-degree normalization;
///   choix (and therefore this port) instead builds a continuous-time
///   generator and takes its stationary distribution, which shares the
///   stationary ranking but is NOT the paper's exact matrix. We
///   implement and verify the choix variant only.
///
/// The chain accrues, on the loser->winner edge, the *ratio*
/// `(alpha + c_win) / (2*alpha + c_win + c_lose)` rather than LSR's
/// worth-weighted count rate — the ratio denominator is a snapshot of
/// the pre-transform counts, exactly like choix's vectorized
/// `chain[idx] / (chain + chain.T)[idx]` (a half-updated in-place
/// denominator gives wrong results; regression-pinned). choix also
/// seeds the diagonal with `alpha` and ratio-transforms it to `1/2`
/// before subtracting the full row sum; the diagonal contribution
/// cancels algebraically in that subtraction (VERIFIED against the pip
/// package to 2e-16 at `alpha = 0.5`), so this port computes
/// off-diagonal ratios only and sets `diag = -offdiag_rowsum`.
///
/// At `alpha = 0` the result is exactly invariant under a global
/// rescaling of all counts (ratios are unchanged); for fixed
/// `alpha > 0` it is not.
///
/// DOCUMENTED DIVERGENCE from choix: an all-zero wins matrix is
/// rejected (shared [`lsr_pairwise`] validation) even when
/// `alpha > 0`, where choix would regularize it to a uniform chain —
/// all-zero data carries no ranking information. Overflowing
/// intermediates (`alpha + c` or the ratio denominator reaching
/// infinity, e.g. `alpha = 1e308`) are an explicit error, whereas
/// choix silently produces near-zero ratios and may fail later in
/// `statdist`.
pub fn rank_centrality(wins: &[f64], n: usize, alpha: f64) -> Result<LsrResult, String> {
    lsr_validate(wins, n, alpha)?;
    // Pre-transform counts snapshot: counts[j*n + i] = alpha + wins[i beats j].
    let mut counts = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                counts[j * n + i] = alpha + wins[i * n + j];
            }
        }
    }
    let mut chain = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                continue;
            }
            let c = counts[i * n + j];
            let denom = c + counts[j * n + i];
            if !c.is_finite() || !denom.is_finite() {
                return Err(
                    "rank_centrality counts overflow (alpha + count or ratio denominator \
                     is not finite)"
                        .into(),
                );
            }
            if c > 0.0 {
                chain[i * n + j] = c / denom;
            }
        }
    }
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| chain[i * n + j]).sum();
        chain[i * n + i] = -row_sum;
    }
    let (params, weights) = statdist_params(chain, n)?;
    Ok(LsrResult {
        params,
        weights,
        iterations: 1,
    })
}

/// Shared input validation for [`lsr_rankings`] / [`ilsr_rankings`].
///
/// CSR layout: ranking `r` is `rankings[starts[r]..starts[r + 1]]`
/// (best-first item indices).
fn rankings_validate(
    rankings: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
) -> Result<(), String> {
    if n < 2 {
        return Err("lsr_rankings needs at least 2 items".into());
    }
    // ponytail: dense O(n^2) chain; hard cap keeps a tiny input from
    // requesting a terabyte allocation and aborting the process. Raise
    // alongside a sparse chain if ever needed.
    if n > 10_000 {
        return Err(format!(
            "n = {n} exceeds the 10000-item cap for the dense O(n^2) chain"
        ));
    }
    if starts.len() < 2 {
        return Err("rankings data is empty (need at least one ranking)".into());
    }
    if starts[0] != 0 {
        return Err("starts[0] must be 0".into());
    }
    if *starts.last().unwrap() != rankings.len() {
        return Err("starts must end at rankings.len()".into());
    }
    let mut seen = vec![usize::MAX; n];
    for r in 0..starts.len() - 1 {
        let (a, b) = (starts[r], starts[r + 1]);
        if b < a {
            return Err("starts must be nondecreasing".into());
        }
        if b - a < 2 {
            return Err(
                "each ranking needs at least 2 items (choix silently no-ops \
                 shorter rankings; rejected here as a documented divergence)"
                    .into(),
            );
        }
        for &item in &rankings[a..b] {
            if item >= n {
                return Err(format!("item index {item} out of range for n = {n}"));
            }
            if seen[item] == r {
                return Err("duplicate item within a ranking (choix accepts duplicates \
                     when the chain stays connected; rejected here as a \
                     documented divergence)"
                    .into());
            }
            seen[item] = r;
        }
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err("alpha must be finite and nonnegative".into());
    }
    Ok(())
}

/// One LSR-rankings pass: build the Markov-chain generator from ranking
/// data under the current worths `w`, solve for its stationary
/// distribution, and return `(params, weights)` (choix `lsr_rankings` +
/// `statdist` + `log_transform`).
fn lsr_rankings_pass(
    rankings: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
    w: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // chain = alpha * ones(n, n); the alpha on the diagonal cancels in
    // the row-sum subtraction (same algebra as lsr_pass; VERIFIED).
    let mut chain = vec![alpha; n * n];
    for r in 0..starts.len() - 1 {
        let rk = &rankings[starts[r]..starts[r + 1]];
        // Ranked-SUBSET worth sum only (an all-items sum is a distinct,
        // wrong model on partial rankings; regression-pinned by the
        // partial-rankings fixture).
        let mut s: f64 = rk.iter().map(|&i| w[i]).sum();
        for i in 0..rk.len() - 1 {
            let winner = rk[i];
            // val is the rate for THIS placement, computed before the
            // placed winner's worth is removed from the running sum.
            let val = 1.0 / s;
            for &loser in &rk[i + 1..] {
                chain[loser * n + winner] += val;
            }
            s -= w[winner];
        }
    }
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| chain[i * n + j]).sum();
        chain[i * n + i] = -row_sum;
    }
    if chain.iter().any(|x| !x.is_finite()) {
        return Err("rankings Markov chain has non-finite transition rates (overflow)".into());
    }
    statdist_params(chain, n)
}

/// Luce Spectral Ranking for full or partial ranking data (one-shot):
/// Plackett-Luce log-worth estimation.
///
/// Implements the algorithm exactly as implemented by the `choix`
/// Python package (v0.4.1, `lsr.py` `lsr_rankings`, `utils.py`
/// `statdist`/`log_transform`).
///
/// Source status:
/// - **READ**: choix 0.4.1 source (`lsr_rankings`, `_init_lsr`,
///   `statdist`, `log_transform`); every formula is traceable to those
///   lines.
/// - **NOT READ (as-cited)**: Maystre, L., & Grossglauser, M. (2015).
///   Fast and accurate inference of Plackett-Luce models. *Advances in
///   Neural Information Processing Systems, 28*, 172-180 — cited as the
///   algorithm origin per choix's docstrings ([MG15]).
///
/// Model (Plackett-Luce): a ranking is a sequence of Luce choices —
/// the item placed at each position is chosen from the not-yet-placed
/// items with probability proportional to its worth. Each ranking
/// `r[0..k-1]` (best first, `k >= 2`) contributes, for every position
/// `i`, the rate `1 / (sum of current worths of r[i..k-1])` on each
/// loser->winner edge `r[j] -> r[i]`, `j > i` (plus `alpha` everywhere
/// as a regularizer). The stationary distribution of that chain, scaled
/// to sum `n`, is the worth estimate; `params` are its centered logs.
/// Errors on disconnected comparison graphs at `alpha = 0` (choix
/// raises there too) and on overflowing transition rates.
///
/// Data layout: CSR — ranking `r` is `rankings[starts[r]..starts[r+1]]`,
/// with `starts[0] == 0` and `starts.last() == rankings.len()`.
///
/// Duplicating the whole dataset is exactly invariant at `alpha = 0`
/// (the generator doubles; its stationary distribution is unchanged)
/// but NOT at `alpha > 0` (regression-pinned).
///
/// DOCUMENTED DIVERGENCES from choix: rankings shorter than 2 items are
/// rejected (choix silently no-ops them); duplicate items within one
/// ranking are rejected (choix accepts them when the chain stays
/// connected, silently corrupting the subset sum); negative indices
/// never reach this API (Python's silent negative-index wrapping is
/// rejected in the wrapper).
pub fn lsr_rankings(
    rankings: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
) -> Result<LsrResult, String> {
    rankings_validate(rankings, starts, n, alpha)?;
    let w = vec![1.0f64; n];
    let (params, weights) = lsr_rankings_pass(rankings, starts, n, alpha, &w)?;
    Ok(LsrResult {
        params,
        weights,
        iterations: 1,
    })
}

/// Iterative Luce Spectral Ranking for ranking data: maximum-likelihood
/// Plackett-Luce estimation by repeated spectral passes (choix
/// `ilsr_rankings`).
///
/// See [`lsr_rankings`] for the model, source status, data layout, and
/// per-pass algebra. Each pass rebuilds the chain with the current
/// worths `w = exp_transform(params)` (`exp(params - mean)` scaled to
/// sum `n`) and re-solves. Convergence (choix `NormOfDifferenceTest`,
/// order 1): fires when the L1 distance between successive centered
/// parameter vectors is `<= tol * n`; the first pass never fires, so at
/// least 2 passes occur. `iterations` is the pass count at convergence.
/// Non-convergence within `max_iter` passes is an error.
pub fn ilsr_rankings(
    rankings: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<LsrResult, String> {
    rankings_validate(rankings, starts, n, alpha)?;
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
        let (newp, weights) = lsr_rankings_pass(rankings, starts, n, alpha, &w)?;
        if let Some(pr) = &prev {
            let dist: f64 = newp.iter().zip(pr.iter()).map(|(a, b)| (a - b).abs()).sum();
            if dist <= tol * nf {
                return Ok(LsrResult {
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
        "ilsr_rankings did not converge after {max_iter} iterations"
    ))
}

/// Shared input validation for [`lsr_top1`] / [`ilsr_top1`].
///
/// CSR layout: observation `r` has winner `winners[r]` and losers
/// `losers[starts[r]..starts[r + 1]]` (the choice set is the winner
/// plus its losers).
fn top1_validate(
    winners: &[usize],
    losers: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
) -> Result<(), String> {
    if n < 2 {
        return Err("lsr_top1 needs at least 2 items".into());
    }
    // ponytail: dense O(n^2) chain; hard cap keeps a tiny input from
    // requesting a terabyte allocation and aborting the process. Raise
    // alongside a sparse chain if ever needed.
    if n > 10_000 {
        return Err(format!(
            "n = {n} exceeds the 10000-item cap for the dense O(n^2) chain"
        ));
    }
    if starts.len() < 2 {
        return Err("top-1 data is empty (need at least one observation)".into());
    }
    if winners.len() != starts.len() - 1 {
        return Err("winners.len() must equal starts.len() - 1".into());
    }
    if starts[0] != 0 {
        return Err("starts[0] must be 0".into());
    }
    if *starts.last().unwrap() != losers.len() {
        return Err("starts must end at losers.len()".into());
    }
    let mut seen = vec![usize::MAX; n];
    for r in 0..winners.len() {
        let (a, b) = (starts[r], starts[r + 1]);
        if b < a {
            return Err("starts must be nondecreasing".into());
        }
        if b == a {
            return Err("empty loser set (choix silently no-ops the observation; \
                 rejected here as a documented divergence)"
                .into());
        }
        let winner = winners[r];
        if winner >= n {
            return Err(format!("winner index {winner} out of range for n = {n}"));
        }
        for &loser in &losers[a..b] {
            if loser >= n {
                return Err(format!("loser index {loser} out of range for n = {n}"));
            }
            if loser == winner {
                return Err("winner appears in its own loser set (choix accepts \
                     this, silently inflating the denominator; rejected \
                     here as a documented divergence)"
                    .into());
            }
            if seen[loser] == r {
                return Err("duplicate loser within an observation (choix accepts \
                     duplicates, inflating the denominator and \
                     double-counting the edge; rejected here as a \
                     documented divergence)"
                    .into());
            }
            seen[loser] = r;
        }
    }
    if !alpha.is_finite() || alpha < 0.0 {
        return Err("alpha must be finite and nonnegative".into());
    }
    Ok(())
}

/// One LSR-top-1 pass: build the Markov-chain generator from top-1
/// choice data under the current worths `w`, solve for its stationary
/// distribution, and return `(params, weights)` (choix `lsr_top1` +
/// `statdist` + `log_transform`).
fn lsr_top1_pass(
    winners: &[usize],
    losers: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
    w: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    // chain = alpha * ones(n, n); the SEEDED DIAGONAL alpha cancels in
    // the row-sum subtraction, but the off-diagonal alpha remains as
    // regularization (same algebra as lsr_rankings; VERIFIED).
    let mut chain = vec![alpha; n * n];
    for r in 0..winners.len() {
        let winner = winners[r];
        let ls = &losers[starts[r]..starts[r + 1]];
        // Choice-SET worth sum: losers plus the winner (choix lsr.py:404;
        // a winner-excluded or all-items sum is a distinct, wrong model
        // on partial choice sets; regression-pinned by fixture TB).
        let s: f64 = ls.iter().map(|&i| w[i]).sum::<f64>() + w[winner];
        let val = 1.0 / s;
        for &loser in ls {
            chain[loser * n + winner] += val;
        }
    }
    for i in 0..n {
        let row_sum: f64 = (0..n).filter(|&j| j != i).map(|j| chain[i * n + j]).sum();
        chain[i * n + i] = -row_sum;
    }
    if chain.iter().any(|x| !x.is_finite()) {
        return Err("top-1 Markov chain has non-finite transition rates (overflow)".into());
    }
    statdist_params(chain, n)
}

/// Luce Spectral Ranking for top-1 choice data (one-shot):
/// Plackett-Luce / Luce-choice log-worth estimation.
///
/// Implements the algorithm exactly as implemented by the `choix`
/// Python package (v0.4.1, `lsr.py` `lsr_top1`, `utils.py`
/// `statdist`/`log_transform`).
///
/// Source status:
/// - **READ**: choix 0.4.1 source (`lsr_top1`, `_init_lsr`, `statdist`,
///   `log_transform`); every formula is traceable to those lines.
/// - **NOT READ (as-cited)**: Maystre, L., & Grossglauser, M. (2015).
///   Fast and accurate inference of Plackett-Luce models. *Advances in
///   Neural Information Processing Systems, 28*, 172-180 — cited as the
///   algorithm origin per choix's docstrings ([MG15]).
///
/// Model (Luce choice): each observation `(winner, losers)` records the
/// winner chosen out of the choice set `{winner} ∪ losers` with
/// probability proportional to its worth. Each observation contributes
/// the rate `1 / (sum of current worths over the choice set)` on every
/// loser->winner edge (plus `alpha` everywhere off-diagonal as a
/// regularizer). The stationary distribution of that chain, scaled to
/// sum `n`, is the worth estimate; `params` are its centered logs.
/// Errors on disconnected comparison graphs at `alpha = 0` (choix
/// raises there too) and on overflowing transition rates.
///
/// Data layout: CSR — observation `r` has winner `winners[r]` and
/// losers `losers[starts[r]..starts[r+1]]`, with `starts[0] == 0` and
/// `starts.last() == losers.len()`.
///
/// Duplicating the whole dataset is exactly invariant at `alpha = 0`
/// (the generator doubles; its stationary distribution is unchanged)
/// but NOT at `alpha > 0` (regression-pinned).
///
/// DOCUMENTED DIVERGENCES from choix: empty loser sets are rejected
/// (choix silently no-ops the observation); a winner appearing in its
/// own loser set is rejected (choix accepts it, adding a self-edge that
/// cancels in the diagonal while still inflating the denominator);
/// duplicate losers within one observation are rejected (choix accepts
/// them, inflating the denominator and double-counting the edge);
/// negative indices never reach this API (rejected in the wrapper).
pub fn lsr_top1(
    winners: &[usize],
    losers: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
) -> Result<LsrResult, String> {
    top1_validate(winners, losers, starts, n, alpha)?;
    let w = vec![1.0f64; n];
    let (params, weights) = lsr_top1_pass(winners, losers, starts, n, alpha, &w)?;
    Ok(LsrResult {
        params,
        weights,
        iterations: 1,
    })
}

/// Iterative Luce Spectral Ranking for top-1 choice data:
/// maximum-likelihood Luce-choice estimation by repeated spectral
/// passes (choix `ilsr_top1`).
///
/// See [`lsr_top1`] for the model, source status, data layout, and
/// per-pass algebra. Each pass rebuilds the chain with the current
/// worths `w = exp_transform(params)` (`exp(params - mean)` scaled to
/// sum `n`) and re-solves. Convergence (choix `NormOfDifferenceTest`,
/// order 1): fires when the L1 distance between successive centered
/// parameter vectors is `<= tol * n`; the first pass never fires, so at
/// least 2 passes occur. `iterations` is the pass count at convergence.
/// Non-convergence within `max_iter` passes is an error.
pub fn ilsr_top1(
    winners: &[usize],
    losers: &[usize],
    starts: &[usize],
    n: usize,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Result<LsrResult, String> {
    top1_validate(winners, losers, starts, n, alpha)?;
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
        let (newp, weights) = lsr_top1_pass(winners, losers, starts, n, alpha, &w)?;
        if let Some(pr) = &prev {
            let dist: f64 = newp.iter().zip(pr.iter()).map(|(a, b)| (a - b).abs()).sum();
            if dist <= tol * nf {
                return Ok(LsrResult {
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
        "ilsr_top1 did not converge after {max_iter} iterations"
    ))
}

// ---------------------------------------------------------------------
// Kendall & Babington Smith (1940) paired-comparison consistency and
// agreement, as implemented by CRAN eba 1.10-0 (Wickelmaier).
//
// SOURCE STATUS:
// - READ: eba `R/circular.R`, `R/kendall.u.R`, `man/circular.Rd`,
//   `man/kendall.u.Rd` (algorithm source of truth; formulas below cite
//   those files).
// - NOT READ (as-cited): Kendall, M. G., & Babington Smith, B. (1940).
//   On the method of paired comparisons. Biometrika, 31(3-4), 324-345.
//   https://doi.org/10.1093/biomet/31.3-4.324. Alway, G. G. (1962).
//   The distribution of the number of circular triads in paired
//   comparisons. Biometrika, 49, 265-269. Attribution is therefore
//   "Kendall & Babington Smith (1940), as implemented in eba" (same
//   governance precedent as psych's thurstone above).
//
// The exact circular-triad frequency tables for n = 2..10 are copied
// verbatim from eba `circular.R` `dcircular()`. Integrity invariant
// (regression-pinned): each row sums to 2^C(n,2), the number of
// orientations of the complete graph on n vertices.
// ---------------------------------------------------------------------

/// Exact frequency of T circular triads over all 2^C(n,2) tournaments,
/// for n = 2..=10 (index 0 <=> n = 2). Copied from eba `circular.R`.
const CIRCULAR_EXACT: [&[u64]; 9] = [
    &[2],
    &[6, 2],
    &[24, 16, 24],
    &[120, 120, 240, 240, 280, 24],
    &[720, 960, 2240, 2880, 6240, 3648, 8640, 4800, 2640],
    &[
        5040, 8400, 21840, 33600, 75600, 90384, 179760, 188160, 277200, 280560, 384048, 244160,
        233520, 72240, 2640,
    ],
    &[
        40320, 80640, 228480, 403200, 954240, 1304576, 3042816, 3870720, 6926080, 8332800,
        15821568, 14755328, 24487680, 24514560, 34762240, 29288448, 37188480, 24487680, 24312960,
        10402560, 3230080,
    ],
    &[
        362880, 846720, 2580480, 5093760, 12579840, 19958400, 44698752, 70785792, 130032000,
        190834560, 361525248, 443931264, 779950080, 1043763840, 1529101440, 1916619264, 2912257152,
        3078407808, 4506485760, 4946417280, 6068256768, 6160876416, 7730384256, 6292581120,
        6900969600, 5479802496, 4327787520, 2399241600, 1197020160, 163094400, 3230080,
    ],
    &[
        3628800,
        9676800,
        31449600,
        68275200,
        175392000,
        311592960,
        711728640,
        1193794560,
        2393475840,
        3784596480,
        7444104192,
        10526745600,
        19533696000,
        27610168320,
        47107169280,
        64016040960,
        107446832640,
        134470425600,
        218941470720,
        272302894080,
        417512148480,
        494080834560,
        743278970880,
        829743344640,
        1202317401600,
        1334577484800,
        1773862272000,
        1878824586240,
        2496636103680,
        2406981104640,
        3032021672960,
        2841072675840,
        3166378709760,
        2743311191040,
        2877794035200,
        2109852702720,
        1840136336640,
        1109253196800,
        689719564800,
        230683084800,
        48251508480,
    ],
];

/// Result of a circular-triads consistency analysis ([`circular_triads`]).
#[derive(Debug, Clone)]
pub struct CircularResult {
    /// Observed number of circular triads T (always integral).
    pub t: f64,
    /// Maximum possible number of circular triads for this n.
    pub t_max: f64,
    /// Expected number of circular triads under random choice, C(n,3)/4.
    pub t_exp: f64,
    /// Kendall's coefficient of consistence, zeta = 1 - T/T_max.
    pub zeta: f64,
    /// Chi-square statistic of the large-sample test (NaN on the exact
    /// path, i.e. for n <= 10).
    pub chi2: f64,
    /// Degrees of freedom of the chi-square approximation (NaN on the
    /// exact path).
    pub df: f64,
    /// P-value of the test (exact for n <= 10, chi-square otherwise).
    pub p_value: f64,
    /// Whether the exact distribution was used for the p-value.
    pub exact: bool,
}

fn parse_alternative(alternative: &str) -> Result<u8, String> {
    match alternative {
        "two.sided" => Ok(0),
        "less" => Ok(1),
        "greater" => Ok(2),
        _ => Err(format!(
            "alternative must be one of 'two.sided', 'less', 'greater'; got '{alternative}'"
        )),
    }
}

/// Circular triads (intransitive cycles) and Kendall's coefficient of
/// consistence for one complete round-robin of paired comparisons, as
/// implemented by eba's `circular()`.
///
/// `mat` is a flat row-major n*n 0/1 matrix; `mat[i*n + j] = 1` means
/// row stimulus `i` was chosen over column stimulus `j` (eba
/// convention, `man/circular.Rd`).
///
/// From `circular.R`:
/// - `T = n(n-1)(2n-1)/12 - (1/2) * sum_j colsum_j^2`, computed here in
///   the algebraically identical integer form
///   `T = C(n,3) - sum_j C(d_j, 2)` with `d_j` the column sums, which
///   proves T is always integral for valid tournaments (DERIVED:
///   `sum_j d_j = C(n,2)` for a complete tournament; expanding the
///   square recovers eba's expression exactly).
/// - `T_max = n(n^2-1)/24` (n odd) or `n(n^2-4)/24` (n even);
///   `T_exp = C(n,3)/4`; `zeta = 1 - T/T_max`.
/// - p-value for n <= 10: EXACT, from the `CIRCULAR_EXACT` tables.
///   `less` is the lower tail `P(T' <= T)`, `greater` the upper tail
///   `P(T' >= T)`, and `two.sided` adds to the own-tail mass `p1` the
///   atoms of the opposite tail whose running (far-end) cumulative sum
///   stays `<= p1`, capped at 1 (eba's `sum(rev(dc)[cumsum(rev(dc)) <=
///   p1])`; the break-loop below is equivalent because the cumulative
///   sum of nonnegative atoms is monotone). All tail masses here are
///   dyadic rationals k/2^C(n,2) with k < 2^53, so the returned p-value
///   is EXACT in f64.
/// - p-value for n >= 11: chi-square approximation with continuity
///   correction (when `correct`): `corr` = -1/2 for `less` (and
///   `two.sided` with T <= T_exp), +1/2 for `greater` (and `two.sided`
///   with T > T_exp); `df = n(n-1)(n-2)/(n-4)^2`;
///   `chi2 = 8/(n-4) * (T_exp - T + corr) + df`. TAIL INVERSION per
///   `circular.R`: the statistic grows as T falls below T_exp, so
///   `less` uses the UPPER chi-square tail, `greater` the lower, and
///   `two.sided` is `2 * min(upper, lower)`.
///
/// DOCUMENTED DIVERGENCES from eba: `n = 2` is rejected (eba returns
/// `zeta = 1 - 0/0 = NaN`); a nonzero diagonal is rejected (eba
/// silently zeroes it); incomplete or non-binary tournaments
/// (`mat[i,j] + mat[j,i] != 1` off-diagonal) are rejected (eba accepts
/// arbitrary matrices); there is no `exact` override or simulated
/// p-value — the exact path is used iff n <= 10.
pub fn circular_triads(
    mat: &[f64],
    n: usize,
    alternative: &str,
    correct: bool,
) -> Result<CircularResult, String> {
    let alt = parse_alternative(alternative)?;
    if n < 3 {
        return Err("circular_triads needs at least 3 objects (n = 2 has T_max = 0)".into());
    }
    if n > 10_000 {
        return Err(format!(
            "circular_triads: n = {n} exceeds the 10000-object cap"
        ));
    }
    if mat.len() != n * n {
        return Err(format!(
            "circular_triads: mat has {} entries, expected n*n = {}",
            mat.len(),
            n * n
        ));
    }
    for i in 0..n {
        if mat[i * n + i] != 0.0 {
            return Err(format!(
                "circular_triads: diagonal entry ({i},{i}) must be 0"
            ));
        }
        for j in (i + 1)..n {
            let a = mat[i * n + j];
            let b = mat[j * n + i];
            if !(a == 0.0 || a == 1.0) || !(b == 0.0 || b == 1.0) {
                return Err(format!(
                    "circular_triads: off-diagonal entries must be 0 or 1 (pair ({i},{j}))"
                ));
            }
            if a + b != 1.0 {
                return Err(format!(
                    "circular_triads: incomplete tournament, mat[{i},{j}] + mat[{j},{i}] != 1"
                ));
            }
        }
    }
    // Integer column sums d_j; T = C(n,3) - sum_j C(d_j, 2).
    let nu = n as u64;
    let mut sum_pairs: u64 = 0;
    for j in 0..n {
        let mut d: u64 = 0;
        for i in 0..n {
            if i != j && mat[i * n + j] == 1.0 {
                d += 1;
            }
        }
        sum_pairs += d * d.saturating_sub(1) / 2;
    }
    let c_n3 = nu * (nu - 1) * (nu - 2) / 6;
    let t_int = c_n3 - sum_pairs.min(c_n3);
    let t_max_int = if n % 2 == 1 {
        nu * (nu * nu - 1) / 24
    } else {
        nu * (nu * nu - 4) / 24
    };
    let t = t_int as f64;
    let t_max = t_max_int as f64;
    let t_exp = c_n3 as f64 / 4.0;
    let zeta = 1.0 - t / t_max;

    if n <= 10 {
        let freq = CIRCULAR_EXACT[n - 2];
        debug_assert!(t_int < freq.len() as u64);
        let ti = t_int as usize;
        let total: u64 = freq.iter().sum();
        let lower: u64 = freq[..=ti].iter().sum();
        let upper: u64 = freq[ti..].iter().sum();
        let k = match alt {
            1 => lower,
            2 => upper,
            _ => {
                // two.sided: own tail plus far-opposite-tail atoms whose
                // running cumulative stays <= p1 (integer arithmetic, so
                // the comparison against p1 is exact).
                let (p1, opposite): (u64, Box<dyn Iterator<Item = &u64>>) = if t <= t_exp {
                    (lower, Box::new(freq.iter().rev()))
                } else {
                    (upper, Box::new(freq.iter()))
                };
                let mut cum: u64 = 0;
                let mut extra: u64 = 0;
                for &a in opposite {
                    cum += a;
                    if cum <= p1 {
                        extra += a;
                    } else {
                        break;
                    }
                }
                (p1 + extra).min(total)
            }
        };
        return Ok(CircularResult {
            t,
            t_max,
            t_exp,
            zeta,
            chi2: f64::NAN,
            df: f64::NAN,
            p_value: k as f64 / total as f64,
            exact: true,
        });
    }

    let corr = if !correct {
        0.0
    } else {
        match alt {
            1 => -0.5,
            2 => 0.5,
            _ => {
                if t <= t_exp {
                    -0.5
                } else {
                    0.5
                }
            }
        }
    };
    let nf = n as f64;
    let df = nf * (nf - 1.0) * (nf - 2.0) / ((nf - 4.0) * (nf - 4.0));
    let chi2 = 8.0 / (nf - 4.0) * (t_exp - t + corr) + df;
    let sf = crate::fitstats::chi2_sf(chi2, df);
    let cdf = 1.0 - sf;
    let p_value = match alt {
        1 => sf,
        2 => cdf,
        _ => (2.0 * sf.min(cdf)).min(1.0),
    };
    Ok(CircularResult {
        t,
        t_max,
        t_exp,
        zeta,
        chi2,
        df,
        p_value,
        exact: false,
    })
}

/// Result of Kendall's coefficient of agreement ([`kendall_u`]).
#[derive(Debug, Clone)]
pub struct KendallUResult {
    /// Sum over ordered pairs of C(M_ij, 2), the number of agreeing
    /// judge pairs.
    pub sigma: f64,
    /// Kendall's u; 1 is maximum agreement.
    pub u: f64,
    /// Minimum attainable u: -1/m (m odd) or -1/(m-1) (m even).
    pub min_u: f64,
    /// Chi-square statistic for the test that agreement is by chance.
    /// Returned RAW — it can be negative under strong disagreement with
    /// continuity correction; only the p-value computation clamps at 0
    /// (matching R's `pchisq(negative, ..., lower.tail=FALSE) = 1`).
    pub chi2: f64,
    /// Degrees of freedom, C(n,2) * m(m-1) / (m-2)^2.
    pub df: f64,
    /// Upper-tail p-value.
    pub p_value: f64,
}

/// Kendall's coefficient of agreement u between m judges over n
/// objects, as implemented by eba's `kendall.u()`.
///
/// `mat` is a flat row-major n*n frequency matrix; `mat[i*n + j]` is
/// the number of judges preferring object `i` over object `j`
/// (`man/kendall.u.Rd` convention). From `kendall.u.R`:
/// - `Sigma = sum_{i != j} C(M_ij, 2)`
/// - `u = 2*Sigma / (C(m,2) * C(n,2)) - 1` with `m` the number of
///   judges per pair
/// - `min_u = -1/m` (m odd) or `-1/(m-1)` (m even)
/// - `chi2 = 4/(m-2) * (Sigma - corr - C(n,2)/2 * C(m,2) * (m-3)/(m-2))`
///   with the continuity correction `corr = 1` when `correct`
/// - `df = C(n,2) * m(m-1) / (m-2)^2`; p-value is the upper chi-square
///   tail (non-integer df supported by [`crate::fitstats::chi2_sf`]).
///
/// DOCUMENTED DIVERGENCES from eba: eba derives `m` from the FIRST
/// pair only (`M[1,2] + M[2,1]`) and never checks its stated
/// equal-observations assumption — here every pair must satisfy
/// `M_ij + M_ji == m` or the input is rejected. `m >= 3` is required
/// (the eba statistic divides by `m - 2`), entries must be
/// nonnegative integers, `n >= 2`, and the diagonal must be zero.
///
/// # ponytail: judge count capped at 1_000_000 so all binomial
/// coefficients stay exactly representable in u64/f64; raise the cap
/// with u128 arithmetic if a use case ever needs more judges.
pub fn kendall_u(mat: &[f64], n: usize, correct: bool) -> Result<KendallUResult, String> {
    if n < 2 {
        return Err("kendall_u needs at least 2 objects".into());
    }
    if n > 10_000 {
        return Err(format!("kendall_u: n = {n} exceeds the 10000-object cap"));
    }
    if mat.len() != n * n {
        return Err(format!(
            "kendall_u: mat has {} entries, expected n*n = {}",
            mat.len(),
            n * n
        ));
    }
    let mut m_int: [Option<u64>; 1] = [None];
    let mut counts = vec![0u64; n * n];
    for i in 0..n {
        if mat[i * n + i] != 0.0 {
            return Err(format!("kendall_u: diagonal entry ({i},{i}) must be 0"));
        }
        for j in 0..n {
            if i == j {
                continue;
            }
            let x = mat[i * n + j];
            if !x.is_finite() || x < 0.0 || x.fract() != 0.0 || x > 1_000_000.0 {
                return Err(format!(
                    "kendall_u: entries must be nonnegative integers <= 1000000 (entry ({i},{j}))"
                ));
            }
            counts[i * n + j] = x as u64;
        }
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let pair = counts[i * n + j] + counts[j * n + i];
            match m_int[0] {
                None => m_int[0] = Some(pair),
                Some(m) if m != pair => {
                    return Err(format!(
                        "kendall_u: unequal observations per pair (pair ({i},{j}) has {pair}, expected {m})"
                    ));
                }
                _ => {}
            }
        }
    }
    let m = m_int[0].expect("n >= 2 guarantees at least one pair");
    if m < 3 {
        return Err(format!(
            "kendall_u: needs at least 3 judges per pair, got {m}"
        ));
    }
    let sigma: u64 = counts.iter().map(|&c| c * c.saturating_sub(1) / 2).sum();
    let nu = n as u64;
    let c_n2 = nu * (nu - 1) / 2;
    let c_m2 = m * (m - 1) / 2;
    let mf = m as f64;
    let u = 2.0 * sigma as f64 / (c_m2 as f64 * c_n2 as f64) - 1.0;
    let min_u = if m % 2 == 1 {
        -1.0 / mf
    } else {
        -1.0 / (mf - 1.0)
    };
    let corr = if correct { 1.0 } else { 0.0 };
    let chi2 = 4.0 / (mf - 2.0)
        * (sigma as f64 - corr - c_n2 as f64 / 2.0 * c_m2 as f64 * (mf - 3.0) / (mf - 2.0));
    let df = c_n2 as f64 * mf * (mf - 1.0) / ((mf - 2.0) * (mf - 2.0));
    let p_value = crate::fitstats::chi2_sf(chi2, df);
    Ok(KendallUResult {
        sigma: sigma as f64,
        u,
        min_u,
        chi2,
        df,
        p_value,
    })
}

// ---------------------------------------------------------------------------
// Elo rating system (batch-per-period update).
//
// Citation governance:
// - READ (implementation source of record): Stephenson, A., & Sonas, J.
//   (2020). PlayerRatings: Dynamic updating methods for player ratings
//   estimation (Version 1.1-0) [R package]. CRAN. `R/ratings.R` `elo()`
//   (lines 1-123: validation, per-period splitting, W/D/L and lag
//   bookkeeping) and `src/ratings.c` `elo_c` (expected-score kernel with
//   per-game advantage `gamma`). Both files were read in full and this
//   implementation was verified against an independently executed
//   exact-rational oracle.
// - NOT READ (cited as origin of the method as described by the
//   PlayerRatings documentation): Elo, A. E. (1978). The rating of
//   chessplayers, past and present. Arco.
//
// Model (derived from the read C kernel): within a rating period, every
// expected score uses the ratings at the START of the period:
//   E_w = 1 / (1 + 10^((r_b - r_w - gamma)/400))
//   E_b = 1 / (1 + 10^((r_w - r_b + gamma)/400))
//   dscore[w] += s - E_w;  dscore[b] += (1 - s) - E_b
// and after all games of the period, r += kfac * dscore (batch update).
// PROVED (hand derivation, confirmed by the spec-verify review): the two
// exponents are exact negations, so E_w + E_b = 1 identically for any
// finite gamma; consequently sum(dscore) = 0 each period and
// sum(ratings) = n * init is conserved for ANY gamma. A refactor
// E_b = 1 - E_w is therefore behaviorally unobservable (documented
// unkillable mutant).
//
// Bookkeeping (R lines 93-108): games/wins/draws/losses count per
// appearance with W/D/L only for scores exactly 1 / 0.5 / 0 (other
// fractional scores count a game but no W/D/L). Lag: after each period,
// every player with cumulative games != 0 (including this period's) gets
// lag += 1, then every player appearing this period resets to lag = 0.
//
// Documented divergences from PlayerRatings `elo()`:
// - `white == black` (self-play) is rejected (R silently permits it).
// - Scalar `kfac` only (function kfac out of scope); any finite value
//   including 0 and negatives is accepted, matching R's lack of a check.
// - No `status` carry-in / `history`: all players start at `init`, lag 0;
//   players in 0..n that never appear keep games = 0, lag = 0.
// - Periods are u64 labels; unsorted input is grouped by ascending period
//   value with original row order preserved within a period (matching R's
//   split() factor-level ordering).
// - Extreme rating differences saturate the expectation to 0/1 without
//   panicking (10^x overflows to +inf in f64); an extreme `kfac` may
//   produce non-finite output ratings and is not treated as an error.
// ---------------------------------------------------------------------------

/// Result of [`elo_rating`]: per-player ratings and bookkeeping counts.
#[derive(Debug, Clone)]
pub struct EloResult {
    /// Post-update rating per player (length `n`).
    pub ratings: Vec<f64>,
    /// Number of game appearances per player.
    pub games: Vec<u64>,
    /// Wins (score exactly 1 as white, or exactly 0 as black).
    pub wins: Vec<u64>,
    /// Draws (score exactly 0.5, both sides).
    pub draws: Vec<u64>,
    /// Losses (score exactly 0 as white, or exactly 1 as black).
    pub losses: Vec<u64>,
    /// Rating periods since the player's last appearance (0 if the player
    /// appeared in the final period or never played).
    pub lag: Vec<u64>,
}

/// Elo ratings from a game schedule, PlayerRatings `elo()` semantics.
///
/// `periods[k]`, `white[k]`, `black[k]`, `score[k]`, `gamma[k]` describe
/// game `k`: rating-period label, player indices in `0..n`, white's score
/// in `[0, 1]`, and white's per-game advantage. Games are grouped by
/// ascending period value; within a period all expectations use the
/// period-start ratings (batch update). All players start at `init`.
pub fn elo_rating(
    periods: &[u64],
    white: &[usize],
    black: &[usize],
    score: &[f64],
    gamma: &[f64],
    n: usize,
    init: f64,
    kfac: f64,
) -> Result<EloResult, String> {
    let g = periods.len();
    if g == 0 {
        return Err("elo_rating: at least one game is required".to_string());
    }
    if white.len() != g || black.len() != g || score.len() != g || gamma.len() != g {
        return Err(format!(
            "elo_rating: length mismatch (periods {}, white {}, black {}, score {}, gamma {})",
            g,
            white.len(),
            black.len(),
            score.len(),
            gamma.len()
        ));
    }
    if n < 2 {
        return Err("elo_rating: at least two players are required".to_string());
    }
    if n > 10_000 {
        return Err(format!(
            "elo_rating: n = {} exceeds the supported cap of 10000",
            n
        ));
    }
    if !init.is_finite() {
        return Err("elo_rating: init must be finite".to_string());
    }
    if !kfac.is_finite() {
        return Err("elo_rating: kfac must be finite".to_string());
    }
    for k in 0..g {
        if white[k] >= n || black[k] >= n {
            return Err(format!(
                "elo_rating: game {} has player index out of range (white {}, black {}, n {})",
                k, white[k], black[k], n
            ));
        }
        if white[k] == black[k] {
            return Err(format!(
                "elo_rating: game {} has white == black == {} (self-play is not supported)",
                k, white[k]
            ));
        }
        if !score[k].is_finite() || !(0.0..=1.0).contains(&score[k]) {
            return Err(format!(
                "elo_rating: game {} has score {} outside [0, 1]",
                k, score[k]
            ));
        }
        if !gamma[k].is_finite() {
            return Err(format!("elo_rating: game {} has non-finite gamma", k));
        }
    }

    // Group by ascending period, preserving row order within a period
    // (R split() orders groups by factor level, i.e. ascending label).
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by_key(|&k| periods[k]); // stable sort keeps row order

    let mut ratings = vec![init; n];
    let mut games = vec![0u64; n];
    let mut wins = vec![0u64; n];
    let mut draws = vec![0u64; n];
    let mut losses = vec![0u64; n];
    let mut lag = vec![0u64; n];
    let mut appeared = vec![false; n];

    let mut i = 0usize;
    while i < g {
        let period = periods[order[i]];
        let mut j = i;
        while j < g && periods[order[j]] == period {
            j += 1;
        }

        let mut dscore = vec![0.0f64; n];
        for player in appeared.iter_mut() {
            *player = false;
        }
        for &k in &order[i..j] {
            let (w, b, s, gam) = (white[k], black[k], score[k], gamma[k]);
            // Expectations from period-START ratings (batch semantics).
            let e_w = 1.0 / (1.0 + 10f64.powf((ratings[b] - ratings[w] - gam) / 400.0));
            let e_b = 1.0 / (1.0 + 10f64.powf((ratings[w] - ratings[b] + gam) / 400.0));
            dscore[w] += s - e_w;
            dscore[b] += (1.0 - s) - e_b;
            games[w] += 1;
            games[b] += 1;
            if s == 1.0 {
                wins[w] += 1;
                losses[b] += 1;
            } else if s == 0.5 {
                draws[w] += 1;
                draws[b] += 1;
            } else if s == 0.0 {
                losses[w] += 1;
                wins[b] += 1;
            }
            appeared[w] = true;
            appeared[b] = true;
        }
        for p in 0..n {
            ratings[p] += kfac * dscore[p];
        }
        // Lag: games are already updated, so first-time players increment
        // to 1 and are immediately reset to 0 (R lines 97-103 order).
        for p in 0..n {
            if games[p] != 0 {
                lag[p] += 1;
            }
        }
        for p in 0..n {
            if appeared[p] {
                lag[p] = 0;
            }
        }
        i = j;
    }

    Ok(EloResult {
        ratings,
        games,
        wins,
        draws,
        losses,
        lag,
    })
}

// ---------------------------------------------------------------------------
// Glicko rating system (batch-per-period update with rating deviations).
//
// Citation governance:
// - READ (formula source): Glickman, M. E. (n.d.). The Glicko system
//   [Technical note]. Harvard University.
//   http://www.glicko.net/glicko/glicko.pdf. Defines Step 1b deviation
//   inflation RD = min(sqrt(RD_old^2 + c^2), 350) and the Step 2 update
//   r' = r + q/(1/RD^2 + 1/d^2) * sum_j g(RD_j)(s_j - E_j),
//   RD' = sqrt((1/RD^2 + 1/d^2)^-1), with q = ln(10)/400,
//   g(RD) = 1/sqrt(1 + 3 q^2 RD^2 / pi^2),
//   E_j = 1/(1 + 10^(-g(RD_j)(r - r_j)/400)),
//   d^2 = (q^2 sum_j g(RD_j)^2 E_j (1 - E_j))^-1,
//   and the worked example (r = 1500, RD = 200 vs opponents (1400, 30),
//   (1550, 100), (1700, 300) scoring 1, 0, 0 -> r' = 1464, RD' = 151.4)
//   which is pinned by test `gk_paper_anchor_ga`.
// - READ (implementation source of record): Stephenson, A., & Sonas, J.
//   (2020). PlayerRatings: Dynamic updating methods for player ratings
//   estimation (Version 1.1-0) [R package]. CRAN. `R/ratings.R` `glicko()`
//   (lines 273-410: per-period splitting, participant-only variance
//   inflation with (lag+1)*cval^2 and rdmax^2 clamp, W/D/L and lag
//   bookkeeping) and `src/ratings.c` `glicko_c` (lines 83-119: per-game
//   accumulation kernel). Both read in full; this implementation was
//   verified against an independently executed oracle whose GA fixture
//   reproduces Glickman's worked example.
// - NOT READ (cited as the method's derivation paper by both read
//   sources): Glickman, M. E. (1999). Parameter estimation in large
//   dynamic paired comparison experiments. Applied Statistics, 48(3),
//   377-394.
//
// Model (derived from the read R/C sources; per rating period, ascending
// period label, batch semantics):
//   1. participants' variances inflate FIRST:
//        v_p = min(v_p + (lag_p + 1) * cval^2, rdmax^2)
//   2. g_k = 1/sqrt(1 + 3 (q/pi)^2 v_k) for ALL players (post-inflation)
//   3. per game (w, b, s, gamma), using period-START ratings:
//        E_w = 1/(1 + 10^(g_b (r_b - r_w - gamma)/400))
//        dval_w += q^2 g_b^2 E_w (1 - E_w); dscore_w += g_b (s - E_w)
//        E_b = 1/(1 + 10^(g_w (r_w - r_b + gamma)/400))
//        dval_b += q^2 g_w^2 E_b (1 - E_b); dscore_b += g_w (1 - s - E_b)
//   4. all players: v <- 1/(1/v + dval), THEN r <- r + v_new * q * dscore
//      (the NEW variance is the paper's q/(1/RD^2 + 1/d^2) factor).
// Bookkeeping identical to `elo_rating`: W/D/L only for scores exactly
// 1 / 0.5 / 0; lag += 1 where cumulative games != 0, then reset to 0 for
// this period's participants.
//
// NOT an invariant (verified by oracle fixture GB): the rating sum is NOT
// conserved in general — updates are weighted by per-player new variance
// and opponent g, which is asymmetric. (A symmetric equal-deviation
// two-player game does conserve; tests must not rely on conservation.)
//
// Documented divergences from PlayerRatings `glicko()`:
// - `white == black` (self-play) is rejected (R silently permits it).
// - No `status` carry-in / `history`: per-player `init_rating`/`init_dev`
//   arrays replace R's status frame; entries are returned for ALL players
//   in 0..n (R's fresh start only creates observed players), and players
//   that never appear keep their init rating/deviation, zero tallies,
//   lag 0.
// - Periods are u64 labels; unsorted input is grouped by ascending period
//   value with row order preserved within a period (R split() ordering).
// ---------------------------------------------------------------------------

/// Result of [`glicko_rating`]: per-player ratings, deviations, and counts.
#[derive(Debug, Clone)]
pub struct GlickoResult {
    /// Post-update rating per player (length `n`).
    pub ratings: Vec<f64>,
    /// Post-update rating deviation (RD) per player.
    pub deviations: Vec<f64>,
    /// Number of game appearances per player.
    pub games: Vec<u64>,
    /// Wins (score exactly 1 as white, or exactly 0 as black).
    pub wins: Vec<u64>,
    /// Draws (score exactly 0.5, both sides).
    pub draws: Vec<u64>,
    /// Losses (score exactly 0 as white, or exactly 1 as black).
    pub losses: Vec<u64>,
    /// Rating periods since the player's last appearance (0 if the player
    /// appeared in the final period or never played).
    pub lag: Vec<u64>,
}

/// Glicko ratings from a game schedule, PlayerRatings `glicko()` semantics.
///
/// `periods[k]`, `white[k]`, `black[k]`, `score[k]`, `gamma[k]` describe
/// game `k`: rating-period label, player indices in `0..n`, white's score
/// in `[0, 1]`, and white's per-game advantage. `init_rating`/`init_dev`
/// give each player's starting rating and deviation (`n` = their length);
/// `cval` is the per-period uncertainty growth constant and `rdmax` the
/// deviation ceiling (Glickman's Step 1b).
pub fn glicko_rating(
    periods: &[u64],
    white: &[usize],
    black: &[usize],
    score: &[f64],
    gamma: &[f64],
    init_rating: &[f64],
    init_dev: &[f64],
    cval: f64,
    rdmax: f64,
) -> Result<GlickoResult, String> {
    let g = periods.len();
    if g == 0 {
        return Err("glicko_rating: at least one game is required".to_string());
    }
    if white.len() != g || black.len() != g || score.len() != g || gamma.len() != g {
        return Err(format!(
            "glicko_rating: length mismatch (periods {}, white {}, black {}, score {}, gamma {})",
            g,
            white.len(),
            black.len(),
            score.len(),
            gamma.len()
        ));
    }
    let n = init_rating.len();
    if n < 2 {
        return Err("glicko_rating: at least two players are required".to_string());
    }
    if n > 10_000 {
        return Err(format!(
            "glicko_rating: n = {} exceeds the supported cap of 10000",
            n
        ));
    }
    if init_dev.len() != n {
        return Err(format!(
            "glicko_rating: init_rating (len {}) and init_dev (len {}) must match",
            n,
            init_dev.len()
        ));
    }
    if !rdmax.is_finite() || rdmax <= 0.0 {
        return Err(format!(
            "glicko_rating: rdmax {} must be finite and > 0",
            rdmax
        ));
    }
    if !cval.is_finite() || cval < 0.0 {
        return Err(format!(
            "glicko_rating: cval {} must be finite and >= 0",
            cval
        ));
    }
    for p in 0..n {
        if !init_rating[p].is_finite() {
            return Err(format!("glicko_rating: init_rating[{}] is not finite", p));
        }
        if !init_dev[p].is_finite() || init_dev[p] <= 0.0 {
            return Err(format!(
                "glicko_rating: init_dev[{}] = {} must be finite and > 0",
                p, init_dev[p]
            ));
        }
        if init_dev[p] > rdmax {
            return Err(format!(
                "glicko_rating: init_dev[{}] = {} exceeds rdmax {}",
                p, init_dev[p], rdmax
            ));
        }
    }
    for k in 0..g {
        if white[k] >= n || black[k] >= n {
            return Err(format!(
                "glicko_rating: game {} has player index out of range (white {}, black {}, n {})",
                k, white[k], black[k], n
            ));
        }
        if white[k] == black[k] {
            return Err(format!(
                "glicko_rating: game {} has white == black == {} (self-play is not supported)",
                k, white[k]
            ));
        }
        if !score[k].is_finite() || !(0.0..=1.0).contains(&score[k]) {
            return Err(format!(
                "glicko_rating: game {} has score {} outside [0, 1]",
                k, score[k]
            ));
        }
        if !gamma[k].is_finite() {
            return Err(format!("glicko_rating: game {} has non-finite gamma", k));
        }
    }

    // q = ln(10)/400 (paper); qip3 = 3 (q/pi)^2 (R line 355).
    let qv = std::f64::consts::LN_10 / 400.0;
    let qip3 = 3.0 * (qv / std::f64::consts::PI).powi(2);

    // Group by ascending period, preserving row order within a period.
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by_key(|&k| periods[k]); // stable sort keeps row order

    let mut ratings: Vec<f64> = init_rating.to_vec();
    // Internal state is the VARIANCE (deviation squared), as in R.
    let mut vars: Vec<f64> = init_dev.iter().map(|d| d * d).collect();
    let mut games = vec![0u64; n];
    let mut wins = vec![0u64; n];
    let mut draws = vec![0u64; n];
    let mut losses = vec![0u64; n];
    let mut lag = vec![0u64; n];
    let mut appeared = vec![false; n];
    let mut gdevs = vec![0.0f64; n];

    let mut i = 0usize;
    while i < g {
        let period = periods[order[i]];
        let mut j = i;
        while j < g && periods[order[j]] == period {
            j += 1;
        }

        for player in appeared.iter_mut() {
            *player = false;
        }
        for &k in &order[i..j] {
            appeared[white[k]] = true;
            appeared[black[k]] = true;
        }
        // Step 1b (participants only): inflate variance, clamp at rdmax^2.
        for p in 0..n {
            if appeared[p] {
                vars[p] = (vars[p] + (lag[p] + 1) as f64 * cval * cval).min(rdmax * rdmax);
            }
        }
        // g(RD) for ALL players from post-inflation variances (R line 370).
        for p in 0..n {
            gdevs[p] = 1.0 / (1.0 + qip3 * vars[p]).sqrt();
        }

        let mut dscore = vec![0.0f64; n];
        let mut dval = vec![0.0f64; n];
        for &k in &order[i..j] {
            let (w, b, s, gam) = (white[k], black[k], score[k], gamma[k]);
            // Expectations from period-START ratings (batch semantics),
            // opponent's g in the exponent (C lines 107, 113).
            let e_w = 1.0 / (1.0 + 10f64.powf(gdevs[b] * (ratings[b] - ratings[w] - gam) / 400.0));
            dval[w] += qv * qv * gdevs[b] * gdevs[b] * e_w * (1.0 - e_w);
            dscore[w] += gdevs[b] * (s - e_w);
            let e_b = 1.0 / (1.0 + 10f64.powf(gdevs[w] * (ratings[w] - ratings[b] + gam) / 400.0));
            dval[b] += qv * qv * gdevs[w] * gdevs[w] * e_b * (1.0 - e_b);
            dscore[b] += gdevs[w] * (1.0 - s - e_b);

            games[w] += 1;
            games[b] += 1;
            if s == 1.0 {
                wins[w] += 1;
                losses[b] += 1;
            } else if s == 0.5 {
                draws[w] += 1;
                draws[b] += 1;
            } else if s == 0.0 {
                losses[w] += 1;
                wins[b] += 1;
            }
        }
        // Variance first, THEN rating with the NEW variance (R lines
        // 377-378; the paper's q/(1/RD^2 + 1/d^2) factor). Non-participants
        // have dval = dscore = 0 and are unchanged.
        for p in 0..n {
            vars[p] = 1.0 / (1.0 / vars[p] + dval[p]);
            ratings[p] += vars[p] * qv * dscore[p];
        }
        for p in 0..n {
            if games[p] != 0 {
                lag[p] += 1;
            }
        }
        for p in 0..n {
            if appeared[p] {
                lag[p] = 0;
            }
        }
        i = j;
    }

    Ok(GlickoResult {
        ratings,
        deviations: vars.iter().map(|v| v.sqrt()).collect(),
        games,
        wins,
        draws,
        losses,
        lag,
    })
}

// ---------------------------------------------------------------------------
// Glicko-2 rating system (Glicko with per-player rating volatility).
//
// Citation governance:
// - READ (formula source): Glickman, M. E. (2022, March 22). Example of
//   the Glicko-2 system [Technical note]. Harvard University.
//   http://www.glicko.net/glicko/glicko2.pdf. Defines the Glicko-2 scale
//   mu = (r - 1500)/173.7178, phi = RD/173.7178 (Step 2), the quantities
//     g(phi) = 1/sqrt(1 + 3 phi^2/pi^2),
//     E(mu, mu_j, phi_j) = 1/(1 + exp(-g(phi_j)(mu - mu_j))),
//     v = (sum_j g(phi_j)^2 E_j (1 - E_j))^-1  (Step 3),
//     Delta = v sum_j g(phi_j)(s_j - E_j)      (Step 4),
//   the Illinois volatility iteration on
//     f(x) = e^x (Delta^2 - phi^2 - v - e^x) / (2 (phi^2 + v + e^x)^2)
//            - (x - a)/tau^2,  a = ln(sigma^2)  (Step 5, revised
//   2012-02-22, epsilon = 1e-6), the pre-update inflation
//   phi*^2 = phi^2 + sigma'^2 (Step 6), the update
//     phi' = 1/sqrt(1/phi*^2 + 1/v), mu' = mu + phi'^2 sum_j g(phi_j)
//     (s_j - E_j) (Step 7), the back-conversion (Step 8), and the worked
//   example (r = 1500, RD = 200, sigma = 0.06, tau = 0.5 vs opponents
//   (1400, 30), (1550, 100), (1700, 300) scoring 1, 0, 0 ->
//   sigma' = 0.05999, r' = 1464.06, RD' = 151.52) pinned by test
//   `g2_paper_anchor_g2a`.
// - READ (implementation source of record): Stephenson, A., & Sonas, J.
//   (2020). PlayerRatings: Dynamic updating methods for player ratings
//   estimation (Version 1.1-0) [R package]. CRAN. `R/ratings.R`
//   `glicko2()` (lines 412-586: per-period splitting, participant-only
//   pre-period inflation lag * sigma^2 with rdmax^2 clamp — the source
//   comments "nlag*(cvols^2) in Glicko-2, (nlag+1)*(cval^2) in Glicko" —
//   volatility ceiling min(sigma', q * rdmax), post-update rdmax^2 clamp,
//   tau > 0 gate, W/D/L and lag bookkeeping) and `src/ratings.c`
//   `glicko2_c` (lines ~121-155: per-game E/dval/dscore accumulation on
//   the Glicko-2 scale, opponent g, gamma signs). Both read in full.
// - NOT READ (cited by the read note as the method's derivation paper):
//   Glickman, M. E. (2001). Dynamic paired comparison models with
//   stochastic variances. Journal of Applied Statistics, 28(6), 673-689.
//
// DERIVED (verified by hand; also confirmed by adversarial spec review):
// PlayerRatings finds sigma' by minimizing
//   nllh(z) = (z - a)^2/tau^2 + ln D + Delta^2/D,  D = phi^2 + v + e^z,
// via a bounded Brent optimize. Glickman's f satisfies
// f(x) = -(1/2) d nllh/dx (check: d/dx ln D = e^x/D and
// d/dx [Delta^2/D] = -Delta^2 e^x/D^2), so the Illinois root of f is the
// same optimum. This implementation follows Glickman's Step 5 Illinois
// algorithm byte-for-byte (endpoint A, `while |B - A| > eps`, halving
// fA <- fA/2, bracket search f(a - k tau) < 0), NOT R's loose-tolerance
// Brent, and was verified against an independently executed float64
// oracle whose G2A fixture reproduces the note's worked example.
//
// Model (per rating period, ascending period label, batch semantics; all
// internal state on the Glicko-2 scale):
//   1. participants only: phi^2 <- min(phi^2 + lag * sigma^2, (q rdmax)^2)
//   2. g_k = 1/sqrt(1 + 3 phi_k^2/pi^2) for ALL players (post-inflation)
//   3. per game (w, b, s, gamma), period-START ratings (C kernel):
//        E_w = 1/(1 + exp(g_b (mu_b - mu_w - gamma)))
//        dval_w += g_b^2 E_w (1 - E_w); dscore_w += g_b (s - E_w)
//        E_b = 1/(1 + exp(g_w (mu_w - mu_b + gamma)))
//        dval_b += g_w^2 E_b (1 - E_b); dscore_b += g_w (1 - s - E_b)
//   4. participants only, iff tau > 0 (R line 532): with v = 1/dval and
//      Delta = v * dscore, sigma <- min(illinois_root, q * rdmax).
//      tau == 0 freezes sigma.
//   5. participants only: phi^2 <- phi^2 + sigma^2  (Step 6, NEW sigma)
//   6. ALL players: phi^2 <- min(1/(1/phi^2 + dval), (q rdmax)^2), THEN
//      mu <- mu + phi^2_new * dscore (Step 7; non-participants have
//      dval = dscore = 0 and only feel the clamp).
// Bookkeeping identical to `glicko_rating`: W/D/L only for scores exactly
// 1 / 0.5 / 0; lag += 1 where cumulative games != 0, then reset for this
// period's participants.
//
// Documented deviation from the read note (R fidelity): Glickman's note
// applies Step 6 alone to idle players every period
// (phi' = sqrt(phi^2 + sigma^2)). PlayerRatings instead leaves idle
// players untouched and applies the accumulated lag as lag * sigma^2
// participant inflation when they next play; between-period idle
// deviation growth is NOT observable in output until the player returns.
// This implementation follows PlayerRatings.
//
// NOT an invariant (verified by oracle fixture G2B): the rating sum is
// NOT conserved (sum 6629.879 != 6600 for the G2B fixture); tests must
// not rely on conservation.
//
// Documented divergences from PlayerRatings `glicko2()` (same set as
// `glicko_rating`): self-play rejected; no `status`/`history`; per-player
// init arrays; u64 period labels grouped ascending with stable row order.
// ---------------------------------------------------------------------------

/// Result of [`glicko2_rating`]: ratings, deviations, volatilities, counts.
#[derive(Debug, Clone)]
pub struct Glicko2Result {
    /// Post-update rating per player (Glicko scale, length `n`).
    pub ratings: Vec<f64>,
    /// Post-update rating deviation (RD) per player (Glicko scale).
    pub deviations: Vec<f64>,
    /// Post-update rating volatility sigma per player (scale-free).
    pub volatilities: Vec<f64>,
    /// Number of game appearances per player.
    pub games: Vec<u64>,
    /// Wins (score exactly 1 as white, or exactly 0 as black).
    pub wins: Vec<u64>,
    /// Draws (score exactly 0.5, both sides).
    pub draws: Vec<u64>,
    /// Losses (score exactly 0 as white, or exactly 1 as black).
    pub losses: Vec<u64>,
    /// Rating periods since the player's last appearance (0 if the player
    /// appeared in the final period or never played).
    pub lag: Vec<u64>,
}

/// Glickman (2022) Step 5 f(x) for the volatility root search.
#[inline]
fn glicko2_fx(x: f64, a: f64, delta2: f64, phi2: f64, v: f64, tau2: f64) -> f64 {
    let ex = x.exp();
    let d = phi2 + v + ex;
    ex * (delta2 - d) / (2.0 * d * d) - (x - a) / tau2
}

/// Glickman (2022) Step 5: new volatility via the Illinois algorithm
/// (revised 2012-02-22), epsilon = 1e-6, returning endpoint A.
fn glicko2_new_volatility(sigma: f64, delta: f64, phi2: f64, v: f64, tau: f64) -> f64 {
    const EPS: f64 = 1e-6;
    let a = (sigma * sigma).ln();
    let tau2 = tau * tau;
    let delta2 = delta * delta;
    let mut big_a = a;
    let mut big_b = if delta2 > phi2 + v {
        (delta2 - phi2 - v).ln()
    } else {
        let mut k = 1.0f64;
        while glicko2_fx(a - k * tau, a, delta2, phi2, v, tau2) < 0.0 {
            k += 1.0;
        }
        a - k * tau
    };
    let mut f_a = glicko2_fx(big_a, a, delta2, phi2, v, tau2);
    let mut f_b = glicko2_fx(big_b, a, delta2, phi2, v, tau2);
    while (big_b - big_a).abs() > EPS {
        let big_c = big_a + (big_a - big_b) * f_a / (f_b - f_a);
        let f_c = glicko2_fx(big_c, a, delta2, phi2, v, tau2);
        if f_c * f_b <= 0.0 {
            big_a = big_b;
            f_a = f_b;
        } else {
            f_a /= 2.0;
        }
        big_b = big_c;
        f_b = f_c;
    }
    (big_a / 2.0).exp()
}

/// Glicko-2 ratings from a game schedule, PlayerRatings `glicko2()`
/// semantics with Glickman's (2022) Illinois volatility step.
///
/// `periods[k]`, `white[k]`, `black[k]`, `score[k]`, `gamma[k]` describe
/// game `k`: rating-period label, player indices in `0..n`, white's score
/// in `[0, 1]`, and white's per-game advantage (Glicko rating points).
/// `init_rating`/`init_dev`/`init_vol` give each player's starting
/// rating, deviation, and volatility (`n` = their common length); `tau`
/// constrains volatility change per period (`tau == 0` freezes it) and
/// `rdmax` is the deviation ceiling (volatility is capped at
/// `ln(10)/400 * rdmax`, the R package's ceiling).
pub fn glicko2_rating(
    periods: &[u64],
    white: &[usize],
    black: &[usize],
    score: &[f64],
    gamma: &[f64],
    init_rating: &[f64],
    init_dev: &[f64],
    init_vol: &[f64],
    tau: f64,
    rdmax: f64,
) -> Result<Glicko2Result, String> {
    let g = periods.len();
    if g == 0 {
        return Err("glicko2_rating: at least one game is required".to_string());
    }
    if white.len() != g || black.len() != g || score.len() != g || gamma.len() != g {
        return Err(format!(
            "glicko2_rating: length mismatch (periods {}, white {}, black {}, score {}, gamma {})",
            g,
            white.len(),
            black.len(),
            score.len(),
            gamma.len()
        ));
    }
    let n = init_rating.len();
    if n < 2 {
        return Err("glicko2_rating: at least two players are required".to_string());
    }
    if n > 10_000 {
        return Err(format!(
            "glicko2_rating: n = {} exceeds the supported cap of 10000",
            n
        ));
    }
    if init_dev.len() != n || init_vol.len() != n {
        return Err(format!(
            "glicko2_rating: init_rating (len {}), init_dev (len {}), and init_vol (len {}) must match",
            n,
            init_dev.len(),
            init_vol.len()
        ));
    }
    if !rdmax.is_finite() || rdmax <= 0.0 {
        return Err(format!(
            "glicko2_rating: rdmax {} must be finite and > 0",
            rdmax
        ));
    }
    if !tau.is_finite() || tau < 0.0 {
        return Err(format!(
            "glicko2_rating: tau {} must be finite and >= 0",
            tau
        ));
    }
    // q = ln(10)/400; the R package caps volatility at q * rdmax
    // (R lines ~420-421: "initial volatility cannot be greater than
    // log(10)*rdmax/400").
    let qv = std::f64::consts::LN_10 / 400.0;
    let vol_max = qv * rdmax;
    for p in 0..n {
        if !init_rating[p].is_finite() {
            return Err(format!("glicko2_rating: init_rating[{}] is not finite", p));
        }
        if !init_dev[p].is_finite() || init_dev[p] <= 0.0 {
            return Err(format!(
                "glicko2_rating: init_dev[{}] = {} must be finite and > 0",
                p, init_dev[p]
            ));
        }
        if init_dev[p] > rdmax {
            return Err(format!(
                "glicko2_rating: init_dev[{}] = {} exceeds rdmax {}",
                p, init_dev[p], rdmax
            ));
        }
        if !init_vol[p].is_finite() || init_vol[p] <= 0.0 {
            return Err(format!(
                "glicko2_rating: init_vol[{}] = {} must be finite and > 0",
                p, init_vol[p]
            ));
        }
        if init_vol[p] > vol_max {
            return Err(format!(
                "glicko2_rating: init_vol[{}] = {} exceeds ln(10)/400 * rdmax = {}",
                p, init_vol[p], vol_max
            ));
        }
    }
    for k in 0..g {
        if white[k] >= n || black[k] >= n {
            return Err(format!(
                "glicko2_rating: game {} has player index out of range (white {}, black {}, n {})",
                k, white[k], black[k], n
            ));
        }
        if white[k] == black[k] {
            return Err(format!(
                "glicko2_rating: game {} has white == black == {} (self-play is not supported)",
                k, white[k]
            ));
        }
        if !score[k].is_finite() || !(0.0..=1.0).contains(&score[k]) {
            return Err(format!(
                "glicko2_rating: game {} has score {} outside [0, 1]",
                k, score[k]
            ));
        }
        if !gamma[k].is_finite() {
            return Err(format!("glicko2_rating: game {} has non-finite gamma", k));
        }
    }

    // Step 2: convert to the Glicko-2 scale. Internal deviation state is
    // phi^2 (variance), as in R; gamma is scaled per game.
    let qip3 = 3.0 / (std::f64::consts::PI * std::f64::consts::PI);
    let rdmax2 = (qv * rdmax) * (qv * rdmax);
    let mut ratings: Vec<f64> = init_rating.iter().map(|r| qv * (r - 1500.0)).collect();
    let mut vars: Vec<f64> = init_dev.iter().map(|d| (qv * d) * (qv * d)).collect();
    let mut vols: Vec<f64> = init_vol.to_vec();
    let mut games = vec![0u64; n];
    let mut wins = vec![0u64; n];
    let mut draws = vec![0u64; n];
    let mut losses = vec![0u64; n];
    let mut lag = vec![0u64; n];
    let mut appeared = vec![false; n];
    let mut gdevs = vec![0.0f64; n];

    // Group by ascending period, preserving row order within a period.
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by_key(|&k| periods[k]); // stable sort keeps row order

    let mut i = 0usize;
    while i < g {
        let period = periods[order[i]];
        let mut j = i;
        while j < g && periods[order[j]] == period {
            j += 1;
        }

        for player in appeared.iter_mut() {
            *player = false;
        }
        for &k in &order[i..j] {
            appeared[white[k]] = true;
            appeared[black[k]] = true;
        }
        // Participants only: lag * sigma^2 inflation (Glicko-2; R lines
        // 520-522 with the explicit "nlag" vs Glicko-1 "(nlag+1)" comment),
        // clamped at (q rdmax)^2.
        for p in 0..n {
            if appeared[p] {
                vars[p] = (vars[p] + lag[p] as f64 * vols[p] * vols[p]).min(rdmax2);
            }
        }
        // g(phi) for ALL players from post-inflation variances.
        for p in 0..n {
            gdevs[p] = 1.0 / (1.0 + qip3 * vars[p]).sqrt();
        }

        let mut dscore = vec![0.0f64; n];
        let mut dval = vec![0.0f64; n];
        for &k in &order[i..j] {
            let (w, b, s, gam) = (white[k], black[k], score[k], gamma[k] * qv);
            // Expectations from period-START ratings, opponent's g in the
            // exponent, natural exp on the Glicko-2 scale (C kernel).
            let e_w = 1.0 / (1.0 + (gdevs[b] * (ratings[b] - ratings[w] - gam)).exp());
            dval[w] += gdevs[b] * gdevs[b] * e_w * (1.0 - e_w);
            dscore[w] += gdevs[b] * (s - e_w);
            let e_b = 1.0 / (1.0 + (gdevs[w] * (ratings[w] - ratings[b] + gam)).exp());
            dval[b] += gdevs[w] * gdevs[w] * e_b * (1.0 - e_b);
            dscore[b] += gdevs[w] * (1.0 - s - e_b);

            games[w] += 1;
            games[b] += 1;
            if s == 1.0 {
                wins[w] += 1;
                losses[b] += 1;
            } else if s == 0.5 {
                draws[w] += 1;
                draws[b] += 1;
            } else if s == 0.0 {
                losses[w] += 1;
                wins[b] += 1;
            }
        }
        // Steps 5-6 (participants only): new volatility iff tau > 0
        // (R line 532 gate; tau == 0 freezes sigma), capped at q * rdmax,
        // then pre-update inflation phi*^2 = phi^2 + sigma'^2.
        if tau > 0.0 {
            for p in 0..n {
                if appeared[p] {
                    let v = 1.0 / dval[p];
                    let delta = v * dscore[p];
                    let sigma = glicko2_new_volatility(vols[p], delta, vars[p], v, tau);
                    vols[p] = sigma.min(vol_max);
                }
            }
        }
        for p in 0..n {
            if appeared[p] {
                vars[p] += vols[p] * vols[p];
            }
        }
        // Step 7, ALL players: variance first (clamped), THEN rating with
        // the NEW variance. Non-participants have dval = dscore = 0.
        for p in 0..n {
            vars[p] = (1.0 / (1.0 / vars[p] + dval[p])).min(rdmax2);
            ratings[p] += vars[p] * dscore[p];
        }
        for p in 0..n {
            if games[p] != 0 {
                lag[p] += 1;
            }
        }
        for p in 0..n {
            if appeared[p] {
                lag[p] = 0;
            }
        }
        i = j;
    }

    Ok(Glicko2Result {
        // Step 8: back to the Glicko scale.
        ratings: ratings.iter().map(|r| r / qv + 1500.0).collect(),
        deviations: vars.iter().map(|v| v.sqrt() / qv).collect(),
        volatilities: vols,
        games,
        wins,
        draws,
        losses,
        lag,
    })
}

// ---------------------------------------------------------------------------
// Stephenson rating system, PlayerRatings `steph()` semantics.
//
// Citation governance:
// - READ: PlayerRatings R package source, R/ratings.R `steph()` driver and
//   src/ratings.c `stephenson_c` kernel (Stephenson, A., & Sonas, J.,
//   PlayerRatings: Dynamic updating methods for player ratings estimation,
//   R package; steph() driver and C kernel read in full). All formulas
//   below cite that source code; R/C line references are to the read copy.
// - NOT READ (provenance only): the system has no standalone journal
//   paper. It originates from Alec Stephenson's winning entry to the 2010
//   Kaggle "Chess ratings: Elo vs the Rest of the World" competition and
//   is documented only in the PlayerRatings package. No formula here is
//   attributed to any unread document.
//
// Model per rating period (ascending unique period labels), with
// qv = ln(10)/400, qip3 = 3 (qv/pi)^2, state ratings r_p and deviation
// VARIANCE c_p, per-period participant set `playi`:
//
// 1. Inflation, participants only (R 689):
//      c_p <- min(c_p + (lag_p + 1) cval^2, rdmax^2)
//    Note (lag+1), unlike Glicko-2's lag * sigma^2: a fresh participant is
//    inflated by one cval^2 every period it plays.
// 2. g factor for ALL players (R 690): g_p = 1/sqrt(1 + qip3 c_p).
// 3. Kernel accumulation per game (C 181-197), B = bval/100 (R 696):
//    white w vs black b with score s and per-game advantage gamma_k:
//      asc_w = s + B
//      e_w = 1/(1 + 10^( g_b (r_b - r_w - gamma_k) / 400 ))
//      dval_w += qv^2 g_b^2 e_w (1 - e_w)
//      dscore_w += g_b (asc_w - e_w)
//      l1t_w += r_b - r_w
//    and mirrored for black with asc_b = 1 - s + B, opponent g_w, and
//    +gamma_k in the exponent. With heterogeneous deviations e_w + e_b
//    != 1 in general (deliberate; inherited from Glicko).
// 4. Posterior for ALL players (R 702-703); non-participants have
//    ngamesi = dval = dscore = 0 so the update is the identity:
//      c_p <- 1/( 1/(c_p + ngamesi_p hval^2) + dval_p )
//      r_p += c_p qv dscore_p
// 5. Lambda drift, participants only (R 704), per-PERIOD game counts:
//      r_p += (lambda/100) l1t_p / ngamesi_p
// 6. Tallies (R 706-712): win iff score exactly 1 (white) / 0 (black);
//    draw iff exactly 0.5 (both sides); loss mirrored. Other scores in
//    (0, 1) update ratings but no W/D/L tally.
// 7. Lag (R 713-714): lag_p += 1 if cumulative games_p != 0, then
//    lag_p = 0 for participants.
//
// Output deviation is sqrt(c_p) (R 726).
//
// Documented divergences from the R driver (same set as `glicko_rating`):
// players are pre-indexed 0..n-1 (no name matching, no `status` frame,
// no `history`); self-play rejected; u64 period labels grouped ascending
// with stable row order. `init_games`/`init_lag` replace the status
// frame's Games/Lag columns; W/D/L outputs are CURRENT-RUN tallies
// (prior Win/Draw/Loss columns are out of scope). Lag continuation for
// indexed non-participants IS reproduced: the step-7 loop over all n
// players matches R 666's `olag += nm` for players with prior games and
// no appearances in the data.
// ---------------------------------------------------------------------------

/// Result of [`stephenson_rating`]: ratings, deviations, counts.
#[derive(Debug, Clone)]
pub struct StephensonResult {
    /// Post-update rating per player (Glicko scale, length `n`).
    pub ratings: Vec<f64>,
    /// Post-update rating deviation per player.
    pub deviations: Vec<f64>,
    /// Cumulative game appearances per player (init_games + current run).
    pub games: Vec<u64>,
    /// Current-run wins (score exactly 1 as white, or exactly 0 as black).
    pub wins: Vec<u64>,
    /// Current-run draws (score exactly 0.5, both sides).
    pub draws: Vec<u64>,
    /// Current-run losses (score exactly 0 as white, or exactly 1 as black).
    pub losses: Vec<u64>,
    /// Rating periods since the player's last appearance (continued from
    /// `init_lag`; 0 if the player appeared in the final period or has
    /// never played).
    pub lag: Vec<u64>,
}

/// Stephenson ratings from a game schedule, PlayerRatings `steph()`
/// semantics (Stephenson's 2010 Kaggle chess-rating system).
///
/// `periods[k]`, `white[k]`, `black[k]`, `score[k]`, `gamma[k]` describe
/// game `k`: rating-period label, player indices in `0..n`, white's score
/// in `[0, 1]`, and white's per-game advantage (rating points).
/// `init_rating`/`init_dev` give each player's starting rating and
/// deviation (`n` = their common length); `init_games`/`init_lag`
/// continue a previous run's cumulative game counts and lags (pass zeros
/// for a fresh run). `cval` is the per-period deviation inflation,
/// `hval` the per-game neighborhood variance, `bval` the per-game bonus
/// added to BOTH players' actual scores (in units of bval/100), `lambda`
/// the drift toward opponents' ratings (percent), and `rdmax` the
/// deviation ceiling. `bval` and `lambda` may be negative (the R package
/// does not restrict them).
#[allow(clippy::too_many_arguments)]
pub fn stephenson_rating(
    periods: &[u64],
    white: &[usize],
    black: &[usize],
    score: &[f64],
    gamma: &[f64],
    init_rating: &[f64],
    init_dev: &[f64],
    init_games: &[u64],
    init_lag: &[u64],
    cval: f64,
    hval: f64,
    bval: f64,
    lambda: f64,
    rdmax: f64,
) -> Result<StephensonResult, String> {
    let g = periods.len();
    if g == 0 {
        return Err("stephenson_rating: at least one game is required".to_string());
    }
    if white.len() != g || black.len() != g || score.len() != g || gamma.len() != g {
        return Err(format!(
            "stephenson_rating: length mismatch (periods {}, white {}, black {}, score {}, gamma {})",
            g,
            white.len(),
            black.len(),
            score.len(),
            gamma.len()
        ));
    }
    let n = init_rating.len();
    if n < 2 {
        return Err("stephenson_rating: at least two players are required".to_string());
    }
    if n > 10_000 {
        return Err(format!(
            "stephenson_rating: n = {} exceeds the supported cap of 10000",
            n
        ));
    }
    if init_dev.len() != n || init_games.len() != n || init_lag.len() != n {
        return Err(format!(
            "stephenson_rating: init_rating (len {}), init_dev (len {}), init_games (len {}), and init_lag (len {}) must match",
            n,
            init_dev.len(),
            init_games.len(),
            init_lag.len()
        ));
    }
    // Counters increment at most once per game (games) / per period (lag),
    // both bounded by g; reject inputs that could overflow u64 rather than
    // panic (debug) or wrap (release) mid-update.
    let g_u64 = g as u64;
    for p in 0..n {
        if init_games[p] > u64::MAX - g_u64 || init_lag[p] > u64::MAX - g_u64 {
            return Err(format!(
                "stephenson_rating: init_games/init_lag for player {} is too large to update without u64 overflow",
                p
            ));
        }
    }
    if !rdmax.is_finite() || rdmax <= 0.0 {
        return Err(format!(
            "stephenson_rating: rdmax {} must be finite and > 0",
            rdmax
        ));
    }
    if !cval.is_finite() || cval < 0.0 {
        return Err(format!(
            "stephenson_rating: cval {} must be finite and >= 0",
            cval
        ));
    }
    if !hval.is_finite() || hval < 0.0 {
        return Err(format!(
            "stephenson_rating: hval {} must be finite and >= 0",
            hval
        ));
    }
    if !bval.is_finite() {
        return Err(format!("stephenson_rating: bval {} must be finite", bval));
    }
    if !lambda.is_finite() {
        return Err(format!(
            "stephenson_rating: lambda {} must be finite",
            lambda
        ));
    }
    for p in 0..n {
        if !init_rating[p].is_finite() {
            return Err(format!(
                "stephenson_rating: init_rating[{}] is not finite",
                p
            ));
        }
        if !init_dev[p].is_finite() || init_dev[p] <= 0.0 {
            return Err(format!(
                "stephenson_rating: init_dev[{}] = {} must be finite and > 0",
                p, init_dev[p]
            ));
        }
        if init_dev[p] > rdmax {
            return Err(format!(
                "stephenson_rating: init_dev[{}] = {} exceeds rdmax {}",
                p, init_dev[p], rdmax
            ));
        }
    }
    for k in 0..g {
        if white[k] >= n || black[k] >= n {
            return Err(format!(
                "stephenson_rating: game {} has player index out of range (white {}, black {}, n {})",
                k, white[k], black[k], n
            ));
        }
        if white[k] == black[k] {
            return Err(format!(
                "stephenson_rating: game {} has white == black == {} (self-play is not supported)",
                k, white[k]
            ));
        }
        if !score[k].is_finite() || !(0.0..=1.0).contains(&score[k]) {
            return Err(format!(
                "stephenson_rating: game {} has score {} outside [0, 1]",
                k, score[k]
            ));
        }
        if !gamma[k].is_finite() {
            return Err(format!(
                "stephenson_rating: game {} has non-finite gamma",
                k
            ));
        }
    }

    let qv = std::f64::consts::LN_10 / 400.0;
    // qip3 = 3 (qv/pi)^2 (R 675). NOT Glicko-2's 3/pi^2: Stephenson works
    // on the raw rating scale, so the qv^2 factor stays in qip3.
    let qip3 = 3.0 * (qv / std::f64::consts::PI) * (qv / std::f64::consts::PI);
    let rdmax2 = rdmax * rdmax;
    let b100 = bval / 100.0;
    let mut ratings: Vec<f64> = init_rating.to_vec();
    // Internal deviation state is the VARIANCE cdevs = dev^2 (R 668).
    let mut cdevs: Vec<f64> = init_dev.iter().map(|d| d * d).collect();
    let mut games: Vec<u64> = init_games.to_vec();
    let mut wins = vec![0u64; n];
    let mut draws = vec![0u64; n];
    let mut losses = vec![0u64; n];
    let mut lag: Vec<u64> = init_lag.to_vec();
    let mut appeared = vec![false; n];
    let mut gdevs = vec![0.0f64; n];

    // Group by ascending period, preserving row order within a period.
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by_key(|&k| periods[k]); // stable sort keeps row order

    let mut i = 0usize;
    while i < g {
        let period = periods[order[i]];
        let mut j = i;
        while j < g && periods[order[j]] == period {
            j += 1;
        }

        for player in appeared.iter_mut() {
            *player = false;
        }
        let mut ngamesi = vec![0u64; n];
        for &k in &order[i..j] {
            appeared[white[k]] = true;
            appeared[black[k]] = true;
            ngamesi[white[k]] += 1;
            ngamesi[black[k]] += 1;
        }
        // Step 1 (R 689): participants only, (lag+1) cval^2, clamped.
        for p in 0..n {
            if appeared[p] {
                cdevs[p] = (cdevs[p] + (lag[p] + 1) as f64 * cval * cval).min(rdmax2);
            }
        }
        // Step 2 (R 690): g factor for ALL players, post-inflation.
        for p in 0..n {
            gdevs[p] = 1.0 / (1.0 + qip3 * cdevs[p]).sqrt();
        }

        // Step 3 (C 181-197): kernel accumulation from period-START
        // ratings; base-10 exponent on the raw scale; tallies inline.
        let mut dscore = vec![0.0f64; n];
        let mut dval = vec![0.0f64; n];
        let mut l1t = vec![0.0f64; n];
        for &k in &order[i..j] {
            let (w, b, s, gam) = (white[k], black[k], score[k], gamma[k]);
            let asc_w = s + b100;
            let e_w = 1.0 / (1.0 + 10f64.powf(gdevs[b] * (ratings[b] - ratings[w] - gam) / 400.0));
            dval[w] += qv * qv * gdevs[b] * gdevs[b] * e_w * (1.0 - e_w);
            dscore[w] += gdevs[b] * (asc_w - e_w);
            l1t[w] += ratings[b] - ratings[w];
            let asc_b = 1.0 - s + b100;
            let e_b = 1.0 / (1.0 + 10f64.powf(gdevs[w] * (ratings[w] - ratings[b] + gam) / 400.0));
            dval[b] += qv * qv * gdevs[w] * gdevs[w] * e_b * (1.0 - e_b);
            dscore[b] += gdevs[w] * (asc_b - e_b);
            l1t[b] += ratings[w] - ratings[b];

            games[w] += 1;
            games[b] += 1;
            if s == 1.0 {
                wins[w] += 1;
                losses[b] += 1;
            } else if s == 0.5 {
                draws[w] += 1;
                draws[b] += 1;
            } else if s == 0.0 {
                losses[w] += 1;
                wins[b] += 1;
            }
        }
        // Step 4 (R 702-703): posterior for ALL players, variance first,
        // THEN rating with the NEW variance (identity for non-participants).
        for p in 0..n {
            cdevs[p] = 1.0 / (1.0 / (cdevs[p] + ngamesi[p] as f64 * hval * hval) + dval[p]);
            ratings[p] += cdevs[p] * qv * dscore[p];
        }
        // Step 5 (R 704): lambda drift, participants only, per-period
        // ngamesi (participants always have ngamesi > 0).
        for p in 0..n {
            if appeared[p] {
                ratings[p] += (lambda / 100.0) * l1t[p] / ngamesi[p] as f64;
            }
        }
        // Step 7 (R 713-714): lag on cumulative games, participants reset.
        for p in 0..n {
            if games[p] != 0 {
                lag[p] += 1;
            }
        }
        for p in 0..n {
            if appeared[p] {
                lag[p] = 0;
            }
        }
        i = j;
    }

    Ok(StephensonResult {
        ratings,
        deviations: cdevs.iter().map(|c| c.sqrt()).collect(),
        games,
        wins,
        draws,
        losses,
        lag,
    })
}

// ---------------------------------------------------------------------------
// elom: multiplayer Elo-style rating for nn-player events (PlayerRatings
// `elom()`).
//
// Citation governance:
// - READ: PlayerRatings R package sources: R driver `elom()` (R/elom.R,
//   lines ~739-932 of the concatenated package R source), C kernel
//   `elom_c` (src/, accumulates ascore/escore per event and computes
//   dscore once per period), and the `kriichi()` K-factor function
//   (lines ~1005-1020). All formulas below were derived from those
//   sources directly and pinned by an executed Python oracle port.
// - NOT READ / does not exist: there is no journal paper for `elom`; it
//   is documented only in the PlayerRatings package (Alec Stephenson),
//   where it is described for multi-player games such as Riichi Mahjong.
//   No formula here is attributed to any unread document.
//
// Model. Each event row has `nn` seats (players and scores); a seat may
// be empty. Per-event preprocessing (R driver ~840-871):
//
// 1. If `placing`, negate scores before ranking (lower placing wins).
// 2. Seat ranks: rank(-zz, ties.method = "min", na.last = "keep"), i.e.
//    rank_j = 1 + #{k : zz_k > zz_j} over occupied seats.
// 3. Base assignment: with `nan` empty seats and base vector b of length
//    nn, the event base vector is b itself when nan = 0; otherwise a
//    ONCE-shrunk copy of b (verified verbatim quirk: `sbase <- basev`
//    sits INSIDE the R shrink loop, so the shrink is applied to the
//    ORIGINAL base exactly once for ANY nan in 1..nn-2, NOT nan times):
//    even length merges the two middle entries into their mean; odd
//    length drops the middle entry. Seat j receives sbase[rank_j - 1].
//
// Per rating period (kernel `elom_c` + R driver update ~887-907), with
// current ratings r and PRE-period cumulative game counts:
//
// 4. For each event in the period with participant set P:
//      avetab = mean(r_p : p in P)
//      ascore_p += base_p;  escore_p += (r_p - avetab) / 40
//    accumulated across ALL events of the period, then
//      dscore_p = ascore_p - escore_p  (0 for non-participants).
// 5. Single update per period: r_p += K_p * dscore_p, where K is either
//    a scalar or kriichi(games): K_p = 1 - (1 - kv) * games_p / gv,
//    clamped to kv when games_p >= gv (kriichi taper; K uses PRE-period
//    games; R computes kfac() before incrementing ngames).
// 6. Tallies: games_p += appearances this period; per-place counts
//    places[p][rank-1] += 1 per appearance; lag_p += 1 for every player
//    with cumulative games != 0, then lag_p = 0 for this period's
//    participants.
//
// Documented divergences from the R driver (REDUCED-SCOPE, mirroring
// `stephenson_rating`): players are pre-indexed 0..n-1 (no name matching,
// no `status` frame -- lag continuation applies to indexed players only;
// there is no status-only roster outside 0..n-1). Rating periods must be
// non-decreasing (R groups by split(); event order within a period
// follows input order). Empty seats are encoded as player = -1 AND score
// = NaN, jointly: in R the score NA count (not player NA) drives ranks
// and base shrink, so a finite score at an empty seat would influence
// other seats' ranks; this port rejects that input instead of
// reproducing it. Duplicate players within one event are rejected (R
// accepts and double-counts them). Kriichi parameters are restricted to
// gv > 0, 0 < kv <= 1 (R does not validate them; the clamp-to-kv form
// equals R's two-step assignment exactly on this domain).
// ---------------------------------------------------------------------------

/// K-factor rule for [`elom_rating`].
#[derive(Debug, Clone, Copy)]
pub enum ElomKFactor {
    /// Constant K for every player and period.
    Scalar(f64),
    /// PlayerRatings `kriichi()`: K = 1 - (1 - kv) * games / gv, clamped
    /// to kv once cumulative (pre-period) games reach gv.
    Kriichi {
        /// Games at which the taper reaches its floor (R default 400).
        gv: f64,
        /// K-factor floor (R default 0.2).
        kv: f64,
    },
}

/// Result of [`elom_rating`]: ratings and tallies.
#[derive(Debug, Clone)]
pub struct ElomResult {
    /// Post-update rating per player (length `n`).
    pub ratings: Vec<f64>,
    /// Cumulative event appearances per player (init_games + current run).
    pub games: Vec<u64>,
    /// Per-place finish counts, row-major `n x nn` (init_places + current
    /// run): `places[p * nn + r]` counts rank `r + 1` finishes.
    pub places: Vec<u64>,
    /// Rating periods since the player's last appearance (continued from
    /// `init_lag`; 0 for final-period participants and never-played
    /// players).
    pub lag: Vec<u64>,
}

/// Multiplayer Elo ratings from an event schedule, PlayerRatings `elom()`
/// semantics (nn-player events, e.g. Riichi Mahjong with nn = 4).
///
/// Events are rows of `nn` seats: `players[e * nn + j]` is seat `j`'s
/// player index in `0..n` or `-1` for an empty seat, and
/// `scores[e * nn + j]` its score (`NaN` exactly at empty seats). At most
/// `nn - 2` seats per event may be empty. `periods` (length `g`,
/// non-decreasing) give each event's rating period. `base` (length `nn`)
/// is the per-rank base score, best rank first (R default
/// `(30, 10, -10, -30)`). If `placing` is true, scores are placings
/// (lower is better). `init_ratings`/`init_games`/`init_lag`/
/// `init_places` continue a previous run (pass zeros for a fresh run;
/// `init_places` is row-major `n x nn`).
#[allow(clippy::too_many_arguments)]
pub fn elom_rating(
    periods: &[u64],
    players: &[i64],
    scores: &[f64],
    base: &[f64],
    init_ratings: &[f64],
    init_games: &[u64],
    init_lag: &[u64],
    init_places: &[u64],
    kfac: ElomKFactor,
    placing: bool,
) -> Result<ElomResult, String> {
    let n = init_ratings.len();
    if n < 2 {
        return Err("elom_rating: at least two players are required".to_string());
    }
    if n > 10_000 {
        return Err(format!(
            "elom_rating: n = {} exceeds the supported cap of 10000",
            n
        ));
    }
    let nn = base.len();
    if !(2..=64).contains(&nn) {
        return Err(format!(
            "elom_rating: base length {} must be in 2..=64 (seats per event)",
            nn
        ));
    }
    if base.iter().any(|b| !b.is_finite()) {
        return Err("elom_rating: base scores must be finite".to_string());
    }
    let g = periods.len();
    if g == 0 {
        return Err("elom_rating: at least one event is required".to_string());
    }
    let cells = g
        .checked_mul(nn)
        .ok_or_else(|| "elom_rating: g * nn overflows usize".to_string())?;
    if players.len() != cells || scores.len() != cells {
        return Err(format!(
            "elom_rating: players (len {}) and scores (len {}) must both have g * nn = {} entries",
            players.len(),
            scores.len(),
            cells
        ));
    }
    if init_ratings.iter().any(|r| !r.is_finite()) {
        return Err("elom_rating: init_ratings must be finite".to_string());
    }
    if init_games.len() != n || init_lag.len() != n {
        return Err(format!(
            "elom_rating: init_games (len {}) and init_lag (len {}) must have length n = {}",
            init_games.len(),
            init_lag.len(),
            n
        ));
    }
    let place_cells = n
        .checked_mul(nn)
        .ok_or_else(|| "elom_rating: n * nn overflows usize".to_string())?;
    if init_places.len() != place_cells {
        return Err(format!(
            "elom_rating: init_places (len {}) must have n * nn = {} entries",
            init_places.len(),
            place_cells
        ));
    }
    if periods.windows(2).any(|w| w[0] > w[1]) {
        return Err("elom_rating: periods must be non-decreasing".to_string());
    }
    // Counters increment at most once per event per player (games, places)
    // or once per period (lag), all bounded by g; reject inputs that could
    // overflow u64 rather than panic (debug) or wrap (release) mid-update.
    let g_u64 = g as u64;
    for p in 0..n {
        if init_games[p] > u64::MAX - g_u64 || init_lag[p] > u64::MAX - g_u64 {
            return Err(format!(
                "elom_rating: init_games/init_lag for player {} is too large to update without u64 overflow",
                p
            ));
        }
    }
    if init_places.iter().any(|&c| c > u64::MAX - g_u64) {
        return Err(
            "elom_rating: an init_places count is too large to update without u64 overflow"
                .to_string(),
        );
    }
    match kfac {
        ElomKFactor::Scalar(k) => {
            if !k.is_finite() || k <= 0.0 {
                return Err(format!(
                    "elom_rating: scalar K-factor {} must be finite and > 0",
                    k
                ));
            }
        }
        ElomKFactor::Kriichi { gv, kv } => {
            if !gv.is_finite() || gv <= 0.0 {
                return Err(format!(
                    "elom_rating: kriichi gv {} must be finite and > 0",
                    gv
                ));
            }
            if !kv.is_finite() || kv <= 0.0 || kv > 1.0 {
                return Err(format!(
                    "elom_rating: kriichi kv {} must be finite and in (0, 1]",
                    kv
                ));
            }
        }
    }

    // Once-shrunk base for events with empty seats (R tmpfun quirk: the
    // shrink applies to the ORIGINAL base exactly once for any nan >= 1).
    let shrunk: Vec<f64> = if nn % 2 == 0 {
        let mut s = Vec::with_capacity(nn - 1);
        s.extend_from_slice(&base[..nn / 2 - 1]);
        s.push((base[nn / 2 - 1] + base[nn / 2]) / 2.0);
        s.extend_from_slice(&base[nn / 2 + 1..]);
        s
    } else {
        let mut s = Vec::with_capacity(nn - 1);
        s.extend_from_slice(&base[..(nn - 1) / 2]);
        s.extend_from_slice(&base[(nn + 1) / 2..]);
        s
    };

    // Per-event validation and rank/base precomputation (R driver
    // ~840-871). rank[e*nn+j] in 1..=(occupied seats); usize::MAX marks
    // empty seats.
    let mut ranks = vec![usize::MAX; cells];
    let mut event_base = vec![0.0f64; cells];
    let mut seen = vec![false; n];
    for e in 0..g {
        let row = e * nn;
        let mut nan = 0usize;
        for j in 0..nn {
            let pl = players[row + j];
            let sc = scores[row + j];
            if pl == -1 {
                if !sc.is_nan() {
                    return Err(format!(
                        "elom_rating: event {} seat {} is empty (player -1) but has a non-NaN score {}; empty seats require NaN scores",
                        e, j, sc
                    ));
                }
                nan += 1;
            } else {
                if pl < 0 || pl as u64 >= n as u64 {
                    return Err(format!(
                        "elom_rating: event {} seat {} has player {} outside 0..{} (or -1 for an empty seat)",
                        e, j, pl, n
                    ));
                }
                if !sc.is_finite() {
                    return Err(format!(
                        "elom_rating: event {} seat {} (player {}) has non-finite score {}",
                        e, j, pl, sc
                    ));
                }
            }
        }
        if nan > nn - 2 {
            return Err(format!(
                "elom_rating: event {} has {} empty seats; at most nn - 2 = {} are supported",
                e,
                nan,
                nn - 2
            ));
        }
        for j in 0..nn {
            let pl = players[row + j];
            if pl == -1 {
                continue;
            }
            let p = pl as usize;
            if seen[p] {
                return Err(format!(
                    "elom_rating: event {} lists player {} more than once",
                    e, p
                ));
            }
            seen[p] = true;
        }
        for j in 0..nn {
            let pl = players[row + j];
            if pl != -1 {
                seen[pl as usize] = false;
            }
        }
        let sbase: &[f64] = if nan == 0 { base } else { &shrunk };
        for j in 0..nn {
            if players[row + j] == -1 {
                continue;
            }
            let zj = if placing {
                -scores[row + j]
            } else {
                scores[row + j]
            };
            // ties.method = "min" on descending scores.
            let mut r = 1usize;
            for k in 0..nn {
                if k == j || players[row + k] == -1 {
                    continue;
                }
                let zk = if placing {
                    -scores[row + k]
                } else {
                    scores[row + k]
                };
                if zk > zj {
                    r += 1;
                }
            }
            ranks[row + j] = r;
            event_base[row + j] = sbase[r - 1];
        }
    }

    let mut ratings = init_ratings.to_vec();
    let mut games = init_games.to_vec();
    let mut lag = init_lag.to_vec();
    let mut places = init_places.to_vec();
    let mut ascore = vec![0.0f64; n];
    let mut escore = vec![0.0f64; n];
    let mut played = vec![0u64; n];

    let mut i = 0usize;
    while i < g {
        let mut j = i + 1;
        while j < g && periods[j] == periods[i] {
            j += 1;
        }
        for v in ascore.iter_mut() {
            *v = 0.0;
        }
        for v in escore.iter_mut() {
            *v = 0.0;
        }
        for v in played.iter_mut() {
            *v = 0;
        }
        // Kernel accumulation across all events of the period (elom_c).
        for e in i..j {
            let row = e * nn;
            let mut sum = 0.0f64;
            let mut cnt = 0usize;
            for k in 0..nn {
                let pl = players[row + k];
                if pl != -1 {
                    sum += ratings[pl as usize];
                    cnt += 1;
                }
            }
            let avetab = sum / cnt as f64;
            for k in 0..nn {
                let pl = players[row + k];
                if pl == -1 {
                    continue;
                }
                let p = pl as usize;
                ascore[p] += event_base[row + k];
                escore[p] += (ratings[p] - avetab) / 40.0;
                played[p] += 1;
                places[p * nn + (ranks[row + k] - 1)] += 1;
            }
        }
        // Single update per period; K uses PRE-period cumulative games.
        for p in 0..n {
            let k = match kfac {
                ElomKFactor::Scalar(k) => k,
                ElomKFactor::Kriichi { gv, kv } => {
                    let gp = games[p] as f64;
                    if gp >= gv {
                        kv
                    } else {
                        1.0 - (1.0 - kv) * gp / gv
                    }
                }
            };
            ratings[p] += k * (ascore[p] - escore[p]);
        }
        // Tallies and lag (R driver ~898-907): games first, then lag on
        // cumulative games != 0, then participant reset.
        for p in 0..n {
            games[p] += played[p];
        }
        for p in 0..n {
            if games[p] != 0 {
                lag[p] += 1;
            }
        }
        for p in 0..n {
            if played[p] != 0 {
                lag[p] = 0;
            }
        }
        i = j;
    }

    Ok(ElomResult {
        ratings,
        games,
        places,
        lag,
    })
}

/// Prediction-quality metrics for binary-outcome forecasts: binomial
/// deviance, root-mean-square error, and mean absolute error, each
/// multiplied by 100 and optionally scaled by the 0.5-constant-predictor
/// baseline.
///
/// # Citation governance
///
/// Normative source (READ): CRAN PlayerRatings 1.1-0, `R/ratings.R`
/// lines 936-957 (`metrics`). NO journal paper exists for this function;
/// its provenance is the CRAN package alone. Verified semantics:
///
/// - The binomial deviance uses the CAPPED predictor column
///   (`pmax.int(pmin.int(pred[,i], cap[2]), cap[1])`, R:946-947) while
///   MSE (R:949) and MAE (R:951) use the RAW UNCAPPED column -- the R
///   source references `pred[,i]`, not `predc`, in those two lines.
/// - `mean(..., na.rm = TRUE)` drops NaN terms elementwise PER METRIC:
///   a numerator term is NaN iff `act[i]` or `pred[i, j]` is NaN. The
///   `scale = TRUE` baselines (R:948, 950, 952) involve only `act`, so
///   their row set is the act-non-NaN rows -- a DIFFERENT row set from
///   the numerator whenever the predictor column has NaNs.
/// - The bdev baseline `-mean(act*log(0.5) + (1-act)*log(0.5))` equals
///   `ln 2` exactly for any FINITE non-NaN act (algebraic identity
///   `a*c + (1-a)*c = c`); this implementation uses `LN_2` directly,
///   which can differ from R\u{2019}s numeric summation by float
///   rounding for non-0/1 act values (within a few ulp).
/// - Results are multiplied by 100 (R:954).
///
/// REDUCED-SCOPE divergences from R (input sanity / presentation):
/// `which`/`sort`/`digits`/`drop` (R:936, 954-956) are presentation and
/// not implemented -- the full `np x 3` unrounded matrix is always
/// returned; `na.rm = FALSE` is not implemented (always elementwise
/// NaN-drop); R\u{2019}s vector recycling of a short `act` is rejected;
/// non-finite non-NaN inputs (Inf) are rejected; an empty per-column
/// row set after NaN removal is rejected (R yields NaN); `scale = TRUE`
/// with an all-0.5 act baseline (zero mae/mse denominator, R yields
/// Inf/NaN) is rejected; caps must satisfy `0 < lo <= hi < 1` (R would
/// take `log` out of domain).
///
/// `pred` is row-major `nr x np` (`pred[i * np + j]`); the output is
/// row-major `np x 3` with per-column `[bdev, mse, mae]`.
pub fn metrics_rating(
    act: &[f64],
    pred: &[f64],
    nr: usize,
    np: usize,
    cap: (f64, f64),
    scale: bool,
) -> Result<Vec<f64>, String> {
    if nr == 0 || nr > 10_000_000 {
        return Err(format!("metrics_rating: nr = {nr} must be in 1..=10000000"));
    }
    if np == 0 || np > 10_000 {
        return Err(format!("metrics_rating: np = {np} must be in 1..=10000"));
    }
    if act.len() != nr {
        return Err(format!(
            "metrics_rating: act has length {} but nr = {nr}",
            act.len()
        ));
    }
    // Checked multiplication BEFORE any indexing: an nr * np overflow must
    // fail loudly instead of wrapping into a bogus expected length.
    let expected = nr
        .checked_mul(np)
        .ok_or_else(|| "metrics_rating: nr * np overflows usize".to_string())?;
    if pred.len() != expected {
        return Err(format!(
            "metrics_rating: pred has length {} but nr * np = {expected}",
            pred.len()
        ));
    }
    let (lo, hi) = cap;
    if !(lo.is_finite() && hi.is_finite() && 0.0 < lo && lo <= hi && hi < 1.0) {
        return Err(format!(
            "metrics_rating: cap ({lo}, {hi}) must satisfy 0 < lo <= hi < 1"
        ));
    }
    for (i, &a) in act.iter().enumerate() {
        if a.is_infinite() {
            return Err(format!(
                "metrics_rating: act[{i}] is infinite (NaN marks missing)"
            ));
        }
    }
    for (k, &p) in pred.iter().enumerate() {
        if p.is_infinite() {
            return Err(format!(
                "metrics_rating: pred[{k}] is infinite (NaN marks missing)"
            ));
        }
    }
    // scale = TRUE baselines run over the act-non-NaN rows only (R:948,
    // 950, 952 involve no pred terms).
    let mut base_sq = 0.0f64;
    let mut base_abs = 0.0f64;
    let mut n_act = 0usize;
    for &a in act {
        if a.is_nan() {
            continue;
        }
        let d = 0.5 - a;
        base_sq += d * d;
        base_abs += d.abs();
        n_act += 1;
    }
    if scale {
        if n_act == 0 {
            return Err(
                "metrics_rating: scale requires at least one non-NaN act value".to_string(),
            );
        }
        if base_sq == 0.0 || base_abs == 0.0 {
            return Err(
                "metrics_rating: scale baseline is zero (all non-NaN act values are 0.5)"
                    .to_string(),
            );
        }
    }
    let base_sq_mean = base_sq / n_act.max(1) as f64;
    let base_abs_mean = base_abs / n_act.max(1) as f64;
    let mut out = Vec::with_capacity(np * 3);
    for j in 0..np {
        let mut sum_bdev = 0.0f64;
        let mut sum_sq = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut n_pair = 0usize;
        for i in 0..nr {
            let a = act[i];
            let p = pred[i * np + j];
            if a.is_nan() || p.is_nan() {
                continue;
            }
            // bdev uses the CAPPED value (R:946-947); mse/mae use the
            // raw pred (R:949, 951) -- the key quirk of the R source.
            let pc = p.clamp(lo, hi);
            sum_bdev += a * pc.ln() + (1.0 - a) * (1.0 - pc).ln();
            let d = p - a;
            sum_sq += d * d;
            sum_abs += d.abs();
            n_pair += 1;
        }
        if n_pair == 0 {
            return Err(format!(
                "metrics_rating: predictor column {j} has no rows where both act and pred are non-NaN"
            ));
        }
        let n = n_pair as f64;
        let mut bdev = -sum_bdev / n;
        let mut mse = (sum_sq / n).sqrt();
        let mut mae = sum_abs / n;
        if scale {
            bdev /= std::f64::consts::LN_2;
            mse /= base_sq_mean.sqrt();
            mae /= base_abs_mean;
        }
        out.push(100.0 * bdev);
        out.push(100.0 * mse);
        out.push(100.0 * mae);
    }
    Ok(out)
}

#[cfg(test)]
#[path = "../../../tests/unit/scaling_tests.rs"]
mod tests;
