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

#[cfg(test)]
#[path = "../../../tests/unit/scaling_tests.rs"]
mod tests;
