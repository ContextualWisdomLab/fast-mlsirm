//! IRT classification accuracy and consistency for cut-score decisions.
//!
//! Two estimators of how reliably a test classifies respondents against one
//! or more cut scores:
//!
//! - **Rudner's normal-approximation method** ([`rudner_classification`]):
//!   the observed score for a respondent with true ability `theta` is taken
//!   as `N(theta, se(theta)^2)`; classification accuracy is the normal mass
//!   of the interval containing `theta`, and consistency is the sum of
//!   squared interval masses (the probability that two independent parallel
//!   administrations land in the same category).
//! - **Lee's summed-score method** ([`lee_classification`]): the exact
//!   summed-score distribution `f(x | theta)` from the Lord-Wingersky (1984)
//!   recursion ([`crate::scoring::lord_wingersky`]) replaces the normal
//!   approximation; categories are raw-score ranges `ceil(c_k) ..
//!   ceil(c_{k+1}) - 1` and the true category is the one whose interval
//!   contains the expected true score `sum_j P_j(theta)`.
//!
//! # Verified sources
//!
//! - Rudner (2001, READ in full): accuracy for a single cut —
//!   `P(observed > c | theta)` is the normal tail beyond `(c - theta) /
//!   se(theta)` with `se(theta) = 1 / sqrt(I(theta))` (eqs. 1-4), aggregated
//!   over the ability distribution (eqs. 5-8).
//! - Rudner (2005, READ in full): the K-category generalization —
//!   `P(obs in [a, b] | theta) = Phi((b - theta)/se) - Phi((a - theta)/se)`
//!   (eq. 1), summed over true-score intervals.
//! - CRAN `cacIRT` 1.4 R sources (Lathrop, READ line by line: `Rud.P.R`,
//!   `Rud.D.R`, `Lee.P.R`, `Lee.D.R`, `recursive.raw.R`, `class.Rud.R`,
//!   `class.Lee.R`): the oracle for the CONSISTENCY formulas (sum of squared
//!   category masses), the per-cut vs simultaneous split, the person-level
//!   (unweighted mean) vs distribution-level (quadrature-weighted mean)
//!   aggregation, and the Lee raw-score mechanics (`bang = ceiling(cuts)`,
//!   category k = raw scores `bang_k .. bang_{k+1} - 1`).
//! - Neither Rudner paper contains a classification-consistency formula; the
//!   consistency implementation follows the cacIRT source, and the concept is
//!   attributed to Lee (2010) as cited in Lathrop's package (Lee 2010 itself
//!   NOT read — paywalled).
//!
//! # Divergences from the cacIRT oracle (deliberate, tested)
//!
//! 1. Category intervals are LEFT-CLOSED everywhere (`theta == cut` or
//!    `expected true score == cut` classifies into the upper category),
//!    matching `Rud.P`/`Rud.D`/`Lee.P` (R `cut(..., right = FALSE)`).
//!    cacIRT's `Lee.D` alone uses right-closed intervals; the two differ only
//!    when a value lands exactly on a cut.
//! 2. `P == 1` item probabilities break cacIRT's `recursive.raw` (its
//!    `Z = P/Q` hazard divides by zero); we reject `P` outside the open
//!    interval `(0, 1)` at the trust boundary instead of silently clamping
//!    (`gen.rec.raw` clamps to `[1e-4, 1 - 1e-4]`). Rejecting `P == 0` is a
//!    stricter policy than the oracle (which tolerates it), not parity.
//! 3. cacIRT emits the "Simultaneous" outputs only for two or more cuts; we
//!    always populate them (for one cut they coincide with the per-cut
//!    values — an identity anchored by multi-cut test fixtures).
//! 4. Lee raw cuts are restricted to `(0, n_items]` after `ceil`-mapping to
//!    strictly increasing integer boundaries; cuts inside
//!    `(n_items, n_items + 1)` would produce an empty top category (a
//!    reversed slice in the R oracle) and are rejected.
//!
//! The normal CDF uses this crate's `erfc` rational approximation
//! (|error| < 1.2e-7); Rudner outputs inherit that accuracy. Lee outputs
//! involve no normal CDF and are exact to f64 rounding.
//!
//! # References
//!
//! Lathrop, Q. N. (2015). *cacIRT: Classification accuracy and consistency
//! under item response theory* (Version 1.4) \[R package\]. CRAN.
//! <https://CRAN.R-project.org/package=cacIRT>
//!
//! Lee, W.-C. (2010). Classification consistency and accuracy for complex
//! assessments using item response theory. *Journal of Educational
//! Measurement, 47*(1), 1-17. (As cited in Lathrop, 2015; not read.)
//!
//! Lord, F. M., & Wingersky, M. S. (1984). Comparison of IRT true-score and
//! equipercentile observed-score "equatings". *Applied Psychological
//! Measurement, 8*(4), 453-461. https://doi.org/10.1177/014662168400800409
//!
//! Rudner, L. M. (2001). Computing the expected proportions of misclassified
//! examinees. *Practical Assessment, Research & Evaluation, 7*(14).
//! https://doi.org/10.7275/an9m-2035
//!
//! Rudner, L. M. (2005). Expected classification accuracy. *Practical
//! Assessment, Research & Evaluation, 10*(13).
//! https://doi.org/10.7275/56a5-6b14

use crate::fitstats::erfc;
use crate::scoring::lord_wingersky;

/// Classification accuracy/consistency summary shared by both methods.
///
/// `m` cuts and `n` evaluation points (persons or quadrature nodes).
/// Marginal values are means over points weighted by the normalized input
/// weights; conditional values are per point.
#[derive(Clone, Debug)]
pub struct ClassificationResult {
    /// Marginal accuracy per cut (`m`), each cut treated as its own
    /// two-category problem.
    pub per_cut_accuracy: Vec<f64>,
    /// Marginal consistency per cut (`m`).
    pub per_cut_consistency: Vec<f64>,
    /// Marginal accuracy of the full `m + 1`-category classification.
    pub simultaneous_accuracy: f64,
    /// Marginal consistency of the full classification.
    pub simultaneous_consistency: f64,
    /// Row-major `m x n` conditional accuracy.
    pub conditional_accuracy: Vec<f64>,
    /// Row-major `m x n` conditional consistency.
    pub conditional_consistency: Vec<f64>,
    /// Per-point simultaneous accuracy (`n`).
    pub conditional_simultaneous_accuracy: Vec<f64>,
    /// Per-point simultaneous consistency (`n`).
    pub conditional_simultaneous_consistency: Vec<f64>,
}

/// Standard normal CDF via the crate's `erfc` approximation.
fn phi(z: f64) -> f64 {
    0.5 * erfc(-z / std::f64::consts::SQRT_2)
}

fn validate_weights(weights: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if weights.len() != n {
        return Err("weights length must match the number of evaluation points".into());
    }
    let mut total = 0.0;
    for &w in weights {
        if !w.is_finite() || w < 0.0 {
            return Err("weights must be finite and non-negative".into());
        }
        total += w;
    }
    if !total.is_finite() {
        return Err("weights sum overflows f64".into());
    }
    if total <= 0.0 {
        return Err("weights must not all be zero".into());
    }
    Ok(weights.iter().map(|&w| w / total).collect())
}

fn validate_cuts(cutscores: &[f64]) -> Result<(), String> {
    if cutscores.is_empty() {
        return Err("at least one cutscore is required".into());
    }
    for &c in cutscores {
        if !c.is_finite() {
            return Err("cutscores must be finite".into());
        }
    }
    if cutscores.windows(2).any(|w| w[0] >= w[1]) {
        return Err("cutscores must be strictly increasing".into());
    }
    Ok(())
}

/// Left-closed category index: number of cuts `c` with `value >= c`.
fn category(value: f64, cutscores: &[f64]) -> usize {
    cutscores.iter().filter(|&&c| value >= c).count()
}

/// Aggregate per-point conditional values into a [`ClassificationResult`].
///
/// `masses(i, cuts)` returns the `cuts.len() + 1` category probability masses
/// at point `i`; `true_value(i)` is the value classified against the cuts.
fn assemble(
    n: usize,
    cutscores: &[f64],
    wn: &[f64],
    masses: impl Fn(usize, &[f64]) -> Vec<f64>,
    true_value: impl Fn(usize) -> f64,
) -> ClassificationResult {
    let m = cutscores.len();
    let mut ca = vec![0.0; m * n];
    let mut cc = vec![0.0; m * n];
    let mut csa = vec![0.0; n];
    let mut csc = vec![0.0; n];
    for i in 0..n {
        let v = true_value(i);
        for (j, &c) in cutscores.iter().enumerate() {
            let ms = masses(i, std::slice::from_ref(&c));
            let k = category(v, std::slice::from_ref(&c));
            ca[j * n + i] = ms[k];
            cc[j * n + i] = ms.iter().map(|p| p * p).sum();
        }
        let ms = masses(i, cutscores);
        let k = category(v, cutscores);
        csa[i] = ms[k];
        csc[i] = ms.iter().map(|p| p * p).sum();
    }
    let wmean = |row: &[f64]| row.iter().zip(wn).map(|(x, w)| x * w).sum::<f64>();
    ClassificationResult {
        per_cut_accuracy: (0..m).map(|j| wmean(&ca[j * n..(j + 1) * n])).collect(),
        per_cut_consistency: (0..m).map(|j| wmean(&cc[j * n..(j + 1) * n])).collect(),
        simultaneous_accuracy: wmean(&csa),
        simultaneous_consistency: wmean(&csc),
        conditional_accuracy: ca,
        conditional_consistency: cc,
        conditional_simultaneous_accuracy: csa,
        conditional_simultaneous_consistency: csc,
    }
}

/// Rudner normal-approximation classification accuracy and consistency.
///
/// `theta[i]` are true abilities (persons or quadrature nodes), `sem[i] > 0`
/// the conditional standard errors of measurement, `weights[i] >= 0` the
/// aggregation weights (uniform weights reproduce cacIRT's person-level
/// `Rud.P` `rowMeans`; quadrature weights reproduce the distribution-level
/// `Rud.D` `weighted.mean` — normalization is internal, so unnormalized
/// weights are accepted). `cutscores` must be finite and strictly increasing.
///
/// Accuracy per Rudner (2001, eqs. 1-3; 2005, eq. 1); consistency per the
/// cacIRT source (see module docs).
pub fn rudner_classification(
    theta: &[f64],
    sem: &[f64],
    weights: &[f64],
    cutscores: &[f64],
) -> Result<ClassificationResult, String> {
    let n = theta.len();
    if n == 0 {
        return Err("at least one evaluation point is required".into());
    }
    if sem.len() != n {
        return Err("theta and sem lengths differ".into());
    }
    if theta.iter().any(|v| !v.is_finite()) {
        return Err("theta must be finite".into());
    }
    if sem.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("sem must be finite and positive".into());
    }
    validate_cuts(cutscores)?;
    let wn = validate_weights(weights, n)?;

    let masses = |i: usize, cuts: &[f64]| -> Vec<f64> {
        let (t, s) = (theta[i], sem[i]);
        let mut out = Vec::with_capacity(cuts.len() + 1);
        let mut lower = 0.0; // Phi(-inf)
        for &c in cuts {
            let upper = phi((c - t) / s);
            out.push(upper - lower);
            lower = upper;
        }
        out.push(1.0 - lower);
        out
    };
    Ok(assemble(n, cutscores, &wn, masses, |i| theta[i]))
}

/// Lee summed-score classification accuracy and consistency (dichotomous).
///
/// `probs` is row-major `n_points x n_items` with `P(X_j = 1 | theta_i)`
/// strictly inside `(0, 1)` — model-agnostic: any binary-response IRF may
/// produce it. `weights` aggregate points as in [`rudner_classification`].
/// `cutscores` are raw-score cuts in `(0, n_items]`; a summed score `x`
/// falls above cut `c` iff `x >= ceil(c)`, and the `ceil`-mapped boundaries
/// must be strictly increasing. The point's true category is the raw-score
/// interval containing its expected true score `sum_j P_ij` (left-closed).
///
/// The summed-score distribution comes from
/// [`crate::scoring::lord_wingersky`]; mechanics follow cacIRT's
/// `Lee.P`/`Lee.D` (see module docs for the divergences).
pub fn lee_classification(
    probs: &[f64],
    n_points: usize,
    n_items: usize,
    weights: &[f64],
    cutscores: &[f64],
) -> Result<ClassificationResult, String> {
    if n_points == 0 || n_items == 0 {
        return Err("at least one evaluation point and one item are required".into());
    }
    let cells = n_points
        .checked_mul(n_items)
        .ok_or_else(|| "n_points * n_items overflows".to_string())?;
    if probs.len() != cells {
        return Err("probs length must equal n_points * n_items".into());
    }
    n_points
        .checked_mul(n_items + 1)
        .ok_or_else(|| "summed-score table size overflows".to_string())?;
    if probs
        .iter()
        .any(|p| !p.is_finite() || *p <= 0.0 || *p >= 1.0)
    {
        return Err("probs must lie strictly inside (0, 1)".into());
    }
    validate_cuts(cutscores)?;
    for &c in cutscores {
        if c <= 0.0 || c > n_items as f64 {
            return Err("raw cutscores must lie in (0, n_items]".into());
        }
    }
    let bang: Vec<usize> = cutscores.iter().map(|&c| c.ceil() as usize).collect();
    if bang.windows(2).any(|w| w[0] >= w[1]) {
        return Err("ceil-mapped raw cutscores must be strictly increasing".into());
    }
    let wn = validate_weights(weights, n_points)?;

    // f(x | theta_i): lord_wingersky treats each point as one "node"; feed it
    // per point so the table stays (n_items + 1) x 1.
    let sc = n_items + 1;
    let mut dist = vec![0.0; n_points * sc];
    let mut ts = vec![0.0; n_points];
    for i in 0..n_points {
        let row = &probs[i * n_items..(i + 1) * n_items];
        ts[i] = row.iter().sum();
        let f = lord_wingersky(row, n_items, 1);
        dist[i * sc..(i + 1) * sc].copy_from_slice(&f);
    }

    let masses = |i: usize, cuts: &[f64]| -> Vec<f64> {
        let f = &dist[i * sc..(i + 1) * sc];
        let mut bounds = Vec::with_capacity(cuts.len() + 2);
        bounds.push(0usize);
        bounds.extend(cuts.iter().map(|&c| c.ceil() as usize));
        bounds.push(sc);
        bounds
            .windows(2)
            .map(|w| f[w[0]..w[1]].iter().sum())
            .collect()
    };
    Ok(assemble(n_points, cutscores, &wn, masses, |i| ts[i]))
}

#[cfg(test)]
#[path = "../../../tests/unit/classification_tests.rs"]
mod tests;
