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

// ===================== Livingston & Lewis (1995) =====================

/// Livingston-Lewis classification consistency and accuracy summary.
///
/// Orientation: "pass" means observed score `>= cut`. Accuracy cells are
/// joint proportions over (true state, observed state) and sum to 1;
/// consistency cells are joint proportions over two hypothetical independent
/// administrations, normalized to sum to 1 (`i` = fail, `j` = pass).
#[derive(Clone, Debug)]
pub struct LivingstonLewisResult {
    /// Unrounded effective test length (Livingston & Lewis, 1995, as
    /// implemented in betafunctions' `ETL`).
    pub effective_test_length: f64,
    /// `round-ties-even(effective_test_length)`, the `N` used in all
    /// binomial integrals (mirrors R `round()`).
    pub etl_rounded: u64,
    /// Lower location of the fitted beta true-score distribution.
    pub lower: f64,
    /// Upper location of the fitted beta true-score distribution.
    pub upper: f64,
    /// First shape parameter of the fitted beta true-score distribution.
    pub alpha: f64,
    /// Second shape parameter of the fitted beta true-score distribution.
    pub beta: f64,
    /// True iff the two-parameter fail-safe replaced the four-parameter fit.
    pub used_two_parameter: bool,
    /// P(true pass, observed pass).
    pub p_tp: f64,
    /// P(true fail, observed pass) — false positive.
    pub p_fp: f64,
    /// P(true fail, observed fail).
    pub p_tf: f64,
    /// P(true pass, observed fail) — false negative.
    pub p_ff: f64,
    /// `p_tp + p_tf`.
    pub accuracy: f64,
    /// `p_tp / (p_tp + p_ff)` — P(observed pass | true pass); `NaN` when
    /// the true-pass margin vanishes (cut outside the fitted support).
    pub sensitivity: f64,
    /// `p_tf / (p_tf + p_fp)` — P(observed fail | true fail); `NaN` when
    /// the true-fail margin vanishes.
    pub specificity: f64,
    /// Consistency cell: fail on both administrations.
    pub p_ii: f64,
    /// Consistency cell: fail then pass (equals `p_ji` by construction here).
    pub p_ij: f64,
    /// Consistency cell: pass then fail.
    pub p_ji: f64,
    /// Consistency cell: pass on both administrations.
    pub p_jj: f64,
    /// `p_ii + p_jj` — expected agreement between two administrations.
    pub consistency: f64,
    /// Chance agreement `(p_ii+p_ij)(p_ii+p_ji) + (p_ij+p_jj)(p_ji+p_jj)`.
    pub chance_consistency: f64,
    /// Cohen's kappa `(p - p_c) / (1 - p_c)`; `NaN` when `p_c == 1`.
    pub kappa: f64,
}

/// Gauss-Legendre nodes/weights on `[-1, 1]` by Newton iteration on the
/// Legendre polynomial recurrence (Press et al., 2007, sec. 4.6 `gauleg`;
/// transcribed from the textbook algorithm and verified in the test suite
/// against the exact 2-node rule and polynomial-exactness identities).
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let m = n.div_ceil(2);
    for i in 0..m {
        // Initial guess: Chebyshev approximation to the i-th root.
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..100 {
            let mut p1 = 1.0;
            let mut p2 = 0.0;
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                let jf = j as f64;
                p1 = ((2.0 * jf + 1.0) * z * p2 - jf * p3) / (jf + 1.0);
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() < 1e-15 {
                break;
            }
        }
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = 2.0 / ((1.0 - z * z) * pp * pp);
        w[n - 1 - i] = w[i];
    }
    (x, w)
}

/// `integral_{t0}^{t1} t^(a-1) (1-t)^(b-1) g(t) dt / B(a, b)`. Endpoint
/// singularities (shape < 1) are absorbed exactly by power substitutions
/// (spec rev 2): on `t <= 1/2` with `a < 1` substitute `v = t^a` so
/// `t^(a-1) dt = dv / a`; on `t >= 1/2` with `b < 1` symmetrically
/// `w = (1-t)^b`. For shape >= 1 the integrand is bounded and integrated
/// directly (substituting there would introduce a `v^(1/a)` derivative kink
/// at 0 and LOSE accuracy — measured 2e-5 mass error on a smooth
/// alpha ~ 8.5 case). Each piece: 64-node Gauss-Legendre over 8
/// subintervals, geometrically graded toward the splitting endpoint on the
/// direct path to handle the unbounded `t^(a-1)` derivative for
/// 1 < shape < 2.
fn beta_weighted_integral(a: f64, b: f64, t0: f64, t1: f64, g: impl Fn(f64) -> f64) -> f64 {
    let ln_b = crate::fitstats::ln_gamma(a) + crate::fitstats::ln_gamma(b)
        - crate::fitstats::ln_gamma(a + b);
    let (nodes, weights) = gauss_legendre(64);
    let gl = |lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| -> f64 {
        let c = 0.5 * (lo + hi);
        let h = 0.5 * (hi - lo);
        nodes
            .iter()
            .zip(&weights)
            .map(|(z, wt)| wt * f(c + h * z))
            .sum::<f64>()
            * h
    };
    // Uniform 8-subinterval composite.
    let composite = |lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| -> f64 {
        if hi <= lo {
            return 0.0;
        }
        let n_sub = 8;
        let h = (hi - lo) / n_sub as f64;
        (0..n_sub)
            .map(|s| gl(lo + s as f64 * h, lo + (s + 1) as f64 * h, f))
            .sum()
    };
    // Composite geometrically graded toward `lo` (ratio 4 per level).
    let graded = |lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| -> f64 {
        if hi <= lo {
            return 0.0;
        }
        let mut total = 0.0;
        let mut right = hi;
        let len = hi - lo;
        for lev in 1..8 {
            let left = lo + len * 4.0_f64.powi(-lev);
            total += gl(left, right, f);
            right = left;
        }
        total + gl(lo, right, f)
    };
    let mut total = 0.0;
    // Left piece: t in [t0, min(t1, 1/2)].
    let tl = t1.min(0.5);
    if t0 < tl {
        if a < 1.0 {
            let f = |v: f64| {
                let t = v.powf(1.0 / a);
                (1.0 - t).powf(b - 1.0) * g(t) / a
            };
            total += composite(t0.powf(a), tl.powf(a), &f);
        } else {
            let f = |t: f64| t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0) * g(t);
            total += graded(t0, tl, &f);
        }
    }
    // Right piece: t in [max(t0, 1/2), t1].
    let tr = t0.max(0.5);
    if tr < t1 {
        if b < 1.0 {
            let f = |wv: f64| {
                let t = 1.0 - wv.powf(1.0 / b);
                t.powf(a - 1.0) * g(t) / b
            };
            total += composite((1.0 - t1).powf(b), (1.0 - tr).powf(b), &f);
        } else {
            let f = |t: f64| t.powf(a - 1.0) * (1.0 - t).powf(b - 1.0) * g(t);
            // graded toward t1 (the potentially singular-derivative end):
            // mirror through u = t0 + t1 - t is unnecessary — grade by
            // integrating the reflected function.
            let fr = |u: f64| f(tr + t1 - u);
            total += graded(tr, t1, &fr);
        }
    }
    total / ln_b.exp()
}

/// Falling factorial `a (a-1) ... (a-r+1)` with the oracle's clamp: for the
/// data path the caller zeroes values below `r` (betafunctions `dfac`).
fn falling_factorial(a: f64, r: u32) -> f64 {
    (0..r).map(|j| a - j as f64).product()
}

/// Livingston-Lewis (1995) classification accuracy and consistency for one
/// cut score, from observed scores and a reliability estimate.
///
/// Pipeline (spec `ll_spec.md` rev 2; each step verified line-by-line against
/// the CRAN betafunctions 1.9.0 sources, the only obtainable oracle — the
/// 1995 paper itself is paywalled and was NOT read):
///
/// 1. Effective test length `ETL = ((m-min)(max-m) - r s^2) / (s^2 (1-r))`
///    with sample variance (`ddof = n-1`); integrals use
///    `N = round_ties_even(ETL)`, moment estimation uses the unrounded ETL
///    (matching `LL.CA`'s call order).
/// 2. True-score raw moments by the binomial factorial-moment identity
///    (`HB.tsm` with Lord's k = 0): `m_i = mean(ff(x', i)) / ff(ETL, i)` on
///    `x' = (x-min)/(max-min) * ETL`, `ff(x', i) := 0` when `x' < i`.
/// 3. Four-parameter beta moment fit (`Beta.4p.fit`); fail-safe to the
///    standard two-parameter fit (`AMS`/`BMS` at `l=0, u=1`) when the 4P
///    solution has `l < 0`, `u > 1`, or is numerically invalid (the last is
///    a documented divergence: the oracle only checks the location bounds).
/// 4. Accuracy cells `integral f(p) * BinTail(p)` split at the rescaled cut;
///    consistency cells `integral f(p) * Tail * Tail` (Hanson-style, per the
///    oracle's own docs; Hanson 1991 NOT read). Passing threshold
///    `k = round_ties_even(N c)` is used in BOTH blocks — a documented
///    divergence from the oracle, which mixes `round` (accuracy) and `floor`
///    (consistency) and is therefore asymmetric in `p_ij`/`p_ji`; under the
///    single threshold `p_ij == p_ji` by construction.
///
/// Orientation: pass = observed `>= cut`; the oracle's `caStats` labels fail
/// as "positive", so its sensitivity is this function's specificity.
///
/// # References
///
/// Haakstad, H. (2023). *betafunctions: Functions for working with two- and
/// four-parameter beta probability distributions* (Version 1.9.0)
/// \[R package\]. CRAN. <https://CRAN.R-project.org/package=betafunctions>
///
/// Hanson, B. A. (1991). *Method of moments estimates for the four-parameter
/// beta compound binomial model and the calculation of classification
/// consistency indexes* (ACT Research Report 91-5). (As cited in Haakstad,
/// 2023; not read.)
///
/// Livingston, S. A., & Lewis, C. (1995). Estimating the consistency and
/// accuracy of classifications based on test scores. *Journal of Educational
/// Measurement, 32*(2), 179-197.
/// https://doi.org/10.1111/j.1745-3984.1995.tb00462.x (As implemented in
/// Haakstad, 2023; the paper itself was not obtainable.)
///
/// Press, W. H., Teukolsky, S. A., Vetterling, W. T., & Flannery, B. P.
/// (2007). *Numerical recipes: The art of scientific computing* (3rd ed.).
/// Cambridge University Press. (Sec. 4.6 Gauss-Legendre `gauleg`.)
pub fn livingston_lewis(
    scores: &[f64],
    reliability: f64,
    min: f64,
    max: f64,
    cut: f64,
) -> Result<LivingstonLewisResult, String> {
    let n = scores.len();
    if n < 10 {
        return Err("at least 10 observed scores are required".into());
    }
    if !min.is_finite() || !max.is_finite() || min >= max {
        return Err("min and max must be finite with min < max".into());
    }
    if scores
        .iter()
        .any(|x| !x.is_finite() || *x < min || *x > max)
    {
        return Err("scores must be finite and within [min, max]".into());
    }
    if !cut.is_finite() || cut <= min || cut >= max {
        return Err("cut must be strictly inside (min, max)".into());
    }
    if !reliability.is_finite() || reliability <= 0.0 || reliability >= 1.0 {
        return Err("reliability must be in the open interval (0, 1)".into());
    }
    let nf = n as f64;
    let mean = scores.iter().sum::<f64>() / nf;
    let s2 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    if s2 <= 0.0 {
        return Err("observed-score variance must be positive".into());
    }
    let etl = ((mean - min) * (max - mean) - reliability * s2) / (s2 * (1.0 - reliability));
    if !etl.is_finite() || etl < 2.0 {
        return Err(format!(
            "effective test length {etl:.4} is not usable (needs >= 2); \
             check the reliability and score bounds"
        ));
    }
    // True-score raw moments (HB.tsm, k = 0) on the unrounded ETL scale.
    let mut m = [0.0_f64; 5];
    for (i, mi) in m.iter_mut().enumerate().skip(1) {
        let r = i as u32;
        let num = scores
            .iter()
            .map(|x| {
                let xp = (x - min) / (max - min) * etl;
                if xp < r as f64 {
                    0.0
                } else {
                    falling_factorial(xp, r)
                }
            })
            .sum::<f64>()
            / nf;
        *mi = num / falling_factorial(etl, r);
    }
    let m1 = m[1];
    let ts2 = m[2] - m1 * m1;
    if !(ts2 > 0.0) {
        return Err("estimated true-score variance is not positive".into());
    }
    let g3 = (m[3] - 3.0 * m1 * m[2] + 2.0 * m1.powi(3)) / ts2.powf(1.5);
    let g4 = (m[4] - 4.0 * m1 * m[3] + 6.0 * m1 * m1 * m[2] - 3.0 * m1.powi(4)) / (ts2 * ts2);
    // Four-parameter beta moment fit with two-parameter fail-safe.
    let mut used_two_parameter = true;
    let (mut a, mut b, mut lower, mut upper) = (f64::NAN, f64::NAN, 0.0, 1.0);
    let rr = 6.0 * (g4 - g3 * g3 - 1.0) / (6.0 + 3.0 * g3 * g3 - 2.0 * g4);
    let d =
        1.0 - 24.0 * (rr + 1.0) / ((rr + 2.0) * (rr + 3.0) * g4 - 3.0 * (rr - 6.0) * (rr + 1.0));
    if d.is_finite() && d >= 0.0 {
        let sq = d.sqrt();
        let (a4, b4) = if g3 < 0.0 {
            (rr / 2.0 * (1.0 + sq), rr / 2.0 * (1.0 - sq))
        } else {
            (rr / 2.0 * (1.0 - sq), rr / 2.0 * (1.0 + sq))
        };
        if a4.is_finite() && b4.is_finite() && a4 > 0.0 && b4 > 0.0 {
            let spread = (ts2 * (a4 + b4 + 1.0)).sqrt() / (a4 * b4).sqrt();
            let l4 = m1 - a4 * spread;
            let u4 = m1 + b4 * spread;
            if l4 >= 0.0 && u4 <= 1.0 {
                a = a4;
                b = b4;
                lower = l4;
                upper = u4;
                used_two_parameter = false;
            }
        }
    }
    if used_two_parameter {
        let scale = m1 * (1.0 - m1) / ts2 - 1.0;
        a = m1 * scale;
        b = (1.0 - m1) * scale;
        lower = 0.0;
        upper = 1.0;
    }
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err("beta true-score fit produced invalid shape parameters".into());
    }
    let n_int = etl.round_ties_even();
    let nn = n_int as u64;
    let c = (cut - min) / (max - min);
    let k = (n_int * c).round_ties_even();
    // P(X <= k-1 | N, p) via the binomial-beta CDF identity
    // P(X <= m) = I_{1-p}(N-m, m+1) with m = k-1.
    let fail_prob = |t: f64| -> f64 {
        let p = (lower + (upper - lower) * t).clamp(0.0, 1.0);
        if k <= 0.0 {
            0.0
        } else if k - 1.0 >= n_int {
            1.0
        } else {
            crate::reliability::inc_beta(n_int - k + 1.0, k, 1.0 - p)
        }
    };
    // x-domain cut mapped to the beta t-domain (density is zero outside).
    let tc = ((c - lower) / (upper - lower)).clamp(0.0, 1.0);
    let p_tp = beta_weighted_integral(a, b, tc, 1.0, |t| 1.0 - fail_prob(t));
    let p_fp = beta_weighted_integral(a, b, 0.0, tc, |t| 1.0 - fail_prob(t));
    let p_ff = beta_weighted_integral(a, b, tc, 1.0, &fail_prob);
    let p_tf = beta_weighted_integral(a, b, 0.0, tc, &fail_prob);
    let p_ii_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| fail_prob(t).powi(2));
    let p_ij_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| fail_prob(t) * (1.0 - fail_prob(t)));
    let p_jj_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| (1.0 - fail_prob(t)).powi(2));
    let tot = p_ii_raw + 2.0 * p_ij_raw + p_jj_raw;
    if !(tot > 0.0) {
        return Err("consistency integrals degenerated to zero mass".into());
    }
    let p_ii = p_ii_raw / tot;
    let p_ij = p_ij_raw / tot;
    let p_jj = p_jj_raw / tot;
    let consistency = p_ii + p_jj;
    let chance_consistency = (p_ii + p_ij) * (p_ii + p_ij) + (p_ij + p_jj) * (p_ij + p_jj);
    // Conditional ratios are undefined when their margin (or the chance
    // correction) vanishes, e.g. a cut outside the fitted beta support;
    // return an explicit NaN rather than an unstable near-0/0 quotient.
    let ratio = |num: f64, den: f64| if den > 1e-12 { num / den } else { f64::NAN };
    Ok(LivingstonLewisResult {
        effective_test_length: etl,
        etl_rounded: nn,
        lower,
        upper,
        alpha: a,
        beta: b,
        used_two_parameter,
        p_tp,
        p_fp,
        p_tf,
        p_ff,
        accuracy: p_tp + p_tf,
        sensitivity: ratio(p_tp, p_tp + p_ff),
        specificity: ratio(p_tf, p_tf + p_fp),
        p_ii,
        p_ij,
        p_ji: p_ij,
        p_jj,
        consistency,
        chance_consistency,
        kappa: ratio(consistency - chance_consistency, 1.0 - chance_consistency),
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/classification_tests.rs"]
mod tests;
