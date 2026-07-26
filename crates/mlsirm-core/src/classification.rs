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
use crate::utility::bvn_upper;

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

/// Hanson-Brennan classification accuracy and consistency under the
/// four-parameter beta compound binomial model, for one cut score.
///
/// Result of [`hanson_brennan`] / [`hanson_brennan_from_params`].
/// Orientation is the crate's pass-positive convention (identical to
/// [`LivingstonLewisResult`]): pass = observed score `>= cut`. CRAN
/// betafunctions `HB.CA` labels *fail* as positive, so its sensitivity is
/// this struct's specificity and vice versa; accuracy, consistency, chance
/// consistency, and kappa are invariant under the relabeling.
#[derive(Debug, Clone)]
pub struct HansonBrennanResult {
    /// Lord's k (Hanson, 1991, Eq. 6). Echoes the input on the params path.
    pub lords_k: f64,
    /// True-score raw moments `m1..m4` (Hanson, 1991, Eqs. 7-8). `NaN` on
    /// the params path, which bypasses moment estimation.
    pub true_score_moments: [f64; 4],
    /// Lower location of the fitted four-parameter beta.
    pub lower: f64,
    /// Upper location of the fitted four-parameter beta.
    pub upper: f64,
    /// First beta shape parameter.
    pub alpha: f64,
    /// Second beta shape parameter.
    pub beta: f64,
    /// Whether the two-parameter `[0, 1]` fail-safe fit was used.
    pub used_two_parameter: bool,
    /// P(true pass and observed pass).
    pub p_tp: f64,
    /// P(true fail and observed pass).
    pub p_fp: f64,
    /// P(true fail and observed fail).
    pub p_tf: f64,
    /// P(true pass and observed fail).
    pub p_ff: f64,
    /// Classification accuracy `p_tp + p_tf`.
    pub accuracy: f64,
    /// `p_tp / (p_tp + p_ff)`; `NaN` when the margin vanishes.
    pub sensitivity: f64,
    /// `p_tf / (p_tf + p_fp)`; `NaN` when the margin vanishes.
    pub specificity: f64,
    /// P(fail on both replications), normalized.
    pub p_ii: f64,
    /// P(fail then pass), normalized.
    pub p_ij: f64,
    /// P(pass then fail) — equals `p_ij` by construction.
    pub p_ji: f64,
    /// P(pass on both replications), normalized.
    pub p_jj: f64,
    /// Classification consistency `p_ii + p_jj`.
    pub consistency: f64,
    /// Chance consistency `(p_ii + p_ij)^2 + (p_ij + p_jj)^2`.
    pub chance_consistency: f64,
    /// Cohen's kappa `(p - p_c) / (1 - p_c)`; `NaN` when `p_c == 1`.
    pub kappa: f64,
}

/// Binomial pmf `C(n, j) p^j (1-p)^(n-j)` via ln-gamma; zero outside
/// `0 <= j <= n` and exact at the `p` endpoints.
fn hb_binom_pmf(j: i64, n: i64, p: f64) -> f64 {
    if n < 0 || j < 0 || j > n {
        return 0.0;
    }
    if p <= 0.0 {
        return if j == 0 { 1.0 } else { 0.0 };
    }
    if p >= 1.0 {
        return if j == n { 1.0 } else { 0.0 };
    }
    let (nf, jf) = (n as f64, j as f64);
    let ln_c = crate::fitstats::ln_gamma(nf + 1.0)
        - crate::fitstats::ln_gamma(jf + 1.0)
        - crate::fitstats::ln_gamma(nf - jf + 1.0);
    (ln_c + jf * p.ln() + (nf - jf) * (1.0 - p).ln()).exp()
}

/// P(X <= cut-1 | n_items, k, p) under Lord's (1965, Eq. 5, as restated by
/// Hanson, 1991, Eq. 3 — Lord not read) two-term compound binomial.
///
/// DERIVED closed form (spec `hanson_brennan_spec.md`; the telescoping of
/// the correction partial sum was proven by exact polynomial identity in
/// the executed oracle for every fixture):
///
/// ```text
/// F(p) = BinCdf(cut-1; K, p)
///        - k p (1-p) [ b(cut-1; K-2, p) - b(cut-2; K-2, p) ]
/// ```
///
/// Raw two-term values are used exactly as CRAN `dcBinom`/`HB.CA` do: no
/// clamping of negative conditional masses, no clamping of `F` to `[0, 1]`,
/// and no renormalization (the full pmf sums to one identically).
fn hb_fail_cdf(cut: usize, n_items: usize, k: f64, p: f64) -> f64 {
    let kk = n_items as f64;
    let m = cut as f64 - 1.0; // BinCdf argument
    let base = if cut == 0 {
        0.0
    } else if m >= kk {
        1.0
    } else if p <= 0.0 {
        1.0
    } else if p >= 1.0 {
        0.0
    } else {
        // P(X <= m) = I_{1-p}(K - m, m + 1).
        crate::reliability::inc_beta(kk - m, m + 1.0, 1.0 - p)
    };
    let n2 = n_items as i64 - 2;
    let c = cut as i64;
    base - k * p * (1.0 - p) * (hb_binom_pmf(c - 1, n2, p) - hb_binom_pmf(c - 2, n2, p))
}

/// Shared index computation for both Hanson-Brennan entry points.
fn hb_indexes(
    n_items: usize,
    lords_k: f64,
    lower: f64,
    upper: f64,
    a: f64,
    b: f64,
    cut: usize,
    moments: [f64; 4],
    used_two_parameter: bool,
) -> Result<HansonBrennanResult, String> {
    let fail = |t: f64| -> f64 {
        let p = (lower + (upper - lower) * t).clamp(0.0, 1.0);
        hb_fail_cdf(cut, n_items, lords_k, p)
    };
    // x-domain truecut cut/K mapped to the beta t-domain.
    let tc = ((cut as f64 / n_items as f64 - lower) / (upper - lower)).clamp(0.0, 1.0);
    let p_tp = beta_weighted_integral(a, b, tc, 1.0, |t| 1.0 - fail(t));
    let p_fp = beta_weighted_integral(a, b, 0.0, tc, |t| 1.0 - fail(t));
    let p_ff = beta_weighted_integral(a, b, tc, 1.0, &fail);
    let p_tf = beta_weighted_integral(a, b, 0.0, tc, &fail);
    let p_ii_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| fail(t).powi(2));
    let p_ij_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| fail(t) * (1.0 - fail(t)));
    let p_jj_raw = beta_weighted_integral(a, b, 0.0, 1.0, |t| (1.0 - fail(t)).powi(2));
    let tot = p_ii_raw + 2.0 * p_ij_raw + p_jj_raw;
    if !(tot > 0.0) {
        return Err("consistency integrals degenerated to zero mass".into());
    }
    let p_ii = p_ii_raw / tot;
    let p_ij = p_ij_raw / tot;
    let p_jj = p_jj_raw / tot;
    let consistency = p_ii + p_jj;
    let chance_consistency = (p_ii + p_ij) * (p_ii + p_ij) + (p_ij + p_jj) * (p_ij + p_jj);
    let ratio = |num: f64, den: f64| if den > 1e-12 { num / den } else { f64::NAN };
    Ok(HansonBrennanResult {
        lords_k,
        true_score_moments: moments,
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

/// Hanson-Brennan classification indexes from fixed model parameters
/// (mirrors CRAN betafunctions `HB.CA` called with a parameter list).
///
/// `lords_k` is Lord's k, `(lower, upper, alpha, beta)` the four-parameter
/// beta true-score distribution, `cut` the raw cut score (pass = observed
/// `>= cut`). `n_items >= 2` suffices here — the data path's `>= 4` bound
/// is a moment-estimation requirement, not a model restriction.
///
/// # References
///
/// Haakstad, H. (2023). *betafunctions: Functions for working with two- and
/// four-parameter beta probability distributions* (Version 1.9.0)
/// \[R package\]. CRAN. <https://CRAN.R-project.org/package=betafunctions>
/// (`HB.CA`, `dcBinom` read line by line.)
///
/// Hanson, B. A. (1991). *Method of moments estimates for the
/// four-parameter beta compound binomial model and the calculation of
/// classification consistency indexes* (ACT Research Report 91-5; ERIC
/// ED344945). (Read: Eqs. 1-3 model, Eq. 6 Lord's k, Eqs. 7-8 moments,
/// Eqs. 9-13 beta fit.)
///
/// Hanson, B. A., & Brennan, R. L. (1990). An investigation of
/// classification consistency indexes estimated under alternative strong
/// true score models. *Journal of Educational Measurement, 27*(4), 345-359.
/// (As cited in Hanson, 1991; not read.)
///
/// Lord, F. M. (1965). A strong true-score theory, with applications.
/// *Psychometrika, 30*(3), 239-270. (As restated by Hanson, 1991; not
/// read.)
pub fn hanson_brennan_from_params(
    n_items: usize,
    lords_k: f64,
    lower: f64,
    upper: f64,
    alpha: f64,
    beta: f64,
    cut: usize,
) -> Result<HansonBrennanResult, String> {
    if n_items < 2 {
        return Err("n_items must be at least 2".into());
    }
    if cut < 1 || cut > n_items {
        return Err("cut must be in 1..=n_items".into());
    }
    if !lords_k.is_finite() {
        return Err("lords_k must be finite".into());
    }
    if !lower.is_finite() || !upper.is_finite() || lower < 0.0 || upper > 1.0 || lower >= upper {
        return Err("beta bounds must satisfy 0 <= lower < upper <= 1".into());
    }
    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        return Err("beta shape parameters must be positive and finite".into());
    }
    hb_indexes(
        n_items,
        lords_k,
        lower,
        upper,
        alpha,
        beta,
        cut,
        [f64::NAN; 4],
        false,
    )
}

/// Hanson-Brennan classification accuracy and consistency for one cut
/// score, from raw number-correct scores and a reliability estimate
/// (mirrors CRAN betafunctions `HB.CA` on raw data).
///
/// Pipeline (spec `hanson_brennan_spec.md` rev 2, adversarially verified;
/// every step checked line by line against the CRAN betafunctions 1.9.0
/// sources and Hanson, 1991):
///
/// 1. Lord's k from the observed mean, sample variance (`ddof = n-1`), and
///    error variance `s2 (1 - reliability)` (Hanson, 1991, Eq. 6;
///    betafunctions `Lords.k`).
/// 2. True-score raw moments `m1..m4` by the compound-binomial
///    factorial-moment recursion (Hanson, 1991, Eqs. 7-8; betafunctions
///    `HB.tsm`), with `ff(x, i) := 0` when `x < i` (betafunctions `dfac`).
/// 3. Four-parameter beta moment fit (Hanson, 1991, Eqs. 9-13), with a
///    fail-safe to the two-parameter `[0, 1]` fit when the 4P solution is
///    out of bounds or numerically invalid; `two_parameter = true` forces
///    the 2P fit (betafunctions `true.model = "2P"`).
/// 4. Accuracy/consistency cells by integrating the DERIVED fail-CDF
///    closed form (see [`hb_fail_cdf`]) against the fitted beta density.
///
/// Orientation: pass = observed `>= cut`; betafunctions `HB.CA` labels fail
/// as "positive", so its sensitivity is this function's specificity.
/// Negative Lord two-term conditional masses are used raw, exactly as
/// `HB.CA` integrates them (no clamping or renormalization).
///
/// # References
///
/// See [`hanson_brennan_from_params`].
pub fn hanson_brennan(
    scores: &[f64],
    n_items: usize,
    reliability: f64,
    cut: usize,
    two_parameter: bool,
) -> Result<HansonBrennanResult, String> {
    let n = scores.len();
    if n < 10 {
        return Err("at least 10 observed scores are required".into());
    }
    if n_items < 4 {
        return Err("n_items must be at least 4 for moment estimation".into());
    }
    if cut < 1 || cut > n_items {
        return Err("cut must be in 1..=n_items".into());
    }
    if !reliability.is_finite() || reliability <= 0.0 || reliability >= 1.0 {
        return Err("reliability must be in the open interval (0, 1)".into());
    }
    let kk = n_items as f64;
    if scores
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0 || *x > kk || x.fract() != 0.0)
    {
        return Err("scores must be integers in [0, n_items]".into());
    }
    let nf = n as f64;
    let mean = scores.iter().sum::<f64>() / nf;
    let s2 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    if s2 <= 0.0 {
        return Err("observed-score variance must be positive".into());
    }
    let s2e = s2 * (1.0 - reliability);
    let k_den = mean * (kk - mean) - (s2 - s2e);
    if !k_den.is_finite() || k_den.abs() < 1e-12 {
        return Err("Lord's k denominator vanishes; check scores/reliability".into());
    }
    let lords_k = kk * ((kk - 1.0) * (s2 - s2e) - kk * s2 + mean * (kk - mean)) / (2.0 * k_den);
    if !lords_k.is_finite() {
        return Err("Lord's k is not finite".into());
    }
    // True-score raw moments m1..m4 (Hanson, 1991, Eqs. 7-8).
    let mut m = [f64::NAN; 4];
    m[0] = mean / kk;
    for i in 2..=4usize {
        let r = i as u32;
        let mean_ff = scores
            .iter()
            .map(|x| {
                if *x < i as f64 {
                    0.0
                } else {
                    falling_factorial(*x, r)
                }
            })
            .sum::<f64>()
            / nf;
        let corr = lords_k * (i * (i - 1)) as f64;
        let den = kk * (kk - 1.0) + corr;
        if !den.is_finite() || den.abs() < 1e-12 {
            return Err("true-score moment recursion denominator vanishes".into());
        }
        m[i - 1] = (mean_ff / falling_factorial(kk - 2.0, r - 2) + corr * m[i - 2]) / den;
    }
    let m1 = m[0];
    let ts2 = m[1] - m1 * m1;
    if !(ts2 > 0.0) {
        return Err("estimated true-score variance is not positive".into());
    }
    // Beta fit: 4P with 2P fail-safe (identical structure to
    // livingston_lewis; Hanson, 1991, Eqs. 9-13).
    let mut used_two_parameter = true;
    let (mut a, mut b, mut lower, mut upper) = (f64::NAN, f64::NAN, 0.0, 1.0);
    if !two_parameter {
        let g3 = (m[2] - 3.0 * m1 * m[1] + 2.0 * m1.powi(3)) / ts2.powf(1.5);
        let g4 = (m[3] - 4.0 * m1 * m[2] + 6.0 * m1 * m1 * m[1] - 3.0 * m1.powi(4)) / (ts2 * ts2);
        let rr = 6.0 * (g4 - g3 * g3 - 1.0) / (6.0 + 3.0 * g3 * g3 - 2.0 * g4);
        let d = 1.0
            - 24.0 * (rr + 1.0) / ((rr + 2.0) * (rr + 3.0) * g4 - 3.0 * (rr - 6.0) * (rr + 1.0));
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
    }
    if used_two_parameter {
        if m1 <= 0.0 || m1 >= 1.0 {
            return Err("mean proportion score must be strictly inside (0, 1)".into());
        }
        let scale = m1 * (1.0 - m1) / ts2 - 1.0;
        a = m1 * scale;
        b = (1.0 - m1) * scale;
        lower = 0.0;
        upper = 1.0;
    }
    if !a.is_finite() || !b.is_finite() || a <= 0.0 || b <= 0.0 {
        return Err("beta true-score fit produced invalid shape parameters".into());
    }
    hb_indexes(
        n_items,
        lords_k,
        lower,
        upper,
        a,
        b,
        cut,
        m,
        used_two_parameter,
    )
}

/// Subkoviak (1976) single-administration coefficient-of-agreement output.
///
/// Citation governance:
/// - READ: Subkoviak, M. J. (1976). *Estimating reliability from a single
///   administration of a mastery test* (ERIC ED120229; AERA paper version of
///   Subkoviak, 1976, *Journal of Educational Measurement, 13*(4), 265-276).
/// - NOT READ, cited as-cited via Subkoviak (1976): Lord & Novick (1968) for
///   the binomial true-score model; Swaminathan, Hambleton, & Algina (1974)
///   for the two-administration p_o; Cohen (1960) for kappa.
pub struct SubkoviakResult {
    /// Reliability used in Eq. 16 (supplied, or KR-21 derived from the data).
    pub alpha: f64,
    /// Per-person regression estimate of the item-domain proportion
    /// (Subkoviak, 1976, Eq. 16): `alpha*(X_i/n) + (1-alpha)*(M/n)`.
    pub p_hat: Vec<f64>,
    /// Per-person coefficient of agreement P(i) (Eqs. 7 and 19).
    pub per_person: Vec<f64>,
    /// Group coefficient of agreement Pc = mean_i P(i) (Eqs. 5 and 20).
    pub agreement: f64,
    /// Chance agreement: sum over categories of the squared marginal
    /// category probability (Eqs. 9-10 and 21-22).
    pub chance_agreement: f64,
    /// Coefficient kappa `(Pc - Pchance) / (1 - Pchance)` (Eq. 11).
    pub kappa: f64,
}

/// Subkoviak's (1976) single-administration coefficient of agreement for
/// mastery classifications under the simple binomial true-score model.
///
/// `cuts` are the strictly increasing integer criteria `C_1 < .. < C_{h-1}`
/// (each in `1..=n_items`); category `j` is `{X : C_{j-1} <= X < C_j}` with
/// `C_0 = 0` and `C_h = n_items + 1`, so mastery at criterion `C` means
/// `X >= C`. VERIFIED against the READ source: the OCR of Eq. 4 prints
/// `X > C`, but Table 1 row 1 (`n = 5`, `C = 4`, `p = .19`) prints `.0055`,
/// which matches `P(X >= 4) = 5(.19)^4(.81) + (.19)^5 = .005526` and not
/// `P(X > 4) = (.19)^5 = .000248`; the two-administration example's
/// exception students likewise require `>=`.
///
/// `alpha = None` derives Kuder-Richardson Formula 21 with the population
/// (ddof = 0) variance, clamped to `[0, 1]`:
/// `a21 = (n/(n-1)) (1 - M(n-M)/(n S^2))`. VERIFIED against the paper's
/// real-data example `(25/24)(1 - 17.40*7.60/(25*5.14)) ≈ 0` (negative,
/// treated as zero), which requires `S^2 = 5.14` as printed. DISCLOSED
/// IRREPRODUCIBILITY: Table 1's footnote value `a21 = .58` does not follow
/// from its own printed `S^2 = 2.61` (which gives `19/29 ≈ .6552`); the
/// printed p-hat column is exactly consistent with `alpha = .58`, so
/// reproducing Table 1 requires supplying `alpha = 0.58` explicitly.
///
/// Category probabilities use the simple binomial (Eq. 8). The compound
/// binomial refinement (Eqs. 12-14) and Lord's (1959) distribution-free
/// p-hat (Eq. 17) are EXCLUDED: both defer to sources not read (Lord &
/// Novick, 1968, pp. 524-526; Lord, 1959). Callers may pass a KR-20-style
/// reliability through `alpha` per the paper's remark that the procedure
/// is analogous.
///
/// All exposed metrics (P(i), Pc, Pchance, kappa) are invariant under a
/// consistent permutation of category labels, and `p_hat` does not depend
/// on the categories at all; a label-permutation mutation is therefore
/// unobservable through this API and behaviorally irrelevant.
pub fn subkoviak_agreement(
    scores: &[f64],
    n_items: usize,
    cuts: &[f64],
    alpha: Option<f64>,
) -> Result<SubkoviakResult, String> {
    if n_items < 2 {
        return Err("n_items must be at least 2".into());
    }
    let n_persons = scores.len();
    if n_persons < 2 {
        return Err("at least 2 observed scores are required".into());
    }
    let nf = n_items as f64;
    if scores
        .iter()
        .any(|x| !x.is_finite() || *x < 0.0 || *x > nf || x.fract() != 0.0)
    {
        return Err("scores must be integers in [0, n_items]".into());
    }
    if cuts.is_empty() {
        return Err("cuts must be nonempty".into());
    }
    if cuts
        .iter()
        .any(|c| !c.is_finite() || c.fract() != 0.0 || *c < 1.0 || *c > nf)
    {
        return Err("cuts must be integers in 1..=n_items".into());
    }
    if cuts.windows(2).any(|w| w[1] <= w[0]) {
        return Err("cuts must be strictly increasing".into());
    }
    let np = n_persons as f64;
    let mean = scores.iter().sum::<f64>() / np;
    let alpha = match alpha {
        Some(a) => {
            if !a.is_finite() || !(0.0..=1.0).contains(&a) {
                return Err("alpha must be finite and in [0, 1]".into());
            }
            a
        }
        None => {
            // KR-21 with population (ddof = 0) variance; see doc comment.
            let s2 = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / np;
            if s2 <= 0.0 {
                return Err("observed-score variance is zero; supply alpha explicitly".into());
            }
            let a21 = nf / (nf - 1.0) * (1.0 - mean * (nf - mean) / (nf * s2));
            a21.clamp(0.0, 1.0)
        }
    };
    // Category boundaries: C_0 = 0, user cuts, C_h = n_items + 1.
    let mut bounds = Vec::with_capacity(cuts.len() + 2);
    bounds.push(0i64);
    bounds.extend(cuts.iter().map(|c| *c as i64));
    bounds.push(n_items as i64 + 1);
    let n_cats = bounds.len() - 1;
    let p_hat: Vec<f64> = scores
        .iter()
        .map(|x| alpha * (x / nf) + (1.0 - alpha) * (mean / nf))
        .collect();
    let mut per_person = Vec::with_capacity(n_persons);
    let mut q_bar = vec![0.0f64; n_cats];
    for &p in &p_hat {
        let mut p_i = 0.0;
        for j in 0..n_cats {
            let q_ij: f64 = (bounds[j]..bounds[j + 1])
                .map(|x| hb_binom_pmf(x, n_items as i64, p))
                .sum();
            p_i += q_ij * q_ij;
            q_bar[j] += q_ij / np;
        }
        per_person.push(p_i);
    }
    let agreement = per_person.iter().sum::<f64>() / np;
    let chance_agreement = q_bar.iter().map(|q| q * q).sum::<f64>();
    let denom = 1.0 - chance_agreement;
    if denom <= 1e-12 {
        return Err("chance agreement is 1; kappa is undefined (all mass in one category)".into());
    }
    let kappa = (agreement - chance_agreement) / denom;
    Ok(SubkoviakResult {
        alpha,
        p_hat,
        per_person,
        agreement,
        chance_agreement,
        kappa,
    })
}

/// Result of Livingston's (1972) criterion-referenced reliability analysis.
pub struct LivingstonResult {
    /// Population (ddof = 0) mean of the observed scores.
    pub mean: f64,
    /// Population (ddof = 0) variance of the observed scores.
    pub var: f64,
    /// Mean squared deviation from the criterion:
    /// `D^2(X) = var + (mean - cut)^2` (Livingston, 1972, Table 1).
    pub msd: f64,
    /// `k^2` at each requested test-length multiplier in `n_lengths`
    /// (Spearman-Brown projected; `n = 1` is the unlengthened value).
    pub k2: Vec<f64>,
}

/// Livingston's (1972) criterion-referenced reliability coefficient `k^2`.
///
/// Source READ: Livingston, S. A. (1972). *A classical test-theory approach
/// to criterion-referenced tests.* AERA paper, ERIC ED069624 (OCR). The
/// published journal version (Livingston, 1972, *Journal of Educational
/// Measurement, 9*(1), 13-26, ERIC EJ053921) was NOT read (abstract only)
/// and is cited only as the archival venue.
///
/// Table 1 of the read source defines the criterion-referenced analogues by
/// replacing central moments with moments about the criterion score `C`:
/// `D^2(X) = E[(X - C_x)^2]` and `D(X,Y) = E[(X - C_x)(Y - C_y)]`. The
/// conversion form implemented here,
///
/// `k^2(X, T) = [rho^2 sigma^2(X) + (mu - C)^2] / [sigma^2(X) + (mu - C)^2]`,
///
/// is an ALGEBRAIC RECONSTRUCTION from those expectation definitions
/// (`E[(X-C)^2] = sigma^2 + (mu-C)^2`), supported by the Table 3/prose
/// discussion; the Table 3 OCR itself is too noisy to serve as a clean
/// symbol-level transcription. `reliability` is the norm-referenced
/// reliability `rho^2(X, T)` in `[0, 1]`, supplied by the caller (the paper's
/// conversion form takes it as input; this function does not estimate it).
///
/// Properties pinned by tests from the read source (pp. 3-5): `k^2 >= rho^2`
/// with equality iff `mu = C`, and a zero-variance group with `mu != C` has
/// `k^2 = 1`. `k^2` is NaN only in the exact degenerate case of scores all
/// exactly equal to `cut` (`D^2` exactly zero); the check compares elements
/// to the cut directly, so it is not defeated by rounding in the summed mean
/// (e.g. constant `0.1` scores at cut `0.1`), and no numerical tolerance
/// widens it. A huge finite `cut` whose squared offset overflows to infinity
/// returns the formula limit `1` instead of `inf/inf` NaN.
///
/// Each entry of `n_lengths` applies the Spearman-Brown step from Table 2 to
/// `k^2` itself: `k^2(n) = n k^2 / (1 + (n - 1) k^2)` (the source states the
/// formula "works exactly the same way" for a test `n` times as long).
/// DISCLOSED EXTRAPOLATION: the source's wording covers integer multiples;
/// accepting positive fractional `n` is a continuous Spearman-Brown
/// projection beyond the literal source wording. `Spearman-Brown of NaN is
/// NaN.`
pub fn livingston_k2(
    scores: &[f64],
    cut: f64,
    reliability: f64,
    n_lengths: &[f64],
) -> Result<LivingstonResult, String> {
    if scores.len() < 2 {
        return Err("at least 2 observed scores are required".into());
    }
    if scores.iter().any(|x| !x.is_finite()) {
        return Err("scores must be finite".into());
    }
    if !cut.is_finite() {
        return Err("cut must be finite".into());
    }
    if !reliability.is_finite() || !(0.0..=1.0).contains(&reliability) {
        return Err("reliability must be finite and in [0, 1]".into());
    }
    if n_lengths.is_empty() {
        return Err("n_lengths must be nonempty".into());
    }
    if n_lengths.iter().any(|n| !n.is_finite() || *n <= 0.0) {
        return Err("n_lengths must be finite and positive".into());
    }
    let np = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / np;
    let var = scores.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / np;
    let off2 = (mean - cut).powi(2);
    let msd = var + off2;
    let base = if scores.iter().all(|&x| x == cut) || (var == 0.0 && mean == cut) {
        // Exact degeneracy: D^2 is mathematically zero. The element-wise
        // check avoids false negatives from rounding in the summed mean.
        f64::NAN
    } else if off2 == f64::INFINITY && var.is_finite() {
        // Squared criterion offset overflowed; the formula limit is 1.
        1.0
    } else {
        (reliability * var + off2) / msd
    };
    let k2 = n_lengths
        .iter()
        .map(|&n| n * base / (1.0 + (n - 1.0) * base))
        .collect();
    Ok(LivingstonResult { mean, var, msd, k2 })
}

/// Livingston's (1972) criterion-referenced correlation `k(X, Y)`.
///
/// Source READ: ERIC ED069624 (see [`livingston_k2`]). Table 1 defines
/// `D(X, Y) = E[(X - C_x)(Y - C_y)] = sigma(X, Y) + (mu_x - C_x)(mu_y - C_y)`
/// and `k(X, Y) = D(X, Y) / sqrt(D^2(X) D^2(Y))` with population (ddof = 0)
/// moments. The read source's Figures 3-4 discussion shows the coefficient
/// can differ in sign and magnitude from the norm-referenced correlation;
/// the test anchors exercising sign flips and asymmetric offsets are
/// formula-derived adversarial pins, not reproductions of a printed figure.
///
/// Returns NaN when either `D^2` is exactly zero (constant scores exactly at
/// their criterion, detected element-wise so summed-mean rounding cannot
/// defeat it); no numerical tolerance widens this. Errs when a squared
/// criterion offset overflows f64 (no clean finite limit exists for the
/// correlation, unlike [`livingston_k2`]).
pub fn livingston_correlation(x: &[f64], y: &[f64], cut_x: f64, cut_y: f64) -> Result<f64, String> {
    if x.len() < 2 {
        return Err("at least 2 observed scores are required".into());
    }
    if x.len() != y.len() {
        return Err("x and y must have the same length".into());
    }
    if x.iter().chain(y.iter()).any(|v| !v.is_finite()) {
        return Err("scores must be finite".into());
    }
    if !cut_x.is_finite() || !cut_y.is_finite() {
        return Err("cuts must be finite".into());
    }
    let np = x.len() as f64;
    let mx = x.iter().sum::<f64>() / np;
    let my = y.iter().sum::<f64>() / np;
    let cov = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| (a - mx) * (b - my))
        .sum::<f64>()
        / np;
    let vx = x.iter().map(|v| (v - mx).powi(2)).sum::<f64>() / np;
    let vy = y.iter().map(|v| (v - my).powi(2)).sum::<f64>() / np;
    let d2x = vx + (mx - cut_x).powi(2);
    let d2y = vy + (my - cut_y).powi(2);
    if x.iter().all(|&v| v == cut_x) || y.iter().all(|&v| v == cut_y) || d2x == 0.0 || d2y == 0.0 {
        return Ok(f64::NAN);
    }
    if !d2x.is_finite() || !d2y.is_finite() {
        return Err("criterion offset too large: squared deviation overflows f64".into());
    }
    Ok((cov + (mx - cut_x) * (my - cut_y)) / (d2x * d2y).sqrt())
}

/// Result of Woodruff & Sawyer's (1988) pass-fail reliability estimation
/// from parallel half-tests (both the split-half/Spearman-Brown method and
/// the bivariate-normal method).
pub struct WoodruffSawyerResult {
    /// Estimated full-test pass rate `p` (from smoothed half-test margins
    /// for the SB method; `1 - Phi(K_q)` for the normal method).
    pub pass_rate: f64,
    /// Half-test agreement coefficient `phi` (eq. 4 of the read source).
    /// NaN for the normal method (not defined there).
    pub phi_half: f64,
    /// Half-test raw agreement `theta = pi00 + pi11`. NaN for the normal
    /// method.
    pub theta_half: f64,
    /// Full-test-length coefficient `phi*` (Spearman-Brown stepped-up for
    /// the SB method, eq. 5; tetrachoric-style BVN value for the normal
    /// method). Equals Cohen's kappa for the 2x2 pass-fail table.
    pub phi: f64,
    /// Full-test-length agreement `theta* = pi*00 + pi*11` (eq. 8).
    pub theta: f64,
    /// Full-test joint fail-fail proportion `pi*00`.
    pub pi00: f64,
    /// Full-test off-diagonal proportion `pi*01` (= `pi*10` by the
    /// parallel-forms symmetry the method imposes).
    pub pi01: f64,
    /// Full-test joint pass-pass proportion `pi*11`.
    pub pi11: f64,
}

/// Woodruff & Sawyer's (1988) split-half / Spearman-Brown estimator of
/// pass-fail reliability (`phi*`, Cohen's kappa) and raw agreement
/// (`theta*`) for a full-length test, from a 2x2 half-test pass-fail table.
///
/// Source READ: Woodruff, D. J., & Sawyer, R. L. (1988). *Estimating
/// measures of pass-fail reliability from parallel half-tests.* AERA paper,
/// ERIC ED292877 (OCR). NOT READ (cited only as cited therein): Huynh
/// (1976); Peng & Subkoviak (1980); Huynh & Sanders (1980); Brennan (1981,
/// ACT TB-38); Cohen (1960); Hambleton & Novick (1973); Swaminathan,
/// Hambleton, & Algina (1974); Subkoviak (1978, 1984).
///
/// `counts = [n00, n01, n10, n11]` where category 0 = fail and 1 = pass on
/// half-tests (X1, X2); `n01` counts fail-on-X1/pass-on-X2. Proportions are
/// normalized by the total count. The parallel-forms symmetrization smooths
/// only the off-diagonal, `pi01_s = (pi01 + pi10) / 2` (source p. 7), so
/// the margins `p = pi01_s + pi11`, `q = 1 - p` are symmetric.
///
/// Half-test coefficient (eq. 4): `phi = 1 - pi01_s / (p q)`; raw
/// agreement `theta = pi00 + pi11` (raw diagonal — unchanged by the
/// off-diagonal smoothing). Full-length step-up (eq. 5) is Spearman-Brown
/// on `phi`: `phi* = 2 phi / (1 + phi)`; the equivalent single-expression
/// form `phi* = 1 - pi01_s / (2 p q - pi01_s)` is a DERIVED algebraic
/// identity (verified by exact rational arithmetic in the session oracle,
/// not printed in the source). The full-length table is reconstructed as
/// `pi*11 = p q phi* + p^2`, `pi*00 = p q phi* + q^2`,
/// `pi*01 = p q (1 - phi*)`, giving `theta* = 2 p q phi* + p^2 + q^2`
/// (eq. 8). That the three cells sum to 1 (with `pi*01` counted twice) is a
/// DERIVED check, also oracle-verified.
///
/// CAVEAT (source pp. 9-10): the source reports `phi*` from this method to
/// be positively biased relative to `phi` estimated from two full-length
/// forms when the halves are not strictly parallel; treat `phi*` as an
/// upper-bound-flavored estimate under half-test parallelism violations.
///
/// `phi` may legitimately be negative (worse-than-chance agreement) and is
/// passed through. Errors: fewer/more than 4 counts (caller contract),
/// negative or non-finite counts, non-finite or zero total, a margin `p` or
/// `q` equal to 0 (phi undefined), and `2 p q == pi01_s` (the `phi = -1`
/// Spearman-Brown singularity).
pub fn woodruff_sawyer_sb(counts: &[f64]) -> Result<WoodruffSawyerResult, String> {
    if counts.len() != 4 {
        return Err("counts must have exactly 4 entries [n00, n01, n10, n11]".into());
    }
    if counts.iter().any(|&c| !c.is_finite() || c < 0.0) {
        return Err("counts must be finite and non-negative".into());
    }
    let total: f64 = counts.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return Err("total count must be finite and positive".into());
    }
    let pi00 = counts[0] / total;
    let pi01 = counts[1] / total;
    let pi10 = counts[2] / total;
    let pi11 = counts[3] / total;
    let pi01_s = 0.5 * (pi01 + pi10);
    let p = pi01_s + pi11;
    let q = 1.0 - p;
    if p <= 0.0 || q <= 0.0 {
        return Err("a smoothed pass/fail margin is zero; phi is undefined".into());
    }
    let pq = p * q;
    let phi_half = 1.0 - pi01_s / pq;
    let theta_half = pi00 + pi11;
    let denom = 2.0 * pq - pi01_s;
    if denom <= 0.0 {
        return Err("phi = -1 singularity: 2*p*q equals the smoothed off-diagonal".into());
    }
    let phi = 1.0 - pi01_s / denom;
    let pi01_star = pq * (1.0 - phi);
    let pi00_star = pq * phi + q * q;
    let pi11_star = pq * phi + p * p;
    let theta = 2.0 * pq * phi + p * p + q * q;
    Ok(WoodruffSawyerResult {
        pass_rate: p,
        phi_half,
        theta_half,
        phi,
        theta,
        pi00: pi00_star,
        pi01: pi01_star,
        pi11: pi11_star,
    })
}

/// Woodruff & Sawyer's (1988) bivariate-normal estimator of pass-fail
/// reliability for a full-length test from a half-test correlation.
///
/// Source READ: ERIC ED292877 (see [`woodruff_sawyer_sb`]); same NOT-READ
/// list. The half-test product-moment correlation `r_half` is stepped up to
/// full length by Spearman-Brown, `r_SB = 2 r / (1 + r)`, and scores on two
/// parallel full-length forms are modeled as bivariate normal with common
/// mean/sd and correlation `r_SB` (source pp. 7-8). With standardized cut
/// `K_q = (cut - mean) / sd`, the fail rate is `q = Phi(K_q)` (LOWER tail;
/// the source's Table 4 uses fail proportions) and
/// `pi*00 = P[Z1 <= K_q, Z2 <= K_q; r_SB]`, evaluated via this crate's
/// upper-tail BVN quadrature through the central-symmetry identity
/// `P[Z1 <= a, Z2 <= a] = P[Z1 > -a, Z2 > -a]` (DERIVED, standard BVN
/// symmetry; oracle-verified against mpmath). Then `pi*01 = q - pi*00`,
/// `pi*11 = p - pi*01`, `theta* = pi*00 + pi*11`, and
/// `phi* = 1 - pi*01 / (p q)` (kappa for the symmetric 2x2 table).
/// `phi_half`/`theta_half` are NaN: the source defines no half-test
/// agreement quantities on this path.
///
/// Errors: non-finite mean/cut, sd not finite and positive, `r_half`
/// outside `[-1, 1]` or non-finite, `r_SB` non-finite or not strictly
/// inside `(-1, 1)` (note `r_half < -1/3` maps below -1 and `r_half = 1`
/// maps to 1), `sqrt(1 - r_SB^2) < 1e-4` (the [`bvn_upper`] caller
/// contract), non-finite `K_q` (e.g. tiny sd overflow), and `q` or `p`
/// rounding to exactly 0 or 1 (cut outside the resolvable score range).
pub fn woodruff_sawyer_normal(
    mean: f64,
    sd: f64,
    cut: f64,
    r_half: f64,
) -> Result<WoodruffSawyerResult, String> {
    if !mean.is_finite() || !cut.is_finite() {
        return Err("mean and cut must be finite".into());
    }
    if !sd.is_finite() || sd <= 0.0 {
        return Err("sd must be finite and positive".into());
    }
    if !r_half.is_finite() || !(-1.0..=1.0).contains(&r_half) {
        return Err("r_half must be finite and in [-1, 1]".into());
    }
    let r_sb = 2.0 * r_half / (1.0 + r_half);
    if !r_sb.is_finite() || r_sb <= -1.0 || r_sb >= 1.0 {
        return Err(format!(
            "Spearman-Brown stepped-up correlation r_SB = {r_sb} must be strictly inside (-1, 1)"
        ));
    }
    if (1.0 - r_sb * r_sb).sqrt() < 1e-4 {
        return Err("r_SB too close to +/-1 for the BVN quadrature (sqrt(1-rho^2) < 1e-4)".into());
    }
    let kq = (cut - mean) / sd;
    if !kq.is_finite() {
        return Err("standardized cut (cut - mean)/sd is not finite".into());
    }
    let q = phi(kq);
    let p = 1.0 - q;
    if q <= 0.0 || q >= 1.0 || p <= 0.0 || p >= 1.0 {
        return Err(
            "cut is outside the resolvable score range (fail rate rounds to 0 or 1)".into(),
        );
    }
    // P[Z1 <= Kq, Z2 <= Kq] = P[Z1 > -Kq, Z2 > -Kq] by central symmetry.
    let pi00 = bvn_upper(-kq, -kq, r_sb);
    let pi01 = q - pi00;
    let pi11 = p - pi01;
    let theta = pi00 + pi11;
    let phi_star = 1.0 - pi01 / (p * q);
    Ok(WoodruffSawyerResult {
        pass_rate: p,
        phi_half: f64::NAN,
        theta_half: f64::NAN,
        phi: phi_star,
        theta,
        pi00,
        pi01,
        pi11,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/classification_tests.rs"]
mod tests;
