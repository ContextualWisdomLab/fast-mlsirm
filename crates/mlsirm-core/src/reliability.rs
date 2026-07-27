//! Guttman (1945) lower-bound reliability coefficients (lambda 1-6) with the
//! split-half machinery of Revelle's `psych` package.
//!
//! # Verified sources
//!
//! - **READ IN FULL** (the oracle): psych 2.6.5 (Revelle, 2025) R sources
//!   `R/guttman.R`, `R/splitHalf.R`, `R/smc.R`, fetched from CRAN. Every
//!   formula below is traced to a line of that code.
//! - **NOT read**: Guttman (1945) itself (paywalled at verification time).
//!   Attribution is therefore "Guttman (1945), as implemented in psych
//!   2.6.5"; no claim is made about the primary text beyond what psych
//!   implements.
//!
//! # Formulas (on the item Pearson correlation matrix `R`, `p` items)
//!
//! With `Vt = sum(R)` (all cells), `sum_off = Vt - p` (`tr(R) = p`),
//! `sumsq_off = sum of squared off-diagonals`:
//!
//! - `lambda1 = 1 - p / Vt`                              (guttman.R line 78)
//! - `lambda2 = (sum_off + sqrt(sumsq_off * p/(p-1))) / Vt`        (line 84)
//! - `lambda3 = p/(p-1) * lambda1` ( = coefficient alpha)          (line 85)
//! - `lambda5 = lambda1 + 2*sqrt(max_j C_j)/Vt` with
//!   `C_j = sum_{i != j} R_ij^2` (`colSums(r^2) - diag(r^2)`, lines 89-91)
//! - `lambda6 = (sum_off + sum_j smc_j) / Vt` with
//!   `smc_j = 1 - 1/[R^{-1}]_jj` clamped to `[0, 1]` (smc.R lines 57, 68-71;
//!   guttman.R line 87 — `sum.r - tr(R) = sum_off`, same expression)
//! - **Split halves** (splitHalf.R): items are split into subset A of size
//!   `m = floor(p/2)` and complement B; for each split
//!   `rb = |4 * S_AB / Vt|` where `S_AB = sum_{i in A, j in B} R_ij`
//!   (line 17: `rab = 4*R[1,2]/sum(R)`; the 2x2 collapsed matrix satisfies
//!   `R11 + R22 + 2 R12 = Vt` because A and B partition the items).
//!   `lambda4 = max rb`, `beta = max(min rb, 0)` (guttman.R lines 121-122),
//!   `mean_split = mean rb`. All `C(p, m)` subsets are enumerated when that
//!   count fits the `n_sample_splits` budget (psych brute-forces at
//!   `<= 15000`, splitHalf.R lines 77-78); otherwise `n_sample_splits`
//!   random subsets are drawn.
//!
//! # Deliberate divergences from psych (all verified against the R source)
//!
//! 1. No `check.keys` auto-reversal of negatively keyed items (splitHalf.R
//!    lines 34-38 call `principal()`; factor analysis is out of scope).
//!    Supply keyed data; for negatively keyed inputs `lambda4`/`beta` are
//!    NOT psych-parity because psych may auto-reverse.
//! 2. `lambda5p`, `alpha.pc`, `r.pc`, `beta.pc`, `glb`, `tenberge` are not
//!    computed (they need `fa`/`glb.fa`, out of scope).
//! 3. The sampled branch draws subsets from the crate LCG stream, not R's
//!    `sample()` — not bit-identical to psych. The exhaustive branch is
//!    deterministic and directly comparable.
//! 4. SMC uses a plain Gauss-Jordan inverse and returns an error on a
//!    singular correlation matrix; psych uses a pseudo-inverse (`Pinv`) and
//!    silently degrades. An exactly singular `R` (e.g. duplicate items) is
//!    an input problem this crate refuses rather than papers over.
//! 5. `|rb|` is taken in BOTH branches. psych's exhaustive branch takes
//!    `abs()` (splitHalf.R lines 85, 107) but its sampled branch does not
//!    (lines 118-128); we prefer internal consistency over replicating that
//!    asymmetry.
//! 6. Sampled subsets may repeat (psych's `sample()` loop has the same
//!    property).
//!
//! # References
//!
//! Guttman, L. (1945). A basis for analyzing test-retest reliability.
//! *Psychometrika, 10*(4), 255-282. (As implemented in psych 2.6.5; primary
//! text not consulted.)
//!
//! Revelle, W. (2025). *psych: Procedures for psychological, psychometric,
//! and personality research* (Version 2.6.5) [R package]. CRAN.
use crate::parallel::{correlation_matrix, lcg_uniform};

/// Guttman lambda coefficients and split-half summaries.
#[derive(Debug, Clone)]
pub struct GuttmanResult {
    pub lambda1: f64,
    pub lambda2: f64,
    pub lambda3: f64,
    /// Maximum absolute split-half reliability over the evaluated splits.
    pub lambda4: f64,
    pub lambda5: f64,
    pub lambda6: f64,
    /// Worst (minimum) split-half, floored at 0 (guttman.R line 122).
    pub beta: f64,
    /// Mean absolute split-half over the evaluated splits.
    pub mean_split: f64,
    /// Number of splits evaluated.
    pub n_splits: usize,
    /// `true` when all `C(p, floor(p/2))` subsets were enumerated.
    pub exhaustive: bool,
}

/// Guttman lambda 1-6 reliability bounds for a row-major `n_persons x
/// n_items` data matrix (complete, finite; see the module docs for the
/// formula provenance and divergences from psych).
///
/// `n_sample_splits` is the split-evaluation budget: when
/// `C(p, floor(p/2))` exceeds it, that many random splits are sampled from
/// the crate's deterministic LCG stream seeded with `seed.max(1)`.
pub fn guttman_lambdas(
    data: &[f64],
    n_persons: usize,
    n_items: usize,
    n_sample_splits: usize,
    seed: u64,
) -> Result<GuttmanResult, String> {
    if n_persons < 3 {
        return Err("guttman_lambdas needs n_persons >= 3".into());
    }
    if n_items < 3 {
        // psych's guttman() stops below 3 items (guttman.R lines 29-30).
        return Err("guttman_lambdas needs n_items >= 3".into());
    }
    if n_sample_splits == 0 {
        return Err("n_sample_splits must be >= 1".into());
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("data dimensions overflow usize")?;
    if data.len() != cells {
        return Err(format!(
            "data length {} does not match n_persons * n_items = {cells}",
            data.len()
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data must be finite (no NaN/inf; complete data required)".into());
    }

    let p = n_items;
    let r = correlation_matrix(data, n_persons, p)?;
    let vt: f64 = r.iter().sum();
    if !vt.is_finite() || vt <= 0.0 {
        return Err(format!(
            "sum of the correlation matrix is {vt}; total-score variance must be positive"
        ));
    }
    let sum_off = vt - p as f64; // tr(R) = p exactly
    let sumsq_off: f64 = (0..p)
        .flat_map(|i| (0..p).map(move |j| (i, j)))
        .filter(|(i, j)| i != j)
        .map(|(i, j)| r[i * p + j] * r[i * p + j])
        .sum();
    let pm1 = (p - 1) as f64;

    let lambda1 = 1.0 - p as f64 / vt;
    let lambda2 = (sum_off + (sumsq_off * p as f64 / pm1).sqrt()) / vt;
    let lambda3 = p as f64 / pm1 * lambda1;

    // lambda5: column sums of squared off-diagonals (guttman.R lines 89-91).
    let mut c_max = f64::NEG_INFINITY;
    for j in 0..p {
        let cj: f64 = (0..p)
            .filter(|&i| i != j)
            .map(|i| r[i * p + j] * r[i * p + j])
            .sum();
        c_max = c_max.max(cj);
    }
    let lambda5 = lambda1 + 2.0 * c_max.sqrt() / vt;

    // lambda6: squared multiple correlations from the inverse diagonal,
    // clamped to [0, 1] as psych's smc() does (smc.R lines 57, 68-71).
    let rinv = invert_symmetric(&r, p)
        .map_err(|e| format!("{e}; lambda6 (SMC) requires an invertible correlation matrix"))?;
    let sum_smc: f64 = (0..p)
        .map(|j| (1.0 - 1.0 / rinv[j * p + j]).clamp(0.0, 1.0))
        .sum();
    let lambda6 = (sum_off + sum_smc) / vt;

    // Split halves.
    let m = p / 2;
    let count = binomial(p, m);
    let mut max_rb = f64::NEG_INFINITY;
    let mut min_rb = f64::INFINITY;
    let mut sum_rb = 0.0_f64;
    let mut n_splits = 0_usize;
    let exhaustive = count <= n_sample_splits as u128;
    if exhaustive {
        // Lexicographic enumeration of all m-subsets of 0..p.
        let mut idx: Vec<usize> = (0..m).collect();
        loop {
            let rb = split_rb(&r, p, vt, &idx);
            max_rb = max_rb.max(rb);
            min_rb = min_rb.min(rb);
            sum_rb += rb;
            n_splits += 1;
            // next combination
            let mut i = m;
            loop {
                if i == 0 {
                    break;
                }
                i -= 1;
                if idx[i] != i + p - m {
                    idx[i] += 1;
                    for k in (i + 1)..m {
                        idx[k] = idx[k - 1] + 1;
                    }
                    break;
                }
            }
            if idx[0] == p - m && (1..m).all(|k| idx[k] == p - m + k) && n_splits as u128 == count {
                break;
            }
            if n_splits as u128 > count {
                return Err("split enumeration overran the binomial count (internal bug)".into());
            }
        }
    } else {
        let mut state = seed.max(1);
        let mut idx: Vec<usize> = (0..p).collect();
        for _ in 0..n_sample_splits {
            // Partial Fisher-Yates: first m entries become subset A.
            for (i, item) in idx.iter_mut().enumerate() {
                *item = i;
            }
            for i in 0..m {
                let u = lcg_uniform(&mut state);
                let j = (i + (u * (p - i) as f64) as usize).min(p - 1);
                idx.swap(i, j);
            }
            let mut a: Vec<usize> = idx[..m].to_vec();
            a.sort_unstable();
            let rb = split_rb(&r, p, vt, &a);
            max_rb = max_rb.max(rb);
            min_rb = min_rb.min(rb);
            sum_rb += rb;
            n_splits += 1;
        }
    }

    Ok(GuttmanResult {
        lambda1,
        lambda2,
        lambda3,
        lambda4: max_rb,
        lambda5,
        lambda6,
        beta: min_rb.max(0.0),
        mean_split: sum_rb / n_splits as f64,
        n_splits,
        exhaustive,
    })
}

/// ten Berge & Zegers mu coefficient series (mu0..mu3).
#[derive(Debug, Clone)]
pub struct TenBergeResult {
    /// mu0 = coefficient alpha (equals Guttman lambda3 exactly).
    pub mu0: f64,
    /// mu1 (equals Guttman lambda2 exactly).
    pub mu1: f64,
    pub mu2: f64,
    pub mu3: f64,
}

/// ten Berge & Zegers (1978) mu0-mu3 reliability lower bounds for a
/// row-major `n_persons x n_items` data matrix (complete, finite).
///
/// Transcribed from psych 2.6.5 `tenberge.R` lines 4-12, read line by line
/// (Revelle, 2025); ten Berge & Zegers (1978), *Psychometrika, 43*(4),
/// 575-579, https://doi.org/10.1007/BF02293811, was NOT read — attribution
/// is "as cited in / as implemented by Revelle (2025)". With `Vt = sum(R)`,
/// `S_k = sum_{i != j} R_ij^k`, and `c = p/(p-1)` applied to the INNERMOST
/// radical only:
///
/// ```text
/// mu0 = c * S_1 / Vt                                (tenberge.R line 9)
/// mu1 = (S_1 + sqrt(c * S_2)) / Vt                  (line 10)
/// mu2 = (S_1 + sqrt(S_2 + sqrt(c * S_4))) / Vt      (line 11)
/// mu3 = (S_1 + sqrt(S_2 + sqrt(S_4 + sqrt(c * S_8)))) / Vt   (line 12)
/// ```
///
/// Verified identities (disclosed, pinned independently in tests):
/// `mu0 == guttman lambda3` (alpha; since `S_1 = Vt - p`) and
/// `mu1 == guttman lambda2` (character-identical formula in guttman.R).
/// `mu0 <= mu1 <= mu2 <= mu3` holds for every valid correlation matrix by
/// Cauchy-Schwarz over the `p*(p-1)` off-diagonal cells
/// (`sqrt(c * S_{2k}) >= S_k / (p-1)` step-wise) given `Vt > 0`.
///
/// Divergences from psych (deliberate): raw-data input only (no
/// correlation-matrix passthrough via the fragile `dim[1] > n` heuristic,
/// no `use = "pairwise"`); hard errors on degenerate input instead of NA
/// propagation. `S_1` is summed directly over off-diagonal cells rather
/// than as `Vt - p` to avoid cancellation.
pub fn tenberge_mu(
    data: &[f64],
    n_persons: usize,
    n_items: usize,
) -> Result<TenBergeResult, String> {
    if n_persons < 3 {
        return Err("tenberge_mu needs n_persons >= 3".into());
    }
    if n_items < 3 {
        return Err("tenberge_mu needs n_items >= 3".into());
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("data dimensions overflow usize")?;
    if data.len() != cells {
        return Err(format!(
            "data length {} does not match n_persons * n_items = {cells}",
            data.len()
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data must be finite (no NaN/inf; complete data required)".into());
    }
    let p = n_items;
    let r = correlation_matrix(data, n_persons, p)?;
    let vt: f64 = r.iter().sum();
    if !vt.is_finite() || vt <= 0.0 {
        return Err(format!(
            "sum of the correlation matrix is {vt}; total-score variance must be positive"
        ));
    }
    let (mut s1, mut s2, mut s4, mut s8) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for i in 0..p {
        for j in 0..p {
            if i == j {
                continue;
            }
            let x = r[i * p + j];
            let x2 = x * x;
            let x4 = x2 * x2;
            s1 += x;
            s2 += x2;
            s4 += x4;
            s8 += x4 * x4;
        }
    }
    let c = p as f64 / (p as f64 - 1.0);
    Ok(TenBergeResult {
        mu0: c * s1 / vt,
        mu1: (s1 + (c * s2).sqrt()) / vt,
        mu2: (s1 + (s2 + (c * s4).sqrt()).sqrt()) / vt,
        mu3: (s1 + (s2 + (s4 + (c * s8).sqrt()).sqrt()).sqrt()) / vt,
    })
}

/// `|4 * S_AB / Vt|` for the split with subset A = `a_idx` (sorted item
/// indices) — splitHalf.R line 17 with abs per divergence 5.
fn split_rb(r: &[f64], p: usize, vt: f64, a_idx: &[usize]) -> f64 {
    let mut in_a = vec![false; p];
    for &i in a_idx {
        in_a[i] = true;
    }
    let mut s_ab = 0.0_f64;
    for i in 0..p {
        if !in_a[i] {
            continue;
        }
        for j in 0..p {
            if !in_a[j] {
                s_ab += r[i * p + j];
            }
        }
    }
    (4.0 * s_ab / vt).abs()
}

/// `C(n, k)` in saturating u128 (only compared against a budget).
fn binomial(n: usize, k: usize) -> u128 {
    let k = k.min(n - k);
    let mut acc: u128 = 1;
    for i in 0..k {
        acc = acc.saturating_mul((n - i) as u128) / (i + 1) as u128;
    }
    acc
}

/// Gauss-Jordan inverse with partial pivoting. Errors when a pivot falls
/// below `1e-12` (singular / numerically singular input).
pub(crate) fn invert_symmetric(matrix: &[f64], p: usize) -> Result<Vec<f64>, String> {
    let mut a = matrix.to_vec();
    let mut inv = vec![0.0_f64; p * p];
    for i in 0..p {
        inv[i * p + i] = 1.0;
    }
    for col in 0..p {
        let (pivot_row, pivot_abs) = (col..p)
            .map(|row| (row, a[row * p + col].abs()))
            .max_by(|x, y| x.1.partial_cmp(&y.1).expect("finite pivots"))
            .expect("non-empty column");
        if pivot_abs < 1e-12 {
            return Err("correlation matrix is singular".into());
        }
        if pivot_row != col {
            for k in 0..p {
                a.swap(col * p + k, pivot_row * p + k);
                inv.swap(col * p + k, pivot_row * p + k);
            }
        }
        let pivot = a[col * p + col];
        for k in 0..p {
            a[col * p + k] /= pivot;
            inv[col * p + k] /= pivot;
        }
        for row in 0..p {
            if row == col {
                continue;
            }
            let factor = a[row * p + col];
            if factor == 0.0 {
                continue;
            }
            for k in 0..p {
                a[row * p + k] -= factor * a[col * p + k];
                inv[row * p + k] -= factor * inv[col * p + k];
            }
        }
    }
    Ok(inv)
}

/// Result of a Feldt (1965) confidence interval for coefficient alpha.
///
/// Feldt, L. S. (1965). The approximate sampling distribution of
/// Kuder-Richardson reliability coefficient twenty. Psychometrika, 30(3),
/// 357-370. https://doi.org/10.1007/BF02289499 (paper unobtainable; formula
/// as cited in and verified against Revelle, W. (2025), psych R package
/// v2.6.5, `alpha.ci` in R/alpha.R, read in full).
#[derive(Debug, Clone, Copy)]
pub struct AlphaCiResult {
    /// The point estimate passed in.
    pub alpha: f64,
    /// Lower confidence bound (may be negative; not clamped, matching psych).
    pub lower: f64,
    /// Upper confidence bound.
    pub upper: f64,
    /// Average inter-item correlation implied by alpha:
    /// `r_bar = alpha / (p - alpha*(p-1))` (Spearman-Brown inversion).
    pub r_bar: f64,
    /// Numerator degrees of freedom, `n - 1`.
    pub df1: f64,
    /// Denominator degrees of freedom, `(n - 1) * (p - 1)`.
    pub df2: f64,
}

/// Cronbach's coefficient alpha from raw data (covariance form).
///
/// `alpha = p/(p-1) * (1 - tr(C)/sum(C))` on the sample covariance matrix C
/// (denominator n-1; the n vs n-1 choice cancels in the ratio).
///
/// Cronbach, L. J. (1951). Coefficient alpha and the internal structure of
/// tests. Psychometrika, 16(3), 297-334. https://doi.org/10.1007/BF02310555
/// (covariance form verified against Revelle (2025) psych v2.6.5 R/alpha.R
/// raw-alpha computation; the 1951 paper itself was not re-read for this
/// implementation).
///
/// `data` is row-major n x p. Divergences from psych::alpha: raw-data input
/// only (no reverse-keying/check.keys), zero-variance items are rejected
/// rather than dropped with a warning, hard errors instead of NA.
pub fn cronbach_alpha(data: &[f64], n: usize, p: usize) -> Result<f64, String> {
    if n < 3 {
        return Err("cronbach_alpha needs at least 3 persons".into());
    }
    if p < 2 {
        return Err("cronbach_alpha needs at least 2 items".into());
    }
    if data.len() != n * p {
        return Err(format!("data length {} != n*p = {}", data.len(), n * p));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data contains non-finite values".into());
    }
    let nf = n as f64;
    let mut means = vec![0.0; p];
    for row in data.chunks_exact(p) {
        for (m, v) in means.iter_mut().zip(row) {
            *m += v;
        }
    }
    for m in &mut means {
        *m /= nf;
    }
    // Covariance matrix accumulators: trace and full sum are all we need.
    let mut trace = 0.0;
    let mut total = 0.0;
    for j in 0..p {
        for k in j..p {
            let mut acc = 0.0;
            for row in data.chunks_exact(p) {
                acc += (row[j] - means[j]) * (row[k] - means[k]);
            }
            let cov = acc / (nf - 1.0);
            if j == k {
                if cov <= 0.0 {
                    return Err(format!("item {j} has non-positive variance"));
                }
                trace += cov;
                total += cov;
            } else {
                total += 2.0 * cov;
            }
        }
    }
    if total <= 0.0 {
        return Err("total-score variance (sum of covariance matrix) is not positive".into());
    }
    let pf = p as f64;
    Ok(pf / (pf - 1.0) * (1.0 - trace / total))
}

/// Regularized incomplete beta I_x(a, b) via Lentz continued fraction
/// (Numerical Recipes 3rd ed., sec. 6.4 `betacf` form; transcribed and
/// verified against scipy.stats fixtures in the test suite).
pub(crate) fn inc_beta(a: f64, b: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = crate::fitstats::ln_gamma(a + b)
        - crate::fitstats::ln_gamma(a)
        - crate::fitstats::ln_gamma(b)
        + a * x.ln()
        + b * (1.0 - x).ln();
    // Symmetry: use the tail where the continued fraction converges fast.
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    const EPS: f64 = 1e-15;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let mf = m as f64;
        let m2 = 2.0 * mf;
        // even step
        let aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        // odd step
        let aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// F-distribution CDF: P(F <= x; d1, d2) = I_z(d1/2, d2/2) with
/// z = d1*x / (d1*x + d2).
fn f_cdf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let z = d1 * x / (d1 * x + d2);
    inc_beta(d1 / 2.0, d2 / 2.0, z)
}

/// F-distribution quantile by bisection on z in (0, 1), then
/// x = d2*z / (d1*(1 - z)). Endpoints: prob <= 0 -> 0, prob >= 1 -> +inf
/// (matching qf/scipy; bisection alone would return a finite cap).
fn f_quantile(prob: f64, d1: f64, d2: f64) -> f64 {
    if prob <= 0.0 {
        return 0.0;
    }
    if prob >= 1.0 {
        return f64::INFINITY;
    }
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        let x = d2 * mid / (d1 * (1.0 - mid));
        if f_cdf(x, d1, d2) < prob {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let z = 0.5 * (lo + hi);
    d2 * z / (d1 * (1.0 - z))
}

/// Feldt (1965) exact-F confidence interval for coefficient alpha.
///
/// The pivot (1-alpha)/(1-alpha_hat) is approximately F(n-1, (n-1)(p-1)), so
/// for a two-sided interval at confidence `level` (delta = 1 - level):
///
/// `lower = 1 - (1-alpha_hat) * F^-1(1-delta/2; df1, df2)`
/// `upper = 1 - (1-alpha_hat) * F^-1(delta/2;   df1, df2)`
///
/// Feldt, L. S. (1965). The approximate sampling distribution of
/// Kuder-Richardson reliability coefficient twenty. Psychometrika, 30(3),
/// 357-370. https://doi.org/10.1007/BF02289499 (paper unobtainable; bound
/// mapping verified line-by-line against Revelle (2025) psych v2.6.5
/// `alpha.ci`, and numerically against scipy.stats.f.ppf fixtures).
///
/// Negative `alpha` is allowed (alpha can be negative); bounds are not
/// clamped, matching psych. Divergence from psych: takes the confidence
/// `level` (e.g. 0.95) rather than p.val, and errors instead of NA.
pub fn feldt_alpha_ci(alpha: f64, n: usize, p: usize, level: f64) -> Result<AlphaCiResult, String> {
    if !alpha.is_finite() {
        return Err("alpha must be finite".into());
    }
    if alpha >= 1.0 {
        return Err("alpha must be < 1 (pivot degenerate at alpha = 1)".into());
    }
    if n < 3 {
        return Err("feldt_alpha_ci needs at least 3 persons".into());
    }
    if p < 2 {
        return Err("feldt_alpha_ci needs at least 2 items".into());
    }
    if !(level > 0.0 && level < 1.0) {
        return Err("level must be in (0, 1)".into());
    }
    let df1 = (n - 1) as f64;
    let df2 = ((n - 1) * (p - 1)) as f64;
    let delta = 1.0 - level;
    let one_minus = 1.0 - alpha;
    let lower = 1.0 - one_minus * f_quantile(1.0 - delta / 2.0, df1, df2);
    let upper = 1.0 - one_minus * f_quantile(delta / 2.0, df1, df2);
    let pf = p as f64;
    let r_bar = alpha / (pf - alpha * (pf - 1.0));
    Ok(AlphaCiResult {
        alpha,
        lower,
        upper,
        r_bar,
        df1,
        df2,
    })
}

/// Person separation reliability (eRm `SepRel`; Wright & Stone, 1999, as
/// cited in Mair et al.'s eRm documentation — neither primary source read).
#[derive(Debug, Clone, PartialEq)]
pub struct SeparationReliabilityResult {
    /// `(ssd - mse) / ssd`; unclamped (negative when `mse > ssd`), NaN when
    /// `ssd <= 1e-12`.
    pub sep_rel: f64,
    /// Sample variance of the measures (n-1 denominator, matching R `var`).
    pub ssd: f64,
    /// Mean squared standard error, `mean(se_i^2)`.
    pub mse: f64,
    /// Separation index `G = sqrt((ssd - mse) / mse)`. HAND-DERIVED, not in
    /// the read source: adjusted true SD over RMSE of measurement, so
    /// `G^2 = R / (1 - R)`. NaN when `mse <= 1e-12` or `ssd < mse`.
    pub sep_index: f64,
}

/// Person separation reliability `(SSD - MSE) / SSD` (transcribed from CRAN
/// eRm `R/SepRel.R`, read in full; the docs cite Wright & Stone, 1999, not
/// read). `measures` are point estimates (any estimation method — eRm's
/// docs stress values differ across methods) and `se` their standard
/// errors.
///
/// Caller responsibility (eRm plumbing NOT reproduced here): eRm drops
/// persons with extreme raw scores (interpolated thetas) and missing
/// estimates before applying the formula — pass already-cleaned vectors.
/// The eRm-backed claim covers person measures; applying this to item
/// measures is a generic extension of the same algebra, not sourced.
pub fn separation_reliability(
    measures: &[f64],
    se: &[f64],
) -> Result<SeparationReliabilityResult, String> {
    let n = measures.len();
    if n < 2 {
        return Err("separation_reliability: need at least 2 measures".into());
    }
    if se.len() != n {
        return Err(format!(
            "separation_reliability: se length {} does not match measures length {n}",
            se.len()
        ));
    }
    if measures.iter().any(|v| !v.is_finite()) {
        return Err("separation_reliability: measures must be finite".into());
    }
    if se.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err("separation_reliability: standard errors must be finite and >= 0".into());
    }
    let nf = n as f64;
    let mean = measures.iter().sum::<f64>() / nf;
    let ssd = measures.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    let mse = se.iter().map(|s| s * s).sum::<f64>() / nf;
    if !ssd.is_finite() || !mse.is_finite() {
        return Err(
            "separation_reliability: variance/MSE overflowed to non-finite (inputs too large)"
                .into(),
        );
    }
    let sep_rel = if ssd <= 1e-12 {
        f64::NAN
    } else {
        (ssd - mse) / ssd
    };
    let sep_index = if mse <= 1e-12 || ssd < mse {
        f64::NAN
    } else {
        ((ssd - mse) / mse).sqrt()
    };
    Ok(SeparationReliabilityResult {
        sep_rel,
        ssd,
        mse,
        sep_index,
    })
}

/// Intraclass correlation coefficient output (irr `icc`).
#[derive(Debug, Clone, PartialEq)]
pub struct IccResult {
    /// The ICC point estimate for the requested variant.
    pub value: f64,
    /// Complete (post listwise-drop) subject rows used.
    pub subjects: u64,
    /// Raters (columns).
    pub raters: u64,
    /// F statistic for H0: icc = r0.
    pub fvalue: f64,
    /// Numerator degrees of freedom.
    pub df1: f64,
    /// Denominator degrees of freedom (Satterthwaite, possibly non-integer,
    /// for the agreement variants).
    pub df2: f64,
    /// Upper-tail p-value `P(F_{df1,df2} > fvalue)`.
    pub p_value: f64,
    /// Lower confidence bound (NOT clamped; can be negative, and below -1
    /// for average-score variants, matching R).
    pub lbound: f64,
    /// Upper confidence bound (not clamped).
    pub ubound: f64,
}

/// Intraclass correlation coefficients (Shrout-Fleiss family), transcribed
/// from CRAN irr 0.85 `R/icc.R` (READ in full; algorithm source of truth).
/// Model origins — cited as origins only, NOT READ: Shrout, P. E., &
/// Fleiss, J. L. (1979). Intraclass correlations: Uses in assessing rater
/// reliability. *Psychological Bulletin, 86*(2), 420-428; McGraw, K. O., &
/// Wong, S. P. (1996). Forming inferences about some intraclass correlation
/// coefficients. *Psychological Methods, 1*(1), 30-46; Bartko, J. J.
/// (1966). The intraclass correlation coefficient as a measure of
/// reliability. *Psychological Reports, 19*, 3-11.
///
/// `ratings` is row-major `ns x nr` (subjects x raters); any row containing
/// NaN is dropped listwise before computation (R `na.omit`). `model` is
/// `"oneway"` or `"twoway"`; `typ` is `"consistency"` or `"agreement"`
/// (ignored for oneway, matching R); `unit` is `"single"` or `"average"`.
/// `r0` is the null ICC for the F test; `conf_level` the CI level.
///
/// ANOVA mean squares (R lines 13-17, `var` = sample variance n-1):
/// `MSr = var(row means)*nr`, `MSw = mean(row variances)`,
/// `MSc = var(col means)*ns`, `MSe = (SStotal - MSr(ns-1) - MSc(nr-1)) /
/// ((ns-1)(nr-1))`. Agreement F tests use the Satterthwaite df with `r0`
/// (R lines 67-75, 128-136); agreement CIs plug the estimated coefficient
/// into the same Satterthwaite form (McGraw & Wong, 1996, as coded at R
/// lines 78-85, 139-146). The average-agreement CI reuses the nr-scaled
/// `a,b` expressions of the single variant verbatim (R lines 139-141) —
/// preserved deliberately.
///
/// Documented deviations from R: explicit errors (instead of NaN
/// propagation) for fewer than 2 complete rows, `nr < 2`, non-finite
/// (non-NaN) input, out-of-range `r0`/`conf_level`, and degenerate
/// zero/non-finite denominators; dimension caps `ns <= 1e6`, `nr <= 1e4`.
pub fn icc(
    ratings: &[f64],
    ns: usize,
    nr: usize,
    model: &str,
    typ: &str,
    unit: &str,
    r0: f64,
    conf_level: f64,
) -> Result<IccResult, String> {
    if !matches!(model, "oneway" | "twoway") {
        return Err("model must be \"oneway\" or \"twoway\"".into());
    }
    if !matches!(typ, "consistency" | "agreement") {
        return Err("type must be \"consistency\" or \"agreement\"".into());
    }
    if !matches!(unit, "single" | "average") {
        return Err("unit must be \"single\" or \"average\"".into());
    }
    if !r0.is_finite() || !(0.0..1.0).contains(&r0) {
        return Err("r0 must be finite and in [0, 1)".into());
    }
    if !conf_level.is_finite() || !(conf_level > 0.0 && conf_level < 1.0) {
        return Err("conf_level must be in (0, 1)".into());
    }
    if nr < 2 {
        return Err("icc needs at least 2 raters".into());
    }
    if ns > 1_000_000 || nr > 10_000 {
        return Err("icc: dimensions exceed caps (ns <= 1e6, nr <= 1e4)".into());
    }
    if ratings.len() != ns * nr {
        return Err(format!(
            "ratings length {} does not match ns*nr = {}",
            ratings.len(),
            ns * nr
        ));
    }
    if ratings.iter().any(|v| v.is_infinite()) {
        return Err("ratings must not contain infinities (use NaN for missing)".into());
    }
    // Listwise drop of rows containing NaN (R na.omit).
    let rows: Vec<&[f64]> = (0..ns)
        .map(|i| &ratings[i * nr..(i + 1) * nr])
        .filter(|r| r.iter().all(|v| v.is_finite()))
        .collect();
    let m = rows.len();
    if m < 2 {
        return Err("icc needs at least 2 complete subject rows after dropping missing".into());
    }
    let mf = m as f64;
    let nrf = nr as f64;

    fn sample_var(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    let all: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let ss_total = sample_var(&all) * (mf * nrf - 1.0);
    let row_means: Vec<f64> = rows.iter().map(|r| r.iter().sum::<f64>() / nrf).collect();
    let ms_r = sample_var(&row_means) * nrf;
    let ms_w = rows.iter().map(|r| sample_var(r)).sum::<f64>() / mf;
    let col_means: Vec<f64> = (0..nr)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / mf)
        .collect();
    let ms_c = sample_var(&col_means) * mf;
    let ms_e = (ss_total - ms_r * (mf - 1.0) - ms_c * (nrf - 1.0)) / ((mf - 1.0) * (nrf - 1.0));
    if ![ms_r, ms_w, ms_c, ms_e].iter().all(|v| v.is_finite()) {
        return Err("icc: ANOVA mean squares are non-finite (inputs too large?)".into());
    }

    let alpha = 1.0 - conf_level;
    let q = 1.0 - alpha / 2.0;
    let oneway = model == "oneway";
    let consistency = typ == "consistency";
    let single = unit == "single";

    // Satterthwaite df for the twoway-agreement F test / CI (R lines
    // 67-69, 78-80, 128-130, 139-141). `scale` is nr for the single-unit
    // a,b and 1 for the average F test; the average CI deliberately reuses
    // the nr-scaled form (R quirk, lines 139-141).
    let satt = |rho: f64, scale: f64| -> (f64, f64, f64) {
        let a = (scale * rho) / (mf * (1.0 - rho));
        let b = 1.0 + (scale * rho * (mf - 1.0)) / (mf * (1.0 - rho));
        let v = (a * ms_c + b * ms_e).powi(2)
            / ((a * ms_c).powi(2) / (nrf - 1.0) + (b * ms_e).powi(2) / ((mf - 1.0) * (nrf - 1.0)));
        (a, b, v)
    };

    let (value, fvalue, df1, df2, lbound, ubound);
    if oneway {
        let denom_s = ms_r + (nrf - 1.0) * ms_w;
        if single {
            value = (ms_r - ms_w) / denom_s;
        } else {
            value = (ms_r - ms_w) / ms_r;
        }
        fvalue = if single {
            ms_r / ms_w * ((1.0 - r0) / (1.0 + (nrf - 1.0) * r0))
        } else {
            ms_r / ms_w * (1.0 - r0)
        };
        df1 = mf - 1.0;
        df2 = mf * (nrf - 1.0);
        let fl = (ms_r / ms_w) / f_quantile(q, df1, df2);
        let fu = (ms_r / ms_w) * f_quantile(q, df2, df1);
        if single {
            lbound = (fl - 1.0) / (fl + nrf - 1.0);
            ubound = (fu - 1.0) / (fu + nrf - 1.0);
        } else {
            lbound = 1.0 - 1.0 / fl;
            ubound = 1.0 - 1.0 / fu;
        }
    } else if consistency {
        if single {
            value = (ms_r - ms_e) / (ms_r + (nrf - 1.0) * ms_e);
        } else {
            value = (ms_r - ms_e) / ms_r;
        }
        fvalue = if single {
            ms_r / ms_e * ((1.0 - r0) / (1.0 + (nrf - 1.0) * r0))
        } else {
            ms_r / ms_e * (1.0 - r0)
        };
        df1 = mf - 1.0;
        df2 = (mf - 1.0) * (nrf - 1.0);
        let fl = (ms_r / ms_e) / f_quantile(q, df1, df2);
        let fu = (ms_r / ms_e) * f_quantile(q, df2, df1);
        if single {
            lbound = (fl - 1.0) / (fl + nrf - 1.0);
            ubound = (fu - 1.0) / (fu + nrf - 1.0);
        } else {
            lbound = 1.0 - 1.0 / fl;
            ubound = 1.0 - 1.0 / fu;
        }
    } else {
        // twoway agreement
        if single {
            value = (ms_r - ms_e) / (ms_r + (nrf - 1.0) * ms_e + (nrf / mf) * (ms_c - ms_e));
        } else {
            value = (ms_r - ms_e) / (ms_r + (ms_c - ms_e) / mf);
        }
        let (a, b, v) = satt(r0, if single { nrf } else { 1.0 });
        fvalue = ms_r / (a * ms_c + b * ms_e);
        df1 = mf - 1.0;
        df2 = v;
        if !(1.0 - value).is_finite() || (1.0 - value).abs() < 1e-12 {
            return Err("icc: degenerate coefficient (icc = 1) — CI undefined".into());
        }
        // McGraw & Wong CI: plug the estimate into the nr-scaled a,b (R
        // lines 78-80 and, deliberately, 139-141 for the average variant).
        let (_a2, _b2, v2) = satt(value, nrf);
        if !v2.is_finite() || v2 <= 0.0 {
            return Err("icc: degenerate Satterthwaite df in CI".into());
        }
        let fl = f_quantile(q, df1, v2);
        let fu = f_quantile(q, v2, df1);
        if single {
            lbound = (mf * (ms_r - fl * ms_e))
                / (fl * (nrf * ms_c + (nrf * mf - nrf - mf) * ms_e) + mf * ms_r);
            ubound = (mf * (fu * ms_r - ms_e))
                / (nrf * ms_c + (nrf * mf - nrf - mf) * ms_e + mf * fu * ms_r);
        } else {
            lbound = (mf * (ms_r - fl * ms_e)) / (fl * (ms_c - ms_e) + mf * ms_r);
            ubound = (mf * (fu * ms_r - ms_e)) / (ms_c - ms_e + mf * fu * ms_r);
        }
    }
    if ![value, fvalue, df2, lbound, ubound]
        .iter()
        .all(|v| v.is_finite())
    {
        return Err(
            "icc: degenerate ratings (zero-variance denominator produced non-finite output)".into(),
        );
    }
    let p_value = 1.0 - f_cdf(fvalue, df1, df2);
    Ok(IccResult {
        value,
        subjects: m as u64,
        raters: nr as u64,
        fvalue,
        df1,
        df2,
        p_value,
        lbound,
        ubound,
    })
}

/// Result of [`kripp_alpha`].
#[derive(Debug, Clone)]
pub struct KrippResult {
    /// Krippendorff's alpha estimate.
    pub value: f64,
    /// Number of subject columns as given (R reports `dim(x)[2]`).
    pub subjects: u64,
    /// Number of raters (matrix rows).
    pub raters: u64,
    /// Number of distinct finite rating levels.
    pub levels: u64,
    /// Total coincidence-matrix mass (R `nmatchval`).
    pub nmatchval: f64,
}

/// Krippendorff's alpha for a raters x subjects rating matrix.
///
/// Transcribed from CRAN irr 0.85 `R/kripp.alpha.R` (READ in full, 66
/// lines; algorithm source of truth). Method origin ? cited as origin
/// only, NOT READ: Krippendorff, K. (1980). *Content analysis: An
/// introduction to its methodology*. Sage.
///
/// Verified against the R source:
/// - Levels are the sorted unique finite values (R `levels(as.factor(x))`,
///   line 7; base R sorts numeric factor levels in ascending numeric
///   order, not lexicographically).
/// - Coincidence matrix (lines 12-25): for every unordered rater pair in
///   a subject column with both values present, cell `(a, b)` gains
///   `(1 + (a == b)) / mc[col]` and the mirror cell is set by assignment
///   (line 21). `mc[col]` is `#nonmissing - 1` only when the matrix
///   contains at least one missing value anywhere, else `1` for every
///   column (lines 12-13). The no-missing divisor of 1 is a documented
///   irr quirk preserved verbatim ? it is NOT the `m - 1` convention and
///   changes both `nmatchval` and alpha on complete data.
/// - `nmatchval` sums all cells (line 26). Fewer than 2 observed levels
///   yields alpha = 1 (line 45).
/// - Distance metrics (lines 50-59) with `nc` the coincidence row sums:
///   nominal `1`; ordinal `(nc_c/2 + sum_{g=c+1}^{k-1} nc_g + nc_k/2)^2`;
///   interval `(v_c - v_k)^2`; ratio `((v_c - v_k)/(v_c + v_k))^2`.
/// - `alpha = 1 - (nmatchval - 1) * sum(utcm * diff2)
///   / sum(nc_c * nc_k * diff2)` over the upper triangle (line 63).
///
/// Documented deviations from R: an all-missing matrix is an error here
/// (R's line-45 path would report alpha = 1 with zero levels); infinite
/// ratings are rejected; the ratio metric errors when any level pair sums
/// to zero (R silently produces Inf/NaN); dimension caps.
///
/// `ratings` is row-major raters x subjects; NaN marks missing. No
/// standard error or CI is produced (the R source computes none).
///
/// # References
/// Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). *irr: Various
///     coefficients of interrater reliability and agreement* (Version 0.85)
///     [Computer software]. CRAN. https://CRAN.R-project.org/package=irr
/// Krippendorff, K. (1980). *Content analysis: An introduction to its
///     methodology*. Sage. (as cited in Gamer et al., 2019; NOT READ)
pub fn kripp_alpha(
    ratings: &[f64],
    nraters: usize,
    nsubjects: usize,
    method: &str,
) -> Result<KrippResult, String> {
    if !matches!(method, "nominal" | "ordinal" | "interval" | "ratio") {
        return Err(
            "method must be one of \"nominal\", \"ordinal\", \"interval\", \"ratio\"".into(),
        );
    }
    if nraters < 2 {
        return Err("kripp_alpha needs at least 2 raters".into());
    }
    if nsubjects < 1 {
        return Err("kripp_alpha needs at least 1 subject".into());
    }
    if nraters > 10_000 || nsubjects > 1_000_000 {
        return Err("kripp_alpha: dimensions exceed caps (raters <= 1e4, subjects <= 1e6)".into());
    }
    if ratings.len() != nraters * nsubjects {
        return Err(format!(
            "ratings length {} does not match raters*subjects = {}",
            ratings.len(),
            nraters * nsubjects
        ));
    }
    if ratings.iter().any(|v| v.is_infinite()) {
        return Err("ratings must not contain infinities (use NaN for missing)".into());
    }
    let mut levels: Vec<f64> = ratings.iter().copied().filter(|v| v.is_finite()).collect();
    levels.sort_by(|a, b| a.partial_cmp(b).expect("finite by filter"));
    levels.dedup();
    let nval = levels.len();
    if nval == 0 {
        return Err("kripp_alpha: all ratings are missing".into());
    }
    let any_na = ratings.iter().any(|v| v.is_nan());
    let lev_index = |v: f64| -> usize {
        levels
            .binary_search_by(|p| p.partial_cmp(&v).expect("finite levels"))
            .expect("observed value is a level by construction")
    };
    let mut cm = vec![0.0_f64; nval * nval];
    for col in 0..nsubjects {
        // R lines 12-13: per-column divisor only under the global-NA path.
        let mc = if any_na {
            let nonmiss = (0..nraters)
                .filter(|&r| !ratings[r * nsubjects + col].is_nan())
                .count();
            nonmiss as f64 - 1.0
        } else {
            1.0
        };
        for i1 in 0..nraters - 1 {
            for i2 in (i1 + 1)..nraters {
                let a = ratings[i1 * nsubjects + col];
                let b = ratings[i2 * nsubjects + col];
                if a.is_nan() || b.is_nan() {
                    continue;
                }
                // A column visited here has >= 2 non-missing values, so
                // mc >= 1 under the NA path (mc = 0 or -1 only occurs for
                // columns that form no pair).
                let (ia, ib) = (lev_index(a), lev_index(b));
                // R line 20: diagonal gains 2/mc, off-diagonal 1/mc with
                // the mirror cell set by assignment (line 21).
                let inc = if ia == ib { 2.0 } else { 1.0 } / mc;
                cm[ia * nval + ib] += inc;
                if ia != ib {
                    cm[ib * nval + ia] = cm[ia * nval + ib];
                }
            }
        }
    }
    let nmatchval: f64 = cm.iter().sum();
    let mut value = 1.0;
    if nval >= 2 {
        let nc: Vec<f64> = (0..nval)
            .map(|i| cm[i * nval..(i + 1) * nval].iter().sum())
            .collect();
        let mut num = 0.0;
        let mut den = 0.0;
        for k in 1..nval {
            for c in 0..k {
                let diff2 = match method {
                    "nominal" => 1.0,
                    "ordinal" => {
                        let s: f64 = nc[c] / 2.0 + nc[c + 1..k].iter().sum::<f64>() + nc[k] / 2.0;
                        s * s
                    }
                    "interval" => {
                        let d = levels[c] - levels[k];
                        d * d
                    }
                    _ => {
                        let s = levels[c] + levels[k];
                        if s == 0.0 {
                            return Err(
                                "kripp_alpha: ratio metric undefined (level pair sums to zero)"
                                    .into(),
                            );
                        }
                        let d = (levels[c] - levels[k]) / s;
                        d * d
                    }
                };
                num += cm[c * nval + k] * diff2;
                den += nc[c] * nc[k] * diff2;
            }
        }
        if den == 0.0 {
            return Err("kripp_alpha: degenerate data (zero denominator)".into());
        }
        value = 1.0 - (nmatchval - 1.0) * num / den;
    }
    if !value.is_finite() || !nmatchval.is_finite() {
        return Err("kripp_alpha: degenerate data (non-finite result)".into());
    }
    Ok(KrippResult {
        value,
        subjects: nsubjects as u64,
        raters: nraters as u64,
        levels: nval as u64,
        nmatchval,
    })
}

/// Finn (1970) reliability coefficient output (irr `finn`).
#[derive(Debug, Clone, PartialEq)]
pub struct FinnResult {
    /// The Finn coefficient `1 - MS/MSexp`.
    pub value: f64,
    /// F statistic `MSexp/MS` (df1 = Inf conceptually; `+Inf` for the
    /// documented perfect-agreement `MS == 0` case).
    pub statistic: f64,
    /// Denominator degrees of freedom `ns*(nr-1)` (convenience field; the
    /// R return encodes it only inside `stat.name = "F(Inf,<df2>)"`).
    pub df2: f64,
    /// Upper-tail p-value `pf(F, Inf, df2, lower.tail = FALSE)`.
    pub p_value: f64,
    /// Complete (post listwise-drop) subject rows used.
    pub subjects: u64,
    /// Raters (columns).
    pub raters: u64,
}

/// Finn (1970) coefficient of reliability for categorical-scale ratings,
/// transcribed from CRAN irr 0.85 `R/finn.R` (READ in full; algorithm
/// source of truth). Origin cited per irr docs, NOT READ: Finn, R. H.
/// (1970). A note on estimating the reliability of categorical data.
/// *Educational and Psychological Measurement, 30*, 71-76.
///
/// `ratings` is row-major `ns x nr` (subjects x raters); rows containing
/// NaN are dropped listwise (R `na.omit`). `s_levels` is the number of
/// discrete scale levels `s >= 2`; the expected mean square under a
/// uniform-random rating model is the discrete-uniform variance
/// `MSexp = (s^2 - 1)/12`. With sample variances (n-1 denominator):
/// `MSw = mean(row variances)` and the two-way residual
/// `MSe = (SStotal - MSr(ns-1) - MSc(nr-1)) / ((ns-1)(nr-1))`,
/// `coeff = 1 - MS/MSexp`, `F = MSexp/MS` where `MS` is `MSw` (oneway) or
/// `MSe` (twoway). Both models use `df2 = ns*(nr-1)` — the R source uses
/// `ns*(nr-1)` for twoway as well (quirk preserved verbatim).
///
/// p-value: R computes `pf(F, Inf, df2, lower.tail = FALSE)`. Limiting
/// identity (hand-derived; convergence-verified against scipy in the
/// session oracle): `F(d1, d2) -> d2 / Chi2_d2` as `d1 -> Inf`, so
/// `P(F > f) = P(Chi2_d2 < d2/f)`, implemented as
/// `1 - chi2_sf(df2/F, df2)`. Valid only for `F > 0`; a negative mean
/// square (possible via floating cancellation only — both `MSw` and `MSe`
/// are nonnegative in exact arithmetic) is rejected as an error rather
/// than emulating R's `pf` on a negative statistic. `MS == 0` (perfect
/// agreement) returns `value = 1`, `statistic = +Inf`, `p_value = 0`
/// (matching the R limit; documented deliberate non-finite statistic).
///
/// Documented deviations from R (deliberate, stricter-than-R — R lets
/// nonsensical inputs propagate to NA/Inf arithmetic): explicit errors for
/// fewer than 2 complete rows, `nr < 2`, `s_levels < 2`, non-finite
/// (non-NaN) input, negative mean squares; dimension caps `ns <= 1e6`,
/// `nr <= 1e4`.
pub fn finn_coefficient(
    ratings: &[f64],
    ns: usize,
    nr: usize,
    s_levels: u32,
    model: &str,
) -> Result<FinnResult, String> {
    if !matches!(model, "oneway" | "twoway") {
        return Err("model must be \"oneway\" or \"twoway\"".into());
    }
    if s_levels < 2 {
        return Err("finn: s_levels must be at least 2".into());
    }
    if nr < 2 {
        return Err("finn needs at least 2 raters".into());
    }
    if ns > 1_000_000 || nr > 10_000 {
        return Err("finn: dimensions exceed caps (ns <= 1e6, nr <= 1e4)".into());
    }
    if ratings.len() != ns * nr {
        return Err(format!(
            "ratings length {} does not match ns*nr = {}",
            ratings.len(),
            ns * nr
        ));
    }
    if ratings.iter().any(|v| v.is_infinite()) {
        return Err("ratings must not contain infinities (use NaN for missing)".into());
    }
    // Listwise drop of rows containing NaN (R na.omit).
    let rows: Vec<&[f64]> = (0..ns)
        .map(|i| &ratings[i * nr..(i + 1) * nr])
        .filter(|r| r.iter().all(|v| v.is_finite()))
        .collect();
    let m = rows.len();
    if m < 2 {
        return Err("finn needs at least 2 complete subject rows after dropping missing".into());
    }
    let mf = m as f64;
    let nrf = nr as f64;

    fn sample_var(xs: &[f64]) -> f64 {
        let n = xs.len() as f64;
        let mean = xs.iter().sum::<f64>() / n;
        xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0)
    }

    let all: Vec<f64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
    let ss_total = sample_var(&all) * (mf * nrf - 1.0);
    let row_means: Vec<f64> = rows.iter().map(|r| r.iter().sum::<f64>() / nrf).collect();
    let ms_r = sample_var(&row_means) * nrf;
    let ms_w = rows.iter().map(|r| sample_var(r)).sum::<f64>() / mf;
    let col_means: Vec<f64> = (0..nr)
        .map(|j| rows.iter().map(|r| r[j]).sum::<f64>() / mf)
        .collect();
    let ms_c = sample_var(&col_means) * mf;
    let ms_e = (ss_total - ms_r * (mf - 1.0) - ms_c * (nrf - 1.0)) / ((mf - 1.0) * (nrf - 1.0));
    if ![ms_r, ms_w, ms_c, ms_e].iter().all(|v| v.is_finite()) {
        return Err("finn: ANOVA mean squares are non-finite (inputs too large?)".into());
    }

    let ms = if model == "oneway" { ms_w } else { ms_e };
    if ms < 0.0 {
        // Only reachable via floating cancellation in MSe; the chi-square
        // limiting identity requires F > 0 (see doc comment).
        return Err("finn: negative mean square (numerically degenerate input)".into());
    }
    let ms_exp = ((s_levels as f64) * (s_levels as f64) - 1.0) / 12.0;
    let df2 = mf * (nrf - 1.0);
    if ms == 0.0 {
        return Ok(FinnResult {
            value: 1.0,
            statistic: f64::INFINITY,
            df2,
            p_value: 0.0,
            subjects: m as u64,
            raters: nr as u64,
        });
    }
    let value = 1.0 - ms / ms_exp;
    let statistic = ms_exp / ms;
    let p_value = 1.0 - crate::fitstats::chi2_sf(df2 / statistic, df2);
    if !value.is_finite() || !statistic.is_finite() || !p_value.is_finite() {
        return Err("finn: non-finite result".into());
    }
    Ok(FinnResult {
        value,
        statistic,
        df2,
        p_value,
        subjects: m as u64,
        raters: nr as u64,
    })
}

/// Result of Maxwell's RE agreement coefficient for two raters.
///
/// Fields mirror the irr `irrlist` return: `value` is the RE coefficient,
/// `subjects` the number of complete (post listwise-deletion) subject rows,
/// `raters` the number of raters (always 2 on success).
#[derive(Debug, Clone)]
pub struct MaxwellResult {
    /// Maxwell's RE coefficient, in [-1, 1].
    pub value: f64,
    /// Number of subjects after listwise deletion of NaN rows.
    pub subjects: u64,
    /// Number of raters (always 2).
    pub raters: u64,
}

/// Maxwell's RE agreement coefficient for two raters with binary ratings.
///
/// Computes `RE = 2 * A / ns - 1`, where `A` is the number of subjects on
/// which the two raters agree exactly and `ns` the number of complete
/// subjects after listwise deletion of rows containing NaN. This replicates
/// `maxwell()` from the irr R package (v0.84.1, `R/maxwell.R`, source READ):
/// the R code builds `table(r1, r2)` over the union of both columns' factor
/// levels and returns `2*sum(diag(ttab))/ns - 1`. Because both columns are
/// refactored with the same level vector, the diagonal sum equals the exact
/// match count, independent of level ordering (verified by hand derivation
/// and an executed exact-Fraction oracle; the R level-ordering quirk of
/// putting the longer level set first cannot affect a diagonal sum).
///
/// Binary check: the distinct-value union across BOTH columns of the kept
/// rows must have cardinality <= 2 (R stops with "Ratings are not binary"
/// otherwise). A single distinct value is allowed and yields RE = 1.
/// Values are compared with exact `==` on f64, mirroring R's factor
/// coercion of numeric ratings.
///
/// Deliberately stricter than R: `nr != 2` is a checked error (R stops only
/// for nr > 2 and fails accidentally with a subscript error for nr < 2), and
/// non-finite values other than NaN (which means missing) are rejected.
///
/// `ratings` is row-major `ns x nr` with `nr == 2`.
///
/// # References
///
/// Gamer, M., Lemon, J., Fellows, I., & Singh, P. (2019). irr:
/// Various coefficients of interrater reliability and agreement
/// (Version 0.84.1) [Computer software]. CRAN.
/// <https://CRAN.R-project.org/package=irr> (R source `R/maxwell.R` READ;
/// normative reference for this implementation.)
///
/// Maxwell, A. E. (1977). Coefficients of agreement between observers and
/// their interpretation. British Journal of Psychiatry, 130(1), 79-83.
/// <https://doi.org/10.1192/bjp.130.1.79> (NOT READ; cited as the
/// historical origin via the irr package documentation only.)
pub fn maxwell_re(ratings: &[f64], ns: usize, nr: usize) -> Result<MaxwellResult, String> {
    if nr != 2 {
        return Err("maxwell: exactly 2 raters required".into());
    }
    if ns == 0 {
        return Err("maxwell: at least one subject required".into());
    }
    if ns > 1_000_000 {
        return Err("maxwell: ns exceeds 1e6 bound".into());
    }
    let expected = ns
        .checked_mul(nr)
        .ok_or_else(|| "maxwell: ns*nr overflows".to_string())?;
    if ratings.len() != expected {
        return Err(format!(
            "maxwell: ratings length {} != ns*nr = {}",
            ratings.len(),
            expected
        ));
    }
    if ratings.iter().any(|v| v.is_infinite()) {
        return Err("maxwell: infinite values not allowed (use NaN for missing)".into());
    }
    // Listwise deletion: keep rows where both entries are non-NaN.
    let kept: Vec<(f64, f64)> = (0..ns)
        .map(|p| (ratings[p * 2], ratings[p * 2 + 1]))
        .filter(|(a, b)| !a.is_nan() && !b.is_nan())
        .collect();
    let m = kept.len();
    if m == 0 {
        return Err("maxwell: no complete subject rows after NaN deletion".into());
    }
    // Distinct-value union across both columns (exact == on f64, as R's
    // factor coercion of numeric labels). Linear scan; stops at 3.
    let mut levels: Vec<f64> = Vec::with_capacity(3);
    for &(a, b) in &kept {
        for v in [a, b] {
            if !levels.iter().any(|&l| l == v) {
                levels.push(v);
                if levels.len() > 2 {
                    return Err("maxwell: ratings are not binary".into());
                }
            }
        }
    }
    let agree = kept.iter().filter(|(a, b)| a == b).count();
    let value = 2.0 * agree as f64 / m as f64 - 1.0;
    if !value.is_finite() {
        return Err("maxwell: non-finite result".into());
    }
    Ok(MaxwellResult {
        value,
        subjects: m as u64,
        raters: 2,
    })
}

/// Result of Robinson's A agreement coefficient.
#[derive(Debug, Clone)]
pub struct RobinsonResult {
    /// Robinson's A = SSb / (SSb + SSr), in [0, 1].
    pub value: f64,
    /// Number of subjects retained after listwise deletion.
    pub subjects: u64,
    /// Number of raters.
    pub raters: u64,
}

/// Robinson's A coefficient of agreement for interval-scale ratings.
///
/// Normative source: the `robinson()` function in the irr R package
/// (version 0.84.1, file R/robinson.R), which was READ in full and used
/// as the behavioural contract. Robinson, W. S. (1957). The statistical
/// measurement of agreement. American Sociological Review, 22(1), 17-25
/// was NOT read; it is cited only as the historical origin per the irr
/// package documentation.
///
/// With R's sample variance var(x) = sum((x - mean)^2) / (n - 1), the
/// (n - 1) factors in robinson.R cancel, giving (verified exactly with
/// Fraction arithmetic in the development oracle):
///
/// ```text
/// SSb = nr * sum_i (rowmean_i - grand)^2
/// SSr = sum_ij (x_ij - rowmean_i - colmean_j + grand)^2   (interaction SS)
/// A   = SSb / (SSb + SSr)
/// ```
///
/// SSr is computed directly as the interaction sum of squares (each term
/// a square, hence SSr >= 0 in f64), never subtractively. Deviation from
/// R: where R silently yields NaN (0/0) for degenerate inputs with no
/// subject variance (SSb + SSr == 0), this function returns Err.
///
/// `ratings` is row-major `ns x nr` (subjects x raters). NaN marks
/// missing values and triggers listwise (whole-row) deletion; infinite
/// values are rejected.
pub fn robinson_a(ratings: &[f64], ns: usize, nr: usize) -> Result<RobinsonResult, String> {
    if nr < 2 {
        return Err("robinson: at least 2 raters required".into());
    }
    if ns == 0 {
        return Err("robinson: at least 1 subject required".into());
    }
    if ns > 1_000_000 {
        return Err("robinson: ns exceeds 1e6".into());
    }
    if nr > 10_000 {
        return Err("robinson: nr exceeds 1e4".into());
    }
    let expected = ns
        .checked_mul(nr)
        .ok_or_else(|| "robinson: ns*nr overflows".to_string())?;
    if ratings.len() != expected {
        return Err(format!(
            "robinson: ratings length {} != ns*nr = {}",
            ratings.len(),
            expected
        ));
    }
    if ratings.iter().any(|v| v.is_infinite()) {
        return Err("robinson: infinite values not allowed (use NaN for missing)".into());
    }
    // Listwise deletion: keep rows with no NaN.
    let kept: Vec<usize> = (0..ns)
        .filter(|&i| (0..nr).all(|j| !ratings[i * nr + j].is_nan()))
        .collect();
    let m = kept.len();
    if m < 2 {
        return Err("robinson: fewer than 2 complete subjects after listwise deletion".into());
    }
    let mf = m as f64;
    let nrf = nr as f64;
    let mut grand = 0.0;
    let mut row_means = vec![0.0; m];
    for (r, &i) in kept.iter().enumerate() {
        let mut s = 0.0;
        for j in 0..nr {
            s += ratings[i * nr + j];
        }
        row_means[r] = s / nrf;
        grand += s;
    }
    grand /= mf * nrf;
    let mut col_means = vec![0.0; nr];
    for (j, cm) in col_means.iter_mut().enumerate() {
        let mut s = 0.0;
        for &i in &kept {
            s += ratings[i * nr + j];
        }
        *cm = s / mf;
    }
    let mut ssb = 0.0;
    for rm in &row_means {
        let d = rm - grand;
        ssb += d * d;
    }
    ssb *= nrf;
    // Direct interaction sum of squares (each term squared, so SSr >= 0).
    let mut ssr = 0.0;
    for (r, &i) in kept.iter().enumerate() {
        for j in 0..nr {
            let d = ratings[i * nr + j] - row_means[r] - col_means[j] + grand;
            ssr += d * d;
        }
    }
    if !ssb.is_finite() || !ssr.is_finite() {
        return Err("robinson: non-finite sums of squares".into());
    }
    let denom = ssb + ssr;
    if denom <= 0.0 {
        return Err(
            "robinson: degenerate input with no subject variance (R would return NaN)".into(),
        );
    }
    let value = ssb / denom;
    if !value.is_finite() {
        return Err("robinson: non-finite result".into());
    }
    Ok(RobinsonResult {
        value,
        subjects: m as u64,
        raters: nr as u64,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/reliability_tests.rs"]
mod tests;
