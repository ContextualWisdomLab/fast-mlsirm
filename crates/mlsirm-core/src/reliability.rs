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

#[cfg(test)]
#[path = "../../../tests/unit/reliability_tests.rs"]
mod tests;
