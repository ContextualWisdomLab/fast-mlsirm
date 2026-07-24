//! Mokken scale analysis: Loevinger scalability coefficients and the
//! automated item selection procedure (AISP).
//!
//! Implements, for a complete integer response matrix (dichotomous or
//! polytomous), the sample scalability coefficients
//!
//! ```text
//! Hij = S_ij / Smax_ij
//! Hi  = sum_{j != i} S_ij / sum_{j != i} Smax_ij
//! H   = sum_{i < j} S_ij / sum_{i < j} Smax_ij
//! ```
//!
//! where `S` is the sample covariance matrix (denominator N-1) and
//! `Smax_ij = cov(sort(X_i), sort(X_j))` is the maximum covariance
//! attainable given the two items' marginal score distributions — the
//! comonotone (sorted-sorted) coupling maximizes `sum x_p y_p` by the
//! rearrangement inequality, and the means are marginal-fixed, so it
//! maximizes the covariance; the N-1 denominators cancel in every ratio.
//!
//! `Hi` uses the ratio of PAIRWISE sums, exactly as the mokken R package
//! computes it (`coefHTiny`). Verified caveat: this is NOT generally equal
//! to a "max Cov(X_j, R_-j) holding the realized rest-score marginal fixed"
//! reading of van der Ark (2007, Eq. 2) — counterexample: X1=X2=[0,0,1,1],
//! X3=[0,1,0,1] gives fixed-marginal max 1/3 for item 1 but pairwise-sum
//! denominator 2/3. The pairwise-sum form is the de-facto MSA standard and
//! is what this module implements.
//!
//! Mokken's Z statistics (null hypothesis of inter-item independence) follow
//! the mokken package's `coefZ` (`type.z = "Z"`):
//!
//! ```text
//! Zij = S_ij * sqrt(N-1) / sqrt(s_ii * s_jj)
//! Zi  = (sum_{j != i} S_ij) * sqrt(N-1) / sqrt(sum_{j != i} s_ii * s_jj)
//! Z   = (sum_{i < j} S_ij) * sqrt(N-1) / sqrt(sum_{i < j} s_ii * s_jj)
//! ```
//!
//! The AISP ("search normal") partitions items into Mokken scales: a start
//! pair maximizing `Hij` among pairs significantly positive (`|Zij| >= Z_c`)
//! with pair `H >= c`, then repeatedly adds the free item that (1) has no
//! negative `Hij` with any selected item (nonnegative allowed), (2) has
//! within-augmented-set `Hi >= c`, (3) has `Zi >= Z_c`, and (4) maximizes the
//! augmented set's total `H`; the scale closes when the best augmented-set
//! `H < c`, and further scales are formed from leftover items. The
//! significance level is Bonferroni-adjusted per scale as
//! `alpha / (K1*(K1-1)/2 + sum of later step candidate counts)`, with the
//! candidate-count vector resetting at each new scale, matching
//! `search.normal.R` (`adjusted.alpha`).
//!
//! Verification status: the coefficient definitions, rules of thumb, and the
//! Mokken-scale definition (all inter-item covariances nonnegative in the
//! selection sense and `Hi >= c > 0`) were read in van der Ark (2007) and
//! Straat et al. (2013); the exact sample statistics, Z forms, tie-breaking,
//! and AISP mechanics were verified line-by-line against the mokken R package
//! source (CRAN, `R/internalFunctions.R::coefHTiny`, `R/coefZ.R`,
//! `R/search.normal.R`). Mokken (1971) and Sijtsma & Molenaar (2002) were NOT
//! read directly; claims from them are relayed via the above sources. No
//! primary-source derivation of the Z normal approximation was verified;
//! it is implementation-verified only.
//!
//! References (APA 7th ed.):
//! - van der Ark, L. A. (2007). Mokken scale analysis in R. *Journal of
//!   Statistical Software, 20*(11), 1-19. https://doi.org/10.18637/jss.v020.i11
//! - Straat, J. H., van der Ark, L. A., & Sijtsma, K. (2013). Comparing
//!   optimization algorithms for item selection in Mokken scale analysis.
//!   *Journal of Classification, 30*(1), 75-99.
//!   https://doi.org/10.1007/s00357-013-9122-y
//! - Mokken, R. J. (1971). *A theory and procedure of scale analysis*.
//!   De Gruyter. (as cited in van der Ark, 2007, and Straat et al., 2013)
//! - Sijtsma, K., & Molenaar, I. W. (2002). *Introduction to nonparametric
//!   item response theory*. Sage. (as cited in Straat et al., 2013)

/// Scalability coefficients and Mokken Z statistics for one item set.
#[derive(Debug, Clone)]
pub struct MokkenH {
    /// Row-major `n_items x n_items`; `hij[i*J + j] = Hij`, diagonal = NaN.
    pub hij: Vec<f64>,
    /// Per-item scalability `Hi`.
    pub hi: Vec<f64>,
    /// Total scale coefficient `H`.
    pub h: f64,
    /// Row-major `n_items x n_items` Mokken Z; diagonal = NaN.
    pub zij: Vec<f64>,
    /// Per-item Z.
    pub zi: Vec<f64>,
    /// Total Z.
    pub z: f64,
}

fn validate(x: &[i64], n_persons: usize, n_items: usize) -> Result<(), String> {
    if n_persons < 3 {
        return Err("mokken requires at least 3 persons".to_string());
    }
    if n_items < 2 {
        return Err("mokken requires at least 2 items".to_string());
    }
    let expected = crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if x.len() != expected {
        return Err(format!(
            "responses length {} != n_persons*n_items {}",
            x.len(),
            expected
        ));
    }
    if x.iter().any(|&v| v < 0) {
        return Err("scores must be nonnegative integers".to_string());
    }
    Ok(())
}

/// Pairwise machinery shared by `coef_h` and `aisp`: covariance matrix `s`,
/// sorted-column max-covariance matrix `smax`, and per-column variances
/// (diagonal of `s`). All with denominator N-1.
fn pairwise(x: &[i64], n_persons: usize, n_items: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    let n = n_persons as f64;
    let j = n_items;
    // column means and centered columns; sorted centered columns for smax
    let mut cols: Vec<Vec<f64>> = Vec::with_capacity(j);
    let mut sorted: Vec<Vec<f64>> = Vec::with_capacity(j);
    for it in 0..j {
        let mut c: Vec<f64> = (0..n_persons).map(|p| x[p * j + it] as f64).collect();
        let mean = c.iter().sum::<f64>() / n;
        for v in c.iter_mut() {
            *v -= mean;
        }
        let mut s = c.clone();
        s.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        cols.push(c);
        sorted.push(s);
    }
    let denom = n - 1.0;
    let mut s = vec![0.0; j * j];
    let mut smax = vec![0.0; j * j];
    for a in 0..j {
        for b in a..j {
            let cov = cols[a]
                .iter()
                .zip(cols[b].iter())
                .map(|(u, v)| u * v)
                .sum::<f64>()
                / denom;
            let cmx = sorted[a]
                .iter()
                .zip(sorted[b].iter())
                .map(|(u, v)| u * v)
                .sum::<f64>()
                / denom;
            s[a * j + b] = cov;
            s[b * j + a] = cov;
            smax[a * j + b] = cmx;
            smax[b * j + a] = cmx;
        }
        if s[a * j + a] <= 0.0 {
            return Err(format!("item {a} has zero variance"));
        }
    }
    Ok((s, smax))
}

/// H and Z coefficients for the item subset `idx` (crate-internal; `idx`
/// indexes into the full `s`/`smax` matrices of width `j_full`).
fn h_subset(s: &[f64], smax: &[f64], j_full: usize, idx: &[usize], n_persons: usize) -> (Vec<f64>, f64, Vec<f64>, f64) {
    let k = idx.len();
    let sqn = ((n_persons - 1) as f64).sqrt();
    let mut hi = vec![0.0; k];
    let mut zi = vec![0.0; k];
    let (mut num, mut den, mut vsum) = (0.0, 0.0, 0.0);
    for (a, &ia) in idx.iter().enumerate() {
        let (mut na, mut da, mut va) = (0.0, 0.0, 0.0);
        for &ib in idx.iter() {
            if ia == ib {
                continue;
            }
            na += s[ia * j_full + ib];
            da += smax[ia * j_full + ib];
            va += s[ia * j_full + ia] * s[ib * j_full + ib];
        }
        hi[a] = na / da;
        zi[a] = na * sqn / va.sqrt();
        num += na;
        den += da;
        vsum += va;
    }
    // each unordered pair counted twice in the row sums
    (hi, num / den, zi, (num / 2.0) * sqn / (vsum / 2.0).sqrt())
}

/// Compute `Hij`, `Hi`, `H` and the Mokken Z statistics for a complete
/// `n_persons x n_items` row-major integer score matrix.
pub fn coef_h(x: &[i64], n_persons: usize, n_items: usize) -> Result<MokkenH, String> {
    validate(x, n_persons, n_items)?;
    let (s, smax) = pairwise(x, n_persons, n_items)?;
    let j = n_items;
    let sqn = ((n_persons - 1) as f64).sqrt();
    let mut hij = vec![f64::NAN; j * j];
    let mut zij = vec![f64::NAN; j * j];
    for a in 0..j {
        for b in 0..j {
            if a != b {
                hij[a * j + b] = s[a * j + b] / smax[a * j + b];
                zij[a * j + b] = s[a * j + b] * sqn / (s[a * j + a] * s[b * j + b]).sqrt();
            }
        }
    }
    let all: Vec<usize> = (0..j).collect();
    let (hi, h, zi, z) = h_subset(&s, &smax, j, &all, n_persons);
    Ok(MokkenH { hij, hi, h, zij, zi, z })
}

/// Standard-normal upper quantile via inverse complementary error function
/// (Acklam-style rational approximation; |error| < 1.15e-9, sufficient for
/// an alpha cut-off). Returns z such that P(N(0,1) > z) = p.
fn normal_upper_quantile(p: f64) -> f64 {
    // invert the CDF at 1 - p using Peter Acklam's approximation
    let q = 1.0 - p;
    debug_assert!(q > 0.0 && q < 1.0);
    const A: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    const B: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    const C: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    const D: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let plow = 0.02425;
    // standard Acklam sign convention: lower branch yields negative values,
    // central passes through, upper is the negated lower expression.
    if q < plow {
        let r = (-2.0 * q.ln()).sqrt();
        (((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0)
    } else if q <= 1.0 - plow {
        let r = q - 0.5;
        let t = r * r;
        (((((A[0] * t + A[1]) * t + A[2]) * t + A[3]) * t + A[4]) * t + A[5]) * r
            / (((((B[0] * t + B[1]) * t + B[2]) * t + B[3]) * t + B[4]) * t + 1.0)
    } else {
        let r = (-2.0 * (1.0 - q).ln()).sqrt();
        -((((((C[0] * r + C[1]) * r + C[2]) * r + C[3]) * r + C[4]) * r + C[5])
            / ((((D[0] * r + D[1]) * r + D[2]) * r + D[3]) * r + 1.0))
    }
}

/// Automated item selection procedure (Mokken's "search normal" AISP).
///
/// Returns a per-item scale label: 0 = unscalable, 1, 2, ... in formation
/// order. `c` is the scalability lower bound (rule of thumb 0.3); `alpha` the
/// nominal significance level (default 0.05 in the literature).
pub fn aisp(
    x: &[i64],
    n_persons: usize,
    n_items: usize,
    c: f64,
    alpha: f64,
) -> Result<Vec<u32>, String> {
    validate(x, n_persons, n_items)?;
    if !(0.0..1.0).contains(&c) {
        return Err("lower bound c must be in [0, 1)".to_string());
    }
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err("alpha must be in (0, 1)".to_string());
    }
    let (s, smax) = pairwise(x, n_persons, n_items)?;
    let j = n_items;
    let sqn = ((n_persons - 1) as f64).sqrt();
    let hij = |a: usize, b: usize| s[a * j + b] / smax[a * j + b];
    let zij = |a: usize, b: usize| s[a * j + b] * sqn / (s[a * j + a] * s[b * j + b]).sqrt();

    let mut in_set = vec![0u32; j];
    let mut scale = 0u32;
    loop {
        scale += 1;
        let free: Vec<usize> = (0..j).filter(|&i| in_set[i] == 0).collect();
        if free.len() < 2 {
            break;
        }
        // Bonferroni accumulation: k_counts[0] = K1 = #free at scale start;
        // later entries are candidate counts of each add step (resets per scale).
        let k1 = free.len() as f64;
        let mut k_rest = 0.0f64;
        let z_c = |k_rest: f64| {
            let adj = alpha / (k1 * (k1 - 1.0) * 0.5 + k_rest);
            normal_upper_quantile(adj)
        };
        // start pair: max Hij among free pairs with |Zij| >= Z_c. Ties mirror
        // mokken's eps rule (search.normal.R subtracts row*1e-10 where row is
        // the LARGER member index): smaller larger-member index wins, then
        // smaller smaller-member index.
        let zc0 = z_c(0.0);
        let mut best: Option<(usize, usize, f64)> = None;
        for (ai, &a) in free.iter().enumerate() {
            for &b in free.iter().skip(ai + 1) {
                if zij(a, b).abs() < zc0 {
                    continue;
                }
                let h = hij(a, b);
                let better = match best {
                    None => true,
                    Some((ba, bb, bh)) => h > bh || (h == bh && (b, a) < (bb, ba)),
                };
                if better {
                    best = Some((a, b, h));
                }
            }
        }
        let Some((a0, b0, h0)) = best else { break };
        // pair Hi == Hij for both members; require >= c
        if h0 < c {
            break;
        }
        let mut selected = vec![a0, b0];
        in_set[a0] = scale;
        in_set[b0] = scale;
        // add loop
        loop {
            let candidates: Vec<usize> = (0..j)
                .filter(|&i| in_set[i] == 0)
                .filter(|&i| selected.iter().all(|&sj| hij(i, sj) >= 0.0))
                .collect();
            if candidates.is_empty() {
                break;
            }
            k_rest += candidates.len() as f64;
            let zc = z_c(k_rest);
            let mut best_h = f64::NEG_INFINITY;
            let mut best_item = None;
            for &cand in &candidates {
                let mut aug = selected.clone();
                aug.push(cand);
                let (hi, h_total, zi, _) = h_subset(&s, &smax, j, &aug, n_persons);
                // candidate is last in aug
                if hi[aug.len() - 1] < c {
                    continue;
                }
                if zi[aug.len() - 1] < zc {
                    continue;
                }
                if h_total > best_h {
                    best_h = h_total;
                    best_item = Some(cand);
                }
            }
            match best_item {
                Some(it) if best_h >= c => {
                    in_set[it] = scale;
                    selected.push(it);
                }
                _ => break,
            }
        }
    }
    Ok(in_set)
}

#[cfg(test)]
#[path = "../../../tests/unit/mokken_tests.rs"]
mod tests;
