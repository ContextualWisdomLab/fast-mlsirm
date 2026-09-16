//! Confirmatory DETECT dimensionality analysis (Zhang & Stout, 1999) via
//! conditional-covariance estimation with sum-score conditioning.
//!
//! Given a persons x items matrix of binary (0/1) responses and a known item
//! clustering, this module estimates the pairwise conditional covariances of
//! the items and computes five DETECT-family indices:
//!
//! ```text
//! ccov_ij   = ( ccov(S) + ccov(R_ij) ) / 2         bias-corrected estimate
//! delta_ij  = +1 if cluster(i) == cluster(j) else -1
//!
//! DETECT    = 100 * mean( ccov_ij * delta_ij )
//! ASSI      =       mean( sign(ccov_ij) * delta_ij )        sign(0) = 0
//! RATIO     = sum( ccov_ij * delta_ij ) / sum |ccov_ij|
//! MADCOV100 = 100 * mean |ccov_ij|
//! MCOV100   = 100 * mean( ccov_ij )
//! ```
//!
//! where `S_p = sum_k X_pk` is the raw total score, `R_ij,p = S_p - X_pi -
//! X_pj` is the pair rest score, and for a conditioning score `c`:
//!
//! ```text
//! ccov(c) = sum_s w_s * MLcov_s(X_i, X_j),   w_s = #{p : c_p = s} / N
//! ```
//!
//! with `MLcov_s` the maximum-likelihood covariance (divide by group size
//! `n_s`, not `n_s - 1`) within the group of persons whose conditioning score
//! equals `s`. A singleton group contributes covariance 0.
//!
//! # Verified sources
//!
//! Formulas were transcribed line by line from the CRAN `sirt` R package
//! sources: `detect.index.R` (the five indices and the `delta` sign
//! convention), `ccov.np.R` (sum-score conditioning, pair rest score
//! `score - X_i - X_j`, bias correction as the average of the total-score and
//! rest-score estimates), `ccov_np_compute_ccov_sum_score.R` (per-group ML
//! covariance aggregated with group-frequency weights recomputed from each
//! conditioning vector), and `conf.detect.R` (driver wiring). The original
//! DETECT papers (Zhang & Stout, 1999a, 1999b; Stout et al., 1996; Zhang,
//! 2007) were NOT read (paywalled) and are cited only as cited in the `sirt`
//! documentation.
//!
//! # Scope: which `sirt` path this matches
//!
//! This implements the explicit non-default oracle path
//! `ccov.np(data, use_sum_score = TRUE, scale_score = FALSE, bias_corr =
//! TRUE)`, equivalently `conf.detect(..., smooth = FALSE, use_sum_score =
//! TRUE)`: the conditioning score is the RAW integer row sum. The `sirt`
//! DEFAULT (`scale_score = TRUE`) z-standardizes the row sum and rounds it to
//! 3 decimals before grouping; because grouping by unique values is invariant
//! to strictly monotone transforms, both paths agree whenever the rounding
//! merges no groups, but parity is only claimed for the raw-score path.
//! Kernel-smoothed conditioning (`smooth = TRUE`), exploratory cluster
//! search (`expl.detect`), polytomous DETECT (Zhang, 2007), and multiple
//! conditioning scores are out of scope.
//!
//! # Divergences from the R package (deliberate)
//!
//! - Missing responses are rejected. `sirt` allows them with pairwise
//!   deletion per item pair (and computes the total score with `rowSums`
//!   WITHOUT `na.rm`, so any-missing rows get `NA` scores even for complete
//!   pairs); v1 requires complete binary data.
//! - When every conditional covariance is exactly zero, R's RATIO is
//!   `0/0 = NaN`; this implementation returns an error instead of a silent NaN.
//! - Because missing data are rejected, every pair has the same person count,
//!   so `sirt`'s `sqrt(N)`-weighted index variants coincide with the
//!   unweighted ones and are not computed.
//!
//! Interpretation thresholds quoted in the `sirt` documentation (as cited
//! there from Jang & Roussos, 2007, and Zhang, 2007): DETECT < 0.2 suggests
//! essential unidimensionality and DETECT >= 1.0 sizeable multidimensionality
//! relative to the supplied partition. These are conventions, not enforced.
//!
//! In LLM-as-a-Judge item-quality management this diagnoses whether a rubric
//! partition of judge items behaves as distinct dimensions (positive DETECT
//! with a coherent partition) or as a single dimension (DETECT near zero).
//!
//! # References (APA 7th ed.)
//!
//! Jang, E. E., & Roussos, L. (2007). An investigation into the
//! dimensionality of TOEFL using conditional covariance-based nonparametric
//! approach. *Journal of Educational Measurement, 44*(1), 1-21. (as cited in
//! Robitzsch, 2024)
//!
//! Robitzsch, A. (2024). *sirt: Supplementary item response theory models*
//! (R package). https://CRAN.R-project.org/package=sirt
//!
//! Stout, W., Habing, B., Douglas, J., & Kim, H. R. (1996). Conditional
//! covariance-based nonparametric multidimensionality assessment. *Applied
//! Psychological Measurement, 20*(4), 331-354. (as cited in Robitzsch, 2024)
//!
//! Zhang, J. (2007). Conditional covariance theory and DETECT for polytomous
//! items. *Psychometrika, 72*(1), 69-91. (as cited in Robitzsch, 2024)
//!
//! Zhang, J., & Stout, W. (1999a). Conditional covariance structure of
//! generalized compensatory multidimensional items. *Psychometrika, 64*(2),
//! 129-152. (as cited in Robitzsch, 2024)
//!
//! Zhang, J., & Stout, W. (1999b). The theoretical DETECT index of
//! dimensionality and its application to approximate simple structure.
//! *Psychometrika, 64*(2), 213-249. (as cited in Robitzsch, 2024)

/// Result of a confirmatory DETECT analysis.
#[derive(Debug, Clone)]
pub struct DetectResult {
    /// DETECT index (x100 scale).
    pub detect: f64,
    /// Approximate simple structure index (unscaled, in [-1, 1]).
    pub assi: f64,
    /// RATIO index (unscaled, in [-1, 1]).
    pub ratio: f64,
    /// Mean absolute conditional covariance (x100 scale).
    pub madcov100: f64,
    /// Mean conditional covariance (x100 scale).
    pub mcov100: f64,
    /// Number of item pairs `I (I - 1) / 2`.
    pub n_pairs: usize,
    /// First item index of each pair, `i < j`, row-major order.
    pub pair_i: Vec<usize>,
    /// Second item index of each pair.
    pub pair_j: Vec<usize>,
    /// Bias-corrected conditional covariance per pair.
    pub ccov: Vec<f64>,
}

/// ML covariance of `(x_i, x_j)` within one conditioning group given by
/// `idx` (divide by group size, not size - 1). Singleton groups return 0.
fn ml_cov_group(xi: &[f64], xj: &[f64], idx: &[usize]) -> f64 {
    let n = idx.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let (mut mi, mut mj) = (0.0, 0.0);
    for &p in idx {
        mi += xi[p];
        mj += xj[p];
    }
    mi /= nf;
    mj /= nf;
    let mut s = 0.0;
    for &p in idx {
        s += (xi[p] - mi) * (xj[p] - mj);
    }
    s / nf
}

/// Aggregate conditional covariance over the groups of a conditioning score
/// vector: `sum_s w_s * MLcov_s` with `w_s` the group frequency / N. The
/// group table is recomputed from `score` on every call (the rest-score pass
/// must NOT reuse total-score weights; `ccov_np_compute_ccov_sum_score.R`
/// builds `wgt_score` from the supplied vector).
fn ccov_sum_score(score: &[i64], xi: &[f64], xj: &[f64]) -> f64 {
    let n = score.len() as f64;
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_unstable_by_key(|&p| score[p]);
    ccov_sum_score_presorted(score, xi, xj, n, &order)
}

/// Aggregate conditional covariance using a pre-sorted person ordering.
/// `order` must already be sorted by ascending `score` value; `n` is
/// `score.len() as f64`. This avoids re-sorting `score` on every pair for
/// vectors whose sort order is constant across pairs (i.e. the total score).
fn ccov_sum_score_presorted(score: &[i64], xi: &[f64], xj: &[f64], n: f64, order: &[usize]) -> f64 {
    let mut total = 0.0;
    let mut start = 0;
    while start < order.len() {
        let s = score[order[start]];
        let mut end = start + 1;
        while end < order.len() && score[order[end]] == s {
            end += 1;
        }
        let group = &order[start..end];
        let w = group.len() as f64 / n;
        total += w * ml_cov_group(xi, xj, group);
        start = end;
    }
    total
}

/// Confirmatory DETECT analysis of a binary response matrix.
///
/// `responses` is row-major `n_persons x n_items` with entries exactly 0.0 or
/// 1.0 (missing data rejected). `cluster` assigns one label per item; labels
/// are opaque (compared for equality only).
pub fn detect_analysis(
    responses: &[f64],
    n_persons: usize,
    n_items: usize,
    cluster: &[i64],
) -> Result<DetectResult, String> {
    if n_persons < 2 {
        return Err("detect: need at least 2 persons".to_string());
    }
    if n_items < 2 {
        return Err("detect: need at least 2 items".to_string());
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "detect: n_persons * n_items overflows".to_string())?;
    if responses.len() != expected {
        return Err(format!(
            "detect: responses length {} != n_persons {} x n_items {}",
            responses.len(),
            n_persons,
            n_items
        ));
    }
    if cluster.len() != n_items {
        return Err(format!(
            "detect: cluster length {} != n_items {}",
            cluster.len(),
            n_items
        ));
    }
    for &x in responses {
        if x != 0.0 && x != 1.0 {
            return Err(
                "detect: responses must be exactly 0 or 1 (missing data not supported)".to_string(),
            );
        }
    }
    let n_pairs = n_items
        .checked_mul(n_items - 1)
        .map(|m| m / 2)
        .ok_or_else(|| "detect: pair count overflows".to_string())?;

    // Column-extract items and raw total scores (integers, exact in f64->i64).
    let mut items: Vec<Vec<f64>> = vec![vec![0.0; n_persons]; n_items];
    let mut total: Vec<i64> = vec![0; n_persons];
    for p in 0..n_persons {
        for i in 0..n_items {
            let x = responses[p * n_items + i];
            items[i][p] = x;
            total[p] += x as i64;
        }
    }

    let mut pair_i = Vec::with_capacity(n_pairs);
    let mut pair_j = Vec::with_capacity(n_pairs);
    let mut ccov = Vec::with_capacity(n_pairs);
    let mut rest: Vec<i64> = vec![0; n_persons];
    // Precompute the total-score person ordering once; it is the same for
    // every pair (total is invariant across pairs). The rest-score must still
    // be re-sorted per pair because it changes with (i, j).
    let n_persons_f = n_persons as f64;
    let mut total_order: Vec<usize> = (0..n_persons).collect();
    total_order.sort_unstable_by_key(|&p| total[p]);
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            let (xi, xj) = (&items[i], &items[j]);
            for p in 0..n_persons {
                rest[p] = total[p] - xi[p] as i64 - xj[p] as i64;
            }
            // Bias correction (ccov.np.R:96-104): average of the covariance
            // conditioned on the total score and on the pair rest score.
            let c1 = ccov_sum_score_presorted(&total, xi, xj, n_persons_f, &total_order);
            let c2 = ccov_sum_score(&rest, xi, xj);
            pair_i.push(i);
            pair_j.push(j);
            ccov.push(0.5 * (c1 + c2));
        }
    }

    let m = n_pairs as f64;
    let (mut sum_cd, mut sum_abs, mut sum_c, mut sum_sd) = (0.0, 0.0, 0.0, 0.0);
    for k in 0..n_pairs {
        let c = ccov[k];
        let delta = if cluster[pair_i[k]] == cluster[pair_j[k]] {
            1.0
        } else {
            -1.0
        };
        sum_cd += c * delta;
        sum_abs += c.abs();
        sum_c += c;
        // R's sign(): sign(0) = 0.
        let sg = if c > 0.0 {
            1.0
        } else if c < 0.0 {
            -1.0
        } else {
            0.0
        };
        sum_sd += sg * delta;
    }
    if sum_abs == 0.0 {
        return Err(
            "detect: all conditional covariances are zero; RATIO is undefined (0/0)".to_string(),
        );
    }
    Ok(DetectResult {
        detect: 100.0 * sum_cd / m,
        assi: sum_sd / m,
        ratio: sum_cd / sum_abs,
        madcov100: 100.0 * sum_abs / m,
        mcov100: 100.0 * sum_c / m,
        n_pairs,
        pair_i,
        pair_j,
        ccov,
    })
}

// ---------------------------------------------------------------------------
// Confirmatory Stout-style DIMTEST statistic for complete binary responses.
//
// READ sources: formulas for the original AT1/AT2 statistic were transcribed
// from Nandakumar & Stout's 1992 ERIC technical-report version (ED351383) of
// "Refinements of Stout's Procedure for Assessing Latent Trait
// Unidimensionality" (published 1993, Journal of Educational Statistics,
// 18(1), 41-68), especially the section that describes Stout (1987, Sec. 4):
// the AT1/AT2/PT split, PT-score subgroups with too-few-examinee elimination
// (Jmin = 20 recommended), the subgroup variance estimates
// `sigma_hat_k^2` (ML, divide by J_k) and unidimensional
// `sigma_hat_U,k^2 = M^-2 sum_i p_hat_i(1 - p_hat_i)`, the standard-error
// estimate
// `S_k^2 = [(mu_hat_4,k - sigma_hat_k^4) + delta_hat_4,k / M^4
//           + 2 sqrt((mu_hat_4,k - sigma_hat_k^4) delta_hat_4,k / M^4)] / J_k`
// with `delta_hat_4,k = sum_i p_hat_i(1-p_hat_i)(1-2 p_hat_i)^2`, the
// normalized sum `T_L = K^{-1/2} sum_k (sigma_hat_k^2 - sigma_hat_U,k^2)/S_k`,
// the same computation on AT2 giving T_B, the bias-corrected statistic
// `T = (T_L - T_B)/sqrt(2)`, and the one-sided rejection rule `T > Z_alpha`.
// Kieftenbeld & Nandakumar (2015, PMC5978610) was READ for the distinction
// between the original second-AT bias correction and the later
// bootstrap-based "current" DIMTEST (Stout et al., 2001).
//
// NOT READ: Stout (1987) original Psychometrika article, Stout et al. (2001),
// Froelich & Habing (2008), and the DIM-Pack source code were not available
// as inspectable sources. Stout (1987) is cited only as described by
// Nandakumar & Stout (1992/1993).
//
// Scope: caller-supplied confirmatory AT1/AT2 only. No ATFIND, no automatic
// difficulty matching, no DIMTEST 2 / bootstrap bias correction, no
// polytomous items, no missing data.
//
// References:
//
// Nandakumar, R., & Stout, W. (1993). Refinements of Stout's procedure for
// assessing latent trait unidimensionality. Journal of Educational
// Statistics, 18(1), 41-68. https://doi.org/10.2307/1165182 (READ as the
// 1992 ERIC technical report ED351383)
//
// Stout, W. (1987). A nonparametric approach for assessing latent trait
// unidimensionality. Psychometrika, 52(4), 589-617. (NOT read; as described
// by Nandakumar & Stout, 1992/1993)
//
// Kieftenbeld, V., & Nandakumar, R. (2015). Alternative hypothesis testing
// procedures for DIMTEST. Applied Psychological Measurement, 39(6), 480-493.
// (READ via PMC5978610)
// ---------------------------------------------------------------------------

/// Result of the confirmatory Stout-style DIMTEST statistic.
#[derive(Debug, Clone)]
pub struct DimtestResult {
    /// Bias-corrected statistic `T = (T_L - T_B) / sqrt(2)`.
    pub t: f64,
    /// AT1 statistic `T_L`.
    pub t_l: f64,
    /// AT2 bias-correction statistic `T_B`.
    pub t_b: f64,
    /// One-sided upper-tail p-value `1 - Phi(T)`.
    pub p_value: f64,
    /// Number of retained PT-score groups (`J_k >= 20`).
    pub groups_used: usize,
    /// Number of examinees discarded with too-small groups.
    pub n_discarded: usize,
    /// Raw PT total scores of the retained groups, ascending.
    pub retained_pt_scores: Vec<i64>,
}

/// Minimum retained PT-score group size (Nandakumar & Stout, 1992/1993:
/// "Jmin=20 recommended"). Fixed, not caller-tunable, to keep the pinned
/// contract exact.
const DIMTEST_JMIN: usize = 20;

/// Groups persons by raw PT total score, keeping only groups with at least
/// `DIMTEST_JMIN` examinees. Returns `(groups, retained_pt_scores,
/// n_discarded)` with groups in ascending PT-score order. Factored out of
/// `dimtest` so tests can pin per-group intermediates against the oracle.
fn dimtest_pt_groups(
    responses: &[f64],
    n_persons: usize,
    n_items: usize,
    pt: &[usize],
) -> (Vec<Vec<usize>>, Vec<i64>, usize) {
    let mut pt_score: Vec<i64> = vec![0; n_persons];
    for (p, s) in pt_score.iter_mut().enumerate() {
        *s = pt.iter().map(|&i| responses[p * n_items + i] as i64).sum();
    }
    let mut order: Vec<usize> = (0..n_persons).collect();
    order.sort_unstable_by_key(|&p| pt_score[p]);
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut retained_pt_scores: Vec<i64> = Vec::new();
    let mut n_discarded = 0usize;
    let mut start = 0;
    while start < order.len() {
        let s = pt_score[order[start]];
        let mut end = start + 1;
        while end < order.len() && pt_score[order[end]] == s {
            end += 1;
        }
        if end - start >= DIMTEST_JMIN {
            groups.push(order[start..end].to_vec());
            retained_pt_scores.push(s);
        } else {
            n_discarded += end - start;
        }
        start = end;
    }
    (groups, retained_pt_scores, n_discarded)
}

/// Per-group DIMTEST intermediates for one assessment subtest, exposed so
/// tests can pin every step of the Nandakumar & Stout (1992/1993) formula
/// against an independent oracle (not just the top-level statistics).
#[derive(Debug, Clone, Copy)]
struct DimtestGroupDiag {
    /// Retained group size `J_k`.
    jk: usize,
    /// Mean raw AT total score in the group.
    mean: f64,
    /// ML (divide-by-`J_k`) variance of the AT total, `sigma_hat_k^2`.
    v: f64,
    /// Unidimensional-null variance `sigma_hat_U,k^2 = sum_i p_i (1 - p_i)`.
    u: f64,
    /// ML fourth central moment of the AT total, `mu4_k`.
    mu4: f64,
    /// `delta4_k = sum_i p_i (1 - p_i) (1 - 2 p_i)^2`.
    delta4: f64,
    /// Refined bias-correction denominator squared, `S_k^2`.
    s2: f64,
    /// Group contribution `(sigma_hat_k^2 - sigma_hat_U,k^2) / S_k`.
    contribution: f64,
}

/// Computes the DIMTEST intermediates for one retained group and one
/// assessment subtest, on the raw-total-score scale.
fn dimtest_group_diag(
    idx: &[usize],
    responses: &[f64],
    n_items: usize,
    subtest: &[usize],
) -> Result<DimtestGroupDiag, String> {
    let jk = idx.len() as f64;
    // Raw AT total per retained person in this group.
    let totals: Vec<f64> = idx
        .iter()
        .map(|&p| subtest.iter().map(|&i| responses[p * n_items + i]).sum())
        .collect();
    let mean = totals.iter().sum::<f64>() / jk;
    let mut v = 0.0;
    let mut mu4 = 0.0;
    for &a in &totals {
        let d = a - mean;
        v += d * d;
        mu4 += d * d * d * d;
    }
    v /= jk;
    mu4 /= jk;
    // Item proportions correct within the group.
    let mut u = 0.0;
    let mut delta4 = 0.0;
    for &i in subtest {
        let p = idx.iter().map(|&q| responses[q * n_items + i]).sum::<f64>() / jk;
        let pq = p * (1.0 - p);
        u += pq;
        let c = 1.0 - 2.0 * p;
        delta4 += pq * c * c;
    }
    let aterm = (mu4 - v * v).max(0.0);
    let s2 = (aterm + delta4 + 2.0 * (aterm * delta4).sqrt()) / jk;
    if s2 <= 0.0 {
        return Err("dimtest: a retained group has zero standard error (S_k = 0)".to_string());
    }
    Ok(DimtestGroupDiag {
        jk: idx.len(),
        mean,
        v,
        u,
        mu4,
        delta4,
        s2,
        contribution: (v - u) / s2.sqrt(),
    })
}

/// Normalized subgroup sum for one assessment subtest: returns
/// `K^{-1/2} sum_k (sigma_hat_k^2 - sigma_hat_U,k^2) / S_k` computed on the
/// raw-total-score scale, which is algebraically identical to the
/// average-score (divide-by-M) scale because numerator and S_k both scale by
/// `M^2` exactly.
fn dimtest_stat(
    groups: &[Vec<usize>],
    responses: &[f64],
    n_items: usize,
    subtest: &[usize],
) -> Result<f64, String> {
    let k_groups = groups.len() as f64;
    let mut sum = 0.0;
    for idx in groups {
        sum += dimtest_group_diag(idx, responses, n_items, subtest)?.contribution;
    }
    Ok(sum / k_groups.sqrt())
}

/// Confirmatory Stout-style DIMTEST of essential unidimensionality.
///
/// `responses` is row-major `n_persons x n_items` with entries exactly 0.0 or
/// 1.0 (missing data rejected). `at1` and `at2` are caller-supplied item
/// index sets (assessment subtests); the partitioning subtest PT is the
/// complement. Persons are grouped by raw PT total score; groups with fewer
/// than 20 examinees are discarded. Returns `T = (T_L - T_B)/sqrt(2)` with a
/// one-sided upper-tail p-value.
pub fn dimtest(
    responses: &[f64],
    n_persons: usize,
    n_items: usize,
    at1: &[usize],
    at2: &[usize],
) -> Result<DimtestResult, String> {
    if n_persons == 0 {
        return Err("dimtest: need at least 1 person".to_string());
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "dimtest: n_persons * n_items overflows".to_string())?;
    if responses.len() != expected {
        return Err(format!(
            "dimtest: responses length {} != n_persons {} x n_items {}",
            responses.len(),
            n_persons,
            n_items
        ));
    }
    for &x in responses {
        if x != 0.0 && x != 1.0 {
            return Err(
                "dimtest: responses must be exactly 0 or 1 (missing data not supported)"
                    .to_string(),
            );
        }
    }
    if at1.len() < 4 || at1.len() != at2.len() {
        return Err("dimtest: AT1 and AT2 must have equal length >= 4 items".to_string());
    }
    let mut role = vec![0u8; n_items]; // 0 = PT, 1 = AT1, 2 = AT2
    for &i in at1 {
        if i >= n_items {
            return Err(format!("dimtest: AT1 index {} out of range", i));
        }
        if role[i] != 0 {
            return Err(format!("dimtest: duplicate AT1 index {}", i));
        }
        role[i] = 1;
    }
    for &i in at2 {
        if i >= n_items {
            return Err(format!("dimtest: AT2 index {} out of range", i));
        }
        if role[i] != 0 {
            return Err(format!("dimtest: AT2 index {} duplicates AT1/AT2", i));
        }
        role[i] = 2;
    }
    let pt: Vec<usize> = (0..n_items).filter(|&i| role[i] == 0).collect();
    if pt.is_empty() {
        return Err("dimtest: partitioning subtest PT is empty".to_string());
    }

    // Group persons by raw PT total score (exact integers in f64).
    let (groups, retained_pt_scores, n_discarded) =
        dimtest_pt_groups(responses, n_persons, n_items, &pt);
    if groups.len() < 2 {
        return Err(format!(
            "dimtest: only {} PT-score group(s) with >= {} examinees; need at least 2",
            groups.len(),
            DIMTEST_JMIN
        ));
    }

    let t_l = dimtest_stat(&groups, responses, n_items, at1)?;
    let t_b = dimtest_stat(&groups, responses, n_items, at2)?;
    let t = (t_l - t_b) / std::f64::consts::SQRT_2;
    // One-sided upper tail: 1 - Phi(t) = 0.5 * erfc(t / sqrt(2)).
    let p_value = 0.5 * crate::fitstats::erfc(t / std::f64::consts::SQRT_2);
    Ok(DimtestResult {
        t,
        t_l,
        t_b,
        p_value,
        groups_used: groups.len(),
        n_discarded,
        retained_pt_scores,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/detect_tests.rs"]
mod tests;
