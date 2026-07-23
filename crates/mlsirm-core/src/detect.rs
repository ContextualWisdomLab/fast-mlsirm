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
//! - When every conditional covariance is exactly zero, R's RATIO is `0/0 =
//! ` `NaN`; this implementation returns an error instead of a silent NaN.
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
    // Group person indices by score value. Scores are small integers
    // (0..=n_items), but rest scores shift them; use sort-based grouping so
    // score VALUES are never used as array indices.
    let mut order: Vec<usize> = (0..score.len()).collect();
    order.sort_unstable_by_key(|&p| score[p]);
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
                "detect: responses must be exactly 0 or 1 (missing data not supported)"
                    .to_string(),
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
    for i in 0..n_items {
        for j in (i + 1)..n_items {
            let (xi, xj) = (&items[i], &items[j]);
            for p in 0..n_persons {
                rest[p] = total[p] - xi[p] as i64 - xj[p] as i64;
            }
            // Bias correction (ccov.np.R:96-104): average of the covariance
            // conditioned on the total score and on the pair rest score.
            let c1 = ccov_sum_score(&total, xi, xj);
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
            "detect: all conditional covariances are zero; RATIO is undefined (0/0)"
                .to_string(),
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

#[cfg(test)]
#[path = "../../../tests/unit/detect_tests.rs"]
mod tests;
