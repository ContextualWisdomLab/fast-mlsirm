//! One-way random-intercept ANOVA variance components and ICC.
//!
//! This module owns domain-neutral numerical arithmetic only. Callers must
//! establish that their observations really form one random classification
//! before invoking it; cross-classified and multiple-membership admission is
//! deliberately outside this API.
//!
//! The method-of-moments variance-component formula follows the NIST Dataplot
//! ONE WAY ANOVA random-effects contract: for `J` clusters and `N` observations,
//! `sigma_between^2 = max(0, (MSB - MSW) / n0)`, `sigma_within^2 = MSW`, and
//! `n0 = (N - sum_j(n_j^2) / N) / (J - 1)`. The ICC is the between component
//! divided by total variance. NIST identifies Searle, Casella, and McCulloch
//! (1992), chapter 3, as a reference for the one-way random classification.
//!
//! # References
//!
//! National Institute of Standards and Technology. (n.d.). *One way ANOVA*.
//! Dataplot reference manual.
//! https://www.itl.nist.gov/div898/software/dataplot/refman1/auxillar/onewayan.htm
//!
//! Searle, S. R., Casella, G., & McCulloch, C. E. (1992). *Variance components*.
//! Wiley. Chapter 3. (Bibliographic scope verified through the NIST reference;
//! this implementation does not claim an equation locator from uninspected text.)

use std::collections::BTreeMap;

/// Method-of-moments result for a one-way random-intercept model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneWayRandomInterceptIcc {
    /// Intraclass correlation `between / (between + within)`.
    pub icc: f64,
    /// Non-negative between-cluster variance component.
    pub between_variance: f64,
    /// Within-cluster variance component (`MSW`).
    pub within_variance: f64,
    /// Unbalanced-design effective cluster size `n0`.
    pub effective_cluster_size_n0: f64,
    /// Number of distinct clusters in the accepted input.
    pub cluster_count: usize,
    /// Number of observations in the accepted input.
    pub sample_size: usize,
}

/// Estimate a one-way random-intercept ICC from finite outcomes and opaque cluster IDs.
///
/// The reduction order is canonicalized by cluster ID and then by `f64::total_cmp`
/// within each cluster. Consequently a row permutation with the same IDs and values
/// produces the same binary64 result; no parallel or hash-map reduction is used in
/// this deterministic CPU reference path.
///
/// This function is **not** a design classifier. Cross-classified,
/// multiple-membership, time-varying, or otherwise non-one-way structures must be
/// rejected or projected by the owning bounded context before this numerical API is
/// called.
///
/// # Errors
///
/// Returns an error for empty or length-mismatched input, non-finite outcomes,
/// fewer than two clusters, non-positive within-cluster residual degrees of freedom,
/// numerical overflow, or degenerate zero total variance.
pub fn one_way_random_intercept_icc(
    cluster_ids: &[u64],
    outcomes: &[f64],
) -> Result<OneWayRandomInterceptIcc, String> {
    if cluster_ids.is_empty() || cluster_ids.len() != outcomes.len() {
        return Err("cluster_ids and outcomes must have the same non-zero length".into());
    }
    if outcomes.iter().any(|value| !value.is_finite()) {
        return Err("outcomes must be finite".into());
    }

    let mut clusters: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for (&cluster_id, &outcome) in cluster_ids.iter().zip(outcomes) {
        clusters.entry(cluster_id).or_default().push(outcome);
    }
    let cluster_count = clusters.len();
    let sample_size = outcomes.len();
    if cluster_count < 2 {
        return Err("one-way ICC requires at least two clusters".into());
    }
    if sample_size <= cluster_count {
        return Err(
            "one-way ICC requires positive within-cluster residual degrees of freedom".into(),
        );
    }

    for values in clusters.values_mut() {
        values.sort_by(f64::total_cmp);
    }

    let mut grand_total = 0.0_f64;
    for values in clusters.values() {
        for &value in values {
            grand_total += value;
            if !grand_total.is_finite() {
                return Err("outcome sum overflowed binary64".into());
            }
        }
    }
    let n = sample_size as f64;
    let j = cluster_count as f64;
    let grand_mean = grand_total / n;

    let mut sum_of_squares_between = 0.0_f64;
    let mut sum_of_squares_within = 0.0_f64;
    let mut sum_cluster_size_squared = 0.0_f64;
    for values in clusters.values() {
        let cluster_size = values.len() as f64;
        sum_cluster_size_squared += cluster_size * cluster_size;

        let mut cluster_total = 0.0_f64;
        for &value in values {
            cluster_total += value;
        }
        let cluster_mean = cluster_total / cluster_size;
        let between_delta = cluster_mean - grand_mean;
        sum_of_squares_between += cluster_size * between_delta * between_delta;
        for &value in values {
            let within_delta = value - cluster_mean;
            sum_of_squares_within += within_delta * within_delta;
        }
    }
    if !sum_of_squares_between.is_finite() || !sum_of_squares_within.is_finite() {
        return Err("ANOVA sums of squares overflowed binary64".into());
    }

    let mean_square_between = sum_of_squares_between / (j - 1.0);
    let mean_square_within = sum_of_squares_within / (n - j);
    let effective_cluster_size_n0 = (n - sum_cluster_size_squared / n) / (j - 1.0);
    let between_variance =
        ((mean_square_between - mean_square_within) / effective_cluster_size_n0).max(0.0);
    let within_variance = mean_square_within;
    let total_variance = between_variance + within_variance;
    if total_variance <= 0.0 {
        return Err("one-way ICC requires positive total variance".into());
    }
    let icc = between_variance / total_variance;

    Ok(OneWayRandomInterceptIcc {
        icc,
        between_variance,
        within_variance,
        effective_cluster_size_n0,
        cluster_count,
        sample_size,
    })
}
