//! Family-wise-error-controlled candidate-superiority arithmetic.
//!
//! This module is a deliberately narrow numerical foundation. It tests the
//! ordered null hypotheses `H0(i,j): mu_i <= mu_j` with one-sided Wald tests,
//! then applies Holm's (1979) sequentially rejective procedure across the full
//! ordered family. A candidate is reported only when every outgoing null is
//! rejected. The caller supplies the family-wise error rate from a governed
//! decision policy; this module does not invent a default loss or threshold.
//!
//! The kernel does not estimate covariance or certify calibration. Consumers
//! must not treat it as route-authorizing evidence until those upstream
//! contracts are supplied and validated.

const ERFC_ABSOLUTE_ERROR_BOUND: f64 = 1.2e-7;

/// One ordered Wald comparison after Holm family-wise correction.
#[derive(Clone, Debug, PartialEq)]
pub struct HolmWaldComparison {
    /// Candidate whose superiority is tested.
    pub candidate_index: usize,
    /// Comparator in `H0: candidate <= comparator`.
    pub comparator_index: usize,
    /// Estimated candidate-minus-comparator contrast.
    pub estimate_difference: f64,
    /// Standard error derived from the full covariance matrix.
    pub standard_error: f64,
    /// Conservative upper bound for the one-sided standard-normal p-value.
    pub p_value_upper_bound: f64,
    /// Whether Holm's step-down procedure rejected this ordered null.
    pub null_rejected: bool,
}

/// Holm--Wald result for one complete candidate family.
#[derive(Clone, Debug, PartialEq)]
pub struct HolmWaldSuperiority {
    /// Unique candidate that rejected every outgoing ordered null, if any.
    pub winner_index: Option<usize>,
    /// All ordered comparisons in input-index order.
    pub comparisons: Vec<HolmWaldComparison>,
}

/// Verify positive definiteness with scale-normalized, residual-checked Cholesky.
///
/// Each pivot must exceed its IEEE-754 dot-product rounding bound. This makes
/// numerically unresolved pivots fail closed instead of converting an absolute
/// eigensolver tolerance into a scale-dependent acceptance rule.
fn positive_definite_cholesky(covariance: &[f64], n: usize) -> bool {
    let scale = covariance
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return false;
    }
    let mut lower = vec![0.0_f64; covariance.len()];
    for row in 0..n {
        for column in 0..=row {
            let normalized = covariance[row * n + column] / scale;
            let mut value = normalized;
            let mut magnitude_sum = normalized.abs();
            for k in 0..column {
                let product = lower[row * n + k] * lower[column * n + k];
                value -= product;
                magnitude_sum += product.abs();
            }
            if !value.is_finite() || !magnitude_sum.is_finite() {
                return false;
            }
            if row == column {
                let operations = 2 * column + 1;
                let roundoff = operations as f64 * f64::EPSILON;
                if roundoff >= 1.0 {
                    return false;
                }
                let rounding_bound = roundoff / (1.0 - roundoff) * magnitude_sum;
                if value <= rounding_bound {
                    return false;
                }
                lower[row * n + column] = value.sqrt();
            } else {
                let factor = value / lower[column * n + column];
                if !factor.is_finite() {
                    return false;
                }
                lower[row * n + column] = factor;
            }
        }
    }

    for row in 0..n {
        for column in 0..n {
            let terms = row.min(column) + 1;
            let mut reconstructed = 0.0;
            let mut magnitude_sum = 0.0;
            for k in 0..terms {
                let product = lower[row * n + k] * lower[column * n + k];
                reconstructed += product;
                magnitude_sum += product.abs();
            }
            let normalized = covariance[row * n + column] / scale;
            let operations = 2 * terms + 1;
            let roundoff = operations as f64 * f64::EPSILON;
            if roundoff >= 1.0 {
                return false;
            }
            let rounding_bound = roundoff / (1.0 - roundoff) * (normalized.abs() + magnitude_sum);
            if !reconstructed.is_finite()
                || !rounding_bound.is_finite()
                || (normalized - reconstructed).abs() > rounding_bound
            {
                return false;
            }
        }
    }
    true
}

/// Conservative one-sided standard-normal survival probability.
///
/// `fitstats::erfc` documents absolute error below `1.2e-7`. Adding half that
/// bound converts its reporting approximation into an upper probability bound,
/// so numerical approximation cannot create a Holm rejection.
fn normal_survival_upper_bound(z: f64) -> Result<f64, String> {
    if !z.is_finite() {
        return Err("pairwise contrast arithmetic must remain finite".into());
    }
    let approximation = 0.5 * crate::fitstats::erfc(z / std::f64::consts::SQRT_2);
    if !approximation.is_finite() {
        return Err("pairwise contrast arithmetic must remain finite".into());
    }
    Ok((approximation + 0.5 * ERFC_ABSOLUTE_ERROR_BOUND).clamp(0.0, 1.0))
}

/// Test whether one candidate is superior to every other candidate.
///
/// `covariance` is row-major `n x n` covariance for `estimates`. It must be
/// finite, exactly symmetric, and positive definite. `familywise_error_rate`
/// must be supplied by the caller's documented decision policy and lie in
/// `(0, 1)`. Holm's procedure provides strong family-wise type-I error control
/// without an independence assumption (Holm, 1979, pp. 65--70).
pub fn holm_wald_superiority(
    estimates: &[f64],
    covariance: &[f64],
    familywise_error_rate: f64,
) -> Result<HolmWaldSuperiority, String> {
    let n = estimates.len();
    if n < 2 {
        return Err("at least two candidate estimates are required".into());
    }
    if estimates.iter().any(|value| !value.is_finite()) {
        return Err("estimates must be finite".into());
    }
    if !familywise_error_rate.is_finite()
        || familywise_error_rate <= 0.0
        || familywise_error_rate >= 1.0
    {
        return Err(
            "familywise_error_rate must be finite and strictly between zero and one".into(),
        );
    }
    let matrix_len = n
        .checked_mul(n)
        .ok_or_else(|| "covariance dimension exceeds supported size".to_string())?;
    if covariance.len() != matrix_len {
        return Err("covariance must be a square matrix matching estimates".into());
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err("covariance must be finite".into());
    }
    for row in 0..n {
        for column in (row + 1)..n {
            if covariance[row * n + column] != covariance[column * n + row] {
                return Err("covariance must be symmetric".into());
            }
        }
    }
    if !positive_definite_cholesky(covariance, n) {
        return Err("covariance must be positive definite".into());
    }

    let comparison_count = n
        .checked_mul(n - 1)
        .ok_or_else(|| "candidate comparison count exceeds supported size".to_string())?;
    let mut comparisons = Vec::new();
    comparisons
        .try_reserve_exact(comparison_count)
        .map_err(|_| "candidate comparison allocation failed".to_string())?;
    for candidate_index in 0..n {
        for comparator_index in 0..n {
            if candidate_index == comparator_index {
                continue;
            }
            let estimate_difference = estimates[candidate_index] - estimates[comparator_index];
            if !estimate_difference.is_finite() {
                return Err("pairwise contrast arithmetic must remain finite".into());
            }
            let variance = covariance[candidate_index * n + candidate_index]
                + covariance[comparator_index * n + comparator_index]
                - 2.0 * covariance[candidate_index * n + comparator_index];
            if !variance.is_finite() || variance <= 0.0 {
                return Err("each pairwise contrast variance must be finite and positive".into());
            }
            let standard_error = variance.sqrt();
            let z = estimate_difference / standard_error;
            let p_value_upper_bound = normal_survival_upper_bound(z)?;
            comparisons.push(HolmWaldComparison {
                candidate_index,
                comparator_index,
                estimate_difference,
                standard_error,
                p_value_upper_bound,
                null_rejected: false,
            });
        }
    }

    let mut order: Vec<usize> = (0..comparisons.len()).collect();
    order.sort_by(|left, right| {
        comparisons[*left]
            .p_value_upper_bound
            .total_cmp(&comparisons[*right].p_value_upper_bound)
            .then_with(|| {
                comparisons[*left]
                    .candidate_index
                    .cmp(&comparisons[*right].candidate_index)
            })
            .then_with(|| {
                comparisons[*left]
                    .comparator_index
                    .cmp(&comparisons[*right].comparator_index)
            })
    });
    for (rank, comparison_index) in order.into_iter().enumerate() {
        let remaining = comparisons.len() - rank;
        let critical_level = familywise_error_rate / remaining as f64;
        if comparisons[comparison_index].p_value_upper_bound <= critical_level {
            comparisons[comparison_index].null_rejected = true;
        } else {
            break;
        }
    }

    let winners: Vec<usize> = (0..n)
        .filter(|candidate_index| {
            comparisons
                .iter()
                .filter(|comparison| comparison.candidate_index == *candidate_index)
                .all(|comparison| comparison.null_rejected)
        })
        .collect();
    let winner_index = match winners.as_slice() {
        [] => None,
        [winner] => Some(*winner),
        _ => return Err("multiple superiority winners violate the ordered contrast model".into()),
    };
    Ok(HolmWaldSuperiority {
        winner_index,
        comparisons,
    })
}

#[cfg(test)]
mod tests {
    use super::holm_wald_superiority;

    #[test]
    fn equal_estimates_are_indeterminate() {
        let result = holm_wald_superiority(&[0.5, 0.5], &[0.01, 0.0, 0.0, 0.01], 0.05)
            .expect("valid covariance");
        assert_eq!(result.winner_index, None);
    }

    #[test]
    fn separated_candidate_rejects_every_outgoing_null() {
        let result = holm_wald_superiority(
            &[3.0, 0.0, -1.0],
            &[0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01],
            0.05,
        )
        .expect("valid covariance");
        assert_eq!(result.winner_index, Some(0));
        assert!(result
            .comparisons
            .iter()
            .filter(|comparison| comparison.candidate_index == 0)
            .all(|comparison| comparison.null_rejected));
    }

    #[test]
    fn tiny_scale_indefinite_covariance_fails_closed() {
        let error =
            holm_wald_superiority(&[1.0, 0.0], &[1.0e-15, -2.0e-15, -2.0e-15, 1.0e-15], 0.05)
                .expect_err("indefinite covariance must fail at every scale");
        assert_eq!(error, "covariance must be positive definite");
    }

    #[test]
    fn huge_scale_indefinite_covariance_fails_closed() {
        let error = holm_wald_superiority(&[1.0, 0.0], &[1.0e300, 2.0e300, 2.0e300, 1.0e300], 0.05)
            .expect_err("indefinite covariance must fail at every scale");
        assert_eq!(error, "covariance must be positive definite");
    }

    #[test]
    fn finite_inputs_with_overflowing_contrast_fail_closed() {
        let error = holm_wald_superiority(&[f64::MAX, -f64::MAX], &[1.0, 0.0, 0.0, 1.0], 0.05)
            .expect_err("non-finite derived arithmetic must fail closed");
        assert_eq!(error, "pairwise contrast arithmetic must remain finite");
    }

    #[test]
    fn reporting_approximation_cannot_reject_above_holm_boundary() {
        let z = 1.959_963_984_54_f64;
        let result = holm_wald_superiority(&[z, 0.0], &[0.5, 0.0, 0.0, 0.5], 0.05)
            .expect("identity contrast variance");
        let forward = result
            .comparisons
            .iter()
            .find(|comparison| comparison.candidate_index == 0)
            .expect("ordered comparison");
        assert!(forward.p_value_upper_bound > 0.025);
        assert!(!forward.null_rejected);
    }
}
