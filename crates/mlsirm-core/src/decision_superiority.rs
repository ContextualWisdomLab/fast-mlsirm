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

use crate::inference::second_order_test;

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
    /// One-sided standard-normal Wald p-value.
    pub p_value: f64,
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
    let (positive_definite, _, _) = second_order_test(covariance, n, 0.0)?;
    if !positive_definite {
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
            let variance = covariance[candidate_index * n + candidate_index]
                + covariance[comparator_index * n + comparator_index]
                - 2.0 * covariance[candidate_index * n + comparator_index];
            if !variance.is_finite() || variance <= 0.0 {
                return Err("each pairwise contrast variance must be finite and positive".into());
            }
            let standard_error = variance.sqrt();
            let z = estimate_difference / standard_error;
            let p_value =
                (0.5 * crate::fitstats::erfc(z / std::f64::consts::SQRT_2)).clamp(0.0, 1.0);
            comparisons.push(HolmWaldComparison {
                candidate_index,
                comparator_index,
                estimate_difference,
                standard_error,
                p_value,
                null_rejected: false,
            });
        }
    }

    let mut order: Vec<usize> = (0..comparisons.len()).collect();
    order.sort_by(|left, right| {
        comparisons[*left]
            .p_value
            .total_cmp(&comparisons[*right].p_value)
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
        if comparisons[comparison_index].p_value <= critical_level {
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
}
