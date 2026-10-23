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

use std::cmp::Ordering;

use num_bigint::{BigInt, Sign};

const ERFC_ABSOLUTE_ERROR_BOUND: f64 = 1.2e-7;

/// One ordered Wald comparison after Holm family-wise correction.
#[derive(Clone, Debug, PartialEq)]
pub struct HolmWaldComparison {
    /// Candidate whose superiority is tested.
    pub candidate_index: usize,
    /// Comparator in `H0: candidate <= comparator`.
    pub comparator_index: usize,
    /// Binary64 summary of the candidate-minus-comparator contrast.
    pub estimate_difference: f64,
    /// Conservative upper standard-error bound from the exact dyadic variance.
    pub standard_error_upper_bound: f64,
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

/// Return the exact signed significand and binary exponent of a finite f64.
fn exact_dyadic_parts(value: f64) -> Option<(bool, u64, i32)> {
    let bits = value.to_bits();
    let negative = bits >> 63 != 0;
    let raw_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if raw_exponent == 0x7ff {
        return None;
    }
    if raw_exponent == 0 {
        return Some((negative, fraction, -1074));
    }
    Some((negative, (1_u64 << 52) | fraction, raw_exponent - 1023 - 52))
}

struct ExactDyadicVector {
    integers: Vec<BigInt>,
    exponent: i32,
}

fn exact_scaled_dyadics(values: &[f64]) -> Option<ExactDyadicVector> {
    let mut parts = Vec::new();
    parts.try_reserve_exact(values.len()).ok()?;
    let mut minimum_exponent = i32::MAX;
    for value in values {
        let part = exact_dyadic_parts(*value)?;
        if part.1 != 0 {
            minimum_exponent = minimum_exponent.min(part.2);
        }
        parts.push(part);
    }
    if minimum_exponent == i32::MAX {
        minimum_exponent = 0;
    }

    let mut integers = Vec::new();
    integers.try_reserve_exact(values.len()).ok()?;
    for (negative, significand, exponent) in parts {
        if significand == 0 {
            integers.push(BigInt::from(0_u8));
            continue;
        }
        let shift = (exponent - minimum_exponent) as usize;
        let magnitude = BigInt::from(significand) << shift;
        integers.push(if negative { -magnitude } else { magnitude });
    }
    Some(ExactDyadicVector {
        integers,
        exponent: minimum_exponent,
    })
}

fn compare_dyadics(
    left_integer: &BigInt,
    left_exponent: i32,
    right_integer: &BigInt,
    right_exponent: i32,
) -> Ordering {
    let common_exponent = left_exponent.min(right_exponent);
    let left_scaled = left_integer << (left_exponent - common_exponent) as usize;
    let right_scaled = right_integer << (right_exponent - common_exponent) as usize;
    left_scaled.cmp(&right_scaled)
}

/// Compare a binary64 p-value bound against the exact Holm `alpha / m` level.
///
/// Cross multiplication keeps both submitted binary64 values exact and avoids
/// admitting a rejection because floating-point division rounded the critical
/// level upward.
fn holm_bound_allows_rejection(
    p_value_upper_bound: f64,
    familywise_error_rate: f64,
    remaining: usize,
) -> Option<bool> {
    let (p_negative, p_significand, p_exponent) = exact_dyadic_parts(p_value_upper_bound)?;
    let (alpha_negative, alpha_significand, alpha_exponent) =
        exact_dyadic_parts(familywise_error_rate)?;
    if p_negative || alpha_negative || remaining == 0 {
        return None;
    }
    let scaled_p = BigInt::from(p_significand) * BigInt::from(remaining);
    let alpha = BigInt::from(alpha_significand);
    Some(compare_dyadics(&scaled_p, p_exponent, &alpha, alpha_exponent) != Ordering::Greater)
}

/// Enclose one exact dyadic value in adjacent finite binary64 values.
fn exact_dyadic_interval(integer: &BigInt, exponent: i32) -> Option<(f64, f64)> {
    if integer.sign() == Sign::NoSign {
        return Some((0.0, 0.0));
    }
    let magnitude = integer.magnitude();
    let bit_count = magnitude.bits();
    let shift = bit_count.saturating_sub(53) as usize;
    let leading = magnitude >> shift;
    let leading_digits = leading.to_u64_digits();
    let leading_u64 = *leading_digits.first()?;
    let scale_exponent = exponent.checked_add(i32::try_from(shift).ok()?)?;
    let mut candidate = leading_u64 as f64 * 2.0_f64.powi(scale_exponent);
    if integer.sign() == Sign::Minus {
        candidate = -candidate;
    }
    if !candidate.is_finite() || candidate == 0.0 {
        return None;
    }

    let (candidate_negative, candidate_significand, candidate_exponent) =
        exact_dyadic_parts(candidate)?;
    let candidate_magnitude = BigInt::from(candidate_significand);
    let candidate_integer = if candidate_negative {
        -candidate_magnitude
    } else {
        candidate_magnitude
    };
    match compare_dyadics(&candidate_integer, candidate_exponent, integer, exponent) {
        Ordering::Equal => Some((candidate, candidate)),
        Ordering::Less => {
            let upper = candidate.next_up();
            upper.is_finite().then_some((candidate, upper))
        }
        Ordering::Greater => {
            let lower = candidate.next_down();
            lower.is_finite().then_some((lower, candidate))
        }
    }
}

fn conservative_wald_inputs(
    exact_difference: &BigInt,
    difference_exponent: i32,
    exact_variance: &BigInt,
    variance_exponent: i32,
) -> Option<(f64, f64)> {
    if exact_variance <= &BigInt::from(0_u8) {
        return None;
    }
    let (difference_lower, difference_upper) =
        exact_dyadic_interval(exact_difference, difference_exponent)?;
    let (variance_lower, variance_upper) =
        exact_dyadic_interval(exact_variance, variance_exponent)?;
    if variance_lower <= 0.0 {
        return None;
    }
    let standard_error_lower = variance_lower.sqrt().next_down();
    let standard_error_upper = variance_upper.sqrt().next_up();
    if standard_error_lower <= 0.0
        || !standard_error_lower.is_finite()
        || !standard_error_upper.is_finite()
    {
        return None;
    }
    let quotient_candidates = [
        difference_lower / standard_error_lower,
        difference_lower / standard_error_upper,
        difference_upper / standard_error_lower,
        difference_upper / standard_error_upper,
    ];
    if quotient_candidates
        .iter()
        .any(|candidate| !candidate.is_finite())
    {
        return None;
    }
    let z_lower = quotient_candidates
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .next_down();
    Some((standard_error_upper, z_lower))
}

/// Certify positive definiteness exactly for the submitted binary64 matrix.
///
/// Every entry is converted to a common-scale dyadic integer. Fraction-free
/// Bareiss elimination then computes the leading principal-minor signs without
/// floating-point rounding. Sylvester's criterion admits the matrix exactly
/// when every leading principal minor is positive.
fn exact_positive_definite(covariance: &[f64], n: usize) -> Option<ExactDyadicVector> {
    let exact_covariance = exact_scaled_dyadics(covariance)?;
    let mut work = exact_covariance.integers.clone();
    let zero = BigInt::from(0_u8);
    let mut previous_pivot = BigInt::from(1_u8);
    for pivot_index in 0..n {
        let pivot = work[pivot_index * n + pivot_index].clone();
        if pivot <= zero {
            return None;
        }
        if pivot_index + 1 == n {
            return Some(exact_covariance);
        }
        for row in (pivot_index + 1)..n {
            for column in (pivot_index + 1)..n {
                let numerator = &work[row * n + column] * &pivot
                    - &work[row * n + pivot_index] * &work[pivot_index * n + column];
                let updated = if pivot_index == 0 {
                    numerator
                } else {
                    let quotient = &numerator / &previous_pivot;
                    if &quotient * &previous_pivot != numerator {
                        return None;
                    }
                    quotient
                };
                work[row * n + column] = updated;
            }
        }
        previous_pivot = pivot;
    }
    None
}

/// Conservative one-sided standard-normal survival probability.
///
/// `fitstats::erfc` documents absolute error below `1.2e-7`. Adding half that
/// bound and rounding the sum upward converts its reporting approximation into
/// an upper probability bound, so numerical approximation cannot create a Holm
/// rejection.
fn normal_survival_upper_bound(z: f64) -> Result<f64, String> {
    if !z.is_finite() {
        return Err("pairwise contrast arithmetic must remain finite".into());
    }
    let approximation = 0.5 * crate::fitstats::erfc(z / std::f64::consts::SQRT_2);
    if !approximation.is_finite() {
        return Err("pairwise contrast arithmetic must remain finite".into());
    }
    let upper_bound = approximation + 0.5 * ERFC_ABSOLUTE_ERROR_BOUND;
    if upper_bound >= 1.0 {
        Ok(1.0)
    } else {
        Ok(upper_bound.next_up())
    }
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
    let Some(exact_covariance) = exact_positive_definite(covariance, n) else {
        return Err("covariance must be positive definite".into());
    };
    let exact_estimates = exact_scaled_dyadics(estimates)
        .ok_or_else(|| "estimate exact arithmetic allocation failed".to_string())?;

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
            let exact_difference = &exact_estimates.integers[candidate_index]
                - &exact_estimates.integers[comparator_index];
            let exact_variance = &exact_covariance.integers[candidate_index * n + candidate_index]
                + &exact_covariance.integers[comparator_index * n + comparator_index]
                - (&exact_covariance.integers[candidate_index * n + comparator_index] << 1);
            let Some((standard_error_upper_bound, z_lower)) = conservative_wald_inputs(
                &exact_difference,
                exact_estimates.exponent,
                &exact_variance,
                exact_covariance.exponent,
            ) else {
                return Err("each pairwise contrast variance must be finite and positive".into());
            };
            let p_value_upper_bound = normal_survival_upper_bound(z_lower)?;
            comparisons.push(HolmWaldComparison {
                candidate_index,
                comparator_index,
                estimate_difference,
                standard_error_upper_bound,
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
        let reject = holm_bound_allows_rejection(
            comparisons[comparison_index].p_value_upper_bound,
            familywise_error_rate,
            remaining,
        )
        .ok_or_else(|| "Holm exact comparison failed".to_string())?;
        if reject {
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
        [winner] => Some(*winner),
        _ => None,
    };
    Ok(HolmWaldSuperiority {
        winner_index,
        comparisons,
    })
}

#[cfg(test)]
mod tests {
    use super::{holm_bound_allows_rejection, holm_wald_superiority};

    #[test]
    fn holm_boundary_uses_exact_cross_multiplication() {
        let rounded_up_division = 0.07_f64 / 5.0;
        assert_eq!(rounded_up_division, 0.014_000_000_000_000_002);
        assert_eq!(
            holm_bound_allows_rejection(rounded_up_division, 0.07, 5),
            Some(false)
        );
    }

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

    #[test]
    fn exact_indefinite_covariance_fails_under_every_permutation() {
        let covariance = [
            0.877_979_360_406_599_1,
            -0.326_872_939_819_592_2,
            0.016_902_198_682_877_17,
            -0.326_872_939_819_592_2,
            0.124_361_918_259_595_37,
            0.045_278_170_900_405_58,
            0.016_902_198_682_877_17,
            0.045_278_170_900_405_58,
            0.997_658_721_333_805_7,
        ];
        let estimates = [2.0, 1.0, 0.0];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for permutation in permutations {
            let permuted_estimates = permutation.map(|index| estimates[index]);
            let mut permuted_covariance = [0.0; 9];
            for row in 0..3 {
                for column in 0..3 {
                    permuted_covariance[row * 3 + column] =
                        covariance[permutation[row] * 3 + permutation[column]];
                }
            }
            let error = holm_wald_superiority(&permuted_estimates, &permuted_covariance, 0.05)
                .expect_err("exact non-PD covariance must fail for every ordering");
            assert_eq!(error, "covariance must be positive definite");
        }
    }

    #[test]
    fn multiple_directional_rejections_are_indeterminate() {
        let result = holm_wald_superiority(&[0.15, 0.0], &[0.5, 0.0, 0.0, 0.5], 0.9)
            .expect("valid high-alpha evidence remains a typed result");
        assert_eq!(result.winner_index, None);
        assert!(result
            .comparisons
            .iter()
            .all(|comparison| comparison.null_rejected));
    }

    #[test]
    fn exact_spd_covariance_is_admitted_under_every_permutation() {
        let covariance = [
            0.899_985_986_740_665_4,
            0.101_832_708_512_720_37,
            -0.224_161_503_975_947_33,
            0.101_832_708_512_720_37,
            0.033_586_996_097_637_64,
            -0.108_147_625_221_174_1,
            -0.224_161_503_975_947_33,
            -0.108_147_625_221_174_1,
            0.366_427_017_161_697_2,
        ];
        let permutations = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for permutation in permutations {
            let mut permuted_covariance = [0.0; 9];
            for row in 0..3 {
                for column in 0..3 {
                    permuted_covariance[row * 3 + column] =
                        covariance[permutation[row] * 3 + permutation[column]];
                }
            }
            let result = holm_wald_superiority(&[0.0; 3], &permuted_covariance, 0.05)
                .expect("exact SPD covariance must be admitted for every ordering");
            assert_eq!(result.winner_index, None);
        }
    }

    #[test]
    fn exact_contrast_variance_prevents_cancellation_rejection() {
        let covariance = [
            1.183_052_186_166_774_7e-271,
            1.183_052_186_166_774_6e-271,
            1.183_052_186_166_774_6e-271,
            1.183_052_186_166_775e-271,
        ];
        let result = holm_wald_superiority(&[1.1e-143, 0.0], &covariance, 0.05)
            .expect("exact SPD covariance");
        assert_eq!(result.winner_index, None);
        let forward = result
            .comparisons
            .iter()
            .find(|comparison| comparison.candidate_index == 0)
            .expect("forward ordered contrast");
        assert!(!forward.null_rejected);
        assert!(forward.p_value_upper_bound > 0.025);
    }
}
