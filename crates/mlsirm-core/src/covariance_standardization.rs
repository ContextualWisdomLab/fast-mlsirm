//! Domain-neutral covariance-to-correlation standardization.
//!
//! This module owns only reusable static numerical normalization. Product- or
//! study-specific temporal admission rules (for example, requiring event time)
//! belong to the consuming bounded context and must wrap this contract rather
//! than being encoded here.
//!
//! For a covariance matrix `Σ` with strictly positive diagonal `D`, the
//! standardized matrix is `R = D^{-1/2} Σ D^{-1/2}`. The scalar specialization
//! is therefore `(1 / sqrt(v)) * v * (1 / sqrt(v)) = 1` for finite `v > 0`.
//! Matrix entries are divided first by the smaller marginal standard deviation
//! and then by the larger one. This avoids forming `sqrt(v_i) * sqrt(v_j)`,
//! whose product can overflow, while also avoiding avoidable intermediate
//! underflow when the two marginal scales differ by many orders of magnitude.
//!
//! The TEPP migration that motivated this owner contract concerns ctsem's
//! `TIPREDVARstd`, but ctsem names, clocks, state equations, and event semantics
//! deliberately do not appear in this reusable kernel.
//!
//! # Research provenance
//!
//! Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time
//! structural equation modeling with R package ctsem. *Journal of Statistical
//! Software, 77*(5), 1–35. https://doi.org/10.18637/jss.v077.i05
//!
//! The ctsem source and paper provide the motivating covariance-standardization
//! use case; the matrix identity implemented here is the ordinary definition
//! converting a covariance matrix to its correlation matrix.

use std::cmp::Ordering;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Versioned public contract for reusable covariance standardization.
pub const COVARIANCE_STANDARDIZATION_CONTRACT_VERSION: &str =
    "fast_mlsirm.covariance_standardization@1.0.0";

/// Fail-closed input and arithmetic errors for covariance standardization.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CovarianceStandardizationError {
    /// Matrix dimension is zero, overflows `usize`, or does not match the slice.
    InvalidShape,
    /// At least one matrix entry or scalar variance is NaN or infinite.
    NonFiniteInput,
    /// A variance on the diagonal is zero or negative and cannot be standardized.
    NonPositiveVariance,
    /// Mirrored covariance cells are not exactly equal in binary64.
    NonSymmetricCovariance,
    /// A represented covariance pair violates `c² <= v_i * v_j` exactly.
    InvalidPairwiseCovariance,
    /// A finite input produced a non-finite standardized result.
    NonFiniteResult,
}

impl Display for CovarianceStandardizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::InvalidShape => "covariance matrix shape is invalid",
            Self::NonFiniteInput => "covariance input must be finite",
            Self::NonPositiveVariance => "covariance diagonal must be strictly positive",
            Self::NonSymmetricCovariance => "covariance matrix must be symmetric",
            Self::InvalidPairwiseCovariance => {
                "covariance pair violates the correlation magnitude bound"
            }
            Self::NonFiniteResult => "covariance standardization produced a non-finite result",
        };
        formatter.write_str(message)
    }
}

impl Error for CovarianceStandardizationError {}

/// Return an exact integer-significand representation of one finite binary64 value.
///
/// The returned pair `(significand, exponent)` satisfies
/// `abs(value) = significand * 2^exponent` without floating-point rounding.
fn binary64_components(value: f64) -> (u64, i32) {
    let bits = value.abs().to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        (fraction, -1074)
    } else {
        ((1_u64 << 52) | fraction, exponent_bits - 1023 - 52)
    }
}

/// Compare two non-negative exact integers multiplied by powers of two.
fn scaled_integer_le(
    left_significand: u128,
    left_exponent: i32,
    right_significand: u128,
    right_exponent: i32,
) -> bool {
    if left_significand == 0 {
        return true;
    }
    if right_significand == 0 {
        return false;
    }

    let left_bits = 128_i32 - left_significand.leading_zeros() as i32;
    let right_bits = 128_i32 - right_significand.leading_zeros() as i32;
    let left_top_bit = left_exponent + left_bits - 1;
    let right_top_bit = right_exponent + right_bits - 1;

    match left_top_bit.cmp(&right_top_bit) {
        Ordering::Less => true,
        Ordering::Greater => false,
        Ordering::Equal => {
            if left_exponent >= right_exponent {
                let shift = (left_exponent - right_exponent) as u32;
                left_significand
                    .checked_shl(shift)
                    .is_some_and(|aligned| aligned <= right_significand)
            } else {
                let shift = (right_exponent - left_exponent) as u32;
                right_significand
                    .checked_shl(shift)
                    .is_some_and(|aligned| left_significand <= aligned)
            }
        }
    }
}

/// Check the covariance Cauchy-Schwarz bound exactly for represented binary64 inputs.
fn pairwise_covariance_is_admissible(
    covariance: f64,
    variance_one: f64,
    variance_two: f64,
) -> bool {
    let (covariance_significand, covariance_exponent) = binary64_components(covariance);
    let (variance_one_significand, variance_one_exponent) = binary64_components(variance_one);
    let (variance_two_significand, variance_two_exponent) = binary64_components(variance_two);

    let covariance_square =
        u128::from(covariance_significand) * u128::from(covariance_significand);
    let variance_product =
        u128::from(variance_one_significand) * u128::from(variance_two_significand);

    scaled_integer_le(
        covariance_square,
        covariance_exponent * 2,
        variance_product,
        variance_one_exponent + variance_two_exponent,
    )
}

/// Standardize one finite, strictly positive variance against itself.
///
/// The arithmetic is evaluated rather than replaced with a hard-coded `1.0` so
/// this scalar reference exercises the same normalization contract consumed by
/// matrix standardization and downstream parity tests.
///
/// # Errors
///
/// Returns [`CovarianceStandardizationError::NonFiniteInput`] for NaN or
/// infinity, [`CovarianceStandardizationError::NonPositiveVariance`] for zero
/// or a negative value, and [`CovarianceStandardizationError::NonFiniteResult`]
/// if the arithmetic cannot produce a finite value.
pub fn standardize_variance(
    variance: f64,
) -> Result<f64, CovarianceStandardizationError> {
    if !variance.is_finite() {
        return Err(CovarianceStandardizationError::NonFiniteInput);
    }
    if variance <= 0.0 {
        return Err(CovarianceStandardizationError::NonPositiveVariance);
    }

    let inverse_sd = 1.0 / variance.sqrt();
    let standardized = (variance * inverse_sd) * inverse_sd;
    if !standardized.is_finite() {
        return Err(CovarianceStandardizationError::NonFiniteResult);
    }
    Ok(standardized)
}

/// Convert a finite symmetric covariance matrix to a correlation matrix.
///
/// `covariance` is row-major with shape `dimension × dimension`. Every
/// diagonal variance must be strictly positive. Mirrored off-diagonal cells
/// must be exactly equal in binary64. Pairwise admissibility is decided from
/// the exact represented binary64 integers using `c² <= v_i * v_j`, so an
/// invalid covariance is never accepted merely because floating-point division
/// rounded its correlation back into range.
///
/// After exact admission, the covariance is divided by the smaller marginal
/// standard deviation before the larger one. For an admissible pair the first
/// quotient is bounded in magnitude by the larger standard deviation, avoiding
/// overflow while minimizing avoidable intermediate underflow. The final
/// division can still round a mathematically valid boundary correlation just
/// outside `[-1, 1]`; only after the exact admission proof is that numerical
/// result projected back to the certified interval. This is not an empirical
/// epsilon or tolerance.
///
/// Callers that need approximate-symmetry preprocessing must define and validate
/// that policy explicitly before calling this kernel. This routine validates
/// pairwise covariance bounds but does not claim a full positive-semidefinite
/// proof; consumers requiring PSD admission retain that model-specific invariant.
///
/// # Errors
///
/// Returns a typed error for invalid shape, non-finite input, non-positive
/// diagonal variance, asymmetric mirrored cells, an impossible pairwise
/// covariance, or non-finite output arithmetic.
pub fn standardize_covariance_matrix(
    covariance: &[f64],
    dimension: usize,
) -> Result<Vec<f64>, CovarianceStandardizationError> {
    let expected_len = dimension
        .checked_mul(dimension)
        .ok_or(CovarianceStandardizationError::InvalidShape)?;
    if dimension == 0 || covariance.len() != expected_len {
        return Err(CovarianceStandardizationError::InvalidShape);
    }
    if covariance.iter().any(|value| !value.is_finite()) {
        return Err(CovarianceStandardizationError::NonFiniteInput);
    }

    let mut standard_deviations = Vec::with_capacity(dimension);
    for index in 0..dimension {
        let variance = covariance[index * dimension + index];
        if variance <= 0.0 {
            return Err(CovarianceStandardizationError::NonPositiveVariance);
        }
        standard_deviations.push(variance.sqrt());
    }

    let mut correlation = vec![0.0; expected_len];
    for index in 0..dimension {
        correlation[index * dimension + index] =
            standardize_variance(covariance[index * dimension + index])?;
    }

    for row in 0..dimension {
        for column in (row + 1)..dimension {
            let upper = covariance[row * dimension + column];
            let lower = covariance[column * dimension + row];
            if upper.to_bits() != lower.to_bits() {
                return Err(CovarianceStandardizationError::NonSymmetricCovariance);
            }

            let variance_row = covariance[row * dimension + row];
            let variance_column = covariance[column * dimension + column];
            if !pairwise_covariance_is_admissible(upper, variance_row, variance_column) {
                return Err(CovarianceStandardizationError::InvalidPairwiseCovariance);
            }

            let (first_sd, second_sd) = if standard_deviations[row] <= standard_deviations[column]
            {
                (standard_deviations[row], standard_deviations[column])
            } else {
                (standard_deviations[column], standard_deviations[row])
            };
            let standardized = (upper / first_sd) / second_sd;
            if !standardized.is_finite() {
                return Err(CovarianceStandardizationError::NonFiniteResult);
            }
            let bounded = standardized.clamp(-1.0, 1.0);
            correlation[row * dimension + column] = bounded;
            correlation[column * dimension + row] = bounded;
        }
    }

    Ok(correlation)
}

#[cfg(test)]
mod tests {
    use super::{
        scaled_integer_le, CovarianceStandardizationError, standardize_covariance_matrix,
        standardize_variance,
    };

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
        );
    }

    #[test]
    fn exact_scaled_integer_comparison_handles_zero_magnitude_and_alignment() {
        assert!(scaled_integer_le(0, -100, 1, -1000));
        assert!(!scaled_integer_le(1, 0, 0, 0));
        assert!(scaled_integer_le(1, 0, 1, 1));
        assert!(!scaled_integer_le(1, 1, 1, 0));
        assert!(scaled_integer_le(1, 1, 2, 0));
        assert!(scaled_integer_le(2, 0, 1, 1));
    }

    #[test]
    fn scalar_reference_recovers_one_across_positive_scales() {
        for variance in [
            f64::MIN_POSITIVE,
            1.0e-200,
            0.25,
            1.0,
            6.4,
            1.0e200,
            f64::MAX,
        ] {
            assert_close(
                standardize_variance(variance).expect("positive variance"),
                1.0,
                8.0e-15,
            );
        }
    }

    #[test]
    fn scalar_reference_fails_closed_for_invalid_variance() {
        for variance in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(standardize_variance(variance).is_err());
        }
        assert_eq!(
            standardize_variance(0.0),
            Err(CovarianceStandardizationError::NonPositiveVariance)
        );
    }

    #[test]
    fn matrix_standardization_recovers_expected_correlation() {
        let covariance = [4.0, 2.0, 2.0, 9.0];
        let correlation = standardize_covariance_matrix(&covariance, 2).expect("covariance");
        assert_close(correlation[0], 1.0, 8.0e-15);
        assert_close(correlation[1], 1.0 / 3.0, 8.0e-15);
        assert_close(correlation[2], 1.0 / 3.0, 8.0e-15);
        assert_close(correlation[3], 1.0, 8.0e-15);
    }

    #[test]
    fn matrix_standardization_is_scale_invariant() {
        let base = [4.0, -3.0, -3.0, 9.0];
        let scaled = [4.0e100, -3.0e100, -3.0e100, 9.0e100];
        let base_result = standardize_covariance_matrix(&base, 2).expect("base");
        let scaled_result = standardize_covariance_matrix(&scaled, 2).expect("scaled");
        for (left, right) in base_result.iter().zip(scaled_result.iter()) {
            assert_close(*left, *right, 8.0e-15);
        }
    }

    #[test]
    fn matrix_standardization_accepts_zero_covariance_with_subnormal_variance() {
        let smallest_subnormal = f64::from_bits(1);
        let covariance = [smallest_subnormal, 0.0, 0.0, 1.0];
        let correlation = standardize_covariance_matrix(&covariance, 2).expect("valid covariance");
        assert_eq!(correlation[1], 0.0);
        assert_eq!(correlation[2], 0.0);
    }

    #[test]
    fn matrix_standardization_fails_closed_for_shape_and_numeric_defects() {
        assert_eq!(
            standardize_covariance_matrix(&[], 0),
            Err(CovarianceStandardizationError::InvalidShape)
        );
        assert_eq!(
            standardize_covariance_matrix(&[1.0, 0.0, 0.0], 2),
            Err(CovarianceStandardizationError::InvalidShape)
        );
        assert_eq!(
            standardize_covariance_matrix(&[1.0, f64::NAN, f64::NAN, 1.0], 2),
            Err(CovarianceStandardizationError::NonFiniteInput)
        );
        assert_eq!(
            standardize_covariance_matrix(&[0.0], 1),
            Err(CovarianceStandardizationError::NonPositiveVariance)
        );
        assert_eq!(
            standardize_covariance_matrix(&[1.0, 0.2, 0.3, 1.0], 2),
            Err(CovarianceStandardizationError::NonSymmetricCovariance)
        );
        assert_eq!(
            standardize_covariance_matrix(&[1.0, 1.1, 1.1, 1.0], 2),
            Err(CovarianceStandardizationError::InvalidPairwiseCovariance)
        );
    }
}
