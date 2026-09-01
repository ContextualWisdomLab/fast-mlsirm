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
//! The matrix implementation divides sequentially by each marginal standard
//! deviation so a representable correlation is not lost because
//! `sqrt(v_i) * sqrt(v_j)` overflows first.
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
    /// Mirrored covariance cells disagree beyond floating-point tolerance.
    NonSymmetricCovariance,
    /// A pairwise covariance implies an absolute correlation materially above one.
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
/// diagonal variance must be strictly positive. Symmetry is checked with a
/// scale-aware binary64 tolerance; off-diagonal pairs that only differ by
/// rounding are averaged safely (`a / 2 + b / 2`) before standardization so
/// the returned matrix is exactly symmetric. Pairwise correlations whose
/// absolute value exceeds one by more than floating-point tolerance fail
/// closed.
///
/// This routine validates pairwise covariance bounds but does not claim a full
/// positive-semidefinite proof. A caller that requires PSD admission must apply
/// that model-specific invariant separately.
///
/// # Errors
///
/// Returns a typed error for invalid shape, non-finite input, non-positive
/// diagonal variance, material asymmetry, an impossible pairwise covariance,
/// or non-finite output arithmetic.
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
            let symmetry_scale = upper.abs().max(lower.abs()).max(1.0);
            let symmetry_tolerance = 64.0 * f64::EPSILON * symmetry_scale;
            if (upper - lower).abs() > symmetry_tolerance {
                return Err(CovarianceStandardizationError::NonSymmetricCovariance);
            }

            let symmetric_covariance = upper * 0.5 + lower * 0.5;
            let standardized =
                (symmetric_covariance / standard_deviations[row]) / standard_deviations[column];
            if !standardized.is_finite() {
                return Err(CovarianceStandardizationError::NonFiniteResult);
            }
            let bound_tolerance = 128.0 * f64::EPSILON;
            if standardized.abs() > 1.0 + bound_tolerance {
                return Err(CovarianceStandardizationError::InvalidPairwiseCovariance);
            }
            let bounded = standardized.clamp(-1.0, 1.0);
            correlation[row * dimension + column] = bounded;
            correlation[column * dimension + row] = bounded;
        }
    }

    Ok(correlation)
}
