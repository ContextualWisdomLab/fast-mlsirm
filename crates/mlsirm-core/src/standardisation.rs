//! Domain-neutral scalar covariance standardisation.
//!
//! This module owns reusable static psychometric arithmetic only. Product-specific
//! clock semantics, temporal state evolution, provenance and admission policy stay
//! in downstream products such as TEPP.

use std::fmt;

/// Versioned Published Language identifier for scalar variance standardisation.
pub const SCALAR_VARIANCE_STANDARDISATION_CONTRACT_V1: &str =
    "fast_mlsirm.scalar_variance_standardisation@1.0.0";

/// Fail-closed input errors for scalar variance standardisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarVarianceStandardisationError {
    /// The supplied variance is NaN or infinite.
    NonFinite,
    /// The supplied variance is zero or negative, so no positive standard deviation exists.
    NonPositive,
}

impl fmt::Display for ScalarVarianceStandardisationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite => formatter.write_str("scalar variance must be finite"),
            Self::NonPositive => formatter.write_str("scalar variance must be strictly positive"),
        }
    }
}

impl std::error::Error for ScalarVarianceStandardisationError {}

/// Standardise a strictly positive finite scalar variance to its unit correlation.
///
/// For a one-dimensional covariance `v > 0`, covariance standardisation is
/// `(1 / sqrt(v)) * v * (1 / sqrt(v)) = 1`. The mathematical result is exactly
/// one, so this reference implementation validates the input and returns the
/// exact binary64 representation of `1.0` rather than re-evaluating the
/// algebraic identity and exposing avoidable rounding such as
/// `1.0000000000000002` for `v = 3`.
///
/// This is intentionally domain-neutral static arithmetic. It does not attach an
/// event clock, infer temporal stationarity, define a ctsem parameter, or promote
/// any model to production status. A downstream adapter may use this primitive
/// only after it has established the named parameter and scientific contract.
///
/// Driver, C. C., Oud, J. H. L., & Voelkle, M. C. (2017). Continuous time
/// structural equation modeling with R package ctsem. *Journal of Statistical
/// Software, 77*(5), 1–35. <https://doi.org/10.18637/jss.v077.i05>
/// documents covariance standardisation in the ctsem reporting contract; that
/// model-specific naming remains outside this generic kernel.
///
/// # Errors
///
/// Returns [`ScalarVarianceStandardisationError::NonFinite`] for NaN or infinite
/// input and [`ScalarVarianceStandardisationError::NonPositive`] when the value
/// is zero or negative.
pub fn standardise_positive_scalar_variance(
    variance: f64,
) -> Result<f64, ScalarVarianceStandardisationError> {
    if !variance.is_finite() {
        return Err(ScalarVarianceStandardisationError::NonFinite);
    }
    if variance <= 0.0 {
        return Err(ScalarVarianceStandardisationError::NonPositive);
    }
    Ok(1.0)
}
