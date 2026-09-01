use mlsirm_core::covariance_standardization::{
    standardize_covariance_matrix, CovarianceStandardizationError,
};

#[test]
fn covariance_matrix_requires_exact_symmetry() {
    let epsilon_offset = 8.0 * f64::EPSILON;
    let covariance = [1.0, 0.25, 0.25 + epsilon_offset, 1.0];

    assert_eq!(
        standardize_covariance_matrix(&covariance, 2),
        Err(CovarianceStandardizationError::NonSymmetricCovariance),
    );
}

#[test]
fn covariance_matrix_never_clamps_an_out_of_range_correlation() {
    let above_one = 1.0 + 64.0 * f64::EPSILON;
    let covariance = [1.0, above_one, above_one, 1.0];

    assert_eq!(
        standardize_covariance_matrix(&covariance, 2),
        Err(CovarianceStandardizationError::InvalidPairwiseCovariance),
    );
}
