use mlsirm_core::covariance_standardization::{
    COVARIANCE_STANDARDIZATION_CONTRACT_VERSION, CovarianceStandardizationError,
    standardize_covariance_matrix, standardize_variance,
};

fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
    );
}

#[test]
fn scalar_contract_is_versioned_and_scale_invariant() {
    assert_eq!(
        COVARIANCE_STANDARDIZATION_CONTRACT_VERSION,
        "fast_mlsirm.covariance_standardization@1.0.0"
    );
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
            standardize_variance(variance).expect("strictly positive finite variance"),
            1.0,
            8.0e-15,
        );
    }
}

#[test]
fn scalar_contract_fails_closed_for_unstandardizable_variance() {
    assert_eq!(
        standardize_variance(0.0),
        Err(CovarianceStandardizationError::NonPositiveVariance)
    );
    assert_eq!(
        standardize_variance(-1.0),
        Err(CovarianceStandardizationError::NonPositiveVariance)
    );
    for variance in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            standardize_variance(variance),
            Err(CovarianceStandardizationError::NonFiniteInput)
        );
    }
}

#[test]
fn matrix_contract_recovers_known_correlation_and_scale_invariance() {
    let base = [4.0, -3.0, -3.0, 9.0];
    let scaled = [4.0e100, -3.0e100, -3.0e100, 9.0e100];
    let base_result = standardize_covariance_matrix(&base, 2).expect("base covariance");
    let scaled_result = standardize_covariance_matrix(&scaled, 2).expect("scaled covariance");
    let expected = [1.0, -0.5, -0.5, 1.0];

    for ((base_value, scaled_value), expected_value) in base_result
        .iter()
        .zip(scaled_result.iter())
        .zip(expected.iter())
    {
        assert_close(*base_value, *expected_value, 8.0e-15);
        assert_close(*scaled_value, *expected_value, 8.0e-15);
    }
}

#[test]
fn matrix_contract_fails_closed_for_invalid_covariance() {
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
