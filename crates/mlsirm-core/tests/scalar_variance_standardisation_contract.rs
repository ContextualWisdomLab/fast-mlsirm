use mlsirm_core::standardisation::{
    SCALAR_VARIANCE_STANDARDISATION_CONTRACT_V1, ScalarVarianceStandardisationError,
    standardise_positive_scalar_variance,
};

#[test]
fn positive_scalar_variance_standardises_to_exact_unit_correlation() {
    for variance in [f64::MIN_POSITIVE, 3.0, 6.4, 1.0e300] {
        let recovered = standardise_positive_scalar_variance(variance)
            .expect("strictly positive finite scalar variance must standardise");
        assert_eq!(recovered.to_bits(), 1.0_f64.to_bits());
    }
}

#[test]
fn invalid_scalar_variance_fails_closed() {
    assert_eq!(
        standardise_positive_scalar_variance(0.0),
        Err(ScalarVarianceStandardisationError::NonPositive)
    );
    assert_eq!(
        standardise_positive_scalar_variance(-1.0),
        Err(ScalarVarianceStandardisationError::NonPositive)
    );
    assert_eq!(
        standardise_positive_scalar_variance(f64::NAN),
        Err(ScalarVarianceStandardisationError::NonFinite)
    );
    assert_eq!(
        standardise_positive_scalar_variance(f64::INFINITY),
        Err(ScalarVarianceStandardisationError::NonFinite)
    );
    assert_eq!(
        standardise_positive_scalar_variance(f64::NEG_INFINITY),
        Err(ScalarVarianceStandardisationError::NonFinite)
    );
}

#[test]
fn public_error_messages_are_stable() {
    assert_eq!(
        ScalarVarianceStandardisationError::NonFinite.to_string(),
        "scalar variance must be finite"
    );
    assert_eq!(
        ScalarVarianceStandardisationError::NonPositive.to_string(),
        "scalar variance must be strictly positive"
    );
}

#[test]
fn published_contract_identity_is_versioned() {
    assert_eq!(
        SCALAR_VARIANCE_STANDARDISATION_CONTRACT_V1,
        "fast_mlsirm.scalar_variance_standardisation@1.0.0"
    );
}
