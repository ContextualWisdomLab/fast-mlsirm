use mlsirm_core::poly::{polytomous_predictions, PolyModel};

#[test]
fn native_predictions_reject_oversized_probability_grid_before_parameter_validation() {
    // 312_501 * 64 = 20_000_064 probability cells. Deliberately keep the
    // parameter slices empty: the resource envelope must be decided from the
    // request shape before any parameter-dependent allocation or validation.
    let error = polytomous_predictions(&[0.0], &[], &[], 312_501, 64, PolyModel::Gpcm)
        .expect_err("oversized native prediction grid must fail closed");

    assert!(
        error.contains("20,000,000") && error.contains("prediction"),
        "unexpected native prediction resource error: {error}"
    );
}

#[test]
fn native_predictions_reject_category_count_above_fitted_model_ceiling_before_parameter_validation() {
    // Fitted polytomous models admit at most POLY_MAX_CAT=64 categories. Keep
    // parameter slices empty so this regression also proves prediction replays
    // the fitted-model category domain before parameter-dependent validation.
    let error = polytomous_predictions(&[0.0], &[], &[], 1, 65, PolyModel::Gpcm)
        .expect_err("prediction must reject category counts the fitter cannot produce");

    assert!(
        error.contains("n_cat") && error.contains("2..=64"),
        "unexpected native prediction category-domain error: {error}"
    );
}

#[test]
fn native_prediction_resource_guard_preserves_small_gpcm_predictions() {
    let prediction = polytomous_predictions(&[0.0], &[1.0], &[0.0], 1, 2, PolyModel::Gpcm)
        .expect("small valid prediction should remain accepted");

    assert_eq!(prediction.probabilities.len(), 2);
    assert_eq!(prediction.expected.len(), 1);
    assert!((prediction.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    assert!((prediction.expected[0] - 0.5).abs() < 1e-12);
}
