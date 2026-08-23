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
fn native_predictions_reject_category_count_above_fitter_limit_before_parameter_validation() {
    let error = polytomous_predictions(&[0.0], &[], &[], 1, 65, PolyModel::Gpcm)
        .expect_err("unsupported native category count must fail closed");

    assert_eq!(error, "n_cat must be in 2..=64");
}

#[test]
fn native_prediction_category_guard_preserves_fitter_upper_boundary() {
    let cat_params = vec![0.0; 63];
    let prediction = polytomous_predictions(
        &[0.0],
        &[1.0],
        &cat_params,
        1,
        64,
        PolyModel::Gpcm,
    )
    .expect("fitter-supported 64-category prediction must remain accepted");

    assert_eq!(prediction.probabilities.len(), 64);
    assert_eq!(prediction.expected.len(), 1);
    assert!((prediction.probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
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
