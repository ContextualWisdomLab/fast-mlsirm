use mlsirm_core::regression::{
    conditional_slope, conditional_slope_weights, design_row_dot, fit_ols, linear_contrast,
    slope_difference,
};

#[test]
fn ols_rejects_invalid_shapes_nonfinite_data_and_rank_deficiency() {
    for (x, y, n, k, message) in [
        (vec![], vec![], 0, 1, "n and k must be positive"),
        (vec![], vec![1.0], 1, 0, "n and k must be positive"),
        (vec![1.0; 3], vec![1.0; 2], 3, 1, "y length"),
        (vec![1.0; 2], vec![1.0; 3], 3, 1, "x length"),
        (vec![f64::NAN; 3], vec![1.0; 3], 3, 1, "must be finite"),
        (vec![1.0; 3], vec![f64::INFINITY; 3], 3, 1, "must be finite"),
        (vec![1.0], vec![1.0], 1, 1, "need n > k"),
        (vec![1.0; 6], vec![1.0; 3], 3, 2, "rank deficient"),
    ] {
        assert!(fit_ols(&x, &y, n, k).unwrap_err().contains(message));
    }
}

#[test]
fn contrast_rejects_invalid_covariance_weights_and_degrees_of_freedom() {
    assert!(linear_contrast(&[1.0], &[1.0], &[], 3.0)
        .unwrap_err()
        .contains("contrast length"));
    assert!(linear_contrast(&[1.0], &[], &[1.0], 3.0)
        .unwrap_err()
        .contains("vcov length"));
    for df in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(linear_contrast(&[1.0], &[1.0], &[1.0], df)
            .unwrap_err()
            .contains("residual df"));
    }
    for variance in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(linear_contrast(&[1.0], &[variance], &[1.0], 3.0)
            .unwrap_err()
            .contains("contrast variance"));
    }
    assert!(linear_contrast(&[f64::NAN], &[1.0], &[1.0], 3.0)
        .unwrap_err()
        .contains("non-finite contrast statistics"));
    assert!(linear_contrast(&[1.0], &[1.0], &[f64::NAN], 3.0)
        .unwrap_err()
        .contains("contrast variance"));
}

#[test]
fn slope_and_difference_use_joint_covariance_with_known_weights() {
    let beta: Vec<f64> = (0..10).map(|i| i as f64).collect();
    let mut covariance = vec![0.0; 100];
    for i in 0..10 {
        covariance[i * 10 + i] = 1.0;
    }
    assert_eq!(
        conditional_slope_weights(" x ", 0.0, 2.0, 3.0, 4.0).unwrap(),
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 4.0, 6.0]
    );
    let slope = conditional_slope(&beta, &covariance, "X", 0.0, 2.0, 3.0, 4.0, 8.0).unwrap();
    assert_eq!(slope.estimate, 115.0);
    assert!((slope.se.powi(2) - 66.0).abs() < 1e-12);
    let difference = slope_difference(
        &beta,
        &covariance,
        "X",
        (0.0, 2.0, 3.0, 4.0),
        (0.0, 0.0, 0.0, 0.0),
        8.0,
    )
    .unwrap();
    assert_eq!(difference.estimate, 114.0);
    assert!((difference.se.powi(2) - 65.0).abs() < 1e-12);
    let z_slope = conditional_slope(&beta, &covariance, "Z", 2.0, 3.0, 0.0, 0.0, 8.0).unwrap();
    assert_eq!(z_slope.estimate, 90.0);
    assert!((z_slope.se.powi(2) - 50.0).abs() < 1e-12);
}

#[test]
fn slope_callers_propagate_invalid_probe_and_contrast_errors() {
    assert!(conditional_slope(&[], &[], "X", 0.0, 0.0, 0.0, 0.0, 3.0)
        .unwrap_err()
        .contains("beta length"));
    assert!(slope_difference(
        &[],
        &[],
        "X",
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0),
        3.0
    )
    .unwrap_err()
    .contains("beta length"));
    assert!(conditional_slope_weights("Y", 0.0, 0.0, 0.0, 0.0)
        .unwrap_err()
        .contains("unknown focal"));
    assert!(conditional_slope_weights("X", 0.0, f64::INFINITY, 1.0, 0.0)
        .unwrap_err()
        .contains("non-finite conditional slope weights"));
    assert!(
        conditional_slope(&[0.0; 10], &[], "X", 0.0, 0.0, 0.0, 0.0, 3.0)
            .unwrap_err()
            .contains("vcov length")
    );
}

#[test]
fn design_dot_rejects_shape_nonfinite_products_and_overflowing_sum() {
    assert_eq!(design_row_dot(&[1.0, 2.0], &[3.0, 4.0]).unwrap(), 11.0);
    assert!(design_row_dot(&[1.0], &[]).unwrap_err().contains("length"));
    assert!(design_row_dot(&[f64::INFINITY], &[1.0])
        .unwrap_err()
        .contains("contribution"));
    assert_eq!(
        design_row_dot(&[f64::MAX, f64::MAX], &[1.0, 1.0]).unwrap_err(),
        "non-finite design_row_dot"
    );
}
