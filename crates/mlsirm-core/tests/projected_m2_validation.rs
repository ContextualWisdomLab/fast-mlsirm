//! Validation contracts for the public projected-M2 Rust boundary.

use mlsirm_core::fitstats::projected_m2;

#[test]
fn projected_m2_rejects_short_residual_without_panicking() {
    let result = projected_m2(
        &[0.25],
        &[1.0, 0.5],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        1,
        100.0,
    );

    assert!(result.is_err(), "short residual must return Err");
}

#[test]
fn projected_m2_rejects_short_delta_without_panicking() {
    let result = projected_m2(
        &[0.25, -0.25],
        &[1.0],
        vec![1.0, 0.0, 0.0, 1.0],
        2,
        1,
        100.0,
    );

    assert!(result.is_err(), "short delta must return Err");
}

#[test]
fn projected_m2_rejects_short_covariance_without_panicking() {
    let result = projected_m2(
        &[0.25, -0.25],
        &[1.0, 0.5],
        vec![1.0, 0.0, 0.0],
        2,
        1,
        100.0,
    );

    assert!(result.is_err(), "short covariance must return Err");
}

#[test]
fn projected_m2_rejects_dimension_overflow_without_allocating() {
    let result = projected_m2(&[], &[], Vec::new(), usize::MAX, 2, 1.0);

    assert!(
        matches!(result, Err(message) if message.contains("overflow")),
        "overflowing dimensions must fail before indexing or allocation"
    );
}

#[test]
fn projected_m2_rejects_nonfinite_sample_size_before_arithmetic() {
    let result = projected_m2(&[0.25], &[1.0], vec![1.0], 1, 1, f64::INFINITY);

    assert!(
        matches!(result, Err(message) if message.contains("n must be finite")),
        "non-finite n must fail before numerical execution"
    );
}
