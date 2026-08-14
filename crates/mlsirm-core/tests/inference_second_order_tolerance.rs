//! Fail-first contracts for observed-information dimension and tolerance semantics.

use mlsirm_core::inference::{
    finite_difference_hessian, second_order_test, standard_errors_from_vcov,
    vcov_from_hessian,
};

#[test]
fn second_order_rejects_negative_tolerance() {
    let information = [1.0_f64, 0.0, 0.0, 1.0];

    let error = second_order_test(&information, 2, -1.0e-8)
        .expect_err("a negative tolerance must not redefine positive definiteness");

    assert_eq!(error, "tol must be a finite non-negative float");
}

#[test]
fn second_order_accepts_zero_tolerance_for_strict_positive_definiteness() {
    let information = [1.0_f64, 0.0, 0.0, 2.0];

    let (passed, min_eigenvalue, eigenvalues) =
        second_order_test(&information, 2, 0.0).expect("zero is a valid strict-PD tolerance");

    assert!(passed);
    assert_eq!(min_eigenvalue, 1.0);
    assert_eq!(eigenvalues, vec![1.0, 2.0]);
}

#[test]
fn second_order_rejects_dimension_product_overflow() {
    let error = second_order_test(&[], usize::MAX, 0.0)
        .expect_err("dimension arithmetic must fail closed instead of overflowing");

    assert_eq!(error, "hessian dimension exceeds supported size");
}

#[test]
fn covariance_rejects_dimension_product_overflow() {
    let error = vcov_from_hessian(&[], usize::MAX, 0.0)
        .expect_err("covariance dimension arithmetic must fail closed");

    assert_eq!(error, "hessian dimension exceeds supported size");
}

#[test]
fn standard_errors_reject_dimension_product_overflow() {
    let error = standard_errors_from_vcov(&[], usize::MAX)
        .expect_err("standard-error dimension arithmetic must fail closed");

    assert_eq!(error, "vcov dimension exceeds supported size");
}

#[test]
fn finite_difference_rejects_dimension_product_overflow() {
    let error = finite_difference_hessian(
        usize::MAX,
        1.0,
        0.0,
        &[],
        &[],
        &[],
        &[],
        &[],
        &[],
    )
    .expect_err("finite-difference dimension arithmetic must fail closed");

    assert_eq!(error, "hessian dimension exceeds supported size");
}
