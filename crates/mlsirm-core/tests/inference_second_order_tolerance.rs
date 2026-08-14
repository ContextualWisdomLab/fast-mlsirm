//! Fail-first contracts for positive-definiteness tolerance semantics.

use mlsirm_core::inference::second_order_test;

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
