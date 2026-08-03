//! Regression contract for zero-variance casewise Vuong comparisons.

use mlsirm_core::fitstats::vuong_nonnested;

#[test]
fn constant_casewise_difference_returns_a_degenerate_result() {
    let a = [1.0, 1.5, 2.0, 2.5];
    let b = [0.8, 1.3, 1.8, 2.3];

    let result = vuong_nonnested(&a, &b, 2, 2, true)
        .expect("zero variance is a decision state rather than malformed input");

    assert!((result.mean_diff - 0.2).abs() < 1e-12);
    assert_eq!(result.omega, 0.0);
    assert!(result.z.is_nan());
    assert!(result.p_two_sided.is_nan());
}
