//! Regression contract for zero-variance casewise Vuong comparisons.

use mlsirm_core::fitstats::vuong_nonnested;

#[test]
fn constant_casewise_difference_returns_a_degenerate_result() {
    let a = [1.0, 1.5, 2.0, 2.5];
    let b = [0.75, 1.25, 1.75, 2.25];

    let result = vuong_nonnested(&a, &b, 2, 2, true)
        .expect("zero variance is a decision state rather than malformed input");

    assert_eq!(result.mean_diff, 0.25);
    assert_eq!(result.omega, 0.0);
    assert!(result.z.is_nan());
    assert!(result.p_two_sided.is_nan());
}
