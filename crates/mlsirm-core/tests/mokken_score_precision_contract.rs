//! Direct-Rust admission contract for Mokken score precision.
//!
//! The numerical kernel converts admitted `i64` scores to `f64` before
//! covariance construction. Values that do not survive that conversion
//! exactly must fail at the public Rust boundary rather than silently changing
//! the observed response evidence.

use mlsirm_core::mokken::{aisp, coef_h};

const TWO_POW_53: i64 = 9_007_199_254_740_992;

fn exact_large_scores() -> Vec<i64> {
    vec![
        TWO_POW_53,
        0,
        TWO_POW_53 + 2,
        1,
        TWO_POW_53 + 4,
        2,
    ]
}

fn lossy_large_scores() -> Vec<i64> {
    vec![
        TWO_POW_53,
        0,
        TWO_POW_53 + 1,
        1,
        TWO_POW_53 + 2,
        2,
    ]
}

#[test]
fn direct_rust_rejects_scores_that_change_when_projected_to_binary64() {
    let x = lossy_large_scores();

    let coef_error = coef_h(&x, 3, 2).expect_err("lossy score must fail before Mokken arithmetic");
    assert_eq!(
        coef_error,
        "scores must be exactly representable as f64",
        "coef_h must not silently round direct-Rust i64 response evidence",
    );

    let aisp_error = aisp(&x, 3, 2, 0.3, 0.05)
        .expect_err("lossy score must fail before AISP arithmetic");
    assert_eq!(
        aisp_error,
        "scores must be exactly representable as f64",
        "aisp must share the canonical direct-Rust score admission boundary",
    );
}

#[test]
fn direct_rust_preserves_large_scores_that_are_exact_binary64_values() {
    let x = exact_large_scores();

    coef_h(&x, 3, 2).expect("exactly representable large scores remain admissible");
    aisp(&x, 3, 2, 0.3, 0.05)
        .expect("AISP keeps the same exact binary64 score domain");
}
