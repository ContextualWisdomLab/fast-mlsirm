//! Regression coverage for direct-core multiple-membership validation.

use mlsirm_core::multilevel::weighted_contextual_effect;

#[test]
fn rejects_duplicate_context_indices_within_one_row() {
    let error =
        weighted_contextual_effect(&[0, 2], &[0, 0], &[0.1, 0.2], &[10.0], 1)
            .expect_err("duplicate context indices in one row must fail closed");

    assert_eq!(error, "context_indices must be unique within each row");
}
