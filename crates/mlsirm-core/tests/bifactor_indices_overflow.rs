//! Numerical-overflow regression contract for bifactor scoreability indices.

use mlsirm_core::bifactor_indices::{bifactor_indices, BifactorIndicesConfig};

#[test]
fn out_of_range_uniqueness_is_rejected_before_accumulation() {
    let error = bifactor_indices(
        &[0.7, 0.2, 0.8, 0.3],
        &[f64::MAX, f64::MAX],
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .expect_err("invalid standardized uniquenesses must not reach omega accumulation");

    assert!(
        error.contains("between zero and one"),
        "unexpected error: {error}"
    );
}
