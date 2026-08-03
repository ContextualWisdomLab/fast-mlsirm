//! Numerical-overflow regression contract for bifactor scoreability indices.

use mlsirm_core::bifactor_indices::{bifactor_indices, BifactorIndicesConfig};

#[test]
fn accumulated_uniqueness_overflow_is_rejected() {
    let error = bifactor_indices(
        &[0.7, 0.2, 0.8, 0.3],
        &[f64::MAX, f64::MAX],
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .expect_err("overflowed omega accumulations must not return NaN or false precision");

    assert!(error.contains("finite"), "unexpected error: {error}");
}
