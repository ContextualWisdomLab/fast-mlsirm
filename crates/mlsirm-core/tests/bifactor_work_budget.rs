//! Work-budget contract for non-iterative bifactor scoreability diagnostics.

use mlsirm_core::bifactor_indices::{bifactor_indices, BifactorIndicesConfig};

#[test]
fn oversized_item_factor_work_is_rejected_before_input_allocation_checks() {
    let error = bifactor_indices(
        &[],
        &[],
        BifactorIndicesConfig::new(12_208, 64, 0),
    )
    .expect_err("work above the deterministic CPU budget must fail closed");

    assert!(
        error.contains("work budget") && error.contains("50000000"),
        "unexpected error: {error}"
    );
}
