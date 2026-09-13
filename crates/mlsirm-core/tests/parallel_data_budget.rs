//! Native resource-budget parity for Horn parallel analysis.

use mlsirm_core::parallel::parallel_analysis;

#[test]
fn oversized_observed_shape_is_rejected_before_data_length_validation() {
    let error = parallel_analysis(&[], 10_000_001, 2, 1, 0, 1)
        .expect_err("observed data above the native cell ceiling must fail closed");

    assert!(
        error.contains("observed matrix") && error.contains("20000000"),
        "unexpected error: {error}"
    );
}
