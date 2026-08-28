//! Direct-Rust resource-admission parity for crossed multilevel estimation.
//!
//! The public Python adapter bounds crossed response evidence to 20,000,000
//! logical cells before dense marshalling. These tests require the Rust owner to
//! reject the same declared work envelope before response-slice traversal.

use mlsirm_core::multilevel::{estimate_crossed_person_effects, CrossedPersonEffectConfig};
use mlsirm_core::Device;

#[test]
fn rejects_oversized_response_work_before_slice_length_validation() {
    let error = estimate_crossed_person_effects(
        &[],
        &[],
        &[],
        &[],
        &[1.0],
        &[0.0],
        &[],
        &[0, 2],
        20_000_001,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("declared crossed response work above the package ceiling must fail closed");

    assert_eq!(
        error,
        "crossed response matrix exceeds the logical-cell cap of 20000000"
    );
}
