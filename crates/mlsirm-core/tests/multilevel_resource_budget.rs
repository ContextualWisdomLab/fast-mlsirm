//! Direct-Rust resource-admission parity for crossed multilevel estimation.
//!
//! The public Python adapter bounds crossed response evidence to 20,000,000
//! logical cells and canonical contextual membership evidence to 100,000 edges
//! before dense/native work. These tests require the public Rust boundaries to
//! reject the same envelopes before response or membership traversal.

use mlsirm_core::multilevel::{
    estimate_crossed_person_effects, weighted_contextual_effect, CrossedPersonEffectConfig,
};
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

#[test]
fn rejects_oversized_membership_work_before_edge_traversal() {
    let context_indices: Vec<usize> = (0..100_001).collect();
    let weights = vec![0.0; context_indices.len()];
    let error = weighted_contextual_effect(
        &[0, context_indices.len()],
        &context_indices,
        &weights,
        &[],
        1,
    )
    .expect_err("membership evidence above the package ceiling must fail closed");

    assert_eq!(
        error,
        "context_indices exceeds the membership-edge cap of 100000"
    );
}
