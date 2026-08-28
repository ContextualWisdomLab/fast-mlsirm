//! Direct-Rust resource-admission parity for crossed multilevel estimation.
//!
//! The public Python adapter bounds crossed response evidence to 20,000,000
//! logical cells, contextual membership evidence to 100,000 edges / 100,001
//! CSR row-pointer entries, and crossed-estimator worker control to 10,000.
//! These tests require the public Rust boundaries to reject the same envelopes
//! before response/membership traversal, allocation, or estimator iteration.

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

#[test]
fn rejects_oversized_row_pointer_work_before_empty_row_allocation() {
    let row_offsets = vec![0usize; 100_002];
    let error = weighted_contextual_effect(&row_offsets, &[], &[], &[], 1)
        .expect_err("row-pointer evidence above the package ceiling must fail closed");

    assert_eq!(
        error,
        "row_offsets exceeds the CSR row-pointer cap of 100001"
    );
}

#[test]
fn rejects_estimator_oversized_membership_work_before_response_value_traversal() {
    let context_indices = vec![0usize; 100_001];
    let weights = vec![0.0; context_indices.len()];
    let error = estimate_crossed_person_effects(
        &[2.0, 0.0],
        &[0, 1, 2],
        &context_indices,
        &weights,
        &[1.0],
        &[0.0],
        &[],
        &[0, 2],
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("estimator membership-edge work must fail before response-value traversal");

    assert_eq!(
        error,
        "context_indices exceeds the membership-edge cap of 100000"
    );
}

#[test]
fn rejects_estimator_oversized_row_pointer_work_before_response_value_traversal() {
    let row_offsets = vec![0usize; 100_002];
    let error = estimate_crossed_person_effects(
        &[2.0, 0.0],
        &row_offsets,
        &[0, 1],
        &[1.0, 1.0],
        &[1.0],
        &[0.0],
        &[],
        &[0, 2],
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("estimator row-pointer work must fail before response-value traversal");

    assert_eq!(
        error,
        "row_offsets exceeds the CSR row-pointer cap of 100001"
    );
}

#[test]
fn rejects_crossed_worker_count_above_public_control_bound() {
    let error = estimate_crossed_person_effects(
        &[1.0, 0.0],
        &[0, 1, 2],
        &[0, 1],
        &[1.0, 1.0],
        &[1.0],
        &[0.0],
        &[],
        &[0, 2],
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            worker_count: 10_001,
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("crossed worker control above the public maximum must fail closed");

    assert_eq!(error, "worker_count must be in 1..=10000");
}

#[test]
fn rejects_invalid_crossed_control_before_response_value_traversal() {
    let error = estimate_crossed_person_effects(
        &[2.0, 0.0],
        &[0, 1, 2],
        &[0, 1],
        &[1.0, 1.0],
        &[1.0],
        &[0.0],
        &[],
        &[0, 2],
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            worker_count: 10_001,
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("inert crossed controls must fail before response-value traversal");

    assert_eq!(error, "worker_count must be in 1..=10000");
}

#[test]
fn rejects_oversized_classification_offset_work_before_shape_validation() {
    let mut classification_offsets: Vec<usize> = (0..=128).collect();
    classification_offsets.push(128);
    let error = estimate_crossed_person_effects(
        &[1.0],
        &[0, 1],
        &[0],
        &[1.0],
        &[1.0],
        &[0.0],
        &[],
        &classification_offsets,
        1,
        1,
        128,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("classification-offset work above the effect envelope must fail closed");

    assert_eq!(error, "classification_offsets exceeds n_effects + 1");
}
