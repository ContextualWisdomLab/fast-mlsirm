//! Direct-Rust classification-structure admission ordering.

use mlsirm_core::multilevel::{
    estimate_crossed_person_effects, CrossedPersonEffectConfig,
};
use mlsirm_core::Device;

fn fit_with_offsets(classification_offsets: &[usize]) -> String {
    estimate_crossed_person_effects(
        &[2.0, 0.0],
        &[0, 1, 2],
        &[0, 1],
        &[1.0, 1.0],
        &[1.0],
        &[0.0],
        &[],
        classification_offsets,
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("invalid classification structure must fail before response-value traversal")
}

#[test]
fn rejects_missing_classification_structure_before_response_value_traversal() {
    assert_eq!(
        fit_with_offsets(&[]),
        "classification_offsets must contain at least one classification"
    );
}

#[test]
fn rejects_malformed_classification_structure_before_response_value_traversal() {
    assert_eq!(
        fit_with_offsets(&[1, 2]),
        "classification_offsets must start at zero, increase strictly, and end at n_effects"
    );
}

#[test]
fn rejects_singleton_classification_before_response_value_traversal() {
    assert_eq!(
        fit_with_offsets(&[0, 1, 2]),
        "each classification must contain at least two context levels"
    );
}

#[test]
fn response_work_cap_precedes_classification_structure() {
    let error = estimate_crossed_person_effects(
        &[],
        &[],
        &[],
        &[],
        &[1.0],
        &[0.0],
        &[],
        &[1, 2],
        20_000_001,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("oversized response work must win before classification structure");

    assert_eq!(
        error,
        "crossed response matrix exceeds the logical-cell cap of 20000000"
    );
}

#[test]
fn response_length_precedes_classification_structure() {
    let error = estimate_crossed_person_effects(
        &[2.0],
        &[],
        &[],
        &[],
        &[1.0],
        &[0.0],
        &[],
        &[1, 2],
        2,
        1,
        2,
        CrossedPersonEffectConfig {
            device: Device::Cpu,
            ..CrossedPersonEffectConfig::default()
        },
    )
    .expect_err("mismatched response length must win before classification structure");

    assert_eq!(error, "y must have length n_persons * n_items");
}
