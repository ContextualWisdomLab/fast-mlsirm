//! Deterministic numerical-contract checks for Oblimax rotation.
//!
//! Formula agreement, bitwise reference identity, and scale invariance are
//! separate acceptance claims. The bitwise fixture below is package-owned and
//! deliberately rejects standard-library transcendental/power routes whose
//! precision Rust leaves unspecified.

use mlsirm_core::rotation::RotationCriterion;

fn scaled_tolerance(expected: f64) -> f64 {
    32.0 * f64::EPSILON * (1.0 + expected.abs())
}

#[test]
fn oblimax_deterministic_reference_avoids_unspecified_std_math() {
    let source = include_str!("../src/rotation/criteria.rs");
    let start = source
        .find("fn oblimax(")
        .expect("Oblimax criterion implementation must exist");
    let end = start
        + source[start..]
            .find("\nfn bentler(")
            .expect("Bentler criterion must follow Oblimax");
    let body = &source[start..end];

    assert!(
        !body.contains(".powi("),
        "Oblimax deterministic reference must not use f64::powi"
    );
    assert!(
        !body.contains(".ln()"),
        "Oblimax deterministic reference must not use f64::ln"
    );
}

#[test]
fn oblimax_has_package_owned_binary64_golden_identity() {
    let loadings = [0.625, -0.75, 0.125, 0.875, -0.75, -0.25, -0.5, 0.25];
    let evaluation = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("finite nonzero loadings satisfy the Oblimax contract");

    assert_eq!(evaluation.value.to_bits(), 0x3ff9_9747_d6de_45d9);
    assert_eq!(
        evaluation
            .gradient
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0x3fd0_8b6c_cc65_67e2,
            0x3fa8_3ba6_e319_0c20,
            0x3fc7_4282_5e9a_9c97,
            0xbfe1_8acb_d433_cd66,
            0x3fa8_3ba6_e319_0c20,
            0xbfd5_2df7_8365_f49c,
            0xbfd9_b798_2d26_a962,
            0x3fd5_2df7_8365_f49c,
        ]
    );
}

#[test]
fn oblimax_repeated_evaluation_is_bitwise_stable() {
    let loadings = [0.625, -0.75, 0.125, 0.875, -0.75, -0.25, -0.5, 0.25];
    let baseline = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("baseline loadings are valid");
    let baseline_value = baseline.value.to_bits();
    let baseline_gradient: Vec<u64> = baseline
        .gradient
        .iter()
        .map(|value| value.to_bits())
        .collect();

    for _ in 0..32 {
        let repeated = RotationCriterion::Oblimax
            .evaluate(&loadings, 4, 2)
            .expect("repeated loadings remain valid");
        assert_eq!(repeated.value.to_bits(), baseline_value);
        assert_eq!(
            repeated
                .gradient
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            baseline_gradient
        );
    }
}

#[test]
fn oblimax_matches_second_and_fourth_moment_formula() {
    let loadings = [0.8, -0.2, 0.1, 0.7, 0.5, -0.4, 0.3, 0.6];
    let evaluation = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("nonzero finite loadings satisfy the Oblimax contract");

    let sum2: f64 = loadings.iter().map(|x| x * x).sum();
    let sum4: f64 = loadings
        .iter()
        .map(|x| {
            let square = x * x;
            square * square
        })
        .sum();
    let expected_value = -(sum4.ln() - 2.0 * sum2.ln());

    assert!(
        (evaluation.value - expected_value).abs() <= scaled_tolerance(expected_value),
        "Oblimax value drifted from the second/fourth-moment formula: actual={} expected={}",
        evaluation.value,
        expected_value
    );

    for (index, (&x, &actual)) in loadings
        .iter()
        .zip(evaluation.gradient.iter())
        .enumerate()
    {
        let cube = (x * x) * x;
        let expected = -(4.0 * cube / sum4 - 4.0 * x / sum2);
        assert!(
            (actual - expected).abs() <= scaled_tolerance(expected),
            "Oblimax gradient drifted at index {index}: actual={actual} expected={expected}"
        );
    }
}

#[test]
fn oblimax_remains_scale_invariant_with_finite_nonzero_loadings() {
    let loadings = [0.8, -0.2, 0.1, 0.7, 0.5, -0.4, 0.3, 0.6];
    let scaled: Vec<f64> = loadings.iter().map(|value| 7.25 * value).collect();

    let baseline = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("baseline loadings are valid");
    let rescaled = RotationCriterion::Oblimax
        .evaluate(&scaled, 4, 2)
        .expect("rescaled loadings are valid");

    assert!(
        (baseline.value - rescaled.value).abs() <= scaled_tolerance(baseline.value),
        "Oblimax value must remain invariant to a common nonzero scale"
    );
}
