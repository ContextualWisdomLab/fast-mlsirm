//! Formula-level numerical-contract checks for Oblimax rotation.
//!
//! These tests intentionally mirror the current production integer-power route
//! while #1747 owns removal of Rust `f64::powi` from the deterministic CPU-f64
//! reference. Passing here is mathematical/formula evidence, not a claim of
//! cross-platform bitwise reproducibility.

use mlsirm_core::rotation::RotationCriterion;

fn scaled_tolerance(expected: f64) -> f64 {
    32.0 * f64::EPSILON * (1.0 + expected.abs())
}

#[test]
fn oblimax_matches_current_second_and_fourth_moment_formula() {
    let loadings = [0.8, -0.2, 0.1, 0.7, 0.5, -0.4, 0.3, 0.6];
    let evaluation = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("nonzero finite loadings satisfy the oblimax contract");

    let sum2: f64 = loadings.iter().map(|x| x * x).sum();
    let sum4: f64 = loadings.iter().map(|x| x.powi(4)).sum();
    let expected_value = -(sum4.ln() - 2.0 * sum2.ln());

    assert!(
        (evaluation.value - expected_value).abs() <= scaled_tolerance(expected_value),
        "oblimax value drifted from the current second/fourth-moment formula: actual={} expected={}",
        evaluation.value,
        expected_value
    );

    for (index, (&x, &actual)) in loadings
        .iter()
        .zip(evaluation.gradient.iter())
        .enumerate()
    {
        let expected = -(4.0 * x.powi(3) / sum4 - 4.0 * x / sum2);
        assert!(
            (actual - expected).abs() <= scaled_tolerance(expected),
            "oblimax gradient drifted at index {index}: actual={actual} expected={expected}"
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
        "oblimax value must remain invariant to a common nonzero scale"
    );
}
