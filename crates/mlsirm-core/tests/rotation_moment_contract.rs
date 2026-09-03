//! Numerical-contract checks for rotation moment reductions.

use mlsirm_core::rotation::RotationCriterion;

fn scaled_tolerance(expected: f64) -> f64 {
    32.0 * f64::EPSILON * (1.0 + expected.abs())
}

#[test]
fn oblimax_matches_explicit_second_and_fourth_moment_formula() {
    let loadings = [0.8, -0.2, 0.1, 0.7, 0.5, -0.4, 0.3, 0.6];
    let evaluation = RotationCriterion::Oblimax
        .evaluate(&loadings, 4, 2)
        .expect("nonzero finite loadings satisfy the oblimax contract");

    let (sum2, sum4) = loadings.iter().fold((0.0_f64, 0.0_f64), |(s2, s4), &x| {
        let x2 = x * x;
        (s2 + x2, s4 + x2 * x2)
    });
    let expected_value = -(sum4.ln() - 2.0 * sum2.ln());

    assert!(
        (evaluation.value - expected_value).abs() <= scaled_tolerance(expected_value),
        "oblimax value drifted from the explicit second/fourth-moment formula: actual={} expected={}",
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
