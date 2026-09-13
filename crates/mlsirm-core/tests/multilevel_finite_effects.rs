//! Regression coverage for finite contextual-effect inputs and outputs.

use mlsirm_core::multilevel::weighted_contextual_effect;

#[test]
fn rejects_non_finite_context_effects_before_weighted_sum() {
    for effect in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = weighted_contextual_effect(&[0, 1], &[0], &[1.0], &[effect], 1)
            .expect_err("non-finite contextual effects must fail closed");

        assert_eq!(error, "effects must be finite");
    }
}

#[test]
fn ignores_non_finite_unreferenced_context_effects() {
    let result = weighted_contextual_effect(
        &[0, 1],
        &[0],
        &[1.0],
        &[2.0, f64::NAN, f64::INFINITY],
        1,
    )
    .expect("unreferenced context effects must not expand validation work or affect output");

    assert_eq!(result, vec![2.0]);
}

#[test]
fn rejects_overflowing_weighted_contextual_output() {
    let error = weighted_contextual_effect(
        &[0, 2],
        &[0, 1],
        &[1.0, 1.0],
        &[f64::MAX, f64::MAX],
        1,
    )
    .expect_err("finite inputs that overflow the weighted sum must fail closed");

    assert_eq!(error, "weighted contextual effects must be finite");
}
