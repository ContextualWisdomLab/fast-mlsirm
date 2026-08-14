//! Regression coverage for finite contextual-effect inputs.

use mlsirm_core::multilevel::weighted_contextual_effect;

#[test]
fn rejects_non_finite_context_effects_before_weighted_sum() {
    for effect in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let error = weighted_contextual_effect(&[0, 1], &[0], &[1.0], &[effect], 1)
            .expect_err("non-finite contextual effects must fail closed");

        assert_eq!(error, "effects must be finite");
    }
}
