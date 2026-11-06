//! Regression coverage for the extreme upper-tail probability used by Mokken AISP.
//!
//! The public AISP contract admits every finite `alpha` in `(0, 1)`. Very
//! small positive values must therefore remain fail-closed significance
//! thresholds even when binary64 tail arithmetic reaches its representable
//! endpoints.

use mlsirm_core::mokken::aisp;

#[test]
fn aisp_tiny_positive_alpha_does_not_turn_the_significance_gate_into_nan() {
    // Six persons x two varying items. Hij = 1/3 and Zij is finite (~0.745),
    // so an alpha of 1e-20 is far too stringent to admit the pair.
    let x = vec![
        0, 0, //
        0, 0, //
        0, 1, //
        1, 0, //
        1, 1, //
        1, 1, //
    ];

    let labels = aisp(&x, 6, 2, 0.3, 1e-20)
        .expect("every finite alpha in (0, 1) must remain numerically admissible");

    assert_eq!(
        labels,
        vec![0, 0],
        "a tiny positive alpha must preserve the significance rejection instead of producing a NaN critical value",
    );
}

#[test]
fn aisp_bonferroni_underflow_is_the_infinite_critical_value_limit() {
    // With three free items the first Bonferroni denominator is 3. The
    // smallest positive f64 divided by 3 rounds to zero even though the
    // nominal alpha itself is valid. The mathematical upper-tail quantile
    // limit at p -> 0+ is +infinity, so every finite Z must remain rejected.
    let x = vec![
        0, 0, 0, //
        0, 0, 1, //
        1, 1, 0, //
        1, 1, 1, //
    ];
    let alpha = f64::from_bits(1);

    let labels = aisp(&x, 4, 3, 0.3, alpha)
        .expect("Bonferroni underflow of a valid positive alpha must remain fail-closed");

    assert_eq!(
        labels,
        vec![0, 0, 0],
        "an adjusted tail probability rounded to zero must act as an infinite critical value",
    );
}
