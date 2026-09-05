//! Regression coverage for the extreme upper-tail probability used by Mokken AISP.
//!
//! The public AISP contract admits every finite `alpha` in `(0, 1)`. A very
//! small positive alpha must therefore remain a finite, fail-closed
//! significance threshold rather than collapsing through `1.0 - p` rounding.

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
