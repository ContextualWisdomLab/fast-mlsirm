//! Regression coverage for tied AISP start-pair formation.
//!
//! `mokken::search.normal` collects every lower-triangle cell tied at the
//! maximum epsilon-adjusted `Hij`, forms one unique `StartSet` from all of
//! those pair endpoints, and then applies the start-set `Hi >= c` gate. A
//! Rust implementation must not collapse that tied maximum to one arbitrary
//! pair before the start-set gate.

use mlsirm_core::mokken::{aisp, coef_h};

#[test]
fn aisp_applies_start_set_gate_to_all_tied_maximum_pairs() {
    // Pair (0, 2) and pair (1, 2) both have Hij = 1 and share the same
    // lower-triangle row, so their epsilon-adjusted start-pair scores tie.
    // Pair (0, 1) has Hij = -0.5. The canonical search.normal StartSet is
    // therefore {0, 1, 2}; its item Hi values include values below c = 0.3,
    // so scale formation stops with every item unassigned.
    //
    // Collapsing the tied maximum to only pair (0, 2) incorrectly forms a
    // two-item scale because the negative (0, 1) relationship is then used
    // only to exclude item 1 from the later candidate step.
    let x = vec![
        0, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    ];

    let coefficients = coef_h(&x, 6, 3).expect("fixture must have finite nonzero item variance");
    assert!((coefficients.hij[0 * 3 + 2] - 1.0).abs() < 1e-12);
    assert!((coefficients.hij[1 * 3 + 2] - 1.0).abs() < 1e-12);
    assert!((coefficients.hij[0 * 3 + 1] + 0.5).abs() < 1e-12);

    let labels = aisp(&x, 6, 3, 0.3, 0.9).expect("AISP should accept the finite integer fixture");
    assert_eq!(
        labels,
        vec![0, 0, 0],
        "all epsilon-adjusted maximum start pairs must participate in the canonical StartSet gate",
    );
}
