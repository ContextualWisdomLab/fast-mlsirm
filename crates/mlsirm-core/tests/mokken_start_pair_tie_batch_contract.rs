//! Regression coverage for exact ties at AISP start-pair selection.
//!
//! The methodological AISP starts each scale with exactly two items: the
//! admissible pair with the highest `Hij`. Exact ties therefore need a
//! deterministic two-item resolution; they must not be unioned into a larger
//! start set before the `Hi >= c` gate. The CRAN implementation vectorizes
//! equal maxima in `search.normal.R`, but that edge behavior can duplicate a
//! shared endpoint and no longer represents the published two-item start step.

use mlsirm_core::mokken::{aisp, coef_h};

#[test]
fn aisp_keeps_exact_start_ties_to_one_deterministic_pair() {
    // Pair (0, 2) and pair (1, 2) both have Hij = 1 and share the same
    // lower-triangle row, so their epsilon-adjusted start-pair scores tie.
    // Pair (0, 1) has Hij = -0.5. The AISP start step is nevertheless one
    // pair, not the union {0, 1, 2}; deterministic scan order selects (0, 2).
    // Item 1 is then excluded from the add step by its negative Hij with item
    // 0, leaving the scientifically admissible two-item scale.
    let x = vec![
        0, 0, 0,
        0, 1, 0,
        0, 1, 0,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    ];

    let coefficients = coef_h(&x, 6, 3).expect("fixture must have finite nonzero item variance");
    assert!((coefficients.hij[2] - 1.0).abs() < 1e-12);
    assert!((coefficients.hij[5] - 1.0).abs() < 1e-12);
    assert!((coefficients.hij[1] + 0.5).abs() < 1e-12);

    let labels = aisp(&x, 6, 3, 0.3, 0.9).expect("AISP should accept the finite integer fixture");
    assert_eq!(
        labels,
        vec![1, 0, 1],
        "an exact maximum tie must resolve to one deterministic two-item start pair",
    );
}
