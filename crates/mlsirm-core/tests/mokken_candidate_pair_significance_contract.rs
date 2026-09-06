//! Regression coverage for pairwise Criterion 1 during AISP candidate addition.
//!
//! Koopman, Zijlstra, and van der Ark (2022) define a Mokken scale with
//! `Hij > 0` for every item pair and describe AISP Step 2 as adding the next
//! item only when both Mokken scale criteria are accepted. For the traditional
//! AISP, Criterion 1 is accepted through the one-sided pairwise `Zij` test.
//! A candidate-level `Zi` result therefore cannot substitute for a failed
//! pairwise relationship with one already-selected item.

use mlsirm_core::mokken::{aisp, coef_h};

#[test]
fn aisp_rejects_candidate_when_one_selected_pair_fails_criterion_one() {
    // Repeat a six-person binary pattern ten times so the ordinary alpha=0.05
    // candidate-level test has useful power without changing the exact H
    // structure. Items 0 and 1 form the deterministic start pair. Item 2 is
    // perfectly scalable with item 0 (Hij = 1) but exactly unrelated to item 1
    // (Hij = 0, Zij = 0). Across all three items, item 2 still has Hi = 0.4
    // and Zi ~= 2.903, enough to clear the current Bonferroni-adjusted
    // candidate-level gate. Criterion 1 nevertheless fails for pair (1, 2).
    let pattern = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 0],
        [1, 1, 1],
    ];
    let mut x = Vec::with_capacity(60 * 3);
    for _ in 0..10 {
        for row in pattern {
            x.extend_from_slice(&row);
        }
    }

    let coefficients =
        coef_h(&x, 60, 3).expect("fixture must have finite nonzero item variance");
    assert!((coefficients.hij[1] - 1.0).abs() < 1e-12);
    assert!((coefficients.hij[2] - 1.0).abs() < 1e-12);
    assert_eq!(coefficients.hij[5], 0.0);
    assert_eq!(coefficients.zij[5], 0.0);
    assert!(coefficients.zi[2] > 2.9);

    let labels =
        aisp(&x, 60, 3, 0.0, 0.05).expect("AISP should accept the finite binary fixture");

    assert_eq!(
        labels,
        vec![1, 1, 0],
        "a candidate-level Zi pass must not override a failed pairwise Criterion 1 test",
    );
}
