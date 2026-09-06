//! Regression coverage for exact ties during AISP candidate addition.
//!
//! The methodological AISP adds one next item at a time: among candidates
//! satisfying both Mokken criteria, choose the item that yields the highest
//! augmented-set `H`, then repeat the step against the enlarged scale. Exact
//! ties therefore need deterministic one-item resolution; batching tied
//! candidates from the same pre-step set can create a final scale containing a
//! negative `Hij` relationship that neither candidate was screened against.

use mlsirm_core::mokken::{aisp, coef_h};

#[test]
fn aisp_rechecks_exactly_tied_candidates_one_item_at_a_time() {
    // Items 0 and 1 are the unambiguous start pair. Items 2 and 3 have the
    // same margins and the same association (Hij = 0.4) with both start
    // items, so each produces the same augmented-set H = 0.6 and clears the
    // pre-step Hi/Z gates. They are negatively associated with each other
    // (Hij = -0.2).
    //
    // A valid Mokken scale requires positive pairwise scalability. The
    // deterministic first tied candidate (item 2) may join the start pair;
    // item 3 must then be reconsidered against the enlarged set and rejected
    // because Hij(2,3) is negative.
    let mut x = Vec::with_capacity(40 * 4);
    for person in 0..40 {
        let start = if person < 20 { 1 } else { 0 };
        let candidate_a = i64::from(person < 14 || (20..26).contains(&person));
        let candidate_b = i64::from((6..20).contains(&person) || (34..40).contains(&person));
        x.extend_from_slice(&[start, start, candidate_a, candidate_b]);
    }

    let coefficients =
        coef_h(&x, 40, 4).expect("fixture must have finite nonzero item variance");
    assert!((coefficients.hij[2 * 4 + 3] + 0.2).abs() < 1e-12);

    let labels =
        aisp(&x, 40, 4, 0.3, 0.05).expect("AISP should accept the finite binary fixture");

    assert_eq!(
        labels,
        vec![1, 1, 1, 0],
        "an exact candidate tie must resolve one item at a time and replay pairwise admissibility",
    );
}
