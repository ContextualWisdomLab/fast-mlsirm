//! Regression coverage for exact-tie candidate batches in Mokken AISP.
//!
//! `mokken::search.normal` assigns every candidate whose augmented-set H
//! equals the step maximum in the same iteration. Recomputing after only one
//! tied candidate can change the reference result when tied candidates are
//! negatively associated with one another.

use mlsirm_core::mokken::aisp;

#[test]
fn aisp_adds_all_exactly_tied_best_candidates_in_one_reference_step() {
    // Items 0 and 1 are the unambiguous start pair. Items 2 and 3 have the
    // same margins and the same association (Hij = 0.4) with both start
    // items, so each produces the same augmented-set H = 0.6 and clears the
    // step's Hi/Z gates. They are negatively associated with each other
    // (Hij = -0.2).
    //
    // search.normal.R computes all candidate H values from the pre-step set,
    // then assigns every item exactly equal to max(result). The reference
    // therefore adds items 2 and 3 together. Adding only the first winner and
    // recomputing incorrectly excludes the other on the next iteration.
    let mut x = Vec::with_capacity(40 * 4);
    for person in 0..40 {
        let start = if person < 20 { 1 } else { 0 };
        let candidate_a = i64::from(person < 14 || (20..26).contains(&person));
        let candidate_b = i64::from((6..20).contains(&person) || (34..40).contains(&person));
        x.extend_from_slice(&[start, start, candidate_a, candidate_b]);
    }

    let labels =
        aisp(&x, 40, 4, 0.3, 0.05).expect("AISP should accept the finite binary fixture");

    assert_eq!(
        labels,
        vec![1, 1, 1, 1],
        "exactly tied best candidates must be admitted as one search.normal step",
    );
}
