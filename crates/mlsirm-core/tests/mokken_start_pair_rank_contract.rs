//! Regression coverage for the start-pair ranking used by Mokken AISP.
//!
//! `mokken::search.normal` subtracts a row-index epsilon from each eligible
//! `Hij` before taking the argmax. That changes ordering for near-ties, not
//! only for bit-identical `Hij` values.

use mlsirm_core::mokken::aisp;

#[test]
fn aisp_applies_search_normal_epsilon_before_start_pair_argmax() {
    // Items 0 and 3 are an exact comonotone binary pair (Hij = 1). Items 1
    // and 2 are an almost-identical polytomous pair whose Hij is lower only
    // by binary64 rounding after one unit perturbation on a 1e9 score scale.
    // Cross-block association is zero or slightly negative, so the chosen
    // start pair determines which block receives scale label 1.
    //
    // In search.normal.R the lower-triangle matrix is ranked after
    // subtracting row * 1e-10. Pair (1,2) therefore outranks pair (0,3)
    // despite the latter's microscopically larger raw Hij. An exact-tie-only
    // comparator incorrectly assigns label 1 to items 0 and 3.
    let mut x = Vec::with_capacity(40 * 4);
    for person in 0..40 {
        let block_a = if person < 20 { 1 } else { 0 };
        let block_b = if person % 2 == 1 { 1_000_000_000_i64 } else { 0 };
        let block_b_peer = if person == 1 { block_b - 1 } else { block_b };
        x.extend_from_slice(&[block_a, block_b, block_b_peer, block_a]);
    }

    let labels =
        aisp(&x, 40, 4, 0.3, 0.05).expect("AISP should accept the finite integer fixture");

    assert_eq!(
        labels,
        vec![2, 1, 1, 2],
        "start-pair ranking must apply the canonical search.normal epsilon before argmax",
    );
}
