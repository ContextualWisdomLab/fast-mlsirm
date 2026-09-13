//! Sparse cross-classified multiple-membership contextual effects.
//!
//! Implements the contextual term of the multilevel latent-trait linear
//! predictor from the atomistic-fallacy foundation RFC
//! (`docs/multilevel_multiple_membership_longitudinal_rfc.md`, this
//! repository's issue #565): for response `Y_pitr` from person/system `p`,
//! item/task `i`, occasion `t`, and rater/engine `r`,
//!
//! ```text
//! eta_pitr = a_i * theta_pt + b_i - rho_rt + z_p^T beta + sum_h w_ph * u_h
//! ```
//!
//! where `u_h` is a contextual random effect and the non-negative membership
//! weights satisfy `sum_h w_ph = 1`. This module implements exactly the
//! `sum_h w_ph * u_h` term, generalized so simultaneous cross-classified
//! dimensions (for example team AND school membership at once) are simply
//! more edges in one person's sparse row: the caller flattens
//! (context_dimension, context) into one global effect index (see
//! `python/fast_mlsirm/multilevel/contracts.py`'s `ContextMembershipDesign`,
//! whose per-dimension weights already sum to one, for the validated sparse
//! design this consumes).
//!
//! Ordinary nesting (each person belongs to exactly one context per
//! dimension) is the one-hot special case `w_ph = 1` for exactly one `h` per
//! dimension -- no separate code path, just a design with singleton weights.
//!
//! # Determinism
//!
//! Each observation's edges are summed in ascending context-index order
//! regardless of the order the caller supplied them in, so the result is
//! identical (bit-for-bit, not just numerically close) under any permutation
//! of a row's edges. Rows are independent, so the result is also identical
//! under any permutation of the *rows themselves*, and under any
//! `worker_count` -- see [`weighted_contextual_effect`].
//!
//! # Verified sources
//!
//! Browne, Goldstein, and Rasbash (2001) define the multiple membership
//! multiple classification (MMMC) linear predictor's contextual term as an
//! explicit weighted sum over classification units, with membership weights
//! constrained to be non-negative and sum to one within each classification
//! (their eq. 1 and surrounding discussion of "weights ... which sum to one
//! for each individual"). This module implements exactly that additive sum;
//! it does not implement MMMC's Bayesian/MCMC estimation of the `u_h`
//! themselves (a later PR in the issue #565 staged plan owns fitting the
//! random effects -- this module only evaluates the linear predictor's
//! contextual term given already-estimated or trial `u_h` values, which is
//! what every iterative fitting method needs to call on every evaluation).
//!
//! # References
//!
//! Browne, W. J., Goldstein, H., & Rasbash, J. (2001). Multiple membership
//! multiple classification (MMMC) models. *Statistical Modelling, 1*(2),
//! 103-124. <https://doi.org/10.1177/1471082X0100100202>

use std::thread;

/// One error class for every rejected sparse-design input.
///
/// Kept as a plain `String` message, matching this crate's existing
/// `Result<_, String>` convention (see `parallel.rs`, `fitstats.rs`) rather
/// than introducing a new error type for one module.
type MultilevelResult<T> = Result<T, String>;

fn validate_worker_count(worker_count: usize) -> MultilevelResult<()> {
    if worker_count == 0 {
        return Err("worker_count must be at least one".to_string());
    }
    Ok(())
}

fn validate_row_offsets(row_offsets: &[usize], edge_count: usize) -> MultilevelResult<usize> {
    if row_offsets.is_empty() {
        return Err("row_offsets must contain at least one entry".to_string());
    }
    if row_offsets[0] != 0 {
        return Err("row_offsets must start at zero".to_string());
    }
    for window in row_offsets.windows(2) {
        if window[1] < window[0] {
            return Err("row_offsets must be non-decreasing".to_string());
        }
    }
    let last = *row_offsets.last().expect("row_offsets is non-empty");
    if last != edge_count {
        return Err(
            "row_offsets must end at the total edge count (context_indices/weights length)"
                .to_string(),
        );
    }
    Ok(row_offsets.len() - 1)
}

fn validate_edges(
    context_indices: &[usize],
    weights: &[f64],
    effect_count: usize,
) -> MultilevelResult<()> {
    if context_indices.len() != weights.len() {
        return Err("context_indices and weights must have equal length".to_string());
    }
    for &index in context_indices {
        if index >= effect_count {
            return Err("context_indices contains an index outside effects".to_string());
        }
    }
    for &weight in weights {
        if !weight.is_finite() || weight < 0.0 {
            return Err("weights must be finite and non-negative".to_string());
        }
    }
    Ok(())
}

/// Sum one observation's edges in ascending context-index order.
///
/// Sorting before summing (rather than summing in whatever order the caller
/// supplied) is what makes the result independent of edge order and worker
/// count, not just numerically close under reordering.
fn row_effect(row_context_indices: &[usize], row_weights: &[f64], effects: &[f64]) -> f64 {
    let mut order: Vec<usize> = (0..row_context_indices.len()).collect();
    order.sort_unstable_by_key(|&position| row_context_indices[position]);
    let mut total = 0.0_f64;
    for position in order {
        total += row_weights[position] * effects[row_context_indices[position]];
    }
    total
}

/// Compute each observation's weighted contextual random-effect contribution
/// `sum_h w_ph * u_h` for a sparse CSR-style cross-classified
/// multiple-membership design (Browne, Goldstein, & Rasbash, 2001; see the
/// module docs for the full predictor and citation).
///
/// # Arguments
///
/// * `row_offsets` - CSR row pointer; `row_offsets[p]..row_offsets[p + 1]`
///   indexes into `context_indices`/`weights` for observation `p`'s edges.
///   Length must be `n_observations + 1`, starting at zero and
///   non-decreasing, ending at `context_indices.len()`.
/// * `context_indices` - global (dimension, context) effect index per edge.
/// * `weights` - membership weight per edge, same length as
///   `context_indices`. Must be finite and non-negative (this function does
///   not itself enforce the per-dimension "sums to one" contract -- that is
///   validated once, at design-construction time, by
///   `python/fast_mlsirm/multilevel/contracts.py::build_context_membership_design`;
///   re-deriving it here from a flattened, dimension-erased edge list would
///   require re-threading dimension boundaries through this hot path for no
///   additional safety, since a design that violates it was already
///   rejected before reaching Rust).
/// * `effects` - per-context random-effect value, indexed by
///   `context_indices`.
/// * `worker_count` - number of deterministic worker threads (must be `>=
///   1`); the result does not depend on this value (see module docs).
///
/// # Errors
///
/// Returns `Err` for malformed `row_offsets`, an out-of-range
/// `context_indices` entry, a non-finite or negative weight, or
/// `worker_count == 0`. Never panics on well-typed input.
pub fn weighted_contextual_effect(
    row_offsets: &[usize],
    context_indices: &[usize],
    weights: &[f64],
    effects: &[f64],
    worker_count: usize,
) -> MultilevelResult<Vec<f64>> {
    validate_worker_count(worker_count)?;
    let n_observations = validate_row_offsets(row_offsets, context_indices.len())?;
    validate_edges(context_indices, weights, effects.len())?;

    let mut output = vec![0.0_f64; n_observations];
    if n_observations == 0 {
        return Ok(output);
    }

    let worker_count = worker_count.min(n_observations);
    let chunk_size = n_observations.div_ceil(worker_count);

    thread::scope(|scope| {
        for (chunk_index, output_chunk) in output.chunks_mut(chunk_size).enumerate() {
            let start_row = chunk_index * chunk_size;
            scope.spawn(move || {
                for (offset, slot) in output_chunk.iter_mut().enumerate() {
                    let row = start_row + offset;
                    let edge_start = row_offsets[row];
                    let edge_end = row_offsets[row + 1];
                    *slot = row_effect(
                        &context_indices[edge_start..edge_end],
                        &weights[edge_start..edge_end],
                        effects,
                    );
                }
            });
        }
    });

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // One-hot nesting parity: exactly one edge per row, weight 1.0, is
    // ordinary single-membership nesting and must reproduce a direct
    // lookup exactly.
    // ------------------------------------------------------------------

    #[test]
    fn one_hot_nesting_reproduces_direct_lookup() {
        // 4 observations, one-hot into 3 contexts (ordinary nesting).
        let row_offsets = [0, 1, 2, 3, 4];
        let context_indices = [0, 1, 2, 0];
        let weights = [1.0, 1.0, 1.0, 1.0];
        let effects = [10.0, 20.0, 30.0];

        let result =
            weighted_contextual_effect(&row_offsets, &context_indices, &weights, &effects, 1)
                .unwrap();

        assert_eq!(result, vec![10.0, 20.0, 30.0, 10.0]);
    }

    #[test]
    fn one_hot_nesting_matches_weighted_membership_at_the_boundary() {
        // A single weighted-membership edge with weight exactly 1.0 must
        // equal the one-hot case -- weighted membership generalizes nesting,
        // it does not diverge from it at the w=1 boundary.
        let row_offsets = [0, 2];
        let context_indices = [0, 1];
        let weights = [0.25, 0.75];
        let effects = [4.0, 4.0]; // equal effects: weighted mix == either alone.

        let weighted =
            weighted_contextual_effect(&row_offsets, &context_indices, &weights, &effects, 1)
                .unwrap();
        let one_hot = weighted_contextual_effect(&[0, 1], &[0], &[1.0], &effects, 1).unwrap();

        assert_eq!(weighted, one_hot);
    }

    #[test]
    fn weighted_membership_computes_the_convex_combination() {
        let row_offsets = [0, 2];
        let context_indices = [0, 1];
        let weights = [0.25, 0.75];
        let effects = [8.0, 4.0];

        let result =
            weighted_contextual_effect(&row_offsets, &context_indices, &weights, &effects, 1)
                .unwrap();

        assert_eq!(result, vec![0.25 * 8.0 + 0.75 * 4.0]);
    }

    // ------------------------------------------------------------------
    // Permutation invariance: edge order within a row, and row order
    // itself, must not change the result.
    // ------------------------------------------------------------------

    #[test]
    fn permuting_edges_within_a_row_does_not_change_its_result() {
        let effects = [1.0, 2.0, 3.0, 4.0];
        let forward =
            weighted_contextual_effect(&[0, 4], &[0, 1, 2, 3], &[0.1, 0.2, 0.3, 0.4], &effects, 1)
                .unwrap();
        let reversed =
            weighted_contextual_effect(&[0, 4], &[3, 2, 1, 0], &[0.4, 0.3, 0.2, 0.1], &effects, 1)
                .unwrap();
        let shuffled =
            weighted_contextual_effect(&[0, 4], &[2, 0, 3, 1], &[0.3, 0.1, 0.4, 0.2], &effects, 1)
                .unwrap();

        assert_eq!(forward, reversed);
        assert_eq!(forward, shuffled);
    }

    #[test]
    fn permuting_rows_does_not_change_any_row_s_own_result() {
        // 3 independent one-edge rows; reordering which row comes first
        // must not change any individual row's own value.
        let context_indices = [0, 1, 2];
        let weights = [1.0, 1.0, 1.0];
        let effects = [100.0, 200.0, 300.0];

        let original =
            weighted_contextual_effect(&[0, 1, 2, 3], &context_indices, &weights, &effects, 1)
                .unwrap();
        let row_permuted_indices = [2, 0, 1];
        let row_permuted_weights = [1.0, 1.0, 1.0];
        let permuted = weighted_contextual_effect(
            &[0, 1, 2, 3],
            &row_permuted_indices,
            &row_permuted_weights,
            &effects,
            1,
        )
        .unwrap();

        assert_eq!(original, vec![100.0, 200.0, 300.0]);
        assert_eq!(permuted, vec![300.0, 100.0, 200.0]);
    }

    // ------------------------------------------------------------------
    // Determinism across worker counts.
    // ------------------------------------------------------------------

    #[test]
    fn result_is_identical_across_worker_counts() {
        let n = 97usize; // deliberately not a multiple of common worker counts
        let row_offsets: Vec<usize> = (0..=n).collect();
        let context_indices: Vec<usize> = (0..n).map(|i| i % 5).collect();
        let weights: Vec<f64> = vec![1.0; n];
        let effects = [1.5, 2.5, 3.5, 4.5, 5.5];

        let single =
            weighted_contextual_effect(&row_offsets, &context_indices, &weights, &effects, 1)
                .unwrap();
        for worker_count in [2, 3, 4, 8, 16, 200] {
            let multi = weighted_contextual_effect(
                &row_offsets,
                &context_indices,
                &weights,
                &effects,
                worker_count,
            )
            .unwrap();
            assert_eq!(single, multi, "worker_count={worker_count} diverged");
        }
    }

    #[test]
    fn cross_classified_dimensions_sum_their_separate_contributions() {
        // Person 0 is one-hot in dimension A (context 0) AND one-hot in
        // dimension B (context 1, flattened to global index 1): the
        // predictor sums both dimensions' contributions.
        let row_offsets = [0, 2];
        let context_indices = [0, 1];
        let weights = [1.0, 1.0];
        let effects = [10.0, 100.0];

        let result =
            weighted_contextual_effect(&row_offsets, &context_indices, &weights, &effects, 1)
                .unwrap();

        assert_eq!(result, vec![110.0]);
    }

    // ------------------------------------------------------------------
    // Fail-closed validation.
    // ------------------------------------------------------------------

    #[test]
    fn rejects_zero_worker_count() {
        assert!(weighted_contextual_effect(&[0], &[], &[], &[], 0).is_err());
    }

    #[test]
    fn rejects_empty_row_offsets() {
        assert!(weighted_contextual_effect(&[], &[], &[], &[], 1).is_err());
    }

    #[test]
    fn rejects_row_offsets_not_starting_at_zero() {
        assert!(weighted_contextual_effect(&[1, 1], &[], &[], &[], 1).is_err());
    }

    #[test]
    fn rejects_decreasing_row_offsets() {
        assert!(
            weighted_contextual_effect(&[0, 3, 2], &[0, 0, 0], &[1.0, 1.0, 1.0], &[1.0], 1)
                .is_err()
        );
    }

    #[test]
    fn rejects_row_offsets_mismatched_with_edge_count() {
        assert!(weighted_contextual_effect(&[0, 5], &[0, 0], &[1.0, 1.0], &[1.0], 1).is_err());
    }

    #[test]
    fn rejects_context_index_out_of_range() {
        assert!(weighted_contextual_effect(&[0, 1], &[3], &[1.0], &[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn rejects_mismatched_context_indices_and_weights_length() {
        assert!(weighted_contextual_effect(&[0, 2], &[0, 1], &[1.0], &[1.0, 2.0], 1).is_err());
    }

    #[test]
    fn rejects_non_finite_weight() {
        assert!(weighted_contextual_effect(&[0, 1], &[0], &[f64::NAN], &[1.0], 1).is_err());
        assert!(weighted_contextual_effect(&[0, 1], &[0], &[f64::INFINITY], &[1.0], 1).is_err());
    }

    #[test]
    fn rejects_negative_weight() {
        assert!(weighted_contextual_effect(&[0, 1], &[0], &[-0.5], &[1.0], 1).is_err());
    }

    #[test]
    fn accepts_zero_weight_as_a_valid_non_contributing_edge() {
        // Zero is finite and non-negative; a zero-weight edge is unusual
        // but not malformed (a design could legitimately transition an
        // edge's weight toward zero across revisions).
        let result = weighted_contextual_effect(&[0, 1], &[0], &[0.0], &[7.0], 1).unwrap();
        assert_eq!(result, vec![0.0]);
    }

    #[test]
    fn empty_design_returns_an_empty_result() {
        let result = weighted_contextual_effect(&[0], &[], &[], &[], 1).unwrap();
        assert_eq!(result, Vec::<f64>::new());
    }

    #[test]
    fn observation_with_no_edges_contributes_zero() {
        // A person with zero contextual memberships in this flattened
        // design (for example, present only through a dimension the caller
        // omitted) contributes no contextual effect, not an error.
        let result = weighted_contextual_effect(&[0, 0, 1], &[0], &[1.0], &[9.0], 1).unwrap();
        assert_eq!(result, vec![0.0, 9.0]);
    }

    #[test]
    fn worker_count_larger_than_observation_count_is_clamped_not_wasted() {
        // 3 observations with worker_count=50 must not panic on an empty
        // chunk_size or spawn 50 threads for 3 rows.
        let result = weighted_contextual_effect(
            &[0, 1, 2, 3],
            &[0, 1, 2],
            &[1.0, 1.0, 1.0],
            &[1.0, 2.0, 3.0],
            50,
        )
        .unwrap();
        assert_eq!(result, vec![1.0, 2.0, 3.0]);
    }
}
