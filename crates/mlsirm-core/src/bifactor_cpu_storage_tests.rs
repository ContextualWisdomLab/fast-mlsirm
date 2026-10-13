// Private full-crate selected controls; original live source is untouched.
// Basis: Cai et al. (2011, pp.231–232, Eqs.15–17; p.248, Eq.C6),
// local full-text sections actually read. Cache/storage are locally derived.
// Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
// item bifactor analysis. Psychological Methods, 16(3), 221–248.
// doi:10.1037/a0023350
#[path = "bifactor_cpu_baseline_tests.rs"]
mod baseline;
#[path = "bifactor_cpu_candidate_tests.rs"]
mod candidate;
fn parity(q: usize, masked: bool, shift: f64, zero_weight: bool) -> (usize, usize, usize) {
    let (original, calls_old, expected_old) = baseline::fixture(q, masked, shift, zero_weight);
    let (cached, calls_new, expected_new) = candidate::fixture(q, masked, shift, zero_weight);
    assert_eq!(original.len(), cached.len());
    assert_eq!(original, cached, "Likelihood/count/moment bits differ");
    assert_eq!(expected_old, expected_new);
    assert_eq!(
        calls_old,
        12 * 2 * q,
        "Baseline counter did not reach original block integration"
    );
    (calls_old, calls_new, expected_new)
}
#[test]
fn desired_unique_pattern_work_is_not_repeated() {
    let (old, new, expected) = parity(121, false, 0.0, false);
    assert!(old > expected);
    assert_eq!(
        new, expected,
        "Repeated group/block response-mask integral was reevaluated"
    );
}
#[test]
fn all_outputs_bit_equal_q121() {
    parity(121, false, 0.0, false);
}
#[test]
fn all_outputs_bit_equal_q241() {
    parity(241, false, 0.0, false);
}
#[test]
fn masked_and_all_missing_blocks_bit_equal() {
    parity(121, true, 0.0, false);
}
#[test]
fn zero_primary_and_specific_weight_bit_equal() {
    parity(121, true, 0.0, true);
}
#[test]
fn changed_parameters_do_not_reuse_previous_invocation() {
    let (a, _, _) = candidate::fixture(121, true, 0.0, false);
    let (b, _, _) = candidate::fixture(121, true, 0.2, false);
    assert_ne!(a, b, "Changed item parameters were ignored");
    parity(121, true, 0.2, false);
    let (again, _, _) = candidate::fixture(121, true, 0.0, false);
    assert_eq!(a, again, "Previous evaluation contaminated fresh call");
}

#[test]
fn cache_stores_only_integral_results() {
    let (_, calls, expected) = candidate::fixture(121, false, 0.0, false);
    assert_eq!(calls, expected);
    assert_eq!(
        candidate::cache_f64(),
        expected,
        "Full block arrays still retained in cache"
    );
}

#[test]
fn count_accumulation_uses_one_numeric_buffer() {
    let (_, _, _) = candidate::fixture(121, true, 0.0, false);
    assert_eq!(
        candidate::count_buffers(),
        1,
        "Counts still use a numeric allocation per node"
    );
}
