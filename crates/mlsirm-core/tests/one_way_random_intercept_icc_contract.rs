use mlsirm_core::reliability::one_way_random_intercept_icc;

#[test]
fn unbalanced_one_way_random_intercept_icc_matches_hand_calculation() {
    let cluster_ids = [10_u64, 10, 20, 20, 20, 30, 30];
    let outcomes = [1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 10.0];

    let result = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect("identified ICC");

    assert_eq!(result.sample_size, 7);
    assert_eq!(result.cluster_count, 3);
    assert!((result.within_variance - 0.75).abs() < 1e-14);
    assert!((result.effective_cluster_size_n0 - 16.0 / 7.0).abs() < 1e-14);
    assert!((result.between_variance - 13.765625).abs() < 1e-12);
    assert!((result.icc - 0.9483315392895587).abs() < 1e-14);
}

#[test]
fn row_permutations_preserve_the_exact_reference_result() {
    let cluster_ids = [10_u64, 10, 20, 20, 20, 30, 30];
    let outcomes = [1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 10.0];
    let reference = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect("reference");

    let permuted_cluster_ids = [30_u64, 20, 10, 20, 30, 10, 20];
    let permuted_outcomes = [10.0, 5.0, 1.0, 6.0, 9.0, 2.0, 4.0];
    let permuted = one_way_random_intercept_icc(&permuted_cluster_ids, &permuted_outcomes)
        .expect("permuted reference");

    assert_eq!(reference.icc.to_bits(), permuted.icc.to_bits());
    assert_eq!(
        reference.between_variance.to_bits(),
        permuted.between_variance.to_bits()
    );
    assert_eq!(
        reference.within_variance.to_bits(),
        permuted.within_variance.to_bits()
    );
    assert_eq!(
        reference.effective_cluster_size_n0.to_bits(),
        permuted.effective_cluster_size_n0.to_bits()
    );
}

#[test]
fn invalid_or_unidentified_inputs_fail_closed() {
    assert!(one_way_random_intercept_icc(&[1_u64], &[]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64], &[1.0, 2.0]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 2], &[1.0, f64::NAN]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 1], &[1.0, 2.0]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 2], &[1.0, 2.0]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 1, 2, 2], &[3.0, 3.0, 3.0, 3.0]).is_err());
}

#[test]
fn negative_method_of_moments_between_component_is_bounded_at_zero() {
    let cluster_ids = [1_u64, 1, 2, 2, 3, 3];
    let outcomes = [-1.0, 1.0, -1.0, 1.0, -1.0, 1.0];

    let result = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect("identified ICC");

    assert_eq!(result.between_variance.to_bits(), 0.0_f64.to_bits());
    assert_eq!(result.icc.to_bits(), 0.0_f64.to_bits());
    assert!(result.within_variance > 0.0);
}
