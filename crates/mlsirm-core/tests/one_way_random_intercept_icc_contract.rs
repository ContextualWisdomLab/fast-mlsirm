use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

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
fn nist_random_effects_fixture_matches_published_variance_components() {
    // NIST Dataplot ONE WAY ANOVA, Program 2 (2023-09-27): five groups,
    // MSB = 36.933333, MSW = 1.8, between component = 11.711111,
    // and ICC = 86.68% after display rounding.
    let cluster_ids = [1_u64, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5];
    let outcomes = [
        74.0, 76.0, 75.0, 68.0, 71.0, 72.0, 75.0, 77.0, 77.0, 72.0, 74.0, 73.0,
        79.0, 81.0, 79.0,
    ];

    let result = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect("NIST fixture");

    assert_eq!(result.effective_cluster_size_n0.to_bits(), 3.0_f64.to_bits());
    assert!((result.within_variance - 1.8).abs() < 1e-12);
    assert!((result.between_variance - 11.711_111_111_111_11).abs() < 1e-12);
    assert!((result.icc - 0.866_776_315_789_473_7).abs() < 1e-12);
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
fn representable_location_shift_preserves_exact_variance_component_result() {
    let cluster_ids = [10_u64, 10, 20, 20, 20, 30, 30];
    let outcomes = [1.0, 2.0, 4.0, 5.0, 6.0, 9.0, 10.0];
    let location_shift = 2_f64.powi(52);
    let shifted = outcomes.map(|value| value + location_shift);

    for (&original, &translated) in outcomes.iter().zip(&shifted) {
        assert_eq!((translated - location_shift).to_bits(), original.to_bits());
    }

    let reference = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect("reference");
    let translated =
        one_way_random_intercept_icc(&cluster_ids, &shifted).expect("translated reference");

    assert_eq!(reference.icc.to_bits(), translated.icc.to_bits());
    assert_eq!(
        reference.between_variance.to_bits(),
        translated.between_variance.to_bits()
    );
    assert_eq!(
        reference.within_variance.to_bits(),
        translated.within_variance.to_bits()
    );
    assert_eq!(
        reference.effective_cluster_size_n0.to_bits(),
        translated.effective_cluster_size_n0.to_bits()
    );
}

#[test]
fn invalid_or_unidentified_inputs_fail_closed() {
    assert!(one_way_random_intercept_icc(&[], &[]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64], &[]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 2], &[1.0, f64::NAN]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 1], &[1.0, 2.0]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 2], &[1.0, 2.0]).is_err());
    assert!(one_way_random_intercept_icc(&[1_u64, 1, 2, 2], &[3.0, 3.0, 3.0, 3.0]).is_err());
}

#[test]
fn unbalanced_exact_constant_binary64_outcomes_fail_closed_before_aggregation() {
    let error = one_way_random_intercept_icc(&[1_u64, 1, 2, 2, 2], &[0.1; 5])
        .expect_err("exactly constant outcomes have zero total variance");

    assert_eq!(error, "one-way ICC requires positive total variance");
}

#[test]
fn positive_within_dispersion_must_not_underflow_into_unit_icc() {
    let cluster_ids = [10_u64, 10, 20, 20, 20, 30, 30];
    let scale = 1.0e-162;
    let outcomes = [
        1.0 * scale,
        2.0 * scale,
        4.0 * scale,
        5.0 * scale,
        6.0 * scale,
        9.0 * scale,
        10.0 * scale,
    ];

    assert_ne!(outcomes[0].to_bits(), outcomes[1].to_bits());
    assert_ne!(outcomes[2].to_bits(), outcomes[3].to_bits());

    let error = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect_err(
        "positive represented within-cluster dispersion must fail closed if squaring underflows binary64",
    );
    assert_eq!(error, "ANOVA squared deviation underflowed binary64");
}

#[test]
fn binary64_overflow_paths_fail_closed() {
    assert!(
        one_way_random_intercept_icc(
            &[1_u64, 1, 2],
            &[f64::MAX, f64::MAX, 0.0],
        )
        .is_err()
    );
    assert!(
        one_way_random_intercept_icc(
            &[1_u64, 1, 2, 2],
            &[-1.0e200, 1.0e200, -1.0, 1.0],
        )
        .is_err()
    );
}

#[test]
fn zero_within_variance_yields_unit_icc_when_between_variance_is_positive() {
    let result = one_way_random_intercept_icc(&[1_u64, 1, 2, 2], &[1.0, 1.0, 2.0, 2.0])
        .expect("identified no-residual-noise case");

    assert_eq!(result.within_variance.to_bits(), 0.0_f64.to_bits());
    assert_eq!(result.icc.to_bits(), 1.0_f64.to_bits());
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
