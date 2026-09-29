use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

#[test]
fn positive_between_sum_of_squares_must_not_underflow_during_mean_square_division() {
    let between_delta = 2_f64.powi(-537);
    let within_residual = 2_f64.powi(-530);
    let cluster_ids = [1_u64, 2, 3, 3, 4, 5, 6];
    let outcomes = [
        -between_delta,
        between_delta,
        -within_residual,
        within_residual,
        0.0,
        0.0,
        0.0,
    ];

    let between_square = between_delta * between_delta;
    assert_eq!(between_square.to_bits(), f64::from_bits(1).to_bits());
    let between_sum_of_squares = between_square + between_square;
    assert!(between_sum_of_squares > 0.0);
    assert_eq!((between_sum_of_squares / 5.0).to_bits(), 0.0_f64.to_bits());
    assert!(within_residual * within_residual > 0.0);

    let error = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect_err(
        "positive represented between sum of squares must fail closed if MSB rounds to zero",
    );
    assert_eq!(error, "ANOVA mean square underflowed binary64");
}
