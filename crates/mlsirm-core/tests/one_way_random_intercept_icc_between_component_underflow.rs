use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

#[test]
fn positive_between_component_must_not_underflow_during_n0_scaling() {
    let unit = 2_f64.powi(-537);
    let cluster_ids = [1_u64, 1, 2, 2];
    let outcomes = [0.0, 2.0 * unit, unit, 4.0 * unit];

    let min_subnormal = f64::from_bits(1);
    let represented_msb = f64::from_bits(4);
    let represented_msw = f64::from_bits(3);
    let represented_method_numerator = represented_msb - represented_msw;

    assert_eq!((unit * unit).to_bits(), min_subnormal.to_bits());
    assert_eq!(represented_method_numerator.to_bits(), min_subnormal.to_bits());
    assert_eq!(
        (represented_method_numerator / 2.0).to_bits(),
        0.0_f64.to_bits(),
        "the witness must lose a positive represented method numerator only at n0 scaling"
    );

    let error = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect_err(
        "positive represented between variance must fail closed if n0 scaling rounds it to zero",
    );
    assert_eq!(error, "ANOVA variance component underflowed binary64");
}
