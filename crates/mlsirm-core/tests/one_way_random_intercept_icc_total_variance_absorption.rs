use mlsirm_core::one_way_random_intercept::one_way_random_intercept_icc;

#[test]
fn positive_within_variance_must_not_be_absorbed_into_false_unit_icc() {
    let unit = 2_f64.powi(-537);
    let min_subnormal = f64::from_bits(1);
    let cluster_ids = [1_u64, 1, 2, 2];
    let outcomes = [0.0, 2.0 * unit, 1.0, 1.0];

    assert_eq!((unit * unit).to_bits(), min_subnormal.to_bits());

    let represented_between_variance = 0.5_f64;
    let represented_within_variance = min_subnormal;
    assert_eq!(
        (represented_between_variance + represented_within_variance).to_bits(),
        represented_between_variance.to_bits(),
        "the witness must lose a positive represented within component only at total-variance composition"
    );

    let error = one_way_random_intercept_icc(&cluster_ids, &outcomes).expect_err(
        "positive represented within variance must fail closed if total-variance composition absorbs it",
    );
    assert_eq!(error, "ANOVA total variance lost positive component in binary64");
}

#[test]
fn exact_zero_within_variance_still_allows_unit_icc() {
    let cluster_ids = [1_u64, 1, 2, 2];
    let outcomes = [0.0, 0.0, 1.0, 1.0];

    let estimate = one_way_random_intercept_icc(&cluster_ids, &outcomes)
        .expect("exact zero within variance with positive between variance is a valid unit ICC");

    assert_eq!(estimate.within_variance.to_bits(), 0.0_f64.to_bits());
    assert!(estimate.between_variance > 0.0);
    assert_eq!(estimate.icc.to_bits(), 1.0_f64.to_bits());
}
