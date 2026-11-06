#![cfg(feature = "gpu")]
use mlsirm_core::samefit_grm::{quadrature, PackedArchive};
use mlsirm_core::samefit_grm_joint::{
    analytic_moments_delta, fit_indices, joint_diagnostics_with_backend, MomentBackend,
};
use mlsirm_core::samefit_strict::{cpu_moments, project};

#[test]
fn tiny_finite_tangent_preserves_direction() {
    let control = project(&[1., 0.], &[1., 0.], vec![1., 0., 0., 1.], 2, 1, 10.).unwrap();
    let actual = project(&[1., 0.], &[1e-200, 0.], vec![1., 0., 0., 1.], 2, 1, 10.).unwrap();
    println!("control={control} tiny={actual}");
    assert_eq!(
        actual["rank"], control["rank"],
        "unit scaling must not discard a nonzero direction"
    );
    assert_eq!(actual["df"], control["df"]);
    assert_eq!(actual["m2"], control["m2"]);
}
#[test]
fn huge_finite_tangent_preserves_direction() {
    let actual = project(&[1., 0.], &[1e200, 0.], vec![1., 0., 0., 1.], 2, 1, 10.);
    println!("huge={actual:?}");
    let v = actual.expect("finite rescaling of one direction must be admitted");
    assert_eq!(v["rank"], 1);
    assert_eq!(v["m2"], 0.);
}
#[test]
fn positive_measured_rank_joint_is_not_rejected_by_raw_columns() {
    let a = PackedArchive::new(vec![true; 2], 2, 2, vec![0., 0., 0., 0., 0., 0.]).unwrap();
    let (nodes, weights) = quadrature(121).unwrap();
    let (_, delta) = analytic_moments_delta(&a, &nodes, &weights).unwrap();
    let m = cpu_moments(&a, &nodes, &weights).unwrap();
    let mut x = vec![0.; 36];
    for g in 0..2 {
        for r in 0..3 {
            for c in 0..3 {
                x[(g * 3 + r) * 6 + g * 3 + c] = 2. * m.xi[g * 9 + r * 3 + c];
            }
        }
    }
    let strict = project(&[0.; 6], &delta, x, 6, 6, 8.).unwrap();
    assert!(strict["df"].as_u64().unwrap() > 0);
    assert_eq!(strict["ridge"], 0.);
    let y = vec![0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1];
    let result =
        joint_diagnostics_with_backend(&y, &[0, 0, 0, 0, 1, 1, 1, 1], &a, 121, MomentBackend::Cpu);
    println!("strict={strict} joint={result:?}");
    assert!(
        result.is_ok(),
        "full consumer must defer df to measured tangent rank"
    );
}
#[test]
fn index_overflow_cannot_be_accepted_as_zero_rmsea() {
    // Finite inputs overflow N*df. Rejection or an explicitly unavailable field
    // is acceptable; a silently accepted infinite denominator and zero are not.
    let result = fit_indices(1.7e308, 1e308, 4, 1.75e308, 1e307);
    println!("index_overflow={result:?}");
    assert!(result.is_err(), "overflow must be refused");
    if let Ok(v) = result {
        assert!(
            v["rmsea_denominator_value"]
                .as_f64()
                .is_some_and(f64::is_finite),
            "accepted output must retain a finite denominator"
        );
    }
}
