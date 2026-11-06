use mlsirm_core::{samefit_grm_joint::fit_indices, samefit_strict::project};
#[test]
fn denominator_is_n_df_for_point_and_interval() {
    let a = fit_indices(25., 10., 100, 200., 20.).unwrap();
    let b = fit_indices(25., 10., 99, 200., 20.).unwrap();
    assert_eq!(a["rmsea_denominator"], "N * df");
    assert_eq!(a["rmsea_denominator_value"], 1000.);
    for field in ["rmsea", "rmsea_ci90_lower", "rmsea_ci90_upper"] {
        assert!(
            (a[field].as_f64().unwrap() * 100f64.sqrt()
                - b[field].as_f64().unwrap() * 99f64.sqrt())
            .abs()
                < 1e-12
        );
    }
}
#[test]
fn projection_refuses_overflow_nonfinite_and_asymmetry() {
    assert!(project(&[], &[], vec![], usize::MAX, 2, 1.).is_err());
    assert!(project(&[f64::NAN, 0.], &[], vec![1., 0., 0., 1.], 2, 0, 1.).is_err());
    assert!(project(&[1., 0.], &[], vec![1., 0.5, 0., 1.], 2, 0, 1.).is_err());
    assert!(project(&[1e308, 1e308], &[], vec![1., 0., 0., 1.], 2, 0, 1.).is_err());
}
#[test]
fn duplicate_columns_allow_positive_measured_df() {
    let v = project(
        &[1., 2.],
        &[1., 1., 0., 0.],
        vec![1., 0., 0., 1.],
        2,
        2,
        10.,
    )
    .unwrap();
    assert_eq!(v["rank"], 1);
    assert_eq!(v["df"], 1);
    assert_eq!(v["ridge"], 0.);
}
#[cfg(feature = "gpu")]
#[test]
fn explicit_cpu_joint_pipeline() {
    use mlsirm_core::samefit_grm::PackedArchive;
    use mlsirm_core::samefit_grm_joint::{joint_diagnostics_with_backend, MomentBackend};
    let a = PackedArchive::new(
        vec![true; 4],
        2,
        2,
        vec![0.7, 0.3, 1.0, -0.2, 1.2, 0.5, 0.8, -0.4, 0.2, 0.1],
    )
    .unwrap();
    let mut y = vec![];
    let mut groups = vec![];
    for g in 0..2 {
        for pattern in 0..16 {
            for _ in 0..(1 + pattern % 3) {
                groups.push(g);
                for i in 0..4 {
                    y.push((pattern >> i) & 1);
                }
            }
        }
    }
    let r = joint_diagnostics_with_backend(&y, &groups, &a, 121, MomentBackend::Cpu).unwrap();
    assert_eq!(r["gpu_dispatch"]["dispatches"], 0);
    assert_eq!(r["strict_projection"]["ridge"], 0.);
    assert!(r["m2"].as_f64().unwrap().is_finite());
}
