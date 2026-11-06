use mlsirm_core::samefit_strict::project;
#[test]
fn unequal_group_weighting() {
    let x = vec![5., 0., 0., 1.25];
    let v = project(&[0.1, 0.2], &[], x, 2, 0, 100.).unwrap();
    assert!((v["m2"].as_f64().unwrap() - 3.4).abs() < 1e-12);
}
#[test]
fn noncentral_ci_order() {
    for (m, df) in [(0., 10.), (10., 10.), (250., 100.), (1e4, 100.)] {
        let v = mlsirm_core::samefit_grm_joint::fit_indices(m, df, 1000, 30000., 1500.).unwrap();
        let lo = v["rmsea_ci90_lower"].as_f64().unwrap();
        let hi = v["rmsea_ci90_upper"].as_f64().unwrap();
        assert!(lo.is_finite() && hi.is_finite() && lo <= hi);
        println!("ci={v}");
    }
}
#[test]
fn scale_invariant_tangent() {
    let e = [0.1, 0.2, 0.3];
    let x = vec![2., 0., 0., 0., 3., 0., 0., 0., 4.];
    let a = project(&e, &[1., 2., 3.], x.clone(), 3, 1, 100.).unwrap();
    let b = project(&e, &[1e-20, 2e-20, 3e-20], x, 3, 1, 100.).unwrap();
    assert_eq!(a["rank"], b["rank"]);
    assert!((a["m2"].as_f64().unwrap() - b["m2"].as_f64().unwrap()).abs() < 1e-12);
}
