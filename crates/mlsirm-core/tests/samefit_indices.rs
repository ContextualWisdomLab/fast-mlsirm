use mlsirm_core::samefit_grm_joint::fit_indices;
#[test]
fn native_fit_indices_have_ordered_rmsea_interval() {
    let x = fit_indices(120.0, 50.0, 500, 500.0, 60.0).unwrap();
    assert!((x["rmsea"].as_f64().unwrap() - (70.0f64 / 25000.0).sqrt()).abs() < 1e-12);
    assert!(x["rmsea_ci90_lower"].as_f64().unwrap() < x["rmsea"].as_f64().unwrap());
    assert!(x["rmsea_ci90_upper"].as_f64().unwrap() > x["rmsea"].as_f64().unwrap());
    assert!((x["cfi"].as_f64().unwrap() - (1.0 - 70.0 / 440.0)).abs() < 1e-12);
    assert!(fit_indices(1.0, 0.0, 10, 2.0, 1.0).is_err());
}
