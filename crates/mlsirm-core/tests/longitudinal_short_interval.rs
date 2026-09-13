use mlsirm_core::longitudinal::fit_longitudinal_state;

#[test]
fn recovers_one_millisecond_respondent_trend() {
    let fit = fit_longitudinal_state(
        &[0, 2],
        &[0, 1],
        &[0, 1],
        &[2.0, 3.0],
        "random_intercept_slope",
        None,
        1,
    )
    .unwrap();

    assert!((fit.intercepts[0] - 2.0).abs() < 1e-12);
    assert!((fit.slopes[0] - 86_400_000.0).abs() < 1e-3);
    assert!((fit.state[0] - 2.0).abs() < 1e-12);
    assert!((fit.state[1] - 3.0).abs() < 1e-12);
    assert!(fit.rmse < 1e-12);
}
