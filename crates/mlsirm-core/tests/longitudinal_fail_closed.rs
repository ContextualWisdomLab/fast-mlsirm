use mlsirm_core::longitudinal::fit_longitudinal_state;

#[test]
fn rejects_accumulated_ar_gap_beyond_i32_exponent_range() {
    let max_gap = i32::MAX as usize;
    let error = fit_longitudinal_state(
        &[0, 3],
        &[0, max_gap, max_gap + 1],
        &[0, 1, 2],
        &[1.0, f64::NAN, 0.25],
        "stationary_autoregressive",
        Some(0.5),
        1,
    )
    .unwrap_err();
    assert!(error.contains("accumulated AR occasion gap"), "{error}");
}
