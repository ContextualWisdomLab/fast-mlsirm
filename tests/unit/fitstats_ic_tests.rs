use super::*;

#[test]
fn information_criteria_reference_values() {
    let ic = information_criteria(-500.0, 10, 200);
    assert!((ic.aic - 1020.0).abs() < 1e-12);
    assert!((ic.bic - (1000.0 + 10.0 * (200.0_f64).ln())).abs() < 1e-12);
    assert!((ic.caic - (1000.0 + 10.0 * ((200.0_f64).ln() + 1.0))).abs() < 1e-12);
    assert!((ic.aicc - (1020.0 + 220.0 / 189.0)).abs() < 1e-9);
    assert!((ic.sabic - (1000.0 + 10.0 * (202.0_f64 / 24.0).ln())).abs() < 1e-9);
    // degenerate n does not panic
    let tiny = information_criteria(-5.0, 10, 10);
    assert!(tiny.aicc.is_nan());
}
