//! Regression coverage for AISP significance thresholds above alpha = 0.5.
//!
//! `mokken::search.normal` uses `abs(qnorm(adjusted.alpha(...)))` for its
//! critical Z threshold. The Rust upper-tail helper is signed when p > 0.5,
//! so AISP must take the magnitude before comparing observed Z statistics.

use mlsirm_core::mokken::aisp;

#[test]
fn aisp_preserves_positive_critical_z_for_high_but_admitted_alpha() {
    // Two nonconstant binary items with exactly zero covariance. With two
    // free items the first adjusted alpha equals the nominal alpha. At 0.9,
    // search.normal therefore uses |qnorm(0.9)| ~= 1.2816 and rejects this
    // pair (Zij = 0). A signed upper-tail quantile would instead be negative
    // and make the significance filter vacuous.
    let x = vec![
        0, 0,
        0, 1,
        1, 0,
        1, 1,
    ];

    let labels = aisp(&x, 4, 2, 0.0, 0.9)
        .expect("AISP should accept finite binary scores and alpha in (0, 1)");

    assert_eq!(
        labels,
        vec![0, 0],
        "zero-association items must remain unscaled when the absolute qnorm threshold is positive",
    );
}
