#![cfg(all(feature = "gpu", not(coverage)))]
//! Independent counterexamples against the frozen source; expected failures are retained.
use mlsirm_core::{
    two_tier_gpu_score::expected_raw_g_only,
    two_tier_recursion::{two_tier_expected_raw_at_q, TwoTierItemParams},
    Device,
};

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn finite_f32_inputs_must_not_silently_lose_exact_cancellation() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let p = TwoTierItemParams {
        a_primary: vec![3e38, -3e38],
        a_specific: vec![0.],
        thresholds: vec![0.],
        specific_map: vec![-1],
        n_primary: 2,
        n_specific: 0,
        n_cat: 2,
    };
    // q=1 is at W=2, S=0. The native f64 predictor is 3e38*2 - 3e38*2 = 0.
    let reference = two_tier_expected_raw_at_q(&p, &[2., 2.], 1).unwrap()[0];
    assert_eq!(reference, 0.5);
    let actual = expected_raw_g_only(&p, &[2.], 2., 1., &[], 0., 1, 1, Device::Gpu);
    println!("FINITE_CANCELLATION native={reference} gpu={actual:?}");
    // Rejection is safe; a finite accepted answer must satisfy the known native value.
    if let Ok(r) = actual {
        assert!((r.scores[0] - reference).abs() < 2e-5);
    }
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn constant_binary_bank_retains_product_weight_mass_at_121_and_241() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let p = TwoTierItemParams {
        a_primary: vec![0.; 48],
        a_specific: vec![0.; 24],
        thresholds: vec![0.; 24],
        specific_map: vec![0; 24],
        n_primary: 2,
        n_specific: 1,
        n_cat: 2,
    };
    let mut accepted = true;
    for q in [15, 121, 241] {
        let native = two_tier_expected_raw_at_q(&p, &[0., 0.], q).unwrap()[0];
        assert!((native - 12.).abs() < 1e-11);
        let r = expected_raw_g_only(&p, &[0.], 0., 1., &[1.], 0., q, q, Device::Gpu).unwrap();
        let error = (r.scores[0] - native).abs();
        println!(
            "CONSTANT_BINARY q_w={q} q_s={q} native={native:.17} gpu={:.17} error={error:.17}",
            r.scores[0]
        );
        // The original module's 2e-5 fixture tolerance is reused, not a scientific cutoff.
        accepted &= error < 2e-5;
    }
    assert!(
        accepted,
        "constant-score normalization exceeds original 2e-5 fixture tolerance"
    );
}
