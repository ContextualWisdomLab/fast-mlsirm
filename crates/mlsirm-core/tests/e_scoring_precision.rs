//! Precision/range regressions for the native strict G-only leaf. Synthetic banks only.
#![cfg(all(feature = "gpu", not(coverage)))]
use mlsirm_core::{
    two_tier_gpu_score::{expected_raw_g_only, ExpectedRawGpu},
    two_tier_recursion::{two_tier_expected_raw, two_tier_expected_raw_at_q, TwoTierItemParams},
    Device,
};
fn binary(a_g: f64, a_w: f64, intercept: f64) -> TwoTierItemParams {
    TwoTierItemParams {
        a_primary: vec![a_g, a_w],
        a_specific: vec![0.],
        thresholds: vec![intercept],
        specific_map: vec![-1],
        n_primary: 2,
        n_specific: 0,
        n_cat: 2,
    }
}
fn receipt(r: &ExpectedRawGpu) {
    assert_eq!(r.dispatches, 1);
    assert_eq!(r.cpu_numerical_fallbacks, 0);
    assert!(!r.backend.is_empty());
    assert!(!r.device.is_empty());
}
#[test]
#[ignore = "requires actual GPU and common host lease"]
fn refuses_unsafe_predictor_stages_with_specific_error() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    // Product overflow hidden by cancellation, transform overflow even with zero slopes,
    // affine-sum overflow, boundary-add overflow and a mapped-specific product overflow.
    let mut specific_transform = binary(0., 0., 0.);
    specific_transform.n_specific = 1;
    specific_transform.specific_map[0] = 0;
    let mut specific_product = specific_transform.clone();
    specific_product.a_specific[0] = 3e38;
    let cases = [
        (
            "primary_product",
            binary(3e38, -3e38, 0.),
            2.,
            2.,
            1.,
            vec![],
            1,
            1,
        ),
        (
            "wording_transform",
            binary(0., 0., 0.),
            0.,
            3e38,
            3e38,
            vec![],
            3,
            1,
        ),
        (
            "specific_transform",
            specific_transform,
            0.,
            0.,
            1.,
            vec![3e38],
            1,
            3,
        ),
        (
            "affine_sum",
            binary(3e38, 3e38, 0.),
            1.,
            1.,
            1.,
            vec![],
            1,
            1,
        ),
        (
            "boundary_add",
            binary(2e38, 0., 2e38),
            1.,
            0.,
            1.,
            vec![],
            1,
            1,
        ),
        (
            "specific_product",
            specific_product,
            0.,
            0.,
            1.,
            vec![2.],
            1,
            3,
        ),
    ];
    for (name, p, g, mean, sd, ss, qw, qs) in cases {
        let r = expected_raw_g_only(&p, &[g], mean, sd, &ss, 0., qw, qs, Device::Gpu);
        println!("RANGE_REFUSAL {name} {r:?}");
        let err =
            r.expect_err("unsafe intermediate must be refused, not saturated to a valid score");
        assert!(err.contains("unsafe f32 predictor range"), "{name}: {err}");
    }
}
#[test]
#[ignore = "requires actual GPU and common host lease"]
fn safe_large_cancellation_and_saturation_match_native_known_values() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    for (name, p, g, mean, known) in [
        (
            "large_exact_cancellation",
            binary(1e37, -1e37, 0.),
            2.,
            2.,
            0.5,
        ),
        ("positive_saturation", binary(40., 0., 0.), 2., 0., 1.),
        ("negative_saturation", binary(-40., 0., 0.), 2., 0., 0.),
    ] {
        let native = two_tier_expected_raw_at_q(&p, &[g, mean], 1).unwrap()[0];
        assert!((native - known).abs() < 1e-30);
        let r = expected_raw_g_only(&p, &[g], mean, 1., &[], 0., 1, 1, Device::Gpu).unwrap();
        receipt(&r);
        assert!(
            (r.scores[0] - native).abs() < 2e-5,
            "{name}: {r:?} vs {native}"
        );
        println!("SAFE_RANGE {name} native={native:.17} {r:?}");
    }
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn refuses_overflowing_reassociated_products_even_at_zero_nodes() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let mut specific = binary(0., 0., 0.);
    specific.n_specific = 1;
    specific.specific_map[0] = 0;
    specific.a_specific[0] = 1e38;
    for (name, p, sd, ss) in [
        ("wording_slope_times_sd", binary(0., 1e38, 0.), 10., vec![]),
        ("specific_slope_times_sd", specific, 1., vec![10.]),
    ] {
        // With one GH node at zero the mathematical contribution vanishes.
        // Reject a nonfinite intermediate in a backend's multiply reassociation.
        let r = expected_raw_g_only(&p, &[0.], 0., sd, &ss, 0., 1, 1, Device::Gpu);
        println!("REASSOCIATION_REFUSAL {name} {r:?}");
        assert!(r
            .expect_err("unsafe reassociated product must be refused")
            .contains("unsafe f32 predictor range"));
    }
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn large_nonconstant_bank_matches_native_lw_at_independent_counts() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let p = TwoTierItemParams {
        a_primary: (0..24)
            .flat_map(|i| {
                [
                    0.2 + (i % 5) as f64 * 0.2,
                    if i % 2 == 0 { 0.8 } else { -0.5 },
                ]
            })
            .collect(),
        a_specific: (0..24).map(|i| if i % 3 == 0 { 0. } else { 0.6 }).collect(),
        thresholds: (0..24)
            .flat_map(|i| [0.7 + (i % 3) as f64 * 0.1, -0.5])
            .collect(),
        specific_map: (0..24).map(|i| if i % 3 == 0 { -1 } else { 0 }).collect(),
        n_primary: 2,
        n_specific: 1,
        n_cat: 3,
    };
    let g = [-0.8, 0., 0.9];
    for (qw, qs) in [(15, 15), (121, 241), (241, 121), (241, 241)] {
        // For each W node use the unchanged native LW distribution/mean, with
        // S nodes scaled in the original coordinate. Only aggregate those native
        // returns with the library's probability GH weights (no external kernel).
        let (wn, ww) = quadrature::gh_rule(qw).unwrap();
        let (sn, sw) = quadrature::gh_rule(qs).unwrap();
        let sn: Vec<_> = sn.iter().map(|x| 0.8 * x).collect();
        let r = expected_raw_g_only(&p, &g, 0.3, 1.2, &[0.8], 0., qw, qs, Device::Gpu).unwrap();
        receipt(&r);
        for (j, &gj) in g.iter().enumerate() {
            let coords: Vec<_> = wn.iter().flat_map(|x| [gj, 0.3 + 1.2 * x]).collect();
            let means = two_tier_expected_raw(&p, &coords, &sn, sw).unwrap();
            let native: f64 = means.iter().zip(ww).map(|(m, w)| m * w).sum();
            let error = (r.scores[j] - native).abs();
            println!("LARGE_NATIVE_LW qw={qw} qs={qs} g={gj} native={native:.17} gpu={:.17} error={error:.17}", r.scores[j]);
            assert!(error < 2e-5);
        }
    }
}
// The quadrature module is crate-private. Include the unchanged actual library
// source for this oracle only; its hash is bound with the full scorer packet.
#[path = "../src/quadrature.rs"]
mod quadrature;
