#![cfg(all(feature = "gpu", not(coverage)))]
//! Independent finite fixtures; no study data and no estimator execution.
use mlsirm_core::{
    two_tier_gpu_score::expected_raw_g_only as score,
    two_tier_recursion::{two_tier_expected_raw, TwoTierItemParams},
    Device,
};

fn bank() -> TwoTierItemParams {
    TwoTierItemParams {
        a_primary: vec![0.7, 1.2, 1.1, -0.6],
        a_specific: vec![0.4, 0.],
        thresholds: vec![0.8, -0.5, 0.4, -0.9],
        specific_map: vec![0, -1],
        n_primary: 2,
        n_specific: 1,
        n_cat: 3,
    }
}
fn gpu(
    p: &TwoTierItemParams,
    g: &[f64],
    mean: f64,
    sd: f64,
    ss: &[f64],
    qw: usize,
    qs: usize,
) -> mlsirm_core::two_tier_gpu_score::ExpectedRawGpu {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let r = score(p, g, mean, sd, ss, 0., qw, qs, Device::Gpu).unwrap();
    assert_eq!(r.dispatches, 1);
    assert_eq!(r.cpu_numerical_fallbacks, 0);
    assert!(!r.backend.is_empty() && !r.device.is_empty());
    assert_eq!((r.q_wording, r.q_specific), (qw, qs));
    r
}
fn close(actual: f64, expected: f64) {
    assert!((actual - expected).abs() < 2e-5, "{actual} vs {expected}");
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn native_invalid_input_controls() {
    let p = bank();
    let invalid = |p: &TwoTierItemParams,
                   g: &[f64],
                   m: f64,
                   sd: f64,
                   ss: &[f64],
                   cov: f64,
                   qw: usize,
                   qs: usize| {
        assert!(score(p, g, m, sd, ss, cov, qw, qs, Device::Gpu).is_err());
    };
    for d in [Device::Cpu, Device::Auto] {
        assert!(score(&p, &[0.], 0., 1., &[1.], 0., 15, 15, d).is_err());
    }
    for c in [0.1, -0.1, f64::NAN, f64::INFINITY] {
        invalid(&p, &[0.], 0., 1., &[1.], c, 15, 15);
    }
    for v in [f64::NAN, f64::INFINITY, 1e300] {
        invalid(&p, &[v], 0., 1., &[1.], 0., 15, 15);
        invalid(&p, &[0.], v, 1., &[1.], 0., 15, 15);
    }
    for v in [0., -1., 1e-300, f64::NAN, f64::INFINITY] {
        invalid(&p, &[0.], 0., v, &[1.], 0., 15, 15);
        invalid(&p, &[0.], 0., 1., &[v], 0., 15, 15);
    }
    invalid(&p, &[], 0., 1., &[1.], 0., 15, 15);
    invalid(&p, &[0.], 0., 1., &[], 0., 15, 15);
    invalid(&p, &[0.], 0., 1., &[1.], 0., 0, 15);
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 0);
    let mut p = bank();
    p.n_primary = 1;
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    let mut p = bank();
    p.a_primary.pop();
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    let mut p = bank();
    p.thresholds.pop();
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    let mut p = bank();
    p.specific_map.pop();
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    for map in [-2, 1] {
        let mut p = bank();
        p.specific_map[0] = map;
        invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    }
    let mut p = bank();
    p.specific_map[0] = -1;
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    let mut p = bank();
    p.thresholds[1] = p.thresholds[0];
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    let mut p = bank();
    p.thresholds[1] = p.thresholds[0] - 1e-12;
    invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    for v in [f64::NAN, f64::INFINITY, 1e300] {
        let mut p = bank();
        p.a_primary[0] = v;
        invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
        let mut p = bank();
        p.a_specific[0] = v;
        invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
        let mut p = bank();
        p.thresholds[0] = v;
        invalid(&p, &[0.], 0., 1., &[1.], 0., 15, 15);
    }
    println!("INVALID_CONTROLS all returned Err; no malformed fixture dispatched");
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn single_node_uses_locations_and_specific_free_coordinate() {
    let p = bank();
    let g = [-0.8, 0., 0.9];
    let actual = gpu(&p, &g, 0.3, 1.2, &[0.8], 1, 1);
    let coords: Vec<f64> = g.iter().flat_map(|&x| [x, 0.3]).collect();
    let expected = two_tier_expected_raw(&p, &coords, &[0.], &[1.]).unwrap();
    for (&a, &b) in actual.scores.iter().zip(&expected) {
        close(a, b);
    }
    println!("SINGLE_NODE {:?}", actual);
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn binary_symmetry_checks_product_weight_normalization_and_f32_scale() {
    let p = TwoTierItemParams {
        a_primary: vec![0., 1.3, 0., -0.8, 0., 0.4],
        a_specific: vec![0.7, -0.5, 0.],
        thresholds: vec![0.; 3],
        specific_map: vec![0, 1, -1],
        n_primary: 2,
        n_specific: 2,
        n_cat: 2,
    };
    for (qw, qs) in [(1, 1), (7, 11), (15, 15), (121, 241)] {
        let r = gpu(&p, &[-1., 0., 2.], 0., 1.6, &[0.4, 1.7], qw, qs);
        for &s in &r.scores {
            close(s, 1.5);
        }
        println!("SYMMETRY {qw} {qs} {:?}", r);
    }
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn known_nonzero_location_uses_mean_plus_sd_times_standard_normal() {
    let p = TwoTierItemParams {
        a_primary: vec![0., 2.],
        a_specific: vec![0.],
        thresholds: vec![0.],
        specific_map: vec![-1],
        n_primary: 2,
        n_specific: 0,
        n_cat: 2,
    };
    let r = gpu(&p, &[0., 2.], 0.5, 1.7, &[], 1, 11);
    // eta=2*0.5=1 with a one-node W rule; these calls exercise native scoring.
    let e = two_tier_expected_raw(&p, &[0., 0.5, 2., 0.5], &[0.], &[1.]).unwrap();
    for (&a, &b) in r.scores.iter().zip(&e) {
        close(a, b);
    }
    println!("LOCATION {:?}", r);
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn multiple_specific_sds_permutation_and_linearity() {
    let p = TwoTierItemParams {
        a_primary: vec![0.5, 0.9, -0.3, 1.1, 1.2, -0.7],
        a_specific: vec![0.8, -0.6, 0.],
        thresholds: vec![0.9, -0.4, 0.5, -1.2, 0.8, -0.3],
        specific_map: vec![1, 0, -1],
        n_primary: 2,
        n_specific: 2,
        n_cat: 3,
    };
    let g = [0.2, -0.8, 0.2, 1.1];
    let r = gpu(&p, &g, 0.4, 1.3, &[0.3, 1.6], 15, 15);
    // Each item gets its mapped SD; match one-specific-scale LW using exact slope rescaling.
    let mut scaled = p.clone();
    scaled.a_specific[0] *= 1.6;
    scaled.a_specific[1] *= 0.3;
    // 15-node reference table from the unchanged native module, not an external package.
    let n = [
        -6.363947888829839,
        -5.190093591304781,
        -4.1962077112690155,
        -3.2890824243987664,
        -2.432436827009758,
        -1.6067100690287297,
        -0.799129068324548,
        0.,
        0.799129068324548,
        1.6067100690287297,
        2.432436827009758,
        3.2890824243987664,
        4.1962077112690155,
        5.190093591304781,
        6.363947888829839,
    ];
    let w = [
        8.589649899633252e-10,
        5.975419597920599e-7,
        5.642146405189029e-5,
        0.001567357503549956,
        0.017365774492137616,
        0.08941779539984437,
        0.23246229360973225,
        0.31825951825951815,
        0.23246229360973225,
        0.08941779539984437,
        0.017365774492137616,
        0.001567357503549956,
        5.642146405189029e-5,
        5.975419597920599e-7,
        8.589649899633252e-10,
    ];
    for (j, &gj) in g.iter().enumerate() {
        let coords: Vec<f64> = n.iter().flat_map(|x| [gj, 0.4 + 1.3 * x]).collect();
        let lw = two_tier_expected_raw(&scaled, &coords, &n, &w).unwrap();
        let e: f64 = lw.iter().zip(w).map(|(m, v)| m * v).sum();
        close(r.scores[j], e);
    }
    assert_eq!(r.scores[0], r.scores[2]);
    let perm = gpu(&p, &[1.1, 0.2, -0.8, 0.2], 0.4, 1.3, &[0.3, 1.6], 15, 15);
    assert_eq!(
        perm.scores,
        vec![r.scores[3], r.scores[0], r.scores[1], r.scores[2]]
    );
    let mut grouped = p.clone();
    grouped.specific_map = vec![0, 0, -1];
    // Total means depend on item marginal scale, not common-vs-separate block membership.
    grouped.a_specific[0] *= 1.6;
    grouped.a_specific[1] *= 0.3;
    grouped.n_specific = 1;
    let linear = gpu(&grouped, &g, 0.4, 1.3, &[1.], 15, 15);
    for (&a, &b) in r.scores.iter().zip(&linear.scores) {
        close(a, b);
    }
    println!("MULTI_SPECIFIC {:?}", r);
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn independent_nuisance_sd_changes_have_native_effects() {
    let mut p = bank();
    p.a_primary[1] = 1.8;
    p.a_specific[0] = 1.4;
    let a = gpu(&p, &[-0.8], 0.3, 0.2, &[0.3], 15, 15);
    let w = gpu(&p, &[-0.8], 0.3, 1.8, &[0.3], 15, 15);
    let s = gpu(&p, &[-0.8], 0.3, 0.2, &[1.8], 15, 15);
    assert!((a.scores[0] - w.scores[0]).abs() > 1e-3);
    assert!((a.scores[0] - s.scores[0]).abs() > 1e-3);
    println!("NUISANCE_SDS {:?} {:?} {:?}", a.scores, w.scores, s.scores);
}

#[test]
#[ignore = "requires actual GPU and common host lease"]
fn finite_extreme_controls_preserve_result_contract() {
    let p = TwoTierItemParams {
        a_primary: vec![0., 0.],
        a_specific: vec![0.],
        thresholds: vec![80.],
        specific_map: vec![-1],
        n_primary: 2,
        n_specific: 0,
        n_cat: 2,
    };
    let r = gpu(&p, &[0.], 0., 1., &[], 121, 241);
    close(r.scores[0], 1.);
    let mut low = p.clone();
    low.thresholds[0] = -80.;
    let r = gpu(&low, &[0.], 0., 1., &[], 121, 241);
    close(r.scores[0], 0.);
    // Individually representable finite inputs can still overflow eta on device.
    // This is characterization, not an invented guarantee on unsupported extremes.
    let mut edge = p.clone();
    edge.a_primary = vec![3e38, -3e38];
    edge.thresholds[0] = 0.;
    let r = score(&edge, &[2.], 2., 1., &[], 0., 1, 1, Device::Gpu);
    println!("FINITE_OVERFLOW_CHARACTERIZATION {r:?}");
}
