use super::*;
use crate::quadrature::gh_rule;

#[test]
fn two_tier_lord_wingersky_matches_direct_enumeration_on_small_grid() {
    let n_items = 4usize;
    let n_cat = 3usize;
    let n_primary = 2usize;
    let n_specific = 2usize;
    let specific_map = vec![0i32, 0, 1, 1];
    let a_primary = vec![
        1.0, 0.0, // item 0: G only
        1.1, 0.2, // item 1: G + W
        0.9, 0.0, // item 2
        0.8, 0.3, // item 3
    ];
    let a_specific = vec![0.7, 0.6, 0.5, 0.4];
    let thresholds = vec![0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2];

    let params = TwoTierItemParams {
        a_primary,
        a_specific,
        thresholds,
        specific_map,
        n_primary,
        n_specific,
        n_cat,
    };

    let theta_primary = vec![0.25, -0.1, -0.5, 0.3];
    let (gh_nodes, gh_weights) = gh_rule(15).expect("gh_rule(15)");

    let lw = two_tier_lord_wingersky(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("two_tier_lord_wingersky");
    let direct = direct_enumeration_two_tier(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("direct_enumeration_two_tier");
    assert_eq!(lw.len(), direct.len());

    let total_max = params.total_max_score();
    for person in 0..2 {
        let row = person * (total_max + 1);
        let sum_lw: f64 = lw[row..row + total_max + 1].iter().sum();
        let sum_dir: f64 = direct[row..row + total_max + 1].iter().sum();
        assert!(
            (sum_lw - 1.0).abs() <= 1e-10,
            "person {person} lw sum={sum_lw}"
        );
        assert!(
            (sum_dir - 1.0).abs() <= 1e-10,
            "person {person} direct sum={sum_dir}"
        );
        for r in 0..=total_max {
            assert!(
                (lw[row + r] - direct[row + r]).abs() <= 1e-12,
                "person {person} score {r}: lw={} direct={}",
                lw[row + r],
                direct[row + r]
            );
        }
    }

    let expected = two_tier_expected_raw(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("two_tier_expected_raw");
    assert_eq!(expected.len(), 2);
    for person in 0..2 {
        let row = person * (total_max + 1);
        let mut mean = 0.0_f64;
        for r in 0..=total_max {
            mean += (r as f64) * lw[row + r];
        }
        assert!((expected[person] - mean).abs() <= 1e-12);
    }
}

fn degenerate_params(n_primary: usize, n_specific: usize, n_cat: usize) -> TwoTierItemParams {
    let m1 = n_cat.saturating_sub(1);
    TwoTierItemParams {
        a_primary: vec![1.0; n_primary],
        a_specific: vec![0.5],
        thresholds: vec![0.0; m1],
        specific_map: vec![0],
        n_primary,
        n_specific,
        n_cat,
    }
}

#[test]
fn degenerate_dimensions_return_err_instead_of_panicking() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule(5)");
    let cases = [
        (degenerate_params(1, 1, 0), vec![0.0]),
        (degenerate_params(1, 1, 1), vec![0.0]),
        (degenerate_params(0, 1, 3), vec![]),
        (degenerate_params(1, 0, 3), vec![0.0]),
    ];
    for (params, theta) in cases.iter() {
        assert!(two_tier_lord_wingersky(params, theta, nodes, weights).is_err());
        assert!(two_tier_expected_raw(params, theta, nodes, weights).is_err());
        assert!(direct_enumeration_two_tier(params, theta, nodes, weights).is_err());
    }
}

#[test]
fn specific_map_below_minus_one_is_rejected_not_treated_as_specific_free() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule(5)");
    let mut params = degenerate_params(1, 1, 2);
    params.specific_map = vec![-2];
    assert!(two_tier_expected_raw(&params, &[0.0], nodes, weights).is_err());
}

#[test]
fn direct_enumeration_rejects_pattern_count_overflow() {
    let n_cat = 65_536usize;
    let params = TwoTierItemParams {
        a_primary: vec![1.0; 4],
        a_specific: vec![0.5; 4],
        thresholds: vec![0.0; 4 * (n_cat - 1)],
        specific_map: vec![0; 4],
        n_primary: 1,
        n_specific: 1,
        n_cat,
    };
    assert!(direct_enumeration_two_tier(&params, &[0.0], &[0.0], &[1.0]).is_err());
}

#[test]
fn empty_quadrature_and_non_finite_primary_return_err() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule(5)");
    let params = degenerate_params(1, 1, 2);
    assert!(two_tier_expected_raw(&params, &[0.0], &[], &[]).is_err());
    assert!(two_tier_expected_raw(&params, &[f64::NAN], nodes, weights).is_err());
    assert!(two_tier_expected_raw(&params, &[f64::INFINITY], nodes, weights).is_err());
}

#[test]
fn non_decreasing_or_non_finite_thresholds_return_err() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule(5)");
    for bad in [
        vec![0.0, 0.5],
        vec![0.5, 0.5],
        vec![f64::NAN, -1.0],
        vec![1.0, f64::NAN],
    ] {
        let mut params = degenerate_params(1, 1, 3);
        params.thresholds = bad.clone();
        assert!(
            two_tier_expected_raw(&params, &[0.0], nodes, weights).is_err(),
            "thresholds {bad:?} must be rejected"
        );
    }
    let mut binary = degenerate_params(1, 1, 2);
    binary.thresholds = vec![f64::INFINITY];
    assert!(two_tier_expected_raw(&binary, &[0.0], nodes, weights).is_err());
    let mut ok = degenerate_params(1, 1, 3);
    ok.thresholds = vec![0.5, -0.5];
    assert!(two_tier_expected_raw(&ok, &[0.0], nodes, weights).is_ok());
}

#[test]
fn non_finite_slopes_return_err() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule(5)");
    let mut primary = degenerate_params(1, 1, 2);
    primary.a_primary = vec![f64::NAN];
    assert!(two_tier_expected_raw(&primary, &[0.0], nodes, weights).is_err());
    let mut specific = degenerate_params(1, 1, 2);
    specific.a_specific = vec![f64::INFINITY];
    assert!(two_tier_expected_raw(&specific, &[0.0], nodes, weights).is_err());
}

fn mixed_params(
    n_items: usize,
    n_cat: usize,
    n_primary: usize,
    n_specific: usize,
    seed: u64,
) -> TwoTierItemParams {
    // Deterministic LCG so the fixture needs no rand dependency.
    let mut state = seed;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let m1 = n_cat - 1;
    let a_primary = (0..n_items * n_primary)
        .map(|_| 0.3 + 1.5 * next())
        .collect();
    let a_specific = (0..n_items).map(|_| 0.2 + 1.2 * next()).collect();
    let mut thresholds = Vec::with_capacity(n_items * m1);
    for _ in 0..n_items {
        let mut t = 1.5 + next();
        for _ in 0..m1 {
            thresholds.push(t);
            t -= 0.4 + next();
        }
    }
    // Last item of every third is specific-free to exercise the -1 block.
    let specific_map = (0..n_items)
        .map(|i| {
            if i % 3 == 2 {
                -1
            } else {
                (i % n_specific) as i32
            }
        })
        .collect();
    TwoTierItemParams {
        a_primary,
        a_specific,
        thresholds,
        specific_map,
        n_primary,
        n_specific,
        n_cat,
    }
}

#[test]
fn closed_form_expected_raw_equals_lord_wingersky_mean() {
    for (q, seed) in [(5usize, 1u64), (15, 7), (21, 42)] {
        let (nodes, weights) = gh_rule(q).expect("gh_rule");
        let params = mixed_params(9, 4, 2, 2, seed);
        let theta = vec![0.0, 0.0, -1.3, 0.7, 2.1, -0.4, 0.25, -2.0];
        let closed = two_tier_expected_raw(&params, &theta, nodes, weights).expect("closed");
        let lw = two_tier_expected_raw_lw(&params, &theta, nodes, weights).expect("lw");
        assert_eq!(closed.len(), 4);
        for (c, l) in closed.iter().zip(lw.iter()) {
            assert!((c - l).abs() <= 1e-12, "closed={c} lw={l}");
        }
    }
}

/// Derived f32 bound (design memo on #2271): per row, the closed form sums
/// S = I*(n_cat-1)*Q terms in [0, 1]; with unit roundoff u = 2^-24, sigmoid
/// error of a few u per term and compensated accumulation across items, the
/// absolute error is bounded by (8 + S) * u * max_score.
fn f32_bound(params: &TwoTierItemParams, q: usize) -> f64 {
    let s = (params.n_items() * (params.n_cat - 1) * q) as f64;
    (8.0 + s) * f64::from(f32::EPSILON / 2.0) * params.total_max_score() as f64
}

#[test]
fn device_dispatch_cpu_is_exact_and_gpu_within_derived_bound() {
    let q = 21usize;
    let (nodes, weights) = gh_rule(q).expect("gh_rule");
    let params = mixed_params(30, 4, 2, 4, 2271);
    // 2_000 rows exercise more than one workgroup; values span the trait range.
    let theta: Vec<f64> = (0..4_000)
        .map(|k| ((k * 37) % 101) as f64 / 20.0 - 2.5)
        .collect();
    let reference = two_tier_expected_raw(&params, &theta, nodes, weights).expect("cpu");
    let (cpu, used) = two_tier_expected_raw_on(&params, &theta, nodes, weights, crate::Device::Cpu)
        .expect("cpu dispatch");
    assert!(!used);
    assert_eq!(cpu, reference);
    let (auto, used_gpu) =
        two_tier_expected_raw_on(&params, &theta, nodes, weights, crate::Device::Auto)
            .expect("auto");
    let bound = f32_bound(&params, q);
    let max_err = auto
        .iter()
        .zip(&reference)
        .map(|(a, r)| (a - r).abs())
        .fold(0.0_f64, f64::max);
    eprintln!("two-tier expected raw device=auto used_gpu={used_gpu} max_abs_err={max_err:e} bound={bound:e}");
    assert!(max_err <= bound, "max_err={max_err} bound={bound}");
    if !used_gpu {
        assert_eq!(auto, reference, "CPU fallback must be the f64 owner");
    }
}

#[test]
fn device_dispatch_validates_before_any_device() {
    let (nodes, weights) = gh_rule(5).expect("gh_rule");
    let mut params = degenerate_params(1, 1, 3);
    params.thresholds = vec![0.0, 0.5];
    for device in [crate::Device::Cpu, crate::Device::Gpu, crate::Device::Auto] {
        assert!(two_tier_expected_raw_on(&params, &[0.0], nodes, weights, device).is_err());
    }
}
