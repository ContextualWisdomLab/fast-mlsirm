//! Fixed-parameter expected-score accumulation regression, not fitted recovery.
//!
//! Conditional-on-primary scoring follows Cai (2015, pp. 542-543, Eqs. 14-17).
//! Summation error depends on accumulator precision and addition order
//! (Higham, 1993, pp. 785-786, Eqs. 2.6-2.8). Same-measure representations
//! below are not quadrature refinements or a universal accuracy guarantee.
//!
//! References (APA 7th ed.):
//! Cai, L. (2015). Lord-Wingersky algorithm version 2.0 for hierarchical item
//! factor models with applications in test scoring, scale alignment, and model
//! fit testing. Psychometrika, 80(2), 535-559. doi:10.1007/s11336-014-9411-3.
//! Higham, N. J. (1993). The accuracy of floating point summation. SIAM Journal
//! on Scientific Computing, 14(4), 783-799. doi:10.1137/0914050.

use mlsirm_core::{
    two_tier_recursion::{two_tier_expected_raw_on, TwoTierItemParams},
    Device,
};
use serde_json::Value;

fn numbers(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("fixture array")
        .iter()
        .map(|v| v.as_f64().expect("finite fixture number"))
        .collect()
}

fn check_representation(representation: &str) {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/two_tier_gpu_accumulation.json"))
            .expect("retained regression fixture");
    let original_nodes = numbers(&fixture["nodes"]);
    let original_weights = numbers(&fixture["weights"]);
    assert_eq!(original_nodes.len(), 3072);
    assert_eq!(original_weights.len(), original_nodes.len());
    let mut pairs: Vec<(f64, f64)> = original_nodes.into_iter().zip(original_weights).collect();
    match representation {
        "original" => {}
        "reverse" => pairs.reverse(),
        "weight_descending" => pairs.sort_by(|a, b| b.1.total_cmp(&a.1)),
        "split_each_weight" => {
            let split: Vec<_> = pairs
                .iter()
                .flat_map(|&(n, w)| [(n, w / 2.0), (n, w / 2.0)])
                .collect();
            for (pair, halves) in pairs.iter().zip(split.chunks_exact(2)) {
                assert_eq!(halves[0].0, pair.0);
                assert_eq!(halves[1].0, pair.0);
                assert_eq!(halves[0].1 + halves[1].1, pair.1);
            }
            pairs = split;
        }
        "zero_weight_tail" => pairs.push((0.0, 0.0)),
        _ => panic!("unknown representation"),
    }
    let (nodes, weights): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    assert!(nodes.iter().all(|n| n.is_finite()));
    assert!(weights.iter().all(|w| w.is_finite() && *w >= 0.0));
    assert!((weights.iter().sum::<f64>() - 1.0).abs() <= 1e-10);
    let theta = numbers(&fixture["theta"]);
    for f in fixture["fixtures"].as_array().expect("fixtures") {
        let name = f["name"].as_str().expect("name");
        let params = TwoTierItemParams {
            a_primary: numbers(&f["a_primary"]),
            a_specific: numbers(&f["a_specific"]),
            thresholds: numbers(&f["thresholds"]),
            specific_map: vec![0, 0, 1, 1, 2, 2, 3, 3],
            n_primary: 2,
            n_specific: 4,
            n_cat: 4,
        };
        let reference = numbers(&f["reference_means"]);
        let (cpu, cpu_used) =
            two_tier_expected_raw_on(&params, &theta, &nodes, &weights, Device::Cpu)
                .expect("CPU control");
        let (actual, used_gpu) =
            two_tier_expected_raw_on(&params, &theta, &nodes, &weights, Device::Auto)
                .expect("automatic scoring");
        let (after, after_used) =
            two_tier_expected_raw_on(&params, &theta, &nodes, &weights, Device::Cpu)
                .expect("CPU after GPU");
        assert!(!cpu_used && !after_used);
        assert_eq!(after, cpu);
        assert_eq!(reference.len(), 7);
        assert_eq!(actual.len(), reference.len());
        assert!(actual.iter().all(|v| v.is_finite()));
        let cpu_error = cpu
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            cpu_error <= 1e-10,
            "{representation}/{name}: CPU reference error {cpu_error}"
        );
        if std::env::var_os("FAST_MLSIRM_REQUIRE_SCORE_GPU").is_some() {
            assert!(
                used_gpu,
                "hardware validation requires actual GPU contributions"
            );
        }
        if !used_gpu {
            assert_eq!(actual, cpu, "fallback must remain exact CPU");
        }
        let error = actual
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("accumulation representation={representation} fixture={name} used_gpu={used_gpu} max_absolute_error={error:e}");
        assert!(
            error <= 1e-5,
            "{representation}/{name}: actual GPU error {error} exceeds fixed 1e-5"
        );
    }
}

#[test]
fn long_rule_original() {
    check_representation("original");
}
#[test]
fn long_rule_reverse() {
    check_representation("reverse");
}
#[test]
fn long_rule_weight_descending() {
    check_representation("weight_descending");
}
#[test]
fn long_rule_split_each_weight() {
    check_representation("split_each_weight");
}
#[test]
fn long_rule_zero_weight_tail() {
    check_representation("zero_weight_tail");
}
