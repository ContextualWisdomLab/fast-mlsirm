//! Integration contract for Rust-native bifactor scoreability indices.

use mlsirm_core::bifactor_indices::{
    bifactor_indices, BifactorIndicesConfig, BifactorIndicesResult,
};

fn example_loadings() -> Vec<f64> {
    vec![
        0.82, 0.10, 0.00, 0.00, // item 1
        0.77, 0.35, 0.00, 0.00, // item 2
        0.79, 0.32, 0.00, 0.00, // item 3
        0.66, 0.39, 0.00, 0.00, // item 4
        0.51, 0.00, 0.71, 0.00, // item 5
        0.56, 0.00, 0.43, 0.00, // item 6
        0.68, 0.00, 0.13, 0.00, // item 7
        0.60, 0.00, 0.50, 0.00, // item 8
        0.83, 0.00, 0.00, 0.47, // item 9
        0.60, 0.00, 0.00, 0.27, // item 10
        0.78, 0.00, 0.00, 0.28, // item 11
        0.55, 0.00, 0.00, 0.75, // item 12
    ]
}

fn uniquenesses(loadings: &[f64], n_items: usize, n_factors: usize) -> Vec<f64> {
    (0..n_items)
        .map(|item| {
            1.0 - (0..n_factors)
                .map(|factor| loadings[item * n_factors + factor].powi(2))
                .sum::<f64>()
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12,
        "expected {expected:.15}, got {actual:.15}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&observed, &target) in actual.iter().zip(expected) {
        assert_close(observed, target);
    }
}

fn example_result() -> BifactorIndicesResult {
    let loadings = example_loadings();
    let uniqueness = uniquenesses(&loadings, 12, 4);
    bifactor_indices(
        &loadings,
        &uniqueness,
        BifactorIndicesConfig {
            n_items: 12,
            n_factors: 4,
            general_factor: 0,
            zero_tolerance: 0.0,
        },
    )
    .expect("published loading example is a valid strict bifactor pattern")
}

#[test]
fn published_example_matches_independent_formula_oracle() {
    let result = example_result();

    assert_eq!(result.factor_item_counts, vec![12, 4, 4, 4]);
    assert!(result.is_strict_bifactor);
    assert_close(
        result.puc.expect("strict pattern has PUC"),
        0.7272727272727273,
    );
    assert_vec_close(
        &result.ecv_ss,
        &[
            0.7138154174781681,
            0.14269911504424782,
            0.40642006802721087,
            0.3229227845914666,
        ],
    );
    assert_vec_close(
        &result.ecv_sg,
        &[
            0.7138154174781681,
            0.048627253879499906,
            0.12011057360055286,
            0.11744675504177923,
        ],
    );
    assert_vec_close(
        &result.ecv_gs,
        &[
            0.7138154174781681,
            0.8573008849557524,
            0.5935799319727892,
            0.6770772154085335,
        ],
    );
    assert_vec_close(
        &result.item_ecv,
        &[
            0.9853458382180539,
            0.8287671232876713,
            0.859050240880936,
            0.7411944869831546,
            0.3403559277676001,
            0.6290872617853561,
            0.9647402461923639,
            0.5901639344262295,
            0.7571993844801055,
            0.8316008316008315,
            0.8858474082702388,
            0.3497109826589596,
        ],
    );
    assert_vec_close(
        &result.omega_total,
        &[
            0.9482359360310674,
            0.8915386688224198,
            0.8400527981054797,
            0.906756072874494,
        ],
    );
    assert_vec_close(
        &result.omega_hierarchical,
        &[
            0.8507481229683099,
            0.11331177580167072,
            0.3040646776792127,
            0.26424595141700413,
        ],
    );
    assert_vec_close(
        &result.construct_replicability,
        &[
            0.9282445565481302,
            0.3070802232706812,
            0.6144805391947925,
            0.6340947986171986,
        ],
    );
}

#[test]
fn puc_is_absent_for_cross_loaded_specific_factors() {
    let mut cross_loaded = example_loadings();
    cross_loaded[2] = 0.20;
    let uniqueness = uniquenesses(&cross_loaded, 12, 4);
    let cross = bifactor_indices(
        &cross_loaded,
        &uniqueness,
        BifactorIndicesConfig::new(12, 4, 0),
    )
    .unwrap();
    assert!(!cross.is_strict_bifactor);
    assert_eq!(cross.puc, None);
}

#[test]
fn incomplete_general_factor_is_rejected() {
    let mut missing_general = example_loadings();
    missing_general[0] = 0.0;
    let uniqueness = uniquenesses(&missing_general, 12, 4);
    let error = bifactor_indices(
        &missing_general,
        &uniqueness,
        BifactorIndicesConfig::new(12, 4, 0),
    )
    .unwrap_err();
    assert!(
        error.contains("general factor"),
        "unexpected error: {error}"
    );
    assert!(error.contains("item 0"), "unexpected error: {error}");
}

#[test]
fn strict_pattern_supports_nonfirst_general_factor_and_single_item_domain() {
    let loadings = vec![
        0.40, 0.70, 0.00, // item 1
        0.30, 0.70, 0.00, // item 2
        0.00, 0.70, 0.50, // item 3
    ];
    let uniqueness = uniquenesses(&loadings, 3, 3);
    let result =
        bifactor_indices(&loadings, &uniqueness, BifactorIndicesConfig::new(3, 3, 1)).unwrap();

    assert_eq!(result.factor_item_counts, vec![2, 3, 1]);
    assert!(result.is_strict_bifactor);
    assert_close(result.puc.unwrap(), 2.0 / 3.0);
}

#[test]
fn structural_zero_tolerance_controls_pattern_not_numeric_sums() {
    let loadings = vec![
        0.70, 0.40, 1e-14, // item 1
        0.70, 0.30, 0.00, // item 2
        0.70, 0.00, 0.50, // item 3
        0.70, 0.00, 0.60, // item 4
    ];
    let uniqueness = uniquenesses(&loadings, 4, 3);
    let tolerant = bifactor_indices(
        &loadings,
        &uniqueness,
        BifactorIndicesConfig {
            n_items: 4,
            n_factors: 3,
            general_factor: 0,
            zero_tolerance: 1e-12,
        },
    )
    .unwrap();
    let exact =
        bifactor_indices(&loadings, &uniqueness, BifactorIndicesConfig::new(4, 3, 0)).unwrap();

    assert!(tolerant.is_strict_bifactor);
    assert_close(tolerant.puc.unwrap(), 2.0 / 3.0);
    assert!(!exact.is_strict_bifactor);
    assert_eq!(exact.puc, None);
    assert_close(tolerant.ecv_sg[2], exact.ecv_sg[2]);
}

#[test]
fn constructor_and_dimension_guards_fail_closed() {
    let valid_loadings = vec![0.7, 0.2, 0.8, 0.3];
    let valid_uniquenesses = vec![0.47, 0.27];

    for (config, message) in [
        (BifactorIndicesConfig::new(1, 2, 0), "at least two items"),
        (BifactorIndicesConfig::new(2, 1, 0), "at least two factors"),
        (BifactorIndicesConfig::new(2, 2, 2), "general_factor"),
    ] {
        let error = bifactor_indices(&valid_loadings, &valid_uniquenesses, config).unwrap_err();
        assert!(error.contains(message), "unexpected error: {error}");
    }

    let error = bifactor_indices(
        &[0.7, 0.2, 0.8],
        &valid_uniquenesses,
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .unwrap_err();
    assert!(error.contains("loading matrix length"));

    let error =
        bifactor_indices(&valid_loadings, &[0.4], BifactorIndicesConfig::new(2, 2, 0)).unwrap_err();
    assert!(error.contains("uniquenesses length"));
}

#[test]
fn numeric_input_guards_reject_undefined_indices() {
    let config = BifactorIndicesConfig::new(2, 2, 0);

    for (loadings, uniquenesses, message) in [
        (vec![f64::NAN, 0.2, 0.8, 0.3], vec![0.4, 0.3], "finite"),
        (
            vec![1.0, 0.0, 0.8, 0.3],
            vec![0.0, 0.3],
            "absolute value below 1",
        ),
        (
            vec![0.7, 0.2, 0.8, 0.3],
            vec![-0.1, 0.3],
            "between zero and one",
        ),
        (
            vec![0.7, 0.2, 0.8, 0.3],
            vec![1.1, 0.27],
            "between zero and one",
        ),
        (vec![0.7, 0.2, 0.8, 0.3], vec![f64::INFINITY, 0.3], "finite"),
        (
            vec![0.0, 0.0, 0.8, 0.3],
            vec![1.0, 0.27],
            "active loading on the declared general factor",
        ),
        (vec![0.7, 0.0, 0.8, 0.0], vec![0.51, 0.36], "every factor"),
        (vec![1e-300, 0.2, 0.8, 0.3], vec![0.96, 0.27], "too small"),
        (vec![0.6, 0.2, 0.8, 0.3], vec![0.47, 0.27], "sum to one"),
    ] {
        let error = bifactor_indices(&loadings, &uniquenesses, config).unwrap_err();
        assert!(error.contains(message), "unexpected error: {error}");
    }

    for tolerance in [-1.0, f64::NAN, f64::INFINITY] {
        let error = bifactor_indices(
            &[0.7, 0.2, 0.8, 0.3],
            &[0.47, 0.27],
            BifactorIndicesConfig {
                zero_tolerance: tolerance,
                ..config
            },
        )
        .unwrap_err();
        assert!(error.contains("zero_tolerance"));
    }
}

#[test]
fn factor_excluded_by_zero_tolerance_is_rejected_explicitly() {
    let loadings = vec![0.7, 1e-4, 0.8, 0.0];
    let uniqueness = uniquenesses(&loadings, 2, 2);
    let error = bifactor_indices(
        &loadings,
        &uniqueness,
        BifactorIndicesConfig {
            n_items: 2,
            n_factors: 2,
            general_factor: 0,
            zero_tolerance: 1e-3,
        },
    )
    .unwrap_err();
    assert!(
        error.contains("at least one loading above zero_tolerance"),
        "unexpected error: {error}"
    );
}

#[test]
fn omega_rejects_zero_variance_composites_created_by_sign_cancellation() {
    let root_half = 0.5_f64.sqrt();
    let error = bifactor_indices(
        &[root_half, root_half, -root_half, -root_half],
        &[0.0, 0.0],
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .unwrap_err();
    assert!(error.contains("omega denominator must be positive"));
}
