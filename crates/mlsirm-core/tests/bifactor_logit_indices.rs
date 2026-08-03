//! Contract for converting fitted logistic bifactor slopes to scoreability indices.

use mlsirm_core::bifactor_indices::{
    bifactor_indices, bifactor_indices_from_logit_slopes, BifactorIndicesConfig,
};

fn standardized_example() -> Vec<f64> {
    vec![
        0.82, 0.10, 0.00, 0.00, 0.77, 0.35, 0.00, 0.00, 0.79, 0.32, 0.00, 0.00, 0.66,
        0.39, 0.00, 0.00, 0.51, 0.00, 0.71, 0.00, 0.56, 0.00, 0.43, 0.00, 0.68, 0.00,
        0.13, 0.00, 0.60, 0.00, 0.50, 0.00, 0.83, 0.00, 0.00, 0.47, 0.60, 0.00, 0.00,
        0.27, 0.78, 0.00, 0.00, 0.28, 0.55, 0.00, 0.00, 0.75,
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

fn raw_logit_slopes(loadings: &[f64], uniquenesses: &[f64], n_factors: usize) -> Vec<f64> {
    let logistic_sd = (std::f64::consts::PI.powi(2) / 3.0).sqrt();
    loadings
        .chunks_exact(n_factors)
        .zip(uniquenesses)
        .flat_map(|(row, &uniqueness)| {
            let multiplier = logistic_sd / uniqueness.sqrt();
            row.iter().map(move |loading| loading * multiplier)
        })
        .collect()
}

fn assert_vec_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (&observed, &target) in actual.iter().zip(expected) {
        assert!(
            (observed - target).abs() <= 2e-12,
            "expected {target:.15}, got {observed:.15}"
        );
    }
}

#[test]
fn logistic_slope_standardization_recovers_direct_loading_indices() {
    let loadings = standardized_example();
    let uniqueness = uniquenesses(&loadings, 12, 4);
    let slopes = raw_logit_slopes(&loadings, &uniqueness, 4);
    let config = BifactorIndicesConfig::new(12, 4, 0);

    let direct = bifactor_indices(&loadings, &uniqueness, config).unwrap();
    let converted = bifactor_indices_from_logit_slopes(&slopes, config).unwrap();

    assert_eq!(converted.factor_item_counts, direct.factor_item_counts);
    assert_eq!(converted.is_strict_bifactor, direct.is_strict_bifactor);
    assert!((converted.puc.unwrap() - direct.puc.unwrap()).abs() <= f64::EPSILON);
    assert_vec_close(&converted.ecv_ss, &direct.ecv_ss);
    assert_vec_close(&converted.ecv_sg, &direct.ecv_sg);
    assert_vec_close(&converted.ecv_gs, &direct.ecv_gs);
    assert_vec_close(&converted.item_ecv, &direct.item_ecv);
    assert_vec_close(&converted.omega_total, &direct.omega_total);
    assert_vec_close(
        &converted.omega_hierarchical,
        &direct.omega_hierarchical,
    );
    assert_vec_close(
        &converted.construct_replicability,
        &direct.construct_replicability,
    );
}

#[test]
fn logistic_conversion_rejects_malformed_or_nondegenerate_limits() {
    let config = BifactorIndicesConfig::new(2, 2, 0);
    let error = bifactor_indices_from_logit_slopes(&[0.5, 0.2, 0.6], config).unwrap_err();
    assert!(error.contains("logit slope matrix length"));

    let error = bifactor_indices_from_logit_slopes(&[0.5, f64::NAN, 0.6, 0.2], config)
        .unwrap_err();
    assert!(error.contains("finite"));

    let error = bifactor_indices_from_logit_slopes(&[0.0, 0.0, 0.6, 0.2], config)
        .unwrap_err();
    assert!(error.contains("every item"));

    let error = bifactor_indices_from_logit_slopes(
        &[f64::MAX, 0.0, 0.6, 0.2],
        config,
    )
    .unwrap_err();
    assert!(error.contains("absolute value below 1"));
}
