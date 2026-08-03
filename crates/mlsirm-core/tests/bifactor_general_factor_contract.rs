//! Fail-closed contracts for bifactor identity and general-factor existence.

use mlsirm_core::bifactor_indices::{bifactor_indices, BifactorIndicesConfig};

#[test]
fn incomplete_declared_general_factor_is_rejected() {
    let loadings = [
        0.0, 0.4, // item 1 has no declared general loading
        0.7, 0.3, // item 2
    ];
    let uniquenesses = [0.84, 0.42];

    let error = bifactor_indices(
        &loadings,
        &uniquenesses,
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .expect_err("bifactor-labelled indices require a general loading for every item");

    assert!(
        error.contains("general factor") && error.contains("item 0"),
        "unexpected error: {error}"
    );
}

#[test]
fn standardized_identity_accepts_small_floating_point_roundoff() {
    let loadings = [
        0.7, 0.2, // item 1: communality .53
        0.8, 0.3, // item 2: communality .73
    ];
    let uniquenesses = [0.47 + 5e-9, 0.27 - 5e-9];

    let result = bifactor_indices(
        &loadings,
        &uniquenesses,
        BifactorIndicesConfig::new(2, 2, 0),
    )
    .expect("roundoff within the documented 1e-8 identity tolerance is valid");

    assert_eq!(result.factor_item_counts, vec![2, 2]);
}
