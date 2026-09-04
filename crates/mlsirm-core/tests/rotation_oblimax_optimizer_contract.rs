//! End-to-end optimizer contract for the deterministic Oblimax reference.
//!
//! This is numerical integration evidence, not VV-SCI-006 scientific recovery.
//! It verifies that the criterion-level exact power-of-two scale contract survives
//! the oblique optimizer, canonicalization, and public `RotationSolution` path.

use mlsirm_core::rotation::{
    rotate_factor_loadings, RotationConfig, RotationCriterion, RotationMode,
};

fn mixed_loadings() -> Vec<f64> {
    vec![
        0.72, 0.39, 0.65, 0.35, 0.60, 0.31, -0.31, 0.70, -0.28, 0.64, -0.25, 0.58,
    ]
}

fn deterministic_oblique_config() -> RotationConfig {
    RotationConfig {
        mode: RotationMode::Oblique,
        normalize: false,
        n_starts: 1,
        seed: 20260904,
        max_iter: 500,
        tolerance: 1e-7,
        function_window: 10,
        max_line_search: 20,
        basin_tolerance: 1e-8,
        max_threads: 1,
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

#[test]
fn oblimax_optimizer_preserves_exact_power_of_two_scale_equivalence() {
    let loadings = mixed_loadings();
    let scale = f64::from_bits(((1023 + 300) as u64) << 52);
    let scaled: Vec<f64> = loadings.iter().map(|value| value * scale).collect();
    let config = deterministic_oblique_config();

    let baseline = rotate_factor_loadings(
        &loadings,
        6,
        2,
        &RotationCriterion::Oblimax,
        &config,
    )
    .expect("ordinary finite loadings must produce an Oblimax solution");
    let scaled_solution = rotate_factor_loadings(
        &scaled,
        6,
        2,
        &RotationCriterion::Oblimax,
        &config,
    )
    .expect("exactly scale-equivalent finite loadings must produce an Oblimax solution");

    assert_eq!(baseline.criterion_value.to_bits(), scaled_solution.criterion_value.to_bits());
    assert_eq!(bits(&baseline.start_values), bits(&scaled_solution.start_values));
    assert_eq!(bits(&baseline.transform_matrix), bits(&scaled_solution.transform_matrix));
    assert_eq!(
        bits(&baseline.factor_correlation),
        bits(&scaled_solution.factor_correlation)
    );
    assert_eq!(baseline.best_start_index, scaled_solution.best_start_index);
    assert_eq!(baseline.converged, scaled_solution.converged);
    assert_eq!(baseline.termination_reason, scaled_solution.termination_reason);

    let restored_scaled_pattern: Vec<f64> = scaled_solution
        .pattern_matrix
        .iter()
        .map(|value| value / scale)
        .collect();
    assert_eq!(bits(&baseline.pattern_matrix), bits(&restored_scaled_pattern));
}
