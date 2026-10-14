//! Category admission with a fixed bank, not population recovery evidence.
//! Basis: Cai (2010, pp. 608–609, Appendix A) separates item category counts
//! from population likelihoods. An absent category is an admission condition
//! for free-item calibration, not an estimated boundary in an all-fixed bank.
//! Reference: Cai, L. (2010). A two-tier full-information item factor analysis
//! model with applications. *Psychometrika, 75*(4), 581–612.
//! doi:10.1007/s11336-010-9178-0.
use super::*;

fn controls() -> BifactorGrmConfig {
    BifactorGrmConfig {
        q_general: 121,
        q_specific: 121,
        max_iter: 1,
        tol: 1e-8,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1e-8,
        slope_prior: SlopePrior::None,
        device: crate::Device::Cpu,
    }
}

#[test]
fn fixed_policy_allows_absent_category_without_changing_declared_categories() {
    let v = validate_with_category_policy(
        &[0, 0, 0, 0, 1, 1, 1, 1],
        None,
        &[0, 0, 1, 1],
        2,
        4,
        2,
        4,
        &controls(),
        false,
    )
    .unwrap();
    assert_eq!(v.n_cat, 4);
    assert_eq!(v.m1, 3);
}

#[test]
fn legacy_eight_argument_validator_preserves_free_category_rejection() {
    let err = validate(
        &[0, 0, 0, 0, 1, 1, 1, 1],
        None,
        &[0, 0, 1, 1],
        2,
        4,
        2,
        4,
        &controls(),
    )
    .unwrap_err();
    assert!(err.contains("item 0 category 2 is never observed"));
}

#[test]
fn fixed_policy_rejects_out_of_range_response() {
    let err = validate_with_category_policy(
        &[4, 0, 0, 0, 1, 1, 1, 1],
        None,
        &[0, 0, 1, 1],
        2,
        4,
        2,
        4,
        &controls(),
        false,
    )
    .unwrap_err();
    assert_eq!(err, "observed response categories must be < n_cat");
}

#[test]
fn fixed_policy_rejects_short_response_shape() {
    let err =
        validate_with_category_policy(&[0], None, &[0, 0, 1, 1], 2, 4, 2, 4, &controls(), false)
            .unwrap_err();
    assert_eq!(err, "y must have length n_persons * n_items");
}

#[test]
fn fixed_policy_rejects_short_observation_mask() {
    let err = validate_with_category_policy(
        &[0; 8],
        Some(&[true]),
        &[0, 0, 1, 1],
        2,
        4,
        2,
        4,
        &controls(),
        false,
    )
    .unwrap_err();
    assert_eq!(err, "observed must have length n_persons * n_items");
}

#[test]
fn fixed_policy_rejects_item_without_observations() {
    let err = validate_with_category_policy(
        &[0; 8],
        Some(&[false, true, true, true, false, true, true, true]),
        &[0, 0, 1, 1],
        2,
        4,
        2,
        4,
        &controls(),
        false,
    )
    .unwrap_err();
    assert_eq!(err, "item 0 has no observed responses");
}

#[test]
fn complete_category_positive_preserves_both_policies() {
    let y = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let legacy = validate(&y, None, &[0, 0, 1, 1], 4, 4, 2, 4, &controls()).unwrap();
    let fixed =
        validate_with_category_policy(&y, None, &[0, 0, 1, 1], 4, 4, 2, 4, &controls(), false)
            .unwrap();
    assert_eq!(legacy.blocks, fixed.blocks);
    assert_eq!(legacy.item_block, fixed.item_block);
    assert_eq!(legacy.n_cat, fixed.n_cat);
}
