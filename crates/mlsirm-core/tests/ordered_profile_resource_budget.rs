//! Resource-governance regressions for ordered proficiency posterior summaries.

use mlsirm_core::ordered_profile::{
    OrderedProfileError, OrderedProfileInput, summarize_ordered_profile,
};

#[test]
fn rejects_posterior_draw_count_above_package_budget_before_summary_work() {
    let samples = vec![0.0; 1_000_001];
    let cut_scores = [0.0];

    let result = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: None,
        cut_scores: &cut_scores,
        credible_mass: 0.95,
    });

    assert!(matches!(
        result,
        Err(OrderedProfileError::PosteriorWorkLimit { samples: 1_000_001 })
    ));
}

#[test]
fn rejects_weight_length_mismatch_before_posterior_value_scan() {
    let result = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[f64::NAN],
        sample_weights: Some(&[]),
        cut_scores: &[0.0],
        credible_mass: 0.95,
    });

    assert!(matches!(
        result,
        Err(OrderedProfileError::WeightLengthMismatch {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn rejects_empty_cut_scores_before_posterior_value_scan() {
    let result = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[f64::NAN],
        sample_weights: None,
        cut_scores: &[],
        credible_mass: 0.95,
    });

    assert!(matches!(result, Err(OrderedProfileError::EmptyCutScores)));
}

#[test]
fn rejects_invalid_credible_mass_before_posterior_value_scan() {
    let result = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[f64::NAN],
        sample_weights: None,
        cut_scores: &[0.0],
        credible_mass: f64::NAN,
    });

    assert!(matches!(result, Err(OrderedProfileError::InvalidCredibleMass)));
}

#[test]
fn rejects_credible_interval_work_budget_before_posterior_value_scan() {
    let cut_scores = (0..6324)
        .map(|index| f64::from(index) + 0.5)
        .collect::<Vec<_>>();
    let result = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[f64::NAN],
        sample_weights: None,
        cut_scores: &cut_scores,
        credible_mass: 0.95,
    });

    assert!(matches!(
        result,
        Err(OrderedProfileError::CredibleIntervalWorkLimit { levels: 6325 })
    ));
}
