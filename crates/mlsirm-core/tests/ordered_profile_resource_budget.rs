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
