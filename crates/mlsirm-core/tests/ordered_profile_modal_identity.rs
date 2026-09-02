//! Exact represented-probability contract for ordered-profile modal decisions.

use mlsirm_core::ordered_profile::{summarize_ordered_profile, OrderedProfileInput};

#[test]
fn distinct_represented_level_probabilities_do_not_become_a_tie() {
    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[-1.0, 1.0],
        sample_weights: Some(&[1.0, 1.0 + 1.0e-13]),
        cut_scores: &[0.0],
        credible_mass: 0.5,
    })
    .expect("finite positive-mass evidence must produce an ordered profile");

    assert_ne!(summary.level_probabilities[0], summary.level_probabilities[1]);
    assert_eq!(summary.credible_level_indices, vec![1]);
    assert_eq!(summary.reported_level_index, Some(1));
}
