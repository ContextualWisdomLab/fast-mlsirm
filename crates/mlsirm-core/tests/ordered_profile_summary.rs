//! Fail-first contracts for ordinal proficiency summaries from calibrated posterior draws.

use mlsirm_core::ordered_profile::{
    OrderedProfileError, OrderedProfileInput, summarize_ordered_profile,
};

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1.0e-12,
        "expected {expected:.16}, got {actual:.16}"
    );
}

#[test]
fn summarizes_unweighted_draws_with_upper_inclusive_cut_scores() {
    let samples = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let cut_scores = [-0.5, 0.5];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: None,
        cut_scores: &cut_scores,
        credible_mass: 0.8,
    })
    .expect("valid posterior draws must produce an ordered profile");

    assert_eq!(summary.level_probabilities, vec![0.2, 0.4, 0.4]);
    assert_eq!(summary.reported_level_index, None);
    assert_eq!(summary.credible_level_indices, vec![1, 2]);
    assert_close(summary.posterior_mean, 0.0);
    assert_close(summary.posterior_standard_deviation, 0.5_f64.sqrt());
}

#[test]
fn normalizes_nonnegative_weights_and_reports_a_unique_modal_level() {
    let samples = [-1.0, 0.0, 1.0];
    let weights = [1.0, 2.0, 1.0];
    let cut_scores = [-0.5, 0.5];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: Some(&weights),
        cut_scores: &cut_scores,
        credible_mass: 0.5,
    })
    .expect("finite positive-mass weights must be normalized");

    assert_eq!(summary.level_probabilities, vec![0.25, 0.5, 0.25]);
    assert_eq!(summary.reported_level_index, Some(1));
    assert_eq!(summary.credible_level_indices, vec![1]);
    assert_close(summary.posterior_mean, 0.0);
    assert_close(summary.posterior_standard_deviation, 0.5_f64.sqrt());
}

#[test]
fn chooses_the_lower_interval_when_equal_mass_intervals_tie() {
    let samples = [-1.0, 1.0];
    let cut_scores = [0.0];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: None,
        cut_scores: &cut_scores,
        credible_mass: 0.5,
    })
    .expect("a tied posterior remains reportable with explicit ambiguity");

    assert_eq!(summary.level_probabilities, vec![0.5, 0.5]);
    assert_eq!(summary.reported_level_index, None);
    assert_eq!(summary.credible_level_indices, vec![0]);
}

#[test]
fn requires_the_selected_interval_to_meet_credible_mass_strictly() {
    let samples = [-1.0, 1.0];
    let cut_scores = [0.0];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: None,
        cut_scores: &cut_scores,
        credible_mass: 0.5000000000005,
    })
    .expect("the full ordered scale meets the requested credible mass");

    assert_eq!(summary.credible_level_indices, vec![0, 1]);
}

#[test]
fn prefers_distinct_probability_mass_below_the_tie_tolerance() {
    let samples = [-1.0, 1.0];
    let weights = [1.0, 1.0 + 1.0e-13];
    let cut_scores = [0.0];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: Some(&weights),
        cut_scores: &cut_scores,
        credible_mass: 0.5,
    })
    .expect("both singleton intervals meet the requested credible mass");

    assert_eq!(summary.credible_level_indices, vec![1]);
}

#[test]
fn prefers_more_probability_mass_before_lower_start_tie_break() {
    let samples = [-1.0, 1.0];
    let weights = [2.0, 3.0];
    let cut_scores = [0.0];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: Some(&weights),
        cut_scores: &cut_scores,
        credible_mass: 0.3,
    })
    .expect("both one-level intervals meet the requested credible mass");

    assert_eq!(summary.level_probabilities, vec![0.4, 0.6]);
    assert_eq!(summary.reported_level_index, Some(1));
    assert_eq!(summary.credible_level_indices, vec![1]);
    assert!(summary.credible_level_indices.contains(&1));
}

#[test]
fn does_not_report_a_unique_modal_level_outside_the_credible_set() {
    let samples = [-1.5, -0.5, 0.5, 1.5];
    let weights = [40.0, 1.0, 30.0, 29.0];
    let cut_scores = [-1.0, 0.0, 1.0];

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: Some(&weights),
        cut_scores: &cut_scores,
        credible_mass: 0.58,
    })
    .expect("valid multimodal evidence must preserve decision ambiguity");

    assert_eq!(summary.level_probabilities, vec![0.4, 0.01, 0.3, 0.29]);
    assert_eq!(summary.credible_level_indices, vec![2, 3]);
    assert_eq!(summary.reported_level_index, None);
}

#[test]
fn uses_compensated_mass_for_credible_interval_admission() {
    let samples = (0..23).map(f64::from).collect::<Vec<_>>();
    let cut_scores = (0..22)
        .map(|index| f64::from(index) + 0.5)
        .collect::<Vec<_>>();
    let mut weights = Vec::with_capacity(23);
    weights.push(0.5);
    weights.extend(std::iter::repeat_n(1.0e-17, 20));
    weights.extend([0.49, 0.01]);

    let summary = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &samples,
        sample_weights: Some(&weights),
        cut_scores: &cut_scores,
        credible_mass: 0.5000000000000001,
    })
    .expect("compensated probability mass must admit the shortest qualifying interval");

    assert_eq!(summary.credible_level_indices, (0..=17).collect::<Vec<_>>());
}

#[test]
fn is_invariant_to_joint_sample_and_weight_permutation() {
    let cut_scores = [-0.5, 0.5];
    let first = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[-1.0, 0.0, 1.0],
        sample_weights: Some(&[1.0, 2.0, 3.0]),
        cut_scores: &cut_scores,
        credible_mass: 0.75,
    })
    .expect("first ordering must be valid");
    let permuted = summarize_ordered_profile(OrderedProfileInput {
        posterior_samples: &[1.0, -1.0, 0.0],
        sample_weights: Some(&[3.0, 1.0, 2.0]),
        cut_scores: &cut_scores,
        credible_mass: 0.75,
    })
    .expect("joint permutation must be valid");

    assert_eq!(first, permuted);
}

#[test]
fn rejects_invalid_posterior_and_weight_inputs_without_repair() {
    let cut_scores = [0.0];

    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &[],
            sample_weights: None,
            cut_scores: &cut_scores,
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::EmptyPosterior)
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &[f64::NAN],
            sample_weights: None,
            cut_scores: &cut_scores,
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::NonFinitePosterior { index: 0 })
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &[0.0, 1.0],
            sample_weights: Some(&[1.0]),
            cut_scores: &cut_scores,
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::WeightLengthMismatch {
            expected: 2,
            actual: 1
        })
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &[0.0],
            sample_weights: Some(&[-1.0]),
            cut_scores: &cut_scores,
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::InvalidWeight { index: 0 })
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &[0.0, 1.0],
            sample_weights: Some(&[0.0, 0.0]),
            cut_scores: &cut_scores,
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::ZeroWeightMass)
    ));
}

#[test]
fn rejects_invalid_cut_scores_and_credible_mass_without_reordering() {
    let samples = [0.0];

    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &samples,
            sample_weights: None,
            cut_scores: &[],
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::EmptyCutScores)
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &samples,
            sample_weights: None,
            cut_scores: &[f64::INFINITY],
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::NonFiniteCutScore { index: 0 })
    ));
    assert!(matches!(
        summarize_ordered_profile(OrderedProfileInput {
            posterior_samples: &samples,
            sample_weights: None,
            cut_scores: &[0.5, 0.5],
            credible_mass: 0.95,
        }),
        Err(OrderedProfileError::NonIncreasingCutScores { index: 1 })
    ));
    for credible_mass in [0.0, -0.1, 1.1, f64::NAN] {
        assert!(matches!(
            summarize_ordered_profile(OrderedProfileInput {
                posterior_samples: &samples,
                sample_weights: None,
                cut_scores: &[0.0],
                credible_mass,
            }),
            Err(OrderedProfileError::InvalidCredibleMass)
        ));
    }
}
