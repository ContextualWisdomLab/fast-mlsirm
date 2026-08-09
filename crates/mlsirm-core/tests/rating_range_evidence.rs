//! Public Rust-core contract tests for paired rating-range evidence.

use mlsirm_core::rating_range::{paired_rating_range_evidence, PairedRatingRangeEvidence};

#[test]
fn public_core_recovers_hand_calculated_range_evidence() {
    let result: PairedRatingRangeEvidence =
        paired_rating_range_evidence(&[1, 1, 2, 3, 3], &[0, 1, 2, 3, 4], 5)
            .expect("valid paired ratings");

    assert_eq!(result.sample_size, 5);
    assert_eq!((result.automated_min, result.automated_max), (1, 3));
    assert_eq!((result.reference_min, result.reference_max), (0, 4));
    assert_eq!(result.automated_distinct_categories, 3);
    assert_eq!(result.reference_distinct_categories, 5);
    assert_eq!(result.automated_span, 2);
    assert_eq!(result.reference_span, 4);
    assert!((result.automated_sd - 0.8_f64.sqrt()).abs() < 1e-12);
    assert!((result.reference_sd - 2.0_f64.sqrt()).abs() < 1e-12);
    assert_eq!(result.span_ratio, Some(0.5));
    assert!((result.distinct_category_ratio - 0.6).abs() < 1e-12);
    assert!((result.sd_ratio.expect("identified") - 0.4_f64.sqrt()).abs() < 1e-12);
    assert_eq!(result.lower_endpoint_gap, 1);
    assert_eq!(result.upper_endpoint_gap, 1);
    assert!(result.narrower_observed_support);
    assert!(result.central_tendency_signal);
}

#[test]
fn public_core_omits_unidentified_relative_ratios() {
    let result = paired_rating_range_evidence(&[1, 2, 2, 3], &[2, 2, 2, 2], 5)
        .expect("valid paired ratings");

    assert_eq!(result.reference_span, 0);
    assert_eq!(result.reference_sd, 0.0);
    assert_eq!(result.span_ratio, None);
    assert_eq!(result.sd_ratio, None);
    assert_eq!(result.distinct_category_ratio, 3.0);
    assert!(!result.narrower_observed_support);
    assert!(!result.central_tendency_signal);
}

#[test]
fn public_core_rejects_invalid_inputs() {
    assert!(paired_rating_range_evidence(&[0], &[0], 2).is_err());
    assert!(paired_rating_range_evidence(&[0, 1], &[0], 2).is_err());
    assert!(paired_rating_range_evidence(&[0, 2], &[0, 1], 2).is_err());
    assert!(paired_rating_range_evidence(&[0, 1], &[0, 1], 1).is_err());
    assert!(paired_rating_range_evidence(&[0, 1], &[0, 1], 1_001).is_err());
}
