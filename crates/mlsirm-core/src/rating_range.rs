//! Descriptive paired rating-range evidence for automated scoring validation.
//!
//! This module summarizes how an automated scorer uses an ordinal category
//! scale relative to paired reference ratings. It is intentionally descriptive:
//! the returned statistics are not a generalized many-facet range-restriction
//! parameter and they do not define universal pass/fail thresholds.

const MAX_CATEGORY_COUNT: usize = 1_000;

/// Descriptive category-support and dispersion evidence for paired ratings.
#[derive(Clone, Debug, PartialEq)]
pub struct PairedRatingRangeEvidence {
    /// Number of paired rating observations.
    pub sample_size: usize,
    /// Smallest observed automated-score category.
    pub automated_min: usize,
    /// Largest observed automated-score category.
    pub automated_max: usize,
    /// Smallest observed reference-score category.
    pub reference_min: usize,
    /// Largest observed reference-score category.
    pub reference_max: usize,
    /// Number of distinct categories used by the automated scorer.
    pub automated_distinct_categories: usize,
    /// Number of distinct categories used by the reference scorer.
    pub reference_distinct_categories: usize,
    /// Observed automated-score range, `automated_max - automated_min`.
    pub automated_span: usize,
    /// Observed reference-score range, `reference_max - reference_min`.
    pub reference_span: usize,
    /// Population-divisor empirical standard deviation of automated ratings.
    pub automated_sd: f64,
    /// Population-divisor empirical standard deviation of reference ratings.
    pub reference_sd: f64,
    /// Automated/reference observed-span ratio, unavailable for zero reference span.
    pub span_ratio: Option<f64>,
    /// Automated/reference distinct-category-count ratio.
    pub distinct_category_ratio: f64,
    /// Automated/reference empirical-SD ratio, unavailable for zero reference SD.
    pub sd_ratio: Option<f64>,
    /// Signed lower-end gap, `automated_min - reference_min`.
    pub lower_endpoint_gap: i64,
    /// Signed upper-end gap, `reference_max - automated_max`.
    pub upper_endpoint_gap: i64,
    /// Whether automated support is narrower in both span and category count.
    pub narrower_observed_support: bool,
    /// Whether narrower support also truncates both reference endpoints inward.
    pub central_tendency_signal: bool,
}

#[derive(Clone, Copy, Debug)]
struct RatingSummary {
    minimum: usize,
    maximum: usize,
    distinct_categories: usize,
    sd: f64,
}

fn summarize_ratings(labels: &[u32], category_count: usize) -> Result<RatingSummary, String> {
    let mut present = vec![false; category_count];
    let mut minimum = usize::MAX;
    let mut maximum = 0usize;
    let mut mean = 0.0;
    let mut m2 = 0.0;

    for (index, &raw) in labels.iter().enumerate() {
        let label = usize::try_from(raw).map_err(|_| "rating label does not fit usize")?;
        if label >= category_count {
            return Err(format!(
                "rating label at paired index {index} must be in 0..category_count-1"
            ));
        }
        present[label] = true;
        minimum = minimum.min(label);
        maximum = maximum.max(label);

        let x = f64::from(raw);
        let n = (index + 1) as f64;
        let delta = x - mean;
        mean += delta / n;
        m2 += delta * (x - mean);
    }

    let distinct_categories = present.into_iter().filter(|seen| *seen).count();
    let variance = m2 / labels.len() as f64;
    Ok(RatingSummary {
        minimum,
        maximum,
        distinct_categories,
        sd: variance.max(0.0).sqrt(),
    })
}

/// Compute descriptive rating-range evidence over the same paired cases.
///
/// `automated` and `reference` must have equal lengths of at least two. Every
/// label must be in `0..category_count`, and `category_count` must be between 2
/// and 1,000 inclusive. Dispersion uses the population divisor (`n`) because
/// these paired observations are the complete validation cases supplied to this
/// diagnostic rather than a sample-SD estimator.
///
/// A zero reference span or reference standard deviation makes the corresponding
/// relative ratio unidentified; that ratio is returned as [`None`] rather than
/// NaN or infinity.
///
/// # Errors
///
/// Returns a bounded descriptive error when lengths, category count, or labels
/// violate the public contract.
pub fn paired_rating_range_evidence(
    automated: &[u32],
    reference: &[u32],
    category_count: usize,
) -> Result<PairedRatingRangeEvidence, String> {
    if !(2..=MAX_CATEGORY_COUNT).contains(&category_count) {
        return Err(format!(
            "category_count must be between 2 and {MAX_CATEGORY_COUNT}"
        ));
    }
    if automated.len() != reference.len() {
        return Err("automated and reference rating lengths must match".to_owned());
    }
    if automated.len() < 2 {
        return Err("paired ratings require at least two observations".to_owned());
    }

    let automated_summary = summarize_ratings(automated, category_count)?;
    let reference_summary = summarize_ratings(reference, category_count)?;
    let automated_span = automated_summary.maximum - automated_summary.minimum;
    let reference_span = reference_summary.maximum - reference_summary.minimum;
    let lower_endpoint_gap = automated_summary.minimum as i64 - reference_summary.minimum as i64;
    let upper_endpoint_gap = reference_summary.maximum as i64 - automated_summary.maximum as i64;
    let narrower_observed_support = automated_span < reference_span
        && automated_summary.distinct_categories < reference_summary.distinct_categories;
    let central_tendency_signal =
        narrower_observed_support && lower_endpoint_gap > 0 && upper_endpoint_gap > 0;

    Ok(PairedRatingRangeEvidence {
        sample_size: automated.len(),
        automated_min: automated_summary.minimum,
        automated_max: automated_summary.maximum,
        reference_min: reference_summary.minimum,
        reference_max: reference_summary.maximum,
        automated_distinct_categories: automated_summary.distinct_categories,
        reference_distinct_categories: reference_summary.distinct_categories,
        automated_span,
        reference_span,
        automated_sd: automated_summary.sd,
        reference_sd: reference_summary.sd,
        span_ratio: (reference_span > 0)
            .then_some(automated_span as f64 / reference_span as f64),
        distinct_category_ratio: automated_summary.distinct_categories as f64
            / reference_summary.distinct_categories as f64,
        sd_ratio: (reference_summary.sd > 0.0)
            .then_some(automated_summary.sd / reference_summary.sd),
        lower_endpoint_gap,
        upper_endpoint_gap,
        narrower_observed_support,
        central_tendency_signal,
    })
}
