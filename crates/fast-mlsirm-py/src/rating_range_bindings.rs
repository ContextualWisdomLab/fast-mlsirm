//! Rust-owned paired rating-range evidence exposed through a modular PyO3 entrypoint.
//!
//! This diagnostic is deliberately descriptive. It summarizes category support
//! and dispersion on paired automated/reference ratings without claiming an
//! inferential rater range-restriction parameter.

use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

const MAX_CATEGORY_COUNT: usize = 1_000;

#[derive(Clone, Debug, PartialEq)]
struct PairedRatingRangeEvidence {
    sample_size: usize,
    automated_min: usize,
    automated_max: usize,
    reference_min: usize,
    reference_max: usize,
    automated_distinct_categories: usize,
    reference_distinct_categories: usize,
    automated_span: usize,
    reference_span: usize,
    automated_sd: f64,
    reference_sd: f64,
    span_ratio: Option<f64>,
    distinct_category_ratio: f64,
    sd_ratio: Option<f64>,
    lower_endpoint_gap: i64,
    upper_endpoint_gap: i64,
    narrower_observed_support: bool,
    central_tendency_signal: bool,
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

fn paired_rating_range_evidence(
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

fn result_dict(py: Python<'_>, result: PairedRatingRangeEvidence) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    out.set_item("sample_size", result.sample_size)?;
    out.set_item("automated_min", result.automated_min)?;
    out.set_item("automated_max", result.automated_max)?;
    out.set_item("reference_min", result.reference_min)?;
    out.set_item("reference_max", result.reference_max)?;
    out.set_item(
        "automated_distinct_categories",
        result.automated_distinct_categories,
    )?;
    out.set_item(
        "reference_distinct_categories",
        result.reference_distinct_categories,
    )?;
    out.set_item("automated_span", result.automated_span)?;
    out.set_item("reference_span", result.reference_span)?;
    out.set_item("automated_sd", result.automated_sd)?;
    out.set_item("reference_sd", result.reference_sd)?;
    out.set_item("span_ratio", result.span_ratio)?;
    out.set_item("distinct_category_ratio", result.distinct_category_ratio)?;
    out.set_item("sd_ratio", result.sd_ratio)?;
    out.set_item("lower_endpoint_gap", result.lower_endpoint_gap)?;
    out.set_item("upper_endpoint_gap", result.upper_endpoint_gap)?;
    out.set_item("narrower_observed_support", result.narrower_observed_support)?;
    out.set_item("central_tendency_signal", result.central_tendency_signal)?;
    Ok(out.into())
}

#[pyfunction(name = "paired_rating_range_evidence")]
fn py_paired_rating_range_evidence(
    py: Python<'_>,
    automated: PyReadonlyArray1<'_, u32>,
    reference: PyReadonlyArray1<'_, u32>,
    category_count: usize,
) -> PyResult<Py<PyDict>> {
    if automated.ndim() != 1 || reference.ndim() != 1 {
        return Err(PyValueError::new_err("ratings must be 1-D arrays"));
    }
    let result = paired_rating_range_evidence(
        automated.as_slice()?,
        reference.as_slice()?,
        category_count,
    )
    .map_err(PyValueError::new_err)?;
    result_dict(py, result)
}

#[pymodule]
#[pyo3(name = "_rating_range_core")]
fn fast_mlsirm_rating_range_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_paired_rating_range_evidence, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compressed_middle_range_matches_hand_calculation() {
        let result = paired_rating_range_evidence(&[1, 1, 2, 3, 3], &[0, 1, 2, 3, 4], 5)
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
    fn degenerate_reference_omits_unidentified_relative_ratios() {
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
    fn malformed_inputs_fail_closed() {
        assert!(paired_rating_range_evidence(&[0], &[0], 2)
            .expect_err("too short")
            .contains("at least two"));
        assert!(paired_rating_range_evidence(&[0, 1], &[0], 2)
            .expect_err("length mismatch")
            .contains("length"));
        assert!(paired_rating_range_evidence(&[0, 2], &[0, 1], 2)
            .expect_err("out of range")
            .contains("0..category_count-1"));
        assert!(paired_rating_range_evidence(&[0, 1], &[0, 1], 1)
            .expect_err("bad category count")
            .contains("category_count"));
        assert!(paired_rating_range_evidence(&[0, 1], &[0, 1], 1_001)
            .expect_err("bad category count")
            .contains("category_count"));
    }
}
