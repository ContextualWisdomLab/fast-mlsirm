//! Rust-owned paired rating-range evidence exposed through a modular PyO3 entrypoint.
//!
//! Numerical statistics live exclusively in `mlsirm-core::rating_range`. This
//! binding validates Python array layout, delegates unchanged rating slices to
//! the core, and marshals the Rust result into a Python dictionary.

use mlsirm_core::rating_range::{paired_rating_range_evidence, PairedRatingRangeEvidence};
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

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
    fn binding_uses_core_hand_calculation() {
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
    fn binding_preserves_core_degenerate_reference_behavior() {
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
    fn binding_preserves_core_validation_errors() {
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
