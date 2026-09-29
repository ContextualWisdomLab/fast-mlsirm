//! Python access to Rust-owned bootstrap Monte Carlo rank diagnostics.

use mlsirm_core::bootstrap_mc::{
    binomial_interval_coverage, binomial_quantile, linear_percentile, mc_rank_interval,
    MAX_BOOTSTRAP_MC_DRAWS,
};
use numpy::{PyReadonlyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

#[pyfunction(name = "binomial_quantile")]
fn py_binomial_quantile(n: usize, p: f64, probability: f64) -> PyResult<usize> {
    binomial_quantile(n, p, probability).map_err(PyValueError::new_err)
}

#[pyfunction(name = "binomial_interval_coverage")]
fn py_binomial_interval_coverage(n: usize, p: f64, low: usize, high: usize) -> PyResult<f64> {
    binomial_interval_coverage(n, p, low, high).map_err(PyValueError::new_err)
}

fn validate_draw_count(draw_count: usize, minimum_draw_count: usize) -> Result<(), String> {
    if !(minimum_draw_count..=MAX_BOOTSTRAP_MC_DRAWS).contains(&draw_count) {
        return Err(format!(
            "values must contain {minimum_draw_count}..={MAX_BOOTSTRAP_MC_DRAWS} finite draws"
        ));
    }
    Ok(())
}

#[pyfunction(name = "linear_percentile")]
fn py_linear_percentile(values: PyReadonlyArray1<'_, f64>, p: f64) -> PyResult<f64> {
    validate_draw_count(values.len(), 1).map_err(PyValueError::new_err)?;
    let draws: Vec<f64> = values.as_array().iter().copied().collect();
    linear_percentile(&draws, p).map_err(PyValueError::new_err)
}

#[pyfunction(name = "mc_rank_interval")]
fn py_mc_rank_interval(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    percentile: f64,
    confidence: f64,
) -> PyResult<Py<PyDict>> {
    validate_draw_count(values.len(), 2).map_err(PyValueError::new_err)?;
    let draws: Vec<f64> = values.as_array().iter().copied().collect();
    let result = mc_rank_interval(&draws, percentile, confidence).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("confidence", result.confidence)?;
    out.set_item("distribution", "Binomial(B, percentile)")?;
    out.set_item("count_low", result.count_low)?;
    out.set_item("count_high", result.count_high)?;
    out.set_item("rank_low_zero_based", result.rank_low_zero_based)?;
    out.set_item("rank_high_zero_based", result.rank_high_zero_based)?;
    out.set_item("attained_coverage", result.attained_coverage)?;
    out.set_item("lo", result.lo)?;
    out.set_item("hi", result.hi)?;
    Ok(out.into())
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_binomial_quantile, m)?)?;
    m.add_function(wrap_pyfunction!(py_binomial_interval_coverage, m)?)?;
    m.add_function(wrap_pyfunction!(py_linear_percentile, m)?)?;
    m.add_function(wrap_pyfunction!(py_mc_rank_interval, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_limit_rejects_before_numpy_collection() {
        assert!(validate_draw_count(MAX_BOOTSTRAP_MC_DRAWS + 1, 1).is_err());
        assert!(validate_draw_count(MAX_BOOTSTRAP_MC_DRAWS + 1, 2).is_err());
    }
}
