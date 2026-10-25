//! Python marshalling for DIF information criteria; all arithmetic stays in Rust.
use mlsirm_core::dif_information_criteria::{compare_dif_information_criteria, DifFitEvidence};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

#[pyfunction(name = "compare_dif_information_criteria")]
fn py_compare_dif_information_criteria(
    py: Python<'_>,
    compact: (f64, usize, usize, bool, String, String),
    augmented: (f64, usize, usize, bool, String, String),
    criterion: &str,
) -> PyResult<Py<PyDict>> {
    let evidence = |row: (f64, usize, usize, bool, String, String)| DifFitEvidence {
        loglik: row.0,
        n_parameters: row.1,
        n_persons: row.2,
        converged: row.3,
        likelihood_kind: row.4,
        comparison_key: row.5,
    };
    let result =
        compare_dif_information_criteria(&evidence(compact), &evidence(augmented), criterion)
            .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item("criterion", result.criterion)?;
    out.set_item("compact_ic", result.compact_ic)?;
    out.set_item("augmented_ic", result.augmented_ic)?;
    out.set_item("delta_ic", result.delta_ic)?;
    out.set_item("prefer_augmented", result.prefer_augmented)?;
    out.set_item("arithmetic_backend", "rust_cpu_f64")?;
    out.set_item("inference", "relative_model_selection")?;
    Ok(out.into())
}

#[pymodule]
#[pyo3(name = "_dif_information_criteria_core")]
fn dif_information_criteria_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_compare_dif_information_criteria, m)?)?;
    Ok(())
}
