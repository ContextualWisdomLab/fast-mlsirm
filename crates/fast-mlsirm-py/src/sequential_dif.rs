#[pyfunction]
fn sequential_dif_complete_gpu(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(|| mlsirm_core::mask_grm::lifecycle::complete_gpu(&request)).map_err(PyValueError::new_err)
}
#[pyfunction]
fn sequential_dif_fit_private(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(|| mlsirm_core::mask_grm::state::fit_private(&request)).map_err(PyValueError::new_err)
}
#[pyfunction]
fn sequential_dif_roundtrip_private(py: Python<'_>, state: String) -> PyResult<String> {
    py.detach(|| mlsirm_core::mask_grm::MaskFit::restore_private(&state)?.export_private()).map_err(PyValueError::new_err)
}
#[pyfunction]
fn sequential_dif_restore_summary(py: Python<'_>, state: String) -> PyResult<String> {
    py.detach(|| {
        let fit=mlsirm_core::mask_grm::MaskFit::restore_private(&state)?;
        Ok::<String,String>(fit.summary()?.to_string())
    }).map_err(PyValueError::new_err)
}
#[pyfunction]
fn sequential_dif_capabilities(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let raw = mlsirm_core::mask_grm::lifecycle::capabilities_json();
    Ok(py.import("json")?.call_method1("loads", (raw,))?.unbind())
}
#[pyfunction]
fn sequential_dif_round_gpu(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(|| mlsirm_core::mask_grm::lifecycle::round_gpu(&request)).map_err(PyValueError::new_err)
}
#[pyfunction]
fn sequential_dif_fit_gpu(py: Python<'_>, request: String) -> PyResult<String> {
    py.detach(|| mlsirm_core::mask_grm::lifecycle::fit_gpu(&request)).map_err(PyValueError::new_err)
}
