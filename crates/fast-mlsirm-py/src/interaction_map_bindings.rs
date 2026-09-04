//! Python bindings for the Rust residual interaction-map bounded context.

use mlsirm_core::interaction_map::residual_interaction_map as core_residual_interaction_map;
use mlsirm_core::interaction_map_envelope::{
    residual_interaction_map_envelope as core_residual_interaction_map_envelope,
};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use pyo3::wrap_pyfunction;

#[pyfunction(name = "residual_interaction_map")]
fn py_residual_interaction_map(
    py: Python<'_>,
    observed: PyReadonlyArray2<'_, f64>,
    expected: PyReadonlyArray2<'_, f64>,
    axis_count: usize,
) -> PyResult<Py<PyDict>> {
    if observed.shape() != expected.shape() {
        return Err(PyValueError::new_err(
            "observed and expected must have the same two-dimensional shape",
        ));
    }
    let shape = observed.shape();
    let result = core_residual_interaction_map(
        observed.as_slice()?,
        expected.as_slice()?,
        shape[0],
        shape[1],
        axis_count,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("person_indices", result.person_indices)?;
    out.set_item("item_indices", result.item_indices)?;
    out.set_item("scored_person_count", result.scored_person_count)?;
    out.set_item("scored_item_count", result.scored_item_count)?;
    out.set_item("person_coordinates", result.person_coordinates)?;
    out.set_item("item_coordinates", result.item_coordinates)?;
    out.set_item("singular_values", result.singular_values)?;
    out.set_item("axis_shares", result.axis_shares)?;
    out.set_item("residual", result.residual)?;
    out.set_item("distance", result.distance)?;
    out.set_item("reconstruction", result.reconstruction)?;
    out.set_item("explained_share", result.explained_share)?;
    out.set_item("unexplained", result.unexplained)?;
    out.set_item("cross_share", result.cross_share)?;
    out.set_item("axis_count", result.axis_count)?;
    Ok(out.into())
}

#[pyfunction(name = "residual_interaction_map_envelope")]
#[allow(clippy::too_many_arguments)]
fn py_residual_interaction_map_envelope(
    py: Python<'_>,
    schema_version: &str,
    input_digest: &str,
    person_ids: Vec<String>,
    item_ids: Vec<String>,
    observed: PyReadonlyArray2<'_, f64>,
    expected: PyReadonlyArray2<'_, f64>,
    axis_count: usize,
) -> PyResult<Py<PyDict>> {
    let observed_shape = observed.shape();
    let expected_shape = expected.shape();
    if observed_shape != expected_shape {
        return Err(PyValueError::new_err(
            "observed and expected must have the same two-dimensional shape",
        ));
    }
    let n_persons = observed_shape[0];
    let n_items = observed_shape[1];
    let envelope = core_residual_interaction_map_envelope(
        schema_version,
        input_digest,
        &person_ids,
        &item_ids,
        observed.as_slice()?,
        expected.as_slice()?,
        n_persons,
        n_items,
        axis_count,
    )
    .map_err(PyValueError::new_err)?;

    let out = PyDict::new(py);
    out.set_item("schema_version", envelope.schema_version)?;
    out.set_item("algorithm_id", envelope.algorithm_id)?;
    out.set_item("implementation_version", envelope.implementation_version)?;
    out.set_item("calculation_provenance", envelope.calculation_provenance)?;
    out.set_item("input_digest", envelope.input_digest)?;
    out.set_item("requested_axis_count", envelope.requested_axis_count)?;
    out.set_item(
        "cell_extrema_tie_policy",
        envelope.cell_extrema_tie_policy,
    )?;
    out.set_item("finite_value_status", envelope.finite_value_status)?;
    out.set_item("retained_person_ids", envelope.retained_person_ids)?;
    out.set_item("retained_item_ids", envelope.retained_item_ids)?;
    out.set_item("closest_cell_ids", envelope.closest_cell_ids)?;
    out.set_item("farthest_cell_ids", envelope.farthest_cell_ids)?;

    let map = envelope.map;
    out.set_item("person_indices", map.person_indices)?;
    out.set_item("item_indices", map.item_indices)?;
    out.set_item("scored_person_count", map.scored_person_count)?;
    out.set_item("scored_item_count", map.scored_item_count)?;
    out.set_item("effective_rank", map.effective_rank)?;
    out.set_item("map_person_count", map.map_person_count)?;
    out.set_item("map_item_count", map.map_item_count)?;
    out.set_item("incomplete_person_count", map.incomplete_person_count)?;
    out.set_item("incomplete_item_count", map.incomplete_item_count)?;
    out.set_item("closest_cell", map.closest_cell)?;
    out.set_item("farthest_cell", map.farthest_cell)?;
    out.set_item("person_coordinates", map.person_coordinates)?;
    out.set_item("item_coordinates", map.item_coordinates)?;
    out.set_item("singular_values", map.singular_values)?;
    out.set_item("axis_shares", map.axis_shares)?;
    out.set_item("observed", map.observed)?;
    out.set_item("expected", map.expected)?;
    out.set_item("residual", map.residual)?;
    out.set_item("distance", map.distance)?;
    out.set_item("reconstruction", map.reconstruction)?;
    out.set_item("explained_share", map.explained_share)?;
    out.set_item("unexplained", map.unexplained)?;
    out.set_item("cross_share", map.cross_share)?;
    Ok(out.into())
}

#[pymodule]
#[pyo3(name = "_interaction_map_core")]
fn fast_mlsirm_interaction_map_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_residual_interaction_map, m)?)?;
    m.add_function(wrap_pyfunction!(py_residual_interaction_map_envelope, m)?)?;
    Ok(())
}
