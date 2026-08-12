//! Rust-owned target-information ATA gain arithmetic exposed through PyO3.
//!
//! Python owns validation, content/exposure orchestration, and deterministic tie
//! breaking. This module owns the result-affecting capped-shortfall gain used to
//! rank eligible candidates. The implementation is allocation-bounded: one
//! output scalar per candidate and no candidate-by-point temporary matrix.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use pyo3::wrap_pyfunction;

fn target_information_gains_core(
    information_matrix: &[f64],
    n_points: usize,
    n_items: usize,
    candidates: &[usize],
    target_info: &[f64],
    accumulated: &[f64],
) -> Result<Vec<f64>, String> {
    if n_points == 0 || n_items == 0 {
        return Err("information matrix must be non-empty".into());
    }
    let expected_len = n_points
        .checked_mul(n_items)
        .ok_or_else(|| "information matrix shape exceeds addressable memory".to_owned())?;
    if information_matrix.len() != expected_len {
        return Err("information matrix storage does not match its shape".into());
    }
    if target_info.len() != n_points || accumulated.len() != n_points {
        return Err("target_info and accumulated must match the matrix row count".into());
    }
    if information_matrix
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("information matrix values must be finite and non-negative".into());
    }
    if target_info
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("target_info values must be finite and non-negative".into());
    }
    if accumulated
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
    {
        return Err("accumulated information must be finite and non-negative".into());
    }

    let mut gains = Vec::with_capacity(candidates.len());
    for &candidate in candidates {
        if candidate >= n_items {
            return Err("candidate item index is out of range".into());
        }
        let mut gain = 0.0;
        for point in 0..n_points {
            let remaining = (target_info[point] - accumulated[point]).max(0.0);
            let item_information = information_matrix[point * n_items + candidate];
            gain += item_information.min(remaining);
        }
        gains.push(gain);
    }
    Ok(gains)
}

#[pyfunction(name = "target_information_gains")]
fn py_target_information_gains(
    information_matrix: PyReadonlyArray2<'_, f64>,
    candidates: PyReadonlyArray1<'_, i64>,
    target_info: PyReadonlyArray1<'_, f64>,
    accumulated: PyReadonlyArray1<'_, f64>,
) -> PyResult<Vec<f64>> {
    let shape = information_matrix.shape();
    let candidate_values = candidates.as_slice()?;
    let mut candidate_indices = Vec::with_capacity(candidate_values.len());
    for &raw in candidate_values {
        if raw < 0 {
            return Err(PyValueError::new_err(
                "candidate item indices must be non-negative",
            ));
        }
        candidate_indices.push(
            usize::try_from(raw)
                .map_err(|_| PyValueError::new_err("candidate item index is too large"))?,
        );
    }
    target_information_gains_core(
        information_matrix.as_slice()?,
        shape[0],
        shape[1],
        &candidate_indices,
        target_info.as_slice()?,
        accumulated.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}

#[pymodule]
#[pyo3(name = "_ata_core")]
fn fast_mlsirm_ata_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_target_information_gains, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_gain_matches_hand_calculation_without_pointwise_temporaries() {
        let matrix = [1.0, 2.0, 4.0, 0.5, 2.0, 3.0];
        let gains = target_information_gains_core(
            &matrix,
            2,
            3,
            &[0, 2],
            &[4.0, 3.0],
            &[1.0, 2.0],
        )
        .expect("valid inputs");
        assert_eq!(gains, vec![1.5, 4.0]);
    }

    #[test]
    fn target_gain_is_zero_after_target_is_already_met() {
        let gains = target_information_gains_core(
            &[10.0, 20.0],
            1,
            2,
            &[0, 1],
            &[2.0],
            &[3.0],
        )
        .expect("valid inputs");
        assert_eq!(gains, vec![0.0, 0.0]);
    }

    #[test]
    fn target_gain_rejects_invalid_shapes_values_and_candidates() {
        assert!(target_information_gains_core(&[], 0, 1, &[], &[], &[])
            .expect_err("zero points")
            .contains("non-empty"));
        assert!(target_information_gains_core(&[1.0], 1, 2, &[], &[1.0], &[0.0])
            .expect_err("storage mismatch")
            .contains("storage"));
        assert!(target_information_gains_core(&[1.0, 2.0], 1, 2, &[], &[], &[0.0])
            .expect_err("target mismatch")
            .contains("row count"));
        assert!(target_information_gains_core(&[f64::NAN], 1, 1, &[], &[1.0], &[0.0])
            .expect_err("non-finite matrix")
            .contains("information matrix values"));
        assert!(target_information_gains_core(&[1.0], 1, 1, &[], &[-1.0], &[0.0])
            .expect_err("negative target")
            .contains("target_info"));
        assert!(target_information_gains_core(&[1.0], 1, 1, &[], &[1.0], &[f64::INFINITY])
            .expect_err("non-finite accumulated")
            .contains("accumulated"));
        assert!(target_information_gains_core(&[1.0], 1, 1, &[1], &[1.0], &[0.0])
            .expect_err("candidate out of range")
            .contains("out of range"));
    }
}