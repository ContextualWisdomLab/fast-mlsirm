//! Checked Rust-owned projected M2 quadratic-form kernel.
//!
//! This module provides the bounded numerical boundary used by composite
//! fit-statistic adapters that already assemble residual, derivative, and
//! covariance matrices. Python may validate and marshal those matrices, but
//! the projection arithmetic itself remains in `mlsirm-core`.

/// In-place lower-triangular Cholesky with an adaptive ridge.
fn cholesky_lower(matrix: &mut [f64], n: usize) -> Result<(), String> {
    let diag_mean = (0..n)
        .map(|index| matrix[index * n + index])
        .sum::<f64>()
        / n.max(1) as f64;
    let base = diag_mean.abs().max(1e-12) * 1e-10;
    let original = matrix.to_vec();
    for attempt in 0..8 {
        if attempt > 0 {
            matrix.copy_from_slice(&original);
            let ridge = base * (10.0_f64).powi(attempt);
            for index in 0..n {
                matrix[index * n + index] += ridge;
            }
        }
        let mut positive_definite = true;
        for column in 0..n {
            let mut diagonal = matrix[column * n + column];
            for prior in 0..column {
                diagonal -= matrix[column * n + prior] * matrix[column * n + prior];
            }
            if diagonal <= 0.0 {
                positive_definite = false;
                break;
            }
            let lower_diagonal = diagonal.sqrt();
            matrix[column * n + column] = lower_diagonal;
            for row in (column + 1)..n {
                let mut dot = matrix[row * n + column];
                for prior in 0..column {
                    dot -= matrix[row * n + prior] * matrix[column * n + prior];
                }
                matrix[row * n + column] = dot / lower_diagonal;
            }
        }
        if positive_definite {
            for column in 0..n {
                for row in 0..column {
                    matrix[row * n + column] = 0.0;
                }
            }
            return Ok(());
        }
    }
    Err("matrix is not positive definite (degenerate margins?)".into())
}

/// Solve `L L' x = b` for a lower Cholesky factor in row-major storage.
fn chol_solve(lower: &[f64], n: usize, rhs: &[f64]) -> Vec<f64> {
    let mut forward = vec![0.0_f64; n];
    for row in 0..n {
        let mut dot = rhs[row];
        for column in 0..row {
            dot -= lower[row * n + column] * forward[column];
        }
        forward[row] = dot / lower[row * n + row];
    }
    let mut solution = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut dot = forward[row];
        for column in (row + 1)..n {
            dot -= lower[column * n + row] * solution[column];
        }
        solution[row] = dot / lower[row * n + row];
    }
    solution
}

/// Evaluate the limited-information M2 projected quadratic form.
///
/// `residual` is length `n_moments`, `delta` is a row-major
/// `n_moments × n_parameters` derivative matrix, and `xi` is a row-major
/// `n_moments × n_moments` covariance matrix. The implementation matches the
/// established projection used by the ordinary and conditional M2 kernels but
/// exposes an explicit checked boundary for Python-composed multigroup blocks.
pub fn projected_m2_stat(
    residual: &[f64],
    delta: &[f64],
    xi: &[f64],
    n_moments: usize,
    n_parameters: usize,
    n: f64,
) -> Result<f64, String> {
    if n_moments == 0 || n_parameters == 0 || n_parameters >= n_moments {
        return Err("projected M2 requires 0 < n_parameters < n_moments".into());
    }
    if !n.is_finite() || n <= 0.0 {
        return Err("projected M2 sample scale must be finite and positive".into());
    }
    let delta_len = crate::checked_mul_usize(
        n_moments,
        n_parameters,
        "projected M2 dimensions overflow usize",
    )?;
    let xi_len = crate::checked_mul_usize(
        n_moments,
        n_moments,
        "projected M2 dimensions overflow usize",
    )?;
    if residual.len() != n_moments || delta.len() != delta_len || xi.len() != xi_len {
        return Err("projected M2 input lengths do not match dimensions".into());
    }
    if residual
        .iter()
        .chain(delta.iter())
        .chain(xi.iter())
        .any(|value| !value.is_finite())
    {
        return Err("projected M2 matrix inputs must be finite".into());
    }

    let s = n_moments;
    let p = n_parameters;
    let mut covariance = xi.to_vec();
    cholesky_lower(&mut covariance, s)?;
    let inverse_residual = chol_solve(&covariance, s, residual);

    let work_len = crate::checked_mul_usize(s, p, "projected M2 workspace overflows usize")?;
    let mut inverse_delta = vec![0.0_f64; work_len];
    let mut column_rhs = vec![0.0_f64; s];
    for column in 0..p {
        for row in 0..s {
            column_rhs[row] = delta[row * p + column];
        }
        let solved = chol_solve(&covariance, s, &column_rhs);
        for row in 0..s {
            inverse_delta[row * p + column] = solved[row];
        }
    }

    let information_len = crate::checked_mul_usize(p, p, "projected M2 workspace overflows usize")?;
    let mut information = vec![0.0_f64; information_len];
    let mut score = vec![0.0_f64; p];
    for row_parameter in 0..p {
        for column_parameter in 0..p {
            let mut total = 0.0;
            for row in 0..s {
                total += delta[row * p + row_parameter]
                    * inverse_delta[row * p + column_parameter];
            }
            information[row_parameter * p + column_parameter] = total;
        }
        let mut total = 0.0;
        for row in 0..s {
            total += inverse_delta[row * p + row_parameter] * residual[row];
        }
        score[row_parameter] = total;
    }

    cholesky_lower(&mut information, p)?;
    let adjustment_solution = chol_solve(&information, p, &score);
    let quadratic: f64 = (0..s)
        .map(|row| residual[row] * inverse_residual[row])
        .sum();
    let adjustment: f64 = (0..p)
        .map(|parameter| score[parameter] * adjustment_solution[parameter])
        .sum();
    Ok((n * (quadratic - adjustment)).max(0.0))
}

#[cfg(test)]
mod tests {
    use super::projected_m2_stat;

    #[test]
    fn projection_matches_identity_hand_calculation() {
        let residual = [1.0, 2.0, 3.0];
        let delta = [1.0, 0.0, 0.0];
        let xi = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        let actual = projected_m2_stat(&residual, &delta, &xi, 3, 1, 2.0).unwrap();
        assert!((actual - 26.0).abs() < 1e-12);
    }

    #[test]
    fn projection_rejects_invalid_dimensions_and_values() {
        assert!(projected_m2_stat(&[], &[], &[], 0, 0, 1.0)
            .unwrap_err()
            .contains("0 < n_parameters < n_moments"));
        assert!(projected_m2_stat(&[0.0; 3], &[0.0; 5], &[0.0; 9], 3, 2, 1.0)
            .unwrap_err()
            .contains("lengths"));
        assert!(projected_m2_stat(
            &[0.0; 3],
            &[0.0; 6],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
            f64::NAN,
        )
        .unwrap_err()
        .contains("finite and positive"));
        assert!(projected_m2_stat(
            &[f64::INFINITY, 0.0, 0.0],
            &[0.0; 6],
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            3,
            2,
            1.0,
        )
        .unwrap_err()
        .contains("matrix inputs must be finite"));
    }

    #[test]
    fn projection_rejects_dimension_overflow_before_allocation() {
        assert!(projected_m2_stat(&[], &[], &[], usize::MAX, 2, 1.0)
            .unwrap_err()
            .contains("overflow"));
    }

    #[test]
    fn projection_rejects_irrecoverably_non_positive_covariance() {
        let error = projected_m2_stat(
            &[0.1, -0.1],
            &[1.0, 0.0],
            &[-1.0, 0.0, 0.0, -1.0],
            2,
            1,
            1.0,
        )
        .unwrap_err();
        assert!(error.contains("not positive definite"));
    }
}
