//! Small deterministic matrix primitives for factor rotation.
//!
//! Rotation works with dense row-major matrices whose factor dimension is
//! normally small. Keeping these operations in the core avoids a new BLAS ABI
//! dependency and gives CPU and future GPU implementations one explicit
//! numerical contract.

/// Return an `n x n` identity matrix.
pub(crate) fn identity(n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        out[i * n + i] = 1.0;
    }
    out
}

/// Transpose a row-major `rows x cols` matrix.
pub(crate) fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}

/// Multiply row-major `a_rows x a_cols` by `a_cols x b_cols` matrices.
pub(crate) fn matmul(
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; a_rows * b_cols];
    for i in 0..a_rows {
        for k in 0..a_cols {
            let aik = a[i * a_cols + k];
            for j in 0..b_cols {
                out[i * b_cols + j] += aik * b[k * b_cols + j];
            }
        }
    }
    out
}

/// Compute `a' b` for two row-major matrices with the same row count.
pub(crate) fn crossprod(
    a: &[f64],
    b: &[f64],
    rows: usize,
    a_cols: usize,
    b_cols: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; a_cols * b_cols];
    for r in 0..rows {
        for i in 0..a_cols {
            let ari = a[r * a_cols + i];
            for j in 0..b_cols {
                out[i * b_cols + j] += ari * b[r * b_cols + j];
            }
        }
    }
    out
}

/// Frobenius inner product.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Frobenius norm.
pub(crate) fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Normalize every column to unit Euclidean length.
pub(crate) fn normalize_columns(a: &mut [f64], rows: usize, cols: usize) -> Result<(), String> {
    for j in 0..cols {
        let mut ss = 0.0;
        for i in 0..rows {
            let value = a[i * cols + j];
            ss += value * value;
        }
        if !ss.is_finite() || ss <= 1e-24 {
            return Err(format!("rotation transform column {j} is singular"));
        }
        let scale = ss.sqrt().recip();
        for i in 0..rows {
            a[i * cols + j] *= scale;
        }
    }
    Ok(())
}

/// Invert a dense square matrix with scaled partial pivoting.
pub(crate) fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if a.len() != n * n {
        return Err("inverse input has an invalid shape".into());
    }
    let mut aug = vec![0.0; n * (2 * n)];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = a[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }
    for col in 0..n {
        let mut pivot = col;
        let mut best = aug[col * 2 * n + col].abs();
        for row in (col + 1)..n {
            let candidate = aug[row * 2 * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        if !best.is_finite() || best <= 1e-12 {
            return Err("rotation transform is singular or ill-conditioned".into());
        }
        if pivot != col {
            for j in 0..(2 * n) {
                aug.swap(col * 2 * n + j, pivot * 2 * n + j);
            }
        }
        let diagonal = aug[col * 2 * n + col];
        for j in 0..(2 * n) {
            aug[col * 2 * n + j] /= diagonal;
        }
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            if factor == 0.0 {
                continue;
            }
            for j in 0..(2 * n) {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    Ok(out)
}

/// Log determinant and inverse of a symmetric positive-definite matrix.
///
/// A lower-triangular Cholesky factorization preserves the symmetry contract,
/// avoids determinant-sign ambiguity from row pivoting, and permits stable
/// inverse construction through forward and backward triangular solves.
pub(crate) fn positive_logdet_inverse(a: &[f64], n: usize) -> Result<(f64, Vec<f64>), String> {
    if n == 0 || a.len() != n * n {
        return Err("log-determinant input has an invalid shape".into());
    }
    if a.iter().any(|value| !value.is_finite()) {
        return Err("criterion matrix must contain only finite values".into());
    }
    let scale = a
        .iter()
        .map(|value| value.abs())
        .fold(1.0_f64, f64::max);
    let symmetry_tolerance = 1e-12 * scale;
    for i in 0..n {
        for j in 0..i {
            if (a[i * n + j] - a[j * n + i]).abs() > symmetry_tolerance {
                return Err("criterion matrix must be symmetric".into());
            }
        }
    }

    // Exact rank deficiency can leave a tiny positive Cholesky residual after
    // floating-point cancellation. Scale the pivot floor to the matrix diagonal
    // so singular Bentler fourth-moment matrices fail closed while genuinely
    // near-singular positive-definite inputs remain supported.
    let diagonal_scale = (0..n)
        .map(|i| a[i * n + i].abs())
        .fold(0.0_f64, f64::max);
    let pivot_tolerance = 64.0 * f64::EPSILON * diagonal_scale.max(f64::MIN_POSITIVE);

    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut residual = a[i * n + j];
            for k in 0..j {
                residual -= lower[i * n + k] * lower[j * n + k];
            }
            if i == j {
                if !residual.is_finite() || residual <= pivot_tolerance {
                    return Err("criterion matrix is not positive definite".into());
                }
                lower[i * n + j] = residual.sqrt();
            } else {
                let diagonal = lower[j * n + j];
                let value = residual / diagonal;
                if !value.is_finite() {
                    return Err("criterion matrix is singular or ill-conditioned".into());
                }
                lower[i * n + j] = value;
            }
        }
    }

    let logdet = 2.0
        * (0..n)
            .map(|i| lower[i * n + i].ln())
            .sum::<f64>();
    if !logdet.is_finite() {
        return Err("criterion matrix log determinant is non-finite".into());
    }

    let mut inverse_matrix = vec![0.0; n * n];
    let mut forward = vec![0.0; n];
    let mut solution = vec![0.0; n];
    for column in 0..n {
        for i in 0..n {
            let mut rhs = if i == column { 1.0 } else { 0.0 };
            for k in 0..i {
                rhs -= lower[i * n + k] * forward[k];
            }
            forward[i] = rhs / lower[i * n + i];
        }
        for reverse in 0..n {
            let i = n - 1 - reverse;
            let mut rhs = forward[i];
            for k in (i + 1)..n {
                rhs -= lower[k * n + i] * solution[k];
            }
            solution[i] = rhs / lower[i * n + i];
        }
        for row in 0..n {
            let value = solution[row];
            if !value.is_finite() {
                return Err("criterion matrix inverse is non-finite".into());
            }
            inverse_matrix[row * n + column] = value;
        }
        forward.fill(0.0);
        solution.fill(0.0);
    }
    Ok((logdet, inverse_matrix))
}

/// Orthogonal Cayley retraction used by gradient projection.
pub(crate) fn cayley_step(
    transform: &[f64],
    projected_gradient: &[f64],
    n: usize,
    step: f64,
) -> Result<Vec<f64>, String> {
    let gt = transpose(projected_gradient, n, n);
    let tt = transpose(transform, n, n);
    let gp_t = matmul(projected_gradient, n, n, &tt, n);
    let t_gp = matmul(transform, n, n, &gt, n);
    let mut skew = vec![0.0; n * n];
    for idx in 0..skew.len() {
        skew[idx] = gp_t[idx] - t_gp[idx];
    }
    let mut plus = identity(n);
    let mut minus = identity(n);
    for idx in 0..skew.len() {
        plus[idx] += 0.5 * step * skew[idx];
        minus[idx] -= 0.5 * step * skew[idx];
    }
    let inv_plus = inverse(&plus, n)?;
    let curve = matmul(&inv_plus, n, n, &minus, n);
    Ok(matmul(&curve, n, n, transform, n))
}

/// Maximum absolute departure of `t' t` from identity.
pub(crate) fn orthogonality_error(transform: &[f64], n: usize) -> f64 {
    let gram = crossprod(transform, transform, n, n, n);
    let mut worst = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            worst = worst.max((gram[i * n + j] - expected).abs());
        }
    }
    worst
}

/// Modified Gram-Schmidt fallback for accumulated orthogonality drift.
pub(crate) fn orthonormalize_columns(a: &mut [f64], rows: usize, cols: usize) -> Result<(), String> {
    for j in 0..cols {
        for prior in 0..j {
            let mut projection = 0.0;
            for i in 0..rows {
                projection += a[i * cols + j] * a[i * cols + prior];
            }
            for i in 0..rows {
                a[i * cols + j] -= projection * a[i * cols + prior];
            }
        }
        let mut ss = 0.0;
        for i in 0..rows {
            ss += a[i * cols + j] * a[i * cols + j];
        }
        if ss <= 1e-24 || !ss.is_finite() {
            return Err("random orthogonal start is rank deficient".into());
        }
        let scale = ss.sqrt().recip();
        for i in 0..rows {
            a[i * cols + j] *= scale;
        }
    }
    Ok(())
}

/// SplitMix64 state transition.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Deterministic open-interval uniform draw.
fn uniform_open(state: &mut u64) -> f64 {
    let bits = splitmix64(state) >> 11;
    ((bits as f64) + 0.5) / ((1_u64 << 53) as f64)
}

/// Deterministic standard normal draw via Box-Muller.
fn standard_normal(state: &mut u64) -> f64 {
    let u1 = uniform_open(state);
    let u2 = uniform_open(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Seeded Haar-like orthogonal start obtained from a Gaussian QR factorization.
pub(crate) fn random_orthogonal(n: usize, seed: u64) -> Result<Vec<f64>, String> {
    let mut state = seed;
    let mut out = vec![0.0; n * n];
    for value in &mut out {
        *value = standard_normal(&mut state);
    }
    orthonormalize_columns(&mut out, n, n)?;
    Ok(out)
}

/// Seeded nonsingular, unit-column oblique start.
pub(crate) fn random_oblique(n: usize, seed: u64) -> Result<Vec<f64>, String> {
    for retry in 0..8_u64 {
        let mut state = seed ^ retry.wrapping_mul(0xD2B74407B1CE6E93);
        let mut out = identity(n);
        for i in 0..n {
            for j in 0..n {
                out[i * n + j] += 0.35 * standard_normal(&mut state);
            }
        }
        normalize_columns(&mut out, n, n)?;
        if inverse(&out, n).is_ok() {
            return Ok(out);
        }
    }
    Err("could not generate a nonsingular oblique start".into())
}

/// Oblique pattern `A T^{-T}` together with `T^{-1}`.
pub(crate) fn oblique_pattern(
    unrotated: &[f64],
    rows: usize,
    factors: usize,
    transform: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let inverse_transform = inverse(transform, factors)?;
    let inverse_transpose = transpose(&inverse_transform, factors, factors);
    Ok((
        matmul(unrotated, rows, factors, &inverse_transpose, factors),
        inverse_transform,
    ))
}

/// Factor correlation matrix `T' T` for a unit-column oblique transform.
pub(crate) fn factor_correlation(transform: &[f64], factors: usize) -> Vec<f64> {
    crossprod(transform, transform, factors, factors, factors)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn determinant_3x3(a: &[f64]) -> f64 {
        a[0] * (a[4] * a[8] - a[5] * a[7])
            - a[1] * (a[3] * a[8] - a[5] * a[6])
            + a[2] * (a[3] * a[7] - a[4] * a[6])
    }

    #[test]
    fn matrix_primitives_cover_success_and_failure_paths() {
        let id = identity(2);
        assert_eq!(transpose(&id, 2, 2), id);
        assert_eq!(matmul(&id, 2, 2, &id, 2), id);
        assert_eq!(crossprod(&id, &id, 2, 2, 2), id);
        assert_eq!(dot(&id, &id), 2.0);
        assert!((norm(&id) - 2.0_f64.sqrt()).abs() < 1e-12);

        let a = vec![4.0, 7.0, 2.0, 6.0];
        let inv = inverse(&a, 2).unwrap();
        let recovered = matmul(&a, 2, 2, &inv, 2);
        assert!(recovered
            .iter()
            .zip(identity(2))
            .all(|(x, y)| (x - y).abs() < 1e-10));
        assert!(inverse(&[1.0], 2).is_err());
        assert!(inverse(&[1.0, 2.0, 2.0, 4.0], 2).is_err());

        let spd = vec![2.0, 0.5, 0.5, 1.0];
        let (logdet, _) = positive_logdet_inverse(&spd, 2).unwrap();
        assert!((logdet - 1.75_f64.ln()).abs() < 1e-10);
        assert!(positive_logdet_inverse(&[1.0], 2).is_err());
        assert!(positive_logdet_inverse(&[0.0, 0.0, 0.0, 0.0], 2).is_err());
        assert!(positive_logdet_inverse(&[-1.0, 0.0, 0.0, 1.0], 2).is_err());

        let mut columns = vec![3.0, 0.0, 4.0, 2.0];
        normalize_columns(&mut columns, 2, 2).unwrap();
        assert!((columns[0] * columns[0] + columns[2] * columns[2] - 1.0).abs() < 1e-12);
        assert!(normalize_columns(&mut [0.0, 0.0, 0.0, 1.0], 2, 2).is_err());
    }

    #[test]
    fn cholesky_logdet_and_inverse_cover_spd_edge_cases() {
        let pivot_provoking = vec![1.0, 2.0, 2.0, 5.0];
        let (logdet, inv) = positive_logdet_inverse(&pivot_provoking, 2).unwrap();
        assert!(logdet.abs() < 1e-12);
        let expected = [5.0, -2.0, -2.0, 1.0];
        assert!(inv
            .iter()
            .zip(expected)
            .all(|(actual, expected)| (actual - expected).abs() < 1e-10));

        let epsilon = 1e-8;
        let near_singular = vec![1.0, 1.0 - epsilon, 1.0 - epsilon, 1.0];
        let (_, near_inverse) = positive_logdet_inverse(&near_singular, 2).unwrap();
        let recovered = matmul(&near_singular, 2, 2, &near_inverse, 2);
        assert!(recovered
            .iter()
            .zip(identity(2))
            .all(|(actual, expected)| (actual - expected).abs() < 1e-6));

        let oracle = vec![4.0, 1.0, 1.0, 1.0, 3.0, 0.5, 1.0, 0.5, 2.0];
        let (oracle_logdet, oracle_inverse) = positive_logdet_inverse(&oracle, 3).unwrap();
        assert!((oracle_logdet - determinant_3x3(&oracle).ln()).abs() < 1e-12);
        let recovered = matmul(&oracle, 3, 3, &oracle_inverse, 3);
        assert!(recovered
            .iter()
            .zip(identity(3))
            .all(|(actual, expected)| (actual - expected).abs() < 1e-10));

        assert!(positive_logdet_inverse(&[1.0, 1.0, 1.0, 1.0], 2).is_err());
        assert!(positive_logdet_inverse(&[1.0, 2.0, 2.0, 1.0], 2).is_err());
        assert!(positive_logdet_inverse(&[1.0, 0.0, 0.5, 1.0], 2).is_err());
        assert!(positive_logdet_inverse(&[f64::NAN, 0.0, 0.0, 1.0], 2).is_err());
        assert!(positive_logdet_inverse(&[], 0).is_err());
    }

    #[test]
    fn starts_and_retractions_are_deterministic() {
        let q1 = random_orthogonal(3, 7).unwrap();
        let q2 = random_orthogonal(3, 7).unwrap();
        assert_eq!(q1, q2);
        assert!(orthogonality_error(&q1, 3) < 1e-12);

        let o1 = random_oblique(3, 11).unwrap();
        let o2 = random_oblique(3, 11).unwrap();
        assert_eq!(o1, o2);
        assert!(inverse(&o1, 3).is_ok());

        let t = identity(2);
        let gp = vec![0.0, 0.5, -0.5, 0.0];
        let stepped = cayley_step(&t, &gp, 2, 0.2).unwrap();
        assert!(orthogonality_error(&stepped, 2) < 1e-10);

        let a = vec![0.8, 0.2, 0.1, 0.7];
        let (pattern, _) = oblique_pattern(&a, 2, 2, &t).unwrap();
        assert_eq!(pattern, a);
        assert_eq!(factor_correlation(&t, 2), t);
    }

    #[test]
    fn orthonormalization_rejects_rank_deficiency() {
        let mut rank_one = vec![1.0, 1.0, 1.0, 1.0];
        assert!(orthonormalize_columns(&mut rank_one, 2, 2).is_err());
        let mut invalid = vec![f64::NAN, 0.0, 0.0, 1.0];
        assert!(normalize_columns(&mut invalid, 2, 2).is_err());
    }
}
