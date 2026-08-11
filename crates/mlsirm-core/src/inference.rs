//! Covariance and standard-error arithmetic for observed information matrices.
//!
//! Python wrappers must marshal only; inversion/pseudoinverse and SE extraction
//! stay on the Rust numeric path so inference contracts stay single-sourced.

/// Invert a square observed-information / Hessian matrix.
///
/// Attempts a partial-pivot Gauss–Jordan inverse; on singularity falls back to
/// a Moore–Penrose pseudoinverse via Jacobi eigen-decomposition with eigenvalue
/// cutoff `rcond * max(|λ|)`. The returned matrix is symmetrised.
pub fn vcov_from_hessian(hessian: &[f64], n: usize, rcond: f64) -> Result<Vec<f64>, String> {
    if n == 0 || hessian.len() != n * n {
        return Err("hessian must be a square matrix".into());
    }
    if !rcond.is_finite() || rcond < 0.0 {
        return Err("rcond must be a finite non-negative float".into());
    }
    // Non-finite observed information is scientifically undefined; fail closed
    // rather than emitting an uncontrolled covariance artifact.
    if hessian.iter().any(|v| !v.is_finite()) {
        return Err("hessian entries must be finite".into());
    }
    let mut inv = match invert_square(hessian, n) {
        Some(v) => v,
        None => pseudoinverse_symmetric(hessian, n, rcond)?,
    };
    // Symmetrise (H^{-1} may carry tiny asymmetric roundoff).
    for i in 0..n {
        for j in (i + 1)..n {
            let mean = 0.5 * (inv[i * n + j] + inv[j * n + i]);
            inv[i * n + j] = mean;
            inv[j * n + i] = mean;
        }
    }
    Ok(inv)
}

/// Standard errors from a covariance diagonal.
///
/// Finite positive diagonal entries become `sqrt(d)`. Finite non-positive
/// entries are clamped to `0.0` (negative numerical noise). Non-finite
/// diagonals (`NaN`, `±∞`) are preserved so undefined or unbounded uncertainty
/// is never misrepresented as zero.
pub fn standard_errors_from_vcov(vcov: &[f64], n: usize) -> Result<Vec<f64>, String> {
    if n == 0 || vcov.len() != n * n {
        return Err("vcov must be a square matrix".into());
    }
    let mut out = vec![0.0_f64; n];
    for i in 0..n {
        let d = vcov[i * n + i];
        out[i] = if !d.is_finite() {
            d
        } else if d > 0.0 {
            d.sqrt()
        } else {
            0.0
        };
    }
    Ok(out)
}

fn invert_square(matrix: &[f64], k: usize) -> Option<Vec<f64>> {
    let mut m = matrix.to_vec();
    let mut inv = vec![0.0_f64; k * k];
    for i in 0..k {
        inv[i * k + i] = 1.0;
    }
    for col in 0..k {
        let mut piv = col;
        for r in (col + 1)..k {
            if m[r * k + col].abs() > m[piv * k + col].abs() {
                piv = r;
            }
        }
        if m[piv * k + col].abs() < 1e-12 {
            return None;
        }
        if piv != col {
            for c in 0..k {
                m.swap(col * k + c, piv * k + c);
                inv.swap(col * k + c, piv * k + c);
            }
        }
        let d = m[col * k + col];
        for c in 0..k {
            m[col * k + c] /= d;
            inv[col * k + c] /= d;
        }
        for r in 0..k {
            if r != col {
                let f = m[r * k + col];
                if f != 0.0 {
                    for c in 0..k {
                        m[r * k + c] -= f * m[col * k + c];
                        inv[r * k + c] -= f * inv[col * k + c];
                    }
                }
            }
        }
    }
    Some(inv)
}

fn pseudoinverse_symmetric(matrix: &[f64], p: usize, rcond: f64) -> Result<Vec<f64>, String> {
    let (evals, evecs) = jacobi_symmetric_eigen(matrix, p)?;
    let max_abs = evals
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max)
        .max(1e-300);
    let cutoff = rcond * max_abs;
    // A+ = V diag(1/λ_i) V^T for |λ_i| > cutoff
    let mut inv = vec![0.0_f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for k in 0..p {
                let lam = evals[k];
                if lam.abs() > cutoff {
                    s += evecs[i * p + k] * (1.0 / lam) * evecs[j * p + k];
                }
            }
            inv[i * p + j] = s;
        }
    }
    Ok(inv)
}

fn jacobi_symmetric_eigen(matrix: &[f64], p: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    const JACOBI_MAX_SWEEPS: usize = 64;
    const JACOBI_TOL: f64 = 1e-14;
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; p * p];
    for i in 0..p {
        v[i * p + i] = 1.0;
    }
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0_f64;
        for i in 0..p {
            for j in (i + 1)..p {
                off = off.max(a[i * p + j].abs());
            }
        }
        if off < JACOBI_TOL {
            let mut evals = vec![0.0; p];
            for i in 0..p {
                evals[i] = a[i * p + i];
            }
            return Ok((evals, v));
        }
        for i in 0..p {
            for j in (i + 1)..p {
                let aij = a[i * p + j];
                if aij.abs() < JACOBI_TOL {
                    continue;
                }
                let theta = (a[j * p + j] - a[i * p + i]) / (2.0 * aij);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..p {
                    let aik = a[i * p + k];
                    let ajk = a[j * p + k];
                    a[i * p + k] = c * aik - s * ajk;
                    a[j * p + k] = s * aik + c * ajk;
                }
                for k in 0..p {
                    let aki = a[k * p + i];
                    let akj = a[k * p + j];
                    a[k * p + i] = c * aki - s * akj;
                    a[k * p + j] = s * aki + c * akj;
                }
                for k in 0..p {
                    let vki = v[k * p + i];
                    let vkj = v[k * p + j];
                    v[k * p + i] = c * vki - s * vkj;
                    v[k * p + j] = s * vki + c * vkj;
                }
            }
        }
    }
    Err("Jacobi eigenvalue iteration did not converge".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_diagonal_information() {
        let h = [4.0, 0.0, 0.0, 9.0];
        let v = vcov_from_hessian(&h, 2, 1e-10).unwrap();
        assert!((v[0] - 0.25).abs() < 1e-12);
        assert!((v[3] - 1.0 / 9.0).abs() < 1e-12);
        let se = standard_errors_from_vcov(&v, 2).unwrap();
        assert!((se[0] - 0.5).abs() < 1e-12);
        assert!((se[1] - 1.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn nonfinite_hessian_fails_closed() {
        for h in [
            [f64::NAN],
            [f64::INFINITY],
            [f64::NEG_INFINITY],
        ] {
            let err = vcov_from_hessian(&h, 1, 1e-10).unwrap_err();
            assert!(
                err.contains("finite"),
                "unexpected nonfinite hessian error: {err}"
            );
        }
    }

    #[test]
    fn standard_errors_preserve_nonfinite_diagonals() {
        // row-major 5x5 with targeted diagonal entries
        let mut v = vec![0.0_f64; 25];
        v[0] = 4.0; // SE=2
        v[6] = 0.0; // SE=0
        v[12] = -1.0; // clamp to 0
        v[18] = f64::NAN; // preserve NaN
        v[24] = f64::INFINITY; // preserve +inf
        let se = standard_errors_from_vcov(&v, 5).unwrap();
        assert!((se[0] - 2.0).abs() < 1e-15);
        assert_eq!(se[1], 0.0);
        assert_eq!(se[2], 0.0);
        assert!(se[3].is_nan());
        assert!(se[4].is_infinite() && se[4].is_sign_positive());
    }

    #[test]
    fn singular_matrix_uses_pseudoinverse_identity() {
        let h = [1.0, 1.0, 1.0, 1.0];
        let v = vcov_from_hessian(&h, 2, 1e-10).unwrap();
        // A A+ A ≈ A
        let mut recon = [0.0; 4];
        for i in 0..2 {
            for j in 0..2 {
                let mut s = 0.0;
                for k in 0..2 {
                    let mut t = 0.0;
                    for l in 0..2 {
                        t += h[i * 2 + l] * v[l * 2 + k];
                    }
                    s += t * h[k * 2 + j];
                }
                recon[i * 2 + j] = s;
            }
        }
        for i in 0..4 {
            assert!((recon[i] - h[i]).abs() < 1e-8, "recon={:?} h={:?}", recon, h);
        }
    }
}
