//! Minimum-residual (minres/ULS) exploratory factor analysis and
//! McDonald's omega_total for the unidimensional (1-factor) case.
//!
//! # Sources actually read
//!
//! - READ: Revelle's CRAN psych `R/fa.R` (fetched from
//!   raw.githubusercontent.com/cran/psych/master/R/fa.R, 2026-07-23), the
//!   minres path: `fit.residuals`, `fit`, `FAgr.minres`, `FAout.wls`, and
//!   the smc-based start values. Transcribed line by line.
//! - READ (prior work in this crate): psych `smc.R` conventions, already
//!   mirrored by `reliability::invert_symmetric` + clamping for lambda6.
//! - NOT READ: McDonald (1999). The omega_total formula below is
//!   hand-derived from the unidimensional standardized factor model
//!   (derivation in the docs of [`omega_total_1f_corr`]) and matches what
//!   secondary sources attribute to McDonald; we cite it as
//!   "as cited in Revelle, 2025".
//!
//! # Algorithm (psych fa.R, minres path)
//!
//! Optimization variable: uniquenesses `psi in [0.005, 1]^p` (correlation
//! metric). Start: `1 - smc_j`. Objective: replace `diag(S)` by `1 - psi`,
//! eigendecompose, clamp eigenvalues `< eps` to `100*eps`, form the
//! rank-`nf` model `Lambda Lambda'` from the top eigenpairs, and sum the
//! squared strictly-lower-triangle residuals of `S - model`
//! (`fit.residuals`, `fm = "minres"`).
//!
//! Search direction: psych's `FAgr.minres`,
//! `g = diag(Lambda Lambda' + diag(psi) - S)` with eigenvalue clamp
//! `max(., 0)`. VERIFIED LIMITATION (adversarial spec review): this is NOT
//! the exact analytic gradient of the lower-triangle objective —
//! finite-difference signs can disagree. It is therefore used only as a
//! cheap search direction inside a projected Barzilai-Borwein descent with
//! an Armijo safeguard and a central finite-difference fallback direction.
//! Convergence is asserted on the ACTUAL objective via a finite-difference
//! box-KKT check (interior: `|fd_j| <= tol`; at the lower bound:
//! `fd_j >= -tol`; at the upper bound: `fd_j <= tol`); the maximum
//! violation is returned as [`MinresFaResult::kkt_violation`] so tests can
//! read it.
//!
//! Final loadings (`FAout.wls`): eigendecompose `S - diag(psi_hat)`,
//! `Lambda = V_{1:nf} diag(sqrt(max(lambda_k, 0)))`, columns in descending
//! eigenvalue order. Sign convention (ours, documented): each column is
//! flipped so its sum is `>= 0`. Loadings are orthogonal (unrotated); no
//! rotation of any kind is implemented.
//!
//! # Divergences from psych (documented)
//!
//! - Optimizer: psych uses `optim(..., method = "L-BFGS-B")`; we use
//!   projected BB descent + Armijo as above. Tests assert agreement with an
//!   independent scipy L-BFGS-B transcription oracle (same algorithm family
//!   as R's optim; NOT claimed bit-identical to any R run) at 1e-5.
//! - Upper bound: psych's `upper = max(smc, 1)` can exceed 1 for a
//!   covariance input; only correlation input is supported here, so the
//!   box is `[0.005, 1]`.
//! - No rotation, no factor scores, no ML/WLS/GLS/minchi methods, no
//!   omega_hierarchical (would need oblimin + Schmid-Leiman; out of scope
//!   per the REDUCED-SCOPE spec decision).
//!
//! # References (APA 7th ed.)
//!
//! Revelle, W. (2025). *psych: Procedures for psychological, psychometric,
//! and personality research* (Version 2.6.5) [R package].
//! https://CRAN.R-project.org/package=psych
//!
//! McDonald, R. P. (1999). *Test theory: A unified treatment*. Erlbaum.
//! (As cited in Revelle, 2025; not read.)

use crate::parallel::correlation_matrix;
use crate::reliability::invert_symmetric;

const LOWER: f64 = 0.005;
const UPPER: f64 = 1.0;
const KKT_TOL: f64 = 1e-6;
const MAX_ITER: usize = 2000;
const JACOBI_MAX_SWEEPS: usize = 128;
const JACOBI_TOL: f64 = 1e-12;

/// Minres factor-analysis output. `loadings` is `p x nf` row-major,
/// unrotated, columns in descending-eigenvalue order, column sums >= 0.
#[derive(Debug, Clone)]
pub struct MinresFaResult {
    pub loadings: Vec<f64>,
    /// Estimated uniquenesses `psi_hat` (box-constrained to [0.005, 1]).
    pub uniquenesses: Vec<f64>,
    /// Communalities `h2_j = sum_k loading_{jk}^2`.
    pub communalities: Vec<f64>,
    /// Final value of the minres objective (sum of squared strictly
    /// lower-triangle residuals).
    pub objective: f64,
    /// Maximum finite-difference box-KKT violation of the actual objective
    /// at the solution (see module docs). Small (< ~1e-6) iff converged.
    pub kkt_violation: f64,
    pub n_iter: usize,
    pub converged: bool,
}

/// McDonald's omega_total for a 1-factor minres solution.
#[derive(Debug, Clone)]
pub struct OmegaResult {
    /// `(sum lambda)^2 / ((sum lambda)^2 + sum psi)`.
    pub omega_total: f64,
    pub fa: MinresFaResult,
}

/// Symmetric Jacobi eigendecomposition returning descending eigenvalues
/// with matching eigenvectors (columns of `v`, row-major `p x p`). Same
/// sweep scheme as `parallel::symmetric_eigenvalues_desc`, extended with
/// vector accumulation.
fn symmetric_eigen_desc(matrix: &[f64], p: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
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
            let mut idx: Vec<usize> = (0..p).collect();
            idx.sort_by(|&x, &y| {
                a[y * p + y]
                    .partial_cmp(&a[x * p + x])
                    .expect("eigenvalues are finite")
            });
            let ev: Vec<f64> = idx.iter().map(|&i| a[i * p + i]).collect();
            let mut vs = vec![0.0; p * p];
            for (col, &i) in idx.iter().enumerate() {
                for r in 0..p {
                    vs[r * p + col] = v[r * p + i];
                }
            }
            return Ok((ev, vs));
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

/// Top-`nf` loadings from `S` with diagonal replaced by `1 - psi`
/// (psych `fit.residuals` eigen step; eigenvalue floor `100 * eps`).
fn loadings_at(
    s: &[f64],
    psi: &[f64],
    p: usize,
    nf: usize,
    floor_eps: bool,
) -> Result<Vec<f64>, String> {
    let mut sstar = s.to_vec();
    for j in 0..p {
        sstar[j * p + j] = 1.0 - psi[j];
    }
    let (ev, vecs) = symmetric_eigen_desc(&sstar, p)?;
    let mut lambda = vec![0.0; p * nf];
    for k in 0..nf {
        let e = if floor_eps {
            if ev[k] < f64::EPSILON {
                100.0 * f64::EPSILON
            } else {
                ev[k]
            }
        } else {
            ev[k].max(0.0)
        };
        let root = e.sqrt();
        for r in 0..p {
            lambda[r * nf + k] = vecs[r * p + k] * root;
        }
    }
    Ok(lambda)
}

/// psych `fit.residuals` (fm = "minres"): sum of squared strictly
/// lower-triangle residuals of `S - Lambda Lambda'`.
fn objective(s: &[f64], psi: &[f64], p: usize, nf: usize) -> Result<f64, String> {
    let l = loadings_at(s, psi, p, nf, true)?;
    let mut f = 0.0;
    for i in 1..p {
        for j in 0..i {
            let mut m = 0.0;
            for k in 0..nf {
                m += l[i * nf + k] * l[j * nf + k];
            }
            let r = s[i * p + j] - m;
            f += r * r;
        }
    }
    Ok(f)
}

/// psych `FAgr.minres` search direction (NOT the exact gradient of
/// `objective`; see module docs).
fn psych_direction(s: &[f64], psi: &[f64], p: usize, nf: usize) -> Result<Vec<f64>, String> {
    let l = loadings_at(s, psi, p, nf, false)?;
    let mut g = vec![0.0; p];
    for j in 0..p {
        let mut m = 0.0;
        for k in 0..nf {
            m += l[j * nf + k] * l[j * nf + k];
        }
        g[j] = m + psi[j] - s[j * p + j];
    }
    Ok(g)
}

/// Central finite-difference gradient of the actual objective.
fn fd_gradient(s: &[f64], psi: &[f64], p: usize, nf: usize) -> Result<Vec<f64>, String> {
    let mut g = vec![0.0; p];
    let mut w = psi.to_vec();
    for j in 0..p {
        let h = 1e-7 * psi[j].abs().max(1.0);
        let orig = w[j];
        w[j] = orig + h;
        let fp = objective(s, &w, p, nf)?;
        w[j] = orig - h;
        let fm = objective(s, &w, p, nf)?;
        w[j] = orig;
        g[j] = (fp - fm) / (2.0 * h);
    }
    Ok(g)
}

/// Max finite-difference box-KKT violation at `psi` (see module docs).
fn kkt_violation(s: &[f64], psi: &[f64], p: usize, nf: usize) -> Result<f64, String> {
    let g = fd_gradient(s, psi, p, nf)?;
    let mut worst = 0.0_f64;
    for j in 0..p {
        let at_lower = psi[j] <= LOWER + 1e-10;
        let at_upper = psi[j] >= UPPER - 1e-10;
        let v = if at_lower {
            (-g[j]).max(0.0)
        } else if at_upper {
            g[j].max(0.0)
        } else {
            g[j].abs()
        };
        worst = worst.max(v);
    }
    Ok(worst)
}

fn clamp_box(v: f64) -> f64 {
    v.clamp(LOWER, UPPER)
}

fn validate_corr(s: &[f64], p: usize, what: &str) -> Result<(), String> {
    if p < 3 {
        return Err(format!("{what}: need at least 3 variables"));
    }
    let pp = p
        .checked_mul(p)
        .ok_or_else(|| format!("{what}: dimension overflow (p = {p})"))?;
    if s.len() != pp {
        return Err(format!(
            "{what}: expected a {p} x {p} correlation matrix, got length {}",
            s.len()
        ));
    }
    if s.iter().any(|v| !v.is_finite()) {
        return Err(format!("{what}: correlation matrix must be finite"));
    }
    for i in 0..p {
        if (s[i * p + i] - 1.0).abs() > 1e-8 {
            return Err(format!(
                "{what}: diagonal must be 1 (correlation metric); got {} at {i}",
                s[i * p + i]
            ));
        }
        for j in 0..i {
            if (s[i * p + j] - s[j * p + i]).abs() > 1e-8 {
                return Err(format!("{what}: matrix must be symmetric"));
            }
        }
    }
    Ok(())
}

/// Minres factor analysis of a correlation matrix (psych fa.R, fm =
/// "minres"; see module docs for the transcription and divergences).
/// `corr` is row-major `p x p`; `1 <= nf < p`.
pub fn minres_fa_corr(corr: &[f64], p: usize, nf: usize) -> Result<MinresFaResult, String> {
    validate_corr(corr, p, "minres_fa")?;
    if nf == 0 || nf >= p {
        return Err("minres_fa: need 1 <= n_factors < n_variables".to_string());
    }

    // Start psi = 1 - smc (psych: diag(S) - smc(S)); errors on a singular
    // correlation matrix, matching the guttman_lambdas precedent.
    let rinv = invert_symmetric(corr, p)
        .map_err(|e| format!("minres_fa: smc start values need an invertible matrix: {e}"))?;
    let mut psi: Vec<f64> = (0..p)
        .map(|j| {
            let smc = (1.0 - 1.0 / rinv[j * p + j]).clamp(0.0, 1.0);
            clamp_box(1.0 - smc)
        })
        .collect();

    let mut f = objective(corr, &psi, p, nf)?;
    let mut g = psych_direction(corr, &psi, p, nf)?;
    let mut prev_psi = psi.clone();
    let mut prev_g = g.clone();
    let mut step = 1.0_f64;
    let mut n_iter = 0;
    for iter in 0..MAX_ITER {
        n_iter = iter + 1;
        // Armijo backtracking along the projected psych direction; central
        // FD direction as fallback when the psych direction fails (it is
        // not the exact gradient — see module docs).
        let mut accepted = false;
        for pass in 0..2 {
            let dir = if pass == 0 {
                g.clone()
            } else {
                fd_gradient(corr, &psi, p, nf)?
            };
            let mut t = step;
            for _ in 0..60 {
                let trial: Vec<f64> = (0..p).map(|j| clamp_box(psi[j] - t * dir[j])).collect();
                let moved: f64 = (0..p).map(|j| (trial[j] - psi[j]).abs()).sum();
                if moved <= 1e-15 {
                    break;
                }
                let ft = objective(corr, &trial, p, nf)?;
                let decrease: f64 = (0..p).map(|j| dir[j] * (psi[j] - trial[j])).sum();
                if ft <= f - 1e-4 * decrease.max(0.0) && ft < f {
                    prev_psi.copy_from_slice(&psi);
                    prev_g.copy_from_slice(&g);
                    psi = trial;
                    f = ft;
                    accepted = true;
                    break;
                }
                t *= 0.5;
            }
            if accepted {
                break;
            }
        }
        if !accepted {
            break;
        }
        g = psych_direction(corr, &psi, p, nf)?;
        // Barzilai-Borwein step for the next iteration.
        let mut ss = 0.0;
        let mut sy = 0.0;
        for j in 0..p {
            let sj = psi[j] - prev_psi[j];
            ss += sj * sj;
            sy += sj * (g[j] - prev_g[j]);
        }
        step = if sy.abs() > 1e-300 {
            (ss / sy).abs().clamp(1e-8, 1e4)
        } else {
            1.0
        };
        let moved: f64 = (0..p).map(|j| (psi[j] - prev_psi[j]).abs()).sum();
        if moved < 1e-13 {
            break;
        }
    }

    let kkt = kkt_violation(corr, &psi, p, nf)?;
    let loadings_raw = loadings_at(corr, &psi, p, nf, false)?;
    // Sign convention: column sums >= 0.
    let mut loadings = loadings_raw;
    for k in 0..nf {
        let colsum: f64 = (0..p).map(|r| loadings[r * nf + k]).sum();
        if colsum < 0.0 {
            for r in 0..p {
                loadings[r * nf + k] = -loadings[r * nf + k];
            }
        }
    }
    let communalities: Vec<f64> = (0..p)
        .map(|r| (0..nf).map(|k| loadings[r * nf + k].powi(2)).sum())
        .collect();
    Ok(MinresFaResult {
        loadings,
        uniquenesses: psi,
        communalities,
        objective: f,
        kkt_violation: kkt,
        n_iter,
        converged: kkt < KKT_TOL,
    })
}

/// Minres factor analysis of raw data (`n x p` row-major); computes the
/// Pearson correlation matrix first (same helper as parallel analysis).
pub fn minres_fa_data(
    data: &[f64],
    n: usize,
    p: usize,
    nf: usize,
) -> Result<MinresFaResult, String> {
    let np = n
        .checked_mul(p)
        .ok_or_else(|| format!("minres_fa: dimension overflow (n = {n}, p = {p})"))?;
    if data.len() != np {
        return Err(format!(
            "minres_fa: data length {} does not match n * p = {np}",
            data.len(),
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("minres_fa: data must be finite (complete data required)".into());
    }
    if n < 3 {
        return Err("minres_fa: need at least 3 observations".into());
    }
    let r = correlation_matrix(data, n, p)?;
    minres_fa_corr(&r, p, nf)
}

/// McDonald's omega_total for the unidimensional case, from a 1-factor
/// minres fit of a correlation matrix.
///
/// Hand-derivation (self-contained; McDonald, 1999, as cited in Revelle,
/// 2025 — original not read): standardized 1-factor model
/// `X_j = lambda_j F + e_j` with `Var(F) = 1`, independent errors
/// `Var(e_j) = psi_j`. The common-score part of the unit-weight total is
/// `(sum_j lambda_j) F` with variance `(sum lambda)^2`; the error variance
/// is `sum psi`. Hence
///
/// ```text
/// omega_total = (sum lambda)^2 / ((sum lambda)^2 + sum psi)
/// ```
pub fn omega_total_1f_corr(corr: &[f64], p: usize) -> Result<OmegaResult, String> {
    let fa = minres_fa_corr(corr, p, 1)?;
    let lsum: f64 = fa.loadings.iter().sum();
    let psum: f64 = fa.uniquenesses.iter().sum();
    let den = lsum * lsum + psum;
    let omega_total = if den <= 1e-12 {
        f64::NAN
    } else {
        lsum * lsum / den
    };
    Ok(OmegaResult { omega_total, fa })
}

/// [`omega_total_1f_corr`] from raw data (`n x p` row-major).
pub fn omega_total_1f_data(data: &[f64], n: usize, p: usize) -> Result<OmegaResult, String> {
    let np = n
        .checked_mul(p)
        .ok_or_else(|| format!("omega_total_1f: dimension overflow (n = {n}, p = {p})"))?;
    if data.len() != np {
        return Err(format!(
            "omega_total_1f: data length {} does not match n * p = {np}",
            data.len(),
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("omega_total_1f: data must be finite".into());
    }
    if n < 3 {
        return Err("omega_total_1f: need at least 3 observations".into());
    }
    let r = correlation_matrix(data, n, p)?;
    omega_total_1f_corr(&r, p)
}

#[cfg(test)]
#[path = "../../../tests/unit/factor_tests.rs"]
mod tests;
