//! Horn's parallel analysis for principal-component retention.
//!
//! Compares the eigenvalues of the Pearson correlation matrix of an observed
//! `n x p` data matrix against the mean (or a Glorfeld upper-centile) of
//! eigenvalues obtained from correlation matrices of `n_iterations` random
//! standard-normal data sets of the same shape. Each observed eigenvalue is
//! adjusted by the sampling bias `random_eigenvalue - 1`, and components are
//! retained while the adjusted eigenvalue stays above 1 (left-to-right scan;
//! the first failure stops retention even if later adjusted values rise back
//! above 1, matching the oracle's loop-and-break logic).
//!
//! # Verified sources
//!
//! - CRAN `paran` 1.5.6 R source (Dinno, READ line by line: `paran.R`): the
//!   oracle for the full pipeline — observed correlation eigenvalues, random
//!   benchmark (`RndEv` = per-position mean, or `quantile(..., centile/100)`
//!   which is R type-7), `Bias = RndEv - 1`, `AdjEv = Ev - Bias`, and the
//!   retention scan (`for` loop with `break` at the first `AdjEv <= 1`).
//! - `paran` man page (READ): documents the PCA adjustment
//!   `lambda_p - (mean lambda^r_p - 1)` and the "adjusted eigenvalue > 1"
//!   retention criterion; attributes the method to Horn (1965) and the
//!   centile variant to Glorfeld (1995).
//! - Horn (1965) and Glorfeld (1995) themselves were NOT read (paywalled);
//!   all attribution is as cited in and implemented by `paran`.
//!
//! # Divergences from the paran oracle (deliberate, documented)
//!
//! 1. **PCA path only.** paran's `cfa = TRUE` common-factor path (which
//!    subtracts a generalized-inverse diagonal from `R`) is out of scope.
//! 2. **RNG.** Random data sets are drawn from this crate's deterministic
//!    LCG + Box-Muller idiom in a single stream seeded once; results are
//!    paran-inspired but NOT bit-identical to any R run. (paran's own
//!    `set.seed(seed*k)` at `paran.R` line 35 references the loop index `k`
//!    before it is defined, so the man page's per-iteration reseeding claim
//!    is not faithfully implemented by the oracle either.)
//! 3. **Guards narrowed.** The oracle feeds `cor()`/`eigen()` whatever it is
//!    given; this implementation rejects `n_persons < 3`, `n_items < 2`,
//!    non-finite cells, zero-variance columns, `n_iterations == 0`, and
//!    `centile > 99` with explicit errors.
//! 4. `iterations = 0` does NOT default to `30 * p` here; the core is
//!    explicit and callers supply the default.
//!
//! Eigenvalues are computed with a cyclic Jacobi sweep (eigenvalues only);
//! non-convergence after the sweep cap is a hard error, never a silent
//! truncation.
//!
//! # References
//!
//! Dinno, A. (2018). *paran: Horn's test of principal components/factors*
//! (Version 1.5.6) \[R package\]. CRAN.
//! <https://CRAN.R-project.org/package=paran>
//!
//! Glorfeld, L. W. (1995). An improvement on Horn's parallel analysis
//! methodology for selecting the correct number of factors to retain.
//! *Educational and Psychological Measurement, 55*(3), 377–393.
//! <https://doi.org/10.1177/0013164495055003002> (not read; as cited in
//! Dinno, 2018)
//!
//! Horn, J. L. (1965). A rationale and a test for the number of factors in
//! factor analysis. *Psychometrika, 30*(2), 179–185.
//! <https://doi.org/10.1007/BF02289447> (not read; as cited in Dinno, 2018)

/// Outputs of Horn's parallel analysis, all vectors of length `n_items` in
/// descending observed-eigenvalue order.
#[derive(Debug, Clone)]
pub struct ParallelAnalysisResult {
    /// Number of components retained (adjusted eigenvalue > 1 until the
    /// first failure).
    pub retained: usize,
    /// Eigenvalues of the observed correlation matrix, descending.
    pub eigenvalues: Vec<f64>,
    /// Random-data benchmark eigenvalues (mean or centile per position).
    pub random_eigenvalues: Vec<f64>,
    /// `random_eigenvalues - 1`, the estimated sampling bias.
    pub bias: Vec<f64>,
    /// `eigenvalues - bias`.
    pub adjusted_eigenvalues: Vec<f64>,
}

const JACOBI_MAX_SWEEPS: usize = 100;
const JACOBI_TOL: f64 = 1e-12;

/// Horn's parallel analysis (PCA path of `paran`, see module docs).
///
/// * `data` — row-major `n_persons x n_items` matrix.
/// * `n_iterations` — number of random data sets (must be >= 1; callers
///   wanting the paran default should pass `30 * n_items`).
/// * `centile` — 0 for the mean benchmark (Horn, as implemented by paran),
///   or 1..=99 for Glorfeld's upper-centile variant (R type-7 quantile).
/// * `seed` — LCG seed for the random benchmark (0 is mapped to 1).
pub fn parallel_analysis(
    data: &[f64],
    n_persons: usize,
    n_items: usize,
    n_iterations: usize,
    centile: u32,
    seed: u64,
) -> Result<ParallelAnalysisResult, String> {
    if n_persons < 3 {
        return Err("parallel analysis needs n_persons >= 3".into());
    }
    if n_items < 2 {
        return Err("parallel analysis needs n_items >= 2".into());
    }
    if n_iterations == 0 {
        return Err("n_iterations must be >= 1".into());
    }
    if centile > 99 {
        return Err("centile must be 0 (mean) or in 1..=99".into());
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("data dimensions overflow usize")?;
    if data.len() != cells {
        return Err(format!(
            "data length {} does not match n_persons * n_items = {cells}",
            data.len()
        ));
    }
    if data.iter().any(|v| !v.is_finite()) {
        return Err("data must be finite (no NaN/inf; complete data required)".into());
    }

    let corr = correlation_matrix(data, n_persons, n_items)?;
    let eigenvalues = symmetric_eigenvalues_desc(&corr, n_items)?;

    // Random benchmark: n_iterations standard-normal data sets from a single
    // deterministic LCG stream (crate idiom; see module docs, divergence 2).
    let mut state = seed.max(1);
    let sim_len = n_iterations
        .checked_mul(n_items)
        .ok_or("n_iterations * n_items overflows usize")?;
    let mut sim = vec![0.0_f64; sim_len];
    let mut rand_data = vec![0.0_f64; cells];
    for k in 0..n_iterations {
        for cell in rand_data.iter_mut() {
            *cell = normal_draw(&mut state);
        }
        let rc = correlation_matrix(&rand_data, n_persons, n_items)
            .map_err(|e| format!("random benchmark iteration {k}: {e}"))?;
        let ev = symmetric_eigenvalues_desc(&rc, n_items)
            .map_err(|e| format!("random benchmark iteration {k}: {e}"))?;
        sim[k * n_items..(k + 1) * n_items].copy_from_slice(&ev);
    }

    let mut random_eigenvalues = vec![0.0_f64; n_items];
    let mut column = vec![0.0_f64; n_iterations];
    for q in 0..n_items {
        for k in 0..n_iterations {
            column[k] = sim[k * n_items + q];
        }
        random_eigenvalues[q] = if centile == 0 {
            column.iter().sum::<f64>() / n_iterations as f64
        } else {
            type7_quantile(&mut column, f64::from(centile) / 100.0)
        };
    }

    let bias: Vec<f64> = random_eigenvalues.iter().map(|r| r - 1.0).collect();
    let adjusted_eigenvalues: Vec<f64> =
        eigenvalues.iter().zip(&bias).map(|(e, b)| e - b).collect();
    let retained = retained_count(&adjusted_eigenvalues);

    Ok(ParallelAnalysisResult {
        retained,
        eigenvalues,
        random_eigenvalues,
        bias,
        adjusted_eigenvalues,
    })
}

/// Left-to-right retention scan (paran.R lines 250-267): retain components
/// until the first adjusted eigenvalue <= 1; later resurgences do not count.
pub(crate) fn retained_count(adjusted: &[f64]) -> usize {
    for (i, a) in adjusted.iter().enumerate() {
        if *a <= 1.0 {
            return i;
        }
    }
    adjusted.len()
}

/// Pearson correlation matrix of the columns of a row-major `n x p` matrix.
pub(crate) fn correlation_matrix(data: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    let mut means = vec![0.0_f64; p];
    for row in 0..n {
        for (j, m) in means.iter_mut().enumerate() {
            *m += data[row * p + j];
        }
    }
    for m in means.iter_mut() {
        *m /= n as f64;
    }
    // Column sums of squared deviations.
    let mut ss = vec![0.0_f64; p];
    for row in 0..n {
        for j in 0..p {
            let d = data[row * p + j] - means[j];
            ss[j] += d * d;
        }
    }
    for (j, s) in ss.iter().enumerate() {
        if !s.is_finite() || !means[j].is_finite() {
            return Err(format!(
                "column {j} magnitude overflows the correlation computation"
            ));
        }
        if *s <= 0.0 {
            return Err(format!(
                "column {j} has zero variance; Pearson correlation undefined"
            ));
        }
    }
    let mut corr = vec![0.0_f64; p * p];
    for i in 0..p {
        corr[i * p + i] = 1.0;
        for j in (i + 1)..p {
            let mut sxy = 0.0_f64;
            for row in 0..n {
                sxy += (data[row * p + i] - means[i]) * (data[row * p + j] - means[j]);
            }
            let r = sxy / (ss[i].sqrt() * ss[j].sqrt());
            if !r.is_finite() {
                return Err(format!(
                    "correlation of columns {i},{j} is not finite (data magnitude overflow)"
                ));
            }
            corr[i * p + j] = r;
            corr[j * p + i] = r;
        }
    }
    Ok(corr)
}

/// Eigenvalues of a symmetric `p x p` matrix by cyclic Jacobi rotations,
/// sorted descending. Errors if the off-diagonal has not converged below
/// `JACOBI_TOL` within `JACOBI_MAX_SWEEPS` sweeps.
fn symmetric_eigenvalues_desc(matrix: &[f64], p: usize) -> Result<Vec<f64>, String> {
    let mut a = matrix.to_vec();
    for _ in 0..JACOBI_MAX_SWEEPS {
        let mut off = 0.0_f64;
        for i in 0..p {
            for j in (i + 1)..p {
                off = off.max(a[i * p + j].abs());
            }
        }
        if off < JACOBI_TOL {
            let mut ev: Vec<f64> = (0..p).map(|i| a[i * p + i]).collect();
            ev.sort_by(|x, y| y.partial_cmp(x).expect("eigenvalues are finite"));
            return Ok(ev);
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
            }
        }
    }
    Err("Jacobi eigenvalue iteration did not converge".into())
}

/// R type-7 quantile of `values` (reordered in place by the sort).
fn type7_quantile(values: &mut [f64], prob: f64) -> f64 {
    values.sort_by(|x, y| x.partial_cmp(y).expect("eigenvalues are finite"));
    let h = (values.len() - 1) as f64 * prob;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        values[lo]
    } else {
        values[lo] + (h - lo as f64) * (values[hi] - values[lo])
    }
}

#[inline]
pub(crate) fn lcg_uniform(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// Box-Muller normal on LCG uniforms (crate idiom, mirrored in the NumPy
/// fixture script for the pinned test literals).
pub(crate) fn normal_draw(state: &mut u64) -> f64 {
    let u1 = lcg_uniform(state).max(1e-12);
    let u2 = lcg_uniform(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[cfg(test)]
#[path = "../../../tests/unit/parallel_tests.rs"]
mod tests;
