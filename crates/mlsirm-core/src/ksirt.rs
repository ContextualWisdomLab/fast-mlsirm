//! Kernel-smoothing nonparametric item response theory (Ramsay-style OCCs).
//!
//! Estimates option characteristic curves (OCCs) by Nadaraya-Watson kernel
//! regression of option indicators on rank-based ordinal ability surrogates,
//! the approach popularized by Ramsay (1991, as cited in Mazza et al., 2014)
//! and implemented in TestGraf and the R package KernSmoothIRT.
//!
//! # Verified sources (citation discipline)
//!
//! Every formula below was verified against sources actually read:
//!
//! - Mazza, Punzo, and McGuire (2014), Sections 2, 2.1, 2.2, 2.3 (full PDF
//!   read): rank transform `r_i = rank(t_i)/(n+1)`, ordinal ability
//!   `theta_i = F^{-1}(r_i)`, Nadaraya-Watson weights, Gaussian /
//!   quadratic / uniform kernels, Silverman bandwidth (their Eq. 5), and
//!   the expected item score `e_j(theta) = sum_l x_{jl} p_{jl}(theta)`.
//! - KernSmoothIRT 1.0.3 R/C++ source, read line by line
//!   (github.com/cran/KernSmoothIRT): `R/ksIRT.R` (ties.method="first",
//!   denominator `n+1`, grid endpoints `F^{-1}(1/(n+1))` to
//!   `F^{-1}(n/(n+1))` with 51 default points, `h = 1.06 * sigma * n^{-1/5}`
//!   with `sigma = 1` for the normal ability metric) and
//!   `src/smoother3.cpp` (per-subject NW smoothing; zero-denominator
//!   fallback returns all-zero weights).
//!
//! Ramsay (1991) itself was NOT obtainable and is cited only through Mazza
//! et al. (2014); no formula here is attributed to it directly.
//!
//! # Deliberate scope reductions and divergences
//!
//! - Pointwise standard errors are omitted: the R package's `stderr`
//!   accumulates `p(1-p)` from a *partially summed* running estimate
//!   (smoother3.cpp lines 148-150, order-dependent), and the JSS paper's
//!   Eq. 6 uses a different per-subject form; neither yields a closed form
//!   verifiable from the read sources, so v1 ships without SEs.
//! - Cross-validation bandwidth selection is omitted (the R implementation
//!   subsamples 10% of subjects at random, making it nondeterministic).
//! - Option lists are reported in ascending score order, whereas R keeps
//!   first-seen order; the estimated curves are unaffected.
//! - Responses must be complete and pre-scored (numeric option scores);
//!   missing-data handling, answer keys, DIF groups, and non-normal ability
//!   metrics are out of scope for v1.
//!
//! # References
//!
//! Mazza, A., Punzo, A., & McGuire, B. (2014). KernSmoothIRT: An R package
//! for kernel smoothing in item response theory. *Journal of Statistical
//! Software, 58*(6), 1-34. https://doi.org/10.18637/jss.v058.i06
//!
//! Nadaraya, E. A. (1964). On estimating regression. *Theory of Probability
//! & Its Applications, 9*(1), 141-142. (As cited in Mazza et al., 2014.)
//!
//! Ramsay, J. O. (1991). Kernel smoothing approaches to nonparametric item
//! characteristic curve estimation. *Psychometrika, 56*(4), 611-630.
//! https://doi.org/10.1007/BF02294494 (As cited in Mazza et al., 2014.)
//!
//! Silverman, B. W. (1986). *Density estimation for statistics and data
//! analysis*. Chapman & Hall. (As cited in Mazza et al., 2014.)
//!
//! Watson, G. S. (1964). Smooth regression analysis. *Sankhya A, 26*(4),
//! 359-372. (As cited in Mazza et al., 2014.)

use crate::mokken::normal_upper_quantile;

/// Kernel function for the Nadaraya-Watson smoother.
///
/// Formulas verified against Mazza et al. (2014, Section 2) and
/// smoother3.cpp: Gaussian `exp(-u^2/2)`, quadratic `(1-u^2)` on `[-1,1]`,
/// uniform indicator on `[-1,1]`. Multiplicative kernel constants cancel in
/// the NW normalization, so the unnormalized forms match the R package.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KsirtKernel {
    Gaussian,
    Quadratic,
    Uniform,
}

impl KsirtKernel {
    fn eval(self, u: f64) -> f64 {
        match self {
            KsirtKernel::Gaussian => (-0.5 * u * u).exp(),
            KsirtKernel::Quadratic => {
                if u.abs() <= 1.0 {
                    1.0 - u * u
                } else {
                    0.0
                }
            }
            KsirtKernel::Uniform => {
                if u.abs() <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

/// Per-item kernel-smoothing output.
#[derive(Debug, Clone)]
pub struct KsirtItem {
    /// Distinct observed option scores, ascending.
    pub options: Vec<f64>,
    /// `options.len() x grid.len()` option characteristic curves; row `l`
    /// gives `p_hat_{jl}(theta_s)` over the evaluation grid. Rows sum to 1
    /// at any grid point with a positive weight denominator and to 0 where
    /// all kernel weights vanish (compact-support kernels far from data).
    pub occ: Vec<Vec<f64>>,
    /// Expected item score curve `sum_l x_{jl} * p_hat_{jl}(theta_s)`.
    pub expected: Vec<f64>,
}

/// Result of [`ksirt`].
#[derive(Debug, Clone)]
pub struct KsirtResult {
    /// Ordinal ability surrogates `Phi^{-1}(rank(t_i)/(n+1))`, subject order.
    pub theta: Vec<f64>,
    /// Evaluation grid (equally spaced, `Phi^{-1}(1/(n+1))` to
    /// `Phi^{-1}(n/(n+1))`).
    pub grid: Vec<f64>,
    /// Per-item bandwidths actually used.
    pub bandwidth: Vec<f64>,
    /// Per-item OCC estimates.
    pub items: Vec<KsirtItem>,
    /// Expected total score curve (sum of per-item expected curves).
    pub expected_total: Vec<f64>,
}

/// Kernel smoothing of option characteristic curves.
///
/// `x[i][j]` is the observed (pre-scored, finite) response of subject `i`
/// to item `j`; the distinct values of column `j` form the option set.
/// `bandwidth` overrides the Silverman default `1.06 * n^{-1/5}` (per-item
/// values, all > 0). See the module docs for the algorithm and sources.
pub fn ksirt(
    x: &[Vec<f64>],
    kernel: KsirtKernel,
    nevalpoints: usize,
    bandwidth: Option<&[f64]>,
) -> Result<KsirtResult, String> {
    let n = x.len();
    if n < 2 {
        return Err("ksirt requires at least 2 subjects".to_string());
    }
    let k = x[0].len();
    if k == 0 {
        return Err("ksirt requires at least 1 item".to_string());
    }
    for (i, row) in x.iter().enumerate() {
        if row.len() != k {
            return Err(format!(
                "ragged response matrix: row {i} has {} items, expected {k}",
                row.len()
            ));
        }
        for (j, &v) in row.iter().enumerate() {
            if !v.is_finite() {
                return Err(format!("non-finite response at subject {i}, item {j}"));
            }
        }
    }
    if nevalpoints < 2 {
        return Err("nevalpoints must be at least 2".to_string());
    }
    let h: Vec<f64> = match bandwidth {
        Some(b) => {
            if b.len() != k {
                return Err(format!(
                    "bandwidth length {} does not match {} items",
                    b.len(),
                    k
                ));
            }
            if b.iter().any(|&v| !(v > 0.0) || !v.is_finite()) {
                return Err("bandwidths must be finite and positive".to_string());
            }
            b.to_vec()
        }
        None => {
            // Silverman rule, sigma = 1 on the normal ability metric
            // (Mazza et al., 2014, Eq. 5; ksIRT.R lines 171-179).
            let hs = 1.06 * (n as f64).powf(-0.2);
            vec![hs; k]
        }
    };

    // Step 1: total scores -> ranks (ties by first occurrence, matching
    // R's ties.method="first"; ksIRT.R line 121) -> normal quantiles.
    let totals: Vec<f64> = x.iter().map(|row| row.iter().sum()).collect();
    let mut order: Vec<usize> = (0..n).collect();
    // stable sort keeps original subject order within ties => "first"
    order.sort_by(|&a, &b| totals[a].partial_cmp(&totals[b]).unwrap());
    let mut rank = vec![0usize; n];
    for (pos, &subj) in order.iter().enumerate() {
        rank[subj] = pos + 1;
    }
    let np1 = (n + 1) as f64;
    // Phi^{-1}(r) = normal_upper_quantile(1 - r): the helper returns z with
    // P(N(0,1) > z) = p, so upper tail 1-r gives the lower quantile at r.
    let theta: Vec<f64> = rank
        .iter()
        .map(|&r| normal_upper_quantile(1.0 - r as f64 / np1))
        .collect();

    // Step 2: evaluation grid (ksIRT.R lines 134-141).
    let lim1 = normal_upper_quantile(1.0 - 1.0 / np1);
    let lim2 = normal_upper_quantile(1.0 - n as f64 / np1);
    let q = nevalpoints;
    let step = (lim2 - lim1) / (q - 1) as f64;
    let grid: Vec<f64> = (0..q).map(|s| lim1 + step * s as f64).collect();

    // Steps 3-4: per grid point, NW weights shared across the item's
    // options (smoother3.cpp lines 76-154, incl. zero-denominator fallback).
    let mut items = Vec::with_capacity(k);
    let mut expected_total = vec![0.0; q];
    for j in 0..k {
        let mut options: Vec<f64> = x.iter().map(|row| row[j]).collect();
        options.sort_by(|a, b| a.partial_cmp(b).unwrap());
        options.dedup();
        let m = options.len();
        let mut occ = vec![vec![0.0; q]; m];
        let mut expected = vec![0.0; q];
        for s in 0..q {
            let kw: Vec<f64> = theta
                .iter()
                .map(|&t| kernel.eval((grid[s] - t) / h[j]))
                .collect();
            let denom: f64 = kw.iter().sum();
            if denom <= 0.0 {
                continue; // all weights zero: occ stays 0 (R fallback)
            }
            for (i, &w) in kw.iter().enumerate() {
                let l = options
                    .iter()
                    .position(|&o| o == x[i][j])
                    .expect("option present by construction");
                occ[l][s] += w / denom;
            }
            for l in 0..m {
                expected[s] += options[l] * occ[l][s];
            }
            expected_total[s] += expected[s];
        }
        items.push(KsirtItem {
            options,
            occ,
            expected,
        });
    }

    Ok(KsirtResult {
        theta,
        grid,
        bandwidth: h,
        items,
        expected_total,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/ksirt_tests.rs"]
mod tests;
