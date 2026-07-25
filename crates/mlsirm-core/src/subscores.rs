//! Haberman (2008) subscore added-value analysis via the proportional
//! reduction in mean squared error (PRMSE).
//!
//! Given a persons x items matrix of scored responses partitioned into `K`
//! disjoint, exhaustive subscales, this module computes for each subscale the
//! three Haberman estimators of the true subscore `s_t` and their PRMSEs:
//!
//! ```text
//! s_hat_s  = E(s) + rho_s (s - E(s))                       PRMSE_s  = rho_s
//! s_hat_x  = E(s) + sqrt(PRMSE_x) (sigma_t / sigma_x)(x - E(x))
//!                                                          PRMSE_x  = rho^2(s_t, x_t) rho_x
//! s_hat_sx = E(s) + beta (s - E(s)) + gamma (x - E(x))     PRMSE_sx = rho_s + tau^2 (1 - r^2)
//! ```
//!
//! with `rho_s`/`rho_x` the Cronbach-alpha reliabilities of the subscale and
//! the total test, `r = corr(s, x)`,
//! `rho^2(s_t, x_t) = cov_k^2 / (V(s_t) V(x_t))` where `cov_k` is the row sum
//! of the true-subscore covariance matrix over the `K` subscore columns
//! (observed covariances off the diagonal, `alpha * observed variance` on the
//! diagonal; the total-score column is EXCLUDED), and
//!
//! ```text
//! tau   = (sqrt(rho_x) sqrt(rho^2(s_t, x_t)) - r sqrt(rho_s)) / (1 - r^2)
//! beta  = sqrt(rho_s) (sqrt(rho_s) - r tau)
//! gamma = sqrt(rho_s) tau (sigma_s / sigma_x)
//! ```
//!
//! Added-value decisions: a subscore has added value iff
//! `PRMSE_s > PRMSE_x` (Haberman's rule); an augmented subscore has added
//! value iff `PRMSE_sx > max(PRMSE_s, PRMSE_x) + 0.01`, which is the
//! operational convention of Sinharay (2010) — the CRAN `subscore` package's
//! `CTTsub` uses a different relative rule (`0.1 * (1 - max)`) that is NOT
//! implemented here.
//!
//! # Verified sources
//!
//! Formulas were verified against (a) the Appendix of Sinharay (2010, ETS
//! RR-10-16), which reproduces the Haberman (2008) methodology, and (b) the
//! CRAN `subscore` package R source (`subscore.s.r`, `subscore.x.R`,
//! `subscore.sx.R`, `data.prep.R`) read line by line; the R code was used to
//! disambiguate Greek symbols lost in PDF extraction. Haberman (2008) itself
//! and Wainer et al. (2001) were NOT read (paywalled) and are cited only as
//! cited in Sinharay (2010).
//!
//! # Divergences from the R package (deliberate)
//!
//! - The partition is validated (every item in exactly one subscale, each
//!   subscale with >= 2 items); CRAN `data.prep()` silently allows totals
//!   that are not the union of the subscales.
//! - Degenerate inputs are rejected instead of propagating NaN or emitting a
//!   warning: any Cronbach alpha outside `(0, 1]`, any non-positive observed
//!   variance, `|corr(s_k, x)| >= 1 - 1e-12`, non-finite moments, or a
//!   computed PRMSE outside `[0, 1 + 1e-9]`.
//! - `s_hat_x` uses the nonnegative root `sqrt(PRMSE_x)` exactly as the R
//!   code does, even when the signed correlation form of the Sinharay
//!   appendix would be negative (i.e. when `cov(s_t, x_t) < 0`, which the
//!   guards do NOT rule out); this follows CRAN's convention.
//! - Missing data are not supported (the R code uses `na.rm`); v1 requires
//!   complete data.
//!
//! All moments are unbiased (`n - 1`), matching R's `var`/`cov`/`cor`.
//!
//! In LLM-as-a-Judge item-quality management this decides whether per-domain
//! judge subscores carry diagnostic information beyond the overall score, or
//! whether reporting them would be statistically misleading.
//!
//! # References (APA 7th ed.)
//!
//! Haberman, S. J. (2008). When can subscores have value? *Journal of
//! Educational and Behavioral Statistics, 33*(2), 204-229.
//! https://doi.org/10.3102/1076998607302636 (as cited in Sinharay, 2010)
//!
//! Sinharay, S. (2010). *When can subscores be expected to have added value?
//! Results from operational and simulated data* (ETS Research Rep. No.
//! RR-10-16). Educational Testing Service.
//!
//! Wainer, H., Vevea, J. L., Camacho, F., Reeve, B. B., Rosa, K., & Nelson,
//! L. (2001). Augmented scores — "borrowing strength" to compute scores based
//! on small numbers of items. In D. Thissen & H. Wainer (Eds.), *Test
//! scoring* (pp. 343-387). Lawrence Erlbaum. (as cited in Sinharay, 2010)

/// Result of the Haberman subscore added-value analysis. All vectors are
/// indexed by subscale `k = 0..K`; person-level estimator matrices are
/// `n_persons x K` in row-major nested `Vec`s.
#[derive(Debug, Clone)]
pub struct SubscoreResult {
    /// Cronbach alpha of each subscale ( = PRMSE_s).
    pub alpha: Vec<f64>,
    /// Cronbach alpha of the total test.
    pub alpha_total: f64,
    /// `(K+1) x (K+1)` correlation matrix of `(s_1..s_K, x)`, total last.
    pub corr: Vec<Vec<f64>>,
    /// `K x K` disattenuated subscore correlations
    /// `corr_kl / sqrt(alpha_k alpha_l)`; diagonal is NaN.
    pub disattenuated_corr: Vec<Vec<f64>>,
    /// PRMSE of the observed-subscore estimator ( = subscale reliability).
    pub prmse_s: Vec<f64>,
    /// PRMSE of the total-score estimator `rho^2(s_t, x_t) rho_x`.
    pub prmse_x: Vec<f64>,
    /// PRMSE of the augmented estimator `rho_s + tau^2 (1 - r^2)`.
    pub prmse_sx: Vec<f64>,
    /// Regression helpers for the augmented estimator.
    pub tau: Vec<f64>,
    pub beta: Vec<f64>,
    pub gamma: Vec<f64>,
    /// `PRMSE_s > PRMSE_x` (Haberman's added-value rule).
    pub added_value_s: Vec<bool>,
    /// `PRMSE_sx > max(PRMSE_s, PRMSE_x) + 0.01` (Sinharay 2010 convention).
    pub added_value_sx: Vec<bool>,
    /// Observed subscores `s_k` per person (`n x K`).
    pub observed: Vec<Vec<f64>>,
    /// Observed total score per person.
    pub total: Vec<f64>,
    /// Estimated true subscores from the observed subscore (`n x K`).
    pub subscore_s: Vec<Vec<f64>>,
    /// Estimated true subscores from the observed total (`n x K`).
    pub subscore_x: Vec<Vec<f64>>,
    /// Estimated true subscores from both (`n x K`).
    pub subscore_sx: Vec<Vec<f64>>,
}

/// Unbiased sample variance (`n - 1`); caller guarantees `v.len() >= 2`.
fn var(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    v.iter().map(|&a| (a - m) * (a - m)).sum::<f64>() / (n - 1.0)
}

/// Unbiased sample covariance (`n - 1`).
fn cov(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (
        a.iter().sum::<f64>() / n,
        b.iter().sum::<f64>() / n,
    );
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - ma) * (y - mb))
        .sum::<f64>()
        / (n - 1.0)
}

/// Cronbach alpha `m/(m-1) (1 - sum_j var(y_j) / var(sum_j y_j))` over the
/// given item columns. Same unbiased convention in numerator and denominator.
fn cronbach_alpha(x: &[Vec<f64>], items: &[usize]) -> f64 {
    let m = items.len() as f64;
    let totals: Vec<f64> = x
        .iter()
        .map(|row| items.iter().map(|&j| row[j]).sum())
        .collect();
    let item_var_sum: f64 = items
        .iter()
        .map(|&j| {
            let col: Vec<f64> = x.iter().map(|row| row[j]).collect();
            var(&col)
        })
        .sum();
    let tv = var(&totals);
    m / (m - 1.0) * (1.0 - item_var_sum / tv)
}

/// Haberman (2008) subscore added-value analysis (see module docs).
///
/// `x` is a complete `n_persons x n_items` matrix of finite scored responses;
/// `groups[j]` in `0..K` assigns item `j` to a subscale. The partition must be
/// exhaustive with at least 2 items per subscale, `n_persons >= 3`, and
/// `K >= 2`. Returns an error on any validation or degeneracy failure (see
/// the module-level guard list).
pub fn subscores(x: &[Vec<f64>], groups: &[usize]) -> Result<SubscoreResult, String> {
    let n = x.len();
    if n < 3 {
        return Err("subscores requires at least 3 persons".into());
    }
    let n_items = x[0].len();
    if groups.len() != n_items {
        return Err("groups must assign every item to a subscale".into());
    }
    for row in x {
        if row.len() != n_items {
            return Err("ragged response matrix".into());
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err("responses must be complete and finite".into());
        }
    }
    let k_count = match groups.iter().max() {
        Some(&g) => g + 1,
        None => return Err("groups must assign every item to a subscale".into()),
    };
    if k_count < 2 {
        return Err("at least 2 subscales are required".into());
    }
    // Bound BEFORE allocating items_of: a hostile sparse index (e.g. 10^9)
    // would otherwise drive a huge allocation. K <= n_items/2 given >= 2
    // items per subscale.
    if k_count > n_items / 2 {
        return Err("subscale indices must be dense: 0..K with K <= n_items / 2".into());
    }
    let mut items_of: Vec<Vec<usize>> = vec![Vec::new(); k_count];
    for (j, &g) in groups.iter().enumerate() {
        items_of[g].push(j);
    }
    if items_of.iter().any(|v| v.len() < 2) {
        return Err("every subscale needs at least 2 items".into());
    }

    // Observed subscores and total (the partition makes x = sum_k s_k).
    let observed: Vec<Vec<f64>> = x
        .iter()
        .map(|row| {
            items_of
                .iter()
                .map(|items| items.iter().map(|&j| row[j]).sum())
                .collect()
        })
        .collect();
    let total: Vec<f64> = observed.iter().map(|s| s.iter().sum()).collect();

    // Columns of the (K+1)-variate score matrix, total last.
    let mut cols: Vec<Vec<f64>> = (0..k_count)
        .map(|k| observed.iter().map(|s| s[k]).collect())
        .collect();
    cols.push(total.clone());
    let kk = k_count + 1;

    let all_items: Vec<usize> = (0..n_items).collect();
    let mut alpha: Vec<f64> = items_of
        .iter()
        .map(|items| cronbach_alpha(x, items))
        .collect();
    let alpha_total = cronbach_alpha(x, &all_items);
    alpha.push(alpha_total);
    for (i, &a) in alpha.iter().enumerate() {
        if !a.is_finite() || a <= 0.0 || a > 1.0 {
            return Err(format!(
                "Cronbach alpha of {} is {a:.6}, outside (0, 1]; the Haberman \
                 analysis is undefined (zero variance or negatively \
                 correlated items)",
                if i < k_count { "a subscale" } else { "the total test" }
            ));
        }
    }

    let var_obs: Vec<f64> = cols.iter().map(|c| var(c)).collect();
    if var_obs.iter().any(|&v| !(v.is_finite() && v > 0.0)) {
        return Err("zero-variance subscore or total score".into());
    }
    let mut c_obs = vec![vec![0.0f64; kk]; kk];
    for a in 0..kk {
        for b in a..kk {
            let v = cov(&cols[a], &cols[b]);
            c_obs[a][b] = v;
            c_obs[b][a] = v;
        }
    }
    let corr: Vec<Vec<f64>> = (0..kk)
        .map(|a| {
            (0..kk)
                .map(|b| c_obs[a][b] / (var_obs[a] * var_obs[b]).sqrt())
                .collect()
        })
        .collect();
    for k in 0..k_count {
        if corr[k][k_count].abs() >= 1.0 - 1e-12 {
            return Err(
                "a subscore is (numerically) collinear with the total score; \
                 the augmented regression is undefined"
                    .into(),
            );
        }
    }

    // True-score covariance matrix C_T: observed off-diagonals, diagonal
    // alpha_k * V(s_k). cov_k sums row k over the K subscore columns ONLY
    // (the total column is excluded).
    let var_true: Vec<f64> = (0..kk).map(|i| var_obs[i] * alpha[i]).collect();
    let cov_rowsum: Vec<f64> = (0..k_count)
        .map(|k| {
            (0..k_count)
                .map(|l| if l == k { var_true[k] } else { c_obs[k][l] })
                .sum()
        })
        .collect();

    let mut prmse_s = Vec::with_capacity(k_count);
    let mut prmse_x = Vec::with_capacity(k_count);
    let mut prmse_sx = Vec::with_capacity(k_count);
    let (mut tau, mut beta, mut gamma) = (
        Vec::with_capacity(k_count),
        Vec::with_capacity(k_count),
        Vec::with_capacity(k_count),
    );
    for k in 0..k_count {
        let r_stxt = cov_rowsum[k] * cov_rowsum[k] / (var_true[k] * var_true[k_count]);
        let r = corr[k][k_count];
        let t = (alpha_total.sqrt() * r_stxt.sqrt() - r * alpha[k].sqrt()) / (1.0 - r * r);
        let ps = alpha[k];
        let px = r_stxt * alpha_total;
        let psx = alpha[k] + t * t * (1.0 - r * r);
        for (name, v) in [("PRMSE_s", ps), ("PRMSE_x", px), ("PRMSE_sx", psx)] {
            if !v.is_finite() || v < 0.0 || v > 1.0 + 1e-9 {
                return Err(format!(
                    "computed {name} = {v:.6} outside [0, 1]; the sample \
                     moments are inconsistent with the CTT assumptions"
                ));
            }
        }
        prmse_s.push(ps);
        prmse_x.push(px);
        prmse_sx.push(psx);
        tau.push(t);
        beta.push(alpha[k].sqrt() * (alpha[k].sqrt() - r * t));
        gamma.push(alpha[k].sqrt() * t * (var_obs[k].sqrt() / var_obs[k_count].sqrt()));
    }

    let disattenuated_corr: Vec<Vec<f64>> = (0..k_count)
        .map(|a| {
            (0..k_count)
                .map(|b| {
                    if a == b {
                        f64::NAN
                    } else {
                        corr[a][b] / (alpha[a] * alpha[b]).sqrt()
                    }
                })
                .collect()
        })
        .collect();

    let mean_s: Vec<f64> = (0..k_count)
        .map(|k| cols[k].iter().sum::<f64>() / n as f64)
        .collect();
    let mean_x = total.iter().sum::<f64>() / n as f64;
    let sd_x = var_obs[k_count].sqrt();

    let mut subscore_s = vec![vec![0.0f64; k_count]; n];
    let mut subscore_x = vec![vec![0.0f64; k_count]; n];
    let mut subscore_sx = vec![vec![0.0f64; k_count]; n];
    for p in 0..n {
        for k in 0..k_count {
            let ds = observed[p][k] - mean_s[k];
            let dx = total[p] - mean_x;
            subscore_s[p][k] = mean_s[k] + alpha[k] * ds;
            subscore_x[p][k] =
                mean_s[k] + prmse_x[k].sqrt() * (var_true[k].sqrt() / sd_x) * dx;
            subscore_sx[p][k] = mean_s[k] + beta[k] * ds + gamma[k] * dx;
        }
    }

    let added_value_s: Vec<bool> = (0..k_count).map(|k| prmse_s[k] > prmse_x[k]).collect();
    let added_value_sx: Vec<bool> = (0..k_count)
        .map(|k| prmse_sx[k] > prmse_s[k].max(prmse_x[k]) + 0.01)
        .collect();

    alpha.pop();
    Ok(SubscoreResult {
        alpha,
        alpha_total,
        corr,
        disattenuated_corr,
        prmse_s,
        prmse_x,
        prmse_sx,
        tau,
        beta,
        gamma,
        added_value_s,
        added_value_sx,
        observed,
        total,
        subscore_s,
        subscore_x,
        subscore_sx,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/subscores_tests.rs"]
mod tests;
