//! Test-security statistics: answer-copying detection.
//!
//! Implements a reduced-scope, no-missing, conditional Wollack-style omega
//! answer-copying statistic. The formula and p-value direction are verified
//! against the inspectable CRAN `CopyDetect` implementation (Zopluoglu;
//! `R/similarity2.r`: `exp.match <- sum(pvec)`,
//! `sd.match <- sqrt(sum(pvec*(1-pvec)))`,
//! `w.value <- (obs.match-exp.match)/sd.match`,
//! `p.value <- pnorm(w.value, lower.tail=FALSE)`) and the independent CRAN
//! `aberrance` implementation (`compute_OMG <- function(s, p, c = 0)
//! (sum(s - p) + c) / sqrt(sum(p * (1 - p)))` with upper-tail p-values).
//! Wollack (1997) is the originating citation for omega, but this
//! implementation does NOT claim to reproduce all paper procedures (the
//! original article was not directly readable during verification); callers
//! supply the suspected copier's per-item option probabilities (e.g. from a
//! nominal response model at the copier's ability estimate, the canonical
//! CopyDetect path). Missing-response handling, NRM fitting inside this
//! function, GBT, g2, K-family indices, and continuity corrections are out
//! of scope.
//!
//! In LLM-as-a-Judge item-quality management this flags judge pairs whose
//! agreement on option-level choices exceeds what the suspected copier's own
//! response model can explain.
//!
//! # References (APA 7th ed.)
//!
//! Wollack, J. A. (1997). A nominal response model approach for detecting
//!   answer copying. *Applied Psychological Measurement, 21*(4), 307-320.
//!   https://doi.org/10.1177/01466216970214002 (originating paper; NOT
//!   directly read — formula verified against the implementations below)
//! Zopluoglu, C. (2018). *CopyDetect: Computing response similarity indices
//!   for multiple-choice tests* (R package version 1.3) [Computer software].
//!   CRAN. (READ: `R/similarity1.r`, `R/similarity2.r`)
//! Man, K., & Harring, J. R. (2023). *aberrance: Detect aberrant behavior in
//!   test data* (R package) [Computer software]. CRAN. (READ:
//!   `R/detect-ac.R` `compute_OMG`, `R/compute.R`; independent check)

/// Result of the Wollack-style omega answer-copying test.
#[derive(Debug, Clone)]
pub struct OmegaResult {
    /// Observed number of identical responses `h`.
    pub observed_matches: usize,
    /// Expected matches under no copying, `E = sum_i P_i[source_i]`.
    pub expected_matches: f64,
    /// Variance of the match count, `V = sum_i p_i (1 - p_i)`.
    pub variance: f64,
    /// Standardized statistic `omega = (h - E) / sqrt(V)`.
    pub omega: f64,
    /// One-sided upper-tail p-value `1 - Phi(omega)`.
    pub p_value: f64,
}

/// Row-sum tolerance for the caller-supplied probability rows.
const OMEGA_ROW_SUM_TOL: f64 = 1e-6;

/// Wollack-style conditional omega answer-copying statistic.
///
/// `copier` and `source` are zero-based chosen-option indices per item (no
/// missing values in this reduced scope). `probs` is row-major
/// `n_items x n_options`: row `i` is the model probability distribution of
/// the SUSPECTED COPIER over the options of item `i` (each row finite,
/// nonnegative, summing to 1 within `1e-6`). The statistic conditions on the
/// source's observed responses: `p_i = probs[i, source[i]]`,
/// `omega = (h - sum p_i) / sqrt(sum p_i (1 - p_i))`, upper-tail p-value.
pub fn wollack_omega(
    copier: &[usize],
    source: &[usize],
    probs: &[f64],
    n_options: usize,
) -> Result<OmegaResult, String> {
    let n_items = copier.len();
    if n_items == 0 {
        return Err("wollack_omega: need at least 1 item".to_string());
    }
    if source.len() != n_items {
        return Err(format!(
            "wollack_omega: copier has {} items but source has {}",
            n_items,
            source.len()
        ));
    }
    if n_options == 0 {
        return Err("wollack_omega: n_options must be positive".to_string());
    }
    let expected = n_items
        .checked_mul(n_options)
        .ok_or_else(|| "wollack_omega: n_items * n_options overflows".to_string())?;
    if probs.len() != expected {
        return Err(format!(
            "wollack_omega: probs length {} != n_items {} x n_options {}",
            probs.len(),
            n_items,
            n_options
        ));
    }
    for (name, resp) in [("copier", copier), ("source", source)] {
        for (i, &k) in resp.iter().enumerate() {
            if k >= n_options {
                return Err(format!(
                    "wollack_omega: {} response {} on item {} out of range (n_options {})",
                    name, k, i, n_options
                ));
            }
        }
    }
    for i in 0..n_items {
        let row = &probs[i * n_options..(i + 1) * n_options];
        let mut s = 0.0;
        for &p in row {
            if !p.is_finite() || p < 0.0 {
                return Err(format!(
                    "wollack_omega: probability row {} has a nonfinite or negative entry",
                    i
                ));
            }
            s += p;
        }
        if (s - 1.0).abs() > OMEGA_ROW_SUM_TOL {
            return Err(format!(
                "wollack_omega: probability row {} sums to {} (must be 1 within 1e-6)",
                i, s
            ));
        }
    }

    let mut h = 0usize;
    let mut e = 0.0;
    let mut v = 0.0;
    for i in 0..n_items {
        if copier[i] == source[i] {
            h += 1;
        }
        let p = probs[i * n_options + source[i]];
        e += p;
        v += p * (1.0 - p);
    }
    if v <= 0.0 {
        return Err(
            "wollack_omega: match-count variance is zero (all source-option probabilities \
             degenerate); normal approximation undefined"
                .to_string(),
        );
    }
    let omega = (h as f64 - e) / v.sqrt();
    // One-sided upper tail: 1 - Phi(omega) = 0.5 * erfc(omega / sqrt(2)).
    let p_value = 0.5 * crate::fitstats::erfc(omega / std::f64::consts::SQRT_2);
    Ok(OmegaResult {
        observed_matches: h,
        expected_matches: e,
        variance: v,
        omega,
        p_value,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/security_tests.rs"]
mod tests;
