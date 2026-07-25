//! Test-security statistics: answer-copying detection.
//!
//! Implements (1) a reduced-scope, no-missing, conditional Wollack-style
//! omega answer-copying statistic and (2) the K-index of matching incorrect
//! answers exactly as implemented by CopyDetect's internal `k()`.
//!
//! The omega formula and p-value direction are verified
//! against the inspectable CRAN `CopyDetect` implementation (Zopluoglu;
//! `R/similarity2.r`: `exp.match <- sum(pvec)`,
//! `sd.match <- sqrt(sum(pvec*(1-pvec)))`,
//! `w.value <- (obs.match-exp.match)/sd.match`,
//! `p.value <- pnorm(w.value, lower.tail=FALSE)`) and the independent CRAN
//! `aberrance` implementation (`compute_OMG <- function(s, p, c = 0)
//! (sum(s - p) + c) / sqrt(sum(p * (1 - p)))` with upper-tail p-values).
//! Documented conflict: CopyDetect's PRINTED documentation shows the sign
//! flipped (`(exp.match - obs.match)/sd.match`), but both source files
//! compute `(obs.match - exp.match)/sd.match` with an upper-tail p; the
//! source convention is implemented here.
//! Wollack (1997) is the originating citation for omega, but this
//! implementation does NOT claim to reproduce all paper procedures (the
//! original article was not directly readable during verification); callers
//! supply the suspected copier's per-item option probabilities (e.g. from a
//! nominal response model at the copier's ability estimate, the canonical
//! CopyDetect path). Missing-response handling, NRM fitting inside this
//! function, GBT, g2, S1/S2/K1/K2/M4 variants, and continuity corrections
//! are out of scope.
//!
//! The K-index (`k_index`) is a faithful port of CopyDetect
//! `R/similarity1.r` internal `k()` (READ; corroborated by the same
//! package's `R/similarity2.r`): subgroup is all rows with the suspected
//! copier's number-incorrect score, including the copier and (if
//! applicable) the source; `p = mean(emp.agg) / ws`; K is the binomial
//! upper tail `P(Bin(ws, p) >= m)`. The `aberrance` package was checked
//! (`R/detect-ac.R`, `R/compute.R`, `src/compute.cpp`) and provides
//! OMG/GBT/M4 but NO K-index, so it does not corroborate K. The ERIC-hosted
//! Sotaridona & Meijer technical report RR-01-07 (ED467373) was READ only
//! for background corroboration of the binomial matching-incorrect
//! framework and the K/K2 distinction; it notes the paper convention
//! excludes the source from number-incorrect groups, which CopyDetect's
//! base `k()` does NOT do — the CopyDetect convention is implemented. NOT
//! READ: Holland (1996) ETS RR-96-07 and Sotaridona & Meijer (2002), so
//! this implementation must not be described as a direct transcription of
//! those papers.
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
//! *aberrance: Detect aberrant behavior in test data* (R package)
//!   [Computer software]. CRAN. (READ: `R/detect-ac.R` `compute_OMG`,
//!   `R/compute.R`; independent check. Authorship not established from the
//!   sources read, so no authors are claimed.)
//! Holland, P. W. (1996). *Assessing unusual agreement between the
//!   incorrect answers of two examinees using the K-index* (Research Report
//!   RR-96-07). Educational Testing Service. (NOT read; K-index cited only
//!   as implemented by CopyDetect.)
//! Sotaridona, L. S., & Meijer, R. R. (2001). *Two new statistics to detect
//!   answer copying* (Research Report RR-01-07; ERIC ED467373). University
//!   of Twente. (READ via ERIC full text; background corroboration of the
//!   binomial matching-incorrect framework only.)
//! Sotaridona, L. S., & Meijer, R. R. (2002). Statistical properties of the
//!   K-index for detecting answer copying. *Journal of Educational
//!   Measurement, 39*(2), 115-132. (NOT read.)

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

/// Result of the CopyDetect-style K-index answer-copying test.
#[derive(Debug, Clone)]
pub struct KIndexResult {
    /// Copier's number-incorrect score `wc`.
    pub wc: usize,
    /// Source's number-incorrect score `ws`.
    pub ws: usize,
    /// Matching incorrect responses `m = #{i : both incorrect}`.
    pub m: usize,
    /// Row indices of the number-incorrect subgroup (all rows with
    /// number-incorrect equal to `wc`, INCLUDING the copier and, when its
    /// number-incorrect equals `wc`, the source — CopyDetect convention).
    pub subgroup: Vec<usize>,
    /// Per-subgroup-member count of items incorrect for both the member and
    /// the source.
    pub emp_agg: Vec<usize>,
    /// Binomial success probability `p = mean(emp_agg) / ws`.
    pub p: f64,
    /// Upper-tail K-index `P(Bin(ws, p) >= m)`; small values suggest copying.
    pub k_index: f64,
}

/// Binomial upper tail `P(Bin(n, p) >= m)` for `0 <= p <= 1`, computed with
/// a coefficient-free term recurrence (no factorials/binomial coefficients
/// materialized), summing whichever tail has fewer terms and clamping
/// roundoff into `[0, 1]`.
fn binom_sf_ge(n: usize, p: f64, m: usize) -> f64 {
    if m == 0 {
        return 1.0;
    }
    if m > n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0; // m >= 1 here
    }
    if p >= 1.0 {
        return 1.0; // m <= n here
    }
    let ratio = p / (1.0 - p);
    // t_k = C(n,k) p^k (1-p)^(n-k); t_0 = (1-p)^n; t_{k+1} = t_k*(n-k)/(k+1)*ratio.
    let mut t = (1.0 - p).powi(n as i32);
    let (lo, hi, complement) = if m <= n / 2 {
        (0usize, m - 1, true) // sum lower tail 0..m-1, return 1 - it
    } else {
        (m, n, false)
    };
    let mut acc = 0.0;
    for k in 0..=hi {
        if k >= lo {
            acc += t;
        }
        if k < hi {
            t *= (n - k) as f64 / (k + 1) as f64 * ratio;
        }
    }
    let out = if complement { 1.0 - acc } else { acc };
    out.clamp(0.0, 1.0)
}

/// K-index of matching incorrect answers, exactly as implemented by the
/// CRAN CopyDetect package's internal `k()` (READ: `R/similarity1.r`,
/// corroborated by the same package's `R/similarity2.r`).
///
/// `responses` is a flattened row-major `n_persons x n_items` scored matrix
/// with entries exactly 0.0 (incorrect) or 1.0 (correct), no missing data.
/// `wc`/`ws` are the copier's/source's number-incorrect scores, `m` the
/// count of items both answered incorrectly. The subgroup is every row with
/// number-incorrect equal to `wc` — including the copier itself and, when
/// applicable, the source (faithful CopyDetect behavior; the mean of
/// `emp_agg` therefore self-includes `m`, and the paper-style
/// source-excluded convention is intentionally NOT applied).
/// `p = mean(emp_agg) / ws` and `K = P(Bin(ws, p) >= m)` (upper tail;
/// small K suggests copying). `ws == 0` is degenerate and returns `Err`.
pub fn k_index(
    responses: &[f64],
    n_persons: usize,
    n_items: usize,
    copier: usize,
    source: usize,
) -> Result<KIndexResult, String> {
    if n_persons < 2 {
        return Err("k_index: need at least 2 persons".to_string());
    }
    if n_items == 0 {
        return Err("k_index: need at least 1 item".to_string());
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "k_index: n_persons * n_items overflows".to_string())?;
    if responses.len() != expected {
        return Err(format!(
            "k_index: responses length {} != n_persons {} x n_items {}",
            responses.len(),
            n_persons,
            n_items
        ));
    }
    if copier >= n_persons || source >= n_persons {
        return Err(format!(
            "k_index: copier {} / source {} out of range (n_persons {})",
            copier, source, n_persons
        ));
    }
    if copier == source {
        return Err("k_index: copier and source must be distinct".to_string());
    }
    for (idx, &x) in responses.iter().enumerate() {
        if x != 0.0 && x != 1.0 {
            return Err(format!(
                "k_index: responses[{}] = {} (entries must be exactly 0 or 1)",
                idx, x
            ));
        }
    }

    let incorrect = |r: usize| -> usize {
        responses[r * n_items..(r + 1) * n_items]
            .iter()
            .filter(|&&x| x == 0.0)
            .count()
    };
    let wc = incorrect(copier);
    let ws = incorrect(source);
    if ws == 0 {
        return Err(
            "k_index: source has no incorrect responses (ws = 0); the K-index is degenerate"
                .to_string(),
        );
    }
    let mut m = 0usize;
    for i in 0..n_items {
        if responses[copier * n_items + i] == 0.0 && responses[source * n_items + i] == 0.0 {
            m += 1;
        }
    }

    let mut subgroup = Vec::new();
    let mut emp_agg = Vec::new();
    for r in 0..n_persons {
        if incorrect(r) == wc {
            subgroup.push(r);
            let mut agg = 0usize;
            for i in 0..n_items {
                if responses[r * n_items + i] == 0.0 && responses[source * n_items + i] == 0.0 {
                    agg += 1;
                }
            }
            debug_assert!(agg <= ws);
            emp_agg.push(agg);
        }
    }
    // The copier always belongs to its own number-incorrect subgroup.
    debug_assert!(!subgroup.is_empty());

    let mean_agg = emp_agg.iter().sum::<usize>() as f64 / emp_agg.len() as f64;
    let p = mean_agg / ws as f64;
    let k = binom_sf_ge(ws, p, m);
    Ok(KIndexResult {
        wc,
        ws,
        m,
        subgroup,
        emp_agg,
        p,
        k_index: k,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/security_tests.rs"]
mod tests;
