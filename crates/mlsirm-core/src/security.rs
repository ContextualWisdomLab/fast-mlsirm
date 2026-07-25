//! Test-security statistics: answer-copying detection.
//!
//! Implements (1) a reduced-scope, no-missing, conditional Wollack-style
//! omega answer-copying statistic, (2) the K-index of matching incorrect
//! answers exactly as implemented by CopyDetect's internal `k()`,
//! (3) the generalized binomial test (GBT) tail kernel exactly as
//! implemented by aberrance's `compute_GBT`, and (4) the K1/K2/S1/S2
//! answer-copying indices exactly as implemented by CopyDetect's internal
//! `ks12()`.
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
//! function, g2, M4 variants, and continuity corrections
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
//! The GBT kernel (`gbt`) is a faithful port of the CRAN aberrance
//! package's `src/compute.cpp` `compute_GBT` (READ): given per-item 0/1
//! match indicators and per-item model match probabilities, it computes the
//! exact Poisson-binomial distribution of the number of matches by a
//! Bernoulli-convolution DP and returns the INCLUSIVE upper tail
//! `P(M >= observed)`. The CopyDetect package's internal `GBT()`
//! (`R/similarity1.r`, READ) computes the same exact distribution and the
//! same inclusive upper tail via an equivalent matrix recursion, providing
//! independent corroboration of the kernel. The two packages differ only in
//! how the per-item match probabilities are CONSTRUCTED (aberrance:
//! directional `P(examinee B produces A's observed response)` per
//! `R/detect-ac.R`; CopyDetect: symmetric
//! `Pi = P1c*P2c + (1-P1c)*(1-P2c)`, both-correct-or-both-incorrect).
//! Probability construction is therefore OUT of scope here: callers supply
//! `match_probs` using either recipe. Missing data is out of scope (the two
//! packages conflict: aberrance skips pairs with any NA, CopyDetect
//! zero-fills common missing). NOT READ: van der Linden & Sotaridona (2006)
//! and (2004), the originating papers; GBT is cited only as implemented by
//! the two packages above. Numerics: the DP uses only nonnegative products
//! and sums of probabilities (no cancellation, no binomial-coefficient
//! overflow); for very large `n` with extreme probabilities, tiny tail
//! masses may underflow ordinary `f64` — this is an exact-DP-in-f64
//! limitation, O(n^2) time, O(n) memory.
//!
//! The K1/K2/S1/S2 indices (`k_variants`) are a faithful port of CopyDetect
//! `R/similarity1.r` internal `ks12()` (READ in full), specialized to the
//! no-missing, scored-0/1 contract. Number-incorrect subgroups EXCLUDE the
//! source row (`subgroups.ind[subgroups.ind!=pa[2]]`) — the opposite of the
//! base `k()` convention used by `k_index` above. Per-subgroup means of
//! matching-incorrect counts (`pr`) are regressed on the proportion-
//! incorrect design (`Qrs = j/n`, ordinary least squares, linear for K1 and
//! quadratic for K2) and, for S1/S2, `ws*pr` (plus a weighted-correct-match
//! shift `ceil(pj)` for S2) is fitted by a log-link Poisson GLM on the raw
//! count design (`Qrs3 = j`). Predictions are taken at the copier's design
//! point (`qc = wc/n` for OLS, integer `wc` for the GLMs — extracted by
//! integer slot, never float equality). Clamps are CopyDetect's:
//! `p1,p2 >= 1 -> 0.999`, `<= 0 -> 0.001`; `s1 >= ws -> ws`;
//! `s2 >= n -> n`. K1/K2 are binomial upper tails `P(Bin(ws, p) >= m)`;
//! S1/S2 are Poisson WINDOW probabilities `P(m <= Pois(s1) <= ws)` and
//! `P(mm <= Pois(s2) <= n)` with `mm = ceil(sum weight[wc][cm]) + m`
//! (raw ceiling, no epsilon, for CopyDetect fidelity — float-boundary
//! sensitivity near integer sums is inherited from R). The match weights
//! use `g = 0.2`, `d2 = -(1+g)/g = -6`,
//! `weight = (((1+g)/(1-g))*e)^(prob*d2)`, so weights lie in `(0, 1]` and
//! `mm <= m + |cm| <= n`. R's Poisson family would warn on the non-integer
//! responses, but `ks12()` suppresses warnings (`options(warn=-1)`) and
//! uses the coefficients; the same score equations are solved here by a
//! guarded Newton/IRLS with step-halving. NOT READ: Sotaridona & Meijer
//! (2002) and (2003), the originating K1/K2 and S1/S2 papers — all four
//! indices are cited only as implemented by CopyDetect; the READ RR-01-07
//! report corroborates only the K-variant regression framing.
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
//! Sotaridona, L. S., & Meijer, R. R. (2003). Two new statistics to detect
//!   answer copying. *Journal of Educational Measurement, 40*(1), 53-69.
//!   (NOT read; S1/S2 cited only as implemented by CopyDetect.)
//! van der Linden, W. J., & Sotaridona, L. (2006). Detecting answer copying
//!   when the regular response process follows a known response model.
//!   *Journal of Educational and Behavioral Statistics, 31*(3), 283-304.
//!   https://doi.org/10.3102/10769986031003283 (originating GBT paper; NOT
//!   read — the statistic is implemented as ported from the aberrance and
//!   CopyDetect sources.)

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

/// Binomial upper tail `P(Bin(n, p) >= m)` for `0 <= p <= 1`, computed by
/// summing the upper-tail terms directly in log space (term recurrence for
/// the log-probabilities, then a max-shifted exponential sum). No binomial
/// coefficients are materialized and no complement subtraction is
/// performed, so extreme `p` / large `n` neither overflow nor underflow
/// (e.g. `n = 1000, p = 0.99, m = 990`, where the linear-space term
/// `(1-p)^n` underflows to zero). Result clamped into `[0, 1]`.
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
    let lp = p.ln();
    let lq = (1.0 - p).ln();
    // log t_k where t_k = C(n,k) p^k (1-p)^(n-k):
    // log t_0 = n log(1-p); log t_{k+1} = log t_k + log((n-k)/(k+1)) + log(p/(1-p)).
    let mut lt = n as f64 * lq;
    let mut logs = Vec::with_capacity(n - m + 1);
    for k in 0..=n {
        if k >= m {
            logs.push(lt);
        }
        if k < n {
            lt += ((n - k) as f64 / (k + 1) as f64).ln() + lp - lq;
        }
    }
    let mx = logs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let s: f64 = logs.iter().map(|&l| (l - mx).exp()).sum();
    (mx + s.ln()).exp().clamp(0.0, 1.0)
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

/// Result of the generalized binomial test (GBT) tail kernel.
#[derive(Debug, Clone)]
pub struct GbtResult {
    /// Observed number of matching items, `sum(matches)`.
    pub observed_matches: usize,
    /// Exact Poisson-binomial pmf of the number of matches, length
    /// `n_items + 1`; `match_dist[k] = P(M = k)`.
    pub match_dist: Vec<f64>,
    /// Inclusive upper tail `P(M >= observed_matches)`; small values
    /// suggest copying.
    pub p_value: f64,
}

/// Generalized binomial test (GBT) tail kernel, exactly as implemented by
/// the CRAN aberrance package's `compute_GBT` (READ: `src/compute.cpp`) and
/// corroborated by CopyDetect's internal `GBT()` (READ: `R/similarity1.r`).
///
/// `matches[i]` is exactly 0.0/1.0 indicating whether the copier's and
/// source's responses on item `i` are identical; `match_probs[i]` is the
/// model probability of a match on item `i` (finite, in closed `[0, 1]`;
/// deterministic 0/1 probabilities are handled exactly by the DP).
/// Probability CONSTRUCTION is the caller's job: aberrance uses the
/// directional `P(examinee B produces A's observed response)`, CopyDetect
/// the symmetric `Pi = P1c*P2c + (1-P1c)*(1-P2c)` — either recipe fits.
///
/// The exact distribution of the match count `M` is built by the standard
/// Bernoulli-convolution DP (seeded at `f = [1]` and folding one item at a
/// time, algebraically identical to aberrance's `f[0]=1-p[0], f[1]=p[0]`
/// seed-then-loop form), and the p-value is the INCLUSIVE upper tail
/// `sum_{k=obs}^{n} f[k]`, clamped into `[0, 1]`. `obs = 0` naturally
/// yields `p = 1`; no special-casing.
pub fn gbt(matches: &[f64], match_probs: &[f64]) -> Result<GbtResult, String> {
    let n = matches.len();
    if n == 0 {
        return Err("gbt: need at least 1 item".to_string());
    }
    if match_probs.len() != n {
        return Err(format!(
            "gbt: matches length {} != match_probs length {}",
            n,
            match_probs.len()
        ));
    }
    for (i, &x) in matches.iter().enumerate() {
        if x != 0.0 && x != 1.0 {
            return Err(format!(
                "gbt: matches[{}] = {} (entries must be exactly 0 or 1)",
                i, x
            ));
        }
    }
    for (i, &p) in match_probs.iter().enumerate() {
        if !p.is_finite() || !(0.0..=1.0).contains(&p) {
            return Err(format!(
                "gbt: match_probs[{}] = {} (must be finite and in [0, 1])",
                i, p
            ));
        }
    }

    let obs = matches.iter().filter(|&&x| x == 1.0).count();

    // Bernoulli-convolution DP over items; f[k] = P(M = k) after each fold.
    let mut f = vec![0.0f64; n + 1];
    f[0] = 1.0;
    for (i, &p) in match_probs.iter().enumerate() {
        let q = 1.0 - p;
        // Descending update so f[j-1] is still the previous item's value.
        for j in (1..=(i + 1)).rev() {
            f[j] = q * f[j] + p * f[j - 1];
        }
        f[0] *= q;
    }

    let p_value = f[obs..].iter().sum::<f64>().clamp(0.0, 1.0);
    Ok(GbtResult {
        observed_matches: obs,
        match_dist: f,
        p_value,
    })
}

/// Result of the K1/K2/S1/S2 answer-copying indices (CopyDetect `ks12()`).
#[derive(Debug, Clone)]
pub struct KVariantsResult {
    /// Copier's number-incorrect score `wc`.
    pub wc: usize,
    /// Source's number-incorrect score `ws`.
    pub ws: usize,
    /// Matching incorrect responses `m`.
    pub m: usize,
    /// S2 shifted match count `mm = ceil(sum weight[wc][cm]) + m`.
    pub mm: usize,
    /// Per-subgroup mean matching-incorrect proportion, length `n_items+1`;
    /// `NaN` where the (source-excluded) subgroup is empty.
    pub pr: Vec<f64>,
    /// Per-subgroup mean weighted correct-match sum, length `n_items+1`;
    /// `NaN` where the subgroup is empty.
    pub pj: Vec<f64>,
    /// Clamped linear-OLS prediction at `qc` (K1 binomial probability).
    pub p1: f64,
    /// Clamped quadratic-OLS prediction at `qc` (K2 binomial probability).
    pub p2: f64,
    /// Capped Poisson-GLM prediction at `wc` (S1 rate).
    pub s1: f64,
    /// Capped Poisson-GLM prediction at `wc` (S2 rate).
    pub s2: f64,
    /// K1 index `P(Bin(ws, p1) >= m)`; small values suggest copying.
    pub k1: f64,
    /// K2 index `P(Bin(ws, p2) >= m)`.
    pub k2: f64,
    /// S1 index `P(m <= Pois(s1) <= ws)`.
    pub s1_index: f64,
    /// S2 index `P(mm <= Pois(s2) <= n_items)`.
    pub s2_index: f64,
}

/// Poisson window probability `P(a <= X <= b)` for `X ~ Pois(lambda)`,
/// summed term-by-term in log space via `ln_gamma` (no complement
/// subtraction, so R's `(1-ppois(a-1,l))-(1-ppois(b,l))` cancellation risk
/// is avoided). `a > b` yields 0; `lambda == 0` yields `1{a == 0}`.
fn pois_window(lambda: f64, a: usize, b: usize) -> f64 {
    if a > b {
        return 0.0;
    }
    if lambda <= 0.0 {
        return if a == 0 { 1.0 } else { 0.0 };
    }
    let ll = lambda.ln();
    let mut s = 0.0f64;
    for k in a..=b {
        s += (-lambda + k as f64 * ll - crate::fitstats::ln_gamma(k as f64 + 1.0)).exp();
    }
    s.clamp(0.0, 1.0)
}

/// Rank-checked least squares for the tiny `ks12()` designs (2 or 3
/// columns), solved via classical Gram-Schmidt QR to match R `lm`'s QR
/// semantics rather than raw normal equations. Returns `Err` on a
/// rank-deficient design (R would drop the aliased column; that silent
/// behavior is NOT ported -- the caller's data is degenerate).
fn ols_qr(xs: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, String> {
    let ncol = xs.len();
    let nrow = y.len();
    if nrow < ncol {
        return Err(format!(
            "k_variants: {} complete design points cannot identify {} coefficients",
            nrow, ncol
        ));
    }
    // QR by modified Gram-Schmidt on the columns.
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(ncol);
    let mut r = vec![vec![0.0f64; ncol]; ncol];
    for (j, col) in xs.iter().enumerate() {
        let mut v = col.clone();
        for (i, qi) in q.iter().enumerate() {
            let dot: f64 = qi.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
            r[i][j] = dot;
            for (vk, qk) in v.iter_mut().zip(qi.iter()) {
                *vk -= dot * qk;
            }
        }
        let nrm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let scale = col.iter().map(|x| x * x).sum::<f64>().sqrt().max(1.0);
        if nrm <= 1e-10 * scale {
            return Err("k_variants: rank-deficient regression design".to_string());
        }
        r[j][j] = nrm;
        for vk in v.iter_mut() {
            *vk /= nrm;
        }
        q.push(v);
    }
    // beta = R^-1 Q^T y (back substitution).
    let qty: Vec<f64> = q
        .iter()
        .map(|qi| qi.iter().zip(y.iter()).map(|(a, b)| a * b).sum())
        .collect();
    let mut beta = vec![0.0f64; ncol];
    for i in (0..ncol).rev() {
        let mut s = qty[i];
        for j in (i + 1)..ncol {
            s -= r[i][j] * beta[j];
        }
        beta[i] = s / r[i][i];
    }
    Ok(beta)
}

/// Two-parameter log-link Poisson GLM `y ~ exp(b0 + b1 x)` solved by
/// guarded Newton on the score equations `sum(y-mu) = 0`,
/// `sum((y-mu) x) = 0` (exactly what R's `glm(..., family=poisson())`
/// converges to; non-integer `y` is permitted -- `ks12()` suppresses R's
/// warning and uses the coefficients). Start `b = (ln mean(y), 0)`;
/// step-halving (up to 30 halvings) enforces monotone log-likelihood
/// `sum(y*eta - mu)`; `eta` is bounded to avoid overflow; tolerance 1e-12
/// on the step, max 200 iterations, `Err` on nonconvergence.
fn poisson_glm2(x: &[f64], y: &[f64]) -> Result<(f64, f64), String> {
    let n = y.len();
    if n < 2 {
        return Err("k_variants: Poisson GLM needs at least 2 complete design points".to_string());
    }
    if x.iter().any(|v| !v.is_finite()) || y.iter().any(|v| !v.is_finite() || *v < 0.0) {
        return Err(
            "k_variants: Poisson GLM requires finite x and finite nonnegative y".to_string(),
        );
    }
    let x0 = x[0];
    if x.iter().all(|&v| v == x0) {
        return Err("k_variants: rank-deficient regression design".to_string());
    }
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut b0 = mean_y.max(1e-9).ln();
    let mut b1 = 0.0f64;
    const ETA_CAP: f64 = 500.0;
    let loglik = |b0: f64, b1: f64| -> f64 {
        let mut s = 0.0;
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let eta = (b0 + b1 * xi).clamp(-ETA_CAP, ETA_CAP);
            s += yi * eta - eta.exp();
        }
        s
    };
    let mut ll = loglik(b0, b1);
    for _ in 0..200 {
        let (mut g0, mut g1) = (0.0f64, 0.0f64);
        let (mut h00, mut h01, mut h11) = (0.0f64, 0.0f64, 0.0f64);
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            let eta = (b0 + b1 * xi).clamp(-ETA_CAP, ETA_CAP);
            let mu = eta.exp();
            g0 += yi - mu;
            g1 += (yi - mu) * xi;
            h00 += mu;
            h01 += mu * xi;
            h11 += mu * xi * xi;
        }
        let det = h00 * h11 - h01 * h01;
        if !det.is_finite() || det.abs() < 1e-300 {
            return Err("k_variants: singular Hessian in Poisson GLM".to_string());
        }
        let d0 = (h11 * g0 - h01 * g1) / det;
        let d1 = (h00 * g1 - h01 * g0) / det;
        // Step-halving: keep the log-likelihood monotone.
        let mut step = 1.0f64;
        let (mut nb0, mut nb1, mut nll);
        loop {
            nb0 = b0 + step * d0;
            nb1 = b1 + step * d1;
            nll = loglik(nb0, nb1);
            if nll >= ll - 1e-12 || step < 1e-9 {
                break;
            }
            step *= 0.5;
        }
        let moved = (step * d0).abs().max((step * d1).abs());
        b0 = nb0;
        b1 = nb1;
        ll = nll;
        if moved < 1e-12 {
            return Ok((b0, b1));
        }
    }
    Err("k_variants: Poisson GLM did not converge in 200 iterations".to_string())
}

/// K1/K2/S1/S2 answer-copying indices, exactly as implemented by the CRAN
/// CopyDetect package's internal `ks12()` (READ: `R/similarity1.r`),
/// specialized to complete scored 0/1 data. See the module header for the
/// full algorithm, clamps, and citation scope. Small index values suggest
/// copying. Errors: invalid inputs (as `k_index`), `ws == 0`, or
/// regression designs with too few / rank-deficient complete subgroups
/// (K2's quadratic needs 3 distinct nonempty number-incorrect scores).
pub fn k_variants(
    responses: &[f64],
    n_persons: usize,
    n_items: usize,
    copier: usize,
    source: usize,
) -> Result<KVariantsResult, String> {
    if n_persons < 2 {
        return Err("k_variants: need at least 2 persons".to_string());
    }
    if n_items == 0 {
        return Err("k_variants: need at least 1 item".to_string());
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "k_variants: n_persons * n_items overflows".to_string())?;
    if responses.len() != expected {
        return Err(format!(
            "k_variants: responses length {} != n_persons {} x n_items {}",
            responses.len(),
            n_persons,
            n_items
        ));
    }
    if copier >= n_persons || source >= n_persons {
        return Err(format!(
            "k_variants: copier {} / source {} out of range (n_persons {})",
            copier, source, n_persons
        ));
    }
    if copier == source {
        return Err("k_variants: copier and source must be distinct".to_string());
    }
    for (idx, &v) in responses.iter().enumerate() {
        if v != 0.0 && v != 1.0 {
            return Err(format!(
                "k_variants: responses[{}] = {} (entries must be exactly 0 or 1)",
                idx, v
            ));
        }
    }

    let n = n_items;
    let row = |r: usize| &responses[r * n..(r + 1) * n];
    let incorrect = |r: usize| row(r).iter().filter(|&&x| x == 0.0).count();
    let wc = incorrect(copier);
    let ws = incorrect(source);
    if ws == 0 {
        return Err(
            "k_variants: source has no incorrect responses (ws = 0); the indices are degenerate"
                .to_string(),
        );
    }
    let src = row(source);
    let mut m = 0usize;
    let mut cm: Vec<usize> = Vec::new();
    for i in 0..n {
        let c = row(copier)[i];
        if c == 0.0 && src[i] == 0.0 {
            m += 1;
        }
        if c == 1.0 && src[i] == 1.0 {
            cm.push(i);
        }
    }
    let qc = wc as f64 / n as f64;

    // Number-incorrect subgroups j = 0..=n, EXCLUDING the source row
    // (ks12(): `subgroups.ind[subgroups.ind!=pa[2]]`; the copier stays in).
    let mut subgroups: Vec<Vec<usize>> = vec![Vec::new(); n + 1];
    for r in 0..n_persons {
        if r != source {
            subgroups[incorrect(r)].push(r);
        }
    }

    let g = 0.2f64;
    let d2 = -(1.0 + g) / g; // -6
    let base = ((1.0 + g) / (1.0 - g)) * std::f64::consts::E;

    let mut pr = vec![f64::NAN; n + 1];
    let mut pj = vec![f64::NAN; n + 1];
    // weight row at the copier's own subgroup (j = wc), needed for `mm`.
    let mut weight_wc = vec![f64::NAN; n];
    for (j, members) in subgroups.iter().enumerate() {
        if members.is_empty() {
            continue;
        }
        let cnt = members.len() as f64;
        // pr[j]: mean matching-incorrect count with the source, over ws.
        let mut agg_sum = 0usize;
        // prob[j][i]: subgroup proportion correct on items the source got
        // correct (0 elsewhere -- `smatrix==1` fails there).
        let mut prob = vec![0.0f64; n];
        for &r in members {
            let rr = row(r);
            for i in 0..n {
                if rr[i] == 0.0 && src[i] == 0.0 {
                    agg_sum += 1;
                }
                if rr[i] == 1.0 && src[i] == 1.0 {
                    prob[i] += 1.0;
                }
            }
        }
        for p in prob.iter_mut() {
            *p /= cnt;
        }
        pr[j] = (agg_sum as f64 / cnt) / ws as f64;
        let w: Vec<f64> = prob.iter().map(|&p| base.powf(p * d2)).collect();
        // pj[j] = mean over members of sum_i cor_match[r][i] * w[i].
        let mut pj_sum = 0.0f64;
        for &r in members {
            let rr = row(r);
            for i in 0..n {
                if rr[i] == 1.0 && src[i] == 1.0 {
                    pj_sum += w[i];
                }
            }
        }
        pj[j] = pj_sum / cnt;
        if j == wc {
            weight_wc.copy_from_slice(&w);
        }
    }
    // The copier always belongs to its own (source-excluded) subgroup.
    debug_assert!(pr[wc].is_finite());

    // Complete cases: subgroup scores j with pr[j] finite (na.omit).
    let complete: Vec<usize> = (0..=n).filter(|&j| pr[j].is_finite()).collect();
    let qrs: Vec<f64> = complete.iter().map(|&j| j as f64 / n as f64).collect();
    let qrs3: Vec<f64> = complete.iter().map(|&j| j as f64).collect();
    let pr_c: Vec<f64> = complete.iter().map(|&j| pr[j]).collect();
    let ones = vec![1.0f64; complete.len()];

    // K1: OLS pr ~ 1 + Qrs, predicted at qc.
    let b1v = ols_qr(&[ones.clone(), qrs.clone()], &pr_c)?;
    let p1_raw = b1v[0] + b1v[1] * qc;
    // K2: OLS pr ~ 1 + Qrs + Qrs^2.
    let qrs2: Vec<f64> = qrs.iter().map(|&x| x * x).collect();
    let b2v = ols_qr(&[ones, qrs, qrs2], &pr_c)?;
    let p2_raw = b2v[0] + b2v[1] * qc + b2v[2] * qc * qc;

    // S1: Poisson GLM ws*pr ~ Qrs3, predicted at integer wc.
    let y1: Vec<f64> = pr_c.iter().map(|&p| ws as f64 * p).collect();
    let (a0, a1) = poisson_glm2(&qrs3, &y1)?;
    let s1_raw = ((a0 + a1 * wc as f64).clamp(-500.0, 500.0)).exp();
    // S2: Poisson GLM ws*pr + ceil(pj) ~ Qrs3.
    let y2: Vec<f64> = complete
        .iter()
        .zip(y1.iter())
        .map(|(&j, &v)| v + pj[j].ceil())
        .collect();
    let (c0, c1) = poisson_glm2(&qrs3, &y2)?;
    let s2_raw = ((c0 + c1 * wc as f64).clamp(-500.0, 500.0)).exp();

    // CopyDetect clamps.
    let clamp_p = |p: f64| {
        if p >= 1.0 {
            0.999
        } else if p <= 0.0 {
            0.001
        } else {
            p
        }
    };
    let p1 = clamp_p(p1_raw);
    let p2 = clamp_p(p2_raw);
    let s1 = if s1_raw >= ws as f64 {
        ws as f64
    } else {
        s1_raw
    };
    let s2 = if s2_raw >= n as f64 { n as f64 } else { s2_raw };

    // mm = ceil(sum weight[wc][cm]) + m; raw ceiling (CopyDetect fidelity;
    // float-boundary flips near integer sums are inherited). Weights are in
    // (0, 1] so mm <= m + |cm| <= n.
    let wsum: f64 = cm.iter().map(|&i| weight_wc[i]).sum();
    let mm = wsum.ceil() as usize + m;

    let k1 = binom_sf_ge(ws, p1, m);
    let k2 = binom_sf_ge(ws, p2, m);
    let s1_index = pois_window(s1, m, ws);
    let s2_index = pois_window(s2, mm, n);

    Ok(KVariantsResult {
        wc,
        ws,
        m,
        mm,
        pr,
        pj,
        p1,
        p2,
        s1,
        s2,
        k1,
        k2,
        s1_index,
        s2_index,
    })
}
#[cfg(test)]
#[path = "../../../tests/unit/security_tests.rs"]
mod tests;
