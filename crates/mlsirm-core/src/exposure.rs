//! Item-exposure control designs for computerized adaptive testing:
//! Sympson-Hetter calibration ([`sympson_hetter`]) and the a-stratified
//! multistage selection design ([`a_stratified`]; see its docs below).
//!
//! Implements the unconditional Sympson-Hetter (1985) probabilistic exposure
//! filter and its iterative calibration for a fixed-length, unidimensional,
//! dichotomous (3PL; 2PL when `c = 0`) maximum-information CAT.
//!
//! Source status: the original Sympson & Hetter (1985) proceedings paper was
//! NOT readable and is cited as attributed. Every implemented rule below was
//! verified against readable secondary sources that restate the algorithm:
//! Georgiadou, Triantafillou & Economides (2007, p. 12: the `P(S)`, `P(A|S)`,
//! `P(A) = P(A|S) P(S)` decomposition, the uniform-draw gate, blocking a
//! rejected item for the rest of the examinee's test, and iterative
//! calibration on a sample from a typical ability distribution) and Barrada,
//! Olea & Ponsoda (2007, Eq. 1-3: the target `max_i P(A_i) <= r_max` and the
//! update `k_i <- 1` if `P(S_i) <= r_max`, else `k_i <- r_max / P(S_i)`),
//! plus the mirtCAT R source (Chalmers, 2016; implementation evidence for
//! the per-encounter uniform draw and item invalidation).
//!
//! Algorithm (one calibration cycle):
//!
//! 1. Simulate `n_simulees` examinees with `theta ~ N(0, 1)` (the target
//!    population; the normal default is an implementation choice also used by
//!    Barrada et al., 2007, not a requirement of the method).
//! 2. Each examinee takes a fixed-length test: at each step the not-yet
//!    administered, not-blocked item with maximum Fisher information at the
//!    current interim EAP is SELECTED; a fresh `u ~ U(0, 1)` is drawn and the
//!    item is ADMINISTERED iff `u <= k_i`, otherwise it is blocked for the
//!    remainder of that examinee's test and the next-best item is considered.
//! 3. After the cycle, with `P(S_i)` and `P(A_i)` the per-simulee selection
//!    and administration proportions: if `max_i P(A_i) <= r_max + tol` the
//!    calibration has reached its target and stops; otherwise
//!    `k_i <- min(1, r_max / P(S_i))` when `P(S_i) > r_max`, else `k_i <- 1`
//!    (Barrada et al., 2007, Eq. 3), and the next cycle runs.
//!
//! Convergence is NOT guaranteed by the method (van der Linden, 2003,
//! abstract: the iterative process "does not guarantee admissibility");
//! the result reports `converged` plus the per-cycle max-exposure history and
//! the calibration loop is bounded by `max_iter`.
//!
//! Exhausted-pool policy (explicit repository choice, not a rule from the
//! read sources): if every remaining item is rejected before the test reaches
//! `test_length`, the run fails with an error rather than force-administering
//! the last selected item.
//!
//! Feasibility (derived here from the counting identity, not from a source):
//! each simulee is administered exactly `test_length` items, so
//! `sum_i P(A_i) = test_length` and `max_i P(A_i) >= test_length / n_items`;
//! `r_max` below that bound is rejected as infeasible. The bound is
//! NECESSARY, not sufficient: at or near `r_max = test_length / n_items`
//! every item must be administered on nearly every selection, so the
//! stochastic gate can still exhaust the pool mid-test and the run then
//! fails with the documented pool-exhausted error rather than force-
//! administering an item (see the exhausted-pool policy above).
//!
//! When `r_max >= 1` every `k_i` stays 1, the uniform gate is skipped
//! entirely (no exposure RNG is consumed), and the procedure reduces exactly
//! to unconstrained maximum-information CAT.
//!
//! The interim/final ability estimate is EAP with an `N(0, 1)` prior on a
//! uniform grid over `[-4, 4]`; item information reuses
//! [`crate::scoring::item_information_4pl`].
//!
//! # References
//!
//! Barrada, J. R., Olea, J., & Ponsoda, V. (2007). Methods for restricting
//! maximum exposure rate in computerized adaptive testing. *Methodology,
//! 3*(1), 14-23. <https://doi.org/10.1027/1614-2241.3.1.14>
//!
//! Chalmers, R. P. (2016). Generating adaptive and non-adaptive test
//! interfaces for multidimensional item response theory applications.
//! *Journal of Statistical Software, 71*(5), 1-38.
//! <https://doi.org/10.18637/jss.v071.i05> (mirtCAT; the package R source
//! was read as implementation evidence.)
//!
//! Georgiadou, E., Triantafillou, E., & Economides, A. A. (2007). A review of
//! item exposure control strategies for computerized adaptive testing
//! developed from 1983 to 2005. *Journal of Technology, Learning, and
//! Assessment, 5*(8).
//!
//! Sympson, J. B., & Hetter, R. D. (1985). Controlling item-exposure rates in
//! computerized adaptive testing. In *Proceedings of the 27th annual meeting
//! of the Military Testing Association* (pp. 973-977). Navy Personnel
//! Research and Development Center. (As cited in Georgiadou et al., 2007, and
//! Barrada et al., 2007; not read.)
//!
//! van der Linden, W. J. (2003). Some alternatives to Sympson-Hetter
//! item-exposure control in computerized adaptive testing. *Journal of
//! Educational and Behavioral Statistics, 28*(3), 249-265. (Abstract only
//! was read.)

use crate::scoring::item_information_4pl;

/// Configuration for [`sympson_hetter`] calibration.
#[derive(Clone, Debug)]
pub struct SympsonHetterConfig {
    /// Target maximum exposure rate `r_max` in `(0, 1]`.
    pub r_max: f64,
    /// Fixed test length `L` (items administered per simulee).
    pub test_length: usize,
    /// Simulees per calibration cycle.
    pub n_simulees: usize,
    /// Maximum calibration cycles (the method does not guarantee
    /// convergence; van der Linden, 2003, abstract).
    pub max_iter: usize,
    /// Monte-Carlo tolerance on the stopping rule `max P(A) <= r_max + tol`.
    pub tol: f64,
    /// RNG seed (deterministic LCG; the crate's inline PRNG idiom).
    pub seed: u64,
    /// EAP quadrature points over `[-4, 4]`.
    pub q_theta: usize,
}

impl Default for SympsonHetterConfig {
    fn default() -> Self {
        Self {
            r_max: 0.25,
            test_length: 20,
            n_simulees: 1000,
            max_iter: 20,
            tol: 0.02,
            seed: 20250724,
            q_theta: 41,
        }
    }
}

/// Result of a Sympson-Hetter calibration run. The returned `k` is always
/// the vector that produced the reported final-cycle rates (the Eq. 3 update
/// is skipped after the last cycle).
#[derive(Clone, Debug)]
pub struct SympsonHetterResult {
    /// Final exposure-control parameters `k_i = P(A_i | S_i)`, in `(0, 1]`.
    pub k: Vec<f64>,
    /// Administration rates `P(A_i)` from the final cycle.
    pub exposure: Vec<f64>,
    /// Selection rates `P(S_i)` from the final cycle.
    pub selection: Vec<f64>,
    /// `max_i P(A_i)` from the final cycle.
    pub max_exposure: f64,
    /// Calibration cycles actually run.
    pub n_iter: usize,
    /// `max_exposure <= r_max + tol` reached within `max_iter` cycles.
    pub converged: bool,
    /// `max_i P(A_i)` after each cycle.
    pub history_max_exposure: Vec<f64>,
}

// Deterministic LCG + Box-Muller (the crate's inline PRNG idiom; see mhrm.rs).
struct Lcg(u64);
impl Lcg {
    #[inline]
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    #[inline]
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[inline]
fn p3pl(theta: f64, a: f64, b: f64, c: f64) -> f64 {
    c + (1.0 - c) / (1.0 + (-a * (theta - b)).exp())
}

/// EAP over a uniform grid on [-4, 4] with an N(0,1) prior, given the
/// administered responses so far (standard posterior-mean point estimate on
/// a discrete grid; the uniform grid is a repository implementation choice).
///
/// `scratch` is a caller-supplied reusable buffer (at least `grid.len()`
/// capacity); it is cleared and overwritten on each call so that no heap
/// allocation is performed inside the hot simulee loop.
fn eap_interim(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    responses: &[(usize, f64)],
    grid: &[f64],
    log_prior: &[f64],
    scratch: &mut Vec<f64>,
) -> f64 {
    scratch.clear();
    scratch.extend_from_slice(log_prior);
    for (q, &t) in grid.iter().enumerate() {
        for &(i, y) in responses {
            let p = p3pl(t, a[i], b[i], c[i]).clamp(1e-12, 1.0 - 1e-12);
            scratch[q] += if y > 0.5 { p.ln() } else { (1.0 - p).ln() };
        }
    }
    let m = scratch.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut num = 0.0;
    let mut den = 0.0;
    for (q, &t) in grid.iter().enumerate() {
        let w = (scratch[q] - m).exp();
        num += w * t;
        den += w;
    }
    num / den
}

/// Calibrate Sympson-Hetter exposure-control parameters by iterative CAT
/// simulation. `a`, `b`, `c` are 3PL item parameters (`c = 0` gives 2PL).
/// See the module docs for the algorithm, sources, and policies.
pub fn sympson_hetter(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    cfg: &SympsonHetterConfig,
) -> Result<SympsonHetterResult, String> {
    let n_items = a.len();
    if b.len() != n_items || c.len() != n_items {
        return Err("a, b, c must have equal lengths".into());
    }
    if n_items == 0 {
        return Err("item pool is empty".into());
    }
    if a.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("discriminations a must be finite and positive".into());
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err("difficulties b must be finite".into());
    }
    if c.iter().any(|v| !v.is_finite() || *v < 0.0 || *v >= 1.0) {
        return Err("guessing c must be finite and in [0, 1)".into());
    }
    if !cfg.r_max.is_finite() || cfg.r_max <= 0.0 || cfg.r_max > 1.0 {
        return Err("r_max must be in (0, 1]".into());
    }
    if cfg.test_length == 0 || cfg.test_length > n_items {
        return Err("test_length must be in 1..=n_items".into());
    }
    // Counting identity (derived here): sum_i P(A_i) = test_length, so
    // max_i P(A_i) >= test_length / n_items; a smaller r_max is infeasible.
    if cfg.r_max < cfg.test_length as f64 / n_items as f64 {
        return Err(format!(
            "r_max = {} is infeasible: max exposure cannot fall below test_length/n_items = {}",
            cfg.r_max,
            cfg.test_length as f64 / n_items as f64
        ));
    }
    if cfg.n_simulees == 0 {
        return Err("n_simulees must be positive".into());
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err("tol must be finite and non-negative".into());
    }
    if cfg.q_theta < 3 {
        return Err("q_theta must be at least 3".into());
    }

    let grid: Vec<f64> = (0..cfg.q_theta)
        .map(|q| -4.0 + 8.0 * q as f64 / (cfg.q_theta - 1) as f64)
        .collect();
    let log_prior: Vec<f64> = grid.iter().map(|&t| -0.5 * t * t).collect();

    let mut k = vec![1.0_f64; n_items];
    let mut rng = Lcg(cfg.seed.wrapping_mul(2654435761).wrapping_add(1));
    let mut history = Vec::with_capacity(cfg.max_iter);
    let mut exposure = vec![0.0; n_items];
    let mut selection = vec![0.0; n_items];
    let mut converged = false;
    let mut n_iter = 0;
    // Reusable scratch buffer for eap_interim: allocated once, reused across
    // all simulees and all administered items to avoid per-call heap allocation.
    let mut eap_scratch: Vec<f64> = Vec::with_capacity(cfg.q_theta);

    for cycle in 0..cfg.max_iter {
        n_iter += 1;
        let mut s_count = vec![0u64; n_items];
        let mut a_count = vec![0u64; n_items];

        for _p in 0..cfg.n_simulees {
            let theta_true = rng.normal();
            let mut usable = vec![true; n_items]; // not administered, not blocked
            let mut responses: Vec<(usize, f64)> = Vec::with_capacity(cfg.test_length);
            let mut theta_hat = 0.0;
            let mut administered = 0usize;

            while administered < cfg.test_length {
                // SELECT: max information among usable items.
                let mut best: Option<usize> = None;
                let mut best_info = f64::NEG_INFINITY;
                for i in 0..n_items {
                    if !usable[i] {
                        continue;
                    }
                    let p = p3pl(theta_hat, a[i], b[i], c[i]);
                    let info = item_information_4pl(a[i], p, c[i], 1.0);
                    if info > best_info {
                        best_info = info;
                        best = Some(i);
                    }
                }
                let Some(s) = best else {
                    // Explicit policy (see module docs): fail, do not force.
                    return Err(
                        "item pool exhausted before reaching test_length (all remaining items rejected)"
                            .into(),
                    );
                };
                s_count[s] += 1;
                // GATE: skip the draw entirely when k = 1 so r_max >= 1
                // consumes no exposure RNG and reduces exactly to
                // unconstrained max-info CAT.
                let admit = k[s] >= 1.0 || rng.next_f64() <= k[s];
                usable[s] = false; // administered or blocked either way
                if !admit {
                    continue;
                }
                a_count[s] += 1;
                let p_true = p3pl(theta_true, a[s], b[s], c[s]);
                let y = if rng.next_f64() < p_true { 1.0 } else { 0.0 };
                responses.push((s, y));
                administered += 1;
                theta_hat = eap_interim(a, b, c, &responses, &grid, &log_prior, &mut eap_scratch);
            }
        }

        let n = cfg.n_simulees as f64;
        for i in 0..n_items {
            selection[i] = s_count[i] as f64 / n;
            exposure[i] = a_count[i] as f64 / n;
        }
        let max_exposure = exposure.iter().cloned().fold(0.0_f64, f64::max);
        history.push(max_exposure);

        if max_exposure <= cfg.r_max + cfg.tol {
            converged = true;
            break;
        }
        // Barrada et al. (2007), Eq. 3. Skipped after the final cycle so the
        // returned k is always the vector that PRODUCED the reported rates.
        if cycle + 1 < cfg.max_iter {
            for i in 0..n_items {
                k[i] = if selection[i] > cfg.r_max {
                    (cfg.r_max / selection[i]).min(1.0)
                } else {
                    1.0
                };
            }
        }
    }

    let max_exposure = *history.last().expect("at least one cycle ran");
    Ok(SympsonHetterResult {
        k,
        exposure,
        selection,
        max_exposure,
        n_iter,
        converged,
        history_max_exposure: history,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/exposure_tests.rs"]
mod tests;

// ===================== a-stratified multistage CAT =====================

/// Configuration for [`a_stratified`] simulation.
#[derive(Clone, Debug)]
pub struct AStratifiedConfig {
    /// Number of strata `K` (also the number of stages).
    pub n_strata: usize,
    /// Fixed test length `L` (items administered per simulee).
    pub test_length: usize,
    /// Number of simulees in the evaluation run.
    pub n_simulees: usize,
    /// RNG seed (deterministic LCG; the crate's inline PRNG idiom).
    pub seed: u64,
    /// EAP quadrature points over `[-4, 4]`.
    pub q_theta: usize,
}

impl Default for AStratifiedConfig {
    fn default() -> Self {
        Self {
            n_strata: 4,
            test_length: 20,
            n_simulees: 1000,
            seed: 20250724,
            q_theta: 41,
        }
    }
}

/// Result of an a-stratified CAT simulation run.
#[derive(Clone, Debug)]
pub struct AStratifiedResult {
    /// Administration rates `P(A_i) = count_i / n_simulees`.
    pub exposure: Vec<f64>,
    /// `max_i P(A_i)`.
    pub max_exposure: f64,
    /// Stratum index (`0..n_strata`, ascending discrimination) per item.
    pub stratum: Vec<usize>,
    /// Items administered in each stage (`sum = test_length`).
    pub stage_lengths: Vec<usize>,
    /// RMSE of the final EAP against the simulated true thetas.
    pub theta_rmse: f64,
    /// Mean of `theta_hat - theta_true`.
    pub theta_bias: f64,
}

/// a-stratified multistage CAT item-selection design (Chang & Ying, 1999),
/// evaluated by simulation.
///
/// Source status: the Chang & Ying (1999) full text was NOT read; the
/// high-level design (early stages administer low-discrimination items,
/// later stages higher-discrimination items) was confirmed from the
/// publisher abstract, and the explicit selection rule was confirmed from
/// Barrada, Mazuela & Olea (2006, read in full), which restates the AS
/// method: sort the pool ascending by `a`, form contiguous strata, and at
/// each step administer the eligible item minimizing `|b_i - theta_hat|`
/// (b-matching, NOT maximum information — the point of the design is that
/// max-info greedily overexposes high-`a` items).
///
/// Algorithm:
/// 1. Sort items ascending by `(a_i, index)` (stable tie-break; the index
///    tie-break is a repository choice). Partition the sorted order into
///    `K` contiguous strata; stratum sizes are near-equal with the first
///    `n mod K` strata one larger (remainder placement is a repository
///    choice — sources leave the partition sizes to the designer).
/// 2. Partition `test_length` into `K` stage lengths the same way (the
///    near-equal split is a repository choice; the paper-supported rule is
///    only that stage `k` draws exclusively from stratum `k`, in order).
/// 3. Within stage `k`, administer the not-yet-administered item of stratum
///    `k` minimizing `|b_i - theta_hat|`; ties break to the lowest original
///    item index (repository choice).
/// 4. Simulate `y ~ Bernoulli(P_3PL(theta_true))` with `theta_true ~
///    N(0, 1)` and update `theta_hat` by interim EAP after each item. The
///    EAP estimator, its grid, and the initial `theta_hat = 0` are
///    repository implementation choices for parity with [`sympson_hetter`];
///    they are NOT claimed as Chang & Ying's estimator (their simulations
///    used ML-based estimation).
///
/// `c` is deliberately unused in stratification and selection (AS ignores
/// the guessing parameter; Barrada et al., 2006) but is used in the
/// simulated 3PL responses and the EAP update.
///
/// Deferred (out of scope): b-blocking stratification (Chang, Qian & Ying,
/// 2001) and maximum-information stratification variants.
///
/// Exact per-stratum counting identity (derived here): every simulee takes
/// exactly `stage_lengths[k]` items from stratum `k`, so
/// `sum_{i in stratum k} P(A_i) = stage_lengths[k]` and globally
/// `sum_i P(A_i) = test_length`.
///
/// # References
///
/// Barrada, J. R., Mazuela, P., & Olea, J. (2006). Maximum information
/// stratification method for controlling item exposure in computerized
/// adaptive testing. *Psicothema, 18*(1), 156-159. (Read in full; source
/// for the AS selection and stratification rules as restated there.)
///
/// Chang, H.-H., & Ying, Z. (1999). a-Stratified multistage computerized
/// adaptive testing. *Applied Psychological Measurement, 23*(3), 211-222.
/// <https://doi.org/10.1177/01466219922031338> (Abstract only was read;
/// cited as the origin of the design.)
///
/// Chang, H.-H., Qian, J., & Ying, Z. (2001). a-Stratified multistage
/// computerized adaptive testing with b blocking. *Applied Psychological
/// Measurement, 25*(4), 333-341. (Not read; cited only to name the deferred
/// b-blocking extension.)
pub fn a_stratified(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    cfg: &AStratifiedConfig,
) -> Result<AStratifiedResult, String> {
    let n_items = a.len();
    if b.len() != n_items || c.len() != n_items {
        return Err("a, b, c must have equal lengths".into());
    }
    if n_items == 0 {
        return Err("item pool is empty".into());
    }
    if a.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err("discriminations a must be finite and positive".into());
    }
    if b.iter().any(|v| !v.is_finite()) {
        return Err("difficulties b must be finite".into());
    }
    if c.iter().any(|v| !v.is_finite() || *v < 0.0 || *v >= 1.0) {
        return Err("guessing c must be finite and in [0, 1)".into());
    }
    if cfg.test_length == 0 || cfg.test_length > n_items {
        return Err("test_length must be in 1..=n_items".into());
    }
    if cfg.n_strata == 0 || cfg.n_strata > cfg.test_length {
        return Err(
            "n_strata must be in 1..=test_length (every stage administers >= 1 item)".into(),
        );
    }
    if cfg.n_simulees == 0 {
        return Err("n_simulees must be positive".into());
    }
    if cfg.q_theta < 3 {
        return Err("q_theta must be at least 3".into());
    }

    let k_strata = cfg.n_strata;
    // Near-equal partition with the first (remainder) parts one larger
    // (repository choice; documented above).
    let part = |total: usize| -> Vec<usize> {
        let q = total / k_strata;
        let r = total % k_strata;
        (0..k_strata).map(|s| q + usize::from(s < r)).collect()
    };
    let stratum_sizes = part(n_items);
    let stage_lengths = part(cfg.test_length);
    for s in 0..k_strata {
        if stratum_sizes[s] < stage_lengths[s] {
            return Err(format!(
                "stratum {} has {} items but stage {} needs {}: pool cannot support this design",
                s, stratum_sizes[s], s, stage_lengths[s]
            ));
        }
    }

    // Stable ascending sort by (a, original index).
    let mut order: Vec<usize> = (0..n_items).collect();
    order.sort_by(|&i, &j| a[i].partial_cmp(&a[j]).expect("finite a").then(i.cmp(&j)));
    let mut stratum = vec![0usize; n_items];
    let mut members: Vec<Vec<usize>> = Vec::with_capacity(k_strata);
    {
        let mut pos = 0usize;
        for (s, &sz) in stratum_sizes.iter().enumerate() {
            let slice = &order[pos..pos + sz];
            for &i in slice {
                stratum[i] = s;
            }
            members.push(slice.to_vec());
            pos += sz;
        }
    }

    let grid: Vec<f64> = (0..cfg.q_theta)
        .map(|q| -4.0 + 8.0 * q as f64 / (cfg.q_theta - 1) as f64)
        .collect();
    let log_prior: Vec<f64> = grid.iter().map(|&t| -0.5 * t * t).collect();

    let mut rng = Lcg(cfg.seed.wrapping_mul(2654435761).wrapping_add(1));
    let mut a_count = vec![0u64; n_items];
    let mut eap_scratch: Vec<f64> = Vec::with_capacity(cfg.q_theta);
    let mut sq_err = 0.0;
    let mut bias = 0.0;

    for _p in 0..cfg.n_simulees {
        let theta_true = rng.normal();
        let mut used = vec![false; n_items];
        let mut responses: Vec<(usize, f64)> = Vec::with_capacity(cfg.test_length);
        let mut theta_hat = 0.0;

        for (stage, &len) in stage_lengths.iter().enumerate() {
            for _ in 0..len {
                // SELECT: b-matching within the active stratum only.
                let mut best: Option<usize> = None;
                let mut best_d = f64::INFINITY;
                for &i in &members[stage] {
                    if used[i] {
                        continue;
                    }
                    let d = (b[i] - theta_hat).abs();
                    let better = match best {
                        None => true,
                        Some(j) => d < best_d || (d == best_d && i < j),
                    };
                    if better {
                        best_d = d;
                        best = Some(i);
                    }
                }
                let s = best.expect("stratum sizes validated >= stage lengths");
                used[s] = true;
                a_count[s] += 1;
                let p_true = p3pl(theta_true, a[s], b[s], c[s]);
                let y = if rng.next_f64() < p_true { 1.0 } else { 0.0 };
                responses.push((s, y));
                theta_hat = eap_interim(a, b, c, &responses, &grid, &log_prior, &mut eap_scratch);
            }
        }
        let e = theta_hat - theta_true;
        sq_err += e * e;
        bias += e;
    }

    let n = cfg.n_simulees as f64;
    let exposure: Vec<f64> = a_count.iter().map(|&x| x as f64 / n).collect();
    let max_exposure = exposure.iter().cloned().fold(0.0_f64, f64::max);
    Ok(AStratifiedResult {
        exposure,
        max_exposure,
        stratum,
        stage_lengths,
        theta_rmse: (sq_err / n).sqrt(),
        theta_bias: bias / n,
    })
}

// ===================== Kullback-Leibler (global information) item selection =====================
//
// Chang and Ying's (1996) global-information criterion for CAT item selection.
//
// Source status: the original paper WAS consulted (Chang & Ying, 1996,
// Definitions 2.1-2.2 define item/test KL information with the expectation
// taken under the true parameter and note non-negativity and non-symmetry;
// Equation 17 constructs the interval index around the provisional estimate;
// Equation 18 motivates the shrinking half-width `delta_n = r / sqrt(n)`, with
// `r = 3` used in their Study 1). The Bernoulli form below was additionally
// verified against the catR implementation (Magis & Raiche, 2012, `KL.R`),
// which computes `P(th0) log[P(th0)/P(th)] + Q(th0) log[Q(th0)/Q(th)]`.
//
// Contract (explicit): [`kl_information`] returns the UNNORMALIZED area
// `integral_{theta0-delta}^{theta0+delta} K_i(theta || theta0) d theta`, not the
// interval average. For a common `delta` across items the argmax is identical;
// the pinned oracle constants in the tests are areas.
//
// Small-delta connection to Fisher information (derived here, verified
// independently by the adversarial spec review): with `u = theta - theta0`,
// `K(theta || theta0) = (1/2) I(theta0) u^2 + O(u^3)`, so the area is
// `I(theta0) delta^3 / 3 + O(delta^5)` -- the criterion collapses to local
// (Fisher) maximum-information selection as the interval shrinks.
//
// Integration is composite Simpson with 2048 panels (an implementation choice,
// not from the paper; error O(h^4) is ~1e-12 at CAT-scale half-widths, well
// inside the 1e-9 oracle tolerance used by the tests).
//
// References (APA 7th):
// Chang, H.-H., & Ying, Z. (1996). A global information approach to
//     computerized adaptive testing. *Applied Psychological Measurement,
//     20*(3), 213-229. https://doi.org/10.1177/014662169602000303
// Magis, D., & Raiche, G. (2012). Random generation of response patterns
//     under computerized adaptive testing with the R package catR. *Journal
//     of Statistical Software, 48*(8), 1-31. https://doi.org/10.18637/jss.v048.i08

/// Result of [`kl_select`].
#[derive(Clone, Debug)]
pub struct KlSelectResult {
    /// KL information index (unnormalized area) per pool item. Administered
    /// items keep their computed index (masking applies to selection only).
    pub index: Vec<f64>,
    /// Selected item: argmax of `index` over non-administered items
    /// (ties broken toward the LOWEST index, deterministically).
    pub selected: usize,
    /// The half-width actually used: `r / sqrt(n_administered)`.
    pub delta: f64,
}

/// Numerically stable `ln(1 + exp(z))` (softplus).
#[inline]
fn softplus(z: f64) -> f64 {
    if z > 0.0 {
        z + (-z).exp().ln_1p()
    } else {
        z.exp().ln_1p()
    }
}

/// `(ln P, ln Q)` for one 3PL item, computed in log space so extreme
/// `a * (theta - b)` never saturates: `Q = (1 - c) * sigma(-z)` gives
/// `ln Q = ln(1 - c) - softplus(z)` exactly, and for `c = 0`
/// `ln P = -softplus(-z)`. For `c > 0`, `P >= c` is bounded away from 0, so
/// the direct log is stable.
#[inline]
fn ln_pq_3pl(a: f64, b: f64, c: f64, theta: f64) -> (f64, f64) {
    let z = a * (theta - b);
    let ln_q = (1.0 - c).ln() - softplus(z);
    let ln_p = if c == 0.0 {
        -softplus(-z)
    } else {
        (c + (1.0 - c) / (1.0 + (-z).exp())).ln()
    };
    (ln_p, ln_q)
}

/// Pointwise Bernoulli KL divergence `K_i(theta || theta0)` for one 3PL item:
/// expectation under `theta0` (Chang & Ying, 1996, Definition 2.1). Computed
/// as `P0 (ln P0 - ln P) + Q0 (ln Q0 - ln Q)` entirely in log space -- no
/// probability clamping, so high-discrimination tails are integrated exactly
/// (a hard clamp at 1e-12 was measured to underestimate the
/// `a = 20, delta = 3` area by ~30%; see the pinned extreme-oracle test).
#[inline]
fn kl_pointwise(a: f64, b: f64, c: f64, theta: f64, theta0: f64) -> f64 {
    let (ln_p0, ln_q0) = ln_pq_3pl(a, b, c, theta0);
    let (ln_p, ln_q) = ln_pq_3pl(a, b, c, theta);
    // Zero-weight terms contribute 0 by the KL convention `0 ln 0 = 0`;
    // without the guard a saturated log pair (ln_q0 = ln_q = -inf at
    // extreme finite `a * (theta - b)`) yields `0 * NaN`.
    let term = |w: f64, d: f64| if w == 0.0 { 0.0 } else { w * d };
    term(ln_p0.exp(), ln_p0 - ln_p) + term(ln_q0.exp(), ln_q0 - ln_q)
}

/// Chang-Ying (1996) KL information index for every item: the unnormalized
/// area of `K_i(theta || theta0)` over `[theta0 - delta, theta0 + delta]`
/// (Eq. 17 form; see the section comment for the exact contract and sources).
///
/// `a`, `b`, `c` are 3PL parameters (`c = 0` gives 2PL). Errors on empty or
/// mismatched inputs, non-finite values, `a <= 0`, `c` outside `[0, 1)`, or
/// non-finite/non-positive `delta`.
pub fn kl_information(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    theta0: f64,
    delta: f64,
) -> Result<Vec<f64>, String> {
    let n = a.len();
    if n == 0 {
        return Err("kl_information: empty item pool".to_string());
    }
    if b.len() != n || c.len() != n {
        return Err(format!(
            "kl_information: length mismatch (a={}, b={}, c={})",
            n,
            b.len(),
            c.len()
        ));
    }
    if !theta0.is_finite() {
        return Err("kl_information: theta0 must be finite".to_string());
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err("kl_information: delta must be finite and > 0".to_string());
    }
    for i in 0..n {
        if !a[i].is_finite() || !b[i].is_finite() || !c[i].is_finite() {
            return Err(format!("kl_information: non-finite parameter (item {i})"));
        }
        if a[i] <= 0.0 {
            return Err(format!("kl_information: a must be > 0 (item {i})"));
        }
        if !(0.0..1.0).contains(&c[i]) {
            return Err(format!("kl_information: c must be in [0, 1) (item {i})"));
        }
    }
    // Composite Simpson, 2048 panels (even), over [theta0 - delta, theta0 + delta].
    const PANELS: usize = 2048;
    let h = 2.0 * delta / PANELS as f64;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let (ai, bi, ci) = (a[i], b[i], c[i]);
        let mut s = kl_pointwise(ai, bi, ci, theta0 - delta, theta0)
            + kl_pointwise(ai, bi, ci, theta0 + delta, theta0);
        for k in 1..PANELS {
            let t = theta0 - delta + k as f64 * h;
            let w = if k % 2 == 1 { 4.0 } else { 2.0 };
            s += w * kl_pointwise(ai, bi, ci, t, theta0);
        }
        out.push(s * h / 3.0);
    }
    Ok(out)
}

/// Select the next CAT item by the Chang-Ying (1996) KL criterion:
/// `argmax_i KL_i(theta0)` over non-administered items, with half-width
/// `delta = r / sqrt(n_administered)` (Eq. 18 rule; the paper's Study 1 uses
/// `r = 3`). Requires `n_administered >= 1` -- the shrinking-interval rule is
/// undefined at `n = 0`; for a first-item or fixed-interval variant call
/// [`kl_information`] with an explicit `delta` and take the argmax yourself.
pub fn kl_select(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    administered: &[bool],
    theta0: f64,
    n_administered: usize,
    r: f64,
) -> Result<KlSelectResult, String> {
    if administered.len() != a.len() {
        return Err(format!(
            "kl_select: administered mask length {} != pool size {}",
            administered.len(),
            a.len()
        ));
    }
    if n_administered == 0 {
        return Err(
            "kl_select: n_administered must be >= 1 (delta = r / sqrt(n) is undefined at n = 0; \
             use kl_information with an explicit delta for the first item)"
                .to_string(),
        );
    }
    if !r.is_finite() || r <= 0.0 {
        return Err("kl_select: r must be finite and > 0".to_string());
    }
    if administered.iter().all(|&u| u) {
        return Err("kl_select: all items already administered".to_string());
    }
    let delta = r / (n_administered as f64).sqrt();
    let index = kl_information(a, b, c, theta0, delta)?;
    let mut selected = usize::MAX;
    let mut best = f64::NEG_INFINITY;
    for (i, &v) in index.iter().enumerate() {
        if !administered[i] && v > best {
            best = v;
            selected = i;
        }
    }
    Ok(KlSelectResult {
        index,
        selected,
        delta,
    })
}
