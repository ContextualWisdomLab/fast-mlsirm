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

// ===================== Owen (1975) approximate Bayesian sequential CAT =====================
//
// Owen's restricted-Bayes sequential procedure for the three-parameter
// normal-ogive model: after each response the exact posterior is replaced by
// a normal with the posterior's exact first two moments, giving closed-form
// mean/variance recursions.
//
// Source status: the original JASA article (Owen, 1975) was NOT read (behind
// a paywall at review time). Every formula below was verified by the
// adversarial spec review against (a) van der Linden (1998), whose Appendix
// reproduces Owen's update equations A.1-A.6 including the guessing
// parameter, and (b) the open-source `irt` R package implementation
// (`src/est_ability_owen.cpp`, lines 31-56), and (c) an independent
// derivation via the joint-Gaussian representation `Y = a(theta - b) + Z`,
// `Z ~ N(0, 1)`: conditioning on `{Y > 0}` / `{Y < 0}` yields the
// truncated-bivariate-normal moment identities used here. High-precision
// numerical integration of the exact posterior confirmed all three pinned
// oracle cases to ~1e-13.
//
// Model: `P(X = 1 | theta) = c + (1 - c) Phi(a (theta - b))` with prior
// `theta ~ N(mu, sig2)`. Let `s2 = 1/a^2 + sig2`, `s = sqrt(s2)`, and
// `D = (mu - b)/s`.
//
// Correct response (`A = c + (1 - c) Phi(D)`, `K = (1 - c) phi(D) / A`):
//   `mu'   = mu + (sig2 / s) K`
//   `sig2' = sig2 - (sig2^2 / s2) K (K + D)`
// Incorrect response (`c` cancels after normalization; `L = phi(D) / (1 - Phi(D))`):
//   `mu'   = mu - (sig2 / s) L`
//   `sig2' = sig2 - (sig2^2 / s2) L (L - D)`
//
// Item selection is Owen's popular rule -- administer the unadministered item
// whose difficulty is closest to the current posterior mean (`|b_i - mu|`;
// argmin over a finite pool, ties to the lowest index). Minimum expected
// posterior variance is a distinct, more expensive criterion van der Linden
// (1998) attributes to Owen only as an alternative; it is NOT implemented
// here. Owen's stopping rule is a posterior-variance threshold; the fixed
// test length is an implementation cap, not the paper's rule.
//
// References (APA 7th):
// Owen, R. J. (1975). A Bayesian sequential procedure for quantal response in
//     the context of adaptive mental testing. *Journal of the American
//     Statistical Association, 70*(350), 351-356.
//     https://doi.org/10.1080/01621459.1975.10479871 (NOT read; historical target)
// van der Linden, W. J. (1998). *Bayesian item selection criteria for adaptive
//     testing* (Research Report 96-01). University of Twente. (ERIC ED424235;
//     Appendix Eqs. A.1-A.6 verified)
// Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in
//     a microcomputer environment. *Applied Psychological Measurement, 6*(4),
//     431-444. https://doi.org/10.1177/014662168200600405 (READ; framing of
//     posterior-mean estimate and variance-based termination)

/// Result of [`owen_cat`].
#[derive(Clone, Debug)]
pub struct OwenCatResult {
    /// Items administered, in administration order.
    pub administered: Vec<usize>,
    /// Posterior mean after each administered item.
    pub mu_trace: Vec<f64>,
    /// Posterior variance after each administered item.
    pub sig2_trace: Vec<f64>,
    /// Final posterior mean (Owen's ability estimate).
    pub mu: f64,
    /// Final posterior variance.
    pub sig2: f64,
}

/// Standard normal CDF via the crate's `erfc` approximation (|error| < 1.2e-7;
/// see `fitstats::erfc`). Oracle tests use tolerances wider than this bound.
#[inline]
fn norm_cdf(z: f64) -> f64 {
    0.5 * crate::fitstats::erfc(-z / std::f64::consts::SQRT_2)
}

#[inline]
fn norm_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// One Owen (1975) posterior moment update for a single 3PNO item. Returns
/// `(mu', sig2')`. See the section comment for the exact formulas, their
/// verification status, and references.
///
/// Errors on non-finite inputs, `a <= 0`, `c` outside `[0, 1)`,
/// `sig2 <= 0`, or when the update degenerates (non-finite or non-positive
/// `sig2'`) because the prior mean sits so deep in the response's
/// improbable tail that the normal approximation breaks down (e.g. an
/// incorrect response at `D >~ 27`, where `Phi(-D)` underflows to zero).
pub fn owen_update(
    a: f64,
    b: f64,
    c: f64,
    response: bool,
    mu: f64,
    sig2: f64,
) -> Result<(f64, f64), String> {
    if !a.is_finite() || a <= 0.0 {
        return Err(format!("owen_update: a must be finite positive, got {a}"));
    }
    if !b.is_finite() {
        return Err("owen_update: b must be finite".to_string());
    }
    if !c.is_finite() || !(0.0..1.0).contains(&c) {
        return Err(format!("owen_update: c must be in [0, 1), got {c}"));
    }
    if !mu.is_finite() {
        return Err("owen_update: mu must be finite".to_string());
    }
    if !sig2.is_finite() || sig2 <= 0.0 {
        return Err(format!(
            "owen_update: sig2 must be finite positive, got {sig2}"
        ));
    }
    let s2 = 1.0 / (a * a) + sig2;
    let s = s2.sqrt();
    let d = (mu - b) / s;
    let (new_mu, new_sig2) = if response {
        let big_a = c + (1.0 - c) * norm_cdf(d);
        let k = (1.0 - c) * norm_pdf(d) / big_a;
        (mu + (sig2 / s) * k, sig2 - (sig2 * sig2 / s2) * k * (k + d))
    } else {
        // 1 - Phi(d) computed as Phi(-d) = 0.5 erfc(d / sqrt 2): the erfc
        // approximation carries an exact exp(-z^2) factor, so the tail is
        // relatively accurate, whereas `1.0 - norm_cdf(d)` cancels
        // catastrophically for d >~ 5 (impl-review finding: at d = 8 the
        // subtractive form inflated sig2' by ~5x instead of shrinking it).
        let l = norm_pdf(d) / norm_cdf(-d);
        (mu - (sig2 / s) * l, sig2 - (sig2 * sig2 / s2) * l * (l - d))
    };
    if !new_mu.is_finite() || !new_sig2.is_finite() || new_sig2 <= 0.0 {
        return Err(format!(
            "owen_update: degenerate posterior (mu' = {new_mu}, sig2' = {new_sig2}); \
             inputs are too extreme for the normal approximation"
        ));
    }
    Ok((new_mu, new_sig2))
}

/// Owen (1975) sequential CAT against a caller-supplied full-pool response
/// vector: repeatedly administer the unadministered item minimizing
/// `|b_i - mu|` (ties to the lowest index), update the posterior moments with
/// [`owen_update`], and stop after `test_length` items or as soon as the
/// posterior variance drops to `sig2_stop` or below (Owen's variance-threshold
/// stopping rule), whichever comes first.
///
/// `responses[i]` must be 0 or 1 and is consulted only if item `i` is
/// administered. Errors on empty/mismatched inputs, invalid item parameters,
/// invalid prior, `test_length` of 0 or exceeding the pool, or a
/// non-finite/non-positive `sig2_stop`.
pub fn owen_cat(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    responses: &[u8],
    mu0: f64,
    sig2_0: f64,
    test_length: usize,
    sig2_stop: Option<f64>,
) -> Result<OwenCatResult, String> {
    let n = a.len();
    if n == 0 {
        return Err("owen_cat: empty item pool".to_string());
    }
    if b.len() != n || c.len() != n || responses.len() != n {
        return Err(format!(
            "owen_cat: length mismatch (a={}, b={}, c={}, responses={})",
            n,
            b.len(),
            c.len(),
            responses.len()
        ));
    }
    for i in 0..n {
        if !a[i].is_finite() || a[i] <= 0.0 {
            return Err(format!("owen_cat: a[{i}] must be finite positive"));
        }
        if !b[i].is_finite() {
            return Err(format!("owen_cat: b[{i}] must be finite"));
        }
        if !c[i].is_finite() || !(0.0..1.0).contains(&c[i]) {
            return Err(format!("owen_cat: c[{i}] must be in [0, 1)"));
        }
        if responses[i] > 1 {
            return Err(format!("owen_cat: responses[{i}] must be 0 or 1"));
        }
    }
    if !mu0.is_finite() {
        return Err("owen_cat: mu0 must be finite".to_string());
    }
    if !sig2_0.is_finite() || sig2_0 <= 0.0 {
        return Err("owen_cat: sig2_0 must be finite positive".to_string());
    }
    if test_length == 0 || test_length > n {
        return Err(format!(
            "owen_cat: test_length must be in 1..={n}, got {test_length}"
        ));
    }
    if let Some(t) = sig2_stop {
        if !t.is_finite() || t <= 0.0 {
            return Err("owen_cat: sig2_stop must be finite positive".to_string());
        }
    }

    let mut administered = Vec::with_capacity(test_length);
    let mut mu_trace = Vec::with_capacity(test_length);
    let mut sig2_trace = Vec::with_capacity(test_length);
    let mut used = vec![false; n];
    let (mut mu, mut sig2) = (mu0, sig2_0);
    for _ in 0..test_length {
        let mut best: Option<(f64, usize)> = None;
        for i in 0..n {
            if used[i] {
                continue;
            }
            let dist = (b[i] - mu).abs();
            if best.map_or(true, |(bd, _)| dist < bd) {
                best = Some((dist, i));
            }
        }
        let (_, item) = best.expect("test_length <= n guarantees a free item");
        used[item] = true;
        let (m, v) = owen_update(a[item], b[item], c[item], responses[item] == 1, mu, sig2)?;
        mu = m;
        sig2 = v;
        administered.push(item);
        mu_trace.push(mu);
        sig2_trace.push(sig2);
        if sig2_stop.is_some_and(|t| sig2 <= t) {
            break;
        }
    }
    Ok(OwenCatResult {
        administered,
        mu_trace,
        sig2_trace,
        mu,
        sig2,
    })
}

// ===================== Kingsbury-Zara (1989) constrained CAT content balancing =====================
//
// Kingsbury and Zara (1989) introduced constrained CAT (C-CAT) procedures
// for selecting items under content-area constraints. CITATION GOVERNANCE:
// the primary full text was NOT read (not obtainable in this environment);
// the exact content-balancing rule implemented here follows the catR
// `nextItem` documentation and source reproduction of the Kingsbury-Zara
// content-balancing control (READ: catR/R/nextItem.R, cbControl branch, and
// the nextItem manual page): first cover any content group with zero
// administered items, then select from the eligible group maximizing
// `target_prop - empirical_prop`, and finally choose the most informative
// item within that group. catR breaks group/item ties RANDOMLY; this
// implementation deterministically takes the LOWEST index (documented
// deviation for reproducible tests). Skipping groups with no unadministered
// item is an implementation safety rule, not verified K&Z text.
//
// The logistic 3PL Fisher information used within the chosen group,
//   I_i(theta) = a_i^2 * (Q_i / P_i) * ((P_i - c_i) / (1 - c_i))^2,
// was verified against catR's `Ii.R` (I = dP^2 / (P Q)) with `Pi.R`'s
// P/dP at D = 1, d = 1, which reduce algebraically to this form
// (adversarial spec review against catR formulas).
//
// References:
// Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items
// for computerized adaptive tests. Applied Measurement in Education, 2(4),
// 359-375. https://doi.org/10.1207/s15324818ame0204_6 (NOT read; rule per
// the catR reproduction cited below)
// Magis, D., & Raiche, G. (2012). Random generation of response patterns
// under computerized adaptive testing with the R package catR. Journal of
// Statistical Software, 48(8), 1-31. https://doi.org/10.18637/jss.v048.i08

/// Result of one Kingsbury-Zara constrained-CAT selection step.
#[derive(Debug, Clone)]
pub struct CcatSelectResult {
    /// Selected item index (unadministered, inside `group`).
    pub selected: usize,
    /// Content group the selection was constrained to.
    pub group: usize,
    /// Per-group `target - empirical` discrepancies (diagnostics; the
    /// zero-coverage priority rule may override the argmax of this vector).
    pub discrepancy: Vec<f64>,
    /// Logistic 3PL Fisher information at `theta0` for ALL items
    /// (administered items keep their value; masking applies to selection
    /// only, matching `kl_select`).
    pub info: Vec<f64>,
}

/// One Kingsbury-Zara content-balanced CAT selection step. See the section
/// comment for the exact rule, its source status, and references.
///
/// `groups[i]` is the content area of item `i` (must be `< targets.len()`);
/// `targets` are strictly positive proportions summing to 1 (tol 1e-8).
/// Errors on empty/mismatched inputs, invalid item parameters, non-finite
/// `theta0`, invalid targets, or when no group has an unadministered item.
pub fn ccat_select(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    groups: &[usize],
    targets: &[f64],
    administered: &[bool],
    theta0: f64,
) -> Result<CcatSelectResult, String> {
    let n = a.len();
    if n == 0 {
        return Err("ccat_select: empty item pool".to_string());
    }
    if b.len() != n || c.len() != n || groups.len() != n || administered.len() != n {
        return Err(format!(
            "ccat_select: length mismatch (a: {n}, b: {}, c: {}, groups: {}, administered: {})",
            b.len(),
            c.len(),
            groups.len(),
            administered.len()
        ));
    }
    let n_groups = targets.len();
    if n_groups == 0 {
        return Err("ccat_select: targets must be non-empty".to_string());
    }
    if !theta0.is_finite() {
        return Err("ccat_select: theta0 must be finite".to_string());
    }
    let mut tsum = 0.0;
    for (g, &t) in targets.iter().enumerate() {
        if !t.is_finite() || t <= 0.0 {
            return Err(format!(
                "ccat_select: targets must be finite positive, got targets[{g}] = {t}"
            ));
        }
        tsum += t;
    }
    if (tsum - 1.0).abs() > 1e-8 {
        return Err(format!("ccat_select: targets must sum to 1, got {tsum}"));
    }
    for i in 0..n {
        if !a[i].is_finite() || a[i] <= 0.0 {
            return Err(format!(
                "ccat_select: a must be finite positive, got a[{i}] = {}",
                a[i]
            ));
        }
        if !b[i].is_finite() {
            return Err(format!("ccat_select: b[{i}] must be finite"));
        }
        if !c[i].is_finite() || !(0.0..1.0).contains(&c[i]) {
            return Err(format!(
                "ccat_select: c must be in [0, 1), got c[{i}] = {}",
                c[i]
            ));
        }
        if groups[i] >= n_groups {
            return Err(format!(
                "ccat_select: groups[{i}] = {} out of range for {n_groups} targets",
                groups[i]
            ));
        }
    }

    // Administered counts and eligibility (a group is eligible if it still
    // has at least one unadministered item).
    let mut k_g = vec![0usize; n_groups];
    let mut eligible = vec![false; n_groups];
    let mut k = 0usize;
    for i in 0..n {
        if administered[i] {
            k_g[groups[i]] += 1;
            k += 1;
        } else {
            eligible[groups[i]] = true;
        }
    }
    if !eligible.iter().any(|&e| e) {
        return Err("ccat_select: all items administered".to_string());
    }

    let discrepancy: Vec<f64> = (0..n_groups)
        .map(|g| targets[g] - if k > 0 { k_g[g] as f64 / k as f64 } else { 0.0 })
        .collect();

    // catR rule: any eligible group with zero administered items has
    // priority (lowest index = documented deterministic substitute for
    // catR's random tie); otherwise the eligible group with the maximal
    // target-minus-empirical discrepancy wins (strict > keeps the lowest
    // index on ties).
    let group = match (0..n_groups).find(|&g| eligible[g] && k_g[g] == 0) {
        Some(g) => g,
        None => {
            let mut best: Option<(f64, usize)> = None;
            for g in 0..n_groups {
                if eligible[g] && best.map_or(true, |(bd, _)| discrepancy[g] > bd) {
                    best = Some((discrepancy[g], g));
                }
            }
            best.expect("at least one eligible group").1
        }
    };

    // Logistic 3PL Fisher information at theta0 for every item, computed in
    // log space for numerical robustness (impl-review rounds 1-2 findings;
    // regression-tested): the naive q/p * r^2 form produced NaN via inf * 0
    // at logistic underflow (P -> c, including subnormal c) and spurious
    // +inf from multiplication order at extreme a, and a pointwise p == 0
    // guard masked genuinely informative extreme items. Algebra: with
    // z = a(theta0 - b) and L = sigmoid(z), P = c + (1 - c) L, so
    // r = (P - c)/(1 - c) = L exactly and
    // I = a^2 (1 - c)(1 - L) L^2 / (c + (1 - c) L); when c = 0 this reduces
    // to I = a^2 L (1 - L). ln L = -softplus(-z) and ln(1-L) = -softplus(z)
    // are stable for all finite (or overflowed-to-inf) z. A genuinely
    // astronomical information (a >= ~1e155 near theta0 = b) still
    // overflows to +inf, which orders correctly in the argmax below.
    let softplus = |x: f64| {
        if x > 0.0 {
            x + (-x).exp().ln_1p()
        } else {
            x.exp().ln_1p()
        }
    };
    let info: Vec<f64> = (0..n)
        .map(|i| {
            let z = a[i] * (theta0 - b[i]);
            let ln_l = -softplus(-z);
            let ln_1ml = -softplus(z);
            let ln_i = if c[i] == 0.0 {
                2.0 * a[i].ln() + ln_l + ln_1ml
            } else {
                // p >= c > 0, so ln(p) is finite even when L underflows.
                let p = c[i] + (1.0 - c[i]) * ln_l.exp();
                2.0 * a[i].ln() + (1.0 - c[i]).ln() + ln_1ml + 2.0 * ln_l - p.ln()
            };
            ln_i.exp()
        })
        .collect();

    // Most informative unadministered item within the chosen group; strict >
    // keeps the lowest index on ties.
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if groups[i] == group && !administered[i] && best.map_or(true, |(bi, _)| info[i] > bi) {
            best = Some((info[i], i));
        }
    }
    let selected = best.expect("chosen group is eligible").1;

    Ok(CcatSelectResult {
        selected,
        group,
        discrepancy,
        info,
    })
}

// ===================== Owen-approximate posterior-predictive EPV item selection =====================
//
// `epv_select` implements Owen-approximate posterior-predictive EPV
// (expected posterior variance) item selection for the three-parameter
// normal-ogive model maintained by [`owen_update`]. For each item it
// computes the normal-posterior predictive probability
//   p*_i = c_i + (1 - c_i) Phi(a_i (mu - b_i) / sqrt(1 + a_i^2 sig2)),
// obtains the two Owen normal-approximation outcome variances from
// [`owen_update`], and scores
//   EPV_i = p*_i sig2_plus_i + (1 - p*_i) sig2_minus_i,
// selecting the unadministered argmin (ties to the lowest index).
//
// CITATION GOVERNANCE / SCOPE (adversarial spec review, epv_spec_review.md):
// this is NOT the exact van der Linden (1998) MEPV criterion, which is
// defined with response probabilities at the current ability estimate and
// true/numerical posterior variances (READ: van der Linden's freely
// available University of Twente/ERIC report, Research Report 96-01, ERIC
// ED424235, Eq. (14); catR EPV.R/EPV.Rd; mirtCAT selection_criteria.R
// 'MEPV'). The Psychometrika 63(2) journal body and Owen (1975) were NOT
// read. The predictive identity E[Phi(alpha + beta Z)] =
// Phi(alpha / sqrt(1 + beta^2)) for Z ~ N(0,1) applied to
// P(theta) = c + (1 - c) Phi(a (theta - b)) under theta ~ N(mu, sig2) was
// hand-derived and verified in the spec review; the outcome variances are
// the crate's Owen closed-form updates, so the whole criterion is an
// explicitly labeled Owen approximation of the posterior-predictive EPV.
//
// References (APA 7th):
// van der Linden, W. J. (1998). Bayesian item selection criteria for
//     adaptive testing (Research Report 96-01). University of Twente.
//     (ERIC ED424235 report text READ; Psychometrika 63(2) body NOT read)
// van der Linden, W. J. (1998). Bayesian item selection criteria for
//     adaptive testing. Psychometrika, 63(2), 201-216.
//     https://doi.org/10.1007/BF02294775 (metadata only)
// Owen, R. J. (1975). A Bayesian sequential procedure for quantal response
//     in the context of adaptive mental testing. Journal of the American
//     Statistical Association, 70(350), 351-356. (NOT read; update formulas
//     per the crate's owen_update, verified against the 1998 report appendix)
// Magis, D., & Raiche, G. (2012). Random generation of response patterns
//     under computerized adaptive testing with the R package catR. Journal
//     of Statistical Software, 48(8), 1-31.
//     https://doi.org/10.18637/jss.v048.i08 (READ: EPV.R structural form)

/// Result of one [`epv_select`] step.
#[derive(Debug, Clone)]
pub struct EpvSelectResult {
    /// Selected item (unadministered argmin of `epv`, ties to lowest index).
    pub selected: usize,
    /// Owen-approximate expected posterior variance for every item.
    pub epv: Vec<f64>,
    /// Posterior-predictive success probability `p*_i` for every item.
    pub predictive: Vec<f64>,
}

/// One Owen-approximate posterior-predictive EPV selection step: given the
/// current normal posterior `theta ~ N(mu, sig2)`, score every item in the
/// pool (administered or not) and select the unadministered item minimizing
/// the expected posterior variance (ties to the lowest index). See the
/// section comment for the exact criterion, its scope label, and references.
///
/// Errors on empty/mismatched inputs, invalid item parameters (mirroring
/// [`owen_cat`]), an invalid prior, an all-administered pool, or when either
/// [`owen_update`] outcome degenerates for an unadministered item.
pub fn epv_select(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    administered: &[bool],
    mu: f64,
    sig2: f64,
) -> Result<EpvSelectResult, String> {
    let n = a.len();
    if n == 0 {
        return Err("epv_select: empty item pool".to_string());
    }
    if b.len() != n || c.len() != n || administered.len() != n {
        return Err(format!(
            "epv_select: length mismatch (a={}, b={}, c={}, administered={})",
            n,
            b.len(),
            c.len(),
            administered.len()
        ));
    }
    for i in 0..n {
        if !a[i].is_finite() || a[i] <= 0.0 {
            return Err(format!("epv_select: a[{i}] must be finite positive"));
        }
        if !b[i].is_finite() {
            return Err(format!("epv_select: b[{i}] must be finite"));
        }
        if !c[i].is_finite() || !(0.0..1.0).contains(&c[i]) {
            return Err(format!("epv_select: c[{i}] must be in [0, 1)"));
        }
    }
    if !mu.is_finite() {
        return Err("epv_select: mu must be finite".to_string());
    }
    if !sig2.is_finite() || sig2 <= 0.0 {
        return Err("epv_select: sig2 must be finite positive".to_string());
    }
    if administered.iter().all(|&x| x) {
        return Err("epv_select: all items administered".to_string());
    }

    let mut epv = Vec::with_capacity(n);
    let mut predictive = Vec::with_capacity(n);
    for i in 0..n {
        // Predictive p*_i = c + (1 - c) Phi(a (mu - b) / sqrt(1 + a^2 sig2)).
        // The argument equals owen_update's d = (mu - b) / sqrt(1/a^2 + sig2)
        // exactly (multiply numerator and denominator by a > 0); the spec
        // review verified this sign convention against the crate.
        let d = (mu - b[i]) / (1.0 / (a[i] * a[i]) + sig2).sqrt();
        let p_star = c[i] + (1.0 - c[i]) * norm_cdf(d);
        let update_plus = owen_update(a[i], b[i], c[i], true, mu, sig2);
        let update_minus = owen_update(a[i], b[i], c[i], false, mu, sig2);
        match (update_plus, update_minus) {
            (Ok((_, sig2_plus)), Ok((_, sig2_minus))) => {
                epv.push(p_star * sig2_plus + (1.0 - p_star) * sig2_minus);
            }
            (Err(_), _) | (_, Err(_)) if administered[i] => {
                // Administered items never affect selection, so keep diagnostics
                // aligned by storing NaN instead of aborting the whole step.
                epv.push(f64::NAN);
            }
            (Err(e), _) | (_, Err(e)) => return Err(e),
        }
        predictive.push(p_star);
    }

    // Unadministered argmin; strict < keeps the lowest index on ties.
    let mut best: Option<(f64, usize)> = None;
    for i in 0..n {
        if !administered[i] && best.map_or(true, |(be, _)| epv[i] < be) {
            best = Some((epv[i], i));
        }
    }
    let selected = best
        .expect("checked above that some item is unadministered")
        .1;

    Ok(EpvSelectResult {
        selected,
        epv,
        predictive,
    })
}

// ===================== Wald SPRT classification for CAT =====================
//
// `sprt_classify` implements single-cut, binary-response SPRT classification
// (Wald's sequential probability ratio test applied to IRT classification
// testing). Two point hypotheses around the cut score,
//   theta0 = theta_cut - delta,  theta1 = theta_cut + delta,
// are compared through the cumulative binary log-likelihood ratio under the
// D = 1 logistic 3PL
//   P_i(theta) = c_i + (1 - c_i) / (1 + exp(-a_i (theta - b_i))),
//   LLR_k = sum_{i<=k} [ u_i ln(P_i(theta1)/P_i(theta0))
//                      + (1 - u_i) ln((1 - P_i(theta1))/(1 - P_i(theta0))) ],
// against the log Wald boundaries
//   A = ln((1 - beta) / alpha),  B = ln(beta / (1 - alpha)).
// Responses are walked in order and the FIRST crossing decides (inclusive
// comparisons, matching catIrt): LLR_k >= A -> "above" with n_used = k;
// LLR_k <= B -> "below" with n_used = k; no crossing -> "continue" with
// n_used = len(responses).
//
// CITATION GOVERNANCE / SCOPE (adversarial spec review, sprt_spec_review.md):
// boundaries and the binary log-likelihood-ratio form were verified against
// READ sources: catIrt R/termSPRT.R + R/logLik.brm.R + R/p.brm.R (GitHub
// swnydick/catIrt) and Thompson (2007), p. 7. Reckase (1983) and Eggen
// (1999) are historical citations via Thompson and were NOT directly read.
// This function implements only a single-cut binary 3PL SPRT with D = 1
// logistic-scale item parameters; it is not a multi-cut, polytomous, or
// D = 1.7 compatibility layer (parameters calibrated on the D = 1.7 metric
// must be rescaled by the caller, a_D1 = 1.7 * a_D17, before use).
//
// The returned decision/n_used are first-crossing SPRT results. llr_trace is
// computed for ALL supplied responses as an offline diagnostic; entries after
// n_used are counterfactual replay values - live CAT would terminate at
// n_used and would not administer later items.
//
// References (APA 7th):
// Wald, A. (1947). Sequential analysis. Wiley. (NOT read; boundary forms
//     verified through the sources below)
// Thompson, N. A. (2007). A practitioner's guide for variable-length
//     computerized classification testing. Practical Assessment, Research &
//     Evaluation, 12(1). https://doi.org/10.7275/fq3r-zz60 (READ: p. 7
//     likelihood-ratio form and Wald decision points)
// Nydick, S. W. (2014). catIrt: An R package for simulating IRT-based
//     computerized adaptive tests. (READ: R/termSPRT.R boundary and
//     inclusive-comparison conventions; R/logLik.brm.R binary log
//     likelihood; R/p.brm.R D = 1 3PL)
// Eggen, T. J. H. M. (1999). Item selection in adaptive testing with the
//     sequential probability ratio test. Applied Psychological Measurement,
//     23(3), 249-261. (NOT read; historical citation via Thompson)
// Reckase, M. D. (1983). A procedure for decision making using tailored
//     testing. (NOT read; historical citation via Thompson)

/// Result of [`sprt_classify`]. `decision` is `"above"`, `"below"`, or
/// `"continue"`; `n_used` is the 1-based count of responses consumed by the
/// first boundary crossing (or all responses when no crossing occurs);
/// `llr_trace` holds the cumulative log-likelihood ratio after every supplied
/// response (entries past `n_used` are offline counterfactuals); `llr` is the
/// final trace entry.
#[derive(Debug, Clone)]
pub struct SprtResult {
    pub decision: &'static str,
    pub n_used: usize,
    pub llr_trace: Vec<f64>,
    pub llr: f64,
}

/// Single-cut binary-response Wald SPRT classification (see module comment
/// above for the exact verified contract and source status).
pub fn sprt_classify(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    responses: &[u8],
    theta_cut: f64,
    delta: f64,
    alpha: f64,
    beta: f64,
) -> Result<SprtResult, String> {
    let n = a.len();
    if n == 0 {
        return Err("sprt_classify: item pool is empty".into());
    }
    if b.len() != n || c.len() != n || responses.len() != n {
        return Err(format!(
            "sprt_classify: length mismatch (a: {}, b: {}, c: {}, responses: {})",
            n,
            b.len(),
            c.len(),
            responses.len()
        ));
    }
    for i in 0..n {
        if !a[i].is_finite() || a[i] <= 0.0 {
            return Err(format!("sprt_classify: a[{i}] must be finite and > 0"));
        }
        if !b[i].is_finite() {
            return Err(format!("sprt_classify: b[{i}] must be finite"));
        }
        if !c[i].is_finite() || !(0.0..1.0).contains(&c[i]) {
            return Err(format!(
                "sprt_classify: c[{i}] must be finite and in [0, 1)"
            ));
        }
        if responses[i] > 1 {
            return Err(format!("sprt_classify: responses[{i}] must be 0 or 1"));
        }
    }
    if !theta_cut.is_finite() {
        return Err("sprt_classify: theta_cut must be finite".into());
    }
    if !delta.is_finite() || delta <= 0.0 {
        return Err("sprt_classify: delta must be finite and > 0".into());
    }
    for (name, v) in [("alpha", alpha), ("beta", beta)] {
        if !v.is_finite() || v <= 0.0 || v >= 1.0 {
            return Err(format!(
                "sprt_classify: {name} must be finite and in (0, 1)"
            ));
        }
    }
    if alpha + beta >= 1.0 {
        return Err("sprt_classify: alpha + beta must be < 1".into());
    }

    let upper = ((1.0 - beta) / alpha).ln();
    let lower = (beta / (1.0 - alpha)).ln();
    let theta0 = theta_cut - delta;
    let theta1 = theta_cut + delta;
    // Stable softplus ln(1 + e^z): shift by max(z, 0) so exp never overflows.
    let softplus = |z: f64| -> f64 {
        if z > 0.0 {
            z + (-z).exp().ln_1p()
        } else {
            z.exp().ln_1p()
        }
    };
    // Stable log-probabilities under the D = 1 logistic 3PL
    // P = c + (1 - c) sigmoid(z), z = a (theta - b) (crate CAT convention;
    // catIrt p.brm.R). ln(1 - P) = ln(1 - c) - softplus(z) always; ln(P)
    // needs the log-sigmoid branch -softplus(-z) only when c = 0 (for c > 0
    // the direct form is bounded below by c and stays finite).
    let ln_p = |z: f64, ci: f64| -> f64 {
        if ci > 0.0 {
            (ci + (1.0 - ci) / (1.0 + (-z).exp())).ln()
        } else {
            -softplus(-z)
        }
    };

    let mut llr_trace = Vec::with_capacity(n);
    let mut cum = 0.0_f64;
    let mut decision = "continue";
    let mut n_used = n;
    for i in 0..n {
        let z0 = a[i] * (theta0 - b[i]);
        let z1 = a[i] * (theta1 - b[i]);
        let inc = if responses[i] == 1 {
            // ln(P(theta1)) - ln(P(theta0)), each log computed stably.
            ln_p(z1, c[i]) - ln_p(z0, c[i])
        } else {
            // ln(1-P(theta1)) - ln(1-P(theta0)); the ln(1-c) terms cancel.
            softplus(z0) - softplus(z1)
        };
        // Defensive: unreachable for validated inputs with the stable forms
        // above (kept as a hard failure rather than silently propagating).
        if !inc.is_finite() {
            return Err(format!(
                "sprt_classify: non-finite log-likelihood-ratio increment at item {i}"
            ));
        }
        cum += inc;
        llr_trace.push(cum);
        // First crossing decides; inclusive comparisons (catIrt termSPRT.R).
        if decision == "continue" {
            if cum >= upper {
                decision = "above";
                n_used = i + 1;
            } else if cum <= lower {
                decision = "below";
                n_used = i + 1;
            }
        }
    }
    Ok(SprtResult {
        decision,
        n_used,
        llr: *llr_trace.last().unwrap(),
        llr_trace,
    })
}

// ================ Confidence-interval (ACI) classification for CAT =========
//
// `ci_classify` implements the confidence-interval classification stopping
// rule for a single cut score with binary responses: after each response,
// compute the interim EAP ability estimate and its posterior SD, form the
// interval theta_hat +/- z_crit * SE, and classify as soon as the whole
// interval lies STRICTLY on one side of the cut (first strict crossing
// decides). EAP uses the crate CAT convention shared with `eap_interim`:
// a fixed uniform grid of 41 points on [-4, 4], standard-normal log prior
// -0.5 * theta^2 (no quadrature-weight multiplier), and the D = 1 logistic
// 3PL P_i(theta) = c_i + (1 - c_i) / (1 + exp(-a_i (theta - b_i))).
//
// For each prefix k = 1..n:
//   theta_hat_k = sum_q w_q theta_q / sum_q w_q
//   se_k        = sqrt(sum_q w_q (theta_q - theta_hat_k)^2 / sum_q w_q)
//   lower_k = theta_hat_k - z_crit * se_k, upper_k = theta_hat_k + z_crit * se_k
//   lower_k > theta_cut -> "above" (n_used = k);
//   upper_k < theta_cut -> "below" (n_used = k); else continue.
// No crossing -> "continue" with n_used = len(responses). Traces are filled
// for ALL supplied responses; entries after n_used are offline counterfactual
// replay values (live CAT would stop at n_used).
//
// CITATION GOVERNANCE / SCOPE (adversarial spec review,
// ci_classify_spec_review.md): the implemented confidence-interval stopping
// rule was verified against catIrt R/termCI.R, R/eapEst.R, and man/catIrt.Rd
// at commit c9e979e4812c27d95d367a7f097edfe8e93ac8eb (READ): form the
// interval theta_hat +/- z * SEM, where the EAP SEM is the posterior SD
// (sqrt(E[theta^2] - theta_hat^2)), and classify only when the full interval
// lies strictly within a category. The fixed 41-point [-4, 4] EAP grid and
// the caller-supplied z_crit (catIrt computes qnorm((1 + conf.lev) / 2) from
// a confidence level; passing that value is equivalent) are repository
// implementation choices. Thompson (2007), Kingsbury & Weiss (1983), and
// Eggen & Straetmans (2000) were NOT method-section verified in this
// iteration and are historical/background context only.
//
// References (APA 7th):
// Nydick, S. W. (2014). catIrt: An R package for simulating IRT-based
//     computerized adaptive tests. (READ: R/termCI.R interval rule and
//     strict within-bounds comparisons; R/eapEst.R posterior-SD SEM;
//     man/catIrt.Rd conf.lev parameterization and first-satisfied-criterion
//     termination)
// Kingsbury, G. G., & Weiss, D. J. (1983). A comparison of IRT-based
//     adaptive mastery testing and a sequential mastery testing procedure.
//     In D. J. Weiss (Ed.), New horizons in testing (pp. 257-283). Academic
//     Press. (NOT read; historical origin of ability-confidence-interval
//     classification)
// Thompson, N. A. (2007). A practitioner's guide for variable-length
//     computerized classification testing. Practical Assessment, Research &
//     Evaluation, 12(1). (NOT read for the CI method section in this
//     iteration; background only)
// Eggen, T. J. H. M., & Straetmans, G. J. J. M. (2000). Computerized
//     adaptive testing for classifying examinees into three categories.
//     Educational and Psychological Measurement, 60(5), 713-734. (NOT read;
//     historical)

/// Result of [`ci_classify`]. `decision` is `"above"`, `"below"`, or
/// `"continue"`; `n_used` is the 1-based count of responses consumed by the
/// first strict interval crossing (or all responses when no crossing
/// occurs); the four traces hold the interim EAP estimate, posterior SD,
/// and interval bounds after every supplied response (entries past `n_used`
/// are offline counterfactuals).
#[derive(Debug, Clone)]
pub struct CiResult {
    pub decision: &'static str,
    pub n_used: usize,
    pub theta_trace: Vec<f64>,
    pub se_trace: Vec<f64>,
    pub lower_trace: Vec<f64>,
    pub upper_trace: Vec<f64>,
}

/// Single-cut binary-response confidence-interval (ACI) classification (see
/// module comment above for the exact verified contract and source status).
pub fn ci_classify(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    responses: &[u8],
    theta_cut: f64,
    z_crit: f64,
) -> Result<CiResult, String> {
    let n = a.len();
    if n == 0 {
        return Err("ci_classify: item pool is empty".into());
    }
    if b.len() != n || c.len() != n || responses.len() != n {
        return Err(format!(
            "ci_classify: length mismatch (a: {}, b: {}, c: {}, responses: {})",
            n,
            b.len(),
            c.len(),
            responses.len()
        ));
    }
    for i in 0..n {
        if !a[i].is_finite() || a[i] <= 0.0 {
            return Err(format!("ci_classify: a[{i}] must be finite and > 0"));
        }
        if !b[i].is_finite() {
            return Err(format!("ci_classify: b[{i}] must be finite"));
        }
        if !c[i].is_finite() || !(0.0..1.0).contains(&c[i]) {
            return Err(format!("ci_classify: c[{i}] must be finite and in [0, 1)"));
        }
        if responses[i] > 1 {
            return Err(format!("ci_classify: responses[{i}] must be 0 or 1"));
        }
    }
    if !theta_cut.is_finite() {
        return Err("ci_classify: theta_cut must be finite".into());
    }
    if !z_crit.is_finite() || z_crit <= 0.0 {
        return Err("ci_classify: z_crit must be finite and > 0".into());
    }

    const Q: usize = 41;
    let grid: Vec<f64> = (0..Q)
        .map(|q| -4.0 + 8.0 * q as f64 / (Q - 1) as f64)
        .collect();
    // Cumulative log posterior weights, updated one response at a time.
    let mut logw: Vec<f64> = grid.iter().map(|&t| -0.5 * t * t).collect();

    let mut theta_trace = Vec::with_capacity(n);
    let mut se_trace = Vec::with_capacity(n);
    let mut lower_trace = Vec::with_capacity(n);
    let mut upper_trace = Vec::with_capacity(n);
    let mut decision = "continue";
    let mut n_used = n;
    for i in 0..n {
        for (q, &t) in grid.iter().enumerate() {
            let p = p3pl(t, a[i], b[i], c[i]).clamp(1e-12, 1.0 - 1e-12);
            logw[q] += if responses[i] == 1 {
                p.ln()
            } else {
                (1.0 - p).ln()
            };
        }
        let m = logw.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut den = 0.0;
        let mut num = 0.0;
        for (q, &t) in grid.iter().enumerate() {
            let w = (logw[q] - m).exp();
            den += w;
            num += w * t;
        }
        let theta_hat = num / den;
        let mut ss = 0.0;
        for (q, &t) in grid.iter().enumerate() {
            let w = (logw[q] - m).exp();
            ss += w * (t - theta_hat) * (t - theta_hat);
        }
        let se = (ss / den).sqrt();
        let lower = theta_hat - z_crit * se;
        let upper = theta_hat + z_crit * se;
        theta_trace.push(theta_hat);
        se_trace.push(se);
        lower_trace.push(lower);
        upper_trace.push(upper);
        // First STRICT crossing decides (catIrt termCI.R uses strict
        // within-bounds comparisons; equality means continue).
        if decision == "continue" {
            if lower > theta_cut {
                decision = "above";
                n_used = i + 1;
            } else if upper < theta_cut {
                decision = "below";
                n_used = i + 1;
            }
        }
    }
    Ok(CiResult {
        decision,
        n_used,
        theta_trace,
        se_trace,
        lower_trace,
        upper_trace,
    })
}

// ================ Lord self-scoring flexilevel testing ======================
//
// `flexilevel_administer` replays Lord's flexilevel design over a FULL 0/1
// response matrix: N (odd) items sorted ASCENDING by difficulty (caller
// responsibility; both sources assume a difficulty-ordered pool), Lord index
// i = column - (n - 1) with n = (N + 1) / 2 so that item 0 is the median.
// Each person starts at the median item; after a RIGHT answer the next item
// is the easiest not-yet-answered harder item, after a WRONG answer the
// hardest not-yet-answered easier item. In Lord's index arithmetic, if item
// i is the v-th administered:
//     i > 0: next = i + 1 (right) or i - v (wrong)
//     i < 0: next = i + v (right) or i - 1 (wrong)
// The i = 0 (first item) case is NOT covered by the index formula in the
// read text (it states i > 0 / i < 0 only); right -> +1 / wrong -> -1 at
// i = 0 follows from the verbal start-at-median rules and coincides with
// the i > 0 branch at v = 1, which is what this code uses.
//
// Self-scoring: after n answers let j be the (n+1)-th item that WOULD be
// administered. j > 0 ("blue": last answer right) gives number-right r = j
// and score x = r; j < 0 ("red": last answer wrong) gives r = n + j and
// x = r + 1/2. The identity r = (number of correct administered answers)
// was verified exhaustively against the routing in the spec oracle.
//
// `flexilevel_score_distribution` computes the exact conditional score
// distribution f(x | theta) by Lord's forward recursion over p_v(i), the
// probability that item i is the v-th administered: p_1(0) = 1 and
// p_{v+1}(next_right) += p_v(i) P_i, p_{v+1}(next_wrong) += p_v(i)(1 - P_i),
// with f(x) = p_{n+1}(j) under the score mapping above (x = j for integer
// scores; the half-integer mapping j = x - 1/2 - n is DERIVED from
// r = n + j and x = r + 1/2 -- the printed Eq. 2 is OCR-garbled in the
// available scan -- and is cross-checked exactly against exhaustive path
// enumeration in the spec oracle). The caller supplies P_i(theta) for the N
// sorted items, keeping the recursion ICC-agnostic (Lord's numerical study
// used a 3-parameter normal ogive; nothing in Eqs. 1-2 depends on it).
//
// Score lattice: x in {1/2, 1, 3/2, ..., n - 1/2, n} (2n points). x = 0 is
// impossible: an all-wrong path is red with r = 0 and scores x = 1/2.
//
// CITATION GOVERNANCE / SCOPE (adversarial spec review,
// flexilevel_spec_review.md): every routing/scoring rule above was verified
// against the two READ primary sources (full OCR text on file). The worked
// example RWWRWRRRWR and its administered sequence
// [0, 1, -1, -2, 2, -3, 3, 4, 5, -4] pin the routing; the answer string is
// readable in the 1971 scan, while the printed sequence line is OCR-blank,
// so the sequence itself is confirmed by applying the (readable) routing
// rules, not by transcription. Out of scope (documented): Lord's Eq. 3
// relative-efficiency ratio (derivable from mean/variance across theta),
// Eq. 4 normal-ogive ICC, and the 1970 answer-sheet layout material.
//
// References (APA 7th):
// Lord, F. M. (1970). The self-scoring flexilevel test (Research Bulletin
//     RB-70-43; ERIC ED042813). Educational Testing Service. (READ: design
//     rules, self-scoring properties 1-9, red +1/2 convention)
// Lord, F. M. (1971). A theoretical study of the measurement effectiveness
//     of flexilevel tests (Research Bulletin RB-71-6; ERIC ED051286).
//     Educational Testing Service. (READ: item-index transition rule,
//     score mapping j > 0 -> r = j / j < 0 -> r = v + j, Eqs. 1-2 forward
//     recursion for f(x | theta); Eq. 2 half-integer branch OCR-garbled,
//     mapping derived as documented above)

/// Result of [`flexilevel_administer`]: per-person administered column
/// indices in administration order (`n_administered` per person, flattened
/// row-major), number-right, red flag (1 iff last answer wrong), and the
/// self-scoring score on the half-integer lattice.
#[derive(Debug, Clone)]
pub struct FlexilevelAdminResult {
    pub n_administered: usize,
    pub items: Vec<usize>,
    pub number_right: Vec<u32>,
    pub is_red: Vec<u8>,
    pub score: Vec<f64>,
}

/// Result of [`flexilevel_score_distribution`]: the ascending score lattice
/// {1/2, 1, ..., n} with exact probabilities, plus their mean and variance.
#[derive(Debug, Clone)]
pub struct FlexilevelDistResult {
    pub scores: Vec<f64>,
    pub probs: Vec<f64>,
    pub mean: f64,
    pub variance: f64,
}

/// Lord-index routing step shared by both entry points: item `i` was the
/// `v`-th administered (v >= 1); returns (next-if-right, next-if-wrong).
#[inline]
fn flexilevel_next(i: i64, v: i64) -> (i64, i64) {
    if i >= 0 {
        (i + 1, i - v)
    } else {
        (i + v, i - 1)
    }
}

/// Deterministic flexilevel routing + self-scoring over a full response
/// matrix (row-major `n_persons x n_items`, entries 0/1; items pre-sorted
/// ascending by difficulty). See the module comment above for the verified
/// contract and source status.
pub fn flexilevel_administer(
    responses: &[u8],
    n_persons: usize,
    n_items: usize,
) -> Result<FlexilevelAdminResult, String> {
    if n_persons == 0 || n_items == 0 {
        return Err("flexilevel_administer: n_persons and n_items must be positive".into());
    }
    if n_items < 3 || n_items % 2 == 0 {
        return Err(format!(
            "flexilevel_administer: n_items must be odd and >= 3 (got {n_items})"
        ));
    }
    let expected = n_persons
        .checked_mul(n_items)
        .ok_or("flexilevel_administer: n_persons * n_items overflows")?;
    if responses.len() != expected {
        return Err(format!(
            "flexilevel_administer: responses has {} entries, expected {} ({} x {})",
            responses.len(),
            expected,
            n_persons,
            n_items
        ));
    }
    let n = (n_items + 1) / 2;
    let median = (n - 1) as i64; // column of Lord index 0
    let mut items = Vec::with_capacity(n_persons * n);
    let mut number_right = Vec::with_capacity(n_persons);
    let mut is_red = Vec::with_capacity(n_persons);
    let mut score = Vec::with_capacity(n_persons);
    for p in 0..n_persons {
        let row = &responses[p * n_items..(p + 1) * n_items];
        let mut i: i64 = 0; // Lord index of the current item
        let mut right: u32 = 0;
        let mut last_correct = false;
        for v in 1..=(n as i64) {
            let col = (i + median) as usize;
            let y = row[col];
            if y > 1 {
                return Err(format!(
                    "flexilevel_administer: responses[{}][{}] must be 0 or 1 (got {y})",
                    p, col
                ));
            }
            items.push(col);
            let (nr, nw) = flexilevel_next(i, v);
            if y == 1 {
                right += 1;
                last_correct = true;
                i = nr;
            } else {
                last_correct = false;
                i = nw;
            }
        }
        // i now holds j, the (n+1)-th item that WOULD be administered.
        let (r, x) = if i > 0 {
            (i as u32, i as f64)
        } else {
            let r = (n as i64 + i) as u32;
            (r, r as f64 + 0.5)
        };
        debug_assert_eq!(r, right, "Lord number-right identity");
        debug_assert_eq!(i > 0, last_correct, "blue iff last answer right");
        number_right.push(r);
        is_red.push(u8::from(i < 0));
        score.push(x);
    }
    Ok(FlexilevelAdminResult {
        n_administered: n,
        items,
        number_right,
        is_red,
        score,
    })
}

/// Exact conditional score distribution f(x | theta) of the flexilevel
/// self-score by Lord's forward recursion (see the module comment above).
/// `p[c]` is P(correct) on the c-th difficulty-sorted item at the fixed
/// ability of interest; `p.len()` = N must be odd and >= 3.
pub fn flexilevel_score_distribution(p: &[f64]) -> Result<FlexilevelDistResult, String> {
    let n_items = p.len();
    if n_items < 3 || n_items % 2 == 0 {
        return Err(format!(
            "flexilevel_score_distribution: p must have odd length >= 3 (got {n_items})"
        ));
    }
    for (c, &pc) in p.iter().enumerate() {
        if !pc.is_finite() || !(0.0..=1.0).contains(&pc) {
            return Err(format!(
                "flexilevel_score_distribution: p[{c}] must be finite and in [0, 1]"
            ));
        }
    }
    let n = (n_items + 1) / 2;
    let median = (n - 1) as i64;
    // p_v over Lord indices, stored on a dense offset grid [-n, n].
    let width = 2 * n + 1;
    let off = n as i64;
    let mut cur = vec![0.0_f64; width];
    cur[n] = 1.0; // p_1(0) = 1
    for v in 1..=(n as i64) {
        let mut nxt = vec![0.0_f64; width];
        for idx in 0..width {
            let pr = cur[idx];
            if pr == 0.0 {
                continue;
            }
            let i = idx as i64 - off;
            let pc = p[(i + median) as usize];
            let (jr, jw) = flexilevel_next(i, v);
            nxt[(jr + off) as usize] += pr * pc;
            nxt[(jw + off) as usize] += pr * (1.0 - pc);
        }
        cur = nxt;
    }
    // cur holds p_{n+1}(j); map to the score lattice {1/2, 1, ..., n}.
    let mut scores = Vec::with_capacity(2 * n);
    let mut probs = Vec::with_capacity(2 * n);
    for k in 1..=(2 * n) {
        let x = k as f64 * 0.5;
        let j = if k % 2 == 0 {
            (k / 2) as i64 // integer x (blue): j = x
        } else {
            (k / 2) as i64 - n as i64 // half-integer x (red): j = x - 1/2 - n
        };
        scores.push(x);
        probs.push(cur[(j + off) as usize]);
    }
    let mean: f64 = scores.iter().zip(&probs).map(|(x, w)| x * w).sum();
    let variance: f64 = scores
        .iter()
        .zip(&probs)
        .map(|(x, w)| (x - mean) * (x - mean) * w)
        .sum();
    Ok(FlexilevelDistResult {
        scores,
        probs,
        mean,
        variance,
    })
}

// ==================== Weiss stradaptive ability test =========================
//
// `stradaptive_administer` replays Weiss's stratified-adaptive (stradaptive)
// test over a FULL 0/1 response vector for one person: the pool is divided
// into S >= 2 difficulty strata (0 = easiest); within a stratum, items are
// administered in pool order (the source orders them by decreasing
// discrimination -- a caller responsibility that is NOT enforced here).
//
// Routing (READ, illustrated rule): start at `entry_stratum`; after a
// correct answer the target is the next more difficult stratum, after an
// incorrect answer the next less difficult stratum. The target is clamped
// to [0, S-1] at the pool boundaries (the source describes same-stratum
// substitution when no more-difficult stratum exists and continuing upward
// when the easiest stratum's supply is exhausted). When the clamped target
// stratum has no unused item, the next item is drawn from the LAST
// ADMINISTERED stratum; when that is also exhausted the test ends
// ("pool_exhausted", the Nancy N. record). DERIVED: the source prints
// same-stratum substitution only for the boundary/lower-exhausted cases;
// its generalization to any exhausted target is a derived choice anchored
// by the synthetic fixtures in the test suite, not by a printed record.
//
// Termination (READ): after each response, the CEILING stratum is the
// lowest stratum with n_administered >= min_items and proportion correct
// <= chance (chance = 1/(number of response options); this implementation
// is multiple-choice-only and requires 0 < chance < 1 -- the source's
// free-response chance = 0 discussion is out of scope). The test stops at
// the first response after which a ceiling exists, on pool exhaustion, or
// at `max_items`.
//
// Scoring methods 1-10 (READ, pp. 22-25) and the consistency score
// (pp. 26-27); NaN encodes the report's indeterminate "I":
//   m1  highest difficulty answered correctly.
//   m2  difficulty of the (n+1)-th item -- the item the routing rule would
//       administer next (NaN when none exists).
//   m3  highest difficulty answered correctly below the ceiling stratum
//       (upper bound = S when no ceiling was identified).
//   m4  mean difficulty (over the FULL pool) of the highest stratum with at
//       least one correct answer.
//   m5  mean difficulty of the (n+1)-th item's stratum. NaN whenever no
//       (n+1)-th item exists; this knowingly omits the Figure 7 record,
//       where the report extrapolates a hypothetical off-pool stratum mean
//       (2.62 + .655 = 3.27) after an off-the-top pool exhaustion.
//   m6  mean difficulty of the highest non-chance stratum (hnc).
//   m7  interpolated stratum difficulty at the hnc stratum (below).
//   m8  mean difficulty of all correctly answered items.
//   m9  mean difficulty of correct items in strata strictly between the
//       basal and ceiling strata (missing basal -> no lower bound; missing
//       ceiling -> upper bound S).
//   m10 mean difficulty of correct items at the hnc stratum.
//   consistency: population variance of the m9 difficulty set (DERIVED
//       definitional choice -- the source proposes "variance or standard
//       deviation" and prints NO worked consistency value; population
//       variance over the between-basal-and-ceiling correct set is the
//       reading implemented and pinned here).
//
// Derived definitional anchors (not printed verbatim in the source):
//   - hnc = ceiling - 1 when a ceiling exists (the Carol C. record prints
//     method 6 = -1.92 = the stratum-2 mean even though a higher stratum
//     reached p = .50, forcing hnc = c - 1 rather than a global search);
//     with no ceiling, hnc = the highest administered stratum with
//     proportion correct > chance (Nancy N.).
//   - basal = highest stratum strictly below the ceiling bound whose
//     administered items were ALL answered correctly (the John J. record
//     accepts a basal "based on only one item").
//
// METHOD 7 FORMULA PROVENANCE: the printed equation is OCR-garbled in the
// available scan ("A =c-1s(Pc-1.50)"). The reconstruction
//     m7 = D_hnc + step * (p_hnc - 1/2),
//     step = D_{hnc+1} - D_hnc   if p_hnc > 1/2,
//            D_hnc - D_{hnc-1}   if p_hnc < 1/2,
//     m7 = D_hnc exactly          if p_hnc == 1/2 (no adjacency needed),
// is DERIVED from the surrounding prose (score equals the stratum mean at
// p = .50 and moves toward the adjacent stratum otherwise) and CONFIRMED
// against five independently printed report values (1.37, -1.73, -.44,
// .80, 2.69). LIMIT (documented): all five printed pins have p > 1/2, so
// they confirm only the upper-step branch; the lower-step branch rests on
// prose plus the synthetic p < 1/2 pin in the test suite. When the
// adjacent stratum needed by the step does not exist (hnc at a pool edge),
// its mean is extrapolated by the mean between-stratum increment
// (D_{S-1} - D_0) / (S - 1); this extrapolation is likewise DERIVED (the
// report applies it once, to the Nancy N. record).
//
// References (APA 7th):
// Weiss, D. J. (1973). The stratified adaptive computerized ability test
//     (Research Report 73-3; ERIC ED084301). University of Minnesota,
//     Psychometric Methods Program. (READ: pool structure, entry,
//     branching, termination, scoring methods 1-10, consistency,
//     Figures 4-9 worked records, Tables 1-2)
// Lord, F. M. (1971). The self-scoring flexilevel test. Journal of
//     Educational Measurement, 8(3), 147-151. (NOT read; cited by Weiss as
//     a fixed-branching contrast -- the flexilevel implementation above
//     uses its own READ ERIC sources)

/// Result of [`stradaptive_administer`]. `administered`/`responses_taken`
/// list the pool indices and 0/1 answers in administration order; `reason`
/// is `"criterion"`, `"pool_exhausted"`, or `"max_items"`; `ceiling`,
/// `basal`, `hnc`, and `next_item` are `None` when undefined; `scores[k]`
/// holds scoring method k+1 (NaN = indeterminate), and `consistency` the
/// population variance of the method-9 set (NaN when that set is empty).
#[derive(Debug, Clone)]
pub struct StradaptiveResult {
    pub administered: Vec<usize>,
    pub responses_taken: Vec<u8>,
    pub reason: &'static str,
    pub ceiling: Option<usize>,
    pub basal: Option<usize>,
    pub hnc: Option<usize>,
    pub next_item: Option<usize>,
    pub scores: [f64; 10],
    pub consistency: f64,
}

/// Stradaptive routing + scoring over a hypothetical full response vector
/// (see the module comment above for the verified contract, derived-choice
/// labels, and source status).
pub fn stradaptive_administer(
    stratum: &[usize],
    difficulty: &[f64],
    responses: &[u8],
    entry_stratum: usize,
    chance: f64,
    min_items: usize,
    max_items: usize,
) -> Result<StradaptiveResult, String> {
    let n = stratum.len();
    if n == 0 {
        return Err("stradaptive_administer: item pool is empty".into());
    }
    if difficulty.len() != n || responses.len() != n {
        return Err(format!(
            "stradaptive_administer: length mismatch (stratum: {}, difficulty: {}, responses: {})",
            n,
            difficulty.len(),
            responses.len()
        ));
    }
    let s_max = *stratum.iter().max().unwrap();
    // Contiguous non-empty strata imply s_max + 1 <= n; guard before the
    // vec![...; n_strata] allocation so a huge stratum id cannot trigger an
    // enormous allocation (or overflow s_max + 1).
    if s_max >= n {
        return Err(format!(
            "stradaptive_administer: stratum {s_max} exceeds the item count {n} \
             (strata must cover 0..S-1 with every stratum non-empty)"
        ));
    }
    let n_strata = s_max + 1;
    if n_strata < 2 {
        return Err("stradaptive_administer: at least 2 strata are required".into());
    }
    let mut by_stratum: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for (i, &s) in stratum.iter().enumerate() {
        by_stratum[s].push(i);
    }
    for (k, items) in by_stratum.iter().enumerate() {
        if items.is_empty() {
            return Err(format!(
                "stradaptive_administer: stratum {k} has no items (strata must cover 0..{s_max})"
            ));
        }
    }
    for (i, &d) in difficulty.iter().enumerate() {
        if !d.is_finite() {
            return Err(format!(
                "stradaptive_administer: difficulty[{i}] must be finite"
            ));
        }
    }
    for (i, &r) in responses.iter().enumerate() {
        if r > 1 {
            return Err(format!(
                "stradaptive_administer: responses[{i}] must be 0 or 1 (got {r})"
            ));
        }
    }
    if entry_stratum >= n_strata {
        return Err(format!(
            "stradaptive_administer: entry_stratum {entry_stratum} out of range (pool has {n_strata} strata)"
        ));
    }
    if !chance.is_finite() || chance <= 0.0 || chance >= 1.0 {
        return Err(
            "stradaptive_administer: chance must be finite and strictly inside (0, 1) \
             (multiple-choice only; free-response chance = 0 is out of scope)"
                .into(),
        );
    }
    if min_items == 0 {
        return Err("stradaptive_administer: min_items must be >= 1".into());
    }
    if max_items == 0 {
        return Err("stradaptive_administer: max_items must be >= 1".into());
    }

    let mut used = vec![0usize; n_strata];
    let mut n_adm = vec![0usize; n_strata];
    let mut n_cor = vec![0usize; n_strata];
    let mut administered = Vec::new();
    let mut responses_taken = Vec::new();
    let mut cur = entry_stratum as i64;
    let mut last = entry_stratum;
    let reason;

    // Clamp the branch target to the pool, then fall back to the last
    // administered stratum when the target is exhausted (DERIVED rule; see
    // module comment).
    let pick = |target: i64, used: &[usize], last: usize| -> Option<usize> {
        let t = target.clamp(0, s_max as i64) as usize;
        if used[t] < by_stratum[t].len() {
            return Some(t);
        }
        if used[last] < by_stratum[last].len() {
            return Some(last);
        }
        None
    };
    let find_ceiling = |n_adm: &[usize], n_cor: &[usize]| -> Option<usize> {
        (0..n_strata)
            .find(|&k| n_adm[k] >= min_items && (n_cor[k] as f64 / n_adm[k] as f64) <= chance)
    };

    loop {
        if administered.len() >= max_items {
            reason = "max_items";
            break;
        }
        let t = match pick(cur, &used, last) {
            Some(t) => t,
            None => {
                reason = "pool_exhausted";
                break;
            }
        };
        let idx = by_stratum[t][used[t]];
        used[t] += 1;
        last = t;
        let r = responses[idx];
        administered.push(idx);
        responses_taken.push(r);
        n_adm[t] += 1;
        n_cor[t] += r as usize;
        cur = if r == 1 { t as i64 + 1 } else { t as i64 - 1 };
        if find_ceiling(&n_adm, &n_cor).is_some() {
            reason = "criterion";
            break;
        }
    }
    let ceiling = find_ceiling(&n_adm, &n_cor);
    let next_item = pick(cur, &used, last).map(|t| by_stratum[t][used[t]]);

    // Full-pool stratum mean difficulties + mean between-stratum increment.
    let mut d_mean = vec![0.0f64; n_strata];
    for k in 0..n_strata {
        let sum: f64 = by_stratum[k].iter().map(|&i| difficulty[i]).sum();
        d_mean[k] = sum / by_stratum[k].len() as f64;
    }
    let incr = (d_mean[n_strata - 1] - d_mean[0]) / (n_strata - 1) as f64;

    let corrects: Vec<usize> = administered
        .iter()
        .zip(&responses_taken)
        .filter(|&(_, &r)| r == 1)
        .map(|(&i, _)| i)
        .collect();
    let upper = ceiling.unwrap_or(n_strata); // exclusive bound for m3/m9
    let hnc = match ceiling {
        Some(c) => c.checked_sub(1),
        None => (0..n_strata)
            .rev()
            .find(|&k| n_adm[k] > 0 && (n_cor[k] as f64 / n_adm[k] as f64) > chance),
    };
    let basal = (0..upper)
        .rev()
        .find(|&k| n_adm[k] > 0 && n_cor[k] == n_adm[k]);

    let mut scores = [f64::NAN; 10];
    scores[0] = corrects
        .iter()
        .map(|&i| difficulty[i])
        .fold(f64::NAN, f64::max);
    if let Some(ni) = next_item {
        scores[1] = difficulty[ni];
        scores[4] = d_mean[stratum[ni]];
    }
    scores[2] = corrects
        .iter()
        .filter(|&&i| stratum[i] < upper)
        .map(|&i| difficulty[i])
        .fold(f64::NAN, f64::max);
    if let Some(hs) = corrects.iter().map(|&i| stratum[i]).max() {
        scores[3] = d_mean[hs];
    }
    if let Some(h) = hnc {
        scores[5] = d_mean[h];
        let p = n_cor[h] as f64 / n_adm[h] as f64;
        scores[6] = if p == 0.5 {
            d_mean[h] // both step branches agree; no adjacent stratum needed
        } else {
            let step = if p > 0.5 {
                let up = if h + 1 < n_strata {
                    d_mean[h + 1]
                } else {
                    d_mean[n_strata - 1] + incr // DERIVED extrapolation
                };
                up - d_mean[h]
            } else {
                let lo = if h > 0 {
                    d_mean[h - 1]
                } else {
                    d_mean[0] - incr // DERIVED extrapolation
                };
                d_mean[h] - lo
            };
            d_mean[h] + step * (p - 0.5)
        };
        let at_h: Vec<f64> = corrects
            .iter()
            .filter(|&&i| stratum[i] == h)
            .map(|&i| difficulty[i])
            .collect();
        if !at_h.is_empty() {
            scores[9] = at_h.iter().sum::<f64>() / at_h.len() as f64;
        }
    }
    if !corrects.is_empty() {
        scores[7] = corrects.iter().map(|&i| difficulty[i]).sum::<f64>() / corrects.len() as f64;
    }
    let lo_bound = basal.map_or(-1i64, |b| b as i64);
    let mid: Vec<f64> = corrects
        .iter()
        .filter(|&&i| (stratum[i] as i64) > lo_bound && stratum[i] < upper)
        .map(|&i| difficulty[i])
        .collect();
    let mut consistency = f64::NAN;
    if !mid.is_empty() {
        let m = mid.iter().sum::<f64>() / mid.len() as f64;
        scores[8] = m;
        consistency = mid.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / mid.len() as f64;
    }

    Ok(StradaptiveResult {
        administered,
        responses_taken,
        reason,
        ceiling,
        basal,
        hnc,
        next_item,
        scores,
        consistency,
    })
}

// ---------------------------------------------------------------------------
// Pyramidal adaptive testing (Larkin & Weiss, 1974)
// ---------------------------------------------------------------------------
//
// Source status: Larkin & Weiss (1974) READ in full (OCR of ERIC ED096343).
// Secondary works cited inside it (Lord, 1970, 1971a, 1971b; Hansen, 1969;
// Bayroff, 1960; Paterson, 1962) were NOT read; they are cited below only
// "as described by Larkin & Weiss (1974)".
//
// Structure (Larkin & Weiss, 1974, pp. 5-7, Figure 1): items arranged in a
// triangular structure by difficulty; stage s (1-based) holds s items; an
// n-stage pyramid needs n(n+1)/2 items (printed formula, p. 13). The first
// item is of median difficulty. Routing is "up-one/down-one" with "equal
// offset": a correct response leads to the harder of the two stage-(s+1)
// neighbours, an incorrect response to the easier. One item per stage; a
// fixed n items are administered.
//
// DERIVED routing recurrence (from the Figure-1 prose; not printed as an
// equation in the source): with 0-based within-stage position j ordered
// easiest -> hardest, j_1 = 0 and j_{s+1} = j_s + u_s where u_s in {0, 1}
// is the correctness of the stage-s response. The row-major flattened node
// index of (stage s, position j) is s(s-1)/2 + j (previous stages hold
// 1 + 2 + ... + (s-1) = s(s-1)/2 items).
//
// Scoring methods 1-6 (Larkin & Weiss, 1974, pp. 15-16):
//   M1 number-correct score: sum of u_s (integer 0..n).
//   M2 mean difficulty of all items attempted.
//   M3 mean difficulty of correctly answered items. The source does not
//      define the 0-correct case; this implementation returns NaN
//      (documented indeterminate, not an error).
//   M4 difficulty of the final (stage-n) item attempted.
//   M5 "final difficulty score" / hypothetical (n+1)th-item score (Hansen,
//      1969, and Lord, 1971b, as described by Larkin & Weiss, 1974): branch
//      once more on the final response into a hypothetical stage n+1 whose
//      difficulties `b_next` (length n+1) the CALLER supplies; the score is
//      b_next[j_n + u_n]. Larkin & Weiss's own construction of b_next
//      (column means with extrapolated extremes, p. 15) is pool-specific
//      and out of scope. When `b_next` is None, M5 is UNAVAILABLE and NaN
//      is returned (not a computed Method-5 score).
//   M6 Hansen (1969, as described by Larkin & Weiss, 1974) all-item score.
//      The per-stage formulas below are DERIVED from the p. 16 prose (they
//      are not printed as equations): a correct response at position j of
//      stage s scores 2 + 2j + [j < s-1] (2 for the item, 2 per easier
//      item, 1 for the next harder item, 0 beyond); an incorrect response
//      scores [j >= 1] + 2*max(j - 1, 0) (0 for the item and all harder,
//      1 for the next easier, 2 per remaining easier item). VERIFIED
//      against the printed 15-stage score range "0 to 240" (p. 16): the
//      all-correct path scores exactly 240 and the all-incorrect path 0
//      (pinned in tests).
//
// Out of scope (variants described but not used in the source's design):
// unequal offsets ("up-one/down-two", correction for guessing, p. 7),
// shrinking step size (Paterson, 1962, as described), multi-item blocks
// per stage (p. 6), and the study's empirical reliability/validity
// analyses. Within-stage difficulty monotonicity is NOT enforced: the
// source's own pyramid 1 contained mis-ordered items (p. 14).
//
// References (APA 7th):
// Larkin, K. C., & Weiss, D. J. (1974). An empirical investigation of
//     computer-administered pyramidal ability testing (Research Report
//     74-3; ERIC ED096343). University of Minnesota, Psychometric Methods
//     Program. (READ: structure, routing, scoring methods 1-6, Table 1)
// Hansen, D. N. (1969). An investigation of computer-based science testing.
//     (NOT read; all-item score and final node score implemented as
//     described by Larkin & Weiss, 1974)
// Lord, F. M. (1971). A theoretical study of the measurement effectiveness
//     of flexilevel tests. Educational and Psychological Measurement,
//     31(4), 805-813. (NOT read; "final difficulty score" naming as
//     described by Larkin & Weiss, 1974)

/// Result of [`pyramidal_administer`]. `path` holds the flattened row-major
/// node indices attempted (one per stage); `positions` the within-stage
/// 0-based positions (easiest -> hardest). Scores follow Larkin & Weiss
/// (1974) methods 1-6: `mean_b_correct` is NaN when no item was answered
/// correctly, and `final_difficulty` is NaN when `b_next` was not supplied
/// (M5 unavailable).
#[derive(Debug, Clone)]
pub struct PyramidalResult {
    pub path: Vec<usize>,
    pub positions: Vec<usize>,
    pub number_correct: f64,
    pub mean_b_attempted: f64,
    pub mean_b_correct: f64,
    pub final_b: f64,
    pub final_difficulty: f64,
    pub all_item_score: f64,
}

/// Administer an n-stage up-one/down-one pyramidal test over a hypothetical
/// full response vector and compute scoring methods 1-6 (see the module
/// comment above for the verified contract and source status).
///
/// `b` is the row-major flattened difficulty vector (stage 1 first; each
/// stage ordered easiest -> hardest) of length n_stages*(n_stages+1)/2;
/// `u[s]` is the 0/1 response to the stage-(s+1) item on the routed path;
/// `b_next`, when supplied, holds the n_stages+1 hypothetical next-stage
/// difficulties used by method 5.
pub fn pyramidal_administer(
    b: &[f64],
    n_stages: usize,
    u: &[u8],
    b_next: Option<&[f64]>,
) -> Result<PyramidalResult, String> {
    if n_stages == 0 {
        return Err("pyramidal_administer: n_stages must be >= 1".into());
    }
    // Checked triangular size: n(n+1)/2 (Larkin & Weiss, 1974, p. 13). A
    // huge n_stages must yield Err, not a debug panic or release wrap.
    let expected = n_stages
        .checked_add(1)
        .and_then(|np1| n_stages.checked_mul(np1))
        .map(|t| t / 2)
        .ok_or_else(|| {
            format!("pyramidal_administer: n_stages {n_stages} overflows the n(n+1)/2 item count")
        })?;
    if b.len() != expected {
        return Err(format!(
            "pyramidal_administer: b has {} items but an {}-stage pyramid needs n(n+1)/2 = {}",
            b.len(),
            n_stages,
            expected
        ));
    }
    if u.len() != n_stages {
        return Err(format!(
            "pyramidal_administer: u has {} responses but n_stages is {}",
            u.len(),
            n_stages
        ));
    }
    for (i, &r) in u.iter().enumerate() {
        if r > 1 {
            return Err(format!(
                "pyramidal_administer: u[{i}] must be 0 or 1 (got {r})"
            ));
        }
    }
    for (i, &d) in b.iter().enumerate() {
        if !d.is_finite() {
            return Err(format!("pyramidal_administer: b[{i}] must be finite"));
        }
    }
    if let Some(bn) = b_next {
        if bn.len() != n_stages + 1 {
            return Err(format!(
                "pyramidal_administer: b_next has {} items but must have n_stages + 1 = {}",
                bn.len(),
                n_stages + 1
            ));
        }
        for (i, &d) in bn.iter().enumerate() {
            if !d.is_finite() {
                return Err(format!("pyramidal_administer: b_next[{i}] must be finite"));
            }
        }
    }

    // Routing: j_1 = 0; j_{s+1} = j_s + u_s (DERIVED; see module comment).
    // `offset` is maintained incrementally (offset += s), so it stays
    // bounded by the validated b.len() and cannot overflow.
    let mut path = Vec::with_capacity(n_stages);
    let mut positions = Vec::with_capacity(n_stages);
    let mut j = 0usize;
    let mut offset = 0usize; // s(s-1)/2 for the current 1-based stage s
    let mut all_item = 0i64;
    let mut n_correct = 0usize;
    let mut sum_attempted = 0.0f64;
    let mut sum_correct = 0.0f64;
    for s in 1..=n_stages {
        positions.push(j);
        path.push(offset + j);
        let bi = b[offset + j];
        sum_attempted += bi;
        let us = u[s - 1];
        if us == 1 {
            n_correct += 1;
            sum_correct += bi;
            // M6 correct: 2 + 2j + [j < s-1] (DERIVED from p. 16 prose).
            all_item += 2 + 2 * j as i64 + i64::from(j < s - 1);
        } else {
            // M6 incorrect: [j >= 1] + 2*max(j-1, 0).
            all_item += i64::from(j >= 1) + 2 * (j.saturating_sub(1)) as i64;
        }
        offset += s;
        if s < n_stages {
            j += us as usize;
        }
    }
    let j_final = positions[n_stages - 1] + u[n_stages - 1] as usize;
    let final_difficulty = match b_next {
        Some(bn) => bn[j_final],
        None => f64::NAN,
    };
    let mean_b_correct = if n_correct > 0 {
        sum_correct / n_correct as f64
    } else {
        f64::NAN
    };

    Ok(PyramidalResult {
        final_b: b[*path.last().unwrap()],
        number_correct: n_correct as f64,
        mean_b_attempted: sum_attempted / n_stages as f64,
        mean_b_correct,
        final_difficulty,
        all_item_score: all_item as f64,
        path,
        positions,
    })
}

// ---------------------------------------------------------------------------
// Two-stage adaptive testing (Betz & Weiss, 1973, 1974)
// ---------------------------------------------------------------------------
//
// Source status: Betz & Weiss (1974, Research Report 74-4; ERIC ED103466)
// and Betz & Weiss (1973, Research Report 73-4; ERIC ED084302) READ in full
// (OCR). Lord (1971), the origin of the scoring method, was NOT read; the
// formulas below are implemented exactly as restated by Betz & Weiss.
//
// Procedure: a routing test of m1 items is administered and scored
// number-correct; an initial ability estimate routes the examinee to the
// measurement test whose mean item difficulty is closest to that estimate
// (Betz & Weiss, 1974, p. 17 and Appendix B); the measurement test of m2
// items is then administered, a second estimate is computed, and the two
// estimates are combined.
//
// Subtest ability estimate (Betz & Weiss, 1974, Equation 2, modifying
// Lord's Equation 1 by using the subtest MEAN discrimination a-bar and MEAN
// difficulty b-bar):
//
//   theta-hat = Phi^-1( ((x'/m) - c) / (1 - c) ) / a-bar + b-bar
//
// with the printed truncation (p. ~18): x' = m - 1/2 when x = m (perfect
// score) and x' = c*m + 1/2 when x <= c*m (chance score or below), else
// x' = x. The reconstruction of the OCR-garbled formula was VERIFIED
// against the printed Appendix B routing table (m = 10, a-bar = .70,
// b-bar = -.23: x = 6 -> -.23 exactly since Phi^-1(1/2) = 0; all 11 rows
// reproduce within +-0.05 of the printed 2-dp values, pinned in tests).
//
// DERIVED validity condition: both subtests must satisfy m*(1 - c) > 1.
// This is a CONSERVATIVE condition guaranteeing the lower truncation
// c*m + 1/2 stays strictly BELOW the upper truncation m - 1/2 (distinct
// endpoints); mere containment of x' in (c*m, m) would only need
// m*(1 - c) > 1/2. The source assumes m = 10/30, c = .2 and states no
// such condition.
//
// Routing: assigned = argmin_k |b_meas[k] - theta1| (Betz & Weiss, 1974,
// p. 17). Ties are broken toward the LOWEST index -- a DERIVED
// deterministic convention; neither source states a tie-break.
//
// Composite (Betz & Weiss, 1974, Equation 3; rationale in Betz & Weiss,
// 1973, p. 15: each subtest estimate "weighted according to the number of
// items on which it was based" -- chosen over Lord's variance weights,
// which produced non-monotonicity):
//
//   theta-hat = (m1*theta1 + m2*theta2) / (m1 + m2)
//
// The papers print only the m1 = 10, m2 = 30 case; the item-count
// generalization above is DERIVED from the quoted weighting rationale and
// is restricted to exactly two subtests.
//
// A single chance level c is shared by both subtests, faithful to the
// sources (all items five-alternative, c = .2). Mixed formats, item
// administration/response simulation (the source's SIMTEST), Lord's
// variance weighting, and the studies' reliability/information analyses
// are out of scope.
//
// References (APA 7th):
// Betz, N. E., & Weiss, D. J. (1973). An empirical study of
//     computer-administered two-stage ability testing (Research Report
//     73-4; ERIC ED084302). University of Minnesota, Psychometric Methods
//     Program. (READ: scoring formulas, weighting rationale)
// Betz, N. E., & Weiss, D. J. (1974). Simulation studies of two-stage
//     ability testing (Research Report 74-4; ERIC ED103466). University of
//     Minnesota, Psychometric Methods Program. (READ: Equations 2-3,
//     routing rule, truncation, Appendix B routing table)
// Lord, F. M. (1971). The self-scoring flexilevel test / theoretical
//     two-stage studies. (NOT read; scoring method implemented as restated
//     by Betz & Weiss, 1973, 1974)

/// Result of [`two_stage_score`]: the routing-test estimate `theta1`, the
/// 0-based `assigned` measurement-test index, the measurement-test estimate
/// `theta2`, and the item-count-weighted `composite` (Betz & Weiss, 1974,
/// Equations 2-3).
#[derive(Debug, Clone)]
pub struct TwoStageResult {
    pub theta1: f64,
    pub assigned: usize,
    pub theta2: f64,
    pub composite: f64,
}

/// Validate one subtest's scalars and return the truncated-score ability
/// estimate theta-hat (Betz & Weiss, 1974, Equation 2). `label` names the
/// subtest in error messages.
fn two_stage_subtest_theta(
    label: &str,
    x: usize,
    m: usize,
    a_bar: f64,
    b_bar: f64,
    c: f64,
) -> Result<f64, String> {
    if m == 0 {
        return Err(format!("two_stage: {label} length m must be >= 1"));
    }
    if x > m {
        return Err(format!(
            "two_stage: {label} number correct {x} exceeds its length {m}"
        ));
    }
    if !a_bar.is_finite() || a_bar <= 0.0 {
        return Err(format!(
            "two_stage: {label} mean discrimination must be finite and > 0 (got {a_bar})"
        ));
    }
    if !b_bar.is_finite() {
        return Err(format!("two_stage: {label} mean difficulty must be finite"));
    }
    let mf = m as f64;
    if mf * (1.0 - c) <= 1.0 {
        return Err(format!(
            "two_stage: {label} needs m*(1-c) > 1 for distinct truncation endpoints \
             (got m = {m}, c = {c})"
        ));
    }
    // Truncation (Betz & Weiss, 1974): perfect -> m - 1/2; at or below
    // chance -> c*m + 1/2; otherwise the observed number correct.
    let x_adj = if x == m {
        mf - 0.5
    } else if x as f64 <= c * mf {
        c * mf + 0.5
    } else {
        x as f64
    };
    let p = ((x_adj / mf) - c) / (1.0 - c);
    debug_assert!(p > 0.0 && p < 1.0);
    Ok(crate::nodes::inv_normal_cdf(p) / a_bar + b_bar)
}

/// Route from a routing-test result to a measurement test: returns
/// `(theta1, assigned)` where `assigned = argmin_k |b_meas[k] - theta1|`
/// (Betz & Weiss, 1974, p. 17; lowest index on ties, a DERIVED convention).
/// Callers administer measurement test `assigned` and then call
/// [`two_stage_score`] with the same inputs plus the observed `x2`.
pub fn two_stage_route(
    x1: usize,
    m1: usize,
    a1: f64,
    b1: f64,
    b_meas: &[f64],
    c: f64,
) -> Result<(f64, usize), String> {
    if !c.is_finite() || !(0.0..1.0).contains(&c) {
        return Err(format!(
            "two_stage: c must be finite and in [0, 1) (got {c})"
        ));
    }
    if b_meas.is_empty() {
        return Err("two_stage: at least one measurement test is required".into());
    }
    for (k, &bk) in b_meas.iter().enumerate() {
        if !bk.is_finite() {
            return Err(format!("two_stage: b_meas[{k}] must be finite"));
        }
    }
    let theta1 = two_stage_subtest_theta("routing test", x1, m1, a1, b1, c)?;
    let mut assigned = 0usize;
    let mut best = (b_meas[0] - theta1).abs();
    for (k, &bk) in b_meas.iter().enumerate().skip(1) {
        let d = (bk - theta1).abs();
        if d < best {
            assigned = k;
            best = d;
        }
    }
    Ok((theta1, assigned))
}

/// Score a completed two-stage test (Betz & Weiss, 1974, Equations 2-3).
///
/// `administered` is the 0-based index of the measurement test the caller
/// actually gave; it is re-derived from `theta1` internally and a mismatch
/// is an error, so `x2` can never be silently scored against the wrong
/// measurement test's parameters.
#[allow(clippy::too_many_arguments)]
pub fn two_stage_score(
    x1: usize,
    m1: usize,
    a1: f64,
    b1: f64,
    x2: usize,
    m2: usize,
    administered: usize,
    a_meas: &[f64],
    b_meas: &[f64],
    c: f64,
) -> Result<TwoStageResult, String> {
    if a_meas.len() != b_meas.len() {
        return Err(format!(
            "two_stage: a_meas has {} entries but b_meas has {}",
            a_meas.len(),
            b_meas.len()
        ));
    }
    let (theta1, assigned) = two_stage_route(x1, m1, a1, b1, b_meas, c)?;
    if administered >= b_meas.len() {
        return Err(format!(
            "two_stage: administered index {administered} out of range for {} measurement tests",
            b_meas.len()
        ));
    }
    if administered != assigned {
        return Err(format!(
            "two_stage: routing assigns measurement test {assigned} but test {administered} \
             was administered; scoring x2 against the wrong test's parameters is refused"
        ));
    }
    let theta2 = two_stage_subtest_theta(
        "measurement test",
        x2,
        m2,
        a_meas[assigned],
        b_meas[assigned],
        c,
    )?;
    let (m1f, m2f) = (m1 as f64, m2 as f64);
    let composite = (m1f * theta1 + m2f * theta2) / (m1f + m2f);
    Ok(TwoStageResult {
        theta1,
        assigned,
        theta2,
        composite,
    })
}
