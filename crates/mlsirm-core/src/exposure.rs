//! Sympson-Hetter item exposure control for computerized adaptive testing.
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
//! `r_max` below that bound is rejected as infeasible.
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
fn eap_interim(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    responses: &[(usize, f64)],
    grid: &[f64],
    log_prior: &[f64],
) -> f64 {
    let mut log_post: Vec<f64> = log_prior.to_vec();
    for (q, &t) in grid.iter().enumerate() {
        for &(i, y) in responses {
            let p = p3pl(t, a[i], b[i], c[i]).clamp(1e-12, 1.0 - 1e-12);
            log_post[q] += if y > 0.5 { p.ln() } else { (1.0 - p).ln() };
        }
    }
    let m = log_post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut num = 0.0;
    let mut den = 0.0;
    for (q, &t) in grid.iter().enumerate() {
        let w = (log_post[q] - m).exp();
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
                theta_hat = eap_interim(a, b, c, &responses, &grid, &log_prior);
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
