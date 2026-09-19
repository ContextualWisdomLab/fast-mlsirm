//! Single-group full-information polytomous two-tier graded response model
//! (Cai, 2010) with dimension reduction over the specific tier (stage 4 of
//! #1912), generalizing the stage-1 bifactor GRM
//! (`crate::bifactor_grm`; Gibbons et al., 2007).
//!
//! # Model
//!
//! Each item `i` has `K = n_cat` ORDERED categories, UNCONSTRAINED primary
//! slopes `a_ip` on a caller-supplied confirmatory subset of the `P`
//! correlated primary dimensions, an UNCONSTRAINED specific slope `a_iS` on
//! at most one orthogonal specific factor, and `K - 1` STRICTLY DECREASING
//! boundary intercepts `d_ik`:
//!
//! ```text
//! P(Y_ij >= k | theta_P, theta_S(i)) = logistic(sum_p a_ip * theta_jp + a_iS * theta_S(i) + d_ik),
//!     k = 1..K - 1,
//! ```
//!
//! with `P(Y >= 0) = 1`, `P(Y >= K) = 0`, and category probabilities from
//! adjacent differences (Cai et al., 2011, eq. 7). The cumulative-logit
//! graded form is Cai, Yang, & Hansen (2011, eq. 6, "A Model for Graded
//! Response" section: `P(y >= k|theta_0, theta_s) = 1/(1 + exp{-[d_k + a_0
//! theta_0 + a_s theta_s]})`) with category probabilities as adjacent
//! differences (Cai et al., 2011, eq. 7); the multi-primary sum replaces
//! their single general term, exactly as the two-tier structure requires
//! (see the covariance paragraph below). The link is logistic rather than
//! the normal ogive of Gibbons et al. (2007, eq. 9) — the same
//! implementation choice as stage 1, matching the `mirt` graded comparison
//! in this repository's fixture.
//!
//! The latent covariance is the two-tier block structure
//! `Sigma = [[G, 0], [0, diag(S)]]` (Chalmers, 2026, mirt `bfactor`
//! documentation, "Details" section: "the secondary latent traits are
//! assumed to be orthogonal to all traits and have a fixed variance of 1,
//! while the primary traits can be organized to vary and covary with other
//! primary traits in the model"): primaries `theta_P ~ MVN(0, Phi)` with
//! `Phi` a correlation matrix (unit diagonal, free off-diagonals — the
//! single-group identification; means are zero), specifics `theta_S ~ N(0,
//! 1)` orthogonal to everything. The bifactor model is the special case of
//! one primary dimension (Chalmers, 2026, mirt `bfactor` documentation:
//! "The bifactor model is a special case of the two-tier model when G above
//! is a 1x1 matrix"); at `P = 1` this fitter evaluates the same finite sums
//! as stage 1 (see the
//! reduction test; note the `>= 2 loading items per primary` validation rule
//! below means degenerate single-loader inputs valid in stage 1 are rejected
//! here — reduction equivalence holds for full-pattern inputs). The two-tier model itself —
//! correlated primaries plus orthogonal specifics, subsuming the bifactor
//! and testlet models — is Cai (2010) (abstract; see the source-access note
//! below).
//!
//! # Source-access note (why Cai 2010 has no equation locator here)
//!
//! The Zotero record for Cai (2010) (key `GT3NQ8K8`) holds the abstract,
//! DOI (`10.1007/s11336-010-9178-0`), and bibliographic data — its abstract
//! confirms the model claims used here (the framework "subsumes standard
//! multidimensional IRT models, bifactor IRT models, and testlet response
//! theory models as special cases", "reduction in the dimensionality of the
//! latent variable space", "an EM algorithm for full-information maximum
//! marginal likelihood estimation") — but the attached file is the Springer
//! article landing page, NOT the full text, and the full text could not be
//! obtained in-run (paywalled at the publisher; the institutional proxy
//! serves the article page without entitlement, Springer WAYF rejects the
//! proxy redirect host, and no author manuscript was found). So no Cai
//! (2010) equation or page number is cited: every locator below names a
//! source whose full text was actually read — the local Gibbons et al.
//! (2007) PDF (eq. 9, 11-12, 15), the open-access Cai, Yang, & Hansen
//! (2011) full text (eq. 6-7, 10-11, "Maximum Marginal Likelihood
//! Estimation" section), and the installed mirt 1.46.1 `bfactor` help topic
//! (two-tier covariance, `ncol(G) + 1` integration). The `mirt::bfactor`
//! two-tier oracle, whose own implementation follows Cai (2010), validates
//! the same MLE empirically.
//!
//! # Estimation: Bock-Aitkin EM with reduction over the specific tier
//!
//! The person marginal integrates the `P` primary dimensions on a product
//! grid and each specific factor within its item block at fixed primary
//! nodes, so numerical integration needs only `P + 1` dimensions (Chalmers,
//! 2026, mirt `bfactor` documentation: "Evaluation of the numerical
//! integrals for the two-tier model requires only ncol(G) + 1 dimensions
//! for integration since the S second order (or 'specific') factors require
//! only 1 integration grid due to the dimension reduction technique"). This
//! is the two-tier instance of the Gibbons-Hedeker reduction: because item
//! `i` depends only on the primaries and its own `theta_S(i)`, the person
//! marginal factors per primary node (Gibbons et al., 2007, eq. 15,
//! "Marginal Maximum Likelihood Estimation" section; Cai, Yang, & Hansen,
//! 2011, p. 221, extend "Gibbons and Hedeker's (1992) bifactor dimension
//! reduction method" and likewise estimate with "the Bock and Aitkin (1981)
//! EM algorithm", "Maximum Marginal Likelihood Estimation" section):
//!
//! ```text
//! L_p = sum_g W_g(Phi) * G_pg * prod_s I_psg,
//! I_psg = sum_h v_h * prod_{i in s} P(Y_pi | z_g, h),
//! ```
//!
//! where `G_pg` is the specific-free item likelihood at primary node `g`.
//! The E-step cost is `O(Q_P^P * sum_s Q_S * |block_s|)` per person instead
//! of `O(Q_P^P * Q_S^S * n_items)`.
//!
//! The primary grid is FIXED independent Gauss-Hermite nodes; the primary
//! correlation enters through density-ratio reweighting,
//! `W_g(Phi) = w_g^0 * N(z_g; 0, Phi) / N(z_g; 0, Phi = I)`, which is the
//! change-of-density identity
//! `log W_g = log w_g^0 - [log|Phi| + z_g'(Phi^{-1} - I)z_g]/2`
//! (implementation choice: the grid never moves, so the fixed-grid EM
//! monotonicity contract of stage 1 is preserved exactly; at `Phi = I` the
//! weights are bitwise the plain Gauss-Hermite weights).
//!
//! # Memory (exact blocked product-grid evaluation; #1992)
//!
//! Category log-probs and M-step node coordinates are evaluated on the fly
//! over the full primary product Gauss–Hermite grid (Golub & Welsch, 1969;
//! node counts remain caller-controlled with no silent cap — #1929;
//! Lesaffre & Spiessens, 2001, warn that low `Q` can bias results). The
//! specific-tier scratch is `O(n_specific * q_specific)` per active primary
//! node rather than `O(n_specific * n_grid * q_specific)`. Finite sums are
//! associative, so the numerical value matches a materialised-table path up
//! to ordinary floating-point roundoff.//!
//! The M-step updates each item by the per-item finite-difference-Hessian
//! Newton of stage 1 (ridge = Hessian conditioning only, NOT a prior;
//! backtracking line search REJECTS non-finite objectives, which is exactly
//! how the ordered-threshold constraint is maintained WITHOUT an explicit
//! reparametrization — the crate's shared ascent convention from `grm.rs`),
//! and updates the primary correlations by Newton on the expected
//! complete-data normal log-likelihood
//! `Q(Phi) = -(N/2)[log|Phi| + tr(Phi^{-1} Sbar)]` over Fisher-`z`
//! transformed correlations (`rho = tanh(z)`, so `|rho| < 1` by
//! construction; non-positive-definite candidates are rejected by the
//! backtracking search, which shrinks toward the positive-definite current
//! iterate). `Sbar` is the mean posterior primary second moment from the
//! E-step; the unconstrained maximizer of `Q` is `Phi = Sbar` (direct
//! calculus: `dQ/dPhi = 0`), and the `z`-space Newton with the crate's
//! shared ascent machinery finds the unit-diagonal-constrained maximizer.
//! Maximizing the expected complete-data log-likelihood in the M-step is
//! the Bock-Aitkin EM principle as implemented by Cai, Yang, & Hansen
//! (2011, "Maximum Marginal Likelihood Estimation" section); the
//! `z`-Newton itself is an implementation choice (same optimizer convention
//! as `grm.rs`).
//!
//! # Identification and reflection
//!
//! Unit trait variances fix the slope scale on every dimension (primaries
//! via the unit-diagonal `Phi`, specifics via `N(0, 1)` — the two-tier
//! covariance of Chalmers, 2026, mirt `bfactor` documentation); ordered
//! thresholds fix the category direction (a direct consequence of the
//! cumulative definition, Cai et al., 2011, eq. 6-7). The primary loading pattern is caller-supplied confirmatory structure (fixed zeros
//! are never estimated — rotation with correlated primaries is the caller's
//! identification responsibility, standard confirmatory practice; Cai,
//! 2010, is a confirmatory model). Validation additionally requires at
//! least two loading items per primary and per specific factor — an
//! implementation choice, not a paper prescription: smaller blocks leave
//! the primary correlation and the primary/specific split weakly identified
//! (mirrors the stage-1 specific-block rule). The per-dimension reflection
//! `(a_.d, theta_d) -> (-a_.d, -theta_d)` (jointly with the `Phi` row/column
//! sign for primary dimensions) leaves every category probability INVARIANT,
//! so it is CANONICALIZED with the same deterministic rule the crate
//! already uses (`poly::canonicalize_slope_reflection`, `grm.rs`:
//! dimension `d` is flipped so its largest-magnitude slope is positive —
//! each primary over its loading items, each specific within its item
//! block — negating that dimension's slopes AND the reported primary EAP
//! column AND the `Phi` row/column signs for primary flips, but NOT the
//! thresholds.
//!
//! # Caller-owned numerics (no hidden clamps, no magic caps)
//!
//! Quadrature densities (`q_primary`, `q_specific`), `max_iter`, `tol`,
//! `n_starts`, and `seed` are CALLER ARGUMENTS; any out-of-range value is a
//! loud `Err`, never a silent clamp. Upper bounds exist only where a real
//! constraint exists: the quadrature counts must resolve a Gauss-Hermite
//! rule (`quadrature::require_gh_rule`, any `n >= 1`, no table cap per
//! #1929), and working-set sizes that would overflow `usize` are rejected
//! by checked arithmetic. Everything else is
//! lower-bounded only (`n_primary >= 1`, `n_specific >= 1`, `n_cat >= 2`,
//! `max_iter >= 1`, `n_starts >= 1`, `newton_iter >= 1`, finite positive
//! `tol`/`ridge`). `seed` drives ONLY the random-start jitter
//! (Gauss-Hermite quadrature is deterministic), and start `t` derives
//! deterministically from `seed ^ f(t)`, so a rerun with the same
//! `(seed, n_starts, ...)` bit-reproduces the fit (#1912 reproducibility
//! requirement).
//!
//! # Failure reporting
//!
//! Every declared category must be observed for every item (with no
//! observations in a category the adjacent boundary intercepts are pinned
//! only to each other by the category-difference definition, Cai et al.,
//! 2011, eq. 7, so the strictly-ordered pair is unidentified): the fitter
//! returns `Err` naming the item and category instead of imputing.
//! Missing cells are dropped under MAR (Cai et al., 2011, eq. 10: a missing
//! observation contributes a factor of 1, i.e. nothing, to the conditional
//! density).
//! Non-convergence at `max_iter` is reported via `converged == false` with
//! `termination_reason == "max_iter_reached"` — never filled with a
//! substitute (#1912 acceptance criterion 3).
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (abstract + metadata read via
//! the Zotero record; full text not accessible — see the source-access note)
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350 (full text read: eq. 6-7, bifactor
//! dimension-reduction discussion p. 221)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485 (full text read: eq. 9, 11-12, 15)
//!
//! Chalmers, R. P. (2026). mirt: Multidimensional item response theory
//! (Version 1.46.1) [R package].
//! https://cran.r-project.org/package=mirt (`bfactor` help topic read:
//! two-tier covariance, `ncol(G) + 1` integration, bifactor special case)

//! Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
//! *Mathematics of Computation, 23*(106), 221-230.
//! https://doi.org/10.1090/S0025-5718-69-99647-1 (full text read via open copy)
//!
//! Lesaffre, E., & Spiessens, B. (2001). On the effect of the number of
//! quadrature points in a logistic random-effects model: an example.
//! *Journal of the Royal Statistical Society Series C: Applied Statistics,
//! 50*(3), 325-335. https://doi.org/10.1111/1467-9876.00237 (full text read:
//! high `Q` often required; do not invent a low default)

use crate::poly::{grm_logprobs, grm_node_gradient, solve_small};

// NOTE (stage-4 design): this module imposes no magic size caps. Upper
// bounds without a documented origin are rejected in favor of correctness
// checks: lower bounds (empty or degenerate problems), the
// embedded-quadrature rule set that actually exists, finiteness/positivity
// of real-valued controls, the correlation-matrix contract on `phi`
// (finite, symmetric, unit diagonal, positive-definite), and checked
// arithmetic that turns size overflow into `Err` instead of a panic.

/// Configuration for [`fit_two_tier_grm`]. Every field is caller-owned and
/// range-validated; nothing is clamped.
///
/// The quadrature densities (`q_primary`, `q_specific`) are REQUIRED caller
/// arguments with no `Default` (Project rule, issue #1929): node counts
/// govern numerical precision and no accuracy target is on file to source a
/// default against. Study settings use >= 121 nodes per dimension (chosen by
/// precision convergence, e.g. 121 vs 241 agreement); any `n >= 1` that
/// `quadrature::require_gh_rule` resolves is accepted (#1929 removed the
/// fixed-table cap), so this module imposes no upper cap of its own.
#[derive(Clone, Copy, Debug)]
pub struct TwoTierGrmConfig {
    /// Gauss-Hermite nodes per primary dimension (any `n >= 1`).
    /// The primary product grid has `q_primary^n_primary` nodes.
    pub q_primary: usize,
    /// Gauss-Hermite nodes per specific factor (any `n >= 1`).
    pub q_specific: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// Number of EM runs from jittered starts; the best loglik wins.
    pub n_starts: usize,
    /// Seeds ONLY the random-start jitter (quadrature is deterministic).
    pub seed: u64,
    /// Inner Newton iterations per M-step (item steps and the Phi step share
    /// this caller-owned budget).
    pub newton_iter: usize,
    /// Newton ridge — Hessian CONDITIONING only, NOT a parameter prior.
    pub ridge: f64,
}

// No `Default` impl: `q_primary`/`q_specific` are quadrature node counts
// with no sourced accuracy target for any particular value (Project rule,
// issue #1929), so every field is a caller-owned, explicit choice.

/// Result of [`fit_two_tier_grm`].
#[derive(Clone, Debug)]
pub struct TwoTierGrmResult {
    /// Primary slopes, row-major `n_items * n_primary` (`0.0` at fixed
    /// pattern positions), reflection-canonicalized per primary dimension.
    pub a_primary: Vec<f64>,
    /// Specific slopes `a_iS`, length `n_items` (`0.0` for specific-free
    /// items), reflection-canonicalized within each block.
    pub a_specific: Vec<f64>,
    /// Ordered boundary intercepts `d_ik`, row-major `n_items * (n_cat - 1)`.
    pub threshold: Vec<f64>,
    /// Estimated primary correlation matrix, row-major
    /// `n_primary * n_primary` (unit diagonal).
    pub phi: Vec<f64>,
    /// Primary-factor EAPs `E[theta_d | Y_p]`, row-major
    /// `n_persons * n_primary`.
    pub theta_p_eap: Vec<f64>,
    /// Primary-factor marginal posterior SDs, row-major
    /// `n_persons * n_primary`.
    pub theta_p_sd: Vec<f64>,
    /// Observed-data category counts, row-major `n_items * n_cat`.
    pub category_counts: Vec<usize>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// Winning start index in `0..n_starts` (deterministic from `seed`).
    pub best_start: usize,
    /// `sum_i (k_i + has_specific(i) + (n_cat - 1)) + P*(P-1)/2` free
    /// parameters, where `k_i` is item `i`'s free primary-slope count.
    pub n_parameters: usize,
}

/// Validated problem structure shared by the fitter and the public
/// marginal-loglik entry points.
pub(crate) struct Validated {
    pub(crate) n_persons: usize,
    pub(crate) n_items: usize,
    pub(crate) n_primary: usize,
    pub(crate) n_specific: usize,
    pub(crate) n_cat: usize,
    pub(crate) m1: usize,
    /// Validated primary product-grid size (`q_primary^n_primary`).
    pub(crate) grid_size: usize,
    /// Per-item free primary dimensions (pattern positions).
    pub(crate) free_primaries: Vec<Vec<usize>>,
    /// Per-specific item-block member lists.
    pub(crate) blocks: Vec<Vec<usize>>,
    /// Specific-free item indices (primary-only items).
    pub(crate) specific_free: Vec<usize>,
    /// Per-item owning block (`None` = specific-free).
    pub(crate) item_block: Vec<Option<usize>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate(
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &TwoTierGrmConfig,
) -> Result<Validated, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if n_primary < 1 {
        return Err("n_primary must be >= 1".into());
    }
    if n_specific < 1 {
        return Err("n_specific must be >= 1".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    crate::quadrature::require_gh_rule(cfg.q_primary, "q_primary")?;
    crate::quadrature::require_gh_rule(cfg.q_specific, "q_specific")?;
    if cfg.max_iter < 1 {
        return Err("max_iter must be >= 1".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if cfg.n_starts < 1 {
        return Err("n_starts must be >= 1".into());
    }
    if cfg.newton_iter < 1 {
        return Err("newton_iter must be >= 1".into());
    }
    if !cfg.ridge.is_finite() || cfg.ridge <= 0.0 {
        return Err("ridge must be finite and positive".into());
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    let primary_cells = n_items.checked_mul(n_primary).ok_or_else(|| {
        "n_items * n_primary overflows usize (primary_map cannot be shaped)".to_string()
    })?;
    if primary_map.len() != primary_cells {
        return Err("primary_map must have length n_items * n_primary (row-major)".into());
    }
    // Primary product-grid size: checked, never wrapped.
    let grid_size = cfg
        .q_primary
        .checked_pow(u32::try_from(n_primary).map_err(|_| {
            "n_primary exceeds u32::MAX; the primary product grid cannot be shaped".to_string()
        })?)
        .ok_or_else(|| "q_primary^n_primary overflows usize".to_string())?;
    // Dominant working-set sizes (per-item expected-count and log-prob
    // tables over the reduced node grid): overflow here is a loud `Err`,
    // never a wrapped index or a capacity panic deeper in the E-step.
    let nodes_per_item = grid_size
        .checked_mul(cfg.q_specific)
        .ok_or_else(|| "primary grid size * q_specific overflows usize".to_string())?;
    n_items
        .checked_mul(nodes_per_item)
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "n_items * primary-grid * q_specific * n_cat overflows usize".to_string())?;
    if specific_map.len() != n_items {
        return Err("specific_map must have length n_items".into());
    }
    let mut free_primaries: Vec<Vec<usize>> = vec![Vec::new(); n_items];
    for i in 0..n_items {
        for d in 0..n_primary {
            if primary_map[i * n_primary + d] {
                free_primaries[i].push(d);
            }
        }
    }
    for d in 0..n_primary {
        let count = free_primaries.iter().filter(|f| f.contains(&d)).count();
        if count < 2 {
            return Err(format!(
                "primary dimension {d} has {count} loading item(s); at least two loading items \
                 per primary dimension are required (implementation stability choice — fewer \
                 leave the primary correlation weakly identified; see the module docs)"
            ));
        }
    }
    let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); n_specific];
    let mut specific_free = Vec::new();
    let mut item_block = vec![None; n_items];
    for (i, &s) in specific_map.iter().enumerate() {
        if s == -1 {
            specific_free.push(i);
        } else if (0..n_specific as i32).contains(&s) {
            blocks[s as usize].push(i);
            item_block[i] = Some(s as usize);
        } else {
            return Err(format!(
                "specific_map[{i}] must be -1 (specific-free) or in 0..{n_specific}; got {s}"
            ));
        }
    }
    for (s, members) in blocks.iter().enumerate() {
        if members.len() < 2 {
            return Err(format!(
                "specific factor {s} has {} item(s); at least two items per specific factor are \
                 required (implementation stability choice — smaller blocks leave the \
                 primary/specific split weakly identified; see the module docs)",
                members.len()
            ));
        }
    }
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) && y[p * n_items + i] >= n_cat {
                return Err("observed response categories must be < n_cat".into());
            }
        }
    }
    // Unobserved categories leave the adjacent ordered boundary pair
    // pinned only to each other (Cai et al., 2011, eq. 7): fail loudly
    // naming the item and category.
    for i in 0..n_items {
        let mut seen = vec![false; n_cat];
        let mut any = false;
        for p in 0..n_persons {
            if is_obs(p, i) {
                any = true;
                seen[y[p * n_items + i]] = true;
            }
        }
        if !any {
            return Err(format!("item {i} has no observed responses"));
        }
        if let Some(k) = (0..n_cat).find(|&k| !seen[k]) {
            return Err(format!(
                "item {i} category {k} is never observed (unidentified GRM boundary); every \
                 declared category must be observed"
            ));
        }
    }
    Ok(Validated {
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        m1: n_cat - 1,
        grid_size,
        free_primaries,
        blocks,
        specific_free,
        item_block,
    })
}

/// Deterministic SplitMix64 stream (no external RNG dependency so the
/// multi-start jitter bit-reproduces from `seed` on every platform).
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform_open(&mut self) -> f64 {
        // (0, 1) exclusive of the endpoints.
        const DEN: f64 = (1u64 << 53) as f64;
        let bits = self.next_u64() >> 11;
        ((bits as f64) + 0.5) / (DEN + 1.0)
    }

    fn standard_normal(&mut self) -> f64 {
        let u1 = self.uniform_open().clamp(1e-12, 1.0 - 1e-12);
        let u2 = self.uniform_open();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// One item's working parameters.
#[derive(Clone, Debug)]
pub(crate) struct ItemParams {
    /// Length `n_primary`; exact `0.0` at fixed pattern positions.
    pub(crate) a_p: Vec<f64>,
    /// `None` for specific-free items.
    pub(crate) a_s: Option<f64>,
    pub(crate) d: Vec<f64>,
}

/// Proportion-based start (start 0): free primary slopes at `1.0`,
/// specific slopes at `0.8`, boundary intercepts at cumulative-logit base
/// rates (ordered decreasing) — the crate's shared proportion-based init
/// convention (`grm::fit_grm`, implementation choice) restricted to the
/// two-tier linear predictor. Starts `t >= 1` add deterministic `N(0, *)`
/// jitter from `SplitMix64(seed ^ GOLDEN * (t + 1))` (slopes `0.4`, thresholds
/// `0.25`, Fisher-`z` correlations `0.25` — the crate's start-jitter
/// convention) and re-sort `d` decreasing.
fn initial_params(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    seed: u64,
    start: usize,
) -> (Vec<ItemParams>, Vec<f64>) {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    let mut rng = SplitMix64(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(start as u64 + 1)));
    let items = (0..v.n_items)
        .map(|i| {
            let mut freq = vec![1e-3f64; v.n_cat];
            for p in 0..v.n_persons {
                if is_obs(p, i) {
                    freq[y[p * v.n_items + i]] += 1.0;
                }
            }
            let tot: f64 = freq.iter().sum();
            let mut d = vec![0.0f64; v.m1];
            let mut cum = 0.0f64;
            for k in (1..v.n_cat).rev() {
                cum += freq[k] / tot;
                let c = cum.clamp(1e-4, 1.0 - 1e-4);
                d[k - 1] = (c / (1.0 - c)).ln();
            }
            let mut a_p = vec![0.0f64; v.n_primary];
            for &dim in &v.free_primaries[i] {
                a_p[dim] = 1.0;
            }
            let mut a_s = v.item_block[i].map(|_| 0.8f64);
            if start > 0 {
                for &dim in &v.free_primaries[i] {
                    a_p[dim] += 0.4 * rng.standard_normal();
                }
                if let Some(a) = a_s.as_mut() {
                    *a += 0.4 * rng.standard_normal();
                }
                for dk in d.iter_mut() {
                    *dk += 0.25 * rng.standard_normal();
                }
                d.sort_by(|x, y| y.total_cmp(x));
            }
            ItemParams { a_p, a_s, d }
        })
        .collect();
    // Primary correlations in Fisher-z space (start 0: Phi = I).
    let m = v.n_primary * (v.n_primary.saturating_sub(1)) / 2;
    let mut z = vec![0.0f64; m];
    if start > 0 {
        for slot in z.iter_mut() {
            *slot = 0.25 * rng.standard_normal();
        }
        // Deterministic positive-definite guard: halve toward z = 0
        // (Phi = I, positive-definite) until the Cholesky succeeds. The
        // loop terminates (z underflows to exactly 0); the bound below is a
        // real termination bound for a geometric contraction, not a model cap.
        for _ in 0..64 {
            if cholesky_lower(&phi_from_z(&z, v.n_primary), v.n_primary).is_some() {
                break;
            }
            for slot in z.iter_mut() {
                *slot *= 0.5;
            }
        }
    }
    (items, z)
}

pub(crate) fn gh_rule(q: usize) -> Result<(&'static [f64], &'static [f64]), String> {
    crate::quadrature::gh_rule(q).ok_or_else(|| format!("unsupported quadrature count {q}"))
}

/// Lower Cholesky factor (row-major) plus `log|Phi|`; `None` when `phi` is
/// not positive-definite.
pub(crate) fn cholesky_lower(phi: &[f64], p: usize) -> Option<(Vec<f64>, f64)> {
    let mut l = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..=i {
            let mut acc = phi[i * p + j];
            for k in 0..j {
                acc -= l[i * p + k] * l[j * p + k];
            }
            if i == j {
                // NaN-safe positive-definiteness gate (NaN and non-positive
                // pivots both reject).
                if !acc.is_finite() || acc <= 0.0 {
                    return None;
                }
                l[i * p + j] = acc.sqrt();
            } else {
                l[i * p + j] = acc / l[j * p + j];
            }
        }
    }
    let logdet = 2.0 * (0..p).map(|i| l[i * p + i].ln()).sum::<f64>();
    if !logdet.is_finite() {
        return None;
    }
    Some((l, logdet))
}

/// Explicit inverse from a lower Cholesky factor (`inv = L^{-T} L^{-1}`).
pub(crate) fn chol_inverse(l: &[f64], p: usize) -> Vec<f64> {
    // Columns of inv(L), then inv = Y'Y.
    let mut y = vec![0.0f64; p * p];
    for col in 0..p {
        for row in 0..p {
            let mut acc = if row == col { 1.0 } else { 0.0 };
            for k in 0..row {
                acc -= l[row * p + k] * y[k * p + col];
            }
            y[row * p + col] = acc / l[row * p + row];
        }
    }
    let mut inv = vec![0.0f64; p * p];
    for i in 0..p {
        for j in 0..p {
            let mut acc = 0.0f64;
            for k in 0..p {
                acc += y[k * p + i] * y[k * p + j];
            }
            inv[i * p + j] = acc;
        }
    }
    inv
}

/// Correlation matrix (row-major `p * p`, unit diagonal) from Fisher-`z`
/// strict-upper-triangle parameters (`rho = tanh(z)`, so `|rho| < 1` by
/// construction; positive-definiteness is checked by the caller via
/// [`cholesky_lower`]).
pub(crate) fn phi_from_z(z: &[f64], p: usize) -> Vec<f64> {
    let mut phi = vec![0.0f64; p * p];
    for i in 0..p {
        phi[i * p + i] = 1.0;
    }
    let mut t = 0usize;
    for i in 0..p {
        for j in (i + 1)..p {
            let rho = z[t].tanh();
            phi[i * p + j] = rho;
            phi[j * p + i] = rho;
            t += 1;
        }
    }
    phi
}

/// Fisher-`z` strict-upper-triangle from a correlation matrix (`z = atanh(rho)`).
/// Unit-diagonal entries are ignored; off-diagonals must satisfy `|rho| < 1`.
pub(crate) fn z_from_phi(phi: &[f64], p: usize) -> Result<Vec<f64>, String> {
    let m = p * (p.saturating_sub(1)) / 2;
    let mut z = vec![0.0f64; m];
    let mut t = 0usize;
    for i in 0..p {
        for j in (i + 1)..p {
            let rho = phi[i * p + j];
            if !rho.is_finite() || rho.abs() >= 1.0 {
                return Err(format!(
                    "phi[{i},{j}] = {rho} is outside (-1, 1); cannot form Fisher-z"
                ));
            }
            z[t] = rho.atanh();
            t += 1;
        }
    }
    Ok(z)
}

/// Fixed independent-Gauss-Hermite primary product grid: row-major
/// `n_grid * n_primary` coordinates plus base log weights
/// (`sum_d log w_d`), little-endian digit order with dimension 0 fastest
/// (so at `n_primary = 1` the coordinates are exactly the GH node vector in
/// rule order). Fixed-grid Gauss-Hermite quadrature is an implementation
/// choice (the embedded rules in `crate::quadrature`); the node counts are
/// caller arguments with no upper cap in this module.
pub(crate) fn build_primary_grid(tz: &[f64], wz: &[f64], p: usize, n_grid: usize) -> (Vec<f64>, Vec<f64>) {
    let q = tz.len();
    let mut coords = vec![0.0f64; n_grid * p];
    let mut log_w0 = vec![0.0f64; n_grid];
    let log_w: Vec<f64> = wz.iter().map(|w| w.ln()).collect();
    for g in 0..n_grid {
        let mut tail = g;
        let mut acc = 0.0f64;
        for d in 0..p {
            let digit = tail % q;
            tail /= q;
            coords[g * p + d] = tz[digit];
            acc += log_w[digit];
        }
        log_w0[g] = acc;
    }
    (coords, log_w0)
}

/// Density-ratio-reweighted primary log weights at the current `Phi`:
/// `log W_g = log w_g^0 - [log|Phi| + z_g'(Phi^{-1} - I)z_g]/2`. At
/// `Phi = I` this is bitwise `log w_g^0` (the inverse is exactly `I`, the
/// log-determinant exactly `0`). This is an implementation choice — the
/// change-of-density identity for two centred multivariate normal densities
/// — that keeps the EM node set fixed (hence the fixed-grid EM
/// monotonicity contract of Cai et al., 2011, "Maximum Marginal Likelihood
/// Estimation" section, is preserved exactly).
pub(crate) fn reweighted_log_weights(
    log_w0: &[f64],
    coords: &[f64],
    phi_inv: &[f64],
    logdet: f64,
    p: usize,
) -> Vec<f64> {
    let n_grid = log_w0.len();
    let mut out = vec![0.0f64; n_grid];
    for g in 0..n_grid {
        let mut quad = 0.0f64;
        for i in 0..p {
            let zi = coords[g * p + i];
            for j in 0..p {
                let mij = phi_inv[i * p + j] - if i == j { 1.0 } else { 0.0 };
                quad += zi * mij * coords[g * p + j];
            }
        }
        out[g] = log_w0[g] - 0.5 * (logdet + quad);
    }
    out
}

fn log_sum_exp(xs: &[f64]) -> f64 {
    let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mx == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let mut acc = 0.0f64;
    for &x in xs {
        acc += (x - mx).exp();
    }
    mx + acc.ln()
}

/// Primary-linear predictor contribution for item `i` at primary node `g`
/// (Cai et al., 2011, eq. 6, p. 227: the multi-primary sum replaces their
/// single general term).
#[inline]
fn item_primary_base(v: &Validated, par: &ItemParams, coords: &[f64], g: usize, i: usize) -> f64 {
    let p = v.n_primary;
    let mut prim = 0.0f64;
    for &dim in &v.free_primaries[i] {
        prim += par.a_p[dim] * coords[g * p + dim];
    }
    prim
}

/// Category log-prob for item `i` at primary node `g` and specific node `h`
/// (`h` ignored / `t_s = 0` for specific-free items). Evaluates Cai et al.
/// (2011, eq. 6–7, p. 227) on the fly so the E-step never materializes a
/// full `n_grid * q_specific * n_cat` table per item (#1992 memory).
#[inline]
fn item_cat_logprob(
    v: &Validated,
    params: &[ItemParams],
    coords: &[f64],
    ts: &[f64],
    i: usize,
    g: usize,
    h: usize,
    cat: usize,
) -> f64 {
    let par = &params[i];
    let prim = item_primary_base(v, par, coords, g, i);
    let base = match par.a_s {
        Some(a_s) => prim + a_s * ts[h],
        None => prim,
    };
    grm_logprobs(base, &par.d)[cat]
}

/// One reduced E-step sweep (Gibbons et al., 2007, eq. 15: the person
/// marginal factored per primary node): observed-data loglik, expected
/// category counts per item (`counts[i][node][k]`, `node = g * qs + h` for
/// block items, `node = g` for specific-free items), and the summed
/// posterior primary second moment (`s_bar_sum[j * p + k] += sum_p sum_g
/// post_pg z_gj z_gk`, divided by `n_persons` by the caller).
///
/// # Memory (exact blocked product-grid evaluation; #1992)
///
/// The primary product Gauss–Hermite grid (Golub & Welsch, 1969) is still
/// fully summed — node counts remain caller-controlled with no silent cap
/// (#1929; Lesaffre & Spiessens, 2001, warn that low `Q` can bias results).
/// Category log-probs are evaluated on the fly, and the specific-tier
/// scratch `block_acc` is sized `n_specific * q_specific` (one primary node
/// at a time) rather than `n_specific * n_grid * q_specific`. Finite sums
/// are associative, so the numerical value matches the materialised-table
/// path up to ordinary floating-point roundoff order.
///
/// Block partial-pattern collapse (#2003) is landed for bifactor GRM; the
/// same Bock–Aitkin / Gibbons–Hedeker basis applies here, but adapting it to
/// this streaming layout is deferred so the bifactor path can land first.
#[allow(clippy::too_many_arguments)]
pub(crate) fn e_step(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    params: &[ItemParams],
    log_w: &[f64],
    log_ws: &[f64],
    coords: &[f64],
    ts: &[f64],
    n_grid: usize,
    qs: usize,
) -> (f64, Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let p = v.n_primary;
    let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * v.n_items + i]);
    let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
    for i in 0..v.n_items {
        let n_nodes = if v.item_block[i].is_some() {
            n_grid * qs
        } else {
            n_grid
        };
        counts.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
    }
    // Per-person scratch: O(n_grid) for primary marginals + O(S * qs) for
    // the active primary node's specific-tier block (not O(S * n_grid * qs)).
    let mut log_i = vec![0.0f64; v.n_specific * n_grid];
    let mut gen_log = vec![0.0f64; n_grid];
    let mut log_like_g = vec![0.0f64; n_grid];
    let mut post_g = vec![0.0f64; n_grid];
    let mut tmp_h = vec![0.0f64; qs];
    let mut block_acc_g = vec![0.0f64; v.n_specific * qs];
    let mut s_bar_sum = vec![0.0f64; p * p];

    let mut loglik = 0.0f64;
    for pp in 0..v.n_persons {
        // Pass 1: person marginal per primary node (specific-free + block
        // integrals), without storing per-(g,h) tables.
        gen_log.copy_from_slice(log_w);
        for &i in &v.specific_free {
            if !is_obs(pp, i) {
                continue;
            }
            let yc = y[pp * v.n_items + i];
            for g in 0..n_grid {
                gen_log[g] += item_cat_logprob(v, params, coords, ts, i, g, 0, yc);
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(pp, i) {
                            continue;
                        }
                        let yc = y[pp * v.n_items + i];
                        acc += item_cat_logprob(v, params, coords, ts, i, g, h, yc);
                    }
                    tmp_h[h] = acc;
                }
                log_i[s * n_grid + g] = log_sum_exp(&tmp_h);
            }
        }
        for g in 0..n_grid {
            let mut acc = gen_log[g];
            for s in 0..v.n_specific {
                acc += log_i[s * n_grid + g];
            }
            log_like_g[g] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        loglik += log_lp;
        for g in 0..n_grid {
            post_g[g] = (log_like_g[g] - log_lp).exp();
        }
        for g in 0..n_grid {
            let post = post_g[g];
            for (jj, slot) in s_bar_sum.iter_mut().enumerate().take(p * p) {
                let j = jj / p;
                let k = jj % p;
                *slot += post * coords[g * p + j] * coords[g * p + k];
            }
        }
        for &i in &v.specific_free {
            if !is_obs(pp, i) {
                continue;
            }
            let yc = y[pp * v.n_items + i];
            for g in 0..n_grid {
                counts[i][g][yc] += post_g[g];
            }
        }
        // Pass 2: joint (g, h) posteriors for block items — recompute the
        // active primary node's specific-tier block on the fly.
        for (s, members) in v.blocks.iter().enumerate() {
            let any_obs = members.iter().any(|&i| is_obs(pp, i));
            if !any_obs {
                continue;
            }
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(pp, i) {
                            continue;
                        }
                        let yc = y[pp * v.n_items + i];
                        acc += item_cat_logprob(v, params, coords, ts, i, g, h, yc);
                    }
                    block_acc_g[s * qs + h] = acc;
                }
                let mut others = gen_log[g] - log_w[g];
                for s2 in 0..v.n_specific {
                    if s2 != s {
                        others += log_i[s2 * n_grid + g];
                    }
                }
                for h in 0..qs {
                    let log_post = log_w[g] + block_acc_g[s * qs + h] + others - log_lp;
                    let post = log_post.exp();
                    for &i in members {
                        if !is_obs(pp, i) {
                            continue;
                        }
                        let yc = y[pp * v.n_items + i];
                        counts[i][g * qs + h][yc] += post;
                    }
                }
            }
        }
    }
    (loglik, counts, s_bar_sum)
}

/// Negative expected complete-data log-lik and gradient for ONE item — the
/// Bock-Aitkin M-step item ascent (Cai et al., 2011, "Maximum Marginal
/// Likelihood Estimation" section) with the crate's shared
/// finite-difference-Hessian Newton convention (`grm.rs`).
/// `params = [free primary slopes..., (a_S?), d_1..d_{K-1}]` (primary slopes
/// in ascending-dimension order). Node coordinates are derived on the fly
/// from the primary product grid and specific GH nodes (`node = g * qs + h`
/// for block items, `node = g` for specific-free) so the M-step never
/// materializes an `n_grid * qs * n_primary` coordinate tensor (#1992).
#[allow(clippy::too_many_arguments)]
fn item_neg_ll_grad(
    params: &[f64],
    free: &[usize],
    has_specific: bool,
    coords: &[f64],
    ts: &[f64],
    n_primary: usize,
    n_grid: usize,
    qs: usize,
    counts: &[Vec<f64>],
    _n_cat: usize,
) -> (f64, Vec<f64>) {
    let k = free.len();
    let off = k + usize::from(has_specific);
    let beta = &params[off..];
    let mut ll = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];
    for (node, cnt) in counts.iter().enumerate() {
        let (g, h) = if has_specific {
            (node / qs, node % qs)
        } else {
            debug_assert!(node < n_grid);
            (node, 0)
        };
        let mut base = 0.0f64;
        for (t, &dim) in free.iter().enumerate() {
            base += params[t] * coords[g * n_primary + dim];
        }
        if has_specific {
            base += params[k] * ts[h];
        }
        let lp = grm_logprobs(base, beta);
        ll += cnt.iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
        let (g_base, g_thr) = grm_node_gradient(base, beta, cnt);
        for (t, &dim) in free.iter().enumerate() {
            grad[t] += g_base * coords[g * n_primary + dim];
        }
        if has_specific {
            grad[k] += g_base * ts[h];
        }
        for (j, gj) in g_thr.iter().enumerate() {
            grad[off + j] += gj;
        }
    }
    (-ll, grad.iter().map(|g| -g).collect())
}

/// Newton M-step for one item — the Bock-Aitkin M-step item ascent (Cai et
/// al., 2011, "Maximum Marginal Likelihood Estimation" section) with the
/// crate's shared ascent convention (FD Hessian, ridge conditioning,
/// backtracking; non-finite rejection keeps `d` strictly ordered, as in
/// `grm.rs`).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // finite-difference Hessian is inherently indexed (mirrors `grm.rs`)
fn m_step_item(
    mut params: Vec<f64>,
    free: &[usize],
    has_specific: bool,
    coords: &[f64],
    ts: &[f64],
    n_primary: usize,
    n_grid: usize,
    qs: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
    ridge: f64,
    n_newton: usize,
) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g) = item_neg_ll_grad(
            &params,
            free,
            has_specific,
            coords,
            ts,
            n_primary,
            n_grid,
            qs,
            counts,
            n_cat,
        );
        let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = item_neg_ll_grad(
                &pj,
                free,
                has_specific,
                coords,
                ts,
                n_primary,
                n_grid,
                qs,
                counts,
                n_cat,
            );
            for r in 0..np {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        for r in 0..np {
            for c in 0..np {
                hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
            }
            hess[r][r] += ridge;
        }
        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }
        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
            max_step = 2.0;
        }
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = params
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let (candidate_f, _) = item_neg_ll_grad(
                &candidate,
                free,
                has_specific,
                coords,
                ts,
                n_primary,
                n_grid,
                qs,
                counts,
                n_cat,
            );
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                params = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    params
}

/// Negative expected complete-data normal log-likelihood (up to constants)
/// for the primary correlations in Fisher-`z` space:
/// `(N/2)[log|Phi(z)| + tr(Phi(z)^{-1} Sbar)]`; `None` (as infinity) when
/// `Phi(z)` is not positive-definite, which the backtracking search
/// rejects. Maximizing this expected complete-data log-likelihood is the
/// Bock-Aitkin M-step principle (Cai et al., 2011, "Maximum Marginal
/// Likelihood Estimation" section); the Fisher-`z` parametrization is an
/// implementation choice (it confines each correlation to `(-1, 1)`).
pub(crate) fn phi_neg_ll(z: &[f64], p: usize, s_bar: &[f64], n_persons: usize) -> f64 {
    let phi = phi_from_z(z, p);
    let Some((l, logdet)) = cholesky_lower(&phi, p) else {
        return f64::INFINITY;
    };
    let inv = chol_inverse(&l, p);
    let mut tr = 0.0f64;
    for i in 0..p {
        for j in 0..p {
            tr += inv[i * p + j] * s_bar[i * p + j];
        }
    }
    0.5 * n_persons as f64 * (logdet + tr)
}

/// Newton M-step for the primary correlations in Fisher-`z` space — the
/// Bock-Aitkin M-step principle (Cai et al., 2011, "Maximum Marginal
/// Likelihood Estimation" section) with the crate's shared ascent
/// convention (FD Hessian, ridge conditioning, backtracking with
/// positive-definiteness rejection), applied to [`phi_neg_ll`]. Skipped
/// entirely at `n_primary = 1` (no correlations).
#[allow(clippy::needless_range_loop)] // finite-difference Hessian is inherently indexed (mirrors `grm.rs`)
fn m_step_phi(
    mut z: Vec<f64>,
    p: usize,
    s_bar: &[f64],
    n_persons: usize,
    ridge: f64,
    n_newton: usize,
) -> Vec<f64> {
    let m = z.len();
    if m == 0 {
        return z;
    }
    for _ in 0..n_newton {
        let f0 = phi_neg_ll(&z, p, s_bar, n_persons);
        if !f0.is_finite() {
            break;
        }
        // Finite-difference gradient in z-space.
        let h = 1e-5;
        let mut g = vec![0.0f64; m];
        for j in 0..m {
            let mut pj = z.clone();
            pj[j] += h;
            let fp = phi_neg_ll(&pj, p, s_bar, n_persons);
            let mut mj = z.clone();
            mj[j] -= h;
            let fm = phi_neg_ll(&mj, p, s_bar, n_persons);
            if fp.is_finite() && fm.is_finite() {
                g[j] = (fp - fm) / (2.0 * h);
            } else if fp.is_finite() {
                g[j] = (fp - f0) / h;
            } else if fm.is_finite() {
                g[j] = (f0 - fm) / h;
            } else {
                g[j] = f64::NAN;
            }
        }
        let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let mut hess = vec![vec![0.0f64; m]; m];
        for j in 0..m {
            let mut pj = z.clone();
            pj[j] += h;
            let mut gj = vec![0.0f64; m];
            let mut ok = true;
            for t in 0..m {
                let mut pp = pj.clone();
                pp[t] += h;
                let fp = phi_neg_ll(&pp, p, s_bar, n_persons);
                let mut mm = pj.clone();
                mm[t] -= h;
                let fm = phi_neg_ll(&mm, p, s_bar, n_persons);
                if fp.is_finite() && fm.is_finite() {
                    gj[t] = (fp - fm) / (2.0 * h);
                } else {
                    ok = false;
                    break;
                }
            }
            if !ok || !gj.iter().all(|x| x.is_finite()) {
                for row in hess.iter_mut() {
                    for slot in row.iter_mut() {
                        *slot = f64::NAN;
                    }
                }
                break;
            }
            for r in 0..m {
                hess[r][j] = (gj[r] - g[r]) / h;
            }
        }
        if !hess.iter().all(|row| row.iter().all(|x| x.is_finite())) {
            break;
        }
        for r in 0..m {
            for c in 0..m {
                hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
            }
            hess[r][r] += ridge;
        }
        let mut step = solve_small(hess, g.clone());
        let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
        if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
            step = g.clone();
            directional = grad_norm * grad_norm;
        }
        let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
        if max_step > 2.0 {
            for s in &mut step {
                *s *= 2.0 / max_step;
            }
            directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
            max_step = 2.0;
        }
        let mut alpha = 1.0f64;
        let mut accepted = false;
        for _ in 0..25 {
            let candidate: Vec<f64> = z
                .iter()
                .zip(&step)
                .map(|(value, direction)| value - alpha * direction)
                .collect();
            let candidate_f = phi_neg_ll(&candidate, p, s_bar, n_persons);
            if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                z = candidate;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted || alpha * max_step < 1e-9 {
            break;
        }
    }
    z
}

fn checked_em_loglik_change(
    current: f64,
    previous: Option<f64>,
    iteration: usize,
) -> Result<Option<f64>, String> {
    if !current.is_finite() {
        return Err(format!(
            "non-finite observed-data log-likelihood at iteration {iteration}"
        ));
    }
    let Some(previous) = previous else {
        return Ok(None);
    };
    let change = current - previous;
    let monotonicity_tolerance = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    if change < -monotonicity_tolerance {
        return Err(format!(
            "EM observed-data log-likelihood decreased at iteration {iteration}: delta={change:.6e}"
        ));
    }
    Ok(Some(change))
}

struct SingleStartOutcome {
    params: Vec<ItemParams>,
    z_phi: Vec<f64>,
    loglik_trace: Vec<f64>,
    n_iter: usize,
    converged: bool,
    termination_reason: String,
    final_loglik_change: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_single_start(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    cfg: &TwoTierGrmConfig,
    coords: &[f64],
    log_w0: &[f64],
    ts: &[f64],
    log_ws: &[f64],
    n_grid: usize,
    qs: usize,
    start: usize,
) -> Result<SingleStartOutcome, String> {
    let p = v.n_primary;
    let (mut params, mut z_phi) = initial_params(v, y, observed, cfg.seed, start);

    // No pre-allocation from `max_iter`: it is caller-owned and unbounded
    // above, so `with_capacity(max_iter + 1)` could overflow; the trace grows
    // amortized instead.
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;

    loop {
        let phi = phi_from_z(&z_phi, p);
        let (l, logdet) = cholesky_lower(&phi, p)
            .ok_or_else(|| format!("primary correlation became non-PD at iteration {n_iter}"))?;
        let phi_inv = chol_inverse(&l, p);
        let log_w = reweighted_log_weights(log_w0, coords, &phi_inv, logdet, p);
        let (ll, counts, s_bar_sum) =
            e_step(v, y, observed, &params, &log_w, log_ws, coords, ts, n_grid, qs);
        let previous = loglik_trace.last().copied();
        let change = checked_em_loglik_change(ll, previous, n_iter)?;
        loglik_trace.push(ll);
        if let Some(change) = change {
            let prev = previous.expect("change requires a previous log-likelihood");
            final_loglik_change = change;
            if final_loglik_change <= cfg.tol * (1.0 + prev.abs()) {
                converged = true;
                termination_reason = "tolerance_met".to_string();
                break;
            }
        }
        if n_iter == cfg.max_iter {
            break;
        }
        let mut s_bar = s_bar_sum;
        for slot in s_bar.iter_mut() {
            *slot /= v.n_persons as f64;
        }
        z_phi = m_step_phi(z_phi, p, &s_bar, v.n_persons, cfg.ridge, cfg.newton_iter);
        for i in 0..v.n_items {
            let free = &v.free_primaries[i];
            let has_specific = v.item_block[i].is_some();
            let mut packed = Vec::with_capacity(free.len() + usize::from(has_specific) + v.m1);
            for &dim in free {
                packed.push(params[i].a_p[dim]);
            }
            if let Some(a_s) = params[i].a_s {
                packed.push(a_s);
            }
            packed.extend_from_slice(&params[i].d);
            let updated = m_step_item(
                packed,
                free,
                has_specific,
                coords,
                ts,
                p,
                n_grid,
                qs,
                &counts[i],
                v.n_cat,
                cfg.ridge,
                cfg.newton_iter,
            );
            for (t, &dim) in free.iter().enumerate() {
                params[i].a_p[dim] = updated[t];
            }
            if has_specific {
                params[i].a_s = Some(updated[free.len()]);
                params[i].d = updated[free.len() + 1..].to_vec();
            } else {
                params[i].d = updated[free.len()..].to_vec();
            }
        }
        n_iter += 1;
    }
    Ok(SingleStartOutcome {
        params,
        z_phi,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
    })
}

/// Fit the single-group polytomous two-tier GRM by Bock-Aitkin marginal ML
/// (Cai et al., 2011, "Maximum Marginal Likelihood Estimation" section)
/// with dimension reduction over the specific tier (Gibbons et al., 2007,
/// eq. 15; Chalmers, 2026, mirt `bfactor` documentation). `y`/`observed`
/// are row-major `n_persons * n_items` (`y` ordered categories
/// `0..n_cat-1`, missing cells dropped MAR under Cai et al., 2011, eq. 10);
/// `primary_map` is row-major `n_items * n_primary` confirmatory
/// free-slope pattern; `specific_map` is length `n_items` with `-1` for
/// specific-free items and `0..n_specific` otherwise. Runs `n_starts` EM
/// runs and keeps the best loglik. Returns `Err` on malformed input,
/// unobserved categories (unidentified ordered boundary pair under Cai et
/// al., 2011, eq. 7), or total numerical failure; per-start
/// non-convergence is reported through the winning run's flags, never
/// substituted.
///
/// # References (APA 7th ed.)
///
/// Cai, L. (2010). A two-tier full-information item factor analysis model
/// with applications. *Psychometrika, 75*(4), 581-612.
/// https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
/// accessible — see the module source-access note)
///
/// Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
/// item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
/// https://doi.org/10.1037/a0023350 (full text read)
///
/// Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
/// Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
/// Stover, A. (2007). Full-information item bifactor analysis of graded
/// response data. *Applied Psychological Measurement, 31*(1), 4-19.
/// https://doi.org/10.1177/0146621606289485 (full text read)
///
/// Chalmers, R. P. (2026). mirt: Multidimensional item response theory
/// (Version 1.46.1) [R package].
/// https://cran.r-project.org/package=mirt (`bfactor` help topic read)
#[allow(clippy::too_many_arguments)]
pub fn fit_two_tier_grm(
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &TwoTierGrmConfig,
) -> Result<TwoTierGrmResult, String> {
    let v = validate(
        y,
        observed,
        primary_map,
        specific_map,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        cfg,
    )?;
    let p = v.n_primary;
    let (tz, wz) = gh_rule(cfg.q_primary)?;
    let (ts, ws) = gh_rule(cfg.q_specific)?;
    let qs = ts.len();
    let n_grid = v.grid_size;
    let (coords, log_w0) = build_primary_grid(tz, wz, p, n_grid);
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

    // Multi-start EM: deterministic starts, best observed loglik wins.
    // A start that fails numerically is SKIPPED (its error is retained for
    // the all-failed report); surviving starts are never mixed.
    let mut best: Option<SingleStartOutcome> = None;
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_start = 0usize;
    let mut first_error: Option<String> = None;
    let mut n_succeeded = 0usize;
    for start in 0..cfg.n_starts {
        match run_single_start(
            &v, y, observed, cfg, &coords, &log_w0, ts, &log_ws, n_grid, qs, start,
        ) {
            Ok(outcome) => {
                n_succeeded += 1;
                let ll = *outcome
                    .loglik_trace
                    .last()
                    .expect("EM trace is never empty");
                if best.is_none() || ll > best_ll {
                    best_ll = ll;
                    best = Some(outcome);
                    best_start = start;
                }
            }
            Err(e) => {
                if first_error.is_none() {
                    first_error = Some(format!("start {start}: {e}"));
                }
            }
        }
    }
    let outcome = best.ok_or_else(|| {
        format!(
            "all {} EM start(s) failed numerically; first error: {}",
            cfg.n_starts,
            first_error.unwrap_or_else(|| "unknown".into())
        )
    })?;
    let _ = n_succeeded;
    let params = outcome.params;
    let phi = phi_from_z(&outcome.z_phi, p);

    // Final EAP pass for the primary tier at the winning parameters.
    // Streaming (same blocked GH product as the E-step; #1992): no full
    // log-prob tables, and specific-tier scratch is O(S * qs) per primary
    // node rather than O(S * n_grid * qs).
    let (l, logdet) = cholesky_lower(&phi, p)
        .ok_or_else(|| "winning primary correlation is non-PD".to_string())?;
    let phi_inv = chol_inverse(&l, p);
    let log_w = reweighted_log_weights(&log_w0, &coords, &phi_inv, logdet, p);
    let mut theta_p_eap = vec![0.0f64; n_persons * p];
    let mut theta_p_sd = vec![0.0f64; n_persons * p];
    let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * n_items + i]);
    let mut log_i = vec![0.0f64; v.n_specific * n_grid];
    let mut log_like_g = vec![0.0f64; n_grid];
    let mut tmp_h = vec![0.0f64; qs];
    for pp in 0..n_persons {
        let mut gen_log = log_w.clone();
        for &i in &v.specific_free {
            if !is_obs(pp, i) {
                continue;
            }
            let yc = y[pp * n_items + i];
            for g in 0..n_grid {
                gen_log[g] += item_cat_logprob(&v, &params, &coords, ts, i, g, 0, yc);
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(pp, i) {
                            continue;
                        }
                        let yc = y[pp * n_items + i];
                        acc += item_cat_logprob(&v, &params, &coords, ts, i, g, h, yc);
                    }
                    tmp_h[h] = acc;
                }
                log_i[s * n_grid + g] = log_sum_exp(&tmp_h);
            }
        }
        for g in 0..n_grid {
            let mut acc = gen_log[g];
            for s in 0..v.n_specific {
                acc += log_i[s * n_grid + g];
            }
            log_like_g[g] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        for d in 0..p {
            let (mut m1, mut m2) = (0.0f64, 0.0f64);
            for (g, &ll) in log_like_g.iter().enumerate() {
                let post = (ll - log_lp).exp();
                let t = coords[g * p + d];
                m1 += post * t;
                m2 += post * t * t;
            }
            theta_p_eap[pp * p + d] = m1;
            theta_p_sd[pp * p + d] = (m2 - m1 * m1).max(0.0).sqrt();
        }
    }

    // Assemble dense outputs.
    let mut a_primary = vec![0.0f64; n_items * p];
    let mut a_specific = vec![0.0f64; n_items];
    let mut threshold = vec![0.0f64; n_items * v.m1];
    let mut n_parameters = p * (p.saturating_sub(1)) / 2;
    for (i, par) in params.iter().enumerate() {
        for &dim in &v.free_primaries[i] {
            a_primary[i * p + dim] = par.a_p[dim];
            n_parameters += 1;
        }
        if let Some(a_s) = par.a_s {
            a_specific[i] = a_s;
            n_parameters += 1;
        }
        n_parameters += v.m1;
        threshold[i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
    }

    // Per-dimension reflection canonicalization (module docs): each primary
    // over its loading items (flipping slopes, the EAP column, and the Phi
    // row/column signs jointly), each specific within its block;
    // thresholds untouched.
    let mut phi_work = phi;
    for d in 0..p {
        let loaders: Vec<usize> = (0..n_items)
            .filter(|&i| v.free_primaries[i].contains(&d))
            .collect();
        let anchor = loaders
            .iter()
            .max_by(|&&i, &&j| {
                a_primary[i * p + d]
                    .abs()
                    .total_cmp(&a_primary[j * p + d].abs())
            })
            .copied()
            .expect("validated primaries all load at least two items");
        if a_primary[anchor * p + d] < 0.0 {
            for &i in &loaders {
                a_primary[i * p + d] = -a_primary[i * p + d];
            }
            for pp in 0..n_persons {
                theta_p_eap[pp * p + d] = -theta_p_eap[pp * p + d];
            }
            for q in 0..p {
                if q != d {
                    phi_work[d * p + q] = -phi_work[d * p + q];
                    phi_work[q * p + d] = -phi_work[q * p + d];
                }
            }
        }
    }
    for members in v.blocks.iter() {
        let anchor = members
            .iter()
            .max_by(|&&i, &&j| a_specific[i].abs().total_cmp(&a_specific[j].abs()))
            .copied()
            .expect("validated blocks are non-empty");
        if a_specific[anchor] < 0.0 {
            for &i in members {
                a_specific[i] = -a_specific[i];
            }
        }
    }

    let mut category_counts = vec![0usize; n_items * n_cat];
    for pp in 0..n_persons {
        for i in 0..n_items {
            if is_obs(pp, i) {
                category_counts[i * n_cat + y[pp * n_items + i]] += 1;
            }
        }
    }

    Ok(TwoTierGrmResult {
        a_primary,
        a_specific,
        threshold,
        phi: phi_work,
        theta_p_eap,
        theta_p_sd,
        category_counts,
        loglik_trace: outcome.loglik_trace,
        n_iter: outcome.n_iter,
        converged: outcome.converged,
        termination_reason: outcome.termination_reason,
        final_loglik_change: outcome.final_loglik_change,
        best_start,
        n_parameters,
    })
}

/// Observed-data marginal loglik at GIVEN parameters via the reduced
/// integration (Gibbons et al., 2007, eq. 15; Chalmers, 2026, mirt
/// `bfactor` documentation). Shared validator with [`fit_two_tier_grm`];
/// `a_primary` is row-major `n_items * n_primary`, `thresholds` row-major
/// `n_items * (n_cat - 1)`, `phi` row-major `n_primary * n_primary`.
/// `a_primary` must be exactly `0.0` at fixed pattern positions (a non-zero
/// value is a loud `Err` — it would otherwise be silently dropped), and
/// `a_specific` must be exactly `0.0` for specific-free items.
///
/// # References (APA 7th ed.)
///
/// Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
/// Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
/// Stover, A. (2007). Full-information item bifactor analysis of graded
/// response data. *Applied Psychological Measurement, 31*(1), 4-19.
/// https://doi.org/10.1177/0146621606289485 (full text read)
///
/// Chalmers, R. P. (2026). mirt: Multidimensional item response theory
/// (Version 1.46.1) [R package].
/// https://cran.r-project.org/package=mirt (`bfactor` help topic read)
#[allow(clippy::too_many_arguments)]
pub fn two_tier_grm_marginal_loglik(
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    phi: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    q_primary: usize,
    q_specific: usize,
) -> Result<f64, String> {
    // max_iter/tol/n_starts/seed/newton_iter/ridge are irrelevant here: this
    // helper only evaluates loglik at given parameters, it does not fit, so
    // `validate` sees them only for its own field-level bounds checks.
    // (No `..Default()` exists: Project rule, issue #1929.)
    let cfg = TwoTierGrmConfig {
        q_primary,
        q_specific,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
    };
    let v = validate(
        y,
        observed,
        primary_map,
        specific_map,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        &cfg,
    )?;
    check_param_shapes(&v, primary_map, a_primary, a_specific, thresholds, phi)?;
    reduced_loglik(&v, a_primary, a_specific, thresholds, phi, y, observed, cfg)
}

#[allow(clippy::too_many_arguments)]
fn reduced_loglik(
    v: &Validated,
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    phi: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    cfg: TwoTierGrmConfig,
) -> Result<f64, String> {
    let p = v.n_primary;
    let (tz, wz) = gh_rule(cfg.q_primary)?;
    let (ts, ws) = gh_rule(cfg.q_specific)?;
    let qs = ts.len();
    let n_grid = v.grid_size;
    let (coords, log_w0) = build_primary_grid(tz, wz, p, n_grid);
    let (l, logdet) =
        cholesky_lower(phi, p).ok_or_else(|| "phi is not positive-definite".to_string())?;
    let phi_inv = chol_inverse(&l, p);
    let log_w = reweighted_log_weights(&log_w0, &coords, &phi_inv, logdet, p);
    let params = pack_params(v, a_primary, a_specific, thresholds);
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    Ok(e_step(
        v, y, observed, &params, &log_w, &log_ws, &coords, ts, n_grid, qs,
    )
    .0)
}

/// Observed-data marginal loglik at GIVEN parameters via BRUTE-FORCE full
/// product-grid integration over `Q_P^P * Q_S^S` nodes — the unrestricted
/// marginal (Gibbons et al., 2007, eq. 10) evaluated by direct summation.
/// Numerically identical to [`two_tier_grm_marginal_loglik`] up to
/// floating-point reorder noise; kept public as the exactness oracle for
/// the reduction (tiny models only — the grid is capped at 2,000,000
/// nodes).
///
/// # References (APA 7th ed.)
///
/// Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
/// Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
/// Stover, A. (2007). Full-information item bifactor analysis of graded
/// response data. *Applied Psychological Measurement, 31*(1), 4-19.
/// https://doi.org/10.1177/0146621606289485 (full text read)
#[allow(clippy::too_many_arguments)]
pub fn two_tier_grm_marginal_loglik_brute(
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    phi: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    q_primary: usize,
    q_specific: usize,
) -> Result<f64, String> {
    // max_iter/tol/n_starts/seed/newton_iter/ridge are irrelevant here: this
    // helper only evaluates loglik at given parameters, it does not fit, so
    // `validate` sees them only for its own field-level bounds checks.
    // (No `..Default()` exists: Project rule, issue #1929.)
    let cfg = TwoTierGrmConfig {
        q_primary,
        q_specific,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
    };
    let v = validate(
        y,
        observed,
        primary_map,
        specific_map,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        &cfg,
    )?;
    check_param_shapes(&v, primary_map, a_primary, a_specific, thresholds, phi)?;
    let p = v.n_primary;
    let (tz, wz) = gh_rule(q_primary)?;
    let (ts, ws) = gh_rule(q_specific)?;
    let qp = tz.len();
    let qs = ts.len();
    let n_grid_p = qp
        .checked_pow(u32::try_from(n_primary).map_err(|_| "grid too large".to_string())?)
        .ok_or_else(|| "primary product grid overflows usize".to_string())?;
    let n_grid_s = qs
        .checked_pow(u32::try_from(n_specific).map_err(|_| "grid too large".to_string())?)
        .ok_or_else(|| "specific product grid overflows usize".to_string())?;
    let n_full = n_grid_p
        .checked_mul(n_grid_s)
        .ok_or_else(|| "product grid overflows usize".to_string())?;
    if n_full > 2_000_000 {
        return Err(format!(
            "brute-force grid {n_full} nodes exceeds the 2,000,000-node oracle cap"
        ));
    }
    let (coords_p, log_w0) = build_primary_grid(tz, wz, p, n_grid_p);
    let (l, logdet) =
        cholesky_lower(phi, p).ok_or_else(|| "phi is not positive-definite".to_string())?;
    let phi_inv = chol_inverse(&l, p);
    let log_w = reweighted_log_weights(&log_w0, &coords_p, &phi_inv, logdet, p);
    let params = pack_params(&v, a_primary, a_specific, thresholds);
    let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * n_items + i]);
    let log_wss: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    // Specific-grid coordinates: little-endian digits over ts.
    let mut log_node = vec![0.0f64; n_full];
    let mut loglik = 0.0f64;
    for pp in 0..n_persons {
        for (node, slot) in log_node.iter_mut().enumerate() {
            let gp = node / n_grid_s;
            let tail = node % n_grid_s;
            let mut acc = log_w[gp];
            let mut tail_tmp = tail;
            let mut spec_coords = vec![0.0f64; n_specific];
            for coord in spec_coords.iter_mut() {
                let digit = tail_tmp % qs;
                tail_tmp /= qs;
                *coord = ts[digit];
                acc += log_wss[digit];
            }
            for i in 0..n_items {
                if !is_obs(pp, i) {
                    continue;
                }
                let mut base = 0.0f64;
                for &dim in &v.free_primaries[i] {
                    base += params[i].a_p[dim] * coords_p[gp * p + dim];
                }
                if let Some(s) = v.item_block[i] {
                    base += params[i].a_s.unwrap_or(0.0) * spec_coords[s];
                }
                let lp = grm_logprobs(base, &params[i].d);
                acc += lp[y[pp * n_items + i]];
            }
            *slot = acc;
        }
        loglik += log_sum_exp(&log_node);
    }
    Ok(loglik)
}

fn check_param_shapes(
    v: &Validated,
    primary_map: &[bool],
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    phi: &[f64],
) -> Result<(), String> {
    if a_primary.len() != v.n_items * v.n_primary {
        return Err("a_primary must have length n_items * n_primary".into());
    }
    if a_specific.len() != v.n_items {
        return Err("a_specific must have length n_items".into());
    }
    if thresholds.len() != v.n_items * v.m1 {
        return Err("thresholds must have length n_items * (n_cat - 1)".into());
    }
    if phi.len() != v.n_primary * v.n_primary {
        return Err("phi must have length n_primary * n_primary".into());
    }
    if [a_primary, a_specific, thresholds, phi]
        .concat()
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err("parameters must be finite".into());
    }
    for (i, &free) in primary_map.iter().enumerate() {
        if !free && a_primary[i] != 0.0 {
            let item = i / v.n_primary;
            let dim = i % v.n_primary;
            return Err(format!(
                "a_primary[{item}, {dim}] must be exactly 0.0 at fixed pattern positions \
                 (primary_map[{item}, {dim}] == false); got {}",
                a_primary[i]
            ));
        }
    }
    for (i, block) in v.item_block.iter().enumerate() {
        if block.is_none() && a_specific[i] != 0.0 {
            return Err(format!(
                "a_specific[{i}] must be exactly 0.0 for specific-free items \
                 (specific_map[{i}] == -1); got {}",
                a_specific[i]
            ));
        }
    }
    for (i, chunk) in thresholds.chunks_exact(v.m1).enumerate() {
        if chunk.windows(2).any(|w| w[0] <= w[1]) {
            return Err(format!(
                "thresholds of item {i} must be strictly decreasing"
            ));
        }
    }
    let p = v.n_primary;
    for i in 0..p {
        if (phi[i * p + i] - 1.0).abs() > 1e-12 {
            return Err(format!(
                "phi must be a correlation matrix (unit diagonal); phi[{i}, {i}] = {}",
                phi[i * p + i]
            ));
        }
        for j in 0..p {
            if (phi[i * p + j] - phi[j * p + i]).abs() > 1e-12 {
                return Err(format!(
                    "phi must be symmetric; |phi[{i}, {j}] - phi[{j}, {i}]| = {}",
                    (phi[i * p + j] - phi[j * p + i]).abs()
                ));
            }
        }
    }
    if cholesky_lower(phi, p).is_none() {
        return Err("phi must be positive-definite".to_string());
    }
    Ok(())
}

pub(crate) fn pack_params(
    v: &Validated,
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
) -> Vec<ItemParams> {
    let p = v.n_primary;
    (0..v.n_items)
        .map(|i| ItemParams {
            a_p: a_primary[i * p..(i + 1) * p].to_vec(),
            a_s: v.item_block[i].map(|_| a_specific[i]),
            d: thresholds[i * v.m1..(i + 1) * v.m1].to_vec(),
        })
        .collect()
}

#[cfg(test)]
#[path = "../../../tests/unit/two_tier_grm_tests.rs"]
mod tests;
