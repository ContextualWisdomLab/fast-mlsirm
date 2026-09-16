//! Single-group full-information polytomous bifactor graded response model
//! with Gibbons-Hedeker dimension reduction (stage 1 of #1912).
//!
//! # Model
//!
//! Each item `i` has `K = n_cat` ORDERED categories, an UNCONSTRAINED general
//! slope `a_iG` on the real line, an UNCONSTRAINED specific slope `a_iS` on at
//! most one orthogonal specific factor, and `K - 1` STRICTLY DECREASING
//! boundary intercepts `d_ik`:
//!
//! ```text
//! P(Y_ij >= k | theta_G, theta_S(i)) = logistic(a_iG * theta_G + a_iS * theta_S(i) + d_ik),
//!     k = 1..K - 1,
//! ```
//!
//! with `P(Y >= 0) = 1`, `P(Y >= K) = 0`, and category probabilities from
//! adjacent differences (Samejima, 1969). The linear predictor follows the
//! bifactor graded model of Gibbons et al. (2007, eq. 9, "The Bifactor Model
//! for Graded Response Data" section): each item loads the general factor
//! plus at most one specific factor ("only one of the k = 2..s values of
//! a_jk is nonzero in addition to a_j1"), with independent standard-normal
//! factors ("Assuming independence of the theta") — the same `N(0, I)`
//! factor assumption as Gibbons & Hedeker (1992, eq. 1). Two deliberate
//! departures from Gibbons et al. (2007), both implementation choices: the
//! link is logistic rather than their normal ogive (to match the `mirt`
//! graded comparison in this repository's fixture), and the intercepts are
//! written directly as `d_ik` rather than split into `c_j + d_t`.
//! The caller supplies the item-to-specific assignment
//! (`specific_map[i] = -1` for a general-only item, otherwise `0..n_specific`);
//! every item loads the general factor. Slopes are UNCONSTRAINED so that
//! reverse-keyed items are representable, matching the crate's unidimensional
//! (`poly::fit_poly_unidim`) and multidimensional (`grm::fit_grm`) GRM
//! estimators since #1879.
//!
//! # Estimation: Bock-Aitkin EM with Gibbons-Hedeker reduction
//!
//! Full product-grid marginal ML would integrate `Q^(S+1)` latent nodes;
//! Gibbons & Hedeker (1992) show the bifactor restriction "reduces the
//! s-dimensional integral in (4) to a two-dimensional integral", via Stuart's
//! (1958) reduction for variates each related to a single dimension only
//! (their eq. 6: "the s dimensions are unconditionally independent, and the
//! joint probability is the product of s unidimensional probabilities").
//! Because item `i` depends only on `theta_G` and its own `theta_S(i)`, the
//! person marginal factors per general node (Gibbons et al., 2007, eq. 15,
//! "Marginal Maximum Likelihood Estimation" section). Cai, Yang, & Hansen
//! (2011) extend "Gibbons and Hedeker's (1992) bifactor dimension reduction
//! method" (p. 221) and likewise estimate with "the Bock and Aitkin (1981)
//! EM algorithm" ("Maximum Marginal Likelihood Estimation" section):
//!
//! ```text
//! L_p = sum_g w_g * G_pg * prod_s I_psg,
//! I_psg = sum_h v_h * prod_{i in s} P(Y_pi | g, h),
//! ```
//!
//! where `G_pg` is the general-only item likelihood at node `g`. The E-step
//! cost is `O(Q_G * sum_s Q_S * |block_s|)` per person instead of
//! `O(Q_G * Q_S^S * n_items)`. The M-step is a per-item finite-difference-
//! Hessian Newton over `[a_G, a_S?, d_1..d_{K-1}]`, byte-for-byte the ascent
//! of `grm::fit_grm`'s item step (ridge = Hessian conditioning only, NOT a
//! prior; backtracking line search REJECTS non-finite objectives, which is
//! exactly how the ordered-threshold constraint is maintained WITHOUT an
//! explicit reparametrization — see `grm.rs`).
//!
//! # Identification and reflection
//!
//! Unit trait variances fix the slope scale on every dimension; ordered
//! thresholds fix the category direction (Samejima, 1969). The bifactor
//! loading pattern (each item on the general factor plus at most one
//! specific; Gibbons & Hedeker, 1992) fixes rotation. Validation additionally
//! requires at least two items per specific factor — an implementation
//! choice, not a paper prescription: no minimum-block-size theorem was found
//! in the cited sources, and smaller blocks leave the general/specific split
//! weakly identified (seen as a slow EM tail in the stage-1 `mirt`
//! comparison). The per-dimension reflection
//! `(a_.d, theta_d) -> (-a_.d, -theta_d)` leaves every category probability
//! INVARIANT, so it is CANONICALIZED with the same deterministic rule the
//! crate already uses (`poly::canonicalize_slope_reflection`, `grm.rs`):
//! dimension `d` is flipped so its largest-magnitude slope is positive — the
//! general dimension over all items, each specific dimension within its item
//! block — negating that dimension's slopes AND (for the general dimension)
//! the reported `theta_G` EAPs, but NOT the thresholds.
//!
//! # Caller-owned numerics (no hidden clamps, no magic caps)
//!
//! Quadrature densities (`q_general`, `q_specific`), `max_iter`, `tol`,
//! `n_starts`, and `seed` are CALLER ARGUMENTS; any out-of-range value is a
//! loud `Err`, never a silent clamp. Upper bounds exist only where a real
//! constraint exists: the quadrature counts must name an embedded
//! Gauss-Hermite rule (`SUPPORTED_Q`), and working-set sizes that would
//! overflow `usize` are rejected by checked arithmetic. Everything else is
//! lower-bounded only (`n_specific >= 1`, `n_cat >= 2`, `max_iter >= 1`,
//! `n_starts >= 1`, `newton_iter >= 1`, finite positive `tol`/`ridge`).
//! `seed` drives ONLY the random-start jitter (Gauss-Hermite quadrature is
//! deterministic), and start `t` derives deterministically from
//! `seed ^ f(t)`, so a rerun with the same `(seed, n_starts, ...)`
//! bit-reproduces the fit (#1912 reproducibility requirement).
//!
//! # Failure reporting
//!
//! Every declared category must be observed for every item (an unobserved
//! category leaves a boundary intercept unidentified; Samejima, 1969): the
//! fitter returns `Err` naming the item and category instead of imputing.
//! Non-convergence at `max_iter` is reported via `converged == false` with
//! `termination_reason == "max_iter_reached"` — never filled with a
//! substitute (#1912 acceptance criterion 3).
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423-436.
//! https://doi.org/10.1007/BF02295430
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350
//!
//! Samejima, F. (1969). Estimation of latent ability using a response pattern
//! of graded scores. *Psychometrika, 34*(S1), 1-97.
//! https://doi.org/10.1007/BF03372160
//!
//! Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of
//! item parameters: Application of an EM algorithm. *Psychometrika, 46*(4),
//! 443-459. https://doi.org/10.1007/BF02293801

use crate::poly::{grm_logprobs, grm_node_gradient, solve_small};
use crate::quadrature::SUPPORTED_Q;

// NOTE (stage-1 review fix-up): this module imposes no magic size caps. Upper
// bounds without a documented origin (a previous revision capped n_specific,
// n_starts, max_iter, newton_iter, n_cat, and the response-cell count) were
// removed; what remains are the correctness checks: lower bounds (empty or
// degenerate problems), the embedded-quadrature rule set that actually
// exists, finiteness/positivity of real-valued controls, and checked
// arithmetic that turns size overflow into `Err` instead of a panic.

/// Configuration for [`fit_bifactor_grm`]. Every field is caller-owned and
/// range-validated; nothing is clamped.
#[derive(Clone, Copy, Debug)]
pub struct BifactorGrmConfig {
    /// Gauss-Hermite nodes for the general factor (one of `SUPPORTED_Q`).
    pub q_general: usize,
    /// Gauss-Hermite nodes per specific factor (one of `SUPPORTED_Q`).
    pub q_specific: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// Number of EM runs from jittered starts; the best loglik wins.
    pub n_starts: usize,
    /// Seeds ONLY the random-start jitter (quadrature is deterministic).
    pub seed: u64,
    /// Inner Newton iterations per item M-step.
    pub newton_iter: usize,
    /// Newton ridge — Hessian CONDITIONING only, NOT a parameter prior.
    pub ridge: f64,
}

impl Default for BifactorGrmConfig {
    fn default() -> Self {
        Self {
            q_general: 21,
            q_specific: 11,
            max_iter: 500,
            tol: 1e-6,
            n_starts: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
            newton_iter: 10,
            ridge: 1e-8,
        }
    }
}

/// Result of [`fit_bifactor_grm`].
#[derive(Clone, Debug)]
pub struct BifactorGrmResult {
    /// General slopes `a_iG`, length `n_items`, reflection-canonicalized.
    pub a_general: Vec<f64>,
    /// Specific slopes `a_iS`, length `n_items` (`0.0` for general-only
    /// items), reflection-canonicalized within each block.
    pub a_specific: Vec<f64>,
    /// Ordered boundary intercepts `d_ik`, row-major `n_items * (n_cat - 1)`.
    pub threshold: Vec<f64>,
    /// General-factor EAP `E[theta_G | Y_p]`, length `n_persons`.
    pub theta_g_eap: Vec<f64>,
    /// General-factor posterior SD, length `n_persons`.
    pub theta_g_sd: Vec<f64>,
    /// Observed-data category counts, row-major `n_items * n_cat`.
    pub category_counts: Vec<usize>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// Winning start index in `0..n_starts` (deterministic from `seed`).
    pub best_start: usize,
    /// `sum_i (1 + has_specific(i) + (n_cat - 1))` free item parameters.
    pub n_parameters: usize,
}

/// Validated problem structure shared by the fitter and the public
/// marginal-loglik entry points.
struct Validated {
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    m1: usize,
    /// Per-specific item-block member lists.
    blocks: Vec<Vec<usize>>,
    /// General-only item indices.
    general_only: Vec<usize>,
    /// Per-item owning block (`None` = general-only).
    item_block: Vec<Option<usize>>,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &BifactorGrmConfig,
) -> Result<Validated, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if n_specific < 1 {
        return Err("n_specific must be >= 1".into());
    }
    if n_cat < 2 {
        return Err("n_cat must be >= 2".into());
    }
    if !SUPPORTED_Q.contains(&cfg.q_general) {
        return Err(format!(
            "q_general must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_general
        ));
    }
    if !SUPPORTED_Q.contains(&cfg.q_specific) {
        return Err(format!(
            "q_specific must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_specific
        ));
    }
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
    // Dominant working-set size (per-item expected-count and log-prob tables
    // over the reduced node grid): overflow here is a loud `Err`, never a
    // wrapped index or a capacity panic deeper in the E-step.
    let nodes_per_item = cfg
        .q_general
        .checked_mul(cfg.q_specific)
        .ok_or_else(|| "q_general * q_specific overflows usize".to_string())?;
    n_items
        .checked_mul(nodes_per_item)
        .and_then(|v| v.checked_mul(n_cat))
        .ok_or_else(|| "n_items * q_general * q_specific * n_cat overflows usize".to_string())?;
    if y.len() != n_cells {
        return Err("y must have length n_persons * n_items".into());
    }
    if let Some(o) = observed {
        if o.len() != n_cells {
            return Err("observed must have length n_persons * n_items".into());
        }
    }
    if specific_map.len() != n_items {
        return Err("specific_map must have length n_items".into());
    }
    let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); n_specific];
    let mut general_only = Vec::new();
    let mut item_block = vec![None; n_items];
    for (i, &s) in specific_map.iter().enumerate() {
        if s == -1 {
            general_only.push(i);
        } else if (0..n_specific as i32).contains(&s) {
            blocks[s as usize].push(i);
            item_block[i] = Some(s as usize);
        } else {
            return Err(format!(
                "specific_map[{i}] must be -1 (general-only) or in 0..{n_specific}; got {s}"
            ));
        }
    }
    for (s, members) in blocks.iter().enumerate() {
        if members.len() < 2 {
            return Err(format!(
                "specific factor {s} has {} item(s); at least two items per specific factor are \
                 required (implementation stability choice — smaller blocks leave the \
                 general/specific split weakly identified; see the module docs)",
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
    // Unobserved categories leave a boundary intercept unidentified
    // (Samejima, 1969): fail loudly naming the item and category.
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
        n_specific,
        n_cat,
        m1: n_cat - 1,
        blocks,
        general_only,
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
struct ItemParams {
    a_g: f64,
    /// `None` for general-only items.
    a_s: Option<f64>,
    d: Vec<f64>,
}

/// Proportion-based start (start 0): slopes at `a_G = 1.0` / `a_S = 0.8`,
/// boundary intercepts at cumulative-logit base rates (ordered decreasing),
/// exactly the `grm::fit_grm` init restricted to the bifactor linear
/// predictor. Starts `t >= 1` add deterministic `N(0, *)` jitter from
/// `SplitMix64(seed ^ GOLDEN * (t + 1))` and re-sort `d` decreasing.
fn initial_params(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    seed: u64,
    start: usize,
) -> Vec<ItemParams> {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    let mut rng = SplitMix64(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(start as u64 + 1)));
    (0..v.n_items)
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
            let (mut a_g, mut a_s) = (1.0f64, v.item_block[i].map(|_| 0.8f64));
            if start > 0 {
                a_g += 0.4 * rng.standard_normal();
                if let Some(a) = a_s.as_mut() {
                    *a += 0.4 * rng.standard_normal();
                }
                for dk in d.iter_mut() {
                    *dk += 0.25 * rng.standard_normal();
                }
                d.sort_by(|x, y| y.total_cmp(x));
            }
            ItemParams { a_g, a_s, d }
        })
        .collect()
}

fn gh_rule(q: usize) -> Result<(&'static [f64], &'static [f64]), String> {
    crate::quadrature::gh_rule(q).ok_or_else(|| format!("unsupported quadrature count {q}"))
}

/// Per-item category log-prob tables at the current parameters:
/// block items `lp[i][g * qs + h][k]`, general-only `lp[i][g][k]`.
fn fill_logprob_tables(
    v: &Validated,
    params: &[ItemParams],
    tg: &[f64],
    ts: &[f64],
    qg: usize,
    qs: usize,
) -> Vec<Vec<f64>> {
    let mut tables = Vec::with_capacity(v.n_items);
    for par in params.iter() {
        match par.a_s {
            Some(a_s) => {
                let mut lp = vec![0.0f64; qg * qs * v.n_cat];
                for g in 0..qg {
                    for h in 0..qs {
                        let base = par.a_g * tg[g] + a_s * ts[h];
                        let probs = grm_logprobs(base, &par.d);
                        lp[(g * qs + h) * v.n_cat..(g * qs + h + 1) * v.n_cat]
                            .copy_from_slice(&probs);
                    }
                }
                tables.push(lp);
            }
            None => {
                let mut lp = vec![0.0f64; qg * v.n_cat];
                for g in 0..qg {
                    let base = par.a_g * tg[g];
                    let probs = grm_logprobs(base, &par.d);
                    lp[g * v.n_cat..(g + 1) * v.n_cat].copy_from_slice(&probs);
                }
                tables.push(lp);
            }
        }
    }
    tables
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

/// One reduced E-step sweep: observed-data loglik plus expected category
/// counts per item (`counts[i][node][k]`, `node = g * qs + h` for block
/// items, `node = g` for general-only items).
#[allow(clippy::too_many_arguments)]
fn e_step(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    tables: &[Vec<f64>],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
) -> (f64, Vec<Vec<Vec<f64>>>) {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
    for (i, par) in tables.iter().enumerate() {
        let _ = par;
        let n_nodes = if v.item_block[i].is_some() {
            qg * qs
        } else {
            qg
        };
        counts.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
    }
    // Per-person scratch.
    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut gen_log = vec![0.0f64; qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut post_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];

    let mut loglik = 0.0f64;
    for p in 0..v.n_persons {
        // General-only log-likelihood per general node.
        gen_log.copy_from_slice(log_wg);
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            let lp = &tables[i];
            for g in 0..qg {
                gen_log[g] += lp[g * v.n_cat + yc];
            }
        }
        // Block accumulations: sum of item log-probs per (s, g, h).
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        acc += tables[i][(g * qs + h) * v.n_cat + yc];
                    }
                    block_acc[(s * qg + g) * qs + h] = acc;
                }
            }
            for g in 0..qg {
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + g) * qs + h];
                }
                log_i[s * qg + g] = log_sum_exp(&tmp_h);
            }
        }
        for g in 0..qg {
            let mut acc = gen_log[g];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + g];
            }
            log_like_g[g] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        loglik += log_lp;
        for g in 0..qg {
            post_g[g] = (log_like_g[g] - log_lp).exp();
        }
        // General-only expected counts share the marginal general posterior.
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            for g in 0..qg {
                counts[i][g][yc] += post_g[g];
            }
        }
        // Block items: joint (g, h) posterior marginalizing the other blocks.
        for (s, members) in v.blocks.iter().enumerate() {
            let any_obs = members.iter().any(|&i| is_obs(p, i));
            if !any_obs {
                continue;
            }
            for g in 0..qg {
                // Sum of the OTHER blocks' log-integrals at g.
                let mut others = gen_log[g] - log_wg[g];
                for s2 in 0..v.n_specific {
                    if s2 != s {
                        others += log_i[s2 * qg + g];
                    }
                }
                for h in 0..qs {
                    let log_post = log_wg[g] + block_acc[(s * qg + g) * qs + h] + others - log_lp;
                    let post = log_post.exp();
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        counts[i][g * qs + h][yc] += post;
                    }
                }
            }
        }
    }
    (loglik, counts)
}

/// Negative expected complete-data log-lik and gradient for ONE item.
/// `params = [a_G, (a_S?), d_1..d_{K-1}]`; `node_g[node]` / `node_s[node]`
/// hold the latent coordinates (`node_s` ignored for general-only items).
fn item_neg_ll_grad(
    params: &[f64],
    has_specific: bool,
    node_g: &[f64],
    node_s: &[f64],
    counts: &[Vec<f64>],
    _n_cat: usize,
) -> (f64, Vec<f64>) {
    let off = if has_specific { 2 } else { 1 };
    let beta = &params[off..];
    let mut ll = 0.0f64;
    let mut grad = vec![0.0f64; params.len()];
    for (node, cnt) in counts.iter().enumerate() {
        let base = if has_specific {
            params[0] * node_g[node] + params[1] * node_s[node]
        } else {
            params[0] * node_g[node]
        };
        let lp = grm_logprobs(base, beta);
        ll += cnt.iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
        let (g_base, g_thr) = grm_node_gradient(base, beta, cnt);
        grad[0] += g_base * node_g[node];
        if has_specific {
            grad[1] += g_base * node_s[node];
        }
        for (j, gj) in g_thr.iter().enumerate() {
            grad[off + j] += gj;
        }
    }
    (-ll, grad.iter().map(|g| -g).collect())
}

/// Newton M-step for one item — the `grm::fit_grm` ascent restricted to the
/// bifactor linear predictor (FD Hessian, ridge conditioning, backtracking;
/// non-finite rejection keeps `d` strictly ordered).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // finite-difference Hessian is inherently indexed (mirrors `grm.rs`)
fn m_step_item(
    mut params: Vec<f64>,
    has_specific: bool,
    node_g: &[f64],
    node_s: &[f64],
    counts: &[Vec<f64>],
    n_cat: usize,
    ridge: f64,
    n_newton: usize,
) -> Vec<f64> {
    let np = params.len();
    for _ in 0..n_newton {
        let (f0, g) = item_neg_ll_grad(&params, has_specific, node_g, node_s, counts, n_cat);
        let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
            break;
        }
        let h = 1e-5;
        let mut hess = vec![vec![0.0f64; np]; np];
        for j in 0..np {
            let mut pj = params.clone();
            pj[j] += h;
            let (_f2, gj) = item_neg_ll_grad(&pj, has_specific, node_g, node_s, counts, n_cat);
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
            let (candidate_f, _) =
                item_neg_ll_grad(&candidate, has_specific, node_g, node_s, counts, n_cat);
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
    cfg: &BifactorGrmConfig,
    tg: &[f64],
    ts: &[f64],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
    start: usize,
) -> Result<SingleStartOutcome, String> {
    let mut params = initial_params(v, y, observed, cfg.seed, start);
    // Latent coordinates per expected-count node.
    let mut node_g: Vec<Vec<f64>> = Vec::with_capacity(v.n_items);
    let mut node_s: Vec<Vec<f64>> = Vec::with_capacity(v.n_items);
    for i in 0..v.n_items {
        if v.item_block[i].is_some() {
            let mut gg = Vec::with_capacity(qg * qs);
            let mut ss = Vec::with_capacity(qg * qs);
            for &t in tg.iter().take(qg) {
                for &u in ts.iter().take(qs) {
                    gg.push(t);
                    ss.push(u);
                }
            }
            node_g.push(gg);
            node_s.push(ss);
        } else {
            node_g.push(tg.to_vec());
            node_s.push(vec![0.0; qg]);
        }
    }

    // No pre-allocation from `max_iter`: it is caller-owned and unbounded
    // above, so `with_capacity(max_iter + 1)` could overflow; the trace grows
    // amortized instead.
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;

    loop {
        let tables = fill_logprob_tables(v, &params, tg, ts, qg, qs);
        let (ll, counts) = e_step(v, y, observed, &tables, log_wg, log_ws, qg, qs);
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
        for i in 0..v.n_items {
            let has_specific = v.item_block[i].is_some();
            let mut packed = Vec::with_capacity(1 + has_specific as usize + v.m1);
            packed.push(params[i].a_g);
            if let Some(a_s) = params[i].a_s {
                packed.push(a_s);
            }
            packed.extend_from_slice(&params[i].d);
            let updated = m_step_item(
                packed,
                has_specific,
                &node_g[i],
                &node_s[i],
                &counts[i],
                v.n_cat,
                cfg.ridge,
                cfg.newton_iter,
            );
            params[i].a_g = updated[0];
            if has_specific {
                params[i].a_s = Some(updated[1]);
                params[i].d = updated[2..].to_vec();
            } else {
                params[i].d = updated[1..].to_vec();
            }
        }
        n_iter += 1;
    }
    Ok(SingleStartOutcome {
        params,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
    })
}

/// Fit the single-group polytomous bifactor GRM by Bock-Aitkin marginal ML
/// with Gibbons-Hedeker dimension reduction. `y`/`observed` are row-major
/// `n_persons * n_items` (`y` ordered categories `0..n_cat-1`, missing cells
/// dropped MAR); `specific_map` is length `n_items` with `-1` for
/// general-only items and `0..n_specific` otherwise. Runs `n_starts` EM runs
/// and keeps the best loglik. Returns `Err` on malformed input, unobserved
/// categories, or total numerical failure; per-start non-convergence is
/// reported through the winning run's flags, never substituted.
#[allow(clippy::too_many_arguments)]
pub fn fit_bifactor_grm(
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &BifactorGrmConfig,
) -> Result<BifactorGrmResult, String> {
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        cfg,
    )?;
    let (tg, wg) = gh_rule(cfg.q_general)?;
    let (ts, ws) = gh_rule(cfg.q_specific)?;
    let qg = tg.len();
    let qs = ts.len();
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
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
            &v, y, observed, cfg, tg, ts, &log_wg, &log_ws, qg, qs, start,
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

    // Final EAP pass for theta_G at the winning parameters.
    let tables = fill_logprob_tables(&v, &params, tg, ts, qg, qs);
    let mut theta_g_eap = vec![0.0f64; n_persons];
    let mut theta_g_sd = vec![0.0f64; n_persons];
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];
    for p in 0..n_persons {
        let mut gen_log = log_wg.clone();
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * n_items + i];
            for g in 0..qg {
                gen_log[g] += tables[i][g * n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * n_items + i];
                        acc += tables[i][(g * qs + h) * n_cat + yc];
                    }
                    block_acc[(s * qg + g) * qs + h] = acc;
                }
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + g) * qs + h];
                }
                log_i[s * qg + g] = log_sum_exp(&tmp_h);
            }
        }
        for g in 0..qg {
            let mut acc = gen_log[g];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + g];
            }
            log_like_g[g] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        let (mut m1, mut m2) = (0.0f64, 0.0f64);
        for (&t, &ll) in tg.iter().zip(log_like_g.iter()) {
            let post = (ll - log_lp).exp();
            m1 += post * t;
            m2 += post * t * t;
        }
        theta_g_eap[p] = m1;
        theta_g_sd[p] = (m2 - m1 * m1).max(0.0).sqrt();
    }

    // Assemble dense outputs.
    let mut a_general = vec![0.0f64; n_items];
    let mut a_specific = vec![0.0f64; n_items];
    let mut threshold = vec![0.0f64; n_items * v.m1];
    let mut n_parameters = 0usize;
    for (i, par) in params.iter().enumerate() {
        a_general[i] = par.a_g;
        n_parameters += 1 + v.m1;
        if let Some(a_s) = par.a_s {
            a_specific[i] = a_s;
            n_parameters += 1;
        }
        threshold[i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
    }

    // Per-dimension reflection canonicalization (module docs): general over
    // all items, each specific within its block; thresholds untouched.
    let anchor_g = (0..n_items)
        .max_by(|&i, &j| a_general[i].abs().total_cmp(&a_general[j].abs()))
        .expect("at least one item");
    if a_general[anchor_g] < 0.0 {
        for a in a_general.iter_mut() {
            *a = -*a;
        }
        for t in theta_g_eap.iter_mut() {
            *t = -*t;
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
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) {
                category_counts[i * n_cat + y[p * n_items + i]] += 1;
            }
        }
    }

    Ok(BifactorGrmResult {
        a_general,
        a_specific,
        threshold,
        theta_g_eap,
        theta_g_sd,
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

/// Observed-data marginal loglik at GIVEN parameters via the Gibbons-Hedeker
/// reduced integration. Shared validator with [`fit_bifactor_grm`]; the
/// `thresholds` layout is row-major `n_items * (n_cat - 1)`. `a_specific` must
/// be exactly `0.0` for general-only items (it is ignored nowhere — a
/// non-zero value is a loud `Err`).
#[allow(clippy::too_many_arguments)]
pub fn bifactor_grm_marginal_loglik(
    a_general: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    q_general: usize,
    q_specific: usize,
) -> Result<f64, String> {
    let cfg = BifactorGrmConfig {
        q_general,
        q_specific,
        ..BifactorGrmConfig::default()
    };
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &cfg,
    )?;
    check_param_shapes(&v, a_general, a_specific, thresholds)?;
    let (tg, wg) = gh_rule(q_general)?;
    let (ts, ws) = gh_rule(q_specific)?;
    let params = pack_params(&v, a_general, a_specific, thresholds);
    let tables = fill_logprob_tables(&v, &params, tg, ts, tg.len(), ts.len());
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    Ok(e_step(
        &v,
        y,
        observed,
        &tables,
        &log_wg,
        &log_ws,
        tg.len(),
        ts.len(),
    )
    .0)
}

/// Observed-data marginal loglik at GIVEN parameters via BRUTE-FORCE full
/// product-grid integration over `Q_G * Q_S^S` nodes. Numerically identical to
/// [`bifactor_grm_marginal_loglik`] up to floating-point reorder noise; kept
/// public as the exactness oracle for the reduction (tiny models only — the
/// grid is capped at 2,000,000 nodes).
#[allow(clippy::too_many_arguments)]
pub fn bifactor_grm_marginal_loglik_brute(
    a_general: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    q_general: usize,
    q_specific: usize,
) -> Result<f64, String> {
    let cfg = BifactorGrmConfig {
        q_general,
        q_specific,
        ..BifactorGrmConfig::default()
    };
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &cfg,
    )?;
    check_param_shapes(&v, a_general, a_specific, thresholds)?;
    let (tg, wg) = gh_rule(q_general)?;
    let (ts, ws) = gh_rule(q_specific)?;
    let qg = tg.len();
    let qs = ts.len();
    let n_full = qg
        .checked_mul(qs.checked_pow(n_specific as u32).ok_or("grid too large")?)
        .ok_or_else(|| "product grid overflows usize".to_string())?;
    if n_full > 2_000_000 {
        return Err(format!(
            "brute-force grid {n_full} nodes exceeds the 2,000,000-node oracle cap"
        ));
    }
    let params = pack_params(&v, a_general, a_specific, thresholds);
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    // Full-grid node coordinates: index = g * qs^S + mixed-radix tail.
    let mut loglik = 0.0f64;
    let mut log_node = vec![0.0f64; n_full];
    for p in 0..n_persons {
        for (node, slot) in log_node.iter_mut().enumerate() {
            let q_pow_s = qs.pow(n_specific as u32);
            let g = node / q_pow_s;
            let tail = node % q_pow_s;
            // Big-endian base-qs digits: digit_s = (tail / qs^(S-1-s)) % qs.
            let digit = |s: usize| (tail / qs.pow((n_specific - 1 - s) as u32)) % qs;
            let mut acc = wg[g].ln();
            for s in 0..n_specific {
                acc += ws[digit(s)].ln();
            }
            // Per-item likelihood at this full node.
            for i in 0..n_items {
                if !is_obs(p, i) {
                    continue;
                }
                let (t_s, a_s) = match v.item_block[i] {
                    Some(s) => (ts[digit(s)], params[i].a_s.unwrap_or(0.0)),
                    None => (0.0, 0.0),
                };
                let base = params[i].a_g * tg[g] + a_s * t_s;
                let lp = grm_logprobs(base, &params[i].d);
                acc += lp[y[p * n_items + i]];
            }
            *slot = acc;
        }
        loglik += log_sum_exp(&log_node);
    }
    Ok(loglik)
}

fn check_param_shapes(
    v: &Validated,
    a_general: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
) -> Result<(), String> {
    if a_general.len() != v.n_items || a_specific.len() != v.n_items {
        return Err("a_general/a_specific must have length n_items".into());
    }
    if thresholds.len() != v.n_items * v.m1 {
        return Err("thresholds must have length n_items * (n_cat - 1)".into());
    }
    if [a_general, a_specific, thresholds]
        .concat()
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err("parameters must be finite".into());
    }
    for (i, block) in v.item_block.iter().enumerate() {
        if block.is_none() && a_specific[i] != 0.0 {
            return Err(format!(
                "a_specific[{i}] must be exactly 0.0 for general-only items \
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
    Ok(())
}

fn pack_params(
    v: &Validated,
    a_general: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
) -> Vec<ItemParams> {
    (0..v.n_items)
        .map(|i| ItemParams {
            a_g: a_general[i],
            a_s: v.item_block[i].map(|_| a_specific[i]),
            d: thresholds[i * v.m1..(i + 1) * v.m1].to_vec(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Multiple-group concurrent calibration (stage 2 of #1912).
//
// Groups share item parameters (anchored items) with optional per-item
// free/anchored flags; the reference group (0) is pinned to `N(0, I)` and each
// focal group's general-factor mean/variance — plus, when requested, its
// specific-factor variances (means fixed at 0) — are estimated by marginal ML
// via EM with the Gibbons-Hedeker reduction applied per group:
//
// ```text
// L_pg = sum_g w_g * G_pg(g) * prod_s I_psg(g),
// I_psg(g) = sum_h v_h * prod_{i in s} P(Y_pi | theta_{G,g}, theta_{S,g,s}),
// theta_{G,g,t} = mu_g + sigma_g * X_t,  theta_{S,g,s,h} = tau_{g,s} * X_h,
// ```
//
// with the shared Gauss-Hermite weights (the node-shift reparameterization
// keeps the weights; cf. `poly::fit_poly_multigroup`'s
// `theta_{g,t} = mu_g + sigma_g x_t` for the Bock-Zimowski pooling).
//
// Two deliberate departures from Gibbons et al. (2007), both implementation
// choices: the link is logistic rather than their normal ogive (to match the
// `mirt` graded comparison in this repository's fixture), and the intercepts
// are written directly as `d_ik` rather than split into `c_j + d_t`.
//
// # Verified paper locators (full texts read; no invented equation numbers)
// periodically re-verified against the PDFs (stage-1 review fix-up)
//
// - Gibbons et al. (2007), `~/papers/Gibbons2007_APM_bifactor_GRM.pdf`
//   (Appl. Psych. Meas. 31(1), 4-19): Samejima GRM category form eq. 4 (p. 6);
//   bifactor linear predictor `z = c + d + sum a_jk theta_k` eq. 9
//   ("The Bifactor Model for Graded Response Data" section);
//   Gibbons & Hedeker (1992) reduce the "s-dimensional integral in (4) to a
//   two-dimensional integral" via Stuart's (1958) reduction for variates each
//   related to a single dimension only (their eq. 6); the person marginal
//   factors per general node (Gibbons et al., 2007, eq. 15, "Marginal Maximum
//   Likelihood Estimation" section); marginal decomposition eq. 15 and loglik
//   eq. 16 (p. 9); general-factor EAP eq. 17 and posterior variance eq. 18
//   (pp. 9-10).
// - Cai, Yang, & Hansen (2011), Zotero `TNQ22C7T` (Psych. Methods 16(3),
//   221-248; full text read via the local Zotero API attachment): extend
//   "Gibbons and Hedeker's (1992) bifactor dimension reduction method"
//   (p. 221) and estimate with "the Bock and Aitkin (1981) EM algorithm"
//   ("Maximum Marginal Likelihood Estimation" section); the multiple-group
//   reference-group paragraph near Fig. 7 — reference latent variables have
//   zero means and identity covariance, focal location/scale are estimated
//   relative to the reference, with at least one common item's parameters
//   set equal across groups to link the scales; graded cumulative
//   `P(y >= k) = 1/(1+exp(-[d_k + a0*theta0 + as*theta_s]))` in the category-
//   response-probabilities section.
// - Bock & Zimowski (1997), Handbook of Modern IRT chap. 25 (pp. 433-448),
//   multiple-group IRT (conceptual; no equation locator claimed — the pooled
//   item M-step stacks each group's nodes and expected counts, exactly the
//   `poly::fit_poly_multigroup` Bock-Zimowski pooling already in this crate).
// - Bafumi et al. (2005) for fixing the per-dimension reflection
//   `(a, theta) -> (-a, -theta)` by a parameter restriction; here the crate
//   rule (largest-magnitude slope positive per dimension, read from the
//   ANCHORED linking items only so a free item's DIF outlier cannot drive the
//   global orientation, applied jointly across groups; general flip also
//   negates every group mean and the reported
//   general EAPs — the #1879 mu-sign fix).
//
// # Caller-owned numerics (no hidden clamps, no magic caps)
//
// `q_general`, `q_specific`, `max_iter`, `tol`, `n_starts`, `seed` are caller
// arguments; any out-of-range value is a loud `Err`, never a silent clamp.
// Upper bounds exist only where a real constraint exists: the quadrature
// counts must name an embedded Gauss-Hermite rule (`SUPPORTED_Q`), and
// working-set sizes that would overflow `usize` are rejected by checked
// arithmetic. Everything else is lower-bounded only. `seed`
// drives ONLY the random-start jitter (quadrature is deterministic); start
// `t` derives from `seed ^ GOLDEN * (t + 1)`. Group variances are estimated
// unconstrained-positive with NO clamping: a non-finite or non-positive
// update fails that start loudly (failed starts are skipped; all-failed
// reports the first error). At least one anchored item is required when
// `n_groups >= 2` (Cai et al. 2011 linking requirement).
//
// # Failure reporting
//
// Every declared category must be observed pooled for every item (Samejima,
// 1969); additionally every free item must show every declared category in
// EVERY group (its per-group parameters are otherwise unidentified). Both
// are loud `Err`s naming item/group/category. Non-convergence reports
// `converged == false` with `termination_reason == "max_iter_reached"`.
//
// # References (APA 7th ed.)
//
// Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
// D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
// Full-information item bifactor analysis of graded response data. *Applied
// Psychological Measurement, 31*(1), 4-19.
// https://doi.org/10.1177/0146621606289485
//
// Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
// item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
// https://doi.org/10.1037/a0023350
//
// Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J.
// van der Linden & R. K. Hambleton (Eds.), *Handbook of modern item response
// theory* (pp. 433-448). Springer. https://doi.org/10.1007/978-1-4757-2691-6_25
//
// Bafumi, J., Gelman, A., Park, D. K., & Kaplan, N. (2005). Practical issues in
// implementing and understanding Bayesian ideal point estimation. *Political
// Analysis, 13*(2), 171-187. https://doi.org/10.1093/pan/mpi010

/// Configuration for [`fit_bifactor_grm_multigroup`]. Numeric fields mirror
/// [`BifactorGrmConfig`]; `estimate_specific_vars` selects whether focal
/// specific-factor variances (means fixed at 0) are estimated (`true`) or
/// held at the reference value 1 (`false`).
#[derive(Clone, Copy, Debug)]
pub struct BifactorMultigroupConfig {
    /// Gauss-Hermite nodes for the general factor (one of `SUPPORTED_Q`).
    pub q_general: usize,
    /// Gauss-Hermite nodes per specific factor (one of `SUPPORTED_Q`).
    pub q_specific: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// Number of EM runs from jittered starts; the best loglik wins.
    pub n_starts: usize,
    /// Seeds ONLY the random-start jitter (quadrature is deterministic).
    pub seed: u64,
    /// Inner Newton iterations per item M-step.
    pub newton_iter: usize,
    /// Newton ridge — Hessian CONDITIONING only, NOT a parameter prior.
    pub ridge: f64,
    /// Estimate focal specific-factor variances (`true`) or fix at 1 (`false`).
    pub estimate_specific_vars: bool,
}

impl Default for BifactorMultigroupConfig {
    fn default() -> Self {
        Self {
            q_general: 21,
            q_specific: 11,
            max_iter: 500,
            tol: 1e-6,
            n_starts: 1,
            seed: 0x9E37_79B9_7F4A_7C15,
            newton_iter: 10,
            ridge: 1e-8,
            estimate_specific_vars: false,
        }
    }
}

/// Result of [`fit_bifactor_grm_multigroup`].
///
/// Item tables are `n_groups` rows: `a_general[g][i]`, `a_specific[g][i]`
/// (`0.0` for general-only items), `threshold[g][i * (n_cat - 1)..]`.
/// Anchored items are identical across rows by construction (a single shared
/// estimate copied to every group); free items hold per-group estimates.
/// `general_mean[0] == 0`, `general_sd[0] == 1`, `specific_sd[0][*] == 1` are
/// the pinned reference. `theta_g_eap`/`theta_g_sd` are per-person
/// general-factor EAPs on the COMMON (reference) scale.
#[derive(Clone, Debug)]
pub struct BifactorMultigroupResult {
    pub a_general: Vec<Vec<f64>>,
    pub a_specific: Vec<Vec<f64>>,
    pub threshold: Vec<Vec<f64>>,
    /// General-factor means, length `n_groups` (`[0] == 0` pinned).
    pub general_mean: Vec<f64>,
    /// General-factor SDs, length `n_groups` (`[0] == 1` pinned).
    pub general_sd: Vec<f64>,
    /// Specific-factor SDs, `n_groups x n_specific` (`[0][*] == 1` pinned;
    /// focal rows are 1 unless `estimate_specific_vars`).
    pub specific_sd: Vec<Vec<f64>>,
    /// General-factor EAP on the common scale, length `n_persons`.
    pub theta_g_eap: Vec<f64>,
    /// General-factor posterior SD, length `n_persons`.
    pub theta_g_sd: Vec<f64>,
    /// Observed-data category counts per group, `n_groups` rows of
    /// `n_items * n_cat`.
    pub group_category_counts: Vec<Vec<usize>>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// Winning start index in `0..n_starts` (deterministic from `seed`).
    pub best_start: usize,
    /// Free item parameters (anchored counted once, free per group) plus
    /// estimated group distribution parameters.
    pub n_parameters: usize,
}

fn validate_multigroup_cfg(cfg: &BifactorMultigroupConfig) -> Result<(), String> {
    // Mirror the single-group checks exactly (lower bounds only, no magic
    // caps — stage-1 review fix-up; quadrature restricted to the embedded
    // rules that exist). Caller-owned numerics behave identically.
    if !SUPPORTED_Q.contains(&cfg.q_general) {
        return Err(format!(
            "q_general must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_general
        ));
    }
    if !SUPPORTED_Q.contains(&cfg.q_specific) {
        return Err(format!(
            "q_specific must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_specific
        ));
    }
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
    Ok(())
}

/// Group-aware initial parameters: anchored items from pooled base rates,
/// free items from group-specific base rates (start 0); starts `t >= 1` add
/// deterministic `N(0, *)` jitter from `SplitMix64(seed ^ GOLDEN * (t + 1))`
/// and re-sort `d` decreasing. Group distributions start at the reference
/// (`mu = 0`, `sigma = 1`, `tau = 1`); starts `t >= 1` jitter focal
/// `mu += 0.2 N`, `sigma *= exp(0.1 N)`, `tau *= exp(0.1 N)` (taus only when
/// estimated — otherwise they stay pinned at 1).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)] // group/item/node indexing is inherently indexed
fn initial_params_multigroup(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    anchor: &[bool],
    seed: u64,
    start: usize,
    estimate_specific_vars: bool,
) -> (Vec<Vec<ItemParams>>, Vec<f64>, Vec<f64>, Vec<Vec<f64>>) {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    let mut rng = SplitMix64(seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul(start as u64 + 1)));
    // Base-rate intercepts: pooled for anchored, per-group for free.
    let pooled_d = |item: usize, rng: &mut SplitMix64| -> Vec<f64> {
        let mut freq = vec![1e-3f64; v.n_cat];
        for p in 0..v.n_persons {
            if is_obs(p, item) {
                freq[y[p * v.n_items + item]] += 1.0;
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
        if start > 0 {
            for dk in d.iter_mut() {
                *dk += 0.25 * rng.standard_normal();
            }
            d.sort_by(|x, y| y.total_cmp(x));
        }
        d
    };
    let group_d = |item: usize, g: usize, rng: &mut SplitMix64| -> Vec<f64> {
        let mut freq = vec![1e-3f64; v.n_cat];
        for p in 0..v.n_persons {
            if group_id[p] == g && is_obs(p, item) {
                freq[y[p * v.n_items + item]] += 1.0;
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
        if start > 0 {
            for dk in d.iter_mut() {
                *dk += 0.25 * rng.standard_normal();
            }
            d.sort_by(|x, y| y.total_cmp(x));
        }
        d
    };
    // Item inits in item order (anchored share one draw, free draw per group
    // in group order) so the sequence is deterministic given
    // `(seed, start, anchor)`.
    let mut anchored_init: Vec<Option<ItemParams>> = vec![None; v.n_items];
    for i in 0..v.n_items {
        if !anchor[i] {
            continue;
        }
        let d = pooled_d(i, &mut rng);
        let (mut a_g, mut a_s) = (1.0f64, v.item_block[i].map(|_| 0.8f64));
        if start > 0 {
            a_g += 0.4 * rng.standard_normal();
            if let Some(a) = a_s.as_mut() {
                *a += 0.4 * rng.standard_normal();
            }
        }
        anchored_init[i] = Some(ItemParams { a_g, a_s, d });
    }
    let mut params_groups: Vec<Vec<ItemParams>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut row = Vec::with_capacity(v.n_items);
        for i in 0..v.n_items {
            if anchor[i] {
                row.push(anchored_init[i].clone().expect("anchored init filled"));
            } else {
                let d = group_d(i, g, &mut rng);
                let (mut a_g, mut a_s) = (1.0f64, v.item_block[i].map(|_| 0.8f64));
                if start > 0 {
                    a_g += 0.4 * rng.standard_normal();
                    if let Some(a) = a_s.as_mut() {
                        *a += 0.4 * rng.standard_normal();
                    }
                }
                row.push(ItemParams { a_g, a_s, d });
            }
        }
        params_groups.push(row);
    }
    let mut mus = vec![0.0f64; n_groups];
    let mut sigmas = vec![1.0f64; n_groups];
    let mut taus = vec![vec![1.0f64; v.n_specific]; n_groups];
    if start > 0 {
        for g in 1..n_groups {
            mus[g] += 0.2 * rng.standard_normal();
            sigmas[g] = (0.1 * rng.standard_normal()).exp();
            if estimate_specific_vars {
                for s in 0..v.n_specific {
                    taus[g][s] = (0.1 * rng.standard_normal()).exp();
                }
            }
        }
    }
    (params_groups, mus, sigmas, taus)
}

struct MultiStartOutcome {
    params_groups: Vec<Vec<ItemParams>>,
    mus: Vec<f64>,
    sigmas: Vec<f64>,
    taus: Vec<Vec<f64>>,
    loglik_trace: Vec<f64>,
    n_iter: usize,
    converged: bool,
    termination_reason: String,
    final_loglik_change: f64,
}

/// One reduced E-step sweep over all groups: total observed-data loglik, plus
/// per-group expected category counts (`counts[g][i][node][k]`,
/// `node = t * qs + h` for block items, `node = t` for general-only items)
/// and the posterior moments that drive the group-distribution M-step
/// (`s1_g/s2_g` for the general factor, `s2_spec[g][s]` for the specifics at
/// zero mean).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
#[allow(clippy::needless_range_loop)] // group/item/node indexing is inherently indexed
fn e_step_multigroup(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    params_groups: &[Vec<ItemParams>],
    tg_groups: &[Vec<f64>],
    ts_groups: &[Vec<Vec<f64>>],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
) -> (
    f64,
    Vec<Vec<Vec<Vec<f64>>>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
) {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    // Per-group logprob tables at the CURRENT group nodes.
    let mut tables_groups: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut tables = Vec::with_capacity(v.n_items);
        for (i, par) in params_groups[g].iter().enumerate() {
            match par.a_s {
                Some(a_s) => {
                    let s = v.item_block[i].expect("block item has an owning block");
                    let mut lp = vec![0.0f64; qg * qs * v.n_cat];
                    for t in 0..qg {
                        for h in 0..qs {
                            let base =
                                par.a_g * tg_groups[g][t] + a_s * ts_groups[g][s][h];
                            let probs = grm_logprobs(base, &par.d);
                            lp[(t * qs + h) * v.n_cat..(t * qs + h + 1) * v.n_cat]
                                .copy_from_slice(&probs);
                        }
                    }
                    tables.push(lp);
                }
                None => {
                    let mut lp = vec![0.0f64; qg * v.n_cat];
                    for t in 0..qg {
                        let base = par.a_g * tg_groups[g][t];
                        let probs = grm_logprobs(base, &par.d);
                        lp[t * v.n_cat..(t + 1) * v.n_cat].copy_from_slice(&probs);
                    }
                    tables.push(lp);
                }
            }
        }
        tables_groups.push(tables);
    }

    let mut counts: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let _ = g;
        let mut cg = Vec::with_capacity(v.n_items);
        for i in 0..v.n_items {
            let n_nodes = if v.item_block[i].is_some() {
                qg * qs
            } else {
                qg
            };
            cg.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
        }
        counts.push(cg);
    }
    let mut w_acc = vec![0.0f64; n_groups];
    let mut s1_g = vec![0.0f64; n_groups];
    let mut s2_g = vec![0.0f64; n_groups];
    let mut s2_spec = vec![vec![0.0f64; v.n_specific]; n_groups];
    // Per-(group, block) posterior mass for the specific-variance M-step:
    // persons with no observed item in block `s` contribute nothing to that
    // block's moments and must not dilute its denominator (review fix for
    // block-wise MAR with `estimate_specific_vars`).
    let mut w_spec = vec![vec![0.0f64; v.n_specific]; n_groups];

    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut gen_log = vec![0.0f64; qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut post_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];

    let mut loglik = 0.0f64;
    for p in 0..v.n_persons {
        let g = group_id[p];
        let tables = &tables_groups[g];
        gen_log.copy_from_slice(log_wg);
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            let lp = &tables[i];
            for t in 0..qg {
                gen_log[t] += lp[t * v.n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for t in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        acc += tables[i][(t * qs + h) * v.n_cat + yc];
                    }
                    block_acc[(s * qg + t) * qs + h] = acc;
                }
            }
            for t in 0..qg {
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + t) * qs + h];
                }
                log_i[s * qg + t] = log_sum_exp(&tmp_h);
            }
        }
        for t in 0..qg {
            let mut acc = gen_log[t];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + t];
            }
            log_like_g[t] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        loglik += log_lp;
        for t in 0..qg {
            post_g[t] = (log_like_g[t] - log_lp).exp();
        }
        // General moments (common-scale nodes) for the group M-step.
        for t in 0..qg {
            w_acc[g] += post_g[t];
            s1_g[g] += post_g[t] * tg_groups[g][t];
            s2_g[g] += post_g[t] * tg_groups[g][t] * tg_groups[g][t];
        }
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            for t in 0..qg {
                counts[g][i][t][yc] += post_g[t];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            let any_obs = members.iter().any(|&i| is_obs(p, i));
            if !any_obs {
                continue;
            }
            for t in 0..qg {
                let mut others = gen_log[t] - log_wg[t];
                for s2 in 0..v.n_specific {
                    if s2 != s {
                        others += log_i[s2 * qg + t];
                    }
                }
                for h in 0..qs {
                    let log_post =
                        log_wg[t] + block_acc[(s * qg + t) * qs + h] + others - log_lp;
                    let post = log_post.exp();
                    // Specific moments at zero mean for the variance M-step.
                    let ts = ts_groups[g][s][h];
                    w_spec[g][s] += post;
                    s2_spec[g][s] += post * ts * ts;
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        counts[g][i][t * qs + h][yc] += post;
                    }
                }
            }
        }
    }
    (loglik, counts, w_acc, s1_g, s2_g, s2_spec, w_spec)
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // group/item/node indexing is inherently indexed
fn run_single_start_multigroup(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    anchor: &[bool],
    cfg: &BifactorMultigroupConfig,
    tg_std: &[f64],
    ts_std: &[f64],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
    start: usize,
) -> Result<MultiStartOutcome, String> {
    let (mut params_groups, mut mus, mut sigmas, mut taus) = initial_params_multigroup(
        v,
        y,
        observed,
        group_id,
        n_groups,
        anchor,
        cfg.seed,
        start,
        cfg.estimate_specific_vars,
    );
    // Latent coordinates per expected-count node (rebuilt when mus/sigmas/
    // taus move): node_g[node]/node_s[node] parallel `counts[g][i]`.
    // No pre-allocation from `max_iter`: it is caller-owned and unbounded
    // above, so `with_capacity(max_iter + 1)` could overflow; the trace grows
    // amortized instead (stage-1 review fix-up).
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;

    loop {
        // Common-scale group nodes for this sweep.
        let mut tg_groups = vec![vec![0.0f64; qg]; n_groups];
        let mut ts_groups = vec![vec![vec![0.0f64; qs]; v.n_specific]; n_groups];
        for g in 0..n_groups {
            for (t, &x) in tg_std.iter().enumerate() {
                tg_groups[g][t] = mus[g] + sigmas[g] * x;
            }
            for s in 0..v.n_specific {
                for (h, &x) in ts_std.iter().enumerate() {
                    ts_groups[g][s][h] = taus[g][s] * x;
                }
            }
        }
        let (ll, counts, w_acc, s1_g, s2_g, s2_spec, w_spec) = e_step_multigroup(
            v,
            y,
            observed,
            group_id,
            n_groups,
            &params_groups,
            &tg_groups,
            &ts_groups,
            log_wg,
            log_ws,
            qg,
            qs,
        );
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
        // M-step, item parameters: anchored items pool every group's nodes
        // and expected counts (Bock-Zimowski pooling); free items fit per
        // group. Stacked node order is groups outer, `(t, h)` inner with
        // `node = t * qs + h` for block items (matching the E-step layout).
        for i in 0..v.n_items {
            let has_specific = v.item_block[i].is_some();
            if anchor[i] {
                let per_group = if has_specific { qg * qs } else { qg };
                let mut stacked_g = Vec::with_capacity(n_groups * per_group);
                let mut stacked_s = Vec::with_capacity(n_groups * per_group);
                let mut stacked_c = Vec::with_capacity(n_groups * per_group);
                for g in 0..n_groups {
                    if has_specific {
                        let s = v.item_block[i].expect("block item has a block");
                        for t in 0..qg {
                            for h in 0..qs {
                                stacked_g.push(tg_groups[g][t]);
                                stacked_s.push(ts_groups[g][s][h]);
                                stacked_c.push(counts[g][i][t * qs + h].clone());
                            }
                        }
                    } else {
                        for t in 0..qg {
                            stacked_g.push(tg_groups[g][t]);
                            stacked_s.push(0.0);
                            stacked_c.push(counts[g][i][t].clone());
                        }
                    }
                }
                let mut packed = Vec::with_capacity(1 + has_specific as usize + v.m1);
                packed.push(params_groups[0][i].a_g);
                if let Some(a_s) = params_groups[0][i].a_s {
                    packed.push(a_s);
                }
                packed.extend_from_slice(&params_groups[0][i].d);
                let updated = m_step_item(
                    packed,
                    has_specific,
                    &stacked_g,
                    &stacked_s,
                    &stacked_c,
                    v.n_cat,
                    cfg.ridge,
                    cfg.newton_iter,
                );
                for g in 0..n_groups {
                    params_groups[g][i].a_g = updated[0];
                    if has_specific {
                        params_groups[g][i].a_s = Some(updated[1]);
                        params_groups[g][i].d = updated[2..].to_vec();
                    } else {
                        params_groups[g][i].d = updated[1..].to_vec();
                    }
                }
            } else {
                for g in 0..n_groups {
                    let (node_g, node_s): (Vec<f64>, Vec<f64>) = if has_specific {
                        let s = v.item_block[i].expect("block item has a block");
                        let mut gg = Vec::with_capacity(qg * qs);
                        let mut ss = Vec::with_capacity(qg * qs);
                        for t in 0..qg {
                            for h in 0..qs {
                                gg.push(tg_groups[g][t]);
                                ss.push(ts_groups[g][s][h]);
                            }
                        }
                        (gg, ss)
                    } else {
                        (tg_groups[g].clone(), vec![0.0; qg])
                    };
                    let mut packed = Vec::with_capacity(1 + has_specific as usize + v.m1);
                    packed.push(params_groups[g][i].a_g);
                    if let Some(a_s) = params_groups[g][i].a_s {
                        packed.push(a_s);
                    }
                    packed.extend_from_slice(&params_groups[g][i].d);
                    let updated = m_step_item(
                        packed,
                        has_specific,
                        &node_g,
                        &node_s,
                        &counts[g][i],
                        v.n_cat,
                        cfg.ridge,
                        cfg.newton_iter,
                    );
                    params_groups[g][i].a_g = updated[0];
                    if has_specific {
                        params_groups[g][i].a_s = Some(updated[1]);
                        params_groups[g][i].d = updated[2..].to_vec();
                    } else {
                        params_groups[g][i].d = updated[1..].to_vec();
                    }
                }
            }
        }
        // M-step, focal group distributions (reference g = 0 pinned to
        // N(0, I)). General: mu = mean EAP, var = mean posterior second
        // moment minus mu^2 (Bock-Aitkin/Bock-Zimowski moment update).
        // Specifics (when estimated): tau^2 = mean posterior second moment
        // at zero mean. All unconstrained-positive with NO clamping: a
        // non-finite or non-positive update fails the start loudly.
        for g in 1..n_groups {
            if w_acc[g] <= 0.0 || !w_acc[g].is_finite() {
                return Err(format!("group {g} has no posterior mass"));
            }
            let mean = s1_g[g] / w_acc[g];
            let var = s2_g[g] / w_acc[g] - mean * mean;
            if !mean.is_finite() || !var.is_finite() {
                return Err(format!("non-finite group-{g} general moment update"));
            }
            if var <= 0.0 {
                return Err(format!(
                    "non-positive group-{g} general variance update ({var:.6e})"
                ));
            }
            mus[g] = mean;
            sigmas[g] = var.sqrt();
            if cfg.estimate_specific_vars {
                for s in 0..v.n_specific {
                    // Denominator counts only persons observed in this block
                    // (block-wise MAR must not dilute the variance update).
                    if w_spec[g][s] <= 0.0 || !w_spec[g][s].is_finite() {
                        return Err(format!(
                            "group {g} specific-{s} has no posterior mass"
                        ));
                    }
                    let vrow = s2_spec[g][s] / w_spec[g][s];
                    if !vrow.is_finite() {
                        return Err(format!("non-finite group-{g} specific-{s} update"));
                    }
                    if vrow <= 0.0 {
                        return Err(format!(
                            "non-positive group-{g} specific-{s} variance update ({vrow:.6e})"
                        ));
                    }
                    taus[g][s] = vrow.sqrt();
                }
            }
        }
        n_iter += 1;
    }
    Ok(MultiStartOutcome {
        params_groups,
        mus,
        sigmas,
        taus,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
    })
}

/// Fit the multiple-group polytomous bifactor GRM by Bock-Aitkin marginal ML
/// with the Gibbons-Hedeker reduction applied per group. `y`/`observed` are
/// row-major `n_persons * n_items` (`y` ordered categories `0..n_cat-1`,
/// missing cells dropped MAR); `group_id` is length `n_persons` with values
/// in `0..n_groups` (group 0 is the reference, pinned to `N(0, I)`);
/// `specific_map` is length `n_items` with `-1` for general-only items and
/// `0..n_specific` otherwise; `anchor` is `None` (all items common) or length
/// `n_items` with `true` = common across groups and `false` = free per group
/// (at least one anchored item is required when `n_groups >= 2` to link the
/// scales — Cai, Yang, & Hansen, 2011). Runs `n_starts` EM runs and keeps the
/// best loglik. With `n_groups == 1` this delegates to [`fit_bifactor_grm`]
/// and bit-reproduces the stage-1 result. Returns `Err` on malformed input,
/// unobserved categories (pooled for every item; per-group for every free
/// item), or total numerical failure; per-start non-convergence is reported
/// through the winning run's flags, never substituted.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // group/item/node indexing is inherently indexed
pub fn fit_bifactor_grm_multigroup(
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    anchor: Option<&[bool]>,
    cfg: &BifactorMultigroupConfig,
) -> Result<BifactorMultigroupResult, String> {
    validate_multigroup_cfg(cfg)?;
    if n_groups < 1 {
        return Err("n_groups must be >= 1".into());
    }
    // Multigroup-shaped inputs are validated BEFORE the single-group
    // delegation so malformed `group_id`/`anchor` fail loudly on every path
    // (review fix: delegation must not silently ignore them).
    if group_id.len() != n_persons {
        return Err("group_id must have length n_persons".into());
    }
    if group_id.iter().any(|&g| g >= n_groups) {
        return Err("group_id labels must be < n_groups".into());
    }
    let anchor_vec: Vec<bool> = match anchor {
        Some(a) => {
            if a.len() != n_items {
                return Err("anchor must have length n_items".into());
            }
            a.to_vec()
        }
        None => vec![true; n_items],
    };
    // With a single group there is no multigroup structure to estimate: the
    // node-shift is the identity and the pooled M-step is the single-group
    // M-step, so delegate to stage 1 and bit-reproduce it exactly.
    if n_groups == 1 {
        let single_cfg = BifactorGrmConfig {
            q_general: cfg.q_general,
            q_specific: cfg.q_specific,
            max_iter: cfg.max_iter,
            tol: cfg.tol,
            n_starts: cfg.n_starts,
            seed: cfg.seed,
            newton_iter: cfg.newton_iter,
            ridge: cfg.ridge,
        };
        let single = fit_bifactor_grm(
            y,
            observed,
            specific_map,
            n_persons,
            n_items,
            n_specific,
            n_cat,
            &single_cfg,
        )?;
        return Ok(BifactorMultigroupResult {
            a_general: vec![single.a_general],
            a_specific: vec![single.a_specific],
            threshold: vec![single.threshold],
            general_mean: vec![0.0],
            general_sd: vec![1.0],
            specific_sd: vec![vec![1.0; n_specific]],
            theta_g_eap: single.theta_g_eap,
            theta_g_sd: single.theta_g_sd,
            group_category_counts: vec![single.category_counts],
            loglik_trace: single.loglik_trace,
            n_iter: single.n_iter,
            converged: single.converged,
            termination_reason: single.termination_reason,
            final_loglik_change: single.final_loglik_change,
            best_start: single.best_start,
            n_parameters: single.n_parameters,
        });
    }
    // Base structural validation (pooled categories, blocks, checked
    // arithmetic) reuses the single-group validator so the two paths accept
    // exactly the same data layouts.
    let single_cfg = BifactorGrmConfig {
        q_general: cfg.q_general,
        q_specific: cfg.q_specific,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_starts: cfg.n_starts,
        seed: cfg.seed,
        newton_iter: cfg.newton_iter,
        ridge: cfg.ridge,
    };
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &single_cfg,
    )?;
    // Working-set sizes with the multigroup factor: overflow is a loud `Err`
    // (the single-group validator covers the per-group grid; the `n_groups`
    // expansion is checked here).
    let nodes_per_item = cfg
        .q_general
        .checked_mul(cfg.q_specific)
        .ok_or_else(|| "q_general * q_specific overflows usize".to_string())?;
    n_items
        .checked_mul(nodes_per_item)
        .and_then(|v| v.checked_mul(n_cat))
        .and_then(|v| v.checked_mul(n_groups))
        .ok_or_else(|| {
            "n_groups * n_items * q_general * q_specific * n_cat overflows usize".to_string()
        })?;
    let mut group_n = vec![0usize; n_groups];
    for &g in group_id {
        group_n[g] += 1;
    }
    if group_n.contains(&0) {
        return Err("every group 0..n_groups-1 must contain at least one person".into());
    }
    if !anchor_vec.iter().any(|&a| a) {
        return Err(
            "at least one anchored (common) item is required to link the group scales \
             (Cai, Yang, & Hansen, 2011)"
                .into(),
        );
    }
    // Free items need every declared category in EVERY group (their per-group
    // boundaries are otherwise unidentified). Anchored items are identified
    // from the pooled data (already checked by `validate`), so a group may
    // lose a category on an anchored item without changing the parameter
    // count — the declared `n_cat` stays fixed across groups (#1912 rule).
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    for i in 0..n_items {
        if !anchor_vec[i] {
            for g in 0..n_groups {
                let mut seen = vec![false; n_cat];
                let mut any = false;
                for p in 0..n_persons {
                    if group_id[p] == g && is_obs(p, i) {
                        any = true;
                        seen[y[p * n_items + i]] = true;
                    }
                }
                if !any {
                    return Err(format!("free item {i} has no observed responses in group {g}"));
                }
                if let Some(k) = (0..n_cat).find(|&k| !seen[k]) {
                    return Err(format!(
                        "free item {i} category {k} is never observed in group {g} \
                         (unidentified per-group GRM boundary)"
                    ));
                }
            }
        }
    }

    let (tg_std, wg) = gh_rule(cfg.q_general)?;
    let (ts_std, ws) = gh_rule(cfg.q_specific)?;
    let qg = tg_std.len();
    let qs = ts_std.len();
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

    let mut best: Option<MultiStartOutcome> = None;
    let mut best_ll = f64::NEG_INFINITY;
    let mut best_start = 0usize;
    let mut first_error: Option<String> = None;
    for start in 0..cfg.n_starts {
        match run_single_start_multigroup(
            &v,
            y,
            observed,
            group_id,
            n_groups,
            &anchor_vec,
            cfg,
            tg_std,
            ts_std,
            &log_wg,
            &log_ws,
            qg,
            qs,
            start,
        ) {
            Ok(outcome) => {
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

    // Final EAP pass for theta_G on the common (reference) scale at the
    // winning parameters (Gibbons et al., 2007, eqs. 17-18, pp. 9-10, applied
    // per group at that group's common-scale nodes).
    let mut tg_groups = vec![vec![0.0f64; qg]; n_groups];
    let mut ts_groups = vec![vec![vec![0.0f64; qs]; v.n_specific]; n_groups];
    for g in 0..n_groups {
        for (t, &x) in tg_std.iter().enumerate() {
            tg_groups[g][t] = outcome.mus[g] + outcome.sigmas[g] * x;
        }
        for s in 0..v.n_specific {
            for (h, &x) in ts_std.iter().enumerate() {
                ts_groups[g][s][h] = outcome.taus[g][s] * x;
            }
        }
    }
    // Rebuild per-group tables at the winning parameters.
    let mut tables_groups: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut tables = Vec::with_capacity(v.n_items);
        for (i, par) in outcome.params_groups[g].iter().enumerate() {
            match par.a_s {
                Some(a_s) => {
                    let s = v.item_block[i].expect("block item has a block");
                    let mut lp = vec![0.0f64; qg * qs * v.n_cat];
                    for t in 0..qg {
                        for h in 0..qs {
                            let base =
                                par.a_g * tg_groups[g][t] + a_s * ts_groups[g][s][h];
                            let probs = grm_logprobs(base, &par.d);
                            lp[(t * qs + h) * v.n_cat..(t * qs + h + 1) * v.n_cat]
                                .copy_from_slice(&probs);
                        }
                    }
                    tables.push(lp);
                }
                None => {
                    let mut lp = vec![0.0f64; qg * v.n_cat];
                    for t in 0..qg {
                        let base = par.a_g * tg_groups[g][t];
                        let probs = grm_logprobs(base, &par.d);
                        lp[t * v.n_cat..(t + 1) * v.n_cat].copy_from_slice(&probs);
                    }
                    tables.push(lp);
                }
            }
        }
        tables_groups.push(tables);
    }
    let mut theta_g_eap = vec![0.0f64; n_persons];
    let mut theta_g_sd = vec![0.0f64; n_persons];
    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];
    for p in 0..n_persons {
        let g = group_id[p];
        let tables = &tables_groups[g];
        let mut gen_log = log_wg.clone();
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * n_items + i];
            for t in 0..qg {
                gen_log[t] += tables[i][t * n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for t in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * n_items + i];
                        acc += tables[i][(t * qs + h) * n_cat + yc];
                    }
                    block_acc[(s * qg + t) * qs + h] = acc;
                }
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + t) * qs + h];
                }
                log_i[s * qg + t] = log_sum_exp(&tmp_h);
            }
        }
        for t in 0..qg {
            let mut acc = gen_log[t];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + t];
            }
            log_like_g[t] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        let (mut m1, mut m2) = (0.0f64, 0.0f64);
        for (t, &ll) in log_like_g.iter().enumerate() {
            let post = (ll - log_lp).exp();
            let theta = tg_groups[g][t];
            m1 += post * theta;
            m2 += post * theta * theta;
        }
        theta_g_eap[p] = m1;
        theta_g_sd[p] = (m2 - m1 * m1).max(0.0).sqrt();
    }

    // Assemble dense per-group outputs (anchored rows already identical).
    let mut a_general = vec![vec![0.0f64; n_items]; n_groups];
    let mut a_specific = vec![vec![0.0f64; n_items]; n_groups];
    let mut threshold = vec![vec![0.0f64; n_items * v.m1]; n_groups];
    for g in 0..n_groups {
        for (i, par) in outcome.params_groups[g].iter().enumerate() {
            a_general[g][i] = par.a_g;
            if let Some(a_s) = par.a_s {
                a_specific[g][i] = a_s;
            }
            threshold[g][i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
        }
    }
    let mut general_mean = outcome.mus;
    let general_sd = outcome.sigmas;
    let specific_sd = outcome.taus;

    // Joint reflection canonicalization across groups: one decision per
    // dimension, so anchored equality is preserved. The sign is read from
    // the ANCHORED (linking) items only — a free item's DIF outlier must not
    // drive the global orientation (review fix; falls back to all items only
    // when a block has no anchored member). General flip negates every
    // group's general slopes, every group mean, and the reported general
    // EAPs (the #1879 mu-sign fix); thresholds, variances, and posterior SDs
    // are invariant. Specific flips negate that block's slopes in every group.
    let anchored_items: Vec<usize> = (0..n_items).filter(|&i| anchor_vec[i]).collect();
    let anchor_g = anchored_items
        .iter()
        .flat_map(|&i| (0..n_groups).map(move |g| (g, i)))
        .max_by(|&(g1, i1), &(g2, i2)| {
            a_general[g1][i1]
                .abs()
                .total_cmp(&a_general[g2][i2].abs())
        })
        .expect("at least one anchored item");
    if a_general[anchor_g.0][anchor_g.1] < 0.0 {
        for row in a_general.iter_mut() {
            for a in row.iter_mut() {
                *a = -*a;
            }
        }
        for m in general_mean.iter_mut() {
            *m = -*m;
        }
        for t in theta_g_eap.iter_mut() {
            *t = -*t;
        }
    }
    for members in v.blocks.iter() {
        let anchored_members: Vec<usize> =
            members.iter().copied().filter(|&i| anchor_vec[i]).collect();
        // Prefer anchored members; fall back to the whole block only when it
        // has no anchored item.
        let candidates: &[usize] = if anchored_members.is_empty() {
            members
        } else {
            &anchored_members
        };
        let anchor = (0..n_groups)
            .flat_map(|g| candidates.iter().map(move |&i| (g, i)))
            .max_by(|&(g1, i1), &(g2, i2)| {
                a_specific[g1][i1]
                    .abs()
                    .total_cmp(&a_specific[g2][i2].abs())
            })
            .expect("validated blocks are non-empty");
        if a_specific[anchor.0][anchor.1] < 0.0 {
            for g in 0..n_groups {
                for &i in members {
                    a_specific[g][i] = -a_specific[g][i];
                }
            }
        }
    }

    let mut group_category_counts = vec![vec![0usize; n_items * n_cat]; n_groups];
    for p in 0..n_persons {
        let g = group_id[p];
        for i in 0..n_items {
            if is_obs(p, i) {
                group_category_counts[g][i * n_cat + y[p * n_items + i]] += 1;
            }
        }
    }

    let mut n_parameters = 0usize;
    for i in 0..n_items {
        let per_item = 1 + v.item_block[i].is_some() as usize + v.m1;
        if anchor_vec[i] {
            n_parameters += per_item;
        } else {
            n_parameters += n_groups * per_item;
        }
    }
    n_parameters += (n_groups - 1) * 2;
    if cfg.estimate_specific_vars {
        n_parameters += (n_groups - 1) * v.n_specific;
    }

    Ok(BifactorMultigroupResult {
        a_general,
        a_specific,
        threshold,
        general_mean,
        general_sd,
        specific_sd,
        theta_g_eap,
        theta_g_sd,
        group_category_counts,
        loglik_trace: outcome.loglik_trace,
        n_iter: outcome.n_iter,
        converged: outcome.converged,
        termination_reason: outcome.termination_reason,
        final_loglik_change: outcome.final_loglik_change,
        best_start,
        n_parameters,
    })
}

// ---------------------------------------------------------------------------
// Fixed-item parameter calibration (FIPC) for the focal group (Kim, 2006,
// MWU-MEM).
//
// A focal group responds to `anchor` items whose general/specific slopes and
// boundary intercepts are FIXED at reference-calibration values plus new
// (free) items. The fitter estimates the free-item parameters AND the focal
// latent distribution — the general-factor mean/variance and, where
// identified, the specific-factor variances — by Bock-Aitkin MML-EM with the
// prior updated after EVERY M-step (Kim, 2006, MWU-MEM eqs. 14-15,
// pp. 361-362; workflow Table 1, p. 362). There is no rescaling of the latent
// points after an EM cycle (Kim, 2006, p. 362) and no reflection
// canonicalization: the fixed anchors pin the orientation, including
// reverse-keyed anchors with negative slopes.
//
// # Implementation basis (APA 7th ed.; every locator verified against the
// # cited source — no invented equation numbers)
//
// - Kim, S. (2006). A comparative study of IRT fixed parameter calibration
//   methods. *Journal of Educational Measurement, 43*(4), 355-381.
//   https://doi.org/10.1111/j.1745-3984.2006.00021.x — the MWU-MEM
//   (multiple-weights updating / multiple EM) procedure: M-step item
//   objective eq. 14 (pp. 361-362), posterior weight update eq. 15 (p. 362),
//   the EM workflow Table 1 (p. 362), the no-rescale rule ("the ability
//   points should not be rescaled after each EM cycle", p. 362), the fixed `N(0, 1)` person prior for the
//   FPC baseline (p. 364), the mean-shift / variance-shift study conditions
//   (pp. 364-365), and the recommendation to update the prior iteratively
//   rather than fix it (pp. 377-378).
// - Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
//   in a fixed-item linking procedure with a fixed-person prior distribution
//   for mixed-format test data. *Applied Measurement in Education, 18*(2),
//   199-215. https://doi.org/10.1207/s15324818ame1802_4 — a fixed person
//   prior biases the focal growth/shift estimate, while iteratively updating
//   the prior recovers it (abstract; cf. Kim, 2006, p. 378).
// - Gibbons et al. (2007), eq. 9 (bifactor graded linear predictor) and eq.
//   15 (person marginal factoring per general node), plus Gibbons & Hedeker
//   (1992) for the bifactor EM with Stuart's reduction — the same reduction
//   this module's E-step already implements (see the module docs).
// - Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J. van
//   der Linden & R. K. Hambleton (Eds.), *Handbook of modern item response
//   theory* (pp. 433-448). Springer.
//   https://doi.org/10.1007/978-1-4757-2691-6_25 — the node-shift
//   distribution form `theta_{G,t} = mu + sigma * X_t`,
//   `theta_{S,s,h} = tau_s * X_h` with the shared Gauss-Hermite weights kept
//   (conceptual; no equation locator claimed — it is exactly the pooling this
//   crate's multigroup stage-2 already uses).
//
// The parametric-normal prior update (`mu`, `sigma`, and, when requested,
// `tau_s` from E-step posterior moments) realizes Kim's discrete weight
// update (eq. 15) inside the Bock-Zimowski parametric family: same
// update-the-prior-after-every-M-step semantics, normal-family form.
//
// # Caller-owned numerics (no hidden clamps, no new constants)
//
// Every numeric choice is a caller argument or mirrors an existing one:
// `q_general`, `q_specific`, `max_iter`, `tol`, `newton_iter`, `ridge` behave
// exactly like the stage-1/stage-2 controls (same loud-`Err` validation, same
// Newton M-step); starts mirror `initial_params` start 0 (`a_G = 1.0`,
// `a_S = 0.8`, cumulative-logit base-rate intercepts, `mu = 0`, `sigma = 1`,
// `tau = 1`) deterministically with NO jitter and a single start; anchors pin
// orientation so no canonicalization runs. Group variances are estimated
// unconstrained-positive with NO clamping: a non-finite or non-positive
// update is a loud `Err`, mirroring the stage-2 focal update.
//
// # References (APA 7th ed.)
//
// Kim, S. (2006). A comparative study of IRT fixed parameter calibration
// methods. *Journal of Educational Measurement, 43*(4), 355-381.
// https://doi.org/10.1111/j.1745-3984.2006.00021.x
//
// Paek, I., & Young, M. J. (2005). Investigation of student growth recovery
// in a fixed-item linking procedure with a fixed-person prior distribution
// for mixed-format test data. *Applied Measurement in Education, 18*(2),
// 199-215. https://doi.org/10.1207/s15324818ame1802_4
//
// Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
// D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
// Full-information item bifactor analysis of graded response data. *Applied
// Psychological Measurement, 31*(1), 4-19.
// https://doi.org/10.1177/0146621606289485
//
// Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
// analysis. *Psychometrika, 57*(3), 423-436.
// https://doi.org/10.1007/BF02295430
//
// Bock, R. D., & Zimowski, M. F. (1997). Multiple group IRT. In W. J.
// van der Linden & R. K. Hambleton (Eds.), *Handbook of modern item response
// theory* (pp. 433-448). Springer. https://doi.org/10.1007/978-1-4757-2691-6_25

/// Configuration for [`fit_bifactor_grm_fipc`]. Every field is caller-owned
/// and range-validated; nothing is clamped. `newton_iter`/`ridge` behave like
/// the stage-1/stage-2 controls.
#[derive(Clone, Copy, Debug)]
pub struct BifactorFipcConfig {
    /// Gauss-Hermite nodes for the general factor (one of `SUPPORTED_Q`).
    pub q_general: usize,
    /// Gauss-Hermite nodes per specific factor (one of `SUPPORTED_Q`).
    pub q_specific: usize,
    pub max_iter: usize,
    pub tol: f64,
    /// Inner Newton iterations per free-item M-step.
    pub newton_iter: usize,
    /// Newton ridge — Hessian CONDITIONING only, NOT a parameter prior.
    pub ridge: f64,
    /// Estimate focal specific-factor variances (`true`) or fix at 1 (`false`).
    pub estimate_specific_vars: bool,
}

impl Default for BifactorFipcConfig {
    fn default() -> Self {
        Self {
            q_general: 21,
            q_specific: 11,
            max_iter: 500,
            tol: 1e-6,
            newton_iter: 10,
            ridge: 1e-8,
            estimate_specific_vars: false,
        }
    }
}

/// Result of [`fit_bifactor_grm_fipc`].
///
/// Anchor items are bit-identical to the fixed inputs. `general_mean` /
/// `general_sd` are the ESTIMATED focal general-factor mean/SD;
/// `specific_sd` holds the estimated focal specific-factor SDs when
/// `estimate_specific_vars` and 1.0 otherwise. `theta_g_eap`/`theta_g_sd`
/// are per-person general-factor EAPs on the FOCAL scale.
#[derive(Clone, Debug)]
pub struct BifactorFipcResult {
    /// General slopes `a_iG`, length `n_items` (anchors bit-exact, signs kept).
    pub a_general: Vec<f64>,
    /// Specific slopes `a_iS`, length `n_items` (`0.0` for general-only
    /// items; anchors bit-exact, signs kept).
    pub a_specific: Vec<f64>,
    /// Ordered boundary intercepts `d_ik`, row-major `n_items * (n_cat - 1)`
    /// (anchor rows bit-exact).
    pub threshold: Vec<f64>,
    /// Estimated focal general-factor mean.
    pub general_mean: f64,
    /// Estimated focal general-factor SD.
    pub general_sd: f64,
    /// Focal specific-factor SDs, length `n_specific` (1.0 unless estimated).
    pub specific_sd: Vec<f64>,
    /// General-factor EAP `E[theta_G | Y_p]` on the focal scale.
    pub theta_g_eap: Vec<f64>,
    /// General-factor posterior SD, length `n_persons`.
    pub theta_g_sd: Vec<f64>,
    /// Observed-data category counts, row-major `n_items * n_cat`.
    pub category_counts: Vec<usize>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: f64,
    /// Free item parameters (anchors are fixed, counted zero) plus estimated
    /// focal distribution parameters (general mean/variance, and the specific
    /// variances when estimated).
    pub n_parameters: usize,
}

/// Fit the focal group with fixed anchor items (Kim, 2006, MWU-MEM) for the
/// polytomous bifactor GRM.
///
/// `y`/`observed` are row-major `n_persons * n_items` focal responses
/// (`y` ordered categories `0..n_cat-1`, missing cells dropped MAR);
/// `specific_map` is length `n_items` with `-1` for general-only items and
/// `0..n_specific` otherwise; `anchor[i] == true` pins item `i` at
/// (`fixed_a_general[i]`, `fixed_a_specific[i]`, `fixed_threshold` row `i`)
/// for ALL EM cycles while the remaining items are freely estimated. Slopes
/// may be NEGATIVE (reverse-keyed anchors keep their signs bit-exact: no
/// reflection canonicalization runs). The focal general mean/variance — plus
/// the focal specific variances iff `cfg.estimate_specific_vars` — are
/// re-estimated after EVERY M-step from E-step posterior moments (Kim, 2006,
/// eqs. 14-15, pp. 361-362), with NO rescaling of the latent points (Kim,
/// 2006, p. 362).
///
/// Returns `Err` on malformed input (shapes and config exactly like the
/// existing fitters; at least one anchor required; fixed arrays finite with
/// correct lengths; `fixed_a_specific[i]` exactly `0.0` for general-only
/// items, mirroring `bifactor_grm_marginal_loglik`; anchor thresholds
/// strictly decreasing per item), unobserved categories, or numerical
/// failure. Non-convergence reports `converged == false` with
/// `termination_reason == "max_iter_reached"`, never a substitute.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::needless_range_loop)] // item/node indexing is inherently indexed
pub fn fit_bifactor_grm_fipc(
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    anchor: &[bool],
    fixed_a_general: &[f64],
    fixed_a_specific: &[f64],
    fixed_threshold: &[f64],
    cfg: &BifactorFipcConfig,
) -> Result<BifactorFipcResult, String> {
    // Config checks mirror `validate_multigroup_cfg` exactly (quadrature must
    // name an embedded rule; lower bounds only; caller-owned numerics behave
    // identically).
    if !SUPPORTED_Q.contains(&cfg.q_general) {
        return Err(format!(
            "q_general must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_general
        ));
    }
    if !SUPPORTED_Q.contains(&cfg.q_specific) {
        return Err(format!(
            "q_specific must be one of {SUPPORTED_Q:?}; got {}",
            cfg.q_specific
        ));
    }
    if cfg.max_iter < 1 {
        return Err("max_iter must be >= 1".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if cfg.newton_iter < 1 {
        return Err("newton_iter must be >= 1".into());
    }
    if !cfg.ridge.is_finite() || cfg.ridge <= 0.0 {
        return Err("ridge must be finite and positive".into());
    }
    // Base structural validation (shapes, blocks, pooled categories, checked
    // arithmetic) reuses the single-group validator so FIPC accepts exactly
    // the same data layouts as the existing fitters.
    let single_cfg = BifactorGrmConfig {
        q_general: cfg.q_general,
        q_specific: cfg.q_specific,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_starts: 1,
        seed: 0x9E37_79B9_7F4A_7C15,
        newton_iter: cfg.newton_iter,
        ridge: cfg.ridge,
    };
    let v = validate(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &single_cfg,
    )?;
    if anchor.len() != n_items {
        return Err("anchor must have length n_items".into());
    }
    if !anchor.iter().any(|&a| a) {
        return Err("at least one anchored (fixed) item is required to identify the focal scale".into());
    }
    if fixed_a_general.len() != n_items || fixed_a_specific.len() != n_items {
        return Err("fixed_a_general/fixed_a_specific must have length n_items".into());
    }
    if fixed_threshold.len() != n_items * v.m1 {
        return Err("fixed_threshold must have length n_items * (n_cat - 1)".into());
    }
    if [fixed_a_general, fixed_a_specific, fixed_threshold]
        .concat()
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err("fixed anchor parameters must be finite".into());
    }
    for (i, block) in v.item_block.iter().enumerate() {
        if block.is_none() && fixed_a_specific[i] != 0.0 {
            return Err(format!(
                "fixed_a_specific[{i}] must be exactly 0.0 for general-only items \
                 (specific_map[{i}] == -1); got {}",
                fixed_a_specific[i]
            ));
        }
    }
    for (i, chunk) in fixed_threshold.chunks_exact(v.m1).enumerate() {
        if anchor[i] && chunk.windows(2).any(|w| w[0] <= w[1]) {
            return Err(format!(
                "fixed thresholds of anchor item {i} must be strictly decreasing"
            ));
        }
    }

    // Init mirrors `initial_params` start 0 deterministically with NO jitter
    // and a single start: anchors at their fixed values; free items at
    // `a_G = 1.0` / `a_S = 0.8` with cumulative-logit base-rate intercepts
    // from the focal data. Focal distribution starts at the reference
    // (`mu = 0`, `sigma = 1`, `tau = 1`).
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    let mut params: Vec<ItemParams> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        if anchor[i] {
            params.push(ItemParams {
                a_g: fixed_a_general[i],
                a_s: v.item_block[i].map(|_| fixed_a_specific[i]),
                d: fixed_threshold[i * v.m1..(i + 1) * v.m1].to_vec(),
            });
        } else {
            let mut freq = vec![1e-3f64; v.n_cat];
            for p in 0..n_persons {
                if is_obs(p, i) {
                    freq[y[p * n_items + i]] += 1.0;
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
            params.push(ItemParams {
                a_g: 1.0,
                a_s: v.item_block[i].map(|_| 0.8),
                d,
            });
        }
    }
    let mut mu = 0.0f64;
    let mut sigma = 1.0f64;
    let mut taus = vec![1.0f64; v.n_specific];

    let (tg_std, wg) = gh_rule(cfg.q_general)?;
    let (ts_std, ws) = gh_rule(cfg.q_specific)?;
    let qg = tg_std.len();
    let qs = ts_std.len();
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    // Single focal group: every person belongs to group 0.
    let group_id = vec![0usize; n_persons];

    // EM loop mirrors `run_single_start_multigroup` with `n_groups == 1`:
    // focal nodes `tg[t] = mu + sigma * tg_std[t]`,
    // `ts[s][h] = tau[s] * ts_std[h]`; the EXISTING `e_step_multigroup`
    // sweep is reused directly; convergence via `checked_em_loglik_change`
    // with the same relative-tolerance rule.
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;

    loop {
        let tg: Vec<f64> = tg_std.iter().map(|&x| mu + sigma * x).collect();
        let ts_focal: Vec<Vec<f64>> = (0..v.n_specific)
            .map(|s| ts_std.iter().map(|&x| taus[s] * x).collect())
            .collect();
        let tg_groups = vec![tg.clone()];
        let ts_groups = vec![ts_focal.clone()];
        let (ll, counts, w_acc, s1_g, s2_g, s2_spec, w_spec) = e_step_multigroup(
            &v,
            y,
            observed,
            &group_id,
            1,
            std::slice::from_ref(&params),
            &tg_groups,
            &ts_groups,
            &log_wg,
            &log_ws,
            qg,
            qs,
        );
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
        // M-step, item parameters: ONLY free items via the existing
        // `m_step_item`, with node coordinates built exactly like the
        // free-item branch of `run_single_start_multigroup` for `g = 0`
        // (stacked `(t, h)` with `node = t * qs + h` for block items;
        // `tg_groups[0]` for general-only items). Anchors stay bit-exact.
        for i in 0..n_items {
            if anchor[i] {
                continue;
            }
            let has_specific = v.item_block[i].is_some();
            let (node_g, node_s): (Vec<f64>, Vec<f64>) = if has_specific {
                let s = v.item_block[i].expect("block item has a block");
                let mut gg = Vec::with_capacity(qg * qs);
                let mut ss = Vec::with_capacity(qg * qs);
                for t in 0..qg {
                    for h in 0..qs {
                        gg.push(tg[t]);
                        ss.push(ts_focal[s][h]);
                    }
                }
                (gg, ss)
            } else {
                (tg.clone(), vec![0.0; qg])
            };
            let mut packed = Vec::with_capacity(1 + has_specific as usize + v.m1);
            packed.push(params[i].a_g);
            if let Some(a_s) = params[i].a_s {
                packed.push(a_s);
            }
            packed.extend_from_slice(&params[i].d);
            let updated = m_step_item(
                packed,
                has_specific,
                &node_g,
                &node_s,
                &counts[0][i],
                v.n_cat,
                cfg.ridge,
                cfg.newton_iter,
            );
            params[i].a_g = updated[0];
            if has_specific {
                params[i].a_s = Some(updated[1]);
                params[i].d = updated[2..].to_vec();
            } else {
                params[i].d = updated[1..].to_vec();
            }
        }
        // M-step, focal distribution (ESTIMATED, not pinned): general
        // `mu = mean EAP`, `var = mean posterior second moment - mu^2`
        // (Bock-Aitkin/Bock-Zimowski moment update); specifics (when
        // estimated) `tau^2 = mean posterior second moment at zero mean`.
        // Mirrors the stage-2 focal update including the loud-`Err` (no
        // clamping) style.
        if w_acc[0] <= 0.0 || !w_acc[0].is_finite() {
            return Err("focal group has no posterior mass".into());
        }
        let mean = s1_g[0] / w_acc[0];
        let var = s2_g[0] / w_acc[0] - mean * mean;
        if !mean.is_finite() || !var.is_finite() {
            return Err("non-finite focal general moment update".into());
        }
        if var <= 0.0 {
            return Err(format!(
                "non-positive focal general variance update ({var:.6e})"
            ));
        }
        mu = mean;
        sigma = var.sqrt();
        if cfg.estimate_specific_vars {
            for s in 0..v.n_specific {
                if w_spec[0][s] <= 0.0 || !w_spec[0][s].is_finite() {
                    return Err(format!("focal specific-{s} has no posterior mass"));
                }
                let vrow = s2_spec[0][s] / w_spec[0][s];
                if !vrow.is_finite() {
                    return Err(format!("non-finite focal specific-{s} update"));
                }
                if vrow <= 0.0 {
                    return Err(format!(
                        "non-positive focal specific-{s} variance update ({vrow:.6e})"
                    ));
                }
                taus[s] = vrow.sqrt();
            }
        }
        n_iter += 1;
    }

    // Final EAP pass for theta_G on the FOCAL scale at the winning parameters
    // (mirrors `fit_bifactor_grm`'s final EAP loop, but with general nodes
    // `tg_focal[t] = mu + sigma * tg_std[t]` and specific nodes
    // `tau[s] * ts_std[h]`; the weights stay the standard ones — the
    // node-shift reparameterization keeps the weights, cf. Bock & Zimowski,
    // 1997).
    let tg_focal: Vec<f64> = tg_std.iter().map(|&x| mu + sigma * x).collect();
    let ts_focal: Vec<Vec<f64>> = (0..v.n_specific)
        .map(|s| ts_std.iter().map(|&x| taus[s] * x).collect())
        .collect();
    let mut tables: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for (i, par) in params.iter().enumerate() {
        match par.a_s {
            Some(a_s) => {
                let s = v.item_block[i].expect("block item has a block");
                let mut lp = vec![0.0f64; qg * qs * v.n_cat];
                for t in 0..qg {
                    for h in 0..qs {
                        let base = par.a_g * tg_focal[t] + a_s * ts_focal[s][h];
                        let probs = grm_logprobs(base, &par.d);
                        lp[(t * qs + h) * v.n_cat..(t * qs + h + 1) * v.n_cat]
                            .copy_from_slice(&probs);
                    }
                }
                tables.push(lp);
            }
            None => {
                let mut lp = vec![0.0f64; qg * v.n_cat];
                for t in 0..qg {
                    let base = par.a_g * tg_focal[t];
                    let probs = grm_logprobs(base, &par.d);
                    lp[t * v.n_cat..(t + 1) * v.n_cat].copy_from_slice(&probs);
                }
                tables.push(lp);
            }
        }
    }
    let mut theta_g_eap = vec![0.0f64; n_persons];
    let mut theta_g_sd = vec![0.0f64; n_persons];
    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];
    for p in 0..n_persons {
        let mut gen_log = log_wg.clone();
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * n_items + i];
            for t in 0..qg {
                gen_log[t] += tables[i][t * n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for t in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * n_items + i];
                        acc += tables[i][(t * qs + h) * n_cat + yc];
                    }
                    block_acc[(s * qg + t) * qs + h] = acc;
                }
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + t) * qs + h];
                }
                log_i[s * qg + t] = log_sum_exp(&tmp_h);
            }
        }
        for t in 0..qg {
            let mut acc = gen_log[t];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + t];
            }
            log_like_g[t] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        let (mut m1, mut m2) = (0.0f64, 0.0f64);
        for (t, &ll) in log_like_g.iter().enumerate() {
            let post = (ll - log_lp).exp();
            let theta = tg_focal[t];
            m1 += post * theta;
            m2 += post * theta * theta;
        }
        theta_g_eap[p] = m1;
        theta_g_sd[p] = (m2 - m1 * m1).max(0.0).sqrt();
    }

    // Assemble dense outputs. NO reflection canonicalization and NO rescaling
    // of the nodes: the fixed anchors pin the orientation (Kim, 2006, p. 362).
    let mut a_general = vec![0.0f64; n_items];
    let mut a_specific = vec![0.0f64; n_items];
    let mut threshold = vec![0.0f64; n_items * v.m1];
    for (i, par) in params.iter().enumerate() {
        a_general[i] = par.a_g;
        if let Some(a_s) = par.a_s {
            a_specific[i] = a_s;
        }
        threshold[i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
    }

    let mut category_counts = vec![0usize; n_items * n_cat];
    for p in 0..n_persons {
        for i in 0..n_items {
            if is_obs(p, i) {
                category_counts[i * n_cat + y[p * n_items + i]] += 1;
            }
        }
    }

    // Free item parameters (anchors are fixed, counted zero) plus estimated
    // focal distribution parameters — mirrors the multigroup/stage-1
    // counting.
    let mut n_parameters = 0usize;
    for i in 0..n_items {
        if !anchor[i] {
            n_parameters += 1 + v.item_block[i].is_some() as usize + v.m1;
        }
    }
    n_parameters += 2;
    if cfg.estimate_specific_vars {
        n_parameters += v.n_specific;
    }

    Ok(BifactorFipcResult {
        a_general,
        a_specific,
        threshold,
        general_mean: mu,
        general_sd: sigma,
        specific_sd: taus,
        theta_g_eap,
        theta_g_sd,
        category_counts,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        n_parameters,
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/bifactor_grm_tests.rs"]
mod tests;
