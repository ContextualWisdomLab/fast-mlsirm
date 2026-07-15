//! Mixed Rasch / mixture IRT (Rost, 1990; Rost & von Davier, 1995): the population
//! is a mixture of `C` latent classes, each with its OWN item parameters and a mixing
//! weight `pi_c`. Within class `c`, responses follow a unidimensional IRT model
//! (`Rasch` fixes `a_ic = 1`; `TwoPl` frees `a_ic`) with ability `theta ~ N(0,1)`
//! fixed per class, estimated by marginal-ML EM over the shared Gauss-Hermite rule.
//! This detects unobserved population heterogeneity — qualitatively different response
//! strategies that a single-class model cannot represent.
//!
//! The E-step forms the joint posterior over (class `c`, ability node `q`); the M-step
//! updates each class's item parameters with the SAME per-item Newton step as
//! [`crate::mmle::fit_mmle_2pl`] (weighted by the class posterior), plus the mixing
//! proportions `pi_c = mean posterior class membership`. The `C = 1` `TwoPl` case at
//! `tol = 0.0` reduces bit-exactly to `fit_mmle_2pl` (the reduction anchor); the
//! `tol = 0.0` is what makes both run the full `max_iter` despite their differing
//! convergence-check placement.
//!
//! Identification. Two problems, both handled: (1) the within-class metric is pinned
//! by `theta_c ~ N(0,1)` with all `b_ic` (and `a_ic`) free — the same standard-normal
//! prior `fit_mmle_2pl` assumes, so every `b_ic` is directly comparable across classes
//! on one metric. (2) Label switching: classes are exchangeable, so the OUTPUT is put
//! in a canonical order (mixing weight descending, ties broken by mean difficulty
//! ascending); recovery studies must additionally match classes by permutation.
//!
//! Provenance. Rost's (1990) original estimator used conditional ML within class with
//! a saturated raw-score distribution; `psychomix` likewise fits Rasch mixtures by
//! conditional ML (Frick, Strobl, Leisch, & Zeileis, 2012). This crate instead combines
//! Rost's latent-class structure with a fixed-standard-normal, Bock-Aitkin marginal-ML
//! EM estimator. That estimator is a repository-specific operationalization, not an
//! assertion that its finite-sample item estimates equal the conditional-ML estimates.
//!
//! Deferred (explicit non-goals): free per-class ability variance `sigma_c`; automatic
//! model selection over `C` (the result returns `n_parameters`/`loglik_trace`, so
//! AIC/BIC/ICL are caller one-liners); concomitant-variable (covariate) mixing; GPU
//! offload of the E-step.
//!
//! References (APA 7th ed.):
//! - Rost, J. (1990). Rasch models in latent classes: An integration of two approaches
//!   to item analysis. *Applied Psychological Measurement, 14*(3), 271–282.
//!   <https://doi.org/10.1177/014662169001400305>
//! - Rost, J., & von Davier, M. (1995). Mixture distribution Rasch models. In G. H.
//!   Fischer & I. W. Molenaar (Eds.), *Rasch models: Foundations, recent developments,
//!   and applications* (pp. 257–268). Springer.
//!   <https://doi.org/10.1007/978-1-4612-4230-7_14>
//! - Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
//!   parameters. *Psychometrika, 46*(4), 443–459. <https://doi.org/10.1007/BF02293801>
//! - Frick, H., Strobl, C., Leisch, F., & Zeileis, A. (2012). Flexible Rasch mixture
//!   models with package psychomix. *Journal of Statistical Software, 48*(7), 1–25.
//!   <https://doi.org/10.18637/jss.v048.i07>
//! - McLachlan, G. J., & Peel, D. (2000). *Finite mixture models*. Wiley.

use crate::mmle::{fit_mmle_2pl, log_sigmoid, sigmoid_stable, MmleConfig, GH_NODES, GH_WEIGHTS};

/// Within-class IRT model: `Rasch` fixes `a_ic = 1` (Rost, 1990); `TwoPl` frees `a_ic`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixtureModel {
    Rasch,
    TwoPl,
}

/// EM configuration for the mixture-IRT estimator.
#[derive(Clone, Copy, Debug)]
pub struct MixtureConfig {
    pub max_iter: usize,
    /// Convergence tolerance on `|delta loglik|`. `0.0` is permitted (runs the full
    /// `max_iter`) — required for the bit-exact `C = 1` reduction anchor.
    pub tol: f64,
    pub ridge_a: f64,
    pub ridge_b: f64,
    pub newton_iter: usize,
    /// Random restarts for the multimodal mixture likelihood (start 0 is the
    /// deterministic warm start). Kept the run with the highest final loglik.
    pub n_starts: usize,
    /// Class separation for the warm start / perturbation scale for restarts.
    pub start_spread: f64,
    /// Floor for `pi_c` (avoids `ln 0`).
    pub pi_floor: f64,
    /// Seed for the restart perturbations (unused when `n_starts == 1`).
    pub seed: u64,
}

impl Default for MixtureConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            ridge_a: 1e-3,
            ridge_b: 1e-3,
            newton_iter: 25,
            n_starts: 1,
            start_spread: 1.0,
            pi_floor: 1e-6,
            seed: 0x2545F491,
        }
    }
}

/// Fitted mixture-IRT model. Item parameters are class-major (`a[c*J + i]`). Classes
/// are in canonical order (mixing weight descending, ties by mean difficulty ascending).
#[derive(Clone, Debug)]
pub struct MixtureResult {
    pub model: MixtureModel,
    pub n_classes: usize,
    /// Per-class item discriminations, class-major `a[c*J + i]` (all 1.0 for `Rasch`).
    pub a: Vec<f64>,
    /// Per-class item difficulties, class-major `b[c*J + i]`.
    pub b: Vec<f64>,
    /// Mixing proportions, length `C`, sum 1.
    pub pi: Vec<f64>,
    /// Class responsibilities `P(class c | x_j)`, row-major `N x C`.
    pub class_posterior: Vec<f64>,
    /// MAP class per person, length `N`.
    pub map_class: Vec<u32>,
    /// Mixture EAP ability per person, length `N`.
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `C*(k*J) + (C-1)`, `k = 2` (TwoPl) | `1` (Rasch).
    pub n_parameters: usize,
}

fn validate(
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    n_classes: usize,
    cfg: &MixtureConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if n_classes < 1 {
        return Err("n_classes must be >= 1".into());
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    // tol == 0.0 is allowed (runs the full max_iter; needed for the C=1 anchor).
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err("tol must be finite and non-negative".into());
    }
    if cfg.newton_iter == 0 {
        return Err("newton_iter must be positive".into());
    }
    if cfg.n_starts == 0 {
        return Err("n_starts must be positive".into());
    }
    if !cfg.ridge_a.is_finite() || cfg.ridge_a < 0.0 {
        return Err("ridge_a must be finite and non-negative".into());
    }
    if !cfg.ridge_b.is_finite() || cfg.ridge_b < 0.0 {
        return Err("ridge_b must be finite and non-negative".into());
    }
    if !cfg.start_spread.is_finite() || cfg.start_spread < 0.0 {
        return Err("start_spread must be finite and non-negative".into());
    }
    if !cfg.pi_floor.is_finite() || !(0.0 < cfg.pi_floor && cfg.pi_floor < 1.0 / n_classes as f64) {
        return Err("pi_floor must be finite and in (0, 1/n_classes)".into());
    }
    if n_classes > u32::MAX as usize {
        return Err("n_classes must fit in the u32 map_class representation".into());
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    let class_items = n_classes
        .checked_mul(n_items)
        .ok_or_else(|| "n_classes * n_items overflows usize".to_string())?;
    n_classes
        .checked_mul(GH_NODES.len())
        .ok_or_else(|| "n_classes * quadrature_nodes overflows usize".to_string())?;
    class_items
        .checked_mul(GH_NODES.len())
        .ok_or_else(|| "n_classes * n_items * quadrature_nodes overflows usize".to_string())?;
    n_persons
        .checked_mul(n_classes)
        .ok_or_else(|| "n_persons * n_classes overflows usize".to_string())?;
    class_items
        .checked_mul(2)
        .and_then(|n| n.checked_add(n_classes - 1))
        .ok_or_else(|| "mixture parameter count overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && v != 0.0 && v != 1.0 {
            return Err(format!("y[{idx}] must be 0 or 1 where observed; got {v}"));
        }
    }
    for i in 0..n_items {
        if !(0..n_persons).any(|p| observed[p * n_items + i]) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    Ok(())
}

/// One penalized Newton item update, shared with (and bit-identical to) the per-item
/// step of [`crate::mmle::fit_mmle_2pl`]. `n_row`/`r_row` are the item's expected node
/// counts (length Q). `fix_slope` holds `a = a0 = 1` (Rasch). The `C = 1` reduction
/// anchor enforces that this stays in lockstep with `fit_mmle_2pl`.
fn newton_item_2pl(
    n_row: &[f64],
    r_row: &[f64],
    a0: f64,
    b0: f64,
    fix_slope: bool,
    newton_iter: usize,
    ridge_a: f64,
    ridge_b: f64,
) -> (f64, f64) {
    let (mut ai, mut bi) = (a0, b0);
    for _ in 0..newton_iter {
        let (mut g_a, mut g_b, mut h_aa, mut h_bb, mut h_ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (qi, &node) in GH_NODES.iter().enumerate() {
            let p_correct = sigmoid_stable(ai * node + bi);
            let n = n_row[qi];
            let w = n * p_correct * (1.0 - p_correct);
            let resid = r_row[qi] - n * p_correct;
            g_a += resid * node;
            g_b += resid;
            h_aa -= w * node * node;
            h_bb -= w;
            h_ab -= w * node;
        }
        if fix_slope {
            // 1-D Newton on b with a held at 1 (Rasch): b -= g_b / h_bb.
            g_b -= ridge_b * bi;
            h_bb -= ridge_b;
            if h_bb.abs() < 1e-12 {
                break;
            }
            let db = g_b / h_bb;
            bi -= db;
            if db.abs() < 1e-8 {
                break;
            }
        } else {
            g_a -= ridge_a * ai;
            g_b -= ridge_b * bi;
            h_aa -= ridge_a;
            h_bb -= ridge_b;
            let det = h_aa * h_bb - h_ab * h_ab;
            if det.abs() < 1e-12 {
                break;
            }
            let da = (h_bb * g_a - h_ab * g_b) / det;
            let db = (h_aa * g_b - h_ab * g_a) / det;
            ai = (ai - da).clamp(1e-3, 10.0);
            bi -= db;
            if da.abs() + db.abs() < 1e-8 {
                break;
            }
        }
    }
    (ai, bi)
}

/// Marginal-ML item-proportion init identical to `fit_mmle_2pl` (a = 1, b = logit of
/// the clamped item proportion). Load-bearing for the bit-exact C=1 anchor.
fn init_mmle_like(y: &[f64], observed: &[bool], n_persons: usize, n_items: usize) -> Vec<f64> {
    let mut b = vec![0.0f64; n_items];
    for i in 0..n_items {
        let (mut num, mut den) = (0.0, 0.0);
        for p in 0..n_persons {
            let idx = p * n_items + i;
            if observed[idx] {
                num += y[idx];
                den += 1.0;
            }
        }
        let prop = if den > 0.0 { (num / den).clamp(0.02, 0.98) } else { 0.5 };
        b[i] = (prop / (1.0 - prop)).ln();
    }
    b
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Run marginal EM from a given init to convergence; classes returned UNORDERED.
#[allow(clippy::too_many_arguments)]
fn run_em(
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    n_classes: usize,
    model: MixtureModel,
    mut a: Vec<f64>,
    mut b: Vec<f64>,
    mut pi: Vec<f64>,
    cfg: &MixtureConfig,
) -> MixtureResult {
    let q = GH_NODES.len();
    let log_w: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();
    let fix_slope = model == MixtureModel::Rasch;
    let cq = n_classes * q;

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut post = vec![0.0f64; cq];

    let build_tables = |a: &[f64], b: &[f64], log_p1: &mut [f64], log_p0: &mut [f64]| {
        for c in 0..n_classes {
            for (qi, &node) in GH_NODES.iter().enumerate() {
                for i in 0..n_items {
                    let eta = a[c * n_items + i] * node + b[c * n_items + i];
                    log_p1[(c * q + qi) * n_items + i] = log_sigmoid(eta);
                    log_p0[(c * q + qi) * n_items + i] = log_sigmoid(-eta);
                }
            }
        }
    };
    // Fill `post` with the joint (class, node) posterior for person j; returns ln P(x_j).
    let person_posterior = |j: usize,
                            log_p1: &[f64],
                            log_p0: &[f64],
                            log_pi: &[f64],
                            post: &mut [f64]|
     -> f64 {
        for c in 0..n_classes {
            for qi in 0..q {
                let mut acc = log_pi[c] + log_w[qi];
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let yy = y[idx];
                        acc += yy * log_p1[(c * q + qi) * n_items + i]
                            + (1.0 - yy) * log_p0[(c * q + qi) * n_items + i];
                    }
                }
                post[c * q + qi] = acc;
            }
        }
        let m = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0;
        for v in post.iter() {
            denom += (v - m).exp();
        }
        for v in post.iter_mut() {
            *v = (*v - m).exp() / denom;
        }
        m + denom.ln()
    };

    let mut log_p1 = vec![0.0f64; cq * n_items];
    let mut log_p0 = vec![0.0f64; cq * n_items];

    for _ in 0..cfg.max_iter {
        build_tables(&a, &b, &mut log_p1, &mut log_p0);
        let log_pi: Vec<f64> = pi.iter().map(|p| p.ln()).collect();

        let mut n_cnt = vec![0.0f64; n_classes * n_items * q];
        let mut r_cnt = vec![0.0f64; n_classes * n_items * q];
        let mut pi_new = vec![0.0f64; n_classes];
        let mut total_ll = 0.0;
        for j in 0..n_persons {
            total_ll += person_posterior(j, &log_p1, &log_p0, &log_pi, &mut post);
            for c in 0..n_classes {
                let mut r_jc = 0.0;
                for qi in 0..q {
                    r_jc += post[c * q + qi];
                }
                pi_new[c] += r_jc;
            }
            for c in 0..n_classes {
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let yy = y[idx];
                        let base = (c * n_items + i) * q;
                        for qi in 0..q {
                            let pv = post[c * q + qi];
                            n_cnt[base + qi] += pv;
                            r_cnt[base + qi] += yy * pv;
                        }
                    }
                }
            }
        }
        loglik_trace.push(total_ll);

        // Convergence check BEFORE the M-step so the returned params match the trace endpoint.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        for c in 0..n_classes {
            for i in 0..n_items {
                let base = (c * n_items + i) * q;
                let (ai, bi) = newton_item_2pl(
                    &n_cnt[base..base + q],
                    &r_cnt[base..base + q],
                    a[c * n_items + i],
                    b[c * n_items + i],
                    fix_slope,
                    cfg.newton_iter,
                    cfg.ridge_a,
                    cfg.ridge_b,
                );
                a[c * n_items + i] = ai;
                b[c * n_items + i] = bi;
            }
        }
        let nf = n_persons as f64;
        let mut z = 0.0;
        for c in 0..n_classes {
            pi[c] = (pi_new[c] / nf).max(cfg.pi_floor);
            z += pi[c];
        }
        for c in 0..n_classes {
            pi[c] /= z;
        }
        n_iter += 1;
    }

    // Final pass: class responsibilities, MAP class, mixture EAP at the converged params.
    build_tables(&a, &b, &mut log_p1, &mut log_p0);
    let log_pi: Vec<f64> = pi.iter().map(|p| p.ln()).collect();
    let mut class_posterior = vec![0.0f64; n_persons * n_classes];
    let mut map_class = vec![0u32; n_persons];
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0;
    for j in 0..n_persons {
        final_ll += person_posterior(j, &log_p1, &log_p0, &log_pi, &mut post);
        let (mut best, mut best_r) = (0usize, -1.0);
        let mut th = 0.0;
        for c in 0..n_classes {
            let mut r_jc = 0.0;
            for qi in 0..q {
                let pv = post[c * q + qi];
                r_jc += pv;
                th += pv * GH_NODES[qi];
            }
            class_posterior[j * n_classes + c] = r_jc;
            if r_jc > best_r {
                best_r = r_jc;
                best = c;
            }
        }
        map_class[j] = best as u32;
        theta[j] = th;
    }
    if !converged {
        loglik_trace.push(final_ll);
    }

    let k = if fix_slope { 1 } else { 2 };
    MixtureResult {
        model,
        n_classes,
        a,
        b,
        pi,
        class_posterior,
        map_class,
        theta,
        loglik_trace,
        n_iter,
        converged,
        n_parameters: n_classes * (k * n_items) + (n_classes - 1),
    }
}

/// Reorder classes into the canonical public order: mixing weight descending, ties by
/// mean difficulty ascending.
fn canonical_order(res: MixtureResult) -> MixtureResult {
    let (c, j) = (res.n_classes, res.b.len() / res.n_classes.max(1));
    let mean_b: Vec<f64> = (0..c)
        .map(|cc| res.b[cc * j..(cc + 1) * j].iter().sum::<f64>() / j as f64)
        .collect();
    let mut order: Vec<usize> = (0..c).collect();
    order.sort_by(|&x, &y| {
        res.pi[y]
            .partial_cmp(&res.pi[x])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(mean_b[x].partial_cmp(&mean_b[y]).unwrap_or(std::cmp::Ordering::Equal))
    });
    let mut inv = vec![0usize; c]; // inv[old] = new position
    for (new_pos, &old) in order.iter().enumerate() {
        inv[old] = new_pos;
    }
    let (mut a2, mut b2, mut pi2) = (vec![0.0; res.a.len()], vec![0.0; res.b.len()], vec![0.0; c]);
    for (new_pos, &old) in order.iter().enumerate() {
        a2[new_pos * j..(new_pos + 1) * j].copy_from_slice(&res.a[old * j..(old + 1) * j]);
        b2[new_pos * j..(new_pos + 1) * j].copy_from_slice(&res.b[old * j..(old + 1) * j]);
        pi2[new_pos] = res.pi[old];
    }
    let n = res.map_class.len();
    let mut cp2 = vec![0.0; res.class_posterior.len()];
    for jj in 0..n {
        for (new_pos, &old) in order.iter().enumerate() {
            cp2[jj * c + new_pos] = res.class_posterior[jj * c + old];
        }
    }
    let map2: Vec<u32> = res.map_class.iter().map(|&m| inv[m as usize] as u32).collect();
    MixtureResult { a: a2, b: b2, pi: pi2, class_posterior: cp2, map_class: map2, ..res }
}

/// Fit a mixture IRT model (Rost, 1990) by marginal EM. `y`/`observed` are row-major
/// `N*J` (`y` in {0,1}); missing cells (`observed == false`) are dropped (MAR). For
/// `C = 1` a single deterministic start is used (`cfg.n_starts` is ignored); with
/// `TwoPl` and `tol = 0.0` it reduces bit-exactly to
/// [`crate::mmle::fit_mmle_2pl`]. For `C >= 2` the fit runs
/// `cfg.n_starts` restarts (start 0 is a deterministic warm start) and keeps the run
/// with the highest final log-likelihood. Classes are returned in canonical order.
pub fn fit_mixture(
    y: &[f64],
    observed: &[bool],
    n_persons: usize,
    n_items: usize,
    n_classes: usize,
    model: MixtureModel,
    cfg: &MixtureConfig,
) -> Result<MixtureResult, String> {
    validate(y, observed, n_persons, n_items, n_classes, cfg)?;

    if n_classes == 1 {
        // Bit-exact single-start reduction to fit_mmle_2pl: a = 1, b = logit(prop).
        let b0 = init_mmle_like(y, observed, n_persons, n_items);
        let a0 = vec![1.0f64; n_items];
        let res = run_em(y, observed, n_persons, n_items, 1, model, a0, b0, vec![1.0], cfg);
        return Ok(canonical_order(res));
    }

    // Warm start from a single-class 2PL fit (its difficulties seed every class).
    let warm = fit_mmle_2pl(
        y,
        observed,
        n_persons,
        n_items,
        &MmleConfig {
            max_iter: cfg.max_iter,
            tol: 1e-4,
            ridge_a: cfg.ridge_a,
            ridge_b: cfg.ridge_b,
            newton_iter: cfg.newton_iter,
        },
    );
    let fix_slope = model == MixtureModel::Rasch;

    let mut best: Option<MixtureResult> = None;
    for start in 0..cfg.n_starts {
        let mut a = vec![0.0f64; n_classes * n_items];
        let mut b = vec![0.0f64; n_classes * n_items];
        for c in 0..n_classes {
            for i in 0..n_items {
                a[c * n_items + i] = if fix_slope { 1.0 } else { warm.a[i] };
            }
        }
        if start == 0 {
            // Deterministic warm start: centered per-class difficulty shift.
            for c in 0..n_classes {
                let delta = cfg.start_spread * (c as f64 - (n_classes as f64 - 1.0) / 2.0);
                for i in 0..n_items {
                    b[c * n_items + i] = warm.b[i] + delta;
                }
            }
        } else {
            // Random restart: per-(class, item) perturbation to explore reordering-type
            // class structure that a global difficulty shift cannot reach.
            let mut rng = Lcg(cfg.seed ^ (start as u64).wrapping_mul(0x9E3779B97F4A7C15));
            for c in 0..n_classes {
                for i in 0..n_items {
                    b[c * n_items + i] = warm.b[i] + cfg.start_spread * (2.0 * rng.next_f64() - 1.0);
                }
            }
        }
        let pi = vec![1.0 / n_classes as f64; n_classes];
        let res = run_em(y, observed, n_persons, n_items, n_classes, model, a, b, pi, cfg);
        let ll = *res.loglik_trace.last().unwrap();
        if best.as_ref().is_none_or(|bst| ll > *bst.loglik_trace.last().unwrap()) {
            best = Some(res);
        }
    }
    Ok(canonical_order(best.unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestRng(u64);
    impl TestRng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
        fn skew(&mut self) -> f64 {
            // Exp(1) - 1: mean 0, var 1, right-skewed (skewness 2).
            -(self.next_f64().max(1e-12)).ln() - 1.0
        }
        fn bern(&mut self, p: f64) -> f64 {
            if self.next_f64() < p {
                1.0
            } else {
                0.0
            }
        }
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
    }
    fn nondecreasing(trace: &[f64]) -> bool {
        trace.windows(2).all(|w| w[1] >= w[0] - 1e-6)
    }

    /// Best of the two C=2 label permutations (identity vs swap) minimizing difficulty
    /// SSE; returns (permutation as new->old, matched-b RMSE).
    fn match_c2(b_fit: &[f64], b_true: &[f64], n_items: usize) -> ([usize; 2], f64) {
        let sse = |perm: [usize; 2]| -> f64 {
            let mut s = 0.0;
            for (c_new, &c_old) in perm.iter().enumerate() {
                for i in 0..n_items {
                    let d = b_fit[c_old * n_items + i] - b_true[c_new * n_items + i];
                    s += d * d;
                }
            }
            s
        };
        let (id, sw) = ([0usize, 1], [1usize, 0]);
        let perm = if sse(id) <= sse(sw) { id } else { sw };
        (perm, (sse(perm) / (2 * n_items) as f64).sqrt())
    }

    /// Adjusted Rand index (Hubert & Arabie, 1985) — label-invariant agreement.
    fn ari(a: &[u32], b: &[u32]) -> f64 {
        let ka = (*a.iter().max().unwrap() + 1) as usize;
        let kb = (*b.iter().max().unwrap() + 1) as usize;
        let mut tab = vec![0u64; ka * kb];
        for (&x, &y) in a.iter().zip(b) {
            tab[x as usize * kb + y as usize] += 1;
        }
        let c2 = |n: u64| (n * n.saturating_sub(1) / 2) as f64;
        let index: f64 = tab.iter().map(|&n| c2(n)).sum();
        let sum_a: f64 = (0..ka).map(|i| c2((0..kb).map(|j| tab[i * kb + j]).sum())).sum();
        let sum_b: f64 = (0..kb).map(|j| c2((0..ka).map(|i| tab[i * kb + j]).sum())).sum();
        let n = a.len() as u64;
        let expected = sum_a * sum_b / c2(n);
        let max_index = 0.5 * (sum_a + sum_b);
        if (max_index - expected).abs() < 1e-12 {
            1.0
        } else {
            (index - expected) / (max_index - expected)
        }
    }

    /// Simulate a two-class mixture with a difficulty REVERSAL (b_1 = -b_0): the
    /// canonical Rost two-strategy structure a single class cannot fit.
    fn simulate_c2(
        n: usize,
        n_items: usize,
        pi: f64,
        b0: &[f64],
        a0: &[f64],
        skew: bool,
        rng: &mut TestRng,
    ) -> (Vec<f64>, Vec<u32>) {
        let mut y = vec![0.0f64; n * n_items];
        let mut cls = vec![0u32; n];
        for j in 0..n {
            let c = if rng.next_f64() < pi { 0usize } else { 1usize };
            cls[j] = c as u32;
            let theta = if skew { rng.skew() } else { rng.normal() };
            for i in 0..n_items {
                let (ai, bi) = if c == 0 { (a0[i], b0[i]) } else { (a0[i], -b0[i]) };
                let p = sigmoid_stable(ai * theta + bi);
                y[j * n_items + i] = rng.bern(p);
            }
        }
        (y, cls)
    }

    /// Anchor 1: C=1 TwoPl reduces bit-exactly to fit_mmle_2pl (tol=0.0 so both run the
    /// full max_iter from the identical init).
    #[test]
    fn mixture_c1_equals_fit_mmle_2pl() {
        let (n, j) = (600usize, 12usize);
        let mut rng = TestRng(7);
        let a_t: Vec<f64> = (0..j).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
        let b_t: Vec<f64> = (0..j).map(|i| -1.2 + 2.4 * i as f64 / (j - 1) as f64).collect();
        let mut y = vec![0.0f64; n * j];
        for p in 0..n {
            let theta = rng.normal();
            for i in 0..j {
                y[p * j + i] = rng.bern(sigmoid_stable(a_t[i] * theta + b_t[i]));
            }
        }
        let observed = vec![true; n * j];
        let mcfg = MmleConfig { max_iter: 60, tol: 0.0, ridge_a: 1e-3, ridge_b: 1e-3, newton_iter: 25 };
        let mmle = fit_mmle_2pl(&y, &observed, n, j, &mcfg);
        let cfg = MixtureConfig { max_iter: 60, tol: 0.0, ridge_a: 1e-3, ridge_b: 1e-3, newton_iter: 25, ..MixtureConfig::default() };
        let mix = fit_mixture(&y, &observed, n, j, 1, MixtureModel::TwoPl, &cfg).unwrap();
        assert_eq!(mix.pi, vec![1.0]);
        assert!(rmse(&mix.a, &mmle.a) < 1e-12, "a RMSE {}", rmse(&mix.a, &mmle.a));
        assert!(rmse(&mix.b, &mmle.b) < 1e-12, "b RMSE {}", rmse(&mix.b, &mmle.b));
        assert_eq!(mix.n_parameters, 2 * j);
    }

    /// Anchor 2: two well-separated classes (difficulty reversal) recovered with
    /// permutation matching, multi-start against local optima.
    #[test]
    fn recovers_mixed_rasch_c2() {
        let (n, j) = (1200usize, 15usize);
        let pi_true = 0.6;
        let b0: Vec<f64> = (0..j).map(|i| -2.0 + 4.0 * i as f64 / (j - 1) as f64).collect();
        let a0 = vec![1.0f64; j];
        let mut rng = TestRng(2024);
        let (y, cls) = simulate_c2(n, j, pi_true, &b0, &a0, false, &mut rng);
        let observed = vec![true; n * j];
        let cfg = MixtureConfig { n_starts: 8, ..MixtureConfig::default() };
        let res = fit_mixture(&y, &observed, n, j, 2, MixtureModel::Rasch, &cfg).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(res.a.iter().all(|&a| (a - 1.0).abs() < 1e-12)); // Rasch: a == 1
        // truth in canonical layout: class 0 = b0, class 1 = -b0
        let mut b_true = vec![0.0f64; 2 * j];
        b_true[..j].copy_from_slice(&b0);
        for i in 0..j {
            b_true[j + i] = -b0[i];
        }
        let (perm, brmse) = match_c2(&res.b, &b_true, j);
        assert!(brmse < 0.25, "matched b RMSE {brmse}");
        // matched mixing proportions (true class 0 has weight pi_true)
        let pi_matched0 = res.pi[perm[0]];
        assert!((pi_matched0 - pi_true).abs() < 0.06, "pi {pi_matched0}");
        // classification: relabel map_class by perm, compare to truth; ARI cross-check
        let inv = if perm == [0, 1] { [0u32, 1] } else { [1u32, 0] };
        let relabeled: Vec<u32> = res.map_class.iter().map(|&m| inv[m as usize]).collect();
        let acc = relabeled.iter().zip(&cls).filter(|(a, b)| a == b).count() as f64 / n as f64;
        assert!(acc > 0.80, "MAP class accuracy {acc}");
        assert!(ari(&res.map_class, &cls) > 0.35, "ARI {}", ari(&res.map_class, &cls));
    }

    /// Missing-at-random cells are dropped from likelihood and counts.
    #[test]
    fn mixture_handles_missing_data() {
        let (n, j) = (800usize, 12usize);
        let b0: Vec<f64> = (0..j).map(|i| -1.5 + 3.0 * i as f64 / (j - 1) as f64).collect();
        let a0 = vec![1.0f64; j];
        let mut rng = TestRng(55);
        let (y, _) = simulate_c2(n, j, 0.5, &b0, &a0, false, &mut rng);
        let mut observed = vec![true; n * j];
        for o in observed.iter_mut() {
            if rng.next_f64() < 0.2 {
                *o = false;
            }
        }
        let cfg = MixtureConfig { n_starts: 6, ..MixtureConfig::default() };
        let res = fit_mixture(&y, &observed, n, j, 2, MixtureModel::Rasch, &cfg).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
    }

    /// The C=1 short-circuit runs a single start regardless of n_starts, and a
    /// non-converged fit still returns (max-iter guard).
    #[test]
    fn mixture_c1_ignores_starts_and_stops_at_max_iter() {
        let (n, j) = (200usize, 8usize);
        let mut rng = TestRng(3);
        let mut y = vec![0.0f64; n * j];
        for p in 0..n {
            let theta = rng.normal();
            for i in 0..j {
                y[p * j + i] = rng.bern(sigmoid_stable(theta - 0.5 + 0.1 * i as f64));
            }
        }
        let observed = vec![true; n * j];
        let cfg = MixtureConfig { max_iter: 1, n_starts: 9, ..MixtureConfig::default() };
        let res = fit_mixture(&y, &observed, n, j, 1, MixtureModel::TwoPl, &cfg).unwrap();
        assert!(!res.converged && res.n_iter == 1 && res.pi == vec![1.0]);
    }

    /// Malformed inputs are rejected (covers each validate branch, incl. tol=0 allowed).
    #[test]
    fn mixture_validate_rejects_malformed() {
        let y = vec![0.0f64; 4 * 3];
        let obs = vec![true; 12];
        let d = MixtureConfig::default();
        let bad = |y: &[f64], obs: &[bool], n, j, c, cfg: &MixtureConfig| {
            fit_mixture(y, obs, n, j, c, MixtureModel::Rasch, cfg).is_err()
        };
        assert!(bad(&y, &obs, 0, 3, 2, &d)); // n_persons < 1
        assert!(bad(&y, &obs, 4, 3, 0, &d)); // n_classes < 1
        assert!(bad(&y, &obs, 4, 3, 2, &MixtureConfig { max_iter: 0, ..d })); // max_iter
        assert!(bad(&y, &obs, 4, 3, 2, &MixtureConfig { tol: -1.0, ..d })); // tol < 0
        assert!(bad(&y, &obs, 4, 3, 2, &MixtureConfig { newton_iter: 0, ..d })); // newton_iter
        assert!(bad(&y, &obs, 4, 3, 2, &MixtureConfig { n_starts: 0, ..d })); // n_starts
        assert!(bad(&y, &obs, 4, 3, 2, &MixtureConfig { pi_floor: 0.6, ..d })); // pi_floor >= 1/C
        assert!(bad(&vec![0.0; 5], &obs, 4, 3, 2, &d)); // y length
        assert!(bad(&vec![2.0; 12], &obs, 4, 3, 2, &d)); // y not 0/1
        let mut obs_gap = vec![true; 12];
        for p in 0..4 {
            obs_gap[p * 3 + 1] = false; // item 1 fully unobserved
        }
        assert!(bad(&y, &obs_gap, 4, 3, 2, &d));
        // tol == 0.0 is accepted
        assert!(fit_mixture(&y, &obs, 4, 3, 1, MixtureModel::Rasch, &MixtureConfig { tol: 0.0, max_iter: 2, ..d }).is_ok());
    }

    #[test]
    fn mixture_validate_rejects_nonfinite_optimizer_config() {
        let y = vec![0.0f64; 4 * 3];
        let obs = vec![true; 12];
        let d = MixtureConfig::default();
        let bad = |cfg: &MixtureConfig| {
            fit_mixture(&y, &obs, 4, 3, 2, MixtureModel::Rasch, cfg).is_err()
        };

        assert!(bad(&MixtureConfig { ridge_a: f64::NAN, ..d }));
        assert!(bad(&MixtureConfig { ridge_b: -1.0, ..d }));
        assert!(bad(&MixtureConfig { start_spread: f64::INFINITY, ..d }));
    }

    #[test]
    fn mixture_validate_rejects_dimension_overflow() {
        let y = vec![0.0f64; 2];
        let obs = vec![true; 2];
        let cfg = MixtureConfig {
            pi_floor: f64::MIN_POSITIVE,
            ..MixtureConfig::default()
        };

        assert!(fit_mixture(
            &y,
            &obs,
            1,
            2,
            usize::MAX,
            MixtureModel::Rasch,
            &cfg,
        )
        .is_err());
    }

    /// Literature-grade Monte-Carlo (>=500 reps): Rost-style two-class reversal recovery
    /// under normal and skew ability, permutation-matched, with ARI cross-check.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_mixture_recovery_500() {
        let (n, j, reps) = (1500usize, 15usize, 500usize);
        let pi_true = 0.6;
        let b0: Vec<f64> = (0..j).map(|i| -2.0 + 4.0 * i as f64 / (j - 1) as f64).collect();
        let a0 = vec![1.0f64; j];
        let mut b_true = vec![0.0f64; 2 * j];
        b_true[..j].copy_from_slice(&b0);
        for i in 0..j {
            b_true[j + i] = -b0[i];
        }
        let n_starts = 8;
        for &skew in [false, true].iter() {
            let (mut sum_brmse, mut sum_bbias, mut sum_pi, mut sum_acc, mut sum_ari) =
                (0.0, 0.0, 0.0, 0.0, 0.0);
            for rep in 0..reps {
                let seed = 0xA1B2C3D4E5F60718u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add(if skew { 0x9E3779B97F4A7C15 } else { 0 });
                let mut rng = TestRng(seed);
                let (y, cls) = simulate_c2(n, j, pi_true, &b0, &a0, skew, &mut rng);
                let observed = vec![true; n * j];
                let cfg = MixtureConfig { n_starts, seed: seed ^ 0xDEAD, ..MixtureConfig::default() };
                let res = fit_mixture(&y, &observed, n, j, 2, MixtureModel::Rasch, &cfg).unwrap();
                let (perm, brmse) = match_c2(&res.b, &b_true, j);
                sum_brmse += brmse;
                let mut bb = 0.0;
                for (c_new, &c_old) in perm.iter().enumerate() {
                    for i in 0..j {
                        bb += res.b[c_old * j + i] - b_true[c_new * j + i];
                    }
                }
                sum_bbias += bb / (2 * j) as f64;
                sum_pi += (res.pi[perm[0]] - pi_true).abs();
                let inv = if perm == [0, 1] { [0u32, 1] } else { [1u32, 0] };
                let relabeled: Vec<u32> = res.map_class.iter().map(|&m| inv[m as usize]).collect();
                sum_acc += relabeled.iter().zip(&cls).filter(|(a, b)| a == b).count() as f64 / n as f64;
                sum_ari += ari(&res.map_class, &cls);
            }
            let r = reps as f64;
            println!(
                "skew={} n_starts={}: RMSE(b)={:.4} bias(b)={:.4} |dpi|={:.4} MAPacc={:.3} ARI={:.3}",
                skew, n_starts, sum_brmse / r, sum_bbias / r, sum_pi / r, sum_acc / r, sum_ari / r
            );
            assert!(sum_brmse / r < 0.20, "mean RMSE(b) {} skew={skew}", sum_brmse / r);
            assert!(sum_pi / r < 0.05, "mean |dpi| {} skew={skew}", sum_pi / r);
            assert!(sum_ari / r > 0.55, "mean ARI {} skew={skew}", sum_ari / r);
        }
    }
}
