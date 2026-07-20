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
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    let class_items =
        crate::checked_mul_usize(n_classes, n_items, "n_classes * n_items overflows usize")?;
    crate::checked_mul_usize(n_classes, GH_NODES.len(), "class-node size overflows")?;
    crate::checked_mul_usize(class_items, GH_NODES.len(), "class-item-node overflow")?;
    crate::checked_mul_usize(n_persons, n_classes, "person-class size overflows")?;
    let doubled = crate::checked_mul_usize(class_items, 2, "parameter count overflows")?;
    crate::checked_add_usize(doubled, n_classes - 1, "parameter count overflows")?;
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
        let prop = if den > 0.0 {
            (num / den).clamp(0.02, 0.98)
        } else {
            0.5
        };
        b[i] = (prop / (1.0 - prop)).ln();
    }
    b
}

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
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
    let person_posterior =
        |j: usize, log_p1: &[f64], log_p0: &[f64], log_pi: &[f64], post: &mut [f64]| -> f64 {
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
            .then(
                mean_b[x]
                    .partial_cmp(&mean_b[y])
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
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
    let map2: Vec<u32> = res
        .map_class
        .iter()
        .map(|&m| inv[m as usize] as u32)
        .collect();
    MixtureResult {
        a: a2,
        b: b2,
        pi: pi2,
        class_posterior: cp2,
        map_class: map2,
        ..res
    }
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
        let res = run_em(
            y,
            observed,
            n_persons,
            n_items,
            1,
            model,
            a0,
            b0,
            vec![1.0],
            cfg,
        );
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
                    b[c * n_items + i] =
                        warm.b[i] + cfg.start_spread * (2.0 * rng.next_f64() - 1.0);
                }
            }
        }
        let pi = vec![1.0 / n_classes as f64; n_classes];
        let res = run_em(
            y, observed, n_persons, n_items, n_classes, model, a, b, pi, cfg,
        );
        let ll = *res.loglik_trace.last().unwrap();
        if best
            .as_ref()
            .is_none_or(|bst| ll > *bst.loglik_trace.last().unwrap())
        {
            best = Some(res);
        }
    }
    Ok(canonical_order(best.unwrap()))
}

#[cfg(test)]
#[path = "../../../tests/unit/mixture_tests.rs"]
mod tests;
