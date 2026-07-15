//! Testlet response model (Bradlow, Wainer, & Wang, 1999; Wang, Bradlow, & Wainer,
//! 2002): a marginal-ML estimator for the local dependence induced when items share a
//! common stimulus (a reading passage, a scenario). Items are partitioned into disjoint
//! *testlets*; each item `i` in testlet `d(i)` carries a person-specific random effect
//! `gamma_{j,d(i)} ~ N(0, sigma^2_d)`, independent across testlets and of `theta_j`:
//!
//! ```text
//! P(X_ij = 1 | theta_j, gamma) = sigmoid(a_i * (theta_j - b_i - gamma_{j,d(i)}))
//!                              = sigmoid(a_i*theta_j + beta_i - a_i*gamma_{j,d(i)})
//! ```
//!
//! with `beta_i = -a_i*b_i` (Rasch fixes `a_i = 1`). The **testlet variance `sigma^2_d`
//! is the estimand of interest**: a large value flags strong within-testlet local
//! dependence (the bundle measures a passage-specific nuisance beyond `theta`); all
//! `sigma^2_d = 0` is exactly the conditional-independence 2PL/Rasch model.
//!
//! Estimation is marginal ML, integrating out `theta` AND the `D` testlet effects.
//! Because each item depends on `theta` and exactly ONE `gamma` and testlets are
//! disjoint, the `D`-dimensional `gamma` integral FACTORS per testlet given `theta`:
//! the marginal likelihood is a `theta`-outer / per-testlet-`gamma`-inner nested
//! Gauss-Hermite quadrature at per-person cost `Q_theta * Q_gamma * n_items`
//! (independent of `D`) — NOT a `(D+1)`-dimensional tensor grid. This is why a
//! dedicated estimator, rather than the general free-loading bifactor
//! ([`crate::ModelType::Bifac2plm`]), is used: the bifactor cannot report the per-
//! testlet variance and its `D`-dimensional secondary-factor grid is exponential.
//!
//! Identification: `theta ~ N(0,1)` pins the trait metric (location -> `beta_i`, scale
//! -> `a_i`); `gamma` is centered (mean absorbed into `beta_i`); only the magnitude
//! `sigma^2_d` is identified, and only for testlets with >= 2 items (a singleton
//! testlet has no within-bundle pair to reveal excess correlation, so its variance is
//! pinned to 0 rather than left to report spurious dependence).
//!
//! Deferred (non-goals): polytomous testlets, 3PL guessing, covariate/second-order
//! structure (the free-loading bifactor already covers that), per-person `gamma` EAP
//! output, GPU offload, and the original paper's fully-Bayesian probit + Gibbs
//! estimator (this is the standard logit + marginal-ML reduction; Wainer, Bradlow, &
//! Wang, 2007).
//!
//! References (APA 7th ed.):
//! - Bradlow, E. T., Wainer, H., & Wang, X. (1999). A Bayesian random effects model
//!   for testlets. *Psychometrika, 64*(2), 153-168.
//!   <https://doi.org/10.1007/BF02294533>
//! - Wang, X., Bradlow, E. T., & Wainer, H. (2002). A general Bayesian model for
//!   testlets: Theory and applications. *Applied Psychological Measurement, 26*(1),
//!   109-128. <https://doi.org/10.1177/0146621602026001007>
//! - Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
//!   parameters. *Psychometrika, 46*(4), 443-459. <https://doi.org/10.1007/BF02293801>

use crate::mmle::{log_sigmoid, sigmoid_stable, GH_NODES, GH_WEIGHTS};
use crate::quadrature::{gh_rule, SUPPORTED_Q};

/// Within-testlet response model: `Rasch` fixes `a_i = 1`; `TwoPl` frees `a_i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestletModel {
    Rasch,
    TwoPl,
}

/// EM configuration for the testlet estimator.
#[derive(Clone, Copy, Debug)]
pub struct TestletConfig {
    pub max_iter: usize,
    /// Convergence tolerance on `|delta loglik|`; `0.0` is permitted (runs the full
    /// `max_iter`) — needed for the exact `sigma -> 0` reduction anchor.
    pub tol: f64,
    /// Inner `gamma` Gauss-Hermite nodes; must be one of `SUPPORTED_Q` (7/11/15/21/31/41).
    pub q_gamma: usize,
    pub ridge_a: f64,
    pub ridge_b: f64,
    pub newton_iter: usize,
    /// Estimate the testlet variances; `false` pins them at `init_sigma2`
    /// (`init_sigma2 = 0` then gives the exact 2PL/Rasch reduction).
    pub estimate_sigma: bool,
    /// Initial (and, if `!estimate_sigma`, fixed) testlet variance for multi-item testlets.
    pub init_sigma2: f64,
}

impl Default for TestletConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            q_gamma: 21,
            ridge_a: 1e-3,
            ridge_b: 1e-3,
            newton_iter: 25,
            estimate_sigma: true,
            init_sigma2: 0.5,
        }
    }
}

/// Fitted testlet model.
#[derive(Clone, Debug)]
pub struct TestletResult {
    pub model: TestletModel,
    /// Per-item discrimination (Rasch: all 1.0), length `J`.
    pub a: Vec<f64>,
    /// Per-item IRT difficulty `b_i = -beta_i / a_i`, length `J`.
    pub b: Vec<f64>,
    /// Per-item intercept `beta_i` (2PL-parity metric; equals `fit_mmle_2pl.b` at
    /// `sigma = 0`), length `J`.
    pub beta: Vec<f64>,
    /// Per-testlet variance `sigma^2_d` — the local-dependence estimand, length `D`.
    pub sigma2: Vec<f64>,
    /// Per-person EAP ability, length `N`.
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `(TwoPl? 2J : J) + D`.
    pub n_parameters: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    y: &[f64],
    observed: &[bool],
    testlet_id: &[usize],
    n_persons: usize,
    n_items: usize,
    n_testlets: usize,
    cfg: &TestletConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 || n_testlets < 1 {
        return Err("n_persons, n_items and n_testlets must be >= 1".into());
    }
    if cfg.max_iter == 0 || cfg.newton_iter == 0 {
        return Err("max_iter and newton_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err("tol must be finite and non-negative".into());
    }
    if !cfg.ridge_a.is_finite() || cfg.ridge_a < 0.0 || !cfg.ridge_b.is_finite() || cfg.ridge_b < 0.0 {
        return Err("ridge_a and ridge_b must be finite and non-negative".into());
    }
    if !cfg.init_sigma2.is_finite() || cfg.init_sigma2 < 0.0 {
        return Err("init_sigma2 must be finite and non-negative".into());
    }
    if !SUPPORTED_Q.contains(&cfg.q_gamma) {
        return Err(format!("q_gamma must be one of {SUPPORTED_Q:?}; got {}", cfg.q_gamma));
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    if testlet_id.len() != n_items {
        return Err("testlet_id must have length n_items".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && v != 0.0 && v != 1.0 {
            return Err(format!("y[{idx}] must be 0 or 1 where observed; got {v}"));
        }
    }
    let mut size = vec![0usize; n_testlets];
    for (i, &d) in testlet_id.iter().enumerate() {
        if d >= n_testlets {
            return Err(format!("testlet_id[{i}] = {d} out of range 0..{n_testlets}"));
        }
        size[d] += 1;
    }
    for (d, &s) in size.iter().enumerate() {
        if s == 0 {
            return Err(format!("testlet {d} has no items"));
        }
    }
    for i in 0..n_items {
        if !(0..n_persons).any(|p| observed[p * n_items + i]) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    Ok(())
}

/// Rasch/2PL easiness init identical to `fit_mmle_2pl`: `beta_i = logit(clamp(prop))`.
fn init_beta(y: &[f64], observed: &[bool], n_persons: usize, n_items: usize) -> Vec<f64> {
    (0..n_items)
        .map(|i| {
            let (mut num, mut den) = (0.0, 0.0);
            for p in 0..n_persons {
                let idx = p * n_items + i;
                if observed[idx] {
                    num += y[idx];
                    den += 1.0;
                }
            }
            let prop = if den > 0.0 { (num / den).clamp(0.02, 0.98) } else { 0.5 };
            (prop / (1.0 - prop)).ln()
        })
        .collect()
}

/// Immutable context shared across E-steps.
struct Ctx<'a> {
    y: &'a [f64],
    observed: &'a [bool],
    testlet_id: &'a [usize],
    items_of: &'a [Vec<usize>],
    n: usize,
    j: usize,
    d_n: usize,
    qt: usize,
    qg: usize,
    u_nodes: &'a [f64],
    log_wt: &'a [f64],
    log_vu: &'a [f64],
}

/// One full E-step: the marginal loglik at `(a, beta, sigma2)`, the expected counts
/// `n_i`/`r_i`, the per-testlet `sum_j E[u_d^2 | y_j]`, and the person theta EAPs. The
/// `sigma == 0` fast path adds each item's term directly into the theta log-numerator,
/// reproducing `fit_mmle_2pl`'s sequential accumulation bit-for-bit (contiguous testlets).
fn full_estep(
    ctx: &Ctx,
    a: &[f64],
    beta: &[f64],
    sigma2: &[f64],
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (n, j, d_n, qt, qg) = (ctx.n, ctx.j, ctx.d_n, ctx.qt, ctx.qg);
    let idx3 = |i: usize, g: usize, h: usize| (i * qt + g) * qg + h;
    let mut logp1 = vec![0.0f64; j * qt * qg];
    let mut logp0 = vec![0.0f64; j * qt * qg];
    for i in 0..j {
        let sd = sigma2[ctx.testlet_id[i]].sqrt();
        for g in 0..qt {
            if sd == 0.0 {
                let eta = a[i] * GH_NODES[g] + beta[i];
                logp1[idx3(i, g, 0)] = log_sigmoid(eta);
                logp0[idx3(i, g, 0)] = log_sigmoid(-eta);
            } else {
                for h in 0..qg {
                    let eta = a[i] * GH_NODES[g] + beta[i] - a[i] * sd * ctx.u_nodes[h];
                    logp1[idx3(i, g, h)] = log_sigmoid(eta);
                    logp0[idx3(i, g, h)] = log_sigmoid(-eta);
                }
            }
        }
    }
    let mut n_i = vec![0.0f64; j * qt * qg];
    let mut r_i = vec![0.0f64; j * qt * qg];
    let mut sum_u2 = vec![0.0f64; d_n];
    let mut theta = vec![0.0f64; n];
    let mut total_ll = 0.0;
    let mut log_a = vec![0.0f64; qt];
    let mut log_g = vec![0.0f64; d_n * qt];
    let mut s_arr = vec![0.0f64; d_n * qt * qg];
    for p in 0..n {
        log_a.copy_from_slice(ctx.log_wt);
        for d in 0..d_n {
            let sd = sigma2[d].sqrt();
            if sd == 0.0 {
                for &i in &ctx.items_of[d] {
                    let idx = p * j + i;
                    if ctx.observed[idx] {
                        let yy = ctx.y[idx];
                        for g in 0..qt {
                            log_a[g] += yy * logp1[idx3(i, g, 0)] + (1.0 - yy) * logp0[idx3(i, g, 0)];
                        }
                    }
                }
            } else {
                for g in 0..qt {
                    for h in 0..qg {
                        let mut s = 0.0;
                        for &i in &ctx.items_of[d] {
                            let idx = p * j + i;
                            if ctx.observed[idx] {
                                let yy = ctx.y[idx];
                                s += yy * logp1[idx3(i, g, h)] + (1.0 - yy) * logp0[idx3(i, g, h)];
                            }
                        }
                        s_arr[(d * qt + g) * qg + h] = s;
                    }
                    let mut m = f64::NEG_INFINITY;
                    for h in 0..qg {
                        let v = ctx.log_vu[h] + s_arr[(d * qt + g) * qg + h];
                        if v > m {
                            m = v;
                        }
                    }
                    let mut denom = 0.0;
                    for h in 0..qg {
                        denom += (ctx.log_vu[h] + s_arr[(d * qt + g) * qg + h] - m).exp();
                    }
                    let lg = m + denom.ln();
                    log_g[d * qt + g] = lg;
                    log_a[g] += lg;
                }
            }
        }
        let mut mg = f64::NEG_INFINITY;
        for &v in log_a.iter() {
            if v > mg {
                mg = v;
            }
        }
        let mut denomg = 0.0;
        for &v in log_a.iter() {
            denomg += (v - mg).exp();
        }
        total_ll += mg + denomg.ln();
        for g in 0..qt {
            log_a[g] = (log_a[g] - mg).exp() / denomg;
        }
        let mut th = 0.0;
        for g in 0..qt {
            th += log_a[g] * GH_NODES[g];
        }
        theta[p] = th;
        for d in 0..d_n {
            let sd = sigma2[d].sqrt();
            if sd == 0.0 {
                for &i in &ctx.items_of[d] {
                    let idx = p * j + i;
                    if ctx.observed[idx] {
                        let yy = ctx.y[idx];
                        for g in 0..qt {
                            let pv = log_a[g];
                            n_i[idx3(i, g, 0)] += pv;
                            r_i[idx3(i, g, 0)] += yy * pv;
                        }
                    }
                }
            } else {
                for g in 0..qt {
                    let lg = log_g[d * qt + g];
                    let pg = log_a[g];
                    for &i in &ctx.items_of[d] {
                        let idx = p * j + i;
                        if ctx.observed[idx] {
                            let yy = ctx.y[idx];
                            for h in 0..qg {
                                let c = (ctx.log_vu[h] + s_arr[(d * qt + g) * qg + h] - lg).exp();
                                let resp = pg * c;
                                n_i[idx3(i, g, h)] += resp;
                                r_i[idx3(i, g, h)] += yy * resp;
                            }
                        }
                    }
                    for h in 0..qg {
                        let c = (ctx.log_vu[h] + s_arr[(d * qt + g) * qg + h] - lg).exp();
                        sum_u2[d] += pg * c * ctx.u_nodes[h] * ctx.u_nodes[h];
                    }
                }
            }
        }
    }
    (total_ll, n_i, r_i, sum_u2, theta)
}

/// One M-step from the expected counts: per-item 2-D Newton on the effective node
/// `z = t_g - sigma_d*u_h` (verbatim `fit_mmle_2pl` arithmetic; `fix_slope` holds
/// `a = 1`) and the closed-form testlet-variance update
/// `sigma^2_d <- sigma^2_d * mean_j E[u_d^2 | y_j]`. Returns the new `(a, beta, sigma2)`.
#[allow(clippy::too_many_arguments)]
fn m_step(
    ctx: &Ctx,
    a: &[f64],
    beta: &[f64],
    sigma2: &[f64],
    n_i: &[f64],
    r_i: &[f64],
    sum_u2: &[f64],
    multi: &[bool],
    fix_slope: bool,
    cfg: &TestletConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (j, d_n, qt, qg) = (ctx.j, ctx.d_n, ctx.qt, ctx.qg);
    let idx3 = |i: usize, g: usize, h: usize| (i * qt + g) * qg + h;
    let mut a = a.to_vec();
    let mut beta = beta.to_vec();
    let mut sigma2 = sigma2.to_vec();
    for i in 0..j {
        let sd = sigma2[ctx.testlet_id[i]].sqrt();
        let (mut ai, mut bi) = (a[i], beta[i]);
        for _ in 0..cfg.newton_iter {
            let (mut g_a, mut g_b, mut h_aa, mut h_bb, mut h_ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
            if sd == 0.0 {
                for g in 0..qt {
                    let z = GH_NODES[g];
                    let pc = sigmoid_stable(ai * z + bi);
                    let nn = n_i[idx3(i, g, 0)];
                    let w = nn * pc * (1.0 - pc);
                    let resid = r_i[idx3(i, g, 0)] - nn * pc;
                    g_a += resid * z;
                    g_b += resid;
                    h_aa -= w * z * z;
                    h_bb -= w;
                    h_ab -= w * z;
                }
            } else {
                for g in 0..qt {
                    for h in 0..qg {
                        let z = GH_NODES[g] - sd * ctx.u_nodes[h];
                        let pc = sigmoid_stable(ai * z + bi);
                        let nn = n_i[idx3(i, g, h)];
                        let w = nn * pc * (1.0 - pc);
                        let resid = r_i[idx3(i, g, h)] - nn * pc;
                        g_a += resid * z;
                        g_b += resid;
                        h_aa -= w * z * z;
                        h_bb -= w;
                        h_ab -= w * z;
                    }
                }
            }
            if fix_slope {
                g_b -= cfg.ridge_b * bi;
                h_bb -= cfg.ridge_b;
                if h_bb.abs() < 1e-12 {
                    break;
                }
                let db = g_b / h_bb;
                bi -= db;
                if db.abs() < 1e-8 {
                    break;
                }
            } else {
                g_a -= cfg.ridge_a * ai;
                g_b -= cfg.ridge_b * bi;
                h_aa -= cfg.ridge_a;
                h_bb -= cfg.ridge_b;
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
        a[i] = ai;
        beta[i] = bi;
    }
    if cfg.estimate_sigma {
        for d in 0..d_n {
            if multi[d] {
                sigma2[d] = (sigma2[d] * sum_u2[d] / ctx.n as f64).clamp(0.0, 100.0);
            }
        }
    }
    (a, beta, sigma2)
}

/// Fit the testlet response model (Bradlow, Wainer, & Wang, 1999) by marginal EM.
/// `y`/`observed` are row-major `N*J` (`y` in {0,1}); `testlet_id[i]` is item `i`'s
/// testlet in `0..n_testlets`. Missing cells are dropped (MAR). Singleton testlets have
/// `sigma^2_d` pinned to 0 (non-identified). `TestletConfig { estimate_sigma: false,
/// init_sigma2: 0.0 }` reduces exactly to a 2PL/Rasch marginal fit.
///
/// The variance-component EM converges only linearly, so when `estimate_sigma` is on the
/// fit is accelerated with SQUAREM (Varadhan & Roland, 2008; monotone, with a plain-EM
/// fallback). Precise `sigma^2_d` may still want a generous `max_iter`.
#[allow(clippy::too_many_arguments)]
pub fn fit_testlet(
    y: &[f64],
    observed: &[bool],
    testlet_id: &[usize],
    n_persons: usize,
    n_items: usize,
    n_testlets: usize,
    model: TestletModel,
    cfg: &TestletConfig,
) -> Result<TestletResult, String> {
    validate(y, observed, testlet_id, n_persons, n_items, n_testlets, cfg)?;
    let (n, j, d_n) = (n_persons, n_items, n_testlets);
    let qt = GH_NODES.len();
    let (u_nodes, u_weights) = gh_rule(cfg.q_gamma).expect("q_gamma validated in SUPPORTED_Q");
    let qg = u_nodes.len();
    let log_wt: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();
    let log_vu: Vec<f64> = u_weights.iter().map(|w| w.ln()).collect();

    // Testlet -> item indices, and per-testlet size (singletons pin sigma^2 = 0).
    let mut items_of: Vec<Vec<usize>> = vec![Vec::new(); d_n];
    for (i, &d) in testlet_id.iter().enumerate() {
        items_of[d].push(i);
    }
    let multi: Vec<bool> = items_of.iter().map(|v| v.len() >= 2).collect();

    let fix_slope = model == TestletModel::Rasch;
    let ctx = Ctx {
        y, observed, testlet_id, items_of: &items_of, n, j, d_n, qt, qg,
        u_nodes, log_wt: &log_wt, log_vu: &log_vu,
    };

    let mut a = vec![1.0f64; j];
    let mut beta = init_beta(y, observed, n, j);
    let mut sigma2: Vec<f64> = (0..d_n).map(|d| if multi[d] { cfg.init_sigma2 } else { 0.0 }).collect();

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    // SQUAREM (Varadhan & Roland, 2008) accelerates the slow variance-component EM;
    // used only when sigma^2 is estimated (plain EM otherwise keeps the sigma->0
    // reduction bit-exact with fit_mmle_2pl).
    let use_squarem = cfg.estimate_sigma && multi.iter().any(|&m| m);

    if use_squarem {
        let len = 2 * j + d_n;
        let pack = |a: &[f64], b: &[f64], s: &[f64]| -> Vec<f64> {
            a.iter().chain(b.iter()).chain(s.iter()).copied().collect()
        };
        let unpack = |p: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            (p[0..j].to_vec(), p[j..2 * j].to_vec(), p[2 * j..2 * j + d_n].to_vec())
        };
        let project = |p: &mut [f64]| {
            for ai in p.iter_mut().take(j) {
                *ai = ai.clamp(1e-3, 10.0);
            }
            for d in 0..d_n {
                let idx = 2 * j + d;
                // Floor multi-testlet sigma^2 above 0: exactly 0 is an absorbing state
                // (the sigma==0 fast path stops accumulating sum_u2, so the
                // multiplicative update could never revive an overshot testlet).
                p[idx] = if multi[d] { p[idx].clamp(1e-8, 100.0) } else { 0.0 };
            }
        };
        let mut params = pack(&a, &beta, &sigma2);
        while n_iter < cfg.max_iter {
            let (a0, b0, s0) = unpack(&params);
            let (l0, ni0, ri0, su0, _) = full_estep(&ctx, &a0, &b0, &s0);
            loglik_trace.push(l0);
            n_iter += 1;
            if loglik_trace.len() > 1 {
                let k = loglik_trace.len();
                if (l0 - loglik_trace[k - 2]).abs() < cfg.tol {
                    converged = true;
                    break;
                }
            }
            if n_iter >= cfg.max_iter {
                break;
            }
            // Two plain EM steps.
            let (a1, b1, s1) = m_step(&ctx, &a0, &b0, &s0, &ni0, &ri0, &su0, &multi, fix_slope, cfg);
            let p1 = pack(&a1, &b1, &s1);
            let (_l1, ni1, ri1, su1, _) = full_estep(&ctx, &a1, &b1, &s1);
            let (a2, b2, s2) = m_step(&ctx, &a1, &b1, &s1, &ni1, &ri1, &su1, &multi, fix_slope, cfg);
            let p2 = pack(&a2, &b2, &s2);
            // SqS3 steplength from r = p1 - p0, v = p2 - 2p1 + p0.
            let mut r = vec![0.0f64; len];
            let mut v = vec![0.0f64; len];
            for k in 0..len {
                r[k] = p1[k] - params[k];
                v[k] = p2[k] - p1[k] - r[k];
            }
            let sr: f64 = r.iter().map(|x| x * x).sum();
            let sv: f64 = v.iter().map(|x| x * x).sum();
            let mut accepted = false;
            if sv > 1e-300 {
                let alpha = (-(sr / sv).sqrt()).min(-1.0);
                let mut pn = vec![0.0f64; len];
                for k in 0..len {
                    pn[k] = params[k] - 2.0 * alpha * r[k] + alpha * alpha * v[k];
                }
                project(&mut pn);
                let (an, bn, sn) = unpack(&pn);
                let (lc, nic, ric, suc, _) = full_estep(&ctx, &an, &bn, &sn);
                // Accept only if not worse than the cycle start (=> monotone after one
                // stabilizing M-step); else fall back to the two plain EM steps.
                if lc.is_finite() && lc >= l0 {
                    let (a3, b3, s3) = m_step(&ctx, &an, &bn, &sn, &nic, &ric, &suc, &multi, fix_slope, cfg);
                    params = pack(&a3, &b3, &s3);
                    accepted = true;
                }
            }
            if !accepted {
                params = p2;
            }
            n_iter += 2;
        }
        let (fa, fb, fs) = unpack(&params);
        a = fa;
        beta = fb;
        sigma2 = fs;
    } else {
        while n_iter < cfg.max_iter {
            let (l0, ni, ri, su, _) = full_estep(&ctx, &a, &beta, &sigma2);
            loglik_trace.push(l0);
            n_iter += 1;
            if loglik_trace.len() > 1 {
                let k = loglik_trace.len();
                if (l0 - loglik_trace[k - 2]).abs() < cfg.tol {
                    converged = true;
                    break;
                }
            }
            let (na, nb, ns) = m_step(&ctx, &a, &beta, &sigma2, &ni, &ri, &su, &multi, fix_slope, cfg);
            a = na;
            beta = nb;
            sigma2 = ns;
        }
    }

    // Final pass at the returned params: theta EAP + final loglik.
    let (final_ll, _, _, _, theta) = full_estep(&ctx, &a, &beta, &sigma2);
    if !converged {
        loglik_trace.push(final_ll);
    }

    let b: Vec<f64> = (0..j).map(|i| -beta[i] / a[i]).collect();
    let k = if fix_slope { 1 } else { 2 };
    // Only FREELY-estimated testlet variances count: singletons are pinned to 0
    // (non-identified) and estimate_sigma=false fixes every variance.
    let n_free_sigma = if cfg.estimate_sigma { multi.iter().filter(|&&m| m).count() } else { 0 };
    Ok(TestletResult {
        model,
        a,
        b,
        beta,
        sigma2,
        theta,
        loglik_trace,
        n_iter,
        converged,
        n_parameters: k * j + n_free_sigma,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmle::{fit_mmle_2pl, MmleConfig};

    struct Lcg(u64);
    impl Lcg {
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
            -(self.next_f64().max(1e-12)).ln() - 1.0 // Exp(1)-1: mean 0, var 1
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
    fn bias(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        a.iter().zip(b).map(|(x, y)| x - y).sum::<f64>() / n
    }
    fn nondecreasing(t: &[f64]) -> bool {
        t.windows(2).all(|w| w[1] >= w[0] - 1e-6)
    }

    /// The gamma quadrature must be the standard normal (unit variance) or the
    /// sigma^2 = sigma^2 * mean(E[u^2]) update converges to a biased fixed point.
    #[test]
    fn gh_rule_is_unit_normal() {
        for &q in &[11usize, 15, 21, 31, 41] {
            let (u, v) = gh_rule(q).unwrap();
            assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-9);
            assert!(u.iter().zip(v).map(|(x, w)| x * w).sum::<f64>().abs() < 1e-9);
            let m2: f64 = u.iter().zip(v).map(|(x, w)| x * x * w).sum();
            assert!((m2 - 1.0).abs() < 1e-6, "gh_rule({q}) E[u^2] = {m2}");
        }
    }

    /// Contiguous testlet assignment: testlet d owns items [d*size .. (d+1)*size).
    fn contiguous_testlets(n_items: usize, n_testlets: usize) -> Vec<usize> {
        let per = n_items / n_testlets;
        (0..n_items).map(|i| (i / per).min(n_testlets - 1)).collect()
    }

    /// Simulate testlet data: draw theta, per-testlet gamma ~ N(0, sigma^2_d), responses.
    fn simulate(
        a: &[f64],
        beta: &[f64],
        sigma2: &[f64],
        testlet_id: &[usize],
        n: usize,
        j: usize,
        skew: bool,
        rng: &mut Lcg,
    ) -> Vec<f64> {
        let d_n = sigma2.len();
        let mut y = vec![0.0f64; n * j];
        for p in 0..n {
            let theta = if skew { rng.skew() } else { rng.normal() };
            let gamma: Vec<f64> = (0..d_n).map(|d| sigma2[d].sqrt() * rng.normal()).collect();
            for i in 0..j {
                let eta = a[i] * theta + beta[i] - a[i] * gamma[testlet_id[i]];
                y[p * j + i] = rng.bern(sigmoid_stable(eta));
            }
        }
        y
    }

    /// PRIMARY anchor: sigma^2 pinned to 0 reduces to fit_mmle_2pl (a/beta/loglik match).
    #[test]
    fn testlet_sigma0_equals_fit_mmle_2pl() {
        let (n, j, d_n) = (700usize, 12usize, 3usize);
        let tid = contiguous_testlets(j, d_n);
        let mut rng = Lcg(7);
        let a_t: Vec<f64> = (0..j).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
        let beta_t: Vec<f64> = (0..j).map(|i| -1.2 + 2.4 * i as f64 / (j - 1) as f64).collect();
        let y = simulate(&a_t, &beta_t, &vec![0.0; d_n], &tid, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let mcfg = MmleConfig { max_iter: 80, tol: 0.0, ridge_a: 1e-3, ridge_b: 1e-3, newton_iter: 25 };
        let mmle = fit_mmle_2pl(&y, &observed, n, j, &mcfg);
        let cfg = TestletConfig {
            max_iter: 80, tol: 0.0, q_gamma: 21, ridge_a: 1e-3, ridge_b: 1e-3,
            newton_iter: 25, estimate_sigma: false, init_sigma2: 0.0,
        };
        let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::TwoPl, &cfg).unwrap();
        // a/beta bit-exact; theta OMITTED (mmle EAP uses a stale posterior — same reason
        // the mixture/lltm anchors assert only item params).
        assert!(rmse(&res.a, &mmle.a) < 1e-12, "a rmse {}", rmse(&res.a, &mmle.a));
        assert!(rmse(&res.beta, &mmle.b) < 1e-12, "beta rmse {}", rmse(&res.beta, &mmle.b));
        // loglik agrees on the common prefix (testlet may push an extra final_ll).
        assert!(
            res.loglik_trace.iter().zip(&mmle.loglik_trace).all(|(x, y)| (x - y).abs() < 1e-12),
            "loglik prefix mismatch"
        );
        assert_eq!(res.n_parameters, 2 * j); // sigma^2 fixed => 0 free variance params
        assert!(res.sigma2.iter().all(|&s| s == 0.0));
    }

    /// No-spurious-LD: pure 2PL data (all true sigma^2=0), full fit must not invent LD.
    /// Ignored by default: shrinking sigma^2 to ~0 needs many iterations (the sigma->0
    /// tail of the variance-component EM is slow even with SQUAREM).
    #[test]
    #[ignore = "slow (sigma->0 convergence); run with: cargo test --release -- --ignored"]
    fn testlet_no_spurious_ld() {
        let (n, j, d_n) = (600usize, 12usize, 3usize);
        let tid = contiguous_testlets(j, d_n);
        let mut rng = Lcg(11);
        let a_t: Vec<f64> = (0..j).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
        let beta_t: Vec<f64> = (0..j).map(|i| -1.5 + 3.0 * i as f64 / (j - 1) as f64).collect();
        let y = simulate(&a_t, &beta_t, &vec![0.0; d_n], &tid, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let cfg = TestletConfig { max_iter: 2000, ..TestletConfig::default() };
        let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::TwoPl, &cfg).unwrap();
        println!("no_spurious: converged={} n_iter={} sigma2={:?}", res.converged, res.n_iter, res.sigma2);
        assert!(nondecreasing(&res.loglik_trace));
        assert!(res.sigma2.iter().all(|&s| s < 0.08), "spurious LD: {:?}", res.sigma2);
    }

    /// Strong-LD: large true sigma^2 recovered, and modeling it improves the loglik over
    /// the sigma=0 (naive-2PL) fit — the signature of unmodeled local dependence.
    #[test]
    fn testlet_recovers_strong_ld() {
        // Rasch (a=1), 8 items per testlet (the well-identified testlet model; the 2PL
        // discrimination trades off against the testlet SD via a_i*sigma_d).
        let (n, j, d_n) = (800usize, 16usize, 2usize);
        let tid = contiguous_testlets(j, d_n);
        let sig2 = vec![0.6f64, 0.3];
        let mut rng = Lcg(2024);
        let a_t = vec![1.0f64; j];
        let beta_t: Vec<f64> = (0..j).map(|i| -1.5 + 3.0 * (i % 8) as f64 / 7.0).collect();
        let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &TestletConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(rmse(&res.sigma2, &sig2) < 0.2, "sigma2 rmse {} ({:?})", rmse(&res.sigma2, &sig2), res.sigma2);
        assert!(res.sigma2[0] > 0.35, "strong LD not recovered: {}", res.sigma2[0]);
        // loglik gain over the naive sigma=0 fit
        let naive = TestletConfig { estimate_sigma: false, init_sigma2: 0.0, ..TestletConfig::default() };
        let res0 = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &naive).unwrap();
        assert!(
            *res.loglik_trace.last().unwrap() > *res0.loglik_trace.last().unwrap() + 5.0,
            "testlet fit did not improve loglik over naive 2PL"
        );
    }

    /// A singleton testlet's variance is non-identified => pinned to 0, not spurious.
    #[test]
    fn testlet_singleton_pinned() {
        let (n, j) = (600usize, 7usize);
        // testlets: {0,1,2}, {3,4,5}, {6} (singleton)
        let tid = vec![0usize, 0, 0, 1, 1, 1, 2];
        let sig2 = vec![0.6f64, 0.6, 0.0];
        let mut rng = Lcg(5);
        let a_t = vec![1.0f64; j];
        let beta_t: Vec<f64> = (0..j).map(|i| -1.0 + 2.0 * i as f64 / (j - 1) as f64).collect();
        let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let res = fit_testlet(&y, &observed, &tid, n, j, 3, TestletModel::Rasch, &TestletConfig::default()).unwrap();
        assert!(res.converged);
        assert_eq!(res.sigma2[2], 0.0, "singleton testlet variance must be pinned to 0");
        // the singleton's pinned variance is NOT a free parameter (Rasch: J + 2 multi).
        assert_eq!(res.n_parameters, j + 2);
    }

    /// Missing-at-random cells are dropped.
    #[test]
    fn testlet_handles_missing_data() {
        let (n, j, d_n) = (500usize, 12usize, 3usize);
        let tid = contiguous_testlets(j, d_n);
        let sig2 = vec![0.5f64, 0.5, 0.5];
        let mut rng = Lcg(9);
        let a_t = vec![1.0f64; j];
        let beta_t: Vec<f64> = (0..j).map(|i| -1.0 + 2.0 * i as f64 / (j - 1) as f64).collect();
        let y = simulate(&a_t, &beta_t, &sig2, &tid, n, j, false, &mut rng);
        let mut observed = vec![true; n * j];
        for o in observed.iter_mut() {
            if rng.next_f64() < 0.2 {
                *o = false;
            }
        }
        let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &TestletConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
    }

    /// Malformed inputs are rejected (covers each validate branch, incl. tol=0 allowed).
    #[test]
    fn testlet_validate_rejects_malformed() {
        let (n, j, d_n) = (5usize, 6usize, 2usize);
        let tid = contiguous_testlets(j, d_n);
        let y = vec![0.0f64; n * j];
        let obs = vec![true; n * j];
        let d = TestletConfig::default();
        let bad = |y: &[f64], obs: &[bool], tid: &[usize], n, j, dn, cfg: &TestletConfig| {
            fit_testlet(y, obs, tid, n, j, dn, TestletModel::Rasch, cfg).is_err()
        };
        assert!(bad(&y, &obs, &tid, 0, j, d_n, &d)); // n_persons
        assert!(bad(&y, &obs, &tid, n, j, 0, &d)); // n_testlets
        assert!(bad(&y, &obs, &tid, n, j, d_n, &TestletConfig { max_iter: 0, ..d }));
        assert!(bad(&y, &obs, &tid, n, j, d_n, &TestletConfig { tol: -1.0, ..d }));
        assert!(bad(&y, &obs, &tid, n, j, d_n, &TestletConfig { q_gamma: 8, ..d })); // not in SUPPORTED_Q
        assert!(bad(&y, &obs, &tid, n, j, d_n, &TestletConfig { init_sigma2: -1.0, ..d }));
        assert!(bad(&vec![0.0; n * j - 1], &obs, &tid, n, j, d_n, &d)); // y length
        assert!(bad(&y, &obs, &vec![0usize; j - 1], n, j, d_n, &d)); // testlet_id length
        assert!(bad(&y, &obs, &vec![0, 0, 0, 5, 0, 0], n, j, d_n, &d)); // testlet_id out of range
        assert!(bad(&vec![2.0; n * j], &obs, &tid, n, j, d_n, &d)); // y not 0/1
        // an empty testlet (n_testlets says 3 but only 0,1 used)
        assert!(bad(&y, &obs, &vec![0, 0, 0, 1, 1, 1], n, j, 3, &d));
        // tol == 0.0 accepted
        assert!(fit_testlet(&y, &obs, &tid, n, j, d_n, TestletModel::Rasch, &TestletConfig { tol: 0.0, max_iter: 2, ..d }).is_ok());
    }

    /// Literature-grade Monte-Carlo (>=500 reps): Bradlow-Wainer-Wang-style design.
    /// Uses the RASCH testlet (the well-identified case; in the 2PL testlet the free
    /// discrimination a_i and the testlet SD sigma_d both scale the LD via a_i*sigma_d
    /// and separate only weakly with few testlets). Recovers the testlet variances and
    /// item difficulties under normal and skew ability.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_testlet_recovery_500() {
        let (n, j, d_n, per, reps) = (1000usize, 24usize, 4usize, 6usize, 500usize);
        let tid = contiguous_testlets(j, d_n);
        let sig2_t = vec![0.2f64, 0.4, 0.6, 0.8];
        assert_eq!(j, d_n * per);
        let a_t = vec![1.0f64; j];
        let cfg = TestletConfig { q_gamma: 15, max_iter: 1500, ..TestletConfig::default() };
        for &skew in [false, true].iter() {
            let (mut s_b, mut s_sig, mut s_bsig, mut n_conv) = (0.0, 0.0, 0.0, 0.0);
            for rep in 0..reps {
                let seed = 0xBADC0FFEE0DDF00Du64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add(if skew { 0x9E3779B97F4A7C15 } else { 0 });
                let mut rng = Lcg(seed);
                let beta_t: Vec<f64> = (0..j).map(|i| -1.5 + 3.0 * (i % per) as f64 / (per - 1) as f64).collect();
                let y = simulate(&a_t, &beta_t, &sig2_t, &tid, n, j, skew, &mut rng);
                let observed = vec![true; n * j];
                let res = fit_testlet(&y, &observed, &tid, n, j, d_n, TestletModel::Rasch, &cfg).unwrap();
                s_b += rmse(&res.beta, &beta_t);
                s_sig += rmse(&res.sigma2, &sig2_t);
                s_bsig += bias(&res.sigma2, &sig2_t);
                if res.converged {
                    n_conv += 1.0;
                }
            }
            let r = reps as f64;
            println!(
                "skew={}: RMSE(beta)={:.4} RMSE(sigma2)={:.4} bias(sigma2)={:.4} converged={:.2}",
                skew, s_b / r, s_sig / r, s_bsig / r, n_conv / r
            );
            assert!(s_b / r < 0.12, "RMSE(beta) {} skew={skew}", s_b / r);
            assert!(s_sig / r < 0.15, "RMSE(sigma2) {} skew={skew}", s_sig / r);
        }
    }
}
