//! Linear Logistic Test Model (LLTM; Fischer, 1973): an *explanatory* Rasch model in
//! which the `J` item difficulties are not free but a fixed linear image of `K` basic
//! (cognitive-operation) parameters through a known weight matrix `Q` (`J x K`,
//! `q_ik` = how many times operation `k` is engaged by item `i`):
//!
//! ```text
//! b_i = c + sum_k q_ik * eta_k,   P(x_ij = 1 | theta_j) = sigmoid(theta_j + b_i)
//! ```
//!
//! with an optional free normalization constant `c` (grand-mean easiness). With
//! `K << J` parameters, LLTM tests whether a small set of cognitive operations
//! *explains* the item difficulties; the likelihood-ratio test against the saturated
//! Rasch model (2·(ll_Rasch − ll_LLTM) ~ χ²(J − K − intercept)) is its classic use.
//!
//! Sign convention: this crate is EASINESS-additive (`eta = a·theta + b` with `a ≡ 1`,
//! consistent with `mmle`/`mixture`), so `eta_k`/`b_i` are easinesses and are directly
//! comparable across the crate. Fischer's classic difficulty form
//! `P = sigmoid(theta − beta)` is a one-line reporting transform: operation difficulty
//! `delta_k = -eta_k`, item difficulty `beta_i = -b_i`.
//!
//! Identification. Two indeterminacies, both resolved: (1) location — `theta ~ N(0,1)`
//! pins the metric directly (no eta-normalization needed), so `c` is estimable as the
//! grand-mean easiness; (2) linear-form injectivity — `eta` is identified iff the map
//! `eta -> Q eta` is injective, i.e. `Q` (and the augmented `[1 | Q]` when the intercept
//! is on) has full COLUMN rank. `validate` rejects a rank-deficient design (e.g. a Q
//! whose rows sum to a constant, which collides with the intercept) rather than letting
//! the Newton ridge paper over a non-identified model.
//!
//! Provenance: Fischer's (1973, 1995) canonical LLTM is conditional-ML (person-free).
//! This is the marginal-ML / `N(0,1)` operationalization mapping onto the crate's
//! Bock-Aitkin infrastructure — the same item contrasts under a different location
//! convention, and it yields the exact `Q = I` Rasch reduction.
//!
//! Deferred (non-goals): conditional-ML estimation, LLTM for 2PL/polytomous models,
//! LLRA / random-weights extensions, person-side covariate design, analytic SE(eta)
//! (the `K x K` `-H_eta` at the fixed point is the observed-information block).
//!
//! References (APA 7th ed.):
//! - Fischer, G. H. (1973). The linear logistic test model as an instrument in
//!   educational research. *Acta Psychologica, 37*(6), 359-374.
//!   <https://doi.org/10.1016/0001-6918(73)90003-6>
//! - Fischer, G. H. (1995). The linear logistic test model. In G. H. Fischer & I. W.
//!   Molenaar (Eds.), *Rasch models: Foundations, recent developments, and
//!   applications* (pp. 131-155). Springer. <https://doi.org/10.1007/978-1-4612-4230-7_8>
//! - Bock, R. D., & Aitkin, M. (1981). Marginal maximum likelihood estimation of item
//!   parameters. *Psychometrika, 46*(4), 443-459. <https://doi.org/10.1007/BF02293801>

use crate::fitstats::chi2_sf;
use crate::mmle::{log_sigmoid, sigmoid_stable, GH_NODES, GH_WEIGHTS};

/// Solve `H x = g` for small dense `H` by Gauss-Jordan with partial pivoting; returns
/// `None` on a singular system. Same arithmetic as `poly::solve_small` (so a diagonal
/// `H` — the `Q = I` case — yields `g[i]/h[i][i]` bit-exactly), but the Newton M-step
/// breaks on `None` rather than taking `poly::solve_small`'s gradient-direction
/// fallback, which would be downhill for this maximization (matches `mmle`/`mixture`).
fn solve_small_checked(mut h: Vec<Vec<f64>>, mut g: Vec<f64>) -> Option<Vec<f64>> {
    let n = g.len();
    for col in 0..n {
        let mut piv = col;
        for r in col + 1..n {
            if h[r][col].abs() > h[piv][col].abs() {
                piv = r;
            }
        }
        if h[piv][col].abs() < 1e-12 {
            return None;
        }
        h.swap(col, piv);
        g.swap(col, piv);
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = h[r][col] / h[col][col];
            for c in col..n {
                h[r][c] -= f * h[col][c];
            }
            g[r] -= f * g[col];
        }
    }
    Some((0..n).map(|i| g[i] / h[i][i]).collect())
}

/// EM configuration for the LLTM estimator.
#[derive(Clone, Copy, Debug)]
pub struct LltmConfig {
    pub max_iter: usize,
    /// Convergence tolerance on `|delta loglik|`; `0.0` is permitted (runs the full
    /// `max_iter`) — required for the exact `Q = I` reduction anchor.
    pub tol: f64,
    /// Ridge on the basic parameters (MAP `N(0, 1/ridge)`); equals the `mmle`/`mixture`
    /// `ridge_b` at `Q = I`.
    pub ridge: f64,
    /// Inner Newton steps per M-step.
    pub newton_iter: usize,
    /// Fit a free grand-mean easiness intercept `c` (prepended ones-column of the design).
    pub fit_intercept: bool,
    /// Also run the `Q = I` Rasch fit and report the LR test of LLTM vs Rasch.
    pub compute_lr: bool,
}

impl Default for LltmConfig {
    fn default() -> Self {
        Self { max_iter: 500, tol: 1e-6, ridge: 1e-3, newton_iter: 25, fit_intercept: true, compute_lr: true }
    }
}

/// Fitted LLTM. `eta` are the basic-operation easinesses (Fischer difficulty = `-eta`);
/// `b` the induced item easinesses `c + Q eta`.
#[derive(Clone, Debug)]
pub struct LltmResult {
    pub eta: Vec<f64>,
    /// Grand-mean easiness `c` (`NaN` when `fit_intercept == false`).
    pub intercept: f64,
    pub b: Vec<f64>,
    pub theta: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `K + fit_intercept`.
    pub n_parameters: usize,
    /// Marginal loglik of the `Q = I` Rasch fit (`NaN` if `!compute_lr`).
    pub loglik_rasch: f64,
    /// `2·(ll_Rasch − ll_LLTM)`, floored at 0 (`NaN` if `!compute_lr`).
    pub lr_stat: f64,
    /// `J − K − fit_intercept`.
    pub lr_df: usize,
    /// LR p-value `chi2_sf(lr_stat, lr_df)` (`NaN` if `!compute_lr` or `lr_df == 0`).
    pub lr_p: f64,
}

/// Build the effective design `D` (`J x M`): `[1 | Q]` when `fit_intercept`, else `Q`.
fn build_design(q_design: &[f64], n_items: usize, n_basic: usize, fit_intercept: bool) -> (Vec<f64>, usize) {
    let intc = fit_intercept as usize;
    let m = n_basic + intc;
    let mut d = vec![0.0f64; n_items * m];
    for i in 0..n_items {
        if fit_intercept {
            d[i * m] = 1.0;
        }
        for k in 0..n_basic {
            d[i * m + intc + k] = q_design[i * n_basic + k];
        }
    }
    (d, m)
}

/// Pivoted Gaussian elimination on the `M x M` Gram: true iff every pivot exceeds
/// `thresh` (i.e. the design has full column rank).
fn gram_full_rank(g: &mut [Vec<f64>], m: usize, thresh: f64) -> bool {
    for col in 0..m {
        let mut piv = col;
        for r in col + 1..m {
            if g[r][col].abs() > g[piv][col].abs() {
                piv = r;
            }
        }
        if g[piv][col].abs() < thresh {
            return false;
        }
        g.swap(col, piv);
        for r in col + 1..m {
            let f = g[r][col] / g[col][col];
            for c in col..m {
                g[r][c] -= f * g[col][c];
            }
        }
    }
    true
}

fn validate(
    y: &[f64],
    observed: &[bool],
    q_design: &[f64],
    n_persons: usize,
    n_items: usize,
    n_basic: usize,
    cfg: &LltmConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 || n_basic < 1 {
        return Err("n_persons, n_items and n_basic must be >= 1".into());
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if cfg.newton_iter == 0 {
        return Err("newton_iter must be positive".into());
    }
    // tol == 0.0 is allowed (runs the full max_iter; needed for the Q=I anchor).
    if !cfg.tol.is_finite() || cfg.tol < 0.0 {
        return Err("tol must be finite and non-negative".into());
    }
    if !cfg.ridge.is_finite() || cfg.ridge < 0.0 {
        return Err("ridge must be finite and non-negative".into());
    }
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_q = n_items
        .checked_mul(n_basic)
        .ok_or_else(|| "n_items * n_basic overflows usize".to_string())?;
    if q_design.len() != n_q {
        return Err("q_design must have length n_items * n_basic".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && v != 0.0 && v != 1.0 {
            return Err(format!("y[{idx}] must be 0 or 1 where observed; got {v}"));
        }
    }
    for &v in q_design.iter() {
        if !v.is_finite() {
            return Err("q_design entries must be finite".into());
        }
    }
    for i in 0..n_items {
        if !(0..n_persons).any(|p| observed[p * n_items + i]) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    // Full-column-rank check on the effective design (identification is a design
    // property, checked here — not papered over by the Newton ridge).
    let (d, m) = build_design(q_design, n_items, n_basic, cfg.fit_intercept);
    if m > n_items {
        return Err(format!("design has more columns ({m}) than items ({n_items}); eta not identified"));
    }
    for a in 0..m {
        if (0..n_items).all(|i| d[i * m + a] == 0.0) {
            return Err(format!("design column {a} is all-zero"));
        }
    }
    let mut gram = vec![vec![0.0f64; m]; m];
    let mut maxg = 0.0f64;
    for a in 0..m {
        for cc in 0..m {
            let mut s = 0.0;
            for i in 0..n_items {
                s += d[i * m + a] * d[i * m + cc];
            }
            gram[a][cc] = s;
            maxg = maxg.max(s.abs());
        }
    }
    if !gram_full_rank(&mut gram, m, 1e-9 * maxg.max(1e-300)) {
        return Err("design matrix (with intercept) is column-rank-deficient; eta is not identified".into());
    }
    Ok(())
}

/// Rasch easiness init: `b_i = logit(clamp(item proportion, 0.02, 0.98))` (identical to
/// `mmle`/`mixture`), load-bearing for the exact `Q = I` reduction.
fn init_b(y: &[f64], observed: &[bool], n_persons: usize, n_items: usize) -> Vec<f64> {
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

/// Induced item easinesses `b = D · params`.
fn induced_b(design: &[f64], m: usize, n_items: usize, params: &[f64]) -> Vec<f64> {
    (0..n_items)
        .map(|i| (0..m).map(|a| design[i * m + a] * params[a]).sum())
        .collect()
}

/// Least-squares projection of `b_init` onto the design column space: solve
/// `(DᵀD) params = Dᵀ b_init`. At `Q = I` the Gram is the identity so `params = b_init`.
fn ls_project(design: &[f64], m: usize, n_items: usize, b_init: &[f64]) -> Vec<f64> {
    let mut gram = vec![vec![0.0f64; m]; m];
    let mut rhs = vec![0.0f64; m];
    for a in 0..m {
        for i in 0..n_items {
            rhs[a] += design[i * m + a] * b_init[i];
        }
        for cc in 0..=a {
            let mut s = 0.0;
            for i in 0..n_items {
                s += design[i * m + a] * design[i * m + cc];
            }
            gram[a][cc] = s;
            gram[cc][a] = s;
        }
    }
    // The Gram is non-singular under a validated full-rank design; fall back to the
    // Rasch init if it ever is not.
    solve_small_checked(gram, rhs).unwrap_or_else(|| b_init.to_vec())
}

/// Log response tables for all (node, item): `log_p1 = log σ(θ_q + b_i)`.
fn build_log_tables(b: &[f64], n_items: usize, log_p1: &mut [f64], log_p0: &mut [f64]) {
    for (qi, &node) in GH_NODES.iter().enumerate() {
        for i in 0..n_items {
            let eta = node + b[i];
            log_p1[qi * n_items + i] = log_sigmoid(eta);
            log_p0[qi * n_items + i] = log_sigmoid(-eta);
        }
    }
}

/// Fill `post[0..Q]` with person `p`'s node posterior; returns `ln P(x_p)`.
#[allow(clippy::too_many_arguments)]
fn person_posterior(
    p: usize,
    y: &[f64],
    observed: &[bool],
    n_items: usize,
    q: usize,
    log_w: &[f64],
    log_p1: &[f64],
    log_p0: &[f64],
    post: &mut [f64],
) -> f64 {
    for (qi, slot) in post.iter_mut().enumerate().take(q) {
        let mut acc = log_w[qi];
        for i in 0..n_items {
            let idx = p * n_items + i;
            if observed[idx] {
                let yy = y[idx];
                acc += yy * log_p1[qi * n_items + i] + (1.0 - yy) * log_p0[qi * n_items + i];
            }
        }
        *slot = acc;
    }
    let mx = post[..q].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut denom = 0.0;
    for &v in post[..q].iter() {
        denom += (v - mx).exp();
    }
    for v in post[..q].iter_mut() {
        *v = (*v - mx).exp() / denom;
    }
    mx + denom.ln()
}

/// One M-step: `newton_iter` chain-rule Newton steps on `params` from the fixed
/// expected counts. `g = Dᵀ g_b`, `H = Dᵀ diag(h_b) D + ridge`, `params -= H⁻¹ g`. At
/// `Q = I` `H` is diagonal and this is exactly the per-item Rasch Newton.
#[allow(clippy::too_many_arguments)]
fn newton_mstep(
    design: &[f64],
    m: usize,
    n_items: usize,
    q: usize,
    n_iq: &[f64],
    r_iq: &[f64],
    mut params: Vec<f64>,
    ridge: f64,
    newton_iter: usize,
) -> Vec<f64> {
    for _ in 0..newton_iter {
        let b = induced_b(design, m, n_items, &params);
        let mut g_b = vec![0.0f64; n_items];
        let mut h_b = vec![0.0f64; n_items];
        for i in 0..n_items {
            for qi in 0..q {
                let p = sigmoid_stable(GH_NODES[qi] + b[i]);
                let n = n_iq[i * q + qi];
                let w = n * p * (1.0 - p);
                g_b[i] += r_iq[i * q + qi] - n * p;
                h_b[i] -= w;
            }
        }
        let mut g = vec![0.0f64; m];
        for a in 0..m {
            for i in 0..n_items {
                g[a] += design[i * m + a] * g_b[i];
            }
        }
        let mut h = vec![vec![0.0f64; m]; m];
        for a in 0..m {
            for cc in 0..=a {
                let mut s = 0.0;
                for i in 0..n_items {
                    s += design[i * m + a] * design[i * m + cc] * h_b[i];
                }
                h[a][cc] = s;
                h[cc][a] = s;
            }
        }
        for a in 0..m {
            g[a] -= ridge * params[a];
            h[a][a] -= ridge;
        }
        // Break on a singular Hessian (matches mmle/mixture): a gradient-direction
        // fallback step would be downhill for this maximization.
        let delta = match solve_small_checked(h, g) {
            Some(d) => d,
            None => break,
        };
        let mut maxd = 0.0f64;
        for a in 0..m {
            params[a] -= delta[a];
            maxd = maxd.max(delta[a].abs());
        }
        if maxd < 1e-8 {
            break;
        }
    }
    params
}

/// Marginal-EM Rasch fit over a fixed design `D` (`J x M`, row-major). Returns
/// `(params, b, theta, loglik_trace, n_iter, converged)`. Mirrors `mixture::run_em`:
/// convergence is checked before the M-step; a final-E-step loglik is pushed on a
/// max-iter exit so the returned params match the trace endpoint.
fn run_em_lltm(
    y: &[f64],
    observed: &[bool],
    design: &[f64],
    n_persons: usize,
    n_items: usize,
    m: usize,
    cfg: &LltmConfig,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize, bool) {
    let q = GH_NODES.len();
    let log_w: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();
    let b_init = init_b(y, observed, n_persons, n_items);
    let mut params = ls_project(design, m, n_items, &b_init);

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut post = vec![0.0f64; q];
    let mut log_p1 = vec![0.0f64; q * n_items];
    let mut log_p0 = vec![0.0f64; q * n_items];

    for _ in 0..cfg.max_iter {
        let b = induced_b(design, m, n_items, &params);
        build_log_tables(&b, n_items, &mut log_p1, &mut log_p0);
        let mut n_iq = vec![0.0f64; n_items * q];
        let mut r_iq = vec![0.0f64; n_items * q];
        let mut total_ll = 0.0;
        for p in 0..n_persons {
            total_ll += person_posterior(p, y, observed, n_items, q, &log_w, &log_p1, &log_p0, &mut post);
            for i in 0..n_items {
                let idx = p * n_items + i;
                if observed[idx] {
                    let yy = y[idx];
                    for qi in 0..q {
                        let pv = post[qi];
                        n_iq[i * q + qi] += pv;
                        r_iq[i * q + qi] += yy * pv;
                    }
                }
            }
        }
        loglik_trace.push(total_ll);
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }
        params = newton_mstep(design, m, n_items, q, &n_iq, &r_iq, params, cfg.ridge, cfg.newton_iter);
        n_iter += 1;
    }

    // Final pass at the converged params: theta EAP + final loglik.
    let b = induced_b(design, m, n_items, &params);
    build_log_tables(&b, n_items, &mut log_p1, &mut log_p0);
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0;
    for p in 0..n_persons {
        final_ll += person_posterior(p, y, observed, n_items, q, &log_w, &log_p1, &log_p0, &mut post);
        theta[p] = (0..q).map(|qi| post[qi] * GH_NODES[qi]).sum();
    }
    if !converged {
        loglik_trace.push(final_ll);
    }
    (params, b, theta, loglik_trace, n_iter, converged)
}

/// Fit the Linear Logistic Test Model (Fischer, 1973) by marginal EM. `y`/`observed`
/// are row-major `N*J` (`y` in {0,1}); `q_design` is row-major `J*K` (real weights).
/// Missing cells (`observed == false`) are dropped (MAR). When `cfg.compute_lr`, the
/// `Q = I` Rasch fit is run too and the LR test of LLTM vs Rasch is reported.
pub fn fit_lltm(
    y: &[f64],
    observed: &[bool],
    q_design: &[f64],
    n_persons: usize,
    n_items: usize,
    n_basic: usize,
    cfg: &LltmConfig,
) -> Result<LltmResult, String> {
    validate(y, observed, q_design, n_persons, n_items, n_basic, cfg)?;
    let intc = cfg.fit_intercept as usize;
    let (design, m) = build_design(q_design, n_items, n_basic, cfg.fit_intercept);

    let (params, b, theta, loglik_trace, n_iter, converged) =
        run_em_lltm(y, observed, &design, n_persons, n_items, m, cfg);
    let ll_lltm = *loglik_trace.last().unwrap();
    let intercept = if cfg.fit_intercept { params[0] } else { f64::NAN };
    let eta = params[intc..].to_vec();

    let lr_df = n_items - n_basic - intc;
    let (loglik_rasch, lr_stat, lr_p) = if cfg.compute_lr {
        // Rasch reference = LLTM(Q = I_J, no intercept) through the same engine.
        let mut id = vec![0.0f64; n_items * n_items];
        for i in 0..n_items {
            id[i * n_items + i] = 1.0;
        }
        let rcfg = LltmConfig { fit_intercept: false, compute_lr: false, ..*cfg };
        let (.., rtrace, _, _) = run_em_lltm(y, observed, &id, n_persons, n_items, n_items, &rcfg);
        let ll_r = *rtrace.last().unwrap();
        let stat = (2.0 * (ll_r - ll_lltm)).max(0.0);
        (ll_r, stat, chi2_sf(stat, lr_df as f64))
    } else {
        (f64::NAN, f64::NAN, f64::NAN)
    };

    Ok(LltmResult {
        eta,
        intercept,
        b,
        theta,
        loglik_trace,
        n_iter,
        converged,
        n_parameters: n_basic + intc,
        loglik_rasch,
        lr_stat,
        lr_df,
        lr_p,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mixture::{fit_mixture, MixtureConfig, MixtureModel};

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
            -(self.next_f64().max(1e-12)).ln() - 1.0 // Exp(1) - 1: mean 0, var 1
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
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }
    fn nondecreasing(t: &[f64]) -> bool {
        t.windows(2).all(|w| w[1] >= w[0] - 1e-6)
    }

    /// A full-column-rank integer design whose rows do NOT sum to a constant (so the
    /// intercept is identified): column `k` cycles with period `k + 2`, so the columns
    /// have distinct fundamental frequencies (independent of each other and of the
    /// constant intercept) and the row sums genuinely vary.
    fn make_q(n_items: usize, n_basic: usize) -> Vec<f64> {
        let mut q = vec![0.0f64; n_items * n_basic];
        for i in 0..n_items {
            for k in 0..n_basic {
                q[i * n_basic + k] = ((i + k) % (k + 2)) as f64;
            }
        }
        q
    }

    fn simulate(
        design_b: &[f64], // induced true b per item
        n_persons: usize,
        n_items: usize,
        skew: bool,
        rng: &mut TestRng,
    ) -> Vec<f64> {
        let mut y = vec![0.0f64; n_persons * n_items];
        for p in 0..n_persons {
            let theta = if skew { rng.skew() } else { rng.normal() };
            for i in 0..n_items {
                y[p * n_items + i] = rng.bern(sigmoid_stable(theta + design_b[i]));
            }
        }
        y
    }

    /// Anchor 1: at Q = I, one chain-rule M-step (fixed single Newton step) is
    /// BIT-IDENTICAL to J independent per-item Rasch Newton steps.
    #[test]
    fn lltm_qi_single_mstep_bit_exact() {
        let (n_items, q) = (10usize, GH_NODES.len());
        let mut id = vec![0.0f64; n_items * n_items];
        for i in 0..n_items {
            id[i * n_items + i] = 1.0;
        }
        // fabricate deterministic expected counts and an init
        let mut rng = TestRng(11);
        let (mut n_iq, mut r_iq) = (vec![0.0f64; n_items * q], vec![0.0f64; n_items * q]);
        for i in 0..n_items {
            for qi in 0..q {
                let n = 5.0 + 20.0 * rng.next_f64();
                n_iq[i * q + qi] = n;
                r_iq[i * q + qi] = n * rng.next_f64();
            }
        }
        let params0: Vec<f64> = (0..n_items).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
        let (ridge, nit) = (1e-3, 1usize);
        let joint = newton_mstep(&id, n_items, n_items, q, &n_iq, &r_iq, params0.clone(), ridge, nit);
        // per-item 1-D Rasch Newton, one step
        let mut per_item = params0.clone();
        for i in 0..n_items {
            let mut b = params0[i];
            let (mut g_b, mut h_bb) = (0.0, 0.0);
            for qi in 0..q {
                let p = sigmoid_stable(GH_NODES[qi] + b);
                let n = n_iq[i * q + qi];
                let w = n * p * (1.0 - p);
                g_b += r_iq[i * q + qi] - n * p;
                h_bb -= w;
            }
            g_b -= ridge * b;
            h_bb -= ridge;
            b -= g_b / h_bb;
            per_item[i] = b;
        }
        for i in 0..n_items {
            assert_eq!(joint[i], per_item[i], "item {i}: joint {} vs per-item {}", joint[i], per_item[i]);
        }
    }

    /// Anchor 2: full LLTM(Q = I, no intercept, tol = 0) equals a single-class Rasch fit.
    #[test]
    fn lltm_qi_equals_rasch_fit() {
        let (n, j) = (700usize, 12usize);
        let mut rng = TestRng(7);
        let b_true: Vec<f64> = (0..j).map(|i| -1.2 + 2.4 * i as f64 / (j - 1) as f64).collect();
        let y = simulate(&b_true, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let mut id = vec![0.0f64; j * j];
        for i in 0..j {
            id[i * j + i] = 1.0;
        }
        let cfg = LltmConfig { max_iter: 80, tol: 0.0, ridge: 1e-3, newton_iter: 25, fit_intercept: false, compute_lr: false };
        let l = fit_lltm(&y, &observed, &id, n, j, j, &cfg).unwrap();
        let mcfg = MixtureConfig { max_iter: 80, tol: 0.0, ridge_b: 1e-3, ..MixtureConfig::default() };
        let mix = fit_mixture(&y, &observed, n, j, 1, MixtureModel::Rasch, &mcfg).unwrap();
        assert!(rmse(&l.b, &mix.b) < 1e-10, "b rmse {}", rmse(&l.b, &mix.b));
        assert!(rmse(&l.eta, &l.b) < 1e-12); // eta == b at Q = I
        assert_eq!(l.n_parameters, j);
        assert_eq!(l.lr_df, 0);
    }

    /// EM ascent guard.
    #[test]
    fn lltm_loglik_nondecreasing() {
        let (n, j, k) = (300usize, 8usize, 3usize);
        let q = make_q(j, k);
        let (design, m) = build_design(&q, j, k, true);
        let params: Vec<f64> = vec![-0.2, 0.5, -0.3, 0.6];
        let b_true = induced_b(&design, m, j, &params);
        let mut rng = TestRng(3);
        let y = simulate(&b_true, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let res = fit_lltm(&y, &observed, &q, n, j, k, &LltmConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
    }

    /// Fast recovery sanity: recover the basic parameters and induced difficulties.
    #[test]
    fn recovers_lltm() {
        let (n, j, k) = (2000usize, 20usize, 5usize);
        let q = make_q(j, k);
        let (design, m) = build_design(&q, j, k, true);
        let eta_true = [0.5f64, -0.3, 0.8, -0.6, 0.4];
        let params: Vec<f64> = std::iter::once(-0.2).chain(eta_true.iter().copied()).collect();
        let b_true = induced_b(&design, m, j, &params);
        let mut rng = TestRng(2024);
        let y = simulate(&b_true, n, j, false, &mut rng);
        let observed = vec![true; n * j];
        let res = fit_lltm(&y, &observed, &q, n, j, k, &LltmConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(corr(&res.eta, &eta_true) > 0.95, "eta corr {}", corr(&res.eta, &eta_true));
        assert!(rmse(&res.b, &b_true) < 0.15, "b rmse {}", rmse(&res.b, &b_true));
        assert_eq!(res.n_parameters, k + 1);
        // LR should NOT reject when the LLTM restriction holds
        assert!(res.lr_p > 0.01, "LR falsely rejected true LLTM: p={}", res.lr_p);
        assert_eq!(res.lr_df, j - k - 1);
    }

    /// Malformed inputs are rejected (covers each validate branch, incl. rank deficiency).
    #[test]
    fn lltm_validate_rejects_malformed() {
        let (n, j, k) = (5usize, 4usize, 2usize);
        let q = make_q(j, k);
        let y = vec![0.0f64; n * j];
        let obs = vec![true; n * j];
        let d = LltmConfig::default();
        let bad = |y: &[f64], obs: &[bool], q: &[f64], n, j, k, cfg: &LltmConfig| {
            fit_lltm(y, obs, q, n, j, k, cfg).is_err()
        };
        assert!(bad(&y, &obs, &q, 0, j, k, &d)); // n_persons < 1
        assert!(bad(&y, &obs, &q, n, j, 0, &d)); // n_basic < 1
        assert!(bad(&y, &obs, &q, n, j, k, &LltmConfig { max_iter: 0, ..d }));
        assert!(bad(&y, &obs, &q, n, j, k, &LltmConfig { newton_iter: 0, ..d }));
        assert!(bad(&y, &obs, &q, n, j, k, &LltmConfig { tol: -1.0, ..d }));
        assert!(bad(&y, &obs, &q, n, j, k, &LltmConfig { ridge: -1.0, ..d }));
        assert!(bad(&vec![0.0; n * j - 1], &obs, &q, n, j, k, &d)); // y length
        assert!(bad(&y, &obs, &vec![0.0; j * k - 1], n, j, k, &d)); // q length
        assert!(bad(&vec![2.0; n * j], &obs, &q, n, j, k, &d)); // y not 0/1
        // rank-deficient design: a duplicated column (K=2, both columns identical)
        let q_dup = vec![1.0f64, 1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0];
        assert!(bad(&y, &obs, &q_dup, n, j, 2, &d));
        // an all-ones column with intercept on => [1|Q] rank-deficient
        let q_const = vec![1.0f64; j * 1];
        assert!(bad(&y, &obs, &q_const, n, j, 1, &LltmConfig { fit_intercept: true, ..d }));
        // an item with no observed responses
        let mut obs_gap = vec![true; n * j];
        for p in 0..n {
            obs_gap[p * j + 1] = false;
        }
        assert!(bad(&y, &obs_gap, &q, n, j, k, &d));
        // tol == 0.0 is accepted
        assert!(fit_lltm(&y, &obs, &q, n, j, k, &LltmConfig { tol: 0.0, max_iter: 2, ..d }).is_ok());
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recover the K basic parameters and
    /// induced difficulties under normal and skew ability, and validate the LR test
    /// (Type I when the LLTM restriction holds, power when it is violated off-model).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_lltm_recovery_500() {
        let (n, j, k, reps) = (1500usize, 30usize, 5usize, 500usize);
        let q = make_q(j, k);
        let (design, m) = build_design(&q, j, k, true);
        let eta_true = [0.6f64, -0.4, 0.9, -0.5, 0.3];
        let c_true = -0.2;
        let params_true: Vec<f64> = std::iter::once(c_true).chain(eta_true.iter().copied()).collect();
        let b_true = induced_b(&design, m, j, &params_true);
        // an off-model perturbation orthogonal to colspace([1|Q]) for the power condition
        let mut eps = vec![0.0f64; j];
        {
            // residual of a vector with a component OUTSIDE the design space after
            // projecting onto the design columns. `make_q`'s columns have periods 2..6,
            // so a period-7 pattern is guaranteed a nonzero residual (a genuinely
            // off-model violation, not one the design can already represent).
            let raw: Vec<f64> = (0..j).map(|i| (i % 7) as f64 - 3.0).collect();
            let proj = ls_project(&design, m, j, &raw);
            let fitted = induced_b(&design, m, j, &proj);
            for i in 0..j {
                eps[i] = raw[i] - fitted[i];
            }
            let nrm = (eps.iter().map(|e| e * e).sum::<f64>() / j as f64).sqrt();
            assert!(nrm > 0.05, "off-model perturbation is (near) in-design: nrm={nrm}");
            for e in eps.iter_mut() {
                *e = *e / nrm * 0.6; // scale the off-model violation to RMS 0.6
            }
        }

        for &skew in [false, true].iter() {
            let (mut sum_re, mut sum_be, mut sum_rb) = (0.0, 0.0, 0.0);
            let (mut type1, mut power) = (0.0, 0.0);
            for rep in 0..reps {
                let seed = 0xC0FFEE1234567u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add(if skew { 0x9E3779B97F4A7C15 } else { 0 });
                let mut rng = TestRng(seed);
                // null (LLTM holds)
                let y0 = simulate(&b_true, n, j, skew, &mut rng);
                let observed = vec![true; n * j];
                let res = fit_lltm(&y0, &observed, &q, n, j, k, &LltmConfig::default()).unwrap();
                sum_re += rmse(&res.eta, &eta_true);
                sum_be += bias(&res.eta, &eta_true);
                sum_rb += rmse(&res.b, &b_true);
                if res.lr_p < 0.05 {
                    type1 += 1.0;
                }
                // alternative (off-model): b = b_true + eps
                let b_alt: Vec<f64> = (0..j).map(|i| b_true[i] + eps[i]).collect();
                let y1 = simulate(&b_alt, n, j, skew, &mut rng);
                let res1 = fit_lltm(&y1, &observed, &q, n, j, k, &LltmConfig::default()).unwrap();
                if res1.lr_p < 0.05 {
                    power += 1.0;
                }
            }
            let r = reps as f64;
            println!(
                "skew={}: RMSE(eta)={:.4} bias(eta)={:.4} RMSE(b)={:.4} LR-typeI={:.3} LR-power={:.3}",
                skew, sum_re / r, sum_be / r, sum_rb / r, type1 / r, power / r
            );
            assert!(sum_re / r < 0.08, "mean RMSE(eta) {} skew={skew}", sum_re / r);
            assert!((sum_be / r).abs() < 0.03, "mean bias(eta) {} skew={skew}", sum_be / r);
            // The LR Type I is properly calibrated under correct specification
            // (normal: ~0.04). A misspecified ability prior (skew = Exp(1)-1 fit with an
            // N(0,1) quadrature) inflates it to ~0.13 because the SATURATED Rasch
            // reference absorbs skew-induced misfit that the CONSTRAINED LLTM cannot —
            // the LR test's known sensitivity to a shared baseline misspecification, not
            // an estimator defect (parameter recovery and power stay excellent in both).
            let type1_bound = if skew { 0.18 } else { 0.08 };
            assert!(type1 / r < type1_bound, "LR Type I {} skew={skew}", type1 / r);
            assert!(power / r > 0.90, "LR power {} skew={skew}", power / r);
        }
    }
}
