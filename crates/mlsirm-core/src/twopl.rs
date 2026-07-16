//! Two-parameter logistic item response model — confirmatory multidimensional form, orthogonal or correlated (Reckase,
//! 2009; Bock, Gibbons, & Muraki, 1988).
//!
//! `fit_2pl` fits a **general compensatory** multidimensional 2PL in which an
//! item may load FREELY on several latent dimensions, which trade off ADDITIVELY inside a
//! single logit:
//!
//! ```text
//! P(X_ij = 1 | theta_j) = sigmoid( sum_{d in S_i} a_id * theta_jd + b_i ),   theta_j ~ MVN(0, I_D)
//! ```
//!
//! `S_i = { d : L_id = 1 }` is item `i`'s loading set from a 0/1 confirmatory pattern `L`
//! (J x D); `a_id` is a free loading for `d in S_i` (zero otherwise); `b_i` is the intercept.
//! This is Reckase's compensatory M2PL / the full-information item factor model of Bock,
//! Gibbons & Muraki (1988), estimated by marginal-ML EM over a product Gauss-Hermite grid.
//!
//! It is genuinely COMPENSATORY (a low standing on one trait can be offset by a high standing
//! on another, because the traits sum in the logit), and it is distinct from the existing
//! estimators: the LSIRM/`marginal.rs` family is simple-structure (one trait dimension per
//! item, which factorizes the quadrature), and the orthogonal bifactor is the special case
//! "one primary + one general per item". Allowing arbitrary within-item cross-loadings breaks
//! that factorization and requires the full `Q^D` product quadrature, so this is a dedicated
//! estimator rather than a mode of `marginal.rs`.
//!
//! **Latent traits.** `theta ~ MVN(0, Sigma)`, `Sigma` a CORRELATION matrix (unit diagonal).
//! With `estimate_corr = false` (default) the factors are ORTHOGONAL (`Sigma = I`); with
//! `estimate_corr = true` the inter-factor correlations are estimated by an ECM step. The
//! correlated case maps the standard Gauss-Hermite grid through the Cholesky factor
//! `theta_g = L z_g` (`Sigma = L L'`) — a measure-preserving change of variables, so the same
//! product-GH weights integrate `phi_Sigma` — and the item M-step is reused verbatim on the
//! mapped nodes; the `Sigma` M-step ascends the Gaussian-prior objective
//! `-0.5[log|Sigma| + tr(Sigma^{-1} C)]` over the free correlations (`C` the posterior second
//! moment) with backtracking + a positive-definite guard so EM stays monotone.
//!
//! **Integration node rule (`xi_rule`).** The product Gauss-Hermite grid is exact for
//! near-polynomial integrands but its `Q^D` node count is exponential in `D`, so it is capped at
//! `D <= 3`. For `D = 4, 5, 6` the E-step integral is instead evaluated by **quasi-Monte-Carlo**
//! (`Halton`, the low-discrepancy default for the QMC path) or plain **Monte-Carlo** (`MonteCarlo`)
//! quadrature: `xi_points` points drawn from the standard-normal prior (Halton radical inverse ->
//! `inv_normal_cdf`, or seeded Gaussian draws), equal weights `1/xi_points`. This is Jank's (2005)
//! QMC-EM — only the E-step nodes/weights change; the per-item Newton M-step and the `Sigma`
//! M-step are byte-for-byte the same code on the swapped node set. Because the standard node set
//! is FIXED for the whole EM run, the ORTHOGONAL fit (`Sigma = I`, nodes never move) stays monotone
//! in the QMC-approximated marginal likelihood. In the CORRELATED fit the `Sigma` M-step
//! reparametrizes the node cloud (`theta_g = L(Sigma) z_g`), so each `Sigma` induces a different
//! QMC quadrature of its own likelihood and EM is monotone only up to the QMC quadrature error
//! (overall ascent with small per-step wobble that shrinks as `xi_points` grows) — use the
//! orthogonal path, or a larger `xi_points`, when strict monotonicity matters. QMC carries an
//! `O(N^{-1} (log N)^D)` finite-node bias that grows with `D`, so `D = 5, 6` and the correlated
//! `Sigma` off-diagonals need materially larger `xi_points`; a Cranley-Patterson random shift
//! (`xi_seed`, nonzero by default) de-correlates the higher-prime Halton axes.
//!
//! Identification: unit trait variances fix the per-dimension loading scale (independently of
//! the free correlations); `E[theta] = 0` fixes the intercepts; the confirmatory pattern
//! labels the dimensions and — with at least one PURE single-loading anchor item per dimension
//! (`validate` enforces this, rejecting rotationally-degenerate patterns such as all-ones) —
//! fixes rotation even with correlated factors (one pure indicator per factor forces the
//! observational-equivalence transform to be diagonal, and the unit diagonal then forces it to
//! `+-I`); the residual per-dimension sign is fixed by a reflection anchor (each dimension is
//! flipped so its largest-magnitude pure anchor loads positively, which also negates that
//! dimension's correlation off-diagonals).
//!
//! # References (APA 7th ed.)
//!
//! Reckase, M. D. (2009). *Multidimensional item response theory*. Springer.
//! https://doi.org/10.1007/978-0-387-89976-3
//!
//! Bock, R. D., Gibbons, R., & Muraki, E. (1988). Full-information item factor analysis.
//! *Applied Psychological Measurement, 12*(3), 261-280.
//! https://doi.org/10.1177/014662168801200305
//!
//! Jank, W. (2005). Quasi-Monte Carlo sampling to improve the efficiency of Monte Carlo EM.
//! *Computational Statistics & Data Analysis, 48*(4), 685-701.
//! https://doi.org/10.1016/j.csda.2004.03.019

use crate::marginal::XiRuleKind;
use crate::mmle::{log_sigmoid, sigmoid_stable};
use crate::nodes::{build_xi_nodes, XiRule};
use crate::poly::solve_small;
use crate::quadrature::{gh_rule, SUPPORTED_Q};

/// Maximum integration node count (bounds the per-iteration `nodes x J` tables) for BOTH the
/// `Q^D` Gauss-Hermite grid and the `xi_points` QMC/MC point set.
const MIRT_MAX_NODES: usize = 200_000;
/// Maximum node-by-item cells in each E-step table. Four dense f64 tables use this shape
/// (log P1, log P0, expected trials, expected successes), so this cap bounds aggregate table
/// memory and must be checked before any of those allocations.
const MIRT_MAX_NODE_ITEM_CELLS: usize = 60_000_000;
/// Maximum latent dimensions for the Gauss-Hermite product grid (`41^3 = 68_921 <= cap`). `D > 3`
/// is served by the quasi-Monte-Carlo (Halton) / Monte-Carlo node rules instead.
const MIRT_MAX_DIMS: usize = 3;
/// Maximum latent dimensions for the Halton/MonteCarlo rules (= `HALTON_PRIMES.len()` in `nodes`,
/// the Halton axis cap; also the sole guard for the MonteCarlo builder, which has no internal cap).
const MIRT_MAX_DIMS_QMC: usize = 6;
/// Symmetric loading bound. Loadings are NOT floored positive: confirmatory MIRT routinely
/// has opposite-sign loadings on a shared dimension (reverse-keyed items, suppressor
/// cross-loadings). The per-dimension reflection anchor fixes only the global sign.
const MIRT_A_BOUND: f64 = 10.0;

/// Configuration for [`fit_2pl`].
#[derive(Clone, Copy, Debug)]
pub struct TwoPlConfig {
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `|delta loglik|`.
    pub tol: f64,
    /// Gauss-Hermite nodes per dimension (must be in `{7, 11, 15, 21, 31, 41}`).
    pub q: usize,
    /// Ridge on the loading Hessian block (Gaussian prior, mirrors `MmleConfig`).
    pub ridge_a: f64,
    /// Ridge on the intercept Hessian entry.
    pub ridge_b: f64,
    /// Inner Newton iterations per item M-step.
    pub newton_iter: usize,
    /// Estimate a free latent CORRELATION matrix `Sigma` (`theta ~ MVN(0, Sigma)`, unit
    /// diagonal). When `false`, `Sigma = I` (orthogonal factors) exactly — the item model is
    /// evaluated on the raw Gauss-Hermite grid, bit-for-bit as the orthogonal fit.
    pub estimate_corr: bool,
    /// Latent-integral node rule. `GaussHermite` (default) uses the `q^D` product grid and caps
    /// `D <= 3`; `Halton` (quasi-Monte-Carlo, Jank 2005) and `MonteCarlo` use `xi_points` nodes
    /// mapped from the prior and unlock `D` up to 6 (the Halton prime axes). The item and `Sigma`
    /// M-steps are identical for every rule — only the E-step quadrature nodes/weights change.
    pub xi_rule: XiRuleKind,
    /// Number of QMC/MC integration points (used only for `Halton`/`MonteCarlo`; `q` is ignored
    /// for those rules). `D = 5, 6` need materially larger `xi_points` to keep the QMC error small.
    pub xi_points: usize,
    /// Halton Cranley-Patterson random-shift seed / Monte-Carlo seed. Nonzero by default so QMC
    /// runs are randomized (helps the high-prime axes at `D >= 5`); deterministic given the seed.
    pub xi_seed: u64,
}

impl Default for TwoPlConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            q: 21,
            ridge_a: 1e-3,
            ridge_b: 1e-3,
            newton_iter: 25,
            estimate_corr: false,
            xi_rule: XiRuleKind::GaussHermite,
            xi_points: 4000,
            xi_seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Result of [`fit_2pl`] (confirmatory compensatory MIRT, orthogonal or
/// correlated latent factors).
#[derive(Clone, Debug)]
pub struct TwoPlResult {
    /// Free loadings `a_id`, row-major `J x D` (exactly `0.0` where `L_id = 0`).
    pub loading: Vec<f64>,
    /// Item intercepts `b_i`, length `J`.
    pub intercept: Vec<f64>,
    /// Per-person trait EAP `E[theta_jd | X_j]`, row-major `N x D`.
    pub theta: Vec<f64>,
    /// Number of latent dimensions `D`.
    pub n_dims: usize,
    /// Latent correlation matrix `Sigma`, row-major `D x D` (identity when `estimate_corr`
    /// is `false`; unit diagonal, estimated off-diagonals otherwise).
    pub corr: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Machine-readable termination status: `converged` or `max_iter_reached`.
    pub termination_reason: String,
    /// Absolute change between the final two evaluated marginal log-likelihoods.
    pub final_loglik_change: f64,
    /// `#{L_id = 1}` loadings `+ J` intercepts `+ D(D-1)/2` correlations (when estimated).
    pub n_parameters: usize,
}

#[allow(clippy::too_many_arguments)]
fn validate(
    y: &[f64],
    observed: &[bool],
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    cfg: &TwoPlConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    for (name, v) in [("ridge_a", cfg.ridge_a), ("ridge_b", cfg.ridge_b)] {
        // Strictly positive: the ridge is what makes A = -Hessian strictly positive-definite,
        // so the Newton solve is always an exact ascent step (never the singular fallback).
        if !v.is_finite() || v <= 0.0 {
            return Err(format!("{name} must be finite and positive"));
        }
    }
    // Rule-dependent dimension bound + node-count cap. The Gauss-Hermite product grid caps at
    // MIRT_MAX_DIMS (Q^D blows up); the QMC/MC rules cap at MIRT_MAX_DIMS_QMC (the Halton primes)
    // and bound the user-supplied point count instead. `q` is validated/used only for GH.
    let n_nodes = match cfg.xi_rule {
        XiRuleKind::GaussHermite => {
            if !(1..=MIRT_MAX_DIMS).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={MIRT_MAX_DIMS} for the Gauss-Hermite grid; use \
                     xi_rule Halton/MonteCarlo for D up to {MIRT_MAX_DIMS_QMC}"
                ));
            }
            if !SUPPORTED_Q.contains(&cfg.q) {
                return Err(format!(
                    "q must be one of {SUPPORTED_Q:?} (Gauss-Hermite rules); got {}",
                    cfg.q
                ));
            }
            // Q^D via an accumulating checked multiply in a fixed order (never wraps).
            let mut n_nodes = 1usize;
            for _ in 0..n_dims {
                n_nodes = n_nodes
                    .checked_mul(cfg.q)
                    .filter(|&n| n <= MIRT_MAX_NODES)
                    .ok_or_else(|| format!("q^n_dims exceeds the node cap {MIRT_MAX_NODES}"))?;
            }
            n_nodes
        }
        XiRuleKind::Halton | XiRuleKind::MonteCarlo => {
            // The MonteCarlo node builder has no internal dimension cap, so this bound is the sole
            // guard for MC at D > MIRT_MAX_DIMS_QMC (Halton's own builder errors past its primes).
            if !(1..=MIRT_MAX_DIMS_QMC).contains(&n_dims) {
                return Err(format!(
                    "n_dims must be in 1..={MIRT_MAX_DIMS_QMC} for the Halton/MonteCarlo rules"
                ));
            }
            if !(1..=MIRT_MAX_NODES).contains(&cfg.xi_points) {
                return Err(format!(
                    "xi_points must be in 1..={MIRT_MAX_NODES} for the Halton/MonteCarlo rules; \
                     got {}",
                    cfg.xi_points
                ));
            }
            cfg.xi_points
        }
    };
    let table_cells = n_nodes
        .checked_mul(n_items)
        .ok_or_else(|| "node * item table size overflows usize".to_string())?;
    if table_cells > MIRT_MAX_NODE_ITEM_CELLS {
        return Err(format!(
            "node * item table has {table_cells} cells, exceeding the cap \
             {MIRT_MAX_NODE_ITEM_CELLS}; reduce nodes or items"
        ));
    }
    n_nodes
        .checked_mul(n_dims)
        .ok_or_else(|| "node * dimension buffer size overflows usize".to_string())?;
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_l = n_items
        .checked_mul(n_dims)
        .ok_or_else(|| "n_items * n_dims overflows usize".to_string())?;
    if loading_pattern.len() != n_l {
        return Err("loading_pattern must have length n_items * n_dims".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && v != 0.0 && v != 1.0 {
            return Err(format!("y[{idx}] must be 0 or 1 where observed; got {v}"));
        }
    }
    for (idx, &v) in loading_pattern.iter().enumerate() {
        if v != 0 && v != 1 {
            return Err(format!("loading_pattern[{idx}] must be 0 or 1; got {v}"));
        }
    }
    // Every item loads >= 1 dimension; every item has >= 1 observed response.
    for i in 0..n_items {
        if !(0..n_dims).any(|d| loading_pattern[i * n_dims + d] != 0) {
            return Err(format!(
                "item {i} loads no dimension (all-zero loading_pattern row)"
            ));
        }
        if !(0..n_persons).any(|p| observed[p * n_items + i]) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    // Identification: every dimension needs a PURE single-loading anchor item (an item that
    // loads ONLY that dimension). This is the sufficient structural condition that fixes the
    // orthogonal rotation (and gives the sign anchor a target); it rejects the all-ones and
    // other rotationally-degenerate patterns that leave the item Hessian block singular.
    for d in 0..n_dims {
        let has_pure_anchor = (0..n_items).any(|i| {
            loading_pattern[i * n_dims + d] != 0
                && (0..n_dims)
                    .filter(|&d2| loading_pattern[i * n_dims + d2] != 0)
                    .count()
                    == 1
        });
        if !has_pure_anchor {
            return Err(format!(
                "dimension {d} has no pure single-loading anchor item (needed for identification; \
                 rotationally-degenerate pattern)"
            ));
        }
    }
    Ok(())
}

/// Build the `D`-fold Cartesian product Gauss-Hermite grid over orthogonal `N(0,1)` axes.
/// Returns row-major `nodes[g*D + d]` and `logw[g] = sum_d ln(w_axis[digit_d])`.
fn build_grid(n_dims: usize, q: usize) -> (Vec<f64>, Vec<f64>) {
    let (axis_nodes, axis_weights) = gh_rule(q).expect("q validated in supported set");
    let log_aw: Vec<f64> = axis_weights.iter().map(|w| w.ln()).collect();
    let n_nodes = q.pow(n_dims as u32);
    let mut nodes = vec![0.0f64; n_nodes * n_dims];
    let mut logw = vec![0.0f64; n_nodes];
    for g in 0..n_nodes {
        let mut rem = g;
        let mut lw = 0.0f64;
        for d in 0..n_dims {
            let digit = rem % q; // mixed-radix base q; digit_d = (g / q^d) % q
            rem /= q;
            nodes[g * n_dims + d] = axis_nodes[digit];
            lw += log_aw[digit];
        }
        logw[g] = lw;
    }
    (nodes, logw)
}

/// Penalized per-item complete-data objective `Q_i` (the M-step ascends this): the expected
/// Bernoulli log-likelihood over the grid minus the ridge Gaussian penalty. Used for the
/// backtracking line search so every M-step step is non-decreasing (keeps EM monotone).
#[allow(clippy::too_many_arguments)]
fn item_obj(
    dims: &[usize],
    a: &[f64],
    b: f64,
    n_ig: &[f64],
    r_ig: &[f64],
    nodes: &[f64],
    n_dims: usize,
    n_nodes: usize,
    ridge_a: f64,
    ridge_b: f64,
) -> f64 {
    let mut acc = 0.0f64;
    for g in 0..n_nodes {
        let mut eta = b;
        for (k, &d) in dims.iter().enumerate() {
            eta += a[k] * nodes[g * n_dims + d];
        }
        acc += r_ig[g] * log_sigmoid(eta) + (n_ig[g] - r_ig[g]) * log_sigmoid(-eta);
    }
    let pen: f64 = a.iter().map(|&ak| ak * ak).sum::<f64>() * ridge_a + b * b * ridge_b;
    acc - 0.5 * pen
}

/// Ascent gradient `g` and the positive-definite `A = -Hessian` of the penalized item
/// objective [`item_obj`] at the current `(a, b)`, over the loaded dimensions `dims`. The
/// diagonal ridge makes `A` strictly positive-definite, so the Newton solve is a well-posed
/// ascent step that never triggers `solve_small`'s singular fallback. `g[k] = sum_g
/// (r_ig - n_ig p_g) z_gk - ridge_k a_k`, `A[k][j] = sum_g n_ig p_g(1-p_g) z_gk z_gj +
/// ridge_k [k=j]`, with `z_gk = theta_{g,dims[k]}` for loadings and `1` for the intercept
/// (last index). This is the Bock-Gibbons-Muraki (1988) full-information item update; at
/// `D = 1` it is `mmle`'s 2x2 block.
#[allow(clippy::too_many_arguments)]
fn item_grad_hess(
    dims: &[usize],
    a: &[f64],
    b: f64,
    n_ig: &[f64],
    r_ig: &[f64],
    nodes: &[f64],
    n_dims: usize,
    n_nodes: usize,
    ridge_a: f64,
    ridge_b: f64,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let ni = dims.len();
    let np = ni + 1;
    let mut grad = vec![0.0f64; np];
    let mut amat = vec![vec![0.0f64; np]; np];
    for g in 0..n_nodes {
        let n = n_ig[g];
        if n == 0.0 {
            continue;
        }
        let mut eta = b;
        for (k, &d) in dims.iter().enumerate() {
            eta += a[k] * nodes[g * n_dims + d];
        }
        let pg = sigmoid_stable(eta);
        let w = n * pg * (1.0 - pg);
        let resid = r_ig[g] - n * pg;
        for k in 0..np {
            let zk = if k < ni {
                nodes[g * n_dims + dims[k]]
            } else {
                1.0
            };
            grad[k] += resid * zk;
            for j in 0..np {
                let zj = if j < ni {
                    nodes[g * n_dims + dims[j]]
                } else {
                    1.0
                };
                amat[k][j] += w * zk * zj;
            }
        }
    }
    for k in 0..np {
        let (rk, pk) = if k < ni {
            (ridge_a, a[k])
        } else {
            (ridge_b, b)
        };
        grad[k] -= rk * pk;
        amat[k][k] += rk;
    }
    (grad, amat)
}

/// Lower Cholesky factor of a `D x D` symmetric matrix (row-major), or `None` if it is not
/// (numerically) positive-definite — the PD gate for the correlation M-step and the node map.
pub(crate) fn chol_lower(sigma: &[f64], d: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0f64; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut s = sigma[i * d + j];
            for k in 0..j {
                s -= l[i * d + k] * l[j * d + k];
            }
            if i == j {
                if s <= 1e-12 {
                    return None;
                }
                l[i * d + i] = s.sqrt();
            } else {
                l[i * d + j] = s / l[j * d + j];
            }
        }
    }
    Some(l)
}

/// Inverse (row-major) and log-determinant of a symmetric PD `D x D` matrix via its Cholesky
/// factor; `None` if not PD.
pub(crate) fn sym_inv_logdet(sigma: &[f64], d: usize) -> Option<(Vec<f64>, f64)> {
    let l = chol_lower(sigma, d)?;
    let logdet = (0..d).map(|i| 2.0 * l[i * d + i].ln()).sum::<f64>();
    let mut inv = vec![0.0f64; d * d];
    for col in 0..d {
        let mut y = vec![0.0f64; d]; // forward solve L y = e_col
        for i in 0..d {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for k in 0..i {
                s -= l[i * d + k] * y[k];
            }
            y[i] = s / l[i * d + i];
        }
        for i in (0..d).rev() {
            // back solve L^T x = y
            let mut s = y[i];
            for k in i + 1..d {
                s -= l[k * d + i] * inv[k * d + col];
            }
            inv[i * d + col] = s / l[i * d + i];
        }
    }
    Some((inv, logdet))
}

/// Gaussian-prior objective the correlation M-step ascends:
/// `Q_prior(Sigma) = -0.5 [ log|Sigma| + tr(Sigma^{-1} C) ]`, `C` the posterior second moment.
/// `None` if `Sigma` is not PD.
fn sigma_qprior(sigma: &[f64], c: &[f64], d: usize) -> Option<f64> {
    let (inv, logdet) = sym_inv_logdet(sigma, d)?;
    let mut tr = 0.0f64;
    for i in 0..d {
        for k in 0..d {
            tr += inv[i * d + k] * c[k * d + i];
        }
    }
    Some(-0.5 * (logdet + tr))
}

/// Off-diagonal gradient of `sigma_qprior` w.r.t. the free correlations (pairs `(i,j)`, `i<j`,
/// length `D(D-1)/2`): `g_{ij} = [Sigma^{-1} C Sigma^{-1} - Sigma^{-1}]_{ij}`. `None` if not PD.
pub(crate) fn sigma_grad(sigma: &[f64], c: &[f64], d: usize) -> Option<Vec<f64>> {
    let (inv, _) = sym_inv_logdet(sigma, d)?;
    let mut ic = vec![0.0f64; d * d]; // inv * C
    for i in 0..d {
        for j in 0..d {
            let mut s = 0.0;
            for k in 0..d {
                s += inv[i * d + k] * c[k * d + j];
            }
            ic[i * d + j] = s;
        }
    }
    let mut g = Vec::with_capacity(d * (d - 1) / 2);
    for i in 0..d {
        for j in i + 1..d {
            let mut ici = 0.0; // (inv * C * inv)_{ij}
            for k in 0..d {
                ici += ic[i * d + k] * inv[k * d + j];
            }
            g.push(ici - inv[i * d + j]);
        }
    }
    Some(g)
}

/// Build a `D x D` correlation matrix (row-major, unit diagonal) from the free off-diagonal
/// correlations (pairs `(i,j)`, `i<j`), clamped to `(-1, 1)` (a cheap first PD reject; the full
/// PD guarantee is enforced by the caller's Cholesky check).
pub(crate) fn build_corr(offdiag: &[f64], d: usize) -> Vec<f64> {
    let mut s = vec![0.0f64; d * d];
    for i in 0..d {
        s[i * d + i] = 1.0;
    }
    let mut m = 0;
    for i in 0..d {
        for j in i + 1..d {
            let r = offdiag[m].clamp(-0.999, 0.999);
            m += 1;
            s[i * d + j] = r;
            s[j * d + i] = r;
        }
    }
    s
}

/// Negate the free correlations (off-diagonal `(i,j)` pairs, `i<j`) that involve dimension
/// `flip`, so a per-dimension reflection `theta_flip -> -theta_flip` stays consistent with the
/// reported correlation matrix (`corr(theta_flip, theta_k) -> -corr`). Correlations not
/// involving `flip` are untouched; the diagonal is implicitly unchanged (it is not stored).
pub(crate) fn flip_corr_dim(offdiag: &mut [f64], d: usize, flip: usize) {
    let mut m = 0;
    for i in 0..d {
        for j in i + 1..d {
            if i == flip || j == flip {
                offdiag[m] = -offdiag[m];
            }
            m += 1;
        }
    }
}

/// Fit the orthogonal OR correlated confirmatory compensatory MIRT by marginal-ML (EC)M.
///
/// `y`/`observed` are row-major `N*J` (`y` in `{0,1}` where observed; missing cells dropped
/// under MAR); `loading_pattern` is row-major `J*D` in `{0,1}`. Returns `Err` on malformed or
/// rotationally-underidentified input.
#[allow(clippy::too_many_arguments)]
pub fn fit_2pl(
    y: &[f64],
    observed: &[bool],
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    cfg: &TwoPlConfig,
) -> Result<TwoPlResult, String> {
    validate(
        y,
        observed,
        loading_pattern,
        n_persons,
        n_items,
        n_dims,
        cfg,
    )?;
    // Build the latent-integral node set once, before the EM loop: a FIXED quadrature keeps EM
    // monotone in the (QMC-)approximated marginal likelihood (Jank, 2005). The Gauss-Hermite path
    // keeps `build_grid` verbatim (bit-for-bit the orthogonal fit); Halton/MonteCarlo delegate to
    // the shared, parity-tested `build_xi_nodes` (prior-sampled points, equal `logw = -ln(n)`).
    let (nodes, logw) = match cfg.xi_rule {
        XiRuleKind::GaussHermite => build_grid(n_dims, cfg.q),
        XiRuleKind::Halton => {
            let xn = build_xi_nodes(
                XiRule::Halton {
                    n: cfg.xi_points,
                    shift_seed: cfg.xi_seed,
                },
                n_dims,
            )?;
            (xn.grid, xn.logw)
        }
        XiRuleKind::MonteCarlo => {
            let xn = build_xi_nodes(
                XiRule::MonteCarlo {
                    n: cfg.xi_points,
                    seed: cfg.xi_seed.max(1),
                },
                n_dims,
            )?;
            (xn.grid, xn.logw)
        }
    };
    let n_nodes = logw.len();

    // Per-item loaded-dimension lists S_i (the free-loading dims).
    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] != 0)
                .collect()
        })
        .collect();

    // Init: loadings 1.0 on the pattern; intercept = logit of the item's observed proportion.
    let mut loading = vec![0.0f64; n_items * n_dims];
    let mut intercept = vec![0.0f64; n_items];
    for i in 0..n_items {
        for &d in &dims_of[i] {
            loading[i * n_dims + d] = 1.0;
        }
        let (mut num, mut den) = (0.0f64, 0.0f64);
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
        intercept[i] = (prop / (1.0 - prop)).ln();
    }

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut theta = vec![0.0f64; n_persons * n_dims];

    let mut post = vec![0.0f64; n_nodes]; // reused per-person buffer (no N x Q^D storage)
    let mut log_p1 = vec![0.0f64; n_nodes * n_items];
    let mut log_p0 = vec![0.0f64; n_nodes * n_items];

    // Correlated traits (estimate_corr): free correlations `r_off` (pairs i<j; Sigma = I at
    // r_off = 0) and the buffer for the correlated nodes theta_g = L z_g. When !estimate_corr
    // the item model reads the raw grid `nodes` directly (bit-identical to the orthogonal fit).
    let d = n_dims;
    let n_off = d * (d - 1) / 2;
    let mut r_off = vec![0.0f64; n_off];
    let mut theta_nodes = if cfg.estimate_corr {
        vec![0.0f64; n_nodes * n_dims]
    } else {
        Vec::new()
    };

    for _ in 0..cfg.max_iter {
        // Map the standard GH grid through L = chol(Sigma): theta_g = L z_g (rt_joint pattern).
        if cfg.estimate_corr {
            let sigma = build_corr(&r_off, d);
            let lchol = chol_lower(&sigma, d).expect("Sigma is PD by construction of r_off");
            for g in 0..n_nodes {
                for k in 0..d {
                    let mut t = 0.0f64;
                    for j in 0..=k {
                        t += lchol[k * d + j] * nodes[g * d + j];
                    }
                    theta_nodes[g * d + k] = t;
                }
            }
        }
        let cur_nodes: &[f64] = if cfg.estimate_corr {
            &theta_nodes
        } else {
            &nodes
        };

        // Node x item log-probabilities under the current parameters.
        for g in 0..n_nodes {
            for i in 0..n_items {
                let mut eta = intercept[i];
                for &d in &dims_of[i] {
                    eta += loading[i * n_dims + d] * cur_nodes[g * n_dims + d];
                }
                log_p1[g * n_items + i] = log_sigmoid(eta);
                log_p0[g * n_items + i] = log_sigmoid(-eta);
            }
        }

        // Streamed E-step: per person, fill `post`, then accumulate counts + theta EAP.
        let mut n_ig = vec![0.0f64; n_items * n_nodes];
        let mut r_ig = vec![0.0f64; n_items * n_nodes];
        let mut m_g = vec![0.0f64; if cfg.estimate_corr { n_nodes } else { 0 }];
        let mut total_ll = 0.0f64;
        for p in 0..n_persons {
            for (g, slot) in post.iter_mut().enumerate() {
                let mut acc = logw[g];
                for i in 0..n_items {
                    let idx = p * n_items + i;
                    if observed[idx] {
                        let yy = y[idx];
                        acc += yy * log_p1[g * n_items + i] + (1.0 - yy) * log_p0[g * n_items + i];
                    }
                }
                *slot = acc;
            }
            let m = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in post.iter() {
                denom += (v - m).exp();
            }
            total_ll += m + denom.ln();
            for v in post.iter_mut() {
                *v = (*v - m).exp() / denom;
            }
            debug_assert!(
                (post.iter().sum::<f64>() - 1.0).abs() < 1e-9,
                "posterior sums to 1"
            );
            if cfg.estimate_corr {
                for (mg, &pg) in m_g.iter_mut().zip(post.iter()) {
                    *mg += pg;
                }
            }
            for i in 0..n_items {
                let idx = p * n_items + i;
                if observed[idx] {
                    let yy = y[idx];
                    let base = i * n_nodes;
                    for g in 0..n_nodes {
                        n_ig[base + g] += post[g];
                        r_ig[base + g] += yy * post[g];
                    }
                }
            }
        }
        loglik_trace.push(total_ll);

        // Converged-check BEFORE the M-step so returned params match the trace endpoint.
        if loglik_trace.len() > 1 {
            let l = loglik_trace.len();
            if (loglik_trace[l - 1] - loglik_trace[l - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: per-item (n_i+1)-dim Newton with ridge + backtracking line search.
        for i in 0..n_items {
            let dims = &dims_of[i];
            let ni = dims.len();
            let ni_off = i * n_nodes;
            let mut a: Vec<f64> = dims.iter().map(|&d| loading[i * n_dims + d]).collect();
            let mut b = intercept[i];
            let ns = &n_ig[ni_off..ni_off + n_nodes];
            let rs = &r_ig[ni_off..ni_off + n_nodes];
            for _ in 0..cfg.newton_iter {
                let (grad, amat) = item_grad_hess(
                    dims,
                    &a,
                    b,
                    ns,
                    rs,
                    cur_nodes,
                    n_dims,
                    n_nodes,
                    cfg.ridge_a,
                    cfg.ridge_b,
                );
                let delta = solve_small(amat, grad); // A positive-definite => exact ascent step
                let q0 = item_obj(
                    dims,
                    &a,
                    b,
                    ns,
                    rs,
                    cur_nodes,
                    n_dims,
                    n_nodes,
                    cfg.ridge_a,
                    cfg.ridge_b,
                );
                // Backtracking: halve until the penalized item objective does not decrease.
                let mut step = 1.0f64;
                let mut accepted = false;
                let (mut a_new, mut b_new) = (a.clone(), b);
                for _ in 0..20 {
                    for k in 0..ni {
                        a_new[k] = (a[k] + step * delta[k]).clamp(-MIRT_A_BOUND, MIRT_A_BOUND);
                    }
                    b_new = b + step * delta[ni];
                    let q1 = item_obj(
                        dims,
                        &a_new,
                        b_new,
                        ns,
                        rs,
                        cur_nodes,
                        n_dims,
                        n_nodes,
                        cfg.ridge_a,
                        cfg.ridge_b,
                    );
                    if q1 >= q0 - 1e-12 {
                        accepted = true;
                        break;
                    }
                    step *= 0.5;
                }
                if !accepted {
                    break; // no uphill step found -> keep previous (rare; near a maximum)
                }
                let moved: f64 =
                    (0..ni).map(|k| (a_new[k] - a[k]).abs()).sum::<f64>() + (b_new - b).abs();
                a = a_new;
                b = b_new;
                if moved < 1e-9 {
                    break;
                }
            }
            for (k, &d) in dims.iter().enumerate() {
                loading[i * n_dims + d] = a[k];
            }
            intercept[i] = b;
        }

        // Correlation (Sigma) M-step: gradient ascent on Q_prior over the free correlations,
        // with backtracking + a full-matrix PD (Cholesky) guard so each step is non-decreasing
        // (keeps the ECM marginal loglik monotone). The complete-data Q separates additively
        // into item terms + the Gaussian prior, so this block is independent of the item M-step.
        if cfg.estimate_corr {
            // C = (1/N) sum_g m_g theta_g theta_g^T (posterior second moment; theta_g is
            // person-independent, so the marginal node mass m_g factors the N-loop out).
            let nf = n_persons as f64;
            let mut cmat = vec![0.0f64; d * d];
            for g in 0..n_nodes {
                let w = m_g[g] / nf;
                for a1 in 0..d {
                    let ta = theta_nodes[g * d + a1];
                    for b1 in 0..d {
                        cmat[a1 * d + b1] += w * ta * theta_nodes[g * d + b1];
                    }
                }
            }
            for _ in 0..cfg.newton_iter {
                let sigma = build_corr(&r_off, d);
                let grad = match sigma_grad(&sigma, &cmat, d) {
                    Some(g) => g,
                    None => break,
                };
                let q0 = match sigma_qprior(&sigma, &cmat, d) {
                    Some(q) => q,
                    None => break,
                };
                let gnorm = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
                if gnorm < 1e-10 {
                    break;
                }
                let mut alpha = 1.0f64;
                let mut moved = false;
                for _ in 0..40 {
                    let r_cand: Vec<f64> = (0..n_off)
                        .map(|m| (r_off[m] + alpha * grad[m]).clamp(-0.999, 0.999))
                        .collect();
                    let cand = build_corr(&r_cand, d);
                    // sigma_qprior returns None unless `cand` is PD -> both the ascent and the
                    // full-matrix PD guard are enforced in one check (the box clamp above is only
                    // a cheap first reject; it does not imply PD at D=3).
                    if let Some(q1) = sigma_qprior(&cand, &cmat, d) {
                        if q1 >= q0 - 1e-12 {
                            r_off = r_cand;
                            moved = true;
                            break;
                        }
                    }
                    alpha *= 0.5;
                }
                if !moved {
                    break;
                }
            }
        }
        n_iter += 1;
    }

    // Final pass under the returned parameters: trait EAP for every person, and the marginal
    // loglik of those parameters (pushed when EM exited on max-iter, so the trace endpoint
    // matches the returned params — on convergence the last E-step already supplied it).
    if cfg.estimate_corr {
        let sigma = build_corr(&r_off, d);
        let lchol = chol_lower(&sigma, d).expect("Sigma is PD by construction of r_off");
        for g in 0..n_nodes {
            for k in 0..d {
                let mut t = 0.0f64;
                for j in 0..=k {
                    t += lchol[k * d + j] * nodes[g * d + j];
                }
                theta_nodes[g * d + k] = t;
            }
        }
    }
    let final_nodes: &[f64] = if cfg.estimate_corr {
        &theta_nodes
    } else {
        &nodes
    };
    for g in 0..n_nodes {
        for i in 0..n_items {
            let mut eta = intercept[i];
            for &d in &dims_of[i] {
                eta += loading[i * n_dims + d] * final_nodes[g * n_dims + d];
            }
            log_p1[g * n_items + i] = log_sigmoid(eta);
            log_p0[g * n_items + i] = log_sigmoid(-eta);
        }
    }
    let mut final_ll = 0.0f64;
    for p in 0..n_persons {
        for (g, slot) in post.iter_mut().enumerate() {
            let mut acc = logw[g];
            for i in 0..n_items {
                let idx = p * n_items + i;
                if observed[idx] {
                    let yy = y[idx];
                    acc += yy * log_p1[g * n_items + i] + (1.0 - yy) * log_p0[g * n_items + i];
                }
            }
            *slot = acc;
        }
        let m = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for v in post.iter() {
            denom += (v - m).exp();
        }
        final_ll += m + denom.ln();
        for (g, v) in post.iter().enumerate() {
            let pg = (v - m).exp() / denom;
            for d in 0..n_dims {
                theta[p * n_dims + d] += pg * final_nodes[g * n_dims + d];
            }
        }
    }
    if !converged {
        loglik_trace.push(final_ll);
        let l = loglik_trace.len();
        if (loglik_trace[l - 1] - loglik_trace[l - 2]).abs() < cfg.tol {
            converged = true;
        }
    }

    // Per-dimension reflection anchor: flip dimension d (all loadings on d and all theta_d) so
    // its largest-|loading| PURE anchor item loads positively. Flipping theta_d -> -theta_d
    // negates corr(theta_d, theta_k), so the correlation off-diagonals of row/col d must flip
    // too (likelihood-invariant relabeling). Flips commute across dimensions.
    for d in 0..n_dims {
        let mut anchor: Option<usize> = None;
        let mut best = 0.0f64;
        for i in 0..n_items {
            let is_pure = dims_of[i].len() == 1 && dims_of[i][0] == d;
            if is_pure && loading[i * n_dims + d].abs() > best {
                best = loading[i * n_dims + d].abs();
                anchor = Some(i);
            }
        }
        if let Some(ai) = anchor {
            if loading[ai * n_dims + d] < 0.0 {
                for i in 0..n_items {
                    loading[i * n_dims + d] = -loading[i * n_dims + d];
                }
                for p in 0..n_persons {
                    theta[p * n_dims + d] = -theta[p * n_dims + d];
                }
                flip_corr_dim(&mut r_off, n_dims, d); // keep Sigma consistent with the sign flip
            }
        }
    }

    let n_free_loadings = loading_pattern.iter().filter(|&&v| v == 1).count();
    let l = loglik_trace.len();
    let final_loglik_change = (loglik_trace[l - 1] - loglik_trace[l - 2]).abs();
    let n_parameters = n_free_loadings + n_items + if cfg.estimate_corr { n_off } else { 0 };
    Ok(TwoPlResult {
        loading,
        intercept,
        theta,
        n_dims,
        corr: build_corr(&r_off, d),
        loglik_trace,
        n_iter,
        converged,
        termination_reason: if converged {
            "converged"
        } else {
            "max_iter_reached"
        }
        .into(),
        final_loglik_change,
        n_parameters,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmle::{fit_mmle_2pl, MmleConfig};

    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
        fn bern(&mut self, p: f64) -> f64 {
            if self.next_f64() < p {
                1.0
            } else {
                0.0
            }
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / n).sqrt()
    }
    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (a, b) in x.iter().zip(y) {
            sxy += (a - mx) * (b - my);
            sxx += (a - mx) * (a - mx);
            syy += (b - my) * (b - my);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    /// Simulate compensatory M2PL responses from loadings (J*D), intercepts (J), and person
    /// traits (N*D) via the same additive-logit model the estimator recovers.
    fn simulate(
        loading: &[f64],
        intercept: &[f64],
        thetas: &[f64],
        n: usize,
        n_items: usize,
        n_dims: usize,
        rng: &mut Lcg,
    ) -> Vec<f64> {
        let mut y = vec![0.0f64; n * n_items];
        for j in 0..n {
            for i in 0..n_items {
                let mut eta = intercept[i];
                for d in 0..n_dims {
                    eta += loading[i * n_dims + d] * thetas[j * n_dims + d];
                }
                y[j * n_items + i] = rng.bern(sigmoid(eta));
            }
        }
        y
    }

    /// The orthogonal product GH grid reproduces the N(0, I) moments (sum w = 1, E[theta_d]=0,
    /// Var=1, Cov=0) - catches a transposed nodes[g*D+d] or a bad Cartesian product.
    #[test]
    fn mirt_grid_moments() {
        let (nodes, logw) = build_grid(2, 15);
        let n = logw.len();
        let w: Vec<f64> = logw.iter().map(|l| l.exp()).collect();
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-10, "sum w");
        let (mut e0, mut e1, mut v0, mut v1, mut c01) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for g in 0..n {
            let (t0, t1) = (nodes[g * 2], nodes[g * 2 + 1]);
            e0 += w[g] * t0;
            e1 += w[g] * t1;
            v0 += w[g] * t0 * t0;
            v1 += w[g] * t1 * t1;
            c01 += w[g] * t0 * t1;
        }
        assert!(e0.abs() < 1e-9 && e1.abs() < 1e-9, "means");
        assert!(
            (v0 - 1.0).abs() < 1e-9 && (v1 - 1.0).abs() < 1e-9,
            "variances"
        );
        assert!(c01.abs() < 1e-9, "cross moment (orthogonality)");
    }

    /// Deterministic anchor: the analytic item gradient AND the full (n_i+1)x(n_i+1) Hessian
    /// block - including the off-diagonal cross-Hessian H_{a0,a1} and the local->pattern-dim
    /// map - match central finite differences of item_obj at D=2 for a BOTH-loading item, to
    /// < 1e-4. A dims[k] indexing bug or a missing cross term fails this with no MC noise.
    #[test]
    fn mirt_item_grad_hess_matches_finite_difference() {
        // Two configs: identity map (dims=[0,1] on a D=2 grid) AND a NON-IDENTITY map
        // (dims=[0,2] on a D=3 grid, so nodes index dims[k]!=k) — the latter genuinely pins
        // the local-param -> pattern-dimension map that a k-vs-dims[k] bug would break.
        for &(n_dims, ref dims) in [(2usize, vec![0usize, 1]), (3usize, vec![0usize, 2])].iter() {
            let (nodes, logw) = build_grid(n_dims, 15);
            let n_nodes = logw.len();
            let mut rng = Lcg(99);
            let (mut n_ig, mut r_ig) = (vec![0.0f64; n_nodes], vec![0.0f64; n_nodes]);
            for g in 0..n_nodes {
                n_ig[g] = 1.0 + rng.next_f64() * 3.0;
                r_ig[g] = n_ig[g] * rng.next_f64();
            }
            let (a, b) = (vec![0.8f64, -0.5], 0.3f64); // dims.len() == 2 for both configs
            let (ra, rb) = (1e-3, 1e-3);
            let np = dims.len() + 1;
            let (grad, amat) =
                item_grad_hess(dims, &a, b, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
            let obj = |aa: &[f64], bb: f64| {
                item_obj(dims, aa, bb, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb)
            };
            let eps = 1e-6;
            let perturb = |k: usize, s: f64| -> (Vec<f64>, f64) {
                let mut aa = a.clone();
                let mut bb = b;
                if k < dims.len() {
                    aa[k] += s;
                } else {
                    bb += s;
                }
                (aa, bb)
            };
            for k in 0..np {
                let (ap, bp) = perturb(k, eps);
                let (am, bm) = perturb(k, -eps);
                let fd = (obj(&ap, bp) - obj(&am, bm)) / (2.0 * eps);
                assert!(
                    (grad[k] - fd).abs() < 1e-4,
                    "grad[{k}] {} vs fd {fd} (D={n_dims})",
                    grad[k]
                );
            }
            for jp in 0..np {
                let (ap, bp) = perturb(jp, eps);
                let (am, bm) = perturb(jp, -eps);
                let (gp, _) =
                    item_grad_hess(dims, &ap, bp, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
                let (gm, _) =
                    item_grad_hess(dims, &am, bm, &n_ig, &r_ig, &nodes, n_dims, n_nodes, ra, rb);
                for k in 0..np {
                    let dfd = (gp[k] - gm[k]) / (2.0 * eps);
                    assert!((dfd + amat[k][jp]).abs() < 1e-4, "H[{k}][{jp}] D={n_dims}");
                }
            }
        }
    }

    /// D=1 (all items load the single dimension) recovers known 2PL parameters and matches
    /// fit_mmle_2pl on the same data (gh_rule(41) is the same 41-node grid as mmle::GH_NODES).
    #[test]
    fn mirt_reduces_to_2pl_at_d1() {
        let (n, n_items) = (1500usize, 12usize);
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.7 + 0.1 * i as f64).collect();
        let b_true: Vec<f64> = (0..n_items).map(|i| -1.0 + 0.18 * i as f64).collect();
        let mut rng = Lcg(2024);
        let thetas: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
        let y = simulate(&a_true, &b_true, &thetas, n, n_items, 1, &mut rng);
        let observed = vec![true; n * n_items];
        let pattern = vec![1u8; n_items];
        let cfg = TwoPlConfig {
            q: 41,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, 1, &cfg).unwrap();
        assert!(
            rmse(&res.loading, &a_true) < 0.12,
            "loading RMSE {}",
            rmse(&res.loading, &a_true)
        );
        assert!(rmse(&res.intercept, &b_true) < 0.12, "intercept RMSE");
        let m = fit_mmle_2pl(&y, &observed, n, n_items, &MmleConfig::default());
        assert!(
            rmse(&res.loading, &m.a) < 1e-2,
            "vs mmle a {}",
            rmse(&res.loading, &m.a)
        );
        assert!(
            rmse(&res.intercept, &m.b) < 1e-2,
            "vs mmle b {}",
            rmse(&res.intercept, &m.b)
        );
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "monotone");
        }
    }

    /// Non-trivial D=2 compensatory recovery: a confirmatory pattern (dim0-only, dim1-only,
    /// AND both-loading items) with ASYMMETRIC, non-centered true loadings INCLUDING genuinely
    /// NEGATIVE loadings. Recovers loadings with correct sign and per-dimension theta EAP
    /// correlation. A dim-swap or a compensation-sign bug fails this.
    #[test]
    fn mirt_recovers_compensatory_d2() {
        let n_dims = 2usize;
        let mut pattern: Vec<u8> = Vec::new();
        for _ in 0..4 {
            pattern.extend_from_slice(&[1, 0]);
        }
        for _ in 0..4 {
            pattern.extend_from_slice(&[0, 1]);
        }
        for _ in 0..3 {
            pattern.extend_from_slice(&[1, 1]);
        }
        let n_items = 11usize;
        let a0 = [1.2, 0.8, 1.5, -0.9];
        let a1 = [1.0, 1.3, 0.7, 1.1];
        let both = [(0.9, 1.1), (1.2, -0.7), (0.8, 0.9)];
        let mut loading = vec![0.0f64; n_items * n_dims];
        for i in 0..4 {
            loading[i * 2] = a0[i];
            loading[(4 + i) * 2 + 1] = a1[i];
        }
        for i in 0..3 {
            loading[(8 + i) * 2] = both[i].0;
            loading[(8 + i) * 2 + 1] = both[i].1;
        }
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.16 * i as f64).collect();
        let n = 4000usize;
        let mut rng = Lcg(777);
        let mut thetas = vec![0.0f64; n * n_dims];
        for j in 0..n {
            thetas[j * 2] = rng.normal();
            thetas[j * 2 + 1] = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = TwoPlConfig {
            q: 21,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
        for i in 0..n_items {
            for d in 0..n_dims {
                if pattern[i * n_dims + d] == 0 {
                    assert_eq!(res.loading[i * n_dims + d], 0.0, "unloaded exactly zero");
                }
            }
        }
        assert!(
            rmse(&res.loading, &loading) < 0.12,
            "loading RMSE {}",
            rmse(&res.loading, &loading)
        );
        assert!(
            res.loading[3 * 2] < -0.5,
            "negative dim0 loading recovered: {}",
            res.loading[3 * 2]
        );
        assert!(
            res.loading[9 * 2 + 1] < -0.3,
            "negative cross-loading: {}",
            res.loading[9 * 2 + 1]
        );
        let t0h: Vec<f64> = (0..n).map(|j| res.theta[j * 2]).collect();
        let t0t: Vec<f64> = (0..n).map(|j| thetas[j * 2]).collect();
        let t1h: Vec<f64> = (0..n).map(|j| res.theta[j * 2 + 1]).collect();
        let t1t: Vec<f64> = (0..n).map(|j| thetas[j * 2 + 1]).collect();
        // EAP shrinks toward the prior, so the true-vs-EAP correlation is bounded by test
        // information (not N); ~0.75-0.85 is the expected range. The POSITIVE sign is the key
        // faithfulness check (a dim-swap or sign bug would give a near-zero or negative corr).
        assert!(corr(&t0h, &t0t) > 0.70, "theta0 corr {}", corr(&t0h, &t0t));
        assert!(corr(&t1h, &t1t) > 0.70, "theta1 corr {}", corr(&t1h, &t1t));
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "monotone");
        }
    }

    /// Deterministic reflection tests. (b) `flip_corr_dim` negates EXACTLY the off-diagonals that
    /// involve the flipped dimension (packed pairs (i,j), i<j) and leaves the rest untouched.
    /// (a) A fit whose largest pure anchor on a dimension is reverse-keyed (true loading strongly
    /// negative) is canonicalized by the reflection so that anchor ends POSITIVE and the dimension's
    /// co-loaders flip sign — deleting the reflection block leaves the raw negative anchor and fails.
    #[test]
    fn mirt_reflection_flips_negative_anchor() {
        // (b) flip_corr_dim on D=4: pairs are m0=(0,1) m1=(0,2) m2=(0,3) m3=(1,2) m4=(1,3) m5=(2,3).
        // Flipping dim 1 must negate exactly m0,(0,1) m3,(1,2) m4,(1,3) and leave m1,m2,m5 alone.
        let mut off = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        flip_corr_dim(&mut off, 4, 1);
        assert_eq!(off, vec![-0.1, 0.2, 0.3, -0.4, -0.5, 0.6]);

        // (a) reverse-keyed largest anchor on dim 0. Items 0,1 pure dim0; 2,3 pure dim1; 4 cross.
        let n_dims = 2usize;
        let n_items = 5usize;
        let pattern: Vec<u8> = vec![1, 0, 1, 0, 0, 1, 0, 1, 1, 1];
        let mut loading = vec![0.0f64; n_items * n_dims];
        loading[0 * 2] = -1.8; // reverse-keyed anchor, largest |loading| on dim 0
        loading[1 * 2] = 1.0;
        loading[2 * 2 + 1] = 1.2;
        loading[3 * 2 + 1] = 1.0;
        loading[4 * 2] = 0.9;
        loading[4 * 2 + 1] = 0.8;
        let intercept = vec![0.1, -0.2, 0.15, -0.1, 0.05];
        let n = 3000usize;
        let mut rng = Lcg(4242);
        let mut thetas = vec![0.0f64; n * n_dims];
        for j in 0..n {
            thetas[j * 2] = rng.normal();
            thetas[j * 2 + 1] = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = TwoPlConfig {
            q: 21,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
        // Canonical output: the largest pure anchor on dim 0 (item 0) ends POSITIVE; because the
        // whole dimension was reflected, the positively-keyed co-item (item 1) ends NEGATIVE.
        assert!(
            res.loading[0 * 2] > 0.8,
            "reflected anchor should be positive: {}",
            res.loading[0 * 2]
        );
        assert!(
            res.loading[1 * 2] < -0.3,
            "co-item flipped negative: {}",
            res.loading[1 * 2]
        );
    }

    /// Two-sided reduction anchor at D=2: the Halton QMC fit AGREES with the Gauss-Hermite fit
    /// within QMC error AND DIFFERS from it bit-wise. The disagreement guard is essential — a
    /// silent fallback to GH nodes on the Halton arm would make the two fits bit-identical and
    /// pass a one-sided within-error check trivially.
    #[test]
    fn qmc_reduces_to_gh_within_error_d2() {
        let n_dims = 2usize;
        let mut pattern: Vec<u8> = Vec::new();
        for _ in 0..4 {
            pattern.extend_from_slice(&[1, 0]);
        }
        for _ in 0..4 {
            pattern.extend_from_slice(&[0, 1]);
        }
        pattern.extend_from_slice(&[1, 1]);
        let n_items = 9usize;
        let mut loading = vec![0.0f64; n_items * n_dims];
        for i in 0..4 {
            loading[i * 2] = 1.0 + 0.15 * i as f64;
            loading[(4 + i) * 2 + 1] = 1.1 - 0.1 * i as f64;
        }
        loading[8 * 2] = 0.9;
        loading[8 * 2 + 1] = 0.8;
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.15 * i as f64).collect();
        let n = 2000usize;
        let mut rng = Lcg(1357);
        let mut thetas = vec![0.0f64; n * n_dims];
        for v in thetas.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let gh = fit_2pl(
            &y,
            &observed,
            &pattern,
            n,
            n_items,
            n_dims,
            &TwoPlConfig {
                q: 21,
                ..TwoPlConfig::default()
            },
        )
        .unwrap();
        let qmc = fit_2pl(
            &y,
            &observed,
            &pattern,
            n,
            n_items,
            n_dims,
            &TwoPlConfig {
                xi_rule: XiRuleKind::Halton,
                xi_points: 6000,
                xi_seed: 0,
                ..TwoPlConfig::default()
            },
        )
        .unwrap();
        let max_abs = gh
            .loading
            .iter()
            .zip(&qmc.loading)
            .chain(gh.intercept.iter().zip(&qmc.intercept))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_abs < 0.10,
            "QMC and GH disagree beyond QMC error: {max_abs}"
        );
        assert!(
            max_abs > 1e-10,
            "QMC fit is bit-identical to GH (silent GH fallback?)"
        );
    }

    /// Deterministic FD anchor on a FIXED Halton node set at D=4 with a NON-IDENTITY dims map
    /// [0,2,3] (so nodes are indexed dims[k] != k). Pins the analytic gradient and the full
    /// (n_i+1)^2 Hessian — including the off-diagonal cross-Hessian and the local->pattern
    /// dimension map — against central differences of item_obj to < 1e-4, on the SAME QMC nodes
    /// the estimator uses. This is deterministic (fixed seed) and node-source specific, so a
    /// cross-Hessian sign error or a dims[k] mis-map at D>3 fails here with no MC noise. (The
    /// grid LAYOUT itself is pinned independently in nodes::halton_grid_layout_is_prime_per_axis.)
    #[test]
    fn qmc_item_grad_hess_matches_fd_on_halton_d4() {
        let n_dims = 4usize;
        let dims = vec![0usize, 2, 3];
        let xn = build_xi_nodes(
            XiRule::Halton {
                n: 240,
                shift_seed: 0,
            },
            n_dims,
        )
        .unwrap();
        let nodes = &xn.grid;
        let n_nodes = xn.logw.len();
        let mut rng = Lcg(2718);
        let (mut n_ig, mut r_ig) = (vec![0.0f64; n_nodes], vec![0.0f64; n_nodes]);
        for g in 0..n_nodes {
            n_ig[g] = 1.0 + rng.next_f64() * 3.0;
            r_ig[g] = n_ig[g] * rng.next_f64();
        }
        let (a, b) = (vec![0.7f64, -0.6, 0.9], 0.2f64);
        let (ra, rb) = (1e-3, 1e-3);
        let np = dims.len() + 1;
        let (grad, amat) =
            item_grad_hess(&dims, &a, b, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
        let obj = |aa: &[f64], bb: f64| {
            item_obj(&dims, aa, bb, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb)
        };
        let eps = 1e-6;
        let perturb = |k: usize, s: f64| -> (Vec<f64>, f64) {
            let mut aa = a.clone();
            let mut bb = b;
            if k < dims.len() {
                aa[k] += s;
            } else {
                bb += s;
            }
            (aa, bb)
        };
        for k in 0..np {
            let (ap, bp) = perturb(k, eps);
            let (am, bm) = perturb(k, -eps);
            let fd = (obj(&ap, bp) - obj(&am, bm)) / (2.0 * eps);
            assert!(
                (grad[k] - fd).abs() < 1e-4,
                "grad[{k}] {} vs fd {fd}",
                grad[k]
            );
        }
        for jp in 0..np {
            let (ap, bp) = perturb(jp, eps);
            let (am, bm) = perturb(jp, -eps);
            let (gp, _) =
                item_grad_hess(&dims, &ap, bp, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
            let (gm, _) =
                item_grad_hess(&dims, &am, bm, &n_ig, &r_ig, nodes, n_dims, n_nodes, ra, rb);
            for k in 0..np {
                let dfd = (gp[k] - gm[k]) / (2.0 * eps);
                assert!((dfd + amat[k][jp]).abs() < 1e-4, "H[{k}][{jp}]");
            }
        }
    }

    /// D=4 orthogonal recovery on Halton QMC nodes (the headline D>3 capability the GH grid cannot
    /// reach). Confirmatory pattern: 2 pure anchors per dimension + cross-loaders INCLUDING a
    /// genuine negative one, which is asserted recovered < 0 explicitly (a compensation-sign bug on
    /// a shared dimension cannot be averaged away by an aggregate RMSE).
    #[test]
    fn qmc_recovers_compensatory_d4() {
        let n_dims = 4usize;
        let mut pattern: Vec<u8> = Vec::new();
        for d in 0..n_dims {
            for _ in 0..2 {
                let mut row = vec![0u8; n_dims];
                row[d] = 1;
                pattern.extend_from_slice(&row);
            }
        }
        // cross-loaders: (0,1) with a NEGATIVE dim-1 loading; (1,2); (2,3).
        pattern.extend_from_slice(&[1, 1, 0, 0]);
        pattern.extend_from_slice(&[0, 1, 1, 0]);
        pattern.extend_from_slice(&[0, 0, 1, 1]);
        let n_items = 2 * n_dims + 3; // 11
        let mut loading = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            loading[(2 * d) * n_dims + d] = 1.2 + 0.1 * d as f64;
            loading[(2 * d + 1) * n_dims + d] = 0.9;
        }
        let cross = 2 * n_dims;
        loading[cross * n_dims + 0] = 1.0;
        loading[cross * n_dims + 1] = -0.8; // the negative cross-loader
        loading[(cross + 1) * n_dims + 1] = 1.1;
        loading[(cross + 1) * n_dims + 2] = 0.7;
        loading[(cross + 2) * n_dims + 2] = 0.8;
        loading[(cross + 2) * n_dims + 3] = 1.0;
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.12 * i as f64).collect();
        let n = 2000usize;
        let mut rng = Lcg(9001);
        let mut thetas = vec![0.0f64; n * n_dims];
        for v in thetas.iter_mut() {
            *v = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 4000,
            xi_seed: 12345,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.n_dims, 4);
        for i in 0..n_items {
            for d in 0..n_dims {
                if pattern[i * n_dims + d] == 0 {
                    assert_eq!(res.loading[i * n_dims + d], 0.0, "unloaded exactly zero");
                }
            }
        }
        assert!(
            rmse(&res.loading, &loading) < 0.18,
            "loading RMSE {}",
            rmse(&res.loading, &loading)
        );
        // the negative cross-loader recovered negative (sign / compensation guard).
        assert!(
            res.loading[cross * n_dims + 1] < -0.3,
            "neg cross-loader: {}",
            res.loading[cross * n_dims + 1]
        );
        for d in 0..n_dims {
            let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
            let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
            assert!(corr(&th, &tt) > 0.55, "theta{d} corr {}", corr(&th, &tt));
        }
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "monotone");
        }
    }

    /// D=4 correlated WIRING on Halton QMC nodes: the correlated path runs at D>3 and returns a
    /// valid positive-definite, unit-diagonal Sigma whose off-diagonals recover the POSITIVE
    /// equicorrelation (truth rho=0.4) directionally, with monotone EM. This exercises the Cholesky
    /// node-map + the Sigma M-step at D>3. It is deliberately a directional/structural check, NOT a
    /// tight per-pair recovery: at an affordable point count the higher-prime Halton axes carry real
    /// QMC error in individual Sigma off-diagonals (documented ceiling), so a broken M-step (Sigma=I,
    /// non-PD, NaN, or sign-flipped) is what this catches. Tight per-pair Sigma recovery needs a much
    /// larger point count (n>=8000 at N>=4000 brings the worst pair within ~0.14 of the realized
    /// correlation) and is out of scope for a fast test.
    #[test]
    fn qmc_recovers_correlated_d4() {
        let n_dims = 4usize;
        // pure anchors: 2 per dim (identification under correlation needs pure indicators).
        let mut pattern: Vec<u8> = Vec::new();
        for d in 0..n_dims {
            for _ in 0..2 {
                let mut row = vec![0u8; n_dims];
                row[d] = 1;
                pattern.extend_from_slice(&row);
            }
        }
        let n_items = 2 * n_dims; // 8, all pure
        let mut loading = vec![0.0f64; n_items * n_dims];
        for d in 0..n_dims {
            loading[(2 * d) * n_dims + d] = 1.3;
            loading[(2 * d + 1) * n_dims + d] = 1.0;
        }
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * i as f64).collect();
        // Build an equicorrelation Sigma (all pairwise correlations = rho) and its Cholesky.
        let rho = 0.4f64;
        let mut sigma = vec![rho; n_dims * n_dims];
        for i in 0..n_dims {
            sigma[i * n_dims + i] = 1.0;
        }
        let lchol = chol_lower(&sigma, n_dims).unwrap();
        let n = 1500usize;
        let mut rng = Lcg(20260716);
        let mut thetas = vec![0.0f64; n * n_dims];
        for j in 0..n {
            let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
            for k in 0..n_dims {
                let mut t = 0.0f64;
                for m in 0..=k {
                    t += lchol[k * n_dims + m] * z[m];
                }
                thetas[j * n_dims + k] = t;
            }
        }
        // realized sample correlation of the drawn traits (the estimable target under finite N).
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 3000,
            xi_seed: 777,
            estimate_corr: true,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.corr.len(), n_dims * n_dims);
        for i in 0..n_dims {
            assert!(
                (res.corr[i * n_dims + i] - 1.0).abs() < 1e-9,
                "unit diagonal"
            );
        }
        // Structural: the returned Sigma is a valid positive-definite correlation matrix, and every
        // off-diagonal is a genuine (non-degenerate) correlation.
        assert!(chol_lower(&res.corr, n_dims).is_some(), "Sigma is PD");
        // Directional: the positive equicorrelation (truth rho=0.4) is recovered as a clearly
        // POSITIVE mean off-diagonal. A broken Sigma M-step returning I gives mean 0; a sign flip
        // gives a negative mean. We do NOT assert closeness to the realized ~0.41: at this
        // affordable point count the higher-prime Halton axes bias the recovered correlations UPWARD
        // by ~0.15 on the mean (the documented QMC ceiling), so tight closeness needs a much larger n.
        let mut rec_sum = 0.0f64;
        let mut cnt = 0.0f64;
        for i in 0..n_dims {
            for j in (i + 1)..n_dims {
                assert!(
                    res.corr[i * n_dims + j].abs() < 0.999,
                    "off-diagonal not degenerate"
                );
                rec_sum += res.corr[i * n_dims + j];
                cnt += 1.0;
            }
        }
        let rec_mean = rec_sum / cnt;
        assert!(
            rec_mean > 0.2,
            "recovered mean correlation {rec_mean} not clearly positive"
        );
        assert!(
            rec_mean < 0.85,
            "recovered mean correlation {rec_mean} implausibly high"
        );
        // The correlated path is NOT strictly step-monotone under QMC: the Sigma M-step
        // reparametrizes the integration nodes (theta_g = L(Sigma) z_g), so each Sigma gives a
        // different QMC quadrature of ITS marginal likelihood and the fixed-node monotonicity that
        // the ORTHOGONAL path enjoys (Sigma = I, nodes never move) no longer holds exactly. What
        // does hold is overall ASCENT and only QMC-scale per-step wobble. (Larger xi_points shrinks
        // the wobble; the orthogonal path is the choice when strict monotonicity is required.)
        // Overall ascent with only QMC-scale per-step wobble. The measured worst decrease here is
        // ~0.1 on a loglik scale of ~8050 (relative ~1e-5); the 1.0 bound gives 10x headroom while
        // still catching a Sigma M-step that genuinely harms the fit (which would drop it by >>1).
        let trace = &res.loglik_trace;
        let max_dec = trace
            .windows(2)
            .map(|w| (w[0] - w[1]).max(0.0))
            .fold(0.0f64, f64::max);
        assert!(
            max_dec < 1.0,
            "per-step decrease {max_dec} exceeds QMC wobble"
        );
        assert!(*trace.last().unwrap() >= trace[0], "overall EM ascent");
    }

    /// Rule-dependent validation: GH stays D<=3, QMC allows D<=6 and bounds xi_points; `q` is
    /// unused on the QMC arms (an out-of-set q must NOT reject a Halton fit).
    #[test]
    fn mirt_qmc_validates() {
        let n = 200usize;
        // GH rejects D=4; Halton accepts it (needs a D=4 pattern with pure anchors).
        let gh4 = TwoPlConfig {
            estimate_corr: false,
            ..TwoPlConfig::default()
        };
        // build a minimal D=4 pattern (one pure anchor per dim) + data of the right shape.
        let n_dims4 = 4usize;
        let mut pat4: Vec<u8> = Vec::new();
        for d in 0..n_dims4 {
            let mut row = vec![0u8; n_dims4];
            row[d] = 1;
            pat4.extend_from_slice(&row);
        }
        let ni4 = n_dims4;
        let y4 = vec![1.0f64; n * ni4];
        let obs4 = vec![true; n * ni4];
        assert!(
            fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &gh4).is_err(),
            "GH D=4 rejected"
        );
        // Halton D=4 with an INVALID GH q (q ignored on the QMC arm) must SUCCEED.
        let ok = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 400,
            xi_seed: 1,
            q: 99,
            max_iter: 3,
            ..TwoPlConfig::default()
        };
        assert!(
            fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &ok).is_ok(),
            "Halton D=4 q=99 ok"
        );
        // Halton D=6 (the UPPER bound MIRT_MAX_DIMS_QMC = HALTON_PRIMES.len()) is ACCEPTED. Pins
        // the boundary so a shrink of the constant to 5 (silently rejecting valid D=6) is caught;
        // D=7 just below is REJECTED (beyond the prime axes).
        let mut pat6 = Vec::new();
        for d in 0..6 {
            let mut r = vec![0u8; 6];
            r[d] = 1;
            pat6.extend_from_slice(&r);
        }
        let y6 = vec![1.0f64; n * 6];
        let obs6 = vec![true; n * 6];
        let d6 = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 200,
            max_iter: 1,
            ..TwoPlConfig::default()
        };
        assert!(
            fit_2pl(&y6, &obs6, &pat6, n, 6, 6, &d6).is_ok(),
            "Halton D=6 accepted"
        );
        let d7 = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 100,
            ..TwoPlConfig::default()
        };
        let mut pat7 = Vec::new();
        for d in 0..7 {
            let mut r = vec![0u8; 7];
            r[d] = 1;
            pat7.extend_from_slice(&r);
        }
        let y7 = vec![1.0f64; n * 7];
        let obs7 = vec![true; n * 7];
        assert!(
            fit_2pl(&y7, &obs7, &pat7, n, 7, 7, &d7).is_err(),
            "Halton D=7 rejected"
        );
        // xi_points bounds: 0 rejected; MAX+1 rejected.
        let zero = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: 0,
            ..TwoPlConfig::default()
        };
        assert!(
            fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &zero).is_err(),
            "xi_points=0 rejected"
        );
        let huge = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: MIRT_MAX_NODES + 1,
            ..TwoPlConfig::default()
        };
        assert!(
            fit_2pl(&y4, &obs4, &pat4, n, ni4, n_dims4, &huge).is_err(),
            "xi_points>MAX rejected"
        );
        // MonteCarlo D=7 also rejected (its builder has no cap; validate is the sole guard).
        let mc7 = TwoPlConfig {
            xi_rule: XiRuleKind::MonteCarlo,
            xi_points: 100,
            ..TwoPlConfig::default()
        };
        assert!(
            fit_2pl(&y7, &obs7, &pat7, n, 7, 7, &mc7).is_err(),
            "MC D=7 rejected"
        );

        // Individually valid xi_points and item counts must not combine into an unbounded dense
        // E-step table. This input is tiny (one response per item), but without the aggregate guard
        // it attempts four 200_000 x 301 f64 tables before doing any statistical work.
        let table_items = MIRT_MAX_NODE_ITEM_CELLS / MIRT_MAX_NODES + 1;
        let table_y = vec![0.0; table_items];
        let table_obs = vec![true; table_items];
        let table_pattern = vec![1u8; table_items];
        let table_cfg = TwoPlConfig {
            xi_rule: XiRuleKind::Halton,
            xi_points: MIRT_MAX_NODES,
            max_iter: 1,
            ..TwoPlConfig::default()
        };
        let err = fit_2pl(
            &table_y,
            &table_obs,
            &table_pattern,
            1,
            table_items,
            1,
            &table_cfg,
        )
        .unwrap_err();
        assert!(err.contains("node * item table"), "{err}");
    }

    fn small_design() -> (Vec<u8>, Vec<f64>, Vec<f64>, usize) {
        let mut pattern: Vec<u8> = Vec::new();
        for _ in 0..3 {
            pattern.extend_from_slice(&[1, 0]);
        }
        for _ in 0..3 {
            pattern.extend_from_slice(&[0, 1]);
        }
        pattern.extend_from_slice(&[1, 1]);
        let n_items = 7usize;
        let mut loading = vec![0.0f64; n_items * 2];
        for i in 0..3 {
            loading[i * 2] = 1.0 + 0.2 * i as f64;
            loading[(3 + i) * 2 + 1] = 1.0 + 0.2 * i as f64;
        }
        loading[6 * 2] = 0.9;
        loading[6 * 2 + 1] = 0.8;
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.15 * i as f64).collect();
        (pattern, loading, intercept, n_items)
    }

    #[test]
    fn mirt_validates_and_handles_missing() {
        let (pattern, loading, intercept, n_items) = small_design();
        let (n, n_dims) = (400usize, 2usize);
        let mut rng = Lcg(31);
        let mut thetas = vec![0.0f64; n * n_dims];
        for j in 0..n {
            thetas[j * 2] = rng.normal();
            thetas[j * 2 + 1] = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let cfg = TwoPlConfig::default();
        let mut observed = vec![true; n * n_items];
        observed[0] = false;
        observed[n_items + 3] = false;
        assert!(fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).is_ok());
        let obs = vec![true; n * n_items];
        let allones = vec![1u8; n_items * n_dims];
        assert!(fit_2pl(&y, &obs, &allones, n, n_items, n_dims, &cfg).is_err());
        let mut badrow = pattern.clone();
        badrow[0] = 0;
        badrow[1] = 0;
        assert!(fit_2pl(&y, &obs, &badrow, n, n_items, n_dims, &cfg).is_err());
        let mut nopure = pattern.clone();
        for i in 0..3 {
            nopure[i * 2 + 1] = 1; // items 0,1,2 now load both dims -> dim0 has no pure anchor
        }
        assert!(fit_2pl(&y, &obs, &nopure, n, n_items, n_dims, &cfg).is_err());
        assert!(fit_2pl(&y, &obs, &vec![1u8; n_items * 4], n, n_items, 4, &cfg).is_err());
        let badq = TwoPlConfig {
            q: 10,
            ..TwoPlConfig::default()
        };
        assert!(fit_2pl(&y, &obs, &pattern, n, n_items, n_dims, &badq).is_err());
        let mut ybad = y.clone();
        ybad[5] = 2.0;
        assert!(fit_2pl(&ybad, &obs, &pattern, n, n_items, n_dims, &cfg).is_err());
    }

    /// The final E-step is a genuine evaluated stopping point: meeting tolerance there is
    /// convergence even when it follows the last permitted M-step; otherwise exhaustion stays
    /// explicit and reports the observed stopping metric.
    #[test]
    fn mirt_reports_final_stopping_evidence() {
        let pattern = vec![1u8, 0, 0, 1];
        let balanced = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let observed = vec![true; balanced.len()];
        let cfg = TwoPlConfig {
            q: 7,
            max_iter: 1,
            ..TwoPlConfig::default()
        };
        let stable = fit_2pl(&balanced, &observed, &pattern, 4, 2, 2, &cfg).unwrap();
        assert!(stable.converged);
        assert_eq!(stable.termination_reason, "converged");
        assert_eq!(stable.n_iter, cfg.max_iter);
        assert_eq!(stable.loglik_trace.len(), 2);
        assert!(stable.final_loglik_change <= cfg.tol);

        let mut y = vec![0.0f64; 20 * 4];
        for p in 0..20 {
            y[p * 4] = if p % 5 == 0 { 0.0 } else { 1.0 };
            y[p * 4 + 1] = if p % 3 == 0 { 1.0 } else { 0.0 };
            y[p * 4 + 2] = if p % 4 == 0 { 0.0 } else { 1.0 };
            y[p * 4 + 3] = if p % 6 == 0 { 1.0 } else { 0.0 };
        }
        let observed = vec![true; y.len()];
        let pattern4 = vec![1u8, 0, 1, 0, 0, 1, 0, 1];
        let strict = TwoPlConfig {
            q: 7,
            max_iter: 1,
            tol: 1e-12,
            ..TwoPlConfig::default()
        };
        let unfinished = fit_2pl(&y, &observed, &pattern4, 20, 4, 2, &strict).unwrap();
        assert!(!unfinished.converged);
        assert_eq!(unfinished.termination_reason, "max_iter_reached");
        assert_eq!(unfinished.n_iter, strict.max_iter);
        assert_eq!(unfinished.loglik_trace.len(), 2);
        assert!(unfinished.final_loglik_change >= strict.tol);
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recover the compensatory loadings and traits
    /// at D=2 and D=3 under BOTH a normal and a right-skew (per-dim z-standardized, so only the
    /// SHAPE is misspecified) trait distribution. Loading RMSE is the primary target; the skew
    /// arm uses a looser bound (recovery is genuinely harder under shape misspecification).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_mirt_recovery_500() {
        let reps = 500usize;
        for &(n_dims, q, n) in [(2usize, 15usize, 3000usize), (3usize, 11usize, 2000usize)].iter() {
            let mut pattern: Vec<u8> = Vec::new();
            for d in 0..n_dims {
                for _ in 0..3 {
                    let mut r = vec![0u8; n_dims];
                    r[d] = 1;
                    pattern.extend_from_slice(&r);
                }
            }
            for d in 0..n_dims {
                let mut r = vec![0u8; n_dims];
                r[d] = 1;
                r[(d + 1) % n_dims] = 1;
                pattern.extend_from_slice(&r);
            }
            let n_items = 3 * n_dims + n_dims;
            let mut loading = vec![0.0f64; n_items * n_dims];
            for d in 0..n_dims {
                for k in 0..3 {
                    loading[(d * 3 + k) * n_dims + d] = 0.9 + 0.3 * k as f64;
                }
            }
            for d in 0..n_dims {
                let base = 3 * n_dims + d;
                loading[base * n_dims + d] = 1.0;
                loading[base * n_dims + (d + 1) % n_dims] = 0.7;
            }
            let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.12 * i as f64).collect();

            for &skew in [false, true].iter() {
                let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
                let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
                let mut nconv = 0usize;
                for rep in 0..reps {
                    let mut rng = Lcg(0x9E3779B97F4A7C15u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                        .wrapping_add(n_dims as u64 * 0x100000001B3));
                    let mut thetas = vec![0.0f64; n * n_dims];
                    for d in 0..n_dims {
                        let col: Vec<f64> = (0..n)
                            .map(|_| {
                                if skew {
                                    let mut cc = 0.0;
                                    for _ in 0..3 {
                                        let z = rng.normal();
                                        cc += z * z;
                                    }
                                    (cc - 3.0) / 6f64.sqrt()
                                } else {
                                    rng.normal()
                                }
                            })
                            .collect();
                        let m = col.iter().sum::<f64>() / n as f64;
                        let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                        let sd = v.sqrt();
                        for j in 0..n {
                            thetas[j * n_dims + d] = (col[j] - m) / sd;
                        }
                    }
                    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                    let observed = vec![true; n * n_items];
                    let cfg = TwoPlConfig {
                        q,
                        ..TwoPlConfig::default()
                    };
                    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                    if res.converged {
                        nconv += 1;
                    }
                    for w in res.loglik_trace.windows(2) {
                        assert!(w[1] >= w[0] - 1e-6, "monotone loglik (rep {rep})");
                    }
                    for i in 0..n_items {
                        for d in 0..n_dims {
                            let v = res.loading[i * n_dims + d];
                            if pattern[i * n_dims + d] == 0 {
                                assert_eq!(v, 0.0, "unloaded exactly zero");
                            } else {
                                assert!(v.is_finite() && v.abs() <= 10.0, "loading in bound");
                                let e = v - loading[i * n_dims + d];
                                lnum += e * e;
                                lden += 1.0;
                                lbias += e;
                            }
                        }
                    }
                    for d in 0..n_dims {
                        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                        let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
                        csum += corr(&th, &tt);
                        ccnt += 1.0;
                    }
                }
                let lrmse = (lnum / lden).sqrt();
                let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
                println!(
                    "[mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                     loadRMSE={lrmse:.4} loadBias={lb:.4} thetaCorr={tc:.3}"
                );
                // Thresholds calibrated from a 40-rep pilot (D2/D3 x normal/skew, N=3000/2000).
                assert!(conv > 0.95, "convergence {conv} (D={n_dims} skew={skew})");
                if skew {
                    // Shape misspecification: loadings attenuate (bias ~ -0.06..-0.09, expected);
                    // recovery is looser but the per-dim trait EAP stays clearly positive.
                    assert!(lrmse < 0.20, "skew loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.62, "skew theta corr {tc} (D={n_dims})");
                } else {
                    // Correctly-specified N(0,I): recovery is UNBIASED (the correctness signal).
                    assert!(lb.abs() < 0.03, "loading bias {lb} (D={n_dims})");
                    assert!(lrmse < 0.14, "loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.68, "theta corr {tc} (D={n_dims})");
                }
            }
        }
    }

    /// Literature-grade Monte-Carlo (>=500 reps) for the HIGH-DIMENSIONAL QMC path (`D > 3`, which
    /// the Gauss-Hermite product grid cannot reach): recover the compensatory loadings and traits
    /// at D=4 and D=5 on Halton QMC nodes, under a normal AND a per-dim-standardized right-skew
    /// trait. The QMC node set is FIXED across the EM run (so EM is monotone) and across reps (a
    /// deterministic quadrature); the finite-node QMC bias is what the looser-than-GH thresholds
    /// absorb, and averaging over reps is what pins the low-variance recovery the single fast test
    /// cannot. Per-rep finiteness + monotone-EM canaries; non-convergence tracked separately.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_qmc_mirt_recovery_500() {
        let reps = 500usize;
        for &(n_dims, xi_points, n) in [
            (4usize, 4000usize, 2000usize),
            (5usize, 6000usize, 1500usize),
        ]
        .iter()
        {
            // 2 pure anchors per dim (identification) + one cross-loader per dim.
            let mut pattern: Vec<u8> = Vec::new();
            for d in 0..n_dims {
                for _ in 0..2 {
                    let mut r = vec![0u8; n_dims];
                    r[d] = 1;
                    pattern.extend_from_slice(&r);
                }
            }
            for d in 0..n_dims {
                let mut r = vec![0u8; n_dims];
                r[d] = 1;
                r[(d + 1) % n_dims] = 1;
                pattern.extend_from_slice(&r);
            }
            let n_items = 2 * n_dims + n_dims;
            let mut loading = vec![0.0f64; n_items * n_dims];
            for d in 0..n_dims {
                loading[(2 * d) * n_dims + d] = 1.2;
                loading[(2 * d + 1) * n_dims + d] = 0.9;
            }
            for d in 0..n_dims {
                let base = 2 * n_dims + d;
                loading[base * n_dims + d] = 1.0;
                // alternate the cross-loader sign so a compensation-sign bug cannot hide.
                loading[base * n_dims + (d + 1) % n_dims] = if d % 2 == 0 { 0.7 } else { -0.7 };
            }
            let intercept: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.1 * i as f64).collect();

            for &skew in [false, true].iter() {
                let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
                let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
                let mut nconv = 0usize;
                for rep in 0..reps {
                    let mut rng = Lcg(0x9E3779B97F4A7C15u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03)
                        .wrapping_add(n_dims as u64 * 0x100000001B3));
                    let mut thetas = vec![0.0f64; n * n_dims];
                    for d in 0..n_dims {
                        let col: Vec<f64> = (0..n)
                            .map(|_| {
                                if skew {
                                    let mut cc = 0.0;
                                    for _ in 0..3 {
                                        let z = rng.normal();
                                        cc += z * z;
                                    }
                                    (cc - 3.0) / 6f64.sqrt()
                                } else {
                                    rng.normal()
                                }
                            })
                            .collect();
                        let m = col.iter().sum::<f64>() / n as f64;
                        let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                        let sd = v.sqrt();
                        for j in 0..n {
                            thetas[j * n_dims + d] = (col[j] - m) / sd;
                        }
                    }
                    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                    let observed = vec![true; n * n_items];
                    let cfg = TwoPlConfig {
                        xi_rule: XiRuleKind::Halton,
                        xi_points,
                        xi_seed: 0x2545_F491_4F6C_DD1D,
                        ..TwoPlConfig::default()
                    };
                    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                    if res.converged {
                        nconv += 1;
                    }
                    assert!(
                        res.loglik_trace.iter().all(|v| v.is_finite()),
                        "finite loglik (rep {rep})"
                    );
                    for w in res.loglik_trace.windows(2) {
                        assert!(w[1] >= w[0] - 1e-6, "monotone loglik (rep {rep})");
                    }
                    for i in 0..n_items {
                        for d in 0..n_dims {
                            let v = res.loading[i * n_dims + d];
                            if pattern[i * n_dims + d] == 0 {
                                assert_eq!(v, 0.0, "unloaded exactly zero");
                            } else {
                                assert!(v.is_finite() && v.abs() <= 10.0, "loading in bound");
                                let e = v - loading[i * n_dims + d];
                                lnum += e * e;
                                lden += 1.0;
                                lbias += e;
                            }
                        }
                    }
                    assert!(
                        res.theta.iter().all(|v| v.is_finite()),
                        "finite theta (rep {rep})"
                    );
                    for d in 0..n_dims {
                        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + d]).collect();
                        let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + d]).collect();
                        csum += corr(&th, &tt);
                        ccnt += 1.0;
                    }
                }
                let lrmse = (lnum / lden).sqrt();
                let (lb, tc, conv) = (lbias / lden, csum / ccnt, nconv as f64 / reps as f64);
                println!(
                    "[qmc-mirt MC D={n_dims} xi={xi_points} N={n} skew={skew}] reps={reps} \
                     conv={conv:.3} loadRMSE={lrmse:.4} loadBias={lb:.4} thetaCorr={tc:.3}"
                );
                // Looser than the GH MC: QMC carries an O(N^-1 (log N)^D) finite-node bias that
                // grows with D. Calibrated from a 50-rep pilot at D=4/5 x normal/skew (conv=1.000;
                // normal loadRMSE 0.13/0.17, bias ~0.01; skew loadRMSE 0.16/0.21, bias ~-0.07/-0.09;
                // thetaCorr 0.58-0.64) with margin for the 500-rep estimate.
                assert!(conv > 0.90, "convergence {conv} (D={n_dims} skew={skew})");
                if skew {
                    assert!(lrmse < 0.26, "skew loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.50, "skew theta corr {tc} (D={n_dims})");
                } else {
                    assert!(lb.abs() < 0.06, "loading bias {lb} (D={n_dims})");
                    assert!(lrmse < 0.19, "loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.55, "theta corr {tc} (D={n_dims})");
                }
            }
        }
    }

    // ----- Correlated-Sigma extension (theta ~ MVN(0, Sigma)) -----

    /// Draw N x D standard normals correlated through L = chol(Sigma): theta = L z.
    fn draw_corr(l: &[f64], n: usize, d: usize, rng: &mut Lcg) -> Vec<f64> {
        let mut th = vec![0.0f64; n * d];
        for j in 0..n {
            let z: Vec<f64> = (0..d).map(|_| rng.normal()).collect();
            for k in 0..d {
                let mut t = 0.0;
                for i in 0..=k {
                    t += l[k * d + i] * z[i];
                }
                th[j * d + k] = t;
            }
        }
        th
    }

    /// Realized sample correlation off-diagonals (pairs i<j) of an N x D trait matrix.
    fn sample_corr(th: &[f64], n: usize, d: usize) -> Vec<f64> {
        let mut mean = vec![0.0f64; d];
        for j in 0..n {
            for k in 0..d {
                mean[k] += th[j * d + k];
            }
        }
        for m in mean.iter_mut() {
            *m /= n as f64;
        }
        let mut var = vec![0.0f64; d];
        let mut off = Vec::new();
        for i in 0..d {
            for j in 0..n {
                var[i] += (th[j * d + i] - mean[i]).powi(2);
            }
        }
        for i in 0..d {
            for k in i + 1..d {
                let mut cov = 0.0;
                for j in 0..n {
                    cov += (th[j * d + i] - mean[i]) * (th[j * d + k] - mean[k]);
                }
                off.push(cov / (var[i] * var[k]).sqrt());
            }
        }
        off
    }

    /// estimate_corr = false reports Sigma = I exactly and keeps the orthogonal parameter count.
    #[test]
    fn mirt_estimate_corr_false_is_identity() {
        let (pattern, loading, intercept, n_items) = small_design();
        let (n, n_dims) = (300usize, 2usize);
        let mut rng = Lcg(5);
        let mut thetas = vec![0.0f64; n * n_dims];
        for t in thetas.iter_mut() {
            *t = rng.normal();
        }
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_2pl(
            &y,
            &observed,
            &pattern,
            n,
            n_items,
            n_dims,
            &TwoPlConfig::default(),
        )
        .unwrap();
        assert_eq!(res.corr, vec![1.0, 0.0, 0.0, 1.0], "Sigma == I exactly");
        let nfree = pattern.iter().filter(|&&v| v == 1).count();
        assert_eq!(res.n_parameters, nfree + n_items, "no extra corr params");
    }

    /// flip_corr_dim negates exactly the correlations that involve the flipped dimension.
    #[test]
    fn mirt_flip_corr_dim_negates_involving_dim() {
        // D=3, off-diagonal order (0,1),(0,2),(1,2).
        let mut r = vec![0.3f64, -0.2, 0.5];
        flip_corr_dim(&mut r, 3, 0); // negate pairs touching dim 0: (0,1),(0,2); (1,2) unchanged
        assert_eq!(r, vec![-0.3, 0.2, 0.5]);
        flip_corr_dim(&mut r, 3, 1); // negate pairs touching dim 1: (0,1),(1,2); (0,2) unchanged
        assert_eq!(r, vec![0.3, 0.2, -0.5]);
    }

    /// Deterministic FD anchor: the analytic correlation gradient matches central finite
    /// differences of Q_prior at a Sigma with NONZERO off-diagonals and a non-diagonal C.
    #[test]
    fn mirt_sigma_grad_matches_finite_difference() {
        for &(d, ref r0, ref c) in [
            (2usize, vec![0.35f64], vec![1.2f64, 0.5, 0.5, 0.9]),
            (
                3usize,
                vec![0.3f64, -0.15, 0.25],
                vec![1.1f64, 0.4, 0.2, 0.4, 0.95, -0.3, 0.2, -0.3, 1.05],
            ),
        ]
        .iter()
        {
            let sigma = build_corr(r0, d);
            let g = sigma_grad(&sigma, c, d).unwrap();
            let eps = 1e-6;
            for m in 0..r0.len() {
                let mut rp = r0.clone();
                let mut rm = r0.clone();
                rp[m] += eps;
                rm[m] -= eps;
                let qp = sigma_qprior(&build_corr(&rp, d), c, d).unwrap();
                let qm = sigma_qprior(&build_corr(&rm, d), c, d).unwrap();
                let fd = (qp - qm) / (2.0 * eps);
                assert!(
                    (g[m] - fd).abs() < 1e-5,
                    "D={d} grad[{m}] {} vs fd {fd}",
                    g[m]
                );
            }
        }
    }

    /// Recover a KNOWN correlated Sigma (rho = 0.5) AND loadings at D=2, with the largest-|loading|
    /// PURE anchor on dim 0 genuinely NEGATIVE so the reflection FIRES: the reported correlation
    /// must then carry the flip-consistent sign (a missing Sigma sign-flip would report +rho).
    #[test]
    fn mirt_recovers_correlated_d2_with_reflection() {
        let n_dims = 2usize;
        let mut pattern: Vec<u8> = Vec::new();
        for _ in 0..4 {
            pattern.extend_from_slice(&[1, 0]);
        }
        for _ in 0..4 {
            pattern.extend_from_slice(&[0, 1]);
        }
        for _ in 0..2 {
            pattern.extend_from_slice(&[1, 1]);
        }
        let n_items = 10usize;
        let mut loading = vec![0.0f64; n_items * n_dims];
        // dim0 pure anchors: largest |.| is -1.6 (NEGATIVE) -> reflection flips dim 0.
        let a0 = [1.0, 0.8, -1.6, 1.1];
        let a1 = [1.2, 0.9, 1.4, 1.0];
        for i in 0..4 {
            loading[i * 2] = a0[i];
            loading[(4 + i) * 2 + 1] = a1[i];
        }
        loading[8 * 2] = 0.9;
        loading[8 * 2 + 1] = 0.8;
        loading[9 * 2] = 1.1;
        loading[9 * 2 + 1] = 0.7;
        let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.13 * i as f64).collect();
        let rho = 0.5;
        let lchol = chol_lower(&build_corr(&[rho], n_dims), n_dims).unwrap();
        let n = 5000usize;
        let mut rng = Lcg(4242);
        let thetas = draw_corr(&lchol, n, n_dims, &mut rng);
        let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = TwoPlConfig {
            q: 15,
            estimate_corr: true,
            ..TwoPlConfig::default()
        };
        let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert!(res.converged);
        // Sigma is a valid unit-diagonal correlation matrix.
        assert!((res.corr[0] - 1.0).abs() < 1e-12 && (res.corr[3] - 1.0).abs() < 1e-12);
        assert!((res.corr[1] - res.corr[2]).abs() < 1e-12, "symmetric");
        // The reflection fired on dim 0 (its true anchor was negative), so the reported theta_0
        // is negated -> the reported correlation is the flip-consistent -rho. The realized sample
        // correlation is the honest recovery target; after the flip its sign is negated.
        let r_true = sample_corr(&thetas, n, n_dims)[0];
        assert!(
            (res.corr[1] - (-r_true)).abs() < 0.06,
            "corr {} vs -R {}",
            res.corr[1],
            -r_true
        );
        assert!(
            res.corr[1] < -0.3,
            "flip-consistent NEGATIVE correlation, got {}",
            res.corr[1]
        );
        // Loadings recovered against the flip-adjusted truth (dim 0 negated by the reflection).
        let mut expected = loading.clone();
        for i in 0..n_items {
            expected[i * 2] = -expected[i * 2]; // dim 0 flipped
        }
        assert!(
            rmse(&res.loading, &expected) < 0.12,
            "loading RMSE {}",
            rmse(&res.loading, &expected)
        );
        assert!(
            res.loading[2 * 2] > 0.9,
            "flipped anchor now positive: {}",
            res.loading[2 * 2]
        );
        assert!(res.n_parameters == pattern.iter().filter(|&&v| v == 1).count() + n_items + 1);
        for w in res.loglik_trace.windows(2) {
            assert!(w[1] >= w[0] - 1e-6, "EM monotone with the Sigma M-step");
        }
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recover loadings AND the latent correlation at
    /// D=2 (rho=0.5) and D=3 (exchangeable rho=0.4, verified PD) under a normal and a NORTA
    /// right-skew marginal (single correlated normal -> monotone per-dim skew, so the copula
    /// keeps the sign; corr is scored against the REALIZED sample correlation R_rep, not nominal).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_corr_mirt_recovery_500() {
        let reps = 500usize;
        for &(n_dims, q, n, ref true_off) in [
            (2usize, 15usize, 3000usize, vec![0.5f64]),
            (3usize, 11usize, 2000usize, vec![0.4f64, 0.4, 0.4]), // exchangeable, eig 1.8,0.6,0.6
        ]
        .iter()
        {
            let sigma_true = build_corr(true_off, n_dims);
            let lchol = chol_lower(&sigma_true, n_dims).expect("true Sigma must be PD");
            // pattern: 3 pure anchors per dim + one cross-loader per consecutive pair.
            let mut pattern: Vec<u8> = Vec::new();
            for dd in 0..n_dims {
                for _ in 0..3 {
                    let mut r = vec![0u8; n_dims];
                    r[dd] = 1;
                    pattern.extend_from_slice(&r);
                }
            }
            for dd in 0..n_dims {
                let mut r = vec![0u8; n_dims];
                r[dd] = 1;
                r[(dd + 1) % n_dims] = 1;
                pattern.extend_from_slice(&r);
            }
            let n_items = 3 * n_dims + n_dims;
            let mut loading = vec![0.0f64; n_items * n_dims];
            for dd in 0..n_dims {
                for k in 0..3 {
                    loading[(dd * 3 + k) * n_dims + dd] = 0.9 + 0.3 * k as f64; // positive anchors
                }
            }
            for dd in 0..n_dims {
                let base = 3 * n_dims + dd;
                loading[base * n_dims + dd] = 1.0;
                loading[base * n_dims + (dd + 1) % n_dims] = 0.7;
            }
            let intercept: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.12 * i as f64).collect();
            let n_off = n_dims * (n_dims - 1) / 2;

            for &skew in [false, true].iter() {
                let (mut lnum, mut lden, mut lbias) = (0.0f64, 0.0f64, 0.0f64);
                let (mut cnum, mut cbias) = (0.0f64, 0.0f64);
                let (mut csum, mut ccnt) = (0.0f64, 0.0f64);
                let (mut nconv, mut interior) = (0usize, 0usize);
                for rep in 0..reps {
                    let mut rng = Lcg(0xD1B54A32D192ED03u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15)
                        .wrapping_add(n_dims as u64 * 0x100000001B3));
                    // NORTA: correlated normals z = L u; per-dim monotone right-skew then
                    // re-standardize (keeps the sign of the correlation, attenuated).
                    let mut thetas = draw_corr(&lchol, n, n_dims, &mut rng);
                    if skew {
                        for k in 0..n_dims {
                            for j in 0..n {
                                let z = thetas[j * n_dims + k];
                                thetas[j * n_dims + k] = (0.5 * z).exp(); // monotone lognormal skew
                            }
                            let col: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + k]).collect();
                            let m = col.iter().sum::<f64>() / n as f64;
                            let v = col.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / n as f64;
                            let sd = v.sqrt();
                            for j in 0..n {
                                thetas[j * n_dims + k] = (thetas[j * n_dims + k] - m) / sd;
                            }
                        }
                    }
                    let r_rep = sample_corr(&thetas, n, n_dims); // honest recovery target
                    let y = simulate(&loading, &intercept, &thetas, n, n_items, n_dims, &mut rng);
                    let observed = vec![true; n * n_items];
                    let cfg = TwoPlConfig {
                        q,
                        estimate_corr: true,
                        ..TwoPlConfig::default()
                    };
                    let res = fit_2pl(&y, &observed, &pattern, n, n_items, n_dims, &cfg).unwrap();
                    if res.converged {
                        nconv += 1;
                    }
                    for w in res.loglik_trace.windows(2) {
                        assert!(w[1] >= w[0] - 1e-6, "EM monotone (rep {rep})");
                    }
                    // Sigma invariants: unit diagonal, symmetric, PD, |off|<1, all finite.
                    for k in 0..n_dims {
                        assert!(
                            (res.corr[k * n_dims + k] - 1.0).abs() < 1e-9,
                            "unit diagonal"
                        );
                    }
                    assert!(chol_lower(&res.corr, n_dims).is_some(), "Sigma PD");
                    let mut pinned = false;
                    let off_est: Vec<f64> = {
                        let mut o = Vec::new();
                        for i in 0..n_dims {
                            for j in i + 1..n_dims {
                                let v = res.corr[i * n_dims + j];
                                assert!(v.is_finite() && v.abs() < 1.0, "corr in (-1,1)");
                                assert!((v - res.corr[j * n_dims + i]).abs() < 1e-12, "symmetric");
                                if v.abs() > 0.99 {
                                    pinned = true;
                                }
                                o.push(v);
                            }
                        }
                        o
                    };
                    if !pinned {
                        interior += 1;
                    }
                    // Loadings: pure anchors positive -> reflection never fires -> no flip; score
                    // vs truth directly.
                    for i in 0..n_items {
                        for dd in 0..n_dims {
                            let v = res.loading[i * n_dims + dd];
                            if pattern[i * n_dims + dd] == 0 {
                                assert_eq!(v, 0.0);
                            } else {
                                assert!(v.is_finite() && v.abs() <= 10.0);
                                let e = v - loading[i * n_dims + dd];
                                lnum += e * e;
                                lden += 1.0;
                                lbias += e;
                            }
                        }
                    }
                    for m in 0..n_off {
                        let e = off_est[m] - r_rep[m]; // vs realized correlation
                        cnum += e * e;
                        cbias += e;
                        // correlation sign matches the (positive) truth
                        assert!(off_est[m] > 0.0, "corr sign matches truth (rep {rep})");
                    }
                    for dd in 0..n_dims {
                        let th: Vec<f64> = (0..n).map(|j| res.theta[j * n_dims + dd]).collect();
                        let tt: Vec<f64> = (0..n).map(|j| thetas[j * n_dims + dd]).collect();
                        csum += corr(&th, &tt);
                        ccnt += 1.0;
                    }
                }
                let lrmse = (lnum / lden).sqrt();
                let crmse = (cnum / (reps * n_off) as f64).sqrt();
                let (lb, cb) = (lbias / lden, cbias / (reps * n_off) as f64);
                let (tc, conv) = (csum / ccnt, nconv as f64 / reps as f64);
                let int_frac = interior as f64 / reps as f64;
                println!(
                    "[corr-mirt MC D={n_dims} q={q} N={n} skew={skew}] reps={reps} conv={conv:.3} \
                     loadRMSE={lrmse:.4} loadBias={lb:.4} corrRMSE={crmse:.4} corrBias={cb:.4} \
                     thetaCorr={tc:.3} interior={int_frac:.3}"
                );
                assert!(conv > 0.95, "convergence {conv} (D={n_dims} skew={skew})");
                assert!(
                    int_frac > 0.95,
                    "Sigma interior fraction {int_frac} (D={n_dims})"
                );
                assert!(
                    crmse < 0.06,
                    "correlation RMSE vs R_rep {crmse} (D={n_dims} skew={skew})"
                );
                if skew {
                    assert!(lrmse < 0.20, "skew loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.62, "skew theta corr {tc} (D={n_dims})");
                } else {
                    assert!(lb.abs() < 0.03, "loading bias {lb} (D={n_dims})");
                    assert!(lrmse < 0.14, "loading RMSE {lrmse} (D={n_dims})");
                    assert!(tc > 0.68, "theta corr {tc} (D={n_dims})");
                }
            }
        }
    }
}
