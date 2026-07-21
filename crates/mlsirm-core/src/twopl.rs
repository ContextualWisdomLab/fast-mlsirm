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
//! correlated case maps the standard-normal node set through `theta_g = L(Sigma) z_g`; the item
//! M-step is reused verbatim on those mapped nodes, while the `Sigma` M-step ascends the
//! Gaussian-prior objective
//! `-0.5[log|Sigma| + tr(Sigma^{-1} C)]` over the free correlations (`C` the posterior second
//! moment). Because remapping a finite QMC set changes its approximated objective, every proposed
//! `Sigma` update is additionally backtracked against the actual finite-node marginal likelihood;
//! this positive-definite observed-objective guard preserves EM ascent and convergence.
//!
//! **Integration node rule (`xi_rule`).** The product Gauss-Hermite grid is exact for
//! near-polynomial integrands but its `Q^D` node count is exponential in `D`, so it is capped at
//! `D <= 3`. For `D = 4, 5, 6` the E-step integral is instead evaluated by **quasi-Monte-Carlo**
//! (`Halton`, the low-discrepancy default for the QMC path) or plain **Monte-Carlo** (`MonteCarlo`)
//! quadrature: `xi_points` points drawn from the standard-normal prior (Halton radical inverse ->
//! `inv_normal_cdf`, or seeded Gaussian draws), equal weights `1/xi_points`. This is Jank's (2005)
//! QMC-EM — only the E-step nodes/weights change; the per-item Newton M-step and the `Sigma`
//! M-step are byte-for-byte the same code on the swapped node set. Because the standard node set
//! is FIXED for the whole EM run, the orthogonal fit optimizes one QMC approximation directly. In
//! the correlated fit, the repository-specific observed-objective backtracking described above
//! compensates for the changing Cholesky-mapped finite node cloud. QMC still carries an
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
/// Upper bound on caller-controlled EM iterations.
const MIRT_MAX_ITER: usize = 100_000;

fn checked_grid_nodes(current: usize, q: usize) -> Result<usize, String> {
    current
        .checked_mul(q)
        .filter(|&n| n <= MIRT_MAX_NODES)
        .ok_or_else(|| format!("q^n_dims exceeds the node cap {MIRT_MAX_NODES}"))
}

fn should_stop_item_newton(accepted: bool, moved: f64) -> bool {
    !accepted || moved < 1e-9
}
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
    if !(1..=MIRT_MAX_ITER).contains(&cfg.max_iter) {
        return Err(format!("max_iter must be in 1..={MIRT_MAX_ITER}"));
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
                n_nodes = checked_grid_nodes(n_nodes, cfg.q)?;
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
    let table_cells =
        crate::checked_mul_usize(n_nodes, n_items, "node * item table size overflows usize")?;
    if table_cells > MIRT_MAX_NODE_ITEM_CELLS {
        return Err(format!(
            "node * item table has {table_cells} cells, exceeding the cap \
             {MIRT_MAX_NODE_ITEM_CELLS}; reduce nodes or items"
        ));
    }
    crate::checked_mul_usize(n_nodes, n_dims, "node-dimension size overflows")?;
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_l = crate::checked_mul_usize(n_items, n_dims, "n_items * n_dims overflows usize")?;
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

fn corr_line_search(
    r_off: &[f64],
    grad: &[f64],
    q0: f64,
    cmat: &[f64],
    d: usize,
) -> Option<Vec<f64>> {
    let mut alpha = 1.0f64;
    for _ in 0..40 {
        let candidate_offdiag: Vec<f64> = (0..r_off.len())
            .map(|m| (r_off[m] + alpha * grad[m]).clamp(-0.999, 0.999))
            .collect();
        let candidate = build_corr(&candidate_offdiag, d);
        if sigma_qprior(&candidate, cmat, d).is_some_and(|q1| q1 >= q0 - 1e-12) {
            return Some(candidate_offdiag);
        }
        alpha *= 0.5;
    }
    None
}

fn map_corr_nodes(r_off: &[f64], base_nodes: &[f64], d: usize, mapped: &mut [f64]) -> bool {
    let Some(lchol) = chol_lower(&build_corr(r_off, d), d) else {
        return false;
    };
    for (source, target) in base_nodes.chunks_exact(d).zip(mapped.chunks_exact_mut(d)) {
        for k in 0..d {
            target[k] = (0..=k).map(|j| lchol[k * d + j] * source[j]).sum();
        }
    }
    true
}

#[allow(clippy::too_many_arguments)]
fn marginal_loglik_on_nodes(
    y: &[f64],
    observed: &[bool],
    loading: &[f64],
    intercept: &[f64],
    dims_of: &[Vec<usize>],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    nodes: &[f64],
    logw: &[f64],
) -> f64 {
    let n_nodes = logw.len();
    let mut log_p1 = vec![0.0; n_nodes * n_items];
    let mut log_p0 = vec![0.0; n_nodes * n_items];
    for g in 0..n_nodes {
        for i in 0..n_items {
            let eta = dims_of[i].iter().fold(intercept[i], |value, &dimension| {
                value + loading[i * n_dims + dimension] * nodes[g * n_dims + dimension]
            });
            log_p1[g * n_items + i] = log_sigmoid(eta);
            log_p0[g * n_items + i] = log_sigmoid(-eta);
        }
    }
    let mut post = vec![0.0; n_nodes];
    let mut total = 0.0;
    for p in 0..n_persons {
        for g in 0..n_nodes {
            post[g] = (0..n_items).fold(logw[g], |value, i| {
                let idx = p * n_items + i;
                if observed[idx] {
                    let response = y[idx];
                    value
                        + response * log_p1[g * n_items + i]
                        + (1.0 - response) * log_p0[g * n_items + i]
                } else {
                    value
                }
            });
        }
        let maximum = post.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        total += maximum
            + post
                .iter()
                .map(|value| (value - maximum).exp())
                .sum::<f64>()
                .ln();
    }
    total
}

#[allow(clippy::too_many_arguments)]
fn corr_marginal_line_search(
    current: &[f64],
    target: &[f64],
    baseline: f64,
    y: &[f64],
    observed: &[bool],
    loading: &[f64],
    intercept: &[f64],
    dims_of: &[Vec<usize>],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    base_nodes: &[f64],
    logw: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut alpha = 1.0;
    let mut mapped = vec![0.0; base_nodes.len()];
    for _ in 0..20 {
        let candidate: Vec<f64> = current
            .iter()
            .zip(target)
            .map(|(&old, &new)| old + alpha * (new - old))
            .collect();
        // Both endpoints are positive-definite correlation matrices and the PD cone is convex, so
        // every interpolation candidate must map successfully.
        assert!(map_corr_nodes(&candidate, base_nodes, n_dims, &mut mapped));
        let candidate_ll = marginal_loglik_on_nodes(
            y, observed, loading, intercept, dims_of, n_persons, n_items, n_dims, &mapped, logw,
        );
        if candidate_ll >= baseline - 1e-10 {
            return Some((candidate, mapped));
        }
        alpha *= 0.5;
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn reflect_mirt_dimensions(
    loading: &mut [f64],
    theta: &mut [f64],
    r_off: &mut [f64],
    dims_of: &[Vec<usize>],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
) {
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
        if let Some(item) = anchor {
            if loading[item * n_dims + d] < 0.0 {
                for i in 0..n_items {
                    loading[i * n_dims + d] = -loading[i * n_dims + d];
                }
                for p in 0..n_persons {
                    theta[p * n_dims + d] = -theta[p * n_dims + d];
                }
                flip_corr_dim(r_off, n_dims, d);
            }
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
            )
            .expect("validated Halton dimensions and point count must build");
            (xn.grid, xn.logw)
        }
        XiRuleKind::MonteCarlo => {
            let xn = build_xi_nodes(
                XiRule::MonteCarlo {
                    n: cfg.xi_points,
                    seed: cfg.xi_seed.max(1),
                },
                n_dims,
            )
            .expect("validated Monte Carlo dimensions and point count must build");
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
        debug_assert!(
            den > 0.0,
            "every item has an observed response by validation"
        );
        let prop = (num / den).clamp(0.02, 0.98);
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
    // r_off = 0) and the buffer for correlated nodes theta_g = L z_g. Candidate Sigma updates are
    // backtracked against the actual finite-node marginal likelihood below.
    let d = n_dims;
    let n_off = d * (d - 1) / 2;
    let mut r_off = vec![0.0f64; n_off];
    let mut theta_nodes = if cfg.estimate_corr {
        vec![0.0f64; n_nodes * n_dims]
    } else {
        Vec::new()
    };

    for _ in 0..cfg.max_iter {
        if cfg.estimate_corr {
            assert!(map_corr_nodes(&r_off, &nodes, d, &mut theta_nodes));
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
                // A rejected backtracking step keeps the previous parameters and stops. Encoding
                // that decision separately makes the rare near-maximum path directly testable.
                let moved = accepted
                    .then(|| {
                        (0..ni).map(|k| (a_new[k] - a[k]).abs()).sum::<f64>() + (b_new - b).abs()
                    })
                    .unwrap_or(f64::INFINITY);
                if accepted {
                    a = a_new;
                    b = b_new;
                }
                if should_stop_item_newton(accepted, moved) {
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
            let current = r_off.clone();
            let baseline = marginal_loglik_on_nodes(
                y,
                observed,
                &loading,
                &intercept,
                &dims_of,
                n_persons,
                n_items,
                n_dims,
                &theta_nodes,
                &logw,
            );
            let mut target = current.clone();
            for _ in 0..cfg.newton_iter {
                let sigma = build_corr(&target, d);
                let grad = sigma_grad(&sigma, &cmat, d)
                    .expect("the accepted correlation matrix remains positive definite");
                let q0 = sigma_qprior(&sigma, &cmat, d)
                    .expect("the accepted correlation matrix remains positive definite");
                let gnorm = grad.iter().map(|x| x * x).sum::<f64>().sqrt();
                if gnorm < 1e-10 {
                    break;
                }
                let next = corr_line_search(&target, &grad, q0, &cmat, d);
                let Some(candidate) = next else { break };
                target = candidate;
            }
            if let Some((candidate, candidate_nodes)) = corr_marginal_line_search(
                &current, &target, baseline, y, observed, &loading, &intercept, &dims_of,
                n_persons, n_items, n_dims, &nodes, &logw,
            ) {
                r_off = candidate;
                theta_nodes = candidate_nodes;
            }
        }
        n_iter += 1;
    }

    // Final pass under the returned parameters: trait EAP for every person, and the marginal
    // loglik of those parameters (pushed when EM exited on max-iter, so the trace endpoint
    // matches the returned params — on convergence the last E-step already supplied it).
    if cfg.estimate_corr {
        assert!(map_corr_nodes(&r_off, &nodes, d, &mut theta_nodes));
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
    reflect_mirt_dimensions(
        &mut loading,
        &mut theta,
        &mut r_off,
        &dims_of,
        n_persons,
        n_items,
        n_dims,
    );

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
#[path = "../../../tests/unit/twopl_tests.rs"]
mod tests;
