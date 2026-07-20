//! Metropolis-Hastings Robbins-Monro (MH-RM) estimation of the confirmatory multidimensional 2PL
//! (Cai, 2010a, 2010b) — a STOCHASTIC-approximation EM that scales item factor analysis to a latent
//! dimensionality where the deterministic Gauss-Hermite (`q^D`) and quasi-Monte-Carlo E-steps of
//! [`crate::twopl::fit_2pl`] become infeasible.
//!
//! The model is the same general compensatory 2PL as [`crate::twopl`]:
//!
//! ```text
//! P(X_ij = 1 | theta_j) = sigmoid( sum_{d in S_i} a_id theta_jd + b_i ),   theta_j ~ MVN(0, I_D)
//! ```
//!
//! but the marginal likelihood integral over `theta_j` is not quadratured. Instead each cycle `k`:
//!
//! 1. **I-step (stochastic imputation).** For each person a short symmetric random-walk Metropolis
//!    chain draws `theta_j` from its current posterior `pi_j(theta) proportional to phi(theta; 0, I)
//!    prod_i P_i(y_ij | theta)`; the chain is PERSISTENT (warm-started from the previous cycle's
//!    draw), so no per-cycle burn-in is needed and a handful of sweeps suffice. The proposal SD `c`
//!    is tuned during burn-in toward a target acceptance rate.
//! 2. **RM step (stochastic approximation).** By Fisher's identity the imputed traits give an
//!    unbiased-in-the-limit Monte-Carlo estimate `s_k` of the marginal score and a complete-data
//!    information `H_k`; a Robbins-Monro recursion smooths `H_k` into `Gamma_k` and takes a single
//!    Newton-like step `xi <- xi + gain_k * Gamma_k^{-1} s_k`. The gain follows a constant-gain
//!    burn-in (`Metropolis-Hastings stochastic EM` that random-walks into the MLE neighbourhood) then
//!    a decreasing `gain_k = 1/(k - k0)^alpha` (`sum gain = inf`, `sum gain^2 < inf`) that converges
//!    almost surely to a marginal-score root (Robbins & Monro, 1951; Cai, 2010a).
//!
//! Because the item blocks are conditionally independent given `theta`, the score, information, and
//! RM step are BLOCK-DIAGONAL by item, so the per-item work reuses the closed-form logistic gradient
//! `X'(y - P)` and information `X'WX` directly (no quadrature, `D`-independent per-node cost). Per-item
//! observed-information standard errors follow the Louis (1982) identity
//! `I_obs = E[-d^2 l_c] - Var[d l_c]`, approximated by a parallel RM filter over the convergence stage
//! that subtracts the UNCENTERED per-person score cross-product `sum_p (y - P)^2 X X'` from the
//! complete-data information (the standard single-imputation `m = 1` form). This is NOT the exact
//! Louis missing information: only the AGGREGATE observed score vanishes at the root, so the
//! per-person score means are not removed and a leading-order PSD term is retained — where the block
//! stays positive-definite the resulting SE is CONSERVATIVE (mildly upward-biased), and where the
//! over-subtraction would leave it non-PD the block falls back to the complete-data (Fisher)
//! information. Exact per-person centering would need `m > 1` imputations per cycle (a follow-up).
//!
//! **Identification.** Unit trait variances fix the loading scale, `E[theta] = 0` the intercepts, and
//! a PURE single-dimension anchor item per dimension pins the rotation to the coordinate axes. The
//! remaining per-dimension reflection `(a_i.d, theta_d) -> (-a_i.d, -theta_d)` leaves the likelihood
//! invariant; because the stochastic iterates could otherwise drift between the two mirror modes and
//! corrupt the Robbins-Monro RUNNING AVERAGE of the loadings, the canonical sign (each dimension's
//! largest pure anchor loads positive) is enforced IN-LOOP every cycle — flipping the loading column,
//! the persistent `theta` chain, and the averaged trait together — and once more at the end.
//!
//! **Correlated factors (`estimate_corr`).** With `estimate_corr = false` (default) the factors are
//! ORTHOGONAL (`theta ~ MVN(0, I)`) and the fit is bit-identical to a run with the flag off. With
//! `estimate_corr = true` a free latent CORRELATION matrix `Phi` (unit diagonal) is estimated (Cai,
//! 2010b confirmatory item factor analysis): the MH acceptance prior uses `Phi^{-1}` (recomputed by
//! Cholesky each cycle) and the free off-diagonals ascend the Gaussian-prior objective
//! `Q(Phi) = -0.5[log|Phi| + tr(Phi^{-1} C)]` (`C` the imputed second moment) by a per-cycle
//! Robbins-Monro gradient step, PD-backtracked, reusing the `twopl.rs` correlation machinery
//! (`build_corr`, `sigma_grad`, `chol_lower`, `sym_inv_logdet`, `flip_corr_dim`). The polytomous item
//! families (reusing the `poly.rs` cell gradients as the complete-data score) are a natural extension
//! of the same MH-RM loop.
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010a). High-dimensional exploratory item factor analysis by a Metropolis-Hastings
//! Robbins-Monro algorithm. *Psychometrika, 75*(1), 33–57. https://doi.org/10.1007/s11336-009-9136-x
//!
//! Cai, L. (2010b). Metropolis-Hastings Robbins-Monro algorithm for confirmatory item factor
//! analysis. *Journal of Educational and Behavioral Statistics, 35*(3), 307–335.
//! https://doi.org/10.3102/1076998609353115
//!
//! Robbins, H., & Monro, S. (1951). A stochastic approximation method. *The Annals of Mathematical
//! Statistics, 22*(3), 400–407. https://doi.org/10.1214/aoms/1177729586
//!
//! Louis, T. A. (1982). Finding the observed information matrix when using the EM algorithm. *Journal
//! of the Royal Statistical Society: Series B (Methodological), 44*(2), 226–233.
//! https://doi.org/10.1111/j.2517-6161.1982.tb01203.x

use crate::mmle::{log_sigmoid, sigmoid_stable};
use crate::poly::solve_small;
use crate::twopl::{build_corr, chol_lower, flip_corr_dim, sigma_grad, sym_inv_logdet};

/// Maximum latent dimensions (MH-RM's whole point is high `D`; this only bounds the per-person
/// proposal work and the `D x D`-ish per-item blocks against pathological inputs).
const MHRM_MAX_DIMS: usize = 64;
/// Maximum persons/items product guard on the response allocation.
const MHRM_MAX_CELLS: usize = 200_000_000;
/// Symmetric loading clamp (loadings are NOT floored positive — reverse-keyed / suppressor
/// cross-loadings are representable; the reflection anchor fixes only the global per-dimension sign).
const MHRM_A_BOUND: f64 = 10.0;
/// Maximum polytomous response categories (bounds the per-item softmax work).
const MHRM_MAX_CAT: usize = 64;

/// Item response family fitted by [`fit_mhrm`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MhrmModel {
    /// Binary two-parameter logistic (default). The per-item parameter block is
    /// `[a_d for d in S_i, b_i]` and `base_i = sum_d a_id theta_d + b_i`.
    TwoPl,
    /// Generalized partial credit model (Muraki, 1992) with `n_cat` ordered categories and INTEGER
    /// scores `0..n_cat-1`. The per-item block is `[a_d for d in S_i, step_i1..step_i,{n_cat-1}]`
    /// (the baseline step `step_i0 = 0` is pinned); `base_i = sum_d a_id theta_d` carries NO
    /// intercept — the steps are the softmax category intercepts, `psi_k = k*base_i + step_ik`,
    /// `P(Y=k) = softmax_k(psi_k)`. `n_cat >= 2` (`n_cat = 2` is mathematically the 2PL but is fit on
    /// the polytomous softmax path, not the exact `log_sigmoid` path).
    Gpcm { n_cat: usize },
}

impl MhrmModel {
    /// Number of free CATEGORY parameters per item (2PL: the single intercept; GPCM: `n_cat - 1`
    /// step intercepts).
    #[inline]
    fn n_free_cat(self) -> usize {
        match self {
            MhrmModel::TwoPl => 1,
            MhrmModel::Gpcm { n_cat } => n_cat - 1,
        }
    }
    /// Response category count (`2` for the binary 2PL).
    #[inline]
    fn n_cat(self) -> usize {
        match self {
            MhrmModel::TwoPl => 2,
            MhrmModel::Gpcm { n_cat } => n_cat,
        }
    }
}

/// Configuration for [`fit_mhrm`].
#[derive(Clone, Copy, Debug)]
pub struct MhrmConfig {
    /// Maximum MH-RM cycles.
    pub max_cycles: usize,
    /// Constant-gain burn-in cycles `k0` (a Metropolis-Hastings stochastic EM; the proposal SD is
    /// tuned here and the decreasing gain starts at `k0 + 1`).
    pub burn_in: usize,
    /// Metropolis sweeps per person per cycle (`T_MH`; the persistent warm-started chain keeps this
    /// small).
    pub mh_steps: usize,
    /// Initial random-walk proposal SD `c` (`theta* = theta + c * N(0, I)`).
    pub proposal_sd: f64,
    /// Adapt `c` toward `target_accept` during burn-in (frozen afterwards).
    pub adapt_proposal: bool,
    /// Target Metropolis acceptance rate for the burn-in proposal tuning (random-walk optimum is
    /// ~0.234 in high `D`; the useful band is ~0.2-0.5).
    pub target_accept: f64,
    /// Constant gain used during burn-in (`gain_k = burn_in_gain` for `k <= burn_in`).
    pub burn_in_gain: f64,
    /// Decreasing-gain exponent `alpha` in `gain_k = 1/(k - k0)^alpha` (Robbins-Monro needs
    /// `alpha in (0.5, 1]`; `1.0` is the canonical `1/(k - k0)`).
    pub gain_exponent: f64,
    /// Convergence window `w`: stop when the running mean of `||xi^(k) - xi^(k-1)||` over the last
    /// `w` post-burn-in cycles falls below `tol` (MH-RM iterates are non-monotone, so a
    /// likelihood-decrease guard is NOT used).
    pub window: usize,
    /// Convergence tolerance on the windowed mean parameter change.
    pub tol: f64,
    /// Ridge added to the RM information diagonal before the per-item solve (conditioning only).
    pub ridge: f64,
    /// Accumulate the Louis (1982) observed-information standard errors over the convergence stage.
    pub estimate_se: bool,
    /// Estimate a free latent CORRELATION matrix `Phi` (`theta ~ MVN(0, Phi)`, unit diagonal;
    /// Cai 2010b confirmatory item factor analysis). When `false` (default), `Phi = I` (orthogonal
    /// factors) exactly — the orthogonal path is BIT-IDENTICAL to a fit with this off. When `true`,
    /// the MH acceptance prior uses `Phi^{-1}` and the free off-diagonals ascend the Gaussian-prior
    /// objective by a per-cycle Robbins-Monro gradient step (PD-backtracked).
    pub estimate_corr: bool,
    /// Item response family: `TwoPl` (default, binary) or `Gpcm { n_cat }` (Muraki 1992 ordered
    /// polytomous). The `TwoPl` path is unchanged / bit-identical.
    pub model: MhrmModel,
    /// PRNG seed (deterministic given the seed).
    pub seed: u64,
}

impl Default for MhrmConfig {
    fn default() -> Self {
        Self {
            max_cycles: 2000,
            burn_in: 200,
            mh_steps: 5,
            proposal_sd: 1.0,
            adapt_proposal: true,
            target_accept: 0.30,
            burn_in_gain: 1.0,
            gain_exponent: 1.0,
            window: 30,
            tol: 1e-3,
            ridge: 1e-6,
            estimate_se: true,
            estimate_corr: false,
            model: MhrmModel::TwoPl,
            seed: 0x9E37_79B9_7F4A_7C15,
        }
    }
}

/// Result of [`fit_mhrm`].
#[derive(Clone, Debug)]
pub struct MhrmResult {
    /// Free loadings `a_id`, row-major `J x D` (exactly `0.0` where `L_id = 0`), per-dimension
    /// reflection-canonicalized so each dimension's largest pure anchor loads positive.
    pub loading: Vec<f64>,
    /// Item intercepts `b_i`, length `J` (the 2PL category parameter; EMPTY for GPCM — see `step`).
    pub intercept: Vec<f64>,
    /// GPCM category step intercepts `step_ik`, row-major `J x (n_cat - 1)` (UNORDERED; EMPTY for the
    /// binary 2PL, which uses `intercept`).
    pub step: Vec<f64>,
    /// Response category count (`2` for the binary 2PL).
    pub n_cat: usize,
    /// Per-person trait EAP (Monte-Carlo mean of the imputed draws over the convergence stage),
    /// row-major `N x D`.
    pub theta: Vec<f64>,
    pub n_dims: usize,
    /// Latent correlation matrix `Phi`, row-major `D x D` (identity when `estimate_corr` is `false`;
    /// unit diagonal, estimated off-diagonals otherwise).
    pub corr: Vec<f64>,
    /// Louis (1982) block-diagonal (per-item) observed-information standard errors for the loadings,
    /// row-major `J x D` (`0.0` off-pattern; empty when `estimate_se` is `false`). Computed from the
    /// uncentered `m = 1` observed information (see the module docs): where the block is
    /// positive-definite the SE is mildly CONSERVATIVE (upward-biased); where the missing-information
    /// subtraction leaves it non-PD the block falls back to the complete-data (Fisher) information,
    /// which OMITS the missing information and so is a mild UNDER-estimate there.
    pub se_loading: Vec<f64>,
    /// Standard errors for the 2PL intercepts, length `J` (empty for GPCM / when `estimate_se` is
    /// `false`).
    pub se_intercept: Vec<f64>,
    /// Standard errors for the GPCM step intercepts, row-major `J x (n_cat - 1)` (empty for the 2PL /
    /// when `estimate_se` is `false`).
    pub se_step: Vec<f64>,
    /// Final tuned Metropolis acceptance rate.
    pub acceptance_rate: f64,
    pub n_cycles: usize,
    pub converged: bool,
    /// `converged` or `max_cycles_reached`.
    pub termination_reason: String,
    /// Windowed mean parameter change at termination.
    pub final_param_change: f64,
    /// `#{L_id = 1}` loadings `+ J * (n_cat - 1)` category parameters (`+ D(D-1)/2` free
    /// correlations when `estimate_corr`). For the binary 2PL `n_cat - 1 = 1` (one intercept per item).
    pub n_parameters: usize,
}

/// Deterministic LCG + Box-Muller normal (the crate's inline PRNG idiom; production because the MH
/// sampler runs inside the fit, not in tests).
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

/// `log P(y | theta)` for one item given its loaded-dimension parameters and the model family.
#[inline]
fn item_logp(model: MhrmModel, params_i: &[f64], dims_i: &[usize], theta_p: &[f64], y: usize) -> f64 {
    let l = dims_i.len();
    match model {
        MhrmModel::TwoPl => {
            let mut base = params_i[l]; // intercept b_i is the last slot
            for (t, &d) in dims_i.iter().enumerate() {
                base += params_i[t] * theta_p[d];
            }
            if y == 1 {
                log_sigmoid(base)
            } else {
                log_sigmoid(-base)
            }
        }
        MhrmModel::Gpcm { n_cat } => {
            // base = sum_d a_id theta_d (NO intercept); the steps are params_i[l..l+n_cat-1].
            let mut base = 0.0;
            for (t, &d) in dims_i.iter().enumerate() {
                base += params_i[t] * theta_p[d];
            }
            gpcm_logp_at(base, &params_i[l..l + n_cat - 1], y, n_cat)
        }
    }
}

/// Scalar `log P(Y = y | theta)` for one GPCM item WITHOUT allocating the full category vector (the
/// MH I-step calls this `O(N * mh_steps * n_items)` per cycle): `psi_k = k*base + step_k`
/// (`step_0 = 0`), `logP(y) = psi_y - logsumexp_k psi_k`.
#[inline]
fn gpcm_logp_at(base: f64, steps: &[f64], y: usize, n_cat: usize) -> f64 {
    let psi = |k: usize| -> f64 { (k as f64) * base + if k == 0 { 0.0 } else { steps[k - 1] } };
    let mut m = f64::NEG_INFINITY;
    for k in 0..n_cat {
        let p = psi(k);
        if p > m {
            m = p;
        }
    }
    let mut se = 0.0;
    for k in 0..n_cat {
        se += (psi(k) - m).exp();
    }
    psi(y) - (m + se.ln())
}

/// Robbins-Monro gain at cycle `k`: the constant `burn_in_gain` during the burn-in
/// (`k <= burn_in`), then the decreasing `1/(k - burn_in)^gain_exponent` (so `sum gain = inf`,
/// `sum gain^2 < inf`; the first convergence-stage cycle `k = burn_in + 1` has gain `1.0`).
#[inline]
pub(crate) fn gain_at(k: usize, burn_in: usize, burn_in_gain: f64, gain_exponent: f64) -> f64 {
    if k <= burn_in {
        burn_in_gain
    } else {
        1.0 / ((k - burn_in) as f64).powf(gain_exponent)
    }
}

/// Per-item complete-data score, complete-data (expected) information `H` (the RM Newton
/// preconditioner), and the Louis missing-information contribution `H - sum_p s_p s_p'`, all at the
/// imputed traits over the observed persons for item `i`. Returns `(score[p_i], H[p_i * p_i],
/// louis[p_i * p_i])`.
///
/// - `TwoPl`: `X_p = [theta_pd for d in S_i, 1]`, score `= sum_p X_p (y_p - P_p)`, `H = sum_p w_p X_p
///   X_p'` (`w_p = P_p(1-P_p)`, the exact 2PL Hessian), `louis = sum_p (w_p - r_p^2) X_p X_p'`;
///   `p_i = |S_i| + 1`.
/// - `Gpcm { n_cat }`: the CLOSED-FORM multinomial Hessian `H = sum_p J_p' (diag(P) - PP') J_p`
///   (data-INDEPENDENT given theta), where the design row `J_p[k]` is `d psi_k / d param`
///   (`psi_k = k*base + step_k`): `d/d a_id = k*theta_pd`, `d/d step_j = [k == j]`. The score is
///   `sum_p J_p' resid_p` (`resid_pk = [k == y_p] - P_pk`, `g_scores` implicitly dropped since the
///   integer scores are fixed), and `louis = H - sum_p s_p s_p'` (exactly the binary structure).
///   `p_i = |S_i| + (n_cat - 1)` (slopes then the `n_cat-1` step intercepts).
#[allow(clippy::too_many_arguments)]
pub(crate) fn item_score_info(
    model: MhrmModel,
    params_i: &[f64],
    dims_i: &[usize],
    theta: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    i: usize,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let li = dims_i.len();
    match model {
        MhrmModel::TwoPl => {
            let pi = li + 1;
            let mut s = vec![0.0f64; pi];
            let mut hmat = vec![0.0f64; pi * pi];
            let mut hobs = vec![0.0f64; pi * pi];
            let mut x = vec![0.0f64; pi];
            for p in 0..n_persons {
                if !observed.map_or(true, |o| o[p * n_items + i]) {
                    continue;
                }
                let mut base = params_i[li];
                for (t, &d) in dims_i.iter().enumerate() {
                    base += params_i[t] * theta[p * n_dims + d];
                }
                let pp = sigmoid_stable(base);
                let resid = y[p * n_items + i] as f64 - pp;
                let w = pp * (1.0 - pp);
                let r2 = resid * resid;
                for (t, &d) in dims_i.iter().enumerate() {
                    x[t] = theta[p * n_dims + d];
                }
                x[li] = 1.0;
                for a in 0..pi {
                    s[a] += resid * x[a];
                    for b in 0..pi {
                        let xx = x[a] * x[b];
                        hmat[a * pi + b] += w * xx;
                        hobs[a * pi + b] += (w - r2) * xx;
                    }
                }
            }
            (s, hmat, hobs)
        }
        MhrmModel::Gpcm { n_cat } => {
            let nf = n_cat - 1;
            let pi = li + nf;
            let mut s = vec![0.0f64; pi];
            let mut hmat = vec![0.0f64; pi * pi];
            let mut hobs = vec![0.0f64; pi * pi];
            // reusable per-person buffers: design J (n_cat x pi), category probs, score, u = J' P
            let mut jmat = vec![0.0f64; n_cat * pi];
            let mut pvec = vec![0.0f64; n_cat];
            let mut sc = vec![0.0f64; pi];
            let mut u = vec![0.0f64; pi];
            for p in 0..n_persons {
                if !observed.map_or(true, |o| o[p * n_items + i]) {
                    continue;
                }
                let mut base = 0.0;
                for (t, &d) in dims_i.iter().enumerate() {
                    base += params_i[t] * theta[p * n_dims + d];
                }
                // softmax P over psi_k = k*base + step_k (step_0 = 0)
                let mut m = f64::NEG_INFINITY;
                for k in 0..n_cat {
                    let psi = (k as f64) * base + if k == 0 { 0.0 } else { params_i[li + k - 1] };
                    pvec[k] = psi;
                    if psi > m {
                        m = psi;
                    }
                }
                let mut denom = 0.0;
                for k in 0..n_cat {
                    pvec[k] = (pvec[k] - m).exp();
                    denom += pvec[k];
                }
                for k in 0..n_cat {
                    pvec[k] /= denom;
                }
                let yy = y[p * n_items + i];
                // design J[k][a] = d psi_k / d param_a: a-slots k*theta_d, step-slot (li+k-1) = 1
                // (off-diagonal step slots stay 0 -- never written, jmat init 0).
                for k in 0..n_cat {
                    for (t, &d) in dims_i.iter().enumerate() {
                        jmat[k * pi + t] = (k as f64) * theta[p * n_dims + d];
                    }
                    if k >= 1 {
                        jmat[k * pi + (li + k - 1)] = 1.0;
                    }
                }
                // score sc[a] = sum_k J[k][a] resid_k, resid_k = [k==yy] - P_k; u[a] = sum_k J[k][a] P_k
                for a in 0..pi {
                    let mut sacc = 0.0;
                    let mut uacc = 0.0;
                    for k in 0..n_cat {
                        let jka = jmat[k * pi + a];
                        sacc += jka * ((if k == yy { 1.0 } else { 0.0 }) - pvec[k]);
                        uacc += jka * pvec[k];
                    }
                    sc[a] = sacc;
                    u[a] = uacc;
                    s[a] += sacc;
                }
                // H[a][b] = sum_k J[k][a] P_k J[k][b] - u[a] u[b]  ( = J'(diag P - P P')J );
                // louis = H - s_p s_p'
                for a in 0..pi {
                    for b in 0..pi {
                        let mut jpj = 0.0;
                        for k in 0..n_cat {
                            jpj += jmat[k * pi + a] * pvec[k] * jmat[k * pi + b];
                        }
                        let hp = jpj - u[a] * u[b];
                        hmat[a * pi + b] += hp;
                        hobs[a * pi + b] += hp - sc[a] * sc[b];
                    }
                }
            }
            (s, hmat, hobs)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    cfg: &MhrmConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(1..=MHRM_MAX_DIMS).contains(&n_dims) {
        return Err(format!(
            "n_dims must be in 1..={MHRM_MAX_DIMS}; got {n_dims}"
        ));
    }
    let cells = n_persons
        .checked_mul(n_items)
        .ok_or("n_persons * n_items overflow")?;
    if cells > MHRM_MAX_CELLS {
        return Err(format!(
            "n_persons * n_items = {cells} exceeds the cap {MHRM_MAX_CELLS}"
        ));
    }
    if y.len() != cells {
        return Err(format!("y has {} entries; expected {cells}", y.len()));
    }
    if let Some(o) = observed {
        if o.len() != cells {
            return Err(format!(
                "observed has {} entries; expected {cells}",
                o.len()
            ));
        }
    }
    if loading_pattern.len() != n_items * n_dims {
        return Err(format!(
            "loading_pattern has {} entries; expected {}",
            loading_pattern.len(),
            n_items * n_dims
        ));
    }
    if loading_pattern.iter().any(|&v| v > 1) {
        return Err("loading_pattern entries must be 0 or 1".into());
    }
    // every observed response is in 0..n_cat (0/1 for the binary 2PL, 0..K-1 for GPCM). The upper
    // bound makes MHRM_MAX_CAT live: the per-item softmax buffers and the coverage `seen` vectors
    // below both allocate on the order of n_cat, so an unbounded n_cat is a DoS/OOM vector.
    let n_cat = cfg.model.n_cat();
    if !(2..=MHRM_MAX_CAT).contains(&n_cat) {
        return Err(format!("model n_cat must be in 2..={MHRM_MAX_CAT}; got {n_cat}"));
    }
    for p in 0..n_persons {
        for i in 0..n_items {
            let seen = observed.map_or(true, |o| o[p * n_items + i]);
            if seen && y[p * n_items + i] >= n_cat {
                return Err(format!(
                    "responses must be in 0..{n_cat} where observed; found {}",
                    y[p * n_items + i]
                ));
            }
        }
    }
    // GPCM: every declared category must be observed at least once per item, else the
    // corresponding step intercept is unidentified (Muraki, 1992). Mirrors gpcm.rs; the
    // binary 2PL path skips this (an all-0 or all-1 item is still a valid regularized cell).
    if matches!(cfg.model, MhrmModel::Gpcm { .. }) {
        for i in 0..n_items {
            let mut seen = vec![false; n_cat];
            let mut any = false;
            for p in 0..n_persons {
                if observed.map_or(true, |o| o[p * n_items + i]) {
                    any = true;
                    seen[y[p * n_items + i]] = true;
                }
            }
            if !any {
                return Err(format!("item {i} has no observed responses"));
            }
            if let Some(k) = (0..n_cat).find(|&k| !seen[k]) {
                return Err(format!(
                    "item {i} category {k} is never observed (unidentified GPCM step); \
                     every declared category must be observed"
                ));
            }
        }
    }
    // every item loads at least one dimension; every dimension has a PURE single-dimension anchor
    for i in 0..n_items {
        if (0..n_dims).all(|d| loading_pattern[i * n_dims + d] == 0) {
            return Err(format!(
                "item {i} loads no dimension (all-zero pattern row)"
            ));
        }
    }
    for d in 0..n_dims {
        let has_pure_anchor = (0..n_items).any(|i| {
            loading_pattern[i * n_dims + d] == 1
                && (0..n_dims)
                    .filter(|&d2| loading_pattern[i * n_dims + d2] == 1)
                    .count()
                    == 1
        });
        if !has_pure_anchor {
            return Err(format!(
                "dimension {d} has no pure single-dimension anchor item (rotation not identified)"
            ));
        }
    }
    if cfg.max_cycles == 0 || cfg.burn_in >= cfg.max_cycles {
        return Err("require 0 < burn_in < max_cycles".into());
    }
    if cfg.mh_steps == 0 {
        return Err("mh_steps must be positive".into());
    }
    if !cfg.proposal_sd.is_finite() || cfg.proposal_sd <= 0.0 {
        return Err("proposal_sd must be finite and positive".into());
    }
    if !(0.0..=1.0).contains(&cfg.target_accept) {
        return Err("target_accept must be in [0, 1]".into());
    }
    if !cfg.burn_in_gain.is_finite() || cfg.burn_in_gain <= 0.0 {
        return Err("burn_in_gain must be finite and positive".into());
    }
    if !(0.5..=1.0).contains(&cfg.gain_exponent) {
        return Err("gain_exponent must be in [0.5, 1.0] (Robbins-Monro)".into());
    }
    if cfg.window == 0 {
        return Err("window must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if !cfg.ridge.is_finite() || cfg.ridge <= 0.0 {
        return Err("ridge must be finite and positive".into());
    }
    Ok(())
}

/// Fit the confirmatory multidimensional 2PL by Metropolis-Hastings Robbins-Monro (Cai, 2010b).
///
/// `y` is a row-major `n_persons * n_items` binary (`0/1`) response array; `observed` an optional
/// row-major bool mask (missing dropped MAR). `loading_pattern` is a row-major `n_items * n_dims`
/// 0/1 confirmatory pattern; every dimension needs a pure single-dimension anchor item.
#[allow(clippy::too_many_arguments)]
pub fn fit_mhrm(
    y: &[usize],
    observed: Option<&[bool]>,
    loading_pattern: &[u8],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    cfg: &MhrmConfig,
) -> Result<MhrmResult, String> {
    validate(
        y,
        observed,
        loading_pattern,
        n_persons,
        n_items,
        n_dims,
        cfg,
    )?;

    let seen = |p: usize, i: usize| observed.map_or(true, |o| o[p * n_items + i]);
    let dims_of: Vec<Vec<usize>> = (0..n_items)
        .map(|i| {
            (0..n_dims)
                .filter(|&d| loading_pattern[i * n_dims + d] == 1)
                .collect()
        })
        .collect();

    let n_cat = cfg.model.n_cat();
    let n_free_cat = cfg.model.n_free_cat();
    // Output routing is keyed on the model FAMILY, not on `n_free_cat`: a GPCM with `n_cat == 2` also
    // has `n_free_cat == 1`, and its single step must still land in `step`/`se_step` (not the 2PL
    // `intercept`/`se_intercept`) to honor the family-based MhrmResult contract.
    let is_2pl = matches!(cfg.model, MhrmModel::TwoPl);
    // Init: loadings 1.0 on loaded dims; category parameters from the item's observed category
    // frequencies — 2PL: a single log-odds intercept; GPCM: plain per-category log-odds steps
    // `step_k = ln(freq_k / freq_0)` (Laplace-smoothed marginal log-odds, matching gpcm.rs init).
    let mut params: Vec<Vec<f64>> = Vec::with_capacity(n_items);
    for i in 0..n_items {
        let li = dims_of[i].len();
        let mut pv = vec![1.0f64; li];
        match cfg.model {
            MhrmModel::TwoPl => {
                let mut n_obs = 0usize;
                let mut n_pos = 0usize;
                for p in 0..n_persons {
                    if seen(p, i) {
                        n_obs += 1;
                        if y[p * n_items + i] == 1 {
                            n_pos += 1;
                        }
                    }
                }
                let pbar = ((n_pos as f64) + 0.5) / ((n_obs as f64) + 1.0); // Laplace-smoothed
                pv.push((pbar / (1.0 - pbar)).ln());
            }
            MhrmModel::Gpcm { .. } => {
                let mut freq = vec![0.5f64; n_cat]; // Laplace prior
                for p in 0..n_persons {
                    if seen(p, i) {
                        freq[y[p * n_items + i]] += 1.0;
                    }
                }
                for k in 1..n_cat {
                    pv.push((freq[k] / freq[0]).ln());
                }
            }
        }
        params.push(pv);
    }

    // Per-item RM information Gamma_i (flat p_i x p_i, p_i = |S_i| + n_free_cat), init to identity
    // (PD); Louis accumulator. `n_free_cat` = 1 for the 2PL (a single intercept) or `n_cat-1` for GPCM.
    let mut gamma: Vec<Vec<f64>> = dims_of
        .iter()
        .map(|d| {
            let p = d.len() + n_free_cat;
            let mut m = vec![0.0f64; p * p];
            for a in 0..p {
                m[a * p + a] = 1.0;
            }
            m
        })
        .collect();
    let mut gamma_obs: Vec<Vec<f64>> = dims_of
        .iter()
        .map(|d| vec![0.0f64; (d.len() + n_free_cat) * (d.len() + n_free_cat)])
        .collect();

    let mut theta = vec![0.0f64; n_persons * n_dims]; // persistent MH chain state
    let mut theta_sum = vec![0.0f64; n_persons * n_dims]; // convergence-stage accumulation
    let mut theta_count = 0usize;

    // Latent correlation Phi (Cai 2010b): free off-diagonals + its inverse (precomputed per cycle for
    // the MH acceptance prior). When estimate_corr is false, phi_inv stays None so the acceptance
    // prior is the bit-identical orthogonal -0.5(||theta*||^2 - ||theta||^2).
    let n_off = n_dims * (n_dims - 1) / 2;
    let mut offdiag = vec![0.0f64; n_off];
    let mut phi_inv: Option<Vec<f64>> = if cfg.estimate_corr {
        let mut m = vec![0.0f64; n_dims * n_dims];
        for a in 0..n_dims {
            m[a * n_dims + a] = 1.0;
        }
        Some(m)
    } else {
        None
    };

    let mut rng = Lcg(cfg.seed | 1);
    let mut c = cfg.proposal_sd;
    let mut converged = false;
    let mut n_cycles = 0usize;
    let mut final_change = 0.0f64;
    let mut acceptance_rate = 0.0f64;
    let mut recent: Vec<f64> = Vec::with_capacity(cfg.window);

    let mut thstar = vec![0.0f64; n_dims];
    for k in 1..=cfg.max_cycles {
        n_cycles = k;

        // ---- I-step: persistent random-walk Metropolis imputation ----
        let mut accepts = 0usize;
        let mut trials = 0usize;
        for p in 0..n_persons {
            for _ in 0..cfg.mh_steps {
                // propose theta* = theta + c N(0, I) (identical RNG stream in both prior branches)
                for d in 0..n_dims {
                    thstar[d] = theta[p * n_dims + d] + c * rng.normal();
                }
                // prior quadratic-form difference: correlated theta*'Phi^-1 theta* - theta'Phi^-1 theta,
                // or the bit-identical orthogonal ||theta*||^2 - ||theta||^2 when phi_inv is None.
                let quad = match phi_inv.as_ref() {
                    Some(pinv) => {
                        let mut q = 0.0;
                        for a in 0..n_dims {
                            for b in 0..n_dims {
                                q += pinv[a * n_dims + b]
                                    * (thstar[a] * thstar[b]
                                        - theta[p * n_dims + a] * theta[p * n_dims + b]);
                            }
                        }
                        q
                    }
                    None => {
                        let mut q = 0.0;
                        for d in 0..n_dims {
                            let cur = theta[p * n_dims + d];
                            q += thstar[d] * thstar[d] - cur * cur;
                        }
                        q
                    }
                };
                let mut lr = -0.5 * quad;
                for i in 0..n_items {
                    if !seen(p, i) {
                        continue;
                    }
                    let yy = y[p * n_items + i];
                    lr += item_logp(cfg.model, &params[i], &dims_of[i], &thstar, yy)
                        - item_logp(
                            cfg.model,
                            &params[i],
                            &dims_of[i],
                            &theta[p * n_dims..(p + 1) * n_dims],
                            yy,
                        );
                }
                trials += 1;
                if lr >= 0.0 || rng.next_f64() < lr.exp() {
                    for d in 0..n_dims {
                        theta[p * n_dims + d] = thstar[d];
                    }
                    accepts += 1;
                }
            }
        }
        acceptance_rate = accepts as f64 / trials.max(1) as f64;
        if cfg.adapt_proposal && k <= cfg.burn_in {
            // multiplicative proposal tuning toward the target acceptance rate
            let adj = 1.0 + 0.5 * (acceptance_rate - cfg.target_accept);
            c = (c * adj.clamp(0.7, 1.4)).clamp(1e-3, 20.0);
        }

        // ---- RM step: per-item stochastic score/information + Newton update ----
        let gain = gain_at(k, cfg.burn_in, cfg.burn_in_gain, cfg.gain_exponent);
        let mut change2 = 0.0f64;
        for i in 0..n_items {
            let li = dims_of[i].len();
            let pi = li + n_free_cat;
            let (s, hmat, hobs) = item_score_info(
                cfg.model,
                &params[i],
                &dims_of[i],
                &theta,
                y,
                observed,
                i,
                n_persons,
                n_items,
                n_dims,
            );
            // RM information recursion Gamma_i += gain (H_k - Gamma_i)
            for idx in 0..pi * pi {
                gamma[i][idx] += gain * (hmat[idx] - gamma[i][idx]);
            }
            // solve (Gamma_i + ridge I) delta = s
            let mut a2: Vec<Vec<f64>> = (0..pi)
                .map(|a| (0..pi).map(|b| gamma[i][a * pi + b]).collect())
                .collect();
            for a in 0..pi {
                a2[a][a] += cfg.ridge;
            }
            let delta = solve_small(a2, s.clone());
            for t in 0..pi {
                let step = gain * delta[t];
                params[i][t] += step;
                change2 += step * step;
            }
            // clamp only the SLOPE slots (0..|S_i|); the intercept/steps are unbounded
            for t in 0..li {
                params[i][t] = params[i][t].clamp(-MHRM_A_BOUND, MHRM_A_BOUND);
            }
            // Louis observed-information accumulation over the convergence stage
            if cfg.estimate_se && k > cfg.burn_in {
                for idx in 0..pi * pi {
                    gamma_obs[i][idx] += gain * (hobs[idx] - gamma_obs[i][idx]);
                }
            }
        }

        // ---- in-loop reflection sign fix (keep the RM average in one mirror mode) ----
        for d in 0..n_dims {
            let mut anchor: Option<usize> = None;
            let mut best = 0.0f64;
            for i in 0..n_items {
                if dims_of[i].len() == 1 && dims_of[i][0] == d {
                    let a = params[i][0].abs();
                    if a > best {
                        best = a;
                        anchor = Some(i);
                    }
                }
            }
            if let Some(ai) = anchor {
                // the pure anchor's slope is params[ai][0] (its sole loaded dim is d)
                if params[ai][0] < 0.0 {
                    for i in 0..n_items {
                        if let Some(t) = dims_of[i].iter().position(|&dd| dd == d) {
                            params[i][t] = -params[i][t];
                            // Keep the RM information accumulators in the SAME mirror mode as the
                            // loadings: `theta_pd -> -theta_pd` negates row t and column t of the
                            // outer products `X_p X_p^T` (the (t, t) diagonal is `theta_pd^2`,
                            // invariant). Without this, a post-burn-in flip would blend +/- oriented
                            // off-diagonals into the Louis SE accumulator (gamma_obs).
                            let pi = dims_of[i].len() + n_free_cat;
                            for a in 0..pi {
                                if a != t {
                                    gamma[i][a * pi + t] = -gamma[i][a * pi + t];
                                    gamma[i][t * pi + a] = -gamma[i][t * pi + a];
                                    gamma_obs[i][a * pi + t] = -gamma_obs[i][a * pi + t];
                                    gamma_obs[i][t * pi + a] = -gamma_obs[i][t * pi + a];
                                }
                            }
                        }
                    }
                    for p in 0..n_persons {
                        theta[p * n_dims + d] = -theta[p * n_dims + d];
                        theta_sum[p * n_dims + d] = -theta_sum[p * n_dims + d];
                    }
                    if cfg.estimate_corr {
                        // theta_d -> -theta_d negates corr(theta_d, theta_k); keep Phi consistent
                        // with the flipped chain BEFORE Phi^{-1} is recomputed below.
                        flip_corr_dim(&mut offdiag, n_dims, d);
                    }
                }
            }
        }

        // ---- correlation Phi Robbins-Monro update (Cai, 2010b) ----
        if cfg.estimate_corr && n_off > 0 {
            // Sample second moment C = (1/N) sum_p theta_p theta_p^T at the imputed traits. RAW /
            // uncentered: E[theta] = 0 is fixed by identification, so this IS the covariance; do NOT
            // standardize to a correlation (that would double-apply the unit-diagonal constraint and
            // bias Phi). Matches twopl::fit_2pl's C exactly.
            let mut cmat = vec![0.0f64; n_dims * n_dims];
            for p in 0..n_persons {
                for a in 0..n_dims {
                    let ta = theta[p * n_dims + a];
                    for b in 0..n_dims {
                        cmat[a * n_dims + b] += ta * theta[p * n_dims + b];
                    }
                }
            }
            for x in cmat.iter_mut() {
                *x /= n_persons as f64;
            }
            let phi = build_corr(&offdiag, n_dims);
            // ponytail: BARE gradient Robbins-Monro step on the free off-diagonals (ascent of the
            // Gaussian-prior objective Q(Phi) = -0.5[log|Phi| + tr(Phi^{-1} C)]), NOT Cai's
            // Newton-preconditioned covariance update. The RM gain still gives a.s. convergence to the
            // same Phi root; only the (un-curvature-adapted) rate differs. Upgrade path: precondition
            // by the Q off-diagonal Hessian if the rate matters. PD is kept by BACKTRACKING the step
            // (halve until the rebuilt Phi is positive-definite), preferred over a full reject to
            // avoid frozen cycles near the PD boundary at high |rho|.
            if let Some(g) = sigma_grad(&phi, &cmat, n_dims) {
                let mut scale = 1.0f64;
                for _ in 0..12 {
                    let cand: Vec<f64> = (0..n_off)
                        .map(|m| offdiag[m] + gain * scale * g[m])
                        .collect();
                    if chol_lower(&build_corr(&cand, n_dims), n_dims).is_some() {
                        offdiag = cand;
                        break;
                    }
                    scale *= 0.5; // never PD after 12 halvings => keep the previous PD offdiag
                }
            }
            // recompute Phi^{-1} for the next cycle's I-step (keep previous if somehow non-PD)
            if let Some((inv, _)) = sym_inv_logdet(&build_corr(&offdiag, n_dims), n_dims) {
                phi_inv = Some(inv);
            }
        }

        // ---- convergence-stage trait accumulation + windowed stopping ----
        if k > cfg.burn_in {
            for idx in 0..n_persons * n_dims {
                theta_sum[idx] += theta[idx];
            }
            theta_count += 1;
        }
        let change = change2.sqrt();
        recent.push(change);
        if recent.len() > cfg.window {
            recent.remove(0);
        }
        // Report the same windowed statistic that defines convergence. Returning only the most
        // recent stochastic step can exceed `tol` even when the window mean legitimately converged.
        final_change = recent.iter().sum::<f64>() / recent.len() as f64;
        if k > cfg.burn_in && recent.len() == cfg.window {
            if final_change < cfg.tol {
                converged = true;
                break;
            }
        }
    }

    // ---- assemble outputs ----
    let mut loading = vec![0.0f64; n_items * n_dims];
    // 2PL: `intercept` (length J); GPCM: `step` (row-major J x (n_cat-1)); the unused one stays empty.
    let mut intercept = if is_2pl {
        vec![0.0f64; n_items]
    } else {
        Vec::new()
    };
    let mut step = if is_2pl {
        Vec::new()
    } else {
        vec![0.0f64; n_items * n_free_cat]
    };
    for i in 0..n_items {
        let li = dims_of[i].len();
        for (t, &d) in dims_of[i].iter().enumerate() {
            loading[i * n_dims + d] = params[i][t];
        }
        if is_2pl {
            intercept[i] = params[i][li];
        } else {
            for j in 0..n_free_cat {
                step[i * n_free_cat + j] = params[i][li + j];
            }
        }
    }
    let mut theta_eap = if theta_count > 0 {
        theta_sum
            .iter()
            .map(|v| v / theta_count as f64)
            .collect::<Vec<f64>>()
    } else {
        theta.clone()
    };

    // final reflection canonicalization (idempotent given the in-loop fix; also aligns theta_eap)
    for d in 0..n_dims {
        let mut anchor: Option<usize> = None;
        let mut best = 0.0f64;
        for i in 0..n_items {
            if dims_of[i].len() == 1 && dims_of[i][0] == d && loading[i * n_dims + d].abs() > best {
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
                    theta_eap[p * n_dims + d] = -theta_eap[p * n_dims + d];
                }
                if cfg.estimate_corr {
                    flip_corr_dim(&mut offdiag, n_dims, d);
                }
            }
        }
    }
    let corr = build_corr(&offdiag, n_dims);

    // Louis SEs: SE = sqrt(diag((Gamma_obs + ridge I)^{-1})) per item block
    let (mut se_loading, mut se_intercept, mut se_step) = (Vec::new(), Vec::new(), Vec::new());
    if cfg.estimate_se {
        se_loading = vec![0.0f64; n_items * n_dims];
        if is_2pl {
            se_intercept = vec![0.0f64; n_items];
        } else {
            se_step = vec![0.0f64; n_items * n_free_cat];
        }
        for i in 0..n_items {
            let li = dims_of[i].len();
            let pi = li + n_free_cat;
            let block = |src: &[f64]| -> Vec<Vec<f64>> {
                let mut m: Vec<Vec<f64>> = (0..pi)
                    .map(|a| (0..pi).map(|b| src[a * pi + b]).collect())
                    .collect();
                for a in 0..pi {
                    m[a][a] += cfg.ridge;
                }
                m
            };
            let diag_inv = |m: &[Vec<f64>]| -> Vec<f64> {
                (0..pi)
                    .map(|t| {
                        let mut e = vec![0.0f64; pi];
                        e[t] = 1.0;
                        solve_small(m.to_vec(), e)[t]
                    })
                    .collect::<Vec<f64>>()
            };
            // Louis observed information; if any variance is non-PD, fall back to the complete-data
            // (Fisher) information block for the whole item (conservative SE).
            let obs = block(&gamma_obs[i]);
            let mut var = diag_inv(&obs);
            if var.iter().any(|v| !v.is_finite() || *v <= 0.0) {
                var = diag_inv(&block(&gamma[i]));
            }
            for t in 0..pi {
                let se = if var[t].is_finite() && var[t] > 0.0 {
                    var[t].sqrt()
                } else {
                    f64::NAN
                };
                if t < li {
                    se_loading[i * n_dims + dims_of[i][t]] = se;
                } else if is_2pl {
                    se_intercept[i] = se;
                } else {
                    se_step[i * n_free_cat + (t - li)] = se;
                }
            }
        }
    }

    let n_free_loadings = loading_pattern.iter().filter(|&&v| v == 1).count();
    let n_corr = if cfg.estimate_corr { n_off } else { 0 };
    Ok(MhrmResult {
        loading,
        intercept,
        step,
        n_cat,
        theta: theta_eap,
        n_dims,
        corr,
        se_loading,
        se_intercept,
        se_step,
        acceptance_rate,
        n_cycles,
        converged,
        termination_reason: if converged {
            "converged"
        } else {
            "max_cycles_reached"
        }
        .into(),
        final_param_change: final_change,
        n_parameters: n_free_loadings + n_items * n_free_cat + n_corr,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// Smoke test: unidimensional 2PL recovery. MH-RM at `D = 1` should recover the loadings and
    /// intercepts within Monte-Carlo tolerance (a fixed-seed anchor, NOT exact equality).
    #[test]
    fn mhrm_recovers_unidimensional_2pl() {
        let (n, n_items) = (1500usize, 12usize);
        let pattern = vec![1u8; n_items]; // D = 1, every item pure
        let mut rng = Lcg(20100507);
        let true_a: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.1 * (i % 5) as f64).collect();
        let true_b: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.15 * i as f64).collect();
        let mut theta = vec![0.0f64; n];
        for v in theta.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let base = true_a[i] * theta[p] + true_b[i];
                let prob = 1.0 / (1.0 + (-base).exp());
                y[p * n_items + i] = if rng.next_f64() < prob { 1 } else { 0 };
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1200,
            burn_in: 150,
            mh_steps: 8,
            seed: 424242,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
        assert_eq!(res.n_dims, 1);
        assert_eq!(res.loading.len(), n_items);
        assert_eq!(res.n_parameters, n_items + n_items);
        // reflection canonical: largest pure anchor positive
        assert!(res.loading.iter().cloned().fold(f64::MIN, f64::max) > 0.0);
        // acceptance in a sane band after tuning
        assert!(
            res.acceptance_rate > 0.1 && res.acceptance_rate < 0.7,
            "acceptance {}",
            res.acceptance_rate
        );
        // recover loadings and intercepts within MC tolerance
        assert!(
            rmse(&res.loading, &true_a) < 0.2,
            "loading RMSE {} loadings {:?}",
            rmse(&res.loading, &true_a),
            res.loading
        );
        assert!(
            rmse(&res.intercept, &true_b) < 0.2,
            "intercept RMSE {}",
            rmse(&res.intercept, &true_b)
        );
        // trait EAP correlates with the truth
        let th: Vec<f64> = (0..n).map(|p| res.theta[p]).collect();
        let mt = th.iter().sum::<f64>() / n as f64;
        let mtt = theta.iter().sum::<f64>() / n as f64;
        let cov: f64 = (0..n).map(|p| (th[p] - mt) * (theta[p] - mtt)).sum();
        let vt: f64 = th.iter().map(|x| (x - mt).powi(2)).sum();
        let vtt: f64 = theta.iter().map(|x| (x - mtt).powi(2)).sum();
        assert!(
            cov / (vt * vtt).sqrt() > 0.8,
            "theta corr {}",
            cov / (vt * vtt).sqrt()
        );
        // Louis SEs finite and positive
        assert!(res.se_loading.iter().all(|s| s.is_finite() && *s > 0.0));
    }

    fn corr(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let ma = a.iter().sum::<f64>() / n;
        let mb = b.iter().sum::<f64>() / n;
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for i in 0..a.len() {
            let (da, db) = (a[i] - ma, b[i] - mb);
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
        sab / (saa * sbb).sqrt()
    }

    fn item_loglik(
        params: &[f64],
        dims: &[usize],
        theta: &[f64],
        y: &[usize],
        np: usize,
        nd: usize,
    ) -> f64 {
        let li = dims.len();
        let mut ll = 0.0;
        for p in 0..np {
            let mut base = params[li];
            for (t, &d) in dims.iter().enumerate() {
                base += params[t] * theta[p * nd + d];
            }
            let pp = 1.0 / (1.0 + (-base).exp());
            ll += if y[p] == 1 { pp.ln() } else { (1.0 - pp).ln() };
        }
        ll
    }

    /// Deterministic anchor: the per-item score and information returned by `item_score_info` are
    /// pinned against finite differences of the complete-data logistic log-likelihood, on ONE D=2
    /// CROSS-loader item with ASYMMETRIC params (a NEGATIVE loading) at fixed asymmetric traits. A
    /// sign flip in the residual, a transposed information layout, or a dropped dims-map entry all
    /// fail here — none of which a centered/symmetric value-recovery test would catch.
    #[test]
    fn mhrm_score_and_info_match_finite_difference() {
        let nd = 2usize;
        let dims = vec![0usize, 1usize];
        let params = vec![0.8f64, -0.5, 0.3]; // [a0, a1, b] — a1 negative
        let theta = vec![0.5, -1.0, -0.7, 0.4, 1.2, 0.9]; // 3 persons x 2 dims (asymmetric)
        let y = vec![1usize, 0, 1];
        let np = 3usize;
        let pi = 3usize;
        let (s, h, hobs) =
            item_score_info(MhrmModel::TwoPl, &params, &dims, &theta, &y, None, 0, np, 1, nd);
        // score[t] = d loglik / d params[t]
        let eps = 1e-6;
        for t in 0..pi {
            let mut pp = params.clone();
            pp[t] += eps;
            let mut pm = params.clone();
            pm[t] -= eps;
            let fd = (item_loglik(&pp, &dims, &theta, &y, np, nd)
                - item_loglik(&pm, &dims, &theta, &y, np, nd))
                / (2.0 * eps);
            assert!((s[t] - fd).abs() < 1e-4, "score[{t}] {} vs FD {}", s[t], fd);
        }
        // info[a][b] = -d^2 loglik / d params[a] d params[b] = sum_p w_p x_a x_b (symmetric, PD)
        let hh = 1e-3;
        for a in 0..pi {
            for b in 0..pi {
                let mut fpp = params.clone();
                fpp[a] += hh;
                fpp[b] += hh;
                let mut fpm = params.clone();
                fpm[a] += hh;
                fpm[b] -= hh;
                let mut fmp = params.clone();
                fmp[a] -= hh;
                fmp[b] += hh;
                let mut fmm = params.clone();
                fmm[a] -= hh;
                fmm[b] -= hh;
                let d2 = (item_loglik(&fpp, &dims, &theta, &y, np, nd)
                    - item_loglik(&fpm, &dims, &theta, &y, np, nd)
                    - item_loglik(&fmp, &dims, &theta, &y, np, nd)
                    + item_loglik(&fmm, &dims, &theta, &y, np, nd))
                    / (4.0 * hh * hh);
                assert!(
                    (h[a * pi + b] - (-d2)).abs() < 1e-2,
                    "info[{a}][{b}] {} vs -FDhess {}",
                    h[a * pi + b],
                    -d2
                );
                assert!(
                    (h[a * pi + b] - h[b * pi + a]).abs() < 1e-12,
                    "info symmetric"
                );
            }
        }
        // non-trivial layout: the cross term is genuinely nonzero (asymmetric traits)
        assert!(h[1].abs() > 0.05, "off-diag info nonzero: {}", h[1]);
        // Louis missing-information term: hobs = sum_p (w_p - r_p^2) X X' = H - sum_p r_p^2 X X'.
        // Pin the SIGN of the r^2 subtraction (the mutant `w + r^2` inverts it) by an INDEPENDENT
        // re-sum of the per-person score outer product r_p^2 X_p X_p'.
        let mut r2_outer = vec![0.0f64; pi * pi];
        for p in 0..np {
            let mut base = params[dims.len()];
            for (t, &d) in dims.iter().enumerate() {
                base += params[t] * theta[p * nd + d];
            }
            let pp = 1.0 / (1.0 + (-base).exp());
            let r2 = (y[p] as f64 - pp).powi(2);
            let x = [theta[p * nd], theta[p * nd + 1], 1.0];
            for a in 0..pi {
                for b in 0..pi {
                    r2_outer[a * pi + b] += r2 * x[a] * x[b];
                }
            }
        }
        for idx in 0..pi * pi {
            assert!(
                (hobs[idx] - (h[idx] - r2_outer[idx])).abs() < 1e-9,
                "louis missing-info sign: hobs[{idx}] {} vs H-r2 {}",
                hobs[idx],
                h[idx] - r2_outer[idx]
            );
        }
    }

    /// White-box anchor on the Robbins-Monro gain schedule: constant `burn_in_gain` through burn-in,
    /// then `1/(k - burn_in)^alpha` (an off-by-one at the boundary is a classic bug the recovery
    /// tests would not localize).
    #[test]
    fn mhrm_gain_schedule() {
        let (b, g0) = (10usize, 0.8f64);
        assert_eq!(gain_at(1, b, g0, 1.0), g0);
        assert_eq!(gain_at(b, b, g0, 1.0), g0); // last burn-in cycle is still constant gain
        assert_eq!(gain_at(b + 1, b, g0, 1.0), 1.0); // first convergence-stage cycle: 1/1
        assert_eq!(gain_at(b + 4, b, g0, 1.0), 0.25); // 1/4
        assert!((gain_at(b + 4, b, g0, 0.5) - 0.5).abs() < 1e-12); // 1/4^0.5 = 0.5
    }

    /// Reduction anchor: at `D = 1`, MH-RM agrees with the established deterministic unidimensional
    /// MMLE (`mmle::fit_mmle_2pl`) within Monte-Carlo tolerance (NOT bit-exact — MH-RM is stochastic).
    #[test]
    fn mhrm_reduces_to_mmle_2pl_at_d1() {
        use crate::mmle::{fit_mmle_2pl, MmleConfig};
        let (n, n_items) = (1200usize, 10usize);
        let pattern = vec![1u8; n_items];
        let mut rng = Lcg(77);
        let a_t: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * (i % 4) as f64).collect();
        let b_t: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.13 * i as f64).collect();
        let mut th = vec![0.0f64; n];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let pr = 1.0 / (1.0 + (-(a_t[i] * th[p] + b_t[i])).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1200,
            burn_in: 150,
            mh_steps: 8,
            seed: 9,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
        let yf: Vec<f64> = y.iter().map(|&v| v as f64).collect();
        let obs = vec![true; n * n_items];
        let m = fit_mmle_2pl(&yf, &obs, n, n_items, &MmleConfig::default());
        assert!(
            rmse(&res.loading, &m.a) < 0.12,
            "MH-RM vs MMLE loading RMSE {}",
            rmse(&res.loading, &m.a)
        );
        assert!(
            rmse(&res.intercept, &m.b) < 0.12,
            "MH-RM vs MMLE intercept RMSE {}",
            rmse(&res.intercept, &m.b)
        );
    }

    /// Headline capability: `D = 6` confirmatory 2PL. The `q^D` Gauss-Hermite grid (`21^6 ~ 8.6e7`)
    /// and even the QMC E-step are infeasible at this dimensionality; MH-RM's stochastic imputation
    /// is `D`-agnostic. Simple structure (3 pure anchors per dimension) plus two cross-loaders, one
    /// genuinely NEGATIVE — recovered with the correct sign.
    #[test]
    fn mhrm_recovers_high_dim_d6() {
        let (n_dims, n) = (6usize, 2500usize);
        let n_items = 20usize;
        let mut pattern = vec![0u8; n_items * n_dims];
        for i in 0..18 {
            pattern[i * n_dims + i / 3] = 1; // items 0..17: 3 pure anchors per dimension
        }
        pattern[18 * n_dims] = 1;
        pattern[18 * n_dims + 3] = 1; // item18 cross-loads dims 0 and 3
        pattern[19 * n_dims + 1] = 1;
        pattern[19 * n_dims + 4] = 1; // item19 cross-loads dims 1 and 4
        let mut a_t = vec![0.0f64; n_items * n_dims];
        for i in 0..18 {
            a_t[i * n_dims + i / 3] = 0.9 + 0.1 * (i % 3) as f64;
        }
        a_t[18 * n_dims] = 1.0;
        a_t[18 * n_dims + 3] = -0.7; // NEGATIVE cross-loader
        a_t[19 * n_dims + 1] = 0.8;
        a_t[19 * n_dims + 4] = 0.6;
        let b_t: Vec<f64> = (0..n_items).map(|i| -0.5 + 0.1 * (i % 7) as f64).collect();
        let mut rng = Lcg(60606);
        let mut th = vec![0.0f64; n * n_dims];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = b_t[i];
                for d in 0..n_dims {
                    base += a_t[i * n_dims + d] * th[p * n_dims + d];
                }
                let pr = 1.0 / (1.0 + (-base).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1000,
            burn_in: 200,
            mh_steps: 6,
            seed: 13,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.n_dims, 6);
        for i in 0..n_items {
            for d in 0..n_dims {
                if pattern[i * n_dims + d] == 0 {
                    assert_eq!(res.loading[i * n_dims + d], 0.0);
                }
            }
        }
        let (mut se2, mut cnt) = (0.0, 0usize);
        for idx in 0..n_items * n_dims {
            if pattern[idx] == 1 {
                se2 += (res.loading[idx] - a_t[idx]).powi(2);
                cnt += 1;
            }
        }
        let load_rmse = (se2 / cnt as f64).sqrt();
        assert!(load_rmse < 0.22, "D=6 on-pattern loading RMSE {load_rmse}");
        assert!(
            res.loading[18 * n_dims + 3] < -0.3,
            "negative cross-loader {}",
            res.loading[18 * n_dims + 3]
        );
        for d in 0..n_dims {
            let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + d]).collect();
            let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims + d]).collect();
            assert!(
                corr(&est, &tru) > 0.5,
                "dim {d} theta corr {}",
                corr(&est, &tru)
            );
        }
    }

    /// The reflection canonicalization FIRES and is WITNESSED. dim0 has a WEAK reverse-keyed SOLE
    /// pure anchor (item0, true `-0.7`) and a STRONG positively-keyed cross-loader (item1, dim0
    /// `+1.7`) that dominates the axis orientation, so raw MH-RM lands the anchor NEGATIVE and
    /// canonicalization must flip dim0: the anchor ends positive, the co-loader negative, and theta_0
    /// correlates NEGATIVELY with the truth. Disabling the flip (in-loop + final) fails all three.
    #[test]
    fn mhrm_reflection_fires_on_negative_anchor() {
        let (n_dims, n) = (2usize, 5000usize);
        let n_items = 4usize;
        // item0 pure d0 (sole d0 anchor), item1 cross d0/d1, item2/3 pure d1
        let pattern = vec![1u8, 0, 1, 1, 0, 1, 0, 1];
        let mut a_t = vec![0.0f64; n_items * n_dims];
        a_t[0] = -0.7; // weak reverse-keyed pure d0 anchor
        a_t[1 * n_dims] = 1.7; // strong positive cross-loader on d0 (sets the axis)
        a_t[1 * n_dims + 1] = 0.6;
        a_t[2 * n_dims + 1] = 1.2;
        a_t[3 * n_dims + 1] = 1.0;
        let b_t = vec![0.2f64, -0.1, 0.3, -0.2];
        let mut rng = Lcg(1357);
        let mut th = vec![0.0f64; n * n_dims];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = b_t[i];
                for d in 0..n_dims {
                    base += a_t[i * n_dims + d] * th[p * n_dims + d];
                }
                let pr = 1.0 / (1.0 + (-base).exp());
                y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1000,
            burn_in: 200,
            mh_steps: 8,
            seed: 24,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert!(
            res.loading[0] > 0.3,
            "reflected anchor positive: {}",
            res.loading[0]
        );
        assert!(
            res.loading[1 * n_dims] < -0.5,
            "co-loader flipped negative: {}",
            res.loading[1 * n_dims]
        );
        let th0: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
        let tt0: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
        let th1: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + 1]).collect();
        let tt1: Vec<f64> = (0..n).map(|p| th[p * n_dims + 1]).collect();
        assert!(
            corr(&th0, &tt0) < -0.4,
            "flipped-dim theta corr negative: {}",
            corr(&th0, &tt0)
        );
        assert!(
            corr(&th1, &tt1) > 0.4,
            "unflipped-dim theta corr positive: {}",
            corr(&th1, &tt1)
        );
    }

    /// Correlated-Sigma MH-RM (Cai, 2010b): with `estimate_corr` the free latent correlation matrix
    /// `Phi` is recovered from `theta ~ MVN(0, Phi)`. Covers a POSITIVE, a near-PD-boundary (D=3,
    /// rho=0.5), and a NEGATIVE correlation (sign correctness); confirms `Phi` stays a valid PD
    /// correlation matrix (unit diagonal) and `n_parameters` counts the `D(D-1)/2` correlations.
    #[test]
    fn mhrm_correlated_recovers_known_phi() {
        for &(n_dims, rho, n) in &[
            (2usize, 0.4f64, 3000usize),
            (3usize, 0.5f64, 3500usize),
            (2usize, -0.5f64, 3000usize),
        ] {
            // exchangeable Phi
            let mut phi = vec![rho; n_dims * n_dims];
            for a in 0..n_dims {
                phi[a * n_dims + a] = 1.0;
            }
            let l = chol_lower(&phi, n_dims).expect("Phi PD");
            let per = 4usize;
            let n_items = per * n_dims;
            let mut pattern = vec![0u8; n_items * n_dims];
            let mut a_t = vec![0.0f64; n_items * n_dims];
            for d in 0..n_dims {
                for a in 0..per {
                    let i = d * per + a;
                    pattern[i * n_dims + d] = 1;
                    a_t[i * n_dims + d] = 1.0 + 0.1 * a as f64;
                }
            }
            let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * (i % 5) as f64).collect();
            let mut rng = Lcg(0x00C0FFEE ^ ((n_dims as u64) << 8) ^ ((rho < 0.0) as u64));
            // theta_p = L z_p ~ MVN(0, Phi)
            let mut th = vec![0.0f64; n * n_dims];
            for p in 0..n {
                let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
                for a in 0..n_dims {
                    let mut v = 0.0;
                    for b in 0..=a {
                        v += l[a * n_dims + b] * z[b];
                    }
                    th[p * n_dims + a] = v;
                }
            }
            let mut y = vec![0usize; n * n_items];
            for p in 0..n {
                for i in 0..n_items {
                    let mut base = b_t[i];
                    for d in 0..n_dims {
                        base += a_t[i * n_dims + d] * th[p * n_dims + d];
                    }
                    let pr = 1.0 / (1.0 + (-base).exp());
                    y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
                }
            }
            let cfg = MhrmConfig {
                max_cycles: 1600,
                burn_in: 350,
                mh_steps: 8,
                estimate_corr: true,
                seed: 42,
                ..MhrmConfig::default()
            };
            let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
            assert_eq!(res.corr.len(), n_dims * n_dims);
            // valid correlation matrix: unit diagonal, symmetric, PD
            for a in 0..n_dims {
                assert!(
                    (res.corr[a * n_dims + a] - 1.0).abs() < 1e-9,
                    "unit diagonal"
                );
                for b in 0..n_dims {
                    assert!((res.corr[a * n_dims + b] - res.corr[b * n_dims + a]).abs() < 1e-12);
                }
            }
            assert!(chol_lower(&res.corr, n_dims).is_some(), "recovered Phi PD");
            // recover the off-diagonals (sign + magnitude) within MC tolerance
            for a in 0..n_dims {
                for b in a + 1..n_dims {
                    let est = res.corr[a * n_dims + b];
                    assert!(
                        (est - rho).abs() < 0.12,
                        "D={n_dims} rho={rho} corr[{a}][{b}]={est}"
                    );
                }
            }
            assert_eq!(
                res.n_parameters,
                n_items + n_items + n_dims * (n_dims - 1) / 2
            );
        }
    }

    /// Validation guards constructed non-vacuously (each input trips the INTENDED guard, not an
    /// earlier one).
    #[test]
    fn mhrm_validates_and_structural_invariants() {
        let (n, n_items, n_dims) = (60usize, 4usize, 2usize);
        let pattern = vec![1u8, 0, 1, 0, 0, 1, 0, 1]; // pure anchors on both dims
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                y[p * n_items + i] = (p + i) % 2; // non-degenerate mixed responses
            }
        }
        let short = MhrmConfig {
            max_cycles: 30,
            burn_in: 5,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &short).unwrap();
        assert_eq!(res.n_parameters, 4 + 4); // 4 loadings + 4 intercepts (no correlations)
        assert_eq!(res.se_loading.len(), n_items * n_dims);
        // estimate_corr=false -> Phi is EXACTLY the identity (orthogonal factors)
        assert_eq!(res.corr, vec![1.0, 0.0, 0.0, 1.0]);
        // no pure anchor on any dimension (every item loads both dims)
        let all_both = vec![1u8; n_items * n_dims];
        assert!(fit_mhrm(&y, None, &all_both, n, n_items, n_dims, &short).is_err());
        // non-binary response where observed
        let mut ybad = y.clone();
        ybad[0] = 2;
        assert!(fit_mhrm(&ybad, None, &pattern, n, n_items, n_dims, &short).is_err());
        // burn_in >= max_cycles
        let bad = MhrmConfig {
            max_cycles: 10,
            burn_in: 10,
            ..MhrmConfig::default()
        };
        assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &bad).is_err());
        // gain_exponent out of (0.5, 1] Robbins-Monro band
        let badgain = MhrmConfig {
            gain_exponent: 0.3,
            ..short
        };
        assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &badgain).is_err());
        // n_dims exceeds MHRM_MAX_DIMS (=64) — the n_dims guard is checked before pattern length
        let big_pat = vec![1u8; n_items * 65];
        assert!(fit_mhrm(&y, None, &big_pat, n, n_items, 65, &short).is_err());
        // y length mismatch (cells != y.len())
        let y_short = vec![0usize; n * n_items - 1];
        assert!(fit_mhrm(&y_short, None, &pattern, n, n_items, n_dims, &short).is_err());
        // loading_pattern entry other than 0/1 (correct length, so the >1 guard is the sole trip)
        let mut pat_bad = pattern.clone();
        pat_bad[0] = 2;
        assert!(fit_mhrm(&y, None, &pat_bad, n, n_items, n_dims, &short).is_err());
    }

    // ================================ GPCM MH-RM (Muraki, 1992) ================================

    /// GPCM category probabilities at a scalar `base = sum_d a_d theta_d`: `psi_k = k*base + step_k`
    /// (`step_0 = 0`), `P_k = softmax_k(psi)`.
    fn gpcm_probs(base: f64, steps: &[f64], n_cat: usize) -> Vec<f64> {
        let mut psi = vec![0.0f64; n_cat];
        let mut m = f64::NEG_INFINITY;
        for k in 0..n_cat {
            psi[k] = (k as f64) * base + if k == 0 { 0.0 } else { steps[k - 1] };
            if psi[k] > m {
                m = psi[k];
            }
        }
        let mut z = 0.0;
        for p in psi.iter_mut() {
            *p = (*p - m).exp();
            z += *p;
        }
        for p in psi.iter_mut() {
            *p /= z;
        }
        psi
    }

    /// Inverse-CDF category draw from a probability vector and a uniform `u`.
    fn gpcm_sample(probs: &[f64], u: f64) -> usize {
        let mut acc = 0.0;
        for (k, &p) in probs.iter().enumerate() {
            acc += p;
            if u < acc {
                return k;
            }
        }
        probs.len() - 1
    }

    /// Complete-data GPCM item log-likelihood at fixed traits (the FD target for the score/Hessian
    /// anchor). `params = [a_d for d in dims, step_1..step_{K-1}]`.
    fn gpcm_item_loglik(
        params: &[f64],
        dims: &[usize],
        theta: &[f64],
        y: &[usize],
        np: usize,
        nd: usize,
        n_cat: usize,
    ) -> f64 {
        let li = dims.len();
        let mut ll = 0.0;
        for p in 0..np {
            let mut base = 0.0;
            for (t, &d) in dims.iter().enumerate() {
                base += params[t] * theta[p * nd + d];
            }
            let mut m = f64::NEG_INFINITY;
            let mut psi = vec![0.0f64; n_cat];
            for k in 0..n_cat {
                psi[k] = (k as f64) * base + if k == 0 { 0.0 } else { params[li + k - 1] };
                if psi[k] > m {
                    m = psi[k];
                }
            }
            let mut z = 0.0;
            for k in 0..n_cat {
                z += (psi[k] - m).exp();
            }
            ll += psi[y[p]] - (m + z.ln());
        }
        ll
    }

    /// Deterministic anchor for the GPCM (Muraki, 1992) per-item score and the CLOSED-FORM multinomial
    /// information, on ONE `D = 2` CROSS-loader item with an ASYMMETRIC NEGATIVE loading and
    /// NON-MONOTONE (unordered) steps at fixed asymmetric traits, `K = 3`. The score is pinned against
    /// finite differences of the complete-data GPCM log-likelihood, and the information block against
    /// the NEGATIVE FD Hessian — which equals the exact multinomial Hessian since it is
    /// data-independent given `theta` (the mutant that uses the BHHH score cross-product as the
    /// information fails here, and would make the Louis SE degenerate). A sign flip in the residual, a
    /// transposed/dropped design-matrix slot, or an over-collapsed step block all fail here — none of
    /// which a centered/symmetric value-recovery test would localize. The Louis block is pinned to
    /// `H - sum_p s_p s_p'` by an INDEPENDENT per-person score outer-product re-sum (the mutant
    /// `H + sum s s'` inverts the sign of the missing-information subtraction).
    #[test]
    fn gpcm_mhrm_score_and_info_match_finite_difference() {
        let nd = 2usize;
        let n_cat = 3usize;
        let dims = vec![0usize, 1usize];
        // [a0, a1, step_1, step_2]; a1 NEGATIVE, steps non-monotone (0.9 then -0.4 -> not increasing)
        let params = vec![0.9f64, -0.6, 0.9, -0.4];
        let pi = dims.len() + (n_cat - 1); // 4
        // 4 persons, asymmetric traits, responses spanning all 3 categories
        let theta = vec![0.5, -1.0, -0.7, 0.4, 1.2, 0.9, -0.3, -1.1];
        let y = vec![2usize, 0, 1, 2];
        let np = 4usize;
        let (s, h, hobs) = item_score_info(
            MhrmModel::Gpcm { n_cat },
            &params,
            &dims,
            &theta,
            &y,
            None,
            0,
            np,
            1,
            nd,
        );
        assert_eq!(s.len(), pi);
        // score[t] = d loglik / d params[t]
        let eps = 1e-6;
        for t in 0..pi {
            let mut pp = params.clone();
            pp[t] += eps;
            let mut pm = params.clone();
            pm[t] -= eps;
            let fd = (gpcm_item_loglik(&pp, &dims, &theta, &y, np, nd, n_cat)
                - gpcm_item_loglik(&pm, &dims, &theta, &y, np, nd, n_cat))
                / (2.0 * eps);
            assert!((s[t] - fd).abs() < 1e-4, "gpcm score[{t}] {} vs FD {}", s[t], fd);
        }
        // info[a][b] = -d^2 loglik / d params[a] d params[b] (exact multinomial Hessian; symmetric, PD)
        let hh = 1e-3;
        for a in 0..pi {
            for b in 0..pi {
                let mut fpp = params.clone();
                fpp[a] += hh;
                fpp[b] += hh;
                let mut fpm = params.clone();
                fpm[a] += hh;
                fpm[b] -= hh;
                let mut fmp = params.clone();
                fmp[a] -= hh;
                fmp[b] += hh;
                let mut fmm = params.clone();
                fmm[a] -= hh;
                fmm[b] -= hh;
                let d2 = (gpcm_item_loglik(&fpp, &dims, &theta, &y, np, nd, n_cat)
                    - gpcm_item_loglik(&fpm, &dims, &theta, &y, np, nd, n_cat)
                    - gpcm_item_loglik(&fmp, &dims, &theta, &y, np, nd, n_cat)
                    + gpcm_item_loglik(&fmm, &dims, &theta, &y, np, nd, n_cat))
                    / (4.0 * hh * hh);
                assert!(
                    (h[a * pi + b] - (-d2)).abs() < 1e-2,
                    "gpcm info[{a}][{b}] {} vs -FDhess {}",
                    h[a * pi + b],
                    -d2
                );
                assert!((h[a * pi + b] - h[b * pi + a]).abs() < 1e-12, "info symmetric");
            }
        }
        // non-trivial layout: the a0-a1 cross term AND an a0-step1 cross term are genuinely nonzero
        assert!(h[1].abs() > 0.05, "a0-a1 cross-info nonzero: {}", h[1]);
        assert!(h[2].abs() > 0.02, "a0-step1 cross-info nonzero: {}", h[2]);
        // Louis: hobs = H - sum_p s_p s_p'. Re-sum the per-person score outer product INDEPENDENTLY
        // (design J[k][t<li] = k*theta_d, J[k][li+k-1] = 1; resid_k = [k==y] - P_k).
        let mut ss = vec![0.0f64; pi * pi];
        for p in 0..np {
            let mut base = 0.0;
            for (t, &d) in dims.iter().enumerate() {
                base += params[t] * theta[p * nd + d];
            }
            let pr = gpcm_probs(base, &params[dims.len()..], n_cat);
            let mut sp = vec![0.0f64; pi];
            for k in 0..n_cat {
                let resid = (if k == y[p] { 1.0 } else { 0.0 }) - pr[k];
                for (t, &d) in dims.iter().enumerate() {
                    sp[t] += (k as f64) * theta[p * nd + d] * resid;
                }
                if k >= 1 {
                    sp[dims.len() + k - 1] += resid;
                }
            }
            for a in 0..pi {
                for b in 0..pi {
                    ss[a * pi + b] += sp[a] * sp[b];
                }
            }
        }
        for idx in 0..pi * pi {
            assert!(
                (hobs[idx] - (h[idx] - ss[idx])).abs() < 1e-9,
                "gpcm louis missing-info sign: hobs[{idx}] {} vs H-ss {}",
                hobs[idx],
                h[idx] - ss[idx]
            );
        }
    }

    /// Reduction anchor: at `D = 1`, GPCM MH-RM agrees with the deterministic unidimensional GPCM MMLE
    /// (`poly::fit_poly_unidim(PolyModel::Gpcm)`, Bock-Aitkin quadrature) within Monte-Carlo tolerance.
    /// NOT bit-exact — MH-RM is stochastic and uses an unconstrained slope (vs `fit_poly_unidim`'s
    /// `log_a > 0`), so it is up to reflection (both land positive here on all-positive truth).
    #[test]
    fn gpcm_mhrm_reduces_to_poly_unidim_at_d1() {
        use crate::poly::{fit_poly_unidim, PolyModel};
        let (n, n_items, n_cat) = (1600usize, 8usize, 3usize);
        let pattern = vec![1u8; n_items];
        let a_t: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * (i % 4) as f64).collect();
        // non-monotone (unordered) steps per item
        let step_t: Vec<[f64; 2]> = (0..n_items)
            .map(|i| [0.6 - 0.1 * (i % 3) as f64, -0.5 + 0.12 * (i % 4) as f64])
            .collect();
        let mut rng = Lcg(2718281);
        let mut th = vec![0.0f64; n];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let probs = gpcm_probs(a_t[i] * th[p], &step_t[i], n_cat);
                y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1200,
            burn_in: 180,
            mh_steps: 8,
            model: MhrmModel::Gpcm { n_cat },
            seed: 31,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, 1, &cfg).unwrap();
        assert_eq!(res.n_cat, n_cat);
        assert!(res.intercept.is_empty());
        assert_eq!(res.step.len(), n_items * (n_cat - 1));
        assert_eq!(res.n_parameters, n_items + n_items * (n_cat - 1));
        // slopes land positive after canonicalization
        assert!(res.loading.iter().all(|&a| a > 0.0));
        let det =
            fit_poly_unidim(&y, None, n, n_items, n_cat, PolyModel::Gpcm, 41, 200, 1e-6).unwrap();
        assert!(
            rmse(&res.loading, &det.slope) < 0.15,
            "GPCM MH-RM vs MMLE slope RMSE {}",
            rmse(&res.loading, &det.slope)
        );
        let det_steps: Vec<f64> = det.cat_params.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(det_steps.len(), res.step.len());
        assert!(
            rmse(&res.step, &det_steps) < 0.2,
            "GPCM MH-RM vs MMLE step RMSE {}",
            rmse(&res.step, &det_steps)
        );
    }

    /// Headline GPCM capability: `D = 5` confirmatory GPCM. The `q^D` Gauss-Hermite grid (`21^5`) and
    /// the QMC E-step are infeasible; MH-RM's stochastic imputation is `D`-agnostic. Simple structure
    /// (3 pure anchors per dimension) plus one genuinely NEGATIVE cross-loader, non-monotone steps,
    /// `K = 3` — loadings and steps recovered with the correct sign.
    #[test]
    fn gpcm_mhrm_recovers_high_dim_d5() {
        let (n_dims, n, n_cat) = (5usize, 2200usize, 3usize);
        let n_items = 16usize;
        let mut pattern = vec![0u8; n_items * n_dims];
        for i in 0..15 {
            pattern[i * n_dims + i / 3] = 1; // items 0..14: 3 pure anchors per dimension
        }
        pattern[15 * n_dims] = 1;
        pattern[15 * n_dims + 2] = 1; // item15 cross-loads dims 0 and 2
        let mut a_t = vec![0.0f64; n_items * n_dims];
        for i in 0..15 {
            a_t[i * n_dims + i / 3] = 0.9 + 0.1 * (i % 3) as f64;
        }
        a_t[15 * n_dims] = 1.0;
        a_t[15 * n_dims + 2] = -0.7; // NEGATIVE cross-loader
        let step_t: Vec<[f64; 2]> = (0..n_items)
            .map(|i| [0.7 - 0.12 * (i % 3) as f64, -0.4 + 0.1 * (i % 4) as f64])
            .collect();
        let mut rng = Lcg(50505);
        let mut th = vec![0.0f64; n * n_dims];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = 0.0;
                for d in 0..n_dims {
                    base += a_t[i * n_dims + d] * th[p * n_dims + d];
                }
                let probs = gpcm_probs(base, &step_t[i], n_cat);
                y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1000,
            burn_in: 200,
            mh_steps: 6,
            model: MhrmModel::Gpcm { n_cat },
            seed: 17,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.n_dims, 5);
        assert_eq!(res.n_cat, n_cat);
        for i in 0..n_items {
            for d in 0..n_dims {
                if pattern[i * n_dims + d] == 0 {
                    assert_eq!(res.loading[i * n_dims + d], 0.0);
                }
            }
        }
        let (mut se2, mut cnt) = (0.0, 0usize);
        for idx in 0..n_items * n_dims {
            if pattern[idx] == 1 {
                se2 += (res.loading[idx] - a_t[idx]).powi(2);
                cnt += 1;
            }
        }
        let load_rmse = (se2 / cnt as f64).sqrt();
        assert!(load_rmse < 0.25, "D=5 GPCM on-pattern loading RMSE {load_rmse}");
        assert!(
            res.loading[15 * n_dims + 2] < -0.25,
            "negative cross-loader {}",
            res.loading[15 * n_dims + 2]
        );
        let true_steps: Vec<f64> = (0..n_items).flat_map(|i| step_t[i]).collect();
        assert!(
            rmse(&res.step, &true_steps) < 0.25,
            "GPCM step RMSE {}",
            rmse(&res.step, &true_steps)
        );
        for d in 0..n_dims {
            let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims + d]).collect();
            let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims + d]).collect();
            assert!(corr(&est, &tru) > 0.5, "dim {d} theta corr {}", corr(&est, &tru));
        }
    }

    /// The reflection canonicalization FIRES for GPCM and is WITNESSED, with the UNORDERED steps left
    /// INVARIANT: `base = k*sum a_d theta_d` flips jointly with `(a, theta)`, so canonicalization
    /// touches only the slope column and the trait chain — never the step intercepts. dim0 has a WEAK
    /// reverse-keyed sole pure anchor (item0, true `-0.7`) and a STRONG positive cross-loader (item1,
    /// dim0 `+1.7`) that sets the axis; raw MH-RM lands the anchor NEGATIVE, so canon must flip dim0.
    /// A mutant that ALSO negated the flipped dimension's items' steps would push item0's step_1 to the
    /// wrong sign — the final assertion catches it.
    #[test]
    fn gpcm_mhrm_reflection_fires_on_negative_anchor() {
        let (n_dims, n, n_cat) = (2usize, 5000usize, 3usize);
        let n_items = 4usize;
        // item0 pure d0 (sole d0 anchor), item1 cross d0/d1, item2/3 pure d1
        let pattern = vec![1u8, 0, 1, 1, 0, 1, 0, 1];
        let mut a_t = vec![0.0f64; n_items * n_dims];
        a_t[0] = -0.7; // weak reverse-keyed pure d0 anchor
        a_t[1 * n_dims] = 1.7; // strong positive cross-loader on d0 (sets the axis)
        a_t[1 * n_dims + 1] = 0.6;
        a_t[2 * n_dims + 1] = 1.2;
        a_t[3 * n_dims + 1] = 1.0;
        // item0's steps are positive-then-negative; if reflection wrongly swept them, step_1 -> ~-0.5
        let step_t = [[0.5f64, -0.3], [0.4, -0.5], [0.6, -0.2], [0.3, -0.4]];
        let mut rng = Lcg(97531);
        let mut th = vec![0.0f64; n * n_dims];
        for v in th.iter_mut() {
            *v = rng.normal();
        }
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                let mut base = 0.0;
                for d in 0..n_dims {
                    base += a_t[i * n_dims + d] * th[p * n_dims + d];
                }
                let probs = gpcm_probs(base, &step_t[i], n_cat);
                y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 1000,
            burn_in: 200,
            mh_steps: 8,
            model: MhrmModel::Gpcm { n_cat },
            seed: 24,
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert!(res.loading[0] > 0.3, "reflected anchor positive: {}", res.loading[0]);
        assert!(
            res.loading[1 * n_dims] < -0.5,
            "co-loader flipped negative: {}",
            res.loading[1 * n_dims]
        );
        let th0: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
        let tt0: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
        assert!(
            corr(&th0, &tt0) < -0.4,
            "flipped-dim theta corr negative: {}",
            corr(&th0, &tt0)
        );
        // steps INVARIANT under reflection: item0's step_1 stays near its (un-flipped) truth +0.5, well
        // away from the mutant's -0.5.
        assert!(
            (res.step[0] - step_t[0][0]).abs() < 0.35,
            "GPCM step not swept by reflection: step_1 {} vs truth {}",
            res.step[0],
            step_t[0][0]
        );
    }

    /// GPCM validation guards constructed non-vacuously: the SAME well-formed GPCM dataset fits (and
    /// exposes the `step`/`n_cat` result shape), then each defect trips its INTENDED guard — an
    /// out-of-range response, and a declared category never observed for an item (an unidentified step,
    /// Muraki, 1992).
    #[test]
    fn gpcm_mhrm_validates_and_structure() {
        let (n, n_items, n_dims, n_cat) = (60usize, 4usize, 2usize, 3usize);
        let pattern = vec![1u8, 0, 1, 0, 0, 1, 0, 1]; // pure anchors on both dims
        // y = (p + i) % 3 -> every item sees all 3 categories across persons
        let mut y = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                y[p * n_items + i] = (p + i) % n_cat;
            }
        }
        let cfg = MhrmConfig {
            max_cycles: 30,
            burn_in: 5,
            model: MhrmModel::Gpcm { n_cat },
            ..MhrmConfig::default()
        };
        let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
        assert_eq!(res.n_cat, n_cat);
        assert!(res.intercept.is_empty());
        assert_eq!(res.step.len(), n_items * (n_cat - 1));
        assert_eq!(res.se_step.len(), n_items * (n_cat - 1));
        assert!(res.se_intercept.is_empty());
        assert_eq!(res.n_parameters, n_items + n_items * (n_cat - 1));
        // (a) response out of 0..n_cat where observed
        let mut ybad = y.clone();
        ybad[0] = n_cat; // == 3, out of range
        assert!(fit_mhrm(&ybad, None, &pattern, n, n_items, n_dims, &cfg).is_err());
        // (b) item0's category-1 responses remapped to 0 -> category 1 never observed for item0
        // (still in range), tripping the coverage guard (the binary 2PL does NOT enforce this).
        let mut ycov = y.clone();
        for p in 0..n {
            if ycov[p * n_items] == 1 {
                ycov[p * n_items] = 0;
            }
        }
        assert!(fit_mhrm(&ycov, None, &pattern, n, n_items, n_dims, &cfg).is_err());
        // (c) n_cat above the MHRM_MAX_CAT cap is rejected (the cap guard fires before the
        // O(n_cat) coverage allocation) -- makes the MHRM_MAX_CAT constant live.
        let cfg_big = MhrmConfig {
            model: MhrmModel::Gpcm {
                n_cat: MHRM_MAX_CAT + 1,
            },
            ..cfg
        };
        assert!(fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg_big).is_err());
        // (d) GPCM with n_cat == 2 (also n_free_cat == 1, colliding with the 2PL) routes its single
        // step to `step`/`se_step` -- NOT the 2PL `intercept`/`se_intercept` -- honoring the
        // family-based contract. y2 = (p + i) % 2 sees both categories per item.
        let mut y2 = vec![0usize; n * n_items];
        for p in 0..n {
            for i in 0..n_items {
                y2[p * n_items + i] = (p + i) % 2;
            }
        }
        let cfg2 = MhrmConfig {
            model: MhrmModel::Gpcm { n_cat: 2 },
            ..cfg
        };
        let res2 = fit_mhrm(&y2, None, &pattern, n, n_items, n_dims, &cfg2).unwrap();
        assert_eq!(res2.n_cat, 2);
        assert!(res2.intercept.is_empty(), "GPCM n_cat=2 must not populate 2PL intercept");
        assert!(res2.se_intercept.is_empty());
        assert_eq!(res2.step.len(), n_items); // J * (2 - 1)
        assert_eq!(res2.se_step.len(), n_items);
        assert!(res2.step.iter().all(|s| s.is_finite()));
        assert_eq!(res2.n_parameters, n_items + n_items); // 4 free loadings + 4 single steps
    }

    /// Literature-grade GPCM Monte-Carlo recovery (>=500 reps), normal + right-skew traits. Run with:
    /// `cargo test -p mlsirm-core --release mc_gpcm_mhrm_recovery_500 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mc_gpcm_mhrm_recovery_500() {
        let reps = 500usize;
        let n_cat = 3usize;
        // D=5 is the regime GH/QMC cannot reach for a polytomous item factor model.
        for &(n_dims, n) in &[(2usize, 2000usize), (5usize, 2500usize)] {
            for &skew in &[false, true] {
                let n_items = if n_dims == 2 { 8 } else { 15 };
                let mut pattern = vec![0u8; n_items * n_dims];
                let mut a_t = vec![0.0f64; n_items * n_dims];
                let per = n_items / n_dims;
                for i in 0..per * n_dims {
                    let d = i / per;
                    pattern[i * n_dims + d] = 1;
                    a_t[i * n_dims + d] = 0.9 + 0.1 * (i % 3) as f64;
                }
                // last item cross-loads dims 0 and 1 (dim0 negative)
                let xi = n_items - 1;
                pattern[xi * n_dims] = 1;
                pattern[xi * n_dims + 1] = 1;
                a_t[xi * n_dims] = -0.8;
                a_t[xi * n_dims + 1] = 0.7;
                let step_t: Vec<[f64; 2]> = (0..n_items)
                    .map(|i| [0.7 - 0.12 * (i % 3) as f64, -0.4 + 0.1 * (i % 4) as f64])
                    .collect();
                let n_free: usize = pattern.iter().filter(|&&v| v == 1).count();

                let (mut conv, mut lse2, mut lbias, mut lcnt) = (0usize, 0.0, 0.0, 0usize);
                let (mut sse2, mut sbias) = (0.0, 0.0);
                let mut corr_sum = 0.0;
                for rep in 0..reps {
                    let mut rng = Lcg(
                        0x6CBC_u64.wrapping_mul((rep as u64) + 1).wrapping_add(n_dims as u64),
                    );
                    let mut th = vec![0.0f64; n * n_dims];
                    for v in th.iter_mut() {
                        *v = if skew {
                            // standardized right-skew (Exp(1) - 1): mean 0, var 1
                            -(rng.next_f64().max(1e-12)).ln() - 1.0
                        } else {
                            rng.normal()
                        };
                    }
                    let mut y = vec![0usize; n * n_items];
                    for p in 0..n {
                        for i in 0..n_items {
                            let mut base = 0.0;
                            for d in 0..n_dims {
                                base += a_t[i * n_dims + d] * th[p * n_dims + d];
                            }
                            let probs = gpcm_probs(base, &step_t[i], n_cat);
                            y[p * n_items + i] = gpcm_sample(&probs, rng.next_f64());
                        }
                    }
                    let cfg = MhrmConfig {
                        max_cycles: 900,
                        burn_in: 180,
                        mh_steps: 6,
                        model: MhrmModel::Gpcm { n_cat },
                        seed: 0xC0DE_u64.wrapping_add(rep as u64),
                        estimate_se: false,
                        ..MhrmConfig::default()
                    };
                    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
                    if res.converged {
                        conv += 1;
                    }
                    for idx in 0..n_items * n_dims {
                        if pattern[idx] == 1 {
                            let e = res.loading[idx] - a_t[idx];
                            lse2 += e * e;
                            lbias += e;
                            lcnt += 1;
                        }
                    }
                    for i in 0..n_items {
                        for j in 0..n_cat - 1 {
                            let e = res.step[i * (n_cat - 1) + j] - step_t[i][j];
                            sse2 += e * e;
                            sbias += e;
                        }
                    }
                    let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
                    let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
                    corr_sum += corr(&est, &tru);
                }
                let scnt = (reps * n_items * (n_cat - 1)) as f64;
                let load_rmse = (lse2 / lcnt as f64).sqrt();
                let step_rmse = (sse2 / scnt).sqrt();
                println!(
                    "[gpcm MC D={n_dims} N={n} n_free={n_free} K={n_cat} skew={skew}] reps={reps} conv={:.3} loadRMSE={:.4} loadBias={:.4} stepRMSE={:.4} stepBias={:.4} thetaCorr={:.3}",
                    conv as f64 / reps as f64,
                    load_rmse,
                    lbias / lcnt as f64,
                    step_rmse,
                    sbias / scnt,
                    corr_sum / reps as f64
                );
                assert!(conv as f64 / reps as f64 > 0.9, "GPCM convergence rate");
                if !skew {
                    assert!(load_rmse < 0.22, "GPCM normal loading RMSE {load_rmse}");
                    assert!(step_rmse < 0.25, "GPCM normal step RMSE {step_rmse}");
                }
            }
        }
        println!("=== gpcm done ===");
    }

    /// Literature-grade Monte-Carlo recovery (>=500 reps). Run with:
    /// `cargo test -p mlsirm-core --release mc_mhrm_recovery_500 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn mc_mhrm_recovery_500() {
        let reps = 500usize;
        // (n_dims, N) conditions; D=6 is the regime GH/QMC cannot reach.
        for &(n_dims, n) in &[(2usize, 2000usize), (6usize, 2500usize)] {
            for &skew in &[false, true] {
                let n_items = if n_dims == 2 { 8 } else { 20 };
                // confirmatory pattern: pure anchors per dim + one negative cross-loader
                let mut pattern = vec![0u8; n_items * n_dims];
                let mut a_t = vec![0.0f64; n_items * n_dims];
                let per = n_items / n_dims;
                for i in 0..per * n_dims {
                    let d = i / per;
                    pattern[i * n_dims + d] = 1;
                    a_t[i * n_dims + d] = 0.9 + 0.1 * (i % 3) as f64;
                }
                // last item cross-loads dims 0 and 1 (dim0 negative)
                let xi = n_items - 1;
                pattern[xi * n_dims] = 1;
                pattern[xi * n_dims + 1] = 1;
                a_t[xi * n_dims] = -0.8;
                a_t[xi * n_dims + 1] = 0.7;
                let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.12 * (i % 5) as f64).collect();
                let n_free: usize = pattern.iter().filter(|&&v| v == 1).count();

                let (mut conv, mut se2, mut sbias, mut cnt) = (0usize, 0.0, 0.0, 0usize);
                let mut corr_sum = 0.0;
                for rep in 0..reps {
                    let mut rng = Lcg(0x51ED_u64
                        .wrapping_mul((rep as u64) + 1)
                        .wrapping_add(n_dims as u64));
                    let mut th = vec![0.0f64; n * n_dims];
                    for v in th.iter_mut() {
                        *v = if skew {
                            // standardized right-skew (Exp(1) - 1): mean 0, var 1
                            -(rng.next_f64().max(1e-12)).ln() - 1.0
                        } else {
                            rng.normal()
                        };
                    }
                    let mut y = vec![0usize; n * n_items];
                    for p in 0..n {
                        for i in 0..n_items {
                            let mut base = b_t[i];
                            for d in 0..n_dims {
                                base += a_t[i * n_dims + d] * th[p * n_dims + d];
                            }
                            let pr = 1.0 / (1.0 + (-base).exp());
                            y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
                        }
                    }
                    let cfg = MhrmConfig {
                        max_cycles: 900,
                        burn_in: 180,
                        mh_steps: 6,
                        seed: 0xABCD_u64.wrapping_add(rep as u64),
                        estimate_se: false,
                        ..MhrmConfig::default()
                    };
                    let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
                    if res.converged {
                        conv += 1;
                    }
                    for idx in 0..n_items * n_dims {
                        if pattern[idx] == 1 {
                            let e = res.loading[idx] - a_t[idx];
                            se2 += e * e;
                            sbias += e;
                            cnt += 1;
                        }
                    }
                    let est: Vec<f64> = (0..n).map(|p| res.theta[p * n_dims]).collect();
                    let tru: Vec<f64> = (0..n).map(|p| th[p * n_dims]).collect();
                    corr_sum += corr(&est, &tru);
                }
                let load_rmse = (se2 / cnt as f64).sqrt();
                let load_bias = sbias / cnt as f64;
                println!(
                    "[mhrm MC D={n_dims} N={n} n_free={n_free} skew={skew}] reps={reps} conv={:.3} loadRMSE={:.4} loadBias={:.4} thetaCorr={:.3}",
                    conv as f64 / reps as f64,
                    load_rmse,
                    load_bias,
                    corr_sum / reps as f64
                );
                assert!(conv as f64 / reps as f64 > 0.9, "convergence rate");
                if !skew {
                    assert!(load_rmse < 0.2, "normal loading RMSE {load_rmse}");
                }
            }
        }

        // correlated-Sigma condition (Cai 2010b): recover an exchangeable Phi at the near-PD-boundary
        // rho = 0.5, D = 3 (so a persistent PD-backtracking stall would surface over 500 reps).
        {
            let (n_dims, n, rho) = (3usize, 3000usize, 0.5f64);
            let per = 4usize;
            let n_items = per * n_dims;
            let mut pattern = vec![0u8; n_items * n_dims];
            let mut a_t = vec![0.0f64; n_items * n_dims];
            for d in 0..n_dims {
                for a in 0..per {
                    let i = d * per + a;
                    pattern[i * n_dims + d] = 1;
                    a_t[i * n_dims + d] = 0.9 + 0.1 * a as f64;
                }
            }
            let b_t: Vec<f64> = (0..n_items).map(|i| -0.4 + 0.1 * (i % 5) as f64).collect();
            let mut phi = vec![rho; n_dims * n_dims];
            for a in 0..n_dims {
                phi[a * n_dims + a] = 1.0;
            }
            let l = chol_lower(&phi, n_dims).unwrap();
            let n_off = n_dims * (n_dims - 1) / 2;
            let (mut conv, mut se2, mut sbias) = (0usize, 0.0f64, 0.0f64);
            for rep in 0..reps {
                let mut rng = Lcg(0x5EED_u64.wrapping_mul((rep as u64) + 1));
                let mut th = vec![0.0f64; n * n_dims];
                for p in 0..n {
                    let z: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
                    for a in 0..n_dims {
                        let mut v = 0.0;
                        for b in 0..=a {
                            v += l[a * n_dims + b] * z[b];
                        }
                        th[p * n_dims + a] = v;
                    }
                }
                let mut y = vec![0usize; n * n_items];
                for p in 0..n {
                    for i in 0..n_items {
                        let mut base = b_t[i];
                        for d in 0..n_dims {
                            base += a_t[i * n_dims + d] * th[p * n_dims + d];
                        }
                        let pr = 1.0 / (1.0 + (-base).exp());
                        y[p * n_items + i] = if rng.next_f64() < pr { 1 } else { 0 };
                    }
                }
                let cfg = MhrmConfig {
                    max_cycles: 1200,
                    burn_in: 300,
                    mh_steps: 6,
                    estimate_corr: true,
                    estimate_se: false,
                    seed: 0xBEEF_u64.wrapping_add(rep as u64),
                    ..MhrmConfig::default()
                };
                let res = fit_mhrm(&y, None, &pattern, n, n_items, n_dims, &cfg).unwrap();
                if res.converged {
                    conv += 1;
                }
                for a in 0..n_dims {
                    for b in a + 1..n_dims {
                        let e = res.corr[a * n_dims + b] - rho;
                        se2 += e * e;
                        sbias += e;
                    }
                }
            }
            let m = (reps * n_off) as f64;
            println!(
                "[mhrm MC correlated D={n_dims} N={n} rho={rho}] reps={reps} conv={:.3} corrRMSE={:.4} corrBias={:.4}",
                conv as f64 / reps as f64,
                (se2 / m).sqrt(),
                sbias / m
            );
            assert!(conv as f64 / reps as f64 > 0.9, "correlated convergence");
            assert!((se2 / m).sqrt() < 0.1, "correlated corr RMSE");
        }
        println!("=== done ===");
    }
}
