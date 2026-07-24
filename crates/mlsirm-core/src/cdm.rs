//! Cognitive diagnosis models: DINA (AND gate) and DINO (OR gate) by marginal-ML
//! EM over the `2^K` binary attribute-mastery profiles.
//!
//! Each respondent `j` has a binary profile `alpha_j in {0,1}^K` (mastery of `K`
//! skills); a `J x K` Q-matrix specifies which attributes each item requires. The
//! ideal (latent) response is
//!
//! * DINA: `eta_ij = prod_k alpha_jk^{q_ik}` — 1 iff the person masters ALL of the
//!   item's required attributes (conjunctive / AND gate);
//! * DINO: `eta_ij = 1 - prod_k (1 - alpha_jk)^{q_ik}` — 1 iff the person masters
//!   ANY required attribute (disjunctive / OR gate).
//!
//! The observed response adds a per-item slip `s_i = P(X=0 | eta=1)` and guess
//! `g_i = P(X=1 | eta=0)`:
//!
//! ```text
//! P(X_ij = 1 | alpha_j) = (1 - s_i)^{eta_ij} * g_i^{1 - eta_ij}
//! ```
//!
//! Estimation is marginal ML by EM over the `L = 2^K` profiles with a free
//! (unstructured) mixing distribution `pi_c = P(alpha = alpha_c)`. The item M-step
//! is closed form (slip = expected fraction of masters answering wrong; guess =
//! expected fraction of non-masters answering right), and the population step is a
//! column-mean of the posteriors. Persons are classified by their posterior mode
//! (MAP profile) and marginal attribute probabilities (attribute EAP).
//!
//! The fixed Q-matrix names the attribute dimensions, so there is no free label
//! switching to align (unlike the continuous latent space in `marginal.rs`). This
//! does **not** by itself identify `s`, `g`, and `pi`: nonzero Q rows and columns
//! are only structural sanity checks, and full DINA identifiability requires
//! stronger Q-matrix conditions (Gu & Xu, 2019). This implementation rejects the
//! degenerate zero-row/zero-column cases but does not certify global
//! identifiability; callers remain responsible for an appropriate study design.
//!
//! The saturated G-DINA ([`fit_gdina`]), empirical Q-matrix validation
//! ([`validate_q_matrix`]), item-level model selection ([`gdina_wald_selection`]),
//! and the higher-order structured attribute prior ([`fit_ho_cdm`], de la Torre &
//! Douglas, 2004) build on this DINA/DINO core in the same module. Deferred
//! (explicit non-goals): full Q-matrix *estimation* and global identifiability
//! certification.
//!
//! References (APA 7th ed.):
//! - de la Torre, J. (2009). DINA model and parameter estimation: A didactic.
//!   *Journal of Educational and Behavioral Statistics, 34*(1), 115–130.
//!   <https://doi.org/10.3102/1076998607309474>
//! - Gu, Y., & Xu, G. (2019). The sufficient and necessary condition for the
//!   identifiability and estimability of the DINA model. *Psychometrika, 84*(2),
//!   468–483. <https://doi.org/10.1007/s11336-018-9619-8>
//! - Junker, B. W., & Sijtsma, K. (2001). Cognitive assessment models with few
//!   assumptions, and connections with nonparametric item response theory.
//!   *Applied Psychological Measurement, 25*(3), 258–272.
//!   <https://doi.org/10.1177/01466210122032064>
//! - Templin, J. L., & Henson, R. A. (2006). Measurement of psychological disorders
//!   using cognitive diagnosis models. *Psychological Methods, 11*(3), 287–305.
//!   <https://doi.org/10.1037/1082-989X.11.3.287>

/// Cognitive-diagnosis gate: `Dina` = conjunctive (AND), `Dino` = disjunctive (OR).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CdmModel {
    Dina,
    Dino,
}

/// EM configuration. Defaults follow the crate's marginal-ML conventions.
#[derive(Clone, Copy, Debug)]
pub struct CdmConfig {
    /// Maximum EM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `|delta loglik|`.
    pub tol: f64,
    /// Clamp for slip/guess and floor for `pi_c` (avoids `ln 0`).
    pub eps: f64,
    /// Interior back-off from the monotonicity boundary `g = 1 - s`.
    pub mono_backoff: f64,
    /// Initial slip.
    pub init_slip: f64,
    /// Initial guess.
    pub init_guess: f64,
    /// Keep the previous slip (resp. guess) when the expected master (resp.
    /// non-master) count for an item falls below this — no information to update it.
    pub count_floor: f64,
}

impl Default for CdmConfig {
    fn default() -> Self {
        Self {
            max_iter: 500,
            tol: 1e-6,
            eps: 1e-6,
            mono_backoff: 1e-3,
            init_slip: 0.2,
            init_guess: 0.2,
            count_floor: 1e-8,
        }
    }
}

/// Fitted DINA/DINO parameters and person classifications.
#[derive(Clone, Debug)]
pub struct CdmResult {
    pub model: CdmModel,
    /// Per-item slip `s_i`, length `J`.
    pub slip: Vec<f64>,
    /// Per-item guess `g_i`, length `J`.
    pub guess: Vec<f64>,
    /// Population mixing proportions `pi_c`, length `2^K`, sum 1.
    pub profile_prob: Vec<f64>,
    /// Bit-encoded MAP profile per person (`argmax_c P(alpha_c | X_j)`), length `N`.
    pub map_profile: Vec<u32>,
    /// Marginal `P(alpha_jk = 1 | X_j)`, row-major `N x K`.
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `2*J + (2^K - 1)`.
    pub n_parameters: usize,
}

fn validate(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    // L = 2^K drives both the eta table (J*L) and the O(N*J*L) E-step; cap K at 15.
    if !(1..=15).contains(&n_attributes) {
        return Err(format!(
            "n_attributes must be in 1..=15 (L = 2^K grid + O(N*J*L) cost); got {n_attributes}"
        ));
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if !cfg.eps.is_finite() || !(0.0 < cfg.eps && cfg.eps < 0.5) {
        return Err("eps must be finite and in (0, 0.5)".into());
    }
    if !cfg.mono_backoff.is_finite() || cfg.mono_backoff <= 2.0 * cfg.eps || cfg.mono_backoff >= 1.0
    {
        return Err("mono_backoff must be finite, greater than 2 * eps, and less than 1".into());
    }
    if !cfg.init_slip.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_slip) {
        return Err("init_slip must be finite and in [eps, 1 - eps]".into());
    }
    if !cfg.init_guess.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_guess) {
        return Err("init_guess must be finite and in [eps, 1 - eps]".into());
    }
    if cfg.init_slip + cfg.init_guess >= 1.0 {
        return Err("init_slip + init_guess must be less than 1".into());
    }
    if !cfg.count_floor.is_finite() || cfg.count_floor < 0.0 {
        return Err("count_floor must be finite and non-negative".into());
    }
    // checked_mul mirrors fit_mmle_2pl: a wrapped product could otherwise pass the
    // length check and let the E-step index out of bounds on adversarial dimensions.
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_q = crate::checked_mul_usize(n_items, n_attributes, "Q-matrix size overflows usize")?;
    if q_matrix.len() != n_q {
        return Err("q_matrix must have length n_items * n_attributes".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && v != 0.0 && v != 1.0 {
            return Err(format!("y[{idx}] must be 0 or 1 where observed; got {v}"));
        }
    }
    for (idx, &v) in q_matrix.iter().enumerate() {
        if v != 0 && v != 1 {
            return Err(format!("q_matrix[{idx}] must be 0 or 1; got {v}"));
        }
    }
    // An entirely unobserved item has no likelihood contribution, so its slip and
    // guess would merely echo the initial values while being reported as estimates.
    for i in 0..n_items {
        if !(0..n_persons).any(|p| observed[p * n_items + i]) {
            return Err(format!("item {i} has no observed responses"));
        }
    }
    // Every Q row nonzero: an all-zero row gives qmask==0, so DINA eta == 1 and
    // DINO eta == 0 for all profiles — the item measures nothing.
    for i in 0..n_items {
        if !(0..n_attributes).any(|k| q_matrix[i * n_attributes + k] != 0) {
            return Err(format!(
                "q_matrix row {i} is all-zero (item measures no attribute)"
            ));
        }
    }
    // Every Q column nonzero: an attribute measured by no item carries zero data
    // information. This is necessary structural validation, not a certificate of
    // the stronger Q-matrix conditions required for global identifiability.
    for k in 0..n_attributes {
        if !(0..n_items).any(|i| q_matrix[i * n_attributes + k] != 0) {
            return Err(format!(
                "q_matrix column {k} is all-zero (attribute measured by no item)"
            ));
        }
    }
    Ok(())
}

/// Normalized posterior over the `L` profiles for person `j`; returns `ln P(X_j)`.
/// `post` is filled in place (reused scratch across persons — no `N*L` storage).
/// `lp1[i*2 + b]` / `lp0[i*2 + b]` are `ln P(X=1|eta=b)` / `ln P(X=0|eta=b)`.
#[allow(clippy::too_many_arguments)]
fn posterior_row(
    j: usize,
    y: &[f64],
    observed: &[bool],
    n_items: usize,
    l: usize,
    eta: &[u8],
    lp1: &[f64],
    lp0: &[f64],
    log_pi: &[f64],
    post: &mut [f64],
) -> f64 {
    for (c, slot) in post.iter_mut().enumerate().take(l) {
        let mut acc = log_pi[c];
        for i in 0..n_items {
            let idx = j * n_items + i;
            if observed[idx] {
                let b = eta[i * l + c] as usize;
                let yy = y[idx];
                acc += yy * lp1[i * 2 + b] + (1.0 - yy) * lp0[i * 2 + b];
            }
        }
        *slot = acc; // log-numerator, exponentiated below
    }
    let m = post[..l].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut denom = 0.0;
    for c in 0..l {
        denom += (post[c] - m).exp();
    }
    for c in 0..l {
        post[c] = (post[c] - m).exp() / denom;
    }
    m + denom.ln()
}

/// Closed-form item M-step (de la Torre, 2009, Eqs. 9-10) with the monotonicity
/// projection. `i1/r1/i0/r0` are the expected master/non-master and correct-count
/// cells for item `i`.
#[allow(clippy::too_many_arguments)]
fn update_item(
    i: usize,
    i1: &[f64],
    r1: &[f64],
    i0: &[f64],
    r0: &[f64],
    s: &mut [f64],
    g: &mut [f64],
    cfg: &CdmConfig,
) {
    // Unconstrained maximisers of Q_i = R1 ln(1-s) + (I1-R1) ln s
    //                                   + R0 ln g  + (I0-R0) ln(1-g):
    //   s_i = 1 - R1_i/I1_i  (masters answering wrong),  g_i = R0_i/I0_i (non-masters right).
    // Count guard: an item with ~no expected mass in a group carries no information
    // for that parameter, so keep the previous value (mirrors the mmle singular break).
    let mut si = if i1[i] > cfg.count_floor {
        1.0 - r1[i] / i1[i]
    } else {
        s[i]
    };
    let mut gi = if i0[i] > cfg.count_floor {
        r0[i] / i0[i]
    } else {
        g[i]
    };
    // Monotonicity / identification 1 - s_i > g_i (equivalently s_i + g_i < 1). If
    // violated, the exact constrained maximiser is on the boundary g = 1 - s, where
    // Q_i collapses to one binomial with maximiser pbar_i = (R1+R0)/(I1+I0); back off
    // by mono_backoff to stay in the open feasible set. When this fires at least one
    // group carried mass (else si,gi kept the previous, whose sum is < 1 by the
    // invariant), so I1+I0 > 0.
    // ponytail: open-set back-off makes this a GEM step, not a strict M-maximiser;
    // it lands mono_backoff/2 interior to the boundary. Never fires under an
    // identifiable Q with literature-grade (s,g); a stricter inner 1-D search buys
    // nothing here.
    if si + gi >= 1.0 {
        let pbar = (r1[i] + r0[i]) / (i1[i] + i0[i]);
        si = (1.0 - pbar) - cfg.mono_backoff / 2.0;
        gi = pbar - cfg.mono_backoff / 2.0;
    }
    s[i] = si.clamp(cfg.eps, 1.0 - cfg.eps);
    g[i] = gi.clamp(cfg.eps, 1.0 - cfg.eps);
}

/// Fit DINA/DINO by marginal EM. `y` and `observed` are row-major `N*J` (`y` in
/// {0,1}); `q_matrix` is row-major `J*K` (entries in {0,1}). Missing cells
/// (`observed == false`) are dropped from both the E-step likelihood and the M-step
/// counts (MAR), mirroring `fit_mmle_2pl`. Returns `Err` on malformed input.
#[allow(clippy::too_many_arguments)]
pub fn fit_cdm(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    model: CdmModel,
    cfg: &CdmConfig,
) -> Result<CdmResult, String> {
    validate(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    let l = 1usize << n_attributes;

    // Per-item required-attribute bitmask (Q is fixed across EM).
    let mut qmask = vec![0usize; n_items];
    for i in 0..n_items {
        let mut mask = 0usize;
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                mask |= 1 << k;
            }
        }
        qmask[i] = mask;
    }
    // Ideal response eta_ic in {0,1} as one bitwise test; the two gates differ here only.
    let mut eta = vec![0u8; n_items * l];
    for i in 0..n_items {
        for c in 0..l {
            eta[i * l + c] = match model {
                CdmModel::Dina => ((c & qmask[i]) == qmask[i]) as u8,
                CdmModel::Dino => ((c & qmask[i]) != 0) as u8,
            };
        }
    }

    let mut s = vec![cfg.init_slip; n_items];
    let mut g = vec![cfg.init_guess; n_items];
    let mut pi = vec![1.0 / l as f64; l]; // deterministic uniform prior
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    let mut post = vec![0.0f64; l];
    let mut lp1 = vec![0.0f64; n_items * 2];
    let mut lp0 = vec![0.0f64; n_items * 2];
    let mut log_pi = vec![0.0f64; l];

    let refresh_tables = |s: &[f64], g: &[f64], lp1: &mut [f64], lp0: &mut [f64]| {
        for i in 0..n_items {
            let sc = s[i].clamp(cfg.eps, 1.0 - cfg.eps);
            let gc = g[i].clamp(cfg.eps, 1.0 - cfg.eps);
            lp1[i * 2 + 1] = (1.0 - sc).ln(); // master, correct
            lp0[i * 2 + 1] = sc.ln(); //          master, wrong
            lp1[i * 2] = gc.ln(); //              non-master, correct
            lp0[i * 2] = (1.0 - gc).ln(); //      non-master, wrong
        }
    };

    for _ in 0..cfg.max_iter {
        refresh_tables(&s, &g, &mut lp1, &mut lp0);
        for c in 0..l {
            log_pi[c] = pi[c].ln();
        }

        // E-step: accumulate the four eta x response expected-count cells on the fly.
        let mut i1 = vec![0.0f64; n_items];
        let mut r1 = vec![0.0f64; n_items];
        let mut i0 = vec![0.0f64; n_items];
        let mut r0 = vec![0.0f64; n_items];
        let mut pi_new = vec![0.0f64; l];
        let mut total_ll = 0.0;
        for j in 0..n_persons {
            total_ll += posterior_row(
                j, y, observed, n_items, l, &eta, &lp1, &lp0, &log_pi, &mut post,
            );
            for c in 0..l {
                pi_new[c] += post[c];
            }
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    // pbar = P(eta_ij = 1 | X_j) = posterior mass that j masters item i.
                    let mut pbar = 0.0;
                    for c in 0..l {
                        if eta[i * l + c] == 1 {
                            pbar += post[c];
                        }
                    }
                    let yy = y[idx];
                    i1[i] += pbar;
                    r1[i] += yy * pbar;
                    i0[i] += 1.0 - pbar;
                    r0[i] += yy * (1.0 - pbar);
                }
            }
        }
        loglik_trace.push(total_ll);

        // The likelihood just evaluated belongs to the current parameters. Stop
        // before another M-step so the returned parameters and trace endpoint agree.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: closed-form items, then population as floored, renormalized mean posterior.
        for i in 0..n_items {
            update_item(i, &i1, &r1, &i0, &r0, &mut s, &mut g, cfg);
        }
        let nf = n_persons as f64;
        let mut z = 0.0;
        for c in 0..l {
            pi[c] = (pi_new[c] / nf).max(cfg.eps);
            z += pi[c];
        }
        for c in 0..l {
            pi[c] /= z;
        }
        n_iter += 1;
    }

    // Final classification: one recompute pass at the converged parameters.
    refresh_tables(&s, &g, &mut lp1, &mut lp0);
    for c in 0..l {
        log_pi[c] = pi[c].ln();
    }
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut final_ll = 0.0;
    for j in 0..n_persons {
        final_ll += posterior_row(
            j, y, observed, n_items, l, &eta, &lp1, &lp0, &log_pi, &mut post,
        );
        let mut best = 0usize;
        for c in 1..l {
            if post[c] > post[best] {
                best = c;
            }
        }
        map_profile[j] = best as u32;
        for k in 0..n_attributes {
            let mut p = 0.0;
            for c in 0..l {
                if (c >> k) & 1 == 1 {
                    p += post[c];
                }
            }
            attr_prob[j * n_attributes + k] = p;
        }
    }
    // A max-iteration exit occurs immediately after an M-step, so record the
    // likelihood of those returned parameters. On convergence the final E-step
    // already supplied the same endpoint.
    if !converged {
        loglik_trace.push(final_ll);
    }

    Ok(CdmResult {
        model,
        slip: s,
        guess: g,
        profile_prob: pi,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        n_parameters: 2 * n_items + (l - 1),
    })
}

// ---------------------------------------------------------------------------
// Generalized DINA (G-DINA; de la Torre, 2011)
// ---------------------------------------------------------------------------

/// Fitted saturated G-DINA model (de la Torre, 2011). Item parameters are stored
/// ragged in CSR layout: item `i` owns the slice `[item_off[i]..item_off[i+1])` of
/// width `2^{K_i}` (`K_i` = number of attributes item `i` requires), indexed by the
/// reduced attribute-mastery class. `item_prob[item_off[i] + l]` is the success
/// probability `P(X_i = 1 | reduced class l)`; `item_delta` is the same slice under
/// the identity link (`delta = M^{-1} p`: intercept, main effects, interactions).
#[derive(Clone, Debug)]
pub struct GdinaResult {
    /// CSR offsets, length `n_items + 1`; item `i` width = `2^{K_i}`.
    pub item_off: Vec<usize>,
    /// Saturated success probabilities `p_il`, CSR-flat.
    pub item_prob: Vec<f64>,
    /// Identity-link parameters `delta_iS = M^{-1} p_i`, same CSR layout.
    pub item_delta: Vec<f64>,
    /// Number of required attributes `K_i` per item (to interpret the ragged rows).
    pub k_required: Vec<u32>,
    /// Population mixing proportions over the full `2^K` grid, sum 1.
    pub profile_prob: Vec<f64>,
    /// Bit-encoded MAP profile per person, length `N`.
    pub map_profile: Vec<u32>,
    /// Marginal `P(alpha_jk = 1 | X_j)`, row-major `N x K`.
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// `sum_i 2^{K_i} + (2^K - 1)`.
    pub n_parameters: usize,
}

/// Bit-encoded reduced attribute-mastery class of full profile `c` for an item with
/// required-attribute bitmask `qmask`: gather the mastery bits at the set positions
/// of `qmask`, packed LSB-ascending in ascending attribute order. This generalizes
/// the DINA gate — `reduce_class(c, qmask) == 2^{K_i} - 1` iff `(c & qmask) == qmask`
/// (all required attributes mastered). The bit convention is load-bearing: it must be
/// identical across `reduce_class`, the design matrix, and `mobius_inverse_inplace`.
#[inline]
fn reduce_class(c: usize, qmask: usize) -> usize {
    let (mut l, mut m, mut q) = (0usize, 0u32, qmask);
    while q != 0 {
        let k = q.trailing_zeros();
        l |= ((c >> k) & 1) << m;
        q &= q - 1; // clear lowest set bit
        m += 1;
    }
    l
}

/// In-place identity-link transform `delta = M^{-1} p` (signed subset Möbius), where
/// `M[l][S] = [(l & S) == S]` is the reduced-class superset (zeta) design. For each
/// required-attribute bit, subtract the value of the pattern without that bit;
/// `k_star` = the item's required-attribute count (`v.len() == 2^{k_star}`). This is
/// the exact inverse of the zeta subset-sum, computed in `O(k_star * 2^{k_star})`
/// without materializing or inverting any matrix.
fn mobius_inverse_inplace(v: &mut [f64], k_star: u32) {
    for t in 0..k_star {
        let bit = 1usize << t;
        for s in 0..v.len() {
            if s & bit != 0 {
                v[s] -= v[s ^ bit];
            }
        }
    }
}

/// Per-person posterior over the full `2^K` profiles for G-DINA; returns `ln P(X_j)`.
/// Mirrors [`posterior_row`] but indexes the ragged per-item CSR success-probability
/// tables through `red` (reduced-class index) + `item_off`.
#[allow(clippy::too_many_arguments)]
fn posterior_row_gdina(
    j: usize,
    y: &[f64],
    observed: &[bool],
    n_items: usize,
    l_full: usize,
    red: &[u16],
    log_p1: &[f64],
    log_p0: &[f64],
    item_off: &[usize],
    log_pi: &[f64],
    post: &mut [f64],
) -> f64 {
    for (c, slot) in post.iter_mut().enumerate().take(l_full) {
        let mut acc = log_pi[c];
        for i in 0..n_items {
            let idx = j * n_items + i;
            if observed[idx] {
                let cell = item_off[i] + red[i * l_full + c] as usize;
                let yy = y[idx];
                acc += yy * log_p1[cell] + (1.0 - yy) * log_p0[cell];
            }
        }
        *slot = acc;
    }
    let m = post[..l_full]
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut denom = 0.0;
    for c in 0..l_full {
        denom += (post[c] - m).exp();
    }
    for c in 0..l_full {
        post[c] = (post[c] - m).exp() / denom;
    }
    m + denom.ln()
}

/// Fit the saturated G-DINA model (de la Torre, 2011) by marginal-ML EM over the
/// `2^K` attribute profiles. Every reduced attribute-mastery class of every item gets
/// a free success probability; DINA, DINO, A-CDM, LLM and R-RUM are all constrained
/// special cases that can be read off the fitted identity-link `item_delta` pattern
/// (they are not refit here — see the module deferred-scope note).
///
/// The E-step and population update are the same profile-grid EM as [`fit_cdm`]; only
/// the item conditional and M-step generalize: the closed-form saturated maximiser is
/// `p_il = R_il / I_il` (expected correct / expected total in reduced class `l`),
/// exactly [`fit_cdm`]'s two-cell slip/guess step generalized to `2^{K_i}` classes.
/// The box constraint `0 <= p_il <= 1` holds for free (`0 <= R_il <= I_il`). This
/// saturated estimator does **not** impose order restrictions: Q-matrix
/// identifiability does not imply that mastering more required attributes raises the
/// success probability, and an all-mastered class need not have the largest estimate.
/// Subset-lattice order-restricted estimation is a separate model choice described by
/// Hong et al. (2016) and is not implemented here. `y`/`observed` are row-major
/// `N*J`, `q_matrix` row-major `J*K`; missing cells are dropped (MAR).
///
/// References (APA 7th ed.):
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
///     76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
///   Hong, C.-Y., Chang, Y.-W., & Tsai, R.-C. (2016). Estimation of generalized
///     DINA model with order restrictions. *Journal of Classification, 33*(3),
///     460-484. https://doi.org/10.1007/s00357-016-9215-5
///   Ma, W., & de la Torre, J. (2020). GDINA: An R package for cognitive diagnosis
///     modeling. *Journal of Statistical Software, 93*(14), 1-26.
///     https://doi.org/10.18637/jss.v093.i14
#[allow(clippy::too_many_arguments)]
pub fn fit_gdina(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<GdinaResult, String> {
    validate(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    let l_full = 1usize << n_attributes;

    // Per-item required-attribute bitmask and count K_i.
    let mut qmask = vec![0usize; n_items];
    let mut k_required = vec![0u32; n_items];
    for i in 0..n_items {
        let mut mask = 0usize;
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                mask |= 1 << k;
            }
        }
        qmask[i] = mask;
        k_required[i] = mask.count_ones();
    }

    // Ragged CSR: item i owns [item_off[i]..item_off[i+1]) of width L_i = 2^{K_i}.
    let mut item_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        item_off[i + 1] = item_off[i] + (1usize << k_required[i]);
    }
    let total = item_off[n_items];

    // Reduced-class index of every (item, full-profile) pair, precomputed once. `u16`
    // holds any class index (max `2^{K_i} - 1 <= 2^15 - 1`) because `validate` caps
    // K <= 15; raising that cap past 15 would require widening this element type.
    let mut red = vec![0u16; n_items * l_full];
    for i in 0..n_items {
        for c in 0..l_full {
            red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
        }
    }

    // Monotone init: p rises with the count of mastered required attributes, from
    // init_guess (none) to 1 - init_slip (all); endpoints match DINA's (g, 1-s).
    let mut p = vec![0.0f64; total];
    for i in 0..n_items {
        let ki = k_required[i] as f64; // >= 1 (validate rejects all-zero Q rows)
        for l in 0..(item_off[i + 1] - item_off[i]) {
            let frac = (l.count_ones() as f64) / ki;
            p[item_off[i] + l] = cfg.init_guess + (1.0 - cfg.init_slip - cfg.init_guess) * frac;
        }
    }
    let mut pi = vec![1.0 / l_full as f64; l_full];
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    let mut post = vec![0.0f64; l_full];
    let mut log_p1 = vec![0.0f64; total];
    let mut log_p0 = vec![0.0f64; total];
    let mut log_pi = vec![0.0f64; l_full];

    let refresh = |p: &[f64], log_p1: &mut [f64], log_p0: &mut [f64]| {
        for x in 0..total {
            let pc = p[x].clamp(cfg.eps, 1.0 - cfg.eps);
            log_p1[x] = pc.ln();
            log_p0[x] = (1.0 - pc).ln();
        }
    };

    for _ in 0..cfg.max_iter {
        refresh(&p, &mut log_p1, &mut log_p0);
        for c in 0..l_full {
            log_pi[c] = pi[c].ln();
        }

        // E-step: scatter expected reduced-class counts I_il / R_il over the posterior.
        let mut ii = vec![0.0f64; total];
        let mut rr = vec![0.0f64; total];
        let mut pi_new = vec![0.0f64; l_full];
        let mut total_ll = 0.0;
        for j in 0..n_persons {
            total_ll += posterior_row_gdina(
                j, y, observed, n_items, l_full, &red, &log_p1, &log_p0, &item_off, &log_pi,
                &mut post,
            );
            for c in 0..l_full {
                pi_new[c] += post[c];
            }
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let (off, yy) = (item_off[i], y[idx]);
                    for c in 0..l_full {
                        let cell = off + red[i * l_full + c] as usize;
                        ii[cell] += post[c];
                        rr[cell] += yy * post[c];
                    }
                }
            }
        }
        loglik_trace.push(total_ll);

        // The likelihood just evaluated belongs to the current parameters. Stop
        // before another M-step so the returned parameters and trace endpoint agree.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: saturated closed form p_il = R_il / I_il (de la Torre, 2011, Eq. 10).
        // Box 0<=p<=1 is free (0<=R<=I); count_floor keeps a class's previous value
        // when the posterior gives it ~no mass (empty reduced class).
        for x in 0..total {
            if ii[x] > cfg.count_floor {
                p[x] = (rr[x] / ii[x]).clamp(cfg.eps, 1.0 - cfg.eps);
            }
        }
        let nf = n_persons as f64;
        let mut z = 0.0;
        for c in 0..l_full {
            pi[c] = (pi_new[c] / nf).max(cfg.eps);
            z += pi[c];
        }
        for c in 0..l_full {
            pi[c] /= z;
        }
        n_iter += 1;
    }

    // Classification pass (mirrors fit_cdm's tail; duplicated to keep the tested DINA
    // core untouched).
    refresh(&p, &mut log_p1, &mut log_p0);
    for c in 0..l_full {
        log_pi[c] = pi[c].ln();
    }
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut final_ll = 0.0;
    for j in 0..n_persons {
        final_ll += posterior_row_gdina(
            j, y, observed, n_items, l_full, &red, &log_p1, &log_p0, &item_off, &log_pi, &mut post,
        );
        let mut best = 0usize;
        for c in 1..l_full {
            if post[c] > post[best] {
                best = c;
            }
        }
        map_profile[j] = best as u32;
        for k in 0..n_attributes {
            let mut pk = 0.0;
            for c in 0..l_full {
                if (c >> k) & 1 == 1 {
                    pk += post[c];
                }
            }
            attr_prob[j * n_attributes + k] = pk;
        }
    }

    // A max-iteration exit occurs immediately after an M-step, so record the
    // likelihood of those returned parameters. On convergence the final E-step
    // already supplied the same endpoint.
    if !converged {
        loglik_trace.push(final_ll);
    }

    // Identity-link parameters delta = M^{-1} p, per item slice.
    let mut item_delta = p.clone();
    for i in 0..n_items {
        mobius_inverse_inplace(&mut item_delta[item_off[i]..item_off[i + 1]], k_required[i]);
    }

    Ok(GdinaResult {
        item_off,
        item_prob: p,
        item_delta,
        k_required,
        profile_prob: pi,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        n_parameters: total + (l_full - 1),
    })
}

fn ensure_gdina_converged(res: &GdinaResult, cfg: &CdmConfig) -> Result<(), String> {
    if res.converged {
        return Ok(());
    }
    let final_delta = res
        .loglik_trace
        .windows(2)
        .last()
        .map(|w| (w[1] - w[0]).abs())
        .unwrap_or(f64::INFINITY);
    Err(format!(
        concat!(
            "G-DINA calibration did not converge after {} of {} M-steps: ",
            "final |delta loglik| = {:.6e} (tol = {:.6e})"
        ),
        res.n_iter, cfg.max_iter, final_delta, cfg.tol
    ))
}

/// Result of [`validate_q_matrix`] (de la Torre & Chiu, 2016). Per item, the
/// method suggests the smallest attribute vector whose PVAF reaches the cutoff.
#[derive(Clone, Debug)]
pub struct QValidationResult {
    pub n_attributes: usize,
    /// The suggested (validated) Q-matrix, row-major `J x K`, entries 0/1.
    pub suggested_q: Vec<u8>,
    /// PVAF of the suggested q-vector, per item (in `[0, 1]`).
    pub suggested_pvaf: Vec<f64>,
    /// PVAF of the caller's provisional q-vector, per item (for comparison).
    pub provisional_pvaf: Vec<f64>,
    /// `true` where the suggested q-vector differs from the provisional one.
    pub flagged: Vec<bool>,
    /// The PVAF cutoff used.
    pub epsilon: f64,
    /// M-steps completed by the provisional G-DINA calibration.
    pub calibration_n_iter: usize,
    /// Configured M-step limit for the provisional calibration.
    pub calibration_max_iter: usize,
    /// Successful calibration stop reason (currently `"tolerance_met"`).
    pub calibration_termination_reason: &'static str,
    /// Absolute final change in the provisional calibration log-likelihood.
    pub calibration_final_loglik_change: f64,
    /// Absolute log-likelihood-change tolerance used by the calibration.
    pub calibration_tol: f64,
}

/// Empirical Q-matrix validation by the PVAF (proportion of variance accounted
/// for) method of de la Torre and Chiu (2016).
///
/// The G-DINA item response function `P(alpha_c)` varies across the `2^K` latent
/// attribute classes. A candidate q-vector groups those classes into equivalence
/// classes (masters vs. non-masters of each *required* attribute), and its
/// captured variance is
///
/// ```text
/// zeta^2(q) = sum_l W_l (Pbar_l(q) - Pbar)^2,   PVAF(q) = zeta^2(q) / zeta^2_full
/// ```
///
/// where `Pbar_l(q)` is the population-weighted mean success probability within
/// reduced class `l` under `q`, `W_l` its total weight, `Pbar` the item's overall
/// mean, and `zeta^2_full` the total across-class variance (the saturated
/// reference). PVAF is monotone in `q` and equals 1 at the full attribute vector.
/// For each item the method returns the q-vector with the **fewest** required
/// attributes whose `PVAF >= epsilon` (ties broken by larger PVAF): an
/// under-specified provisional q falls short of the cutoff and is enlarged, an
/// over-specified one is trimmed because a smaller vector already reaches it.
///
/// The class weights `pi_c` and the reference IRF are read off a G-DINA fit with
/// the **provisional** Q ([`fit_gdina`]): that structural model identifies the
/// attribute labels (which latent bit is which attribute), and each item's
/// *saturated* success probability over all `2^K` classes,
/// `p_{i,c} = E[X_i | alpha_c]`, is then recovered nonparametrically from the
/// fitted posteriors (expected-correct / expected-count per full class). Because a
/// mis-specified item's responses still correlate with the attributes recovered
/// from the *other* items, its saturated IRF exposes the attributes it truly
/// depends on — so an under-specified provisional q shows `PVAF < epsilon` and is
/// enlarged. The method assumes the provisional Q is mostly correct (enough items
/// to identify the attributes). `y`/`observed` are row-major `N*J` (missing
/// dropped, MAR); `provisional_q` is row-major `J*K`, entries 0/1, each item
/// loading at least one attribute. Cost is `O(J * 4^K)` for the exhaustive
/// q-vector search, so `K` is capped at 10. Validation returns an error rather
/// than computing PVAF from an unconverged provisional G-DINA calibration.
///
/// References (APA 7th ed.):
///   de la Torre, J., & Chiu, C.-Y. (2016). A general method of empirical Q-matrix
///     validation. *Psychometrika, 81*(2), 253–273.
///     <https://doi.org/10.1007/s11336-015-9467-8>
///   de la Torre, J. (2008). An empirically based method of Q-matrix validation for
///     the DINA model: Development and applications. *Journal of Educational
///     Measurement, 45*(4), 343-362. https://doi.org/10.1111/j.1745-3984.2008.00069.x
#[allow(clippy::too_many_arguments)]
pub fn validate_q_matrix(
    y: &[f64],
    observed: &[bool],
    provisional_q: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    epsilon: f64,
    cfg: &CdmConfig,
) -> Result<QValidationResult, String> {
    if n_attributes == 0 || n_attributes > 10 {
        return Err("n_attributes must be in 1..=10 for the PVAF q-vector search".into());
    }
    if !(epsilon > 0.0 && epsilon <= 1.0) {
        return Err("epsilon must be in (0, 1]".into());
    }
    if provisional_q.len() != n_items * n_attributes {
        return Err("provisional_q must have length n_items * n_attributes".into());
    }
    for &v in provisional_q {
        if v > 1 {
            return Err("provisional_q entries must be 0 or 1".into());
        }
    }

    for i in 0..n_items {
        if (0..n_attributes).all(|k| provisional_q[i * n_attributes + k] == 0) {
            return Err("each provisional_q row must load at least one attribute".into());
        }
    }

    let l_full = 1usize << n_attributes;
    let full_mask = l_full - 1;

    // Fit the structural G-DINA under the provisional Q (identifies the attribute
    // labels; also validates y/observed shapes and the config).
    let res = fit_gdina(
        y,
        observed,
        provisional_q,
        n_persons,
        n_items,
        n_attributes,
        cfg,
    )?;
    ensure_gdina_converged(&res, cfg)?;
    let calibration_final_loglik_change = res
        .loglik_trace
        .windows(2)
        .last()
        .map(|w| (w[1] - w[0]).abs())
        .unwrap_or(f64::INFINITY);

    // Recover each item's SATURATED IRF over all 2^K full classes and the class
    // weights pi_c from one posterior pass at the fitted parameters. The provisional
    // fit's own item probabilities are only over reduced classes; the saturated IRF
    // p_{i,c} = E[X_i | alpha_c] over full classes is what PVAF needs.
    let mut qmask = vec![0usize; n_items];
    for i in 0..n_items {
        for k in 0..n_attributes {
            if provisional_q[i * n_attributes + k] != 0 {
                qmask[i] |= 1 << k;
            }
        }
    }
    let total = res.item_off[n_items];
    let mut red = vec![0u16; n_items * l_full];
    for i in 0..n_items {
        for c in 0..l_full {
            red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
        }
    }
    let mut log_p1 = vec![0.0f64; total];
    let mut log_p0 = vec![0.0f64; total];
    for x in 0..total {
        let pc = res.item_prob[x].clamp(cfg.eps, 1.0 - cfg.eps);
        log_p1[x] = pc.ln();
        log_p0[x] = (1.0 - pc).ln();
    }
    let log_pi: Vec<f64> = res
        .profile_prob
        .iter()
        .map(|v| v.max(cfg.eps).ln())
        .collect();

    let mut icount = vec![0.0f64; n_items * l_full]; // I_{i,c} expected count
    let mut rcount = vec![0.0f64; n_items * l_full]; // R_{i,c} expected correct
    let mut pi_c = vec![0.0f64; l_full];
    let mut post = vec![0.0f64; l_full];
    for j in 0..n_persons {
        posterior_row_gdina(
            j,
            y,
            observed,
            n_items,
            l_full,
            &red,
            &log_p1,
            &log_p0,
            &res.item_off,
            &log_pi,
            &mut post,
        );
        for c in 0..l_full {
            pi_c[c] += post[c];
        }
        for i in 0..n_items {
            let idx = j * n_items + i;
            if observed[idx] {
                let yy = y[idx];
                for c in 0..l_full {
                    icount[i * l_full + c] += post[c];
                    rcount[i * l_full + c] += yy * post[c];
                }
            }
        }
    }
    let ntot: f64 = pi_c.iter().sum();
    let pi: Vec<f64> = pi_c.iter().map(|v| v / ntot).collect();

    let mut suggested_q = vec![0u8; n_items * n_attributes];
    let mut suggested_pvaf = vec![0.0f64; n_items];
    let mut provisional_pvaf = vec![0.0f64; n_items];
    let mut flagged = vec![false; n_items];

    // Reusable per-mask group accumulators (sized to the largest reduced class set).
    let mut num = vec![0.0f64; l_full];
    let mut den = vec![0.0f64; l_full];
    let mut p_full = vec![0.0f64; l_full];

    for i in 0..n_items {
        // Saturated IRF: p_{i,c} = R_{i,c} / I_{i,c}; empty classes (~zero weight)
        // fall back to the overall item mean so they never distort the variance.
        let mut mean_num = 0.0f64;
        let mut mean_den = 0.0f64;
        for c in 0..l_full {
            let ic = icount[i * l_full + c];
            if ic > cfg.count_floor {
                p_full[c] = (rcount[i * l_full + c] / ic).clamp(0.0, 1.0);
                mean_num += rcount[i * l_full + c];
                mean_den += ic;
            }
        }
        let item_mean = if mean_den > 0.0 {
            mean_num / mean_den
        } else {
            0.0
        };
        for c in 0..l_full {
            if icount[i * l_full + c] <= cfg.count_floor {
                p_full[c] = item_mean;
            }
        }
        let p_c: &[f64] = &p_full;

        // Overall mean and total across-class variance (saturated reference).
        let mut pbar = 0.0f64;
        for c in 0..l_full {
            pbar += pi[c] * p_c[c];
        }
        let mut var_tot = 0.0f64;
        for c in 0..l_full {
            let d = p_c[c] - pbar;
            var_tot += pi[c] * d * d;
        }

        // PVAF of a candidate q-vector (bit-mask of required attributes).
        let mut pvaf_of = |mask: usize| -> f64 {
            if var_tot <= cfg.eps {
                return 0.0; // non-discriminating item: no variance to explain
            }
            let lred = 1usize << mask.count_ones();
            for l in 0..lred {
                num[l] = 0.0;
                den[l] = 0.0;
            }
            for c in 0..l_full {
                let l = reduce_class(c, mask);
                num[l] += pi[c] * p_c[c];
                den[l] += pi[c];
            }
            let mut var_q = 0.0f64;
            for l in 0..lred {
                if den[l] > 0.0 {
                    let d = num[l] / den[l] - pbar;
                    var_q += den[l] * d * d;
                }
            }
            (var_q / var_tot).clamp(0.0, 1.0)
        };

        let prov_mask = {
            let mut m = 0usize;
            for k in 0..n_attributes {
                if provisional_q[i * n_attributes + k] != 0 {
                    m |= 1 << k;
                }
            }
            m
        };
        debug_assert_ne!(prov_mask, 0, "provisional rows were validated above");
        provisional_pvaf[i] = pvaf_of(prov_mask);

        if var_tot <= cfg.eps {
            // Uninformative item: cannot be validated. Keep the provisional vector.
            for k in 0..n_attributes {
                suggested_q[i * n_attributes + k] = provisional_q[i * n_attributes + k];
            }
            suggested_pvaf[i] = 0.0;
            flagged[i] = false;
            continue;
        }

        // Search for the fewest-attribute q-vector reaching the cutoff. The full
        // vector always qualifies (PVAF == 1), so a solution always exists; ties on
        // attribute count are broken by the larger PVAF, then the smaller mask.
        let mut best_mask = full_mask;
        let mut best_pvaf = 1.0f64;
        let mut best_pc = n_attributes as u32;
        for mask in 1..l_full {
            let pv = pvaf_of(mask);
            if pv + 1e-12 >= epsilon {
                let pc = (mask as u32).count_ones();
                if pc < best_pc || (pc == best_pc && pv > best_pvaf + 1e-12) {
                    best_mask = mask;
                    best_pvaf = pv;
                    best_pc = pc;
                }
            }
        }

        for k in 0..n_attributes {
            suggested_q[i * n_attributes + k] = ((best_mask >> k) & 1) as u8;
        }
        suggested_pvaf[i] = best_pvaf;
        flagged[i] = best_mask != prov_mask;
    }

    Ok(QValidationResult {
        n_attributes,
        suggested_q,
        suggested_pvaf,
        provisional_pvaf,
        flagged,
        epsilon,
        calibration_n_iter: res.n_iter,
        calibration_max_iter: cfg.max_iter,
        calibration_termination_reason: "tolerance_met",
        calibration_final_loglik_change,
        calibration_tol: cfg.tol,
    })
}

/// Result of [`gdina_wald_selection`] (de la Torre & Lee, 2013). Per item, each
/// candidate reduced model is Wald-tested against the saturated G-DINA, and
/// `selected` names the most parsimonious model not rejected at level `alpha`.
#[derive(Clone, Debug)]
pub struct WaldSelectionResult {
    /// Candidate reduced models (`["dina", "dino", "acdm"]`); DINA and DINO cost two
    /// parameters, A-CDM costs `1 + K_i`.
    pub models: Vec<String>,
    /// Wald statistic per `(item, model)`, row-major `n_items * n_models`; `NaN`
    /// where the test is undefined (an item requiring `< 2` attributes).
    pub wald_stat: Vec<f64>,
    /// Degrees of freedom per `(item, model)`, row-major (`0` when undefined).
    pub wald_df: Vec<usize>,
    /// Upper-tail p-value per `(item, model)`, row-major (`NaN` where undefined).
    pub p_value: Vec<f64>,
    /// Selected model index into `models` per item, or `-1` for the saturated
    /// G-DINA (all reduced models rejected, or the item requires `< 2` attributes).
    pub selected: Vec<i64>,
    pub alpha: f64,
}

fn select_wald_model(df: &[usize], p_value: &[f64], alpha: f64, k: usize) -> i64 {
    let param_count = |model: usize| if model <= 1 { 2 } else { 1 + k };
    let mut best: Option<usize> = None;
    for model in 0..p_value.len() {
        if df[model] == 0 {
            continue;
        }
        let candidate_p = p_value[model];
        if candidate_p.is_finite() && candidate_p > alpha {
            best = match best {
                None => Some(model),
                Some(current) => {
                    let (current_n, candidate_n) = (param_count(current), param_count(model));
                    if candidate_n < current_n
                        || (candidate_n == current_n && candidate_p > p_value[current])
                    {
                        Some(model)
                    } else {
                        Some(current)
                    }
                }
            };
        }
    }
    best.map_or(-1, |model| model as i64)
}

/// Item-level cognitive-diagnosis model selection by the Wald test (de la Torre &
/// Lee, 2013). For each item the saturated G-DINA is compared with reduced models that
/// are exact linear restrictions of the reduced-class success probabilities `P` (the
/// `2^{K_i}` values; `M[l][S] = [S subseteq l]` the subset-sum design, see [`fit_gdina`]).
/// DINA/DINO/A-CDM restrict the identity-link parameters `delta = M^{-1} P`; LLM and
/// R-RUM are additive on the logit and log links, so they restrict the transformed
/// parameters `delta^h = M^{-1} h(P)`:
///
/// - **DINA** (purely conjunctive): only the intercept `delta_0` and the top
///   interaction `delta_{1..K}` are free; the middle `2^{K_i} - 2` coordinates are 0.
/// - **DINO** (purely disjunctive): the non-intercept coordinates are tied onto one
///   line `delta_S = (-1)^{|S|+1} Delta` (a general, non-coordinate linear
///   restriction with `df = 2^{K_i} - 2`).
/// - **A-CDM** (additive on the identity link): all interaction coordinates (`|S| >= 2`)
///   are 0, leaving the intercept and `K_i` main effects.
/// - **LLM** (linear logistic model; additive on the logit link): the interaction
///   coordinates of `delta^{logit} = M^{-1} logit(P)` are 0 (`df = 2^{K_i} - 1 - K_i`).
/// - **R-RUM** (reduced reparameterized unified model; additive on the log link): the
///   interaction coordinates of `delta^{log} = M^{-1} log(P)` are 0 (same `df`).
///
/// The Wald statistic for the restriction `R delta = 0` (`df = rank(R)`) is
/// `W = (R delta)^T (R Sigma_delta R^T)^{-1} (R delta) ~ chi^2(df)` under the reduced
/// model; for the coordinate restrictions (DINA, A-CDM, LLM, R-RUM) `R` selects
/// coordinates and `R Sigma_delta R^T` is the corresponding *block* of `Sigma_delta`,
/// while DINO uses a general sparse `R`. Under the complete-data model each
/// `P_hat_l = R_l / I_l` is a binomial proportion over the disjoint persons of reduced
/// class `l`, so `Var(P_hat) = diag(P_l (1 - P_l) / I_l)` is *exact* there (`I_l` =
/// expected count in reduced class `l`). For the identity link
/// `Sigma_delta = M^{-1} Var(P_hat) M^{-T}`; for a transformed link `h` the first-order
/// delta method gives `Var(h(P_hat_l)) = h'(P_hat_l)^2 Var(P_hat_l)`, so LLM uses
/// `h' = 1/(P(1-P))` (`Var = 1/(I_l P_l(1-P_l))`) and R-RUM `h' = 1/P`
/// (`Var = (1-P_l)/(I_l P_l)`), with the same Mobius sandwich. This estimator uses
/// complete-data (expected) rather than observed information; by the missing-information
/// principle `I_complete >= I_observed`, so `Sigma_delta` is under-estimated and the test
/// is mildly **liberal** (Type I `>=` alpha), the gap shrinking with `N` and with item
/// discrimination (small slip/guess). Per item the fewest-parameter model with
/// `p > alpha` is selected (DINA and DINO cost two parameters; A-CDM, LLM and R-RUM each
/// cost `1 + K_i`, so ties are broken by the larger p-value); if all reduced models are
/// rejected, the saturated G-DINA.
///
/// `y`/`observed` are row-major `N*J`; `q_matrix` row-major `J*K` (0/1). Deferred: the
/// incomplete-data (observed-information) covariance. A nonconverged saturated G-DINA
/// calibration is rejected rather than used to form Wald statistics from unfinished
/// parameters.
///
/// References (APA 7th ed.):
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
///     76*(2), 179–199. https://doi.org/10.1007/s11336-011-9207-7
///   de la Torre, J., & Lee, Y.-S. (2013). Evaluating the Wald test for item-level
///     comparison of saturated and reduced models in cognitive diagnosis. *Journal
///     of Educational Measurement, 50*(4), 355–373.
///     https://doi.org/10.1111/jedm.12022
///   Ma, W., Iaconangelo, C., & de la Torre, J. (2016). Model similarity, model
///     selection, and attribute classification. *Applied Psychological Measurement,
///     40*(3), 200–217. https://doi.org/10.1177/0146621615621717
#[allow(clippy::too_many_arguments)]
pub fn gdina_wald_selection(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    alpha: f64,
    cfg: &CdmConfig,
) -> Result<WaldSelectionResult, String> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err("alpha must be in (0, 1)".into());
    }
    let l_full = 1usize << n_attributes;

    // Saturated G-DINA under the given Q (also validates y/observed/q shapes+config).
    let res = fit_gdina(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    ensure_gdina_converged(&res, cfg)?;

    // Reduced-class index tables (mirror fit_gdina), then one posterior pass to
    // recover the expected reduced-class counts I_l (the Var(P_l) denominators).
    let mut qmask = vec![0usize; n_items];
    for i in 0..n_items {
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                qmask[i] |= 1 << k;
            }
        }
    }
    let total = res.item_off[n_items];
    let mut red = vec![0u16; n_items * l_full];
    for i in 0..n_items {
        for c in 0..l_full {
            red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
        }
    }
    let mut log_p1 = vec![0.0f64; total];
    let mut log_p0 = vec![0.0f64; total];
    for x in 0..total {
        let pc = res.item_prob[x].clamp(cfg.eps, 1.0 - cfg.eps);
        log_p1[x] = pc.ln();
        log_p0[x] = (1.0 - pc).ln();
    }
    let log_pi: Vec<f64> = res
        .profile_prob
        .iter()
        .map(|v| v.max(cfg.eps).ln())
        .collect();
    let mut icount = vec![0.0f64; total]; // I_l, CSR layout matching item_prob
    let mut post = vec![0.0f64; l_full];
    for j in 0..n_persons {
        posterior_row_gdina(
            j,
            y,
            observed,
            n_items,
            l_full,
            &red,
            &log_p1,
            &log_p0,
            &res.item_off,
            &log_pi,
            &mut post,
        );
        for i in 0..n_items {
            let idx = j * n_items + i;
            if observed[idx] {
                for c in 0..l_full {
                    icount[res.item_off[i] + red[i * l_full + c] as usize] += post[c];
                }
            }
        }
    }

    let models = vec![
        "dina".to_string(),
        "dino".to_string(),
        "acdm".to_string(),
        "llm".to_string(),
        "rrum".to_string(),
    ];
    let n_models = models.len();
    let mut wald_stat = vec![f64::NAN; n_items * n_models];
    let mut wald_df = vec![0usize; n_items * n_models];
    let mut p_value = vec![f64::NAN; n_items * n_models];
    let mut selected = vec![-1i64; n_items];

    for i in 0..n_items {
        let k = res.k_required[i] as usize;
        if k < 2 {
            continue; // no interactions: DINA = A-CDM = saturated, nothing to test
        }
        let off = res.item_off[i];
        let w = res.item_off[i + 1] - off; // 2^k
        let p = &res.item_prob[off..off + w];
        let delta = &res.item_delta[off..off + w];
        let ic = &icount[off..off + w];

        // Sigma_delta = sum_l v_l c_l c_l^T, c_l = M^{-1} e_l (Mobius applied to the
        // l-th unit vector = column l of M^{-1}). The per-class variance v_l depends on
        // the link on which the reduced model is additive. On the identity link
        // (DINA/DINO/A-CDM) v_l = Var(P_l) = P_l(1-P_l)/I_l. On a transformed link h the
        // delta method gives Var(h(P_l)) = h'(P_l)^2 Var(P_l): LLM uses the logit
        // (h' = 1/(P(1-P)) -> v_l = 1/(I_l P_l(1-P_l))) and R-RUM the log
        // (h' = 1/P -> v_l = (1-P_l)/(I_l P_l)). The Mobius columns c_l are shared across
        // links, so all three covariances accumulate in one pass. An empty reduced class
        // (I_l ~ 0) is floored so its variance is finite-but-huge, making any delta
        // touching it effectively untestable (conservative) rather than NaN.
        let mut sigma = vec![vec![0.0f64; w]; w]; // identity link (DINA/DINO/A-CDM)
        let mut sigma_logit = vec![vec![0.0f64; w]; w]; // logit link (LLM)
        let mut sigma_log = vec![vec![0.0f64; w]; w]; // log link (R-RUM)
        for l in 0..w {
            // Floor at count_floor (an empty class -> huge, conservative variance)
            // and additionally at a strictly positive constant so a wholly-unobserved
            // item under a `count_floor == 0` config cannot divide by zero.
            let denom = ic[l].max(cfg.count_floor).max(1e-12);
            let pl = p[l].clamp(cfg.eps, 1.0 - cfg.eps); // guard the logit/log transforms
            let base = pl * (1.0 - pl); // P_l(1-P_l) > 0 under the clamp
            let v = base / denom; // identity-link Var(P_l), matches the reduced-model baseline
            debug_assert!(
                v > 0.0,
                "clamped probabilities and positive counts imply variance"
            );
            let v_logit = 1.0 / (denom * base); // (1/base)^2 * base/denom
            let v_log = (1.0 - pl) / (denom * pl); // (1/P_l)^2 * base/denom
            let mut c = vec![0.0f64; w];
            c[l] = 1.0;
            mobius_inverse_inplace(&mut c, k as u32);
            for a in 0..w {
                let ca = c[a];
                if ca == 0.0 {
                    continue;
                }
                let (vca, vca_logit, vca_log) = (v * ca, v_logit * ca, v_log * ca);
                for b in 0..w {
                    let cb = c[b];
                    sigma[a][b] += vca * cb;
                    sigma_logit[a][b] += vca_logit * cb;
                    sigma_log[a][b] += vca_log * cb;
                }
            }
        }
        // Link-transformed deltas delta^h = M^{-1} h(P) restricted by LLM (logit) and
        // R-RUM (log); the identity-link `delta` above serves DINA/DINO/A-CDM.
        let mut delta_logit = vec![0.0f64; w];
        let mut delta_log = vec![0.0f64; w];
        for l in 0..w {
            let pl = p[l].clamp(cfg.eps, 1.0 - cfg.eps);
            delta_logit[l] = (pl / (1.0 - pl)).ln();
            delta_log[l] = pl.ln();
        }
        mobius_inverse_inplace(&mut delta_logit, k as u32);
        mobius_inverse_inplace(&mut delta_log, k as u32);

        let full = w - 1;
        // Restriction rows in the subset-index layout: each row is a sparse linear
        // combination sum_j coeff_j * delta_{S_j} that a reduced model sets to zero.
        // DINA and A-CDM are coordinate restrictions (single unit entry per row); DINO
        // is a general restriction that ties the non-intercept deltas onto one line
        // delta_S = (-1)^{|S|+1} * Delta (reference coordinate s=1, so Delta = delta_1).
        let restriction_rows = |model: usize| -> Vec<Vec<(usize, f64)>> {
            match model {
                // DINA: intercept and top interaction free, middle coordinates zero.
                0 => (0..w)
                    .filter(|&s| s != 0 && s != full)
                    .map(|s| vec![(s, 1.0)])
                    .collect(),
                // DINO: delta_S - (-1)^{|S|+1} delta_1 = 0 for every S != {empty, ref=1}.
                1 => (0..w)
                    .filter(|&s| s != 0 && s != 1)
                    .map(|s| {
                        let sign = if (s as u32).count_ones() % 2 == 1 {
                            1.0
                        } else {
                            -1.0
                        };
                        vec![(s, 1.0), (1usize, -sign)]
                    })
                    .collect(),
                // A-CDM / LLM / R-RUM: all interaction coordinates zero. The three share
                // this restriction pattern but on different links (identity/logit/log),
                // so they differ only in which (delta, Sigma) pair the caller feeds in.
                _ => (0..w)
                    .filter(|&s| (s as u32).count_ones() >= 2)
                    .map(|s| vec![(s, 1.0)])
                    .collect(),
            }
        };

        for m in 0..n_models {
            let rows = restriction_rows(m);
            let df = rows.len();
            wald_df[i * n_models + m] = df;
            debug_assert!(
                df > 0,
                "items with at least two attributes have restrictions"
            );
            // DINA/DINO/A-CDM restrict the identity-link delta; LLM restricts the
            // logit-link delta and R-RUM the log-link delta, each with the matching
            // delta-method covariance.
            let (dvec, svec): (&[f64], &[Vec<f64>]) = match m {
                3 => (&delta_logit, &sigma_logit),
                4 => (&delta_log, &sigma_log),
                _ => (delta, &sigma),
            };
            // R*delta and R*Sigma*R^T; relative ridge for a well-posed solve.
            let mut rd = vec![0.0f64; df];
            let mut sr = vec![vec![0.0f64; df]; df];
            for a in 0..df {
                for &(ca, va) in &rows[a] {
                    rd[a] += va * dvec[ca];
                }
            }
            for a in 0..df {
                for b in 0..df {
                    let mut acc = 0.0f64;
                    for &(ca, va) in &rows[a] {
                        for &(cb, vb) in &rows[b] {
                            acc += va * vb * svec[ca][cb];
                        }
                    }
                    sr[a][b] = acc;
                }
            }
            let diag_sum: f64 = (0..df).map(|a| sr[a][a]).sum();
            let ridge = 1e-9 * (diag_sum / df as f64).max(1e-300);
            for a in 0..df {
                sr[a][a] += ridge;
            }
            // W = (R delta)^T (R Sigma R^T)^{-1} (R delta): solve (R Sigma R^T) x = R delta.
            let x = crate::poly::solve_small(sr, rd.clone());
            let wstat = (0..df).map(|a| rd[a] * x[a]).sum::<f64>().max(0.0);
            wald_stat[i * n_models + m] = wstat;
            p_value[i * n_models + m] = crate::fitstats::chi2_sf(wstat, df as f64);
        }

        // Fewest-parameter reduced model not rejected (DINA=2, DINO=2, A-CDM=1+K);
        // ties (DINA vs DINO) broken by the larger p-value; else the saturated G-DINA.
        selected[i] = select_wald_model(
            &wald_df[i * n_models..(i + 1) * n_models],
            &p_value[i * n_models..(i + 1) * n_models],
            alpha,
            k,
        );
    }

    Ok(WaldSelectionResult {
        models,
        wald_stat,
        wald_df,
        p_value,
        selected,
        alpha,
    })
}

/// Mild Gaussian ridge on the higher-order attribute parameters, mirroring
/// `fit_mmle_2pl`'s ridge so the per-attribute Newton stays well-posed.
const HO_RIDGE: f64 = 1e-3;

/// Result of [`fit_ho_cdm`] (de la Torre & Douglas, 2004). The `2^K` class
/// distribution is not free: it is generated by the higher-order trait through
/// `attr_slope`/`attr_intercept`, and `profile_prob` is the *implied* marginal.
#[derive(Clone, Debug)]
pub struct HoCdmResult {
    pub model: CdmModel,
    /// Per-item slip `s_i` and guess `g_i`.
    pub slip: Vec<f64>,
    pub guess: Vec<f64>,
    /// Higher-order attribute slope `a_k` (discrimination on the trait), length `K`.
    pub attr_slope: Vec<f64>,
    /// Higher-order attribute intercept `d_k` (easiness), length `K`.
    pub attr_intercept: Vec<f64>,
    /// Implied marginal class probabilities `pi_c` (length `2^K`, sum 1).
    pub profile_prob: Vec<f64>,
    /// Per-person EAP higher-order trait score.
    pub theta: Vec<f64>,
    /// Bit-encoded MAP profile per person.
    pub map_profile: Vec<u32>,
    /// Marginal `P(alpha_jk = 1 | X_j)`, row-major `N x K`.
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Configured outer EM iteration limit.
    pub max_iter: usize,
    /// Stable public reason for termination: `tolerance_met` or `max_iter_reached`.
    pub termination_reason: &'static str,
    /// Last observed-data log-likelihood increment at the returned parameters.
    pub final_loglik_change: f64,
    /// Requested absolute log-likelihood-change stopping tolerance.
    pub stopping_tolerance: f64,
    /// `2*J + 2*K`.
    pub n_parameters: usize,
}

/// Marginal class probabilities `pi_c = integral P(alpha_c | theta) phi(theta) dtheta`
/// implied by the higher-order parameters, on the 41-node Gauss-Hermite grid. With
/// every slope zero this is exactly the independent-attribute Bernoulli product
/// `prod_k sigmoid(d_k)^{alpha_ck} (1 - sigmoid(d_k))^{1 - alpha_ck}` (theta drops out).
fn ho_pi_from_params(attr_slope: &[f64], attr_intercept: &[f64], n_attributes: usize) -> Vec<f64> {
    use crate::mmle::{log_sigmoid, GH_NODES, GH_WEIGHTS};
    let l = 1usize << n_attributes;
    let q = GH_NODES.len();
    let mut logp = vec![0.0f64; n_attributes * q];
    let mut log1mp = vec![0.0f64; n_attributes * q];
    for k in 0..n_attributes {
        for (qi, &node) in GH_NODES.iter().enumerate() {
            let z = attr_slope[k] * node + attr_intercept[k];
            logp[k * q + qi] = log_sigmoid(z);
            log1mp[k * q + qi] = log_sigmoid(-z);
        }
    }
    let mut pi = vec![0.0f64; l];
    for (c, pic) in pi.iter_mut().enumerate() {
        let mut acc = 0.0f64;
        for (qi, &w) in GH_WEIGHTS.iter().enumerate() {
            let mut lp = 0.0f64;
            for k in 0..n_attributes {
                lp += if (c >> k) & 1 == 1 {
                    logp[k * q + qi]
                } else {
                    log1mp[k * q + qi]
                };
            }
            acc += w * lp.exp();
        }
        *pic = acc;
    }
    pi
}

/// Newton step for one attribute's higher-order 2PL `sigmoid(a*theta + d)` from the
/// expected node counts `r[q]` (masters) and `w[q]` (total) at the Gauss-Hermite
/// nodes. Arithmetically identical to `fit_mmle_2pl`'s inner `(a, b)` Newton.
fn newton_attr_2pl(mut a: f64, mut d: f64, r: &[f64], w: &[f64], newton_iter: usize) -> (f64, f64) {
    use crate::mmle::{log_sigmoid, sigmoid_stable, GH_NODES};
    let q_value = |aa: f64, dd: f64| -> f64 {
        GH_NODES
            .iter()
            .enumerate()
            .map(|(qi, &node)| {
                let z = aa * node + dd;
                r[qi] * log_sigmoid(z) + (w[qi] - r[qi]) * log_sigmoid(-z)
            })
            .sum()
    };
    for _ in 0..newton_iter {
        let (mut g_a, mut g_d, mut h_aa, mut h_dd, mut h_ad) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (qi, &node) in GH_NODES.iter().enumerate() {
            let p = sigmoid_stable(a * node + d);
            let ww = w[qi] * p * (1.0 - p);
            let resid = r[qi] - w[qi] * p;
            g_a += resid * node;
            g_d += resid;
            h_aa -= ww * node * node;
            h_dd -= ww;
            h_ad -= ww * node;
        }
        g_a -= HO_RIDGE * a;
        g_d -= HO_RIDGE * d;
        h_aa -= HO_RIDGE;
        h_dd -= HO_RIDGE;
        let det = h_aa * h_dd - h_ad * h_ad;
        debug_assert!(
            det >= HO_RIDGE * HO_RIDGE,
            "the positive ridge keeps the information matrix nonsingular"
        );
        let da = (h_dd * g_a - h_ad * g_d) / det;
        let dd = (h_aa * g_d - h_ad * g_a) / det;
        let (old_a, old_d) = (a, d);
        let old_q = q_value(old_a, old_d);
        let mut step = 1.0f64;
        let mut accepted = false;
        // Near-separated expected counts can make an undamped Newton step overshoot.
        // Backtrack on the unpenalized EM auxiliary function so the numerical ridge
        // cannot make the reported marginal log-likelihood move backwards.
        for _ in 0..30 {
            // A zero loading is the independent-attribute boundary of the
            // higher-order model. Keep the identification anchor non-negative,
            // but do not exclude that valid boundary with an arbitrary epsilon.
            let cand_a = (old_a - step * da).clamp(0.0, 10.0);
            let cand_d = old_d - step * dd;
            let cand_q = q_value(cand_a, cand_d);
            if cand_q.is_finite() && cand_q >= old_q - 1e-12 {
                a = cand_a;
                d = cand_d;
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if !accepted {
            break;
        }
        if (a - old_a).abs() + (d - old_d).abs() < 1e-8 {
            break;
        }
    }
    (a, d)
}

/// Fit the higher-order DINA/DINO model of de la Torre and Douglas (2004) using this
/// crate's marginal EM over the joint `(alpha_c, theta_q)` grid. The source paper
/// estimates the model by Bayesian MCMC; quadrature EM is the implementation choice
/// here, not an algorithm claimed by that paper. A continuous higher-order trait
/// `theta ~ N(0,1)` structures attribute mastery,
/// `P(alpha_k = 1 | theta) = sigmoid(a_k theta + d_k)` with attributes conditionally
/// independent given `theta`, so the `2^K` class distribution is a `2K`-parameter
/// structured family rather than the free distribution of [`fit_cdm`]. The item part
/// (slip/guess DINA or DINO gate) is unchanged; the population update is replaced by
/// `K` independent 2PL calibrations of attribute mastery on `theta`.
///
/// `y`/`observed` are row-major `N*J` (`y` in {0,1}); `q_matrix` is row-major `J*K`.
/// Missing cells (MAR) are dropped. With every `a_k = 0` the class prior reduces to
/// independent attributes; the trait `theta` fixes its own scale via the `N(0,1)`
/// prior. Reuses `mmle::GH_NODES/GH_WEIGHTS` (41-node) and the DINA item M-step.
///
/// The observed-data likelihood depends on `(a_k, d_k)` only through the implied
/// class distribution `pi_c`, so the higher-order parameters are a genuine
/// restriction (and identified) only for `K >= 3` (`2K` structural parameters vs a
/// `2^K - 1`-dimensional simplex); at `K <= 2` they are over-parameterized and only
/// `pi_c` (and the attribute classification) is identified. `attr_slope` is anchored
/// non-negative (`a_k >= 1e-3`), the standard orientation that the trait raises every
/// attribute's mastery.
///
/// References (APA 7th ed.):
///   de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models for
///     cognitive diagnosis. *Psychometrika, 69*(3), 333-353.
///     <https://doi.org/10.1007/BF02295640>
#[allow(clippy::too_many_arguments)]
pub fn fit_ho_cdm(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    model: CdmModel,
    cfg: &CdmConfig,
) -> Result<HoCdmResult, String> {
    use crate::mmle::{log_sigmoid, GH_NODES, GH_WEIGHTS};
    validate(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    let l = 1usize << n_attributes;
    let q = GH_NODES.len();
    let log_w: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();

    // Ideal-response gate (same as fit_cdm).
    let mut qmask = vec![0usize; n_items];
    for i in 0..n_items {
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                qmask[i] |= 1 << k;
            }
        }
    }
    let mut eta = vec![0u8; n_items * l];
    for i in 0..n_items {
        for c in 0..l {
            eta[i * l + c] = match model {
                CdmModel::Dina => ((c & qmask[i]) == qmask[i]) as u8,
                CdmModel::Dino => ((c & qmask[i]) != 0) as u8,
            };
        }
    }

    let mut s = vec![cfg.init_slip; n_items];
    let mut g = vec![cfg.init_guess; n_items];
    let mut a = vec![1.0f64; n_attributes];
    let mut d = vec![0.0f64; n_attributes];

    let mut lp1 = vec![0.0f64; n_items * 2];
    let mut lp0 = vec![0.0f64; n_items * 2];
    let refresh = |s: &[f64], g: &[f64], lp1: &mut [f64], lp0: &mut [f64]| {
        for i in 0..n_items {
            let sc = s[i].clamp(cfg.eps, 1.0 - cfg.eps);
            let gc = g[i].clamp(cfg.eps, 1.0 - cfg.eps);
            lp1[i * 2 + 1] = (1.0 - sc).ln();
            lp0[i * 2 + 1] = sc.ln();
            lp1[i * 2] = gc.ln();
            lp0[i * 2] = (1.0 - gc).ln();
        }
    };

    // Build the structural class-log-prior table logPalpha[c*q + qi] for the current
    // (a, d), plus the marginal logL_j(c) for one person, then the joint posterior.
    let structural_table = |a: &[f64], d: &[f64]| -> Vec<f64> {
        let mut logp = vec![0.0f64; n_attributes * q];
        let mut log1mp = vec![0.0f64; n_attributes * q];
        for k in 0..n_attributes {
            for (qi, &node) in GH_NODES.iter().enumerate() {
                let z = a[k] * node + d[k];
                logp[k * q + qi] = log_sigmoid(z);
                log1mp[k * q + qi] = log_sigmoid(-z);
            }
        }
        let mut logpa = vec![0.0f64; l * q];
        for c in 0..l {
            for qi in 0..q {
                let mut lp = 0.0f64;
                for k in 0..n_attributes {
                    lp += if (c >> k) & 1 == 1 {
                        logp[k * q + qi]
                    } else {
                        log1mp[k * q + qi]
                    };
                }
                logpa[c * q + qi] = lp;
            }
        }
        logpa
    };

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut post = vec![0.0f64; l * q]; // reused joint-posterior scratch

    for _ in 0..cfg.max_iter {
        refresh(&s, &g, &mut lp1, &mut lp0);
        let logpa = structural_table(&a, &d);

        // E-step over the joint (c, q) grid.
        let mut i1 = vec![0.0f64; n_items];
        let mut r1 = vec![0.0f64; n_items];
        let mut i0 = vec![0.0f64; n_items];
        let mut r0 = vec![0.0f64; n_items];
        let mut wq = vec![0.0f64; q]; // node mass W_q
        let mut rkq = vec![0.0f64; n_attributes * q]; // masters per (k, q)
        let mut total_ll = 0.0;
        for j in 0..n_persons {
            // logL_j(c) then joint = logL + logPalpha + log_w.
            for c in 0..l {
                let mut ll = 0.0f64;
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let b = eta[i * l + c] as usize;
                        let yy = y[idx];
                        ll += yy * lp1[i * 2 + b] + (1.0 - yy) * lp0[i * 2 + b];
                    }
                }
                for qi in 0..q {
                    post[c * q + qi] = ll + logpa[c * q + qi] + log_w[qi];
                }
            }
            let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in post.iter() {
                denom += (v - mx).exp();
            }
            total_ll += mx + denom.ln();
            for v in post.iter_mut() {
                *v = (*v - mx).exp() / denom;
            }
            // marginal class posterior -> item counts (same as fit_cdm)
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let mut pbar = 0.0f64;
                    for c in 0..l {
                        if eta[i * l + c] == 1 {
                            // sum over q of post(c,q) restricted to masters of item i
                            let base = c * q;
                            for v in &post[base..base + q] {
                                pbar += v;
                            }
                        }
                    }
                    let yy = y[idx];
                    i1[i] += pbar;
                    r1[i] += yy * pbar;
                    i0[i] += 1.0 - pbar;
                    r0[i] += yy * (1.0 - pbar);
                }
            }
            // node mass + structural masters
            for qi in 0..q {
                let mut wnode = 0.0f64;
                for c in 0..l {
                    let p = post[c * q + qi];
                    wnode += p;
                    let mut cc = c;
                    let mut k = 0;
                    while cc != 0 {
                        if cc & 1 == 1 {
                            rkq[k * q + qi] += p;
                        }
                        cc >>= 1;
                        k += 1;
                    }
                }
                wq[qi] += wnode;
            }
        }
        loglik_trace.push(total_ll);

        // Converge check before the M-step so returned params match the trace endpoint.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: items (slip/guess), then structure (per-attribute 2PL).
        for i in 0..n_items {
            update_item(i, &i1, &r1, &i0, &r0, &mut s, &mut g, cfg);
        }
        for k in 0..n_attributes {
            // 25 inner Newton steps (mirrors fit_mmle_2pl's default newton_iter).
            let (ak, dk) = newton_attr_2pl(a[k], d[k], &rkq[k * q..(k + 1) * q], &wq, 25);
            a[k] = ak;
            d[k] = dk;
        }
        n_iter += 1;
    }

    // Final classification / theta pass at the converged parameters.
    refresh(&s, &g, &mut lp1, &mut lp0);
    let logpa = structural_table(&a, &d);
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0;
    for j in 0..n_persons {
        for c in 0..l {
            let mut ll = 0.0f64;
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let b = eta[i * l + c] as usize;
                    let yy = y[idx];
                    ll += yy * lp1[i * 2 + b] + (1.0 - yy) * lp0[i * 2 + b];
                }
            }
            for qi in 0..q {
                post[c * q + qi] = ll + logpa[c * q + qi] + log_w[qi];
            }
        }
        let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for v in post.iter() {
            denom += (v - mx).exp();
        }
        final_ll += mx + denom.ln();
        for v in post.iter_mut() {
            *v = (*v - mx).exp() / denom;
        }
        // MAP profile (over marginal class posterior), attribute marginals, theta EAP.
        let (mut best, mut best_p) = (0usize, f64::NEG_INFINITY);
        for c in 0..l {
            let mut pc = 0.0f64;
            for v in &post[c * q..c * q + q] {
                pc += v;
            }
            if pc > best_p {
                best_p = pc;
                best = c;
            }
            for k in 0..n_attributes {
                if (c >> k) & 1 == 1 {
                    attr_prob[j * n_attributes + k] += pc;
                }
            }
        }
        map_profile[j] = best as u32;
        for qi in 0..q {
            let mut wnode = 0.0f64;
            for c in 0..l {
                wnode += post[c * q + qi];
            }
            theta[j] += wnode * GH_NODES[qi];
        }
    }
    if !converged {
        loglik_trace.push(final_ll);
    }
    let final_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| pair[1] - pair[0])
        .unwrap_or(f64::NAN);
    let termination_reason = if converged {
        "tolerance_met"
    } else {
        "max_iter_reached"
    };

    let profile_prob = ho_pi_from_params(&a, &d, n_attributes);
    Ok(HoCdmResult {
        model,
        slip: s,
        guess: g,
        attr_slope: a,
        attr_intercept: d,
        profile_prob,
        theta,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        max_iter: cfg.max_iter,
        termination_reason,
        final_loglik_change,
        stopping_tolerance: cfg.tol,
        n_parameters: 2 * n_items + 2 * n_attributes,
    })
}

/// Result of [`fit_ho_gdina`] (higher-order G-DINA). Combines the saturated per-item
/// reduced-class probabilities of [`fit_gdina`] (CSR-laid-out) with the higher-order
/// structural attribute parameters of [`fit_ho_cdm`].
#[derive(Clone, Debug)]
pub struct HoGdinaResult {
    pub item_off: Vec<usize>,
    pub item_prob: Vec<f64>,
    pub item_delta: Vec<f64>,
    pub k_required: Vec<u32>,
    /// Higher-order attribute slope `a_k` and intercept `d_k`, length `K`.
    pub attr_slope: Vec<f64>,
    pub attr_intercept: Vec<f64>,
    /// Implied marginal class probabilities `pi_c` (length `2^K`).
    pub profile_prob: Vec<f64>,
    pub theta: Vec<f64>,
    pub map_profile: Vec<u32>,
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Stable public reason for termination: `tolerance_met` or `max_iter_reached`.
    pub termination_reason: &'static str,
    /// Last observed-data log-likelihood increment at the returned parameters.
    pub final_loglik_change: f64,
    /// Last scale-free increment `|delta log L| / (1 + |log L_previous|)`.
    pub final_relative_loglik_change: f64,
    /// Requested relative log-likelihood stopping tolerance.
    pub stopping_tolerance: f64,
    /// `sum_i 2^{K_i} + 2*K`.
    pub n_parameters: usize,
}

/// Fit the higher-order G-DINA model by marginal-ML EM over the joint
/// `(alpha_c, theta_q)` grid: the saturated G-DINA item model of [`fit_gdina`] (each
/// reduced attribute-mastery class of each item gets a free success probability)
/// under the higher-order structural attribute prior of [`fit_ho_cdm`] (a continuous
/// trait `theta ~ N(0,1)` drives mastery, `P(alpha_k=1|theta) = sigmoid(a_k theta +
/// d_k)`, attributes conditionally independent given `theta`). This is the
/// higher-order form of the general G-DINA framework: it generalizes [`fit_ho_cdm`]
/// (which restricts the item model to DINA slip/guess) and constrains [`fit_gdina`]'s
/// free class distribution to the `2K`-parameter structured family.
///
/// The item response is conditionally independent of `theta` given `alpha`, so the
/// saturated item M-step `p_il = R_il / I_il` marginalizes `theta` out (it uses only
/// the marginal class posterior), exactly as [`fit_gdina`]; the structural M-step is
/// `K` independent 2PL calibrations of attribute mastery on `theta`, exactly as
/// [`fit_ho_cdm`]. Identification mirrors [`fit_ho_cdm`] (`theta ~ N(0,1)` fixes the
/// scale, `a_k` anchored non-negative, higher-order parameters identified for
/// `K >= 3`) and [`fit_gdina`] (the Q-matrix must identify the saturated item probs).
/// This implementation stops on the scale-free observed-data likelihood change
/// `|delta log L| / (1 + |log L_previous|) < tol`; this numerical rule is a package
/// choice rather than a claim from the cited Bayesian source estimator.
///
/// References (APA 7th ed.):
///   de la Torre, J., & Douglas, J. A. (2004). Higher-order latent trait models for
///     cognitive diagnosis. *Psychometrika, 69*(3), 333-353.
///     https://doi.org/10.1007/BF02295640
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
///     76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
#[allow(clippy::too_many_arguments)]
pub fn fit_ho_gdina(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<HoGdinaResult, String> {
    use crate::mmle::{log_sigmoid, GH_NODES, GH_WEIGHTS};
    validate(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    if n_attributes < 3 {
        return Err("higher-order G-DINA requires at least 3 attributes for identified structural parameters".into());
    }
    let l = 1usize << n_attributes;
    let q = GH_NODES.len();
    let log_w: Vec<f64> = GH_WEIGHTS.iter().map(|w| w.ln()).collect();

    // Saturated-item CSR layout (verbatim from fit_gdina).
    let mut qmask = vec![0usize; n_items];
    let mut k_required = vec![0u32; n_items];
    for i in 0..n_items {
        let mut mask = 0usize;
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                mask |= 1 << k;
            }
        }
        qmask[i] = mask;
        k_required[i] = mask.count_ones();
    }
    let mut item_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        item_off[i + 1] = item_off[i] + (1usize << k_required[i]);
    }
    let total = item_off[n_items];
    let mut red = vec![0u16; n_items * l];
    for i in 0..n_items {
        for c in 0..l {
            red[i * l + c] = reduce_class(c, qmask[i]) as u16;
        }
    }
    let mut p = vec![0.0f64; total];
    for i in 0..n_items {
        let ki = k_required[i] as f64;
        for li in 0..(item_off[i + 1] - item_off[i]) {
            let frac = (li.count_ones() as f64) / ki;
            p[item_off[i] + li] = cfg.init_guess + (1.0 - cfg.init_slip - cfg.init_guess) * frac;
        }
    }

    // Higher-order structural parameters (verbatim from fit_ho_cdm).
    let mut a = vec![1.0f64; n_attributes];
    let mut d = vec![0.0f64; n_attributes];
    let structural_table = |a: &[f64], d: &[f64]| -> Vec<f64> {
        let mut logp = vec![0.0f64; n_attributes * q];
        let mut log1mp = vec![0.0f64; n_attributes * q];
        for k in 0..n_attributes {
            for (qi, &node) in GH_NODES.iter().enumerate() {
                let z = a[k] * node + d[k];
                logp[k * q + qi] = log_sigmoid(z);
                log1mp[k * q + qi] = log_sigmoid(-z);
            }
        }
        let mut logpa = vec![0.0f64; l * q];
        for c in 0..l {
            for qi in 0..q {
                let mut lp = 0.0f64;
                for k in 0..n_attributes {
                    lp += if (c >> k) & 1 == 1 {
                        logp[k * q + qi]
                    } else {
                        log1mp[k * q + qi]
                    };
                }
                logpa[c * q + qi] = lp;
            }
        }
        logpa
    };

    let mut log_p1 = vec![0.0f64; total];
    let mut log_p0 = vec![0.0f64; total];
    let refresh_p = |p: &[f64], log_p1: &mut [f64], log_p0: &mut [f64]| {
        for x in 0..total {
            let pc = p[x].clamp(cfg.eps, 1.0 - cfg.eps);
            log_p1[x] = pc.ln();
            log_p0[x] = (1.0 - pc).ln();
        }
    };

    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;
    let mut post = vec![0.0f64; l * q];
    let mut pc = vec![0.0f64; l]; // marginal class posterior scratch

    for _ in 0..cfg.max_iter {
        refresh_p(&p, &mut log_p1, &mut log_p0);
        let logpa = structural_table(&a, &d);

        let mut ii = vec![0.0f64; total];
        let mut rr = vec![0.0f64; total];
        let mut wq = vec![0.0f64; q];
        let mut rkq = vec![0.0f64; n_attributes * q];
        let mut total_ll = 0.0;
        for j in 0..n_persons {
            for c in 0..l {
                let mut ll = 0.0f64;
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let cell = item_off[i] + red[i * l + c] as usize;
                        let yy = y[idx];
                        ll += yy * log_p1[cell] + (1.0 - yy) * log_p0[cell];
                    }
                }
                for qi in 0..q {
                    post[c * q + qi] = ll + logpa[c * q + qi] + log_w[qi];
                }
            }
            let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in post.iter() {
                denom += (v - mx).exp();
            }
            total_ll += mx + denom.ln();
            for v in post.iter_mut() {
                *v = (*v - mx).exp() / denom;
            }
            // marginal class posterior (theta integrated out) -> saturated item counts
            for c in 0..l {
                let mut s = 0.0f64;
                for v in &post[c * q..c * q + q] {
                    s += v;
                }
                pc[c] = s;
            }
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let yy = y[idx];
                    for c in 0..l {
                        let cell = item_off[i] + red[i * l + c] as usize;
                        ii[cell] += pc[c];
                        rr[cell] += yy * pc[c];
                    }
                }
            }
            // structural node mass + masters (same as fit_ho_cdm)
            for qi in 0..q {
                let mut wnode = 0.0f64;
                for c in 0..l {
                    let pv = post[c * q + qi];
                    wnode += pv;
                    let mut cc = c;
                    let mut k = 0;
                    while cc != 0 {
                        if cc & 1 == 1 {
                            rkq[k * q + qi] += pv;
                        }
                        cc >>= 1;
                        k += 1;
                    }
                }
                wq[qi] += wnode;
            }
        }
        loglik_trace.push(total_ll);

        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            let delta = loglik_trace[n - 1] - loglik_trace[n - 2];
            let relative_delta = delta.abs() / (1.0 + loglik_trace[n - 2].abs());
            if relative_delta < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: saturated item probs (closed form) then per-attribute 2PL Newton.
        for x in 0..total {
            if ii[x] > cfg.count_floor {
                p[x] = (rr[x] / ii[x]).clamp(cfg.eps, 1.0 - cfg.eps);
            }
        }
        for k in 0..n_attributes {
            let (ak, dk) = newton_attr_2pl(a[k], d[k], &rkq[k * q..(k + 1) * q], &wq, 25);
            a[k] = ak;
            d[k] = dk;
        }
        n_iter += 1;
    }

    // Final classification / theta pass at the returned parameters.
    refresh_p(&p, &mut log_p1, &mut log_p0);
    let logpa = structural_table(&a, &d);
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut theta = vec![0.0f64; n_persons];
    let mut final_ll = 0.0;
    for j in 0..n_persons {
        for c in 0..l {
            let mut ll = 0.0f64;
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let cell = item_off[i] + red[i * l + c] as usize;
                    let yy = y[idx];
                    ll += yy * log_p1[cell] + (1.0 - yy) * log_p0[cell];
                }
            }
            for qi in 0..q {
                post[c * q + qi] = ll + logpa[c * q + qi] + log_w[qi];
            }
        }
        let mx = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for v in post.iter() {
            denom += (v - mx).exp();
        }
        final_ll += mx + denom.ln();
        for v in post.iter_mut() {
            *v = (*v - mx).exp() / denom;
        }
        let (mut best, mut best_p) = (0usize, f64::NEG_INFINITY);
        for c in 0..l {
            let mut cpost = 0.0f64;
            for v in &post[c * q..c * q + q] {
                cpost += v;
            }
            if cpost > best_p {
                best_p = cpost;
                best = c;
            }
            for k in 0..n_attributes {
                if (c >> k) & 1 == 1 {
                    attr_prob[j * n_attributes + k] += cpost;
                }
            }
        }
        map_profile[j] = best as u32;
        for qi in 0..q {
            let mut wnode = 0.0f64;
            for c in 0..l {
                wnode += post[c * q + qi];
            }
            theta[j] += wnode * GH_NODES[qi];
        }
    }
    if !converged {
        loglik_trace.push(final_ll);
    }
    let final_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| pair[1] - pair[0])
        .unwrap_or(f64::NAN);
    let final_relative_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| (pair[1] - pair[0]).abs() / (1.0 + pair[0].abs()))
        .unwrap_or(f64::NAN);
    let termination_reason = if converged {
        "tolerance_met"
    } else {
        "max_iter_reached"
    };

    // Identity-link parameters delta = M^{-1} p, per item slice.
    let mut item_delta = p.clone();
    for i in 0..n_items {
        mobius_inverse_inplace(&mut item_delta[item_off[i]..item_off[i + 1]], k_required[i]);
    }
    let profile_prob = ho_pi_from_params(&a, &d, n_attributes);

    Ok(HoGdinaResult {
        item_off,
        item_prob: p,
        item_delta,
        k_required,
        attr_slope: a,
        attr_intercept: d,
        profile_prob,
        theta,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        final_relative_loglik_change,
        stopping_tolerance: cfg.tol,
        n_parameters: total + 2 * n_attributes,
    })
}

/// Result of [`fit_seq_gdina`] (Ma & de la Torre, 2016): the shared-Q sequential
/// (continuation-ratio) G-DINA for ordered polytomous responses.
///
/// Ragged, CLASS-MAJOR CSR. Item `i` has `M_i = max_cat[i]` ordered steps over
/// `2^{K_i}` reduced attribute classes:
///   `step_prob[s_off[i] + l * M_i + (k-1)] = s_ik(l) = P(X_i >= k | X_i >= k-1, class l)`
/// for step `k in 1..=M_i` and reduced class `l`; the implied category probabilities are
///   `cat_prob[cat_off[i] + l * (M_i + 1) + x] = P(X_i = x | class l)` for `x in 0..=M_i`.
#[derive(Clone, Debug)]
pub struct SeqGdinaResult {
    /// Per-item step-prob block offsets into `step_prob` (length `n_items + 1`).
    pub s_off: Vec<usize>,
    /// Continuation probabilities `s_ik(l)`, class-major ragged (see struct doc).
    pub step_prob: Vec<f64>,
    /// Per-item category-prob block offsets into `cat_prob` (length `n_items + 1`).
    pub cat_off: Vec<usize>,
    /// Implied category probabilities `P(X_i = x | class l)`, class-major ragged.
    pub cat_prob: Vec<f64>,
    /// Maximum observed category `M_i` (number of ordered steps) per item.
    pub max_cat: Vec<u32>,
    /// Required-attribute count `K_i` per item.
    pub k_required: Vec<u32>,
    /// Free profile distribution `pi_c` (length `2^K`, sums to 1).
    pub profile_prob: Vec<f64>,
    /// Bit-encoded MAP profile per person.
    pub map_profile: Vec<u32>,
    /// Marginal `P(alpha_jk = 1 | X_j)`, row-major `N x K`.
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    /// Stable public reason for termination: `tolerance_met` or `max_iter_reached`.
    pub termination_reason: &'static str,
    /// Last observed-data log-likelihood increment at the returned parameters.
    pub final_loglik_change: f64,
    /// Last scale-free increment `|delta log L| / (1 + |log L_previous|)`.
    pub final_relative_loglik_change: f64,
    /// Requested absolute log-likelihood stopping tolerance.
    pub stopping_tolerance: f64,
    /// `sum_i M_i * 2^{K_i}` step probs `+ (2^K - 1)` free profile probs.
    pub n_parameters: usize,
}

/// Category cap for the sequential G-DINA: an ordered item may have at most this many
/// categories (`0..=SEQ_MAX_CAT`). Bounds the ragged `(M_i + 1) * 2^{K_i}` allocation
/// against an adversarial category label, mirroring the `K <= 15` cap on the profile grid.
const SEQ_MAX_CAT: usize = 50;

/// Sequential category probabilities from a single reduced class's step (continuation)
/// probabilities `s = [s_1, .., s_M]` (Ma & de la Torre, 2016; Tutz, 1990):
/// `P(X=0) = 1 - s_1`, `P(X=k) = (prod_{v<=k} s_v)(1 - s_{k+1})` for `1 <= k < M`, and
/// `P(X=M) = prod_{v<=M} s_v` (the top category has NO trailing continuation factor — the
/// stop sentinel `s_{M+1} = 0` makes `1 - s_{M+1} = 1`, so it must not be routed through
/// any eps clamp). The `M + 1` probabilities telescope to 1 for any `s in [0, 1]^M`, so
/// the sequential form is a valid multinomial for free (no simplex projection needed).
#[cfg(test)]
pub(crate) fn seq_category_probs(steps: &[f64]) -> Vec<f64> {
    let m = steps.len();
    let mut probs = vec![0.0f64; m + 1];
    let mut cum = 1.0f64; // prod_{v < k} s_v
    for k in 1..=m {
        let sk = steps[k - 1];
        probs[k - 1] = cum * (1.0 - sk); // P(X = k-1) = (prod_{v<k} s_v)(1 - s_k)
        cum *= sk;
    }
    probs[m] = cum; // P(X = M) = prod_{v<=M} s_v (no trailing 1 - s_{M+1} factor)
    probs
}

/// Log category probabilities for one reduced class, written into `out` (length
/// `steps.len() + 1`). This is the production E-step transform: the same sequential
/// decomposition as [`seq_category_probs`] in log space, with the `M` real step probs
/// eps-clamped for numerical stability and the top-category stop sentinel exact (never
/// clamped, no trailing factor). `exp(out)` equals `seq_category_probs(steps)` for
/// interior `steps` (a test pins this against hand-computed literals).
#[inline]
fn seq_category_logprobs_into(steps: &[f64], eps: f64, out: &mut [f64]) {
    let m = steps.len();
    let mut cum = 0.0f64; // sum_{v<k} ln s_v
    for k in 1..=m {
        let sk = steps[k - 1].clamp(eps, 1.0 - eps);
        out[k - 1] = cum + (1.0 - sk).ln(); // ln P(X = k-1)
        cum += sk.ln();
    }
    out[m] = cum; // ln P(X = M), sentinel s_{M+1}=0 => no trailing factor
}

/// Scatter one weighted response `x` into a reduced class's per-step at-risk (`I`) and
/// advanced (`R`) count cells (`i_cells`/`r_cells` length `M`, contiguous per class). Step
/// `k` is *at risk* when `x >= k-1` and *advanced* when `x >= k`, so `I[k-1] += w` for
/// `k <= x+1` and `R[k-1] += w` for `k <= x`. This is the sequential factorization's
/// Bernoulli bookkeeping: conditional on reaching category `k-1`, advancing to `>= k` is
/// `Bernoulli(s_ik)`, independent across steps, so the saturated MLE is `s_ik = R/I`.
#[inline]
fn seq_scatter_counts(x: usize, w: f64, m: usize, i_cells: &mut [f64], r_cells: &mut [f64]) {
    let kmax = m.min(x + 1); // steps with x >= k-1 (at risk); x < k-1 for larger k
    for k in 1..=kmax {
        i_cells[k - 1] += w;
        if k <= x {
            r_cells[k - 1] += w; // advanced past step k (x >= k)
        }
    }
}

/// Validate polytomous sequential-CDM input and return the per-item maximum observed
/// category `M_i` (the number of ordered steps). Unlike [`validate`], responses are
/// ordered integers `0..=M_i` (not just 0/1); `M_i` is *derived from the data* (max
/// observed category), and an item whose observed max is `< 1` (never leaves category 0)
/// measures nothing and is rejected, mirroring the all-zero-Q-row and unobserved-item
/// rejections. A zero-frequency *interior* category is NOT rejected: in a
/// continuation-ratio model it simply means `s_{i,k+1}(l) ~ 1` and is legitimate.
#[allow(clippy::too_many_arguments)]
fn validate_seq_gdina(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<Vec<u32>, String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(1..=15).contains(&n_attributes) {
        return Err(format!(
            "n_attributes must be in 1..=15 (L = 2^K grid + O(N*J*L) cost); got {n_attributes}"
        ));
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if !cfg.eps.is_finite() || !(0.0 < cfg.eps && cfg.eps < 0.5) {
        return Err("eps must be finite and in (0, 0.5)".into());
    }
    if !cfg.init_slip.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_slip) {
        return Err("init_slip must be finite and in [eps, 1 - eps]".into());
    }
    if !cfg.init_guess.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_guess) {
        return Err("init_guess must be finite and in [eps, 1 - eps]".into());
    }
    if cfg.init_slip + cfg.init_guess >= 1.0 {
        return Err("init_slip + init_guess must be less than 1".into());
    }
    if !cfg.count_floor.is_finite() || cfg.count_floor < 0.0 {
        return Err("count_floor must be finite and non-negative".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_q = crate::checked_mul_usize(n_items, n_attributes, "Q-matrix size overflows usize")?;
    if q_matrix.len() != n_q {
        return Err("q_matrix must have length n_items * n_attributes".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && (!v.is_finite() || v < 0.0 || v.fract() != 0.0) {
            return Err(format!(
                "y[{idx}] must be a non-negative integer category where observed; got {v}"
            ));
        }
    }
    for (idx, &v) in q_matrix.iter().enumerate() {
        if v != 0 && v != 1 {
            return Err(format!("q_matrix[{idx}] must be 0 or 1; got {v}"));
        }
    }
    // Per item: at least one observed response, and a maximum observed category >= 1
    // (an item stuck at category 0 measures nothing). M_i = max observed category.
    let mut max_cat = vec![0u32; n_items];
    for i in 0..n_items {
        let mut any = false;
        let mut mi = 0u32;
        for p in 0..n_persons {
            let idx = p * n_items + i;
            if observed[idx] {
                any = true;
                mi = mi.max(y[idx] as u32);
            }
        }
        if !any {
            return Err(format!("item {i} has no observed responses"));
        }
        if mi < 1 {
            return Err(format!(
                "item {i} never leaves category 0 (max observed category 0; measures nothing)"
            ));
        }
        if mi as usize > SEQ_MAX_CAT {
            return Err(format!(
                "item {i} max category {mi} exceeds SEQ_MAX_CAT = {SEQ_MAX_CAT}"
            ));
        }
        max_cat[i] = mi;
    }
    for i in 0..n_items {
        if !(0..n_attributes).any(|k| q_matrix[i * n_attributes + k] != 0) {
            return Err(format!(
                "q_matrix row {i} is all-zero (item measures no attribute)"
            ));
        }
    }
    for k in 0..n_attributes {
        if !(0..n_items).any(|i| q_matrix[i * n_attributes + k] != 0) {
            return Err(format!(
                "q_matrix column {k} is all-zero (attribute measured by no item)"
            ));
        }
    }
    Ok(max_cat)
}

/// Fit the **shared-Q sequential (continuation-ratio) G-DINA** for ordered polytomous
/// responses (Ma & de la Torre, 2016) by marginal-ML EM over the `2^K` attribute
/// profiles.
///
/// For item `i` with maximum observed category `M_i`, each ordered *step*
/// `k in 1..=M_i` has a free continuation probability that is a saturated G-DINA over the
/// item's `2^{K_i}` reduced attribute classes:
/// `s_ik(l) = P(X_i >= k | X_i >= k-1, reduced class l)`. The category probabilities are
/// the sequential decomposition `P(X_i = k | l) = (prod_{v<=k} s_iv(l))(1 - s_{i,k+1}(l))`
/// with the stop sentinel `s_{i,M_i+1} = 0`. The population is a free profile distribution
/// `pi_c` (as in [`fit_gdina`]; the higher-order structural prior is the alternative
/// offered by [`fit_ho_cdm`]/[`fit_ho_gdina`]).
///
/// **Scope (restriction).** This is the *shared item-level Q-vector* sequential G-DINA:
/// every step of item `i` is a saturated G-DINA over the SAME required attributes (Q row
/// `i`), each with its own step-specific probability table. It is a restriction of the
/// general per-step (per-category) `q_ik` model of Ma & de la Torre (2016), whose headline
/// feature is *step-distinct* attribute requirements (e.g. step 1 needs attribute A, step 2
/// needs A and B). Use [`fit_seq_gdina_qr`] when the steps need distinct Q-vectors. For this
/// shared-Q entry point, supply the item Q-vector as the UNION of every step's required
/// attributes so no step depends on an attribute outside it (any step that truly needs only
/// a subset is still representable — its table is flat in the irrelevant attribute).
///
/// Estimation reuses the CDM machinery: the closed-form saturated M-step
/// `s_ik(l) = R_ik(l) / I_ik(l)` where `R = expected count reaching category >= k` and
/// `I = expected count reaching >= k-1` in reduced class `l` (the sequential likelihood
/// factorizes into independent per-step Bernoullis on the at-risk set, so this ratio is the
/// exact complete-data MLE — [`fit_gdina`]'s saturated step on continuation counts). An
/// unreached step in a reduced class (`I ~ 0`) keeps its previous value (`count_floor`
/// guard) — that cell is non-identified and inert, since only response patterns that cross
/// the step depend on it. With `M_i = 1` for every item the model is exactly [`fit_gdina`].
///
/// `y`/`observed` are row-major `N*J` (`y` holds ordered integer categories `0..=M_i` where
/// observed; `M_i` is derived as the maximum observed category); `q_matrix` is row-major
/// `J*K` (0/1). Missing cells are dropped (MAR). Returns `Err` on malformed input. The
/// nonzero Q-row/Q-column guards are necessary sanity checks, not a certificate of global
/// model identifiability; callers must provide an identified design and inspect `converged`.
/// Convergence uses the absolute observed-data log-likelihood increment and is checked
/// before another M-step, so the trace endpoint and returned parameters agree. The stable
/// termination reason, signed and relative terminal increments, completed M-step count, and
/// requested tolerance are returned explicitly.
///
/// References (APA 7th ed.):
///   Ma, W., & de la Torre, J. (2016). A sequential cognitive diagnosis model for
///     polytomous responses. *British Journal of Mathematical and Statistical Psychology,
///     69*(3), 253-275. https://doi.org/10.1111/bmsp.12070
///   Tutz, G. (1990). Sequential item response models with an ordered response. *British
///     Journal of Mathematical and Statistical Psychology, 43*(1), 39-55.
///     https://doi.org/10.1111/j.2044-8317.1990.tb00925.x
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika, 76*(2),
///     179-199. https://doi.org/10.1007/s11336-011-9207-7
#[allow(clippy::too_many_arguments)]
pub fn fit_seq_gdina(
    y: &[f64],
    observed: &[bool],
    q_matrix: &[u8],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<SeqGdinaResult, String> {
    let max_cat = validate_seq_gdina(y, observed, q_matrix, n_persons, n_items, n_attributes, cfg)?;
    let l_full = 1usize << n_attributes;

    // Per-item required-attribute bitmask and K_i.
    let mut qmask = vec![0usize; n_items];
    let mut k_required = vec![0u32; n_items];
    for i in 0..n_items {
        let mut mask = 0usize;
        for k in 0..n_attributes {
            if q_matrix[i * n_attributes + k] != 0 {
                mask |= 1 << k;
            }
        }
        qmask[i] = mask;
        k_required[i] = mask.count_ones();
    }

    // Ragged CLASS-MAJOR CSR: item i owns M_i * 2^{K_i} step probs (class l, step k at
    // s_off[i] + l*M_i + (k-1)) and (M_i+1) * 2^{K_i} category log-probs.
    let mut s_off = vec![0usize; n_items + 1];
    let mut cat_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        let rw = 1usize << k_required[i];
        let m = max_cat[i] as usize;
        s_off[i + 1] = s_off[i] + m * rw;
        cat_off[i + 1] = cat_off[i] + (m + 1) * rw;
    }
    let total_steps = s_off[n_items];
    let total_cats = cat_off[n_items];

    // Reduced-class index of every (item, full-profile) pair.
    let mut red = vec![0u16; n_items * l_full];
    for i in 0..n_items {
        for c in 0..l_full {
            red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
        }
    }

    // Monotone init: each step's probability rises with the count of mastered required
    // attributes, from init_guess (none) to 1 - init_slip (all). At M_i = 1 this is exactly
    // fit_gdina's `p` init, which (with the shared refresh/M-step below) makes M=1 reduce to
    // fit_gdina bit-for-bit.
    let mut s = vec![0.0f64; total_steps];
    for i in 0..n_items {
        let ki = k_required[i] as f64; // >= 1
        let rw = 1usize << k_required[i];
        let m = max_cat[i] as usize;
        for l in 0..rw {
            let frac = (l.count_ones() as f64) / ki;
            let val = cfg.init_guess + (1.0 - cfg.init_slip - cfg.init_guess) * frac;
            for k in 0..m {
                s[s_off[i] + l * m + k] = val;
            }
        }
    }
    let mut pi = vec![1.0 / l_full as f64; l_full];
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    let mut post = vec![0.0f64; l_full];
    let mut clp = vec![0.0f64; total_cats]; // category log-probs, class-major
    let mut log_pi = vec![0.0f64; l_full];

    // Fill category log-probs from step probs: clp[cat_off[i] + l*(M+1) + x] = ln P(X=x|l).
    // The M_i real step probs are eps-clamped; the stop sentinel is NOT clamped (top
    // category = cumulative log-product with no trailing 1 - s_{M+1} factor).
    let refresh = |s: &[f64], clp: &mut [f64]| {
        for i in 0..n_items {
            let rw = 1usize << k_required[i];
            let m = max_cat[i] as usize;
            let (so, co) = (s_off[i], cat_off[i]);
            for l in 0..rw {
                let (sbase, cbase) = (so + l * m, co + l * (m + 1));
                seq_category_logprobs_into(
                    &s[sbase..sbase + m],
                    cfg.eps,
                    &mut clp[cbase..cbase + (m + 1)],
                );
            }
        }
    };

    for _ in 0..cfg.max_iter {
        refresh(&s, &mut clp);
        for c in 0..l_full {
            log_pi[c] = pi[c].ln();
        }

        let mut i_acc = vec![0.0f64; total_steps];
        let mut r_acc = vec![0.0f64; total_steps];
        let mut pi_new = vec![0.0f64; l_full];
        let mut total_ll = 0.0f64;
        for j in 0..n_persons {
            // E-step posterior over the 2^K profiles using the category log-probs.
            for c in 0..l_full {
                let mut acc = log_pi[c];
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let m1 = max_cat[i] as usize + 1;
                        let l = red[i * l_full + c] as usize;
                        let x = y[idx] as usize;
                        acc += clp[cat_off[i] + l * m1 + x];
                    }
                }
                post[c] = acc;
            }
            let mmax = post[..l_full]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for c in 0..l_full {
                denom += (post[c] - mmax).exp();
            }
            for c in 0..l_full {
                post[c] = (post[c] - mmax).exp() / denom;
            }
            total_ll += mmax + denom.ln();

            for c in 0..l_full {
                pi_new[c] += post[c];
            }
            // M-step counts: scatter each response into its at-risk/advanced step cells.
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let m = max_cat[i] as usize;
                    let x = y[idx] as usize;
                    let so = s_off[i];
                    for c in 0..l_full {
                        let l = red[i * l_full + c] as usize;
                        let base = so + l * m;
                        seq_scatter_counts(
                            x,
                            post[c],
                            m,
                            &mut i_acc[base..base + m],
                            &mut r_acc[base..base + m],
                        );
                    }
                }
            }
        }
        loglik_trace.push(total_ll);

        // Converged-check before the M-step: returned params match the trace endpoint.
        if loglik_trace.len() > 1 {
            let n = loglik_trace.len();
            if (loglik_trace[n - 1] - loglik_trace[n - 2]).abs() < cfg.tol {
                converged = true;
                break;
            }
        }

        // M-step: saturated per-step closed form s_ik(l) = R/I. Empty at-risk cell keeps
        // its previous value (non-identified, inert). Box 0<=s<=1 is free (0<=R<=I).
        for x in 0..total_steps {
            if i_acc[x] > cfg.count_floor {
                s[x] = (r_acc[x] / i_acc[x]).clamp(cfg.eps, 1.0 - cfg.eps);
            }
        }
        let nf = n_persons as f64;
        let mut z = 0.0f64;
        for c in 0..l_full {
            pi[c] = (pi_new[c] / nf).max(cfg.eps);
            z += pi[c];
        }
        for c in 0..l_full {
            pi[c] /= z;
        }
        n_iter += 1;
    }

    // Classification pass + category probabilities from the final step probs.
    refresh(&s, &mut clp);
    for c in 0..l_full {
        log_pi[c] = pi[c].ln();
    }
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut final_ll = 0.0f64;
    for j in 0..n_persons {
        for c in 0..l_full {
            let mut acc = log_pi[c];
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let m1 = max_cat[i] as usize + 1;
                    let l = red[i * l_full + c] as usize;
                    let x = y[idx] as usize;
                    acc += clp[cat_off[i] + l * m1 + x];
                }
            }
            post[c] = acc;
        }
        let mmax = post[..l_full]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for c in 0..l_full {
            denom += (post[c] - mmax).exp();
        }
        for c in 0..l_full {
            post[c] = (post[c] - mmax).exp() / denom;
        }
        final_ll += mmax + denom.ln();
        let mut best = 0usize;
        for c in 1..l_full {
            if post[c] > post[best] {
                best = c;
            }
        }
        map_profile[j] = best as u32;
        for k in 0..n_attributes {
            let mut pk = 0.0;
            for c in 0..l_full {
                if (c >> k) & 1 == 1 {
                    pk += post[c];
                }
            }
            attr_prob[j * n_attributes + k] = pk;
        }
    }
    if !converged {
        loglik_trace.push(final_ll);
    }

    let cat_prob: Vec<f64> = clp.iter().map(|v| v.exp()).collect();
    let final_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| pair[1] - pair[0])
        .unwrap_or(f64::NAN);
    let final_relative_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| (pair[1] - pair[0]).abs() / (1.0 + pair[0].abs()))
        .unwrap_or(f64::NAN);
    let termination_reason = if converged {
        "tolerance_met"
    } else {
        "max_iter_reached"
    };

    Ok(SeqGdinaResult {
        s_off,
        step_prob: s,
        cat_off,
        cat_prob,
        max_cat,
        k_required,
        profile_prob: pi,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        final_relative_loglik_change,
        stopping_tolerance: cfg.tol,
        n_parameters: total_steps + (l_full - 1),
    })
}

/// Result of [`fit_seq_gdina_qr`] (Ma & de la Torre, 2016): the PER-STEP-Q sequential
/// (continuation-ratio) G-DINA, where each ordered step has its own attribute requirement
/// `q_ik` (the restricted Q-matrix `Q_r`). Step probabilities are STEP-ROW-major: step row
/// `g = step_off[i] + (k-1)` (item `i`, step `k`) owns `2^{|q_ik|}` reduced classes at
/// `step_prob[spo[g] + l]`, `l` the reduced class of the profile under `q_ik`. Category
/// probabilities are UNION-class-major: item `i`'s union `u_i = OR_k q_ik` has `2^{K^u_i}`
/// classes and `cat_prob[cat_off[i] + uc*(M_i+1) + x] = P(X_i = x | union class uc)`.
#[derive(Clone, Debug)]
pub struct SeqGdinaQrResult {
    /// Per-item offsets into the step-row arrays (`spo`, `step_kq`), length `n_items + 1`.
    pub step_off: Vec<usize>,
    /// Per-step-row offsets into `step_prob` (length `sum_i M_i + 1`).
    pub spo: Vec<usize>,
    /// Continuation probabilities `s_ik(l)`, step-row-major (see struct doc).
    pub step_prob: Vec<f64>,
    /// Required-attribute count `|q_ik|` per step row (length `sum_i M_i`).
    pub step_kq: Vec<u32>,
    /// Per-item category-prob block offsets into `cat_prob` (length `n_items + 1`).
    pub cat_off: Vec<usize>,
    /// Implied category probabilities `P(X_i = x | union class uc)`, union-class-major.
    pub cat_prob: Vec<f64>,
    /// Number of ordered steps `M_i` per item.
    pub max_cat: Vec<u32>,
    /// Union required-attribute count `K^u_i = |OR_k q_ik|` per item.
    pub union_k: Vec<u32>,
    /// Free profile distribution `pi_c` (length `2^K`, sums to 1).
    pub profile_prob: Vec<f64>,
    /// Bit-encoded MAP profile per person.
    pub map_profile: Vec<u32>,
    /// Marginal `P(alpha_jk = 1 | X_j)`, row-major `N x K`.
    pub attr_prob: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: &'static str,
    pub final_loglik_change: f64,
    pub final_relative_loglik_change: f64,
    pub stopping_tolerance: f64,
    /// `sum_{i,k} 2^{|q_ik|}` step probs `+ (2^K - 1)` free profile probs.
    pub n_parameters: usize,
}

/// Validate per-step-Q sequential-CDM input and return `(n_steps as usize vector)`. `step_q`
/// is row-major `(sum_i n_steps[i]) x K` (0/1); `n_steps[i] = M_i` is the declared step count
/// of item `i`. Rejects: shape/overflow, non-0/1 or non-integer data, a step measuring nothing
/// (all-zero `q_ik` row), an attribute measured by NO step of any item (all-zero column over the
/// union), an item with no observed responses, and `M_i != ` the maximum OBSERVED category
/// (a declared step no one reaches, or data beyond the declared steps).
#[allow(clippy::too_many_arguments)]
fn validate_seq_gdina_qr(
    y: &[f64],
    observed: &[bool],
    step_q: &[u8],
    n_steps: &[usize],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<(), String> {
    if n_persons < 1 || n_items < 1 {
        return Err("n_persons and n_items must be >= 1".into());
    }
    if !(1..=15).contains(&n_attributes) {
        return Err(format!(
            "n_attributes must be in 1..=15 (L = 2^K grid + O(N*J*L) cost); got {n_attributes}"
        ));
    }
    if cfg.max_iter == 0 {
        return Err("max_iter must be positive".into());
    }
    if !cfg.tol.is_finite() || cfg.tol <= 0.0 {
        return Err("tol must be finite and positive".into());
    }
    if !cfg.eps.is_finite() || !(0.0 < cfg.eps && cfg.eps < 0.5) {
        return Err("eps must be finite and in (0, 0.5)".into());
    }
    if !cfg.init_slip.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_slip) {
        return Err("init_slip must be finite and in [eps, 1 - eps]".into());
    }
    if !cfg.init_guess.is_finite() || !(cfg.eps..=1.0 - cfg.eps).contains(&cfg.init_guess) {
        return Err("init_guess must be finite and in [eps, 1 - eps]".into());
    }
    if cfg.init_slip + cfg.init_guess >= 1.0 {
        return Err("init_slip + init_guess must be less than 1".into());
    }
    if !cfg.count_floor.is_finite() || cfg.count_floor < 0.0 {
        return Err("count_floor must be finite and non-negative".into());
    }
    if n_steps.len() != n_items {
        return Err("n_steps must have length n_items".into());
    }
    // Total step rows and the step_q length, both via checked arithmetic.
    let mut total_step_rows = 0usize;
    for (i, &m) in n_steps.iter().enumerate() {
        if m < 1 {
            return Err(format!(
                "item {i} has n_steps < 1 (an item must leave category 0)"
            ));
        }
        if m > SEQ_MAX_CAT {
            return Err(format!(
                "item {i} n_steps {m} exceeds SEQ_MAX_CAT = {SEQ_MAX_CAT}"
            ));
        }
        total_step_rows =
            crate::checked_add_usize(total_step_rows, m, "sum of n_steps overflows usize")?;
    }
    let n_sq =
        crate::checked_mul_usize(total_step_rows, n_attributes, "step-Q size overflows usize")?;
    if step_q.len() != n_sq {
        return Err("step_q must have length sum(n_steps) * n_attributes".into());
    }
    let n_cells =
        crate::checked_mul_usize(n_persons, n_items, "n_persons * n_items overflows usize")?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    for (idx, &v) in y.iter().enumerate() {
        if observed[idx] && (!v.is_finite() || v < 0.0 || v.fract() != 0.0) {
            return Err(format!(
                "y[{idx}] must be a non-negative integer category where observed; got {v}"
            ));
        }
    }
    for (idx, &v) in step_q.iter().enumerate() {
        if v != 0 && v != 1 {
            return Err(format!("step_q[{idx}] must be 0 or 1; got {v}"));
        }
    }
    // Each declared step measures at least one attribute (no all-zero step-q row).
    for g in 0..total_step_rows {
        if !(0..n_attributes).any(|k| step_q[g * n_attributes + k] != 0) {
            return Err(format!(
                "step row {g} is all-zero (a step measuring no attribute)"
            ));
        }
    }
    // Every attribute is required by at least one step of some item (union column non-empty).
    for k in 0..n_attributes {
        if !(0..total_step_rows).any(|g| step_q[g * n_attributes + k] != 0) {
            return Err(format!(
                "attribute {k} is required by no step (all-zero column; non-identified)"
            ));
        }
    }
    // Per item: at least one observed response, and the maximum observed category equals the
    // declared step count M_i (so every declared step has a globally non-empty at-risk set and
    // no observed category exceeds the declared steps).
    let mut step_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        step_off[i + 1] = step_off[i] + n_steps[i];
    }
    for i in 0..n_items {
        let mut any = false;
        let mut mi = 0u32;
        for p in 0..n_persons {
            let idx = p * n_items + i;
            if observed[idx] {
                any = true;
                mi = mi.max(y[idx] as u32);
            }
        }
        if !any {
            return Err(format!("item {i} has no observed responses"));
        }
        if mi as usize != n_steps[i] {
            return Err(format!(
                "item {i}: max observed category {mi} != declared n_steps {} (a declared step is \
                 unreached, or data exceeds the declared steps)",
                n_steps[i]
            ));
        }
    }
    Ok(())
}

/// Fit the **per-step-Q sequential (continuation-ratio) G-DINA** for ordered polytomous
/// responses (Ma & de la Torre, 2016), the full restricted-Q model in which each ordered STEP
/// `k` of item `i` is a saturated G-DINA over its OWN required attributes `q_ik` (step 1 may
/// need attribute A, step 2 need A and B, etc.).
///
/// Generalizes the shared-Q [`fit_seq_gdina`]: when every step of an item shares the item's
/// Q-vector this reduces to it exactly. Step `k`'s continuation probability
/// `s_ik(l) = P(X_i >= k | X_i >= k-1, reduced class of alpha under q_ik)` is free per step
/// reduced class; the category probability is the unchanged sequential product
/// `P(X_i = x | alpha) = (prod_{v<=x} s_iv)(1 - s_{i,x+1})`. Estimated by marginal-ML EM with
/// the closed-form saturated step ratio `s_ik(l) = R/I` (reached >= k over reached >= k-1 in
/// step `k`'s reduced class `l`) — the sequential likelihood still factorizes into independent
/// per-step Bernoullis, so this is the exact complete-data MLE. Free profile distribution `pi_c`.
///
/// Each step's reduced class is computed DIRECTLY from the full profile (`reduce_class(c,
/// q_ik)`); the item's UNION class `reduce_class(c, OR_k q_ik)` indexes the category
/// probabilities. `y`/`observed` are row-major `N*J` (ordered integer categories `0..=M_i`);
/// `step_q` is row-major `(sum_i n_steps[i]) * K` (0/1), step `k` of item `i` at row
/// `step_off[i] + (k-1)`; `n_steps[i] = M_i` (the number of steps, which must equal item `i`'s
/// maximum observed category). Missing cells are dropped (MAR). Nonzero step-Q rows and
/// columns are necessary sanity checks, not a certificate of global model identifiability;
/// callers must provide an identified design and inspect `converged`.
///
/// References (APA 7th ed.):
///   Ma, W., & de la Torre, J. (2016). A sequential cognitive diagnosis model for polytomous
///     responses. *British Journal of Mathematical and Statistical Psychology, 69*(3),
///     253-275. https://doi.org/10.1111/bmsp.12070
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika, 76*(2),
///     179-199. https://doi.org/10.1007/s11336-011-9207-7
#[allow(clippy::too_many_arguments)]
pub fn fit_seq_gdina_qr(
    y: &[f64],
    observed: &[bool],
    step_q: &[u8],
    n_steps: &[usize],
    n_persons: usize,
    n_items: usize,
    n_attributes: usize,
    cfg: &CdmConfig,
) -> Result<SeqGdinaQrResult, String> {
    validate_seq_gdina_qr(
        y,
        observed,
        step_q,
        n_steps,
        n_persons,
        n_items,
        n_attributes,
        cfg,
    )?;
    let l_full = 1usize << n_attributes;

    // Per-item offsets into the step-row arrays; total step rows = sum_i M_i.
    let mut step_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        step_off[i + 1] = step_off[i] + n_steps[i];
    }
    let n_rows = step_off[n_items];

    // Per step row: its own attribute mask, |q_ik|, reduced-class width, and step_prob offset.
    let mut step_qmask = vec![0usize; n_rows];
    let mut step_kq = vec![0u32; n_rows];
    let mut spo = vec![0usize; n_rows + 1];
    for g in 0..n_rows {
        let mut mask = 0usize;
        for k in 0..n_attributes {
            if step_q[g * n_attributes + k] != 0 {
                mask |= 1 << k;
            }
        }
        step_qmask[g] = mask;
        step_kq[g] = mask.count_ones();
        spo[g + 1] = spo[g] + (1usize << step_kq[g]);
    }
    let total_steps = spo[n_rows];

    // Each step's reduced class as a function of the FULL profile (mirror shared-Q's `red`;
    // this avoids the union-renumber / bit-gather hazard entirely).
    let mut step_red = vec![0u16; n_rows * l_full];
    for g in 0..n_rows {
        for c in 0..l_full {
            step_red[g * l_full + c] = reduce_class(c, step_qmask[g]) as u16;
        }
    }

    // Per item: union mask u_i, union width, union reduced class of each profile, category
    // offsets (union-class-major: (M_i+1) * 2^{K^u_i}).
    let mut union_k = vec![0u32; n_items];
    let mut union_rw = vec![0usize; n_items];
    let mut red_u = vec![0u16; n_items * l_full];
    let mut cat_off = vec![0usize; n_items + 1];
    for i in 0..n_items {
        let mut u = 0usize;
        for g in step_off[i]..step_off[i + 1] {
            u |= step_qmask[g];
        }
        union_k[i] = u.count_ones();
        union_rw[i] = 1usize << union_k[i];
        for c in 0..l_full {
            red_u[i * l_full + c] = reduce_class(c, u) as u16;
        }
        cat_off[i + 1] = cat_off[i] + (n_steps[i] + 1) * union_rw[i];
    }
    let total_cats = cat_off[n_items];

    // Monotone init per step row, using the STEP's own |q_ik| in the denominator so the
    // shared-Q case reproduces fit_seq_gdina's init exactly.
    let mut s = vec![0.0f64; total_steps];
    for g in 0..n_rows {
        let kq = step_kq[g] as f64; // >= 1 (validate rejects all-zero step rows)
        let rw = 1usize << step_kq[g];
        for l in 0..rw {
            let frac = (l.count_ones() as f64) / kq;
            s[spo[g] + l] = cfg.init_guess + (1.0 - cfg.init_slip - cfg.init_guess) * frac;
        }
    }
    let mut pi = vec![1.0 / l_full as f64; l_full];
    let mut loglik_trace: Vec<f64> = Vec::new();
    let mut converged = false;
    let mut n_iter = 0usize;

    let mut post = vec![0.0f64; l_full];
    let mut clp = vec![0.0f64; total_cats]; // union-class category log-probs
    let mut log_pi = vec![0.0f64; l_full];
    let mut sbuf = vec![0.0f64; *n_steps.iter().max().unwrap_or(&1)]; // per-item step gather

    // Fill union-class category log-probs by walking the full profile grid: each step's own
    // reduced class comes from step_red[g][c]; multiple profiles map to the same union class and
    // (because q_ik subset of u_i) give the same gathered step vector, so the writes agree.
    let refresh = |s: &[f64], clp: &mut [f64], sbuf: &mut [f64]| {
        for i in 0..n_items {
            let m = n_steps[i];
            let m1 = m + 1;
            let co = cat_off[i];
            for c in 0..l_full {
                let uc = red_u[i * l_full + c] as usize;
                debug_assert!(uc < union_rw[i], "union class (clp) within bound");
                for v in 0..m {
                    let g = step_off[i] + v;
                    let l_v = step_red[g * l_full + c] as usize;
                    debug_assert!(
                        l_v < (1usize << step_kq[g]),
                        "step class (step_prob) within bound"
                    );
                    sbuf[v] = s[spo[g] + l_v];
                }
                seq_category_logprobs_into(
                    &sbuf[..m],
                    cfg.eps,
                    &mut clp[co + uc * m1..co + uc * m1 + m1],
                );
            }
        }
    };

    for _ in 0..cfg.max_iter {
        refresh(&s, &mut clp, &mut sbuf);
        for c in 0..l_full {
            log_pi[c] = pi[c].ln();
        }

        let mut i_acc = vec![0.0f64; total_steps];
        let mut r_acc = vec![0.0f64; total_steps];
        let mut pi_new = vec![0.0f64; l_full];
        let mut total_ll = 0.0f64;
        for j in 0..n_persons {
            // E-step posterior over the 2^K profiles via the union-class category log-probs.
            for c in 0..l_full {
                let mut acc = log_pi[c];
                for i in 0..n_items {
                    let idx = j * n_items + i;
                    if observed[idx] {
                        let m1 = n_steps[i] + 1;
                        let uc = red_u[i * l_full + c] as usize;
                        let x = y[idx] as usize;
                        acc += clp[cat_off[i] + uc * m1 + x];
                    }
                }
                post[c] = acc;
            }
            let mmax = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut denom = 0.0f64;
            for v in post.iter() {
                denom += (v - mmax).exp();
            }
            total_ll += mmax + denom.ln();
            for v in post.iter_mut() {
                *v = (*v - mmax).exp() / denom;
            }
            for c in 0..l_full {
                pi_new[c] += post[c];
            }
            // M-step counts: scatter PER STEP into its own (step_row, step-class) cell.
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let x = y[idx] as usize;
                    let m = n_steps[i];
                    for c in 0..l_full {
                        let pc = post[c];
                        for v in 1..=m {
                            let g = step_off[i] + (v - 1);
                            let l_v = step_red[g * l_full + c] as usize;
                            let cell = spo[g] + l_v;
                            if x >= v - 1 {
                                i_acc[cell] += pc; // at risk for step v
                                if x >= v {
                                    r_acc[cell] += pc; // advanced past step v
                                }
                            }
                        }
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

        for x in 0..total_steps {
            if i_acc[x] > cfg.count_floor {
                s[x] = (r_acc[x] / i_acc[x]).clamp(cfg.eps, 1.0 - cfg.eps);
            }
        }
        let nf = n_persons as f64;
        let mut z = 0.0f64;
        for c in 0..l_full {
            pi[c] = (pi_new[c] / nf).max(cfg.eps);
            z += pi[c];
        }
        for c in 0..l_full {
            pi[c] /= z;
        }
        n_iter += 1;
    }

    // Classification pass + category probabilities from the final step probs.
    refresh(&s, &mut clp, &mut sbuf);
    for c in 0..l_full {
        log_pi[c] = pi[c].ln();
    }
    let mut map_profile = vec![0u32; n_persons];
    let mut attr_prob = vec![0.0f64; n_persons * n_attributes];
    let mut final_ll = 0.0f64;
    for j in 0..n_persons {
        for c in 0..l_full {
            let mut acc = log_pi[c];
            for i in 0..n_items {
                let idx = j * n_items + i;
                if observed[idx] {
                    let m1 = n_steps[i] + 1;
                    let uc = red_u[i * l_full + c] as usize;
                    let x = y[idx] as usize;
                    acc += clp[cat_off[i] + uc * m1 + x];
                }
            }
            post[c] = acc;
        }
        let mmax = post.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut denom = 0.0f64;
        for v in post.iter() {
            denom += (v - mmax).exp();
        }
        for v in post.iter_mut() {
            *v = (*v - mmax).exp() / denom;
        }
        final_ll += mmax + denom.ln();
        let mut best = 0usize;
        for c in 1..l_full {
            if post[c] > post[best] {
                best = c;
            }
        }
        map_profile[j] = best as u32;
        for k in 0..n_attributes {
            let mut pk = 0.0;
            for c in 0..l_full {
                if (c >> k) & 1 == 1 {
                    pk += post[c];
                }
            }
            attr_prob[j * n_attributes + k] = pk;
        }
    }
    if !converged {
        loglik_trace.push(final_ll);
    }

    let cat_prob: Vec<f64> = clp.iter().map(|v| v.exp()).collect();
    let max_cat: Vec<u32> = n_steps.iter().map(|&m| m as u32).collect();
    let final_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| pair[1] - pair[0])
        .unwrap_or(f64::NAN);
    let final_relative_loglik_change = loglik_trace
        .windows(2)
        .last()
        .map(|pair| (pair[1] - pair[0]).abs() / (1.0 + pair[0].abs()))
        .unwrap_or(f64::NAN);
    let termination_reason = if converged {
        "tolerance_met"
    } else {
        "max_iter_reached"
    };

    Ok(SeqGdinaQrResult {
        step_off,
        spo,
        step_prob: s,
        step_kq,
        cat_off,
        cat_prob,
        max_cat,
        union_k,
        profile_prob: pi,
        map_profile,
        attr_prob,
        loglik_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        final_relative_loglik_change,
        stopping_tolerance: cfg.tol,
        n_parameters: total_steps + (l_full - 1),
    })
}

#[cfg(test)]
#[path = "../../../tests/unit/cdm_tests.rs"]
mod tests;
