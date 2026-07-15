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
    if !cfg.mono_backoff.is_finite() || cfg.mono_backoff <= 2.0 * cfg.eps || cfg.mono_backoff >= 1.0 {
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
    let n_cells = n_persons
        .checked_mul(n_items)
        .ok_or_else(|| "n_persons * n_items overflows usize".to_string())?;
    if y.len() != n_cells || observed.len() != n_cells {
        return Err("y and observed must have length n_persons * n_items".into());
    }
    let n_q = n_items
        .checked_mul(n_attributes)
        .ok_or_else(|| "n_items * n_attributes overflows usize".to_string())?;
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
            return Err(format!("q_matrix row {i} is all-zero (item measures no attribute)"));
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
    let mut si = if i1[i] > cfg.count_floor { 1.0 - r1[i] / i1[i] } else { s[i] };
    let mut gi = if i0[i] > cfg.count_floor { r0[i] / i0[i] } else { g[i] };
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
    let m = post[..l_full].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
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
/// The box constraint `0 <= p_il <= 1` holds for free (`0 <= R_il <= I_il`); the
/// all-mastered class has the highest success probability under an identifiable Q,
/// which the recovery tests assert rather than the estimator projecting (matching de
/// la Torre's unconstrained-in-`[0,1]` saturated MLE; full subset-lattice isotonicity
/// — Hong et al., 2016 — is a deferred add-on). `y`/`observed` are row-major `N*J`,
/// `q_matrix` row-major `J*K`; missing cells are dropped (MAR).
///
/// References (APA 7th ed.):
///   de la Torre, J. (2011). The generalized DINA model framework. *Psychometrika,
///     76*(2), 179-199. https://doi.org/10.1007/s11336-011-9207-7
///   Chen, H., & Zhou, H. (2016). ... order restrictions. *Journal of Classification,
///     33*(3), 460-484. https://doi.org/10.1007/s00357-016-9216-4
///   Ma, W., & de la Torre, J. (2020). GDINA: An R package. *Journal of Statistical
///     Software, 93*(14), 1-26. https://doi.org/10.18637/jss.v093.i14
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
///     validation. *Psychometrika, 81*(2), 253-273.
///     https://doi.org/10.1007/s11336-015-9467-8
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
    let res = fit_gdina(y, observed, provisional_q, n_persons, n_items, n_attributes, cfg)?;
    ensure_gdina_converged(&res, cfg)?;

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
    let log_pi: Vec<f64> = res.profile_prob.iter().map(|v| v.max(cfg.eps).ln()).collect();

    let mut icount = vec![0.0f64; n_items * l_full]; // I_{i,c} expected count
    let mut rcount = vec![0.0f64; n_items * l_full]; // R_{i,c} expected correct
    let mut pi_c = vec![0.0f64; l_full];
    let mut post = vec![0.0f64; l_full];
    for j in 0..n_persons {
        posterior_row_gdina(
            j, y, observed, n_items, l_full, &red, &log_p1, &log_p0, &res.item_off, &log_pi,
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
        let item_mean = if mean_den > 0.0 { mean_num / mean_den } else { 0.0 };
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
        provisional_pvaf[i] = if prov_mask == 0 { 0.0 } else { pvaf_of(prov_mask) };

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
    let log_pi: Vec<f64> = res.profile_prob.iter().map(|v| v.max(cfg.eps).ln()).collect();
    let mut icount = vec![0.0f64; total]; // I_l, CSR layout matching item_prob
    let mut post = vec![0.0f64; l_full];
    for j in 0..n_persons {
        posterior_row_gdina(
            j, y, observed, n_items, l_full, &red, &log_p1, &log_p0, &res.item_off, &log_pi,
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
            if v <= 0.0 {
                continue;
            }
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
                0 => (0..w).filter(|&s| s != 0 && s != full).map(|s| vec![(s, 1.0)]).collect(),
                // DINO: delta_S - (-1)^{|S|+1} delta_1 = 0 for every S != {empty, ref=1}.
                1 => (0..w)
                    .filter(|&s| s != 0 && s != 1)
                    .map(|s| {
                        let sign = if (s as u32).count_ones() % 2 == 1 { 1.0 } else { -1.0 };
                        vec![(s, 1.0), (1usize, -sign)]
                    })
                    .collect(),
                // A-CDM / LLM / R-RUM: all interaction coordinates zero. The three share
                // this restriction pattern but on different links (identity/logit/log),
                // so they differ only in which (delta, Sigma) pair the caller feeds in.
                _ => (0..w).filter(|&s| (s as u32).count_ones() >= 2).map(|s| vec![(s, 1.0)]).collect(),
            }
        };

        for m in 0..n_models {
            let rows = restriction_rows(m);
            let df = rows.len();
            wald_df[i * n_models + m] = df;
            if df == 0 {
                continue;
            }
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
        let param_count = |m: usize| -> usize { if m <= 1 { 2 } else { 1 + k } };
        let mut best: Option<usize> = None;
        for m in 0..n_models {
            if wald_df[i * n_models + m] == 0 {
                continue;
            }
            let pv = p_value[i * n_models + m];
            if pv.is_finite() && pv > alpha {
                best = match best {
                    None => Some(m),
                    Some(b) => {
                        let (pb, pm) = (param_count(b), param_count(m));
                        if pm < pb || (pm == pb && pv > p_value[i * n_models + b]) {
                            Some(m)
                        } else {
                            Some(b)
                        }
                    }
                };
            }
        }
        selected[i] = best.map_or(-1, |m| m as i64);
    }

    Ok(WaldSelectionResult { models, wald_stat, wald_df, p_value, selected, alpha })
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
                lp += if (c >> k) & 1 == 1 { logp[k * q + qi] } else { log1mp[k * q + qi] };
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
        if det.abs() < 1e-12 {
            break;
        }
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
///     https://doi.org/10.1007/BF02295640
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
                    lp += if (c >> k) & 1 == 1 { logp[k * q + qi] } else { log1mp[k * q + qi] };
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
                    lp += if (c >> k) & 1 == 1 { logp[k * q + qi] } else { log1mp[k * q + qi] };
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
    let termination_reason = if converged { "tolerance_met" } else { "max_iter_reached" };

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
        fn bern(&mut self, p: f64) -> f64 {
            if self.next_f64() < p {
                1.0
            } else {
                0.0
            }
        }
        fn profile(&mut self, l: usize) -> usize {
            ((self.next_f64() * l as f64) as usize).min(l - 1)
        }
        fn normal(&mut self) -> f64 {
            let u1 = self.next_f64().max(1e-12);
            let u2 = self.next_f64();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

    fn qmask_of(q: &[u8], i: usize, k: usize) -> usize {
        let mut m = 0usize;
        for a in 0..k {
            if q[i * k + a] != 0 {
                m |= 1 << a;
            }
        }
        m
    }
    fn eta_of(model: CdmModel, c: usize, mask: usize) -> u8 {
        match model {
            CdmModel::Dina => ((c & mask) == mask) as u8,
            CdmModel::Dino => ((c & mask) != 0) as u8,
        }
    }

    /// Draw responses for the given true profiles using the same bit encoding as the estimator.
    fn simulate(
        model: CdmModel,
        q: &[u8],
        s: &[f64],
        g: &[f64],
        profiles: &[usize],
        n_items: usize,
        n_attr: usize,
        rng: &mut Lcg,
    ) -> Vec<f64> {
        let n = profiles.len();
        let mut y = vec![0.0f64; n * n_items];
        for j in 0..n {
            for i in 0..n_items {
                let mask = qmask_of(q, i, n_attr);
                let eta = eta_of(model, profiles[j], mask);
                let p = if eta == 1 { 1.0 - s[i] } else { g[i] };
                y[j * n_items + i] = rng.bern(p);
            }
        }
        y
    }

    fn pattern_agreement(map: &[u32], truth: &[usize]) -> f64 {
        let ok = map.iter().zip(truth).filter(|(m, t)| **m as usize == **t).count();
        ok as f64 / map.len() as f64
    }
    fn attribute_agreement(attr_prob: &[f64], truth: &[usize], n: usize, k: usize) -> f64 {
        let mut ok = 0usize;
        for j in 0..n {
            for a in 0..k {
                let est = (attr_prob[j * k + a] >= 0.5) as usize;
                let tru = (truth[j] >> a) & 1;
                if est == tru {
                    ok += 1;
                }
            }
        }
        ok as f64 / (n * k) as f64
    }
    fn nondecreasing(trace: &[f64]) -> bool {
        trace.windows(2).all(|w| w[1] >= w[0] - 1e-6)
    }
    fn monotone_items(res: &CdmResult) -> bool {
        // 1 - s_i > g_i, with slack for the extreme clamp corner (1-s = g = eps).
        res.slip.iter().zip(&res.guess).all(|(s, g)| 1.0 - s > g - 1e-9)
    }

    /// Anchor 1: the eta bitmask + likelihood algebra, with zero estimation. `P(X_j)`
    /// from the module's log-space path must equal a naive enumeration that expands
    /// `eta = prod_k alpha^{q}` in plain arithmetic.
    #[test]
    fn anchor_brute_force_likelihood() {
        let (n_attr, n_items, l) = (2usize, 2usize, 4usize);
        let q: Vec<u8> = vec![1, 0, /* */ 1, 1];
        let s = [0.1f64, 0.2];
        let g = [0.15f64, 0.2];
        let pi = [0.4f64, 0.2, 0.1, 0.3];
        let x = [1.0f64, 0.0];
        let model = CdmModel::Dina;

        let mut eta = vec![0u8; n_items * l];
        let mut lp1 = vec![0.0f64; n_items * 2];
        let mut lp0 = vec![0.0f64; n_items * 2];
        for i in 0..n_items {
            let mask = qmask_of(&q, i, n_attr);
            for c in 0..l {
                eta[i * l + c] = eta_of(model, c, mask);
            }
            lp1[i * 2 + 1] = (1.0 - s[i]).ln();
            lp0[i * 2 + 1] = s[i].ln();
            lp1[i * 2] = g[i].ln();
            lp0[i * 2] = (1.0 - g[i]).ln();
        }
        let log_pi: Vec<f64> = pi.iter().map(|p| p.ln()).collect();
        let observed = vec![true; n_items];
        let mut post = vec![0.0f64; l];
        let log_px =
            posterior_row(0, &x, &observed, n_items, l, &eta, &lp1, &lp0, &log_pi, &mut post);

        let mut px = 0.0;
        for c in 0..l {
            let mut lik = pi[c];
            for i in 0..n_items {
                let mut e = 1u8;
                for k in 0..n_attr {
                    if q[i * n_attr + k] == 1 {
                        e *= ((c >> k) & 1) as u8; // AND gate as a product
                    }
                }
                let pc = if e == 1 { 1.0 - s[i] } else { g[i] };
                let xi = x[i];
                lik *= pc.powf(xi) * (1.0 - pc).powf(1.0 - xi);
            }
            px += lik;
        }
        assert!((log_px.exp() - px).abs() < 1e-12, "module {} vs naive {}", log_px.exp(), px);
        assert!((post.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Anchor 2: deterministic limit s=g=0 => X = eta exactly. Recovery of the ideal
    /// pattern must be perfect and recovered slip/guess near zero.
    #[test]
    fn anchor_deterministic_limit() {
        let (n_attr, n_items) = (2usize, 3usize);
        let q: Vec<u8> = vec![1, 0, /* */ 0, 1, /* */ 1, 1];
        let s = vec![0.0f64; n_items];
        let g = vec![0.0f64; n_items];
        let n = 400usize;
        let profiles: Vec<usize> = (0..n).map(|j| j % 4).collect();
        let mut rng = Lcg(12345);
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
                .unwrap();
        assert!(res.converged);
        assert!(nondecreasing(&res.loglik_trace));
        assert!(monotone_items(&res));
        assert!(pattern_agreement(&res.map_profile, &profiles) > 0.99);
        assert!(res.slip.iter().all(|&s| s < 1e-2), "slip {:?}", res.slip);
        assert!(res.guess.iter().all(|&g| g < 1e-2), "guess {:?}", res.guess);
    }

    /// Anchor 3: with a single-attribute-per-item Q, `(c & mask) == mask` and
    /// `(c & mask) != 0` coincide, so DINA and DINO share bit-identical eta and, from
    /// the deterministic init, must produce identical fits. Pure algebraic identity.
    #[test]
    fn anchor_dina_dino_gate_identity() {
        let (n_attr, n_items) = (2usize, 4usize);
        let q: Vec<u8> = vec![1, 0, /* */ 1, 0, /* */ 0, 1, /* */ 0, 1];
        let s = vec![0.15f64; n_items];
        let g = vec![0.2f64; n_items];
        let n = 500usize;
        let mut rng = Lcg(999);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = CdmConfig::default();
        let a = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg).unwrap();
        let b = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dino, &cfg).unwrap();
        assert!(rmse(&a.slip, &b.slip) < 1e-9);
        assert!(rmse(&a.guess, &b.guess) < 1e-9);
        assert!(rmse(&a.profile_prob, &b.profile_prob) < 1e-9);
    }

    /// Anchor 4: K=1, Q all-ones reduces to a 2-class latent-class model. Recover the
    /// master proportion, slip and guess.
    #[test]
    fn anchor_k1_two_class_reduction() {
        let (n_attr, n_items) = (1usize, 10usize);
        let q: Vec<u8> = vec![1u8; n_items];
        let (s_true, g_true, pi1) = (0.15f64, 0.2f64, 0.6f64);
        let s = vec![s_true; n_items];
        let g = vec![g_true; n_items];
        let n = 2000usize;
        let mut rng = Lcg(7);
        let profiles: Vec<usize> = (0..n).map(|_| if rng.next_f64() < pi1 { 1 } else { 0 }).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
                .unwrap();
        assert!(res.converged && monotone_items(&res));
        let mean_s = res.slip.iter().sum::<f64>() / n_items as f64;
        let mean_g = res.guess.iter().sum::<f64>() / n_items as f64;
        assert!((mean_s - s_true).abs() < 0.05, "mean slip {mean_s}");
        assert!((mean_g - g_true).abs() < 0.05, "mean guess {mean_g}");
        assert!((res.profile_prob[1] - pi1).abs() < 0.05, "pi1 {}", res.profile_prob[1]);
    }

    /// Tier-1 fast recovery guard: K=2, J=15, N=1000, s=g=0.2, identifiable Q.
    #[test]
    fn recovery_guard() {
        let (n_attr, n_items, n) = (2usize, 15usize, 1000usize);
        // 5 items {a0}, 5 items {a1}, 5 items {a0,a1}.
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..15 {
            if i < 5 {
                q[i * 2] = 1;
            } else if i < 10 {
                q[i * 2 + 1] = 1;
            } else {
                q[i * 2] = 1;
                q[i * 2 + 1] = 1;
            }
        }
        let s = vec![0.2f64; n_items];
        let g = vec![0.2f64; n_items];
        let mut rng = Lcg(2024);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
                .unwrap();
        assert!(res.converged);
        assert!(nondecreasing(&res.loglik_trace));
        assert!(monotone_items(&res));
        assert!(rmse(&res.slip, &s) < 0.05, "rmse slip {}", rmse(&res.slip, &s));
        assert!(rmse(&res.guess, &g) < 0.05, "rmse guess {}", rmse(&res.guess, &g));
        assert!(pattern_agreement(&res.map_profile, &profiles) > 0.80);
        assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85);
        assert_eq!(res.n_parameters, 2 * n_items + ((1 << n_attr) - 1));
    }

    /// Missing-data (MAR) path: masked cells are dropped from likelihood and counts.
    #[test]
    fn handles_missing_data() {
        let (n_attr, n_items, n) = (2usize, 8usize, 400usize);
        let q: Vec<u8> = vec![
            1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1,
        ];
        let s = vec![0.15f64; n_items];
        let g = vec![0.2f64; n_items];
        let mut rng = Lcg(555);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let mut observed = vec![true; n * n_items];
        for (idx, o) in observed.iter_mut().enumerate() {
            if rng.next_f64() < 0.2 {
                *o = false; // ~20% MCAR missing
            }
            let _ = idx;
        }
        let res =
            fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
                .unwrap();
        assert!(res.converged && monotone_items(&res));
        assert!(nondecreasing(&res.loglik_trace));
    }

    /// Directly exercise every M-step branch (normal, both count guards, projection).
    #[test]
    fn update_item_branches() {
        let cfg = CdmConfig::default();
        let mut s = vec![0.2, 0.2, 0.2, 0.2];
        let mut g = vec![0.2, 0.2, 0.2, 0.2];
        // 0: normal — masters mostly right, non-masters mostly wrong.
        // 1: I1 below floor -> keep previous slip.
        // 2: I0 below floor -> keep previous guess.
        // 3: monotonicity violation (masters worse than non-masters) -> projection.
        let i1 = vec![100.0, 1e-12, 100.0, 100.0];
        let r1 = vec![80.0, 0.0, 80.0, 20.0];
        let i0 = vec![100.0, 100.0, 1e-12, 100.0];
        let r0 = vec![20.0, 20.0, 0.0, 80.0];
        for i in 0..4 {
            update_item(i, &i1, &r1, &i0, &r0, &mut s, &mut g, &cfg);
        }
        assert!((s[0] - 0.2).abs() < 1e-9 && (g[0] - 0.2).abs() < 1e-9);
        assert!((s[1] - 0.2).abs() < 1e-9, "kept prev slip {}", s[1]); // guard held slip
        assert!((g[2] - 0.2).abs() < 1e-9, "kept prev guess {}", g[2]); // guard held guess
        assert!(1.0 - s[3] > g[3], "projection kept monotonicity: 1-s={} g={}", 1.0 - s[3], g[3]);
    }

    /// The non-converged exit path (max_iter reached without meeting tol).
    #[test]
    fn stops_at_max_iter() {
        let (n_attr, n_items, n) = (1usize, 4usize, 50usize);
        let q = vec![1u8; n_items];
        let s = vec![0.1f64; n_items];
        let g = vec![0.2f64; n_items];
        let mut rng = Lcg(3);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(2)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = CdmConfig { max_iter: 1, ..CdmConfig::default() };
        let res = fit_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg).unwrap();
        assert!(!res.converged);
        assert_eq!(res.n_iter, 1);
        assert_eq!(res.loglik_trace.len(), 2);
        assert!(nondecreasing(&res.loglik_trace));
    }

    /// Malformed inputs are rejected with `Err` (covers each validate branch).
    #[test]
    fn validate_rejects_malformed() {
        let q_ok = vec![1u8, 0, 0, 1];
        let y = vec![0.0f64; 2 * 2];
        let obs = vec![true; 4];
        let cfg = CdmConfig::default();
        let bad = |q: &[u8], y: &[f64], obs: &[bool], n: usize, j: usize, k: usize| {
            fit_cdm(y, obs, q, n, j, k, CdmModel::Dina, &cfg).is_err()
        };
        assert!(bad(&q_ok, &y, &obs, 0, 2, 2)); // n_persons < 1
        assert!(bad(&q_ok, &y, &obs, 2, 2, 0)); // K < 1
        assert!(bad(&vec![1u8; 2 * 16], &vec![0.0; 2 * 2], &vec![true; 4], 2, 2, 16)); // K > 15
        assert!(bad(&q_ok, &vec![0.0; 3], &obs, 2, 2, 2)); // y length
        assert!(bad(&q_ok, &y, &vec![true; 3], 2, 2, 2)); // observed length
        assert!(bad(&vec![1u8; 3], &y, &obs, 2, 2, 2)); // q length
        assert!(bad(&q_ok, &vec![2.0, 0.0, 0.0, 0.0], &obs, 2, 2, 2)); // y not in {0,1}
        assert!(bad(&vec![2u8, 0, 0, 1], &y, &obs, 2, 2, 2)); // q not in {0,1}
        assert!(bad(&vec![0u8, 0, 1, 1], &y, &obs, 2, 2, 2)); // all-zero Q row 0
        assert!(bad(&vec![1u8, 0, 1, 0], &y, &obs, 2, 2, 2)); // all-zero Q column 1
        // Item 1 is entirely missing, so its slip/guess cannot be estimated.
        assert!(bad(&q_ok, &y, &[true, false, true, false], 2, 2, 2));
        // A well-formed call still succeeds.
        assert!(fit_cdm(&y, &obs, &q_ok, 2, 2, 2, CdmModel::Dina, &cfg).is_ok());
    }

    #[test]
    fn validate_rejects_invalid_config() {
        let q = vec![1u8, 0, 0, 1];
        let y = vec![0.0f64; 4];
        let observed = vec![true; 4];
        let rejected = |cfg: CdmConfig| {
            fit_cdm(&y, &observed, &q, 2, 2, 2, CdmModel::Dina, &cfg).is_err()
        };
        assert!(rejected(CdmConfig { max_iter: 0, ..CdmConfig::default() }));
        assert!(rejected(CdmConfig { tol: f64::NAN, ..CdmConfig::default() }));
        assert!(rejected(CdmConfig { eps: 0.5, ..CdmConfig::default() }));
        assert!(rejected(CdmConfig {
            eps: 1e-3,
            mono_backoff: 2e-3,
            ..CdmConfig::default()
        }));
        assert!(rejected(CdmConfig { init_slip: f64::INFINITY, ..CdmConfig::default() }));
        assert!(rejected(CdmConfig { init_slip: 0.6, init_guess: 0.4, ..CdmConfig::default() }));
        assert!(rejected(CdmConfig { count_floor: -1.0, ..CdmConfig::default() }));
    }

    /// Literature-grade Monte-Carlo (>=500 reps): de la Torre (2009)-style design,
    /// recovering slip/guess (RMSE/bias) and attribute/pattern classification accuracy.
    /// Q is held to moderate complexity (1-2 attribute items) so the aggregate RMSE
    /// bound holds (a 3-attribute item shrinks the eta=1 group to ~N/8 and inflates SE).
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_cdm_recovery() {
        let (n_attr, n_items, n, reps) = (5usize, 30usize, 1000usize, 500usize);
        let l = 1usize << n_attr;
        // 20 single-attribute items (4 per attribute) + 10 two-attribute items (pairs).
        let mut q = vec![0u8; n_items * n_attr];
        for a in 0..5 {
            for r in 0..4 {
                q[(a * 4 + r) * n_attr + a] = 1;
            }
        }
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 3), (1, 4), (0, 4)];
        for (t, &(a, b)) in pairs.iter().enumerate() {
            q[(20 + t) * n_attr + a] = 1;
            q[(20 + t) * n_attr + b] = 1;
        }

        for (cond, &sg) in [0.1f64, 0.2].iter().enumerate() {
            let s_true = vec![sg; n_items];
            let g_true = vec![sg; n_items];
            let (mut sum_rs, mut sum_rg, mut sum_bs, mut sum_bg) = (0.0, 0.0, 0.0, 0.0);
            let (mut ss_rs, mut ss_rg) = (0.0, 0.0);
            let (mut sum_pat, mut sum_attr) = (0.0, 0.0);
            for rep in 0..reps {
                let seed = 0xD1B54A32D192ED03u64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((cond as u64 + 1) * 0x9E3779B97F4A7C15);
                let mut rng = Lcg(seed);
                let profiles: Vec<usize> = (0..n).map(|_| rng.profile(l)).collect();
                let y = simulate(CdmModel::Dina, &q, &s_true, &g_true, &profiles, n_items, n_attr, &mut rng);
                let observed = vec![true; n * n_items];
                let res = fit_cdm(
                    &y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default(),
                )
                .unwrap();
                let (rs, rg) = (rmse(&res.slip, &s_true), rmse(&res.guess, &g_true));
                sum_rs += rs;
                sum_rg += rg;
                ss_rs += rs * rs;
                ss_rg += rg * rg;
                sum_bs += bias(&res.slip, &s_true);
                sum_bg += bias(&res.guess, &g_true);
                sum_pat += pattern_agreement(&res.map_profile, &profiles);
                sum_attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
            }
            let r = reps as f64;
            let (m_rs, m_rg) = (sum_rs / r, sum_rg / r);
            let sd_rs = (ss_rs / r - m_rs * m_rs).max(0.0).sqrt();
            let sd_rg = (ss_rg / r - m_rg * m_rg).max(0.0).sqrt();
            println!(
                "s=g={:.1}: RMSE(s)={:.4}(SD {:.4}) RMSE(g)={:.4}(SD {:.4}) bias(s)={:.4} bias(g)={:.4} pattern={:.3} attribute={:.3}",
                sg, m_rs, sd_rs, m_rg, sd_rg, sum_bs / r, sum_bg / r, sum_pat / r, sum_attr / r
            );
            assert!(m_rs < 0.03, "mean RMSE(s) {m_rs} at s=g={sg}");
            assert!(m_rg < 0.03, "mean RMSE(g) {m_rg} at s=g={sg}");
            if sg == 0.1 {
                assert!(sum_attr / r > 0.90, "mean attribute agreement {} at s=g=0.1", sum_attr / r);
            }
        }
    }

    // ----- G-DINA (saturated) tests -----

    /// Build the ragged CSR layout (item_off, qmask, k_required) from a Q-matrix,
    /// matching fit_gdina exactly.
    fn gdina_layout(q: &[u8], n_items: usize, n_attr: usize) -> (Vec<usize>, Vec<usize>, Vec<u32>) {
        let mut qmask = vec![0usize; n_items];
        let mut kreq = vec![0u32; n_items];
        for i in 0..n_items {
            let m = qmask_of(q, i, n_attr);
            qmask[i] = m;
            kreq[i] = m.count_ones();
        }
        let mut off = vec![0usize; n_items + 1];
        for i in 0..n_items {
            off[i + 1] = off[i] + (1usize << kreq[i]);
        }
        (off, qmask, kreq)
    }

    /// Draw responses from a CSR-flat truth table, using the SAME reduce_class + item_off
    /// convention as the estimator so RMSE compares matched classes (spec fix 3).
    fn simulate_gdina(
        qmask: &[usize],
        item_off: &[usize],
        truth_p: &[f64],
        profiles: &[usize],
        n_items: usize,
        rng: &mut Lcg,
    ) -> Vec<f64> {
        let n = profiles.len();
        let mut y = vec![0.0f64; n * n_items];
        for j in 0..n {
            for i in 0..n_items {
                let l = reduce_class(profiles[j], qmask[i]);
                y[j * n_items + i] = rng.bern(truth_p[item_off[i] + l]);
            }
        }
        y
    }

    /// The all-mastered reduced class has the highest success probability per item.
    fn top_class_is_max(res: &GdinaResult) -> bool {
        (0..res.k_required.len()).all(|i| {
            let (a, b) = (res.item_off[i], res.item_off[i + 1]);
            let top = res.item_prob[b - 1];
            res.item_prob[a..b].iter().all(|&p| p <= top + 1e-9)
        })
    }

    /// reduce_class packs the required-attribute mastery bits LSB-ascending, and
    /// equals L_i-1 iff all required attributes are mastered (the DINA eta identity).
    #[test]
    fn gdina_reduce_class_matches_bruteforce() {
        for k in 1..=4usize {
            for qmask in 1..(1usize << k) {
                let li = 1usize << (qmask.count_ones());
                for c in 0..(1usize << k) {
                    let (mut expect, mut m) = (0usize, 0u32);
                    for bit in 0..k {
                        if (qmask >> bit) & 1 == 1 {
                            expect |= ((c >> bit) & 1) << m;
                            m += 1;
                        }
                    }
                    assert_eq!(reduce_class(c, qmask), expect);
                    assert_eq!(reduce_class(c, qmask) == li - 1, (c & qmask) == qmask);
                }
            }
        }
    }

    /// mobius_inverse_inplace is the exact inverse of the zeta subset-sum, and matches
    /// the explicit K=2 identity-link formulas.
    #[test]
    fn gdina_mobius_roundtrip() {
        let mut rng = Lcg(42);
        for ki in 1..=3u32 {
            let li = 1usize << ki;
            let p: Vec<f64> = (0..li).map(|_| 0.05 + 0.9 * rng.next_f64()).collect();
            let mut delta = p.clone();
            mobius_inverse_inplace(&mut delta, ki);
            for l in 0..li {
                // reconstruct p_l = sum_{S subset of l} delta_S
                let recon: f64 = (0..li).filter(|&s| (l & s) == s).map(|s| delta[s]).sum();
                assert!((recon - p[l]).abs() < 1e-12, "roundtrip K={ki} l={l}");
            }
        }
        let mut d = vec![0.2, 0.5, 0.6, 0.9]; // p00, p10, p01, p11
        mobius_inverse_inplace(&mut d, 2);
        assert!((d[0] - 0.2).abs() < 1e-12);
        assert!((d[1] - (0.5 - 0.2)).abs() < 1e-12);
        assert!((d[2] - (0.6 - 0.2)).abs() < 1e-12);
        assert!((d[3] - (0.9 - 0.5 - 0.6 + 0.2)).abs() < 1e-12);
    }

    /// Brute-force likelihood: the CSR log-space path equals a naive enumeration.
    #[test]
    fn gdina_brute_force_likelihood() {
        let (n_attr, n_items) = (2usize, 2usize);
        let l_full = 1usize << n_attr;
        let q: Vec<u8> = vec![1, 0, /* */ 1, 1]; // item 0: K=1, item 1: K=2
        let (item_off, qmask, _k) = gdina_layout(&q, n_items, n_attr);
        let total = item_off[n_items];
        let p = vec![0.15f64, 0.8, /* */ 0.1, 0.3, 0.4, 0.85];
        assert_eq!(p.len(), total);
        let mut red = vec![0u16; n_items * l_full];
        for i in 0..n_items {
            for c in 0..l_full {
                red[i * l_full + c] = reduce_class(c, qmask[i]) as u16;
            }
        }
        let (mut log_p1, mut log_p0) = (vec![0.0f64; total], vec![0.0f64; total]);
        for x in 0..total {
            log_p1[x] = p[x].ln();
            log_p0[x] = (1.0 - p[x]).ln();
        }
        let pi = [0.4f64, 0.2, 0.1, 0.3];
        let log_pi: Vec<f64> = pi.iter().map(|v| v.ln()).collect();
        let x = [1.0f64, 0.0];
        let observed = vec![true; n_items];
        let mut post = vec![0.0f64; l_full];
        let log_px = posterior_row_gdina(
            0, &x, &observed, n_items, l_full, &red, &log_p1, &log_p0, &item_off, &log_pi, &mut post,
        );
        let mut px = 0.0;
        for c in 0..l_full {
            let mut lik = pi[c];
            for i in 0..n_items {
                let pc = p[item_off[i] + reduce_class(c, qmask[i])];
                let xi = x[i];
                lik *= pc.powf(xi) * (1.0 - pc).powf(1.0 - xi);
            }
            px += lik;
        }
        assert!((log_px.exp() - px).abs() < 1e-12, "module {} vs naive {}", log_px.exp(), px);
        assert!((post.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// THE CRUX ANCHOR: DINA-generated data => the saturated fit recovers p = g for
    /// every non-top reduced class and 1-s at the top, so delta has only the intercept
    /// and the highest-order interaction nonzero (the exact DINA identity-link constraint).
    #[test]
    fn gdina_recovers_dina() {
        let (n_attr, n_items, n) = (2usize, 12usize, 2500usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..n_items {
            if i < 4 {
                q[i * 2] = 1;
            } else if i < 8 {
                q[i * 2 + 1] = 1;
            } else {
                q[i * 2] = 1;
                q[i * 2 + 1] = 1;
            }
        }
        let s = vec![0.15f64; n_items];
        let g = vec![0.2f64; n_items];
        let mut rng = Lcg(2011);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace) && top_class_is_max(&res));
        let (item_off, _qm, _k) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a, b) = (item_off[i], item_off[i + 1]);
            for l in a..b {
                truth[l] = g[i];
            }
            truth[b - 1] = 1.0 - s[i];
        }
        assert!(rmse(&res.item_prob, &truth) < 0.03, "DINA p RMSE {}", rmse(&res.item_prob, &truth));
        for i in 0..n_items {
            let (a, b) = (item_off[i], item_off[i + 1]);
            let d = &res.item_delta[a..b];
            assert!((d[0] - g[i]).abs() < 0.05, "delta0 {} vs g {}", d[0], g[i]);
            assert!((d[b - a - 1] - ((1.0 - s[i]) - g[i])).abs() < 0.05, "delta_full item {i}");
            for l in 1..(b - a - 1) {
                assert!(d[l].abs() < 0.05, "interior delta item {i} idx {l} = {}", d[l]);
            }
        }
    }

    /// DINO-generated data: p = g at the empty reduced class, 1-s elsewhere. Uses a
    /// mixed Q (single-attribute items identify the attributes; an all-two-attribute Q
    /// would leave profiles 10/01/11 response-equivalent under the OR gate).
    #[test]
    fn gdina_recovers_dino() {
        let (n_attr, n_items, n) = (2usize, 12usize, 2500usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..n_items {
            if i < 4 {
                q[i * 2] = 1;
            } else if i < 8 {
                q[i * 2 + 1] = 1;
            } else {
                q[i * 2] = 1;
                q[i * 2 + 1] = 1;
            }
        }
        let s = vec![0.15f64; n_items];
        let g = vec![0.2f64; n_items];
        let mut rng = Lcg(77);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate(CdmModel::Dino, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        let (item_off, _qm, _k) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a, b) = (item_off[i], item_off[i + 1]);
            for l in a..b {
                truth[l] = 1.0 - s[i];
            }
            truth[a] = g[i];
        }
        assert!(rmse(&res.item_prob, &truth) < 0.03, "DINO p RMSE {}", rmse(&res.item_prob, &truth));
    }

    /// A-CDM (additive) data: recover p and confirm the interaction delta is ~0.
    #[test]
    fn gdina_recovers_acdm() {
        let (n_attr, n_items, n) = (2usize, 10usize, 4000usize);
        let q = vec![1u8; n_items * n_attr];
        let base = [0.1f64, 0.35, 0.4, 0.65]; // additive: p11 = 0.1 + 0.25 + 0.3, no interaction
        let (item_off, qmask, _k) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            for l in 0..4 {
                truth[item_off[i] + l] = base[l];
            }
        }
        let mut rng = Lcg(303);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(rmse(&res.item_prob, &truth) < 0.05, "A-CDM p RMSE {}", rmse(&res.item_prob, &truth));
        // Additive truth => interaction terms are negligible RELATIVE to the main
        // effects (an interaction is a 4-probability contrast, so its absolute noise
        // (~0.05) makes a fixed bound flaky; the additivity claim is a small ratio).
        let (mut sum_int, mut sum_main) = (0.0, 0.0);
        for i in 0..n_items {
            let base = item_off[i];
            sum_int += res.item_delta[base + 3].abs(); // both-attribute interaction
            sum_main += (res.item_delta[base + 1].abs() + res.item_delta[base + 2].abs()) / 2.0;
        }
        assert!(sum_int / sum_main < 0.35, "A-CDM interaction/main ratio {}", sum_int / sum_main);
        assert!(top_class_is_max(&res));
    }

    /// Deterministic s=g=0 limit: ideal responses => exact pattern recovery.
    #[test]
    fn gdina_deterministic_limit() {
        let (n_attr, n_items, n) = (2usize, 3usize, 400usize);
        let q: Vec<u8> = vec![1, 0, /* */ 0, 1, /* */ 1, 1];
        let s = vec![0.0f64; n_items];
        let g = vec![0.0f64; n_items];
        let profiles: Vec<usize> = (0..n).map(|j| j % 4).collect();
        let mut rng = Lcg(9);
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, n_attr, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.converged && top_class_is_max(&res));
        assert!(pattern_agreement(&res.map_profile, &profiles) > 0.99);
    }

    /// Tier-1 fast recovery guard: K=2, J=15, N=1000, monotone saturated truth.
    #[test]
    fn gdina_recovery_guard() {
        let (n_attr, n_items, n) = (2usize, 15usize, 1000usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..15 {
            if i < 5 {
                q[i * 2] = 1;
            } else if i < 10 {
                q[i * 2 + 1] = 1;
            } else {
                q[i * 2] = 1;
                q[i * 2 + 1] = 1;
            }
        }
        let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let a = item_off[i];
            if kreq[i] == 1 {
                truth[a] = 0.2;
                truth[a + 1] = 0.8;
            } else {
                truth[a] = 0.2;
                truth[a + 1] = 0.5;
                truth[a + 2] = 0.55;
                truth[a + 3] = 0.85;
            }
        }
        let mut rng = Lcg(2024);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(rmse(&res.item_prob, &truth) < 0.05, "guard p RMSE {}", rmse(&res.item_prob, &truth));
        assert!(top_class_is_max(&res));
        assert!(pattern_agreement(&res.map_profile, &profiles) > 0.80);
        assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85);
        let total: usize = (0..n_items).map(|i| 1usize << kreq[i]).sum();
        assert_eq!(res.n_parameters, total + ((1 << n_attr) - 1));
    }

    /// Missing-at-random cells are dropped from both likelihood and reduced-class counts.
    #[test]
    fn gdina_handles_missing_data() {
        let (n_attr, n_items, n) = (2usize, 9usize, 500usize);
        let q: Vec<u8> = vec![
            1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1, /* */ 1, 1, /* */ 1, 0, /* */ 0, 1, /* */ 1, 1,
        ];
        let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let a = item_off[i];
            if kreq[i] == 1 {
                truth[a] = 0.2;
                truth[a + 1] = 0.8;
            } else {
                truth[a] = 0.15;
                truth[a + 1] = 0.5;
                truth[a + 2] = 0.55;
                truth[a + 3] = 0.85;
            }
        }
        let mut rng = Lcg(555);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << n_attr)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let mut observed = vec![true; n * n_items];
        for o in observed.iter_mut() {
            if rng.next_f64() < 0.2 {
                *o = false;
            }
        }
        let res = fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.converged && top_class_is_max(&res));
        assert!(nondecreasing(&res.loglik_trace));
    }

    /// Literature-grade Monte-Carlo (>=500 reps): de la Torre (2011)-style design.
    /// Attributes are drawn from a STOCHASTIC higher-order logistic model (de la Torre
    /// & Douglas, 2004) so every reduced class gets positive, correlated mass; RMSE(p)
    /// is mass-weighted so near-empty classes don't dominate (spec fixes 1 & 2). Q is
    /// held to 1-2 required attributes per item to keep the reduced classes populated.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_gdina_recovery() {
        let (n_attr, n_items, n, reps) = (5usize, 30usize, 1000usize, 500usize);
        let mut q = vec![0u8; n_items * n_attr];
        for a in 0..5 {
            for r in 0..4 {
                q[(a * 4 + r) * n_attr + a] = 1;
            }
        }
        let pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2), (1, 3), (2, 4), (0, 3), (1, 4), (0, 4)];
        for (t, &(a, b)) in pairs.iter().enumerate() {
            q[(20 + t) * n_attr + a] = 1;
            q[(20 + t) * n_attr + b] = 1;
        }
        let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
        let total = item_off[n_items];
        let bk = [-1.0f64, -0.5, 0.0, 0.5, 1.0];
        let lambda = 1.5f64;

        for &skew in [false, true].iter() {
            for &sg in [0.1f64, 0.2].iter() {
                // Additive monotone truth: p_il = sg + (1-2sg)*popcount(l)/K_i.
                let mut truth = vec![0.0f64; total];
                for i in 0..n_items {
                    let ki = kreq[i] as f64;
                    for l in 0..(item_off[i + 1] - item_off[i]) {
                        truth[item_off[i] + l] = sg + (1.0 - 2.0 * sg) * (l.count_ones() as f64) / ki;
                    }
                }
                let mut dtruth = truth.clone();
                for i in 0..n_items {
                    mobius_inverse_inplace(&mut dtruth[item_off[i]..item_off[i + 1]], kreq[i]);
                }
                let (mut sum_wp, mut sum_bp, mut sum_dp, mut sum_pat, mut sum_attr) =
                    (0.0, 0.0, 0.0, 0.0, 0.0);
                for rep in 0..reps {
                    let seed = 0xD1B54A32D192ED03u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 * 2 + (sg == 0.1) as u64 + 1) * 0x9E3779B97F4A7C15);
                    let mut rng = Lcg(seed);
                    let profiles: Vec<usize> = (0..n)
                        .map(|_| {
                            let theta =
                                if skew { -(rng.next_f64().max(1e-12)).ln() - 1.0 } else { rng.normal() };
                            let mut c = 0usize;
                            for k in 0..n_attr {
                                let pk = 1.0 / (1.0 + (-lambda * (theta - bk[k])).exp());
                                if rng.next_f64() < pk {
                                    c |= 1 << k;
                                }
                            }
                            c
                        })
                        .collect();
                    let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
                    let observed = vec![true; n * n_items];
                    let res =
                        fit_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
                    // mass-weighted RMSE(p): weight each class by realized frequency.
                    let mut mass = vec![0.0f64; total];
                    for &c in &profiles {
                        for i in 0..n_items {
                            mass[item_off[i] + reduce_class(c, qmask[i])] += 1.0;
                        }
                    }
                    let (mut num, mut den) = (0.0, 0.0);
                    for x in 0..total {
                        let e = res.item_prob[x] - truth[x];
                        num += mass[x] * e * e;
                        den += mass[x];
                    }
                    sum_wp += (num / den).sqrt();
                    sum_bp += bias(&res.item_prob, &truth);
                    sum_dp += rmse(&res.item_delta, &dtruth);
                    sum_pat += pattern_agreement(&res.map_profile, &profiles);
                    sum_attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
                }
                let r = reps as f64;
                println!(
                    "skew={} s=g={:.1}: wRMSE(p)={:.4} bias(p)={:.4} RMSE(delta)={:.4} pattern={:.3} attribute={:.3}",
                    skew, sg, sum_wp / r, sum_bp / r, sum_dp / r, sum_pat / r, sum_attr / r
                );
                assert!(sum_wp / r < 0.03, "mass-weighted RMSE(p) {} skew={skew} sg={sg}", sum_wp / r);
                if sg == 0.1 {
                    assert!(sum_attr / r > 0.90, "attribute agreement {} skew={skew}", sum_attr / r);
                }
            }
        }
    }

    // ----- Q-matrix validation (de la Torre & Chiu, 2016) tests -----

    /// A canonical K=3, 15-item Q-matrix: six single-attribute items (two per
    /// attribute), six two-attribute items (two per pair), three full-triple items.
    fn canonical_q3() -> Vec<u8> {
        let k = 3usize;
        let mut q = vec![0u8; 15 * k];
        let set = |q: &mut [u8], i: usize, attrs: &[usize]| {
            for &a in attrs {
                q[i * k + a] = 1;
            }
        };
        let rows: [&[usize]; 15] = [
            &[0], &[1], &[2], &[0], &[1], &[2], // singles
            &[0, 1], &[0, 2], &[1, 2], &[0, 1], &[0, 2], &[1, 2], // pairs
            &[0, 1, 2], &[0, 1, 2], &[0, 1, 2], // triples
        ];
        for (i, r) in rows.iter().enumerate() {
            set(&mut q, i, r);
        }
        q
    }

    fn q_rows_equal(a: &[u8], b: &[u8], i: usize, k: usize) -> bool {
        (0..k).all(|c| (a[i * k + c] != 0) == (b[i * k + c] != 0))
    }

    /// ANCHOR: DINA-generated data whose provisional Q is the TRUE Q must validate
    /// to itself — every item's true q-vector is the fewest-attribute vector whose
    /// PVAF clears the cutoff, so nothing is flagged.
    #[test]
    fn qval_true_q_validates_to_itself() {
        let (k, n_items, n) = (3usize, 15usize, 3000usize);
        let q = canonical_q3();
        let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
        let mut rng = Lcg(20240715);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
        let y = simulate(CdmModel::Dina, &q, &s, &g, &profiles, n_items, k, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            validate_q_matrix(&y, &observed, &q, n, n_items, k, 0.95, &CdmConfig::default()).unwrap();
        let correct = (0..n_items).filter(|&i| q_rows_equal(&res.suggested_q, &q, i, k)).count();
        assert!(correct >= n_items - 1, "recovered {correct}/{n_items} true q-vectors");
        // The true q-vector explains ~all the item variance.
        assert!(
            res.provisional_pvaf.iter().all(|&p| p > 0.9),
            "min provisional PVAF {}",
            res.provisional_pvaf.iter().cloned().fold(f64::INFINITY, f64::min)
        );
    }

    /// A provisional Q with BOTH under-specified pairs (one attribute dropped) and
    /// over-specified singles (one spurious attribute added) is corrected back to
    /// the truth, and exactly the mis-specified items are flagged.
    #[test]
    fn qval_corrects_over_and_under_specification() {
        let (k, n_items, n) = (3usize, 15usize, 4000usize);
        let truth = canonical_q3();
        let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
        let mut rng = Lcg(13579);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
        let y = simulate(CdmModel::Dina, &truth, &s, &g, &profiles, n_items, k, &mut rng);
        let observed = vec![true; n * n_items];

        // Mis-specify a FEW items only (the method needs the rest of the Q to keep
        // the attributes identified): over-specify singles 0 & 3, under-specify
        // pairs 6 & 9.
        let mut prov = truth.clone();
        prov[0 * k + 1] = 1; // item 0 {0} -> {0,1}
        prov[3 * k + 2] = 1; // item 3 {0} -> {0,2}
        prov[6 * k + 1] = 0; // item 6 {0,1} -> {0}
        prov[9 * k + 0] = 0; // item 9 {0,1} -> {1}
        let perturbed = [0usize, 3, 6, 9];

        let res = validate_q_matrix(&y, &observed, &prov, n, n_items, k, 0.95, &CdmConfig::default())
            .unwrap();
        let correct = (0..n_items).filter(|&i| q_rows_equal(&res.suggested_q, &truth, i, k)).count();
        assert!(correct >= n_items - 1, "corrected {correct}/{n_items} to truth");
        for &i in &perturbed {
            assert!(res.flagged[i], "item {i} was mis-specified but not flagged");
            assert!(
                q_rows_equal(&res.suggested_q, &truth, i, k),
                "item {i} not corrected back to truth"
            );
        }
    }

    #[test]
    fn qval_rejects_malformed() {
        let n = 4usize;
        let y = vec![0.0f64; n * 3];
        let obs = vec![true; n * 3];
        let q = vec![1u8; 3 * 2];
        // bad epsilon
        assert!(validate_q_matrix(&y, &obs, &q, n, 3, 2, 0.0, &CdmConfig::default()).is_err());
        assert!(validate_q_matrix(&y, &obs, &q, n, 3, 2, 1.5, &CdmConfig::default()).is_err());
        // n_attributes out of range
        assert!(validate_q_matrix(&y, &obs, &q, n, 3, 0, 0.95, &CdmConfig::default()).is_err());
        assert!(validate_q_matrix(&y, &obs, &[1u8; 3 * 11], n, 3, 11, 0.95, &CdmConfig::default())
            .is_err());
        // wrong provisional_q length
        assert!(validate_q_matrix(&y, &obs, &[1u8; 5], n, 3, 2, 0.95, &CdmConfig::default()).is_err());
        // non-binary provisional entry
        assert!(
            validate_q_matrix(&y, &obs, &[2, 0, 1, 1, 0, 1], n, 3, 2, 0.95, &CdmConfig::default())
                .is_err()
        );
    }

    #[test]
    fn qval_rejects_nonconverged_calibration() {
        let n = 8usize;
        let y = vec![
            0.0, 0.0, 0.0, // 00
            0.0, 1.0, 0.0, // 01
            1.0, 0.0, 0.0, // 10
            1.0, 1.0, 1.0, // 11
            0.0, 0.0, 0.0, // repeated response patterns keep every item observed
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            1.0, 1.0, 1.0,
        ];
        let observed = vec![true; y.len()];
        let q = vec![1, 0, 0, 1, 1, 1];
        let cfg = CdmConfig {
            max_iter: 1,
            tol: 1e-12,
            ..CdmConfig::default()
        };

        let err =
            validate_q_matrix(&y, &observed, &q, n, 3, 2, 0.95, &cfg).unwrap_err();
        assert!(err.contains("did not converge"), "unexpected error: {err}");
        assert!(
            err.contains("1 of 1 M-steps"),
            "unexpected error: {err}"
        );
        assert!(
            err.contains("tol = 1.000000e-12"),
            "unexpected error: {err}"
        );
    }

    /// Literature-grade Monte-Carlo (>=500 reps): recovery of the true Q-matrix by
    /// PVAF validation starting from a mis-specified provisional Q, under a uniform
    /// (independent) and a correlated/skew (higher-order) attribute distribution.
    /// Reported as a *procedure* recovery: per-item exact q-vector rate plus
    /// attribute-level true-positive / false-positive rates.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_qval_recovery_500() {
        let (k, n_items, n, reps) = (3usize, 15usize, 1000usize, 500usize);
        let truth = canonical_q3();
        let (s, g) = (vec![0.1f64; n_items], vec![0.1f64; n_items]);
        let bk = [-0.6f64, 0.0, 0.6];
        let lambda = 1.5f64;

        for &skew in [false, true].iter() {
            let (mut sum_qrec, mut sum_tpr, mut sum_fpr) = (0.0f64, 0.0f64, 0.0f64);
            for rep in 0..reps {
                let seed = 0x2545F4914F6CDD1Du64
                    .wrapping_mul(rep as u64 + 1)
                    .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15);
                let mut rng = Lcg(seed);
                // attribute profiles
                let profiles: Vec<usize> = (0..n)
                    .map(|_| {
                        if skew {
                            // correlated higher-order logistic (de la Torre & Douglas, 2004)
                            let theta = -(rng.next_f64().max(1e-12)).ln() - 1.0;
                            let mut c = 0usize;
                            for a in 0..k {
                                let pk = 1.0 / (1.0 + (-lambda * (theta - bk[a])).exp());
                                if rng.next_f64() < pk {
                                    c |= 1 << a;
                                }
                            }
                            c
                        } else {
                            rng.profile(1 << k) // independent uniform over classes
                        }
                    })
                    .collect();
                let y = simulate(CdmModel::Dina, &truth, &s, &g, &profiles, n_items, k, &mut rng);
                let observed = vec![true; n * n_items];

                // mis-specify ~1/6 of items (flip one attribute bit); the rest keep
                // the attributes identified, as the method requires.
                let mut prov = truth.clone();
                for i in 0..n_items {
                    if rng.next_f64() < 0.17 {
                        let a = (rng.next_f64() * k as f64) as usize % k;
                        prov[i * k + a] ^= 1;
                    }
                    // guard against an all-zero provisional row (validation needs >=1)
                    if (0..k).all(|a| prov[i * k + a] == 0) {
                        prov[i * k] = 1;
                    }
                }
                let res =
                    validate_q_matrix(&y, &observed, &prov, n, n_items, k, 0.95, &CdmConfig::default())
                        .unwrap();

                let mut qrec = 0usize;
                let (mut tp, mut fp, mut pos, mut neg) = (0usize, 0usize, 0usize, 0usize);
                for i in 0..n_items {
                    if q_rows_equal(&res.suggested_q, &truth, i, k) {
                        qrec += 1;
                    }
                    for a in 0..k {
                        let t = truth[i * k + a] != 0;
                        let hcap = res.suggested_q[i * k + a] != 0;
                        if t {
                            pos += 1;
                            if hcap {
                                tp += 1;
                            }
                        } else {
                            neg += 1;
                            if hcap {
                                fp += 1;
                            }
                        }
                    }
                }
                sum_qrec += qrec as f64 / n_items as f64;
                sum_tpr += tp as f64 / pos as f64;
                sum_fpr += fp as f64 / neg as f64;
            }
            let r = reps as f64;
            println!(
                "[qval MC skew={skew}] reps={reps} q-recovery={:.3} attr-TPR={:.3} attr-FPR={:.3}",
                sum_qrec / r,
                sum_tpr / r,
                sum_fpr / r
            );
            assert!(sum_qrec / r > 0.80, "q-vector recovery {} skew={skew}", sum_qrec / r);
            assert!(sum_tpr / r > 0.90, "attribute TPR {} skew={skew}", sum_tpr / r);
            assert!(sum_fpr / r < 0.10, "attribute FPR {} skew={skew}", sum_fpr / r);
        }
    }

    // ----- CDM item-level Wald model selection (de la Torre, 2011) tests -----

    /// K=2 Q with `n_single` single-attribute items per attribute (strong attribute
    /// identification keeps the complete-data Wald covariance accurate) plus
    /// `n_pair` two-attribute items (the ones the Wald test evaluates). The first
    /// `2*n_single` items are singletons; the pair items follow.
    fn wald_q2(n_single: usize, n_pair: usize) -> (Vec<u8>, usize) {
        let k = 2usize;
        let mut rows: Vec<[u8; 2]> = Vec::new();
        for _ in 0..n_single {
            rows.push([1, 0]);
        }
        for _ in 0..n_single {
            rows.push([0, 1]);
        }
        for _ in 0..n_pair {
            rows.push([1, 1]);
        }
        let n_items = rows.len();
        let mut q = vec![0u8; n_items * k];
        for (i, r) in rows.iter().enumerate() {
            q[i * k] = r[0];
            q[i * k + 1] = r[1];
        }
        (q, n_items)
    }

    /// CSR truth table for the K=2 scenario. Single items are 2PL-like (low/high);
    /// pair items follow `kind`: DINA (conjunctive), DINO (disjunctive), A-CDM
    /// (additive), or "sat" (main effects AND interaction, so no reduced model fits).
    fn wald_truth(
        q: &[u8],
        n_items: usize,
        kind: &str,
    ) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
        let (item_off, qmask, kreq) = gdina_layout(q, n_items, 2);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let a = item_off[i];
            if kreq[i] == 1 {
                truth[a] = 0.15;
                truth[a + 1] = 0.85;
            } else {
                // reduce_class layout: [none, a0, a1, both]
                let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
                let (p00, p10, p01, p11) = match kind {
                    "dina" => (0.15, 0.15, 0.15, 0.85), // conjunctive
                    "dino" => (0.15, 0.85, 0.85, 0.85), // disjunctive (any mastered -> 1-s)
                    "acdm" => (0.10, 0.45, 0.45, 0.80), // additive 0.1 + .35a0 + .35a1
                    // LLM: additive on the logit, logit(P) = -3 + 2 a0 + 2 a1. Chosen
                    // asymmetric (2*(-3)+2+2 = -2 != 0) so the four points are NOT
                    // reflection-symmetric about 0 -> genuinely identity-NONadditive
                    // (A-CDM must reject) yet exactly logit-additive (LLM must not). Also
                    // log-nonadditive (P10/P00 != P11/P01), so R-RUM rejects too.
                    "llm" => (sig(-3.0), sig(-1.0), sig(-1.0), sig(1.0)),
                    // R-RUM: additive on the log, P = pi* r0^(1-a0) r1^(1-a1) with
                    // pi*=0.92, r0=0.3, r1=0.4. Log-additive (P10/P00 = P11/P01 = 1/r0)
                    // but strongly identity- AND logit-NONadditive (the high pi* makes
                    // logit(P) depart from log(P) sharply), so only R-RUM survives.
                    "rrum" => (0.92 * 0.3 * 0.4, 0.92 * 0.4, 0.92 * 0.3, 0.92),
                    _ => (0.10, 0.35, 0.35, 0.90), // main effects + interaction (saturated)
                };
                truth[a] = p00;
                truth[a + 1] = p10;
                truth[a + 2] = p01;
                truth[a + 3] = p11;
            }
        }
        (item_off, qmask, truth)
    }

    /// DINA-generated pair items are classified as DINA (the conjunctive reduced
    /// model is not rejected while the additive one is).
    #[test]
    fn wald_dina_data_selects_dina() {
        let (q, n_items) = wald_q2(5, 8);
        let n = 5000usize;
        let first_pair = 10usize;
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "dina");
        let mut rng = Lcg(4011);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        assert_eq!(
            res.models,
            vec![
                "dina".to_string(),
                "dino".to_string(),
                "acdm".to_string(),
                "llm".to_string(),
                "rrum".to_string(),
            ]
        );
        let nm = res.models.len();
        let pair_dina = (first_pair..n_items).filter(|&i| res.selected[i] == 0).count();
        assert!(pair_dina >= 7, "DINA selected for {pair_dina}/8 pair items");
        // single-attribute items are trivial (df=0) -> saturated, NaN stats
        for i in 0..first_pair {
            assert_eq!(res.selected[i], -1);
            assert!(res.wald_stat[i * nm].is_nan());
        }
    }

    /// DINO-generated pair items are classified as DINO (the disjunctive reduced
    /// model is not rejected while DINA and A-CDM are). Exercises the general
    /// (non-coordinate) linear restriction and the DINA/DINO parameter-count tie.
    #[test]
    fn wald_dino_data_selects_dino() {
        let (q, n_items) = wald_q2(5, 8);
        let n = 8000usize;
        let first_pair = 10usize;
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "dino");
        let mut rng = Lcg(6060);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        let nm = res.models.len();
        let pair_dino = (first_pair..n_items).filter(|&i| res.selected[i] == 1).count();
        assert!(pair_dino >= 7, "DINO selected for {pair_dino}/8 pair items");
        // DINO and DINA both have df = 2^K - 2 = 2 at K=2
        assert_eq!(res.wald_df[first_pair * nm], 2); // DINA
        assert_eq!(res.wald_df[first_pair * nm + 1], 2); // DINO
    }

    /// Additive-generated pair items are classified as A-CDM (additive not rejected,
    /// conjunctive DINA and disjunctive DINO rejected). A-CDM is candidate index 2.
    #[test]
    fn wald_acdm_data_selects_acdm() {
        let (q, n_items) = wald_q2(5, 8);
        let n = 5000usize;
        let first_pair = 10usize;
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "acdm");
        let mut rng = Lcg(2027);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        let pair_acdm = (first_pair..n_items).filter(|&i| res.selected[i] == 2).count();
        assert!(pair_acdm >= 7, "A-CDM selected for {pair_acdm}/8 pair items");
    }

    /// Faithfulness anchor for the link-transformed reduced models. The LLM and R-RUM
    /// truths are constructed to be additive ONLY on their own link (logit / log) and
    /// genuinely NON-additive on the identity link, so a correct implementation must
    /// (a) select LLM (index 3) / R-RUM (index 4) and (b) *reject* the identity-link
    /// A-CDM (index 2) — a sign/identity bug in the Jacobian covariance or the
    /// transformed delta would collapse this distinction. This is deliberately a
    /// non-centered, non-trivial truth: A-CDM, LLM and R-RUM all cost 1+K parameters,
    /// so only the transform can break the tie.
    #[test]
    fn wald_llm_and_rrum_data_select_their_link() {
        let (q, n_items) = wald_q2(5, 8);
        let n = 8000usize;
        let first_pair = 10usize;

        // LLM truth (logit-additive; identity- and log-NONadditive) -> LLM selected.
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "llm");
        let mut rng = Lcg(770011);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        let nm = res.models.len();
        let pair_llm = (first_pair..n_items).filter(|&i| res.selected[i] == 3).count();
        assert!(pair_llm >= 7, "LLM selected for {pair_llm}/8 pair items");
        // The identity-link A-CDM must be rejected on these identity-nonadditive items.
        let acdm_rej = (first_pair..n_items).filter(|&i| res.p_value[i * nm + 2] < 0.05).count();
        assert!(acdm_rej >= 7, "A-CDM rejected on {acdm_rej}/8 LLM items (identity-nonadditive)");

        // R-RUM truth (log-additive; identity- and logit-NONadditive) -> R-RUM selected.
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "rrum");
        let mut rng = Lcg(880022);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        let pair_rrum = (first_pair..n_items).filter(|&i| res.selected[i] == 4).count();
        assert!(pair_rrum >= 7, "R-RUM selected for {pair_rrum}/8 pair items");
        // The logit-link LLM must be rejected on these logit-nonadditive items.
        let llm_rej = (first_pair..n_items).filter(|&i| res.p_value[i * nm + 3] < 0.05).count();
        assert!(llm_rej >= 7, "LLM rejected on {llm_rej}/8 R-RUM items (logit-nonadditive)");
    }

    /// Items with both main effects and an interaction reject every reduced model,
    /// so the saturated G-DINA is kept.
    #[test]
    fn wald_saturated_data_selects_saturated() {
        let (q, n_items) = wald_q2(5, 8);
        let n = 5000usize;
        let first_pair = 10usize;
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "sat");
        let mut rng = Lcg(9091);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &CdmConfig::default())
                .unwrap();
        let nm = res.models.len();
        let pair_sat = (first_pair..n_items).filter(|&i| res.selected[i] == -1).count();
        assert!(pair_sat >= 7, "saturated kept for {pair_sat}/8 pair items");
        // every reduced model (DINA/DINO/A-CDM/LLM/R-RUM) carries a positive, finite stat
        for i in first_pair..n_items {
            for m in 0..nm {
                assert!(res.wald_stat[i * nm + m].is_finite() && res.wald_stat[i * nm + m] >= 0.0);
                assert!(res.p_value[i * nm + m].is_finite());
            }
        }
    }

    /// Degrees of freedom are exactly the restriction sizes: DINA & DINO df = 2^K-2,
    /// A-CDM df = 2^K-1-K, for K=3 items.
    #[test]
    fn wald_degrees_of_freedom() {
        // K=3 Q: single items (identification) + one triple item to read df off.
        let k = 3usize;
        let mut rows: Vec<[u8; 3]> = Vec::new();
        for a in 0..3 {
            for _ in 0..3 {
                let mut r = [0u8; 3];
                r[a] = 1;
                rows.push(r);
            }
        }
        rows.push([1, 1, 1]); // one K=3 item
        let n_items = rows.len();
        let mut q = vec![0u8; n_items * k];
        for (i, r) in rows.iter().enumerate() {
            q[i * k..i * k + k].copy_from_slice(r);
        }
        let n = 3000usize;
        let (item_off, qmask, _kr) = gdina_layout(&q, n_items, k);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let a = item_off[i];
            let w = item_off[i + 1] - a;
            for l in 0..w {
                truth[a + l] = 0.15 + 0.7 * (l.count_ones() as f64) / (w.trailing_zeros() as f64);
            }
        }
        let mut rng = Lcg(31337);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(1 << k)).collect();
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let res =
            gdina_wald_selection(&y, &observed, &q, n, n_items, k, 0.05, &CdmConfig::default())
                .unwrap();
        let nm = res.models.len();
        let triple = n_items - 1;
        assert_eq!(res.wald_df[triple * nm], (1 << k) - 2, "DINA df"); // 6
        assert_eq!(res.wald_df[triple * nm + 1], (1 << k) - 2, "DINO df"); // 6
        assert_eq!(res.wald_df[triple * nm + 2], (1 << k) - 1 - k, "A-CDM df"); // 4
        assert_eq!(res.wald_df[triple * nm + 3], (1 << k) - 1 - k, "LLM df"); // 4
        assert_eq!(res.wald_df[triple * nm + 4], (1 << k) - 1 - k, "R-RUM df"); // 4
        // single-attribute items: no test (df=0), saturated
        assert_eq!(res.wald_df[0], 0);
        assert_eq!(res.selected[0], -1);
    }

    #[test]
    fn wald_rejects_malformed() {
        let (q, n_items) = wald_q2(2, 2);
        let n = 10usize;
        let y = vec![0.0f64; n * n_items];
        let obs = vec![true; n * n_items];
        // alpha out of (0,1)
        assert!(gdina_wald_selection(&y, &obs, &q, n, n_items, 2, 0.0, &CdmConfig::default()).is_err());
        assert!(gdina_wald_selection(&y, &obs, &q, n, n_items, 2, 1.0, &CdmConfig::default()).is_err());
        // shape errors are delegated to fit_gdina's validate
        assert!(gdina_wald_selection(&y[..5], &obs, &q, n, n_items, 2, 0.05, &CdmConfig::default())
            .is_err());
    }

    #[test]
    fn wald_rejects_nonconverged_gdina_calibration() {
        let (q, n_items) = wald_q2(2, 2);
        let n = 80usize;
        let mut rng = Lcg(20260715);
        let profiles: Vec<usize> = (0..n).map(|_| rng.profile(4)).collect();
        let (item_off, qmask, truth) = wald_truth(&q, n_items, "dina");
        let y = simulate_gdina(&qmask, &item_off, &truth, &profiles, n_items, &mut rng);
        let observed = vec![true; n * n_items];
        let cfg = CdmConfig { max_iter: 1, tol: 1e-12, ..CdmConfig::default() };

        let err = gdina_wald_selection(&y, &observed, &q, n, n_items, 2, 0.05, &cfg)
            .expect_err("Wald selection must not use unfinished G-DINA parameters");
        assert!(err.contains("G-DINA calibration did not converge after 1 of 1 M-steps"));
        assert!(err.contains("final |delta loglik| ="));
        assert!(err.contains("tol = 1.000000e-12"));
    }

    /// Literature-grade Monte-Carlo (>=500 reps): Type I error (reject the TRUE
    /// reduced model ~ alpha) and power (reject a false, over-restrictive model),
    /// under uniform and correlated/skew attribute distributions.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_wald_type1_power_500() {
        let reps = 500usize;
        let (q, n_items) = wald_q2(5, 8);
        let n = 3000usize;
        let first_pair = 10usize;
        let k = 2usize;
        let bk = [-0.4f64, 0.4];
        let lambda = 1.5f64;
        let draw_profiles = |rng: &mut Lcg, skew: bool| -> Vec<usize> {
            (0..n)
                .map(|_| {
                    if skew {
                        let theta = -(rng.next_f64().max(1e-12)).ln() - 1.0;
                        let mut c = 0usize;
                        for a in 0..k {
                            let pk = 1.0 / (1.0 + (-lambda * (theta - bk[a])).exp());
                            if rng.next_f64() < pk {
                                c |= 1 << a;
                            }
                        }
                        c
                    } else {
                        rng.profile(1 << k)
                    }
                })
                .collect()
        };

        // Candidate columns: DINA=0, DINO=1, A-CDM=2, LLM=3, R-RUM=4.
        for &skew in [false, true].iter() {
            let (mut t1_acdm, mut t1_dina, mut t1_dino, mut t1_llm, mut t1_rrum) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            // Power of over-restrictive models against each additive-family truth: the
            // identity-link A-CDM and cross-link LLM/R-RUM must reject the wrong link.
            let (mut pow_dina, mut pow_dino, mut pow_acdm_llm, mut pow_rrum_llm, mut pow_llm_rrum) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
            let mut den = 0.0f64;
            for rep in 0..reps {
                let mut rng = Lcg(
                    0x9E3779B97F4A7C15u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0xD1B54A32D192ED03),
                );
                let obs = vec![true; n * n_items];
                let run = |kind: &str, rng: &mut Lcg| {
                    let (io, qm, tr) = wald_truth(&q, n_items, kind);
                    let prof = draw_profiles(rng, skew);
                    let y = simulate_gdina(&qm, &io, &tr, &prof, n_items, rng);
                    gdina_wald_selection(&y, &obs, &q, n, n_items, k, 0.05, &CdmConfig::default())
                        .unwrap()
                };
                // A-CDM truth: Type I of A-CDM (col 2) + power of the false DINA (col 0).
                let ra = run("acdm", &mut rng);
                // DINA truth: Type I of DINA (col 0) + power of the false DINO (col 1).
                let rd = run("dina", &mut rng);
                // DINO truth: Type I of DINO (col 1).
                let rn = run("dino", &mut rng);
                // LLM truth: Type I of LLM (col 3) + power of the false identity A-CDM
                // (col 2) and false log-link R-RUM (col 4).
                let rl = run("llm", &mut rng);
                // R-RUM truth: Type I of R-RUM (col 4) + power of the false logit LLM (col 3).
                let rr = run("rrum", &mut rng);
                let nm = ra.models.len();
                for i in first_pair..n_items {
                    if ra.p_value[i * nm + 2] < 0.05 {
                        t1_acdm += 1.0;
                    }
                    if ra.p_value[i * nm] < 0.05 {
                        pow_dina += 1.0; // DINA false under A-CDM truth
                    }
                    if rd.p_value[i * nm] < 0.05 {
                        t1_dina += 1.0;
                    }
                    if rd.p_value[i * nm + 1] < 0.05 {
                        pow_dino += 1.0; // DINO false under DINA truth
                    }
                    if rn.p_value[i * nm + 1] < 0.05 {
                        t1_dino += 1.0;
                    }
                    if rl.p_value[i * nm + 3] < 0.05 {
                        t1_llm += 1.0;
                    }
                    if rl.p_value[i * nm + 2] < 0.05 {
                        pow_acdm_llm += 1.0; // A-CDM false under LLM truth
                    }
                    if rl.p_value[i * nm + 4] < 0.05 {
                        pow_rrum_llm += 1.0; // R-RUM false under LLM truth
                    }
                    if rr.p_value[i * nm + 4] < 0.05 {
                        t1_rrum += 1.0;
                    }
                    if rr.p_value[i * nm + 3] < 0.05 {
                        pow_llm_rrum += 1.0; // LLM false under R-RUM truth
                    }
                    den += 1.0;
                }
            }
            println!(
                "[wald MC skew={skew}] reps={reps} TypeI(dina)={:.3} TypeI(dino)={:.3} \
                 TypeI(acdm)={:.3} TypeI(llm)={:.3} TypeI(rrum)={:.3} power(dina|acdm)={:.3} \
                 power(dino|dina)={:.3} power(acdm|llm)={:.3} power(rrum|llm)={:.3} \
                 power(llm|rrum)={:.3}",
                t1_dina / den,
                t1_dino / den,
                t1_acdm / den,
                t1_llm / den,
                t1_rrum / den,
                pow_dina / den,
                pow_dino / den,
                pow_acdm_llm / den,
                pow_rrum_llm / den,
                pow_llm_rrum / den
            );
            // Complete-data covariance is mildly liberal; allow up to ~2.5x nominal.
            assert!(t1_acdm / den < 0.13, "A-CDM Type I {}", t1_acdm / den);
            assert!(t1_dina / den < 0.13, "DINA Type I {}", t1_dina / den);
            assert!(t1_dino / den < 0.13, "DINO Type I {}", t1_dino / den);
            assert!(t1_llm / den < 0.13, "LLM Type I {}", t1_llm / den);
            assert!(t1_rrum / den < 0.13, "R-RUM Type I {}", t1_rrum / den);
            assert!(pow_dina / den > 0.95, "DINA power {}", pow_dina / den);
            assert!(pow_dino / den > 0.95, "DINO power {}", pow_dino / den);
            assert!(pow_acdm_llm / den > 0.95, "A-CDM|LLM power {}", pow_acdm_llm / den);
            assert!(pow_rrum_llm / den > 0.90, "R-RUM|LLM power {}", pow_rrum_llm / den);
            assert!(pow_llm_rrum / den > 0.90, "LLM|R-RUM power {}", pow_llm_rrum / den);
        }
    }

    // ----- Higher-order structured attribute prior (de la Torre & Douglas, 2004) -----

    /// Simulate higher-order DINA data: theta -> attribute mastery via
    /// sigmoid(a_k theta + d_k), then the DINA gate with slip/guess.
    #[allow(clippy::too_many_arguments)]
    fn simulate_ho_dina(
        a: &[f64],
        d: &[f64],
        s: &[f64],
        g: &[f64],
        q: &[u8],
        n: usize,
        n_items: usize,
        n_attr: usize,
        skew: bool,
        rng: &mut Lcg,
    ) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let mut y = vec![0.0f64; n * n_items];
        let mut profiles = vec![0usize; n];
        let mut thetas = vec![0.0f64; n];
        for j in 0..n {
            let theta = if skew {
                // standardized shifted chi-square(3): mean 0, var 1, right-skewed
                let mut cc = 0.0;
                for _ in 0..3 {
                    let z = rng.normal();
                    cc += z * z;
                }
                (cc - 3.0) / (6.0_f64).sqrt()
            } else {
                rng.normal()
            };
            thetas[j] = theta;
            let mut c = 0usize;
            for k in 0..n_attr {
                let p = 1.0 / (1.0 + (-(a[k] * theta + d[k])).exp());
                if rng.next_f64() < p {
                    c |= 1 << k;
                }
            }
            profiles[j] = c;
            for i in 0..n_items {
                let mask = qmask_of(q, i, n_attr);
                let eta = (c & mask) == mask;
                let p = if eta { 1.0 - s[i] } else { g[i] };
                y[j * n_items + i] = rng.bern(p);
            }
        }
        (y, profiles, thetas)
    }

    fn corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for i in 0..x.len() {
            sxy += (x[i] - mx) * (y[i] - my);
            sxx += (x[i] - mx).powi(2);
            syy += (y[i] - my).powi(2);
        }
        sxy / (sxx.sqrt() * syy.sqrt())
    }

    /// ANCHOR: with every attribute slope zero, the implied class prior is exactly the
    /// independent-attribute Bernoulli product (theta drops out), bit-for-bit.
    #[test]
    fn ho_pi_independent_when_slope_zero() {
        let k = 3usize;
        let a = vec![0.0f64; k];
        let d = vec![0.7f64, -0.4, 0.2];
        let pi = ho_pi_from_params(&a, &d, k);
        let pk: Vec<f64> = d.iter().map(|&dk| 1.0 / (1.0 + (-dk).exp())).collect();
        for c in 0..(1 << k) {
            let mut prod = 1.0f64;
            for (bit, &p) in pk.iter().enumerate() {
                prod *= if (c >> bit) & 1 == 1 { p } else { 1.0 - p };
            }
            assert!((pi[c] - prod).abs() < 1e-12, "class {c}: {} vs {}", pi[c], prod);
        }
        assert!((pi.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    /// Higher-order DINA recovery: attribute slopes/intercepts, slip/guess, the trait,
    /// and attribute classification under a known higher-order structure.
    #[test]
    fn ho_recovers_params() {
        let (n_attr, n_items, n) = (3usize, 15usize, 4000usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..n_items {
            // 4 single-attribute items per attribute + 3 pair items
            if i < 12 {
                q[i * n_attr + (i / 4)] = 1;
            } else {
                q[i * n_attr + (i - 12)] = 1;
                q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
            }
        }
        let a_true = vec![1.2f64, 1.5, 0.9];
        let d_true = vec![0.3f64, -0.5, 0.6];
        let s = vec![0.12f64; n_items];
        let g = vec![0.12f64; n_items];
        let mut rng = Lcg(70424);
        let (y, profiles, thetas) =
            simulate_ho_dina(&a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
            .unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(res.n_parameters == 2 * n_items + 2 * n_attr);
        assert!((res.profile_prob.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // slip/guess
        assert!(rmse(&res.slip, &s) < 0.05, "slip RMSE {}", rmse(&res.slip, &s));
        assert!(rmse(&res.guess, &g) < 0.05, "guess RMSE {}", rmse(&res.guess, &g));
        // higher-order parameters (identified up to the N(0,1) trait scale)
        assert!(rmse(&res.attr_slope, &a_true) < 0.4, "a RMSE {}", rmse(&res.attr_slope, &a_true));
        assert!(rmse(&res.attr_intercept, &d_true) < 0.3, "d RMSE {}", rmse(&res.attr_intercept, &d_true));
        assert!(res.attr_slope.iter().all(|&x| x > 0.0));
        // trait recovery (EAP is shrunk, so correlation is the right metric)
        assert!(corr(&res.theta, &thetas) > 0.6, "theta corr {}", corr(&res.theta, &thetas));
        // attribute classification
        assert!(
            attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.85,
            "attribute agreement {}",
            attribute_agreement(&res.attr_prob, &profiles, n, n_attr)
        );
    }

    /// Data from independent attributes (all true slopes 0) -> the *implied class
    /// distribution* `pi_c` recovers the independent-attribute product. (The
    /// individual slopes are not the right target: independence is also consistent
    /// with a single nonzero slope, since one attribute loading on theta induces no
    /// cross-attribute correlation. The likelihood identifies only `pi_c`.)
    #[test]
    fn ho_independent_data_recovers_pi() {
        let (n_attr, n_items, n) = (3usize, 15usize, 4000usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..n_items {
            if i < 12 {
                q[i * n_attr + (i / 4)] = 1;
            } else {
                q[i * n_attr + (i - 12)] = 1;
                q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
            }
        }
        let a_true = vec![0.0f64; n_attr];
        let d_true = vec![0.4f64, -0.3, 0.2];
        let s = vec![0.1f64; n_items];
        let g = vec![0.1f64; n_items];
        let mut rng = Lcg(9021);
        let (y, _p, _t) =
            simulate_ho_dina(&a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
            .unwrap();
        let pi_true = ho_pi_from_params(&a_true, &d_true, n_attr);
        assert!(
            rmse(&res.profile_prob, &pi_true) < 0.03,
            "implied pi RMSE {}",
            rmse(&res.profile_prob, &pi_true)
        );
    }

    /// Single-attribute Q: DINA and DINO share the ideal-response gate, so the
    /// higher-order fits coincide. Also exercises missing-at-random data.
    #[test]
    fn ho_reduces_dino_and_handles_missing() {
        let (n_attr, n_items, n) = (2usize, 8usize, 1000usize);
        let q: Vec<u8> = (0..n_items)
            .flat_map(|i| if i % 2 == 0 { [1u8, 0] } else { [0u8, 1] })
            .collect();
        let a_true = vec![1.0f64, 1.0];
        let d_true = vec![0.0f64, 0.0];
        let s = vec![0.15f64; n_items];
        let g = vec![0.15f64; n_items];
        let mut rng = Lcg(4242);
        let (mut y, _p, _t) =
            simulate_ho_dina(&a_true, &d_true, &s, &g, &q, n, n_items, n_attr, false, &mut rng);
        let mut observed = vec![true; n * n_items];
        // DINA == DINO on single-attribute items
        let da = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
            .unwrap();
        let di = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dino, &CdmConfig::default())
            .unwrap();
        assert!(rmse(&da.slip, &di.slip) < 1e-9 && rmse(&da.guess, &di.guess) < 1e-9);
        // missing-at-random cells dropped, still converges
        for o in observed.iter_mut() {
            if rng.next_f64() < 0.15 {
                *o = false;
            }
        }
        for (idx, o) in observed.iter().enumerate() {
            if !o {
                y[idx] = 0.0;
            }
        }
        let rm = fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
            .unwrap();
        assert!(rm.loglik_trace.iter().all(|v| v.is_finite()));
    }

    /// Full structural Newton steps used to make the observed log-likelihood fall
    /// (seed 12) and could then satisfy `abs(delta) < tol` on a negative change,
    /// falsely reporting convergence (seed 6).
    #[test]
    fn ho_structural_newton_preserves_em_ascent() {
        let (n_attr, n_items, n) = (3usize, 9usize, 40usize);
        let q = vec![
            1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1,
            1, 0, 1,
        ];
        let item_prob = [0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        for (seed, max_iter) in [(12u64, 100usize), (6, 500)] {
            let mut rng = Lcg(seed);
            let mut y = vec![0.0; n * n_items];
            for j in 0..n {
                for i in 0..n_items {
                    y[j * n_items + i] = rng.bern(item_prob[i]);
                }
            }
            let observed = vec![true; y.len()];
            let cfg = CdmConfig { max_iter, ..CdmConfig::default() };
            let res = fit_ho_cdm(
                &y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &cfg,
            )
            .unwrap();
            assert!(
                nondecreasing(&res.loglik_trace),
                "higher-order GEM lowered log-likelihood for seed {seed}: {:?}",
                res.loglik_trace
            );
            if seed == 6 {
                let delta = res.loglik_trace[res.loglik_trace.len() - 1]
                    - res.loglik_trace[res.loglik_trace.len() - 2];
                assert!(res.converged, "safeguarded seed-6 fit did not converge");
                assert!(
                    (0.0..cfg.tol).contains(&delta),
                    "convergence must be a non-negative improvement below tol; delta={delta:e}"
                );
            }
        }
    }

    #[test]
    fn ho_validate_rejects_malformed() {
        let cfg = CdmConfig::default();
        // y length mismatch (expects n_persons * n_items = 2)
        assert!(fit_ho_cdm(&[0.0], &[true], &[1, 1], 1, 2, 1, CdmModel::Dina, &cfg).is_err());
        // all-zero Q column: attribute 1 measured by no item
        assert!(fit_ho_cdm(&[0.0, 1.0], &[true, true], &[1, 0, 1, 0], 1, 2, 2, CdmModel::Dina, &cfg)
            .is_err());
    }

    /// Literature-grade Monte-Carlo (>=500 reps): higher-order DINA parameter recovery
    /// under normal and skew (mis-specified prior) trait distributions.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_ho_recovery_500() {
        let (n_attr, n_items, n, reps) = (3usize, 15usize, 1000usize, 500usize);
        let mut q = vec![0u8; n_items * n_attr];
        for i in 0..n_items {
            if i < 12 {
                q[i * n_attr + (i / 4)] = 1;
            } else {
                q[i * n_attr + (i - 12)] = 1;
                q[i * n_attr + ((i - 12) + 1) % n_attr] = 1;
            }
        }
        let a_true = vec![1.2f64, 1.5, 0.9];
        let d_true = vec![0.3f64, -0.5, 0.6];
        let s = vec![0.12f64; n_items];
        let g = vec![0.12f64; n_items];
        for &skew in [false, true].iter() {
            let (mut ra, mut rd, mut ba, mut bd, mut attr, mut nconv) =
                (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
            for rep in 0..reps {
                let mut rng = Lcg(
                    0xA24BAED4963EE407u64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15),
                );
                let (y, profiles, _t) =
                    simulate_ho_dina(&a_true, &d_true, &s, &g, &q, n, n_items, n_attr, skew, &mut rng);
                let observed = vec![true; n * n_items];
                let res =
                    fit_ho_cdm(&y, &observed, &q, n, n_items, n_attr, CdmModel::Dina, &CdmConfig::default())
                        .unwrap();
                if res.converged {
                    nconv += 1;
                    ra += rmse(&res.attr_slope, &a_true);
                    rd += rmse(&res.attr_intercept, &d_true);
                    ba += bias(&res.attr_slope, &a_true);
                    bd += bias(&res.attr_intercept, &d_true);
                    attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr);
                }
            }
            let conv_rate = nconv as f64 / reps as f64;
            assert!(
                conv_rate >= 0.95,
                "higher-order MC convergence rate {conv_rate:.3} below 0.95 for skew={skew}"
            );
            let den = nconv as f64;
            ra /= den;
            rd /= den;
            ba /= den;
            bd /= den;
            attr /= den;
            println!(
                "[HO-DINA MC skew={skew}] reps={reps} converged={nconv} ({conv_rate:.3}) \
                 RMSE(a)={ra:.3} RMSE(d)={rd:.3} bias(a)={ba:.3} bias(d)={bd:.3} \
                 attr-agree={attr:.3}"
            );
            // The trait prior is fixed N(0,1); under a skewed true trait the
            // structural slope/intercept degrade (prior mis-specification, as in 2PL
            // MMLE), while the attribute classification stays robust. Observed:
            // normal RMSE(a)~0.28 / RMSE(d)~0.09; skew RMSE(a)~0.37 / RMSE(d)~0.18;
            // attribute agreement ~0.98 in both. Bounds are condition-specific.
            let (a_bound, d_bound) = if skew { (0.45, 0.25) } else { (0.32, 0.15) };
            assert!(ra < a_bound, "RMSE(a) {ra} skew={skew}");
            assert!(rd < d_bound, "RMSE(d) {rd} skew={skew}");
            assert!(attr > 0.90, "attribute agreement {attr} skew={skew}");
        }
    }

    // ----- Higher-order G-DINA (de la Torre & Douglas, 2004 x de la Torre, 2011) -----

    /// Simulate higher-order G-DINA data: theta -> attribute mastery via
    /// sigmoid(a_k theta + d_k), then draw responses from the SATURATED per-reduced-
    /// class truth table (CSR, indexed by reduce_class), returning (y, profiles, thetas).
    #[allow(clippy::too_many_arguments)]
    fn simulate_ho_gdina(
        a: &[f64],
        d: &[f64],
        qmask: &[usize],
        item_off: &[usize],
        truth_p: &[f64],
        n: usize,
        n_items: usize,
        n_attr: usize,
        skew: bool,
        rng: &mut Lcg,
    ) -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let mut y = vec![0.0f64; n * n_items];
        let mut profiles = vec![0usize; n];
        let mut thetas = vec![0.0f64; n];
        for j in 0..n {
            let theta = if skew {
                let mut cc = 0.0;
                for _ in 0..3 {
                    let z = rng.normal();
                    cc += z * z;
                }
                (cc - 3.0) / (6.0_f64).sqrt()
            } else {
                rng.normal()
            };
            thetas[j] = theta;
            let mut c = 0usize;
            for k in 0..n_attr {
                let pk = 1.0 / (1.0 + (-(a[k] * theta + d[k])).exp());
                if rng.next_f64() < pk {
                    c |= 1 << k;
                }
            }
            profiles[j] = c;
            for i in 0..n_items {
                let l = reduce_class(c, qmask[i]);
                y[j * n_items + i] = rng.bern(truth_p[item_off[i] + l]);
            }
        }
        (y, profiles, thetas)
    }

    /// A canonical K=3 Q: single-attribute items (identification) + pair + triple.
    fn hogdina_q3() -> Vec<u8> {
        let k = 3usize;
        let mut q = vec![0u8; 15 * k];
        let rows: [&[usize]; 15] = [
            &[0], &[1], &[2], &[0], &[1], &[2], &[0], &[1], &[2], // 9 singles
            &[0, 1], &[1, 2], &[0, 2], &[0, 1], &[1, 2], // 5 pairs
            &[0, 1, 2], // 1 triple
        ];
        for (i, r) in rows.iter().enumerate() {
            for &at in *r {
                q[i * k + at] = 1;
            }
        }
        q
    }

    /// NON-TRIVIAL anchor: HO structure with SATURATED item probs set to the DINA
    /// pattern (g off-top, 1-s at top). The free saturated fit recovers those probs
    /// (so the item-level identity-link delta shows the DINA pattern) and the
    /// higher-order (a, d).
    #[test]
    fn ho_gdina_recovers_dina_pattern() {
        let (n_attr, n_items, n) = (3usize, 15usize, 3000usize);
        let q = hogdina_q3();
        let (item_off, qmask, _kreq) = gdina_layout(&q, n_items, n_attr);
        let (s, g) = (0.15f64, 0.2f64);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a0, b0) = (item_off[i], item_off[i + 1]);
            for l in a0..b0 {
                truth[l] = g;
            }
            truth[b0 - 1] = 1.0 - s; // DINA: only the all-mastered reduced class is high
        }
        let a_true = vec![1.2f64, 1.5, 0.9];
        let d_true = vec![0.3f64, -0.5, 0.6];
        let mut rng = Lcg(20242011);
        let (y, profiles, thetas) =
            simulate_ho_gdina(&a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, false, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.converged && nondecreasing(&res.loglik_trace));
        assert!(res.n_parameters == item_off[n_items] + 2 * n_attr);
        // saturated item probs recover the DINA pattern
        assert!(rmse(&res.item_prob, &truth) < 0.04, "item p RMSE {}", rmse(&res.item_prob, &truth));
        // identity-link delta: intercept ~ g, top interaction ~ (1-s)-g, interior ~ 0
        for i in 0..n_items {
            let (a0, b0) = (item_off[i], item_off[i + 1]);
            let dl = &res.item_delta[a0..b0];
            assert!((dl[0] - g).abs() < 0.06, "delta0 item {i}");
            assert!((dl[b0 - a0 - 1] - ((1.0 - s) - g)).abs() < 0.06, "delta_full item {i}");
            for l in 1..(b0 - a0 - 1) {
                assert!(dl[l].abs() < 0.06, "interior delta item {i} idx {l}");
            }
        }
        // higher-order recovery (identified at K=3) + trait + classification
        assert!(rmse(&res.attr_slope, &a_true) < 0.45, "a RMSE {}", rmse(&res.attr_slope, &a_true));
        assert!(res.attr_slope.iter().all(|&x| x > 0.0));
        assert!(attribute_agreement(&res.attr_prob, &profiles, n, n_attr) > 0.9);
        let tc = {
            let corr = |x: &[f64], y: &[f64]| {
                let nn = x.len() as f64;
                let (mx, my) = (x.iter().sum::<f64>() / nn, y.iter().sum::<f64>() / nn);
                let (mut sxy, mut sx, mut sy) = (0.0, 0.0, 0.0);
                for i in 0..x.len() {
                    sxy += (x[i] - mx) * (y[i] - my);
                    sx += (x[i] - mx).powi(2);
                    sy += (y[i] - my).powi(2);
                }
                sxy / (sx.sqrt() * sy.sqrt())
            };
            corr(&res.theta, &thetas)
        };
        assert!(tc > 0.55, "theta corr {tc}");
    }

    /// Independent-attribute data (all slopes 0) -> the implied class distribution
    /// recovers the independent-attribute product (K=3; the identified quantity).
    #[test]
    fn ho_gdina_independent_recovers_pi() {
        let (n_attr, n_items, n) = (3usize, 15usize, 3000usize);
        let q = hogdina_q3();
        let (item_off, qmask, _kr) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a0, b0) = (item_off[i], item_off[i + 1]);
            for (li, l) in (a0..b0).enumerate() {
                truth[l] = 0.15 + 0.7 * (li.count_ones() as f64) / (b0 - a0).trailing_zeros() as f64;
            }
        }
        let a_true = vec![0.0f64; n_attr];
        let d_true = vec![0.4f64, -0.3, 0.2];
        let mut rng = Lcg(7777);
        let (y, _p, _t) =
            simulate_ho_gdina(&a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, false, &mut rng);
        let observed = vec![true; n * n_items];
        let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        let pi_true = ho_pi_from_params(&a_true, &d_true, n_attr);
        assert!(
            res.converged,
            "termination={} n_iter={} relative_change={} tolerance={} attr_slope={:?}",
            res.termination_reason,
            res.n_iter,
            res.final_relative_loglik_change,
            res.stopping_tolerance,
            res.attr_slope
        );
        assert_eq!(res.termination_reason, "tolerance_met");
        assert!(res.final_relative_loglik_change < res.stopping_tolerance);
        assert!(nondecreasing(&res.loglik_trace));
        println!(
            "[HO-GDINA independent] n_iter={} delta_loglik={:.3e} relative_delta={:.3e} tol={:.1e}",
            res.n_iter,
            res.final_loglik_change,
            res.final_relative_loglik_change,
            res.stopping_tolerance
        );
        assert!(rmse(&res.profile_prob, &pi_true) < 0.03, "pi RMSE {}", rmse(&res.profile_prob, &pi_true));
    }

    #[test]
    fn ho_gdina_handles_missing_and_validates() {
        let (n_attr, n_items, n) = (3usize, 15usize, 1000usize);
        let q = hogdina_q3();
        let (item_off, qmask, _kr) = gdina_layout(&q, n_items, n_attr);
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a0, b0) = (item_off[i], item_off[i + 1]);
            for l in a0..b0 {
                truth[l] = 0.2;
            }
            truth[b0 - 1] = 0.85;
        }
        let mut rng = Lcg(99);
        let (mut y, _p, _t) = simulate_ho_gdina(
            &[1.0, 1.0, 1.0], &[0.0, 0.0, 0.0], &qmask, &item_off, &truth, n, n_items, n_attr, false, &mut rng,
        );
        let mut observed = vec![true; n * n_items];
        for o in observed.iter_mut() {
            if rng.next_f64() < 0.15 {
                *o = false;
            }
        }
        for (idx, o) in observed.iter().enumerate() {
            if !o {
                y[idx] = 0.0;
            }
        }
        let res = fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
        assert!(res.loglik_trace.iter().all(|v| v.is_finite()));
        // malformed
        let cfg = CdmConfig::default();
        assert!(fit_ho_gdina(&[0.0], &[true], &[1, 1], 1, 2, 1, &cfg).is_err()); // y length mismatch
        assert!(fit_ho_gdina(&[0.0, 1.0], &[true, true], &[0, 0, 0, 0], 1, 2, 2, &cfg).is_err()); // all-zero Q row
        let err = fit_ho_gdina(
            &[0.0, 1.0, 1.0, 0.0],
            &[true; 4],
            &[1, 0, 0, 1],
            2,
            2,
            2,
            &cfg,
        )
        .unwrap_err();
        assert!(err.contains("at least 3 attributes"), "{err}");

        let one_step = fit_ho_gdina(
            &y,
            &observed,
            &q,
            n,
            n_items,
            n_attr,
            &CdmConfig { max_iter: 1, tol: 1e-12, ..CdmConfig::default() },
        )
        .unwrap();
        assert!(!one_step.converged);
        assert_eq!(one_step.n_iter, 1);
        assert_eq!(one_step.termination_reason, "max_iter_reached");
        assert!(one_step.final_loglik_change.is_finite());
        assert!(one_step.final_relative_loglik_change.is_finite());
    }

    /// Literature-grade Monte-Carlo (>=500 reps): higher-order G-DINA recovery of the
    /// saturated item probabilities and the higher-order parameters under a normal and
    /// a skewed (mis-specified prior) trait distribution.
    #[test]
    #[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
    fn mc_ho_gdina_recovery_500() {
        let (n_attr, n_items, n, reps) = (3usize, 15usize, 1500usize, 500usize);
        let q = hogdina_q3();
        let (item_off, qmask, kreq) = gdina_layout(&q, n_items, n_attr);
        // additive saturated truth: p_il = 0.15 + 0.7 * popcount(l)/K_i
        let mut truth = vec![0.0f64; item_off[n_items]];
        for i in 0..n_items {
            let (a0, b0) = (item_off[i], item_off[i + 1]);
            for (li, l) in (a0..b0).enumerate() {
                truth[l] = 0.15 + 0.7 * (li.count_ones() as f64) / kreq[i] as f64;
            }
        }
        let a_true = vec![1.2f64, 1.5, 0.9];
        let d_true = vec![0.3f64, -0.5, 0.6];
        for &skew in [false, true].iter() {
            let (mut wp, mut ra, mut attr, mut nconv) = (0.0f64, 0.0f64, 0.0f64, 0usize);
            for rep in 0..reps {
                let mut rng = Lcg(
                    0x27BB2EE687B0B0FDu64
                        .wrapping_mul(rep as u64 + 1)
                        .wrapping_add((skew as u64 + 1) * 0x9E3779B97F4A7C15),
                );
                let (y, profiles, _t) =
                    simulate_ho_gdina(&a_true, &d_true, &qmask, &item_off, &truth, n, n_items, n_attr, skew, &mut rng);
                let observed = vec![true; n * n_items];
                let res =
                    fit_ho_gdina(&y, &observed, &q, n, n_items, n_attr, &CdmConfig::default()).unwrap();
                if res.converged {
                    nconv += 1;
                }
                // mass-weighted RMSE(p) so near-empty classes don't dominate
                let mut mass = vec![0.0f64; item_off[n_items]];
                for &c in &profiles {
                    for i in 0..n_items {
                        mass[item_off[i] + reduce_class(c, qmask[i])] += 1.0;
                    }
                }
                let (mut num, mut den) = (0.0f64, 0.0f64);
                for x in 0..item_off[n_items] {
                    let e = res.item_prob[x] - truth[x];
                    num += mass[x] * e * e;
                    den += mass[x];
                }
                wp += (num / den).sqrt() / reps as f64;
                ra += rmse(&res.attr_slope, &a_true) / reps as f64;
                attr += attribute_agreement(&res.attr_prob, &profiles, n, n_attr) / reps as f64;
            }
            println!(
                "[HO-GDINA MC skew={skew}] reps={reps} conv={:.2} wRMSE(p)={:.4} RMSE(a)={:.3} attr-agree={:.3}",
                nconv as f64 / reps as f64,
                wp,
                ra,
                attr
            );
            assert_eq!(nconv, reps, "nonconverged replications: {} of {reps} (skew={skew})", reps - nconv);
            assert!(wp < 0.04, "wRMSE(p) {wp} skew={skew}");
            assert!(attr > 0.90, "attribute agreement {attr} skew={skew}");
        }
    }
}
