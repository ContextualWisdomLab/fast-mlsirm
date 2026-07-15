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
//! Deferred (explicit non-goals): the general G-DINA/saturated CDM (de la Torre,
//! 2011), Q-matrix estimation or full identifiability certification, and
//! higher-order structured attribute priors (de la Torre & Douglas, 2004).
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
/// q-vector search, so `K` is capped at 10.
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
}
