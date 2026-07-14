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
}
