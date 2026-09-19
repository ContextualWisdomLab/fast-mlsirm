//! Observed-information standard errors for the single-group polytomous
//! bifactor graded response model via the Oakes (1999) identity (stage 3 of
//! #1912), structured to extend to the stage-2 multigroup calibration.
//!
//! # Method
//!
//! At the marginal-ML solution, the observed-data log-likelihood `l(xi)`
//! satisfies the Oakes (1999, eq. 6, p. 480) identity, valid for every `xi`
//! (not only at the MLE):
//!
//! ```text
//! d^2 l(xi) / d xi d xi' = [d^2 Q(xi' | xi) / d xi' d xi'
//!                          + d^2 Q(xi' | xi) / d xi' d xi] at xi' = xi,
//! ```
//!
//! where `Q(xi' | xi)` is the EM criterion — the expected complete-data
//! log-likelihood under the posterior at `xi` (Oakes, 1999, p. 479).
//! Term A (the M-step Hessian at a FIXED posterior) is computed here
//! ANALYTICALLY from the graded-response cell Hessian
//! ([`crate::poly::grm_node_hessian`]); the cross term (Term B) reflects the
//! sensitivity of the imputed sufficient statistics to the posterior
//! (Oakes, 1999, section 3: "the second term ... reflects the sensitivity of
//! the imputed complete-data sufficient statistic to changes in the
//! hypothesized parameter value") and needs one E-step per perturbed
//! coordinate — `k + 1` E-steps total for `k` free item parameters, versus
//! `2k` marginal-score evaluations for a central difference of the observed
//! score (the same costing argument documented in `crate::oakes`).
//!
//! # Complete-data model (what `Q` is)
//!
//! The E-step is the Gibbons-Hedeker reduced E-step: the person marginal
//! factors per general node (Gibbons et al., 2007, eq. 15, "Marginal Maximum
//! Likelihood Estimation" section), so the expected category counts
//! `r_ijk(node)` per item/node/category are the complete-data sufficient
//! statistics (Gibbons et al., 2007, Appendix, eqs. A4-A6). Conditional on
//! those counts, `Q` separates per item (each item loads the general factor
//! plus at most one specific; Gibbons et al., 2007, eq. 9), hence Term A is
//! BLOCK-DIAGONAL per item while the cross term is dense. The estimator is
//! unpenalized marginal ML, so `Q` carries NO prior: the Newton `ridge` in
//! `bifactor_grm.rs` is Hessian conditioning only, not a parameter prior,
//! and is NOT part of the information.
//!
//! # Free parameters and multigroup extension
//!
//! The free vector is per-item `[a_G, a_S?, d_1..d_{K-1}]` (slopes
//! unconstrained; thresholds strictly decreasing). Population parameters are
//! fixed by identification (orthogonal `N(0, 1)` factors; Gibbons et al.,
//! 2007, "Model" section), so the vcov is conditional on them. The assembly
//! is written against the [`PosteriorProvider`] trait: stage 1 fills
//! expected counts from the single-group reduced E-step; the stage-2
//! multigroup calibration reuses the same assembly with group-specific
//! E-steps sharing this item-parameter packing (group means/variances stay
//! conditional, held at their MLE).
//!
//! # Failure reporting
//!
//! The observed information is returned ALWAYS. When it is not positive
//! definite, `positive_definite` is `false` with a `non_pd_reason`, and
//! `vcov`/`se` are `None` — never a generalized inverse, never absolute or
//! clipped values (#1912 acceptance criterion 3: failures are reported, not
//! substituted).
//!
//! # References (APA 7th ed.)
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use crate::bifactor_grm::{
    check_param_shapes, e_step, fill_logprob_tables, gh_rule, pack_params, validate, ItemParams,
    Validated,
};
use crate::poly::{grm_node_gradient, grm_node_hessian};

/// Configuration for [`bifactor_oakes_se`]. Every field is caller-owned and
/// range-validated; nothing is clamped.
#[derive(Clone, Copy, Debug)]
pub struct BifactorOakesConfig {
    /// Gauss-Hermite nodes for the general factor (#1929: any `q >= 1`; the
    /// eigensolve overflow guard fires in `gh_rule` when nodes are drawn).
    pub q_general: usize,
    /// Gauss-Hermite nodes per specific factor (#1929: any `q >= 1`).
    pub q_specific: usize,
    /// Relative finite-difference step for the Oakes cross term
    /// (`h_j = fd_step * (1 + |xi_j|)`); the complete-data gradient and
    /// Hessian are analytic, so this touches only the posterior sensitivity.
    pub fd_step: f64,
}

/// Result of [`bifactor_oakes_se`].
#[derive(Clone, Debug)]
pub struct BifactorOakesResult {
    /// Parameter labels in vector order (`a_general:{i}`, `a_specific:{i}`
    /// only for block items, `d:{i}:{k}`).
    pub labels: Vec<String>,
    /// Observed information, row-major `k x k`. ALWAYS returned, even when
    /// it is not positive definite (it is the factual curvature).
    pub information: Vec<f64>,
    /// Full item-parameter vcov (inverse information), row-major `k x k`.
    /// `None` when the information is not positive definite — never a
    /// generalized inverse or any other substitute.
    pub vcov: Option<Vec<f64>>,
    /// Standard errors (`sqrt(diag(vcov))`). `None` exactly when `vcov` is
    /// `None`.
    pub se: Option<Vec<f64>>,
    /// Whether the observed information is positive definite.
    pub positive_definite: bool,
    /// Why the information is not positive definite. `None` exactly when
    /// `positive_definite` is `true`.
    pub non_pd_reason: Option<String>,
}

/// Expected complete-data sufficient statistics for the Oakes assembly: per
/// item, per latent node, per category expected counts, with the node
/// coordinates. Stage 1 fills this from the Gibbons-Hedeker reduced E-step
/// (Gibbons et al., 2007, eq. 15); stage 2 (multigroup) fills one per group.
#[derive(Clone, Debug)]
pub(crate) struct OakesPosterior {
    /// Per item: `counts[node][k]`, `node = g * q_s + h` for block items,
    /// `node = g` for general-only items.
    pub counts: Vec<Vec<Vec<f64>>>,
    /// Per item: general-factor coordinates per count node.
    pub node_g: Vec<Vec<f64>>,
    /// Per item: specific-factor coordinates per count node (`0.0` for
    /// general-only items).
    pub node_s: Vec<Vec<f64>>,
}

/// One item's slot in the free vector: local order `[a_G, (a_S?), d..]`
/// mapped to global free indices. The per-item `Q` derivatives are written
/// against this layout, so the stage-2 multigroup provider reuses them
/// unchanged with its own global packing.
#[derive(Clone, Debug)]
pub(crate) struct ItemSpec {
    pub has_specific: bool,
    pub slots: Vec<usize>,
}

/// How the Oakes assembly re-runs the E-step at perturbed parameters (the
/// cross term, Oakes, 1999, section 3). Stage 1 uses the single-group
/// reduced E-step; stage 2 plugs group-specific E-steps sharing the same
/// item-parameter packing.
pub(crate) trait PosteriorProvider {
    fn posterior_at(&self, packed: &[f64]) -> Result<OakesPosterior, String>;
    fn free_len(&self) -> usize;
    fn labels(&self) -> Vec<String>;
    fn item_specs(&self) -> Vec<ItemSpec>;
}

/// Single-group [`PosteriorProvider`] (stage 1): the Gibbons-Hedeker reduced
/// E-step at the caller quadrature, sharing the item-parameter packing that
/// stage 2 (multigroup) reuses.
#[derive(Clone, Debug)]
pub(crate) struct Stage1Provider {
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
    v: Validated,
    tg: Vec<f64>,
    ts: Vec<f64>,
    log_wg: Vec<f64>,
    log_ws: Vec<f64>,
    qg: usize,
    qs: usize,
    specs: Vec<ItemSpec>,
    labels: Vec<String>,
}

impl Stage1Provider {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        y: &[usize],
        observed: Option<&[bool]>,
        specific_map: &[i32],
        n_persons: usize,
        n_items: usize,
        n_specific: usize,
        n_cat: usize,
        cfg: &BifactorOakesConfig,
    ) -> Result<Self, String> {
        if !cfg.fd_step.is_finite() || cfg.fd_step <= 0.0 {
            return Err("fd_step must be finite and positive".into());
        }
        // Reuse the stage-1 validator (loud `Err` on malformed input,
        // unobserved categories, unsupported quadrature — never clamped).
        // `validate` needs a `BifactorGrmConfig`; the EM budgets it also
        // checks are satisfied with inert values because this provider never
        // runs the EM — only the quadrature E-step.
        let iter_cfg = crate::bifactor_grm::BifactorGrmConfig {
            q_general: cfg.q_general,
            q_specific: cfg.q_specific,
            max_iter: 1,
            tol: 1.0,
            n_starts: 1,
            seed: 0,
            newton_iter: 1,
            ridge: 1e-8,
            // The Oakes assembly's E-step reruns are exact f64 scalar work
            // (the cross term needs analytic precision), never the f32 GPU
            // kernels.
            device: crate::Device::Cpu,
        };
        let v = validate(
            y,
            observed,
            specific_map,
            n_persons,
            n_items,
            n_specific,
            n_cat,
            &iter_cfg,
        )?;
        let (tg, wg) = gh_rule(cfg.q_general)?;
        let (ts, ws) = gh_rule(cfg.q_specific)?;
        let (qg, qs) = (tg.len(), ts.len());
        let mut specs = Vec::with_capacity(n_items);
        let mut labels = Vec::new();
        let mut cursor = 0usize;
        for i in 0..n_items {
            let mut slots = vec![cursor];
            labels.push(format!("a_general:{i}"));
            cursor += 1;
            let has_specific = v.item_block[i].is_some();
            if has_specific {
                slots.push(cursor);
                labels.push(format!("a_specific:{i}"));
                cursor += 1;
            }
            for k in 0..v.m1 {
                slots.push(cursor);
                labels.push(format!("d:{i}:{k}"));
                cursor += 1;
            }
            specs.push(ItemSpec {
                has_specific,
                slots,
            });
        }
        Ok(Self {
            y: y.to_vec(),
            observed: observed.map(|o| o.to_vec()),
            v,
            tg: tg.to_vec(),
            ts: ts.to_vec(),
            log_wg: wg.iter().map(|w| w.ln()).collect(),
            log_ws: ws.iter().map(|w| w.ln()).collect(),
            qg,
            qs,
            specs,
            labels,
        })
    }

    pub(crate) fn check_params(
        &self,
        a_general: &[f64],
        a_specific: &[f64],
        thresholds: &[f64],
    ) -> Result<(), String> {
        check_param_shapes(&self.v, a_general, a_specific, thresholds)
    }

    pub(crate) fn pack(
        &self,
        a_general: &[f64],
        a_specific: &[f64],
        thresholds: &[f64],
    ) -> Vec<f64> {
        let params = pack_params(&self.v, a_general, a_specific, thresholds);
        let mut out = vec![0.0f64; self.free_len()];
        for (i, spec) in self.specs.iter().enumerate() {
            out[spec.slots[0]] = params[i].a_g;
            let mut s = 1;
            if spec.has_specific {
                out[spec.slots[1]] = params[i].a_s.unwrap_or(0.0);
                s = 2;
            }
            for (j, d) in params[i].d.iter().enumerate() {
                out[spec.slots[s + j]] = *d;
            }
        }
        out
    }

    /// Unpack the free vector back to per-item working parameters.
    pub(crate) fn unpack(&self, packed: &[f64]) -> Vec<ItemParams> {
        self.specs
            .iter()
            .map(|spec| {
                let a_g = packed[spec.slots[0]];
                let (a_s, off) = if spec.has_specific {
                    (Some(packed[spec.slots[1]]), 2)
                } else {
                    (None, 1)
                };
                let d = spec.slots[off..].iter().map(|&s| packed[s]).collect();
                ItemParams { a_g, a_s, d }
            })
            .collect()
    }
}

impl PosteriorProvider for Stage1Provider {
    fn posterior_at(&self, packed: &[f64]) -> Result<OakesPosterior, String> {
        let params = self.unpack(packed);
        let tables = fill_logprob_tables(&self.v, &params, &self.tg, &self.ts, self.qg, self.qs);
        let (_, counts, _) = e_step(
            &self.v,
            &self.y,
            self.observed.as_deref(),
            &tables,
            &self.log_wg,
            &self.log_ws,
            self.qg,
            self.qs,
            &self.tg,
            &self.ts,
            crate::Device::Cpu,
        );
        let mut node_g = Vec::with_capacity(self.v.n_items);
        let mut node_s = Vec::with_capacity(self.v.n_items);
        for i in 0..self.v.n_items {
            if self.v.item_block[i].is_some() {
                let mut gg = Vec::with_capacity(self.qg * self.qs);
                let mut ss = Vec::with_capacity(self.qg * self.qs);
                for &t in self.tg.iter().take(self.qg) {
                    for &u in self.ts.iter().take(self.qs) {
                        gg.push(t);
                        ss.push(u);
                    }
                }
                node_g.push(gg);
                node_s.push(ss);
            } else {
                node_g.push(self.tg.clone());
                node_s.push(vec![0.0; self.qg]);
            }
        }
        Ok(OakesPosterior {
            counts,
            node_g,
            node_s,
        })
    }

    fn free_len(&self) -> usize {
        self.labels.len()
    }

    fn labels(&self) -> Vec<String> {
        self.labels.clone()
    }

    fn item_specs(&self) -> Vec<ItemSpec> {
        self.specs.clone()
    }
}

/// Analytic gradient of the expected complete-data log-likelihood `Q` at
/// FIXED posterior counts, w.r.t. the packed free vector. Per item, per
/// node, the GRM cell gradient ([`grm_node_gradient`]) is chained through
/// `eta(node) = a_G * tG + a_S * tS + d` exactly as the M-step gradient in
/// `bifactor_grm.rs` (`item_neg_ll_grad`, up to the overall sign).
pub(crate) fn q_gradient_analytic(
    packed: &[f64],
    posterior: &OakesPosterior,
    provider: &dyn PosteriorProvider,
) -> Vec<f64> {
    let specs = provider.item_specs();
    let k = provider.free_len();
    let mut grad = vec![0.0f64; k];
    // Recover per-item local params from the packed vector via slots.
    for (i, spec) in specs.iter().enumerate() {
        let a_g = packed[spec.slots[0]];
        let (a_s, off) = if spec.has_specific {
            (packed[spec.slots[1]], 2)
        } else {
            (0.0, 1)
        };
        let d: Vec<f64> = spec.slots[off..].iter().map(|&s| packed[s]).collect();
        let (tg, ts, counts) = (
            &posterior.node_g[i],
            &posterior.node_s[i],
            &posterior.counts[i],
        );
        for (node, cnt) in counts.iter().enumerate() {
            let base = a_g * tg[node] + a_s * ts[node];
            let (g_base, g_thr) = grm_node_gradient(base, &d, cnt);
            grad[spec.slots[0]] += g_base * tg[node];
            if spec.has_specific {
                grad[spec.slots[1]] += g_base * ts[node];
            }
            for (j, gj) in g_thr.iter().enumerate() {
                grad[spec.slots[off + j]] += gj;
            }
        }
    }
    grad
}

/// Analytic Hessian of `Q` at FIXED posterior counts (Term A of the Oakes
/// identity), row-major `k x k`, symmetric. Per item, per node, the GRM cell
/// Hessian ([`grm_node_hessian`]) is chained through
/// `eta(node) = a_G * tG + a_S * tS + d` (all second derivatives of `eta`
/// vanish, so only first-order chain terms appear):
///
/// ```text
/// H[a_G,a_G] += tG^2 * S_n,  H[a_S,a_S] += tS^2 * S_n,
/// H[a_G,a_S] += tG * tS * S_n,
/// H[a_G,d_j] += tG * R^{(n)}_j,  H[a_S,d_j] += tS * R^{(n)}_j,
/// H[d_j,d_l] += H^{(n)}_{jl},
/// ```
///
/// with `S_n` the node grand sum and `R^{(n)}_j` the node row sums. `Q`
/// separates per item given the counts, so the result is block-diagonal per
/// item.
pub(crate) fn q_hessian_analytic(
    packed: &[f64],
    posterior: &OakesPosterior,
    provider: &dyn PosteriorProvider,
) -> Vec<f64> {
    let specs = provider.item_specs();
    let k = provider.free_len();
    let mut hess = vec![0.0f64; k * k];
    for (i, spec) in specs.iter().enumerate() {
        let a_g = packed[spec.slots[0]];
        let (a_s, off) = if spec.has_specific {
            (packed[spec.slots[1]], 2)
        } else {
            (0.0, 1)
        };
        let d: Vec<f64> = spec.slots[off..].iter().map(|&s| packed[s]).collect();
        let (tg, ts, counts) = (
            &posterior.node_g[i],
            &posterior.node_s[i],
            &posterior.counts[i],
        );
        let (g_slot, s_slot, s_off) =
            (spec.slots[0], spec.has_specific.then(|| spec.slots[1]), off);
        for (node, cnt) in counts.iter().enumerate() {
            let base = a_g * tg[node] + a_s * ts[node];
            let (grand, row_sums, mat) = grm_node_hessian(base, &d, cnt);
            let (tgn, tsn) = (tg[node], ts[node]);
            hess[g_slot * k + g_slot] += tgn * tgn * grand;
            if let Some(ss) = s_slot {
                hess[ss * k + ss] += tsn * tsn * grand;
                let cross = tgn * tsn * grand;
                hess[g_slot * k + ss] += cross;
                hess[ss * k + g_slot] += cross;
            }
            for (j, rj) in row_sums.iter().enumerate() {
                let dj = spec.slots[s_off + j];
                let cg = tgn * rj;
                hess[g_slot * k + dj] += cg;
                hess[dj * k + g_slot] += cg;
                if let Some(ss) = s_slot {
                    let cs = tsn * rj;
                    hess[ss * k + dj] += cs;
                    hess[dj * k + ss] += cs;
                }
                for (l, hjl) in mat[j].iter().enumerate() {
                    hess[dj * k + spec.slots[s_off + l]] += hjl;
                }
            }
        }
    }
    hess
}

/// Cholesky-based inversion of a symmetric matrix with the crate's existing
/// singularity floor (`crate::oakes::invert` rejects pivots below `1e-12` in
/// absolute value). Returns `Err` naming the pivot when the matrix is not
/// (numerically) positive definite — the caller reports this as the
/// non-PD flag + reason, never as substituted SEs.
fn cholesky_inverse(info: &[f64], k: usize) -> Result<Vec<f64>, String> {
    const PIVOT_FLOOR: f64 = 1e-12;
    let mut lower = vec![0.0f64; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut acc = info[i * k + j];
            for t in 0..j {
                acc -= lower[i * k + t] * lower[j * k + t];
            }
            if i == j {
                if !acc.is_finite() {
                    return Err(format!(
                        "observed information is not positive definite \
                         (cholesky pivot {i} is non-finite); SEs unavailable"
                    ));
                }
                if acc <= PIVOT_FLOOR {
                    return Err(format!(
                        "observed information is not positive definite \
                         (cholesky pivot {i} = {acc:.3e} <= {PIVOT_FLOOR:.0e}); SEs unavailable"
                    ));
                }
                lower[i * k + i] = acc.sqrt();
            } else {
                lower[i * k + j] = acc / lower[j * k + j];
            }
        }
    }
    // Solve L L' X = I column by column (forward then backward substitution).
    let mut inv = vec![0.0f64; k * k];
    let (mut fwd, mut sol) = (vec![0.0f64; k], vec![0.0f64; k]);
    for col in 0..k {
        for i in 0..k {
            let mut acc = if i == col { 1.0 } else { 0.0 };
            for t in 0..i {
                acc -= lower[i * k + t] * fwd[t];
            }
            fwd[i] = acc / lower[i * k + i];
        }
        for i in (0..k).rev() {
            let mut acc = fwd[i];
            for t in (i + 1)..k {
                acc -= lower[t * k + i] * sol[t];
            }
            sol[i] = acc / lower[i * k + i];
        }
        for (r, s) in sol.iter().enumerate() {
            inv[r * k + col] = *s;
        }
    }
    Ok(inv)
}

/// Observed-information standard errors via the Oakes (1999, eq. 6) identity
/// at GIVEN item parameters (valid for every `xi`, not only the MLE).
/// `a_specific` must be exactly `0.0` for general-only items; `thresholds`
/// is row-major `n_items * (n_cat - 1)` strictly decreasing per item.
/// Quadrature counts and `fd_step` are caller arguments (validated, never
/// clamped).
#[allow(clippy::too_many_arguments)]
pub fn bifactor_oakes_se(
    a_general: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &BifactorOakesConfig,
) -> Result<BifactorOakesResult, String> {
    // #1929: no node-count cap. Only the lower bound (>= 1) is checked here;
    // `gh_rule` (called wherever nodes are actually generated) also guards
    // the eigensolve allocation against usize overflow for absurd q.
    if cfg.q_general < 1 {
        return Err(format!("q_general must be >= 1; got {}", cfg.q_general));
    }
    if cfg.q_specific < 1 {
        return Err(format!("q_specific must be >= 1; got {}", cfg.q_specific));
    }
    let provider = Stage1Provider::new(
        y,
        observed,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        cfg,
    )?;
    provider.check_params(a_general, a_specific, thresholds)?;
    let labels = provider.labels();
    let k = provider.free_len();
    let packed = provider.pack(a_general, a_specific, thresholds);

    // Term A: analytic M-step Hessian at the FIXED base posterior.
    let posterior0 = provider.posterior_at(&packed)?;
    let term_a = q_hessian_analytic(&packed, &posterior0, &provider);

    // Term B: cross derivative — forward FD over the posterior argument
    // (one E-step per coordinate), gradient evaluated at the base xi. A
    // perturbation can invert a tight threshold gap (making the perturbed
    // posterior non-finite); that surfaces as non-finite information and is
    // reported through the non-PD flag, never as Err.
    let g0 = q_gradient_analytic(&packed, &posterior0, &provider);
    let mut cross = vec![0.0f64; k * k];
    for j in 0..k {
        let hj = cfg.fd_step * (1.0 + packed[j].abs());
        let mut perturbed = packed.clone();
        perturbed[j] += hj;
        let posterior_p = provider.posterior_at(&perturbed)?;
        let gp = q_gradient_analytic(&packed, &posterior_p, &provider);
        for c in 0..k {
            cross[j * k + c] = (gp[c] - g0[c]) / hj;
        }
    }

    // Observed information = -(A + B) by Oakes (1999, eq. 6, p. 480),
    // symmetrized to remove forward-FD asymmetry (the exact sum is
    // symmetric). Note `cross[j * k + c]` holds d g_c / d xi_j, i.e. the
    // transpose of the mixed partials, so the average recovers (B+B')/2.
    let mut information = vec![0.0f64; k * k];
    for r in 0..k {
        for c in 0..k {
            information[r * k + c] = -0.5
                * (term_a[r * k + c] + cross[r * k + c] + term_a[c * k + r] + cross[c * k + r]);
        }
    }
    if information.iter().any(|v| !v.is_finite()) {
        return Ok(BifactorOakesResult {
            labels,
            information,
            vcov: None,
            se: None,
            positive_definite: false,
            non_pd_reason: Some(
                "observed information has non-finite entries; SEs unavailable".into(),
            ),
        });
    }
    match cholesky_inverse(&information, k) {
        Ok(vcov) => {
            // A successful Cholesky certifies positive definiteness, hence
            // every diagonal of the inverse is positive; take exact square
            // roots (no clipping — a non-positive diagonal here would be an
            // arithmetic bug, surfaced by the debug assertion, not data).
            let se: Vec<f64> = (0..k)
                .map(|j| {
                    debug_assert!(
                        vcov[j * k + j] > 0.0,
                        "Cholesky-certified PD inverse must have a positive diagonal"
                    );
                    vcov[j * k + j].sqrt()
                })
                .collect();
            Ok(BifactorOakesResult {
                labels,
                information,
                vcov: Some(vcov),
                se: Some(se),
                positive_definite: true,
                non_pd_reason: None,
            })
        }
        Err(reason) => Ok(BifactorOakesResult {
            labels,
            information,
            vcov: None,
            se: None,
            positive_definite: false,
            non_pd_reason: Some(reason),
        }),
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/bifactor_oakes_tests.rs"]
mod tests;
