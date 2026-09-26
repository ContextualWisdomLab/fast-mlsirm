//! Observed-information standard errors for the single-group polytomous
//! two-tier / confirmatory multi-primary graded response model via the
//! Oakes (1999) identity (#1992), extending the single-general bifactor
//! assembly in [`crate::bifactor_oakes`].
//!
//! # Method
//!
//! At any parameter point `xi` (not only the MLE), the observed-data
//! log-likelihood satisfies Oakes (1999, eq. 6, p. 480):
//!
//! ```text
//! d^2 l(xi) / d xi d xi' = [d^2 Q(xi' | xi) / d xi' d xi'
//!                          + d^2 Q(xi' | xi) / d xi' d xi] at xi' = xi,
//! ```
//!
//! where `Q(xi' | xi)` is the expected complete-data log-likelihood under
//! the posterior at `xi` (Oakes, 1999, p. 479). Term A (M-step Hessian at a
//! FIXED posterior) is analytic for item parameters via the graded cell
//! Hessian ([`crate::poly::grm_node_hessian`]) chained through the two-tier
//! linear predictor (Cai, Yang, & Hansen, 2011, eq. 6, p. 227); for the
//! Fisher-`z` primary correlations it is the Hessian of the expected
//! complete-data normal criterion at fixed `Sbar` (Bock–Aitkin M-step
//! principle; Cai et al., 2011, "Maximum Marginal Likelihood Estimation"
//! section). Term B (posterior sensitivity) uses one forward finite-
//! difference E-step per free coordinate — the same costing argument as
//! [`crate::bifactor_oakes`].
//!
//! # Model and free vector
//!
//! The response model is the confirmatory two-tier GRM of Cai (2010)
//! (full text read, pp. 583-584): correlated
//! primaries plus orthogonal specifics with dimension reduction over the
//! specific tier (Gibbons et al., 2007, eq. 15; Chalmers, 2026, mirt
//! `bfactor` documentation: `ncol(G) + 1` integration). The free vector is
//! per-item `[a_p for free primary slots..., a_S?, d_1..d_{K-1}]` followed
//! by Fisher-`z` primary correlations (`rho = tanh(z)`). Population
//! specific variances remain fixed at 1 by identification.
//!
//! # Failure reporting
//!
//! The observed information is ALWAYS returned. When it is not positive
//! definite, `positive_definite` is `false` with a `non_pd_reason`, and
//! `vcov`/`se` are `None` — never a generalized inverse, never clipped
//! values (same contract as [`crate::bifactor_oakes`]).
//!
//! # References (APA 7th ed.)
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188
//! (full text read: eq. 6, p. 480)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (full text read, pp. 583-584)
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350 (full text read: eq. 6–7, p. 227;
//! "Maximum Marginal Likelihood Estimation" section)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E.,
//! Bhaumik, D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., &
//! Stover, A. (2007). Full-information item bifactor analysis of graded
//! response data. *Applied Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485 (full text read: eq. 15)

use crate::poly::{grm_node_gradient, grm_node_hessian};
use crate::two_tier_grm::{
    build_primary_grid, cholesky_lower, chol_inverse, e_step, gh_rule, pack_params, phi_from_z,
    phi_neg_ll, reweighted_log_weights, validate, z_from_phi, ItemParams, TwoTierGrmConfig,
    Validated,
};

/// Configuration for [`two_tier_oakes_se`]. Every field is caller-owned;
/// nothing is clamped or defaulted (ADR-0028 / #1929).
#[derive(Clone, Copy, Debug)]
pub struct TwoTierOakesConfig {
    /// Estimate primary correlations; false excludes Phi from the free vector.
    pub estimate_primary_correlation: bool,
    /// Gauss–Hermite nodes per primary dimension (any `q >= 1`).
    pub q_primary: usize,
    /// Gauss–Hermite nodes per specific factor (any `q >= 1`).
    pub q_specific: usize,
    /// Relative finite-difference step for the Oakes cross term
    /// (`h_j = fd_step * (1 + |xi_j|)`).
    pub fd_step: f64,
}

/// Result of [`two_tier_oakes_se`].
#[derive(Clone, Debug)]
pub struct TwoTierOakesResult {
    pub labels: Vec<String>,
    /// Observed information, row-major `k x k` (always present).
    pub information: Vec<f64>,
    pub vcov: Option<Vec<f64>>,
    pub se: Option<Vec<f64>>,
    pub positive_definite: bool,
    pub non_pd_reason: Option<String>,
}

/// Expected complete-data sufficient statistics for the Oakes assembly.
#[derive(Clone, Debug)]
struct TwoTierPosterior {
    /// Per item: `counts[node][k]`.
    counts: Vec<Vec<Vec<f64>>>,
    /// Mean posterior primary second moment `Sbar` (row-major `p * p`).
    s_bar: Vec<f64>,
}

#[derive(Clone, Debug)]
struct ItemSpec {
    free_dims: Vec<usize>,
    has_specific: bool,
    /// Global free indices for `[a_p..., a_S?, d...]`.
    slots: Vec<usize>,
}

struct Provider {
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
    v: Validated,
    coords: Vec<f64>,
    log_w0: Vec<f64>,
    ts: Vec<f64>,
    log_ws: Vec<f64>,
    n_grid: usize,
    qs: usize,
    specs: Vec<ItemSpec>,
    labels: Vec<String>,
    n_phi: usize,
    phi_slots: Vec<usize>,
}

impl Provider {
    #[allow(clippy::too_many_arguments)]
    fn new(
        y: &[usize],
        observed: Option<&[bool]>,
        primary_map: &[bool],
        specific_map: &[i32],
        n_persons: usize,
        n_items: usize,
        n_primary: usize,
        n_specific: usize,
        n_cat: usize,
        cfg: &TwoTierOakesConfig,
    ) -> Result<Self, String> {
        if !cfg.fd_step.is_finite() || cfg.fd_step <= 0.0 {
            return Err("fd_step must be finite and positive".into());
        }
        if cfg.q_primary < 1 {
            return Err(format!("q_primary must be >= 1; got {}", cfg.q_primary));
        }
        if cfg.q_specific < 1 {
            return Err(format!("q_specific must be >= 1; got {}", cfg.q_specific));
        }
        let iter_cfg = TwoTierGrmConfig {
            estimate_primary_correlation: cfg.estimate_primary_correlation,
            q_primary: cfg.q_primary,
            q_specific: cfg.q_specific,
            max_iter: 1,
            tol: 1.0,
            n_starts: 1,
            seed: 0,
            newton_iter: 1,
            ridge: 1e-8,
        };
        let v = validate(
            y,
            observed,
            primary_map,
            specific_map,
            n_persons,
            n_items,
            n_primary,
            n_specific,
            n_cat,
            &iter_cfg,
        )?;
        let (tz, wz) = gh_rule(cfg.q_primary)?;
        let (ts, ws) = gh_rule(cfg.q_specific)?;
        let qs = ts.len();
        let n_grid = v.grid_size;
        let (coords, log_w0) = build_primary_grid(tz, wz, n_primary, n_grid);
        let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

        let mut specs = Vec::with_capacity(n_items);
        let mut labels = Vec::new();
        let mut cursor = 0usize;
        for i in 0..n_items {
            let mut slots = Vec::new();
            let free_dims = v.free_primaries[i].clone();
            for &dim in &free_dims {
                slots.push(cursor);
                labels.push(format!("a_primary:{i}:{dim}"));
                cursor += 1;
            }
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
                free_dims,
                has_specific,
                slots,
            });
        }
        let n_phi = if cfg.estimate_primary_correlation {
            n_primary * (n_primary.saturating_sub(1)) / 2
        } else {
            0
        };
        let mut phi_slots = Vec::with_capacity(n_phi);
        let mut t = 0usize;
        for i in 0..n_primary {
            for j in (i + 1)..n_primary {
                if !cfg.estimate_primary_correlation {
                    continue;
                }
                phi_slots.push(cursor);
                labels.push(format!("phi_z:{i}:{j}"));
                cursor += 1;
                t += 1;
            }
        }
        debug_assert_eq!(t, n_phi);
        Ok(Self {
            y: y.to_vec(),
            observed: observed.map(|o| o.to_vec()),
            v,
            coords,
            log_w0,
            ts: ts.to_vec(),
            log_ws,
            n_grid,
            qs,
            specs,
            labels,
            n_phi,
            phi_slots,
        })
    }

    fn free_len(&self) -> usize {
        self.labels.len()
    }

    fn pack(
        &self,
        a_primary: &[f64],
        a_specific: &[f64],
        thresholds: &[f64],
        phi: &[f64],
    ) -> Result<Vec<f64>, String> {
        let p = self.v.n_primary;
        let params = pack_params(&self.v, a_primary, a_specific, thresholds);
        let z = z_from_phi(phi, p)?;
        if self.n_phi == 0 && phi != phi_from_z(&vec![0.0; p * (p - 1) / 2], p) {
            return Err("fixed primary correlation requires phi = I".into());
        }
        let mut out = vec![0.0f64; self.free_len()];
        for (i, spec) in self.specs.iter().enumerate() {
            let mut s = 0usize;
            for &dim in &spec.free_dims {
                out[spec.slots[s]] = params[i].a_p[dim];
                s += 1;
            }
            if spec.has_specific {
                out[spec.slots[s]] = params[i].a_s.unwrap_or(0.0);
                s += 1;
            }
            for (j, d) in params[i].d.iter().enumerate() {
                out[spec.slots[s + j]] = *d;
            }
        }
        for (t, &slot) in self.phi_slots.iter().enumerate() {
            out[slot] = z[t];
        }
        Ok(out)
    }

    fn unpack_items_phi(&self, packed: &[f64]) -> (Vec<ItemParams>, Vec<f64>) {
        let p = self.v.n_primary;
        let mut items = Vec::with_capacity(self.v.n_items);
        for (i, spec) in self.specs.iter().enumerate() {
            let mut a_p = vec![0.0f64; p];
            let mut s = 0usize;
            for &dim in &spec.free_dims {
                a_p[dim] = packed[spec.slots[s]];
                s += 1;
            }
            let a_s = if spec.has_specific {
                let v = packed[spec.slots[s]];
                s += 1;
                Some(v)
            } else {
                None
            };
            let d: Vec<f64> = spec.slots[s..].iter().map(|&idx| packed[idx]).collect();
            let _ = i;
            items.push(ItemParams { a_p, a_s, d });
        }
        let z: Vec<f64> = if self.n_phi == 0 {
            vec![0.0; p * (p - 1) / 2]
        } else {
            self.phi_slots.iter().map(|&s| packed[s]).collect()
        };
        (items, z)
    }

    fn posterior_at(&self, packed: &[f64]) -> Result<TwoTierPosterior, String> {
        let (params, z) = self.unpack_items_phi(packed);
        let p = self.v.n_primary;
        let phi = phi_from_z(&z, p);
        let (l, logdet) = cholesky_lower(&phi, p)
            .ok_or_else(|| "primary correlation became non-PD during Oakes E-step".to_string())?;
        let phi_inv = chol_inverse(&l, p);
        let log_w = reweighted_log_weights(&self.log_w0, &self.coords, &phi_inv, logdet, p);
        let (ll, counts, s_bar_sum) = e_step(
            &self.v,
            &self.y,
            self.observed.as_deref(),
            &params,
            &log_w,
            &self.log_ws,
            &self.coords,
            &self.ts,
            self.n_grid,
            self.qs,
        );
        if !ll.is_finite() {
            return Err("non-finite loglik in Oakes E-step".into());
        }
        let mut s_bar = s_bar_sum;
        let n = self.v.n_persons as f64;
        for slot in s_bar.iter_mut() {
            *slot /= n;
        }
        Ok(TwoTierPosterior { counts, s_bar })
    }
}

/// Analytic complete-data gradient of `Q` at FIXED posterior (items + Phi).
fn q_gradient(
    packed: &[f64],
    posterior: &TwoTierPosterior,
    provider: &Provider,
) -> Vec<f64> {
    let k = provider.free_len();
    let p = provider.v.n_primary;
    let mut grad = vec![0.0f64; k];
    let (params, z) = provider.unpack_items_phi(packed);

    for (i, spec) in provider.specs.iter().enumerate() {
        let par = &params[i];
        let counts = &posterior.counts[i];
        let has_s = spec.has_specific;
        let off = spec.free_dims.len() + usize::from(has_s);
        for (node, cnt) in counts.iter().enumerate() {
            let (g, h) = if has_s {
                (node / provider.qs, node % provider.qs)
            } else {
                (node, 0)
            };
            let mut base = 0.0f64;
            for &dim in &spec.free_dims {
                base += par.a_p[dim] * provider.coords[g * p + dim];
            }
            if let Some(a_s) = par.a_s {
                base += a_s * provider.ts[h];
            }
            let (g_base, g_thr) = grm_node_gradient(base, &par.d, cnt);
            let mut s = 0usize;
            for &dim in &spec.free_dims {
                grad[spec.slots[s]] += g_base * provider.coords[g * p + dim];
                s += 1;
            }
            if has_s {
                grad[spec.slots[s]] += g_base * provider.ts[h];
                s += 1;
            }
            for (j, gj) in g_thr.iter().enumerate() {
                grad[spec.slots[off + j]] += gj;
                let _ = s;
            }
        }
    }

    // Phi: Q = -phi_neg_ll (up to additive constants independent of z).
    // Gradient via central FD at fixed Sbar.
    if provider.n_phi > 0 {
        let h = 1e-5;
        let n = provider.v.n_persons;
        let f0 = phi_neg_ll(&z, p, &posterior.s_bar, n);
        for (t, &slot) in provider.phi_slots.iter().enumerate() {
            let mut zp = z.clone();
            zp[t] += h;
            let fp = phi_neg_ll(&zp, p, &posterior.s_bar, n);
            let mut zm = z.clone();
            zm[t] -= h;
            let fm = phi_neg_ll(&zm, p, &posterior.s_bar, n);
            // dQ/dz = -d(phi_neg_ll)/dz
            let d_neg = if fp.is_finite() && fm.is_finite() {
                (fp - fm) / (2.0 * h)
            } else if fp.is_finite() && f0.is_finite() {
                (fp - f0) / h
            } else if fm.is_finite() && f0.is_finite() {
                (f0 - fm) / h
            } else {
                f64::NAN
            };
            grad[slot] = -d_neg;
        }
    }
    grad
}

/// Analytic / FD Hessian of `Q` at FIXED posterior (Term A).
fn q_hessian(
    packed: &[f64],
    posterior: &TwoTierPosterior,
    provider: &Provider,
) -> Vec<f64> {
    let k = provider.free_len();
    let p = provider.v.n_primary;
    let mut hess = vec![0.0f64; k * k];
    let (params, z) = provider.unpack_items_phi(packed);

    for (i, spec) in provider.specs.iter().enumerate() {
        let par = &params[i];
        let counts = &posterior.counts[i];
        let has_s = spec.has_specific;
        let n_free = spec.free_dims.len();
        let off = n_free + usize::from(has_s);
        for (node, cnt) in counts.iter().enumerate() {
            let (g, h) = if has_s {
                (node / provider.qs, node % provider.qs)
            } else {
                (node, 0)
            };
            let mut base = 0.0f64;
            for &dim in &spec.free_dims {
                base += par.a_p[dim] * provider.coords[g * p + dim];
            }
            let t_s = if has_s { provider.ts[h] } else { 0.0 };
            if let Some(a_s) = par.a_s {
                base += a_s * t_s;
            }
            let (grand, row_sums, mat) = grm_node_hessian(base, &par.d, cnt);
            // Chain through eta = sum_p a_p t_p + a_s t_s + d.
            for (r, &dim_r) in spec.free_dims.iter().enumerate() {
                let tr = provider.coords[g * p + dim_r];
                let sr = spec.slots[r];
                hess[sr * k + sr] += tr * tr * grand;
                for (c, &dim_c) in spec.free_dims.iter().enumerate().skip(r + 1) {
                    let tc = provider.coords[g * p + dim_c];
                    let sc = spec.slots[c];
                    let cross = tr * tc * grand;
                    hess[sr * k + sc] += cross;
                    hess[sc * k + sr] += cross;
                }
                if has_s {
                    let ss = spec.slots[n_free];
                    let cross = tr * t_s * grand;
                    hess[sr * k + ss] += cross;
                    hess[ss * k + sr] += cross;
                }
                for (j, rj) in row_sums.iter().enumerate() {
                    let dj = spec.slots[off + j];
                    let cg = tr * rj;
                    hess[sr * k + dj] += cg;
                    hess[dj * k + sr] += cg;
                }
            }
            if has_s {
                let ss = spec.slots[n_free];
                hess[ss * k + ss] += t_s * t_s * grand;
                for (j, rj) in row_sums.iter().enumerate() {
                    let dj = spec.slots[off + j];
                    let cs = t_s * rj;
                    hess[ss * k + dj] += cs;
                    hess[dj * k + ss] += cs;
                }
            }
            for (j, row) in mat.iter().enumerate() {
                let dj = spec.slots[off + j];
                for (l, hjl) in row.iter().enumerate() {
                    hess[dj * k + spec.slots[off + l]] += hjl;
                }
            }
        }
    }

    // Phi block of Term A: FD Hessian of Q = -phi_neg_ll at fixed Sbar.
    if provider.n_phi > 0 {
        let h = 1e-5;
        let n = provider.v.n_persons;
        let m = provider.n_phi;
        let mut g0 = vec![0.0f64; m];
        let f0 = phi_neg_ll(&z, p, &posterior.s_bar, n);
        for t in 0..m {
            let mut zp = z.clone();
            zp[t] += h;
            let fp = phi_neg_ll(&zp, p, &posterior.s_bar, n);
            let mut zm = z.clone();
            zm[t] -= h;
            let fm = phi_neg_ll(&zm, p, &posterior.s_bar, n);
            g0[t] = if fp.is_finite() && fm.is_finite() {
                (fp - fm) / (2.0 * h)
            } else {
                f64::NAN
            };
        }
        for t in 0..m {
            let mut zp = z.clone();
            zp[t] += h;
            let mut gt = vec![0.0f64; m];
            for u in 0..m {
                let mut pp = zp.clone();
                pp[u] += h;
                let fp = phi_neg_ll(&pp, p, &posterior.s_bar, n);
                let mut mm = zp.clone();
                mm[u] -= h;
                let fm = phi_neg_ll(&mm, p, &posterior.s_bar, n);
                gt[u] = if fp.is_finite() && fm.is_finite() {
                    (fp - fm) / (2.0 * h)
                } else {
                    f64::NAN
                };
            }
            let st = provider.phi_slots[t];
            for u in 0..m {
                let su = provider.phi_slots[u];
                // d²Q = -d²(phi_neg_ll)
                let h_neg = (gt[u] - g0[u]) / h;
                hess[st * k + su] += -h_neg;
            }
        }
        let _ = f0;
    }
    hess
}

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

/// Observed-information standard errors via Oakes (1999, eq. 6, p. 480) at
/// GIVEN two-tier item and primary-correlation parameters.
/// With fixed Phi=I, only item parameters enter the free vector (Cai, 2010,
/// pp. 583-584; Oakes, 1999, eq. 6, p. 480).
#[allow(clippy::too_many_arguments)]
pub fn two_tier_oakes_se(
    a_primary: &[f64],
    a_specific: &[f64],
    thresholds: &[f64],
    phi: &[f64],
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    cfg: &TwoTierOakesConfig,
) -> Result<TwoTierOakesResult, String> {
    let provider = Provider::new(
        y,
        observed,
        primary_map,
        specific_map,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        cfg,
    )?;
    if a_primary.len() != n_items * n_primary {
        return Err("a_primary must have length n_items * n_primary".into());
    }
    if a_specific.len() != n_items {
        return Err("a_specific must have length n_items".into());
    }
    if thresholds.len() != n_items * (n_cat - 1) {
        return Err("thresholds must have length n_items * (n_cat - 1)".into());
    }
    if phi.len() != n_primary * n_primary {
        return Err("phi must have length n_primary * n_primary".into());
    }
    let labels = provider.labels.clone();
    let k = provider.free_len();
    let packed = provider.pack(a_primary, a_specific, thresholds, phi)?;

    let posterior0 = provider.posterior_at(&packed)?;
    let term_a = q_hessian(&packed, &posterior0, &provider);
    let g0 = q_gradient(&packed, &posterior0, &provider);
    let mut cross = vec![0.0f64; k * k];
    for j in 0..k {
        let hj = cfg.fd_step * (1.0 + packed[j].abs());
        let mut perturbed = packed.clone();
        perturbed[j] += hj;
        match provider.posterior_at(&perturbed) {
            Ok(posterior_p) => {
                let gp = q_gradient(&packed, &posterior_p, &provider);
                for c in 0..k {
                    cross[j * k + c] = (gp[c] - g0[c]) / hj;
                }
            }
            Err(_) => {
                for c in 0..k {
                    cross[j * k + c] = f64::NAN;
                }
            }
        }
    }

    let mut information = vec![0.0f64; k * k];
    for r in 0..k {
        for c in 0..k {
            information[r * k + c] = -0.5
                * (term_a[r * k + c] + cross[r * k + c] + term_a[c * k + r] + cross[c * k + r]);
        }
    }
    if information.iter().any(|v| !v.is_finite()) {
        return Ok(TwoTierOakesResult {
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
            let se: Vec<f64> = (0..k)
                .map(|j| {
                    debug_assert!(vcov[j * k + j] > 0.0);
                    vcov[j * k + j].sqrt()
                })
                .collect();
            Ok(TwoTierOakesResult {
                labels,
                information,
                vcov: Some(vcov),
                se: Some(se),
                positive_definite: true,
                non_pd_reason: None,
            })
        }
        Err(reason) => Ok(TwoTierOakesResult {
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
#[path = "../../../tests/unit/two_tier_oakes_tests.rs"]
mod tests;
