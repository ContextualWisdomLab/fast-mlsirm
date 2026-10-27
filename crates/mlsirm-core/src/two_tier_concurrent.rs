//! Concurrent orthogonal-primary two-tier GRM with GPU item maximization.
//!
//! This is a hybrid native path: reduced posterior/count/specific moments and
//! Newton item stages dispatch on GPU; probability tables, scalar f64 likelihood,
//! primary posterior moments and Gaussian updates are native CPU arithmetic.
//! No CPU optimizer fallback is permitted. It is not a full-GPU study producer.
//!
//! Cai (2010, pp. 586–590, Eqs. 2,4–6,11–16; pp. 608–609, Appendix A)
//! supplies the two-tier normal model, reduced likelihood and complete-data
//! criteria. Summing independent group criteria with shared item coordinates
//! gives concurrent common-item maximization. His empirical development is
//! single-group (p. 606); this explicit multigroup extension, affine GH rule,
//! strict Newton controls and convergence policy are implementation choices.
//! Group zero fixes the metric. Caller-supplied fixed/free diagonal moments
//! define focal constraints; no study-specific choice is inferred here.
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model with
//! applications. *Psychometrika, 75*(4), 581–612.
//! https://doi.org/10.1007/s11336-010-9178-0
use super::*;
use crate::gpu_mstep::{NewtonControl, StageReceipt};

#[cfg(test)]
#[path = "../../../tests/unit/two_tier_concurrent_moment_tests.rs"]
mod moment_tests;
#[cfg(test)]
pub(super) use moment_tests::synthetic_group_sweep_oracle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MomentConstraint {
    pub mean: f64,
    pub sd: f64,
    pub free_mean: bool,
    pub free_sd: bool,
}
impl MomentConstraint {
    /// Reference metric N(0,1); caller must explicitly request this constraint.
    pub fn fixed_standard() -> Self {
        Self {
            mean: 0.,
            sd: 1.,
            free_mean: false,
            free_sd: false,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ConcurrentConfig {
    pub em: TwoTierGrmConfig,
    /// Must match em.newton_iter/ridge; gradient and step remain exactly 1e-9.
    pub newton: NewtonControl,
}

#[derive(Clone, Debug)]
pub struct GroupSweepReceipt {
    pub gpu_used: bool,
    pub backend: String,
    pub device: String,
    pub fallback_reason: Option<String>,
    pub cpu_probability_tables: usize,
    pub cpu_scalar_likelihood: usize,
    pub cpu_primary_moment_reductions: usize,
    /// Completely unobserved blocks have their unchanged prior posterior.
    /// The shared device kernel omits these; native CPU adds the actual GH
    /// prior moments once per missing person/block, never from item counts.
    pub cpu_missing_block_prior_moment_completions: usize,
}
#[derive(Clone, Debug, Default)]
pub struct ItemSweepReceipt {
    pub common_updates: usize,
    pub free_updates: usize,
    pub stages: StageReceipt,
}
#[derive(Clone, Debug)]
pub struct SweepReceipt {
    pub iteration: usize,
    pub groups: Vec<GroupSweepReceipt>,
    pub m_step: Option<ItemSweepReceipt>,
    pub cpu_gaussian_parameter_updates: usize,
}
#[derive(Clone, Debug)]
pub struct StartReceipt {
    pub start: usize,
    pub sweeps: Vec<SweepReceipt>,
    pub error: Option<String>,
    pub converged: bool,
}
#[derive(Clone, Debug)]
pub struct ConcurrentFailure {
    pub message: String,
    pub starts: Vec<StartReceipt>,
}
impl std::fmt::Display for ConcurrentFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.message.fmt(f)
    }
}
impl std::error::Error for ConcurrentFailure {}
impl From<String> for ConcurrentFailure {
    fn from(message: String) -> Self {
        Self {
            message,
            starts: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ConcurrentFit {
    /// Group-major dense item coefficients; common rows are bit-identical.
    pub a_primary: Vec<Vec<f64>>,
    pub a_specific: Vec<Vec<f64>>,
    pub threshold: Vec<Vec<f64>>,
    /// Group-major [primary dimensions..., specific dimensions...].
    pub moments: Vec<MomentConstraint>,
    /// Primary correlation, exactly identity. SDs are separate from Phi.
    pub phi: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_param_change: f64,
    pub best_start: usize,
    pub n_parameters: usize,
    /// Includes every attempted start and initial/terminal E sweep.
    pub starts: Vec<StartReceipt>,
}

struct GroupData {
    v: Validated,
    y: Vec<usize>,
    observed: Option<Vec<bool>>,
}
struct GroupState {
    counts: Vec<Vec<Vec<f64>>>,
    first: Vec<f64>,
    second: Vec<f64>,
    mass: Vec<f64>,
    coords: Vec<f64>,
    specific_nodes: Vec<Vec<f64>>,
    loglik: f64,
}

/// Normal sufficient-statistic update on actual common-scale posterior nodes
/// (Cai, 2010, pp. 608–609, Appendix A). A fixed mean stays fixed even when
/// the SD is free: E[(theta-fixed_mean)^2], not Var(theta), is maximized.
fn update_moment(
    old: MomentConstraint,
    mass: f64,
    first: f64,
    second: f64,
) -> Result<MomentConstraint, String> {
    if !mass.is_finite() || mass <= 0. || !first.is_finite() || !second.is_finite() {
        return Err("nonfinite or empty Gaussian posterior sufficient statistics".into());
    }
    let mean = if old.free_mean {
        first / mass
    } else {
        old.mean
    };
    let sd = if old.free_sd {
        let variance = second / mass - 2. * mean * first / mass + mean * mean;
        if !variance.is_finite() || variance <= 0. {
            return Err("nonpositive Gaussian variance update; no clamp permitted".into());
        }
        variance.sqrt()
    } else {
        old.sd
    };
    if !mean.is_finite() || !sd.is_finite() || sd <= 0. {
        return Err("nonfinite Gaussian update".into());
    }
    Ok(MomentConstraint { mean, sd, ..old })
}

/// Joint shared-anchor location screen for precisely the free focal means.
/// This checks the null location shift of the anchored linear predictors,
/// not full information/scale identification (Cai, 2010, pp. 586–587).
fn location_rank(
    v: &Validated,
    params: &[ItemParams],
    common: &[bool],
    moments: &[MomentConstraint],
) -> Result<(), String> {
    let cols: Vec<usize> = moments
        .iter()
        .enumerate()
        .filter_map(|(d, m)| m.free_mean.then_some(d))
        .collect();
    if cols.is_empty() {
        return Ok(());
    }
    let rows: Vec<Vec<f64>> = params
        .iter()
        .enumerate()
        .filter(|(i, _)| common[*i])
        .map(|(i, item)| {
            cols.iter()
                .map(|&d| {
                    if d < v.n_primary {
                        item.a_p[d]
                    } else if v.item_block[i] == Some(d - v.n_primary) {
                        item.a_s.unwrap_or(0.)
                    } else {
                        0.
                    }
                })
                .collect()
        })
        .collect();
    let error = || "common-item location rank deficient for free group means".to_string();
    if rows.len() < cols.len() {
        return Err(error());
    }
    let mut basis: Vec<Vec<f64>> = Vec::new();
    for d in 0..cols.len() {
        let mut x: Vec<f64> = rows.iter().map(|r| r[d]).collect();
        let scale = x.iter().fold(0_f64, |s, a| s.max(a.abs()));
        if !scale.is_finite() || scale == 0. {
            return Err(error());
        }
        for a in &mut x {
            *a /= scale;
        }
        let norm = x.iter().fold(0_f64, |s, a| s.hypot(*a));
        for a in &mut x {
            *a /= norm;
        }
        for _ in 0..2 {
            for q in &basis {
                let dot = x.iter().zip(q).map(|(a, b)| a * b).sum::<f64>();
                for (a, b) in x.iter_mut().zip(q) {
                    *a -= dot * b;
                }
            }
        }
        let norm = x.iter().fold(0_f64, |s, a| s.hypot(*a));
        if !norm.is_finite() || norm <= 64. * f64::EPSILON * rows.len().max(cols.len()) as f64 {
            return Err(error());
        }
        for a in &mut x {
            *a /= norm;
        }
        basis.push(x);
    }
    Ok(())
}

/// Reuse the actual reduced device kernel without its optional CPU fallback.
/// Specific first/second moments come from the full block posterior, even when
/// an item's response is missing. Primary moments use the returned postg.
fn group_sweep(
    data: &GroupData,
    params: &[ItemParams],
    moments: &[MomentConstraint],
    base: &[f64],
    log_w: &[f64],
    ts: &[f64],
    log_ws: &[f64],
) -> Result<(GroupState, GroupSweepReceipt), String> {
    let v = &data.v;
    let p = v.n_primary;
    let ng = v.grid_size;
    let qs = ts.len();
    let coords: Vec<f64> = base
        .chunks_exact(p)
        .flat_map(|row| {
            row.iter()
                .enumerate()
                .map(|(d, x)| moments[d].mean + moments[d].sd * x)
        })
        .collect();
    let specific_nodes: Vec<Vec<f64>> = moments[p..]
        .iter()
        .map(|m| ts.iter().map(|x| m.mean + m.sd * x).collect())
        .collect();
    let order = canonical_person_order(v, &data.y, data.observed.as_deref());
    let y: Vec<usize> = order
        .iter()
        .flat_map(|&n| data.y[n * v.n_items..(n + 1) * v.n_items].iter().copied())
        .collect();
    let observed = data.observed.as_ref().map(|o| {
        order
            .iter()
            .flat_map(|&n| o[n * v.n_items..(n + 1) * v.n_items].iter().copied())
            .collect::<Vec<_>>()
    });
    let tables = focal_gpu_tables(v, params, &coords, &specific_nodes, ng, qs);
    let tables_groups = [tables];
    let tg_groups = [vec![0.; ng]];
    let ts_groups = [specific_nodes.clone()];
    crate::gpu_bifactor::reset_gpu_dispatch_receipt();
    let inputs = crate::gpu_bifactor::ReducedEstepInputs {
        y: &y,
        observed: observed.as_deref(),
        group_id: None,
        n_persons: v.n_persons,
        n_items: v.n_items,
        n_specific: v.n_specific,
        n_cat: v.n_cat,
        qg: ng,
        qs,
        n_groups: 1,
        tables_groups: &tables_groups,
        item_block: &v.item_block,
        blocks: &v.blocks,
        tg_groups: &tg_groups,
        ts_groups: &ts_groups,
        log_wg: log_w,
        log_ws,
    };
    let result = crate::gpu_bifactor::e_step_reduced_gpu(&inputs).ok_or_else(|| {
        format!(
            "concurrent GPU E-step refused; no CPU fallback: {:?}",
            crate::gpu_bifactor::gpu_dispatch_receipt()
        )
    })?;
    let receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
    if !receipt.used
        || receipt.fallback_reason.is_some()
        || receipt.backend.as_deref().is_none_or(str::is_empty)
        || receipt.device_name.as_deref().is_none_or(str::is_empty)
    {
        return Err("missing or fallback-bearing concurrent GPU E-step receipt".into());
    }
    let mut counts = Vec::new();
    for i in 0..v.n_items {
        let rows = ng * if v.item_block[i].is_some() { qs } else { 1 };
        let base = i * result.counts_stride_nodes * v.n_cat;
        counts.push(
            result.counts[base..base + rows * v.n_cat]
                .chunks_exact(v.n_cat)
                .map(<[f64]>::to_vec)
                .collect(),
        );
    }
    let dims = p + v.n_specific;
    let mut first = vec![0.; dims];
    let mut second = vec![0.; dims];
    let mut mass = vec![0.; dims];
    for person in 0..v.n_persons {
        for g in 0..ng {
            let post = result.postg[person * ng + g];
            if !post.is_finite() || post < 0. {
                return Err("invalid device primary posterior".into());
            }
            for d in 0..p {
                let x = coords[g * p + d];
                first[d] += post * x;
                second[d] += post * x * x;
                mass[d] += post;
            }
        }
    }
    first[p..].copy_from_slice(&result.s1_spec);
    second[p..].copy_from_slice(&result.s2_spec);
    mass[p..].copy_from_slice(&result.w_spec);
    // The existing reduced kernel intentionally omits unobserved blocks from
    // w_spec. For population estimation their conditional posterior is the
    // independent prior (Cai, 2010, Eqs.4,6,15). Complete that exact finite GH
    // measure natively; this is disclosed CPU arithmetic, not GPU evidence.
    let mut missing_block_completions = 0;
    for s in 0..v.n_specific {
        let missing = (0..v.n_persons)
            .filter(|&person| {
                v.blocks[s].iter().all(|&i| {
                    data.observed
                        .as_ref()
                        .is_some_and(|o| !o[person * v.n_items + i])
                })
            })
            .count();
        if missing == 0 {
            continue;
        }
        let mut w = 0.;
        let mut sx = 0.;
        let mut sx2 = 0.;
        for h in 0..qs {
            let weight = log_ws[h].exp();
            let x = specific_nodes[s][h];
            w += weight;
            sx += weight * x;
            sx2 += weight * x * x;
        }
        let n = missing as f64;
        mass[p + s] += n * w;
        first[p + s] += n * sx;
        second[p + s] += n * sx2;
        missing_block_completions += missing;
    }
    let log_ws_by_specific = vec![log_ws.to_vec(); v.n_specific];
    let loglik = focal_loglik_cpu(
        v,
        &data.y,
        data.observed.as_deref(),
        params,
        log_w,
        &log_ws_by_specific,
        &coords,
        &specific_nodes,
        ng,
        qs,
    );
    if !loglik.is_finite() {
        return Err("nonfinite concurrent scalar f64 likelihood".into());
    }
    Ok((
        GroupState {
            counts,
            first,
            second,
            mass,
            coords,
            specific_nodes,
            loglik,
        },
        GroupSweepReceipt {
            gpu_used: receipt.used,
            backend: receipt.backend.unwrap(),
            device: receipt.device_name.unwrap(),
            fallback_reason: receipt.fallback_reason,
            cpu_probability_tables: 1,
            cpu_scalar_likelihood: 1,
            cpu_primary_moment_reductions: 1,
            cpu_missing_block_prior_moment_completions: missing_block_completions,
        },
    ))
}

fn packed_item(v: &Validated, item: &ItemParams, i: usize) -> Vec<f64> {
    let mut out: Vec<f64> = v.free_primaries[i].iter().map(|&d| item.a_p[d]).collect();
    if let Some(x) = item.a_s {
        out.push(x);
    }
    out.extend_from_slice(&item.d);
    out
}
fn unpack_item(v: &Validated, packed: &[f64], i: usize) -> ItemParams {
    let free = &v.free_primaries[i];
    let specific = v.item_block[i].is_some();
    let mut a_p = vec![0.; v.n_primary];
    for (k, &d) in free.iter().enumerate() {
        a_p[d] = packed[k];
    }
    ItemParams {
        a_p,
        a_s: specific.then(|| packed[free.len()]),
        d: packed[free.len() + usize::from(specific)..].to_vec(),
    }
}

/// Concatenate group-specific node/count pairs, never sum counts across
/// nonidentical coordinates. Each flattened design row uses exactly the
/// existing adapter order [free primary..., specific?] (Cai, 2010, p.609).
fn update_shared(
    v: &Validated,
    item: &ItemParams,
    i: usize,
    states: &[GroupState],
    control: &NewtonControl,
) -> Result<(ItemParams, StageReceipt), String> {
    let free = &v.free_primaries[i];
    let block = v.item_block[i];
    let off = free.len() + usize::from(block.is_some());
    let rows = states
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.counts[i].len()))
        .ok_or("pooled node rows overflow")?;
    let cells = rows
        .checked_mul(off)
        .and_then(|n| n.checked_mul(2))
        .ok_or("pooled node design overflow")?;
    let (limit, _) = crate::gpu_mstep::limits()?;
    crate::gpu_mstep::checked_buffer(cells, limit)?;
    let mut nodes = Vec::new();
    let mut counts = Vec::new();
    for s in states {
        let ts = block.map_or(&[][..], |b| s.specific_nodes[b].as_slice());
        nodes.extend(crate::two_tier_gpu_item::pack_nodes(
            free,
            block.is_some(),
            &s.coords,
            ts,
            v.n_primary,
            v.grid_size,
            ts.len(),
        )?);
        counts.extend(s.counts[i].iter().cloned());
    }
    // Already expanded rows: use all design columns as adapter primaries.
    // This is transport only, not an extra factor or a model substitution.
    let all: Vec<usize> = (0..off).collect();
    let (packed, receipt) = crate::two_tier_gpu_item::update_item(
        &packed_item(v, item, i),
        &all,
        false,
        &nodes,
        &[],
        off,
        rows,
        0,
        &counts,
        v.n_cat,
        control,
        crate::Device::Gpu,
    )?;
    Ok((unpack_item(v, &packed, i), receipt))
}

struct Outcome {
    params: Vec<Vec<ItemParams>>,
    moments: Vec<MomentConstraint>,
    trace: Vec<f64>,
    n_iter: usize,
    converged: bool,
    change: f64,
}

/// Fit a shared/free item-mask model with explicit fixed/free diagonal group
/// moments. Group0 must be fixed N(0,I), Phi stays identity in every group.
/// This does not choose study focal W/specific moments, implement DIF inference,
/// or provide same-fit diagnostics. All starts and terminal E sweeps are kept.
#[allow(clippy::too_many_arguments)]
pub fn fit_concurrent_gpu(
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    common: &[bool],
    moment_constraints: &[MomentConstraint],
    cfg: &ConcurrentConfig,
) -> Result<ConcurrentFit, ConcurrentFailure> {
    let em = &cfg.em;
    let fail = |s: &str| ConcurrentFailure::from(s.to_string());
    cfg.newton.validate().map_err(ConcurrentFailure::from)?;
    if em.estimate_primary_correlation {
        return Err(fail(
            "concurrent GPU increment requires explicit identity Phi",
        ));
    }
    if cfg.newton.gradient_tolerance != 1e-9
        || cfg.newton.step_tolerance != 1e-9
        || cfg.newton.ridge != em.ridge
        || cfg.newton.iterations != em.newton_iter
    {
        return Err(fail(
            "strict Newton gradient/step1e-9 and EM/control iteration/ridge identity required",
        ));
    }
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
        em,
    )
    .map_err(ConcurrentFailure::from)?;
    let dims = n_primary
        .checked_add(n_specific)
        .ok_or_else(|| fail("factor dimensions overflow"))?;
    if n_groups == 0
        || group_id.len() != n_persons
        || group_id.iter().any(|g| *g >= n_groups)
        || common.len() != n_items
        || n_groups.checked_mul(dims) != Some(moment_constraints.len())
    {
        return Err(fail(
            "invalid group, common-mask or moment-constraint geometry",
        ));
    }
    if moment_constraints
        .iter()
        .any(|m| !m.mean.is_finite() || !m.sd.is_finite() || m.sd <= 0.)
    {
        return Err(fail("group moments require finite means and positive SDs"));
    }
    if moment_constraints[..dims]
        .iter()
        .any(|m| *m != MomentConstraint::fixed_standard())
    {
        return Err(fail("reference group must fix every factor to N(0,1)"));
    }
    // Direct anchored loading coverage is a conservative necessary scale screen,
    // not a sufficient full-model information-rank assertion.
    for group in 1..n_groups {
        for d in 0..dims {
            let m = moment_constraints[group * dims + d];
            if (m.free_mean || m.free_sd)
                && !(0..n_items).any(|i| {
                    common[i]
                        && if d < n_primary {
                            primary_map[i * n_primary + d]
                        } else {
                            specific_map[i] == (d - n_primary) as i32
                        }
                })
            {
                return Err(fail(
                    "free group moment lacks a directly common loading item",
                ));
            }
        }
    }
    let mut data = Vec::new();
    for g in 0..n_groups {
        let people: Vec<usize> = group_id
            .iter()
            .enumerate()
            .filter_map(|(p, &s)| (s == g).then_some(p))
            .collect();
        if people.is_empty() {
            return Err(fail("declared group has no persons"));
        }
        let yy: Vec<usize> = people
            .iter()
            .flat_map(|&p| y[p * n_items..(p + 1) * n_items].iter().copied())
            .collect();
        let oo = observed.map(|o| {
            people
                .iter()
                .flat_map(|&p| o[p * n_items..(p + 1) * n_items].iter().copied())
                .collect::<Vec<_>>()
        });
        for i in 0..n_items {
            let mut seen = vec![false; n_cat];
            for p in 0..people.len() {
                if oo.as_ref().is_none_or(|o| o[p * n_items + i]) {
                    seen[yy[p * n_items + i]] = true;
                }
            }
            if !common[i] && seen.iter().any(|b| !*b) {
                return Err(fail("free group item has an unobserved category"));
            }
        }
        data.push(GroupData {
            v: Validated {
                n_persons: people.len(),
                n_items,
                n_primary,
                n_specific,
                n_cat,
                m1: v.m1,
                grid_size: v.grid_size,
                free_primaries: v.free_primaries.clone(),
                blocks: v.blocks.clone(),
                specific_free: v.specific_free.clone(),
                item_block: v.item_block.clone(),
            },
            y: yy,
            observed: oo,
        });
    }
    let (z, w) = gh_rule(em.q_primary).map_err(ConcurrentFailure::from)?;
    let (ts, ws) = gh_rule(em.q_specific).map_err(ConcurrentFailure::from)?;
    let (base, log_w) = build_primary_grid(z, w, n_primary, v.grid_size);
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let mut starts = Vec::new();
    let mut best: Option<Outcome> = None;
    let mut best_start = 0;
    for start in 0..em.n_starts {
        let mut receipt = StartReceipt {
            start,
            sweeps: Vec::new(),
            error: None,
            converged: false,
        };
        let outcome = (|| -> Result<Outcome, String> {
            // The ordinary start0 makes every primary slope1 and every specific
            // slope0.8. Free primary+specific means then have exactly dependent
            // location columns even for an identified design. Reuse the existing
            // deterministic jitter stream for every concurrent start; still reject
            // deficient fitted anchor columns rather than bypassing the rank gate.
            let jitter_start = start.checked_add(1).ok_or("start index overflow")?;
            let (initial, _) = initial_params(&v, y, observed, em.seed, jitter_start, false);
            let mut params = vec![initial; n_groups];
            let mut moments = moment_constraints.to_vec();
            let mut trace: Vec<f64> = Vec::new();
            let mut last_limited = false;
            let mut change = f64::INFINITY;
            let mut n_iter = 0;
            loop {
                for g in 1..n_groups {
                    location_rank(&v, &params[g], common, &moments[g * dims..(g + 1) * dims])?;
                }
                let mut states = Vec::new();
                receipt.sweeps.push(SweepReceipt {
                    iteration: n_iter,
                    groups: Vec::new(),
                    m_step: None,
                    cpu_gaussian_parameter_updates: 0,
                });
                let sweep = receipt.sweeps.last_mut().unwrap();
                for g in 0..n_groups {
                    let (state, r) = group_sweep(
                        &data[g],
                        &params[g],
                        &moments[g * dims..(g + 1) * dims],
                        &base,
                        &log_w,
                        ts,
                        &log_ws,
                    )?;
                    states.push(state);
                    sweep.groups.push(r);
                }
                let ll = states.iter().map(|s| s.loglik).sum::<f64>();
                let delta = checked_em_loglik_change(ll, trace.last().copied(), n_iter)?;
                let stable = delta
                    .is_some_and(|d| d <= em.tol * (1. + trace.last().unwrap().abs()))
                    && change <= em.tol
                    && !last_limited;
                trace.push(ll);
                if stable || n_iter == em.max_iter {
                    receipt.converged = stable;
                    return Ok(Outcome {
                        params,
                        moments,
                        trace,
                        n_iter,
                        converged: stable,
                        change,
                    });
                }
                change = 0.;
                last_limited = false;
                sweep.m_step = Some(ItemSweepReceipt::default());
                for i in 0..n_items {
                    if common[i] {
                        let (item, r) = update_shared(&v, &params[0][i], i, &states, &cfg.newton)?;
                        for g in 0..n_groups {
                            change = change.max(
                                packed_item(&v, &params[g][i], i)
                                    .iter()
                                    .zip(packed_item(&v, &item, i))
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0., f64::max),
                            );
                            params[g][i] = item.clone();
                        }
                        last_limited |= r.iteration_limit_updates > 0;
                        let m = sweep.m_step.as_mut().unwrap();
                        m.common_updates += 1;
                        m.stages.add(&r);
                    } else {
                        for g in 0..n_groups {
                            let s = &states[g];
                            let block = v.item_block[i];
                            let nodes = block.map_or(&[][..], |b| s.specific_nodes[b].as_slice());
                            let old = packed_item(&v, &params[g][i], i);
                            let (new, r) = crate::two_tier_gpu_item::update_item(
                                &old,
                                &v.free_primaries[i],
                                block.is_some(),
                                &s.coords,
                                nodes,
                                n_primary,
                                v.grid_size,
                                ts.len(),
                                &s.counts[i],
                                n_cat,
                                &cfg.newton,
                                crate::Device::Gpu,
                            )?;
                            change = change.max(
                                old.iter()
                                    .zip(&new)
                                    .map(|(a, b)| (a - b).abs())
                                    .fold(0., f64::max),
                            );
                            params[g][i] = unpack_item(&v, &new, i);
                            last_limited |= r.iteration_limit_updates > 0;
                            let m = sweep.m_step.as_mut().unwrap();
                            m.free_updates += 1;
                            m.stages.add(&r);
                        }
                    }
                }
                for g in 1..n_groups {
                    for d in 0..dims {
                        let k = g * dims + d;
                        let old = moments[k];
                        if !old.free_mean && !old.free_sd {
                            continue;
                        }
                        let new = update_moment(
                            old,
                            states[g].mass[d],
                            states[g].first[d],
                            states[g].second[d],
                        )?;
                        change = change
                            .max((new.mean - old.mean).abs())
                            .max((new.sd.ln() - old.sd.ln()).abs());
                        moments[k] = new;
                        sweep.cpu_gaussian_parameter_updates +=
                            usize::from(old.free_mean) + usize::from(old.free_sd);
                    }
                }
                n_iter += 1;
            }
        })();
        match outcome {
            Ok(outcome) => {
                if best
                    .as_ref()
                    .is_none_or(|b| outcome.trace.last().unwrap() > b.trace.last().unwrap())
                {
                    best = Some(outcome);
                    best_start = start;
                }
            }
            Err(error) => receipt.error = Some(error),
        }
        starts.push(receipt);
    }
    let Some(outcome) = best else {
        return Err(ConcurrentFailure {
            message: format!(
                "all {} concurrent GPU starts failed: {}",
                em.n_starts,
                starts
                    .iter()
                    .filter_map(|s| s.error.as_deref())
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            starts,
        });
    };
    let a_primary = outcome
        .params
        .iter()
        .map(|bank| bank.iter().flat_map(|i| i.a_p.iter().copied()).collect())
        .collect();
    let a_specific = outcome
        .params
        .iter()
        .map(|bank| bank.iter().map(|i| i.a_s.unwrap_or(0.)).collect())
        .collect();
    let threshold = outcome
        .params
        .iter()
        .map(|bank| bank.iter().flat_map(|i| i.d.iter().copied()).collect())
        .collect();
    let mut phi = vec![0.; n_primary * n_primary];
    for d in 0..n_primary {
        phi[d * n_primary + d] = 1.;
    }
    let n_parameters = (0..n_items)
        .map(|i| {
            (v.free_primaries[i].len() + usize::from(v.item_block[i].is_some()) + v.m1)
                * if common[i] { 1 } else { n_groups }
        })
        .sum::<usize>()
        + moment_constraints
            .iter()
            .map(|m| usize::from(m.free_mean) + usize::from(m.free_sd))
            .sum::<usize>();
    Ok(ConcurrentFit {
        a_primary,
        a_specific,
        threshold,
        moments: outcome.moments,
        phi,
        loglik_trace: outcome.trace,
        n_iter: outcome.n_iter,
        converged: outcome.converged,
        termination_reason: if outcome.converged {
            "tolerance_met"
        } else {
            "max_iter_reached"
        }
        .into(),
        final_param_change: outcome.change,
        best_start,
        n_parameters,
        starts,
    })
}
