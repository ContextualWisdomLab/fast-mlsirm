//! All-fixed bifactor focal calibration with every orthogonal mean/variance free.
//! Basis: Cai, Yang, and Hansen (2011, pp. 230–232, Eqs. 15–17) describe
//! population means/variances and reduced integration. Kim (2006, pp. 360–363)
//! describes fixed-item calibration with iterated population updates.
//! References: Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized
//! full-information item bifactor analysis. Psychological Methods, 16(3),
//! 221–248. doi:10.1037/a0023350. Kim, S. (2006). A comparative study of IRT
//! fixed parameter calibration methods. Journal of Educational Measurement,
//! 43(4), 355–381. doi:10.1111/j.1745-3984.2006.00021.x.
use super::*;

/// Caller-owned quadrature and stopping controls; no node-count default/cap.
#[derive(Clone, Debug)]
pub struct BifactorFullFipcConfig {
    pub q_general: usize,
    pub q_specific: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub device: crate::Device,
}

/// Population-only result. No participant scores or item maximization.
#[derive(Clone, Debug)]
pub struct BifactorFullFipcResult {
    pub mean: Vec<f64>,
    pub sd: Vec<f64>,
    pub a_general: Vec<f64>,
    pub a_specific: Vec<f64>,
    pub threshold: Vec<f64>,
    pub loglik_trace: Vec<f64>,
    pub n_iter: usize,
    pub converged: bool,
    pub termination_reason: String,
    pub final_loglik_change: Option<f64>,
    pub em_map_displacement: f64,
    pub n_parameters: usize,
    pub gpu_execution_used: bool,
    pub gpu_backend: Option<String>,
    pub gpu_device_name: Option<String>,
    pub cpu_fallback_reason: Option<String>,
}

/// Estimate all focal means/diagonal variances under fixed item parameters.
///
/// Basis: Cai et al. (2011, pp. 230–232, Eqs. 15–17); Kim (2006, pp. 360–363),
/// full citations above. Orthogonal normal support is `mean + sd * GH_node`.
/// With complete responses, category-summed expected counts of one block
/// item equal the joint posterior mass at each reduced node. Their first and
/// second moments give the full population EM map. Missing responses are
/// refused, not treated as representative counts. Item parameters are copied
/// unchanged. Every step is full: no damping, rollback, or variance clamp.
/// Convergence requires finite nondecreasing likelihood plus both relative
/// likelihood change and full EM-map mean/log-SD displacement <= caller tol.
/// GPU uses the existing binary32 reduced E-step; this is not an accuracy
/// certificate or the stochastic MHRM estimator used by a retained R fit.
#[allow(clippy::too_many_arguments)]
pub fn fit_bifactor_grm_fipc_full(
    y: &[usize],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_specific: usize,
    n_cat: usize,
    fixed_ag: &[f64],
    fixed_as: &[f64],
    fixed_threshold: &[f64],
    initial_mean: &[f64],
    initial_sd: &[f64],
    cfg: &BifactorFullFipcConfig,
) -> Result<BifactorFullFipcResult, String> {
    crate::gpu_bifactor::reset_gpu_dispatch_receipt();
    let validator = BifactorGrmConfig {
        q_general: cfg.q_general,
        q_specific: cfg.q_specific,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
        device: cfg.device,
    };
    let v = validate(
        y,
        None,
        specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &validator,
    )?;
    let dim = n_specific
        .checked_add(1)
        .ok_or("factor dimension overflow")?;
    if initial_mean.len() != dim || initial_sd.len() != dim {
        return Err("initial_mean/initial_sd must have length n_specific + 1".into());
    }
    if initial_mean.iter().any(|x| !x.is_finite())
        || initial_sd.iter().any(|x| !x.is_finite() || *x <= 0.0)
    {
        return Err("initial means must be finite and SDs finite positive".into());
    }
    if fixed_ag.len() != n_items
        || fixed_as.len() != n_items
        || fixed_threshold.len() != n_items * v.m1
    {
        return Err("fixed item parameter shapes do not match items/categories".into());
    }
    if fixed_ag
        .iter()
        .chain(fixed_as)
        .chain(fixed_threshold)
        .any(|x| !x.is_finite())
    {
        return Err("fixed item parameters must be finite".into());
    }
    for i in 0..n_items {
        if v.item_block[i].is_none() && fixed_as[i] != 0.0 {
            return Err("general-only fixed specific slopes must be zero".into());
        }
        if fixed_threshold[i * v.m1..(i + 1) * v.m1]
            .windows(2)
            .any(|w| w[0] <= w[1])
        {
            return Err("fixed thresholds must be strictly decreasing".into());
        }
    }
    let params: Vec<ItemParams> = (0..n_items)
        .map(|i| ItemParams {
            a_g: fixed_ag[i],
            a_s: v.item_block[i].map(|_| fixed_as[i]),
            d: fixed_threshold[i * v.m1..(i + 1) * v.m1].to_vec(),
        })
        .collect();
    let (xg, wg) = gh_rule(cfg.q_general)?;
    let (xs, ws) = gh_rule(cfg.q_specific)?;
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let group = vec![0; n_persons];
    let mut mean = initial_mean.to_vec();
    let mut sd = initial_sd.to_vec();
    let mut trace: Vec<f64> = Vec::new();
    let mut n_iter = 0;
    let mut converged = false;
    let mut final_change;
    let displacement;
    loop {
        let tg: Vec<f64> = xg.iter().map(|x| mean[0] + sd[0] * x).collect();
        let ts: Vec<Vec<f64>> = (0..n_specific)
            .map(|s| xs.iter().map(|x| mean[s + 1] + sd[s + 1] * x).collect())
            .collect();
        if tg.iter().chain(ts.iter().flatten()).any(|x| !x.is_finite()) {
            return Err("non-finite affine quadrature support".into());
        }
        let (kernel_ll, counts, mass, s1, s2, _, _) = e_step_multigroup(
            &v,
            y,
            None,
            &group,
            1,
            std::slice::from_ref(&params),
            std::slice::from_ref(&tg),
            std::slice::from_ref(&ts),
            &log_wg,
            &log_ws,
            cfg.q_general,
            cfg.q_specific,
            cfg.device,
        );
        let receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
        if cfg.device == crate::Device::Gpu
            && (!receipt.used
                || receipt.fallback_reason.is_some()
                || receipt.backend.is_none()
                || receipt.device_name.is_none())
        {
            return Err(format!(
                "GPU-only E-step required: {:?}",
                receipt.fallback_reason
            ));
        }
        let ll = if receipt.used {
            observed_loglik_binary64(&v, y, &params, &tg, &ts, &log_wg, &log_ws)?
        } else {
            kernel_ll
        };
        final_change = checked_em_loglik_change(ll, trace.last().copied(), n_iter)?;
        trace.push(ll);
        let mut next_mean = vec![0.0; dim];
        let mut next_sd = vec![0.0; dim];
        let mut first = vec![s1[0]];
        let mut second = vec![s2[0]];
        let mut masses = vec![mass[0]];
        for s in 0..n_specific {
            let item = v.blocks[s][0];
            let mut m = 0.0;
            let mut a = 0.0;
            let mut b = 0.0;
            for t in 0..cfg.q_general {
                for h in 0..cfg.q_specific {
                    let p: f64 = counts[0][item][t * cfg.q_specific + h].iter().sum();
                    let x = ts[s][h];
                    m += p;
                    a += p * x;
                    b += p * x * x;
                }
            }
            masses.push(m);
            first.push(a);
            second.push(b);
        }
        let mut delta: f64 = 0.0;
        for d in 0..dim {
            if !masses[d].is_finite() || masses[d] <= 0.0 {
                return Err("nonpositive/nonfinite posterior mass".into());
            }
            next_mean[d] = first[d] / masses[d];
            let variance = second[d] / masses[d] - next_mean[d] * next_mean[d];
            if !next_mean[d].is_finite() || !variance.is_finite() || variance <= 0.0 {
                return Err("nonfinite mean or nonpositive posterior variance".into());
            }
            next_sd[d] = variance.sqrt();
            delta = delta
                .max((next_mean[d] - mean[d]).abs())
                .max((next_sd[d].ln() - sd[d].ln()).abs());
        }
        if let Some(change) = final_change {
            let previous = trace[trace.len() - 2];
            if change <= cfg.tol * (1.0 + previous.abs()) && delta <= cfg.tol {
                converged = true;
                displacement = delta;
                break;
            }
        }
        if n_iter >= cfg.max_iter {
            displacement = delta;
            break;
        }
        mean = next_mean;
        sd = next_sd;
        n_iter += 1;
    }
    let receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
    Ok(BifactorFullFipcResult {
        mean,
        sd,
        a_general: fixed_ag.to_vec(),
        a_specific: fixed_as.to_vec(),
        threshold: fixed_threshold.to_vec(),
        loglik_trace: trace,
        n_iter,
        converged,
        termination_reason: if converged {
            "tolerance_met"
        } else {
            "max_iter_reached"
        }
        .into(),
        final_loglik_change: final_change,
        em_map_displacement: displacement,
        n_parameters: dim.checked_mul(2).ok_or("parameter count overflow")?,
        gpu_execution_used: receipt.used,
        gpu_backend: receipt.backend,
        gpu_device_name: receipt.device_name,
        cpu_fallback_reason: receipt.fallback_reason,
    })
}

/// Binary64 observed likelihood on the identical affine support/weights.
/// Basis: Cai et al. (2011, pp. 231–232, Eqs. 15–17); full reference above.
/// CPU verification of the scalar likelihood does not replace GPU moments
/// or permit CPU fallback. It avoids using rounded binary32 likelihoods for
/// strict monotonicity/termination. No tolerance is altered.
fn observed_loglik_binary64(
    v: &Validated,
    y: &[usize],
    params: &[ItemParams],
    tg: &[f64],
    ts: &[Vec<f64>],
    log_wg: &[f64],
    log_ws: &[f64],
) -> Result<f64, String> {
    let qg = tg.len();
    let qs = log_ws.len();
    let mut tables = Vec::with_capacity(v.n_items);
    for (i, par) in params.iter().enumerate() {
        let mut table = Vec::new();
        for &g in tg {
            if let Some(s) = v.item_block[i] {
                for &h in &ts[s] {
                    let base = par.a_g * g + par.a_s.expect("validated specific item") * h;
                    table.extend(grm_logprobs(base, &par.d));
                }
            } else {
                table.extend(grm_logprobs(par.a_g * g, &par.d));
            }
        }
        tables.push(table);
    }
    let mut likelihood = 0.0;
    let mut compensation = 0.0;
    let mut general = vec![0.0; qg];
    let mut specific = vec![0.0; qs];
    for p in 0..v.n_persons {
        general.copy_from_slice(log_wg);
        for t in 0..qg {
            for &item in &v.general_only {
                general[t] += tables[item][t * v.n_cat + y[p * v.n_items + item]];
            }
            for members in &v.blocks {
                specific.copy_from_slice(log_ws);
                for h in 0..qs {
                    for &item in members {
                        specific[h] +=
                            tables[item][(t * qs + h) * v.n_cat + y[p * v.n_items + item]];
                    }
                }
                general[t] += log_sum_exp(&specific);
            }
        }
        let person_ll = log_sum_exp(&general);
        if !person_ll.is_finite() {
            return Err("non-finite binary64 observed likelihood".into());
        }
        let add = person_ll - compensation;
        let total = likelihood + add;
        compensation = (total - likelihood) - add;
        likelihood = total;
    }
    Ok(likelihood)
}
