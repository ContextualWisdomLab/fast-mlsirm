use super::*;

#[allow(clippy::too_many_arguments)]
pub(super) fn legacy_fit_two_tier_grm_fipc(
    y: &[usize],
    observed: Option<&[bool]>,
    primary_map: &[bool],
    specific_map: &[i32],
    n_persons: usize,
    n_items: usize,
    n_primary: usize,
    n_specific: usize,
    n_cat: usize,
    anchor: &[bool],
    fixed_a_primary: &[f64],
    fixed_a_specific: &[f64],
    fixed_threshold: &[f64],
    cfg: &TwoTierFipcConfig,
) -> Result<TwoTierFipcResult, String> {
    crate::gpu_bifactor::reset_gpu_dispatch_receipt();
    let validation_cfg = TwoTierGrmConfig {
        estimate_primary_correlation: true,
        q_primary: cfg.q_primary,
        q_specific: cfg.q_specific,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_starts: 1,
        seed: 0,
        newton_iter: cfg.newton_iter,
        ridge: cfg.ridge,
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
        &validation_cfg,
    )?;
    if anchor.len() != n_items {
        return Err("anchor must have length n_items".into());
    }
    if !anchor.iter().any(|&a| a) {
        return Err("at least one anchored item is required to identify the focal scale".into());
    }
    if fixed_a_primary.len() != n_items * n_primary {
        return Err("fixed_a_primary must have length n_items * n_primary".into());
    }
    if fixed_a_specific.len() != n_items {
        return Err("fixed_a_specific must have length n_items".into());
    }
    if fixed_threshold.len() != n_items * v.m1 {
        return Err("fixed_threshold must have length n_items * (n_cat - 1)".into());
    }
    if [fixed_a_primary, fixed_a_specific, fixed_threshold]
        .concat()
        .iter()
        .any(|x| !x.is_finite())
    {
        return Err("fixed anchor parameters must be finite".into());
    }
    for i in 0..n_items {
        if v.item_block[i].is_none() && fixed_a_specific[i] != 0.0 {
            return Err(format!(
                "fixed_a_specific[{i}] must be exactly 0.0 for specific-free items"
            ));
        }
        if anchor[i] {
            for d in 0..n_primary {
                if !primary_map[i * n_primary + d] && fixed_a_primary[i * n_primary + d] != 0.0 {
                    return Err(format!(
                        "fixed_a_primary[{i},{d}] must be exactly 0.0 at fixed pattern positions"
                    ));
                }
            }
            let row = &fixed_threshold[i * v.m1..(i + 1) * v.m1];
            if row.windows(2).any(|w| w[0] <= w[1]) {
                return Err(format!(
                    "fixed thresholds of anchor item {i} must be strictly decreasing"
                ));
            }
        }
    }

    let (tz, wz) = gh_rule(cfg.q_primary)?;
    let (ts_std, ws) = gh_rule(cfg.q_specific)?;
    let n_grid = v.grid_size;
    let (base_coords, log_w0) = build_primary_grid(tz, wz, n_primary, n_grid);
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * n_items + i]);
    let mut params = Vec::with_capacity(n_items);
    for i in 0..n_items {
        if anchor[i] {
            params.push(ItemParams {
                a_p: fixed_a_primary[i * n_primary..(i + 1) * n_primary].to_vec(),
                a_s: v.item_block[i].map(|_| fixed_a_specific[i]),
                d: fixed_threshold[i * v.m1..(i + 1) * v.m1].to_vec(),
            });
        } else {
            let mut freq = vec![1e-3; n_cat];
            for p in 0..n_persons {
                if is_obs(p, i) {
                    freq[y[p * n_items + i]] += 1.0;
                }
            }
            let total: f64 = freq.iter().sum();
            let mut d = vec![0.0; v.m1];
            let mut cum = 0.0;
            for k in (1..n_cat).rev() {
                cum += freq[k] / total;
                let c = cum.clamp(1e-4, 1.0 - 1e-4);
                d[k - 1] = (c / (1.0 - c)).ln();
            }
            let mut a_p = vec![0.0; n_primary];
            for &dim in &v.free_primaries[i] {
                a_p[dim] = 1.0;
            }
            params.push(ItemParams {
                a_p,
                a_s: v.item_block[i].map(|_| 0.8),
                d,
            });
        }
    }
    let mut mean = vec![0.0; n_primary];
    let mut covariance = vec![0.0; n_primary * n_primary];
    for d in 0..n_primary {
        covariance[d * n_primary + d] = 1.0;
    }
    let mut specific_sd = vec![1.0; n_specific];
    let mut loglik_trace = Vec::new();
    let mut fixed_loglik_trace = Vec::new();
    let mut fixed_primary_first_moment_trace = Vec::new();
    let mut fixed_primary_second_moment_trace = Vec::new();
    let mut fixed_specific_second_moment_trace = Vec::new();
    let mut prior_mean_trace = Vec::new();
    let mut prior_covariance_trace = Vec::new();
    let mut prior_specific_sd_trace = Vec::new();
    let mut converged = false;
    let mut n_iter = 0;
    let mut termination_reason = "max_iter_reached".to_string();
    let mut final_loglik_change = f64::NAN;
    let mut rolled_back = false;
    let mut n_accepted_prior_steps = 0;
    let mut n_rollback_full = 0;
    let mut consecutive_rollback = 0;
    let mut prior_update_decision_trace = Vec::new();
    // Full EM-map displacement of the previous cycle (mirt's TOL quantity).
    let mut em_map_displacement = f64::INFINITY;
    let mut full_step_accepted = false;
    const MAX_CONSECUTIVE_ROLLBACKS: usize = 3;

    // Diagnostic-only initial N(0, I) support and standard GH weights.
    // These are immutable across iterations; current item parameters are evaluated.
    let fixed_coords = base_coords.clone();
    let fixed_log_w = log_w0.clone();
    let fixed_ts_by_specific: Vec<Vec<f64>> = (0..n_specific).map(|_| ts_std.to_vec()).collect();
    let fixed_log_ws_by_specific: Vec<Vec<f64>> = (0..n_specific).map(|_| log_ws.clone()).collect();
    loop {
        let (chol, _) = cholesky_lower(&covariance, n_primary).ok_or_else(|| {
            format!("focal primary covariance became non-PD at iteration {n_iter}")
        })?;
        let coords = fipc_primary_coords(&base_coords, &mean, &chol, n_primary, n_grid);
        let ts_by_specific: Vec<Vec<f64>> = (0..n_specific)
            .map(|s| ts_std.iter().map(|&x| x * specific_sd[s]).collect())
            .collect();
        // The affine maps change node locations, not the probability measure:
        // these are direct standard-normal GH rules under the focal prior.
        let log_ws_by_specific: Vec<Vec<f64>> = (0..n_specific).map(|_| log_ws.clone()).collect();
        let log_w = log_w0.clone();
        let (fixed_ll, fixed_m1, fixed_m2, fixed_specific_m2) = fixed_fipc_e_step(
            &v,
            y,
            observed,
            &params,
            &fixed_coords,
            &fixed_log_w,
            &fixed_log_ws_by_specific,
            &fixed_ts_by_specific,
            n_grid,
            ts_std.len(),
            cfg.device,
        );
        fixed_loglik_trace.push(fixed_ll);
        fixed_primary_first_moment_trace.extend_from_slice(&fixed_m1);
        fixed_primary_second_moment_trace.extend_from_slice(&fixed_m2);
        fixed_specific_second_moment_trace.extend_from_slice(&fixed_specific_m2);
        let (ll, counts, sum_primary, sum_primary2, sum_specific2, specific_mass, _, _) =
            e_step_fipc(
                &v,
                y,
                observed,
                &params,
                &log_w,
                &log_ws_by_specific,
                &coords,
                &ts_by_specific,
                n_grid,
                ts_std.len(),
                cfg.device,
            );
        let previous = loglik_trace.last().copied();
        // loglik_trace contains consecutive production evaluations, including
        // the current evaluation when convergence exits before another M-step.
        loglik_trace.push(ll);
        if let Some(change) = checked_em_loglik_change(ll, previous, n_iter).map_err(|error| {
            let fixed_change = fixed_loglik_trace
                .windows(2)
                .last()
                .map(|w| w[1] - w[0]);
            format!(
                "{error}; fixed_eval_ll={fixed_ll:.6e}, fixed_eval_delta={}, fixed_eval_trace={fixed_loglik_trace:?}, remapped_eval_trace={loglik_trace:?}, fixed_eval_primary_m1={fixed_m1:?}, fixed_eval_primary_m2={fixed_m2:?}, fixed_eval_specific_m2={fixed_specific_m2:?}, last_prior_mean={:?}, last_prior_covariance={:?}, last_prior_specific_sd={:?}",
                fixed_change
                    .map(|value| format!("{value:.6e}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                prior_mean_trace.rchunks(n_primary).next().unwrap_or(&[]),
                prior_covariance_trace
                    .rchunks(n_primary * n_primary)
                    .next()
                    .unwrap_or(&[]),
                prior_specific_sd_trace.rchunks(n_specific).next().unwrap_or(&[]),
            )
        })? {
            final_loglik_change = change;
            // A rejected combined update restores the prior iteration state;
            // its next LL is flat by construction, not evidence of convergence.
            if !rolled_back
                && fipc_convergence_decision(
                    change,
                    previous.expect("previous loglik exists"),
                    em_map_displacement,
                    cfg.tol,
                    full_step_accepted,
                ) == FipcStopDecision::Converged
            {
                converged = true;
                termination_reason = "tolerance_met".to_string();
                break;
            }
        }
        rolled_back = false;
        if n_iter == cfg.max_iter {
            if n_iter > 0 && !full_step_accepted {
                termination_reason = "step_limited".to_string();
            }
            break;
        }
        let previous_params = params.clone();
        let previous_mean = mean.clone();
        let previous_covariance = covariance.clone();
        let previous_specific_sd = specific_sd.clone();
        let baseline_mean = previous_mean.clone();
        let baseline_covariance = previous_covariance.clone();
        let baseline_specific_sd = previous_specific_sd.clone();
        for i in 0..n_items {
            if anchor[i] {
                continue;
            }
            let free = &v.free_primaries[i];
            let has_specific = v.item_block[i].is_some();
            // Counts use node g * qs + h. The objective indexes the compact
            // primary support at g and this item's specific support at h;
            // expanding primary nodes here would repeat the g index twice.
            // Basis: Cai (2010, pp. 608-609, Appendix A), item complete-data
            // likelihood evaluated at its corresponding quadrature tuple.
            // Reference: Cai, L. (2010). A two-tier full-information item factor
            // analysis model with applications. Psychometrika, 75(4), 581-612.
            // doi:10.1007/s11336-010-9178-0.
            let item_ts = v.item_block[i]
                .map(|specific| ts_by_specific[specific].as_slice())
                .unwrap_or(&[]);
            let mut packed = Vec::with_capacity(free.len() + usize::from(has_specific) + v.m1);
            for &d in free {
                packed.push(params[i].a_p[d]);
            }
            if let Some(a_s) = params[i].a_s {
                packed.push(a_s);
            }
            packed.extend_from_slice(&params[i].d);
            let updated = m_step_item(
                packed,
                free,
                has_specific,
                &coords,
                item_ts,
                n_primary,
                n_grid,
                ts_std.len(),
                &counts[i],
                n_cat,
                cfg.ridge,
                cfg.newton_iter,
            );
            for (slot, &d) in free.iter().enumerate() {
                params[i].a_p[d] = updated[slot];
            }
            if has_specific {
                params[i].a_s = Some(updated[free.len()]);
                params[i].d = updated[free.len() + 1..].to_vec();
            } else {
                params[i].d = updated[free.len()..].to_vec();
            }
        }
        let mass = n_persons as f64;
        for d in 0..n_primary {
            mean[d] = sum_primary[d] / mass;
        }
        for j in 0..n_primary {
            for k in 0..n_primary {
                covariance[j * n_primary + k] =
                    sum_primary2[j * n_primary + k] / mass - mean[j] * mean[k];
            }
        }
        for d in 0..n_primary {
            covariance[d * n_primary + d] += 1e-10;
        }
        if cfg.estimate_specific_vars {
            for s in 0..n_specific {
                if specific_mass[s] <= 0.0 {
                    return Err(format!("focal specific-{s} has no posterior mass"));
                }
                let variance = sum_specific2[s] / specific_mass[s];
                if !variance.is_finite() || variance <= 0.0 {
                    return Err(format!(
                        "non-positive focal specific-{s} variance update ({variance:.6e})"
                    ));
                }
                specific_sd[s] = variance.sqrt();
            }
        }
        em_map_displacement = fipc_max_param_change(
            &previous_params,
            &params,
            anchor,
            &baseline_mean,
            &mean,
            &baseline_covariance,
            &covariance,
            &baseline_specific_sd,
            &specific_sd,
            n_primary,
            cfg.estimate_specific_vars,
        );
        // The direct-GH reparameterization keeps standard weights but moves
        // the support after each focal-prior update. Item sufficient
        // statistics were formed on the pre-update support, so accept the
        // prior update only when its remapped observed-data objective does not
        // regress; backtracking keeps the adopted direct-GH target intact.
        let candidate_ll = {
            let candidate_chol = cholesky_lower(&covariance, n_primary);
            candidate_chol.map(|(candidate_chol, _)| {
                let candidate_coords =
                    fipc_primary_coords(&base_coords, &mean, &candidate_chol, n_primary, n_grid);
                let candidate_ts: Vec<Vec<f64>> = (0..n_specific)
                    .map(|s| ts_std.iter().map(|&x| x * specific_sd[s]).collect())
                    .collect();
                let candidate_weights: Vec<Vec<f64>> =
                    (0..n_specific).map(|_| log_ws.clone()).collect();
                focal_loglik(
                    &v,
                    y,
                    observed,
                    &params,
                    &log_w0,
                    &candidate_weights,
                    &candidate_coords,
                    &candidate_ts,
                    n_grid,
                    ts_std.len(),
                    cfg.device,
                )
            })
        };
        let acceptance_tolerance = 32.0 * f64::EPSILON * (1.0 + ll.abs());
        // Before frozen diagnostics were separated, the recovery baseline was
        // the same current remapped E-step likelihood as `ll`. Keep that measure
        // here; frozen standard-normal diagnostics must not gate recovery.
        let recovery_improve_eps = 32.0 * f64::EPSILON * (1.0 + ll.abs());
        full_step_accepted = candidate_ll
            .is_some_and(|value| value.is_finite() && value >= ll - acceptance_tolerance);
        if !full_step_accepted {
            let target_mean = mean.clone();
            let target_covariance = covariance.clone();
            let target_specific_sd = specific_sd.clone();
            let target_params = params.clone();
            let joint_ll = candidate_ll;
            let joint_pd = cholesky_lower(&covariance, n_primary).is_some();
            let mut alpha = 0.5;
            let mut accepted = false;
            let mut decision = format!(
                "iter={n_iter};branch=joint_reject;ll={ll:.10e};fixed_ll={fixed_ll:.10e};joint_ll={};joint_pd={joint_pd};target_mean={target_mean:?};baseline_mean={baseline_mean:?}",
                joint_ll
                    .map(|v| format!("{v:.10e}"))
                    .unwrap_or_else(|| "none".into())
            );
            while alpha >= 1e-6 {
                for i in 0..n_items {
                    if anchor[i] {
                        continue;
                    }
                    for j in 0..params[i].a_p.len() {
                        params[i].a_p[j] = previous_params[i].a_p[j]
                            + alpha * (target_params[i].a_p[j] - previous_params[i].a_p[j]);
                    }
                    params[i].a_s = match (previous_params[i].a_s, target_params[i].a_s) {
                        (Some(previous), Some(target)) => {
                            Some(previous + alpha * (target - previous))
                        }
                        (None, None) => None,
                        _ => target_params[i].a_s,
                    };
                    for j in 0..params[i].d.len() {
                        params[i].d[j] = previous_params[i].d[j]
                            + alpha * (target_params[i].d[j] - previous_params[i].d[j]);
                    }
                }
                for d in 0..n_primary {
                    mean[d] = previous_mean[d] + alpha * (target_mean[d] - previous_mean[d]);
                }
                for j in 0..n_primary * n_primary {
                    covariance[j] = previous_covariance[j]
                        + alpha * (target_covariance[j] - previous_covariance[j]);
                }
                for s in 0..n_specific {
                    specific_sd[s] = previous_specific_sd[s]
                        + alpha * (target_specific_sd[s] - previous_specific_sd[s]);
                }
                let Some((candidate_chol, _)) = cholesky_lower(&covariance, n_primary) else {
                    alpha *= 0.5;
                    continue;
                };
                let candidate_coords =
                    fipc_primary_coords(&base_coords, &mean, &candidate_chol, n_primary, n_grid);
                let candidate_ts: Vec<Vec<f64>> = (0..n_specific)
                    .map(|s| ts_std.iter().map(|&x| x * specific_sd[s]).collect())
                    .collect();
                let candidate_weights: Vec<Vec<f64>> =
                    (0..n_specific).map(|_| log_ws.clone()).collect();
                let remapped_ll = focal_loglik(
                    &v,
                    y,
                    observed,
                    &params,
                    &log_w0,
                    &candidate_weights,
                    &candidate_coords,
                    &candidate_ts,
                    n_grid,
                    ts_std.len(),
                    cfg.device,
                );
                if remapped_ll.is_finite() && remapped_ll >= ll - acceptance_tolerance {
                    accepted = true;
                    decision = format!(
                        "iter={n_iter};branch=joint_backtrack_accept;alpha={alpha:.3e};ll={ll:.10e};cand_ll={remapped_ll:.10e};mean={mean:?}"
                    );
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                params = previous_params.clone();
                mean = previous_mean.clone();
                covariance = previous_covariance.clone();
                specific_sd = previous_specific_sd.clone();
                // Mean-first recovery on the restored baseline. A scale-first
                // trust region previously accepted covariance drift while the
                // mean loop either never ran (pre-pairing) or only tried
                // alpha<=0.1; when a remapped mean step improves the Class-A
                // objective, take the largest feasible step from 1.0.
                let mut mean_accepted = false;
                let mut mean_alpha = 1.0;
                let mut mean_reject_detail = String::from("mean_not_tried");
                while !mean_accepted && mean_alpha >= 1e-6 {
                    let candidate_mean: Vec<f64> = mean
                        .iter()
                        .zip(&target_mean)
                        .map(|(&old, &target)| old + mean_alpha * (target - old))
                        .collect();
                    let mean_ll = direct_fipc_loglik(
                        &v,
                        y,
                        observed,
                        &params,
                        &base_coords,
                        &log_w0,
                        &log_ws,
                        ts_std,
                        &candidate_mean,
                        &covariance,
                        &specific_sd,
                        n_grid,
                        cfg.device,
                    );
                    let mean_pd = cholesky_lower(&covariance, n_primary).is_some();
                    let passes_ll_guard = mean_ll.is_some_and(|value| {
                        value.is_finite() && value >= ll - acceptance_tolerance
                    });
                    let passes_fixed_improve = mean_ll.is_some_and(|value| {
                        value.is_finite() && value > ll + recovery_improve_eps
                    });
                    if passes_ll_guard && passes_fixed_improve {
                        mean = candidate_mean;
                        mean_accepted = true;
                        decision = format!(
                            "iter={n_iter};branch=mean_accept;alpha={mean_alpha:.3e};ll={ll:.10e};fixed_ll={fixed_ll:.10e};mean_ll={:.10e};scale_first=false;mean_pd={mean_pd};mean={mean:?}",
                            mean_ll.unwrap_or(f64::NAN)
                        );
                        break;
                    }
                    mean_reject_detail = format!(
                        "alpha={mean_alpha:.3e};mean_ll={};passes_ll_guard={passes_ll_guard};passes_fixed_improve={passes_fixed_improve};cand_mean={candidate_mean:?}",
                        mean_ll
                            .map(|v| format!("{v:.10e}"))
                            .unwrap_or_else(|| "none".into())
                    );
                    mean_alpha *= 0.5;
                }
                // Scale recovery after mean: keep any improving covariance /
                // specific-SD step under the same remapped LL guard. Do not
                // undo a valid scale step when mean already had its chance.
                let mut scale_accepted = false;
                let mut scale_alpha = 0.1;
                let mut scale_ll_best = None;
                while scale_alpha >= 1e-6 {
                    let candidate_covariance: Vec<f64> = covariance
                        .iter()
                        .zip(&target_covariance)
                        .map(|(&old, &target)| old + scale_alpha * (target - old))
                        .collect();
                    let candidate_specific_sd: Vec<f64> = specific_sd
                        .iter()
                        .zip(&target_specific_sd)
                        .map(|(&old, &target)| old + scale_alpha * (target - old))
                        .collect();
                    let scale_ll = direct_fipc_loglik(
                        &v,
                        y,
                        observed,
                        &params,
                        &base_coords,
                        &log_w0,
                        &log_ws,
                        ts_std,
                        &mean,
                        &candidate_covariance,
                        &candidate_specific_sd,
                        n_grid,
                        cfg.device,
                    );
                    let scale_pd = cholesky_lower(&candidate_covariance, n_primary).is_some();
                    let scale_moved_materially = candidate_covariance
                        .iter()
                        .zip(&covariance)
                        .any(|(&new, &old)| (new - old).abs() > 1e-3)
                        || candidate_specific_sd
                            .iter()
                            .zip(&specific_sd)
                            .any(|(&new, &old)| (new - old).abs() > 1e-3);
                    if scale_ll.is_some_and(|value| {
                        value.is_finite()
                            && value >= ll - acceptance_tolerance
                            && value > ll + recovery_improve_eps
                            && scale_moved_materially
                    }) {
                        covariance = candidate_covariance;
                        specific_sd = candidate_specific_sd;
                        scale_accepted = true;
                        scale_ll_best = scale_ll;
                        if mean_accepted {
                            decision = format!(
                                "iter={n_iter};branch=mean_then_scale_accept;mean_alpha={mean_alpha:.3e};scale_alpha={scale_alpha:.3e};ll={ll:.10e};fixed_ll={fixed_ll:.10e};scale_ll={:.10e};scale_pd={scale_pd};mean={mean:?}",
                                scale_ll.unwrap_or(f64::NAN)
                            );
                        } else {
                            decision = format!(
                                "iter={n_iter};branch=scale_only_accept;alpha={scale_alpha:.3e};ll={ll:.10e};fixed_ll={fixed_ll:.10e};scale_ll={:.10e};scale_pd={scale_pd};{mean_reject_detail}",
                                scale_ll.unwrap_or(f64::NAN)
                            );
                        }
                        break;
                    }
                    scale_alpha *= 0.5;
                }
                accepted = mean_accepted || scale_accepted;
                if !accepted {
                    decision = format!(
                        "iter={n_iter};branch=full_rollback;ll={ll:.10e};fixed_ll={fixed_ll:.10e};scale_ll={};{mean_reject_detail};target_mean={target_mean:?}",
                        scale_ll_best
                            .map(|v| format!("{v:.10e}"))
                            .unwrap_or_else(|| "none".into())
                    );
                }
            }
            prior_update_decision_trace.push(decision);
            if accepted {
                n_accepted_prior_steps += 1;
                consecutive_rollback = 0;
            } else {
                rolled_back = true;
                n_rollback_full += 1;
                consecutive_rollback += 1;
            }
        } else {
            prior_update_decision_trace.push(format!(
                "iter={n_iter};branch=joint_full_accept;ll={ll:.10e};cand_ll={:.10e};mean={mean:?}",
                candidate_ll.unwrap_or(f64::NAN)
            ));
            n_accepted_prior_steps += 1;
            consecutive_rollback = 0;
        }
        prior_mean_trace.extend_from_slice(&mean);
        prior_covariance_trace.extend_from_slice(&covariance);
        prior_specific_sd_trace.extend_from_slice(&specific_sd);
        n_iter += 1;
        if consecutive_rollback >= MAX_CONSECUTIVE_ROLLBACKS {
            termination_reason = "prior_update_stalled".to_string();
            break;
        }
    }
    let (chol, _) = cholesky_lower(&covariance, n_primary)
        .ok_or_else(|| "final focal primary covariance is not positive-definite".to_string())?;
    let coords = fipc_primary_coords(&base_coords, &mean, &chol, n_primary, n_grid);
    let ts_by_specific: Vec<Vec<f64>> = (0..n_specific)
        .map(|s| ts_std.iter().map(|&x| x * specific_sd[s]).collect())
        .collect();
    // Keep the final EAP pass on exactly the same direct-quadrature measure.
    let log_ws_by_specific: Vec<Vec<f64>> = (0..n_specific).map(|_| log_ws.clone()).collect();
    let log_w = log_w0.clone();
    let (_, _, _, _, _, _, theta_p_eap, theta_p_sd) = e_step_fipc(
        &v,
        y,
        observed,
        &params,
        &log_w,
        &log_ws_by_specific,
        &coords,
        &ts_by_specific,
        n_grid,
        ts_std.len(),
        cfg.device,
    );
    let mut a_primary = vec![0.0; n_items * n_primary];
    let mut a_specific = vec![0.0; n_items];
    let mut threshold = vec![0.0; n_items * v.m1];
    let mut category_counts = vec![0usize; n_items * n_cat];
    for (i, par) in params.iter().enumerate() {
        a_primary[i * n_primary..(i + 1) * n_primary].copy_from_slice(&par.a_p);
        if let Some(a_s) = par.a_s {
            a_specific[i] = a_s;
        }
        threshold[i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
        for p in 0..n_persons {
            if is_obs(p, i) {
                category_counts[i * n_cat + y[p * n_items + i]] += 1;
            }
        }
    }
    let mut n_parameters = n_primary + n_primary * (n_primary + 1) / 2;
    for i in 0..n_items {
        if !anchor[i] {
            n_parameters +=
                v.free_primaries[i].len() + usize::from(v.item_block[i].is_some()) + v.m1;
        }
    }
    if cfg.estimate_specific_vars {
        n_parameters += n_specific;
    }
    let primary_sd = (0..n_primary)
        .map(|d| covariance[d * n_primary + d].max(0.0).sqrt())
        .collect();
    let gpu_receipt = crate::gpu_bifactor::gpu_dispatch_receipt();
    Ok(TwoTierFipcResult {
        a_primary,
        a_specific,
        threshold,
        primary_mean: mean,
        primary_cov: covariance,
        primary_sd,
        specific_sd,
        theta_p_eap,
        theta_p_sd,
        category_counts,
        loglik_trace,
        fixed_loglik_trace,
        fixed_primary_first_moment_trace,
        fixed_primary_second_moment_trace,
        fixed_specific_second_moment_trace,
        prior_mean_trace,
        prior_covariance_trace,
        prior_specific_sd_trace,
        n_iter,
        converged,
        termination_reason,
        final_loglik_change,
        final_param_change: em_map_displacement,
        n_parameters,
        n_accepted_prior_steps,
        n_rollback_full,
        consecutive_rollback,
        prior_update_decision_trace,
        gpu_execution_used: gpu_receipt.used,
        gpu_backend: gpu_receipt.backend,
        gpu_device_name: gpu_receipt.device_name,
        cpu_fallback_reason: gpu_receipt.fallback_reason,
    })
}
