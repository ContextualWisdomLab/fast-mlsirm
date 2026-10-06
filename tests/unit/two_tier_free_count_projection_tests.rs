use super::*;

/// Fixed item rows do not consume their category count tables in the M-step.
/// Preserve likelihood, moments, EAP and free counts exactly.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn anchored_count_projection_preserves_consumed_outputs_without_fixed_count_tables() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 8,
        n_items: 4,
        n_primary: 1,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: q + 1,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1, 2], vec![]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(0), None],
    };
    let y: Vec<usize> = (0..32).map(|i| (i / 4 + i % 4) % 3).collect();
    let mask: Vec<bool> = (0..32)
        .map(|i| !(i / 4 % 4 == 1 && i % 4 < 3) && i % 7 != 0)
        .collect();
    let mut params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![if i == 0 { -1.1 } else { 1.0 + i as f64 * 0.1 }],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut coords: Vec<f64> = x.iter().map(|t| 0.35 + 1.2 * t).collect();
    coords.insert(0, -43.0);
    let mut lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    lw.insert(0, f64::NEG_INFINITY);
    let mut lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>(); 2];
    let mut ts = vec![x.iter().map(|t| 0.75 * t).collect::<Vec<_>>(); 2];
    for phase in 0..4 {
        let observed = if phase % 2 == 0 {
            Some(mask.as_slice())
        } else {
            None
        };
        let legacy = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
            &v,
            &y,
            observed,
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            q + 1,
            q,
        );
        let anchor = [true, false, true, true];
        let actual = e_step_fipc_anchored(
            &v,
            &y,
            observed,
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            q + 1,
            q,
            crate::Device::Cpu,
            &anchor,
        );
        assert_eq!(actual.0.to_bits(), legacy.0.to_bits(), "LL phase{phase}");
        for (a, b) in [
            (&actual.2, &legacy.2),
            (&actual.3, &legacy.3),
            (&actual.4, &legacy.4),
            (&actual.5, &legacy.5),
            (&actual.6, &legacy.6),
            (&actual.7, &legacy.7),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "full output phase{phase}"
            );
        }
        assert_eq!(actual.1.len(), legacy.1.len());
        for (i, (a, b)) in actual.1.iter().zip(&legacy.1).enumerate() {
            if anchor[i] {
                continue;
            }
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.len(), b.len());
                assert!(
                    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "counts phase{phase}"
                );
            }
        }
        let unused_rows: usize = actual
            .1
            .iter()
            .enumerate()
            .filter(|(i, _)| anchor[*i])
            .map(|(_, rows)| rows.len())
            .sum();
        assert_eq!(
            unused_rows, 0,
            "constructed {unused_rows} unused fixed-item count rows"
        );
        params[0].d[0] += 0.125;
        coords[10] += 0.25;
        ts[0][10] += 0.125;
        lws[0][10] -= 0.01;
        lw[10] -= 0.015;
    }
}

/// Characterize the actual fitter's use of anchor-count projection.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn focal_fitter_requests_only_free_item_count_tables() {
    let n = 6;
    let ni = 4;
    let nc = 3;
    let y: Vec<usize> = (0..n * ni).map(|i| (i / ni + i % ni) % nc).collect();
    let cfg = TwoTierFipcConfig {
        q_primary: 121,
        q_specific: 121,
        max_iter: 1,
        tol: 1e-5,
        newton_iter: 1,
        ridge: 1e-8,
        estimate_specific_vars: false,
        device: crate::Device::Cpu,
    };
    FIT_ANCHOR_PROJECTION_CALLS.with(|c| c.set(0));
    let result = fit_two_tier_grm_fipc(
        &y,
        None,
        &[true; 4],
        &[0, 0, 0, 0],
        n,
        ni,
        1,
        1,
        nc,
        &[true, true, false, true],
        &[1.1; 4],
        &[0.8; 4],
        &[0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8],
        &cfg,
    )
    .unwrap();
    assert_eq!(result.n_iter, 1);
    assert!(!result.converged);
    assert!(result.loglik_trace.iter().all(|v| v.is_finite()));
    assert_eq!(
        FIT_ANCHOR_PROJECTION_CALLS.with(|c| c.get()),
        2,
        "focal fitter bypassed free-item-count projection"
    );
    let previous = free_count_legacy_fitter::legacy_fit_two_tier_grm_fipc(
        &y,
        None,
        &[true; 4],
        &[0, 0, 0, 0],
        n,
        ni,
        1,
        1,
        nc,
        &[true, true, false, true],
        &[1.1; 4],
        &[0.8; 4],
        &[0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8],
        &cfg,
    )
    .unwrap();
    for (a, b) in [
        (&result.a_primary, &previous.a_primary),
        (&result.a_specific, &previous.a_specific),
        (&result.threshold, &previous.threshold),
        (&result.primary_mean, &previous.primary_mean),
        (&result.primary_cov, &previous.primary_cov),
        (&result.primary_sd, &previous.primary_sd),
        (&result.specific_sd, &previous.specific_sd),
        (&result.theta_p_eap, &previous.theta_p_eap),
        (&result.theta_p_sd, &previous.theta_p_sd),
        (&result.loglik_trace, &previous.loglik_trace),
        (&result.fixed_loglik_trace, &previous.fixed_loglik_trace),
        (
            &result.fixed_primary_first_moment_trace,
            &previous.fixed_primary_first_moment_trace,
        ),
        (
            &result.fixed_primary_second_moment_trace,
            &previous.fixed_primary_second_moment_trace,
        ),
        (
            &result.fixed_specific_second_moment_trace,
            &previous.fixed_specific_second_moment_trace,
        ),
        (&result.prior_mean_trace, &previous.prior_mean_trace),
        (
            &result.prior_covariance_trace,
            &previous.prior_covariance_trace,
        ),
        (
            &result.prior_specific_sd_trace,
            &previous.prior_specific_sd_trace,
        ),
    ] {
        assert_eq!(a.len(), b.len());
        assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
    }
    assert_eq!(result.category_counts, previous.category_counts);
    assert_eq!(result.n_iter, previous.n_iter);
    assert_eq!(result.converged, previous.converged);
    assert_eq!(result.termination_reason, previous.termination_reason);
    assert_eq!(
        result.final_loglik_change.to_bits(),
        previous.final_loglik_change.to_bits()
    );
    assert_eq!(
        result.final_param_change.to_bits(),
        previous.final_param_change.to_bits()
    );
    assert_eq!(result.n_parameters, previous.n_parameters);
    assert_eq!(
        result.n_accepted_prior_steps,
        previous.n_accepted_prior_steps
    );
    assert_eq!(result.n_rollback_full, previous.n_rollback_full);
    assert_eq!(result.consecutive_rollback, previous.consecutive_rollback);
    assert_eq!(
        result.prior_update_decision_trace,
        previous.prior_update_decision_trace
    );
    assert_eq!(result.gpu_execution_used, previous.gpu_execution_used);
    assert_eq!(result.gpu_backend, previous.gpu_backend);
    assert_eq!(result.gpu_device_name, previous.gpu_device_name);
    assert_eq!(result.cpu_fallback_reason, previous.cpu_fallback_reason);
}

/// Characterize full P2 output parity without changing the fitting contract.
/// Basis: Cai (2010, pp. 607–608, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn anchor_projection_p2_preserves_all_consumed_outputs() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let ng = q * q;
    let v = Validated {
        n_persons: 2,
        n_items: 2,
        n_primary: 2,
        n_specific: 1,
        n_cat: 3,
        m1: 2,
        grid_size: ng,
        free_primaries: vec![vec![0, 1]; 2],
        blocks: vec![vec![0]],
        specific_free: vec![1],
        item_block: vec![Some(0), None],
    };
    let y = vec![1, 0, 1, 2];
    let params = vec![
        ItemParams {
            a_p: vec![-1.1, 0.8],
            a_s: Some(-0.7),
            d: vec![0.8, -0.8],
        },
        ItemParams {
            a_p: vec![0.9, 0.7],
            a_s: None,
            d: vec![0.8, -0.8],
        },
    ];
    let mut coords = Vec::new();
    let mut lw = Vec::new();
    for g in 0..q {
        for h in 0..q {
            coords.extend_from_slice(&[0.3 + 1.2 * x[g], -0.2 + 0.25 * x[g] + 0.9 * x[h]]);
            lw.push(w[g].ln() + w[h].ln());
        }
    }
    let mut ts = vec![x.iter().map(|t| 0.75 * t).collect::<Vec<_>>()];
    let lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>()];
    for phase in 0..2 {
        let previous = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
            &v, &y, None, &params, &lw, &lws, &coords, &ts, ng, q,
        );
        FULL_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.set(0));
        let anchor = if phase == 0 {
            [true, false]
        } else {
            [false, true]
        };
        let actual = e_step_fipc_anchored(
            &v,
            &y,
            None,
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            ng,
            q,
            crate::Device::Cpu,
            &anchor,
        );
        assert_eq!(actual.0.to_bits(), previous.0.to_bits());
        for (a, b) in [
            (&actual.2, &previous.2),
            (&actual.3, &previous.3),
            (&actual.4, &previous.4),
            (&actual.5, &previous.5),
            (&actual.6, &previous.6),
            (&actual.7, &previous.7),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "P2 output phase{phase}"
            );
        }
        for (i, (a, b)) in actual.1.iter().zip(&previous.1).enumerate() {
            if anchor[i] {
                assert!(a.is_empty());
                continue;
            }
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.len(), b.len());
                assert!(
                    a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "P2 counts phase{phase}"
                );
            }
        }
        assert_eq!(FULL_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.get()), ng);
        coords[10] += 0.125;
        ts[0][10] += 0.1;
        lw[10] -= 0.01;
    }
}
