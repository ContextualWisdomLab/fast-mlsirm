use super::*;

/// Fixed-specific variance and all-anchor blocks need no specific posterior outputs.
/// Likelihood and primary posteriors still integrate every observed item.
/// Basis: Cai (2010, pp. 608–609, Appendix A, distinct item and latent E-step tables).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn fixed_specific_projection_skips_only_unconsumed_block_posteriors() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 4,
        n_items: 4,
        n_primary: 1,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: q + 1,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1], vec![2]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(1), None],
    };
    let y: Vec<usize> = (0..16).map(|i| (i / 4 + i % 4) % 3).collect();
    let mask: Vec<bool> = (0..16).map(|i| i != 2 && i != 9).collect();
    let params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![-1.1 + i as f64 * 0.1],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut coords: Vec<f64> = x.iter().map(|t| 0.3 + 1.2 * t).collect();
    coords.insert(0, -43.);
    let mut lw: Vec<f64> = w.iter().map(|t| t.ln()).collect();
    lw.insert(0, f64::NEG_INFINITY);
    let lws = vec![w.iter().map(|t| t.ln()).collect::<Vec<_>>(); 2];
    let mut ts = vec![x.iter().map(|t| 0.75 * t).collect::<Vec<_>>(); 2];
    let anchor = [true, true, false, true];
    for phase in 0..2 {
        let observed = if phase == 0 {
            None
        } else {
            Some(mask.as_slice())
        };
        let previous = e_step_fipc_anchored(
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
        SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.set(0));
        let actual = e_step_fipc_anchored_moments(
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
            false,
        );
        assert_eq!(actual.0.to_bits(), previous.0.to_bits());
        for (a, b) in [
            (&actual.2, &previous.2),
            (&actual.3, &previous.3),
            (&actual.6, &previous.6),
            (&actual.7, &previous.7),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        for (a, b) in actual.1.iter().zip(&previous.1) {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                assert_eq!(a.len(), b.len());
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
        }
        let observed_free = (0..v.n_persons)
            .filter(|&pp| observed.is_none_or(|o| o[pp * 4 + 2]))
            .count();
        assert_eq!(
            SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.get()),
            observed_free * (q + 1) * q,
            "unused specific posterior materialized for all-fixed block"
        );
        // Estimating specific variances must retain the complete moment vectors.
        let retained = e_step_fipc_anchored_moments(
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
            true,
        );
        for (a, b) in [(&retained.4, &previous.4), (&retained.5, &previous.5)] {
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        // When every observed free item is masked, no block posterior is consumed.
        let anchor_only: Vec<bool> = (0..16).map(|i| i % 4 != 2).collect();
        SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.set(0));
        let masked = e_step_fipc_anchored_moments(
            &v,
            &y,
            Some(&anchor_only),
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            q + 1,
            q,
            crate::Device::Cpu,
            &anchor,
            false,
        );
        assert_eq!(SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.get()), 0);
        let masked_full = e_step_fipc_anchored(
            &v,
            &y,
            Some(&anchor_only),
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
        assert_eq!(masked.0.to_bits(), masked_full.0.to_bits());
        for (a, b) in [
            (&masked.2, &masked_full.2),
            (&masked.3, &masked_full.3),
            (&masked.6, &masked_full.6),
            (&masked.7, &masked_full.7),
        ] {
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        coords[10] += 0.125;
        ts[1][10] += 0.1;
        lw[10] -= 0.01;
    }
}

/// Actual fit caller must project unused fixed-specific outputs in moving EM only.
/// Basis and reference: Cai (2010), same Appendix A and citation above.
#[test]
fn fitter_fixed_specific_posterior_work_matches_consumed_blocks() {
    let n = 6;
    let q = 121;
    let y: Vec<usize> = (0..24).map(|i| (i / 4 + i % 4) % 3).collect();
    let cfg = TwoTierFipcConfig {
        q_primary: q,
        q_specific: q,
        max_iter: 1,
        tol: 1e-5,
        newton_iter: 1,
        ridge: 1e-8,
        estimate_specific_vars: false,
        device: crate::Device::Cpu,
    };
    SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.set(0));
    let fit = fit_two_tier_grm_fipc(
        &y,
        None,
        &[true; 4],
        &[0, 0, 1, 1],
        n,
        4,
        1,
        2,
        3,
        &[true, true, false, true],
        &[1.1; 4],
        &[0.8; 4],
        &[0.8, -0.8, 0.8, -0.8, 0.8, -0.8, 0.8, -0.8],
        &cfg,
    )
    .unwrap();
    assert_eq!(fit.n_iter, 1);
    assert!(!fit.converged);
    // Initial+next moving pass each need one block; unchanged final EAP needs two.
    assert_eq!(
        SPECIFIC_POSTERIOR_EVALUATIONS.with(|c| c.get()),
        4 * n * q * q,
        "fitter bypassed fixed-specific posterior projection"
    );
}

/// Compare every returned numerical bit, state and device field to exact97c4dd3.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
fn assert_same_fit(result: &TwoTierFipcResult, previous: &TwoTierFipcResult) {
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

/// Original structural pattern has one all-anchor block and one mixed block.
/// Pair fixed and estimated specific variances, missing items and all-fixed fits.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn original_structural_pattern_and_estimated_specific_fitter_are_bitwise_unchanged() {
    let n = 6;
    let y: Vec<usize> = (0..n * 6).map(|i| (i / 6 + i % 6) % 4).collect();
    let map = [
        true, false, true, false, true, false, false, true, false, true, false, true,
    ];
    let sm = [0, 0, 1, 0, 1, 1];
    let anchors = [true, true, false, true, true, false];
    let ap = [1.4, 0., 1.1, 0., 1., 0., 0., 1.2, 0., 0.9, 0., 1.1];
    let ass = [1., 0.9, 1.1, 1., 0.8, 0.9];
    let thresholds: Vec<f64> = (0..6)
        .flat_map(|i| [1.2 + i as f64 * 0.01, 0., -1.2])
        .collect();
    // Mask only a duplicated observed category; every category remains identified.
    let masks: Vec<bool> = (0..36).map(|i| i != 1).collect();
    for phase in 0..4 {
        let cfg = TwoTierFipcConfig {
            q_primary: 121,
            q_specific: 121,
            max_iter: 2,
            tol: 1e-5,
            newton_iter: 1,
            ridge: 1e-8,
            estimate_specific_vars: phase == 1,
            device: crate::Device::Cpu,
        };
        let obs = if phase == 2 {
            Some(masks.as_slice())
        } else {
            None
        };
        let an = if phase == 3 { &[true; 6] } else { &anchors };
        let previous = specific_projection_legacy_fitter::predecessor_fit_two_tier_grm_fipc(
            &y,
            obs,
            &map,
            &sm,
            n,
            6,
            2,
            2,
            4,
            an,
            &ap,
            &ass,
            &thresholds,
            &cfg,
        )
        .unwrap();
        let result = fit_two_tier_grm_fipc(
            &y,
            obs,
            &map,
            &sm,
            n,
            6,
            2,
            2,
            4,
            an,
            &ap,
            &ass,
            &thresholds,
            &cfg,
        )
        .unwrap();
        assert_same_fit(&result, &previous);
    }
}
