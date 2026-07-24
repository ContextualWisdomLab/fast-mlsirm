use super::*;

#[test]
fn fitters_reject_unbounded_categories_and_iterations() {
    let y = [0usize];
    assert!(
        fit_poly_unidim(&y, None, 1, 1, POLY_MAX_CAT + 1, PolyModel::Grm, 7, 1, 1e-6,).is_err()
    );
    assert!(fit_poly_unidim(
        &y,
        None,
        1,
        1,
        2,
        PolyModel::Grm,
        7,
        POLY_MAX_ITER + 1,
        1e-6,
    )
    .is_err());
}

#[test]
fn poly_public_boundaries_and_small_diagnostic_paths() {
    assert_eq!(grm_logprobs(0.0, &[]), vec![0.0]);
    assert_eq!(grm_node_gradient(0.0, &[], &[]), (0.0, Vec::new()));
    let (base, threshold) = grm_node_gradient(0.0, &[0.5], &[0.0, 0.0]);
    assert_eq!(base, 0.0);
    assert_eq!(threshold, vec![0.0]);
    assert_eq!(solve_small(vec![vec![0.0]], vec![3.0]), vec![3.0]);
    let swapped = solve_small(vec![vec![0.0, 1.0], vec![2.0, 3.0]], vec![1.0, 5.0]);
    assert!(swapped.iter().all(|value| value.is_finite()));
    let (fallback, directional, maximum) = stabilized_newton_step(vec![f64::NAN], &[3.0], 3.0);
    assert_eq!(fallback, vec![2.0]);
    assert_eq!(directional, 6.0);
    assert_eq!(maximum, 2.0);
    assert_eq!(
        stabilized_newton_step(vec![1.0], &[1.0], 1.0),
        (vec![1.0], 1.0, 1.0)
    );
    assert_eq!(checked_em_delta(-4.0, None, 1e-6, 0).unwrap(), None);
    assert!(checked_em_delta(f64::NAN, None, 1e-6, 0).is_err());
    assert!(checked_em_delta(-5.0, Some(-4.0), 1e-6, 1).is_err());
    let accepted = checked_em_delta(-3.5, Some(-4.0), 1e-6, 1)
        .unwrap()
        .unwrap();
    assert_eq!(accepted.0, 0.5);
    assert!((accepted.1 - 5e-6).abs() < 1e-15);
    assert!(matches!(
        multigroup_em_status(f64::NAN, None, 1e-6),
        MultigroupEmStatus::NonFinite
    ));
    assert!(matches!(
        multigroup_em_status(-10.0, None, 1e-6),
        MultigroupEmStatus::First
    ));
    assert!(matches!(
        multigroup_em_status(-11.0, Some(-10.0), 1e-6),
        MultigroupEmStatus::NonMonotone
    ));
    assert!(matches!(
        multigroup_em_status(-9.999_999, Some(-10.0), 1e-6),
        MultigroupEmStatus::Converged { .. }
    ));
    assert!(matches!(
        multigroup_em_status(-9.0, Some(-10.0), 1e-6),
        MultigroupEmStatus::Continue { .. }
    ));

    for (status, reason, stops, trace_len, did_converge) in [
        (MultigroupEmStatus::NonFinite, "non_finite", true, 0, false),
        (
            MultigroupEmStatus::NonMonotone,
            "non_monotone",
            true,
            1,
            false,
        ),
        (
            MultigroupEmStatus::Converged {
                delta: 1e-7,
                tolerance: 1e-6,
            },
            "tolerance",
            true,
            1,
            true,
        ),
        (
            MultigroupEmStatus::Continue {
                delta: 0.5,
                tolerance: 1e-6,
            },
            "max_iter",
            false,
            1,
            false,
        ),
        (MultigroupEmStatus::First, "max_iter", false, 1, false),
    ] {
        let mut trace = Vec::new();
        let mut converged = false;
        let mut termination_reason = "max_iter".to_owned();
        let mut final_delta = f64::NAN;
        let mut stopping_tolerance = 0.0;
        let stopped = record_multigroup_em_status(
            status,
            -10.0,
            &mut trace,
            &mut converged,
            &mut termination_reason,
            &mut final_delta,
            &mut stopping_tolerance,
        );
        assert_eq!(stopped, stops);
        assert_eq!(termination_reason, reason);
        assert_eq!(trace.len(), trace_len);
        assert_eq!(converged, did_converge);
    }

    let compact = TwoGroupPolyFit {
        slope: vec![],
        cat_params: vec![],
        studied_slope: vec![],
        studied_cat: vec![vec![0.0], vec![1.0]],
        mu: vec![],
        sigma: vec![],
        loglik: f64::NAN,
        n_iter: 1,
        converged: false,
        termination_reason: "non_finite".into(),
        loglik_trace: vec![],
        final_delta: f64::NAN,
        stopping_tolerance: 1e-6,
    };
    assert!(validate_poly_dif_compact(&compact, 10).is_err());
    let compact = TwoGroupPolyFit {
        loglik: -10.0,
        ..compact
    };
    assert!(validate_poly_dif_compact(&compact, 10).is_err());
    assert!(poly_dif_metrics(&compact, -11.0, 3).0.is_nan());
    let augmented = TwoGroupPolyFit {
        converged: true,
        loglik: -9.0,
        ..compact
    };
    let metrics = poly_dif_metrics(&augmented, -10.0, 3);
    assert_eq!(metrics.0, 2.0);
    assert_eq!(metrics.2, 1.0);

    let y = [0usize, 1, 2, 1];
    let observed = [true, false, true, true];
    for args in [
        fit_poly_unidim(&[], None, 0, 1, 3, PolyModel::Gpcm, 7, 1, 1e-6),
        fit_poly_unidim(&y, None, 2, 2, 3, PolyModel::Gpcm, 7, 1, 0.0),
        fit_poly_unidim(&y[..3], None, 2, 2, 3, PolyModel::Gpcm, 7, 1, 1e-6),
        fit_poly_unidim(&[0, 1, 3, 1], None, 2, 2, 3, PolyModel::Gpcm, 7, 1, 1e-6),
        fit_poly_unidim(
            &y,
            Some(&observed[..3]),
            2,
            2,
            3,
            PolyModel::Gpcm,
            7,
            1,
            1e-6,
        ),
    ] {
        assert!(args.is_err());
    }

    for result in [
        fit_nominal(&y, None, 2, 2, 1, 7, 1, 1e-6),
        fit_nominal(&y[..3], None, 2, 2, 3, 7, 1, 1e-6),
        fit_nominal(&y, Some(&observed[..3]), 2, 2, 3, 7, 1, 1e-6),
        fit_nominal(&[0, 1, 3, 1], None, 2, 2, 3, 7, 1, 1e-6),
    ] {
        assert!(result.is_err());
    }
    let nominal_missing = fit_nominal(&y, Some(&observed), 2, 2, 3, 7, 1, 1e9).unwrap();
    assert!(nominal_missing.converged);

    let slope = [0.8, 1.2];
    let cat = [0.5, -0.5, 0.8, -0.4];
    for result in [
        poly_person_fit(
            &y,
            None,
            2,
            2,
            1,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            0.0,
            1.0,
            -2.0,
        ),
        poly_person_fit(
            &y,
            None,
            2,
            2,
            3,
            &[1.0],
            &cat,
            PolyModel::Gpcm,
            7,
            0.0,
            1.0,
            -2.0,
        ),
        poly_person_fit(
            &y,
            None,
            2,
            2,
            3,
            &slope,
            &[0.0],
            PolyModel::Gpcm,
            7,
            0.0,
            1.0,
            -2.0,
        ),
        poly_person_fit(
            &y,
            None,
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            0.0,
            0.0,
            -2.0,
        ),
        poly_person_fit(
            &[0, 1, 3, 1],
            None,
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            0.0,
            1.0,
            -2.0,
        ),
    ] {
        assert!(result.is_err());
    }
    let sparse_observed = [false, false, true, true];
    let person_fit = poly_person_fit(
        &y,
        Some(&sparse_observed),
        2,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Grm,
        7,
        0.0,
        1.0,
        100.0,
    )
    .unwrap();
    assert!(person_fit.lz[0].is_nan());

    assert!(poly_cat_simulate(
        &[0.0],
        &slope,
        &cat,
        2,
        1,
        PolyModel::Gpcm,
        7,
        0.0,
        1,
        2,
        true,
        1
    )
    .is_err());
    assert!(poly_cat_simulate(
        &[0.0],
        &[1.0],
        &cat,
        2,
        3,
        PolyModel::Gpcm,
        7,
        0.0,
        1,
        2,
        true,
        1
    )
    .is_err());
    assert!(poly_cat_simulate(
        &[0.0],
        &[1.0],
        &[0.0, 0.0],
        1,
        3,
        PolyModel::Gpcm,
        7,
        0.0,
        1,
        1,
        true,
        1
    )
    .is_err());
    let cat_fit = poly_cat_simulate(
        &[0.0],
        &slope,
        &cat,
        2,
        3,
        PolyModel::Grm,
        7,
        f64::INFINITY,
        1,
        2,
        false,
        0,
    )
    .unwrap();
    assert_eq!(cat_fit.n_used, vec![1]);
    let scored_cat = poly_cat_simulate(
        &[0.0],
        &slope,
        &cat,
        2,
        3,
        PolyModel::Gpcm,
        7,
        0.0,
        2,
        2,
        true,
        2,
    )
    .unwrap();
    assert_eq!(scored_cat.n_used, vec![2]);

    let groups = [0usize, 1];
    for result in [
        fit_poly_multigroup(
            &[],
            None,
            &[],
            2,
            0,
            1,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &groups,
            2,
            2,
            2,
            1,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            0,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            0.0,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &groups,
            1,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y[..3],
            None,
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &[0],
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &[0, 2],
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &[0, 1, 3, 1],
            None,
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            Some(&observed[..3]),
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            None,
            7,
            1,
            1e-6,
        ),
        fit_poly_multigroup(
            &y,
            None,
            &groups,
            2,
            2,
            2,
            3,
            PolyModel::Gpcm,
            Some(2),
            7,
            1,
            1e-6,
        ),
    ] {
        assert!(result.is_err());
    }
    let group_y = [0usize, 99, 1, 2, 0, 2, 1, 2];
    let group_id = [0usize, 0, 1, 1];
    let group_observed = [true, false, true, true, true, true, true, true];
    let grouped = fit_poly_multigroup(
        &group_y,
        Some(&group_observed),
        &group_id,
        2,
        4,
        2,
        3,
        PolyModel::Grm,
        Some(0),
        7,
        1,
        1e9,
    )
    .unwrap();
    // Reads crate output. Kills the mutation that validates masked categories
    // before consulting `observed`.
    assert!(grouped.loglik.is_finite());
    let mut group_observed_bad = group_observed;
    group_observed_bad[1] = true;
    assert!(fit_poly_multigroup(
        &group_y,
        Some(&group_observed_bad),
        &group_id,
        2,
        4,
        2,
        3,
        PolyModel::Grm,
        Some(0),
        7,
        1,
        1e9,
    )
    .is_err());
    assert!(grouped.converged);
    assert_eq!(grouped.termination_reason, "tolerance");
    assert!(poly_dif_sweep(
        &[],
        None,
        &[],
        2,
        0,
        1,
        3,
        PolyModel::Gpcm,
        None,
        7,
        1,
        1e-6,
        0.05,
    )
    .is_err());
    assert!(poly_dif_sweep(
        &group_y,
        None,
        &group_id,
        2,
        4,
        2,
        3,
        PolyModel::Gpcm,
        Some(&[2]),
        7,
        1,
        1e9,
        0.05,
    )
    .is_err());

    for result in [
        u3_poly_person_fit(&y, None, 2, 2, 1, None),
        u3_poly_person_fit(&y[..3], None, 2, 2, 3, None),
        u3_poly_person_fit(&[0, 1, 3, 1], None, 2, 2, 3, None),
        u3_poly_person_fit(&y, Some(&observed[..3]), 2, 2, 3, None),
        u3_poly_person_fit(&y, None, 2, 2, 3, Some(f64::NAN)),
    ] {
        assert!(result.is_err());
    }
    let none_observed = [false, false, true, false];
    let y_masked = [99usize, 99, 1, 99];
    let u3 = u3_poly_person_fit(&y_masked, Some(&none_observed), 2, 2, 3, Some(-1.0)).unwrap();
    // Reads crate output. Kills the mutation that validates masked categories
    // before consulting `observed`.
    assert!(u3.u3poly[0].is_nan());
    assert_eq!(u3.flagged.len(), 2);
    assert!(
        u3_poly_person_fit(&y_masked, Some(&[true, false, true, false]), 2, 2, 3, None).is_err()
    );

    for model in [PolyModel::Gpcm, PolyModel::Grm] {
        let cutoff = u3_poly_bootstrap_cutoff(4, 2, 3, &slope, &cat, model, 0.1, 2, 0).unwrap();
        assert!(cutoff.is_finite());
    }
    let exact_order_statistic =
        u3_poly_bootstrap_cutoff(3, 2, 3, &slope, &cat, PolyModel::Gpcm, 0.5, 1, 7).unwrap();
    assert!(exact_order_statistic.is_finite());
    for result in [
        u3_poly_bootstrap_cutoff(2, 2, 1, &slope, &cat, PolyModel::Gpcm, 0.1, 1, 1),
        u3_poly_bootstrap_cutoff(2, 2, 3, &[1.0], &cat, PolyModel::Gpcm, 0.1, 1, 1),
        u3_poly_bootstrap_cutoff(0, 2, 3, &slope, &cat, PolyModel::Gpcm, 0.1, 1, 1),
        u3_poly_bootstrap_cutoff(2, 2, 3, &slope, &cat, PolyModel::Gpcm, 1.0, 1, 1),
        u3_poly_bootstrap_cutoff(2, 2, 3, &slope, &cat, PolyModel::Gpcm, 0.1, 0, 1),
    ] {
        assert!(result.is_err());
    }

    assert!(poly_information_curves(&[0.0], &slope, &cat, 2, 1, PolyModel::Gpcm).is_err());
    assert!(poly_information_curves(&[0.0], &slope, &[0.0], 2, 3, PolyModel::Gpcm).is_err());
    for model in [PolyModel::Gpcm, PolyModel::Grm] {
        let information =
            poly_information_curves(&[-1.0, 0.0, 1.0], &slope, &cat, 2, 3, model).unwrap();
        assert_eq!(information.len(), 6);
        assert!(information
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0));
    }

    for result in [
        score_poly_eap(&y, None, 2, 2, 1, &slope, &cat, PolyModel::Gpcm, 7),
        score_poly_eap(&y[..3], None, 2, 2, 3, &slope, &cat, PolyModel::Gpcm, 7),
        score_poly_eap(
            &y,
            Some(&observed[..3]),
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
        ),
        score_poly_eap(&y, None, 2, 2, 3, &[1.0], &cat, PolyModel::Gpcm, 7),
    ] {
        assert!(result.is_err());
    }
    let (eap, sd) = score_poly_eap(
        &y,
        Some(&observed),
        2,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Grm,
        7,
    )
    .unwrap();
    assert_eq!(eap.len(), 2);
    assert!(sd.iter().all(|value| value.is_finite() && *value >= 0.0));

    for result in [
        poly_s_x2(&y, None, 2, 2, 1, &slope, &cat, PolyModel::Gpcm, 7, 1.0),
        poly_s_x2(
            &[0],
            None,
            1,
            1,
            3,
            &[1.0],
            &[0.0, 0.0],
            PolyModel::Gpcm,
            7,
            1.0,
        ),
        poly_s_x2(
            &y[..3],
            None,
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            1.0,
        ),
        poly_s_x2(&y, None, 2, 2, 3, &[1.0], &cat, PolyModel::Gpcm, 7, 1.0),
        poly_s_x2(&y, None, 2, 2, 3, &slope, &[0.0], PolyModel::Gpcm, 7, 1.0),
        poly_s_x2(
            &y,
            Some(&observed[..3]),
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            1.0,
        ),
        poly_s_x2(
            &[0, 1, 3, 1],
            None,
            2,
            2,
            3,
            &slope,
            &cat,
            PolyModel::Gpcm,
            7,
            1.0,
        ),
    ] {
        assert!(result.is_err());
    }
    let empty_sx2 = poly_s_x2(
        &[99, 99, 1, 99],
        Some(&[false; 4]),
        2,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Gpcm,
        7,
        f64::INFINITY,
    )
    .unwrap();
    // Reads crate output. Kills the mutation that validates masked categories
    // before consulting `observed`.
    assert_eq!(empty_sx2.n_cells, vec![0, 0]);
    assert!(poly_s_x2(
        &[0, 99, 1, 2],
        Some(&[true, true, true, true]),
        2,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Gpcm,
        7,
        f64::INFINITY,
    )
    .is_err());
    let residual_sx2 = poly_s_x2(
        &[0, 1],
        None,
        1,
        2,
        3,
        &slope,
        &cat,
        PolyModel::Gpcm,
        7,
        f64::INFINITY,
    )
    .unwrap();
    assert_eq!(residual_sx2.n_cells, vec![0, 0]);
}

fn logsumexp0(v: &[f64]) -> f64 {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    m + v.iter().map(|&x| (x - m).exp()).sum::<f64>().ln()
}

#[test]
fn grm_logprobs_normalize_and_binary_parity() {
    // K=2, one threshold: P(Y=1)=sigmoid(base+beta), P(Y=0)=sigmoid(-(base+beta))
    let base = 0.4;
    let beta = -0.3;
    let lp = grm_logprobs(base, &[beta]);
    let z = logsumexp0(&lp);
    assert!(z.abs() < 1e-12, "not normalized: {z}");
    assert!((lp[1] - log_sigmoid(base + beta)).abs() < 1e-12);
    assert!((lp[0] - log_sigmoid(-(base + beta))).abs() < 1e-12);
    // K=4 normalization
    let lp4 = grm_logprobs(0.2, &[1.0, 0.0, -1.2]);
    assert!(logsumexp0(&lp4).abs() < 1e-10);
    assert!(lp4.iter().all(|v| v.is_finite()));
}

#[test]
fn grm_logprobs_and_gradient_remain_finite_at_extreme_bases() {
    let thresholds = [1.0, 0.0];
    let expected_middle = -1000.0 + (-(-1.0_f64).exp()).ln_1p();
    let lp = grm_logprobs(1000.0, &thresholds);
    assert!(lp.iter().all(|value| value.is_finite()), "{lp:?}");
    assert!((lp[1] - expected_middle).abs() < 1e-12, "{lp:?}");
    let (g_base, g_thresholds) = grm_node_gradient(1000.0, &thresholds, &[3.0, 5.0, 2.0]);
    assert!(g_base.is_finite(), "g_base={g_base}");
    assert!(
        g_thresholds.iter().all(|value| value.is_finite()),
        "{g_thresholds:?}"
    );
}

#[test]
fn grm_gradient_matches_finite_difference() {
    let base = 0.3;
    let thr = vec![1.1, 0.1, -0.9]; // decreasing => valid
    let counts = vec![4.0, 6.0, 3.0, 5.0];
    let q = |b: f64, t: &[f64]| -> f64 {
        grm_logprobs(b, t)
            .iter()
            .zip(&counts)
            .map(|(l, r)| r * l)
            .sum()
    };
    let (g_base, g_t) = grm_node_gradient(base, &thr, &counts);
    let h = 1e-6;
    assert!(((q(base + h, &thr) - q(base - h, &thr)) / (2.0 * h) - g_base).abs() < 1e-5);
    for j in 0..thr.len() {
        let mut tp = thr.clone();
        let mut tm = thr.clone();
        tp[j] += h;
        tm[j] -= h;
        let fd = (q(base, &tp) - q(base, &tm)) / (2.0 * h);
        assert!(
            (fd - g_t[j]).abs() < 1e-5,
            "grm g_t[{j}]: {} vs {}",
            fd,
            g_t[j]
        );
    }
}

#[test]
fn gpcm_logprobs_binary_parity_and_monotone() {
    let base = 0.5;
    let b = 0.2;
    let lp = gpcm_logprobs(base, &[0.0, 1.0], &[0.0, b]);
    assert!(logsumexp0(&lp).abs() < 1e-12);
    assert!((lp[1] - log_sigmoid(base + b)).abs() < 1e-12);
    // higher base -> more mass on top category (scores 0,1,2)
    let lo = gpcm_logprobs(-2.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]);
    let hi = gpcm_logprobs(2.0, &[0.0, 1.0, 2.0], &[0.0, 0.0, 0.0]);
    assert!(hi[2].exp() > lo[2].exp());
}

#[test]
fn poly_k2_matches_trusted_binary_mmle() {
    // Cross-validation against an ALREADY-VALIDATED reference (not self-
    // recovery): at K=2 the GPCM cell is exactly the 2PL, P(Y=1) =
    // sigmoid(a*theta + c_1). The polytomous fitter must reproduce the
    // repo's binary MMLE-EM (mmle::fit_mmle_2pl, NumPy-parity + real-data
    // validated) item parameters on the same data, to a small RMSE.
    use crate::mmle::{fit_mmle_2pl, MmleConfig};
    let (n_persons, n_items) = (4000usize, 8usize);
    let mut st = 271828u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.12 * i as f64).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -0.9 + 0.25 * i as f64).collect();
    let mut yf = vec![0.0_f64; n_persons * n_items];
    let mut yi = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let eta = a_true[i] * theta + b_true[i];
            let pr = 1.0 / (1.0 + (-eta).exp());
            let v = if u() < pr { 1.0 } else { 0.0 };
            yf[p * n_items + i] = v;
            yi[p * n_items + i] = v as usize;
        }
    }
    let observed = vec![true; n_persons * n_items];
    let bin = fit_mmle_2pl(
        &yf,
        &observed,
        n_persons,
        n_items,
        &MmleConfig {
            max_iter: 500,
            tol: 1e-7,
            ridge_a: 1e-4,
            ridge_b: 1e-4,
            newton_iter: 25,
        },
    );
    let rmse = |a: &[f64], b: &[f64]| {
        (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64).sqrt()
    };
    // BOTH cells reduce to the 2PL at K=2 (GRM is the default): each must
    // match the trusted binary MMLE's item parameters on the same data.
    for model in [PolyModel::Gpcm, PolyModel::Grm] {
        let poly = fit_poly_unidim(&yi, None, n_persons, n_items, 2, model, 41, 300, 1e-7).unwrap();
        let c1: Vec<f64> = poly.cat_params.iter().map(|c| c[0]).collect();
        let ra = rmse(&poly.slope, &bin.a);
        let rb = rmse(&c1, &bin.b);
        assert!(
            ra < 0.1,
            "{model:?} slope RMSE vs trusted binary MMLE: {ra}"
        );
        assert!(
            rb < 0.1,
            "{model:?} intercept RMSE vs trusted binary MMLE: {rb}"
        );
    }
}

#[test]
fn fit_poly_unidim_recovers_gpcm() {
    let (n_persons, n_items, k) = (4000usize, 6usize, 3usize);
    let mut st = 20260714u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.8 + 0.16 * i as f64).collect();
    let c_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.0, 0.3 - 0.1 * i as f64, -0.2 + 0.15 * i as f64])
        .collect();
    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let lp = gpcm_logprobs(a_true[i] * theta, &scores, &c_true[i]);
            let uu = u();
            let mut cum = 0.0_f64;
            let mut cat = k - 1;
            for (c, l) in lp.iter().enumerate() {
                cum += l.exp();
                if uu < cum {
                    cat = c;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    let fit = fit_poly_unidim(
        &y,
        None,
        n_persons,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        80,
        1e-6,
    )
    .unwrap();
    assert!(fit.loglik.is_finite());
    assert!(fit.converged, "termination={}", fit.termination_reason);
    assert_eq!(fit.termination_reason, "tolerance");
    assert!(fit.n_iter < 80);
    assert_eq!(fit.loglik_trace.len(), fit.n_iter + 1);
    assert_eq!(fit.loglik, *fit.loglik_trace.last().unwrap());
    let previous = fit.loglik_trace[fit.loglik_trace.len() - 2];
    let monotonic_tolerance = 32.0 * f64::EPSILON * (1.0 + previous.abs());
    assert!(fit.final_delta >= -monotonic_tolerance);
    assert!(fit.final_delta <= fit.stopping_tolerance);

    let limited = fit_poly_unidim(
        &y,
        None,
        n_persons,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        1,
        1e-12,
    )
    .unwrap();
    assert!(!limited.converged);
    assert_eq!(limited.termination_reason, "max_iter");
    assert_eq!(limited.n_iter, 1);
    assert_eq!(limited.loglik_trace.len(), 2);
    assert_eq!(limited.loglik, *limited.loglik_trace.last().unwrap());
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mh) = (mean(&a_true), mean(&fit.slope));
    let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
    for i in 0..n_items {
        num += (a_true[i] - ma) * (fit.slope[i] - mh);
        da += (a_true[i] - ma).powi(2);
        dh += (fit.slope[i] - mh).powi(2);
    }
    let corr = num / (da.sqrt() * dh.sqrt());
    assert!(corr > 0.9, "slope corr {corr}; hat={:?}", fit.slope);
}

#[test]
fn poly_item_information_matches_finite_difference() {
    // I(theta) = sum_k (dP_k/dtheta)^2 / P_k, checked against a central FD of the cell.
    let h = 1e-6;
    let cases: [(PolyModel, &[f64]); 2] = [
        (PolyModel::Gpcm, &[0.2, -0.3]),
        (PolyModel::Grm, &[1.1, -0.9]),
    ];
    for (model, cat) in cases.iter().copied() {
        let (a, theta) = (1.3_f64, 0.4_f64);
        let cell = |t: f64| -> Vec<f64> {
            let base = a * t;
            match model {
                PolyModel::Gpcm => {
                    let k = cat.len() + 1;
                    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                    let mut ic = vec![0.0; k];
                    ic[1..].copy_from_slice(cat);
                    gpcm_logprobs(base, &scores, &ic)
                        .iter()
                        .map(|l| l.exp())
                        .collect()
                }
                PolyModel::Grm => grm_logprobs(base, cat).iter().map(|l| l.exp()).collect(),
            }
        };
        let (pp, pm, p0) = (cell(theta + h), cell(theta - h), cell(theta));
        let mut fd_info = 0.0_f64;
        for k in 0..p0.len() {
            let dp = (pp[k] - pm[k]) / (2.0 * h);
            fd_info += dp * dp / p0[k];
        }
        let ana = poly_item_information(theta, a, cat, model);
        assert!(
            (ana - fd_info).abs() < 1e-4,
            "{model:?}: analytic {ana} vs fd {fd_info}"
        );
    }
}

#[test]
fn poly_information_curves_rejects_nonfinite_or_empty_inputs() {
    for (theta, slope, cat_params) in [
        (&[f64::NAN][..], &[1.0][..], &[0.0, 0.0][..]),
        (&[0.0][..], &[f64::INFINITY][..], &[0.0, 0.0][..]),
        (&[0.0][..], &[1.0][..], &[0.0, f64::NEG_INFINITY][..]),
    ] {
        assert!(poly_information_curves(theta, slope, cat_params, 1, 3, PolyModel::Gpcm,).is_err());
    }
    assert!(poly_information_curves(&[], &[1.0], &[0.0, 0.0], 1, 3, PolyModel::Gpcm).is_err());
    assert!(poly_information_curves(&[0.0], &[], &[], 0, 3, PolyModel::Gpcm).is_err());
}

#[test]
fn fit_poly_unidim_recovers_with_missing_data() {
    let (n_persons, n_items, k) = (5000usize, 6usize, 3usize);
    let mut st = 5150u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
    let c_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.0, 0.3 - 0.1 * i as f64, -0.2 + 0.1 * i as f64])
        .collect();
    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
    let mut y = vec![0usize; n_persons * n_items];
    let mut observed = vec![true; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            if u() < 0.25 {
                observed[p * n_items + i] = false; // ~25% MCAR missing
                continue;
            }
            let lp = gpcm_logprobs(a_true[i] * theta, &scores, &c_true[i]);
            let uu = u();
            let mut cum = 0.0_f64;
            let mut cat = k - 1;
            for (c, l) in lp.iter().enumerate() {
                cum += l.exp();
                if uu < cum {
                    cat = c;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    let fit = fit_poly_unidim(
        &y,
        Some(&observed),
        n_persons,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        80,
        1e-6,
    )
    .unwrap();
    assert!(fit.loglik.is_finite());
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mh) = (mean(&a_true), mean(&fit.slope));
    let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
    for i in 0..n_items {
        num += (a_true[i] - ma) * (fit.slope[i] - mh);
        da += (a_true[i] - ma).powi(2);
        dh += (fit.slope[i] - mh).powi(2);
    }
    assert!(
        num / (da.sqrt() * dh.sqrt()) > 0.9,
        "slope corr under missingness"
    );
    // absolute agreement, not just association
    let s_rmse = (a_true
        .iter()
        .zip(&fit.slope)
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        / n_items as f64)
        .sqrt();
    assert!(s_rmse < 0.2, "slope RMSE under missingness {s_rmse}");
}

#[test]
fn score_poly_eap_recovers_true_theta() {
    let (n_persons, n_items, k) = (3000usize, 8usize, 3usize);
    let mut st = 424242u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.1 * i as f64).collect();
    let c_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.0, 0.2 - 0.05 * i as f64, -0.3 + 0.08 * i as f64])
        .collect();
    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
    let mut theta_true = vec![0.0_f64; n_persons];
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        theta_true[p] = theta;
        for i in 0..n_items {
            let lp = gpcm_logprobs(a_true[i] * theta, &scores, &c_true[i]);
            let uu = u();
            let mut cum = 0.0_f64;
            let mut cat = k - 1;
            for (c, l) in lp.iter().enumerate() {
                cum += l.exp();
                if uu < cum {
                    cat = c;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    // score with the TRUE item params (isolates the scorer from fit error)
    let cat_flat: Vec<f64> = c_true.iter().flat_map(|c| c[1..].iter().copied()).collect();
    let (eap, sd) = score_poly_eap(
        &y,
        None,
        n_persons,
        n_items,
        k,
        &a_true,
        &cat_flat,
        PolyModel::Gpcm,
        41,
    )
    .unwrap();
    assert!(sd.iter().all(|s| s.is_finite() && *s > 0.0));
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mt, me) = (mean(&theta_true), mean(&eap));
    let (mut num, mut dt, mut de) = (0.0, 0.0, 0.0);
    for p in 0..n_persons {
        num += (theta_true[p] - mt) * (eap[p] - me);
        dt += (theta_true[p] - mt).powi(2);
        de += (eap[p] - me).powi(2);
    }
    let corr = num / (dt.sqrt() * de.sqrt());
    assert!(corr > 0.8, "theta EAP corr {corr}");
}

#[test]
fn score_poly_eap_rejects_invalid_inputs() {
    let y = vec![3usize];
    let slope = vec![1.0];
    let cat_params = vec![0.2, -0.3];
    let err =
        score_poly_eap(&y, None, 1, 1, 3, &slope, &cat_params, PolyModel::Gpcm, 21).unwrap_err();
    assert!(err.contains("categories"));

    let err = score_poly_eap(
        &[1],
        None,
        1,
        1,
        3,
        &[f64::NAN],
        &cat_params,
        PolyModel::Gpcm,
        21,
    )
    .unwrap_err();
    assert!(err.contains("finite"));
}

#[test]
fn fit_poly_unidim_recovers_grm() {
    let (n_persons, n_items, k) = (4000usize, 6usize, 4usize);
    let mut st = 99887766u64;
    let mut u = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
    // ordered-decreasing thresholds (valid GRM)
    let thr_true: Vec<Vec<f64>> = (0..n_items).map(|_| vec![1.4, 0.1, -1.2]).collect();
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let lp = grm_logprobs(a_true[i] * theta, &thr_true[i]);
            let uu = u();
            let mut cum = 0.0_f64;
            let mut cat = k - 1;
            for (c, l) in lp.iter().enumerate() {
                cum += l.exp();
                if uu < cum {
                    cat = c;
                    break;
                }
            }
            y[p * n_items + i] = cat;
        }
    }
    let fit = fit_poly_unidim(
        &y,
        None,
        n_persons,
        n_items,
        k,
        PolyModel::Grm,
        21,
        80,
        1e-6,
    )
    .unwrap();
    assert!(fit.loglik.is_finite());
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (ma, mh) = (mean(&a_true), mean(&fit.slope));
    let (mut num, mut da, mut dh) = (0.0, 0.0, 0.0);
    for i in 0..n_items {
        num += (a_true[i] - ma) * (fit.slope[i] - mh);
        da += (a_true[i] - ma).powi(2);
        dh += (fit.slope[i] - mh).powi(2);
    }
    let corr = num / (da.sqrt() * dh.sqrt());
    assert!(corr > 0.9, "grm slope corr {corr}; hat={:?}", fit.slope);
    // thresholds recovered near truth (pooled mean abs error, item 0)
    let mae: f64 = (0..3)
        .map(|j| (fit.cat_params[0][j] - thr_true[0][j]).abs())
        .sum::<f64>()
        / 3.0;
    assert!(
        mae < 0.25,
        "grm threshold MAE {mae}: {:?}",
        fit.cat_params[0]
    );
}

#[test]
fn gpcm_gradient_matches_finite_difference() {
    let scores = vec![0.0, 1.0, 2.0, 3.0];
    let intercepts = vec![0.0, 0.2, -0.1, 0.3];
    let counts = vec![3.0, 5.0, 2.0, 4.0];
    let base = 0.4;
    let q = |b: f64, ic: &[f64], sc: &[f64]| -> f64 {
        gpcm_logprobs(b, sc, ic)
            .iter()
            .zip(&counts)
            .map(|(l, r)| r * l)
            .sum()
    };
    let (g_ic, g_base, g_sc) = gpcm_node_gradient(base, &scores, &intercepts, &counts);
    let h = 1e-6;
    assert!(
        ((q(base + h, &intercepts, &scores) - q(base - h, &intercepts, &scores)) / (2.0 * h)
            - g_base)
            .abs()
            < 1e-5
    );
    for m in 1..scores.len() {
        let mut ip = intercepts.clone();
        let mut im = intercepts.clone();
        ip[m] += h;
        im[m] -= h;
        let fd = (q(base, &ip, &scores) - q(base, &im, &scores)) / (2.0 * h);
        assert!((fd - g_ic[m - 1]).abs() < 1e-5);
        let mut sp = scores.clone();
        let mut sm = scores.clone();
        sp[m] += h;
        sm[m] -= h;
        let fds = (q(base, &intercepts, &sp) - q(base, &intercepts, &sm)) / (2.0 * h);
        assert!((fds - g_sc[m - 1]).abs() < 1e-5);
    }
}

// deterministic uniform draws for the item-fit tests
fn rng(seed: u64) -> impl FnMut() -> f64 {
    let mut st = seed;
    move || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

#[test]
fn poly_s_x2_reduces_to_binary_orlando_thissen() {
    // At K=2 the generalized S-X² must equal the trusted binary Orlando-
    // Thissen s_x2 (crate::fitstats) EXACTLY on the same quadrature grid:
    // both GRM and GPCM cells reduce to the 2PL P(Y=1)=sigmoid(a*theta+b),
    // and the summed-score recursion / expected proportions coincide. Large
    // N + few centered items keep either statistic out of its collapsing
    // regime, so the agreement is bit-for-bit (min_expected tiny on both).
    use crate::fitstats::{s_x2, SX2Config};
    use crate::nodes::XiRule;
    use crate::scoring::{ItemBank, PriorSpec};
    use crate::ModelType;
    let (n_persons, n_items, q_theta) = (4000usize, 6usize, 41usize);
    let mut u = rng(13579);
    let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.1 * i as f64).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -0.6 + 0.24 * i as f64).collect();
    let mut yi = vec![0usize; n_persons * n_items];
    for _p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let pr = 1.0 / (1.0 + (-(a_true[i] * theta + b_true[i])).exp());
            yi[_p * n_items + i] = if u() < pr { 1 } else { 0 };
        }
    }
    let yf: Vec<f64> = yi.iter().map(|&v| v as f64).collect();
    let observed_bool = vec![true; n_persons * n_items];
    let alpha: Vec<f64> = a_true.iter().map(|a| a.ln()).collect();
    let zeta = vec![0.0_f64; n_items];
    let fid = vec![0usize; n_items];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b_true,
        zeta: &zeta,
        tau: -50.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let bin = s_x2(
        &bank,
        &yf,
        &observed_bool,
        n_persons,
        &PriorSpec::standard(1),
        &SX2Config {
            q_theta,
            xi_rule: XiRule::GaussHermite { q_xi: 1 },
            min_expected: 1e-9,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    for model in [PolyModel::Grm, PolyModel::Gpcm] {
        let poly = poly_s_x2(
            &yi, None, n_persons, n_items, 2, &a_true, &b_true, model, q_theta, 1e-9,
        )
        .unwrap();
        for i in 0..n_items {
            assert!(
                (poly.statistic[i] - bin.statistic[i]).abs() < 1e-8,
                "{model:?} item {i}: poly {} vs binary {}",
                poly.statistic[i],
                bin.statistic[i]
            );
            assert_eq!(
                poly.df[i], bin.df[i],
                "{model:?} item {i} df: poly {:?} vs binary {:?}",
                poly.df[i], bin.df[i]
            );
        }
    }
}

#[test]
fn poly_s_x2_is_calibrated_at_true_parameters() {
    // Kang & Chen (2008/2011) headline: under the true model the generalized
    // S-X² tracks its reference chi-square. Evaluated at the KNOWN generating
    // parameters the reference df is the retained cell count (no −m estimation
    // adjustment), so E[S-X²] ≈ Σ cells. We reproduce this — an ABSOLUTE
    // agreement of the sampling mean with its theoretical value, the analogue
    // of an RMSE recovery check for a fit statistic — for both GPCM (2008) and
    // GRM (2011), which is exactly what a mis-calibrated index (e.g. Yen's
    // Q1 / PARSCALE G², inflated to many times its df) would fail.
    let (n_persons, n_items, n_cat, reps) = (1500usize, 8usize, 4usize, 24usize);
    for model in [PolyModel::Gpcm, PolyModel::Grm] {
        let a_true: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.08 * i as f64).collect();
        let cat_true: Vec<f64> = (0..n_items)
            .flat_map(|i| match model {
                // GPCM additive intercepts (any reals)
                PolyModel::Gpcm => vec![0.8 - 0.06 * i as f64, 0.0, -0.8 + 0.06 * i as f64],
                // GRM thresholds must be strictly decreasing for a valid cdf
                PolyModel::Grm => vec![1.1 + 0.04 * i as f64, 0.0, -1.1 - 0.04 * i as f64],
            })
            .collect();
        let z = n_cat - 1;
        let (mut stat_sum, mut cell_sum) = (0.0_f64, 0.0_f64);
        let mut n_flagged = 0usize;
        let mut n_tested = 0usize;
        for r in 0..reps {
            let mut u = rng(2024_0714 + r as u64 * 97);
            let mut yi = vec![0usize; n_persons * n_items];
            for p in 0..n_persons {
                let u1 = u().max(1e-12);
                let u2 = u();
                let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                for i in 0..n_items {
                    let base = a_true[i] * theta;
                    let cp = &cat_true[i * z..(i + 1) * z];
                    let lp = match model {
                        PolyModel::Gpcm => {
                            let scores: Vec<f64> = (0..n_cat).map(|c| c as f64).collect();
                            let mut ic = vec![0.0_f64; n_cat];
                            ic[1..].copy_from_slice(cp);
                            gpcm_logprobs(base, &scores, &ic)
                        }
                        PolyModel::Grm => grm_logprobs(base, cp),
                    };
                    let draw = u();
                    let mut acc = 0.0_f64;
                    let mut cat = n_cat - 1;
                    for (c, l) in lp.iter().enumerate() {
                        acc += l.exp();
                        if draw <= acc {
                            cat = c;
                            break;
                        }
                    }
                    yi[p * n_items + i] = cat;
                }
            }
            let res = poly_s_x2(
                &yi, None, n_persons, n_items, n_cat, &a_true, &cat_true, model, 21, 1.0,
            )
            .unwrap();
            for i in 0..n_items {
                if res.n_cells[i] >= 1 && res.statistic[i].is_finite() {
                    stat_sum += res.statistic[i];
                    cell_sum += res.n_cells[i] as f64;
                    n_tested += 1;
                    if res.p_value[i].is_finite() && res.p_value[i] < 0.05 {
                        n_flagged += 1;
                    }
                }
            }
        }
        let ratio = stat_sum / cell_sum;
        assert!(
            (0.85..=1.15).contains(&ratio),
            "{model:?}: mean S-X² / cells = {ratio} (stat {stat_sum}, cells {cell_sum})"
        );
        // df uses the −m adjustment, so p-values at true params are mildly
        // conservative; the flag rate stays far below the >30% seen for G².
        let flag_rate = n_flagged as f64 / n_tested as f64;
        assert!(
            flag_rate < 0.15,
            "{model:?}: flag rate {flag_rate} too high for the true model"
        );
    }
}

/// One ability condition's aggregate recovery: absolute-agreement RMSE and
/// mean |bias| for the slope and the category intercepts.
struct McRecovery {
    cond: &'static str,
    a_rmse: f64,
    a_bias: f64,
    c_rmse: f64,
    c_bias: f64,
}

/// Monte-Carlo parameter-recovery study for the GPCM fitter, generating from
/// the published item-parameter scheme of Kang & Chen (2008, p. 397): slopes
/// `a_i ~ lognormal(0, 0.5²)` and four step difficulties `b_{i,c} ~
/// N(means −1.5, −0.5, 0.5, 1.5; SD 0.5)`. Two ability conditions are run —
/// NORMAL `θ ~ N(0, 1)` (the fitter's prior, so recovery is near-unbiased)
/// and right-SKEWED `θ = Exp(1) − 1` (mean 0, var 1, skewness 2), a prior
/// misspecification Kang & Chen flag as future work. Returns per-condition
/// RMSE and mean |bias| (absolute agreement, not correlation) over `reps`
/// replications on a fixed true item bank.
///
/// # References (APA 7th ed.)
///
/// Kang, T., & Chen, T. T. (2008). Performance of the generalized S-X² item
///   fit index for polytomous IRT models. *Journal of Educational
///   Measurement, 45*(4), 391–406.
///   https://doi.org/10.1111/j.1745-3984.2008.00070.x
/// Muraki, E. (1992). A generalized partial credit model: Application of an
///   EM algorithm. *Applied Psychological Measurement, 16*(2), 159–176.
///   https://doi.org/10.1177/014662169201600206
fn mc_gpcm_recovery(reps: usize, n_persons: usize) -> Vec<McRecovery> {
    let (n_items, k) = (5usize, 5usize);
    let z_steps = k - 1; // 4 step difficulties
    let step_means = [-1.5_f64, -0.5, 0.5, 1.5];

    // fixed "true" item bank (drawn once) from the published scheme
    let mut bu = rng(96100);
    let mut bnorm = || {
        let u1 = bu().max(1e-12);
        let u2 = bu();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut a_true = vec![0.0_f64; n_items];
    let mut cat_true = vec![0.0_f64; n_items * z_steps]; // additive intercepts
    for i in 0..n_items {
        a_true[i] = (0.5 * bnorm()).exp(); // lognormal(0, 0.5²)
        let mut cum = 0.0_f64;
        for c in 0..z_steps {
            let b = step_means[c] + 0.5 * bnorm(); // step difficulty
            cum += b;
            cat_true[i * z_steps + c] = -a_true[i] * cum; // GPCM intercept
        }
    }

    let mut out = Vec::new();
    for (cond, skew) in [("normal", false), ("skew", true)] {
        // accumulate signed error and squared error per parameter over reps
        let mut a_err = vec![0.0_f64; n_items];
        let mut a_sq = vec![0.0_f64; n_items];
        let mut c_err = vec![0.0_f64; n_items * z_steps];
        let mut c_sq = vec![0.0_f64; n_items * z_steps];
        for rep in 0..reps {
            let mut u = rng(4242 + rep as u64 * 131 + if skew { 7 } else { 0 });
            let mut yi = vec![0usize; n_persons * n_items];
            for p in 0..n_persons {
                let theta = if skew {
                    -(u().max(1e-12)).ln() - 1.0 // Exp(1) − 1: mean 0, var 1, skew 2
                } else {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                };
                for i in 0..n_items {
                    let base = a_true[i] * theta;
                    let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                    let mut ic = vec![0.0_f64; k];
                    ic[1..].copy_from_slice(&cat_true[i * z_steps..(i + 1) * z_steps]);
                    let lp = gpcm_logprobs(base, &scores, &ic);
                    let draw = u();
                    let (mut acc, mut cat) = (0.0_f64, k - 1);
                    for (c, l) in lp.iter().enumerate() {
                        acc += l.exp();
                        if draw <= acc {
                            cat = c;
                            break;
                        }
                    }
                    yi[p * n_items + i] = cat;
                }
            }
            let fit = fit_poly_unidim(
                &yi,
                None,
                n_persons,
                n_items,
                k,
                PolyModel::Gpcm,
                21,
                100,
                1e-6,
            )
            .unwrap();
            assert!(
                fit.converged,
                "GPCM recovery replicate {rep} ({cond}) did not converge: \
                 reason={}, n_iter={}/{}, delta={:.6e}, tolerance={:.6e}",
                fit.termination_reason, fit.n_iter, 100, fit.final_delta, fit.stopping_tolerance
            );
            for i in 0..n_items {
                let ea = fit.slope[i] - a_true[i];
                a_err[i] += ea;
                a_sq[i] += ea * ea;
                for c in 0..z_steps {
                    let ec = fit.cat_params[i][c] - cat_true[i * z_steps + c];
                    c_err[i * z_steps + c] += ec;
                    c_sq[i * z_steps + c] += ec * ec;
                }
            }
        }
        let r = reps as f64;
        let rmse = |sq: &[f64]| (sq.iter().sum::<f64>() / (sq.len() as f64 * r)).sqrt();
        let mean_bias =
            |er: &[f64]| er.iter().map(|e| (e / r).abs()).sum::<f64>() / er.len() as f64;
        out.push(McRecovery {
            cond,
            a_rmse: rmse(&a_sq),
            a_bias: mean_bias(&a_err),
            c_rmse: rmse(&c_sq),
            c_bias: mean_bias(&c_err),
        });
    }
    out
}

fn assert_recovery(out: &[McRecovery], reps: usize, n_persons: usize) {
    for s in out {
        println!(
            "[MC recovery, θ={}] reps={reps} N={n_persons}  \
             slope: RMSE={:.4} |bias|={:.4}  intercept: RMSE={:.4} |bias|={:.4}",
            s.cond, s.a_rmse, s.a_bias, s.c_rmse, s.c_bias
        );
        assert!(s.a_rmse.is_finite() && s.c_rmse.is_finite());
        if s.cond == "skew" {
            // prior misspecification: recovery holds but degrades (reported)
            assert!(s.a_rmse < 0.45, "skew slope RMSE too large: {}", s.a_rmse);
            assert!(
                s.c_rmse < 1.2,
                "skew intercept RMSE too large: {}",
                s.c_rmse
            );
        } else {
            // matched prior: tight, near-unbiased recovery
            assert!(s.a_rmse < 0.20, "normal slope RMSE too large: {}", s.a_rmse);
            assert!(
                s.c_rmse < 0.45,
                "normal intercept RMSE too large: {}",
                s.c_rmse
            );
            assert!(s.a_bias < 0.10, "normal slope bias too large: {}", s.a_bias);
        }
    }
}

#[test]
fn fit_poly_unidim_recovery_ci_guard() {
    // Fast regression guard (few reps). The authoritative >=500-replication
    // study is `fit_poly_unidim_recovery_monte_carlo_500` (ignored below);
    // run it with: cargo test --release -- --ignored --nocapture
    let (reps, n_persons) = (20usize, 1500usize);
    assert_recovery(&mc_gpcm_recovery(reps, n_persons), reps, n_persons);
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn fit_poly_unidim_recovery_monte_carlo_500() {
    // 500-replication recovery study (the sample size common in the IRT
    // Monte-Carlo literature), N = 2000 per replication.
    let (reps, n_persons) = (500usize, 2000usize);
    assert_recovery(&mc_gpcm_recovery(reps, n_persons), reps, n_persons);
}

#[test]
fn fit_nominal_nests_gpcm() {
    // The nominal model contains the GPCM (scores linear in k, a_k = a*k), so
    // fitting nominal to GPCM data must (a) reach a log-likelihood at least as
    // high as the GPCM fit and (b) recover linear scores: a_2/a_1 ≈ 2.
    let (n_persons, n_items, k) = (3000usize, 5usize, 3usize);
    let mut u = rng(778899);
    let a_gpcm: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.15 * i as f64).collect();
    let c_gpcm: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.3 - 0.1 * i as f64, -0.4 + 0.1 * i as f64])
        .collect();
    let mut yi = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let base = a_gpcm[i] * theta;
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut ic = vec![0.0_f64; k];
            ic[1..].copy_from_slice(&c_gpcm[i]);
            let lp = gpcm_logprobs(base, &scores, &ic);
            let draw = u();
            let (mut acc, mut cat) = (0.0_f64, k - 1);
            for (c, l) in lp.iter().enumerate() {
                acc += l.exp();
                if draw <= acc {
                    cat = c;
                    break;
                }
            }
            yi[p * n_items + i] = cat;
        }
    }
    let gpcm = fit_poly_unidim(
        &yi,
        None,
        n_persons,
        n_items,
        k,
        PolyModel::Gpcm,
        41,
        300,
        1e-7,
    )
    .unwrap();
    let nom = fit_nominal(&yi, None, n_persons, n_items, k, 41, 300, 1e-7).unwrap();
    assert!(
        nom.loglik >= gpcm.loglik - 0.5,
        "nominal loglik {} should be >= GPCM {}",
        nom.loglik,
        gpcm.loglik
    );
    for i in 0..n_items {
        let (a1, a2) = (nom.scores[i][0], nom.scores[i][1]);
        assert!(
            (a2 / a1 - 2.0).abs() < 0.4,
            "item {i}: recovered scores not linear (a2/a1={})",
            a2 / a1
        );
    }
}

#[test]
fn fit_nominal_reports_convergence_and_rejects_invalid_controls() {
    let (n_persons, n_items, n_cat) = (60usize, 3usize, 3usize);
    let y: Vec<usize> = (0..n_persons)
        .flat_map(|p| (0..n_items).map(move |i| (p + i) % n_cat))
        .collect();
    let fit = fit_nominal(&y, None, n_persons, n_items, n_cat, 21, 1, 1e-12).unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.termination_reason, "max_iter");
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.loglik_trace.len(), fit.n_iter + 1);
    assert_eq!(fit.loglik, *fit.loglik_trace.last().unwrap());
    assert!(fit.final_delta.is_finite());
    assert!(fit.final_delta > fit.stopping_tolerance);
    assert!(fit
        .loglik_trace
        .windows(2)
        .all(|pair| pair[1] >= pair[0] - 1e-10));

    assert!(fit_nominal(&[], None, 0, n_items, n_cat, 21, 10, 1e-6).is_err());
    assert!(fit_nominal(&y, None, n_persons, n_items, n_cat, 21, 0, 1e-6).is_err());
    assert!(fit_nominal(&y, None, n_persons, n_items, n_cat, 21, 10, f64::INFINITY).is_err());
    let observed: Vec<bool> = (0..n_persons)
        .flat_map(|_| (0..n_items).map(|i| i != 1))
        .collect();
    assert!(fit_nominal(&y, Some(&observed), n_persons, n_items, n_cat, 21, 10, 1e-6).is_err());
}

/// Aggregate nominal-model recovery (RMSE and mean |bias|) for the free
/// scores and intercepts over `reps` datasets at fixed true parameters, with
/// per-item sign alignment (the model is identified up to (a_k,θ)→(−a_k,−θ)).
fn mc_nominal_recovery(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64, f64) {
    let (n_items, k) = (6usize, 4usize);
    let z = k - 1;
    let a_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| {
            vec![
                0.9 + 0.04 * i as f64,
                2.0 - 0.03 * i as f64,
                2.7 + 0.05 * i as f64,
            ]
        })
        .collect();
    let c_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.5 - 0.05 * i as f64, 0.0, -0.6 + 0.05 * i as f64])
        .collect();
    let (mut a_err, mut a_sq, mut c_err, mut c_sq) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let mut cnt = 0.0_f64;
    for rep in 0..reps {
        let mut u = rng(31337 + rep as u64 * 131 + if skew { 9 } else { 0 });
        let mut yi = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let theta = if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            for i in 0..n_items {
                let mut scores = vec![0.0_f64; k];
                let mut intercepts = vec![0.0_f64; k];
                for m in 0..z {
                    scores[m + 1] = a_true[i][m];
                    intercepts[m + 1] = c_true[i][m];
                }
                let lp = gpcm_logprobs(theta, &scores, &intercepts);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[p * n_items + i] = cat;
            }
        }
        let fit = fit_nominal(&yi, None, n_persons, n_items, k, 21, 200, 1e-6).unwrap();
        assert!(
            fit.converged,
            "nominal recovery replicate {rep} did not converge: reason={} n_iter={} \
             final_delta={:.6e} tolerance={:.6e}",
            fit.termination_reason, fit.n_iter, fit.final_delta, fit.stopping_tolerance
        );
        for i in 0..n_items {
            // align the reflection sign to the truth for this item
            let dot: f64 = (0..z).map(|m| fit.scores[i][m] * a_true[i][m]).sum();
            let s = if dot >= 0.0 { 1.0 } else { -1.0 };
            for m in 0..z {
                let ea = s * fit.scores[i][m] - a_true[i][m];
                a_err += ea;
                a_sq += ea * ea;
                let ec = fit.intercepts[i][m] - c_true[i][m];
                c_err += ec;
                c_sq += ec * ec;
                cnt += 1.0;
            }
        }
    }
    (
        (a_sq / cnt).sqrt(),
        (a_err / cnt).abs(),
        (c_sq / cnt).sqrt(),
        (c_err / cnt).abs(),
    )
}

#[test]
fn fit_nominal_recovery_ci_guard() {
    // Fast guard. Authoritative >=500-rep study is
    // fit_nominal_recovery_monte_carlo_500 (ignored).
    let (reps, n) = (12usize, 2000usize);
    let (ar, ab, cr, cb) = mc_nominal_recovery(reps, n, false);
    let (asr, _, csr, _) = mc_nominal_recovery(reps, n, true);
    println!(
        "[nominal recovery] reps={reps} N={n}  normal: score RMSE={ar:.4} |bias|={ab:.4} \
         intercept RMSE={cr:.4} |bias|={cb:.4}  skew: score RMSE={asr:.4} intercept RMSE={csr:.4}"
    );
    assert!(
        ar < 0.25 && cr < 0.30,
        "normal recovery too loose: a={ar}, c={cr}"
    );
    assert!(ab < 0.12, "normal score bias too large: {ab}");
    assert!(
        asr > ar,
        "skew should degrade score recovery: {asr} vs {ar}"
    );
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn fit_nominal_recovery_monte_carlo_500() {
    let (reps, n) = (500usize, 2000usize);
    let (ar, ab, cr, cb) = mc_nominal_recovery(reps, n, false);
    let (asr, asb, csr, _) = mc_nominal_recovery(reps, n, true);
    println!(
        "[nominal recovery 500] N={n}  normal: score RMSE={ar:.4} |bias|={ab:.4} \
         intercept RMSE={cr:.4} |bias|={cb:.4}  skew: score RMSE={asr:.4} |bias|={asb:.4} \
         intercept RMSE={csr:.4}"
    );
    assert!(
        ar < 0.15 && cr < 0.20,
        "normal recovery too loose: a={ar}, c={cr}"
    );
    assert!(ab < 0.05, "normal score bias not near zero: {ab}");
    assert!(
        asr > ar + 0.03,
        "skew should measurably degrade recovery: {asr} vs {ar}"
    );
}

#[test]
fn poly_person_fit_matches_binary_lz_at_k2() {
    // At K=2 the polytomous l_z must equal the trusted binary person_fit l_z
    // on the same EAP trait (both cells reduce to the 2PL); l_z* matches to
    // finite-difference tolerance (poly uses a numerical trait derivative).
    use crate::fitstats::person_fit;
    use crate::scoring::ItemBank;
    use crate::ModelType;
    let (n_persons, n_items) = (1000usize, 12usize);
    let mut u = rng(56789);
    let a: Vec<f64> = (0..n_items).map(|i| 0.9 + 0.06 * i as f64).collect();
    let b: Vec<f64> = (0..n_items).map(|i| -0.8 + 0.14 * i as f64).collect();
    let mut yf = vec![0.0_f64; n_persons * n_items];
    let mut yi = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let u1 = u().max(1e-12);
        let u2 = u();
        let th = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let pr = 1.0 / (1.0 + (-(a[i] * th + b[i])).exp());
            let v = if u() < pr { 1.0 } else { 0.0 };
            yf[p * n_items + i] = v;
            yi[p * n_items + i] = v as usize;
        }
    }
    let obs = vec![true; n_persons * n_items];
    let poly = poly_person_fit(
        &yi,
        None,
        n_persons,
        n_items,
        2,
        &a,
        &b,
        PolyModel::Gpcm,
        41,
        0.0,
        1.0,
        -1.645,
    )
    .unwrap();
    let alpha: Vec<f64> = a.iter().map(|x| x.ln()).collect();
    let zeta = vec![0.0_f64; n_items];
    let fid = vec![0usize; n_items];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -50.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let xi = vec![0.0_f64; n_persons];
    let bin = person_fit(
        &bank,
        &yf,
        &obs,
        n_persons,
        &poly.theta_eap,
        &xi,
        &[],
        -1.645,
    )
    .unwrap();
    let (mut d_lz, mut d_lzs) = (0.0_f64, 0.0_f64);
    for p in 0..n_persons {
        if poly.lz[p].is_finite() && bin.lz[p].is_finite() {
            d_lz = d_lz.max((poly.lz[p] - bin.lz[p]).abs());
            d_lzs = d_lzs.max((poly.lz_star[p] - bin.lz_star[p]).abs());
        }
    }
    assert!(d_lz < 1e-6, "l_z max diff vs binary: {d_lz}");
    assert!(d_lzs < 5e-3, "l_z* max diff vs binary: {d_lzs}");
}

// GPCM person-fit Monte-Carlo: a fraction of respondents answer carelessly
// (uniform random categories) and the rest come from the model; evaluated at
// the true item parameters. Returns (Type I flag rate among model
// respondents, power among careless respondents, mean l_z*, sd l_z*).
fn mc_poly_person_fit(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64, f64) {
    let (n_items, k) = (20usize, 3usize);
    let z = k - 1;
    let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.03 * i as f64).collect();
    let cat_true: Vec<f64> = (0..n_items)
        .flat_map(|i| vec![0.6 - 0.01 * i as f64, -0.6 + 0.01 * i as f64])
        .collect();
    let n_care = n_persons / 10; // first 10% are careless
    let (mut n_norm, mut flag_norm, mut flag_care) = (0usize, 0usize, 0usize);
    let (mut sum, mut sum2) = (0.0_f64, 0.0_f64);
    for rep in 0..reps {
        let mut u = rng(7000 + rep as u64 * 131 + if skew { 3 } else { 0 });
        let mut yi = vec![0usize; n_persons * n_items];
        for p in 0..n_persons {
            let careless = p < n_care;
            let theta = if skew {
                -(u().max(1e-12)).ln() - 1.0
            } else {
                let u1 = u().max(1e-12);
                let u2 = u();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
            };
            for i in 0..n_items {
                // careless / inconsistent responder: the implied trait alternates
                // +-1.6 across items, so no single theta fits the pattern.
                let theta_use = if careless {
                    if i % 2 == 0 {
                        1.6
                    } else {
                        -1.6
                    }
                } else {
                    theta
                };
                let base = a_true[i] * theta_use;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&cat_true[i * z..(i + 1) * z]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut cat) = (0.0_f64, k - 1);
                for (c, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        cat = c;
                        break;
                    }
                }
                yi[p * n_items + i] = cat;
            }
        }
        let pf = poly_person_fit(
            &yi,
            None,
            n_persons,
            n_items,
            k,
            &a_true,
            &cat_true,
            PolyModel::Gpcm,
            21,
            0.0,
            1.0,
            -1.645,
        )
        .unwrap();
        for p in 0..n_persons {
            if p < n_care {
                if pf.flagged[p] {
                    flag_care += 1;
                }
            } else {
                n_norm += 1;
                if pf.flagged[p] {
                    flag_norm += 1;
                }
                if pf.lz_star[p].is_finite() {
                    sum += pf.lz_star[p];
                    sum2 += pf.lz_star[p] * pf.lz_star[p];
                }
            }
        }
    }
    let mean = sum / n_norm as f64;
    let sd = (sum2 / n_norm as f64 - mean * mean).max(0.0).sqrt();
    (
        flag_norm as f64 / n_norm as f64,
        flag_care as f64 / (n_care * reps) as f64,
        mean,
        sd,
    )
}

#[test]
fn poly_person_fit_type1_and_power() {
    // Fast guard. Authoritative >=500-rep study is
    // poly_person_fit_monte_carlo_500 (ignored).
    let (reps, n) = (8usize, 800usize);
    let (t1, power, mean, sd) = mc_poly_person_fit(reps, n, false);
    let (t1s, _, _, _) = mc_poly_person_fit(reps, n, true);
    println!(
        "[poly person-fit] normal: Type I(l_z*<-1.645)={t1:.3} power(careless)={power:.3} \
         mean(l_z*)={mean:.3} sd(l_z*)={sd:.3}  skew: Type I={t1s:.3}"
    );
    assert!((0.01..=0.12).contains(&t1), "Type I off nominal: {t1}");
    assert!(
        power > 0.5,
        "power to flag careless responders too low: {power}"
    );
    assert!(
        mean.abs() < 0.4 && (0.75..=1.3).contains(&sd),
        "l_z* not ~N(0,1): mean={mean}, sd={sd}"
    );
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn poly_person_fit_monte_carlo_500() {
    let (reps, n) = (500usize, 600usize);
    let (t1, power, mean, sd) = mc_poly_person_fit(reps, n, false);
    println!(
        "[poly person-fit 500] normal: Type I={t1:.4} power={power:.4} mean(l_z*)={mean:.4} \
         sd(l_z*)={sd:.4}"
    );
    // l_z* runs slightly high at a 20-item test (a documented finite-length
    // effect); it converges to nominal as the test lengthens.
    assert!((0.02..=0.11).contains(&t1), "Type I off nominal: {t1}");
    assert!(power > 0.7, "power too low: {power}");
    assert!(
        mean.abs() < 0.25 && (0.85..=1.2).contains(&sd),
        "l_z* not ~N(0,1): mean={mean}, sd={sd}"
    );
}

/// A GPCM item bank for the CAT tests: `n_items` items with difficulties
/// spread across the trait range so the adaptive selector has informative
/// items at every ability level.
fn cat_bank(n_items: usize, k: usize) -> (Vec<f64>, Vec<f64>) {
    let z = k - 1;
    let mut slope = vec![0.0_f64; n_items];
    let mut cat = vec![0.0_f64; n_items * z];
    for i in 0..n_items {
        let a = 1.0 + 0.25 * (i % 3) as f64; // 1.0 / 1.25 / 1.5, cycling
        slope[i] = a;
        let b = -2.2 + 4.4 * i as f64 / (n_items - 1) as f64; // spread difficulty
        let mut cum = 0.0_f64;
        for m in 0..z {
            let step = b + (m as f64 - (z as f64 - 1.0) / 2.0) * 0.9;
            cum += step;
            cat[i * z + m] = -a * cum;
        }
    }
    (slope, cat)
}

fn cat_rmse(eap: &[f64], true_theta: &[f64]) -> f64 {
    let n = true_theta.len() as f64;
    (eap.iter()
        .zip(true_theta)
        .map(|(e, t)| (e - t).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
}

#[test]
fn poly_cat_recovers_and_beats_random() {
    // Fast guard. Authoritative >=500-simulee study is
    // poly_cat_monte_carlo_500 (ignored).
    let (n_items, k) = (40usize, 4usize);
    let (slope, cat) = cat_bank(n_items, k);
    let n_sim = 300usize;
    let mut u = rng(9001);
    let true_theta: Vec<f64> = (0..n_sim)
        .map(|_| {
            let u1 = u().max(1e-12);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        })
        .collect();
    // adaptive, variable length: stop at SE < 0.30
    let var = poly_cat_simulate(
        &true_theta,
        &slope,
        &cat,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        0.30,
        5,
        30,
        true,
        111,
    )
    .unwrap();
    let rmse_var = cat_rmse(&var.theta_eap, &true_theta);
    let mean_items = var.n_used.iter().sum::<usize>() as f64 / n_sim as f64;
    println!("[poly CAT] var-len(SE<.30): RMSE={rmse_var:.3} mean_items={mean_items:.1}/{n_items}");
    assert!(rmse_var < 0.40, "CAT theta RMSE too high: {rmse_var}");
    assert!(
        mean_items < 0.75 * n_items as f64,
        "CAT should use fewer than the bank: {mean_items}"
    );
    // fixed length L=12: maximum-information beats random selection
    let adap = poly_cat_simulate(
        &true_theta,
        &slope,
        &cat,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        0.0,
        12,
        12,
        true,
        222,
    )
    .unwrap();
    let rand = poly_cat_simulate(
        &true_theta,
        &slope,
        &cat,
        n_items,
        k,
        PolyModel::Gpcm,
        21,
        0.0,
        12,
        12,
        false,
        333,
    )
    .unwrap();
    let (ra, rr) = (
        cat_rmse(&adap.theta_eap, &true_theta),
        cat_rmse(&rand.theta_eap, &true_theta),
    );
    println!("[poly CAT] fixed L=12: adaptive RMSE={ra:.3} random RMSE={rr:.3}");
    assert!(
        ra < rr,
        "max-information CAT should beat random selection: {ra} vs {rr}"
    );
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 simulees); run with: cargo test --release -- --ignored --nocapture"]
fn poly_cat_monte_carlo_500() {
    let (n_items, k) = (40usize, 4usize);
    let (slope, cat) = cat_bank(n_items, k);
    let n_sim = 500usize;
    for (label, skew) in [("normal", false), ("skew", true)] {
        let mut u = rng(if skew { 7001 } else { 7000 });
        let true_theta: Vec<f64> = (0..n_sim)
            .map(|_| {
                if skew {
                    -(u().max(1e-12)).ln() - 1.0
                } else {
                    let u1 = u().max(1e-12);
                    let u2 = u();
                    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
                }
            })
            .collect();
        let var = poly_cat_simulate(
            &true_theta,
            &slope,
            &cat,
            n_items,
            k,
            PolyModel::Gpcm,
            21,
            0.30,
            5,
            30,
            true,
            4242,
        )
        .unwrap();
        let rmse = cat_rmse(&var.theta_eap, &true_theta);
        let mean_items = var.n_used.iter().sum::<usize>() as f64 / n_sim as f64;
        let adap = poly_cat_simulate(
            &true_theta,
            &slope,
            &cat,
            n_items,
            k,
            PolyModel::Gpcm,
            21,
            0.0,
            12,
            12,
            true,
            5,
        )
        .unwrap();
        let rand = poly_cat_simulate(
            &true_theta,
            &slope,
            &cat,
            n_items,
            k,
            PolyModel::Gpcm,
            21,
            0.0,
            12,
            12,
            false,
            6,
        )
        .unwrap();
        let (ra, rr) = (
            cat_rmse(&adap.theta_eap, &true_theta),
            cat_rmse(&rand.theta_eap, &true_theta),
        );
        println!(
            "[poly CAT 500 θ={label}] var-len RMSE={rmse:.4} mean_items={mean_items:.2}/{n_items}  \
             fixed L=12: adaptive RMSE={ra:.4} random RMSE={rr:.4}"
        );
        assert!(rmse < 0.42, "{label} CAT RMSE too high: {rmse}");
        assert!(
            mean_items < 0.7 * n_items as f64,
            "{label} CAT not saving items: {mean_items}"
        );
        assert!(ra < rr, "{label} adaptive should beat random: {ra} vs {rr}");
    }
}

// Two-group GPCM dataset generator for the DIF tests. group 0 = reference
// theta~N(0,1); group 1 = focal theta~N(0.5, 1.2^2) (impact). `dif` on item 0
// for the focal group: 0=none, 1=uniform (difficulty shift), 2=non-uniform
// (slope 1.6x). `skew` draws the focal trait from Exp(1)-1 instead.
fn gen_two_group_gpcm(
    n_per_group: usize,
    n_items: usize,
    k: usize,
    dif: u8,
    skew: bool,
    seed: u64,
) -> (Vec<usize>, Vec<usize>) {
    let a_true: Vec<f64> = (0..n_items).map(|i| 1.0 + 0.05 * i as f64).collect();
    let int_true: Vec<Vec<f64>> = (0..n_items)
        .map(|i| vec![0.7 - 0.05 * i as f64, -0.7 + 0.05 * i as f64])
        .collect();
    let n_persons = 2 * n_per_group;
    let mut u = rng(seed);
    let mut yi = vec![0usize; n_persons * n_items];
    let mut gid = vec![0usize; n_persons];
    for p in 0..n_persons {
        let focal = p >= n_per_group;
        gid[p] = focal as usize;
        let theta = if !focal {
            let u1 = u().max(1e-12);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        } else if skew {
            -(u().max(1e-12)).ln() - 1.0
        } else {
            let u1 = u().max(1e-12);
            let u2 = u();
            0.5 + 1.2 * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        for i in 0..n_items {
            let (a, ints) = if i == 0 && focal && dif == 1 {
                let d = 0.6; // uniform: shift difficulty => intercept_k += k*a*d
                (
                    a_true[0],
                    vec![
                        int_true[0][0] + a_true[0] * d,
                        int_true[0][1] + 2.0 * a_true[0] * d,
                    ],
                )
            } else if i == 0 && focal && dif == 2 {
                (a_true[0] * 1.6, int_true[0].clone())
            } else {
                (a_true[i], int_true[i].clone())
            };
            let base = a * theta;
            let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
            let mut ic = vec![0.0_f64; k];
            ic[1..].copy_from_slice(&ints);
            let lp = gpcm_logprobs(base, &scores, &ic);
            let draw = u();
            let (mut acc, mut cat) = (0.0_f64, k - 1);
            for (c, l) in lp.iter().enumerate() {
                acc += l.exp();
                if draw <= acc {
                    cat = c;
                    break;
                }
            }
            yi[p * n_items + i] = cat;
        }
    }
    (yi, gid)
}

#[test]
fn poly_dif_structural_recovers_impact_and_nesting() {
    // No DIF, but the focal group has impact N(0.5, 1.2^2): the estimator
    // must recover the focal distribution and keep the reference pinned; the
    // augmented (item-0-free) model must not fall below the compact one.
    let (n_items, k) = (10usize, 3usize);
    let (yi, gid) = gen_two_group_gpcm(1200, n_items, k, 0, false, 909);
    let np = gid.len();
    let con = fit_poly_multigroup(
        &yi,
        None,
        &gid,
        2,
        np,
        n_items,
        k,
        PolyModel::Gpcm,
        None,
        21,
        200,
        1e-6,
    )
    .unwrap();
    assert!(con.converged, "compact fit: {}", con.termination_reason);
    assert!(con.n_iter < 200);
    assert_eq!(con.loglik_trace.last().copied(), Some(con.loglik));
    assert!(con.final_delta <= con.stopping_tolerance);
    assert_eq!(con.mu[0], 0.0);
    assert_eq!(con.sigma[0], 1.0);
    assert!(
        (con.mu[1] - 0.5).abs() < 0.15,
        "focal mean not recovered: {}",
        con.mu[1]
    );
    assert!(
        (con.sigma[1] - 1.2).abs() < 0.2,
        "focal sd not recovered: {}",
        con.sigma[1]
    );
    let aug = fit_poly_multigroup(
        &yi,
        None,
        &gid,
        2,
        np,
        n_items,
        k,
        PolyModel::Gpcm,
        Some(0),
        21,
        200,
        1e-6,
    )
    .unwrap();
    assert!(aug.converged, "augmented fit: {}", aug.termination_reason);
    assert!(aug.n_iter < 200);
    assert_eq!(aug.loglik_trace.last().copied(), Some(aug.loglik));
    assert!(aug.final_delta <= aug.stopping_tolerance);
    for fit in [&con, &aug] {
        assert!(fit.loglik_trace.iter().all(|v| v.is_finite()));
        assert!(fit.loglik_trace.windows(2).all(|w| w[1] >= w[0] - 1e-9));
    }
    println!(
        "[poly DIF convergence] compact: reason={} iter={}/200 delta={:.3e} tol={:.3e} ll={:.6}; \
         augmented: reason={} iter={}/200 delta={:.3e} tol={:.3e} ll={:.6}",
        con.termination_reason,
        con.n_iter,
        con.final_delta,
        con.stopping_tolerance,
        con.loglik,
        aug.termination_reason,
        aug.n_iter,
        aug.final_delta,
        aug.stopping_tolerance,
        aug.loglik,
    );
    // nesting, with tolerance-scaled numerical slack
    let slack = 1e-6_f64.max(1e-6 * (1.0 + con.loglik.abs()));
    assert!(
        aug.loglik >= con.loglik - slack,
        "nesting violated: ll_aug={} ll_con={}",
        aug.loglik,
        con.loglik
    );
    assert_eq!(aug.studied_slope.len(), 2);
}

#[test]
fn poly_dif_rejects_empty_declared_group() {
    // Declaring a group with no persons would make df = (n_groups-1)*n_cat
    // count parameters no data can identify (conservative, miscalibrated LR).
    // The data uses labels {0,1}; declaring n_groups=3 leaves group 2 empty.
    let (yi, gid) = gen_two_group_gpcm(300, 6, 3, 0, false, 4242);
    let np = gid.len();
    let err = fit_poly_multigroup(
        &yi,
        None,
        &gid,
        3,
        np,
        6,
        3,
        PolyModel::Gpcm,
        None,
        21,
        50,
        1e-4,
    );
    assert!(err.is_err(), "empty declared group should be rejected");
}

#[test]
fn poly_dif_rejects_unconverged_compact_fit() {
    let (yi, gid) = gen_two_group_gpcm(100, 4, 3, 0, false, 1701);
    let np = gid.len();
    let result = poly_dif_sweep(
        &yi,
        None,
        &gid,
        2,
        np,
        4,
        3,
        PolyModel::Gpcm,
        Some(&[0]),
        7,
        1,
        1e-12,
        0.05,
    );
    let err = match result {
        Ok(_) => panic!("iteration-limited compact fit must fail closed"),
        Err(err) => err,
    };
    assert!(err.contains("did not converge"), "unexpected error: {err}");
    assert!(err.contains("reason=max_iter"), "unexpected error: {err}");
    assert!(err.contains("iteration=1/1"), "unexpected error: {err}");
}

// (Type I over non-DIF items, power on item 0 when DIF is present, mean LR
// among null items) over `reps` two-group datasets. df = (G-1)*K = K.
fn mc_poly_dif(
    reps: usize,
    n_per_group: usize,
    n_items: usize,
    dif: u8,
    skew: bool,
) -> (f64, f64, f64) {
    let k = 3usize;
    let (mut t1_rej, mut t1_cnt) = (0usize, 0usize);
    let (mut pow_rej, mut lr_sum, mut lr_cnt) = (0usize, 0.0_f64, 0usize);
    for rep in 0..reps {
        let seed = 88_000 + rep as u64 * 131 + skew as u64 * 3 + dif as u64 * 7;
        let (yi, gid) = gen_two_group_gpcm(n_per_group, n_items, k, dif, skew, seed);
        let np = gid.len();
        let rows = poly_dif_sweep(
            &yi,
            None,
            &gid,
            2,
            np,
            n_items,
            k,
            PolyModel::Gpcm,
            None,
            21,
            80,
            1e-5,
            0.05,
        )
        .unwrap();
        for r in &rows {
            let rej = r.p_value < 0.05;
            if r.item == 0 && dif != 0 {
                if rej {
                    pow_rej += 1;
                }
            } else {
                // non-DIF items (and item 0 when dif==0) measure Type I
                if rej {
                    t1_rej += 1;
                }
                t1_cnt += 1;
                lr_sum += r.lr;
                lr_cnt += 1;
            }
        }
    }
    let type1 = t1_rej as f64 / t1_cnt as f64;
    let power = if dif != 0 {
        pow_rej as f64 / reps as f64
    } else {
        0.0
    };
    (type1, power, lr_sum / lr_cnt as f64)
}

#[test]
fn poly_dif_type1_and_power() {
    // Fast guard (few reps => Type I lower bound is unmeasurable; mean(LR)~df
    // is the robust cheap calibration). Authoritative >=500-rep study with a
    // tight Type I band is poly_dif_monte_carlo_500.
    let df = 3.0; // (G-1)*K = K = 3
    let (t1, _, mean_lr) = mc_poly_dif(3, 400, 6, 0, false); // no DIF
    let (t1u, pow_u, _) = mc_poly_dif(3, 400, 6, 1, false); // uniform DIF on item 0
    println!(
        "[poly DIF] df={df}  no-DIF: Type I={t1:.3} mean(LR)={mean_lr:.2}  \
         uniform: Type I(others)={t1u:.3} power(item0)={pow_u:.3}"
    );
    assert!(t1 < 0.18, "Type I inflated: {t1}"); // lower bound needs the 500-rep test
    assert!(
        (df - 1.2..=df + 1.4).contains(&mean_lr),
        "mean LR should ~ df={df}: {mean_lr}"
    );
    assert!(pow_u > 0.6, "uniform DIF power too low: {pow_u}");
    assert!(t1u < 0.2, "non-DIF items over-flagged under DIF: {t1u}");
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn poly_dif_monte_carlo_500() {
    let reps = 500usize;
    let (t1, _, mean_lr) = mc_poly_dif(reps, 500, 8, 0, false);
    let (_, pow_u, _) = mc_poly_dif(reps, 500, 8, 1, false);
    let (_, pow_n, _) = mc_poly_dif(reps, 500, 8, 2, false);
    let (t1s, _, _) = mc_poly_dif(reps, 500, 8, 0, true);
    println!(
        "[poly DIF 500] df=3  no-DIF: Type I={t1:.4} mean(LR)={mean_lr:.3}  \
         power: uniform={pow_u:.3} non-uniform={pow_n:.3}  skew: Type I={t1s:.4}"
    );
    assert!((0.03..=0.075).contains(&t1), "Type I off nominal: {t1}");
    assert!(
        (2.6..=3.4).contains(&mean_lr),
        "mean LR should ~ df=3: {mean_lr}"
    );
    assert!(
        pow_u > 0.85 && pow_n > 0.7,
        "DIF power too low: uniform={pow_u} nonuniform={pow_n}"
    );
}

// Hand-coded van der Flier dichotomous U3 (the trusted binary reference the
// polytomous U3 must reduce to at n_cat=2), with the same den=1 boundary.
fn u3_binary_vdf(y: &[usize], n_persons: usize, n_items: usize) -> Vec<f64> {
    let mut w = vec![0.0_f64; n_items];
    for i in 0..n_items {
        let s: usize = (0..n_persons).map(|p| y[p * n_items + i]).sum();
        let pi = s as f64 / n_persons as f64;
        w[i] = if pi <= 0.0 || pi >= 1.0 {
            0.0
        } else {
            (pi / (1.0 - pi)).ln()
        };
    }
    let mut sorted = w.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let mut topsum = vec![0.0_f64; n_items + 1];
    let mut botsum = vec![0.0_f64; n_items + 1];
    for s in 1..=n_items {
        topsum[s] = topsum[s - 1] + sorted[s - 1];
        botsum[s] = botsum[s - 1] + sorted[n_items - s];
    }
    let mut out = vec![0.0_f64; n_persons];
    for p in 0..n_persons {
        let (mut sc, mut wsum) = (0usize, 0.0_f64);
        for i in 0..n_items {
            if y[p * n_items + i] == 1 {
                sc += 1;
                wsum += w[i];
            }
        }
        let den = if sc == 0 || sc == n_items {
            1.0
        } else {
            topsum[sc] - botsum[sc]
        };
        out[p] = if den > 1e-9 {
            (topsum[sc] - wsum) / den
        } else {
            f64::NAN
        };
    }
    out
}

#[test]
fn poly_u3_reduces_to_binary_vdf() {
    // At n_cat=2 the polytomous U3 must be identical to van der Flier's U3
    // (the "reduce to a trusted binary" correctness anchor).
    let mut u = rng(1234);
    let (n_persons, n_items) = (400usize, 12usize);
    let mut y = vec![0usize; n_persons * n_items];
    for v in y.iter_mut() {
        *v = if u() < 0.5 { 1 } else { 0 };
    }
    let res = u3_poly_person_fit(&y, None, n_persons, n_items, 2, None).unwrap();
    let vdf = u3_binary_vdf(&y, n_persons, n_items);
    let mut maxdev = 0.0_f64;
    for p in 0..n_persons {
        let (a, b) = (res.u3poly[p], vdf[p]);
        if a.is_nan() && b.is_nan() {
            continue;
        }
        maxdev = maxdev.max((a - b).abs());
    }
    assert!(
        maxdev < 1e-10,
        "U3poly(K=2) must equal vdF U3: maxdev={maxdev}"
    );
    // orientation: a popularity-inconsistent person scores higher than a
    // consistent one. Build two persons on a fixed 4-item bank.
    let ni = 4;
    // popularities descending: item 0 easiest .. item 3 hardest
    let mut yy = vec![0usize; 40 * ni];
    let mut u2 = rng(99);
    for p in 0..40 {
        for i in 0..ni {
            let pi = 0.8 - 0.18 * i as f64; // 0.80,0.62,0.44,0.26
            yy[p * ni + i] = if u2() < pi { 1 } else { 0 };
        }
    }
    // consistent person (easy items 1, hard 0) vs reversed (hard 1, easy 0)
    yy[0 * ni..1 * ni].copy_from_slice(&[1, 1, 0, 0]);
    yy[1 * ni..2 * ni].copy_from_slice(&[0, 0, 1, 1]);
    let r2 = u3_poly_person_fit(&yy, None, 40, ni, 2, None).unwrap();
    assert!(
        r2.u3poly[1] > r2.u3poly[0],
        "reversed person must have larger U3"
    );
    assert!(
        r2.u3poly[0] < 0.5 && r2.u3poly[1] > 0.5,
        "orientation off: {:?}",
        &r2.u3poly[..2]
    );
}

// GPCM data generator: first `n_care` persons are careless (uniform-random
// categories, ignoring item popularity); the rest respond from the model.
fn gen_u3_data(
    slope: &[f64],
    cat: &[f64],
    n_persons: usize,
    n_items: usize,
    k: usize,
    n_care: usize,
    skew: bool,
    seed: u64,
) -> Vec<usize> {
    let z = k - 1;
    let mut u = rng(seed);
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        let careless = p < n_care;
        let theta = if skew {
            -(u().max(1e-12)).ln() - 1.0
        } else {
            let u1 = u().max(1e-12);
            let u2 = u();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        for i in 0..n_items {
            if careless {
                y[p * n_items + i] = ((u() * k as f64) as usize).min(k - 1);
            } else {
                let base = slope[i] * theta;
                let scores: Vec<f64> = (0..k).map(|c| c as f64).collect();
                let mut ic = vec![0.0_f64; k];
                ic[1..].copy_from_slice(&cat[i * z..(i + 1) * z]);
                let lp = gpcm_logprobs(base, &scores, &ic);
                let draw = u();
                let (mut acc, mut c) = (0.0_f64, k - 1);
                for (cc, l) in lp.iter().enumerate() {
                    acc += l.exp();
                    if draw <= acc {
                        c = cc;
                        break;
                    }
                }
                y[p * n_items + i] = c;
            }
        }
    }
    y
}

fn quantile_sorted(v: &mut Vec<f64>, q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let idx = (n as f64 - 1.0) * q;
    let (lo, hi) = (idx.floor() as usize, idx.ceil() as usize);
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (idx - lo as f64) * (v[hi] - v[lo])
    }
}

// Returns (marginal Type I, max |flag_rate - alpha| across total-score bins,
// power on careless responders). The cutoff is the (1-alpha) quantile of null
// U3poly estimated under the MATCHING latent shape from disjoint seeds.
fn mc_u3poly(reps: usize, n_persons: usize, skew: bool) -> (f64, f64, f64) {
    let (n_items, k) = (20usize, 5usize);
    let alpha = 0.05_f64;
    let (slope, cat) = cat_bank(n_items, k);
    let maxnc = n_items * (k - 1);
    let so = if skew { 7 } else { 0 };
    // cutoff from pooled null U3poly (seed base 900000, disjoint from eval)
    let mut pool = Vec::new();
    for b in 0..6u64 {
        let y = gen_u3_data(
            &slope,
            &cat,
            n_persons,
            n_items,
            k,
            0,
            skew,
            900_000 + b * 131 + so,
        );
        let r = u3_poly_person_fit(&y, None, n_persons, n_items, k, None).unwrap();
        pool.extend(r.u3poly.into_iter().filter(|v| v.is_finite()));
    }
    let cutoff = quantile_sorted(&mut pool, 1.0 - alpha);
    let n_bins = 3usize;
    let (mut bin_flag, mut bin_tot) = (vec![0usize; n_bins], vec![0usize; n_bins]);
    let (mut t1_flag, mut t1_tot) = (0usize, 0usize);
    let (mut pw_flag, mut pw_tot) = (0usize, 0usize);
    let n_care = n_persons / 5; // 20% careless in the power datasets
    for rep in 0..reps as u64 {
        // null eval (disjoint seed base 100000)
        let yn = gen_u3_data(
            &slope,
            &cat,
            n_persons,
            n_items,
            k,
            0,
            skew,
            100_000 + rep * 131 + so,
        );
        let rn = u3_poly_person_fit(&yn, None, n_persons, n_items, k, Some(cutoff)).unwrap();
        for p in 0..n_persons {
            if rn.u3poly[p].is_finite() {
                t1_tot += 1;
                if rn.flagged[p] {
                    t1_flag += 1;
                }
                let bin = (rn.total_score[p] * n_bins / (maxnc + 1)).min(n_bins - 1);
                bin_tot[bin] += 1;
                if rn.flagged[p] {
                    bin_flag[bin] += 1;
                }
            }
        }
        // power eval (careless responders, seed base 200000)
        let ya = gen_u3_data(
            &slope,
            &cat,
            n_persons,
            n_items,
            k,
            n_care,
            skew,
            200_000 + rep * 131 + so,
        );
        let ra = u3_poly_person_fit(&ya, None, n_persons, n_items, k, Some(cutoff)).unwrap();
        for p in 0..n_care {
            if ra.u3poly[p].is_finite() {
                pw_tot += 1;
                if ra.flagged[p] {
                    pw_flag += 1;
                }
            }
        }
    }
    let type1 = t1_flag as f64 / t1_tot.max(1) as f64;
    let bin_maxdev = (0..n_bins)
        .map(|b| (bin_flag[b] as f64 / bin_tot[b].max(1) as f64 - alpha).abs())
        .fold(0.0_f64, f64::max);
    let power = pw_flag as f64 / pw_tot.max(1) as f64;
    (type1, bin_maxdev, power)
}

#[test]
fn poly_u3_type1_and_power() {
    // Fast guard. Authoritative >=500-rep study is poly_u3_monte_carlo_500.
    let (t1, _bindev, power) = mc_u3poly(6, 500, false);
    println!("[u3poly] normal: Type I={t1:.3} power(careless)={power:.3}");
    assert!((0.01..=0.12).contains(&t1), "Type I off nominal: {t1}");
    assert!(power > 0.5, "careless-detection power too low: {power}");
}
#[test]
#[ignore = "literature-grade Monte-Carlo (>=500 reps); run with: cargo test --release -- --ignored --nocapture"]
fn poly_u3_monte_carlo_500() {
    let reps = 500usize;
    let (t1n, bindev_n, pow_n) = mc_u3poly(reps, 600, false);
    let (t1s, bindev_s, pow_s) = mc_u3poly(reps, 600, true);
    println!(
        "[u3poly 500] normal: Type I={t1n:.4} bin-maxdev={bindev_n:.3} power={pow_n:.3}  \
         skew: Type I={t1s:.4} bin-maxdev={bindev_s:.3} power={pow_s:.3}"
    );
    // marginal Type I calibrated by the simulated cutoff; per-NC-bin deviation
    // reported (a single pooled cutoff cannot perfectly condition on the total
    // score — Emons 2008 uses simulated critical values for this reason).
    assert!(
        (0.03..=0.08).contains(&t1n),
        "normal Type I off nominal: {t1n}"
    );
    assert!(pow_n > 0.7, "normal careless power too low: {pow_n}");
    assert!(
        bindev_n < 0.10,
        "per-score-group miscalibration too large: {bindev_n}"
    );
}
