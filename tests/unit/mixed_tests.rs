use super::*;

#[test]
fn ggum_probabilities_match_paired_subjective_category_formula() {
    let spec = MixedItemSpec {
        kind: MixedItemKind::Ggum,
        n_categories: 4,
    };
    let a = 1.2_f64;
    let delta = -0.3;
    let thresholds = [0.8, 0.2, -0.4];
    let mut params = vec![a.ln(), delta];
    params.extend(ordered_raw(&thresholds));

    let theta = 0.7;
    let actual = item_logprobs(&spec, &params, theta, &[], 0);

    // Roberts et al. (2000): P(Z=z) is proportional to
    // f(z) + f(M-z), where the subjective-category thresholds are
    // symmetric around the zero middle threshold.
    let c = spec.n_categories - 1;
    let m = 2 * c + 1;
    let mut tau = vec![0.0; m + 1];
    tau[1..=c].copy_from_slice(&thresholds);
    for z in 1..=c {
        tau[m - z + 1] = -thresholds[z - 1];
    }
    let mut cumulative_tau = 0.0;
    let mut log_f = Vec::with_capacity(m + 1);
    for (w, &tau_w) in tau.iter().enumerate() {
        cumulative_tau += tau_w;
        log_f.push(a * (w as f64 * (theta - delta) - cumulative_tau));
    }
    let paired: Vec<f64> = (0..=c).map(|z| logaddexp(log_f[z], log_f[m - z])).collect();
    let expected = softmax_log(&paired);

    for (category, (got, want)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (got - want).abs() < 1e-12,
            "category {category}: got {got}, expected {want}"
        );
    }
}

#[test]
fn every_mixed_cell_normalizes() {
    let cases = [
        (MixedItemKind::Rasch, 2),
        (MixedItemKind::TwoPl, 2),
        (MixedItemKind::ThreePl, 2),
        (MixedItemKind::ThreePlUpper, 2),
        (MixedItemKind::FourPl, 2),
        (MixedItemKind::Cll, 2),
        (MixedItemKind::Grm, 4),
        (MixedItemKind::Pcm, 4),
        (MixedItemKind::Gpcm, 4),
        (MixedItemKind::Sequential, 4),
        (MixedItemKind::Tutz, 4),
        (MixedItemKind::Nominal, 4),
        (MixedItemKind::Ideal, 2),
        (MixedItemKind::Ggum, 4),
        (MixedItemKind::Lsirm, 2),
        (MixedItemKind::LsirmGrm, 4),
        (MixedItemKind::LsirmGpcm, 4),
    ];
    for (kind, n_categories) in cases {
        let spec = MixedItemSpec { kind, n_categories };
        let latent_dim = if kind.is_spatial() { 2 } else { 0 };
        let freq = vec![1.0 / n_categories as f64; n_categories];
        let params = initial_params(&spec, &freq, 0, 1, latent_dim);
        for theta in [-4.0, 0.0, 4.0] {
            let xi = if latent_dim == 0 {
                &[][..]
            } else {
                &[0.3, -0.2][..]
            };
            let lp = item_logprobs(&spec, &params, theta, xi, latent_dim);
            assert_eq!(lp.len(), n_categories);
            assert!(lp.iter().all(|v| v.is_finite()), "{kind:?}: {lp:?}");
            let total: f64 = lp.iter().map(|v| v.exp()).sum();
            assert!((total - 1.0).abs() < 1e-10, "{kind:?}: {total}");
        }
    }
}

#[test]
fn binary_cells_match_their_defining_formulas() {
    let theta = 0.4;
    let rasch = MixedItemSpec {
        kind: MixedItemKind::Rasch,
        n_categories: 2,
    };
    let lp = item_logprobs(&rasch, &[-0.3], theta, &[], 0);
    assert!((lp[1].exp() - logistic(theta + 0.3)).abs() < 1e-12);

    let two = MixedItemSpec {
        kind: MixedItemKind::TwoPl,
        n_categories: 2,
    };
    let lp = item_logprobs(&two, &[1.2_f64.ln(), -0.3], theta, &[], 0);
    let expected = 1.0 / (1.0 + (-(1.2 * theta - 0.3)).exp());
    assert!((lp[1].exp() - expected).abs() < 1e-12);

    let three = MixedItemSpec {
        kind: MixedItemKind::ThreePl,
        n_categories: 2,
    };
    let raw_lower = logit(0.2);
    let lp = item_logprobs(&three, &[1.2_f64.ln(), -0.3, raw_lower], theta, &[], 0);
    let expected = 0.2 + 0.8 * logistic(1.2 * theta - 0.3);
    assert!((lp[1].exp() - expected).abs() < 1e-12);

    let upper = MixedItemSpec {
        kind: MixedItemKind::ThreePlUpper,
        n_categories: 2,
    };
    let lp = item_logprobs(&upper, &[1.2_f64.ln(), -0.3, logit(0.85)], theta, &[], 0);
    let expected = 0.85 * logistic(1.2 * theta - 0.3);
    assert!((lp[1].exp() - expected).abs() < 1e-12);

    let four = MixedItemSpec {
        kind: MixedItemKind::FourPl,
        n_categories: 2,
    };
    let raw_gap = logit((0.85 - 0.2) / (1.0 - 0.2));
    let params = [1.2_f64.ln(), -0.3, raw_lower, raw_gap];
    let lp = item_logprobs(&four, &params, theta, &[], 0);
    let expected = 0.2 + 0.65 * logistic(1.2 * theta - 0.3);
    assert!((lp[1].exp() - expected).abs() < 1e-12);
    let estimate = public_estimate(&four, &params, 0);
    assert!((estimate.lower_asymptote.unwrap() - 0.2).abs() < 1e-12);
    assert!((estimate.upper_asymptote.unwrap() - 0.85).abs() < 1e-12);

    let cll = MixedItemSpec {
        kind: MixedItemKind::Cll,
        n_categories: 2,
    };
    let lp = item_logprobs(&cll, &[-0.3], theta, &[], 0);
    let expected = 1.0 - (-(theta + 0.3).exp()).exp();
    assert!((lp[1].exp() - expected).abs() < 1e-12);

    let ideal = MixedItemSpec {
        kind: MixedItemKind::Ideal,
        n_categories: 2,
    };
    let lp = item_logprobs(&ideal, &[1.5_f64.ln(), -0.2], theta, &[], 0);
    let expected = (-0.5 * (1.5 * (theta + 0.2)).powi(2)).exp();
    assert!((lp[1].exp() - expected).abs() < 1e-12);
}

#[test]
fn partial_credit_and_sequential_cells_match_definitions() {
    let theta = 0.35;
    let pcm = MixedItemSpec {
        kind: MixedItemKind::Pcm,
        n_categories: 3,
    };
    let pcm_lp = item_logprobs(&pcm, &[0.2, -0.4], theta, &[], 0);
    let expected = gpcm_logprobs(theta, &[0.0, 1.0, 2.0], &[0.0, 0.2, -0.4]);
    for (got, want) in pcm_lp.iter().zip(expected) {
        assert!((*got - want).abs() < 1e-12);
    }

    let sequential = MixedItemSpec {
        kind: MixedItemKind::Sequential,
        n_categories: 3,
    };
    let params = [1.4_f64.ln(), 0.2, -0.5];
    let lp = item_logprobs(&sequential, &params, theta, &[], 0);
    let q1 = logistic(1.4 * theta + 0.2);
    let q2 = logistic(1.4 * theta - 0.5);
    let expected = [1.0 - q1, q1 * (1.0 - q2), q1 * q2];
    for (got, want) in lp.iter().map(|v| v.exp()).zip(expected) {
        assert!((got - want).abs() < 1e-12);
    }
    let estimate = public_estimate(&sequential, &params, 0);
    assert_eq!(estimate.intercepts, vec![0.2, -0.5]);

    let tutz = MixedItemSpec {
        kind: MixedItemKind::Tutz,
        n_categories: 3,
    };
    let lp = item_logprobs(&tutz, &[0.2, -0.5], theta, &[], 0);
    let q1 = logistic(theta + 0.2);
    let q2 = logistic(theta - 0.5);
    let expected = [1.0 - q1, q1 * (1.0 - q2), q1 * q2];
    for (got, want) in lp.iter().map(|v| v.exp()).zip(expected) {
        assert!((got - want).abs() < 1e-12);
    }
    let estimate = public_estimate(&tutz, &[0.2, -0.5], 0);
    assert_eq!(estimate.intercepts, vec![0.2, -0.5]);
}

#[test]
fn new_family_aliases_and_public_constraints_are_explicit() {
    let aliases = [
        ("1pl", MixedItemKind::Rasch, "rasch"),
        ("partial_credit", MixedItemKind::Pcm, "pcm"),
        ("upper_3pl", MixedItemKind::ThreePlUpper, "3plu"),
        ("complementary_log_log", MixedItemKind::Cll, "cll"),
        ("sequential", MixedItemKind::Sequential, "sequential"),
        ("tutz", MixedItemKind::Tutz, "tutz"),
    ];
    for (alias, kind, canonical) in aliases {
        assert_eq!(MixedItemKind::parse(alias).unwrap(), kind);
        assert_eq!(kind.as_str(), canonical);
    }
    assert!(MixedItemKind::parse("not-a-family").is_err());

    let four = MixedItemSpec {
        kind: MixedItemKind::FourPl,
        n_categories: 2,
    };
    let mut extreme = [8.0, 20.0, -20.0, 20.0];
    clamp_params(&four, &mut extreme, 0);
    assert_eq!(extreme[0], 4.0);
    assert_eq!(extreme[1], 12.0);
    let estimate = public_estimate(&four, &extreme, 0);
    let lower = estimate.lower_asymptote.unwrap();
    let upper = estimate.upper_asymptote.unwrap();
    assert!(0.0 < lower && lower < upper && upper < 1.0);
}

#[test]
fn numeric_hessian_is_symmetrized_without_order_bias() {
    let mut hessian = vec![vec![2.0, 4.0], vec![8.0, 6.0]];
    symmetrize_and_ridge(&mut hessian, 0.25);
    assert_eq!(hessian, vec![vec![2.25, 6.0], vec![6.0, 6.25]]);
}

#[test]
fn mixed_item_line_search_stops_at_a_clamped_boundary() {
    let spec = MixedItemSpec {
        kind: MixedItemKind::Rasch,
        n_categories: 2,
    };
    let grid = build_grid(std::slice::from_ref(&spec), 0, 7, 7).unwrap();
    let mut counts = vec![0.0; grid.cell() * 2];
    for node in 0..grid.cell() {
        counts[node * 2] = 1.0;
    }
    let fitted = m_step_item(&spec, &[12.0], &grid, &counts, 1);
    assert_eq!(fitted, vec![12.0]);
}

#[test]
fn rejects_hidden_nonconvergence_as_success() {
    let y = vec![0, 0, 1, 1, 0, 1, 1, 0];
    let specs = vec![
        MixedItemSpec {
            kind: MixedItemKind::TwoPl,
            n_categories: 2,
        },
        MixedItemSpec {
            kind: MixedItemKind::TwoPl,
            n_categories: 2,
        },
    ];
    let fit = fit_mixed_items(&y, None, 4, 2, &specs, 1, 7, 7, 1, 1e-14, 1).unwrap();
    assert!(!fit.converged);
    assert_eq!(fit.termination_reason, "max_iter_reached");
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.loglik_trace.len(), 2);
}

#[test]
fn mixed_fit_executes_every_response_family() {
    let cases = [
        (MixedItemKind::Rasch, 2),
        (MixedItemKind::TwoPl, 2),
        (MixedItemKind::ThreePl, 2),
        (MixedItemKind::ThreePlUpper, 2),
        (MixedItemKind::FourPl, 2),
        (MixedItemKind::Cll, 2),
        (MixedItemKind::Grm, 3),
        (MixedItemKind::Pcm, 3),
        (MixedItemKind::Gpcm, 3),
        (MixedItemKind::Sequential, 3),
        (MixedItemKind::Tutz, 3),
        (MixedItemKind::Nominal, 3),
        (MixedItemKind::Ideal, 2),
        (MixedItemKind::Ggum, 3),
        (MixedItemKind::Lsirm, 2),
        (MixedItemKind::LsirmGrm, 3),
        (MixedItemKind::LsirmGpcm, 3),
    ];
    let specs: Vec<MixedItemSpec> = cases
        .iter()
        .map(|&(kind, n_categories)| MixedItemSpec { kind, n_categories })
        .collect();
    let n_persons = 4;
    let n_items = specs.len();
    let mut y = vec![0usize; n_persons * n_items];
    for person in 0..n_persons {
        for (item, spec) in specs.iter().enumerate() {
            y[person * n_items + item] = if person % 2 == 0 {
                0
            } else {
                spec.n_categories - 1
            };
        }
    }

    let fit = fit_mixed_items(&y, None, n_persons, n_items, &specs, 1, 7, 7, 1, 1e-12, 0)
        .expect("all documented mixed response families must execute in one calibration");
    assert_eq!(fit.items.len(), n_items);
    assert_eq!(fit.theta_eap.len(), n_persons);
    assert_eq!(fit.theta_sd.len(), n_persons);
    assert_eq!(fit.xi_eap.len(), n_persons);
    assert_eq!(fit.latent_dim, 1);
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.termination_reason, "max_iter_reached");
    assert!(fit.loglik.is_finite());
    assert!(fit.loglik_trace.iter().all(|value| value.is_finite()));
    assert!(fit.theta_eap.iter().all(|value| value.is_finite()));
    assert!(fit.theta_sd.iter().all(|value| value.is_finite()));
    assert!(fit.xi_eap.iter().all(|value| value.is_finite()));
}

#[test]
fn mixed_fit_covers_parallel_masked_and_converged_paths() {
    let n_persons = 256;
    let n_items = 4;
    let specs = vec![
        MixedItemSpec {
            kind: MixedItemKind::TwoPl,
            n_categories: 2,
        };
        n_items
    ];
    let y: Vec<usize> = (0..n_persons * n_items)
        .map(|index| (index / n_items + index % n_items) % 2)
        .collect();
    let mut observed = vec![true; y.len()];
    observed[0] = false;

    let fit = fit_mixed_items(
        &y,
        Some(&observed),
        n_persons,
        n_items,
        &specs,
        1,
        7,
        7,
        2,
        1e12,
        2,
    )
    .unwrap();
    assert!(fit.converged);
    assert_eq!(fit.termination_reason, "converged");
    assert!(fit.n_threads >= 1);
    assert!(fit
        .loglik_trace
        .windows(2)
        .all(|pair| pair[1] + 1e-8 >= pair[0]));
}

#[test]
fn mixed_fit_validation_and_helper_boundaries() {
    assert_eq!(contextualize_mixed_update(Ok(0.25)).unwrap(), 0.25);
    assert_eq!(
        contextualize_mixed_update(Err("non_monotone_update")).unwrap_err(),
        "mixed-format EM update failed: non_monotone_update"
    );
    let binary = MixedItemSpec {
        kind: MixedItemKind::TwoPl,
        n_categories: 2,
    };
    let spatial = MixedItemSpec {
        kind: MixedItemKind::Lsirm,
        n_categories: 2,
    };
    let call = |y: &[usize],
                observed: Option<&[bool]>,
                n_persons,
                n_items,
                specs: &[MixedItemSpec],
                max_iter,
                tol| {
        fit_mixed_items(
            y, observed, n_persons, n_items, specs, 1, 7, 7, max_iter, tol, 1,
        )
    };

    assert!(call(&[], None, 0, 1, &[binary.clone()], 1, 1e-6).is_err());
    assert!(call(
        &[],
        None,
        usize::MAX,
        2,
        &[binary.clone(), binary.clone()],
        1,
        1e-6
    )
    .is_err());
    assert!(call(&[0], None, 1, 2, &[binary.clone(), binary.clone()], 1, 1e-6).is_err());
    assert!(call(&[0, 1], None, 2, 1, &[], 1, 1e-6).is_err());
    assert!(call(&[0, 1], Some(&[true]), 2, 1, &[binary.clone()], 1, 1e-6).is_err());
    assert!(call(&[0, 1], None, 2, 1, &[binary.clone()], 0, 1e-6).is_err());
    assert!(call(&[0, 1], None, 2, 1, &[binary.clone()], 1, f64::NAN).is_err());
    assert!(call(
        &[0, 1],
        None,
        2,
        1,
        &[MixedItemSpec {
            kind: MixedItemKind::Pcm,
            n_categories: 1,
        }],
        1,
        1e-6,
    )
    .is_err());
    assert!(call(
        &[0, 1],
        None,
        2,
        1,
        &[MixedItemSpec {
            kind: MixedItemKind::Rasch,
            n_categories: 3,
        }],
        1,
        1e-6,
    )
    .is_err());
    assert!(call(&[0, 2], None, 2, 1, &[binary.clone()], 1, 1e-6).is_err());
    assert!(call(&[0, 0], None, 2, 1, &[binary.clone()], 1, 1e-6).is_err());
    assert!(build_grid(&[spatial.clone()], 0, 7, 7).is_err());
    assert!(build_grid(&[binary.clone()], 1, 5, 7).is_err());
    assert!(build_grid(&[spatial], 1, 7, 5).is_err());
    assert!(tensor_grid(41, 4).is_err());
    assert!(ordered_values(&[]).is_empty());
    assert!(ordered_raw(&[]).is_empty());
    assert_eq!(asymptotes(MixedItemKind::Rasch, &[0.0]), (0.0, 1.0));

    let aliases = [
        ("binary", MixedItemKind::TwoPl),
        ("3pl", MixedItemKind::ThreePl),
        ("4pl", MixedItemKind::FourPl),
        ("graded", MixedItemKind::Grm),
        ("gpcm", MixedItemKind::Gpcm),
        ("nrm", MixedItemKind::Nominal),
        ("ideal_point", MixedItemKind::Ideal),
        ("ggum", MixedItemKind::Ggum),
        ("lsirm", MixedItemKind::Lsirm),
        ("lsirm_grm", MixedItemKind::LsirmGrm),
        ("lsirm_gpcm", MixedItemKind::LsirmGpcm),
    ];
    for (name, kind) in aliases {
        assert_eq!(MixedItemKind::parse(name).unwrap(), kind);
        assert_eq!(kind.as_str(), MixedItemKind::parse(name).unwrap().as_str());
    }

    let params = initial_params(
        &MixedItemSpec {
            kind: MixedItemKind::LsirmGpcm,
            n_categories: 3,
        },
        &[0.3, 0.4, 0.3],
        1,
        3,
        3,
    );
    assert_eq!(params.len(), 6);
    assert!(params.iter().all(|value| value.is_finite()));

    assert_eq!(assess_loglik_update(-10.0, -9.5), Ok(0.5));
    assert_eq!(
        assess_loglik_update(-10.0, f64::NAN),
        Err("non_finite_loglik")
    );
    assert_eq!(
        assess_loglik_update(-10.0, -11.0),
        Err("non_monotone_update")
    );

    let grid = build_grid(&[binary.clone()], 1, 7, 7).unwrap();
    let params = vec![initial_params(&binary, &[0.5, 0.5], 0, 1, 0)];
    let counts = vec![vec![f64::NAN; grid.cell() * 2]];
    let fitted = m_step(&[binary], &params, &grid, &counts, 1);
    assert_eq!(fitted.len(), 1);

    let binary = MixedItemSpec {
        kind: MixedItemKind::TwoPl,
        n_categories: 2,
    };
    let grid = build_grid(&[binary.clone()], 1, 7, 7).unwrap();
    let initial = initial_params(&binary, &[0.5, 0.5], 0, 1, 0);
    let zero_counts = vec![0.0; grid.cell() * 2];
    let stationary = m_step_item(&binary, &initial, &grid, &zero_counts, 6);
    assert_eq!(stationary.len(), initial.len());
}
