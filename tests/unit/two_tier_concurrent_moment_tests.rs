use super::*;

pub(crate) fn synthetic_group_sweep_oracle() {
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        q_primary: 3,
        q_specific: 3,
        max_iter: 1,
        tol: 1e-8,
        n_starts: 1,
        seed: 9178,
        newton_iter: 20,
        ridge: 1e-8,
    };
    let primary = [true, false, true, true, true, false, true, true];
    let specific = [0, 0, 0, 0];
    let y: Vec<usize> = (0..16)
        .flat_map(|p| (0..4).map(move |i| (p >> i) & 1))
        .collect();
    // One entire specific block is missing for one person. Its Gaussian
    // posterior is the prior, and must not vanish from population moments.
    let observed: Vec<bool> = (0..64).map(|k| k >= 4 && k % 7 != 0).collect();
    let v = validate(
        &y,
        Some(&observed),
        &primary,
        &specific,
        16,
        4,
        2,
        1,
        2,
        &cfg,
    )
    .unwrap();
    let (params, _) = initial_params(&v, &y, Some(&observed), cfg.seed, 1, false);
    let (z, w) = gh_rule(3).unwrap();
    let (base, logw) = build_primary_grid(z, w, 2, 9);
    let lw: Vec<f64> = w.iter().map(|x| x.ln()).collect();
    let moments = vec![
        MomentConstraint {
            mean: 0.25,
            sd: 1.1,
            free_mean: true,
            free_sd: true,
        },
        MomentConstraint {
            mean: -0.2,
            sd: 0.8,
            free_mean: true,
            free_sd: true,
        },
        MomentConstraint {
            mean: 0.3,
            sd: 1.2,
            free_mean: true,
            free_sd: true,
        },
    ];
    let data = GroupData {
        v,
        y,
        observed: Some(observed),
    };
    let (state, receipt) = group_sweep(&data, &params, &moments, &base, &logw, z, &lw).unwrap();
    let (ll, counts, s1, s2, ss2, mass, _, _) = e_step_fipc_cpu(
        &data.v,
        &data.y,
        data.observed.as_deref(),
        &params,
        &logw,
        &[lw],
        &state.coords,
        &state.specific_nodes,
        9,
        3,
    );
    let counts_error = state
        .counts
        .iter()
        .flatten()
        .flatten()
        .zip(counts.iter().flatten().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0., f64::max);
    assert!((ll - state.loglik).abs() < 1e-12);
    assert!(
        counts_error < 1e-5,
        "native f32 count transport error={counts_error:e}"
    );
    for d in 0..2 {
        assert!((s1[d] - state.first[d]).abs() < 1e-5);
        assert!((s2[d * 2 + d] - state.second[d]).abs() < 1e-5);
        assert!((state.mass[d] - 16.).abs() < 1e-5);
    }
    assert_eq!(receipt.cpu_missing_block_prior_moment_completions, 1);
    let prior_second = w
        .iter()
        .zip(&state.specific_nodes[0])
        .map(|(w, x)| w * x * x)
        .sum::<f64>();
    assert!((ss2[0] + prior_second - state.second[2]).abs() < 2e-5);
    assert!((mass[0] + 1. - state.mass[2]).abs() < 1e-5);
    // Separate brute conditional posterior for the one-specific synthetic model;
    // it integrates all primary/specific nodes even when individual items miss.
    let mut first = 0.;
    let mut second = 0.;
    let mut total = 0.;
    let lw: Vec<f64> = w.iter().map(|x| x.ln()).collect();
    for person in 0..16 {
        let mut logs = Vec::new();
        for g in 0..9 {
            for h in 0..3 {
                let mut l = logw[g] + lw[h];
                for i in 0..4 {
                    if data.observed.as_ref().unwrap()[person * 4 + i] {
                        l += item_cat_logprob_fipc(
                            &data.v,
                            &params,
                            &state.coords,
                            &state.specific_nodes,
                            i,
                            g,
                            h,
                            data.y[person * 4 + i],
                        );
                    }
                }
                logs.push(l);
            }
        }
        let denom = log_sum_exp(&logs);
        for (node, l) in logs.into_iter().enumerate() {
            let post = (l - denom).exp();
            let x = state.specific_nodes[0][node % 3];
            total += post;
            first += post * x;
            second += post * x * x;
        }
    }
    assert!((total - state.mass[2]).abs() < 1e-5);
    assert!((first - state.first[2]).abs() < 2e-5);
    assert!((second - state.second[2]).abs() < 2e-5);
    eprintln!("SYNTHETIC_MOMENT_ORACLE count_max_abs={counts_error:e}");
}

#[test]
fn gaussian_update_respects_separate_fixed_free_constraints() {
    let fixed = MomentConstraint {
        mean: 2.,
        sd: 3.,
        free_mean: false,
        free_sd: false,
    };
    assert_eq!(update_moment(fixed, 2., 2., 8.).unwrap(), fixed);
    let mean = update_moment(
        MomentConstraint {
            free_mean: true,
            ..fixed
        },
        2.,
        2.,
        8.,
    )
    .unwrap();
    assert_eq!((mean.mean, mean.sd), (1., 3.));
    let sd = update_moment(
        MomentConstraint {
            free_sd: true,
            ..fixed
        },
        2.,
        2.,
        8.,
    )
    .unwrap();
    assert_eq!((sd.mean, sd.sd), (2., 2.)); // second about fixed mean2
    let both = update_moment(
        MomentConstraint {
            free_mean: true,
            free_sd: true,
            ..fixed
        },
        2.,
        2.,
        8.,
    )
    .unwrap();
    assert_eq!(both.mean, 1.);
    assert_eq!(both.sd, 3_f64.sqrt());
    assert!(update_moment(fixed, 0., 0., 0.).is_err());
    assert!(update_moment(
        MomentConstraint {
            free_sd: true,
            ..fixed
        },
        1.,
        2.,
        4.
    )
    .is_err());
}
