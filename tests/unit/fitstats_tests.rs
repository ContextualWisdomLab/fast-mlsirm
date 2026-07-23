use super::*;
use crate::ModelType;

#[test]
fn chi2_sf_reference_values() {
    assert!((chi2_sf(3.841, 1.0) - 0.05).abs() < 1e-3);
    assert!((chi2_sf(18.307, 10.0) - 0.05).abs() < 1e-3);
    assert!((chi2_sf(0.0, 5.0) - 1.0).abs() < 1e-12);
    assert!(chi2_sf(1e6, 2.0) < 1e-12);
}

#[test]
fn bh_step_up_known_case() {
    let p = [
        0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216,
    ];
    let r = benjamini_hochberg(&p, 0.05);
    assert_eq!(r.iter().filter(|&&v| v).count(), 2);
    assert!(r[0] && r[1]);
}

#[test]
fn leniency_residuals_respect_mask_and_sign() {
    let y = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let observed = vec![true, true, true, false, false, false];
    let prob = vec![0.2, 0.2, 0.2, 0.2, 0.2, 0.2];
    let result = leniency_residuals(&y, &observed, &prob, 3).unwrap();
    // Reads crate-returned values and kills mutations that flip residual sign
    // or ignore observed-mask filtering; the empty third row kills mutations
    // that leak NaN/empty rows into public outputs or summary statistics.
    assert!(result.residual[0] > 0.75);
    assert!(result.residual[1] < -0.15);
    assert!(result.residual[0] > result.residual[1]);
    assert_eq!(result.residual[2], 0.0);
    assert_eq!(result.n_observed, vec![2, 1, 0]);
    assert!(result.mean > 0.29);
    assert!(result.abs_p95 > result.residual[1].abs());
    assert!(result.abs_p95 < result.residual[0].abs());
}

fn toy_bank_data() -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<f64>,
    Vec<bool>,
    Vec<f64>,
    Vec<f64>,
) {
    // 1 dim, 20 items, 2000 persons simulated from a plain 1PL (MIRT
    // flags); person-fit asymptotics are in the item count, and the S-X2
    // effect size needs enough persons per score group to separate
    // sampling noise from systematic misfit.
    let n_items = 20usize;
    let n_persons = 2000usize;
    let alpha = vec![0.0; n_items];
    let b: Vec<f64> = (0..n_items).map(|i| -1.2 + 0.12 * i as f64).collect();
    let zeta = vec![0.0; n_items];
    let fid = vec![0usize; n_items];
    let mut state = 777u64;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut theta = vec![0.0_f64; n_persons];
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let u1: f64 = unif().max(1e-12);
        let u2: f64 = unif();
        theta[p] = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let eta: f64 = theta[p] + b[i];
            let prob = 1.0 / (1.0 + (-eta).exp());
            y[p * n_items + i] = if unif() < prob { 1.0 } else { 0.0 };
        }
    }
    let observed = vec![true; n_persons * n_items];
    let xi = vec![0.0_f64; n_persons];
    (alpha, b, zeta, fid, y, observed, theta, xi)
}

#[test]
fn sx2_runs_and_effect_size_is_small_for_true_model() {
    let (alpha, b, zeta, fid, y, observed, _, _) = toy_bank_data();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let res = s_x2(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config {
            q_theta: 21,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    let finite = res.statistic.iter().filter(|v| v.is_finite()).count();
    assert!(finite >= 15);
    // data simulated from the scoring model: typical effect sizes stay low
    // (the residual RMS at this N is dominated by ~sqrt(p(1-p)/N_s) noise)
    let mean_effect: f64 = res
        .rms_residual
        .iter()
        .filter(|v| v.is_finite())
        .sum::<f64>()
        / finite as f64;
    assert!(
        mean_effect < 0.05,
        "effect size too large for a true model: {mean_effect}"
    );
}

#[test]
fn sx2_rejects_non_dichotomous_responses() {
    // A non-0/1 observed value would index the summed-score table out of bounds.
    let (alpha, b, zeta, fid, mut y, observed, _, _) = toy_bank_data();
    y[0] = 2.0;
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let res = s_x2(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config {
            q_theta: 21,
            ..Default::default()
        },
        None,
    );
    let err = res.err().expect("expected an error");
    assert!(err.contains("dichotomous"), "got: {err}");
}

#[test]
fn sx2_rejects_malformed_bank_controls_and_weights() {
    let (alpha, b, zeta, mut fid, y, observed, _, _) = toy_bank_data();
    fid.pop();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let err = s_x2(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config::default(),
        None,
    )
    .err()
    .expect("expected malformed bank error");
    assert!(err.contains("inconsistent lengths"), "got: {err}");

    let valid_fid = vec![0usize; b.len()];
    let valid_bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &valid_fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let bad_weight = vec![0.5; 2000];
    let err = s_x2(
        &valid_bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config::default(),
        Some(&bad_weight),
    )
    .err()
    .expect("expected invalid weight error");
    assert!(err.contains("0/1"), "got: {err}");

    let err = s_x2(
        &valid_bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config {
            fdr_q: f64::NAN,
            ..Default::default()
        },
        None,
    )
    .err()
    .expect("expected invalid fdr error");
    assert!(err.contains("fdr_q"), "got: {err}");
}

#[test]
fn sx2_extreme_item_probabilities_remain_finite() {
    let (alpha, _b, zeta, fid, mut y, observed, _, _) = toy_bank_data();
    let b = vec![-1000.0; alpha.len()];
    for (index, value) in y.iter_mut().enumerate() {
        *value = ((index.wrapping_mul(17).wrapping_add(3)) % 5 < 2) as u8 as f64;
    }
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let result = s_x2(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config::default(),
        None,
    )
    .unwrap();
    assert!(result.statistic.iter().all(|value| value.is_finite()));
    assert!(result
        .rms_residual
        .iter()
        .zip(&result.n_score_groups)
        .all(|(value, &groups)| value.is_finite() || groups == 0));
    assert!(result
        .g2_statistic
        .iter()
        .zip(&result.n_score_groups)
        .all(|(value, &groups)| value.is_finite() || groups == 0));
    assert!(result
        .g2_p_value
        .iter()
        .zip(&result.df)
        .all(|(value, &df)| value.is_finite() || !df.is_finite() || df < 1.0));
}

#[test]
fn sx2_g2_p_values_follow_chi2_sf_mapping() {
    let (alpha, b, zeta, fid, y, observed, _, _) = toy_bank_data();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let result = s_x2(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        &SX2Config::default(),
        None,
    )
    .unwrap();
    // Reads crate-returned g2_statistic/g2_p_value and fails if the implementation
    // mutates to use the wrong p-value mapping.
    for i in 0..result.df.len() {
        if result.df[i].is_finite() && result.df[i] >= 1.0 && result.g2_statistic[i].is_finite() {
            let expected = chi2_sf(result.g2_statistic[i], result.df[i]);
            assert!(
                (result.g2_p_value[i] - expected).abs() < 1e-12,
                "item {i} mismatch: {} vs {}",
                result.g2_p_value[i],
                expected
            );
        }
    }
}

#[test]
fn sx2_g2_stays_finite_when_observed_success_rate_hits_zero() {
    let n_items = 20usize;
    let n_persons = 2000usize;
    let alpha = vec![0.0; n_items];
    let b = vec![0.0; n_items];
    let zeta = vec![0.0; n_items];
    let fid = vec![0usize; n_items];
    let y = vec![0.0_f64; n_persons * n_items];
    let observed = vec![true; n_persons * n_items];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let result = s_x2(
        &bank,
        &y,
        &observed,
        n_persons,
        &PriorSpec::standard(1),
        &SX2Config::default(),
        None,
    )
    .unwrap();
    // Reads crate-returned g2_statistic and kills the mutation where x*ln(x/y)
    // is implemented naively (log(0) * 0 => NaN).
    let finite: Vec<(f64, f64)> = result
        .g2_statistic
        .iter()
        .zip(&result.statistic)
        .filter_map(|(&g2, &x2)| {
            if g2.is_finite() && x2.is_finite() {
                Some((g2, x2))
            } else {
                None
            }
        })
        .collect();
    assert!(!finite.is_empty());
    assert!(finite.iter().all(|(g2, _)| *g2 >= 0.0));
    assert!(result
        .g2_p_value
        .iter()
        .zip(&result.df)
        .all(|(value, &df)| value.is_finite() || !df.is_finite() || df < 1.0));
}

#[test]
fn infit_outfit_rejects_wrong_theta_length() {
    let (alpha, b, zeta, fid, y, observed, _, xi) = toy_bank_data();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let short_theta = vec![0.0_f64; 3]; // not n_persons * n_dims
    let err = infit_outfit(&bank, &y, &observed, 2000, &short_theta, &xi)
        .err()
        .expect("expected an error");
    assert!(err.contains("theta/xi"), "got: {err}");
}

#[test]
fn person_diagnostics_reject_malformed_bank_and_non_dichotomous_responses() {
    let (alpha, b, zeta, fid, mut y, observed, theta, xi) = toy_bank_data();
    let malformed_bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid[..fid.len() - 1],
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    for err in [
        person_fit(
            &malformed_bank,
            &y,
            &observed,
            2000,
            &theta,
            &xi,
            &[],
            -1.645,
        )
        .err()
        .expect("expected malformed bank error"),
        infit_outfit(&malformed_bank, &y, &observed, 2000, &theta, &xi)
            .err()
            .expect("expected malformed bank error"),
    ] {
        assert!(err.contains("inconsistent lengths"), "got: {err}");
    }

    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    y[0] = 2.0;
    for err in [
        person_fit(&bank, &y, &observed, 2000, &theta, &xi, &[], -1.645)
            .err()
            .expect("expected dichotomous response error"),
        infit_outfit(&bank, &y, &observed, 2000, &theta, &xi)
            .err()
            .expect("expected dichotomous response error"),
    ] {
        assert!(err.contains("0 or 1"), "got: {err}");
    }
}

#[test]
fn person_fit_and_msq_finite_for_true_model() {
    let (alpha, b, zeta, fid, y, observed, _theta_true, _xi_true) = toy_bank_data();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    // designed usage: the Snijders correction applies to ESTIMATED scores
    let eap = crate::scoring::score_eap(
        &bank,
        &y,
        &observed,
        2000,
        &PriorSpec::standard(1),
        21,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    let pf = person_fit(
        &bank,
        &y,
        &observed,
        2000,
        &eap.theta_eap,
        &eap.xi_eap,
        &[],
        -1.645,
    )
    .unwrap();
    let finite = pf.lz_star.iter().filter(|v| v.is_finite()).count();
    assert!(finite > 1800);
    let flag_rate = pf.flagged.iter().filter(|&&f| f).count() as f64 / 2000.0;
    assert!(
        flag_rate < 0.12,
        "flag rate should approach the nominal 5%: {flag_rate}"
    );
    let msq = infit_outfit(&bank, &y, &observed, 2000, &eap.theta_eap, &eap.xi_eap).unwrap();
    let mean_infit: f64 = msq.infit.iter().sum::<f64>() / 20.0;
    assert!(
        (mean_infit - 1.0).abs() < 0.25,
        "infit should center near 1: {mean_infit}"
    );
}

#[test]
fn fitstats_public_boundaries_and_interaction_paths() {
    assert_eq!(at_least_tiny(0.0, 1e-9), 1e-9);
    assert_eq!(at_least_tiny(2.0, 1e-9), 2.0);
    assert!(gammainc_upper_reg(1.0, -1.0).is_nan());
    assert!(gammainc_upper_reg(0.0, 1.0).is_nan());
    assert!(ln_gamma(0.25).is_finite());
    assert!(chi2_sf(1.0, 0.0).is_nan());
    assert_eq!(benjamini_hochberg(&[f64::NAN], 0.05), vec![false]);
    assert!(erfc(-1.0) > 1.0);
    assert!(vuong_nonnested(&[0.0], &[0.0], 1, 1, false).is_err());

    let alpha = [0.0, 0.1, -0.1];
    let b = [-0.5, 0.0, 0.5];
    let zeta = [0.2, -0.1, 0.3];
    let factor = [0usize, 0, 0];
    let y = [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let observed = [true, false, true, true, true, true, false, false, false];
    let theta = [-0.5, 0.0, 0.5];
    let xi = [0.1, -0.2, 0.3];
    let prior = PriorSpec::standard(1);

    for model_type in [ModelType::Mls2plm, ModelType::Bifac2plm] {
        let bank = ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: 0.0,
            factor_id: &factor,
            model_type,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        let pf = person_fit(&bank, &y, &observed, 3, &theta, &xi, &theta, 100.0).unwrap();
        assert_eq!(pf.flagged.len(), 3);
        let msq = infit_outfit(&bank, &y, &observed, 3, &theta, &xi).unwrap();
        assert_eq!(msq.infit.len(), 3);
        let sx2 = s_x2(
            &bank,
            &y,
            &observed,
            3,
            &prior,
            &SX2Config {
                q_theta: 7,
                ..Default::default()
            },
            Some(&[0.0, 0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(sx2.statistic.len(), 3);
        let residual_y: Vec<f64> = (0..10)
            .flat_map(|person| {
                [
                    (person % 2) as f64,
                    ((person + 1) % 2) as f64,
                    (person % 3 == 0) as u8 as f64,
                ]
            })
            .collect();
        let residual_observed = vec![true; residual_y.len()];
        let residual_theta: Vec<f64> = (0..10).map(|person| person as f64 / 3.0 - 1.5).collect();
        let residual_xi: Vec<f64> = (0..10).map(|person| person as f64 / 10.0 - 0.5).collect();
        let residual = residual_item_fit(
            &bank,
            &residual_y,
            &residual_observed,
            10,
            &residual_theta,
            &residual_xi,
            2,
        )
        .unwrap();
        assert_eq!(residual.max_abs_z.len(), 3);
        let resampled =
            person_fit_resampling(&bank, &y, &observed, 3, &theta, &xi, &theta, 1, 0).unwrap();
        assert_eq!(resampled.len(), 3);
    }

    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: 0.0,
        factor_id: &factor,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let rasch_bank = ItemBank {
        model_type: ModelType::Mlsrm,
        ..bank
    };
    assert!(s_x2(
        &rasch_bank,
        &y,
        &observed,
        3,
        &prior,
        &SX2Config {
            q_theta: 7,
            min_expected: 100.0,
            ..Default::default()
        },
        None,
    )
    .is_ok());
    let all_zero = [0.0; 9];
    let all_observed = [true; 9];
    let sx2_extreme = s_x2(
        &bank,
        &all_zero,
        &all_observed,
        3,
        &prior,
        &SX2Config {
            q_theta: 7,
            ..Default::default()
        },
        None,
    )
    .unwrap();
    assert!(sx2_extreme.rms_residual.iter().all(|value| value.is_nan()));

    let two_alpha = [0.0, 0.0];
    let two_b = [0.0, 0.0];
    let two_zeta = [0.0, 0.0];
    let split_factor = [0usize, 1];
    let split_bank = ItemBank {
        alpha: &two_alpha,
        b: &two_b,
        zeta: &two_zeta,
        tau: 0.0,
        factor_id: &split_factor,
        model_type: ModelType::Mirt,
        n_dims: 2,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    assert!(s_x2(
        &split_bank,
        &[0.0, 1.0, 1.0, 0.0],
        &[true; 4],
        2,
        &PriorSpec::standard(2),
        &SX2Config {
            q_theta: 7,
            ..Default::default()
        },
        None,
    )
    .is_ok());

    let mut item_missing = observed;
    for p in 0..3 {
        item_missing[p * 3 + 2] = false;
    }
    let empty_item_msq = infit_outfit(&bank, &y, &item_missing, 3, &theta, &xi).unwrap();
    assert!(empty_item_msq.infit[2].is_nan() && empty_item_msq.outfit[2].is_nan());

    let long_y: Vec<f64> = (0..30).map(|index| (index % 2) as f64).collect();
    let long_observed = vec![true; 30];
    let long_theta: Vec<f64> = (0..10).map(|p| p as f64 / 5.0 - 1.0).collect();
    let long_xi: Vec<f64> = (0..10).map(|p| 0.1 * p as f64).collect();
    for model_type in [ModelType::Mls2plm, ModelType::Bifac2plm] {
        let interaction_bank = ItemBank { model_type, ..bank };
        let residual = residual_item_fit(
            &interaction_bank,
            &long_y,
            &long_observed,
            10,
            &long_theta,
            &long_xi,
            2,
        )
        .unwrap();
        assert!(residual.max_abs_z.iter().all(|value| value.is_finite()));
    }
    assert!(s_x2(
        &bank,
        &y[..2],
        &observed[..2],
        1,
        &prior,
        &Default::default(),
        None
    )
    .is_err());
    assert!(s_x2(
        &bank,
        &y,
        &observed,
        3,
        &prior,
        &Default::default(),
        Some(&[1.0])
    )
    .is_err());
    assert!(person_fit(&bank, &y[..2], &observed[..2], 1, &[0.0], &[0.0], &[], -1.0).is_err());
    assert!(person_fit(&bank, &y, &observed, 3, &[0.0], &xi, &[], -1.0).is_err());
    assert!(person_fit(&bank, &y, &observed, 3, &theta, &xi, &[0.0], -1.0).is_err());
    assert!(infit_outfit(&bank, &y[..2], &observed[..2], 1, &[0.0], &[0.0]).is_err());
    assert!(residual_item_fit(&bank, &y[..2], &observed[..2], 1, &[0.0], &[0.0], 2).is_err());
    assert!(residual_item_fit(&bank, &y, &observed, 3, &[0.0], &xi, 2).is_err());
    assert!(residual_item_fit(&bank, &y, &observed, 3, &theta, &xi, 1).is_err());
    assert!(person_fit_resampling(&bank, &y, &observed, 3, &theta, &xi, &[], 0, 1).is_err());
    assert!(person_fit_resampling(&bank, &y, &observed, 3, &theta, &xi, &[], 10_001, 1).is_err());
    assert!(
        person_fit_resampling(&bank, &y, &observed, usize::MAX, &theta, &xi, &[], 2, 1).is_err()
    );

    let mut nonfinite_theta = theta.clone();
    nonfinite_theta[0] = f64::NAN;
    assert!(person_fit(&bank, &y, &observed, 3, &nonfinite_theta, &xi, &[], -1.0).is_err());
    assert!(person_fit(&bank, &y, &observed, 3, &theta, &xi, &[f64::NAN; 3], -1.0).is_err());
    assert!(person_fit(&bank, &y, &observed, 3, &theta, &xi, &[], f64::NAN).is_err());
    assert!(person_fit_resampling(&bank, &[], &[], 0, &[], &[], &[], 2, 1).is_err());
    assert!(adjusted_chi2_pairs(
        &bank,
        &y[..2],
        &observed[..2],
        1,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 }
    )
    .is_err());
    let adjusted = adjusted_chi2_pairs(
        &bank,
        &y,
        &observed,
        3,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .unwrap();
    assert!(adjusted.ratio.iter().all(|value| value.is_nan()));

    assert!(dimensionality_residuals(&[0.0], 2, 1).is_err());
    assert!(dimensionality_residuals(&[], 0, 1).is_err());
    assert!(dimensionality_residuals(&[], 1, 0).is_err());
    assert!(dimensionality_residuals(&[f64::INFINITY], 1, 1).is_err());
    let sparse = dimensionality_residuals(&[f64::NAN, 0.0, f64::NAN, 0.0], 2, 2).unwrap();
    assert!(sparse.q3[0].is_nan());
    assert!(sparse.q3_max_abs.is_nan());
    assert!(sparse.q3_mean_abs.is_nan());
    let constant = dimensionality_residuals(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 3, 2).unwrap();
    assert!(constant.q3[0].is_nan());
    assert!(constant.q3_max_abs.is_nan());
    assert!(constant.q3_mean_abs.is_nan());
    let no_pairs = dimensionality_residuals(&[0.0, 1.0, 2.0], 3, 1).unwrap();
    assert!(no_pairs.gddm.is_nan());
    assert!(no_pairs.mean_abs_residual_cross_product.is_nan());

    let one_alpha = [0.0];
    let one_b = [0.0];
    let one_zeta = [0.0];
    let one_factor = [0usize];
    let one_bank = ItemBank {
        alpha: &one_alpha,
        b: &one_b,
        zeta: &one_zeta,
        tau: 0.0,
        factor_id: &one_factor,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    assert!(ld_indices(
        &one_bank,
        &[0.0],
        &[true],
        1,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .is_err());
    assert!(ld_indices(
        &bank,
        &[0.0],
        &[true],
        1,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .is_err());
    let ld_small = ld_indices(
        &bank,
        &y,
        &observed,
        3,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .unwrap();
    assert!(ld_small.x2_signed.iter().all(|value| value.is_nan()));

    let mut indefinite = [-1.0, 0.0, 0.0, -1.0];
    assert!(cholesky_lower(&mut indefinite, 2).is_err());
    let mut positive = [4.0, 2.0, 2.0, 3.0];
    cholesky_lower(&mut positive, 2).unwrap();
    let solved = chol_solve(&positive, 2, &[1.0, 2.0]);
    assert!(solved.iter().all(|value| value.is_finite()));
    assert_eq!(ncchi2_cdf(3.0, 2.0, 0.0), chi2_cdf(3.0, 2.0));
    assert!(ncchi2_cdf(f64::NAN, 2.0, 1.0).is_nan());
    assert_eq!(nc_lambda_for(0.0, 2.0, 0.95), 0.0);

    let other_b = [0.0, 0.1];
    let other_alpha = [0.0, 0.0];
    let other_zeta = [0.0, 0.0];
    let other_factor = [0usize, 0];
    let other_bank = ItemBank {
        alpha: &other_alpha,
        b: &other_b,
        zeta: &other_zeta,
        tau: 0.0,
        factor_id: &other_factor,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    assert!(tcc_drift(
        &bank,
        &other_bank,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
        0.1
    )
    .is_err());
    let spatial_bank = ItemBank {
        model_type: ModelType::Mls2plm,
        ..bank
    };
    let quadrature_error = match tcc_drift(
        &bank,
        &spatial_bank,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 7 },
        0.1,
    ) {
        Err(error) => error,
        Ok(_) => panic!("mismatched quadrature unexpectedly accepted"),
    };
    assert!(
        quadrature_error.contains("quadrature"),
        "{quadrature_error}"
    );

    let pm = crate::poly::PolyModel::Gpcm;
    assert!(poly_local_dependence(&[], None, 0, 1, 2, &[1.0], &[0.0], pm, 7).is_err());
    assert!(poly_local_dependence(&[], None, 0, 2, 1, &[1.0, 1.0], &[], pm, 7).is_err());
    assert!(poly_local_dependence(&[0], None, 1, 2, 2, &[1.0, 1.0], &[0.0, 0.0], pm, 7).is_err());
    assert!(poly_local_dependence(
        &[0, 0],
        Some(&[true]),
        1,
        2,
        2,
        &[1.0, 1.0],
        &[0.0, 0.0],
        pm,
        7,
    )
    .is_err());
    assert!(poly_local_dependence(&[0, 0], None, 1, 2, 2, &[1.0], &[0.0, 0.0], pm, 7).is_err());
    assert!(poly_local_dependence(&[0, 0], None, 1, 2, 2, &[1.0, 1.0], &[0.0], pm, 7).is_err());
    assert!(
        poly_local_dependence(&[0, 2], None, 1, 2, 2, &[1.0, 1.0], &[0.0, 0.0], pm, 7).is_err()
    );
    let masked_ld = poly_local_dependence(
        &[0, 99],
        Some(&[true, false]),
        1,
        2,
        2,
        &[1.0, 1.0],
        &[0.0, 0.0],
        pm,
        7,
    )
    .unwrap();
    // Reads crate output (`x2`). Kills the mutation that validates all y cells
    // without checking `observed`.
    assert!(masked_ld.x2[0].is_nan());
    assert!(poly_local_dependence(
        &[0, 99],
        Some(&[true, true]),
        1,
        2,
        2,
        &[1.0, 1.0],
        &[0.0, 0.0],
        pm,
        7,
    )
    .is_err());
    let poly_y: Vec<usize> = (0..20).flat_map(|p| [p % 3, (p + 1) % 3]).collect();
    for model in [crate::poly::PolyModel::Gpcm, crate::poly::PolyModel::Grm] {
        let result = poly_local_dependence(
            &poly_y,
            None,
            20,
            2,
            3,
            &[1.0, 0.8],
            &[-0.5, 0.5, -0.25, 0.75],
            model,
            7,
        )
        .unwrap();
        assert_eq!(result.pairs, vec![(0, 1)]);
        assert!(result.x2[0].is_finite());
    }
    let sparse_pair =
        poly_local_dependence(&[0, 1], None, 1, 2, 3, &[1.0, 1.0], &[0.0; 4], pm, 7).unwrap();
    assert!(sparse_pair.x2[0].is_nan());

    assert!(poly_m2(&[], None, 0, 2, 2, &[1.0; 2], &[0.0; 2], pm, 7).is_err());
    assert!(poly_m2(&[], None, 0, 3, 1, &[1.0; 3], &[], pm, 7).is_err());
    assert!(poly_m2(&[0], None, 1, 3, 2, &[1.0; 3], &[0.0; 3], pm, 7).is_err());
    assert!(poly_m2(
        &[0, 0, 0],
        Some(&[true]),
        1,
        3,
        2,
        &[1.0; 3],
        &[0.0; 3],
        pm,
        7,
    )
    .is_err());
    assert!(poly_m2(&[0, 0, 0], None, 1, 3, 2, &[1.0; 2], &[0.0; 3], pm, 7).is_err());
    assert!(poly_m2(&[0, 0, 0], None, 1, 3, 2, &[1.0; 3], &[0.0; 2], pm, 7).is_err());
    assert!(poly_m2(&[0, 0, 2], None, 1, 3, 2, &[1.0; 3], &[0.0; 3], pm, 7).is_err());
    let mut masked_y: Vec<usize> = (0..20)
        .flat_map(|p| [p % 2, (p / 2) % 2, (p / 3) % 2, (p / 5) % 2])
        .collect();
    let mut masked_obs = vec![true; masked_y.len()];
    masked_y[2] = 99;
    masked_obs[2] = false;
    let masked_poly_m2 = poly_m2(
        &masked_y,
        Some(&masked_obs),
        20,
        4,
        2,
        &[1.0; 4],
        &[0.0; 4],
        pm,
        7,
    )
    .unwrap();
    // Reads crate output (`n_complete`). Kills the mutation that validates all y
    // cells without checking `observed`.
    assert_eq!(masked_poly_m2.n_complete, 19);
    masked_obs[2] = true;
    assert!(poly_m2(
        &masked_y,
        Some(&masked_obs),
        20,
        4,
        2,
        &[1.0; 4],
        &[0.0; 4],
        pm,
        7
    )
    .is_err());
    assert!(poly_m2(&[], None, 0, 3, 2, &[1.0; 3], &[0.0; 3], pm, 7).is_err());
    assert!(poly_m2(&[], None, 0, 4, 2, &[1.0; 4], &[0.0; 4], pm, 7).is_err());
    let four_y: Vec<usize> = (0..12)
        .flat_map(|p| [p % 2, (p / 2) % 2, (p / 3) % 2, (p / 5) % 2])
        .collect();
    assert!(poly_m2(
        &four_y,
        None,
        12,
        4,
        2,
        &[1.0; 4],
        &[100.0; 4],
        crate::poly::PolyModel::Grm,
        7,
    )
    .is_err());
}

#[test]
fn person_fit_resampling_distinguishes_zero_and_one_seeds() {
    let probs = [0.25_f64, 0.5, 0.75];
    let alpha = [0.0; 3];
    let b = probs.map(|p| (p / (1.0 - p)).ln());
    let zeta = [0.0; 3];
    let factor = [0_usize; 3];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &factor,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let y = [0.0; 3];
    let observed = [true; 3];
    let theta = [0.0];
    let xi = [0.0];

    let seed_zero = person_fit_resampling(&bank, &y, &observed, 1, &theta, &xi, &[], 1, 0).unwrap();
    let seed_one = person_fit_resampling(&bank, &y, &observed, 1, &theta, &xi, &[], 1, 1).unwrap();

    assert_eq!(seed_zero, vec![1.0]);
    assert_eq!(seed_one, vec![0.5]);
}
