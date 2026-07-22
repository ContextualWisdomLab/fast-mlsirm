use super::*;
use crate::nodes::XiRule;

fn small_bank() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let alpha = vec![0.1, -0.1, 0.2, 0.0, 0.05, -0.05];
    let b = vec![0.4, -0.3, 0.1, -0.6, 0.2, 0.0];
    let zeta = vec![
        0.5, -0.4, -0.6, 0.3, 0.2, 0.7, -0.1, -0.5, 0.4, 0.4, -0.3, 0.1,
    ];
    let factor_id = vec![0, 1, 0, 1, 0, 1];
    (alpha, b, zeta, factor_id)
}

fn bank<'a>(
    alpha: &'a [f64],
    b: &'a [f64],
    zeta: &'a [f64],
    factor_id: &'a [usize],
) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: 0.0,
        factor_id,
        model_type: ModelType::Mls2plm,
        n_dims: 2,
        latent_dim: 2,
        eps_distance: 1e-8,
    }
}

#[test]
fn bank_information_rejects_overflowing_point_shapes() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);

    let err = bank_information_device(&bk, &[], &[], usize::MAX, crate::Device::Cpu).unwrap_err();

    assert_eq!(err, "n_points * n_dims overflows");
}

#[test]
fn default_eap_policy_matches_auto_device() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let prior = PriorSpec::standard(2);
    let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let observed = vec![true; y.len()];
    let rule = XiRule::GaussHermite { q_xi: 7 };

    let default = score_eap(&bk, &y, &observed, 1, &prior, 15, rule).unwrap();
    let auto =
        score_eap_device(&bk, &y, &observed, 1, &prior, 15, rule, crate::Device::Auto).unwrap();

    assert_eq!(default.theta_eap, auto.theta_eap);
    assert_eq!(default.theta_sd, auto.theta_sd);
    assert_eq!(default.xi_eap, auto.xi_eap);
    assert_eq!(default.loglik, auto.loglik);
}

#[test]
fn eap_map_agree_and_react_to_data() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let prior = PriorSpec::standard(2);
    // all-pass vs all-fail on dim 0 items (0, 2, 4)
    let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let observed = vec![true; 12];
    let eap = score_eap(
        &bk,
        &y,
        &observed,
        2,
        &prior,
        21,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert!(
        eap.theta_eap[0] > eap.theta_eap[2],
        "dim-0 pass > dim-0 fail"
    );
    let map = score_map(&bk, &y, &observed, 2, &prior, 50, 1e-6).unwrap();
    assert!(map.converged.iter().all(|&c| c));
    // EAP and MAP should agree loosely for these smooth posteriors
    for p in 0..2 {
        for d in 0..2 {
            let diff = (eap.theta_eap[p * 2 + d] - map.theta_map[p * 2 + d]).abs();
            assert!(diff < 0.6, "EAP/MAP disagree: {diff}");
        }
        assert!(map.theta_se[p * 2].is_finite() && map.theta_se[p * 2] > 0.0);
    }
}

#[test]
fn map_does_not_relax_the_requested_gradient_tolerance() {
    let alpha = [
        0.04400557738268765,
        -0.04623670215195566,
        0.2241479276551487,
        0.036715041003563896,
        -0.18748428060638883,
        0.12655826921831964,
        0.456400015795548,
        0.3314783370952347,
        -0.24630733253244738,
        -0.44289751486611834,
        -0.21814606188807326,
        0.014464092771535259,
    ];
    let b = [
        -2.3250307746388343,
        -0.21879166393254573,
        -1.2459109472530652,
        -0.7322673547034516,
        -0.5442589828573099,
        -0.31630015636915454,
        0.4116305363741328,
        1.0425133694426776,
        -0.12853466294403426,
        1.3664634705496859,
        -0.6651946734866135,
        0.3515100700930197,
    ];
    let zeta = [
        0.9034701816518086,
        0.09401229776087457,
        -0.7434992493538084,
        -0.9217253762584194,
        -0.45772582566733916,
        0.2201951234700494,
        -1.009618183538736,
        -0.20917557487171307,
        -0.15922500991447772,
        0.5408455846858077,
        0.2146591225063409,
        0.3553727090399214,
        -0.6538286094183394,
        -0.12961363369276946,
        0.7839754700613295,
        1.4934311452207607,
        -1.2590655321041202,
        1.5139237747390626,
        1.3458754237823045,
        0.7813114007004275,
        0.2644556303293035,
        -0.3139228145364278,
        1.4580206835369587,
        1.9602583164499647,
    ];
    let factor_id: Vec<usize> = (0..12).map(|i| i % 2).collect();
    let y = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: 0.9008174349330625,
        factor_id: &factor_id,
        model_type: ModelType::Mls2plm,
        n_dims: 2,
        latent_dim: 2,
        eps_distance: 1e-8,
    };
    let prior = PriorSpec {
        mean: vec![0.3, -0.2],
        sd: vec![1.2, 0.8],
    };

    let map = score_map(&bank, &y, &[true; 12], 1, &prior, 100, 1e-12).unwrap();
    assert!(
        !map.converged[0],
        "a failed line search must not relax tol=1e-12 to the internal 1e-4 scale"
    );
}

#[test]
fn map_theta_se_uses_the_full_distance_observed_information() {
    let alpha = [0.0];
    let b = [0.2];
    let zeta = [0.7];
    let factor_id = [0usize];
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: 0.0,
        factor_id: &factor_id,
        model_type: ModelType::Mls2plm,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 0.1,
    };
    let map = score_map(
        &bank,
        &[1.0],
        &[true],
        1,
        &PriorSpec::standard(1),
        100,
        1e-6,
    )
    .unwrap();
    let theta = map.theta_map[0];
    let xi = map.xi_map[0];
    let diff = xi - zeta[0];
    let dist = (bank.eps_distance + diff * diff).sqrt();
    let eta = theta + b[0] - dist;
    let probability = 1.0 / (1.0 + (-eta).exp());
    let residual = 1.0 - probability;
    let weight = probability * (1.0 - probability);
    let derivative = -diff / dist;
    let second_derivative = -(1.0 / dist - diff * diff / dist.powi(3));
    let info_tt = weight + 1.0;
    let info_tx = weight * derivative;
    let info_xx = weight * derivative * derivative - residual * second_derivative + 1.0;
    let expected_se = (info_xx / (info_tt * info_xx - info_tx * info_tx)).sqrt();

    assert!(map.converged[0]);
    assert!(
        (map.theta_se[0] - expected_se).abs() < 1e-8,
        "theta SE {} omitted nonlinear distance curvature; expected {expected_se}",
        map.theta_se[0]
    );
}

#[test]
fn prior_shift_moves_scores() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let empty_y = vec![0.0; 6];
    let none_obs = vec![false; 6];
    let base = score_eap_device(
        &bk,
        &empty_y,
        &none_obs,
        1,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
        crate::Device::Cpu,
    )
    .unwrap();
    assert!(base.theta_eap[0].abs() < 1e-9, "no data -> prior mean");
    let shifted_prior = PriorSpec {
        mean: vec![0.7, -0.2],
        sd: vec![1.0, 1.0],
    };
    let shifted = score_eap_device(
        &bk,
        &empty_y,
        &none_obs,
        1,
        &shifted_prior,
        15,
        XiRule::GaussHermite { q_xi: 7 },
        crate::Device::Cpu,
    )
    .unwrap();
    assert!((shifted.theta_eap[0] - 0.7).abs() < 1e-9);
    assert!((shifted.theta_eap[1] + 0.2).abs() < 1e-9);
}

#[test]
fn lord_wingersky_sums_to_one_and_matches_enumeration() {
    let probs = vec![0.3, 0.6, 0.2, 0.8, 0.5, 0.5];
    let f = lord_wingersky(&probs, 3, 2);
    for x in 0..2 {
        let total: f64 = (0..4).map(|r| f[r * 2 + x]).sum();
        assert!((total - 1.0).abs() < 1e-12);
    }
    // enumeration for node 0: p = (0.3, 0.2, 0.5)
    let (p1, p2, p3) = (0.3, 0.2, 0.5);
    let expect0 = (1.0 - p1) * (1.0 - p2) * (1.0 - p3);
    assert!((f[0] - expect0).abs() < 1e-12);
    let expect3 = p1 * p2 * p3;
    assert!((f[3 * 2] - expect3).abs() < 1e-12);
}

#[test]
fn eapsum_tables_are_monotone_in_score() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let tables = eapsum_tables(
        &bk,
        &PriorSpec::standard(2),
        21,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert_eq!(tables.len(), 2);
    for tab in &tables {
        assert_eq!(tab.eap.len(), tab.n_items_dim + 1);
        let total: f64 = tab.score_prob.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "score probs must sum to 1");
        for s in 1..tab.eap.len() {
            assert!(
                tab.eap[s] > tab.eap[s - 1] - 1e-9,
                "EAPsum must be nondecreasing in the summed score"
            );
        }
    }
}

#[test]
fn multilevel_marginal_prior_widens_sd() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let sigma_u = 0.8_f64;
    let marginal_prior = PriorSpec {
        mean: vec![0.0; 2],
        sd: vec![(1.0 + sigma_u * sigma_u).sqrt(); 2],
    };
    let t1 = eapsum_tables(
        &bk,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    let t2 = eapsum_tables(&bk, &marginal_prior, 15, XiRule::GaussHermite { q_xi: 7 }).unwrap();
    // wider prior -> more extreme conversion at the top score
    let top1 = *t1[0].eap.last().unwrap();
    let top2 = *t2[0].eap.last().unwrap();
    assert!(
        top2 > top1,
        "marginal multilevel prior should widen the scale"
    );
}

#[test]
fn rejects_bad_inputs() {
    let (alpha, b, zeta, fid) = small_bank();
    let bk = bank(&alpha, &b, &zeta, &fid);
    let prior = PriorSpec::standard(2);
    assert!(score_eap(
        &bk,
        &[0.0; 5],
        &[true; 5],
        1,
        &prior,
        21,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
    let bad_prior = PriorSpec {
        mean: vec![0.0],
        sd: vec![1.0],
    };
    assert!(score_eap(
        &bk,
        &[0.0; 6],
        &[true; 6],
        1,
        &bad_prior,
        21,
        XiRule::GaussHermite { q_xi: 7 }
    )
    .is_err());
    let neg_sd = PriorSpec {
        mean: vec![0.0; 2],
        sd: vec![1.0, -1.0],
    };
    assert!(eapsum_tables(&bk, &neg_sd, 21, XiRule::GaussHermite { q_xi: 7 }).is_err());
}

#[test]
fn scoring_public_boundaries_and_interaction_paths() {
    assert_eq!(lord_wingersky(&[], 0, 3), vec![1.0, 1.0, 1.0]);
    assert!(solve_sym(vec![0.0, 0.0, 0.0, 0.0], vec![1.0, 1.0], 2).is_none());
    let swapped = solve_sym(vec![1.0, 2.0, 3.0, 4.0], vec![1.0, 0.0], 2).unwrap();
    assert!(swapped.iter().all(|value| value.is_finite()));

    let alpha = [0.0, 0.2, -0.1];
    let b = [-0.5, 0.0, 0.5];
    let zeta = [0.2, -0.1, 0.3];
    let factor = [0usize, 0, 0];
    let y = [0.0, 1.0, 1.0];
    let observed = [true, false, true];
    let prior = PriorSpec::standard(1);
    for model_type in [ModelType::Mls2plm, ModelType::Bifac2plm, ModelType::Mirt] {
        let bk = ItemBank {
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
        let map = score_map(&bk, &y, &observed, 1, &prior, 10, 1e-6).unwrap();
        assert!(map.log_posterior[0].is_finite());
        let (item, test) = bank_information(&bk, &[0.25], &[0.1], 1).unwrap();
        assert!(item.iter().chain(&test).all(|value| value.is_finite()));
        let pv = plausible_values(
            &bk,
            &y,
            &observed,
            1,
            &prior,
            7,
            XiRule::GaussHermite { q_xi: 7 },
            2,
            0,
        )
        .unwrap();
        assert_eq!(pv.len(), 2);
    }

    let plain_bank = ItemBank {
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
    assert!(score_map(&plain_bank, &y, &observed, 1, &prior, 0, 1e-6).is_err());
    assert!(score_map(&plain_bank, &y, &observed, 1, &prior, 10, f64::NAN).is_err());
    assert!(cat_next_item(
        &plain_bank,
        &y[..2],
        &observed[..2],
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .is_err());

    // Finite but extreme calibration can underflow a summed-score cell to zero. The documented
    // prior fallback must remain finite and deterministic for that representational boundary.
    let extreme_b = [1e308];
    let one_alpha = [0.0];
    let one_zeta = [0.0];
    let one_factor = [0usize];
    let extreme_bank = ItemBank {
        alpha: &one_alpha,
        b: &extreme_b,
        zeta: &one_zeta,
        tau: 0.0,
        factor_id: &one_factor,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let extreme_table = eapsum_tables(
        &extreme_bank,
        &PriorSpec::standard(1),
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert_eq!(extreme_table[0].eap[0], 0.0);
    assert_eq!(extreme_table[0].sd[0], 1.0);

    let two_dim_factor = [0usize, 0, 0];
    let two_dim = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: 0.0,
        factor_id: &two_dim_factor,
        model_type: ModelType::Mirt,
        n_dims: 2,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    let empty_dimension = eapsum_tables(
        &two_dim,
        &PriorSpec::standard(2),
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert_eq!(empty_dimension[1].n_items_dim, 0);
    let cat = cat_next_item(
        &two_dim,
        &y,
        &[true; 3],
        &PriorSpec::standard(2),
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert!(cat.ranked_items.is_empty());

    let bad_mean = PriorSpec {
        mean: vec![f64::NAN],
        sd: vec![1.0],
    };
    assert!(score_eap(
        &ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: 0.0,
            factor_id: &factor,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        },
        &y,
        &observed,
        1,
        &bad_mean,
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .is_err());
    assert!(bank_information(
        &ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: 0.0,
            factor_id: &factor,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        },
        &[],
        &[],
        1,
    )
    .is_err());
    assert!(plausible_values(
        &ItemBank {
            alpha: &alpha,
            b: &b,
            zeta: &zeta,
            tau: 0.0,
            factor_id: &factor,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        },
        &y,
        &observed,
        1,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 7 },
        0,
        1,
    )
    .is_err());
    assert!(empirical_reliability(&[0.0], &[1.0], 1, 1).is_err());

    assert!(score_wle(&[], &[], &[], &[], &[], &[], 0, 6.0, 1e-6).is_err());
    assert!(score_wle(&[1.0], &[], &[0.0], &[1.0], &[0.0], &[true], 1, 6.0, 1e-6).is_err());
    assert!(score_wle(
        &[f64::NAN],
        &[0.0],
        &[0.0],
        &[1.0],
        &[0.0],
        &[true],
        1,
        6.0,
        1e-6,
    )
    .is_err());
    assert!(score_wle(
        &[1.0],
        &[0.0],
        &[0.5],
        &[0.5],
        &[0.0],
        &[true],
        1,
        6.0,
        1e-6
    )
    .is_err());
    assert!(score_wle(
        &[1.0],
        &[0.0],
        &[0.0],
        &[1.0],
        &[0.0],
        &[true],
        1,
        0.0,
        1e-6
    )
    .is_err());
    assert!(score_wle(&[1.0], &[0.0], &[0.0], &[1.0], &[0.0], &[true], 1, 6.0, 0.0).is_err());
    let no_data = score_wle(
        &[1.0],
        &[0.0],
        &[0.0],
        &[1.0],
        &[0.0],
        &[false],
        1,
        6.0,
        1e-6,
    )
    .unwrap();
    assert!(no_data.theta[0].is_nan() && no_data.boundary[0]);
}
