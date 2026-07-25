use super::*;
use crate::nodes::XiRule;
use crate::scoring::{score_eap, ItemBank, PriorSpec};
use crate::ModelType;

fn sim_bank(
    n_persons: usize,
    n_items: usize,
    seed: u64,
) -> (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<f64>,
    Vec<bool>,
) {
    let alpha = vec![0.0_f64; n_items];
    let b: Vec<f64> = (0..n_items)
        .map(|i| -1.2 + 2.4 * i as f64 / n_items as f64)
        .collect();
    let zeta = vec![0.0_f64; n_items];
    let fid = vec![0usize; n_items];
    let mut state = seed;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let u1: f64 = unif().max(1e-12);
        let u2: f64 = unif();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let eta: f64 = theta + b[i];
            if unif() < 1.0 / (1.0 + (-eta).exp()) {
                y[p * n_items + i] = 1.0;
            }
        }
    }
    (alpha, b, zeta, fid, y, vec![true; n_persons * n_items])
}

fn mk_bank<'a>(alpha: &'a [f64], b: &'a [f64], zeta: &'a [f64], fid: &'a [usize]) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: -30.0,
        factor_id: fid,
        model_type: ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    }
}

#[test]
fn residual_fit_and_adjusted_chi2_calibrate_on_true_model() {
    // Use a long test so the repository's plug-in EAP-bin approximation is
    // not dominated by EAP shrinkage.
    let (alpha, b, zeta, fid, y, observed) = sim_bank(1500, 40, 99);
    let bank = mk_bank(&alpha, &b, &zeta, &fid);
    let eap = score_eap(
        &bank,
        &y,
        &observed,
        1500,
        &PriorSpec::standard(1),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    let rf = residual_item_fit(&bank, &y, &observed, 1500, &eap.theta_eap, &eap.xi_eap, 8).unwrap();
    let finite = rf.max_abs_z.iter().filter(|v| v.is_finite()).count();
    assert!(finite >= 35);
    let flagged = rf.p_value.iter().filter(|&&p| p < 0.05).count();
    assert!(flagged <= 8, "true model should rarely flag: {flagged}");
    let adj = adjusted_chi2_pairs(
        &bank,
        &y,
        &observed,
        1500,
        &PriorSpec::standard(1),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert!(
        adj.mean_ratio < 3.0,
        "true-model mean adjusted ratio: {}",
        adj.mean_ratio
    );
}

#[test]
fn adjusted_chi2_rejects_nonbinary_data_and_marks_empty_summary_undefined() {
    let (alpha, b, zeta, fid, mut y, observed) = sim_bank(20, 2, 17);
    let bank = mk_bank(&alpha, &b, &zeta, &fid);
    let prior = PriorSpec::standard(1);
    y[0] = 2.0;
    let err = adjusted_chi2_pairs(
        &bank,
        &y,
        &observed,
        20,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .err()
    .expect("non-binary observed responses must be rejected");
    assert!(err.contains("0 or 1"), "unexpected error: {err}");

    let (_, _, _, _, sparse_y, sparse_observed) = sim_bank(19, 2, 18);
    let sparse = adjusted_chi2_pairs(
        &bank,
        &sparse_y,
        &sparse_observed,
        19,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
    )
    .unwrap();
    assert!(sparse.ratio.iter().all(|value| value.is_nan()));
    assert!(sparse.mean_ratio.is_nan());
    assert!(sparse.max_ratio.is_nan());
}

#[test]
fn residual_fit_rejects_inputs_that_can_hide_misfit() {
    let (alpha, b, zeta, fid, mut y, observed) = sim_bank(10, 1, 7);
    let bank = mk_bank(&alpha, &b, &zeta, &fid);
    let mut theta = vec![0.0; 10];
    let xi = vec![0.0; 10];

    theta[0] = f64::NAN;
    let err = residual_item_fit(&bank, &y, &observed, 10, &theta, &xi, 2)
        .err()
        .expect("non-finite EAP scores must be rejected");
    assert!(err.contains("finite"), "unexpected error: {err}");

    theta[0] = 0.0;
    y[0] = 2.0;
    let err = residual_item_fit(&bank, &y, &observed, 10, &theta, &xi, 2)
        .err()
        .expect("non-binary observed responses must be rejected");
    assert!(err.contains("0 or 1"), "unexpected error: {err}");

    y[0] = 0.0;
    let err = residual_item_fit(&bank, &y, &observed, 10, &theta, &xi, 3)
        .err()
        .expect("undersized bins must be rejected");
    assert!(err.contains("five persons"), "unexpected error: {err}");

    let err = residual_item_fit(&bank, &y, &observed, 10, &theta, &xi, usize::MAX)
        .err()
        .expect("overflowing bin work must be rejected");
    assert!(err.contains("overflows"), "unexpected error: {err}");
}

#[test]
fn resampling_person_fit_flags_reversed_pattern() {
    let (alpha, b, zeta, fid, mut y, observed) = sim_bank(60, 20, 5);
    // person 0: reversed responses (passes hard, fails easy) — aberrant
    for i in 0..20 {
        y[i] = if b[i] < 0.0 { 1.0 } else { 0.0 };
    }
    let bank = mk_bank(&alpha, &b, &zeta, &fid);
    let eap = score_eap(
        &bank,
        &y,
        &observed,
        60,
        &PriorSpec::standard(1),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    let pv = person_fit_resampling(
        &bank,
        &y,
        &observed,
        60,
        &eap.theta_eap,
        &eap.xi_eap,
        &[],
        200,
        11,
    )
    .unwrap();
    assert!(pv[0].is_finite());
    let median_rest = {
        let mut rest: Vec<f64> = (1..60).map(|p| pv[p]).filter(|v| v.is_finite()).collect();
        rest.sort_by(|a, b| a.partial_cmp(b).unwrap());
        rest[rest.len() / 2]
    };
    assert!(
        pv[0] < median_rest,
        "aberrant person must sit low in the bootstrap null: {} vs median {}",
        pv[0],
        median_rest
    );
}

#[test]
fn tcc_drift_isolates_the_shifted_item() {
    let (alpha, b, zeta, fid, _y, _obs) = sim_bank(10, 10, 1);
    let mut b_new = b.clone();
    b_new[4] += 1.0; // drift on item 4
    let bank_old = mk_bank(&alpha, &b, &zeta, &fid);
    let bank_new = mk_bank(&alpha, &b_new, &zeta, &fid);
    let res = tcc_drift(
        &bank_old,
        &bank_new,
        &PriorSpec::standard(1),
        21,
        XiRule::GaussHermite { q_xi: 7 },
        1e-3,
    )
    .unwrap();
    assert!(
        res.drifted.contains(&4),
        "shifted item must be flagged: {:?}",
        res.drifted
    );
    assert!(res.area_trace[0] > *res.area_trace.last().unwrap());
    assert_eq!(res.termination_reason, "threshold_met");
    assert_eq!(res.iterations, res.drifted.len());
    assert_eq!(res.max_iterations, 8);
}

#[test]
fn tcc_drift_rejects_invalid_thresholds_and_reports_the_item_floor() {
    let (alpha, b, zeta, fid, _y, _obs) = sim_bank(10, 6, 2);
    let b_new: Vec<f64> = b.iter().map(|value| value + 0.5).collect();
    let bank_old = mk_bank(&alpha, &b, &zeta, &fid);
    let bank_new = mk_bank(&alpha, &b_new, &zeta, &fid);
    let prior = PriorSpec::standard(1);

    for threshold in [f64::NAN, -1.0] {
        let err = tcc_drift(
            &bank_old,
            &bank_new,
            &prior,
            7,
            XiRule::GaussHermite { q_xi: 3 },
            threshold,
        )
        .err()
        .expect("invalid threshold must be rejected");
        assert!(
            err.contains("finite and non-negative"),
            "unexpected error: {err}"
        );
    }

    let res = tcc_drift(
        &bank_old,
        &bank_new,
        &prior,
        7,
        XiRule::GaussHermite { q_xi: 3 },
        0.0,
    )
    .unwrap();
    assert_eq!(res.termination_reason, "minimum_items_reached");
    assert_eq!(res.iterations, 4);
    assert_eq!(res.max_iterations, 4);
    assert_eq!(res.area_trace.len(), res.iterations + 1);
    assert!(*res.area_trace.last().unwrap() > 0.0);
}
