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
    // long test: the residual method's design regime (EAP shrinkage is
    // negligible); short tests belong to S-X2
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
}
