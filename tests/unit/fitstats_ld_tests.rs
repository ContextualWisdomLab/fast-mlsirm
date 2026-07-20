use super::*;
use crate::nodes::XiRule;
use crate::scoring::{ItemBank, PriorSpec};
use crate::ModelType;

fn two_item_bank<'a>(
    alpha: &'a [f64],
    b: &'a [f64],
    zeta: &'a [f64],
    fid: &'a [usize],
) -> ItemBank<'a> {
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
fn ld_indices_reject_non_binary_observed_responses() {
    let alpha = vec![0.0; 2];
    let b = vec![0.0; 2];
    let zeta = vec![0.0; 2];
    let fid = vec![0usize; 2];
    let bank = two_item_bank(&alpha, &b, &zeta, &fid);
    let observed = vec![true; 40];

    for invalid in [2.0, f64::NAN] {
        let mut y = vec![0.0; 40];
        y[0] = invalid;
        assert!(
            ld_indices(
                &bank,
                &y,
                &observed,
                20,
                &PriorSpec::standard(1),
                7,
                XiRule::GaussHermite { q_xi: 7 },
            )
            .is_err(),
            "observed response {invalid:?} must be rejected"
        );
    }
}

#[test]
fn ld_indices_returns_error_for_malformed_prior() {
    let alpha = vec![0.0; 2];
    let b = vec![0.0; 2];
    let zeta = vec![0.0; 2];
    let fid = vec![0usize; 2];
    let bank = two_item_bank(&alpha, &b, &zeta, &fid);
    let y = vec![0.0; 40];
    let observed = vec![true; 40];
    let malformed = PriorSpec {
        mean: Vec::new(),
        sd: Vec::new(),
    };

    assert!(ld_indices(
        &bank,
        &y,
        &observed,
        20,
        &malformed,
        7,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .is_err());
}

#[test]
fn ld_indices_flag_a_dependent_pair() {
    // simulate 1PL data, then force item 1 to copy item 0 (max LD)
    let n_items = 6usize;
    let n_persons = 800usize;
    let alpha = vec![0.0; n_items];
    let b: Vec<f64> = (0..n_items).map(|i| -1.0 + 0.4 * i as f64).collect();
    let zeta = vec![0.0; n_items];
    let fid = vec![0usize; n_items];
    let mut state = 21u64;
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
        y[p * n_items + 1] = y[p * n_items]; // item 1 duplicates item 0
    }
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
    let res = ld_indices(
        &bank,
        &y,
        &observed,
        n_persons,
        &PriorSpec::standard(1),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    // pair (0,1) is the first upper-triangle entry
    assert!(
        res.x2_signed[0] > 50.0,
        "duplicated pair must show large positive LD X2: {}",
        res.x2_signed[0]
    );
    assert!(res.g2_signed[0] > 50.0);
    // an unrelated pair stays modest
    let pair_23 = (n_items - 1) + (n_items - 2) + 0; // (2,3) index in triangle
    assert!(res.x2_signed[pair_23].abs() < 50.0);
}
