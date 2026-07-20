use super::*;
use crate::nodes::XiRule;
use crate::ModelType;

fn bank_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let alpha = vec![0.2, -0.1, 0.4, 0.0, 0.3, -0.2, 0.1, 0.25];
    let b = vec![0.5, -0.5, 0.0, 1.0, -1.0, 0.3, -0.3, 0.8];
    let zeta = vec![0.0; 8];
    let factor_id = vec![0, 1, 0, 1, 0, 1, 0, 1];
    (alpha, b, zeta, factor_id)
}

#[test]
fn information_reduces_to_2pl_and_peaks_at_b() {
    // 4PL formula with c=0, d=1 equals a^2 P (1-P)
    let i1 = item_information_4pl(1.5, 0.4, 0.0, 1.0);
    assert!((i1 - 1.5f64 * 1.5 * 0.4 * 0.6).abs() < 1e-12);
    // guessing shrinks information (Magis 2013)
    let i3pl = item_information_4pl(1.5, 0.4, 0.2, 1.0);
    assert!(i3pl < i1);
    assert_eq!(item_information_4pl(1.5, 0.0, 0.0, 1.0), 0.0);
}

#[test]
fn cat_selects_informative_item_on_target_dim() {
    let (alpha, b, zeta, fid) = bank_fixture();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 2,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    // dim 0 already has two answers; dim 1 has none -> target dim 1
    let y = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let administered = vec![true, false, true, false, false, false, false, false];
    let step = cat_next_item(
        &bank,
        &y,
        &administered,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .unwrap();
    assert_eq!(step.target_dim, 1, "unmeasured dimension must be targeted");
    assert!(step
        .ranked_items
        .iter()
        .all(|&i| fid[i] == 1 && !administered[i]));
    // ranked by information: descending
    for w in step.ranked_info.windows(2) {
        assert!(w[0] >= w[1]);
    }
    let mut invalid_y = y.clone();
    invalid_y[0] = 2.0;
    assert!(cat_next_item(
        &bank,
        &invalid_y,
        &administered,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .is_err());
    invalid_y[0] = f64::NAN;
    assert!(cat_next_item(
        &bank,
        &invalid_y,
        &administered,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
    )
    .is_err());
}

#[test]
fn plausible_values_track_the_posterior() {
    let (alpha, b, zeta, fid) = bank_fixture();
    let bank = ItemBank {
        alpha: &alpha,
        b: &b,
        zeta: &zeta,
        tau: -30.0,
        factor_id: &fid,
        model_type: ModelType::Mirt,
        n_dims: 2,
        latent_dim: 1,
        eps_distance: 1e-8,
    };
    // person 0 passes everything on dim 0, person 1 fails everything
    let y = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let observed = vec![true; 16];
    let pv = plausible_values(
        &bank,
        &y,
        &observed,
        2,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
        200,
        7,
    )
    .unwrap();
    let mean_p0_d0: f64 = (0..200).map(|r| pv[(r) * 2]).sum::<f64>() / 200.0;
    let mean_p1_d0: f64 = (0..200).map(|r| pv[(200 + r) * 2]).sum::<f64>() / 200.0;
    assert!(
        mean_p0_d0 > mean_p1_d0 + 0.5,
        "PV means must separate pass-all from fail-all: {mean_p0_d0} vs {mean_p1_d0}"
    );
    // draws are reproducible
    let pv2 = plausible_values(
        &bank,
        &y,
        &observed,
        2,
        &PriorSpec::standard(2),
        15,
        XiRule::GaussHermite { q_xi: 7 },
        200,
        7,
    )
    .unwrap();
    assert_eq!(pv, pv2);
}
