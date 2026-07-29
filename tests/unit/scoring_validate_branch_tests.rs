use super::*;
use crate::nodes::XiRule;

fn ok_bank<'a>(alpha: &'a [f64], b: &'a [f64], zeta: &'a [f64], fid: &'a [usize]) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: -30.0,
        factor_id: fid,
        model_type: crate::ModelType::Mirt,
        n_dims: 1,
        latent_dim: 1,
        eps_distance: 1e-8,
    }
}

#[test]
fn validate_bank_rejects_malformed_banks() {
    let y = vec![0.0; 3];
    let obs = vec![true; 3];
    let prior = PriorSpec::standard(1);
    let rule = XiRule::GaussHermite { q_xi: 7 };
    // inconsistent alpha length
    let (a, b, z, f) = (vec![0.0; 2], vec![0.0; 3], vec![0.0; 3], vec![0usize; 3]);
    assert!(score_eap(&ok_bank(&a, &b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
    // factor_id out of range (>= n_dims)
    let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![5usize, 0, 0]);
    assert!(score_eap(&ok_bank(&a, &b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
    // latent_dim zero
    let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 0], vec![0usize; 3]);
    let mut bk = ok_bank(&a, &b, &z, &f);
    bk.latent_dim = 0;
    assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
    // eps_distance non-positive
    let (a, b, z, f) = (vec![0.0; 3], vec![0.0; 3], vec![0.0; 3], vec![0usize; 3]);
    let mut bk = ok_bank(&a, &b, &z, &f);
    bk.eps_distance = 0.0;
    assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
    // y/observed length mismatch
    let bk = ok_bank(&a, &b, &z, &f);
    assert!(score_eap(&bk, &vec![0.0; 6], &vec![true; 6], 1, &prior, 7, rule).is_err());

    // Public Rust scoring must reject non-finite calibrated parameters rather
    // than returning an apparently successful result filled with NaNs.
    let mut bad_b = b.clone();
    bad_b[0] = f64::NAN;
    assert!(score_eap(&ok_bank(&a, &bad_b, &z, &f), &y, &obs, 1, &prior, 7, rule).is_err());
    let mut bk = ok_bank(&a, &b, &z, &f);
    bk.tau = f64::INFINITY;
    assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());
    let mut bk = ok_bank(&a, &b, &z, &f);
    bk.eps_distance = f64::NAN;
    assert!(score_eap(&bk, &y, &obs, 1, &prior, 7, rule).is_err());

    // Observed responses are dichotomous. NaN and other categories were
    // previously classified as zero by index_responses.
    for bad in [f64::NAN, f64::INFINITY, -1.0, 2.0] {
        let mut bad_y = y.clone();
        bad_y[0] = bad;
        assert!(score_eap(&ok_bank(&a, &b, &z, &f), &bad_y, &obs, 1, &prior, 7, rule).is_err());
    }

    // Adversarial dimensions must return an error instead of overflowing
    // n_persons * n_items in a debug-build panic.
    assert!(score_eap(
        &ok_bank(&a, &b, &z, &f),
        &[],
        &[],
        usize::MAX,
        &prior,
        7,
        rule
    )
    .is_err());

    let overflow_alpha = [0.0, 0.0];
    let overflow_b = [0.0, 0.0];
    let overflow_factor = [0usize, 0];
    let overflow_bank = ItemBank {
        alpha: &overflow_alpha,
        b: &overflow_b,
        zeta: &[],
        tau: 0.0,
        factor_id: &overflow_factor,
        model_type: crate::ModelType::Mirt,
        n_dims: 1,
        latent_dim: usize::MAX,
        eps_distance: 1e-8,
    };
    assert!(score_eap(&overflow_bank, &[], &[], 0, &prior, 7, rule).is_err());
    assert!(score_eap(&ok_bank(&a, &b, &z, &f), &y, &obs, 1, &prior, 3, rule).is_err());
}
