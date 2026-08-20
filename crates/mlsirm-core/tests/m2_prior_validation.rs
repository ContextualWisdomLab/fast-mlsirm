use mlsirm_core::fitstats::m2_rmsea2_structured;
use mlsirm_core::nodes::XiRule;
use mlsirm_core::scoring::{ItemBank, PriorSpec};
use mlsirm_core::ModelType;

fn fixture<'a>(
    alpha: &'a [f64],
    b: &'a [f64],
    zeta: &'a [f64],
    factor_id: &'a [usize],
) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: -30.0,
        factor_id,
        model_type: ModelType::Mirt,
        n_dims: 2,
        latent_dim: 1,
        eps_distance: 1e-8,
    }
}

#[test]
fn structured_m2_rejects_prior_dimension_mismatch_before_quadrature() {
    let n_items = 8usize;
    let n_persons = 64usize;
    let alpha = vec![0.0; n_items];
    let b = vec![0.0; n_items];
    let zeta = vec![0.0; n_items];
    let factor_id: Vec<usize> = (0..n_items).map(|item| item % 2).collect();
    let bank = fixture(&alpha, &b, &zeta, &factor_id);
    let y = vec![0.0; n_persons * n_items];
    let observed = vec![true; y.len()];

    for prior in [
        PriorSpec {
            mean: vec![0.0],
            sd: vec![1.0, 1.0],
        },
        PriorSpec {
            mean: vec![0.0, 0.0],
            sd: vec![1.0],
        },
    ] {
        let error = m2_rmsea2_structured(
            &bank,
            &y,
            &observed,
            n_persons,
            &prior,
            11,
            XiRule::GaussHermite { q_xi: 1 },
            None,
            true,
            false,
        )
        .expect_err("mismatched prior dimensions must fail closed");

        assert_eq!(
            error,
            "prior mean/sd must have one entry per trait dimension"
        );
    }
}
