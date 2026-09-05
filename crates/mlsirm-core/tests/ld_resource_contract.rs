use mlsirm_core::fitstats::{
    ld_resource_preflight, LD_MAX_PAIR_OUTPUTS, LD_RESOURCE_CONTRACT_VERSION,
};
use mlsirm_core::nodes::XiRule;
use mlsirm_core::scoring::ItemBank;
use mlsirm_core::ModelType;

fn bank<'a>(
    alpha: &'a [f64],
    b: &'a [f64],
    zeta: &'a [f64],
    factor_id: &'a [usize],
    model_type: ModelType,
    latent_dim: usize,
) -> ItemBank<'a> {
    ItemBank {
        alpha,
        b,
        zeta,
        tau: 0.0,
        factor_id,
        model_type,
        n_dims: 1,
        latent_dim,
        eps_distance: 1e-8,
    }
}

#[test]
fn ld_resource_contract_is_versioned_and_rejects_pair_quadrature_before_allocation() {
    assert_eq!(LD_RESOURCE_CONTRACT_VERSION, "ld-resource-v1");

    let n_items = 1_000usize;
    let latent_dim = 2usize;
    let alpha = vec![0.0; n_items];
    let b = vec![0.0; n_items];
    let zeta = vec![0.0; n_items * latent_dim];
    let factor_id = vec![0usize; n_items];
    let bank = bank(
        &alpha,
        &b,
        &zeta,
        &factor_id,
        ModelType::Mls2plm,
        latent_dim,
    );

    let error = ld_resource_preflight(
        &bank,
        20,
        41,
        XiRule::GaussHermite { q_xi: 21 },
    )
    .expect_err("9,031,459,500 pair-quadrature cells must fail before node allocation");
    assert!(error.contains("pair-quadrature"), "unexpected error: {error}");
}

#[test]
fn ld_resource_contract_rejects_pair_output_and_pair_person_surfaces() {
    let output_items = 3_163usize;
    let alpha = vec![0.0; output_items];
    let b = vec![0.0; output_items];
    let zeta = vec![0.0; output_items];
    let factor_id = vec![0usize; output_items];
    let output_bank = bank(
        &alpha,
        &b,
        &zeta,
        &factor_id,
        ModelType::Mirt,
        1,
    );
    let error = ld_resource_preflight(
        &output_bank,
        1,
        7,
        XiRule::GaussHermite { q_xi: usize::MAX },
    )
    .expect_err("pair output above the canonical ceiling must fail");
    assert!(error.contains("pair output"), "unexpected error: {error}");
    assert_eq!(LD_MAX_PAIR_OUTPUTS, 5_000_000);

    let person_items = 100usize;
    let alpha = vec![0.0; person_items];
    let b = vec![0.0; person_items];
    let zeta = vec![0.0; person_items];
    let factor_id = vec![0usize; person_items];
    let person_bank = bank(
        &alpha,
        &b,
        &zeta,
        &factor_id,
        ModelType::Mirt,
        1,
    );
    let error = ld_resource_preflight(
        &person_bank,
        40_405,
        7,
        XiRule::GaussHermite { q_xi: usize::MAX },
    )
    .expect_err("pair-person work above the canonical ceiling must fail");
    assert!(error.contains("pair-person"), "unexpected error: {error}");
}

#[test]
fn ld_resource_contract_preserves_mirt_xi_nonuse() {
    let alpha = vec![0.0; 2];
    let b = vec![0.0; 2];
    let zeta = vec![0.0; 2];
    let factor_id = vec![0usize; 2];
    let bank = bank(&alpha, &b, &zeta, &factor_id, ModelType::Mirt, 1);

    let usage = ld_resource_preflight(
        &bank,
        20,
        7,
        XiRule::GaussHermite { q_xi: usize::MAX },
    )
    .expect("MIRT does not consume latent-space quadrature");
    assert_eq!(usage.probability_cells, 14);
    assert_eq!(usage.pair_outputs, 1);
    assert_eq!(usage.pair_person_cells, 20);
    assert_eq!(usage.pair_quadrature_cells, 7);
}
