use super::{m_step_delta, Contexts, EStep, Grids};
use crate::{ModelConfig, ModelType, PenaltyConfig};

#[test]
fn bifactor_delta_step_uses_inner_product_predictor() {
    let mut delta = 0.0;
    let config = ModelConfig {
        n_persons: 100,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Bifac2plm,
        eps_distance: 1e-8,
    };
    let estep = EStep {
        nbar: vec![100.0],
        rbar: vec![50.0],
        mbar: vec![0.0],
        loglik: 0.0,
        zi_resp: Vec::new(),
        sum_e_v2: 0.0,
        cluster_post: Vec::new(),
    };
    let ctx = Contexts {
        n_ctx: 1,
        shift: vec![0.0],
        scale: vec![1.0],
        u_nodes: Vec::new(),
        u_logw: Vec::new(),
    };
    let grids = Grids {
        t_nodes: vec![0.0],
        t_logw: vec![0.0],
        x_grid: vec![2.0],
        x_logw: vec![0.0],
        q_t: 1,
        n_x: 1,
    };

    m_step_delta(
        &[0.0],
        &[0.0],
        &[2.0],
        -30.0,
        &mut delta,
        &[1.0],
        &estep,
        &ctx,
        &grids,
        &config,
        &[0],
        &PenaltyConfig::default(),
    );

    assert!(
        delta < -1.0,
        "the inner-product eta is 4 at delta=0, so a 50% success rate must move delta negative; got {delta}"
    );
}
