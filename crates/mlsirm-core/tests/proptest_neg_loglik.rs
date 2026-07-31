//! Property-based (proptest) fuzz harness for the core numeric kernel.
//!
//! `neg_loglik_and_grad` is the hot path every backend funnels response data,
//! item->factor mappings and parameter vectors through. CodeGraph
//! (`codegraph explore "neg_loglik_and_grad config Params ModelConfig"`) shows
//! it as the single Rust entry point with the widest blast radius. It carries
//! documented length preconditions (`assert_eq!` on `y`, `factor_id`, `mask`);
//! this harness always *satisfies* those preconditions and then fuzzes the
//! numeric content, asserting the invariants that must hold for every well
//! formed problem instance:
//!
//!   * no panic / no arithmetic UB on arbitrary finite inputs,
//!   * every returned gradient vector has exactly the length the config implies,
//!   * with finite inputs the objective and log-likelihood are finite.
//!
//! proptest is MIT / Apache-2.0 licensed, so it is safe for this MIT crate. It
//! runs under the standard `cargo test`, so CI needs no extra toolchain.

use mlsirm_core::{ModelConfig, ModelType, Params, PenaltyConfig};
use proptest::prelude::*;

const MODEL_TYPES: [ModelType; 5] = [
    ModelType::Mirt,
    ModelType::Mls2plm,
    ModelType::Mlsrm,
    ModelType::Uls2plm,
    ModelType::Ulsrm,
];

/// A structurally-consistent problem instance: all vector lengths agree with
/// the config, so the kernel's own `assert_eq!` preconditions are met and we
/// are fuzzing behaviour, not argument plumbing.
#[derive(Debug, Clone)]
struct Instance {
    y: Vec<f64>,
    mask: Option<Vec<bool>>,
    factor_id: Vec<usize>,
    params: Params,
    config: ModelConfig,
    penalty: PenaltyConfig,
}

fn finite() -> impl Strategy<Value = f64> {
    // Bounded finite reals: exercises the numerics without trivially
    // overflowing exp()/softplus into +inf on every draw.
    -50.0f64..50.0f64
}

fn instance_strategy() -> impl Strategy<Value = Instance> {
    (1usize..6, 1usize..6, 1usize..4, 1usize..4, 0usize..5).prop_flat_map(
        |(n_persons, n_items, n_dims, latent_dim, model_idx)| {
            let model_type = MODEL_TYPES[model_idx];
            let config = ModelConfig {
                n_persons,
                n_items,
                n_dims,
                latent_dim,
                model_type,
                eps_distance: 1e-8,
            };

            let y_len = n_persons * n_items;
            let theta_len = n_persons * n_dims;
            let xi_len = n_persons * latent_dim;
            let zeta_len = n_items * latent_dim;

            (
                // Binary-ish responses, but allow arbitrary finite values too.
                prop::collection::vec(prop_oneof![Just(0.0f64), Just(1.0f64), finite()], y_len),
                prop::option::of(prop::collection::vec(any::<bool>(), y_len)),
                // factor_id must index theta's dimension: 0..n_dims.
                prop::collection::vec(0..n_dims, n_items),
                prop::collection::vec(finite(), theta_len),
                prop::collection::vec(finite(), n_items),
                prop::collection::vec(finite(), n_items),
                prop::collection::vec(finite(), xi_len),
                prop::collection::vec(finite(), zeta_len),
                finite(),
            )
                .prop_map(
                    move |(y, mask, factor_id, theta, alpha, b, xi, zeta, tau)| Instance {
                        y,
                        mask,
                        factor_id,
                        params: Params {
                            theta,
                            alpha,
                            b,
                            xi,
                            zeta,
                            tau,
                        },
                        config: config.clone(),
                        penalty: PenaltyConfig::default(),
                    },
                )
        },
    )
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn neg_loglik_and_grad_holds_invariants(inst in instance_strategy()) {
        let Instance { y, mask, factor_id, params, config, penalty } = inst;
        let mask_ref = mask.as_deref();

        let (objective, grad, loglik) =
            mlsirm_core::neg_loglik_and_grad(&y, mask_ref, &factor_id, &params, &config, &penalty);

        // Gradient shapes must exactly match the configured dimensions.
        prop_assert_eq!(grad.theta.len(), config.n_persons * config.n_dims);
        prop_assert_eq!(grad.alpha.len(), config.n_items);
        prop_assert_eq!(grad.b.len(), config.n_items);
        prop_assert_eq!(grad.xi.len(), config.n_persons * config.latent_dim);
        prop_assert_eq!(grad.zeta.len(), config.n_items * config.latent_dim);

        // With finite inputs the outputs stay finite (no NaN/inf leakage).
        prop_assert!(objective.is_finite(), "objective was not finite: {}", objective);
        prop_assert!(loglik.is_finite(), "loglik was not finite: {}", loglik);
        for g in grad.theta.iter().chain(&grad.alpha).chain(&grad.b).chain(&grad.xi).chain(&grad.zeta) {
            prop_assert!(g.is_finite(), "gradient component was not finite: {}", g);
        }
        prop_assert!(grad.tau.is_finite(), "grad.tau was not finite: {}", grad.tau);
    }
}
