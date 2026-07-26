use super::{
    fit_marginal, fit_marginal_anchored, fit_marginal_full, m_step_delta, m_step_items, m_step_tau,
    pca_align, validate, Anchors, Contexts, EStep, Grids, ItemCovariate, MarginalConfig,
    PopulationSpec, XiRuleKind,
};
use crate::{Device, ModelConfig, ModelType, PenaltyConfig};

#[test]
fn trace_endpoint_matches_returned_parameters_after_max_iter() {
    let n_persons = 8;
    let n_items = 3;
    let y = vec![
        0.0, 0.0, 0.0, // person 0
        0.0, 0.0, 1.0, // person 1
        0.0, 1.0, 0.0, // person 2
        0.0, 1.0, 1.0, // person 3
        1.0, 0.0, 0.0, // person 4
        1.0, 0.0, 1.0, // person 5
        1.0, 1.0, 0.0, // person 6
        1.0, 1.0, 1.0, // person 7
    ];
    let observed = vec![true; n_persons * n_items];
    let factor_id = vec![0; n_items];
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mirt,
        eps_distance: 1e-8,
    };
    let mcfg = MarginalConfig {
        q_theta: 7,
        q_xi: 7,
        q_u: 7,
        max_iter: 1,
        m_steps: 2,
        ..MarginalConfig::default()
    };
    let result = fit_marginal(
        &y,
        &observed,
        &factor_id,
        &config,
        &PopulationSpec::Single,
        &mcfg,
        &PenaltyConfig::default(),
        Device::Cpu,
    )
    .unwrap();
    let anchors = Anchors {
        fixed: vec![true; n_items],
        alpha: result.alpha.clone(),
        b: result.b.clone(),
        zeta: result.zeta.clone(),
        tau: Some(result.tau),
    };
    let reevaluated = fit_marginal_anchored(
        &y,
        &observed,
        &factor_id,
        &config,
        &PopulationSpec::Single,
        &mcfg,
        &PenaltyConfig::default(),
        Device::Cpu,
        Some(&anchors),
    )
    .unwrap();

    assert_eq!(result.n_iter, 1);
    assert!(
        (result.loglik_trace.last().unwrap() - reevaluated.loglik_trace[0]).abs() < 1e-10,
        "trace endpoint must be the likelihood of the returned parameters: {:?} vs {:?}",
        result.loglik_trace,
        reevaluated.loglik_trace
    );
}

fn tiny_config(model_type: ModelType) -> ModelConfig {
    ModelConfig {
        n_persons: 2,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        model_type,
        eps_distance: 1e-8,
    }
}

fn tiny_mcfg() -> MarginalConfig {
    MarginalConfig {
        q_theta: 7,
        q_xi: 7,
        q_u: 7,
        max_iter: 1,
        m_steps: 1,
        ..MarginalConfig::default()
    }
}

#[test]
fn validation_covers_every_shape_population_and_rule_guard() {
    let y = [0.0, 1.0];
    let observed = [true, true];
    let factor = [0];
    let config = tiny_config(ModelType::Mirt);
    let mcfg = tiny_mcfg();

    assert!(validate(&y, &observed, &[], &config, &PopulationSpec::Single, &mcfg).is_err());

    let mut bad = config.clone();
    bad.model_type = ModelType::Ulsrm;
    bad.n_dims = 2;
    assert!(validate(&y, &observed, &factor, &bad, &PopulationSpec::Single, &mcfg).is_err());

    bad = config.clone();
    bad.n_items = 0;
    bad.n_dims = 0;
    assert!(validate(&[], &[], &[], &bad, &PopulationSpec::Single, &mcfg).is_err());

    bad = config.clone();
    bad.latent_dim = 7;
    assert!(validate(&y, &observed, &factor, &bad, &PopulationSpec::Single, &mcfg).is_err());

    for eps in [0.0, f64::NAN, f64::INFINITY] {
        bad = config.clone();
        bad.eps_distance = eps;
        assert!(validate(&y, &observed, &factor, &bad, &PopulationSpec::Single, &mcfg).is_err());
    }

    for rule in [XiRuleKind::Halton, XiRuleKind::MonteCarlo] {
        let bad_rule = MarginalConfig {
            xi_rule: rule,
            xi_points: 0,
            ..mcfg
        };
        assert!(validate(
            &y,
            &observed,
            &factor,
            &config,
            &PopulationSpec::Single,
            &bad_rule,
        )
        .is_err());
    }

    assert!(validate(
        &y,
        &observed,
        &factor,
        &config,
        &PopulationSpec::Multigroup {
            group_id: vec![0],
            n_groups: 1,
        },
        &mcfg,
    )
    .is_err());
    assert!(validate(
        &y,
        &observed,
        &factor,
        &config,
        &PopulationSpec::Multilevel {
            cluster_id: vec![0, 0],
            n_clusters: 0,
        },
        &mcfg,
    )
    .is_err());
}

#[test]
fn anchored_and_covariate_contracts_cover_all_early_errors() {
    let y = [0.0, 1.0];
    let observed = [true, true];
    let factor = [0];
    let config = tiny_config(ModelType::Mlsrm);
    let mcfg = tiny_mcfg();
    let penalty = PenaltyConfig::default();
    let call = |pop: &PopulationSpec, anchors: Option<&Anchors>, cov: Option<&ItemCovariate>| {
        fit_marginal_full(
            &y,
            &observed,
            &factor,
            &config,
            pop,
            &mcfg,
            &penalty,
            Device::Cpu,
            anchors,
            cov,
        )
    };

    let wrong_shape = Anchors {
        fixed: vec![true],
        alpha: vec![],
        b: vec![0.0],
        zeta: vec![0.0],
        tau: None,
    };
    assert!(call(&PopulationSpec::Single, Some(&wrong_shape), None).is_err());

    let no_fixed = Anchors {
        fixed: vec![false],
        alpha: vec![0.0],
        b: vec![0.0],
        zeta: vec![0.0],
        tau: None,
    };
    assert!(call(&PopulationSpec::Single, Some(&no_fixed), None).is_err());

    let nonfinite = Anchors {
        fixed: vec![true],
        alpha: vec![f64::NAN],
        b: vec![0.0],
        zeta: vec![0.0],
        tau: None,
    };
    assert!(call(&PopulationSpec::Single, Some(&nonfinite), None).is_err());

    let cov = ItemCovariate {
        w: vec![0.0],
        init_delta: 0.0,
    };
    assert!(call(
        &PopulationSpec::Multilevel {
            cluster_id: vec![0, 0],
            n_clusters: 1,
        },
        None,
        Some(&cov),
    )
    .is_err());
    assert!(call(
        &PopulationSpec::Multigroup {
            group_id: vec![0, 1],
            n_groups: 2,
        },
        None,
        Some(&cov),
    )
    .is_err());
}

#[test]
fn small_fits_cover_device_zero_inflation_initialization_and_empty_groups() {
    let y = [0.0, 1.0];
    let observed = [true, true];
    let factor = [0];
    let penalty = PenaltyConfig::default();

    let mirt = tiny_config(ModelType::Mirt);
    let auto = fit_marginal(
        &y,
        &observed,
        &factor,
        &mirt,
        &PopulationSpec::Single,
        &tiny_mcfg(),
        &penalty,
        Device::Auto,
    )
    .unwrap();
    assert_eq!(auto.n_iter, 1);

    let zi_cfg = MarginalConfig {
        zero_inflation: true,
        ..tiny_mcfg()
    };
    for pop in [
        PopulationSpec::Multigroup {
            group_id: vec![0, 1],
            n_groups: 2,
        },
        PopulationSpec::Multilevel {
            cluster_id: vec![0, 0],
            n_clusters: 1,
        },
    ] {
        let fit = fit_marginal(
            &y,
            &observed,
            &factor,
            &mirt,
            &pop,
            &zi_cfg,
            &penalty,
            Device::Cpu,
        )
        .unwrap();
        assert_eq!(fit.zero_responsibility.len(), 2);
    }

    let empty_group = fit_marginal(
        &y,
        &observed,
        &factor,
        &mirt,
        &PopulationSpec::Multigroup {
            group_id: vec![0, 0],
            n_groups: 2,
        },
        &tiny_mcfg(),
        &penalty,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(empty_group.mu[1], 0.0);

    let all_missing = fit_marginal(
        &y,
        &[false, false],
        &factor,
        &mirt,
        &PopulationSpec::Single,
        &tiny_mcfg(),
        &penalty,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(all_missing.b[0], 0.0);

    let spatial_d3 = ModelConfig {
        latent_dim: 3,
        model_type: ModelType::Mlsrm,
        ..mirt.clone()
    };
    let d3 = fit_marginal(
        &y,
        &observed,
        &factor,
        &spatial_d3,
        &PopulationSpec::Single,
        &tiny_mcfg(),
        &penalty,
        Device::Cpu,
    )
    .unwrap();
    assert_eq!(d3.zeta.len(), 3);

    let anchors = Anchors {
        fixed: vec![true],
        alpha: vec![0.0],
        b: vec![0.0],
        zeta: vec![0.25],
        tau: Some(0.25),
    };
    let anchored = fit_marginal_anchored(
        &y,
        &observed,
        &factor,
        &tiny_config(ModelType::Mlsrm),
        &PopulationSpec::Single,
        &tiny_mcfg(),
        &penalty,
        Device::Cpu,
        Some(&anchors),
    )
    .unwrap();
    assert_eq!(anchored.tau, 0.25);
}

#[test]
fn marginal_numeric_noop_paths_are_stable() {
    let config = ModelConfig {
        n_persons: 1,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mlsrm,
        eps_distance: 1e-8,
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
        x_grid: vec![0.0],
        x_logw: vec![0.0],
        q_t: 1,
        n_x: 1,
    };
    let estep = EStep {
        nbar: vec![0.0],
        rbar: vec![0.0],
        mbar: vec![0.0],
        loglik: 0.0,
        zi_resp: Vec::new(),
        sum_e_v2: 0.0,
        cluster_post: Vec::new(),
    };
    let mut zero_penalty = PenaltyConfig::default();
    zero_penalty.lambda_b = 0.0;
    zero_penalty.lambda_alpha = 0.0;
    zero_penalty.lambda_zeta = 0.0;
    zero_penalty.lambda_tau = 0.0;

    let (mut alpha, mut b, mut zeta) = (vec![0.0], vec![0.0], vec![0.0]);
    m_step_items(
        &mut alpha,
        &mut b,
        &mut zeta,
        0.0,
        &estep,
        &ctx,
        &grids,
        &config,
        &[0],
        &zero_penalty,
        1,
        None,
        None,
    );
    assert_eq!((alpha[0], b[0], zeta[0]), (0.0, 0.0, 0.0));

    let mut tau = 0.0;
    m_step_tau(
        &alpha,
        &b,
        &zeta,
        &mut tau,
        &estep,
        &ctx,
        &grids,
        &config,
        &[0],
        &zero_penalty,
        None,
    );
    assert_eq!(tau, 0.0);

    let mut ridge_penalty = zero_penalty.clone();
    ridge_penalty.lambda_tau = 1.0;
    m_step_tau(
        &alpha,
        &b,
        &zeta,
        &mut tau,
        &estep,
        &ctx,
        &grids,
        &config,
        &[0],
        &ridge_penalty,
        None,
    );
    assert_eq!(tau, 0.0);

    let mut delta = 0.0;
    m_step_delta(
        &alpha,
        &b,
        &zeta,
        tau,
        &mut delta,
        &[1.0],
        &estep,
        &ctx,
        &grids,
        &config,
        &[0],
        &zero_penalty,
    );
    assert_eq!(delta, 0.0);

    let (mut positive, mut xi) = (vec![0.5], vec![0.25]);
    pca_align(&mut positive, &mut xi, 1, 1, 1);
    pca_align(&mut [], &mut [], 0, 0, 0);
    assert_eq!((positive[0], xi[0]), (0.5, 0.25));
}
