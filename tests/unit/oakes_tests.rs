use super::*;
use crate::marginal::{fit_marginal, MarginalConfig, PopulationSpec};
use crate::{Device, ModelType, PenaltyConfig};

#[test]
fn oakes_matches_central_difference_of_the_score() {
    // simulate a small 1PL-with-space fit, then check the Oakes assembly
    // against the full central difference of the marginal score, and the
    // SEs against 1/sqrt(n) scaling expectations.
    let mut state = 4242u64;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let (n_persons, n_items) = (400usize, 6usize);
    let factor_id = vec![0usize; n_items];
    let b_true: Vec<f64> = (0..n_items).map(|i| -1.0 + 0.4 * i as f64).collect();
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let u1: f64 = unif().max(1e-12);
        let u2: f64 = unif();
        let theta = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        for i in 0..n_items {
            let eta: f64 = theta + b_true[i];
            if unif() < 1.0 / (1.0 + (-eta).exp()) {
                y[p * n_items + i] = 1.0;
            }
        }
    }
    let observed = vec![true; n_persons * n_items];
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mirt,
        eps_distance: 1e-8,
    };
    let mcfg = MarginalConfig {
        q_theta: 15,
        q_xi: 7,
        max_iter: 80,
        ..Default::default()
    };
    let pen = PenaltyConfig::lsirm_prior();
    let fitted = fit_marginal(
        &y,
        &observed,
        &factor_id,
        &config,
        &PopulationSpec::Single,
        &mcfg,
        &pen,
        Device::Cpu,
    )
    .unwrap();
    let final_change = (fitted.loglik_trace[fitted.loglik_trace.len() - 1]
        - fitted.loglik_trace[fitted.loglik_trace.len() - 2])
        .abs();
    assert!(fitted.converged, "Oakes reference fit did not converge");
    assert!(fitted.n_iter < mcfg.max_iter, "fit exhausted max_iter");
    assert!(
        final_change < mcfg.tol,
        "final likelihood change {final_change} exceeds tolerance {}",
        mcfg.tol
    );
    assert!(fitted.loglik_trace.iter().all(|value| value.is_finite()));
    assert!(
        fitted
            .loglik_trace
            .windows(2)
            .all(|pair| pair[1] + 1e-10 >= pair[0]),
        "reference EM likelihood was not monotone"
    );
    eprintln!(
        "Oakes reference: converged=true reason=tolerance_met n_iter={}/{} final_change={} tolerance={}",
        fitted.n_iter, mcfg.max_iter, final_change, mcfg.tol
    );
    let res = observed_information_oakes(
        &y,
        &observed,
        &factor_id,
        &config,
        &PopulationSpec::Single,
        &mcfg,
        &pen,
        &fitted.alpha,
        &fitted.b,
        &fitted.zeta,
        fitted.tau,
        &fitted.mu,
        &fitted.sigma,
        fitted.sigma_u,
        1e-5,
    )
    .unwrap();
    // MIRT free-alpha: labels alternate alpha/b per item
    assert_eq!(res.labels.len(), 2 * n_items);
    assert!(res.se.iter().all(|s| s.is_finite() && *s > 0.0));
    // b SEs at n=400 for a 1PL-ish item live in the 0.05..0.5 band
    for (lab, se) in res.labels.iter().zip(&res.se) {
        if lab.starts_with("b:") {
            assert!((0.03..0.6).contains(se), "implausible SE for {lab}: {se}");
        }
    }
    // internal consistency: Oakes total equals the central FD of the
    // marginal score for a couple of probe coordinates
    let pv_probe = [1usize, 4usize];
    let pv = ParamVec {
        free_alpha: true,
        uses_space: false,
        tau_free: false,
        n_items,
        latent_dim: 1,
    };
    let (t_nodes, t_weights) = gh_rule(15).unwrap();
    let grids = Grids {
        t_nodes: t_nodes.to_vec(),
        t_logw: t_weights.iter().map(|w| w.ln()).collect(),
        x_grid: vec![0.0; 1],
        x_logw: vec![0.0],
        q_t: 15,
        n_x: 1,
    };
    let ctx = build_contexts(&PopulationSpec::Single, &[], &[], 0.0, 1, 15);
    let resp = index_responses(&y, &observed, n_persons, n_items);
    let xi0 = pv.pack(&fitted.alpha, &fitted.b, &fitted.zeta, fitted.tau);
    let score_at = |xv: &[f64]| -> Vec<f64> {
        let (a0, b0, z0, t0) = pv.unpack(xv);
        let tables = build_tables(&a0, &b0, &z0, t0, &config, &factor_id, &ctx, &grids);
        let counts = e_step(
            &tables,
            &resp,
            &factor_id,
            &config,
            &PopulationSpec::Single,
            &ctx,
            &grids,
        );
        q_gradient(&pv, xv, &counts, &ctx, &grids, &config, &factor_id, &pen)
    };
    for &j in &pv_probe {
        let hj = 1e-5 * (1.0 + xi0[j].abs());
        let mut xp = xi0.clone();
        xp[j] += hj;
        let mut xm = xi0.clone();
        xm[j] -= hj;
        let sp = score_at(&xp);
        let sm = score_at(&xm);
        for c in 0..pv.len() {
            let fd = -(sp[c] - sm[c]) / (2.0 * hj);
            let oakes = res.information[j * pv.len() + c];
            assert!(
                (fd - oakes).abs() < 1e-2 * (1.0 + fd.abs()),
                "Oakes[{j},{c}] = {oakes} vs FD {fd}"
            );
        }
    }
}

#[test]
fn inner_product_q_gradient_does_not_write_a_tau_slot() {
    let pv = ParamVec {
        free_alpha: true,
        uses_space: true,
        tau_free: false,
        n_items: 1,
        latent_dim: 1,
    };
    let counts = EStepCounts {
        nbar: vec![1.0],
        rbar: vec![0.5],
        mbar: vec![0.0],
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
        x_grid: vec![0.25],
        x_logw: vec![0.0],
        q_t: 1,
        n_x: 1,
    };
    let config = ModelConfig {
        n_persons: 1,
        n_items: 1,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Bifac2plm,
        eps_distance: 1e-8,
    };

    let gradient = q_gradient(
        &pv,
        &[0.0, 0.0, 0.1],
        &counts,
        &ctx,
        &grids,
        &config,
        &[0],
        &PenaltyConfig::lsirm_prior(),
    );

    assert_eq!(gradient.len(), pv.len());
    assert!(gradient.iter().all(|value| value.is_finite()));
}

#[test]
fn oakes_private_numeric_helpers_cover_all_model_shapes() {
    let pv = ParamVec {
        free_alpha: true,
        uses_space: true,
        tau_free: true,
        n_items: 1,
        latent_dim: 2,
    };
    let packed = pv.pack(&[0.2], &[-0.3], &[0.4, -0.5], 0.6);
    assert_eq!(packed, vec![0.2, -0.3, 0.4, -0.5, 0.6]);
    assert_eq!(
        pv.unpack(&packed),
        (vec![0.2], vec![-0.3], vec![0.4, -0.5], 0.6)
    );
    assert_eq!(
        pv.labels(),
        vec!["alpha:0", "b:0", "zeta:0:0", "zeta:0:1", "tau"]
    );
    assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-12);

    let swapped = invert(vec![0.0, 1.0, 1.0, 0.0], 2).unwrap();
    assert_eq!(swapped, vec![0.0, 1.0, 1.0, 0.0]);
    assert!(invert(vec![0.0, 0.0, 0.0, 0.0], 2).is_none());
    assert!(invert_information(vec![0.0; 4], 2).is_err());
    assert_eq!(invert_information(vec![1.0], 1).unwrap(), vec![1.0]);

    let ctx = Contexts {
        n_ctx: 1,
        shift: vec![0.0],
        scale: vec![1.0],
        u_nodes: Vec::new(),
        u_logw: Vec::new(),
    };
    let grids = Grids {
        t_nodes: vec![0.5],
        t_logw: vec![0.0],
        x_grid: vec![0.25, -0.5],
        x_logw: vec![0.0],
        q_t: 1,
        n_x: 1,
    };
    let config = ModelConfig {
        n_persons: 1,
        n_items: 1,
        n_dims: 1,
        latent_dim: 2,
        model_type: ModelType::Mls2plm,
        eps_distance: 1e-8,
    };
    let penalty = PenaltyConfig::lsirm_prior();
    let gradient = q_gradient(
        &pv,
        &packed,
        &EStepCounts {
            nbar: vec![1.0],
            rbar: vec![0.75],
            mbar: vec![0.0],
        },
        &ctx,
        &grids,
        &config,
        &[0],
        &penalty,
    );
    assert_eq!(gradient.len(), packed.len());
    assert!(gradient.iter().all(|value| value.is_finite()));
    let empty_gradient = q_gradient(
        &pv,
        &packed,
        &EStepCounts {
            nbar: vec![0.0],
            rbar: vec![0.0],
            mbar: vec![0.0],
        },
        &ctx,
        &grids,
        &config,
        &[0],
        &penalty,
    );
    assert!(empty_gradient.iter().all(|value| value.is_finite()));
}

#[test]
fn oakes_rejects_unsupported_modes_and_builds_every_xi_rule() {
    let y = [0.0, 1.0, 1.0, 0.0];
    let observed = [true; 4];
    let factor_id = [0, 0];
    let config = ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mlsrm,
        eps_distance: 1e-8,
    };
    let penalty = PenaltyConfig::lsirm_prior();
    let call = |mcfg: &MarginalConfig, h: f64| {
        observed_information_oakes(
            &y,
            &observed,
            &factor_id,
            &config,
            &PopulationSpec::Single,
            mcfg,
            &penalty,
            &[0.0; 2],
            &[0.0; 2],
            &[0.0; 2],
            -2.0,
            &[],
            &[],
            0.0,
            h,
        )
    };

    assert!(call(
        &MarginalConfig {
            zero_inflation: true,
            ..Default::default()
        },
        1e-5
    )
    .is_err());
    assert!(call(
        &MarginalConfig {
            q_theta: 9,
            ..Default::default()
        },
        1e-5
    )
    .is_err());
    assert!(call(&MarginalConfig::default(), 0.0).is_err());
    assert!(call(&MarginalConfig::default(), f64::NAN).is_err());
    for xi_rule in [
        XiRuleKind::GaussHermite,
        XiRuleKind::Halton,
        XiRuleKind::MonteCarlo,
    ] {
        let _ = call(
            &MarginalConfig {
                q_theta: 7,
                q_xi: 7,
                xi_points: 8,
                xi_seed: 0,
                xi_rule,
                ..Default::default()
            },
            1e-5,
        );
    }
}

#[test]
fn oakes_rejects_malformed_core_inputs_without_panicking() {
    let config = ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Mlsrm,
        eps_distance: 1e-8,
    };
    let call = |y: &[f64], alpha: &[f64], b: &[f64]| {
        observed_information_oakes(
            y,
            &[true; 4],
            &[0, 0],
            &config,
            &PopulationSpec::Single,
            &MarginalConfig::default(),
            &PenaltyConfig::lsirm_prior(),
            alpha,
            b,
            &[0.0; 2],
            -2.0,
            &[],
            &[],
            0.0,
            1e-5,
        )
    };

    assert!(call(&[0.0], &[0.0; 2], &[0.0; 2]).is_err());
    assert!(call(&[0.0; 4], &[0.0], &[0.0; 2]).is_err());
    assert!(call(&[0.0; 4], &[0.0; 2], &[0.0, f64::NAN]).is_err());
}
