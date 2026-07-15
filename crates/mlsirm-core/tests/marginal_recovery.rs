//! Recovery and contract tests for the marginal (MMLE-EM) estimator.

use mlsirm_core::marginal::{fit_marginal, MarginalConfig, PopulationSpec};
use mlsirm_core::{Device, ModelConfig, ModelType, PenaltyConfig};

struct Lcg(u64);
impl Lcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        sxy += (x[i] - mx) * (y[i] - my);
        sxx += (x[i] - mx).powi(2);
        syy += (y[i] - my).powi(2);
    }
    sxy / (sxx.sqrt() * syy.sqrt())
}

struct Sim {
    y: Vec<f64>,
    observed: Vec<bool>,
    factor_id: Vec<usize>,
    b_true: Vec<f64>,
    a_true: Vec<f64>,
    theta_true: Vec<f64>,
    zeta_true: Vec<f64>,
}

/// Interaction-adjusted easiness `b_i - gamma * E_xi[d(xi, zeta_i)]` — the
/// item quantity the distance model identifies (raw `b_i` is confounded with
/// the item's radius; cf. the adjusted summaries in the design-doc sources).
fn adjusted_easiness(
    b: &[f64],
    zeta: &[f64],
    gamma: f64,
    latent_dim: usize,
    rng: &mut Lcg,
) -> Vec<f64> {
    let n_items = b.len();
    let draws: Vec<f64> = (0..2000 * latent_dim).map(|_| rng.normal()).collect();
    (0..n_items)
        .map(|i| {
            let mut mean_d = 0.0;
            for s in 0..2000 {
                let mut d2 = 1e-8;
                for k in 0..latent_dim {
                    let diff = draws[s * latent_dim + k] - zeta[i * latent_dim + k];
                    d2 += diff * diff;
                }
                mean_d += d2.sqrt();
            }
            b[i] - gamma * mean_d / 2000.0
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn simulate(
    rng: &mut Lcg,
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
    latent_dim: usize,
    gamma: f64,
    group_shift: &[f64],
    group_id: &[usize],
    cluster_sd: f64,
    cluster_id: &[usize],
    n_clusters: usize,
) -> Sim {
    let factor_id: Vec<usize> = (0..n_items).map(|i| i % n_dims).collect();
    let b_true: Vec<f64> = (0..n_items).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
    let a_true: Vec<f64> = (0..n_items).map(|_| 0.8 + 0.8 * rng.next_f64()).collect();
    let zeta_true: Vec<f64> = (0..n_items * latent_dim).map(|_| rng.normal() * 0.8).collect();
    let u_true: Vec<f64> = (0..n_clusters).map(|_| rng.normal() * cluster_sd).collect();
    let mut y = vec![0.0_f64; n_persons * n_items];
    let observed = vec![true; n_persons * n_items];
    let mut theta_true = vec![0.0_f64; n_persons * n_dims];
    for p in 0..n_persons {
        let shift = if group_shift.is_empty() { 0.0 } else { group_shift[group_id[p]] };
        let u = if n_clusters > 0 { u_true[cluster_id[p]] } else { 0.0 };
        let xi_p: Vec<f64> = (0..latent_dim).map(|_| rng.normal()).collect();
        for d in 0..n_dims {
            theta_true[p * n_dims + d] = shift + u + rng.normal();
        }
        for i in 0..n_items {
            let d = factor_id[i];
            let mut dist2 = 1e-8;
            for k in 0..latent_dim {
                let diff = xi_p[k] - zeta_true[i * latent_dim + k];
                dist2 += diff * diff;
            }
            let eta = a_true[i] * theta_true[p * n_dims + d] + b_true[i] - gamma * dist2.sqrt();
            let prob = 1.0 / (1.0 + (-eta).exp());
            y[p * n_items + i] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
        }
    }
    Sim { y, observed, factor_id, b_true, a_true, theta_true, zeta_true }
}

fn small_cfg() -> MarginalConfig {
    MarginalConfig { q_theta: 15, q_xi: 7, q_u: 11, max_iter: 150, ..Default::default() }
}

fn assert_monotone(trace: &[f64]) {
    // Exact EM is monotone; with adaptive population nodes (multigroup /
    // multilevel updates move the quadrature grid) the quadrature
    // APPROXIMATION of the marginal can dip by discretization error, so allow
    // a small absolute slack.
    for w in trace.windows(2) {
        assert!(w[1] >= w[0] - 1e-3, "marginal loglik decreased: {} -> {}", w[0], w[1]);
    }
}

#[test]
fn recovers_mls2plm_single_population() {
    let mut rng = Lcg(2024);
    let (n_persons, n_items, n_dims, latent_dim) = (800usize, 16usize, 2usize, 2usize);
    let sim =
        simulate(&mut rng, n_persons, n_items, n_dims, latent_dim, 1.0, &[], &[], 0.0, &[], 0);
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Mls2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Single,
        &MarginalConfig { q_theta: 21, q_xi: 11, max_iter: 150, ..Default::default() },
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("fit should succeed");
    assert_monotone(&res.loglik_trace);
    let mut mc = Lcg(4242);
    let b_adj_true = adjusted_easiness(&sim.b_true, &sim.zeta_true, 1.0, latent_dim, &mut mc);
    let mut mc = Lcg(4242);
    let b_adj_est = adjusted_easiness(&res.b, &res.zeta, res.tau.exp(), latent_dim, &mut mc);
    let cb = corr(&b_adj_est, &b_adj_true);
    assert!(cb > 0.85, "adjusted easiness recovery too low: {cb}");
    let a_est: Vec<f64> = res.alpha.iter().map(|a| a.exp()).collect();
    let ca = corr(&a_est, &sim.a_true);
    assert!(ca > 0.55, "a recovery too low: {ca}");
    let theta_est_d0: Vec<f64> = (0..n_persons).map(|p| res.theta_eap[p * n_dims]).collect();
    let theta_true_d0: Vec<f64> = (0..n_persons).map(|p| sim.theta_true[p * n_dims]).collect();
    let ct = corr(&theta_est_d0, &theta_true_d0);
    assert!(ct > 0.6, "theta recovery too low: {ct}");
    assert!(res.tau.exp() > 0.3, "gamma should stay clearly positive");
    assert!(res.theta_sd.iter().all(|s| s.is_finite() && *s >= 0.0));
}

#[test]
fn recovers_multigroup_mean_shift() {
    let mut rng = Lcg(7);
    let (n_persons, n_items, n_dims, latent_dim) = (500usize, 12usize, 1usize, 1usize);
    let group_id: Vec<usize> = (0..n_persons).map(|p| p % 2).collect();
    let sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.8, &[0.0, 1.0], &group_id, 0.0, &[],
        0,
    );
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Multigroup { group_id, n_groups: 2 },
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("fit should succeed");
    assert!((res.mu[0] - 0.0).abs() < 1e-12, "reference group mean must stay pinned");
    assert!(res.mu[1] > 0.5 && res.mu[1] < 1.6, "group-2 mean should recover ~1.0, got {}", res.mu[1]);
    assert_monotone(&res.loglik_trace);
}

#[test]
fn recovers_multilevel_intercept_sd() {
    let mut rng = Lcg(99);
    let (n_persons, n_items, n_dims, latent_dim) = (600usize, 12usize, 1usize, 1usize);
    let n_clusters = 30usize;
    let cluster_id: Vec<usize> = (0..n_persons).map(|p| p % n_clusters).collect();
    let sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.8, &[], &[], 0.8, &cluster_id,
        n_clusters,
    );
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Ulsrm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Multilevel { cluster_id, n_clusters },
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("fit should succeed");
    assert!(
        res.sigma_u > 0.35 && res.sigma_u < 1.4,
        "sigma_u should recover ~0.8, got {}",
        res.sigma_u
    );
    assert_eq!(res.u_eap.len(), n_clusters);
    assert!(res.u_eap.iter().all(|u| u.is_finite()));
    assert_monotone(&res.loglik_trace);
}

#[test]
fn mirt_runs_without_latent_space() {
    let mut rng = Lcg(5);
    let (n_persons, n_items, n_dims, latent_dim) = (200usize, 8usize, 2usize, 2usize);
    let sim =
        simulate(&mut rng, n_persons, n_items, n_dims, latent_dim, 0.0, &[], &[], 0.0, &[], 0);
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Mirt,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Single,
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("fit should succeed");
    assert!(res.zeta.iter().all(|z| *z == 0.0), "MIRT must not move item positions");
    assert_monotone(&res.loglik_trace);
}

#[test]
fn tolerates_missing_and_all_missing_rows() {
    let mut rng = Lcg(13);
    let (n_persons, n_items, n_dims, latent_dim) = (150usize, 10usize, 1usize, 2usize);
    let mut sim =
        simulate(&mut rng, n_persons, n_items, n_dims, latent_dim, 1.0, &[], &[], 0.0, &[], 0);
    for p in 0..n_persons {
        for i in 0..n_items {
            if rng.next_f64() < 0.25 {
                sim.observed[p * n_items + i] = false;
            }
        }
    }
    for i in 0..n_items {
        sim.observed[i] = false; // person 0: all missing
    }
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Single,
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("fit should succeed");
    assert!(res.theta_eap[0].abs() < 1e-6, "all-missing person shrinks to prior mean");
    assert!(res.theta_eap.iter().all(|t| t.is_finite()));
    assert_monotone(&res.loglik_trace);
}

#[test]
fn rejects_invalid_inputs() {
    let config = ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 2,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let ok_y = vec![0.0, 1.0, 1.0, 0.0];
    let ok_obs = vec![true; 4];
    let base = MarginalConfig::default();
    let pen = PenaltyConfig::default();
    let single = PopulationSpec::Single;
    // wrong y length
    assert!(fit_marginal(&[0.0; 3], &ok_obs, &[0, 0], &config, &single, &base, &pen, Device::Cpu)
        .is_err());
    // bad factor id
    assert!(
        fit_marginal(&ok_y, &ok_obs, &[0, 5], &config, &single, &base, &pen, Device::Cpu).is_err()
    );
    // non-binary response
    assert!(fit_marginal(
        &[0.0, 2.0, 1.0, 0.0],
        &ok_obs,
        &[0, 0],
        &config,
        &single,
        &base,
        &pen,
        Device::Cpu
    )
    .is_err());
    // unsupported quadrature
    let bad_q = MarginalConfig { q_theta: 12, ..MarginalConfig::default() };
    assert!(
        fit_marginal(&ok_y, &ok_obs, &[0, 0], &config, &single, &bad_q, &pen, Device::Cpu)
            .is_err()
    );
    // bad group id
    assert!(fit_marginal(
        &ok_y,
        &ok_obs,
        &[0, 0],
        &config,
        &PopulationSpec::Multigroup { group_id: vec![0, 7], n_groups: 2 },
        &base,
        &pen,
        Device::Cpu
    )
    .is_err());
    // bad cluster id length
    assert!(fit_marginal(
        &ok_y,
        &ok_obs,
        &[0, 0],
        &config,
        &PopulationSpec::Multilevel { cluster_id: vec![0], n_clusters: 1 },
        &base,
        &pen,
        Device::Cpu
    )
    .is_err());
    // latent_dim too large for grid quadrature
    let big_k = ModelConfig { latent_dim: 4, ..config.clone() };
    assert!(
        fit_marginal(&ok_y, &ok_obs, &[0, 0], &big_k, &single, &base, &pen, Device::Cpu).is_err()
    );
}

#[test]
fn qmc_and_mc_rules_recover_like_gauss_hermite() {
    use mlsirm_core::marginal::XiRuleKind;
    let mut rng = Lcg(31);
    let (n_persons, n_items, n_dims, latent_dim) = (500usize, 14usize, 2usize, 2usize);
    let sim =
        simulate(&mut rng, n_persons, n_items, n_dims, latent_dim, 1.0, &[], &[], 0.0, &[], 0);
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Mls2plm,
        eps_distance: 1e-8,
    };
    let fit_with = |rule: XiRuleKind, points: usize| {
        fit_marginal(
            &sim.y,
            &sim.observed,
            &sim.factor_id,
            &config,
            &PopulationSpec::Single,
            &MarginalConfig {
                q_theta: 15,
                q_xi: 7,
                max_iter: 70,
                tol: 5e-2,
                xi_rule: rule,
                xi_points: points,
                xi_seed: 7,
                ..Default::default()
            },
            &PenaltyConfig::lsirm_prior(),
            Device::Cpu,
        )
        .expect("fit should succeed")
    };
    let gh = fit_with(XiRuleKind::GaussHermite, 0);
    let qmc = fit_with(XiRuleKind::Halton, 128);
    let mc = fit_with(XiRuleKind::MonteCarlo, 256);
    for (name, fit) in [("GH", &gh), ("QMC", &qmc), ("MC", &mc)] {
        let final_change = fit.loglik_trace[fit.loglik_trace.len() - 1]
            - fit.loglik_trace[fit.loglik_trace.len() - 2];
        assert!(
            fit.converged,
            "{name} did not converge in {} M-steps; final likelihood change={final_change}",
            fit.n_iter
        );
        assert!(
            (0.0..5e-2).contains(&final_change),
            "{name} final likelihood change {final_change} did not meet tolerance 0.05"
        );
        eprintln!(
            "[{name}] converged=true reason=tolerance_met n_iter={}/70 final_change={final_change} tolerance=0.05",
            fit.n_iter
        );
    }
    // the integration rule must not change the answer materially
    assert!(corr(&gh.b, &qmc.b) > 0.98, "QMC b diverges from GH: {}", corr(&gh.b, &qmc.b));
    assert!(corr(&gh.b, &mc.b) > 0.95, "MC b diverges from GH: {}", corr(&gh.b, &mc.b));
    assert!(
        (gh.tau.exp() - qmc.tau.exp()).abs() < 0.4,
        "gamma mismatch GH={} QMC={}",
        gh.tau.exp(),
        qmc.tau.exp()
    );
    assert_monotone(&gh.loglik_trace);
    // QMC/MC traces are deterministic too (fixed point sets), so still monotone
    assert_monotone(&qmc.loglik_trace);
    assert_monotone(&mc.loglik_trace);
}

#[test]
fn fipc_recovers_shifted_population_with_anchors() {
    use mlsirm_core::marginal::Anchors;
    let mut rng = Lcg(55);
    let (n_persons, n_items, n_dims, latent_dim) = (700usize, 12usize, 1usize, 1usize);
    // simulate a shifted population theta ~ N(0.8, 1) WITHOUT a latent-space
    // term: with gamma > 0 the raw b is not the identified anchor quantity
    // (it confounds with the item's map radius), so valid anchors require a
    // distance-free generating model here.
    let sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.0, &[0.8], &vec![0; n_persons],
        0.0, &[], 0,
    );
    // "old calibration": treat the first 6 items' TRUE parameters as anchors
    let mut fixed = vec![false; n_items];
    let mut anchor_alpha = vec![0.0_f64; n_items];
    let mut anchor_b = vec![0.0_f64; n_items];
    let anchor_zeta = vec![0.0_f64; n_items * latent_dim]; // unknown -> only used where fixed
    for i in 0..6 {
        fixed[i] = true;
        anchor_alpha[i] = sim.a_true[i].ln();
        anchor_b[i] = sim.b_true[i];
    }
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let anchors = Anchors {
        fixed: fixed.clone(),
        alpha: anchor_alpha.clone(),
        b: anchor_b.clone(),
        zeta: anchor_zeta,
        tau: Some(-30.0), // anchor calibration had no usable space; freeze gamma ~ 0
    };
    let res = mlsirm_core::marginal::fit_marginal_anchored(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::SingleFree,
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
        Some(&anchors),
    )
    .expect("FIPC fit should succeed");
    // anchored items must not move
    for i in 0..6 {
        assert_eq!(res.alpha[i], anchor_alpha[i], "anchored alpha moved");
        assert_eq!(res.b[i], anchor_b[i], "anchored b moved");
    }
    // the free population mean must absorb the shift
    assert!(
        res.mu[0] > 0.4 && res.mu[0] < 1.3,
        "FIPC population mean should recover ~0.8, got {}",
        res.mu[0]
    );
    assert_monotone(&res.loglik_trace);
}

#[test]
fn fipc_requires_anchors_for_free_population() {
    let config = ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &[0.0, 1.0, 1.0, 0.0],
        &[true; 4],
        &[0, 0],
        &config,
        &PopulationSpec::SingleFree,
        &MarginalConfig::default(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    );
    assert!(res.is_err(), "SingleFree without anchors must be rejected");
}

#[test]
fn fipc_rejects_unidentified_and_nonfinite_anchor_contracts() {
    use mlsirm_core::marginal::{fit_marginal_anchored, Anchors};
    let config = ModelConfig {
        n_persons: 2,
        n_items: 4,
        n_dims: 2,
        latent_dim: 1,
        model_type: ModelType::Mirt,
        eps_distance: 1e-8,
    };
    let y = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0];
    let factor_id = [0, 0, 1, 1];
    let mut anchors = Anchors {
        fixed: vec![true, true, false, false],
        alpha: vec![0.0; 4],
        b: vec![0.0; 4],
        zeta: vec![0.0; 4],
        tau: None,
    };
    let err = fit_marginal_anchored(
        &y,
        &[true; 8],
        &factor_id,
        &config,
        &PopulationSpec::SingleFree,
        &MarginalConfig::default(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
        Some(&anchors),
    )
    .unwrap_err();
    assert!(err.contains("at least two fixed anchor items"), "{err}");

    anchors.fixed.fill(true);
    anchors.tau = Some(f64::NAN);
    let err = fit_marginal_anchored(
        &y,
        &[true; 8],
        &factor_id,
        &config,
        &PopulationSpec::SingleFree,
        &MarginalConfig::default(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
        Some(&anchors),
    )
    .unwrap_err();
    assert!(err.contains("anchor tau must be finite"), "{err}");
}

#[test]
fn concurrent_calibration_two_forms_with_anchor_block() {
    // Hanson-Beguin common-item design: two groups, each sees its own unique
    // block plus a shared anchor block; one concurrent multigroup run.
    let mut rng = Lcg(77);
    let (n_persons, n_items, n_dims, latent_dim) = (800usize, 15usize, 1usize, 1usize);
    let group_id: Vec<usize> = (0..n_persons).map(|p| p % 2).collect();
    let mut sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.8, &[0.0, 0.7], &group_id, 0.0,
        &[], 0,
    );
    // items 0..5 unique to form A, 5..10 anchors, 10..15 unique to form B
    for p in 0..n_persons {
        for i in 0..n_items {
            let unique_a = i < 5;
            let unique_b = i >= 10;
            if (group_id[p] == 1 && unique_a) || (group_id[p] == 0 && unique_b) {
                sim.observed[p * n_items + i] = false;
            }
        }
    }
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Multigroup { group_id, n_groups: 2 },
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("concurrent calibration should succeed");
    assert!((res.mu[0]).abs() < 1e-12, "reference group stays pinned");
    assert!(
        res.mu[1] > 0.3 && res.mu[1] < 1.2,
        "concurrent run should recover the ~0.7 group shift, got {}",
        res.mu[1]
    );
    assert_monotone(&res.loglik_trace);
    // every item calibrated despite the structural missingness
    assert!(res.b.iter().all(|v| v.is_finite()));
}

#[test]
fn zero_inflation_recovers_mixing_weight() {
    let mut rng = Lcg(2027);
    let (n_persons, n_items, n_dims, latent_dim) = (800usize, 12usize, 1usize, 1usize);
    let mut sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.5, &[], &[], 0.0, &[], 0,
    );
    // structural zeros: 30% of persons produce all-zero patterns regardless
    let pi_true = 0.30;
    let n_zero = (n_persons as f64 * pi_true) as usize;
    for p in 0..n_zero {
        for i in 0..n_items {
            sim.y[p * n_items + i] = 0.0;
        }
    }
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let mcfg = MarginalConfig { zero_inflation: true, ..small_cfg() };
    let res = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Single,
        &mcfg,
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("ZI fit should succeed");
    assert!(
        res.pi_zero > 0.15 && res.pi_zero < 0.45,
        "pi should approach ~0.30 (injected zeros + natural all-zero patterns), got {}",
        res.pi_zero
    );
    // injected structural zeros carry high responsibility
    let mean_resp_zero: f64 =
        res.zero_responsibility[..n_zero].iter().sum::<f64>() / n_zero as f64;
    let mean_resp_rest: f64 = res.zero_responsibility[n_zero..].iter().sum::<f64>()
        / (n_persons - n_zero) as f64;
    assert!(
        mean_resp_zero > mean_resp_rest + 0.3,
        "structural zeros must get higher responsibility: {mean_resp_zero} vs {mean_resp_rest}"
    );
    assert_monotone(&res.loglik_trace);
    // without the mixture the fit runs unchanged and reports pi = 0
    let plain = fit_marginal(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Single,
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("plain fit should succeed");
    assert_eq!(plain.pi_zero, 0.0);
    assert!(
        res.loglik_trace.last().unwrap() > plain.loglik_trace.last().unwrap(),
        "the mixture must improve the marginal loglik on ZI data"
    );
}

#[test]
fn item_position_covariate_recovers_delta() {
    use mlsirm_core::marginal::{fit_marginal_full, ItemCovariate};
    let mut rng = Lcg(404);
    let (n_persons, n_items, n_dims, latent_dim) = (900usize, 12usize, 1usize, 1usize);
    let group_id: Vec<usize> = (0..n_persons).map(|p| p % 2).collect();
    // two booklets: item i sits at position i in booklet 0, reversed in booklet 1
    let mut w = vec![0.0_f64; 2 * n_items];
    for i in 0..n_items {
        w[i] = i as f64 / (n_items - 1) as f64;
        w[n_items + i] = (n_items - 1 - i) as f64 / (n_items - 1) as f64;
    }
    let delta_true = -0.8; // later positions get harder (fatigue effect)
    let mut sim = simulate(
        &mut rng, n_persons, n_items, n_dims, latent_dim, 0.0, &[0.0, 0.0], &group_id, 0.0,
        &[], 0,
    );
    // re-simulate responses with the position effect applied
    let mut rng2 = Lcg(405);
    for p in 0..n_persons {
        let booklet = group_id[p];
        for i in 0..n_items {
            let eta = sim.a_true[i] * sim.theta_true[p] + sim.b_true[i]
                + delta_true * w[booklet * n_items + i];
            let prob = 1.0 / (1.0 + (-eta).exp());
            sim.y[p * n_items + i] = if rng2.next_f64() < prob { 1.0 } else { 0.0 };
        }
    }
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let cov = ItemCovariate { w, init_delta: 0.0 };
    let res = fit_marginal_full(
        &sim.y,
        &sim.observed,
        &sim.factor_id,
        &config,
        &PopulationSpec::Multigroup { group_id, n_groups: 2 },
        &small_cfg(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
        None,
        Some(&cov),
    )
    .expect("covariate fit should succeed");
    assert!(
        (res.delta - delta_true).abs() < 0.35,
        "position coefficient should recover ~{delta_true}, got {}",
        res.delta
    );
    assert_monotone(&res.loglik_trace);
}

#[test]
fn covariate_guards() {
    use mlsirm_core::marginal::{fit_marginal_full, ItemCovariate};
    let config = ModelConfig {
        n_persons: 2,
        n_items: 2,
        n_dims: 1,
        latent_dim: 1,
        model_type: ModelType::Uls2plm,
        eps_distance: 1e-8,
    };
    let cov = ItemCovariate { w: vec![0.0, 1.0], init_delta: 0.0 };
    // single-context covariate without anchors: collinear with b -> rejected
    let res = fit_marginal_full(
        &[0.0, 1.0, 1.0, 0.0],
        &[true; 4],
        &[0, 0],
        &config,
        &PopulationSpec::Single,
        &MarginalConfig::default(),
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
        None,
        Some(&cov),
    );
    assert!(res.is_err());
}


#[test]
fn bifactor_recovers_general_loadings() {
    // dichotomous bifactor (Gibbons-Hedeker): specifics via simple structure,
    // general factor via the inner-product kind at latent_dim = 1
    let mut rng = Lcg(606);
    let (n_persons, n_items, n_dims, latent_dim) = (900usize, 12usize, 2usize, 1usize);
    let factor_id: Vec<usize> = (0..n_items).map(|i| i % n_dims).collect();
    let b_true: Vec<f64> = (0..n_items).map(|_| -1.0 + 2.0 * rng.next_f64()).collect();
    let lambda_true: Vec<f64> = (0..n_items).map(|_| 0.6 + 0.9 * rng.next_f64()).collect();
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let g = rng.normal();
        let th: Vec<f64> = (0..n_dims).map(|_| rng.normal()).collect();
        for i in 0..n_items {
            let eta = th[factor_id[i]] + b_true[i] + lambda_true[i] * g;
            let prob = 1.0 / (1.0 + (-eta).exp());
            y[p * n_items + i] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
        }
    }
    let observed = vec![true; n_persons * n_items];
    let config = ModelConfig {
        n_persons,
        n_items,
        n_dims,
        latent_dim,
        model_type: ModelType::Bifac2plm,
        eps_distance: 1e-8,
    };
    let res = fit_marginal(
        &y,
        &observed,
        &factor_id,
        &config,
        &PopulationSpec::Single,
        &MarginalConfig { q_theta: 15, q_xi: 15, max_iter: 150, ..Default::default() },
        &PenaltyConfig::lsirm_prior(),
        Device::Cpu,
    )
    .expect("bifactor fit should succeed");
    assert_monotone(&res.loglik_trace);
    // general-factor loadings recovered up to a global sign (fixed by the
    // alignment); check correlation with truth
    let lam: Vec<f64> = res.zeta.clone();
    let c = corr(&lam, &lambda_true);
    assert!(c.abs() > 0.6, "lambda recovery too low: {c}");
    // tau is not a free parameter for the inner kind
    assert!(res.tau < -20.0, "tau must stay inert for BIFAC2PLM: {}", res.tau);
}


#[test]
fn m2_calibration_and_local_dependence() {
    use mlsirm_core::fitstats::m2_rmsea2;
    use mlsirm_core::nodes::XiRule;
    use mlsirm_core::scoring::{ItemBank, PriorSpec};

    // unidimensional 2PL (MIRT kind: no latent-space term), fit by MMLE
    let mut rng = Lcg(9091);
    let (n_persons, n_items) = (2500usize, 12usize);
    let factor_id = vec![0usize; n_items];
    let slope: Vec<f64> = (0..n_items).map(|_| 0.8 + 0.7 * rng.next_f64()).collect();
    let b_true: Vec<f64> = (0..n_items).map(|i| -1.4 + 0.25 * i as f64).collect();
    let mut y = vec![0.0_f64; n_persons * n_items];
    for p in 0..n_persons {
        let th = rng.normal();
        for i in 0..n_items {
            let eta = slope[i] * th + b_true[i];
            let prob = 1.0 / (1.0 + (-eta).exp());
            y[p * n_items + i] = if rng.next_f64() < prob { 1.0 } else { 0.0 };
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
    let mcfg = MarginalConfig { q_theta: 21, max_iter: 300, ..Default::default() };
    let fit = |resp: &[f64]| {
        fit_marginal(
            resp,
            &observed,
            &factor_id,
            &config,
            &PopulationSpec::Single,
            &mcfg,
            &PenaltyConfig::default(),
            Device::Cpu,
        )
        .expect("fit should succeed")
    };
    let run_m2 = |resp: &[f64], res: &mlsirm_core::marginal::MarginalResult| {
        let bank = ItemBank {
            alpha: &res.alpha,
            b: &res.b,
            zeta: &res.zeta,
            tau: res.tau,
            factor_id: &factor_id,
            model_type: ModelType::Mirt,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-8,
        };
        m2_rmsea2(&bank, resp, &observed, n_persons, &PriorSpec::standard(1), 21, XiRule::GaussHermite { q_xi: 7 })
            .expect("m2 should succeed")
    };

    // well-specified: df = (12 + 66) - 24 = 54; RMSEA2 near zero
    let res = fit(&y);
    let m2 = run_m2(&y, &res);
    assert_eq!(m2.n_moments, 78);
    assert_eq!(m2.n_parameters, 24);
    assert_eq!(m2.df, 54.0);
    assert!(m2.rmsea2 < 0.03, "well-specified RMSEA2 too high: {}", m2.rmsea2);
    assert!(
        m2.rmsea2_ci_lower <= m2.rmsea2 + 1e-9 && m2.rmsea2 <= m2.rmsea2_ci_upper + 1e-9,
        "CI must bracket point estimate: [{}, {}] vs {}",
        m2.rmsea2_ci_lower,
        m2.rmsea2_ci_upper,
        m2.rmsea2
    );
    assert!(m2.srmsr < 0.05, "well-specified SRMSR too high: {}", m2.srmsr);

    // inject strong local dependence: item 1 becomes an exact copy of item 0
    let mut y_ld = y.clone();
    for p in 0..n_persons {
        y_ld[p * n_items + 1] = y_ld[p * n_items + 0];
    }
    let res_ld = fit(&y_ld);
    let m2_ld = run_m2(&y_ld, &res_ld);
    assert!(
        m2_ld.rmsea2 > 0.08,
        "local dependence should inflate RMSEA2: {}",
        m2_ld.rmsea2
    );
    assert!(m2_ld.m2 > m2.m2, "LD M2 must exceed well-specified M2");
    assert!(m2_ld.srmsr > m2.srmsr, "LD SRMSR must exceed well-specified SRMSR");
}
