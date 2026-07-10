#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    Mirt,
    Mls2plm,
    Mlsrm,
    Uls2plm,
    Ulsrm,
}

#[derive(Clone, Debug)]
pub struct ModelConfig {
    pub n_persons: usize,
    pub n_items: usize,
    pub n_dims: usize,
    pub latent_dim: usize,
    pub model_type: ModelType,
    pub eps_distance: f64,
}

#[derive(Clone, Debug)]
pub struct PenaltyConfig {
    pub lambda_theta: f64,
    pub lambda_xi: f64,
    pub lambda_zeta: f64,
    pub lambda_b: f64,
    pub lambda_alpha: f64,
    pub lambda_tau: f64,
    pub mu_alpha: f64,
    pub mu_tau: f64,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            lambda_theta: 0.01,
            lambda_xi: 0.01,
            lambda_zeta: 0.01,
            lambda_b: 0.001,
            lambda_alpha: 0.001,
            lambda_tau: 0.001,
            mu_alpha: 0.0,
            mu_tau: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Params {
    pub theta: Vec<f64>,
    pub alpha: Vec<f64>,
    pub b: Vec<f64>,
    pub xi: Vec<f64>,
    pub zeta: Vec<f64>,
    pub tau: f64,
}

#[derive(Clone, Debug)]
pub struct Gradients {
    pub theta: Vec<f64>,
    pub alpha: Vec<f64>,
    pub b: Vec<f64>,
    pub xi: Vec<f64>,
    pub zeta: Vec<f64>,
    pub tau: f64,
}

pub fn neg_loglik_and_grad(
    y: &[f64],
    mask: Option<&[bool]>,
    factor_id: &[usize],
    params: &Params,
    config: &ModelConfig,
    penalty: &PenaltyConfig,
) -> (f64, Gradients, f64) {
    assert_eq!(y.len(), config.n_persons * config.n_items);
    assert_eq!(factor_id.len(), config.n_items);
    if let Some(m) = mask {
        assert_eq!(m.len(), y.len());
    }

    let free_alpha = !matches!(config.model_type, ModelType::Mlsrm | ModelType::Ulsrm);
    let uses_space = !matches!(config.model_type, ModelType::Mirt);
    let gamma = if uses_space { params.tau.exp() } else { 0.0 };

    let mut objective = 0.0;
    let mut grad = Gradients {
        theta: vec![0.0; config.n_persons * config.n_dims],
        alpha: vec![0.0; config.n_items],
        b: vec![0.0; config.n_items],
        xi: vec![0.0; config.n_persons * config.latent_dim],
        zeta: vec![0.0; config.n_items * config.latent_dim],
        tau: 0.0,
    };

    for p in 0..config.n_persons {
        for (i, &d) in factor_id.iter().enumerate().take(config.n_items) {
            let idx = p * config.n_items + i;
            if mask.is_some_and(|m| !m[idx]) {
                continue;
            }
            let alpha = if free_alpha { params.alpha[i] } else { 0.0 };
            let a = alpha.exp();
            let mut dist2 = config.eps_distance;
            for k in 0..config.latent_dim {
                let diff =
                    params.xi[p * config.latent_dim + k] - params.zeta[i * config.latent_dim + k];
                dist2 += diff * diff;
            }
            let r = if uses_space { dist2.sqrt() } else { 0.0 };
            let eta = a * params.theta[p * config.n_dims + d] + params.b[i] - gamma * r;
            let pi = sigmoid(eta);
            let response = y[idx];
            objective += softplus(eta) - response * eta;
            let e = pi - response;

            grad.b[i] += e;
            if free_alpha {
                grad.alpha[i] += e * a * params.theta[p * config.n_dims + d];
            }
            grad.theta[p * config.n_dims + d] += e * a;
            if uses_space {
                grad.tau += e * (-gamma * r);
                for k in 0..config.latent_dim {
                    let diff = params.xi[p * config.latent_dim + k]
                        - params.zeta[i * config.latent_dim + k];
                    let common = gamma * diff / r;
                    grad.xi[p * config.latent_dim + k] += e * (-common);
                    grad.zeta[i * config.latent_dim + k] += e * common;
                }
            }
        }
    }

    let loglik = -objective;
    objective += add_penalty(params, config, penalty, free_alpha, uses_space, &mut grad);
    (objective, grad, loglik)
}

pub fn initial_theta(
    y: &[f64],
    observed: &[bool],
    factor_id: &[usize],
    n_persons: usize,
    n_items: usize,
    n_dims: usize,
) -> Vec<f64> {
    assert_eq!(y.len(), n_persons * n_items);
    assert_eq!(observed.len(), y.len());
    assert_eq!(factor_id.len(), n_items);
    for &factor in factor_id {
        assert!(factor < n_dims);
    }

    let mut raw = vec![0.0; n_persons * n_dims];
    let mut counts = vec![0usize; n_persons * n_dims];
    for p in 0..n_persons {
        for (i, &d) in factor_id.iter().enumerate().take(n_items) {
            let response_idx = p * n_items + i;
            if observed[response_idx] {
                let theta_idx = p * n_dims + d;
                raw[theta_idx] += y[response_idx];
                counts[theta_idx] += 1;
            }
        }
    }

    for (value, count) in raw.iter_mut().zip(counts.iter()) {
        if *count > 0 {
            *value /= *count as f64;
        }
    }

    let mut theta = vec![0.0; n_persons * n_dims];
    for d in 0..n_dims {
        let mean = (0..n_persons).map(|p| raw[p * n_dims + d]).sum::<f64>() / n_persons as f64;
        let variance = (0..n_persons)
            .map(|p| {
                let delta = raw[p * n_dims + d] - mean;
                delta * delta
            })
            .sum::<f64>()
            / n_persons as f64;
        let sd = variance.sqrt();
        if sd.is_finite() && sd >= 1e-12 {
            for p in 0..n_persons {
                theta[p * n_dims + d] = (raw[p * n_dims + d] - mean) / sd;
            }
        }
    }
    theta
}

fn add_penalty(
    params: &Params,
    config: &ModelConfig,
    penalty: &PenaltyConfig,
    free_alpha: bool,
    uses_space: bool,
    grad: &mut Gradients,
) -> f64 {
    let mut value = 0.0;

    value += add_l2(&params.theta, penalty.lambda_theta, 0.0, &mut grad.theta);
    value += add_l2(&params.b, penalty.lambda_b, 0.0, &mut grad.b);
    if free_alpha {
        value += add_l2(
            &params.alpha,
            penalty.lambda_alpha,
            penalty.mu_alpha,
            &mut grad.alpha,
        );
    }
    if uses_space {
        value += add_l2(&params.xi, penalty.lambda_xi, 0.0, &mut grad.xi);
        value += add_l2(&params.zeta, penalty.lambda_zeta, 0.0, &mut grad.zeta);
        let tau_delta = params.tau - penalty.mu_tau;
        value += 0.5 * penalty.lambda_tau * tau_delta * tau_delta;
        grad.tau += penalty.lambda_tau * tau_delta;
    } else {
        debug_assert_eq!(config.model_type, ModelType::Mirt);
    }
    value
}

fn add_l2(values: &[f64], lambda: f64, center: f64, grad: &mut [f64]) -> f64 {
    let mut value = 0.0;
    for (idx, item) in values.iter().enumerate() {
        let delta = item - center;
        value += 0.5 * lambda * delta * delta;
        grad[idx] += lambda * delta;
    }
    value
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

fn softplus(x: f64) -> f64 {
    x.max(0.0) + (-x.abs()).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ModelConfig {
        ModelConfig {
            n_persons: 2,
            n_items: 2,
            n_dims: 1,
            latent_dim: 2,
            model_type: ModelType::Mls2plm,
            eps_distance: 1e-8,
        }
    }

    fn params() -> Params {
        Params {
            theta: vec![0.2, -0.4],
            alpha: vec![0.1, -0.2],
            b: vec![0.3, -0.1],
            xi: vec![0.1, 0.2, -0.2, 0.4],
            zeta: vec![0.0, -0.1, 0.3, -0.4],
            tau: 0.2,
        }
    }

    #[test]
    fn single_item_matches_manual_nll() {
        let cfg = ModelConfig {
            n_persons: 1,
            n_items: 1,
            n_dims: 1,
            latent_dim: 1,
            model_type: ModelType::Mls2plm,
            eps_distance: 1e-8,
        };
        let p = Params {
            theta: vec![0.5],
            alpha: vec![0.0],
            b: vec![0.1],
            xi: vec![0.2],
            zeta: vec![-0.3],
            tau: 0.0,
        };
        let penalty = PenaltyConfig {
            lambda_theta: 0.0,
            lambda_xi: 0.0,
            lambda_zeta: 0.0,
            lambda_b: 0.0,
            lambda_alpha: 0.0,
            lambda_tau: 0.0,
            mu_alpha: 0.0,
            mu_tau: 0.0,
        };
        let (got, _, _) = neg_loglik_and_grad(&[1.0], None, &[0], &p, &cfg, &penalty);
        let r = ((0.2_f64 - -0.3_f64).powi(2) + 1e-8).sqrt();
        let eta = 0.5 + 0.1 - r;
        let expected = softplus(eta) - eta;
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn gradient_matches_finite_difference_for_tau() {
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let (base, grad, _) = neg_loglik_and_grad(&y, None, &[0, 0], &p, &cfg, &penalty);

        let h = 1e-6;
        let mut plus = p.clone();
        plus.tau += h;
        let (obj_plus, _, _) = neg_loglik_and_grad(&y, None, &[0, 0], &plus, &cfg, &penalty);
        let finite_diff = (obj_plus - base) / h;
        assert!((finite_diff - grad.tau).abs() < 1e-5);
    }

    #[test]
    fn initial_theta_matches_standardized_group_means() {
        let y = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let observed = vec![true, true, false, true, true, true, false, true, true];
        let factor_id = vec![0, 1, 1];

        let theta = initial_theta(&y, &observed, &factor_id, 3, 3, 2);

        let sqrt2 = 2.0_f64.sqrt();
        let sqrt6 = 6.0_f64.sqrt();
        let expected = vec![sqrt2, -sqrt6 / 2.0, -sqrt2 / 2.0, sqrt6 / 2.0, -sqrt2 / 2.0, 0.0];
        for (got, want) in theta.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn initial_theta_leaves_empty_dimensions_at_zero() {
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let observed = vec![true, true, true, true];
        let factor_id = vec![0, 1];

        let theta = initial_theta(&y, &observed, &factor_id, 2, 2, 3);

        assert_eq!(theta[2], 0.0);
        assert_eq!(theta[5], 0.0);
    }

    #[test]
    fn mask_excludes_entries() {
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let mask = vec![true, true, true, false];

        let (objective, grad, loglik) =
            neg_loglik_and_grad(&y, Some(&mask), &[0, 0], &p, &cfg, &penalty);

        assert!(objective.is_finite());
        assert!(loglik.is_finite());
        assert_eq!(grad.theta.len(), cfg.n_persons * cfg.n_dims);
    }

    #[test]
    fn mirt_ignores_latent_space_terms() {
        let mut cfg = config();
        cfg.model_type = ModelType::Mirt;
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];

        let (objective, grad, loglik) = neg_loglik_and_grad(&y, None, &[0, 0], &p, &cfg, &penalty);

        assert!(objective.is_finite());
        assert!(loglik.is_finite());
        assert_eq!(grad.tau, 0.0);
        assert!(grad.xi.iter().all(|value| *value == 0.0));
        assert!(grad.zeta.iter().all(|value| *value == 0.0));
    }
}
