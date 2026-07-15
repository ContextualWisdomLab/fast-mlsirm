pub mod agreement;
pub mod cdm;
pub mod crm;
pub mod equating;
pub mod fitstats;
pub mod linking;
pub mod lltm;
pub mod marginal;
pub mod mixed;
pub mod mixture;
pub mod mirt;
pub mod mmle;
pub mod nodes;
pub mod poly;
pub mod poly_marginal;
pub mod oakes;
pub mod rsm;
pub mod rt;
pub mod rt_joint;
pub(crate) mod quadrature;
pub mod scoring;
pub mod testlet;

// cargo-llvm-cov runs in CPU-only CI against the repository-owned line-coverage
// baseline. Keep the hardware-backed wgpu module in normal builds, and cover
// the deterministic CPU fallback contract during coverage builds.
#[cfg(all(feature = "gpu", not(coverage)))]
mod gpu;
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) mod gpu_marginal;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    Mirt,
    Mls2plm,
    Mlsrm,
    Uls2plm,
    Ulsrm,
    /// Full-information dichotomous bifactor (Gibbons & Hedeker 1992; Cai,
    /// Yang & Hansen 2011) as the inner-product interaction kind:
    /// `eta = a_i theta_d(i) + b_i + dot(zeta_i, x)` with `x ~ MVN(0, I)` the
    /// general factor(s); at `latent_dim = 1`, `zeta_i` is the general-factor
    /// loading `lambda_i`. Marginal (MMLE) estimation only.
    ///
    /// # References
    ///
    /// Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
    /// analysis. *Psychometrika, 57*(3), 423–436.
    /// https://doi.org/10.1007/BF02295430
    ///
    /// Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
    /// item bifactor analysis. *Psychological Methods, 16*(3), 221–248.
    /// https://doi.org/10.1037/a0023350
    Bifac2plm,
}

/// The item-person interaction kind a model places on the latent-space axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InteractionKind {
    /// No interaction term (MIRT).
    None,
    /// Distance: `- exp(tau) * ||x - zeta_i||` (the LSIRM family).
    Distance,
    /// Inner product: `+ dot(zeta_i, x)` (bifactor / bilinear family).
    Inner,
}

/// Interaction kind of a model (shared by every eta evaluation site).
pub fn interaction_kind(model_type: ModelType) -> InteractionKind {
    match model_type {
        ModelType::Mirt => InteractionKind::None,
        ModelType::Bifac2plm => InteractionKind::Inner,
        _ => InteractionKind::Distance,
    }
}

/// Execution device for the likelihood/gradient hot path.
///
/// This is a sub-option of the Rust backend, not a separate compute backend
/// axis: `Gpu`/`Auto` run the wgpu GPGPU kernels when a GPU adapter is present
/// and otherwise fall back to the identical CPU implementation. The numerical
/// contract is the same for every variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    /// Always use the scalar CPU implementation.
    Cpu,
    /// Prefer the wgpu GPGPU path; fall back to CPU (with a warning) if no GPU.
    Gpu,
    /// Use the GPGPU path when a GPU is available, otherwise CPU. No warning.
    Auto,
}

impl Device {
    /// Parse a case-insensitive device string (`cpu` / `gpu` / `auto`).
    pub fn parse(name: &str) -> Option<Device> {
        match name.trim().to_ascii_lowercase().as_str() {
            "cpu" => Some(Device::Cpu),
            "gpu" => Some(Device::Gpu),
            "auto" => Some(Device::Auto),
            _ => None,
        }
    }
}

/// Whether a model frees the item discrimination and uses the latent space.
///
/// Shared between the CPU and GPGPU code paths so both agree on model algebra.
pub(crate) fn model_exec_flags(model_type: ModelType) -> (bool, bool) {
    let free_alpha = !matches!(model_type, ModelType::Mlsrm | ModelType::Ulsrm);
    let uses_space = !matches!(model_type, ModelType::Mirt);
    (free_alpha, uses_space)
}

/// Guard for numeric paths that only implement the distance kind (the JML
/// objective and its GPU kernels): `Bifac2plm` is marginal-only.
pub(crate) fn assert_distance_kind(model_type: ModelType) {
    debug_assert!(
        !matches!(model_type, ModelType::Bifac2plm),
        "BIFAC2PLM is supported by the marginal estimator only"
    );
}

/// Compute the negative log-likelihood and gradients on the requested device.
///
/// `Device::Cpu` runs the scalar reference implementation. `Device::Gpu` and
/// `Device::Auto` attempt the wgpu GPGPU kernels and transparently fall back to
/// the CPU implementation when no compatible GPU adapter is available (for
/// example in CI), so the call never fails for lack of a GPU.
pub fn neg_loglik_and_grad_device(
    device: Device,
    y: &[f64],
    mask: Option<&[bool]>,
    factor_id: &[usize],
    params: &Params,
    config: &ModelConfig,
    penalty: &PenaltyConfig,
) -> (f64, Gradients, f64) {
    match device {
        Device::Cpu => neg_loglik_and_grad(y, mask, factor_id, params, config, penalty),
        Device::Gpu | Device::Auto => {
            #[cfg(all(feature = "gpu", not(coverage)))]
            {
                let gpu_result =
                    gpu::neg_loglik_and_grad_gpu(y, mask, factor_id, params, config, penalty);
                finish_device(
                    device, gpu_result, y, mask, factor_id, params, config, penalty,
                )
            }
            #[cfg(any(not(feature = "gpu"), coverage))]
            neg_loglik_and_grad(y, mask, factor_id, params, config, penalty)
        }
    }
}

/// Resolve the outcome of a GPU device request: use the GPU result when the
/// adapter produced one, otherwise warn (only when the GPU was explicitly
/// requested, not for `Auto`) and fall back to the scalar CPU path.
///
/// This is deliberately split out of [`neg_loglik_and_grad_device`] so that both
/// the GPU-succeeded and GPU-unavailable branches are exercised by unit tests on
/// every machine, regardless of whether the host running the tests actually has
/// a usable GPU adapter (a GPU-equipped host only ever takes the success branch,
/// a GPU-less host only ever takes the fallback branch).
#[cfg(all(feature = "gpu", not(coverage)))]
#[allow(clippy::too_many_arguments)]
fn finish_device(
    device: Device,
    gpu_result: Option<(f64, Gradients, f64)>,
    y: &[f64],
    mask: Option<&[bool]>,
    factor_id: &[usize],
    params: &Params,
    config: &ModelConfig,
    penalty: &PenaltyConfig,
) -> (f64, Gradients, f64) {
    match gpu_result {
        Some(result) => result,
        None => {
            if matches!(device, Device::Gpu) {
                eprintln!(
                    "fast-mlsirm: GPU device requested but no usable GPU adapter was found; \
                     falling back to the CPU implementation."
                );
            }
            neg_loglik_and_grad(y, mask, factor_id, params, config, penalty)
        }
    }
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

impl PenaltyConfig {
    /// MAP penalties equal to the default LSIRM priors of Jeon et al. (2021)
    /// and the `lsirm12pl` package: `beta_i ~ N(0, 4)`, `log alpha_i ~
    /// N(0.5, 1)`, `zeta_i ~ MVN(0, I)`, `log gamma ~ N(0.5, 1)`. Used by the
    /// marginal (MMLE) estimator, where the person-side penalties are moot
    /// (persons are integrated out) and the item-side priors prevent slope
    /// collapse and latent-space blow-up on sparse items.
    pub fn lsirm_prior() -> Self {
        Self {
            lambda_theta: 0.0,
            lambda_xi: 0.0,
            lambda_zeta: 1.0,
            lambda_b: 0.25,
            lambda_alpha: 1.0,
            lambda_tau: 1.0,
            mu_alpha: 0.5,
            mu_tau: 0.5,
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
    assert_distance_kind(config.model_type);
    assert_eq!(y.len(), config.n_persons * config.n_items);
    assert_eq!(factor_id.len(), config.n_items);
    if let Some(m) = mask {
        assert_eq!(m.len(), y.len());
    }

    let (free_alpha, uses_space) = model_exec_flags(config.model_type);
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

pub(crate) fn add_penalty(
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
    fn device_parse_accepts_known_names() {
        assert_eq!(Device::parse("cpu"), Some(Device::Cpu));
        assert_eq!(Device::parse("GPU"), Some(Device::Gpu));
        assert_eq!(Device::parse(" Auto "), Some(Device::Auto));
        assert_eq!(Device::parse("cuda"), None);
    }

    #[test]
    fn device_auto_matches_cpu_on_fallback() {
        // On machines/CI without a GPU adapter, Auto/Gpu must fall back to the
        // CPU path and reproduce it bit-for-bit. When a GPU is present the f32
        // kernels are exercised instead and this asserts close (not exact)
        // agreement, which is the guarantee we make for the GPGPU path.
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let mask = vec![true, true, true, false];

        let (cpu_obj, cpu_grad, cpu_ll) =
            neg_loglik_and_grad_device(Device::Cpu, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
        for device in [Device::Auto, Device::Gpu] {
            let (obj, grad, ll) =
                neg_loglik_and_grad_device(device, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
            assert!(
                (obj - cpu_obj).abs() < 1e-4,
                "objective mismatch for {device:?}"
            );
            assert!((ll - cpu_ll).abs() < 1e-4, "loglik mismatch for {device:?}");
            assert!((grad.tau - cpu_grad.tau).abs() < 1e-4);
            for (a, b) in grad.theta.iter().zip(&cpu_grad.theta) {
                assert!((a - b).abs() < 1e-4);
            }
            for (a, b) in grad.xi.iter().zip(&cpu_grad.xi) {
                assert!((a - b).abs() < 1e-4);
            }
            for (a, b) in grad.zeta.iter().zip(&cpu_grad.zeta) {
                assert!((a - b).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn device_gpu_handles_absent_mask() {
        // Exercises the `mask: None` host-side path (dense all-observed matrix)
        // through the device entry point. On a GPU-equipped host this drives the
        // GPGPU kernels with the `None` mask branch; on a GPU-less host it is the
        // CPU fallback. Either way the device result must match the CPU reference
        // computed the same way.
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];

        let (cpu_obj, cpu_grad, cpu_ll) =
            neg_loglik_and_grad_device(Device::Cpu, &y, None, &[0, 0], &p, &cfg, &penalty);
        for device in [Device::Auto, Device::Gpu] {
            let (obj, grad, ll) =
                neg_loglik_and_grad_device(device, &y, None, &[0, 0], &p, &cfg, &penalty);
            assert!(
                (obj - cpu_obj).abs() < 1e-4,
                "objective mismatch for {device:?}"
            );
            assert!((ll - cpu_ll).abs() < 1e-4, "loglik mismatch for {device:?}");
            assert!((grad.tau - cpu_grad.tau).abs() < 1e-4);
            assert_eq!(grad.b.len(), cpu_grad.b.len());
        }
    }

    #[cfg(all(feature = "gpu", not(coverage)))]
    #[test]
    fn finish_device_prefers_gpu_result_when_present() {
        // When the GPU adapter produced a result, `finish_device` must return it
        // verbatim without invoking the CPU fallback, for both Gpu and Auto.
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let sentinel = Gradients {
            theta: vec![1.0, 2.0],
            alpha: vec![3.0, 4.0],
            b: vec![5.0, 6.0],
            xi: vec![7.0, 8.0, 9.0, 10.0],
            zeta: vec![11.0, 12.0, 13.0, 14.0],
            tau: 42.0,
        };
        for device in [Device::Gpu, Device::Auto] {
            let (obj, grad, ll) = finish_device(
                device,
                Some((123.0, sentinel.clone(), -7.0)),
                &y,
                None,
                &[0, 0],
                &p,
                &cfg,
                &penalty,
            );
            assert_eq!(obj, 123.0);
            assert_eq!(ll, -7.0);
            assert_eq!(grad.tau, 42.0);
            assert_eq!(grad.b, vec![5.0, 6.0]);
        }
    }

    #[cfg(all(feature = "gpu", not(coverage)))]
    #[test]
    fn finish_device_falls_back_to_cpu_when_gpu_absent() {
        // When the GPU produced no result, `finish_device` must reproduce the CPU
        // reference. `Device::Gpu` additionally emits the fallback warning while
        // `Device::Auto` stays silent; both must return the CPU numbers.
        let cfg = config();
        let p = params();
        let penalty = PenaltyConfig::default();
        let y = vec![1.0, 0.0, 0.0, 1.0];
        let mask = vec![true, true, true, false];
        let expected = neg_loglik_and_grad(&y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
        for device in [Device::Gpu, Device::Auto] {
            let (obj, grad, ll) =
                finish_device(device, None, &y, Some(&mask), &[0, 0], &p, &cfg, &penalty);
            assert_eq!(obj, expected.0);
            assert_eq!(ll, expected.2);
            assert_eq!(grad.tau, expected.1.tau);
        }
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

#[cfg(test)]
mod additional_tests {
    use super::*;

    #[test]
    fn test_mask_and_mirt() {
        let params = Params {
            theta: vec![0.0],
            alpha: vec![0.0],
            b: vec![0.0],
            xi: vec![0.0],
            zeta: vec![0.0],
            tau: 0.0,
        };
        let config = ModelConfig {
            n_persons: 1,
            n_items: 1,
            n_dims: 1,
            latent_dim: 1,
            eps_distance: 1e-12,
            model_type: ModelType::Mirt,
        };
        let penalty = PenaltyConfig {
            lambda_theta: 0.0,
            lambda_b: 0.0,
            lambda_alpha: 0.0,
            lambda_xi: 0.0,
            lambda_zeta: 0.0,
            lambda_tau: 0.0,
            mu_alpha: 0.0,
            mu_tau: 0.0,
        };
        let y = vec![1.0];
        let mask = vec![false];
        let (obj, _, _) = neg_loglik_and_grad(&y, Some(&mask), &[0], &params, &config, &penalty);
        assert_eq!(obj, 0.0);

        let mask_true = vec![true];
        let (obj_mirt, _, _) =
            neg_loglik_and_grad(&y, Some(&mask_true), &[0], &params, &config, &penalty);
        assert!(obj_mirt > 0.0);
    }
}
