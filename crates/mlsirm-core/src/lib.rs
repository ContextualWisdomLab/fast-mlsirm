pub mod agreement;
pub mod cdm;
pub mod crm;
pub mod dif;
pub mod equating;
pub mod fitstats;
pub mod gpcm;
pub mod grm;
pub mod linking;
pub mod lltm;
pub mod marginal;
pub mod mhrm;
pub mod mixed;
pub mod mixture;
pub mod mmle;
pub mod nodes;
pub mod nominal;
pub mod oakes;
pub mod poly;
pub mod poly_marginal;
pub(crate) mod quadrature;
pub mod rasch_cml;
pub mod rsm;
pub mod rt;
pub mod rt_joint;
pub mod scoring;
pub mod testlet;
pub mod twopl;

/// Checked size arithmetic shared by public-input validators.
///
/// Keeping the overflow branch in one non-generic function avoids creating a
/// separate, partially covered closure for every validated buffer product.
pub(crate) fn checked_mul_usize(a: usize, b: usize, message: &str) -> Result<usize, String> {
    match a.checked_mul(b) {
        Some(value) => Ok(value),
        None => Err(message.to_owned()),
    }
}

/// Checked addition companion to [`checked_mul_usize`].
pub(crate) fn checked_add_usize(a: usize, b: usize, message: &str) -> Result<usize, String> {
    match a.checked_add(b) {
        Some(value) => Ok(value),
        None => Err(message.to_owned()),
    }
}

// cargo-llvm-cov runs in CPU-only CI against the repository-owned line-coverage
// baseline. Keep the hardware-backed wgpu module in normal builds, and cover
// the deterministic CPU fallback contract during coverage builds.
#[cfg(all(feature = "gpu", not(coverage)))]
mod gpu;
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) mod gpu_eapsum;
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) mod gpu_marginal;
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) mod gpu_plausible;
#[cfg(all(feature = "gpu", not(coverage)))]
pub(crate) mod gpu_scoring;
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
#[path = "../../../tests/unit/lib_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "../../../tests/unit/lib_additional_tests.rs"]
mod additional_tests;

#[cfg(test)]
#[path = "../../../tests/unit/marginal_recovery_tests.rs"]
mod marginal_recovery_tests;

#[cfg(test)]
#[path = "../../../tests/unit/proptest_neg_loglik_tests.rs"]
mod proptest_neg_loglik_tests;
