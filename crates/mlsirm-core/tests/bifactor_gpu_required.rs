//! Device-contract regression: an explicit GPU request must use hardware or
//! fail. wgpu 30.0.0 classifies `DeviceType::Cpu` as software rendering:
//! https://docs.rs/wgpu/30.0.0/wgpu/enum.DeviceType.html

use mlsirm_core::bifactor_grm::{
    fit_bifactor_grm, fit_bifactor_grm_multigroup, BifactorGrmConfig,
    BifactorMultigroupConfig, SlopePrior,
};
use mlsirm_core::Device;

#[test]
#[cfg_attr(feature = "gpu", ignore = "requires hardware GPU")]
fn explicit_gpu_request_uses_hardware_or_fails() {
    let y = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1];
    let smap = [0, 0, 1, 1];
    let config = BifactorGrmConfig {
        q_general: 3,
        q_specific: 3,
        max_iter: 1,
        tol: 1e-5,
        n_starts: 1,
        seed: 7,
        newton_iter: 1,
        ridge: 1e-4,
        slope_prior: SlopePrior::None,
        device: Device::Gpu,
    };
    let single = fit_bifactor_grm(&y, None, &smap, 4, 4, 2, 2, &config);
    #[cfg(not(feature = "gpu"))]
    assert!(single.unwrap_err().contains("GPU bifactor E-step requested"));
    #[cfg(feature = "gpu")]
    assert!(single.is_ok(), "single-group GPU fit failed: {single:?}");

    let multigroup = BifactorMultigroupConfig {
        q_general: config.q_general,
        q_specific: config.q_specific,
        max_iter: config.max_iter,
        tol: config.tol,
        n_starts: config.n_starts,
        seed: config.seed,
        newton_iter: config.newton_iter,
        ridge: config.ridge,
        slope_prior: config.slope_prior,
        estimate_specific_vars: false,
        device: Device::Gpu,
    };
    let group_id = [0, 0, 1, 1];
    let grouped = fit_bifactor_grm_multigroup(
        &y, None, &group_id, 2, &smap, 4, 4, 2, 2, None, &multigroup,
    );
    #[cfg(not(feature = "gpu"))]
    assert!(grouped.unwrap_err().contains("GPU bifactor E-step requested"));
    #[cfg(feature = "gpu")]
    assert!(grouped.is_ok(), "multigroup GPU fit failed: {grouped:?}");
}
