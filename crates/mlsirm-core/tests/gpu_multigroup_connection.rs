//! Generic multigroup dispatch regression; no study data.
#![cfg(feature = "gpu")]
use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig, enable_mstep_sweep_counters, reset_mstep_sweep_counters, mstep_sweep_counters};

#[test]
#[ignore = "requires owner-held physical GPU lease"]
fn multigroup_gpu_common_and_free_items_do_not_enter_cpu_newton() {
    let n = 27;
    let items = 7;
    let y: Vec<usize> = (0..n).flat_map(|p| (0..items).map(move |i| (p+i)%3)).collect();
    let groups: Vec<usize> = (0..n).map(|p| p/9).collect();
    let cfg = BifactorMultigroupConfig {
        q_general: 3, q_specific: 3, max_iter: 1, tol: 1e-8,
        n_starts: 1, seed: 20261026, newton_iter: 20, ridge: 1e-8,
        estimate_specific_means: true, estimate_specific_vars: true,
        device: mlsirm_core::Device::Gpu, ..Default::default()
    };
    enable_mstep_sweep_counters(true);
    reset_mstep_sweep_counters();
    let result = fit_bifactor_grm_multigroup(&y, None, &groups, 3,
        &[0,0,1,1,2,2,-1], n, items, 3, 3,
        Some(&[true,false,true,true,true,true,true]), &cfg);
    let counters = mstep_sweep_counters();
    enable_mstep_sweep_counters(false);
    eprintln!("MULTIGROUP_RESULT {:?}; CPU_NEWTON_COUNTERS {:?}", result.as_ref().map(|r| (r.n_iter,r.converged)), counters);
    assert_eq!(counters, (0,0,0,0), "GPU multigroup fitter executes CPU numerical M-step");
    assert!(result.is_ok(), "native fit failed: {:?}", result.err());
}
