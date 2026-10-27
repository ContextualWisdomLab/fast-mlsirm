use super::*;
fn fit() -> MaskFit {
    let cfg = BifactorMultigroupConfig {
        q_general: 7,
        q_specific: 1,
        max_iter: 4,
        tol: 1e-8,
        n_starts: 1,
        seed: 42,
        newton_iter: 20,
        ridge: 1e-6,
        slope_prior: crate::bifactor_grm::SlopePrior::None,
        estimate_specific_vars: false,
        estimate_specific_means: false,
        device: crate::Device::Gpu,
    };
    let result: BifactorMultigroupResult=serde_json::from_value(serde_json::json!({
  "scoring_posterior_gpu":true,"start_outcomes":[],"dispatch":[],"dispatch_complete":true,"m_step_gpu_dispatches":0,
  "a_general":[[1.0],[1.0]],"a_specific":[[0.0],[0.0]],"threshold":[[0.2],[0.2]],"general_mean":[0.0,0.1],"general_sd":[1.0,1.1],"specific_sd":[[],[]],"specific_mean":[[],[]],"theta_g_eap":[],"theta_g_sd":[],"group_category_counts":[[],[]],"loglik_trace":[-3.0,-2.5],"n_iter":1,"converged":true,"termination_reason":"tolerance_met","final_loglik_change":0.5,"best_start":0,"n_parameters":6,"slope_prior":"None","em_objective_trace":[-3.0,-2.5]
 })).unwrap();
    MaskFit {
        y: vec![],
        observed: None,
        groups: vec![0, 1],
        n_groups: 2,
        n_items: 1,
        n_cat: 2,
        anchor: vec![true],
        cfg,
        result,
        specific_map: vec![-1],
        n_specific: 0,
        embedded_start: None,
        embedded_parent: None,
    }
}
#[test]
fn forged_dispatch_never_reaches_likelihood_or_inference() {
    let f = fit();
    let c = InferenceControls {
        fd_step: 1e-4,
        rank_atol: 1e-9,
        rank_rtol: 1e-9,
        objective_call_budget: 0,
        workspace_bytes: 0,
        ll_roundoff_tolerance: 1e-9,
        fit_ll_tolerance: 1e-9,
        lr_backend_tolerance: 1e-9,
        hessian_rtol: 1e-9,
        require_gpu: true,
    };
    let out = f.nested_lr(&fit(), 0, &c).unwrap();
    assert_eq!(out["status"], "gpu_receipt_required");
    assert!(out["information"].is_null());
    assert!(out["p_value"].is_null());
}

#[test]
fn execution_summary_does_not_recompute_likelihood_and_preserves_fields() {
    let f = fit();
    let view = f.execution_summary();
    assert_eq!(view["dispatch_complete"], false);
    assert_eq!(
        view["dispatch"],
        serde_json::to_value(&f.result.dispatch).unwrap()
    );
    assert_eq!(
        view["starts"],
        serde_json::to_value(&f.result.start_outcomes).unwrap()
    );
    assert_eq!(
        view["execution_evidence"],
        serde_json::to_value(&f.result.execution_evidence).unwrap()
    );
}
