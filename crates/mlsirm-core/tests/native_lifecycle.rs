use mlsirm_core::{mask_grm, bifactor_grm::*, Device};
#[cfg(feature = "gpu")]
#[path = "support/fifa_hardware.rs"]
mod hardware_support;

#[cfg(feature = "gpu")]
#[test]
fn native_hardware_enrollment_is_explicit() {
    hardware_support::assert_hardware_cases_are_ignored(&[
        "native_gpu_bifactor_mixed_control",
        "native_gpu_general_only_control",
    ]);
}
#[cfg(not(feature="gpu"))]
#[test]
fn native_gpu_unavailable_no_fallback() {
    let cfg=BifactorMultigroupConfig { device:Device::Gpu,q_general:7,q_specific:1,..Default::default() };
    let error=fit_bifactor_grm_multigroup(&[0,0,1,1,0,1,1,0],None,&[0,0,1,1],2,&[-1,-1],4,2,0,2,None,&cfg).unwrap_err();
    assert!(error.contains("GPU required"));
}
#[test]
fn native_mask_policy_controls() {
    let (p,j)=mask_grm::select_removal(&[true,true,false], &[0.001,0.01,0.0001],0.05).unwrap();
    assert_eq!(p.len(),3); assert_eq!(j,Some(0));
    assert!(mask_grm::select_removal(&[true],&[f64::NAN],0.05).is_err());
    let c=mask_grm::bifactor_coordinates(&[true,false],2,3,&[-1,-1],0,false,false).unwrap();
    assert_eq!(c.len(),11);
}
#[test]
fn native_explicit_gpu_round_refuses_cpu() {
    assert!(mask_grm::lifecycle::complete_gpu(r#"{"device":"cpu"}"#).is_err());
    assert!(mask_grm::lifecycle::round_gpu(r#"{"device":"cpu","controls":{"require_gpu":true},"max_rounds":1}"#).is_err());
}
#[test]
fn native_zero_specific_validator_is_real_reduction() {
    let y=vec![0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0];
    let cfg=BifactorMultigroupConfig{q_general:7,q_specific:1,max_iter:1,newton_iter:2,..Default::default()};
    let r=fit_bifactor_grm_multigroup(&y,None,&[0,0,0,0,1,1,1,1],2,&[-1,-1],8,2,0,2,None,&cfg).unwrap();
    assert!(r.specific_mean.iter().all(|x|x.is_empty()));
    assert_eq!(r.n_parameters,6);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "required physical GPU cohort: tools/verify_lifecycle_hardware.py with the canonical host lease"]
fn native_gpu_bifactor_mixed_control() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(),Ok("1"));
    let mut y=Vec::new(); let mut groups=Vec::new(); let mut seed=77231u64;
    for g in 0..2 { for _ in 0..64 {
        groups.push(g);
        for _ in 0..8 { seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1); y.push((seed>>62) as usize); }
    }}
    let cfg=BifactorMultigroupConfig{q_general:7,q_specific:7,max_iter:1,newton_iter:20,
        estimate_specific_means:true,estimate_specific_vars:true,device:Device::Gpu,..Default::default()};
    let r=fit_bifactor_grm_multigroup(&y,None,&groups,2,&[0,0,0,1,1,1,2,2],128,8,3,4,
        Some(&[true,true,false,true,true,true,true,true]),&cfg).unwrap();
    assert!(r.dispatch_complete); assert_eq!(r.dispatch.len(),2);
    assert_eq!(r.m_step_gpu_dispatches,9);
    assert_eq!(r.specific_mean[1].len(),3);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "required physical GPU cohort: tools/verify_lifecycle_hardware.py with the canonical host lease"]
fn native_gpu_general_only_control() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(),Ok("1"));
    let y=vec![0,0,1,1,0,1,1,0,0,0,1,1,0,1,1,0];
    let cfg=BifactorMultigroupConfig{q_general:7,q_specific:1,max_iter:1,newton_iter:20,device:Device::Gpu,..Default::default()};
    let r=fit_bifactor_grm_multigroup(&y,None,&[0,0,0,0,1,1,1,1],2,&[-1,-1],8,2,0,2,None,&cfg).unwrap();
    assert!(r.dispatch_complete); assert_eq!(r.dispatch.len(),2);
    assert_eq!(r.m_step_gpu_dispatches,2);
    assert!(r.dispatch.iter().all(|d|d.gpu_used&&d.fallback_reason.is_none()));
    assert!(r.scoring_posterior_gpu);
    assert!(r.theta_g_sd.iter().all(|x| x.is_finite() && *x >= 0.0));
    let request=serde_json::json!({"y":y,"group_id":[0,0,0,0,1,1,1,1],"n_groups":2,"n_cat":2,
        "item_ids":["a","b"],"anchor":[true,true],"q_general":7,"max_iter":1,"tol":1e-8,
        "n_starts":1,"seed":42,"newton_iter":20,"ridge":1e-6,"device":"gpu",
        "alpha":0.05,"max_rounds":1,"controls":{"fd_step":0.001,"rank_atol":1e-9,"rank_rtol":1e-7,
        "objective_call_budget":10000,"workspace_bytes":10000000,"ll_roundoff_tolerance":1e-7,
        "fit_ll_tolerance":1e-4,"lr_backend_tolerance":1e-4,"hessian_rtol":0.01,"require_gpu":true}});
    let state=mask_grm::lifecycle::complete_gpu(&request.to_string()).unwrap();
    let state:serde_json::Value=serde_json::from_str(&state).unwrap();
    assert_eq!(state["status"],"initial_nonconverged");
    assert_eq!(state["initial"]["fit"]["final_posterior_device"],"gpu_f32");
    assert!(state.get("dif").is_none());
    assert!(state.get("final").is_none());
    let mut cpu_cfg=cfg.clone(); cpu_cfg.device=Device::Cpu;
    let cpu=fit_bifactor_grm_multigroup(&y,None,&[0,0,0,0,1,1,1,1],2,&[-1,-1],8,2,0,2,None,&cpu_cfg).unwrap();
    assert!(!cpu.scoring_posterior_gpu);
    for (a,b) in r.theta_g_eap.iter().zip(cpu.theta_g_eap.iter()) { assert!((a-b).abs()<1e-5); }
}
