//! Composition of the recovered native mask producer; no Python statistics.
//! Host probability tables, group-moment updates and information remain explicit.
use super::*;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
#[path = "mask_grm_resume.rs"]
pub mod resume;

pub fn capabilities_json() -> String {
    json!({"schema_version":1,"fit_backend":"gpu_two_word_e_tables_item_m_group_cm_eap_host_information",
        "full_matrix_single_removal":true,"all_items_retested":true,
        "native_identity":format!("{:x}",Sha256::digest(include_bytes!("mask_grm.rs"))),
        "group_moment_backend":"gpu_two_word_finite_grid_conditional_em",
        "final_posterior_backend":"gpu_two_word_multigroup","final_eap_backend":"gpu_two_word",
        "probability_table_backend":"gpu_two_word","full_gpu":false,"information_backend":"cpu_f64",
        "gradient_tolerance":1e-9,"step_tolerance":1e-9}).to_string()
}

pub fn round_gpu(request: &str) -> Result<String,String> {
    let value: Value = serde_json::from_str(request).map_err(|_| "invalid request JSON")?;
    if value["device"] != "gpu" || value["controls"]["require_gpu"] != true
        || !value["studied_item"].is_null() || value["max_rounds"] != 1 {
        return Err("one full-family explicit GPU round required".into());
    }
    run_json(request)
}

/// Initial fit, full-family purification, selected final fit. Never emits person rows.
/// An incomplete scientific gate is retained as a terminal stage, not promoted.
pub fn complete_gpu(request: &str) -> Result<String,String> {
    complete_with(request,fit_gpu,run_json)
}
// Private deterministic IO seam: public entry uses the actual native fit/DIF producers.
fn complete_with(request:&str,mut fit:impl FnMut(&str)->Result<String,String>,mut dif_run:impl FnMut(&str)->Result<String,String>)->Result<String,String>{
    let mut value: Value = serde_json::from_str(request).map_err(|_| "invalid request JSON")?;
    if value["device"] != "gpu" || value["controls"]["require_gpu"] != true
        || !value["studied_item"].is_null() {
        return Err("explicit full-family GPU lifecycle required".into());
    }
    progress_event(json!({"event":"lifecycle_initial_start","request_sha256":format!("{:x}",Sha256::digest(request.as_bytes()))}))?;
    let initial: Value = serde_json::from_str(&fit(request)?).map_err(|_| "invalid initial output")?;
    progress_event(json!({"event":"lifecycle_initial_terminal","result":initial}))?;
    let mut output = json!({"schema_version":1,
        "request_sha256":format!("{:x}",Sha256::digest(request.as_bytes())),
        "initial":initial,"status":"initial_nonconverged","full_gpu":false});
    if output["initial"]["fit"]["converged"] != true { return Ok(output.to_string()); }
    if output["initial"]["fit"]["dispatch_complete"] != true {
        output["status"]=json!("initial_gpu_receipt_required");return Ok(output.to_string());
    }
    let dif: Value = serde_json::from_str(&dif_run(request)?).map_err(|_| "invalid DIF output")?;
    let stable = dif["result"]["status"] == "stable_no_further_removal";
    value["anchor"] = dif["result"]["final_anchor"].clone();
    output["dif"] = dif;
    output["status"] = json!("dif_incomplete");
    if !stable { return Ok(output.to_string()); }
    let final_fit: Value = serde_json::from_str(&fit(&value.to_string())?).map_err(|_| "invalid final output")?;
    let complete = final_fit["fit"]["converged"] == true && final_fit["fit"]["dispatch_complete"] == true;
    output["final"] = final_fit;
    output["status"] = json!(if complete {"complete_host_eap_information"} else {"final_nonconverged"});
    Ok(output.to_string())
}

pub fn fit_gpu(request: &str) -> Result<String,String> {
    let r: Request = serde_json::from_str(request).map_err(|_| "invalid request JSON")?;
    if r.device != "gpu" || !r.controls.require_gpu {
        return Err("explicit GPU initial/final fit required".into());
    }
    let n = r.item_ids.len();
    if n == 0 || r.anchor.len()!=n || r.item_ids.iter().any(|s|s.is_empty())
        || r.item_ids.iter().collect::<std::collections::HashSet<_>>().len()!=n {
        return Err("invalid item identity".into());
    }
    let (map, ns, qs, means, vars) = match &r.bifactor {
        Some(b) => (b.specific_map.clone(), b.n_specific,b.q_specific,b.estimate_specific_means,b.estimate_specific_vars),
        None => (vec![-1;n],0,1,false,false),
    };
    let cfg = BifactorMultigroupConfig { q_general:r.q_general,q_specific:qs,
        max_iter:r.max_iter,tol:r.tol,n_starts:r.n_starts,seed:r.seed,
        newton_iter:r.newton_iter,ridge:r.ridge,estimate_specific_means:means,
        estimate_specific_vars:vars,device:crate::Device::Gpu,
        slope_prior:crate::bifactor_grm::SlopePrior::None };
    let fit = MaskFit::fit_bifactor(&r.y,r.observed.as_deref(),&r.group_id,r.n_groups,
        n,r.n_cat,&r.anchor,&cfg,&map,ns)?;
    fit.validate_gpu_evidence()?;
    Ok(json!({"request_sha256":format!("{:x}",Sha256::digest(request.as_bytes())),
        "fit":fit.summary()?,"capabilities":serde_json::from_str::<Value>(&capabilities_json()).unwrap()}).to_string())
}

#[cfg(test)]mod receipt_transition_tests{
 use super::*;
 fn request()->String{json!({"device":"gpu","controls":{"require_gpu":true},"studied_item":null}).to_string()}
 #[test]fn missing_initial_receipt_stops_before_dif(){
  let calls=std::cell::Cell::new(0);
  let r=complete_with(&request(),|_|Ok(json!({"fit":{"converged":true,"dispatch_complete":false}}).to_string()),|_|{calls.set(calls.get()+1);Err("DIF sentinel".into())}).unwrap();
  let r:Value=serde_json::from_str(&r).unwrap();assert_eq!(r["status"],"initial_gpu_receipt_required");assert_eq!(calls.get(),0);
 }
 #[test]fn missing_final_receipt_is_not_complete(){
  let calls=std::cell::Cell::new(0);
  let r=complete_with(&request(),|_|{calls.set(calls.get()+1);Ok(json!({"fit":{"converged":true,"dispatch_complete":calls.get()==1}}).to_string())},|_|Ok(json!({"result":{"status":"stable_no_further_removal","final_anchor":[true]}}).to_string())).unwrap();
  let r:Value=serde_json::from_str(&r).unwrap();assert_ne!(r["status"],"complete_host_eap_information");assert_eq!(calls.get(),2);
 }
}
