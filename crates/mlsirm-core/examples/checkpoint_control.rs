//! Native typed persistence control, no external fit and no printed response/score arrays.
use mlsirm_core::{mask_grm::{self, MaskFit},bifactor_grm::BifactorMultigroupConfig,Device};
use sha2::{Digest,Sha256};
fn main(){
 let args:Vec<_>=std::env::args().collect();
 let path=std::path::Path::new(&args[2]);
 if args[1]=="produce" {
  assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(),Ok("1"));
  let y=[0,0,0,1,1,0,1,1,0,1,1,0,0,0,1,1];
  let groups=[0,0,0,0,1,1,1,1];
  let cfg=BifactorMultigroupConfig {q_general:7,q_specific:1,max_iter:1,tol:1e-8,n_starts:1,seed:42,newton_iter:20,ridge:1e-6,device:Device::Gpu,..Default::default()};
  let fit=mask_grm::state::with_private_checkpoints(path,||MaskFit::fit(&y,None,&groups,2,2,2,&[true,true],&cfg)).unwrap();
  fit.validate_gpu_evidence().unwrap();
  let raw=fit.export_private().unwrap();
  let name=format!("{:x}.maskfit.json",Sha256::digest(raw.as_bytes()));
  assert_eq!(std::fs::read_to_string(path.join(name)).unwrap(),raw);
  println!("{}",serde_json::json!({"state_sha256":format!("{:x}",Sha256::digest(raw.as_bytes())),"loglik_bits":fit.loglik_at(&fit.packed().unwrap()).unwrap().to_bits(),"trace_bits":fit.trace_loglik().to_bits(),"fit_identity":fit.summary().unwrap()["fit_identity"],"device_evidence_valid":true,"converged":fit.summary().unwrap()["converged"]}));
 } else {
  assert_eq!(args[1],"restore");
  let raw=std::fs::read_to_string(path).unwrap();
  let fit=MaskFit::restore_private(&raw).unwrap();
  assert_eq!(fit.export_private().unwrap(),raw);
  fit.validate_gpu_evidence().unwrap();
  println!("{}",serde_json::json!({"state_sha256":format!("{:x}",Sha256::digest(raw.as_bytes())),"loglik_bits":fit.loglik_at(&fit.packed().unwrap()).unwrap().to_bits(),"trace_bits":fit.trace_loglik().to_bits(),"fit_identity":fit.summary().unwrap()["fit_identity"],"device_evidence_valid":true,"converged":fit.summary().unwrap()["converged"]}));
 }
}
