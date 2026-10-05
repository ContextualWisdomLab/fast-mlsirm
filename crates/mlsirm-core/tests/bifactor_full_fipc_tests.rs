use mlsirm_core::bifactor_grm::full_fipc::{fit_bifactor_grm_fipc_full, BifactorFullFipcConfig};
use mlsirm_core::Device;

fn config() -> BifactorFullFipcConfig {
    BifactorFullFipcConfig { q_general:121,q_specific:121,max_iter:1,tol:1e-8,device:Device::Cpu }
}

#[test]
fn full_specific_means_update_and_budget_remains_nonconverged() {
    let y=[0,0,0,0,0,1,1,1,1,0,0,1,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,0,0,1,1];
    let ag=[0.7,1.1,0.8,1.2];let a_s=[1.2,-0.9,1.1,0.6];let th=[0.1,-0.3,0.4,-0.2];
    let r=fit_bifactor_grm_fipc_full(&y,&[0,0,1,1],8,4,2,2,&ag,&a_s,&th,&[0.2,-0.4,0.6],&[0.9,1.2,0.8],&config()).unwrap();
    assert_eq!(r.n_iter,1);assert!(!r.converged);assert_eq!(r.n_parameters,6);
    assert_eq!(r.a_general,ag);assert_eq!(r.a_specific,a_s);assert_eq!(r.threshold,th);
    assert!(r.mean[1]!=-0.4 && r.mean[2]!=0.6);
    assert!(r.em_map_displacement>1e-8);
    assert_eq!(r.loglik_trace.len(),2);
}

#[test]
fn malformed_direct_core_controls_refuse_without_panic() {
    let ag=[0.7,1.1,0.8,1.2];let a_s=[1.2,-0.9,1.1,0.6];let th=[0.1,-0.3,0.4,-0.2];
    let call=|y:&[usize],c:&BifactorFullFipcConfig|fit_bifactor_grm_fipc_full(y,&[0,0,1,1],8,4,2,2,&ag,&a_s,&th,&[0.2,-0.4,0.6],&[0.9,1.2,0.8],c);
    assert!(call(&[0],&config()).unwrap_err().contains("y must have length"));
    let mut c=config();c.q_general=0;
    assert!(call(&[0;32],&c).unwrap_err().contains("q_general"));
    c=config();c.tol=f64::NAN;
    assert!(call(&[0;32],&c).unwrap_err().contains("tol"));
}
