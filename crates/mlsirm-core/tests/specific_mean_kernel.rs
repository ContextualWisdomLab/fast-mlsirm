use mlsirm_core::bifactor_grm::{fit_bifactor_grm_multigroup, BifactorMultigroupConfig};
use mlsirm_core::Device;

// Synthetic complete category support, focal shift on the specific block.
fn data() -> (Vec<usize>, Vec<usize>) {
    let mut y = Vec::new(); let mut groups = Vec::new();
    for g in 0..2 { for pattern in 0..81 {
        for rep in 0..(if g == 1 && pattern % 3 == 2 { 5 } else { 1 }) {
            let _ = rep; groups.push(g);
            for i in 0..4 { y.push((pattern / 3usize.pow(i)) % 3); }
        }
    }}
    (y, groups)
}
#[test]
fn specific_mean_updates_reference_pinned_and_budget_not_convergence() {
    let (y,g) = data();
    let cfg = BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:1,
        estimate_specific_means:true,estimate_specific_vars:true,device:Device::Cpu,..Default::default()};
    let r=fit_bifactor_grm_multigroup(&y,None,&g,2,&[0,0,-1,-1],g.len(),4,1,3,None,&cfg).unwrap();
    assert_eq!(r.specific_mean[0],vec![0.0]);
    assert_eq!(r.specific_sd[0],vec![1.0]);
    assert!(r.specific_mean[1][0].abs()>1e-4,"focal specific posterior mean must update: {:?}",r.specific_mean);
    assert!(!r.converged); assert_eq!(r.termination_reason,"max_iter_reached");
    assert_eq!(r.n_parameters,18); // 2 block*4 + 2 general*3 + 2 general prior + mean/SD
}
#[test]
fn specific_mean_general_only_preserves_existing_rejection() {
    let (y,g)=data();
    let cfg=BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:2,..Default::default()};
    for means in [false, true] {
        let error=fit_bifactor_grm_multigroup(&y,None,&g,2,&[-1;4],g.len(),4,0,3,None,
            &BifactorMultigroupConfig {estimate_specific_means:means,..cfg}).unwrap_err();
        assert_eq!(error, "n_specific must be >= 1");
    }
}
#[test]
fn specific_mean_single_group_reduces_exactly() {
    let (y,g)=data(); let one=vec![0;g.len()];
    let cfg=BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:2,..Default::default()};
    let a=fit_bifactor_grm_multigroup(&y,None,&one,1,&[0,0,-1,-1],g.len(),4,1,3,None,&cfg).unwrap();
    let b=fit_bifactor_grm_multigroup(&y,None,&one,1,&[0,0,-1,-1],g.len(),4,1,3,None,
        &BifactorMultigroupConfig {estimate_specific_means:true,estimate_specific_vars:true,..cfg}).unwrap();
    assert_eq!(a.loglik_trace,b.loglik_trace); assert_eq!(a.a_general,b.a_general);
    assert_eq!(a.a_specific,b.a_specific); assert_eq!(a.threshold,b.threshold);
    assert_eq!(a.theta_g_eap,b.theta_g_eap); assert_eq!(a.n_parameters,b.n_parameters);
    assert_eq!(b.specific_mean,vec![vec![0.0]]); assert_eq!(b.specific_sd,vec![vec![1.0]]);
}
#[test]
fn specific_mean_fixed_sd_and_disabled_mean_contracts() {
    let (y,g)=data();
    for (means,vars) in [(true,false),(false,true),(false,false)] {
        let cfg=BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:1,
            estimate_specific_means:means,estimate_specific_vars:vars,..Default::default()};
        let r=fit_bifactor_grm_multigroup(&y,None,&g,2,&[0,0,-1,-1],g.len(),4,1,3,None,&cfg).unwrap();
        if !vars { assert_eq!(r.specific_sd,vec![vec![1.0];2]); }
        if !means { assert_eq!(r.specific_mean,vec![vec![0.0];2]); }
        else { assert!(r.specific_mean[1][0].abs()>1e-4); }
        assert!(!r.converged);
    }
}
#[cfg(feature="gpu")]
#[test]
#[ignore="requires owner-held host GPU lease"]
fn specific_mean_gpu_fit_matches_cpu_and_terminal_dispatch() {
    let (y,g)=data();
    let cfg=BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:2,tol:1e-12,
        estimate_specific_means:true,estimate_specific_vars:true,..Default::default()};
    let a=fit_bifactor_grm_multigroup(&y,None,&g,2,&[0,0,-1,-1],g.len(),4,1,3,None,&cfg).unwrap();
    let b=fit_bifactor_grm_multigroup(&y,None,&g,2,&[0,0,-1,-1],g.len(),4,1,3,None,
        &BifactorMultigroupConfig {device:Device::Gpu,..cfg}).unwrap();
    assert!(!a.converged && !b.converged);
    assert_eq!(b.termination_reason,"max_iter_reached");
    for (x,y) in a.specific_mean.iter().flatten().zip(b.specific_mean.iter().flatten()) {assert!((x-y).abs()<2e-4);}
    for (x,y) in a.specific_sd.iter().flatten().zip(b.specific_sd.iter().flatten()) {assert!((x-y).abs()<2e-4);}
    for (x,y) in a.loglik_trace.iter().zip(&b.loglik_trace) {assert!((x-y).abs()<2e-3);}

}
