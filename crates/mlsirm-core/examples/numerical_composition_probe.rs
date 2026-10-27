//! Synthetic integration probe; no study input and no estimator implementation.
use mlsirm_core::mask_grm;
fn main() {
    assert_eq!(std::env::var("FMLS_DS_GPU_LEASED").as_deref(), Ok("1"));
    let mut seed=98231u64;
    let mut uniform=||{seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);((seed>>11) as f64+0.5)/9007199254740992.0};
    let mut y=Vec::new(); let mut groups=Vec::new();
    for g in 0..2 { for _ in 0..128 {
        let theta=(uniform()+uniform()+uniform()+uniform()+uniform()+uniform()-3.)*2f64.sqrt();
        groups.push(g);
        for i in 0..4 {let eta=(0.8+0.2*i as f64)*theta+(-0.6+0.4*i as f64);y.push(usize::from(uniform()<1./(1.+(-eta).exp())));}
    }}
    let request=serde_json::json!({"y":y,"group_id":groups,"n_groups":2,"n_cat":2,
        "item_ids":["a","b","c","d"],"anchor":[true,true,true,true],"q_general":7,"max_iter":100,"tol":1e-8,
        "n_starts":1,"seed":42,"newton_iter":20,"ridge":1e-6,"device":"gpu",
        "alpha":0.05,"max_rounds":1,"controls":{"fd_step":0.001,"rank_atol":1e-9,"rank_rtol":1e-7,
        "objective_call_budget":10000,"workspace_bytes":10000000,"ll_roundoff_tolerance":1e-7,
        "fit_ll_tolerance":1e-4,"lr_backend_tolerance":1e-4,"hessian_rtol":0.01,"require_gpu":true}});
    let mode=std::env::args().nth(1).expect("initial or complete");
    let out=if mode=="initial" {mask_grm::lifecycle::fit_gpu(&request.to_string())} else {assert_eq!(mode,"complete");mask_grm::lifecycle::complete_gpu(&request.to_string())};
    match out {Ok(s)=>println!("{}",s),Err(e)=>{eprintln!("NATIVE_REFUSAL {}",e);std::process::exit(2);}}
}
