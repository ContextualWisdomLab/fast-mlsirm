//! Native synthetic workload discriminator; not identification or study fit.
#![cfg(feature = "gpu")]
use mlsirm_core::{bifactor_grm::SlopePrior, gpu_mstep::{update_grm, NewtonControl}};
#[test]
#[ignore = "requires canonical GPU lease"]
fn four_category_single_item() {
    assert_eq!(std::env::var("FMLS_E_GPU_LEASED").as_deref(), Ok("1"));
    let n: usize = std::env::var("POLY_ROWS").unwrap_or("1".into()).parse().unwrap();
    let iterations: usize = std::env::var("POLY_ITERATIONS").unwrap_or("1".into()).parse().unwrap();
    let nodes: Vec<f64> = (0..n).flat_map(|i| [((i%3) as f64-1.)*0.5, ((i/3%3) as f64-1.)*0.5]).collect();
    let counts = vec![vec![1.,2.,2.,1.]; n];
    let control = NewtonControl { iterations, backtracks:25, ridge:1e-8, gradient_tolerance:1e-9, step_tolerance:1e-9, pivot_tolerance:1e-14, armijo:1e-4 };
    let start=std::time::Instant::now();
    eprintln!("POLY_SINGLE_BEGIN rows={n} np=5 off=2 categories=4 iterations={iterations}");
    let (p,r)=update_grm(&[1.,0.6,1.,0.,-1.],2,&nodes,&counts,SlopePrior::None,&control).unwrap();
    assert!(p.iter().all(|v|v.is_finite()));
    assert!(p[2..].windows(2).all(|w|w[0]>w[1]));
    assert_eq!(r.cpu_numerical_fallbacks,0);
    assert!(r.gradient>0 && r.hessian>0);
    eprintln!("POLY_SINGLE_END seconds={} receipt={r:?}", start.elapsed().as_secs_f64());
}
