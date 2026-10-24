use super::*;

fn fixture() -> (Validated, Vec<usize>, Vec<bool>, Vec<usize>, Vec<Vec<ItemParams>>) {
    let y: Vec<usize> = (0..9).flat_map(|p| (0..7).map(move |i| (p+i)%3)).collect();
    let mut obs=vec![true;y.len()];
    // The last person's block 1 is entirely missing: no block moment mass.
    obs[8*7+2]=false; obs[8*7+3]=false;
    let v=validate(&y,Some(&obs),&[0,0,1,1,2,2,-1],9,7,3,3,&BifactorGrmConfig {q_general:3,q_specific:3,max_iter:1,tol:1e-6,n_starts:1,seed:1,newton_iter:10,ridge:1e-8,slope_prior:SlopePrior::None,device:crate::Device::Cpu}).unwrap();
    let params:Vec<ItemParams>=(0..7).map(|i| ItemParams {a_g:0.4+0.1*i as f64,
        a_s:if i<6 {Some(0.8)} else {None},d:vec![0.7,-0.6]}).collect();
    (v,y,obs,vec![0,0,0,1,1,1,1,1,1],vec![params.clone(),params])
}

fn moments(device:crate::Device) {
    let (v,y,obs,gid,pars)=fixture();
    let (x,w)=gh_rule(3).unwrap(); let lw:Vec<_>=w.iter().map(|v|v.ln()).collect();
    let tg=vec![x.to_vec(),x.iter().map(|x|0.3+1.1*x).collect()];
    let ts:Vec<Vec<Vec<f64>>>=(0..2).map(|g|(0..3).map(|s|x.iter().map(|x|
        if g==0 {*x} else {0.4*(s+1) as f64+(0.7+0.2*s as f64)*x}).collect()).collect()).collect();
    let (ll,_,_,_,_,s1,s2,mass)=e_step_multigroup(&v,&y,Some(&obs),&gid,2,&pars,&tg,&ts,&lw,&lw,3,3,device);
    if device==crate::Device::Gpu {
        let r=crate::gpu_bifactor::gpu_dispatch_receipt();
        assert!(r.used && r.backend.is_some() && r.device_name.is_some() && r.fallback_reason.is_none());
        eprintln!("physical_gpu backend={:?} device={:?}",r.backend,r.device_name);
    }
    let mut bll=0.0; let mut b1=vec![vec![0.0;3];2]; let mut b2=b1.clone(); let mut bm=b1.clone();
    for person in 0..9 {
        let g=gid[person]; let mut den=0.0; let mut a=[0.0;3]; let mut b=[0.0;3];
        // Full 4D product grid is independent of the reduced posterior assembly.
        for t in 0..3 { for h0 in 0..3 { for h1 in 0..3 { for h2 in 0..3 {
            let hs=[h0,h1,h2]; let mut value=w[t]*w[h0]*w[h1]*w[h2];
            for i in 0..7 { if obs[person*7+i] {
                let par=&pars[g][i];
                let eta=par.a_g*tg[g][t]+if i<6 {par.a_s.unwrap()*ts[g][i/2][hs[i/2]]} else {0.0};
                value*=grm_logprobs(eta,&par.d)[y[person*7+i]].exp();
            }}
            den+=value;
            for s in 0..3 { let z=ts[g][s][hs[s]]; a[s]+=value*z; b[s]+=value*z*z; }
        }}}}
        bll+=den.ln();
        for s in 0..3 { if v.blocks[s].iter().any(|&i|obs[person*7+i]) {
            bm[g][s]+=1.0; b1[g][s]+=a[s]/den; b2[g][s]+=b[s]/den;
        }}
    }
    let eps=if device==crate::Device::Gpu {2e-5} else {1e-11};
    assert!((ll-bll).abs()<eps,"ll {ll} vs {bll}");
    for g in 0..2 { for s in 0..3 {
        assert!((mass[g][s]-bm[g][s]).abs()<eps);
        assert!((s1[g][s]-b1[g][s]).abs()<eps,"first moment g={g} s={s}: {} vs {}",s1[g][s],b1[g][s]);
        assert!((s2[g][s]-b2[g][s]).abs()<eps);
    }}
}
#[test]
fn specific_mean_cpu_full_product_moments_with_missing_block() { moments(crate::Device::Cpu); }
#[test]
fn specific_mean_reflection_moves_means_and_preserves_linear_predictor() {
    let mut slopes=vec![vec![-1.2,-0.8,0.7,1.1],vec![-1.2,-4.0,0.7,1.2]];
    let before=slopes.clone(); let mut means=vec![vec![0.0,0.0],vec![0.9,-0.4]];
    let means_before=means.clone();
    canonicalize_multigroup_specific(&[vec![0,1],vec![2,3]], &[true,false,true,false], &mut slopes,&mut means);
    assert_eq!(means,vec![vec![-0.0,0.0],vec![-0.9,-0.4]]);
    for g in 0..2 {for i in 0..4 {
        let s=i/2; let node=0.37;
        let sign=if s==0 {-1.0} else {1.0};
        assert!((before[g][i]*(means_before[g][s]+node)-slopes[g][i]*(means[g][s]+sign*node)).abs()<1e-12);
    }}
}
#[test]
fn specific_mean_final_g3_likelihood_and_eap_use_affine_nodes() {
    let (_,y,obs,gid,_)=fixture();
    let cfg=BifactorMultigroupConfig {q_general:3,q_specific:3,max_iter:1,
        estimate_specific_means:true,estimate_specific_vars:true,n_starts:2,..Default::default()};
    let r=fit_bifactor_grm_multigroup(&y,Some(&obs),&gid,2,&[0,0,1,1,2,2,-1],9,7,3,3,None,&cfg).unwrap();
    assert_eq!(r.n_parameters,35); // 6*4 + 3 + 2 + 3 means + 3 SDs
    let (x,w)=gh_rule(3).unwrap(); let mut total=0.0;
    for person in 0..9 {
        let g=gid[person]; let mut den=0.0; let mut num=0.0; let mut second=0.0;
        for t in 0..3 {for h0 in 0..3 {for h1 in 0..3 {for h2 in 0..3 {
            let hs=[h0,h1,h2]; let theta=r.general_mean[g]+r.general_sd[g]*x[t];
            let mut prob=w[t]*w[h0]*w[h1]*w[h2];
            for i in 0..7 {if obs[person*7+i] {
                let eta=r.a_general[g][i]*theta+if i<6 {
                    r.a_specific[g][i]*(r.specific_mean[g][i/2]+r.specific_sd[g][i/2]*x[hs[i/2]])
                } else {0.0};
                prob*=grm_logprobs(eta,&r.threshold[g][i*2..i*2+2])[y[person*7+i]].exp();
            }}
            den+=prob; num+=prob*theta; second+=prob*theta*theta;
        }}}}
        total+=den.ln(); let mean=num/den;
        assert!((r.theta_g_eap[person]-mean).abs()<1e-11);
        assert!((r.theta_g_sd[person].powi(2)-(second/den-mean*mean)).abs()<1e-11);
    }
    assert!((total-r.loglik_trace.last().unwrap()).abs()<1e-11);
}

#[cfg(feature="gpu")]
#[test]
#[ignore = "requires owner-held host GPU lease; never silently skipped inside test"]
fn specific_mean_gpu_full_product_moments_with_missing_block() { moments(crate::Device::Gpu); }

#[test]
fn specific_mean_centered_variance_matches_first_estep() {
    let (v,y,obs,gid,_)=fixture();
    let (x,w)=gh_rule(7).unwrap(); let lw:Vec<_>=w.iter().map(|v|v.ln()).collect();
    let cfg=BifactorMultigroupConfig {q_general:7,q_specific:7,max_iter:1,estimate_specific_means:true,
        estimate_specific_vars:true,..Default::default()};
    let anchor=vec![true;7];
    let (pars,_,_,_,_)=initial_params_multigroup(&v,&y,Some(&obs),&gid,2,&anchor,cfg.seed,0,true,true);
    let tg=vec![x.to_vec();2]; let ts=vec![vec![x.to_vec();3];2];
    let (_,_,_,_,_,s1,s2,mass)=e_step_multigroup(&v,&y,Some(&obs),&gid,2,&pars,&tg,&ts,&lw,&lw,7,7,crate::Device::Cpu);
    let result=run_single_start_multigroup(&v,&y,Some(&obs),&gid,2,&anchor,&cfg,x,x,&lw,&lw,7,7,0).unwrap();
    assert!(!result.converged);
    for s in 0..3 {
        let mean=s1[1][s]/mass[1][s]; let variance=s2[1][s]/mass[1][s]-mean*mean;
        assert!((result.specific_mus[1][s]-mean).abs()<1e-12);
        assert!((result.taus[1][s].powi(2)-variance).abs()<1e-12);
        assert_eq!(result.specific_mus[0][s],0.0); assert_eq!(result.taus[0][s],1.0);
    }
}
