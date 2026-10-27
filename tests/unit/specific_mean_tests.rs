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
fn specific_mean_unanchored_block_shift_preserves_cpu_likelihood() {
    let (v, y, obs, gid, pars) = fixture();
    let (x, w) = gh_rule(3).unwrap();
    let lw: Vec<f64> = w.iter().map(|x| x.ln()).collect();
    let tg = vec![x.to_vec(); 2];
    let ts = vec![vec![x.to_vec(); 3]; 2];
    // Blocks 0/2 and the general-only item are linked; block 1 is free.
    let anchor = [true, true, false, false, true, true, true];
    let mut shifted_pars = pars.clone();
    let mut shifted_ts = ts.clone();
    let c = 1.25;
    for node in &mut shifted_ts[1][1] { *node += c; }
    for &i in &v.blocks[1] {
        assert!(!anchor[i]);
        let a_s = shifted_pars[1][i].a_s.unwrap();
        for d in &mut shifted_pars[1][i].d { *d -= a_s * c; }
        assert!(shifted_pars[1][i].d[0] > shifted_pars[1][i].d[1]);
        for t in 0..3 { for h in 0..3 {
            for k in 0..v.m1 {
                let eta = pars[1][i].a_g * tg[1][t] + a_s * ts[1][1][h] + pars[1][i].d[k];
                let shifted = shifted_pars[1][i].a_g * tg[1][t] + a_s * shifted_ts[1][1][h] + shifted_pars[1][i].d[k];
                assert!((eta - shifted).abs() < 2e-15);
            }
        }}
    }
    let before = e_step_multigroup(&v, &y, Some(&obs), &gid, 2, &pars, &tg, &ts,
        &lw, &lw, 3, 3, crate::Device::Cpu);
    let after = e_step_multigroup(&v, &y, Some(&obs), &gid, 2, &shifted_pars, &tg, &shifted_ts,
        &lw, &lw, 3, 3, crate::Device::Cpu);
    assert!((before.0 - after.0).abs() < 1e-12);
    assert!((after.5[1][1] - before.5[1][1] - c * before.7[1][1]).abs() < 1e-12);
    // The slope-only prior cannot distinguish the two coordinates either.
    assert_eq!(pars.iter().flatten().map(|p| (p.a_g, p.a_s)).collect::<Vec<_>>(),
        shifted_pars.iter().flatten().map(|p| (p.a_g, p.a_s)).collect::<Vec<_>>());
}

#[test]
fn specific_mean_location_rank_controls() {
    let (v, _, _, _, mut pars) = fixture();
    let all = vec![true; 7];
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &all).is_ok());
    // One anchor per block supplies only 3 rows for 4 free locations.
    assert!(validate_specific_mean_location_rank(&v, &pars[0],
        &[true, false, true, false, true, false, false]).is_err());
    // Without a general-only item, nonproportional rows can still span all
    // locations. Presence of a general-only anchor is not a requirement.
    let no_general = [true, true, true, true, true, true, false];
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &no_general).is_ok());
    // Two common items per block can all remain proportional to general.
    for par in &mut pars[0][..6] { par.a_s = Some(2.0 * par.a_g); }
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &no_general).is_err());
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &all).is_ok());
    // A common item with zero loading cannot link its specific location.
    for &i in &v.blocks[1] { pars[0][i].a_s = Some(0.0); }
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &all).is_err());
    pars[0][2].a_s = Some(f64::NAN);
    assert!(validate_specific_mean_location_rank(&v, &pars[0], &all).is_err());
}

// Private observer: one actual receipt per reduced E-step, for every start.
// It is inert outside cfg(test) and never supplies the numerical result.
#[derive(Clone, Debug)]
struct SpecificMeanSweep {
    start: usize,
    iteration: usize,
    terminal: bool,
    receipt: crate::gpu_bifactor::GpuDispatchReceipt,
}
thread_local! {
    static FIT_SWEEPS: std::cell::RefCell<Option<Vec<SpecificMeanSweep>>> = const {
        std::cell::RefCell::new(None)
    };
}
pub(super) fn begin_fit_sweep() {
    FIT_SWEEPS.with(|events| {
        if events.borrow().is_some() { crate::gpu_bifactor::reset_gpu_dispatch_receipt(); }
    });
}
pub(super) fn record_fit_sweep(start: usize, iteration: usize) -> Option<usize> {
    FIT_SWEEPS.with(|events| {
        let mut capture = events.borrow_mut();
        let events = capture.as_mut()?;
        let index = events.len();
        events.push(SpecificMeanSweep { start, iteration, terminal: false,
            receipt: crate::gpu_bifactor::gpu_dispatch_receipt() });
        Some(index)
    })
}
pub(super) fn mark_terminal_fit_sweep(index: Option<usize>) {
    if let Some(index) = index {
        FIT_SWEEPS.with(|events| events.borrow_mut().as_mut().unwrap()[index].terminal = true);
    }
}

// Shared by the ignored hardware fit and deterministic CPU negative controls.
fn assert_specific_mean_gpu_fit(
    cpu: &BifactorMultigroupResult,
    gpu: &BifactorMultigroupResult,
    sweeps: &[SpecificMeanSweep],
    n_starts: usize,
) {
    assert!(!sweeps.is_empty(), "missing actual GPU sweep receipts");
    for sweep in sweeps {
        let r = &sweep.receipt;
        assert!(r.used && r.backend.as_deref().is_some_and(|s| !s.is_empty())
            && r.device_name.as_deref().is_some_and(|s| !s.is_empty())
            && r.fallback_reason.is_none(),
            "every fit/terminal sweep must use an identified GPU without fallback: {sweep:?}");
        assert!(sweep.start < n_starts);
    }
    for start in 0..n_starts {
        let events: Vec<_> = sweeps.iter().filter(|e| e.start == start).collect();
        assert!(!events.is_empty(), "missing start {start}");
        for (iteration, event) in events.iter().enumerate() {
            assert_eq!(event.iteration, iteration);
            assert_eq!(event.terminal, iteration + 1 == events.len());
        }
        let terminal = events.last().unwrap();
        assert!(terminal.terminal && terminal.iteration > 0);
        if start == gpu.best_start {
            assert_eq!(terminal.iteration, gpu.n_iter);
            assert_eq!(events.len(), gpu.loglik_trace.len());
        }
    }
    assert!(!cpu.converged && !gpu.converged);
    assert_eq!(gpu.termination_reason, "max_iter_reached");
    assert_eq!(cpu.specific_mean.len(), gpu.specific_mean.len());
    assert_eq!(cpu.specific_sd.len(), gpu.specific_sd.len());
    assert_eq!(cpu.loglik_trace.len(), gpu.loglik_trace.len());
    for (x, y) in cpu.specific_mean.iter().flatten().zip(gpu.specific_mean.iter().flatten()) {
        assert!((x - y).abs() < 2e-4);
    }
    for (x, y) in cpu.specific_sd.iter().flatten().zip(gpu.specific_sd.iter().flatten()) {
        assert!((x - y).abs() < 2e-4);
    }
    for (x, y) in cpu.loglik_trace.iter().zip(&gpu.loglik_trace) {
        assert!((x - y).abs() < 2e-3);
    }
}

fn dispatch_fit(device: crate::Device) -> (BifactorMultigroupResult, Vec<SpecificMeanSweep>) {
    FIT_SWEEPS.with(|events| *events.borrow_mut() = Some(Vec::new()));
    // Keep the predecessor integration fixture and quadrature/parity budgets;
    // use two starts to check nonwinning-start dispatch as well.
    let mut y = Vec::new();
    let mut gid = Vec::new();
    for g in 0..2 { for pattern in 0..81 {
        for _ in 0..(if g == 1 && pattern % 3 == 2 { 5 } else { 1 }) {
            gid.push(g);
            for i in 0..4 { y.push((pattern / 3usize.pow(i)) % 3); }
        }
    }}
    let cfg = BifactorMultigroupConfig {
        q_general: 7, q_specific: 7, max_iter: 2, tol: 1e-12,
        estimate_specific_means: true, estimate_specific_vars: true,
        n_starts: 2, device, ..Default::default()
    };
    let result = fit_bifactor_grm_multigroup(&y, None, &gid, 2,
        &[0, 0, -1, -1], gid.len(), 4, 1, 3, None, &cfg);
    let events = FIT_SWEEPS.with(|events| events.borrow_mut().take().unwrap());
    (result.unwrap(), events)
}

#[test]
fn specific_mean_gpu_fit_oracle_rejects_cpu_substitute() {
    let (cpu, sweeps) = dispatch_fit(crate::Device::Cpu);
    // Exactly equal CPU estimates satisfy the old parity/termination oracle.
    // There is no GPU request or adapter initialization in this control.
    let admitted = std::panic::catch_unwind(|| assert_specific_mean_gpu_fit(&cpu, &cpu, &sweeps, 2));
    assert!(admitted.is_err(), "GPU fit oracle accepts a CPU-only substitute");
    assert_eq!(sweeps.len(), 6);
    assert_eq!(sweeps.iter().filter(|e| e.terminal).count(), 2);
    assert!(sweeps.iter().all(|e| !e.receipt.used));
}

#[test]
fn specific_mean_gpu_fit_receipt_negative_controls() {
    let (cpu, cpu_sweeps) = dispatch_fit(crate::Device::Cpu);
    // Synthetic receipt-shaped positive only: the numeric result is CPU.
    // This tests the oracle, and is never physical GPU execution evidence.
    let mut good = cpu_sweeps.clone();
    for event in &mut good {
        event.receipt = crate::gpu_bifactor::GpuDispatchReceipt {
            used: true, backend: Some("synthetic-backend".into()),
            device_name: Some("synthetic-device".into()), fallback_reason: None,
        };
    }
    assert_specific_mean_gpu_fit(&cpu, &cpu, &good, 2);
    let rejects = |events: &[SpecificMeanSweep]| {
        assert!(std::panic::catch_unwind(||
            assert_specific_mean_gpu_fit(&cpu, &cpu, events, 2)).is_err());
    };
    rejects(&[]);
    rejects(&good[..good.len() - 1]);
    let mut bad = good.clone();
    bad[0].receipt = cpu_sweeps[0].receipt.clone(); // early fallback, later successes
    rejects(&bad);
    let mut bad = good.clone();
    bad.last_mut().unwrap().receipt = cpu_sweeps[0].receipt.clone(); // terminal CPU
    rejects(&bad);
    let mut bad = good.clone();
    bad[0].receipt.fallback_reason = Some("synthetic_early_fallback".into());
    rejects(&bad);
    let mut bad = good.clone();
    bad.last_mut().unwrap().terminal = false;
    rejects(&bad);
    let mut bad = good.clone();
    bad[0].receipt.backend = None;
    rejects(&bad);
    let mut bad = good.clone();
    bad[0].receipt.device_name = Some(String::new());
    rejects(&bad);
    let mut bad = good.clone();
    bad[0].iteration = 99;
    rejects(&bad);
    rejects(&good.iter().filter(|e| e.start == cpu.best_start).cloned().collect::<Vec<_>>());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires owner-held host GPU lease; no CPU fallback acceptance"]
fn specific_mean_gpu_fit_matches_cpu_and_terminal_dispatch() {
    let (cpu, _) = dispatch_fit(crate::Device::Cpu);
    let (gpu, sweeps) = dispatch_fit(crate::Device::Gpu);
    assert_specific_mean_gpu_fit(&cpu, &gpu, &sweeps, 2);
}

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
    let result=run_single_start_multigroup(&v,&y,Some(&obs),&gid,2,&anchor,&cfg,x,x,&lw,&lw,7,7,0,None,&mut Vec::new(),&mut 0).unwrap();
    assert!(!result.converged);
    for s in 0..3 {
        let mean=s1[1][s]/mass[1][s]; let variance=s2[1][s]/mass[1][s]-mean*mean;
        assert!((result.specific_mus[1][s]-mean).abs()<1e-12);
        assert!((result.taus[1][s].powi(2)-variance).abs()<1e-12);
        assert_eq!(result.specific_mus[0][s],0.0); assert_eq!(result.taus[0][s],1.0);
    }
}
