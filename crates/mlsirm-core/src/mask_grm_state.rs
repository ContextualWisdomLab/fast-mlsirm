//! Trusted private fitted-state persistence, not an optimizer checkpoint or attestation.
//! The payload contains responses and scores: callers must use private storage.
use super::*;
use sha2::{Digest, Sha256};

impl MaskFit {
    pub fn export_private(&self) -> Result<String, String> {
        let state = serde_json::to_string(self).map_err(|_| "state serialization failed")?;
        // String payload retains binary64 round-trip spelling and prevents Value conversion loss.
        Ok(serde_json::json!({"schema":1,"source":state_source(),
            "sha256":format!("{:x}",Sha256::digest(state.as_bytes())),"state":state}).to_string())
    }
    pub fn restore_private(raw: &str) -> Result<Self, String> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Envelope { schema:u32, source:String, sha256:String, state:String }
        let e:Envelope=serde_json::from_str(raw).map_err(|_|"invalid private envelope")?;
        if e.schema!=1 || e.source!=state_source()
            || e.sha256!=format!("{:x}",Sha256::digest(e.state.as_bytes())) {
            return Err("private state schema/source/digest mismatch".into());
        }
        let f:Self=serde_json::from_str(&e.state).map_err(|_|"invalid private state")?;
        let g=f.n_groups; let n=f.n_items; let s=f.n_specific; let r=&f.result;
        let matrix=|v:&Vec<Vec<f64>>,width:usize| v.len()==g && v.iter().all(|row|row.len()==width && row.iter().all(|x|x.is_finite()));
        let cells=f.groups.len().checked_mul(n).ok_or("state geometry overflow")?;
        let thresholds=n.checked_mul(f.n_cat.checked_sub(1).ok_or("invalid categories")?).ok_or("state geometry overflow")?;
        if g<1 || n<1 || f.n_cat<2 || f.y.len()!=cells || f.groups.iter().any(|&v|v>=g)
            || f.observed.as_ref().is_some_and(|v|v.len()!=cells)
            || f.anchor.len()!=n || f.specific_map.len()!=n
            || f.specific_map.iter().any(|&v|v < -1 || (v>=0 && v as usize>=s))
            || !matrix(&r.a_general,n) || !matrix(&r.a_specific,n) || !matrix(&r.threshold,thresholds)
            || !matrix(&r.specific_mean,s) || !matrix(&r.specific_sd,s)
            || r.general_mean.len()!=g || r.general_sd.len()!=g
            || r.theta_g_eap.len()!=f.groups.len() || r.theta_g_sd.len()!=f.groups.len()
            || f.cfg.q_general==0 || f.cfg.q_specific==0 {
            return Err("invalid private state geometry".into());
        }
        let packed=f.packed()?;
        if packed.len()!=r.n_parameters || packed.iter().any(|v|!v.is_finite())
            || !f.loglik_at(&packed)?.is_finite() {return Err("invalid private fit coordinates".into());}
        Ok(f)
    }
}
fn state_source()->String {
    let mut h=Sha256::new();
    h.update(include_bytes!("mask_grm.rs"));
    h.update(include_bytes!("bifactor_grm.rs"));
    h.update(include_bytes!("mask_grm_state.rs"));
    format!("{:x}",h.finalize())
}

/// Explicit private-state output; no printing and no automatic filesystem writes.
pub fn fit_private(request:&str)->Result<String,String> {
    let r:Request=serde_json::from_str(request).map_err(|_|"invalid request JSON")?;
    let n=r.item_ids.len();
    if n==0 || r.anchor.len()!=n {return Err("invalid item identity".into());}
    let device=match r.device.as_str(){"cpu"=>crate::Device::Cpu,"gpu"=>crate::Device::Gpu,_=>return Err("explicit device required".into())};
    if r.controls.require_gpu && device!=crate::Device::Gpu {return Err("GPU required".into());}
    let (map,ns,qs,means,vars)=match &r.bifactor {
        Some(b)=>(b.specific_map.clone(),b.n_specific,b.q_specific,b.estimate_specific_means,b.estimate_specific_vars),
        None=>(vec![-1;n],0,1,false,false),
    };
    let cfg=BifactorMultigroupConfig {q_general:r.q_general,q_specific:qs,max_iter:r.max_iter,tol:r.tol,
        n_starts:r.n_starts,seed:r.seed,newton_iter:r.newton_iter,ridge:r.ridge,
        estimate_specific_means:means,estimate_specific_vars:vars,device,slope_prior:crate::bifactor_grm::SlopePrior::None};
    MaskFit::fit_bifactor(&r.y,r.observed.as_deref(),&r.group_id,r.n_groups,n,r.n_cat,&r.anchor,&cfg,&map,ns)?.export_private()
}
