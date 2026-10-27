//! Likelihood-only CPU-f64 reduction for the validated private MaskFit consumer.
//! Preserve the fitting E-step's node/item/block/person summation order.
//! Cache only exact factor values, never differences or rounded parameters.
use super::MaskFit;
use crate::bifactor_grm::MultigroupParameters;

fn log_sum_exp(xs: &[f64]) -> f64 {
    let mx = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if mx == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    let mut acc = 0.0;
    for &x in xs { acc += (x - mx).exp(); }
    mx + acc.ln()
}
struct Entry { key: Vec<u64>, values: Vec<f64> }
#[derive(Default)]
struct Factor { base: Option<Entry>, last: Option<Entry> }
/// Per-information-call bounded cache. At most two arrays per group/factor:
/// the base and the most recently changed parameter tuple. No global state.
/// The fit, responses, missingness, map and quadrature are borrowed immutably.
pub(super) struct Cache<'a> {
    fit: &'a MaskFit,
    factors: Vec<Factor>,
    persons: Vec<Vec<usize>>,
    local: Vec<usize>,
    members: Vec<Vec<usize>>,
    x: &'static [f64], z: &'static [f64], lw: Vec<f64>, lv: Vec<f64>,
}
impl<'a> Cache<'a> {
    pub(super) fn new(f: &'a MaskFit) -> Result<Self,String> {
        let (x,w)=crate::bifactor_grm::gh_rule(f.cfg.q_general)?;
        let (z,v)=crate::bifactor_grm::gh_rule(f.cfg.q_specific)?;
        let mut persons=vec![Vec::new();f.n_groups];
        let mut local=vec![0;f.groups.len()];
        for (p,&g) in f.groups.iter().enumerate() {local[p]=persons[g].len();persons[g].push(p);}
        let members=(0..=f.n_specific).map(|s|(0..f.n_items).filter(|&i|if s==f.n_specific {f.specific_map[i]<0} else {f.specific_map[i]==s as i32}).collect()).collect();
        Ok(Self {fit:f,factors:(0..f.n_groups*(f.n_specific+1)).map(|_|Factor::default()).collect(),persons,local,members,x,z,lw:w.iter().map(|v|v.ln()).collect(),lv:v.iter().map(|v|v.ln()).collect()})
    }
    pub(super) fn loglik(&mut self, r: &MultigroupParameters<'_>) -> Result<f64,String> {
        let f=self.fit;let qg=self.x.len();let qs=self.z.len();let m=f.n_cat-1;
        let nf=f.n_specific+1;
        let mut selected=vec![false;self.factors.len()];
        for g in 0..f.n_groups {for s in 0..nf {
            let general=s==f.n_specific;
            let members=&self.members[s];
            let mut key=vec![r.general_mean[g].to_bits(),r.general_sd[g].to_bits()];
            if !general {key.extend([r.specific_mean[g][s].to_bits(),r.specific_sd[g][s].to_bits()]);}
            for &i in members {
                key.push(r.a_general[g][i].to_bits());
                if !general {key.push(r.a_specific[g][i].to_bits());}
                key.extend(r.threshold[g][i*m..(i+1)*m].iter().map(|v|v.to_bits()));
            }
            let idx=g*nf+s;
            let factor=&mut self.factors[idx];
            if factor.base.as_ref().is_some_and(|e|e.key==key) {continue;}
            selected[idx]=true;
            if factor.last.as_ref().is_some_and(|e|e.key==key) {continue;}
            let nh=if general {1} else {qs};
            let tg:Vec<_>=self.x.iter().map(|x|r.general_mean[g]+r.general_sd[g]*x).collect();
            let ts:Vec<_>=if general {Vec::new()} else {self.z.iter().map(|z|r.specific_mean[g][s]+r.specific_sd[g][s]*z).collect()};
            let mut tables=Vec::with_capacity(members.len());
            for &i in members {
                let mut lp=vec![0.0;qg*nh*f.n_cat];
                for t in 0..qg {for h in 0..nh {
                    let eta=if general {r.a_general[g][i]*tg[t]} else {r.a_general[g][i]*tg[t]+r.a_specific[g][i]*ts[h]};
                    let probs=crate::poly::grm_logprobs(eta,&r.threshold[g][i*m..(i+1)*m]);
                    lp[(t*nh+h)*f.n_cat..(t*nh+h+1)*f.n_cat].copy_from_slice(&probs);
                }}
                tables.push(lp);
            }
            let mut values=vec![0.0;self.persons[g].len()*qg];
            let mut tmp=vec![0.0;qs];
            for (row,&p) in self.persons[g].iter().enumerate() {
                // Remove missing items once, outside the integration loops.
                let observed:Vec<_>=members.iter().enumerate().filter(|(_,i)|f.observed.as_ref().is_none_or(|o|o[p*f.n_items+**i])).map(|(j,&i)|(j,f.y[p*f.n_items+i])).collect();
                for t in 0..qg {
                    if general {
                        let mut acc=self.lw[t];
                        for &(j,yc) in &observed {acc+=tables[j][t*f.n_cat+yc];}
                        values[row*qg+t]=acc;
                    } else {
                        for h in 0..qs {
                            let mut acc=self.lv[h];
                            for &(j,yc) in &observed {acc+=tables[j][(t*qs+h)*f.n_cat+yc];}
                            tmp[h]=acc;
                        }
                        values[row*qg+t]=log_sum_exp(&tmp);
                    }
                }
            }
            let entry=Entry {key,values};
            if factor.base.is_none() {factor.base=Some(entry);selected[idx]=false;} else {factor.last=Some(entry);}
        }}
        let mut marginal=vec![0.0;qg];let mut ll=0.0;
        for (p,&g) in f.groups.iter().enumerate() {
            let row=self.local[p];
            let get=|s:usize| {let idx=g*nf+s;let fact=&self.factors[idx];if selected[idx] {&fact.last.as_ref().unwrap().values} else {&fact.base.as_ref().unwrap().values}};
            for t in 0..qg {
                let mut acc=get(f.n_specific)[row*qg+t];
                for s in 0..f.n_specific {acc+=get(s)[row*qg+t];}
                marginal[t]=acc;
            }
            ll+=log_sum_exp(&marginal);
        }
        if !ll.is_finite() {return Err("nonfinite marginal likelihood".into());}
        Ok(ll)
    }
}
pub(super) fn loglik(f: &MaskFit, r: &MultigroupParameters<'_>) -> Result<f64,String> {
    Cache::new(f)?.loglik(r)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bifactor_grm::BifactorMultigroupConfig;
    use std::time::Instant;
    fn fixture() -> MaskFit {
        let map=[2,0,1,0,1,0,2,2,1,1,2,0,2,1];
        let mut y=Vec::new(); let mut groups=Vec::new(); let mut obs=Vec::new();
        let mut seed=20261023u64;
        for p in 0..36 {groups.push(p%3);for i in 0..14 {
            seed=seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            y.push(if i==0 {(p/3)%4} else {(seed>>32) as usize%4});
            obs.push(i==0 || (p+i)%11!=0);
        }}
        let cfg=BifactorMultigroupConfig {q_general:5,q_specific:5,max_iter:2,n_starts:1,estimate_specific_means:true,estimate_specific_vars:true,device:crate::Device::Cpu,..Default::default()};
        MaskFit::fit_bifactor(&y,Some(&obs),&groups,3,14,4,&[true;14],&cfg,&map,3).unwrap()
    }
    #[test]
    fn bounded_q21_objective_profile_and_cache_invalidation() {
        let mut f=fixture();
        // Fixed-parameter likelihood profile only: repeated synthetic rows,
        // not a convergence or quadrature-accuracy claim.
        f.y=(0..1200).flat_map(|p|f.y[(p%36)*14..(p%36+1)*14].iter().copied()).collect();
        f.observed=Some((0..1200).flat_map(|p|f.observed.as_ref().unwrap()[(p%36)*14..(p%36+1)*14].iter().copied()).collect());
        f.groups=(0..1200).map(|p|f.groups[p%36]).collect();
        f.cfg.q_general=21;f.cfg.q_specific=21;
        let x=f.packed().unwrap();
        let probes:Vec<_>=(0..100).map(|i|{let mut z=x.clone();z[i%x.len()]+=if i%2==0 {0.001} else {-0.001};z[(i*17+13)%x.len()]-=0.0005;z}).collect();
        let start=Instant::now();
        let old:Vec<_>=probes.iter().map(|z|f.loglik_at_impl(z,false).unwrap()).collect();
        let old_s=start.elapsed().as_secs_f64();
        let mut cache=Cache::new(&f).unwrap();
        let start=Instant::now();
        let new:Vec<_>=probes.iter().map(|z|f.loglik_at_cached(z,true,Some(&mut cache)).unwrap()).collect();
        let new_s=start.elapsed().as_secs_f64();
        assert_eq!(old.iter().map(|v|v.to_bits()).collect::<Vec<_>>(),new.iter().map(|v|v.to_bits()).collect::<Vec<_>>());
        assert!(cache.factors.iter().all(|f|f.base.is_some()));
        eprintln!("PROFILE_FIXED_PARAMETERS N=1200 quadpts=21 objective_calls=100 legacy_seconds={old_s} cached_seconds={new_s} speedup={} likelihoods_bitwise_equal=true",old_s/new_s);
    }
    #[test]
    fn likelihood_only_matches_estep_stencils_and_hessian() {
        let f=fixture();let x=f.packed().unwrap();
        // Every coordinate class, anchored and released parameters, focal priors,
        // partial/block missingness, two stencil steps and mixed perturbations.
        for f in [&f,&f.release_item(0).unwrap()] {
            let x=f.packed().unwrap();
            for step in [0.001,0.0005] {
                for i in 0..x.len() {for sign in [-1.0,1.0] {
                    let mut z=x.clone();z[i]+=sign*step;
                    let old=f.loglik_at_impl(&z,false).unwrap();
                    let new=f.loglik_at_impl(&z,true).unwrap();
                    assert_eq!(old.to_bits(),new.to_bits(),"coordinate {i}");
                }}
            }
        }
        let start=Instant::now();
        let old=f.information_impl(0.001,1e-6,1e-8,30000,100000000,false).unwrap();
        let old_s=start.elapsed().as_secs_f64();
        let start=Instant::now();
        let new=f.information_impl(0.001,1e-6,1e-8,30000,100000000,true).unwrap();
        let new_s=start.elapsed().as_secs_f64();
        assert_eq!(old.matrix,new.matrix);
        assert_eq!(old.eigenvalues,new.eigenvalues);
        assert_eq!(old.rank,new.rank);
        assert_eq!(old.negative_eigenvalues,new.negative_eigenvalues);
        assert_eq!(old.objective_calls,new.objective_calls);
        assert_eq!(new.objective_calls,2*x.len()*x.len()+1);
        assert!(f.information(0.001,1e-6,1e-8,new.objective_calls-1,100000000).is_err());
        assert!(f.information(0.001,1e-6,1e-8,30000,1).is_err());
        eprintln!("PROFILE N=36 items=14 G+3 groups=3 quadpts=5 dimension={} objective_calls={} legacy_seconds={} likelihood_seconds={} speedup={} matrix_bitwise_equal=true rank={}",x.len(),new.objective_calls,old_s,new_s,old_s/new_s,new.rank);
        let alt=f.release_item(0).unwrap();
        for (fit,step) in [(&f,0.0005),(&alt,0.001),(&alt,0.0005)] {
            let old=fit.information_impl(step,1e-6,1e-8,30000,100000000,false).unwrap();
            let new=fit.information_impl(step,1e-6,1e-8,30000,100000000,true).unwrap();
            assert_eq!(old.matrix,new.matrix);
            assert_eq!(old.eigenvalues,new.eigenvalues);
            assert_eq!((old.rank,old.negative_eigenvalues,old.min_eigenvalue,old.threshold),(new.rank,new.negative_eigenvalues,new.min_eigenvalue,new.threshold));
            assert_eq!(old.objective_calls,new.objective_calls);
            eprintln!("EQUIVALENCE dimension={} step={} calls={} rank={} negative={} exact_matrix=true",new.dimension,step,new.objective_calls,new.rank,new.negative_eigenvalues);
        }
    }
}
