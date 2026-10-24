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
#[test]
fn specific_mean_rejects_unanchored_specific_block() {
    // Reviewer F1: block 0/general have anchors; block 1 is entirely free.
    // All declared categories occur in every group/item.
    let mut y = Vec::new();
    let mut groups = Vec::new();
    for g in 0..2 {
        for pattern in 0..243 {
            groups.push(g);
            for i in 0..5 {
                y.push((pattern / 3usize.pow(i)) % 3);
            }
        }
    }
    let cfg = BifactorMultigroupConfig {
        q_general: 3, q_specific: 3, max_iter: 1,
        estimate_specific_means: true, estimate_specific_vars: false,
        device: Device::Cpu, ..Default::default()
    };
    let err = fit_bifactor_grm_multigroup(
        &y, None, &groups, 2, &[0, 0, 1, 1, -1], groups.len(), 5, 2, 3,
        Some(&[true, true, false, false, true]), &cfg,
    ).expect_err("an entirely free block cannot link a focal specific mean");
    assert!(err.contains("specific-1") && err.contains("anchor"), "{err}");
    // Opt-out keeps the previous zero-specific-mean contract.
    assert!(fit_bifactor_grm_multigroup(
        &y, None, &groups, 2, &[0, 0, 1, 1, -1], groups.len(), 5, 2, 3,
        Some(&[true, true, false, false, true]),
        &BifactorMultigroupConfig { estimate_specific_means: false, ..cfg },
    ).is_ok());
}

#[test]
fn specific_mean_rejects_rank_deficient_joint_anchor_locations() {
    // Both items are common and the block has an anchor. Equal response
    // columns keep their fitted loading rows identical, so only one joint
    // location contrast is linked despite two anchor rows.
    let groups: Vec<usize> = (0..2).flat_map(|g| std::iter::repeat_n(g, 9)).collect();
    let y: Vec<usize> = (0..18).flat_map(|p| [p % 3, p % 3]).collect();
    let cfg = BifactorMultigroupConfig {
        q_general: 3, q_specific: 3, max_iter: 1,
        estimate_specific_means: true, device: Device::Cpu, ..Default::default()
    };
    let err = fit_bifactor_grm_multigroup(
        &y, None, &groups, 2, &[0, 0], groups.len(), 2, 1, 3, None, &cfg,
    ).expect_err("per-block anchor presence cannot prove joint location rank");
    assert!(err.contains("location") && err.contains("rank"), "{err}");
}

#[test]
fn specific_mean_rejects_unobserved_focal_block_anchor() {
    let (y, groups) = data();
    let mut obs = vec![true; y.len()];
    for (p, &g) in groups.iter().enumerate() {
        if g == 1 { obs[p * 4] = false; }
    }
    let cfg = BifactorMultigroupConfig {
        q_general: 3, q_specific: 3, max_iter: 1,
        estimate_specific_means: true, device: Device::Cpu, ..Default::default()
    };
    // A nominally common item absent from the focal data cannot link that
    // block's location; its free sibling still has complete category support.
    let err = fit_bifactor_grm_multigroup(&y, Some(&obs), &groups, 2,
        &[0, 0, -1, -1], groups.len(), 4, 1, 3,
        Some(&[true, false, true, true]), &cfg).unwrap_err();
    assert!(err.contains("group 1") && err.contains("location") && err.contains("rank"), "{err}");
}

#[test]
fn specific_mean_rejects_disjoint_reference_and_focal_anchor_observations() {
    let mut y = Vec::new();
    let mut groups = Vec::new();
    let mut obs = Vec::new();
    for g in 0..2 { for pattern in 0..81 {
        groups.push(g);
        for i in 0..8 {
            y.push((pattern / 3usize.pow((i % 4) as u32)) % 3);
            obs.push((i % 4 < 2) == (g == 0));
        }
    }}
    let cfg = BifactorMultigroupConfig {
        q_general: 3, q_specific: 3, max_iter: 1,
        estimate_specific_means: true, device: Device::Cpu, ..Default::default()
    };
    // Each group separately observes block + general-only anchors, but no
    // common item has data on both sides of the reference/focal link.
    let err = fit_bifactor_grm_multigroup(&y, Some(&obs), &groups, 2,
        &[0, 0, 0, 0, -1, -1, -1, -1], groups.len(), 8, 1, 3, None, &cfg)
        .expect_err("disjoint observed anchor banks cannot link locations");
    assert!(err.contains("location") && err.contains("rank"), "{err}");
}

// The ignored physical-GPU fit/terminal test lives in
// tests/unit/specific_mean_tests.rs, next to the private dispatch observer.
