//! Reduced expected-count ownership regression.
//! Basis: Cai (2010, pp. 608-609, Appendix A), category expected frequencies
//! at the same latent-node tuple. General-only items use primary posteriors;
//! a later block-only pass must not erase those counts.
//! Reference: Cai, L. (2010). A two-tier full-information item factor analysis
//! model with applications. Psychometrika, 75(4), 581-612.
//! doi:10.1007/s11336-010-9178-0.
use super::*;

#[cfg(all(feature = "gpu", not(coverage)))]
fn general_counts_preserved(qg: usize, qs: usize, require_hardware: bool) {
    let (tg, wg) = crate::quadrature::require_gh_rule(qg, "q_primary").unwrap();
    let (ts, ws) = crate::quadrature::require_gh_rule(qs, "q_specific").unwrap();
    let np = 12;
    let ni = 4;
    let nc = 3;
    let y: Vec<usize> = (0..np * ni)
        .map(|index| (index / ni + index % ni) % nc)
        .collect();
    let observed: Vec<bool> = (0..y.len()).map(|index| index % 7 != 0).collect();
    let groups: Vec<usize> = (0..np).map(|person| person % 2).collect();
    let item_block = [Some(0), None, Some(0), None];
    let blocks = vec![vec![0, 2]];
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let make_tables = || {
        item_block
            .iter()
            .map(|block| vec![-(nc as f64).ln(); qg * if block.is_some() { qs } else { 1 } * nc])
            .collect::<Vec<_>>()
    };
    let tables = vec![make_tables(), make_tables()];
    let primary_nodes = vec![tg.to_vec(), tg.to_vec()];
    let specific_nodes = vec![vec![ts.to_vec()], vec![ts.to_vec()]];
    let inputs = ReducedEstepInputs {
        y: &y,
        observed: Some(&observed),
        group_id: Some(&groups),
        n_persons: np,
        n_items: ni,
        n_specific: 1,
        n_cat: nc,
        qg,
        qs,
        n_groups: 2,
        tables_groups: &tables,
        item_block: &item_block,
        blocks: &blocks,
        tg_groups: &primary_nodes,
        ts_groups: &specific_nodes,
        log_wg: &log_wg,
        log_ws: &log_ws,
    };
    reset_gpu_dispatch_receipt();
    let Some(result) = e_step_reduced_gpu(&inputs) else {
        assert!(
            !require_hardware,
            "physical GPU required: {:?}",
            gpu_dispatch_receipt().fallback_reason
        );
        assert!(gpu_dispatch_receipt().fallback_reason.is_some());
        return;
    };
    let receipt = gpu_dispatch_receipt();
    assert!(receipt.used && receipt.backend.is_some() && receipt.device_name.is_some());
    let expected_weights_sum: f64 = wg.iter().sum();
    for group in 0..2 {
        for item in 0..ni {
            let nodes = if item_block[item].is_some() {
                qg * qs
            } else {
                qg
            };
            let base = (group * ni + item) * result.counts_stride_nodes * nc;
            let observed_count = (0..np)
                .filter(|&p| groups[p] == group && observed[p * ni + item])
                .count();
            let mass: f64 = result.counts[base..base + nodes * nc].iter().sum();
            assert!(mass.is_finite() && (mass - observed_count as f64).abs() <= 1e-5,
                "expected-count observed mass erased: group={group}, item={item}, mass={mass}, expected={observed_count}");
            if item_block[item].is_none() {
                for node in 0..qg {
                    for category in 0..nc {
                        let count = (0..np)
                            .filter(|&p| {
                                groups[p] == group
                                    && observed[p * ni + item]
                                    && y[p * ni + item] == category
                            })
                            .count();
                        let expected = count as f64 * wg[node] / expected_weights_sum;
                        let actual = result.counts[base + node * nc + category];
                        assert!(actual.is_finite() && (actual - expected).abs() <= 1e-6,
                            "general-only posterior count changed: group={group}, item={item}, node={node}, category={category}");
                    }
                }
            }
        }
    }
    eprintln!(
        "expected-count ownership Q{qg}/Q{qs}: backend={:?};device={:?};fallback={:?}",
        receipt.backend, receipt.device_name, receipt.fallback_reason
    );
}

#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn mixed_general_block_counts_keep_observed_mass() {
    general_counts_preserved(121, 241, false);
    general_counts_preserved(241, 121, false);
}

#[test]
#[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
#[cfg(all(feature = "gpu", not(coverage)))]
fn mixed_general_block_counts_physical_gpu_gate() {
    general_counts_preserved(121, 241, true);
    general_counts_preserved(241, 121, true);
}

/// Checks moment accumulation with likelihood-independent posterior weights.
///
/// Basis: Higham (1993, pp. 787-788, Eqs. 3.3-3.6) analyzes addition trees;
/// the posterior normalization and moments follow Cai (2010, pp. 608-609,
/// Appendix A). The arithmetic-only comparison bound is not a fit-acceptance
/// threshold and does not certify an arbitrary quadrature approximation.
///
/// References:
/// Cai, L. (2010). A two-tier full-information item factor analysis model with
/// applications. Psychometrika, 75(4), 581-612. doi:10.1007/s11336-010-9178-0.
/// Higham, N. J. (1993). The accuracy of floating point summation. SIAM Journal
/// on Scientific Computing, 14(4), 783-799. doi:10.1137/0914050.
#[cfg(all(feature = "gpu", not(coverage)))]
fn prior_moments_preserved(qg: usize, qs: usize, require_hardware: bool) {
    let (tg, wg) = crate::quadrature::require_gh_rule(qg, "q_primary").unwrap();
    let (ts, ws) = crate::quadrature::require_gh_rule(qs, "q_specific").unwrap();
    let np = 48;
    let ni = 3;
    let nc = 3;
    let ng = 3;
    let ns = 2;
    let y: Vec<usize> = (0..np * ni).map(|i| (i / ni + i % ni) % nc).collect();
    let groups: Vec<usize> = (0..np).map(|p| p % 2).collect();
    let observed: Vec<bool> = (0..np * ni)
        .map(|i| !(i % ni == 1 && groups[i / ni] == 1) && i % 11 != 0)
        .collect();
    let item_block = [Some(0), Some(1), None];
    let blocks = vec![vec![0], vec![1]];
    let log_wg: Vec<f64> = wg.iter().map(|x| x.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|x| x.ln()).collect();
    let tables: Vec<Vec<Vec<f64>>> = (0..ng)
        .map(|_| {
            item_block
                .iter()
                .map(|b| vec![-(nc as f64).ln(); qg * if b.is_some() { qs } else { 1 } * nc])
                .collect()
        })
        .collect();
    let primary: Vec<Vec<f64>> = (0..ng)
        .map(|g| {
            tg.iter()
                .map(|t| [0.35, -0.65, 0.0][g] + [0.8, 1.25, 1.0][g] * t)
                .collect()
        })
        .collect();
    let specific: Vec<Vec<Vec<f64>>> = (0..ng)
        .map(|g| {
            (0..ns)
                .map(|s| {
                    ts.iter()
                        .map(|t| [-0.4, 1.1][s] + (0.8 + 0.2 * g as f64) * t)
                        .collect()
                })
                .collect()
        })
        .collect();
    let inputs = ReducedEstepInputs {
        y: &y,
        observed: Some(&observed),
        group_id: Some(&groups),
        n_persons: np,
        n_items: ni,
        n_specific: ns,
        n_cat: nc,
        qg,
        qs,
        n_groups: ng,
        tables_groups: &tables,
        item_block: &item_block,
        blocks: &blocks,
        tg_groups: &primary,
        ts_groups: &specific,
        log_wg: &log_wg,
        log_ws: &log_ws,
    };
    reset_gpu_dispatch_receipt();
    let Some(result) = e_step_reduced_gpu(&inputs) else {
        assert!(
            !require_hardware,
            "physical GPU required: {:?}",
            gpu_dispatch_receipt()
        );
        assert!(gpu_dispatch_receipt().fallback_reason.is_some());
        return;
    };
    let receipt = gpu_dispatch_receipt();
    assert!(receipt.used && receipt.backend.is_some() && receipt.device_name.is_some());
    let wgtotal: f64 = wg.iter().sum();
    let wstotal: f64 = ws.iter().sum();
    for g in 0..ng {
        let mass = groups.iter().filter(|&&id| id == g).count() as f64;
        let first = mass
            * primary[g]
                .iter()
                .zip(wg.iter())
                .map(|(t, w)| t * w)
                .sum::<f64>()
            / wgtotal;
        let second = mass
            * primary[g]
                .iter()
                .zip(wg.iter())
                .map(|(t, w)| t * t * w)
                .sum::<f64>()
            / wgtotal;
        for (name, actual, expected) in [
            ("primary mass", result.w_acc[g], mass),
            ("primary first", result.s1_g[g], first),
            ("primary second", result.s2_g[g], second),
        ] {
            let bound = 8.0 * f64::from(f32::EPSILON) * expected.abs().max(mass).max(1.0);
            assert!(actual.is_finite() && (actual-expected).abs() <= bound,
                "{name} reduction loss: group={g}, actual={actual}, expected={expected}, bound={bound}");
        }
        for s in 0..ns {
            let mass = (0..np)
                .filter(|&p| groups[p] == g && blocks[s].iter().any(|&i| observed[p * ni + i]))
                .count() as f64;
            let second = mass
                * specific[g][s]
                    .iter()
                    .zip(ws.iter())
                    .map(|(t, w)| t * t * w)
                    .sum::<f64>()
                / wstotal;
            for (name, actual, expected) in [
                ("specific mass", result.w_spec[g * ns + s], mass),
                ("specific second", result.s2_spec[g * ns + s], second),
            ] {
                let bound = 8.0 * f64::from(f32::EPSILON) * expected.abs().max(1.0);
                assert!(actual.is_finite() && (actual-expected).abs() <= bound,
                    "{name} reduction loss: group={g}, block={s}, actual={actual}, expected={expected}, bound={bound}");
                if mass == 0.0 {
                    assert_eq!(actual, 0.0);
                }
            }
        }
    }
    eprintln!("moment reduction Q{qg}/Q{qs}: {:?}", receipt);
}

#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn grouped_masked_high_q_moments_keep_prior_mass() {
    prior_moments_preserved(121, 241, false);
    prior_moments_preserved(241, 121, false);
}

#[test]
#[ignore = "requires an actual GPU adapter; fallback is an unmet hardware gate"]
#[cfg(all(feature = "gpu", not(coverage)))]
fn grouped_masked_high_q_moments_physical_gpu_gate() {
    prior_moments_preserved(121, 241, true);
    prior_moments_preserved(241, 121, true);
}
