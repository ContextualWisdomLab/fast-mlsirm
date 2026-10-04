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
