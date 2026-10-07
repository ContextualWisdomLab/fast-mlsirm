use super::*;

/// Scratch-only equivalence discriminator for compacted group/mask/tail leaves.
/// Basis: Higham (1993, pp. 783, 797), pairwise summation context.
/// This partition is an implementation hypothesis, not a new statistical model.
/// Reference: Higham, N. J. (1993). The accuracy of floating point summation.
/// SIAM Journal on Scientific Computing, 14(4), 783–799. doi:10.1137/0914050.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn specific_leaf_dispatch_is_partitioned() {
    let (qg, qs, np, ni, ng, ns, nc) = (121, 241, 13, 3, 3, 3, 3);
    let (tg, wg) = crate::quadrature::require_gh_rule(qg, "q_primary").unwrap();
    let (ts, ws) = crate::quadrature::require_gh_rule(qs, "q_specific").unwrap();
    let y: Vec<usize> = (0..np * ni).map(|i| (i / ni + i % ni) % nc).collect();
    let groups: Vec<usize> = (0..np).map(|p| p % 2).collect();
    let block = [Some(0), Some(1), None];
    let blocks = vec![vec![0], vec![1], vec![]];
    let lwg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let lws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let primary: Vec<Vec<f64>> = (0..ng)
        .map(|g| {
            tg.iter()
                .map(|t| t * (1.0 + 0.1 * g as f64) + 0.2 * g as f64)
                .collect()
        })
        .collect();
    let specific: Vec<Vec<Vec<f64>>> = (0..ng)
        .map(|g| {
            (0..ns)
                .map(|s| {
                    ts.iter()
                        .map(|t| t * (1.0 + 0.05 * s as f64) + 0.1 * g as f64)
                        .collect()
                })
                .collect()
        })
        .collect();
    for phase in 0..4 {
        let observed: Vec<bool> = (0..y.len())
            .map(|i| match phase {
                0 => true,
                1 => i % 7 != 0,
                2 => i / ni % 3 != 0 && i % ni != 1,
                _ => false,
            })
            .collect();
        let tables: Vec<Vec<Vec<f64>>> = (0..ng)
            .map(|group| {
                (0..ni)
                    .map(|i| {
                        let mut table = Vec::new();
                        for &t in &primary[group] {
                            for h in 0..if block[i].is_some() { qs } else { 1 } {
                                let base = 0.3 * group as f64
                                    + (0.8 + 0.1 * i as f64) * t
                                    + block[i].map_or(0.0, |s| 0.6 * specific[group][s][h])
                                    + 0.125 * phase as f64;
                                table.extend(crate::poly::grm_logprobs(base, &[0.8, -0.8]));
                            }
                        }
                        table
                    })
                    .collect()
            })
            .collect();
        let inputs = ReducedEstepInputs {
            y: &y,
            observed: if phase == 0 { None } else { Some(&observed) },
            group_id: Some(&groups),
            n_persons: np,
            n_items: ni,
            n_specific: ns,
            n_cat: nc,
            qg,
            qs,
            n_groups: ng,
            tables_groups: &tables,
            item_block: &block,
            blocks: &blocks,
            tg_groups: &primary,
            ts_groups: &specific,
            log_wg: &lwg,
            log_ws: &lws,
        };
        reset_gpu_dispatch_receipt();
        ORIGINAL_SPECIFIC_TREE.with(|v| v.set(true));
        let Some(original) = e_step_reduced_gpu(&inputs) else {
            ORIGINAL_SPECIFIC_TREE.with(|v| v.set(false));
            assert!(gpu_dispatch_receipt().fallback_reason.is_some());
            return;
        };
        let original_span = SPECIFIC_LEAF_SPAN.with(|v| v.get());
        ORIGINAL_SPECIFIC_TREE.with(|v| v.set(false));
        SPECIFIC_LEAF_SPAN.with(|v| v.set(0));
        let full = e_step_reduced_gpu(&inputs).expect("same GPU must retain compatible dispatch");
        let receipt = gpu_dispatch_receipt();
        assert!(receipt.used && receipt.fallback_reason.is_none());
        assert_eq!(original.loglik.to_bits(), full.loglik.to_bits());
        for (label, a, b) in [
            ("counts", &original.counts, &full.counts),
            ("postg", &original.postg, &full.postg),
            ("w_acc", &original.w_acc, &full.w_acc),
            ("s1_g", &original.s1_g, &full.s1_g),
            ("s2_g", &original.s2_g, &full.s2_g),
            ("s2_spec", &original.s2_spec, &full.s2_spec),
            ("w_spec", &original.w_spec, &full.w_spec),
        ] {
            assert_eq!(a.len(), b.len(), "{label} phase={phase}");
            for (i, (a, b)) in a.iter().zip(b).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{label} index={i} phase={phase}");
            }
        }
        let span = SPECIFIC_LEAF_SPAN.with(|v| v.get());
        assert!(
            span > 0 && span <= 256,
            "specific leaf dispatch still has an unpartitioned {span}-leaf serial span"
        );
        assert_eq!(original_span, np * qg * qs);
        eprintln!("ACTUAL_SPECIFIC_LEAF_SPAN original={original_span} partitioned={span} phase={phase} allbits=true backend={:?} device={:?}",receipt.backend,receipt.device_name);
        let omitted = e_step_reduced_gpu_consumed(&inputs, false).expect("same adapter");
        assert_eq!(omitted.loglik.to_bits(), full.loglik.to_bits());
        assert_eq!(omitted.counts, full.counts);
        assert_eq!(omitted.postg, full.postg);
        assert!(omitted
            .s2_spec
            .iter()
            .chain(&omitted.w_spec)
            .all(|v| v.to_bits() == 0));
    }
}

/// Checked payload boundaries are storage/index guards, not Q caps.
/// Basis: Higham (1993, pp. 783, 797), tree partition only.
/// Reference: Higham, N. J. (1993). The accuracy of floating point summation.
/// SIAM Journal on Scientific Computing, 14(4), 783–799. doi:10.1137/0914050.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn specific_tree_layout_checks_payload_without_quadrature_cap() {
    for q in [121usize, 241, 481] {
        let leaves = 13 * q * q;
        let tiles = leaves.div_ceil(256);
        let counts = 3 * 3 * q * q * 3;
        assert_eq!(
            specific_tree_layout(13, q, q, 3, 3, 3, 3, true),
            Some((tiles, counts + 18 * tiles))
        );
        assert_eq!(
            specific_tree_layout(13, q, q, 3, 3, 3, 3, false),
            Some((0, counts))
        );
    }
    for leaves in [255, 256, 257, 511, 512, 513] {
        assert_eq!(
            specific_tree_layout(leaves, 1, 1, 1, 1, 1, 1, true),
            Some((leaves.div_ceil(256), 1 + 2 * leaves.div_ceil(256)))
        );
    }
    assert_eq!(
        specific_tree_layout(usize::MAX, 2, 1, 1, 1, 1, 1, true),
        None
    );
    assert_eq!(
        specific_tree_layout(1, 1, 1, usize::MAX, 1, 2, 1, true),
        None
    );
    assert_eq!(
        specific_tree_layout(u32::MAX as usize, 1, 1, 1, 1, 1, 1, true),
        None
    );
    assert_eq!(
        specific_tree_layout(0, 121, 241, 3, 3, 3, 3, true),
        Some((0, 3 * 3 * 121 * 241 * 3))
    );
}
