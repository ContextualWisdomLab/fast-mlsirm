use super::*;

/// A consuming projection must not launch an unused specific-moment reduction.
/// Item counts and primary outputs remain original GPU bits.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
#[cfg(all(feature = "gpu", not(coverage)))]
fn omitted_specific_moments_preserve_consumed_gpu_bits() {
    let qg = 121;
    let qs = 241;
    let np = 12;
    let ni = 3;
    let ng = 2;
    let ns = 2;
    let nc = 3;
    let (tg, wg) = crate::quadrature::require_gh_rule(qg, "q_primary").unwrap();
    let (ts, ws) = crate::quadrature::require_gh_rule(qs, "q_specific").unwrap();
    let y: Vec<usize> = (0..np * ni).map(|i| (i / ni + i % ni) % nc).collect();
    let observed: Vec<bool> = (0..y.len()).map(|i| i % 7 != 0).collect();
    let groups: Vec<usize> = (0..np).map(|p| p % ng).collect();
    let block = [Some(0), Some(1), None];
    let blocks = vec![vec![0], vec![1]];
    let lwg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let lws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let primary = vec![tg.to_vec(); ng];
    let specific = vec![vec![ts.to_vec(); ns]; ng];
    for phase in 0..2 {
        let tables: Vec<Vec<Vec<f64>>> = (0..ng)
            .map(|group| {
                (0..ni)
                    .map(|i| {
                        let mut table = Vec::new();
                        for &t in tg {
                            for h in 0..if block[i].is_some() { qs } else { 1 } {
                                let base = 0.3 * group as f64
                                    + (0.8 + 0.1 * i as f64) * t
                                    + if block[i].is_some() { 0.6 * ts[h] } else { 0.0 }
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
            item_block: &block,
            blocks: &blocks,
            tg_groups: &primary,
            ts_groups: &specific,
            log_wg: &lwg,
            log_ws: &lws,
        };
        reset_gpu_dispatch_receipt();
        SPECIFIC_MOMENT_DISPATCHES.with(|v| v.set(0));
        let Some(full) = e_step_reduced_gpu(&inputs) else {
            assert!(gpu_dispatch_receipt().fallback_reason.is_some());
            return;
        };
        assert!(gpu_dispatch_receipt().used);
        let projected =
            e_step_reduced_gpu_consumed(&inputs, false).expect("same adapter and dimensions");
        assert_eq!(full.loglik.to_bits(), projected.loglik.to_bits());
        for (a, b) in [
            (&full.counts, &projected.counts),
            (&full.postg, &projected.postg),
            (&full.w_acc, &projected.w_acc),
            (&full.s1_g, &projected.s1_g),
            (&full.s2_g, &projected.s2_g),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert_eq!(
            SPECIFIC_MOMENT_DISPATCHES.with(|v| v.get()),
            1,
            "unconsumed specific moments still dispatched"
        );
        assert!(projected
            .w_spec
            .iter()
            .chain(&projected.s2_spec)
            .all(|x| *x == 0.0));
        eprintln!("CONSUMED_MOMENTS_GPU {:?}", gpu_dispatch_receipt());
    }
}
