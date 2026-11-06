use super::*;

/// Repeated block responses require one identical marginal per current support.
/// Basis: Cai (2010, pp. 589–590, Eqs. 15–16; pp. 607–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn scalar_block_marginals_reuse_observed_patterns_with_exact_likelihood() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 12,
        n_items: 4,
        n_primary: 1,
        n_specific: 1,
        n_cat: 3,
        m1: 2,
        grid_size: q + 1,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1, 2]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(0), None],
    };
    let y: Vec<usize> = (0..48).map(|i| (i / 4 + i % 4) % 3).collect();
    let observed: Vec<bool> = (0..48)
        .map(|i| !(i / 4 % 4 == 1 && i % 4 < 3) && i % 7 != 0)
        .collect();
    let mut params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![if i == 0 { -1.1 } else { 1.0 + i as f64 * 0.1 }],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut coords: Vec<f64> = x.iter().map(|v| 0.35 + 1.2 * v).collect();
    coords.insert(0, -43.0);
    let mut lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    lw.insert(0, f64::NEG_INFINITY);
    let mut ts = vec![x.iter().map(|v| 0.75 * v).collect::<Vec<_>>()];
    let lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>()];
    for phase in 0..4 {
        let mask = if phase % 2 == 0 {
            Some(observed.as_slice())
        } else {
            None
        };
        let signatures: std::collections::HashSet<Vec<Option<usize>>> = (0..v.n_persons)
            .map(|pp| {
                v.blocks[0]
                    .iter()
                    .map(|&i| {
                        if mask.is_none_or(|m| m[pp * v.n_items + i]) {
                            Some(y[pp * v.n_items + i])
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let legacy = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
            &v,
            &y,
            mask,
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            q + 1,
            q,
        )
        .0;
        BLOCK_MARGINAL_EVALUATIONS.with(|c| c.set(0));
        BLOCK_MARGINAL_EVALUATIONS_ACTIVE.with(|c| c.set(true));
        let actual = focal_loglik_cpu(&v, &y, mask, &params, &lw, &lws, &coords, &ts, q + 1, q);
        BLOCK_MARGINAL_EVALUATIONS_ACTIVE.with(|c| c.set(false));
        let actual_cells = BLOCK_MARGINAL_EVALUATIONS.with(|c| c.get());
        assert_eq!(actual.to_bits(), legacy.to_bits(), "LL phase{phase}");
        let expected = signatures.len() * (q + 1);
        assert_eq!(
            actual_cells, expected,
            "block repeats evaluated {actual_cells}; distinct current signatures need {expected}"
        );
        params[0].d[0] += 0.125;
        coords[10] += 0.25;
        ts[0][10] += 0.125;
    }
}
