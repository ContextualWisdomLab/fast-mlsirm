use super::*;

/// Identical ordered block patterns have identical marginal integrals.
/// Keep individual posterior moment addition order and support changes intact.
/// Basis: Cai (2010, pp. 607–608, Appendix A, definition of E_is).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn fixed_block_marginals_reuse_identical_patterns_without_changing_moment_bits() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 8,
        n_items: 4,
        n_primary: 1,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: q + 1,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1, 2], vec![]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(0), None],
    };
    let y: Vec<usize> = (0..32).map(|i| (i / 4 + i % 4) % 3).collect();
    let mask: Vec<bool> = (0..32)
        .map(|i| !(i / 4 % 4 == 1 && i % 4 < 3) && i % 7 != 0)
        .collect();
    let mut params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![if i == 0 { -1.1 } else { 1.0 + i as f64 * 0.1 }],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut coords: Vec<f64> = x.iter().map(|t| 0.35 + 1.2 * t).collect();
    coords.insert(0, -43.0);
    let mut lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    lw.insert(0, f64::NEG_INFINITY);
    let mut lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>(); 2];
    let mut ts = vec![x.iter().map(|t| 0.75 * t).collect::<Vec<_>>(); 2];
    for phase in 0..4 {
        let observed = if phase % 2 == 0 {
            Some(mask.as_slice())
        } else {
            None
        };
        let legacy = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
            &v,
            &y,
            observed,
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            q + 1,
            q,
        );
        FIXED_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.set(0));
        let actual = fixed_fipc_e_step(
            &v,
            &y,
            observed,
            &params,
            &coords,
            &lw,
            &lws,
            &ts,
            q + 1,
            q,
            crate::Device::Cpu,
        );
        let calls = FIXED_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.get());
        assert_eq!(actual.0.to_bits(), legacy.0.to_bits(), "LL phase{phase}");
        for (a, b) in [
            (&actual.1, &legacy.2),
            (&actual.2, &legacy.3),
            (&actual.3, &legacy.4),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "moments phase{phase}"
            );
        }
        let patterns: usize = v
            .blocks
            .iter()
            .map(|members| {
                let distinct: std::collections::HashSet<Vec<Option<usize>>> = (0..v.n_persons)
                    .map(|pp| {
                        members
                            .iter()
                            .map(|&i| {
                                if observed.is_none_or(|o| o[pp * v.n_items + i]) {
                                    Some(y[pp * v.n_items + i])
                                } else {
                                    None
                                }
                            })
                            .collect()
                    })
                    .collect();
                distinct.len()
            })
            .sum();
        let bound = patterns * (q + 1);
        assert_eq!(
            calls, bound,
            "fixed block LSE repeated {calls} times; distinct bound {bound}; phase{phase}"
        );
        params[0].d[0] += 0.125;
        coords[10] += 0.25;
        ts[0][10] += 0.125;
        lws[0][10] -= 0.01;
        lw[10] -= 0.015;
    }
}

/// P2 correlated and shifted support parity is existing-behavior characterization.
/// Basis: Cai (2010, pp. 607–608, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn fixed_marginal_p2_matches_exact_predecessor_on_changed_support() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let n_grid = q * q;
    let v = Validated {
        n_persons: 2,
        n_items: 4,
        n_primary: 2,
        n_specific: 1,
        n_cat: 3,
        m1: 2,
        grid_size: n_grid,
        free_primaries: vec![vec![0, 1]; 4],
        blocks: vec![vec![0, 1, 2]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(0), None],
    };
    let y = vec![0, 1, 2, 0, 0, 1, 2, 2];
    let params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![-1.1, 0.7 + i as f64 * 0.1],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut coords = Vec::new();
    let mut lw = Vec::new();
    for g in 0..q {
        for h in 0..q {
            coords.extend_from_slice(&[0.3 + 1.2 * x[g], -0.2 + 0.25 * x[g] + 0.9 * x[h]]);
            lw.push(w[g].ln() + w[h].ln());
        }
    }
    let mut ts = vec![x.iter().map(|t| 0.75 * t).collect::<Vec<_>>()];
    let lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>()];
    for phase in 0..2 {
        let previous = fixed_block_marginal_legacy::legacy_fixed_fipc_moments_cpu(
            &v, &y, None, &params, &lw, &lws, &coords, &ts, n_grid, q,
        );
        FIXED_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.set(0));
        let actual =
            fixed_fipc_moments_cpu(&v, &y, None, &params, &lw, &lws, &coords, &ts, n_grid, q);
        assert_eq!(actual.0.to_bits(), previous.0.to_bits());
        for (a, b) in [
            (&actual.1, &previous.1),
            (&actual.2, &previous.2),
            (&actual.3, &previous.3),
        ] {
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "P2 phase{phase}"
            );
        }
        assert_eq!(FIXED_BLOCK_MARGINAL_EVALUATIONS.with(|c| c.get()), n_grid);
        coords[10] += 0.125;
        ts[0][10] += 0.1;
        lw[10] -= 0.01;
    }
}
