use super::*;

/// Fixed-support diagnostic consumes LL and three population moments, not counts.
/// Basis: Cai (2010, pp. 589–590, Eqs. 15–16; pp. 607–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn fixed_moments_omit_counts_and_preserve_all_consumed_bits() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 4,
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
    let y: Vec<usize> = (0..16).map(|i| (i / 4 + i % 4) % 3).collect();
    let mask: Vec<bool> = (0..16)
        .map(|i| !(i / 4 == 1 && i % 4 < 3) && i % 7 != 0)
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
        FOCAL_COUNT_ROWS.with(|c| c.set(0));
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
        let rows = FOCAL_COUNT_ROWS.with(|c| c.get());
        assert_eq!(actual.0.to_bits(), legacy.0.to_bits(), "LL phase{phase}");
        for (a, b) in [
            (&actual.1, &legacy.2),
            (&actual.2, &legacy.3),
            (&actual.3, &legacy.4),
        ] {
            assert_eq!(a.len(), b.len());
            assert!(
                a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()),
                "moment phase{phase}"
            );
        }
        assert_eq!(
            rows, 0,
            "fixed diagnostic constructed {rows} discarded posterior rows"
        );
        params[0].d[0] += 0.125;
        coords[10] += 0.25;
        ts[0][10] += 0.125;
        lws[0][10] -= 0.01;
    }
}
