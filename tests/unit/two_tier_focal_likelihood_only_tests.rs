use super::*;

/// Scalar observed-data likelihood does not need posterior count tables.
/// Basis: Cai (2010, pp. 589–590, Eqs. 15–16; pp. 607–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn scalar_likelihood_omits_count_tables_and_preserves_likelihood_bits() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 4,
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
    let y: Vec<usize> = (0..16).map(|i| (i / 4 + i % 4) % 3).collect();
    let observed: Vec<bool> = (0..16)
        .map(|i| !(i / 4 == 1 && i % 4 < 3) && i % 7 != 0)
        .collect();
    let mut params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![if i == 0 { -1.1 } else { 1.0 + i as f64 * 0.1 }],
            a_s: if i == 3 { None } else { Some(-0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let mut base = x.to_vec();
    base.insert(0, -43.0);
    let mut lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    lw.insert(0, f64::NEG_INFINITY);
    let lws: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    for phase in 0..4 {
        let mask = if phase % 2 == 0 {
            Some(observed.as_slice())
        } else {
            None
        };
        let mean = vec![0.35 + 0.1 * phase as f64];
        let covariance = vec![1.44];
        let sd = vec![0.75];
        let coords = fipc_primary_coords(&base, &mean, &[1.2], 1, q + 1);
        let ts = vec![x.iter().map(|v| 0.75 * v).collect::<Vec<_>>()];
        let legacy = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
            &v,
            &y,
            mask,
            &params,
            &lw,
            &[lws.clone()],
            &coords,
            &ts,
            q + 1,
            q,
        )
        .0;
        FOCAL_COUNT_ROWS.with(|c| c.set(0));
        let actual = direct_fipc_loglik(
            &v,
            &y,
            mask,
            &params,
            &base,
            &lw,
            &lws,
            x,
            &mean,
            &covariance,
            &sd,
            q + 1,
            crate::Device::Cpu,
        )
        .unwrap();
        let rows = FOCAL_COUNT_ROWS.with(|c| c.get());
        assert_eq!(
            actual.to_bits(),
            legacy.to_bits(),
            "scalar LL phase {phase}"
        );
        assert_eq!(
            rows, 0,
            "scalar likelihood constructed {rows} unused posterior count rows"
        );
        params[0].d[0] += 0.125;
        base[10] += 0.25;
    }
}
