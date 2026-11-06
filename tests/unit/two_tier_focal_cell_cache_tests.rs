mod focal_cell_cache_regression {
    use super::super::*;
    /// Finite probability-cell reuse, not a timing or study acceptance test.
    /// Cai (2010, pp.589–590,Eqs.11–16; pp.608–609,Appendix A).
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika,75(4),581–612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn focal_cells_evaluated_no_more_than_current_node_category_table() {
        let q = 121usize;
        let (x, w) = gh_rule(q).unwrap();
        let v = Validated {
            n_persons: 12,
            n_items: 4,
            n_primary: 1,
            n_specific: 1,
            n_cat: 3,
            m1: 2,
            grid_size: q,
            free_primaries: vec![vec![0]; 4],
            blocks: vec![vec![0, 1, 2]],
            specific_free: vec![3],
            item_block: vec![Some(0), Some(0), Some(0), None],
        };
        let y: Vec<usize> = (0..48).map(|i| (i / 4 + i % 4) % 3).collect();
        let params: Vec<ItemParams> = (0..4)
            .map(|i| ItemParams {
                a_p: vec![1.0 + i as f64 * 0.1],
                a_s: if i == 3 { None } else { Some(0.8) },
                d: vec![0.8, -0.8],
            })
            .collect();
        let lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
        CATEGORY_EVALUATIONS.with(|c| c.set(0));
        CATEGORY_EVALUATIONS_ACTIVE.with(|c| c.set(true));
        let out = e_step_fipc_cpu(
            &v,
            &y,
            None,
            &params,
            &lw,
            &[lw.clone()],
            x,
            &[x.to_vec()],
            q,
            q,
        );
        CATEGORY_EVALUATIONS_ACTIVE.with(|c| c.set(false));
        let actual = CATEGORY_EVALUATIONS.with(|c| c.get());
        let bound = (3 * q * q + q) * 3;
        assert!(out.0.is_finite());
        assert!(
            actual <= bound,
            "focal repeat category calls={actual}; current node/category upperbound={bound}"
        );
    }
    /// Exact probability cache parity on shifted support with block missingness.
    /// Basis: Cai (2010,pp.589–590,Eqs.11–16;pp.608–609,Appendix A).
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika,75(4),581–612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn focal_cache_matches_original_all_outputs_and_rebuilds() {
        let q = 121usize;
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
        let mask: Vec<bool> = (0..48)
            .map(|i| i % 7 != 0 && !(i / 4 == 1 && i % 4 < 3))
            .collect();
        let mut params: Vec<ItemParams> = (0..4)
            .map(|i| ItemParams {
                a_p: vec![1.0 + i as f64 * 0.1],
                a_s: if i == 3 { None } else { Some(-0.8) },
                d: vec![0.8, -0.8],
            })
            .collect();
        let mut coords: Vec<f64> = x.iter().map(|t| 0.35 + 1.2 * t).collect();
        coords.insert(0, -43.0);
        let mut lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
        lw.insert(0, f64::NEG_INFINITY);
        let ts = vec![x.iter().map(|t| t * 0.75).collect::<Vec<_>>()];
        let lws = vec![w.iter().map(|v| v.ln()).collect::<Vec<_>>()];
        for phase in 0..2 {
            let a = e_step_fipc_cpu(
                &v,
                &y,
                Some(&mask),
                &params,
                &lw,
                &lws,
                &coords,
                &ts,
                q + 1,
                q,
            );
            let b = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
                &v,
                &y,
                Some(&mask),
                &params,
                &lw,
                &lws,
                &coords,
                &ts,
                q + 1,
                q,
            );
            assert_eq!(a.0.to_bits(), b.0.to_bits(), "LL phase{phase}");
            assert_eq!(a.1, b.1, "counts phase{phase}");
            for (a, b) in [
                (&a.2, &b.2),
                (&a.3, &b.3),
                (&a.4, &b.4),
                (&a.5, &b.5),
                (&a.6, &b.6),
                (&a.7, &b.7),
            ] {
                assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
            }
            params[0].d[0] += 0.125;
            coords[10] += 0.25;
        }
    }
}
