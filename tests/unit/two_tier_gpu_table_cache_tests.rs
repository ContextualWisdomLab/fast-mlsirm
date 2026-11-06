use super::*;

/// Table entries are original binary64 cells even when primary predictors repeat.
/// Basis: Cai (2010, p. 589, Eqs. 11–12; pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn gpu_table_preparation_reuses_predictors_but_preserves_all_table_bits() {
    let q = 121;
    let (x, _) = gh_rule(q).unwrap();
    let ng = q * q;
    let v = Validated {
        n_persons: 1,
        n_items: 4,
        n_primary: 2,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: ng,
        free_primaries: vec![vec![0, 1], vec![1], vec![0, 1], vec![1]],
        blocks: vec![vec![0, 1], vec![2]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(1), None],
    };
    let mut coords: Vec<f64> = (0..ng)
        .flat_map(|g| [0.3 + 1.2 * x[g / q], -0.2 + 0.8 * x[g % q]])
        .collect();
    let mut ts = vec![
        x.iter().map(|t| 0.75 * t).collect::<Vec<_>>(),
        x.iter().map(|t| 1.3 * t).collect::<Vec<_>>(),
    ];
    let mut params = vec![
        ItemParams {
            a_p: vec![-1.1, 0.],
            a_s: Some(-0.8),
            d: vec![0.8, -0.8],
        },
        ItemParams {
            a_p: vec![0., 1.2],
            a_s: Some(0.6),
            d: vec![0.9, -0.7],
        },
        ItemParams {
            a_p: vec![1.3, 0.],
            a_s: Some(0.5),
            d: vec![1.0, -0.6],
        },
        ItemParams {
            a_p: vec![0., -0.7],
            a_s: None,
            d: vec![0.7, -0.9],
        },
    ];
    for phase in 0..5 {
        CATEGORY_EVALUATIONS_ACTIVE.with(|c| c.set(false));
        let expected: Vec<Vec<f64>> = (0..v.n_items)
            .map(|i| {
                let mut table = Vec::new();
                for g in 0..ng {
                    for h in 0..if v.item_block[i].is_some() { q } else { 1 } {
                        for cat in 0..v.n_cat {
                            table.push(item_cat_logprob_fipc(
                                &v, &params, &coords, &ts, i, g, h, cat,
                            ));
                        }
                    }
                }
                table
            })
            .collect();
        let unique: Vec<usize> = (0..v.n_items)
            .map(|i| {
                (0..ng)
                    .map(|g| item_primary_base(&v, &params[i], &coords, g, i).to_bits())
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            })
            .collect();
        if phase >= 3 {
            let actual_keys: std::collections::HashSet<u64> = (0..ng)
                .map(|g| item_primary_base(&v, &params[0], &coords, g, 0).to_bits())
                .collect();
            let mut without_crossloading = params[0].clone();
            without_crossloading.a_p[1] = 0.0;
            let zero_keys: std::collections::HashSet<u64> = (0..ng)
                .map(|g| item_primary_base(&v, &without_crossloading, &coords, g, 0).to_bits())
                .collect();
            assert!(
                actual_keys.len() > zero_keys.len(),
                "GPU-TABLE-TEST-CROSSLOAD-1: second coefficient did not change predictor keys"
            );
            assert!(
                (0..ng).any(
                    |g| item_primary_base(&v, &params[0], &coords, g, 0).to_bits()
                        != item_primary_base(&v, &without_crossloading, &coords, g, 0).to_bits()
                ),
                "GPU-TABLE-TEST-CROSSLOAD-1: cross-loading coefficient is ignored"
            );
        }
        let bound: usize = unique
            .iter()
            .enumerate()
            .map(|(i, n)| n * if v.item_block[i].is_some() { q } else { 1 } * v.n_cat)
            .sum();
        CATEGORY_EVALUATIONS.with(|c| c.set(0));
        CATEGORY_EVALUATIONS_ACTIVE.with(|c| c.set(true));
        let actual = focal_gpu_tables(&v, &params, &coords, &ts, ng, q);
        CATEGORY_EVALUATIONS_ACTIVE.with(|c| c.set(false));
        let evaluated = CATEGORY_EVALUATIONS.with(|c| c.get());
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(&expected) {
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        assert!(evaluated<=bound,"GPU table category evaluations={evaluated} exact-predictor bound={bound} phase={phase}");
        coords[10] += 0.125;
        ts[1][10] += 0.25;
        params[0].d[0] += 0.1;
        if phase == 2 {
            // Cross-loadings and correlated support exercise near-full-key cardinality.
            params[0].a_p[1] = 0.37;
            params[2].a_s = None;
            for g in 0..ng {
                coords[g * 2 + 1] += 0.2 * coords[g * 2];
            }
        }
        if phase == 3 {
            params[1].a_p = [0., 0.].to_vec();
            params[1].a_s = Some(0.);
            coords[0] = -0.;
            coords[1] = 0.;
        }
    }
}
