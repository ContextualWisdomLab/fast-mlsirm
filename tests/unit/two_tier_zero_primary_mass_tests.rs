//! Zero-primary-mass posterior regression.
//! Basis: Cai (2010), pp. 589-590, Eqs. 15-16; pp. 608-609, Appendix A.
//! Reference: Cai, L. (2010). A two-tier full-information item factor analysis
//! model with applications. Psychometrika, 75(4), 581-612.
//! doi:10.1007/s11336-010-9178-0.
mod zero_primary_mass_regression {
    use super::super::*;
    use crate::quadrature;

    fn fixture(
        qp: usize,
        qs: usize,
        mixed: bool,
    ) -> (
        Validated,
        Vec<usize>,
        Vec<ItemParams>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
    ) {
        let (coords, weights) = quadrature::require_gh_rule(qp, "q_primary").unwrap();
        let (specific, sw) = quadrature::require_gh_rule(qs, "q_specific").unwrap();
        let v = Validated {
            n_persons: 12,
            n_items: 4,
            n_primary: 1,
            n_specific: 1,
            n_cat: 3,
            m1: 2,
            grid_size: qp,
            free_primaries: vec![vec![0]; 4],
            blocks: vec![if mixed {
                vec![0, 1, 2]
            } else {
                vec![0, 1, 2, 3]
            }],
            specific_free: if mixed { vec![3] } else { vec![] },
            item_block: if mixed {
                vec![Some(0), Some(0), Some(0), None]
            } else {
                vec![Some(0); 4]
            },
        };
        let rows = [
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [1, 2, 0, 1],
            [1, 2, 0, 1],
            [1, 2, 0, 1],
            [2, 0, 1, 2],
            [2, 0, 1, 2],
            [2, 0, 1, 2],
            [0, 1, 2, 0],
            [0, 1, 2, 0],
            [0, 1, 2, 0],
        ];
        let y = rows.into_iter().flatten().collect();
        let params = (0..4)
            .map(|i| ItemParams {
                a_p: vec![[1.0, 1.2, 0.9, 1.1][i]],
                a_s: if mixed && i == 3 {
                    None
                } else {
                    Some([0.6, 0.7, 0.8, 0.5][i])
                },
                d: vec![[0.8, 1.0, 0.6, 0.9][i], [-0.8, -1.0, -0.6, -0.9][i]],
            })
            .collect();
        (
            v,
            y,
            params,
            coords.to_vec(),
            weights.iter().map(|w| w.ln()).collect(),
            specific.to_vec(),
            sw.iter().map(|w| w.ln()).collect(),
        )
    }
    fn check(qp: usize, qs: usize, mixed: bool, padding: bool, ordinary: bool) {
        let (mut v, y, params, mut coords, mut lw, ts, lws) = fixture(qp, qs, mixed);
        if padding {
            coords.insert(0, -43.0);
            lw.insert(0, f64::NEG_INFINITY);
            v.grid_size += 1;
        }
        let qg = lw.len();
        if qp == 481 || padding {
            assert!(lw.iter().any(|x| *x == f64::NEG_INFINITY));
        }
        if ordinary {
            let (ll, counts, moments) =
                e_step(&v, &y, None, &params, &lw, &lws, &coords, &ts, qg, qs);
            assert!(ll.is_finite());
            assert!(moments.iter().all(|x| x.is_finite()));
            assert!(
                counts.iter().flatten().flatten().all(|x| x.is_finite()),
                "ordinary counts contain NaN with primary Q{qp}"
            );
            for i in 0..4 {
                let mass: f64 = counts[i].iter().flatten().sum();
                assert!((mass - 12.0).abs() < 1e-10, "mass={mass}");
            }
        } else {
            let (ll, counts, m1, m2, s2, mass, eap, sd) =
                e_step_fipc_cpu(&v, &y, None, &params, &lw, &[lws], &coords, &[ts], qg, qs);
            assert!(ll.is_finite());
            assert!(
                s2.iter().all(|x| x.is_finite()),
                "FIPC specific moments contain NaN with primary Q{qp}"
            );
            assert!(counts.iter().flatten().flatten().all(|x| x.is_finite()));
            assert!([&m1, &m2, &mass, &eap, &sd]
                .into_iter()
                .all(|xs| xs.iter().all(|x| x.is_finite())));
            assert!((mass[0] - 12.0).abs() < 1e-10, "specific mass={}", mass[0]);
            for i in 0..4 {
                let total: f64 = counts[i].iter().flatten().sum();
                assert!((total - 12.0).abs() < 1e-10, "mass={total}");
            }
        }
    }
    #[test]
    fn fipc_primary481_axis_finite() {
        check(481, 241, false, false, false);
    }
    #[test]
    fn fipc_specific481_axis_control() {
        check(241, 481, false, false, false);
    }
    #[test]
    fn fipc_both481_finite() {
        check(481, 481, false, false, false);
    }
    #[test]
    fn ordinary_primary481_counts_finite() {
        check(481, 241, false, false, true);
    }
    #[test]
    fn fipc_zero_mass_padding_mixed_specific_free() {
        check(121, 121, true, true, false);
    }
    #[test]
    fn ordinary_zero_mass_padding_mixed_specific_free() {
        check(121, 121, true, true, true);
    }
    #[test]
    fn fipc_positive121_control() {
        check(121, 121, false, false, false);
    }
    #[test]
    fn ordinary_positive121_control() {
        check(121, 121, false, false, true);
    }

    #[test]
    fn fipc_zero_mass_removal_keeps_posterior_measure() {
        let (v, y, params, coords, lw, ts, lws) = fixture(481, 121, true);
        let active: Vec<usize> = lw
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.is_finite().then_some(i))
            .collect();
        assert!(active.len() < lw.len());
        let lc: Vec<f64> = active.iter().map(|&i| coords[i]).collect();
        let llw: Vec<f64> = active.iter().map(|&i| lw[i]).collect();
        let full = e_step_fipc_cpu(
            &v,
            &y,
            None,
            &params,
            &lw,
            &[lws.clone()],
            &coords,
            &[ts.clone()],
            lw.len(),
            ts.len(),
        );
        let trimmed = e_step_fipc_cpu(
            &v,
            &y,
            None,
            &params,
            &llw,
            &[lws],
            &lc,
            &[ts],
            active.len(),
            121,
        );
        assert!((full.0 - trimmed.0).abs() < 1e-10);
        for (a, b) in [
            (&full.2, &trimmed.2),
            (&full.3, &trimmed.3),
            (&full.4, &trimmed.4),
            (&full.5, &trimmed.5),
            (&full.6, &trimmed.6),
            (&full.7, &trimmed.7),
        ] {
            assert!(a
                .iter()
                .zip(b)
                .all(|(x, y)| x.is_finite() && y.is_finite() && (x - y).abs() < 1e-10));
        }
        for i in 0..v.n_items {
            let stride = if v.item_block[i].is_some() { 121 } else { 1 };
            for (ng, &g) in active.iter().enumerate() {
                for h in 0..stride {
                    for k in 0..v.n_cat {
                        assert!(
                            (full.1[i][g * stride + h][k] - trimmed.1[i][ng * stride + h][k]).abs()
                                < 1e-10
                        );
                    }
                }
            }
            for (g, l) in lw.iter().enumerate().filter(|(_, l)| !l.is_finite()) {
                let _ = l;
                for h in 0..stride {
                    assert!(full.1[i][g * stride + h].iter().all(|x| *x == 0.0));
                }
            }
        }
    }
}
