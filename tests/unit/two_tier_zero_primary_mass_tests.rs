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

    #[test]
    fn selected_category_preserves_all_vector_cells_bitwise() {
        let banks: Vec<Vec<f64>> = vec![
            vec![],
            vec![0.0],
            vec![2.0, 0.3, -1.2],
            vec![1000.0, 999.999999999, -1000.0],
            vec![1e-12, 0.0, -1e-12],
            (0..16).map(|i| 4.0 - i as f64 * 0.5).collect(),
        ];
        let bases = [
            -1e6, -1000.0, -40.0, -1.0, -0.0, 0.0, 0.3, 40.0, 1000.0, 1e6,
        ];
        let mut pairs = 0;
        for thresholds in &banks {
            for base in bases {
                let expected = grm_logprobs(base, thresholds);
                for (category, value) in expected.iter().enumerate() {
                    assert_eq!(
                        grm_selected_logprob(base, thresholds, category).to_bits(),
                        value.to_bits(),
                        "base={base}, category={category}, thresholds={thresholds:?}",
                    );
                    pairs += 1;
                }
            }
        }
        assert_eq!(pairs, 320);
    }

    #[test]
    #[should_panic]
    fn selected_category_rejects_invalid_category() {
        grm_selected_logprob(0.0, &[], 1);
    }

    // The item M-step must use each count cell's matching primary/specific
    // quadrature tuple (Cai, 2010, pp. 608-609, Appendix A).
    // Reference: Cai, L. (2010). A two-tier full-information item factor
    // analysis model with applications. Psychometrika, 75(4), 581-612.
    // doi:10.1007/s11336-010-9178-0.
    fn compact_support_fit(qp: usize, qs: usize, device: crate::Device) -> TwoTierFipcResult {
        let (v, y, parameters, _, _, _, _) = fixture(qp, qs, false);
        let primary: Vec<f64> = parameters
            .iter()
            .flat_map(|p| p.a_p.iter().copied())
            .collect();
        let specific: Vec<f64> = parameters.iter().map(|p| p.a_s.unwrap()).collect();
        let threshold: Vec<f64> = parameters
            .iter()
            .flat_map(|p| p.d.iter().copied())
            .collect();
        let result = fit_two_tier_grm_fipc(
            &y,
            None,
            &[true; 4],
            &[0; 4],
            v.n_persons,
            v.n_items,
            v.n_primary,
            v.n_specific,
            v.n_cat,
            &[true, true, false, false],
            &primary,
            &specific,
            &threshold,
            &TwoTierFipcConfig {
                q_primary: qp,
                q_specific: qs,
                max_iter: 1,
                tol: 1e-6,
                newton_iter: 1,
                ridge: 1e-8,
                estimate_specific_vars: false,
                device,
            },
        )
        .expect("compact-support focal item update");
        assert!(
            !result.converged,
            "one-step control must not claim convergence"
        );
        for i in 0..2 {
            assert_eq!(result.a_primary[i].to_bits(), primary[i].to_bits());
            assert_eq!(result.a_specific[i].to_bits(), specific[i].to_bits());
            assert_eq!(
                &result.threshold[i * 2..(i + 1) * 2],
                &threshold[i * 2..(i + 1) * 2]
            );
        }
        result
    }

    #[test]
    fn fipc_first_update_compact_support_stabilizes_q121_q241() {
        let a = compact_support_fit(121, 121, crate::Device::Cpu);
        let b = compact_support_fit(241, 241, crate::Device::Cpu);
        for (x, y) in a
            .a_primary
            .iter()
            .zip(&b.a_primary)
            .chain(a.a_specific.iter().zip(&b.a_specific))
            .chain(a.threshold.iter().zip(&b.threshold))
        {
            assert!(
                x.is_finite() && y.is_finite() && (x - y).abs() <= 1e-6,
                "first free-item update must stabilize across explicit Q121 and Q241: {x} vs {y}"
            );
        }
    }

    #[test]
    fn fipc_first_update_compact_support_gpu_cpu_parity() {
        for q in [121, 241] {
            let cpu = compact_support_fit(q, q, crate::Device::Cpu);
            let gpu = compact_support_fit(q, q, crate::Device::Gpu);
            if !gpu.gpu_execution_used {
                assert!(gpu.cpu_fallback_reason.is_some());
            }
            for (x, y) in cpu
                .a_primary
                .iter()
                .zip(&gpu.a_primary)
                .chain(cpu.a_specific.iter().zip(&gpu.a_specific))
                .chain(cpu.threshold.iter().zip(&gpu.threshold))
            {
                assert!(
                    x.is_finite() && y.is_finite() && (x - y).abs() <= 1e-6,
                    "free-item GPU/CPU update differs beyond unchanged precision: {x} vs {y}"
                );
            }
        }
    }
}
