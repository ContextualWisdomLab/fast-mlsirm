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

    // Same category probabilities in the complete-data objective and derivative
    // (Cai, 2010, p. 589, Eqs. 11-12; pp. 608-609, Appendix A).
    // Reference: Cai, L. (2010). A two-tier full-information item factor
    // analysis model with applications. Psychometrika, 75(4), 581-612.
    // doi:10.1007/s11336-010-9178-0.
    fn legacy_log_sigmoid(x: f64) -> f64 {
        if x >= 0.0 {
            -(-x).exp().ln_1p()
        } else {
            x - x.exp().ln_1p()
        }
    }
    fn legacy_node_gradient(base: f64, thresholds: &[f64], counts: &[f64]) -> (f64, Vec<f64>) {
        let kb = thresholds.len();
        let mut g_t = vec![0.0_f64; kb];
        let mut g_base = 0.0_f64;
        if kb == 0 {
            return (0.0, g_t);
        }
        let log_p = grm_logprobs(base, thresholds);
        // Evaluate v/P in log space. Directly exponentiating a valid tail category
        // can underflow P to zero even though its score contribution is finite.
        for j in 0..kb {
            let eta = base + thresholds[j];
            let log_v = legacy_log_sigmoid(eta) + legacy_log_sigmoid(-eta);
            // d q / d s_j = r_{j+1}/P_{j+1} - r_j/P_j  (boundary j sits between cats j and j+1)
            let right = if counts[j + 1] == 0.0 {
                0.0
            } else {
                counts[j + 1] * (log_v - log_p[j + 1]).exp()
            };
            let left = if counts[j] == 0.0 {
                0.0
            } else {
                counts[j] * (log_v - log_p[j]).exp()
            };
            g_t[j] = right - left;
            g_base += right - left;
        }
        (g_base, g_t)
    }
    fn legacy_objective(
        params: &[f64],
        free: &[usize],
        has_specific: bool,
        coords: &[f64],
        ts: &[f64],
        n_primary: usize,
        n_grid: usize,
        qs: usize,
        counts: &[Vec<f64>],
        _n_cat: usize,
    ) -> (f64, Vec<f64>) {
        let k = free.len();
        let off = k + usize::from(has_specific);
        let beta = &params[off..];
        let mut ll = 0.0f64;
        let mut grad = vec![0.0f64; params.len()];
        for (node, cnt) in counts.iter().enumerate() {
            let (g, h) = if has_specific {
                (node / qs, node % qs)
            } else {
                debug_assert!(node < n_grid);
                (node, 0)
            };
            let mut base = 0.0f64;
            for (t, &dim) in free.iter().enumerate() {
                base += params[t] * coords[g * n_primary + dim];
            }
            if has_specific {
                base += params[k] * ts[h];
            }
            let lp = grm_logprobs(base, beta);
            ll += cnt.iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
            let (g_base, g_thr) = legacy_node_gradient(base, beta, cnt);
            for (t, &dim) in free.iter().enumerate() {
                grad[t] += g_base * coords[g * n_primary + dim];
            }
            if has_specific {
                grad[k] += g_base * ts[h];
            }
            for (j, gj) in g_thr.iter().enumerate() {
                grad[off + j] += gj;
            }
        }
        (-ll, grad.iter().map(|g| -g).collect())
    }
    fn legacy_newton(
        mut params: Vec<f64>,
        free: &[usize],
        has_specific: bool,
        coords: &[f64],
        ts: &[f64],
        n_primary: usize,
        n_grid: usize,
        qs: usize,
        counts: &[Vec<f64>],
        n_cat: usize,
        ridge: f64,
        n_newton: usize,
    ) -> Vec<f64> {
        let np = params.len();
        for _ in 0..n_newton {
            let (f0, g) = legacy_objective(
                &params,
                free,
                has_specific,
                coords,
                ts,
                n_primary,
                n_grid,
                qs,
                counts,
                n_cat,
            );
            let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
                break;
            }
            let h = 1e-5;
            let mut hess = vec![vec![0.0f64; np]; np];
            for j in 0..np {
                let mut pj = params.clone();
                pj[j] += h;
                let (_f2, gj) = legacy_objective(
                    &pj,
                    free,
                    has_specific,
                    coords,
                    ts,
                    n_primary,
                    n_grid,
                    qs,
                    counts,
                    n_cat,
                );
                for r in 0..np {
                    hess[r][j] = (gj[r] - g[r]) / h;
                }
            }
            for r in 0..np {
                for c in 0..np {
                    hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
                }
                hess[r][r] += ridge;
            }
            let mut step = solve_small(hess, g.clone());
            let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
            if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
                step = g.clone();
                directional = grad_norm * grad_norm;
            }
            let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
            if max_step > 2.0 {
                for s in &mut step {
                    *s *= 2.0 / max_step;
                }
                directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
                max_step = 2.0;
            }
            let mut alpha = 1.0f64;
            let mut accepted = false;
            for _ in 0..25 {
                let candidate: Vec<f64> = params
                    .iter()
                    .zip(&step)
                    .map(|(value, direction)| value - alpha * direction)
                    .collect();
                let (candidate_f, _) = legacy_objective(
                    &candidate,
                    free,
                    has_specific,
                    coords,
                    ts,
                    n_primary,
                    n_grid,
                    qs,
                    counts,
                    n_cat,
                );
                if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                    params = candidate;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted || alpha * max_step < 1e-9 {
                break;
            }
        }
        params
    }
    #[test]
    fn reused_probability_gradient_preserves_boundary_cells() {
        let banks = vec![
            vec![],
            vec![0.],
            vec![2., 0.3, -1.2],
            vec![1000., 999.999999999, -1000.],
            vec![1e-12, 0., -1e-12],
            (0..16).map(|i| 4. - i as f64 * 0.5).collect::<Vec<_>>(),
        ];
        let mut comparisons = 0;
        for thr in &banks {
            for base in [-1e6, -1000., -40., -1., -0., 0., 0.3, 40., 1000., 1e6] {
                for pattern in 0..3 {
                    let counts = (0..=thr.len())
                        .map(|i| match pattern {
                            0 => 0.,
                            1 => {
                                if i % 2 == 0 {
                                    0.
                                } else {
                                    (i + 1) as f64 / 7.
                                }
                            }
                            _ => 1. + i as f64 / 3.,
                        })
                        .collect::<Vec<_>>();
                    let lp = grm_logprobs(base, thr);
                    let old = legacy_node_gradient(base, thr, &counts);
                    let new = crate::poly::grm_node_gradient_from_logprobs(base, thr, &counts, &lp);
                    assert_eq!(old.0.to_bits(), new.0.to_bits());
                    assert_eq!(old.1.len(), new.1.len());
                    for (a, b) in old.1.iter().zip(&new.1) {
                        assert_eq!(a.to_bits(), b.to_bits());
                    }
                    comparisons += 1;
                }
            }
        }
        assert_eq!(comparisons, 180);
        println!("gradient_boundary_pairs={comparisons}");
    }
    #[test]
    fn reused_probability_objective_and_newton_preserve_full_outputs() {
        let (v, y, pars, coords, lw, ts, lws) = fixture(121, 121, true);
        let mask = (0..y.len()).map(|i| i % 5 != 0).collect::<Vec<_>>();
        let (_, counts, _) = e_step(
            &v,
            &y,
            Some(&mask),
            &pars,
            &lw,
            &lws,
            &coords,
            &ts,
            lw.len(),
            121,
        );
        for i in [0, 3] {
            let free = &v.free_primaries[i];
            let specific = v.item_block[i].is_some();
            let par = &pars[i];
            let mut packed = free.iter().map(|&d| par.a_p[d]).collect::<Vec<_>>();
            if let Some(a) = par.a_s {
                packed.push(a);
            }
            packed.extend(&par.d);
            let old = legacy_objective(
                &packed,
                free,
                specific,
                &coords,
                &ts,
                v.n_primary,
                lw.len(),
                121,
                &counts[i],
                3,
            );
            let new = item_neg_ll_grad(
                &packed,
                free,
                specific,
                &coords,
                &ts,
                v.n_primary,
                lw.len(),
                121,
                &counts[i],
                3,
            );
            assert_eq!(old.0.to_bits(), new.0.to_bits());
            assert_eq!(old.1.len(), new.1.len());
            for (a, b) in old.1.iter().zip(&new.1) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
            let old = legacy_newton(
                packed.clone(),
                free,
                specific,
                &coords,
                &ts,
                v.n_primary,
                lw.len(),
                121,
                &counts[i],
                3,
                1e-8,
                1,
            );
            let new = m_step_item(
                packed,
                free,
                specific,
                &coords,
                &ts,
                v.n_primary,
                lw.len(),
                121,
                &counts[i],
                3,
                1e-8,
                1,
            );
            assert_eq!(old.len(), new.len());
            for (a, b) in old.iter().zip(&new) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }

    fn legacy_ordinary_e_step(
        v: &Validated,
        y: &[usize],
        observed: Option<&[bool]>,
        params: &[ItemParams],
        log_w: &[f64],
        log_ws: &[f64],
        coords: &[f64],
        ts: &[f64],
        n_grid: usize,
        qs: usize,
    ) -> (f64, Vec<Vec<Vec<f64>>>, Vec<f64>) {
        let p = v.n_primary;
        let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * v.n_items + i]);
        let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
        for i in 0..v.n_items {
            let n_nodes = if v.item_block[i].is_some() {
                n_grid * qs
            } else {
                n_grid
            };
            counts.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
        }
        // Per-person scratch: O(n_grid) for primary marginals + O(S * qs) for
        // the active primary node's specific-tier block (not O(S * n_grid * qs)).
        let mut log_i = vec![0.0f64; v.n_specific * n_grid];
        let mut gen_log = vec![0.0f64; n_grid];
        let mut log_like_g = vec![0.0f64; n_grid];
        let mut post_g = vec![0.0f64; n_grid];
        let mut tmp_h = vec![0.0f64; qs];
        let mut block_acc_g = vec![0.0f64; v.n_specific * qs];
        let mut s_bar_sum = vec![0.0f64; p * p];

        let mut loglik = 0.0f64;
        for pp in 0..v.n_persons {
            // Pass 1: person marginal per primary node (specific-free + block
            // integrals), without storing per-(g,h) tables.
            gen_log.copy_from_slice(log_w);
            for &i in &v.specific_free {
                if !is_obs(pp, i) {
                    continue;
                }
                let yc = y[pp * v.n_items + i];
                for g in 0..n_grid {
                    gen_log[g] += item_cat_logprob(v, params, coords, ts, i, g, 0, yc);
                }
            }
            for (s, members) in v.blocks.iter().enumerate() {
                for g in 0..n_grid {
                    for h in 0..qs {
                        let mut acc = log_ws[h];
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            acc += item_cat_logprob(v, params, coords, ts, i, g, h, yc);
                        }
                        tmp_h[h] = acc;
                    }
                    log_i[s * n_grid + g] = log_sum_exp(&tmp_h);
                }
            }
            for g in 0..n_grid {
                let mut acc = gen_log[g];
                for s in 0..v.n_specific {
                    acc += log_i[s * n_grid + g];
                }
                log_like_g[g] = acc;
            }
            let log_lp = log_sum_exp(&log_like_g);
            loglik += log_lp;
            for g in 0..n_grid {
                post_g[g] = (log_like_g[g] - log_lp).exp();
            }
            for g in 0..n_grid {
                let post = post_g[g];
                for (jj, slot) in s_bar_sum.iter_mut().enumerate().take(p * p) {
                    let j = jj / p;
                    let k = jj % p;
                    *slot += post * coords[g * p + j] * coords[g * p + k];
                }
            }
            for &i in &v.specific_free {
                if !is_obs(pp, i) {
                    continue;
                }
                let yc = y[pp * v.n_items + i];
                for g in 0..n_grid {
                    counts[i][g][yc] += post_g[g];
                }
            }
            // Pass 2: joint (g, h) posteriors for block items — recompute the
            // active primary node's specific-tier block on the fly.
            for (s, members) in v.blocks.iter().enumerate() {
                let any_obs = members.iter().any(|&i| is_obs(pp, i));
                if !any_obs {
                    continue;
                }
                for g in 0..n_grid {
                    for h in 0..qs {
                        let mut acc = log_ws[h];
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            acc += item_cat_logprob(v, params, coords, ts, i, g, h, yc);
                        }
                        block_acc_g[s * qs + h] = acc;
                    }
                    // Keep the prior in the joint numerator: subtracting its log
                    // first is undefined at a zero-mass primary node. This is
                    // the unchanged reduced posterior product (Cai, 2010,
                    // pp. 589-590, Eqs. 15-16; pp. 608-609, Appendix A).
                    // Reference: Cai, L. (2010). A two-tier full-information
                    // item factor analysis model with applications.
                    // Psychometrika, 75(4), 581-612. doi:10.1007/s11336-010-9178-0.
                    let mut others = gen_log[g];
                    for s2 in 0..v.n_specific {
                        if s2 != s {
                            others += log_i[s2 * n_grid + g];
                        }
                    }
                    for h in 0..qs {
                        let log_post = block_acc_g[s * qs + h] + others - log_lp;
                        let post = log_post.exp();
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            counts[i][g * qs + h][yc] += post;
                        }
                    }
                }
            }
        }
        (loglik, counts, s_bar_sum)
    }

    // Ordinary reduced posterior sums and GRM cells retain the same measure
    // (Cai, 2010, p. 589, Eqs. 11-12; pp. 589-590, Eqs. 15-16).
    // Reference: Cai, L. (2010). A two-tier full-information item factor
    // analysis model with applications. Psychometrika, 75(4), 581-612.
    // doi:10.1007/s11336-010-9178-0.
    #[test]
    fn projected_item_cells_preserve_observed_node_probabilities() {
        let (mut v, y, mut parameters, base, lw, ts, _) = fixture(121, 121, true);
        // Two-primary product grid: item0 ignores dimension1, item1 cross-loads.
        let mut coords = Vec::new();
        for &x in &base {
            for &z in &base {
                coords.extend([x, z]);
            }
        }
        v.n_primary = 2;
        v.grid_size = base.len() * base.len();
        for i in 0..v.n_items {
            v.free_primaries[i] = vec![0];
            parameters[i].a_p = vec![parameters[i].a_p[0], 0.0];
        }
        v.free_primaries[1] = vec![0, 1];
        parameters[1].a_p[1] = 0.3;
        let mask = (0..y.len()).map(|i| i % 5 != 0).collect::<Vec<_>>();
        let table = ordinary_item_logprobs(
            &v,
            &y,
            Some(&mask),
            &parameters,
            &coords,
            &ts,
            0,
            v.grid_size,
            ts.len(),
        );
        assert_eq!(
            table.log_probabilities.len(),
            base.len() * ts.len() * v.n_cat,
            "inactive primary axis must not multiply stored category cells"
        );
        for i in [0, 1, 3] {
            let original = ordinary_item_logprobs(
                &v,
                &y,
                Some(&mask),
                &parameters,
                &coords,
                &ts,
                i,
                v.grid_size,
                ts.len(),
            );
            for g in 0..v.grid_size {
                for h in 0..if parameters[i].a_s.is_some() {
                    ts.len()
                } else {
                    1
                } {
                    for c in 0..v.n_cat {
                        assert_eq!(
                            original.get(g, h, c).to_bits(),
                            item_cat_logprob(&v, &parameters, &coords, &ts, i, g, h, c).to_bits()
                        );
                    }
                }
            }
        }
        let _ = lw;
        parameters[0].d[0] += 0.25;
        let rebuilt = ordinary_item_logprobs(
            &v,
            &y,
            Some(&mask),
            &parameters,
            &coords,
            &ts,
            0,
            v.grid_size,
            ts.len(),
        );
        assert_ne!(
            table.get(0, 0, 0).to_bits(),
            rebuilt.get(0, 0, 0).to_bits(),
            "a new E-step must not reuse old item parameters"
        );
    }

    #[test]
    fn projected_item_lookup_preserves_ordinary_estep_tuple() {
        for mixed in [false, true] {
            for mode in 0..6 {
                let (v, mut y, mut parameters, coords, mut lw, mut ts, mut lws) =
                    fixture(7, 11, mixed);
                let mut mask = (0..y.len()).map(|i| i % 5 != 0).collect::<Vec<_>>();
                match mode {
                    0 => {}
                    1 => {
                        for pp in 0..v.n_persons {
                            mask[pp * v.n_items] = false;
                        }
                    }
                    2 => {
                        for pp in 0..v.n_persons {
                            for &i in &v.blocks[0] {
                                mask[pp * v.n_items + i] = false;
                            }
                        }
                    }
                    3 => {
                        lw[0] = f64::NEG_INFINITY;
                        lws[0] = f64::NEG_INFINITY;
                    }
                    4 => {
                        parameters[0].d = vec![1e-12, 0.0];
                        parameters[1].a_p[0] = -1000.0;
                        ts.iter_mut().for_each(|x| *x *= 1.7);
                    }
                    _ => {
                        y.fill(0);
                    }
                }
                let old = legacy_ordinary_e_step(
                    &v,
                    &y,
                    Some(&mask),
                    &parameters,
                    &lw,
                    &lws,
                    &coords,
                    &ts,
                    lw.len(),
                    ts.len(),
                );
                let new = e_step(
                    &v,
                    &y,
                    Some(&mask),
                    &parameters,
                    &lw,
                    &lws,
                    &coords,
                    &ts,
                    lw.len(),
                    ts.len(),
                );
                assert_eq!(old.0.to_bits(), new.0.to_bits());
                assert_eq!(old.1.len(), new.1.len());
                for (a, b) in old.1.iter().zip(&new.1) {
                    assert_eq!(a.len(), b.len());
                    for (a, b) in a.iter().zip(b) {
                        assert_eq!(a.len(), b.len());
                        for (x, y) in a.iter().zip(b) {
                            assert_eq!(x.to_bits(), y.to_bits());
                        }
                    }
                }
                assert_eq!(old.2.len(), new.2.len());
                for (x, y) in old.2.iter().zip(&new.2) {
                    assert_eq!(x.to_bits(), y.to_bits());
                }
            }
        }
    }

    // Counts only allocations in the explicitly enabled current test thread.
    // The guard restores counting even if an assertion unwinds.
    struct ObjectiveAllocationCounter;
    std::thread_local! {
        static ALLOCATION_ACTIVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
        static ALLOCATION_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    impl ObjectiveAllocationCounter {
        fn record() {
            let enabled = ALLOCATION_ACTIVE.try_with(|v| v.get()).unwrap_or(false);
            if enabled {
                let _ = ALLOCATION_COUNT.try_with(|v| v.set(v.get() + 1));
            }
        }
    }
    unsafe impl std::alloc::GlobalAlloc for ObjectiveAllocationCounter {
        unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
            Self::record();
            unsafe { std::alloc::GlobalAlloc::alloc(&std::alloc::System, layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
            Self::record();
            unsafe { std::alloc::GlobalAlloc::alloc_zeroed(&std::alloc::System, layout) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: std::alloc::Layout) {
            unsafe { std::alloc::GlobalAlloc::dealloc(&std::alloc::System, pointer, layout) }
        }
        unsafe fn realloc(
            &self,
            pointer: *mut u8,
            layout: std::alloc::Layout,
            size: usize,
        ) -> *mut u8 {
            Self::record();
            unsafe { std::alloc::GlobalAlloc::realloc(&std::alloc::System, pointer, layout, size) }
        }
    }
    #[global_allocator]
    static OBJECTIVE_ALLOCATOR: ObjectiveAllocationCounter = ObjectiveAllocationCounter;
    struct AllocationGuard;
    impl Drop for AllocationGuard {
        fn drop(&mut self) {
            ALLOCATION_ACTIVE.with(|v| v.set(false));
        }
    }

    /// Checks caller-local scratch reuse without changing the graded objective.
    /// Basis: Cai (2010, p. 589, Eqs. 11-12; pp. 608-609, Appendix A).
    /// The allocation budget is an implementation resource invariant, not a
    /// quadrature, convergence, model fit or recovery acceptance threshold.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn objective_buffers_do_not_allocate_per_latent_node() {
        let (v, y, pars, coords, lw, ts, lws) = fixture(121, 121, true);
        let (_, counts, _) = e_step(
            &v,
            &y,
            None,
            &pars,
            &lw,
            &lws,
            &coords,
            &ts,
            lw.len(),
            ts.len(),
        );
        let i = 0;
        let mut packed = vec![pars[i].a_p[0], pars[i].a_s.unwrap()];
        packed.extend_from_slice(&pars[i].d);
        ALLOCATION_COUNT.with(|v| v.set(0));
        ALLOCATION_ACTIVE.with(|v| v.set(true));
        let guard = AllocationGuard;
        let actual = item_neg_ll_grad(
            &packed,
            &v.free_primaries[i],
            true,
            &coords,
            &ts,
            v.n_primary,
            lw.len(),
            ts.len(),
            &counts[i],
            v.n_cat,
        );
        drop(guard);
        let allocations = ALLOCATION_COUNT.with(|v| v.get());
        let expected = legacy_objective(
            &packed,
            &v.free_primaries[i],
            true,
            &coords,
            &ts,
            v.n_primary,
            lw.len(),
            ts.len(),
            &counts[i],
            v.n_cat,
        );
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(
            actual.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        eprintln!(
            "objective_nodes={} objective_allocations={allocations}",
            counts[i].len()
        );
        assert!(
            allocations <= 5,
            "objective allocation grows per latent node: {allocations}"
        );
    }

    fn frozen_buffer_logprobs(base: f64, thresholds: &[f64]) -> Vec<f64> {
        let kb = thresholds.len(); // number of boundaries = K-1
        let mut out = vec![0.0_f64; kb + 1];
        if kb == 0 {
            out[0] = 0.0;
            return out;
        }
        // category 0: 1 - sigmoid(base + beta_0) = sigmoid(-(base + beta_0))
        out[0] = legacy_log_sigmoid(-(base + thresholds[0]));
        // middle categories 1..K-2: P = sigmoid(base+beta_{k-1}) - sigmoid(base+beta_k)
        for k in 1..kb {
            let upper = base + thresholds[k - 1];
            let lower = base + thresholds[k];
            // sigmoid(upper) - sigmoid(lower)
            // = sigmoid(upper) * sigmoid(-lower) * (1 - exp(lower - upper)).
            // `-expm1` preserves a narrow category and avoids subtracting two
            // rounded log-sigmoids in the same extreme tail.
            out[k] = legacy_log_sigmoid(upper)
                + legacy_log_sigmoid(-lower)
                + (-(lower - upper).exp_m1()).ln();
        }
        // top category K-1: sigmoid(base + beta_{K-2})
        out[kb] = legacy_log_sigmoid(base + thresholds[kb - 1]);
        out
    }

    /// Exercises overwritten caller buffers against the frozen prior formula.
    /// Basis: Cai (2010, p. 589, Eqs. 11-12), not a new probability model.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn reused_buffers_preserve_boundary_cells_and_overwrite_prior_values() {
        let banks = vec![
            vec![],
            vec![0.],
            vec![2., 0.3, -1.2],
            vec![1000., 999.999999999, -1000.],
            vec![1e-12, 0., -1e-12],
            vec![0., 0.],
            vec![-1., 1.],
        ];
        let mut comparisons = 0;
        for thresholds in &banks {
            let mut out = vec![f64::NAN; thresholds.len() + 1];
            let mut gout = vec![f64::NAN; thresholds.len()];
            for base in [-1e6, -1000., -40., -1., -0., 0., 0.3, 40., 1000., 1e6] {
                let expected = frozen_buffer_logprobs(base, thresholds);
                crate::poly::grm_logprobs_into(base, thresholds, &mut out);
                assert_eq!(
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    out.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    grm_logprobs(base, thresholds)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                );
                for pattern in 0..3 {
                    let counts = (0..=thresholds.len())
                        .map(|i| {
                            if pattern == 0 {
                                0.
                            } else if pattern == 1 && i % 2 == 0 {
                                0.
                            } else {
                                1. + i as f64 / 3.
                            }
                        })
                        .collect::<Vec<_>>();
                    let old = legacy_node_gradient(base, thresholds, &counts);
                    let gb = crate::poly::grm_node_gradient_into(
                        base, thresholds, &counts, &out, &mut gout,
                    );
                    assert_eq!(old.0.to_bits(), gb.to_bits());
                    assert_eq!(
                        old.1.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        gout.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    comparisons += 1;
                }
            }
        }
        assert_eq!(comparisons, 210);
    }

    #[test]
    fn allocation_counter_restores_thread_scope_on_unwind() {
        ALLOCATION_COUNT.with(|v| v.set(0));
        let rejected = std::panic::catch_unwind(|| {
            ALLOCATION_ACTIVE.with(|v| v.set(true));
            let _guard = AllocationGuard;
            std::thread::spawn(|| {
                assert!(!ALLOCATION_ACTIVE.with(|v| v.get()));
                let _values = vec![1.0_f64; 32];
            })
            .join()
            .unwrap();
            panic!("fixture-owned unwind");
        });
        assert!(rejected.is_err());
        assert!(!ALLOCATION_ACTIVE.with(|v| v.get()));
        ALLOCATION_COUNT.with(|v| v.set(0));
        let _values = vec![1.0_f64; 32];
        assert_eq!(ALLOCATION_COUNT.with(|v| v.get()), 0);
    }

    /// Response cells are person-independent at fixed item parameters (Cai,
    /// 2010, p. 589, Eqs. 11-12); final EAP uses their reduced posterior
    /// weights (p. 591; p. 609, Appendix B), without changing node counts.
    ///
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn ordinary_final_eap_reuses_person_independent_cells() {
        struct Scope;
        impl Drop for Scope {
            fn drop(&mut self) {
                CATEGORY_EVALUATIONS_ACTIVE.with(|v| v.set(false));
            }
        }
        let (_, y0, _, _, _, _, _) = fixture(121, 121, true);
        let y: Vec<usize> = y0.iter().copied().cycle().take(60 * 4).collect();
        let cfg = TwoTierGrmConfig {
            q_primary: 121,
            q_specific: 121,
            max_iter: 1,
            newton_iter: 1,
            n_starts: 1,
            estimate_primary_correlation: false,
            seed: 20260921,
            tol: 1e-6,
            ridge: 1e-8,
        };
        CATEGORY_EVALUATIONS.with(|v| v.set(0));
        CATEGORY_EVALUATIONS_ACTIVE.with(|v| v.set(true));
        let scope = Scope;
        let result =
            fit_two_tier_grm(&y, None, &[true; 4], &[0, 0, 0, -1], 60, 4, 1, 1, 3, &cfg).unwrap();
        drop(scope);
        let evaluated = CATEGORY_EVALUATIONS.with(|v| v.get());
        // Exactly two E-steps (initial and post-update), plus final EAP.
        // The full all-category per-item table is an independent upper bound.
        let bound = 3 * 4 * 121 * 121 * 3;
        eprintln!("final_eap_category_evaluations={evaluated} upper_bound={bound}");
        assert!(result
            .theta_p_eap
            .iter()
            .chain(&result.theta_p_sd)
            .all(|x| x.is_finite()));
        assert!(!result.converged && result.n_iter == 1);
        assert!(
            evaluated <= bound,
            "final EAP repeats person-independent item probabilities: {evaluated} > {bound}"
        );
    }

    /// Frozen predecessor fitter, including its original final posterior
    /// loop and reflection order (Cai, 2010, p. 591; p. 609, Appendix B).
    /// Only the function name and visibility differ from the parent fitter.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[allow(clippy::too_many_arguments)]
    fn predecessor_final_eap(
        y: &[usize],
        observed: Option<&[bool]>,
        primary_map: &[bool],
        specific_map: &[i32],
        n_persons: usize,
        n_items: usize,
        n_primary: usize,
        n_specific: usize,
        n_cat: usize,
        cfg: &TwoTierGrmConfig,
    ) -> Result<TwoTierGrmResult, String> {
        let v = validate(
            y,
            observed,
            primary_map,
            specific_map,
            n_persons,
            n_items,
            n_primary,
            n_specific,
            n_cat,
            cfg,
        )?;
        let p = v.n_primary;
        let (tz, wz) = gh_rule(cfg.q_primary)?;
        let (ts, ws) = gh_rule(cfg.q_specific)?;
        let qs = ts.len();
        let n_grid = v.grid_size;
        let (coords, log_w0) = build_primary_grid(tz, wz, p, n_grid);
        let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

        // Multi-start EM: deterministic starts, best observed loglik wins.
        // A start that fails numerically is SKIPPED (its error is retained for
        // the all-failed report); surviving starts are never mixed.
        let mut best: Option<SingleStartOutcome> = None;
        let mut best_ll = f64::NEG_INFINITY;
        let mut best_start = 0usize;
        let mut first_error: Option<String> = None;
        let mut n_succeeded = 0usize;
        for start in 0..cfg.n_starts {
            match run_single_start(
                &v, y, observed, cfg, &coords, &log_w0, ts, &log_ws, n_grid, qs, start,
            ) {
                Ok(outcome) => {
                    n_succeeded += 1;
                    let ll = *outcome
                        .loglik_trace
                        .last()
                        .expect("EM trace is never empty");
                    if best.is_none() || ll > best_ll {
                        best_ll = ll;
                        best = Some(outcome);
                        best_start = start;
                    }
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error = Some(format!("start {start}: {e}"));
                    }
                }
            }
        }
        let outcome = best.ok_or_else(|| {
            format!(
                "all {} EM start(s) failed numerically; first error: {}",
                cfg.n_starts,
                first_error.unwrap_or_else(|| "unknown".into())
            )
        })?;
        let _ = n_succeeded;
        let params = outcome.params;
        let phi = phi_from_z(&outcome.z_phi, p);

        // Final EAP pass for the primary tier at the winning parameters.
        // Streaming (same blocked GH product as the E-step; #1992): no full
        // log-prob tables, and specific-tier scratch is O(S * qs) per primary
        // node rather than O(S * n_grid * qs).
        let (l, logdet) = cholesky_lower(&phi, p)
            .ok_or_else(|| "winning primary correlation is non-PD".to_string())?;
        let phi_inv = chol_inverse(&l, p);
        let log_w = reweighted_log_weights(&log_w0, &coords, &phi_inv, logdet, p);
        let mut theta_p_eap = vec![0.0f64; n_persons * p];
        let mut theta_p_sd = vec![0.0f64; n_persons * p];
        let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * n_items + i]);
        let mut log_i = vec![0.0f64; v.n_specific * n_grid];
        let mut log_like_g = vec![0.0f64; n_grid];
        let mut tmp_h = vec![0.0f64; qs];
        for pp in 0..n_persons {
            let mut gen_log = log_w.clone();
            for &i in &v.specific_free {
                if !is_obs(pp, i) {
                    continue;
                }
                let yc = y[pp * n_items + i];
                for g in 0..n_grid {
                    gen_log[g] += item_cat_logprob(&v, &params, &coords, ts, i, g, 0, yc);
                }
            }
            for (s, members) in v.blocks.iter().enumerate() {
                for g in 0..n_grid {
                    for h in 0..qs {
                        let mut acc = log_ws[h];
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * n_items + i];
                            acc += item_cat_logprob(&v, &params, &coords, ts, i, g, h, yc);
                        }
                        tmp_h[h] = acc;
                    }
                    log_i[s * n_grid + g] = log_sum_exp(&tmp_h);
                }
            }
            for g in 0..n_grid {
                let mut acc = gen_log[g];
                for s in 0..v.n_specific {
                    acc += log_i[s * n_grid + g];
                }
                log_like_g[g] = acc;
            }
            let log_lp = log_sum_exp(&log_like_g);
            for d in 0..p {
                let (mut m1, mut m2) = (0.0f64, 0.0f64);
                for (g, &ll) in log_like_g.iter().enumerate() {
                    let post = (ll - log_lp).exp();
                    let t = coords[g * p + d];
                    m1 += post * t;
                    m2 += post * t * t;
                }
                theta_p_eap[pp * p + d] = m1;
                theta_p_sd[pp * p + d] = (m2 - m1 * m1).max(0.0).sqrt();
            }
        }

        // Assemble dense outputs.
        let mut a_primary = vec![0.0f64; n_items * p];
        let mut a_specific = vec![0.0f64; n_items];
        let mut threshold = vec![0.0f64; n_items * v.m1];
        let mut n_parameters = if cfg.estimate_primary_correlation {
            p * (p.saturating_sub(1)) / 2
        } else {
            0
        };
        for (i, par) in params.iter().enumerate() {
            for &dim in &v.free_primaries[i] {
                a_primary[i * p + dim] = par.a_p[dim];
                n_parameters += 1;
            }
            if let Some(a_s) = par.a_s {
                a_specific[i] = a_s;
                n_parameters += 1;
            }
            n_parameters += v.m1;
            threshold[i * v.m1..(i + 1) * v.m1].copy_from_slice(&par.d);
        }

        // Per-dimension reflection canonicalization (module docs): each primary
        // over its loading items (flipping slopes, the EAP column, and the Phi
        // row/column signs jointly only when Phi is estimated), each specific
        // within its block; fixed Phi remains bit-exact I;
        // thresholds untouched.
        let mut phi_work = phi;
        for d in 0..p {
            let loaders: Vec<usize> = (0..n_items)
                .filter(|&i| v.free_primaries[i].contains(&d))
                .collect();
            let anchor = loaders
                .iter()
                .max_by(|&&i, &&j| {
                    a_primary[i * p + d]
                        .abs()
                        .total_cmp(&a_primary[j * p + d].abs())
                })
                .copied()
                .expect("validated primaries all load at least two items");
            if a_primary[anchor * p + d] < 0.0 {
                for &i in &loaders {
                    a_primary[i * p + d] = -a_primary[i * p + d];
                }
                for pp in 0..n_persons {
                    theta_p_eap[pp * p + d] = -theta_p_eap[pp * p + d];
                }
                if cfg.estimate_primary_correlation {
                    for q in 0..p {
                        if q != d {
                            phi_work[d * p + q] = -phi_work[d * p + q];
                            phi_work[q * p + d] = -phi_work[q * p + d];
                        }
                    }
                }
            }
        }
        for members in v.blocks.iter() {
            let anchor = members
                .iter()
                .max_by(|&&i, &&j| a_specific[i].abs().total_cmp(&a_specific[j].abs()))
                .copied()
                .expect("validated blocks are non-empty");
            if a_specific[anchor] < 0.0 {
                for &i in members {
                    a_specific[i] = -a_specific[i];
                }
            }
        }

        let mut category_counts = vec![0usize; n_items * n_cat];
        for pp in 0..n_persons {
            for i in 0..n_items {
                if is_obs(pp, i) {
                    category_counts[i * n_cat + y[pp * n_items + i]] += 1;
                }
            }
        }

        Ok(TwoTierGrmResult {
            a_primary,
            a_specific,
            threshold,
            phi: phi_work,
            theta_p_eap,
            theta_p_sd,
            category_counts,
            loglik_trace: outcome.loglik_trace,
            n_iter: outcome.n_iter,
            converged: outcome.converged,
            termination_reason: outcome.termination_reason,
            final_loglik_change: outcome.final_loglik_change,
            best_start,
            n_parameters,
            primary_identification: if cfg.estimate_primary_correlation {
                "correlated"
            } else {
                "orthogonal"
            },
        })
    }

    /// Response probabilities and posterior weights retain the predecessor
    /// arithmetic (Cai, 2010, p. 589, Eqs. 11-12; p. 609, Appendix B).
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn ordinary_final_eap_cache_preserves_masked_posterior_vectors() {
        for (p, qp, qs, mixed) in [(1, 121, 241, false), (1, 241, 121, true), (2, 7, 9, true)] {
            let np = 12;
            let ni = 6;
            let y: Vec<usize> = (0..np * ni).map(|k| (k / ni + 2 * (k % ni)) % 3).collect();
            let mut mask: Vec<bool> = (0..np * ni).map(|k| k % 13 != 0).collect();
            // Preserve an entirely unobserved person's prior-only posterior.
            mask[..ni].fill(false);
            let primary_map: Vec<bool> = (0..ni * p)
                .map(|k| p == 1 || k % p == usize::from(k / p >= 3))
                .collect();
            let specific_map = if mixed {
                vec![0, 0, 0, 1, 1, -1]
            } else {
                vec![0, 0, 0, 1, 1, 1]
            };
            let cfg = TwoTierGrmConfig {
                estimate_primary_correlation: p == 2,
                q_primary: qp,
                q_specific: qs,
                max_iter: 1,
                tol: 1e-8,
                n_starts: 1,
                seed: 20260921,
                newton_iter: 1,
                ridge: 1e-8,
            };
            let result = fit_two_tier_grm(
                &y,
                Some(&mask),
                &primary_map,
                &specific_map,
                np,
                ni,
                p,
                2,
                3,
                &cfg,
            )
            .unwrap();
            let expected = predecessor_final_eap(
                &y,
                Some(&mask),
                &primary_map,
                &specific_map,
                np,
                ni,
                p,
                2,
                3,
                &cfg,
            )
            .unwrap();
            assert_eq!(
                result
                    .theta_p_eap
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .theta_p_eap
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                result
                    .theta_p_sd
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected
                    .theta_p_sd
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            for (actual, original) in [
                (&result.a_primary, &expected.a_primary),
                (&result.a_specific, &expected.a_specific),
                (&result.threshold, &expected.threshold),
                (&result.phi, &expected.phi),
                (&result.loglik_trace, &expected.loglik_trace),
            ] {
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    original.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
            assert_eq!(result.category_counts, expected.category_counts);
            assert_eq!(
                result.final_loglik_change.to_bits(),
                expected.final_loglik_change.to_bits()
            );
            assert_eq!(result.termination_reason, expected.termination_reason);
            assert_eq!(result.n_parameters, expected.n_parameters);
            assert_eq!(result.best_start, expected.best_start);
            assert!(!result.converged && result.n_iter == 1);
            assert_eq!(result.converged, expected.converged);
            assert_eq!(result.n_iter, expected.n_iter);
        }
    }

    /// Reuse exact GRM cells without combining posterior-count rows.
    /// Basis: Cai (2010), p. 589, Eqs. 11-12; p. 590, M-step description.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn item_objective_reuses_exact_probability_cells() {
        let q = 121usize;
        let grid = q * q;
        let coords: Vec<f64> = (0..grid)
            .flat_map(|g| [(g % q) as f64 / 16.0 - 3.75, (g / q) as f64 / 16.0 - 3.75])
            .collect();
        let ts: Vec<f64> = (0..q).map(|h| h as f64 / 16.0 - 3.75).collect();
        let counts: Vec<Vec<f64>> = (0..grid * q)
            .map(|node| vec![0.0, (node % 11) as f64 / 100.0, 0.1, 0.2])
            .collect();
        let params = [1.2, 0.7, 1.4, 0.0, -1.3];
        let mut distinct = std::collections::HashSet::new();
        for node in 0..counts.len() {
            let (g, h) = (node / q, node % q);
            let mut base = 0.0;
            base += params[0] * coords[g * 2];
            base += params[1] * ts[h];
            distinct.insert(base.to_bits());
        }
        assert!(distinct.len() < counts.len());
        OBJECTIVE_CELL_EVALUATIONS.with(|count| count.set(0));
        OBJECTIVE_CELL_EVALUATIONS_ACTIVE.with(|active| active.set(true));
        let result = std::panic::catch_unwind(|| {
            item_neg_ll_grad(&params, &[0], true, &coords, &ts, 2, grid, q, &counts, 4)
        });
        OBJECTIVE_CELL_EVALUATIONS_ACTIVE.with(|active| active.set(false));
        let result = result.unwrap();
        assert!(result.0.is_finite() && result.1.iter().all(|x| x.is_finite()));
        let evaluations = OBJECTIVE_CELL_EVALUATIONS.with(|count| count.get());
        assert_eq!(
            evaluations,
            distinct.len(),
            "identical predictor cells were recomputed"
        );
    }

    fn predecessor_item_objective(
        params: &[f64],
        free: &[usize],
        has_specific: bool,
        coords: &[f64],
        ts: &[f64],
        n_primary: usize,
        n_grid: usize,
        qs: usize,
        counts: &[Vec<f64>],
        _n_cat: usize,
    ) -> (f64, Vec<f64>) {
        let k = free.len();
        let off = k + usize::from(has_specific);
        let beta = &params[off..];
        let mut ll = 0.0f64;
        let mut grad = vec![0.0f64; params.len()];
        let mut lp = vec![0.0; beta.len() + 1];
        let mut g_thr = vec![0.0; beta.len()];
        for (node, cnt) in counts.iter().enumerate() {
            let (g, h) = if has_specific {
                (node / qs, node % qs)
            } else {
                debug_assert!(node < n_grid);
                (node, 0)
            };
            let mut base = 0.0f64;
            for (t, &dim) in free.iter().enumerate() {
                base += params[t] * coords[g * n_primary + dim];
            }
            if has_specific {
                base += params[k] * ts[h];
            }
            crate::poly::grm_logprobs_into(base, beta, &mut lp);
            ll += cnt.iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
            let g_base = crate::poly::grm_node_gradient_into(base, beta, cnt, &lp, &mut g_thr);
            for (t, &dim) in free.iter().enumerate() {
                grad[t] += g_base * coords[g * n_primary + dim];
            }
            if has_specific {
                grad[k] += g_base * ts[h];
            }
            for (j, gj) in g_thr.iter().enumerate() {
                grad[off + j] += gj;
            }
        }
        (-ll, grad.iter().map(|g| -g).collect())
    }

    fn predecessor_item_newton(
        mut params: Vec<f64>,
        free: &[usize],
        has_specific: bool,
        coords: &[f64],
        ts: &[f64],
        n_primary: usize,
        n_grid: usize,
        qs: usize,
        counts: &[Vec<f64>],
        n_cat: usize,
        ridge: f64,
        n_newton: usize,
    ) -> Vec<f64> {
        let np = params.len();
        for _ in 0..n_newton {
            let (f0, g) = predecessor_item_objective(
                &params,
                free,
                has_specific,
                coords,
                ts,
                n_primary,
                n_grid,
                qs,
                counts,
                n_cat,
            );
            let grad_norm = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if !f0.is_finite() || !grad_norm.is_finite() || grad_norm < 1e-9 {
                break;
            }
            let h = 1e-5;
            let mut hess = vec![vec![0.0f64; np]; np];
            for j in 0..np {
                let mut pj = params.clone();
                pj[j] += h;
                let (_f2, gj) = predecessor_item_objective(
                    &pj,
                    free,
                    has_specific,
                    coords,
                    ts,
                    n_primary,
                    n_grid,
                    qs,
                    counts,
                    n_cat,
                );
                for r in 0..np {
                    hess[r][j] = (gj[r] - g[r]) / h;
                }
            }
            for r in 0..np {
                for c in 0..np {
                    hess[r][c] = 0.5 * (hess[r][c] + hess[c][r]);
                }
                hess[r][r] += ridge;
            }
            let mut step = solve_small(hess, g.clone());
            let mut directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum::<f64>();
            if !step.iter().all(|s| s.is_finite()) || directional <= 0.0 {
                step = g.clone();
                directional = grad_norm * grad_norm;
            }
            let mut max_step = step.iter().map(|s| s.abs()).fold(0.0f64, f64::max);
            if max_step > 2.0 {
                for s in &mut step {
                    *s *= 2.0 / max_step;
                }
                directional = g.iter().zip(&step).map(|(gi, si)| gi * si).sum();
                max_step = 2.0;
            }
            let mut alpha = 1.0f64;
            let mut accepted = false;
            for _ in 0..25 {
                let candidate: Vec<f64> = params
                    .iter()
                    .zip(&step)
                    .map(|(value, direction)| value - alpha * direction)
                    .collect();
                let (candidate_f, _) = predecessor_item_objective(
                    &candidate,
                    free,
                    has_specific,
                    coords,
                    ts,
                    n_primary,
                    n_grid,
                    qs,
                    counts,
                    n_cat,
                );
                if candidate_f.is_finite() && candidate_f <= f0 - 1e-4 * alpha * directional {
                    params = candidate;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted || alpha * max_step < 1e-9 {
                break;
            }
        }
        params
    }

    /// Compare exact predecessor accumulation and solver under explicit Q121.
    /// Basis: Cai (2010), p. 589, Eqs. 11-12; p. 590, E-step-table M-step.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn item_objective_cache_preserves_gradient_and_newton_bits() {
        let q = 121;
        let (nodes, _) = quadrature::require_gh_rule(q, "q_primary").unwrap();
        let (ts, _) = quadrature::require_gh_rule(q, "q_specific").unwrap();
        for n_primary in [1usize, 2] {
            let coords: Vec<f64> = nodes
                .iter()
                .flat_map(|&x| {
                    if n_primary == 1 {
                        vec![x]
                    } else {
                        vec![x, 0.0]
                    }
                })
                .collect();
            for has_specific in [false, true] {
                let counts: Vec<Vec<f64>> = (0..if has_specific { q * q } else { q })
                    .map(|n| {
                        vec![
                            if n % 5 == 0 { 0.0 } else { 0.1 },
                            (n % 7) as f64 / 20.0,
                            0.2,
                            0.0,
                        ]
                    })
                    .collect();
                for slope in [0.0, -0.0, 1.2, -0.9] {
                    let mut params = vec![slope];
                    if has_specific {
                        params.push(0.7);
                    }
                    params.extend_from_slice(&[1.4, 0.0, -1.3]);
                    let expected = predecessor_item_objective(
                        &params,
                        &[0],
                        has_specific,
                        &coords,
                        ts,
                        n_primary,
                        q,
                        q,
                        &counts,
                        4,
                    );
                    let actual = item_neg_ll_grad(
                        &params,
                        &[0],
                        has_specific,
                        &coords,
                        ts,
                        n_primary,
                        q,
                        q,
                        &counts,
                        4,
                    );
                    assert_eq!(actual.0.to_bits(), expected.0.to_bits());
                    assert_eq!(
                        actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                    // Use a distinct second call; values must not leak across parameters.
                    params[0] += 0.31;
                    let expected = predecessor_item_newton(
                        params.clone(),
                        &[0],
                        has_specific,
                        &coords,
                        ts,
                        n_primary,
                        q,
                        q,
                        &counts,
                        4,
                        1e-8,
                        1,
                    );
                    let actual = m_step_item(
                        params,
                        &[0],
                        has_specific,
                        &coords,
                        ts,
                        n_primary,
                        q,
                        q,
                        &counts,
                        4,
                        1e-8,
                        1,
                    );
                    assert_eq!(
                        actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
        }
    }

    /// Retain zero-count and invalid-trial outcomes without normalizing them.
    /// Basis: Cai (2010), p. 589, Eqs. 11-12; p. 590, M-step description.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn item_objective_cache_retains_zero_counts_and_invalid_trials() {
        let q = 121usize;
        let coords: Vec<f64> = (0..q)
            .map(|n| if n % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let coords: Vec<f64> = coords.iter().flat_map(|&x| [x, 0.0]).collect();
        let ts = vec![0.0; q];
        for counts in [
            vec![vec![0.0; 4]; q * q],
            vec![vec![0.0, 0.2, 0.1, 0.0]; q * q],
        ] {
            for params in [
                [1.2, 0.7, 1.4, 0.0, -1.3],
                [1.2, 0.7, 0.0, 0.0, -1.3],
                [1.2, 0.7, -1.0, 1.0, -1.3],
                [f64::INFINITY, 0.7, 1.4, 0.0, -1.3],
            ] {
                let expected = predecessor_item_objective(
                    &params,
                    &[0],
                    true,
                    &coords,
                    &ts,
                    2,
                    q,
                    q,
                    &counts,
                    4,
                );
                let actual =
                    item_neg_ll_grad(&params, &[0], true, &coords, &ts, 2, q, q, &counts, 4);
                assert_eq!(actual.0.to_bits(), expected.0.to_bits());
                assert_eq!(
                    actual.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.1.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    /// Inspect repeated block signatures without combining respondent sums.
    /// Basis: Cai (2010), pp. 589-590, Eqs. 14-16; pp. 607-608, Appendix A.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn ordinary_estep_reuses_exact_block_response_marginals() {
        let (v, y, params, coords, lw, ts, lws) = fixture(121, 121, true);
        let mask: Vec<bool> = (0..v.n_persons)
            .flat_map(|person| [true, person % 4 != 0, person % 5 != 0, true])
            .collect();
        let distinct: usize = v
            .blocks
            .iter()
            .map(|items| {
                (0..v.n_persons)
                    .map(|person| {
                        items
                            .iter()
                            .map(|&item| {
                                if mask[person * v.n_items + item] {
                                    Some(y[person * v.n_items + item])
                                } else {
                                    None
                                }
                            })
                            .collect::<Vec<_>>()
                    })
                    .collect::<std::collections::HashSet<_>>()
                    .len()
            })
            .sum();
        assert!(distinct < v.n_persons * v.n_specific);
        BLOCK_MARGINAL_EVALUATIONS.with(|count| count.set(0));
        BLOCK_MARGINAL_EVALUATIONS_ACTIVE.with(|active| active.set(true));
        let result = std::panic::catch_unwind(|| {
            e_step(
                &v,
                &y,
                Some(&mask),
                &params,
                &lw,
                &lws,
                &coords,
                &ts,
                lw.len(),
                ts.len(),
            )
        });
        BLOCK_MARGINAL_EVALUATIONS_ACTIVE.with(|active| active.set(false));
        let (ll, counts, moments) = result.unwrap();
        assert!(ll.is_finite());
        assert!(counts
            .iter()
            .flatten()
            .flatten()
            .chain(moments.iter())
            .all(|x| x.is_finite()));
        assert_eq!(
            BLOCK_MARGINAL_EVALUATIONS.with(|count| count.get()),
            distinct * lw.len(),
            "identical block-response likelihoods were recomputed"
        );
    }

    /// Retain the exact pre-memoization E-step arithmetic as a test oracle.
    /// Basis: Cai (2010), pp. 589-590, Eqs. 15-16; pp. 607-608, Appendix A.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    fn predecessor_block_pattern_e_step(
        v: &Validated,
        y: &[usize],
        observed: Option<&[bool]>,
        params: &[ItemParams],
        log_w: &[f64],
        log_ws: &[f64],
        coords: &[f64],
        ts: &[f64],
        n_grid: usize,
        qs: usize,
    ) -> (f64, Vec<Vec<Vec<f64>>>, Vec<f64>) {
        let logprobs: Vec<OrdinaryItemLogprobs> = (0..v.n_items)
            .map(|i| ordinary_item_logprobs(v, y, observed, params, coords, ts, i, n_grid, qs))
            .collect();
        let p = v.n_primary;
        let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * v.n_items + i]);
        let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
        for i in 0..v.n_items {
            let n_nodes = if v.item_block[i].is_some() {
                n_grid * qs
            } else {
                n_grid
            };
            counts.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
        }
        // Per-person scratch: O(n_grid) for primary marginals + O(S * qs) for
        // the active primary node's specific-tier block (not O(S * n_grid * qs)).
        let mut log_i = vec![0.0f64; v.n_specific * n_grid];
        let mut gen_log = vec![0.0f64; n_grid];
        let mut log_like_g = vec![0.0f64; n_grid];
        let mut post_g = vec![0.0f64; n_grid];
        let mut tmp_h = vec![0.0f64; qs];
        let mut block_acc_g = vec![0.0f64; v.n_specific * qs];
        let mut s_bar_sum = vec![0.0f64; p * p];

        let mut loglik = 0.0f64;
        for pp in 0..v.n_persons {
            // Pass 1: person marginal per primary node (specific-free + block
            // integrals), without storing per-(g,h) tables.
            gen_log.copy_from_slice(log_w);
            for &i in &v.specific_free {
                if !is_obs(pp, i) {
                    continue;
                }
                let yc = y[pp * v.n_items + i];
                for g in 0..n_grid {
                    gen_log[g] += logprobs[i].get(g, 0, yc);
                }
            }
            for (s, members) in v.blocks.iter().enumerate() {
                for g in 0..n_grid {
                    for h in 0..qs {
                        let mut acc = log_ws[h];
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            acc += logprobs[i].get(g, h, yc);
                        }
                        tmp_h[h] = acc;
                    }
                    log_i[s * n_grid + g] = log_sum_exp(&tmp_h);
                }
            }
            for g in 0..n_grid {
                let mut acc = gen_log[g];
                for s in 0..v.n_specific {
                    acc += log_i[s * n_grid + g];
                }
                log_like_g[g] = acc;
            }
            let log_lp = log_sum_exp(&log_like_g);
            loglik += log_lp;
            for g in 0..n_grid {
                post_g[g] = (log_like_g[g] - log_lp).exp();
            }
            for g in 0..n_grid {
                let post = post_g[g];
                for (jj, slot) in s_bar_sum.iter_mut().enumerate().take(p * p) {
                    let j = jj / p;
                    let k = jj % p;
                    *slot += post * coords[g * p + j] * coords[g * p + k];
                }
            }
            for &i in &v.specific_free {
                if !is_obs(pp, i) {
                    continue;
                }
                let yc = y[pp * v.n_items + i];
                for g in 0..n_grid {
                    counts[i][g][yc] += post_g[g];
                }
            }
            // Pass 2: joint (g, h) posteriors for block items — recompute the
            // active primary node's specific-tier block on the fly.
            for (s, members) in v.blocks.iter().enumerate() {
                let any_obs = members.iter().any(|&i| is_obs(pp, i));
                if !any_obs {
                    continue;
                }
                for g in 0..n_grid {
                    for h in 0..qs {
                        let mut acc = log_ws[h];
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            acc += logprobs[i].get(g, h, yc);
                        }
                        block_acc_g[s * qs + h] = acc;
                    }
                    // Keep the prior in the joint numerator: subtracting its log
                    // first is undefined at a zero-mass primary node. This is
                    // the unchanged reduced posterior product (Cai, 2010,
                    // pp. 589-590, Eqs. 15-16; pp. 608-609, Appendix A).
                    // Reference: Cai, L. (2010). A two-tier full-information
                    // item factor analysis model with applications.
                    // Psychometrika, 75(4), 581-612. doi:10.1007/s11336-010-9178-0.
                    let mut others = gen_log[g];
                    for s2 in 0..v.n_specific {
                        if s2 != s {
                            others += log_i[s2 * n_grid + g];
                        }
                    }
                    for h in 0..qs {
                        let log_post = block_acc_g[s * qs + h] + others - log_lp;
                        let post = log_post.exp();
                        for &i in members {
                            if !is_obs(pp, i) {
                                continue;
                            }
                            let yc = y[pp * v.n_items + i];
                            counts[i][g * qs + h][yc] += post;
                        }
                    }
                }
            }
        }
        (loglik, counts, s_bar_sum)
    }

    /// Verify block reuse without changing ordered person statistics.
    /// Basis: Cai (2010), pp. 589-590, Eqs. 15-16; pp. 607-608, Appendix A.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn ordinary_block_pattern_cache_preserves_masked_highq_tuple() {
        for (qp, qs) in [(121, 121), (241, 121), (121, 241), (481, 121), (121, 481)] {
            let (mut v, mut y, mut params, mut coords, mut lw, mut ts, mut lws) =
                fixture(qp, qs, true);
            // Block-specific maps must not collide with a sibling's identical signature.
            v.n_specific = 2;
            v.blocks = vec![vec![0, 1], vec![2]];
            v.item_block = vec![Some(0), Some(0), Some(1), None];
            let mask: Vec<bool> = (0..v.n_persons)
                .flat_map(|person| [person % 4 != 0, person % 4 != 0, person % 5 != 0, true])
                .collect();
            for (n, observed) in mask.iter().enumerate() {
                if !observed {
                    y[n] = usize::MAX;
                }
            }
            // Include a zero-primary-mass node without changing the previous guard.
            coords.insert(0, -43.0);
            lw.insert(0, f64::NEG_INFINITY);
            v.grid_size += 1;
            for shifted in [false, true] {
                if shifted {
                    for par in &mut params {
                        par.d[0] += 0.21;
                        par.a_p[0] *= 1.07;
                    }
                    for x in &mut coords {
                        *x += 0.03;
                    }
                    for x in &mut ts {
                        *x -= 0.04;
                    }
                    lw[1] -= 0.15;
                    lws[0] -= 0.2;
                }
                let expected = predecessor_block_pattern_e_step(
                    &v,
                    &y,
                    Some(&mask),
                    &params,
                    &lw,
                    &lws,
                    &coords,
                    &ts,
                    lw.len(),
                    ts.len(),
                );
                let actual = e_step(
                    &v,
                    &y,
                    Some(&mask),
                    &params,
                    &lw,
                    &lws,
                    &coords,
                    &ts,
                    lw.len(),
                    ts.len(),
                );
                assert!(actual.0.is_finite());
                assert_eq!(actual.0.to_bits(), expected.0.to_bits());
                assert_eq!(
                    actual
                        .1
                        .iter()
                        .flatten()
                        .flatten()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    expected
                        .1
                        .iter()
                        .flatten()
                        .flatten()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    actual.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    expected.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    /// Characterize missing and empty helper blocks independently of public validation.
    /// Basis: Cai (2010), pp. 589-590, Eqs. 15-16; pp. 607-608, Appendix A.
    /// Reference: Cai, L. (2010). A two-tier full-information item factor
    /// analysis model with applications. Psychometrika, 75(4), 581-612.
    /// doi:10.1007/s11336-010-9178-0.
    #[test]
    fn ordinary_block_pattern_cache_preserves_unobserved_and_empty_helpers() {
        let (mut v, mut y, params, coords, lw, ts, lws) = fixture(121, 121, true);
        v.n_specific = 2;
        v.blocks.push(vec![]);
        let observed = vec![false; y.len()];
        y.fill(usize::MAX);
        let expected = predecessor_block_pattern_e_step(
            &v,
            &y,
            Some(&observed),
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            lw.len(),
            ts.len(),
        );
        let actual = e_step(
            &v,
            &y,
            Some(&observed),
            &params,
            &lw,
            &lws,
            &coords,
            &ts,
            lw.len(),
            ts.len(),
        );
        assert!(actual.0.is_finite());
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(
            actual
                .1
                .iter()
                .flatten()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            expected
                .1
                .iter()
                .flatten()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            actual.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.2.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
    }
}
