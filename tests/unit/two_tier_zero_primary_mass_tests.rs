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
}
