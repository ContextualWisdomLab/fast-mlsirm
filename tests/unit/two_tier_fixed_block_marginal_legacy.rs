use super::*;

pub(super) fn legacy_fixed_fipc_moments_cpu(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    params: &[ItemParams],
    log_w: &[f64],
    log_ws_by_specific: &[Vec<f64>],
    coords: &[f64],
    ts_by_specific: &[Vec<f64>],
    n_grid: usize,
    qs: usize,
) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let p = v.n_primary;
    let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * v.n_items + i]);
    let cells: Vec<OrdinaryItemLogprobs> = (0..v.n_items)
        .map(|i| focal_item_logprobs(v, params, coords, ts_by_specific, i, n_grid, qs))
        .collect();
    let mut log_i = vec![0.0; v.n_specific * n_grid];
    let mut gen_log = vec![0.0; n_grid];
    let mut log_like_g = vec![0.0; n_grid];
    let mut post_g = vec![0.0; n_grid];
    let mut tmp_h = vec![0.0; qs];
    let mut block_acc_g = vec![0.0; v.n_specific * qs];
    let mut sum_primary = vec![0.0; p];
    let mut sum_primary2 = vec![0.0; p * p];
    let mut sum_specific2 = vec![0.0; v.n_specific];
    let mut loglik = 0.0;

    let person_order = canonical_person_order(v, y, observed);
    for pp in person_order {
        gen_log.copy_from_slice(log_w);
        for &i in &v.specific_free {
            if !is_obs(pp, i) {
                continue;
            }
            let yc = y[pp * v.n_items + i];
            for g in 0..n_grid {
                gen_log[g] += cells[i].get(g, 0, yc);
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws_by_specific[s][h];
                    for &i in members {
                        if is_obs(pp, i) {
                            acc += cells[i].get(g, h, y[pp * v.n_items + i]);
                        }
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
        for d in 0..p {
            let mut m1 = 0.0;
            for g in 0..n_grid {
                let t = coords[g * p + d];
                m1 += post_g[g] * t;
            }
            sum_primary[d] += m1;
        }
        for j in 0..p {
            for k in 0..p {
                let mut m = 0.0;
                for g in 0..n_grid {
                    m += post_g[g] * coords[g * p + j] * coords[g * p + k];
                }
                sum_primary2[j * p + k] += m;
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            if !members.iter().any(|&i| is_obs(pp, i)) {
                continue;
            }
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws_by_specific[s][h];
                    for &i in members {
                        if is_obs(pp, i) {
                            acc += cells[i].get(g, h, y[pp * v.n_items + i]);
                        }
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
                    let post = (block_acc_g[s * qs + h] + others - log_lp).exp();
                    sum_specific2[s] += post * ts_by_specific[s][h] * ts_by_specific[s][h];
                }
            }
        }
    }
    (loglik, sum_primary, sum_primary2, sum_specific2)
}
