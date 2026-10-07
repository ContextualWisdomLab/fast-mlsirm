use super::*;
pub(super) fn legacy_e_step_fipc_cpu(
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
) -> (
    f64,
    Vec<Vec<Vec<f64>>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
) {
    let p = v.n_primary;
    let is_obs = |pp: usize, i: usize| observed.is_none_or(|o| o[pp * v.n_items + i]);
    let mut counts = Vec::with_capacity(v.n_items);
    for i in 0..v.n_items {
        let nodes = if v.item_block[i].is_some() {
            n_grid * qs
        } else {
            n_grid
        };
        counts.push(vec![vec![0.0; v.n_cat]; nodes]);
    }
    let mut log_i = vec![0.0; v.n_specific * n_grid];
    let mut gen_log = vec![0.0; n_grid];
    let mut log_like_g = vec![0.0; n_grid];
    let mut post_g = vec![0.0; n_grid];
    let mut tmp_h = vec![0.0; qs];
    let mut block_acc_g = vec![0.0; v.n_specific * qs];
    let mut sum_primary = vec![0.0; p];
    let mut sum_primary2 = vec![0.0; p * p];
    let mut sum_specific2 = vec![0.0; v.n_specific];
    let mut specific_mass = vec![0.0; v.n_specific];
    let mut person_eap = vec![0.0; v.n_persons * p];
    let mut person_sd = vec![0.0; v.n_persons * p];
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
                gen_log[g] += item_cat_logprob_fipc(v, params, coords, ts_by_specific, i, g, 0, yc);
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..n_grid {
                for h in 0..qs {
                    let mut acc = log_ws_by_specific[s][h];
                    for &i in members {
                        if is_obs(pp, i) {
                            acc += item_cat_logprob_fipc(
                                v,
                                params,
                                coords,
                                ts_by_specific,
                                i,
                                g,
                                h,
                                y[pp * v.n_items + i],
                            );
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
            let mut m2 = 0.0;
            for g in 0..n_grid {
                let t = coords[g * p + d];
                m1 += post_g[g] * t;
                m2 += post_g[g] * t * t;
            }
            person_eap[pp * p + d] = m1;
            person_sd[pp * p + d] = (m2 - m1 * m1).max(0.0).sqrt();
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
        for &i in &v.specific_free {
            if !is_obs(pp, i) {
                continue;
            }
            let yc = y[pp * v.n_items + i];
            for g in 0..n_grid {
                counts[i][g][yc] += post_g[g];
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
                            acc += item_cat_logprob_fipc(
                                v,
                                params,
                                coords,
                                ts_by_specific,
                                i,
                                g,
                                h,
                                y[pp * v.n_items + i],
                            );
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
                    specific_mass[s] += post;
                    sum_specific2[s] += post * ts_by_specific[s][h] * ts_by_specific[s][h];
                    for &i in members {
                        if is_obs(pp, i) {
                            counts[i][g * qs + h][y[pp * v.n_items + i]] += post;
                        }
                    }
                }
            }
        }
    }
    (
        loglik,
        counts,
        sum_primary,
        sum_primary2,
        sum_specific2,
        specific_mass,
        person_eap,
        person_sd,
    )
}
