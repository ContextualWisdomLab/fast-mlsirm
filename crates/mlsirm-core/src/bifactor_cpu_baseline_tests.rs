use crate::bifactor_grm::{general_only_without_prior, log_sum_exp, ItemParams, Validated};
use crate::poly::grm_logprobs;
fn reset_calls() {}
pub(crate) fn e_step_multigroup(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    group_id: &[usize],
    n_groups: usize,
    params_groups: &[Vec<ItemParams>],
    tg_groups: &[Vec<f64>],
    ts_groups: &[Vec<Vec<f64>>],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
    device: crate::Device,
) -> (
    f64,
    Vec<Vec<Vec<Vec<f64>>>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<Vec<f64>>,
    Vec<Vec<f64>>,
) {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    // Per-group logprob tables at the CURRENT group nodes.
    let mut tables_groups: Vec<Vec<Vec<f64>>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let mut tables = Vec::with_capacity(v.n_items);
        for (i, par) in params_groups[g].iter().enumerate() {
            match par.a_s {
                Some(a_s) => {
                    let s = v.item_block[i].expect("block item has an owning block");
                    let mut lp = vec![0.0f64; qg * qs * v.n_cat];
                    for t in 0..qg {
                        for h in 0..qs {
                            let base = par.a_g * tg_groups[g][t] + a_s * ts_groups[g][s][h];
                            let probs = grm_logprobs(base, &par.d);
                            lp[(t * qs + h) * v.n_cat..(t * qs + h + 1) * v.n_cat]
                                .copy_from_slice(&probs);
                        }
                    }
                    tables.push(lp);
                }
                None => {
                    let mut lp = vec![0.0f64; qg * v.n_cat];
                    for t in 0..qg {
                        let base = par.a_g * tg_groups[g][t];
                        let probs = grm_logprobs(base, &par.d);
                        lp[t * v.n_cat..(t + 1) * v.n_cat].copy_from_slice(&probs);
                    }
                    tables.push(lp);
                }
            }
        }
        tables_groups.push(tables);
    }

    #[cfg(all(feature = "gpu", not(coverage)))]
    {
        if device == crate::Device::Gpu || device == crate::Device::Auto {
            let inputs = crate::gpu_bifactor::ReducedEstepInputs {
                y,
                observed,
                group_id: Some(group_id),
                n_persons: v.n_persons,
                n_items: v.n_items,
                n_specific: v.n_specific,
                n_cat: v.n_cat,
                qg,
                qs,
                n_groups,
                tables_groups: &tables_groups,
                item_block: &v.item_block,
                blocks: &v.blocks,
                tg_groups,
                ts_groups,
                log_wg,
                log_ws,
                want_node_post: false,
            };
            if let Some(res) = crate::gpu_bifactor::e_step_reduced_gpu(&inputs) {
                let stride = res.counts_stride_nodes;
                let mut counts: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_groups);
                for g in 0..n_groups {
                    let mut cg = Vec::with_capacity(v.n_items);
                    for i in 0..v.n_items {
                        let base = (g * v.n_items + i) * stride * v.n_cat;
                        let n_nodes = if v.item_block[i].is_some() {
                            qg * qs
                        } else {
                            qg
                        };
                        cg.push(
                            res.counts[base..base + n_nodes * v.n_cat]
                                .chunks_exact(v.n_cat)
                                .map(<[f64]>::to_vec)
                                .collect(),
                        );
                    }
                    counts.push(cg);
                }
                let mut s2_spec = vec![vec![0.0; v.n_specific]; n_groups];
                let mut w_spec = vec![vec![0.0; v.n_specific]; n_groups];
                for g in 0..n_groups {
                    for s in 0..v.n_specific {
                        s2_spec[g][s] = res.s2_spec[g * v.n_specific + s];
                        w_spec[g][s] = res.w_spec[g * v.n_specific + s];
                    }
                }
                return (
                    res.loglik, counts, res.w_acc, res.s1_g, res.s2_g, s2_spec, w_spec,
                );
            }
        }
    }
    if device == crate::Device::Gpu {
        eprintln!(
            "fast-mlsirm: GPU bifactor E-step requested but no usable GPU adapter was found; \
             falling back to CPU implementation."
        );
    }

    let mut counts: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_groups);
    for g in 0..n_groups {
        let _ = g;
        let mut cg = Vec::with_capacity(v.n_items);
        for i in 0..v.n_items {
            let n_nodes = if v.item_block[i].is_some() {
                qg * qs
            } else {
                qg
            };
            cg.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
        }
        counts.push(cg);
    }
    let mut w_acc = vec![0.0f64; n_groups];
    let mut s1_g = vec![0.0f64; n_groups];
    let mut s2_g = vec![0.0f64; n_groups];
    let mut s2_spec = vec![vec![0.0f64; v.n_specific]; n_groups];
    // Per-(group, block) posterior mass for the specific-variance M-step:
    // persons with no observed item in block `s` contribute nothing to that
    // block's moments and must not dilute its denominator (review fix for
    // block-wise MAR with `estimate_specific_vars`).
    let mut w_spec = vec![vec![0.0f64; v.n_specific]; n_groups];

    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut gen_log = vec![0.0f64; qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut post_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];

    let mut loglik = 0.0f64;
    for p in 0..v.n_persons {
        let g = group_id[p];
        let tables = &tables_groups[g];
        gen_log.copy_from_slice(log_wg);
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            let lp = &tables[i];
            for t in 0..qg {
                gen_log[t] += lp[t * v.n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for t in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        acc += tables[i][(t * qs + h) * v.n_cat + yc];
                    }
                    block_acc[(s * qg + t) * qs + h] = acc;
                }
            }
            for t in 0..qg {
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + t) * qs + h];
                }
                log_i[s * qg + t] = log_sum_exp(&tmp_h);
            }
        }
        for t in 0..qg {
            let mut acc = gen_log[t];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + t];
            }
            log_like_g[t] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        loglik += log_lp;
        for t in 0..qg {
            post_g[t] = (log_like_g[t] - log_lp).exp();
        }
        // General moments (common-scale nodes) for the group M-step.
        for t in 0..qg {
            w_acc[g] += post_g[t];
            s1_g[g] += post_g[t] * tg_groups[g][t];
            s2_g[g] += post_g[t] * tg_groups[g][t] * tg_groups[g][t];
        }
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            for t in 0..qg {
                counts[g][i][t][yc] += post_g[t];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            let any_obs = members.iter().any(|&i| is_obs(p, i));
            if !any_obs {
                continue;
            }
            for t in 0..qg {
                let Some(mut others) = general_only_without_prior(gen_log[t], log_wg[t]) else {
                    continue;
                };
                for s2 in 0..v.n_specific {
                    if s2 != s {
                        others += log_i[s2 * qg + t];
                    }
                }
                for h in 0..qs {
                    let log_post = log_wg[t] + block_acc[(s * qg + t) * qs + h] + others - log_lp;
                    let post = log_post.exp();
                    if !post.is_finite() {
                        continue;
                    }
                    // Specific moments at zero mean for the variance M-step.
                    let ts = ts_groups[g][s][h];
                    w_spec[g][s] += post;
                    s2_spec[g][s] += post * ts * ts;
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        counts[g][i][t * qs + h][yc] += post;
                    }
                }
            }
        }
    }
    (loglik, counts, w_acc, s1_g, s2_g, s2_spec, w_spec)
}
// Test-only fixed inputs and complete returned-array projection.
// Shared fixture bytes are included inside each private module.
pub(crate) fn fixture(
    q: usize,
    masked: bool,
    shift: f64,
    zero_weight: bool,
) -> (Vec<u64>, usize, usize) {
    assert!(q >= 121);
    let np = 12;
    let ni = 5;
    let ns = 2;
    let nc = 4;
    let v = Validated {
        n_persons: np,
        n_items: ni,
        n_specific: ns,
        n_cat: nc,
        m1: nc - 1,
        blocks: vec![vec![0, 1], vec![2, 3]],
        general_only: vec![4],
        item_block: vec![Some(0), Some(0), Some(1), Some(1), None],
    };
    let group: Vec<usize> = (0..np).map(|p| p / 6).collect();
    let mut y = vec![0usize; np * ni];
    let mut mask = vec![true; np * ni];
    for p in 0..np {
        let pattern = p % 3;
        for i in 0..ni {
            y[p * ni + i] = (pattern + i) % nc;
            if masked && ((pattern == 1 && i < 2) || (pattern == 2 && (i == 0 || i == 3))) {
                mask[p * ni + i] = false;
                y[p * ni + i] = usize::MAX; // Unobserved payload must never index tables.
            }
        }
    }
    let observed = if masked { Some(mask.as_slice()) } else { None };
    let mut patterns = std::collections::HashSet::new();
    for p in 0..np {
        for (block, members) in v.blocks.iter().enumerate() {
            let values: Vec<Option<usize>> = members
                .iter()
                .map(|&i| {
                    if mask[p * ni + i] {
                        Some(y[p * ni + i])
                    } else {
                        None
                    }
                })
                .collect();
            patterns.insert((group[p], block, values));
        }
    }
    let expected = patterns.len() * q;
    let params: Vec<Vec<ItemParams>> = (0..2)
        .map(|g| {
            (0..ni)
                .map(|i| ItemParams {
                    a_g: (if i == 3 { -0.9 } else { 1.1 }) + 0.08 * (g as f64),
                    a_s: if i == 4 {
                        None
                    } else {
                        Some(0.7 + 0.05 * (g as f64))
                    },
                    d: vec![1.2 + shift + 0.03 * (i as f64), 0.1 + shift, -1.1 + shift],
                })
                .collect()
        })
        .collect();
    let (x, w) = crate::quadrature::gh_rule(q).expect("caller quadrature");
    let mut log_wg: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let mut log_ws = log_wg.clone();
    if zero_weight {
        log_wg[0] = f64::NEG_INFINITY;
        log_ws[0] = f64::NEG_INFINITY;
    }
    let tg: Vec<Vec<f64>> = (0..2)
        .map(|g| {
            x.iter()
                .map(|&n| if g == 0 { n } else { 0.4 + 1.2 * n })
                .collect()
        })
        .collect();
    let ts: Vec<Vec<Vec<f64>>> = (0..2)
        .map(|g| {
            (0..ns)
                .map(|s| {
                    x.iter()
                        .map(|&n| (1.0 + 0.1 * (g + s) as f64) * n)
                        .collect()
                })
                .collect()
        })
        .collect();
    reset_calls();
    let (ll, counts, w_acc, s1_g, s2_g, s2_spec, w_spec) = e_step_multigroup(
        &v,
        &y,
        observed,
        &group,
        2,
        &params,
        &tg,
        &ts,
        &log_wg,
        &log_ws,
        q,
        q,
        crate::Device::Cpu,
    );
    let calls = np * ns * q;
    let mut output = vec![ll];
    for group in counts {
        for item in group {
            for node in item {
                output.extend(node);
            }
        }
    }
    output.extend(w_acc);
    output.extend(s1_g);
    output.extend(s2_g);
    for row in s2_spec {
        output.extend(row);
    }
    for row in w_spec {
        output.extend(row);
    }
    assert!(
        output.iter().all(|v| v.is_finite()),
        "Nonfinite returned component"
    );
    (
        output.into_iter().map(f64::to_bits).collect(),
        calls,
        expected,
    )
}
