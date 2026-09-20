//! Same-host CPU+GPU concurrent bifactor E-step person split (#2001 L3).
//!
//! Persons are partitioned into disjoint shards; each shard runs on one device.
//! Partials merge in fixed ascending `person_start` order: per-person `f64`
//! log-likelihood terms concatenate in that person order (bit-exact across
//! shard counts), while expected-count tensors sum commutatively after the
//! same sort so their totals do not depend on completion order.

use crate::bifactor_grm::{log_sum_exp, general_only_without_prior, Validated};

/// Per-shard execution record for provenance (#2001 §3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EstepShardProvenance {
    pub shard_index: usize,
    pub device: String,
    pub person_start: usize,
    pub person_end: usize,
}

/// Effective-device provenance for one E-step sweep (#2001 §3.4).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EstepExecutionProvenance {
    pub requested_device: String,
    pub effective_device: String,
    pub shards: Vec<EstepShardProvenance>,
}

/// One partial E-step over a disjoint person range.
#[derive(Clone, Debug)]
pub(crate) struct EstepPartial {
    pub person_start: usize,
    pub person_end: usize,
    pub device_label: &'static str,
    /// Per-person marginal log-likelihood terms in ascending person index.
    pub per_person_loglik: Vec<f64>,
    pub counts: Vec<Vec<Vec<f64>>>,
}

/// Zero-initialized expected-count tensor matching `v`.
pub(crate) fn zero_estep_counts(v: &Validated, qg: usize, qs: usize) -> Vec<Vec<Vec<f64>>> {
    let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
    for i in 0..v.n_items {
        let n_nodes = if v.item_block[i].is_some() {
            qg * qs
        } else {
            qg
        };
        counts.push(vec![vec![0.0f64; v.n_cat]; n_nodes]);
    }
    counts
}

/// Scalar CPU E-step over `person_start..person_end` (half-open).
#[allow(clippy::too_many_arguments)]
pub(crate) fn e_step_cpu_person_range(
    v: &Validated,
    y: &[usize],
    observed: Option<&[bool]>,
    tables: &[Vec<f64>],
    log_wg: &[f64],
    log_ws: &[f64],
    qg: usize,
    qs: usize,
    person_start: usize,
    person_end: usize,
) -> EstepPartial {
    let is_obs = |p: usize, i: usize| observed.is_none_or(|o| o[p * v.n_items + i]);
    let mut counts = zero_estep_counts(v, qg, qs);
    let mut block_acc = vec![0.0f64; v.n_specific * qg * qs];
    let mut log_i = vec![0.0f64; v.n_specific * qg];
    let mut gen_log = vec![0.0f64; qg];
    let mut log_like_g = vec![0.0f64; qg];
    let mut post_g = vec![0.0f64; qg];
    let mut tmp_h = vec![0.0f64; qs];

    let mut per_person_loglik = Vec::with_capacity(person_end - person_start);
    for p in person_start..person_end {
        gen_log.copy_from_slice(log_wg);
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            let lp = &tables[i];
            for g in 0..qg {
                gen_log[g] += lp[g * v.n_cat + yc];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            for g in 0..qg {
                for h in 0..qs {
                    let mut acc = log_ws[h];
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        acc += tables[i][(g * qs + h) * v.n_cat + yc];
                    }
                    block_acc[(s * qg + g) * qs + h] = acc;
                }
            }
            for g in 0..qg {
                for h in 0..qs {
                    tmp_h[h] = block_acc[(s * qg + g) * qs + h];
                }
                log_i[s * qg + g] = log_sum_exp(&tmp_h);
            }
        }
        for g in 0..qg {
            let mut acc = gen_log[g];
            for s in 0..v.n_specific {
                acc += log_i[s * qg + g];
            }
            log_like_g[g] = acc;
        }
        let log_lp = log_sum_exp(&log_like_g);
        per_person_loglik.push(log_lp);
        for g in 0..qg {
            post_g[g] = (log_like_g[g] - log_lp).exp();
        }
        for &i in &v.general_only {
            if !is_obs(p, i) {
                continue;
            }
            let yc = y[p * v.n_items + i];
            for g in 0..qg {
                counts[i][g][yc] += post_g[g];
            }
        }
        for (s, members) in v.blocks.iter().enumerate() {
            let any_obs = members.iter().any(|&i| is_obs(p, i));
            if !any_obs {
                continue;
            }
            for g in 0..qg {
                let Some(mut others) = general_only_without_prior(gen_log[g], log_wg[g]) else {
                    continue;
                };
                for s2 in 0..v.n_specific {
                    if s2 != s {
                        others += log_i[s2 * qg + g];
                    }
                }
                for h in 0..qs {
                    let log_post = log_wg[g] + block_acc[(s * qg + g) * qs + h] + others - log_lp;
                    let post = log_post.exp();
                    if !post.is_finite() {
                        continue;
                    }
                    for &i in members {
                        if !is_obs(p, i) {
                            continue;
                        }
                        let yc = y[p * v.n_items + i];
                        counts[i][g * qs + h][yc] += post;
                    }
                }
            }
        }
    }
    EstepPartial {
        person_start,
        person_end,
        device_label: "cpu",
        per_person_loglik,
        counts,
    }
}

/// Merge partials in ascending `person_start` order (fixed-order f64 reduction).
pub(crate) fn merge_estep_partials(mut partials: Vec<EstepPartial>) -> (f64, Vec<Vec<Vec<f64>>>) {
    partials.sort_by_key(|p| p.person_start);
    let mut loglik = 0.0f64;
    let mut merged: Option<Vec<Vec<Vec<f64>>>> = None;
    for part in &partials {
        for &ll_p in &part.per_person_loglik {
            loglik += ll_p;
        }
    }
    for part in partials {
        match &mut merged {
            None => merged = Some(part.counts),
            Some(base) => add_estep_counts(base, &part.counts),
        }
    }
    (loglik, merged.unwrap_or_default())
}

fn add_estep_counts(base: &mut [Vec<Vec<f64>>], add: &[Vec<Vec<f64>>]) {
    for (item_base, item_add) in base.iter_mut().zip(add.iter()) {
        for (node_base, node_add) in item_base.iter_mut().zip(item_add.iter()) {
            for (c_base, &c_add) in node_base.iter_mut().zip(node_add.iter()) {
                *c_base += c_add;
            }
        }
    }
}

/// Build provenance from merged shard partials.
pub(crate) fn provenance_from_partials(
    requested: &str,
    partials: &[EstepPartial],
) -> EstepExecutionProvenance {
    let mut devices: Vec<&str> = Vec::new();
    for p in partials {
        if !devices.contains(&p.device_label) {
            devices.push(p.device_label);
        }
    }
    let effective_device = devices.join("+");
    EstepExecutionProvenance {
        requested_device: requested.to_string(),
        effective_device,
        shards: partials
            .iter()
            .enumerate()
            .map(|(idx, p)| EstepShardProvenance {
                shard_index: idx,
                device: p.device_label.to_string(),
                person_start: p.person_start,
                person_end: p.person_end,
            })
            .collect(),
    }
}

/// Deterministic person boundaries for `n_shards` equal-sized CPU shards.
pub(crate) fn cpu_shard_bounds(n_persons: usize, n_shards: usize) -> Vec<(usize, usize)> {
    let n_shards = n_shards.max(1);
    let chunk = n_persons.div_ceil(n_shards);
    let mut bounds = Vec::with_capacity(n_shards);
    let mut start = 0usize;
    while start < n_persons {
        let end = (start + chunk).min(n_persons);
        if start < end {
            bounds.push((start, end));
        }
        start = end;
    }
    bounds
}

