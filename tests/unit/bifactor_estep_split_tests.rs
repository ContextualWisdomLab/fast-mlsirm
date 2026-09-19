//! L3 same-host CPU+GPU bifactor E-step split contracts (#2001).

use super::{
    e_step, e_step_cpu_sharded, fill_logprob_tables, gh_rule, initial_params, validate,
    BifactorGrmConfig,
};
use crate::bifactor_estep_split::{cpu_shard_bounds, merge_estep_partials, EstepPartial};

const TINY_N_ITEMS: usize = 6;
const TINY_N_SPECIFIC: usize = 2;
const TINY_N_CAT: usize = 3;
const TINY_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [0, 0, 0, 1, 1, 1];

fn tiny_data() -> (Vec<usize>, usize) {
    let n_persons = 12;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for (p, row) in y.chunks_mut(TINY_N_ITEMS).enumerate() {
        for (i, cell) in row.iter_mut().enumerate() {
            *cell = (p + i) % TINY_N_CAT;
        }
    }
    (y, n_persons)
}

fn tiny_estep_tables() -> (
    super::Validated,
    Vec<usize>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
) {
    let (y, n_persons) = tiny_data();
    let cfg = BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 10,
        tol: 1e-6,
        n_starts: 1,
        seed: 1,
        newton_iter: 10,
        ridge: 1e-8,
        device: crate::Device::Cpu,
    };
    let v = validate(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("tiny fixture validates");
    let (tg, wg) = gh_rule(7).unwrap();
    let (ts, ws) = gh_rule(7).unwrap();
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, &tg, &ts, 7, 7);
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    (v, y, log_wg, log_ws, tg.to_vec(), ts.to_vec(), 7, 7)
}

#[test]
fn cpu_sharded_estep_matches_serial_bit_exact() {
    let (v, y, log_wg, log_ws, tg, ts, qg, qs) = tiny_estep_tables();
    let params = initial_params(&v, &y, None, 1, 0);
    let tables = fill_logprob_tables(&v, &params, &tg, &ts, qg, qs);

    let (ll_serial, counts_serial, _) = e_step(
        &v,
        &y,
        None,
        &tables,
        &log_wg,
        &log_ws,
        qg,
        qs,
        &tg,
        &ts,
        crate::Device::Cpu,
    );

    for n_shards in [2, 3, 4, 6] {
        let (ll_shard, counts_shard) = e_step_cpu_sharded(
            &v, &y, None, &tables, &log_wg, &log_ws, qg, qs, n_shards,
        );
        assert_eq!(
            ll_serial, ll_shard,
            "loglik must be bit-identical for {n_shards} CPU shards"
        );
        assert_eq!(counts_serial, counts_shard);
    }
}

#[test]
fn merge_estep_partials_is_order_invariant_for_disjoint_shards() {
    let (v, y, log_wg, log_ws, tg, ts, qg, qs) = tiny_estep_tables();
    let params = initial_params(&v, &y, None, 1, 0);
    let tables = fill_logprob_tables(&v, &params, &tg, &ts, qg, qs);

    let bounds = cpu_shard_bounds(v.n_persons, 4);
    let partials: Vec<EstepPartial> = bounds
        .iter()
        .map(|&(start, end)| {
            crate::bifactor_estep_split::e_step_cpu_person_range(
                &v, &y, None, &tables, &log_wg, &log_ws, qg, qs, start, end,
            )
        })
        .collect();

    let forward = merge_estep_partials(partials.clone());
    let reverse = merge_estep_partials(partials.into_iter().rev().collect());
    assert_eq!(forward.0, reverse.0);
    assert_eq!(forward.1, reverse.1);
}

#[test]
fn split_device_records_effective_device_provenance() {
    let (v, y, log_wg, log_ws, tg, ts, qg, qs) = tiny_estep_tables();
    let params = initial_params(&v, &y, None, 1, 0);
    let tables = fill_logprob_tables(&v, &params, &tg, &ts, qg, qs);

    let (_, _, prov) = e_step(
        &v,
        &y,
        None,
        &tables,
        &log_wg,
        &log_ws,
        qg,
        qs,
        &tg,
        &ts,
        crate::Device::Split {
            gpu_person_start: v.n_persons / 2,
        },
    );
    let prov = prov.expect("split E-step must record provenance");
    assert_eq!(prov.requested_device, "split");
    assert!(
        prov.effective_device == "cpu+gpu" || prov.effective_device == "cpu",
        "effective_device must reflect adapter availability, got {}",
        prov.effective_device
    );
    assert!(!prov.shards.is_empty());
}

fn measure_fixture(q: usize) -> (
    super::Validated,
    Vec<usize>,
    Vec<Vec<f64>>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    usize,
    usize,
) {
    let n_persons = 120usize;
    let n_items = TINY_N_ITEMS;
    let mut y = vec![0usize; n_persons * n_items];
    for (p, row) in y.chunks_mut(n_items).enumerate() {
        for (i, cell) in row.iter_mut().enumerate() {
            *cell = (p + i) % TINY_N_CAT;
        }
    }
    let cfg = BifactorGrmConfig {
        q_general: q,
        q_specific: q,
        max_iter: 10,
        tol: 1e-6,
        n_starts: 1,
        seed: 42,
        newton_iter: 10,
        ridge: 1e-8,
        device: crate::Device::Cpu,
    };
    let v = validate(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        n_items,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("measure fixture validates");
    let (tg, wg) = gh_rule(q).unwrap();
    let (ts, ws) = gh_rule(q).unwrap();
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, &tg, &ts, q, q);
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    (
        v,
        y,
        tables,
        log_wg,
        log_ws,
        tg.to_vec(),
        ts.to_vec(),
        q,
        q,
    )
}

fn max_count_abs_diff(a: &[Vec<Vec<f64>>], b: &[Vec<Vec<f64>>]) -> f64 {
    let mut max_diff = 0.0f64;
    for (aa, bb) in a.iter().zip(b.iter()) {
        for (an, bn) in aa.iter().zip(bb.iter()) {
            for (ac, bc) in an.iter().zip(bn.iter()) {
                max_diff = max_diff.max((ac - bc).abs());
            }
        }
    }
    max_diff
}

/// Manual overlap + parity evidence for PR #2043 (not a CI gate).
#[test]
fn measure_concurrent_split_estep_vs_cpu_reference() {
    use std::time::Instant;

    const SPLIT_AT: usize = 60;

    for q in [21usize, 121usize] {
        let (v, y, tables, log_wg, log_ws, tg, ts, qg, qs) = measure_fixture(q);

        let (ll_cpu, counts_cpu, _) = e_step(
            &v,
            &y,
            None,
            &tables,
            &log_wg,
            &log_ws,
            qg,
            qs,
            &tg,
            &ts,
            crate::Device::Cpu,
        );

        let tables_wrapped = vec![tables.clone()];
        let tg_wrapped = vec![tg.clone()];
        let ts_wrapped = vec![vec![ts.clone(); v.n_specific]];
        let inputs = crate::gpu_bifactor::ReducedEstepInputs {
            y: &y,
            observed: None,
            group_id: None,
            n_persons: v.n_persons,
            n_items: v.n_items,
            n_specific: v.n_specific,
            n_cat: v.n_cat,
            qg,
            qs,
            n_groups: 1,
            tables_groups: &tables_wrapped,
            item_block: &v.item_block,
            blocks: &v.blocks,
            tg_groups: &tg_wrapped,
            ts_groups: &ts_wrapped,
            log_wg: &log_wg,
            log_ws: &log_ws,
        };

        let wall_start = Instant::now();
        let gpu_submit = crate::gpu_bifactor::e_step_reduced_gpu_submit(
            &inputs,
            SPLIT_AT,
            v.n_persons,
        );
        let gpu_dispatched = Instant::now();

        let Some((pending, meta)) = gpu_submit else {
            println!(
                "\n[q={q}] SKIP: no GPU adapter (split would fall back to CPU-only)"
            );
            continue;
        };

        let cpu_start = Instant::now();
        let cpu_partial = crate::bifactor_estep_split::e_step_cpu_person_range(
            &v,
            &y,
            None,
            &tables,
            &log_wg,
            &log_ws,
            qg,
            qs,
            0,
            SPLIT_AT,
        );
        let cpu_end = Instant::now();

        let gpu_out = crate::gpu_bifactor::complete_reduced_gpu_submit(&pending, &meta)
            .expect("GPU readback must succeed when submit succeeded");
        let gpu_end = Instant::now();

        let gpu_partial = crate::bifactor_estep_split::EstepPartial {
            person_start: SPLIT_AT,
            person_end: v.n_persons,
            device_label: "gpu",
            per_person_loglik: gpu_out.per_person_loglik,
            counts: {
                let stride = gpu_out.counts_stride_nodes;
                let mut counts: Vec<Vec<Vec<f64>>> = Vec::with_capacity(v.n_items);
                for i in 0..v.n_items {
                    let base = i * stride * v.n_cat;
                    if v.item_block[i].is_some() {
                        counts.push(
                            gpu_out.counts[base..base + qg * qs * v.n_cat]
                                .chunks_exact(v.n_cat)
                                .map(<[f64]>::to_vec)
                                .collect(),
                        );
                    } else {
                        counts.push(
                            gpu_out.counts[base..base + qg * v.n_cat]
                                .chunks_exact(v.n_cat)
                                .map(<[f64]>::to_vec)
                                .collect(),
                        );
                    }
                }
                counts
            },
        };
        let partials = vec![cpu_partial, gpu_partial];
        let prov = crate::bifactor_estep_split::provenance_from_partials("split", &partials);
        let (ll_split, counts_split) = merge_estep_partials(partials);
        let wall_end = Instant::now();

        let cpu_secs = (cpu_end - cpu_start).as_secs_f64();
        let gpu_active_secs = (gpu_end - gpu_dispatched).as_secs_f64();
        let overlap = cpu_start < gpu_end && gpu_dispatched < cpu_end;
        let ll_diff = (ll_cpu - ll_split).abs();
        let count_diff = max_count_abs_diff(&counts_cpu, &counts_split);

        println!("\n=== measure_concurrent_split_estep_vs_cpu_reference q={q} ===");
        println!("effective_device={}", prov.effective_device);
        for shard in &prov.shards {
            println!(
                "  shard {}: device={} persons=[{}, {})",
                shard.shard_index, shard.device, shard.person_start, shard.person_end
            );
        }
        println!(
            "cpu_shard: start={:.6}s end={:.6}s duration={cpu_secs:.6}s (monotonic offset from wall_start)",
            (cpu_start - wall_start).as_secs_f64(),
            (cpu_end - wall_start).as_secs_f64(),
        );
        println!(
            "gpu_shard: dispatched={:.6}s end={:.6}s active_duration={gpu_active_secs:.6}s",
            (gpu_dispatched - wall_start).as_secs_f64(),
            (gpu_end - wall_start).as_secs_f64(),
        );
        println!("intervals_overlap={overlap}");
        println!("cpu_loglik={ll_cpu:.12e} split_loglik={ll_split:.12e}");
        println!("max_abs_loglik_diff={ll_diff:.6e}");
        println!("max_abs_count_diff={count_diff:.6e}");
        let wall_secs = (wall_end - wall_start).as_secs_f64();
        println!("wall_seconds={wall_secs:.6}");

        assert_eq!(
            prov.effective_device, "cpu+gpu",
            "measurement requires both CPU and GPU shards, not provenance-only naming"
        );
        assert!(
            overlap,
            "CPU and GPU shard intervals must overlap (concurrent execution evidence)"
        );
    }
}
