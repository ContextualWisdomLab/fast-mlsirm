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
