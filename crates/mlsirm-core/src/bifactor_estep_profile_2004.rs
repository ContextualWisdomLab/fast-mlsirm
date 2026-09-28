//! Synthetic characterization of the loglik-only bifactor E-step (#2004).
//!
//! Quadrature node counts are explicit caller arguments (ADR-0028 / #1929).
//! Timings here are a regression guard on this fixture, not product-data
//! acceptance evidence, and they do not close #2004.

use std::time::Instant;

use super::{
    bifactor_grm_marginal_loglik, e_step, enable_estep_nest_profile, fill_logprob_tables, gh_rule,
    pack_params, take_estep_nest_profile, validate, BifactorGrmConfig,
};

const N_PERSONS: usize = 240;
const N_ITEMS: usize = 13;
const N_SPECIFIC: usize = 3;
const N_CAT: usize = 4;
const Q: usize = 41;
/// CP3-like map: blocks of 4 + one general-only item.
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1];

fn synthetic_y(seed: u64) -> Vec<usize> {
    let mut state = seed;
    let mut y = vec![0usize; N_PERSONS * N_ITEMS];
    for slot in &mut y {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *slot = ((state >> 33) as usize) % N_CAT;
    }
    y
}

fn synthetic_params() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a_g = vec![1.0f64; N_ITEMS];
    let mut a_s = vec![0.0f64; N_ITEMS];
    for (i, &s) in SPECIFIC_MAP.iter().enumerate() {
        a_g[i] = 0.8 + 0.05 * (i as f64);
        if s >= 0 {
            a_s[i] = 0.7 + 0.03 * (i as f64);
        }
    }
    let mut thr = vec![0.0f64; N_ITEMS * (N_CAT - 1)];
    for i in 0..N_ITEMS {
        thr[i * 3] = 1.0;
        thr[i * 3 + 1] = 0.0;
        thr[i * 3 + 2] = -1.0;
    }
    (a_g, a_s, thr)
}

struct Prepared {
    v: super::Validated,
    y: Vec<usize>,
    tables: Vec<Vec<f64>>,
    log_wg: Vec<f64>,
    log_ws: Vec<f64>,
    tg: &'static [f64],
    ts: &'static [f64],
    a_g: Vec<f64>,
    a_s: Vec<f64>,
    thr: Vec<f64>,
}

fn prepare() -> Prepared {
    let y = synthetic_y(2004);
    let (a_g, a_s, thr) = synthetic_params();
    let cfg = BifactorGrmConfig {
        q_general: Q,
        q_specific: Q,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
        device: crate::Device::Cpu,
    };
    let v = validate(
        &y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &cfg,
    )
    .expect("synthetic fixture must validate");
    let params = pack_params(&v, &a_g, &a_s, &thr);
    let (tg, wg) = gh_rule(Q).expect("q=41 rule must exist");
    let (ts, ws) = gh_rule(Q).expect("q=41 rule must exist");
    let tables = fill_logprob_tables(&v, &params, tg, ts, tg.len(), ts.len());
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    Prepared {
        v,
        y,
        tables,
        log_wg,
        log_ws,
        tg,
        ts,
        a_g,
        a_s,
        thr,
    }
}

fn run_estep(prep: &Prepared, accumulate_counts: bool) -> (f64, Vec<Vec<Vec<f64>>>) {
    e_step(
        &prep.v,
        &prep.y,
        None,
        &prep.tables,
        &prep.log_wg,
        &prep.log_ws,
        prep.tg.len(),
        prep.ts.len(),
        prep.tg,
        prep.ts,
        crate::Device::Cpu,
        accumulate_counts,
    )
}

#[test]
fn estep_nest_profile_2004_loglik_only_skips_count_fill() {
    let prep = prepare();
    // Warmup (untimed).
    let _ = run_estep(&prep, false);

    let t_counts = Instant::now();
    let (ll_counts, counts) = run_estep(&prep, true);
    let counts_ns = t_counts.elapsed().as_nanos();

    enable_estep_nest_profile();
    let t_skip = Instant::now();
    let (ll_skip, skipped) = run_estep(&prep, false);
    let skip_ns = t_skip.elapsed().as_nanos();
    let prof = take_estep_nest_profile().expect("profile was enabled");

    let ll_marginal = bifactor_grm_marginal_loglik(
        &prep.a_g,
        &prep.a_s,
        &prep.thr,
        &prep.y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Q,
        Q,
    )
    .expect("marginal loglik");

    assert!(ll_counts.is_finite() && ll_skip.is_finite());
    assert_eq!(
        ll_counts, ll_skip,
        "loglik-only E-step must match the count-filling sweep"
    );
    assert_eq!(
        ll_marginal, ll_skip,
        "bifactor_grm_marginal_loglik must use the loglik-only sweep"
    );
    assert!(!counts.is_empty(), "count-filling sweep must return counts");
    assert!(
        counts
            .iter()
            .any(|item| item.iter().flatten().any(|c| *c != 0.0)),
        "synthetic responses must produce a non-zero expected count"
    );
    assert!(
        skipped.is_empty(),
        "loglik-only sweep must return no counts"
    );
    assert_eq!(prof.n_persons, N_PERSONS);
    let timed = prof.gen_only_ns + prof.block_acc_ns + prof.posterior_ns;
    assert!(timed > 0, "section timers must record work");
    let block_share = prof.block_acc_ns as f64 / timed as f64;
    eprintln!(
        "2004 synthetic loglik-only N={N_PERSONS} items={N_ITEMS} q={Q}: \
         counts_ms={:.3} skip_ms={:.3} block_share={:.1}% ll={ll_skip:.6}",
        counts_ns as f64 / 1e6,
        skip_ns as f64 / 1e6,
        100.0 * block_share,
    );
    assert!(
        block_share >= 0.70,
        "after skipping counts, block_acc should dominate; share={block_share:.3}"
    );
}

/// Documents that defect-A hoist of `is_obs`/`y` is NOT a free win on the
/// common fully-observed path (measured regression; see ledger).
#[test]
fn estep_nest_a_hoist_regression_documented_2004() {
    let qg = Q;
    let qs = Q;
    let n_cat = N_CAT;
    let members: [usize; 4] = [0, 1, 2, 3];
    let y = [0usize, 1, 2, 3];
    let log_ws: Vec<f64> = (0..qs).map(|h| -((h + 1) as f64).ln()).collect();
    let mut tables: Vec<Vec<f64>> = Vec::with_capacity(4);
    for i in 0..4 {
        let mut t = vec![0.0f64; qg * qs * n_cat];
        for g in 0..qg {
            for h in 0..qs {
                for c in 0..n_cat {
                    t[(g * qs + h) * n_cat + c] = -0.01 * ((i + g + h + c) as f64);
                }
            }
        }
        tables.push(t);
    }
    let is_obs = |_p: usize, _i: usize| true;
    let reps = 80usize;

    let mut naive = vec![0.0f64; qg * qs];
    let t_naive = Instant::now();
    for _ in 0..reps {
        for g in 0..qg {
            for h in 0..qs {
                let mut acc = log_ws[h];
                for &i in &members {
                    if !is_obs(0, i) {
                        continue;
                    }
                    let yc = y[i];
                    acc += tables[i][(g * qs + h) * n_cat + yc];
                }
                naive[g * qs + h] = acc;
            }
        }
    }
    let naive_ns = t_naive.elapsed().as_nanos();

    let mut hoist = vec![0.0f64; qg * qs];
    let mut obs: Vec<(usize, usize)> = Vec::new();
    let t_hoist = Instant::now();
    for _ in 0..reps {
        obs.clear();
        for &i in &members {
            if is_obs(0, i) {
                obs.push((i, y[i]));
            }
        }
        for g in 0..qg {
            for h in 0..qs {
                let mut acc = log_ws[h];
                for &(i, yc) in &obs {
                    acc += tables[i][(g * qs + h) * n_cat + yc];
                }
                hoist[g * qs + h] = acc;
            }
        }
    }
    let hoist_ns = t_hoist.elapsed().as_nanos();

    assert_eq!(
        naive, hoist,
        "A-hoist must be bit-identical on this fixture"
    );
    let speedup = naive_ns as f64 / hoist_ns as f64;
    eprintln!(
        "2004 A-hoist microbench (fully observed) q={Q} reps={reps}: \
         naive_ms={:.3} hoist_ms={:.3} speedup={speedup:.3}x (not applied; regression risk)",
        naive_ns as f64 / 1e6,
        hoist_ns as f64 / 1e6,
    );
}
