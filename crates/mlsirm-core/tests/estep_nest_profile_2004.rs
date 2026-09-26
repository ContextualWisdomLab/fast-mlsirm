//! #2004 nest-audit profile + loglik-only E-step evidence for bifactor GRM.
//!
//! Runs a CP3-shaped CPU E-step with section timers. Quadrature node counts
//! are explicit caller arguments (ADR-0028 / #1929), not crate defaults.

use mlsirm_core::bifactor_grm::{
    bifactor_grm_marginal_loglik, enable_estep_nest_profile, take_estep_nest_profile,
};
use std::time::Instant;

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
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
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

#[test]
fn estep_nest_profile_2004_loglik_only_skips_count_fill() {
    let y = synthetic_y(2004);
    let (a_g, a_s, thr) = synthetic_params();
    // Warmup (untimed).
    let _ = bifactor_grm_marginal_loglik(
        &a_g,
        &a_s,
        &thr,
        &y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Q,
        Q,
    )
    .expect("warmup marginal loglik");

    enable_estep_nest_profile();
    let t0 = Instant::now();
    let ll = bifactor_grm_marginal_loglik(
        &a_g,
        &a_s,
        &thr,
        &y,
        None,
        &SPECIFIC_MAP,
        N_PERSONS,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        Q,
        Q,
    )
    .expect("profiled marginal loglik");
    let wall_ns = t0.elapsed().as_nanos();
    let prof = take_estep_nest_profile().expect("profile was enabled");
    let timed = prof.gen_only_ns + prof.block_acc_ns + prof.posterior_ns;
    assert!(ll.is_finite(), "loglik must be finite");
    assert_eq!(prof.n_persons, N_PERSONS);
    assert!(timed > 0, "section timers must record work");
    let block_share = prof.block_acc_ns as f64 / timed as f64;
    eprintln!(
        "2004 AFTER loglik-only N={N_PERSONS} items={N_ITEMS} q={Q}: wall_ms={:.3} \
         gen_only_ms={:.3} block_acc_ms={:.3} posterior_ms={:.3} \
         block_share={:.1}% ll={ll:.6}",
        wall_ns as f64 / 1e6,
        prof.gen_only_ns as f64 / 1e6,
        prof.block_acc_ns as f64 / 1e6,
        prof.posterior_ns as f64 / 1e6,
        100.0 * block_share,
    );
    // Baseline (counts filled) on the same host/fixture was wall≈470ms with
    // posterior_ms≈311 (67% of timed sections). After skip, wall must drop
    // below that measured posterior term and block_acc must dominate.
    assert!(
        wall_ns as f64 / 1e6 < 350.0,
        "expected wall below baseline-with-counts (~470ms), got {:.1}ms",
        wall_ns as f64 / 1e6
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

    assert_eq!(naive, hoist, "A-hoist must be bit-identical on this fixture");
    let speedup = naive_ns as f64 / hoist_ns as f64;
    eprintln!(
        "2004 A-hoist microbench (fully observed) q={Q} reps={reps}: \
         naive_ms={:.3} hoist_ms={:.3} speedup={speedup:.3}x (not applied; regression risk)",
        naive_ns as f64 / 1e6,
        hoist_ns as f64 / 1e6,
    );
}
