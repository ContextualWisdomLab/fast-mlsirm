//! Unit contract for the single-group polytomous bifactor GRM fitter
//! (`mlsirm_core::bifactor_grm`, stage 1 of #1912).
//!
//! * Dimension-reduction exactness: the Gibbons-Hedeker reduced marginal
//!   log-likelihood must equal brute-force full product-grid integration to
//!   `<= 1e-10` on a tiny model (4 items, 2 specifics, `Q = 7`). The two paths
//!   reorder the same finite sum, so any larger gap is an implementation bug,
//!   not quadrature error.
//! * Argument validation and failure reporting: unsupported quadrature,
//!   bad iteration/tolerance/start budgets, malformed specific maps,
//!   out-of-range categories, unobserved categories, and the non-convergence
//!   report (`converged == false`, `termination_reason == "max_iter_reached"`).
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Gibbons, R. D., & Hedeker, D. R. (1992). Full-information item bi-factor
//! analysis. *Psychometrika, 57*(3), 423-436.
//! https://doi.org/10.1007/BF02295430

use crate::bifactor_grm::{
    bifactor_grm_marginal_loglik, bifactor_grm_marginal_loglik_brute, fit_bifactor_grm,
    BifactorGrmConfig,
};

// ---------------------------------------------------------------------------
// Dimension-reduction exactness (tiny model, Q = 7).
// ---------------------------------------------------------------------------

const TINY_N_ITEMS: usize = 4;
const TINY_N_SPECIFIC: usize = 2;
const TINY_N_CAT: usize = 3;
/// Item -> specific map: items 0-1 on specific 0, items 2-3 on specific 1.
const TINY_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [0, 0, 1, 1];

fn tiny_params() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let a_general = vec![1.2, -0.9, 1.0, 0.7];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6];
    // Strictly decreasing within each item (n_cat - 1 = 2 intercepts).
    let thresholds = vec![0.9, -0.7, 0.5, -1.0, 1.1, -0.4, 0.2, -1.3];
    (a_general, a_specific, thresholds)
}

fn tiny_data() -> (Vec<usize>, usize) {
    // Deterministic toy responses covering every category of every item.
    let n_persons = 12usize;
    let rows: [[usize; TINY_N_ITEMS]; 12] = [
        [0, 1, 2, 0],
        [1, 2, 0, 1],
        [2, 0, 1, 2],
        [0, 2, 1, 0],
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [0, 0, 0, 0],
        [2, 1, 0, 2],
        [1, 0, 2, 1],
        [0, 2, 0, 1],
        [1, 2, 1, 0],
        [2, 0, 2, 1],
    ];
    let mut y = Vec::with_capacity(n_persons * TINY_N_ITEMS);
    for row in rows {
        y.extend_from_slice(&row);
    }
    (y, n_persons)
}

#[test]
fn reduced_marginal_loglik_matches_brute_force_product_grid() {
    let (a_g, a_s, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    let reduced = bifactor_grm_marginal_loglik(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect("reduced loglik on valid tiny input must succeed");
    let brute = bifactor_grm_marginal_loglik_brute(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect("brute-force loglik on valid tiny input must succeed");
    assert!(
        reduced.is_finite() && brute.is_finite(),
        "both logliks must be finite: reduced={reduced}, brute={brute}"
    );
    let gap = (reduced - brute).abs();
    assert!(
        gap <= 1e-10,
        "Gibbons-Hedeker reduction only reorders the finite marginal sum, so the gap \
         must be floating-point noise (<= 1e-10); got gap={gap:.3e} \
         (reduced={reduced:.10}, brute={brute:.10})"
    );
}

// ---------------------------------------------------------------------------
// Argument validation and failure reporting.
// ---------------------------------------------------------------------------

fn valid_config() -> BifactorGrmConfig {
    BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 5,
        tol: 1e-4,
        n_starts: 1,
        seed: 42,
        newton_iter: 3,
        ridge: 1e-8,
        device: crate::Device::Cpu,
    }
}

fn valid_data() -> (Vec<usize>, usize) {
    tiny_data()
}

#[test]
fn oracle_rejects_nonzero_specific_slope_on_general_only_items() {
    // A general-only item (specific_map == -1) owns no specific slope: a
    // non-zero value would be silently dropped, so the oracle fails loudly.
    let (a_g, _, thresholds) = tiny_params();
    let (y, n_persons) = tiny_data();
    // One specific factor on items 0, 1, 3; item 2 is general-only.
    let map: [i32; TINY_N_ITEMS] = [0, 0, -1, 0];
    let mut a_s = vec![0.8, 1.1, 0.0, 0.6];
    a_s[2] = 0.5; // general-only item with a stray specific slope
    bifactor_grm_marginal_loglik(
        &a_g,
        &a_s,
        &thresholds,
        &y,
        None,
        &map,
        n_persons,
        TINY_N_ITEMS,
        1,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-zero a_specific on a general-only item must fail loudly");
}

#[test]
fn uncapped_budgets_are_accepted() {
    // Stage-1 review fix-up: no magic upper caps on caller budgets. Lower
    // bounds (and only lower bounds) are validated; size overflow is a loud
    // `Err` from checked arithmetic instead.
    let (y, n_persons) = valid_data();
    let cfg = BifactorGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        n_starts: 40,
        ..valid_config()
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("n_starts=40 must be accepted (no upper cap)");
    assert!(
        fit.best_start < 40,
        "best_start must index the 40 runs; got {}",
        fit.best_start
    );

    // n_specific=17 validates: 34 items in 17 pairs, 12 persons cycling all
    // three categories so every boundary stays identified.
    let n_items_17 = 34usize;
    let n_persons_17 = 12usize;
    let map_17: Vec<i32> = (0..n_items_17).map(|i| (i / 2) as i32).collect();
    let mut y_17 = vec![0usize; n_persons_17 * n_items_17];
    for p in 0..n_persons_17 {
        for i in 0..n_items_17 {
            y_17[p * n_items_17 + i] = (p + i) % TINY_N_CAT;
        }
    }
    fit_bifactor_grm(
        &y_17,
        None,
        &map_17,
        n_persons_17,
        n_items_17,
        17,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect("n_specific=17 must be accepted (no upper cap)");
}

#[test]
fn arbitrary_quadrature_counts_above_the_old_fixed_table_are_accepted() {
    // #1929: node count controls integration precision and must not be capped
    // at a fixed table; 5/22/100 used to be rejected, now must fit cleanly.
    let (y, n_persons) = valid_data();
    for (qg, qs) in [(5, 7), (7, 5), (21, 22), (100, 7)] {
        let cfg = BifactorGrmConfig {
            q_general: qg,
            q_specific: qs,
            ..valid_config()
        };
        fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q_general={qg}, q_specific={qs} must be accepted: {e}"));
    }
}

#[test]
fn rejects_zero_quadrature_counts() {
    let (y, n_persons) = valid_data();
    for (qg, qs) in [(0, 7), (7, 0)] {
        let cfg = BifactorGrmConfig {
            q_general: qg,
            q_specific: qs,
            ..valid_config()
        };
        let err = fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("q_general/q_specific == 0 must fail loudly, never clamp");
        assert!(
            err.contains("q_general") || err.contains("q_specific") || err.contains('q'),
            "quadrature error must name the offending argument; got: {err}"
        );
    }
}

#[test]
fn rejects_bad_iteration_and_tolerance_budgets() {
    let (y, n_persons) = valid_data();
    for cfg in [
        BifactorGrmConfig {
            max_iter: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            tol: 0.0,
            ..valid_config()
        },
        BifactorGrmConfig {
            tol: f64::NAN,
            ..valid_config()
        },
        BifactorGrmConfig {
            n_starts: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            newton_iter: 0,
            ..valid_config()
        },
        BifactorGrmConfig {
            ridge: -1e-8,
            ..valid_config()
        },
    ] {
        fit_bifactor_grm(
            &y,
            None,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("out-of-range caller budgets must fail loudly, never clamp");
    }
}

#[test]
fn rejects_malformed_specific_map() {
    let (y, n_persons) = valid_data();
    // Wrong length.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map length must equal n_items");
    // Out-of-range block index.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1, 2],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map entries must be -1 or in 0..n_specific");
    // Negative other than the -1 general-only sentinel.
    fit_bifactor_grm(
        &y,
        None,
        &[0, 0, 1, -2],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("specific_map entries below -1 must be rejected");
    // A specific factor with fewer than two items is rejected as an
    // implementation stability choice (weakly identified general/specific
    // split; no minimum-block-size theorem in the cited sources).
    fit_bifactor_grm(
        &y,
        None,
        &[0, 1, 1, 1],
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("a specific factor with a single item must be rejected");
}

#[test]
fn rejects_out_of_range_categories() {
    let (mut y, n_persons) = valid_data();
    y[0] = TINY_N_CAT; // one past the top category
    fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("observed categories must be < n_cat");
}

#[test]
fn unobserved_category_fails_loudly_with_item_and_category() {
    // Item 0 never sees category 2: its top boundary intercept is unidentified
    // (Samejima, 1969), so the fitter must refuse instead of imputing.
    let n_persons = 30usize;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for p in 0..n_persons {
        y[p * TINY_N_ITEMS] = p % 2; // item 0: only categories 0-1
        for i in 1..TINY_N_ITEMS {
            y[p * TINY_N_ITEMS + i] = p % TINY_N_CAT;
        }
    }
    let err = fit_bifactor_grm(
        &y,
        None,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("an unobserved category must fail loudly, never be imputed");
    assert!(
        err.contains("item 0") && err.contains("category 2"),
        "the error must name the item and the unobserved category; got: {err}"
    );
}

#[test]
fn non_convergence_is_reported_not_substituted() {
    // One EM sweep cannot converge: the fitter must return Ok with
    // converged == false and termination_reason == "max_iter_reached" so the
    // caller sees the failure instead of receiving a substitute.
    let (y, _) = tiny_data();
    // Larger toy sample so a single sweep is nowhere near convergence.
    let n_big = 200usize;
    let mut y_big = Vec::with_capacity(n_big * TINY_N_ITEMS);
    for p in 0..n_big {
        for i in 0..TINY_N_ITEMS {
            y_big.push(y[(p % 12) * TINY_N_ITEMS + i]);
        }
    }
    let cfg = BifactorGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..valid_config()
    };
    let fit = fit_bifactor_grm(
        &y_big,
        None,
        &TINY_SPECIFIC_MAP,
        n_big,
        TINY_N_ITEMS,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("max_iter exhaustion must be reported via flags, not via Err");
    assert!(
        !fit.converged,
        "a single EM sweep must not report convergence"
    );
    assert_eq!(
        fit.termination_reason, "max_iter_reached",
        "termination reason must say max_iter_reached"
    );
    assert!(
        fit.loglik_trace.len() == 2,
        "trace must hold the initial and one updated loglik; got {}",
        fit.loglik_trace.len()
    );
}

// ---------------------------------------------------------------------------
// Stage 5: GPU/CPU E-step parity on expected counts and loglik.
// ---------------------------------------------------------------------------

#[test]
fn estep_gpu_matches_cpu_counts_and_loglik() {
    // The WGSL person-parallel sweep accumulates in f32 while the CPU
    // reference is f64: per-count-entry error scales with n_persons × eps
    // (≈ 12 × 1.2e-7 here), so the envelope below is orders of magnitude
    // above the expected f32 noise. The tiny fixture carries a negative
    // general slope (reverse-keyed item 1), so sign handling is covered.
    // Without a GPU adapter the GPU entry falls back to CPU and the
    // comparison is trivially exact; the fit-level Python test pins the
    // real-device numbers.
    use super::{
        e_step, fill_logprob_tables, gh_rule, initial_params, validate,
    };

    let (y, n_persons) = tiny_data();
    let cfg = valid_config();
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
    .expect("tiny fixture must validate");
    let (tg, wg) = gh_rule(7).expect("Q=7 rule must exist");
    let (ts, ws) = gh_rule(7).expect("Q=7 rule must exist");
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, tg, ts, 7, 7);
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();

    let (ll_cpu, counts_cpu) = e_step(
        &v, &y, None, &tables, &log_wg, &log_ws, 7, 7, tg, ts,
        crate::Device::Cpu,
    );
    let (ll_gpu, counts_gpu) = e_step(
        &v, &y, None, &tables, &log_wg, &log_ws, 7, 7, tg, ts,
        crate::Device::Gpu,
    );

    assert!(
        (ll_cpu - ll_gpu).abs() <= 1e-3,
        "E-step loglik must agree within f32 envelope: cpu={ll_cpu}, gpu={ll_gpu}"
    );
    assert_eq!(counts_cpu.len(), counts_gpu.len());
    let mut max_diff = 0.0f64;
    for (cc, gc) in counts_cpu.iter().zip(counts_gpu.iter()) {
        assert_eq!(cc.len(), gc.len());
        for (cn, gn) in cc.iter().zip(gc.iter()) {
            assert_eq!(cn.len(), gn.len());
            for (&a, &b) in cn.iter().zip(gn.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
        }
    }
    assert!(
        max_diff <= 1e-4,
        "expected counts must agree within f32 envelope; got max|diff|={max_diff:.3e}"
    );
}

// ---------------------------------------------------------------------------
// #1976: zero Gauss-Hermite prior mass must not NaN-poison E-step counts;
// a fit that never leaves its start must not report tolerance_met.
// ---------------------------------------------------------------------------

#[test]
fn zero_prior_weight_nodes_do_not_nan_estep_counts() {
    use super::{e_step, fill_logprob_tables, gh_rule, initial_params, validate};

    let (y, n_persons) = tiny_data();
    let cfg = valid_config();
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
    .expect("tiny fixture must validate");
    let (tg, wg) = gh_rule(7).expect("Q=7 rule must exist");
    let (ts, ws) = gh_rule(7).expect("Q=7 rule must exist");
    let params = initial_params(&v, &y, None, cfg.seed, 0);
    let tables = fill_logprob_tables(&v, &params, tg, ts, 7, 7);
    let mut log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    // Inject exactly-zero prior mass at both tails (the large-q Golub–Welsch
    // failure mode from #1976).
    log_wg[0] = f64::NEG_INFINITY;
    log_wg[6] = f64::NEG_INFINITY;

    let (ll, counts) = e_step(
        &v,
        &y,
        None,
        &tables,
        &log_wg,
        &log_ws,
        7,
        7,
        tg,
        ts,
        crate::Device::Cpu,
    );
    assert!(ll.is_finite(), "observed-data loglik must stay finite; got {ll}");
    for (i, item_counts) in counts.iter().enumerate() {
        for (node, cat) in item_counts.iter().enumerate() {
            for (k, &c) in cat.iter().enumerate() {
                assert!(
                    c.is_finite(),
                    "count[{i}][{node}][{k}] must be finite; got {c}"
                );
            }
        }
    }
}

#[test]
fn refuse_tolerance_reclassifies_bit_identical_start() {
    use super::{refuse_tolerance_on_frozen_start, ItemParams};

    let start = vec![ItemParams {
        a_g: 1.0,
        a_s: Some(0.8),
        d: vec![1.0, -1.0],
    }];
    let frozen = start.clone();
    let mut converged = true;
    let mut reason = "tolerance_met".to_string();
    refuse_tolerance_on_frozen_start(&mut converged, &mut reason, &frozen, &start);
    assert!(!converged);
    assert_eq!(reason, "numerical_em_stall");

    let mut moved = start.clone();
    moved[0].a_g = 1.5;
    converged = true;
    reason = "tolerance_met".to_string();
    refuse_tolerance_on_frozen_start(&mut converged, &mut reason, &moved, &start);
    assert!(converged);
    assert_eq!(reason, "tolerance_met");
}

#[test]
fn dense_quadrature_fit_never_claims_tolerance_at_start_slopes() {
    // Compact #1976-shaped design (13 items / 3 specifics / 4 cats). q=421 is
    // where Golub–Welsch produces exactly-zero prior weights; the pre-fix
    // path reported tolerance_met with a_general still at 1.0.
    let n_persons = 60usize;
    let n_items = 13usize;
    let n_specific = 3usize;
    let n_cat = 4usize;
    let specific_map: [i32; 13] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, -1];
    let mut y = vec![0usize; n_persons * n_items];
    for p in 0..n_persons {
        for i in 0..n_items {
            y[p * n_items + i] = (p + 3 * i) % n_cat;
        }
    }
    let cfg = BifactorGrmConfig {
        q_general: 421,
        q_specific: 421,
        max_iter: 12,
        tol: 1e-6,
        n_starts: 1,
        seed: 20260917,
        newton_iter: 5,
        ridge: 1e-4,
        device: crate::Device::Cpu,
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &cfg,
    )
    .expect("dense-q fit must return a result");
    if fit.termination_reason == "tolerance_met" {
        assert!(fit.converged);
        assert!(
            fit.a_general.iter().any(|&a| a != 1.0),
            "tolerance_met must not leave a_general at the 1.0 start; got {:?}",
            &fit.a_general[..3]
        );
    } else if fit.a_general.iter().all(|&a| a == 1.0) {
        assert!(
            !fit.converged,
            "parameters frozen at start must not report converged"
        );
        assert_eq!(fit.termination_reason, "numerical_em_stall");
    }
}

// ---------------------------------------------------------------------------
// Block partial-pattern collapse (#2003).
// ---------------------------------------------------------------------------

#[test]
fn block_partial_patterns_collapse_duplicates_and_separate_missingness() {
    // Two specific blocks of two items; duplicate rows + one MAR mask variant.
    let n_persons = 8usize;
    let n_items = 4usize;
    let n_cat = 3usize;
    let blocks = [vec![0usize, 1], vec![2usize, 3]];
    let general_only: Vec<usize> = Vec::new();
    // Persons 0..3 share [0,1 | 2,0]; 4..5 share [1,1 | 0,0]; 6 unique;
    // 7 matches 0 on observed categories but misses item 1 → distinct pattern.
    let rows: [[usize; 4]; 8] = [
        [0, 1, 2, 0],
        [0, 1, 2, 0],
        [0, 1, 2, 0],
        [0, 1, 2, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [2, 2, 1, 1],
        [0, 1, 2, 0],
    ];
    let mut y = Vec::with_capacity(n_persons * n_items);
    for row in rows {
        y.extend_from_slice(&row);
    }
    let mut observed = vec![true; n_persons * n_items];
    observed[7 * n_items + 1] = false;

    let prov = crate::bifactor_block_patterns::block_pattern_collapse_provenance(
        &y,
        Some(&observed),
        n_persons,
        n_items,
        n_cat,
        &blocks,
        &general_only,
    )
    .expect("provenance");
    assert_eq!(prov.n_persons, n_persons);
    assert_eq!(prov.n_persons_per_block, vec![n_persons, n_persons]);
    // Block 0: [0,1] x4, [1,1] x2, [2,2] x1, [0, MISSING] x1 → 4 unique.
    assert_eq!(prov.n_unique_patterns_per_block[0], 4);
    // Block 1: [2,0] x5 (persons 0-3 and 7), [0,0] x2, [1,1] x1 → 3 unique.
    assert_eq!(prov.n_unique_patterns_per_block[1], 3);
    assert!(
        prov.n_unique_patterns_per_block[0] < prov.n_persons_per_block[0],
        "duplicates must reduce unique count"
    );
}

#[test]
fn collapsed_e_step_matches_personwise_aggregates() {
    // Tiny bifactor; duplicate patterns so collapse is active.
    let n_persons = 6usize;
    let n_items = 4usize;
    let n_specific = 2usize;
    let n_cat = 3usize;
    let specific_map = [0i32, 0, 1, 1];
    let rows: [[usize; 4]; 6] = [
        [0, 1, 2, 0],
        [0, 1, 2, 0],
        [0, 1, 2, 0],
        [1, 2, 0, 1],
        [1, 2, 0, 1],
        [2, 0, 1, 2],
    ];
    let mut y = Vec::with_capacity(n_persons * n_items);
    for row in rows {
        y.extend_from_slice(&row);
    }
    let cfg = BifactorGrmConfig {
        q_general: 7,
        q_specific: 7,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
        device: crate::Device::Cpu,
    };
    let v = super::validate(
        &y,
        None,
        &specific_map,
        n_persons,
        n_items,
        n_specific,
        n_cat,
        &cfg,
    )
    .expect("validate");
    let (tg, wg) = super::gh_rule(7).expect("gh");
    let (ts, ws) = super::gh_rule(7).expect("gh");
    let qg = tg.len();
    let qs = ts.len();
    let log_wg: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let a_g = vec![1.0, 0.8, 1.1, 0.9];
    let a_s = vec![0.7, 0.6, 0.5, 0.8];
    let thr = vec![0.5, -0.5, 0.4, -0.6, 0.3, -0.7, 0.2, -0.8];
    let params = super::pack_params(&v, &a_g, &a_s, &thr);
    let tables = super::fill_logprob_tables(&v, &params, tg, ts, qg, qs);

    let (ll_c, counts_c) = super::e_step(
        &v,
        &y,
        None,
        &tables,
        &log_wg,
        &log_ws,
        qg,
        qs,
        tg,
        ts,
        crate::Device::Cpu,
    );
    let (ll_p, counts_p) = super::e_step_personwise_reference(
        &v, &y, None, &tables, &log_wg, &log_ws, qg, qs,
    );
    // Measured on this fixture: relative loglik gap and max |count| gap stay
    // under these bounds (associative reordering of the same finite sums).
    let rel = (ll_c - ll_p).abs() / (1.0 + ll_p.abs());
    assert!(
        rel <= 1e-12,
        "loglik rel gap {rel} (collapsed={ll_c}, personwise={ll_p})"
    );
    let mut max_abs = 0.0f64;
    for i in 0..n_items {
        for node in 0..counts_c[i].len() {
            for k in 0..n_cat {
                max_abs = max_abs.max((counts_c[i][node][k] - counts_p[i][node][k]).abs());
            }
        }
    }
    assert!(
        max_abs <= 1e-10,
        "max |count| gap {max_abs} exceeds measured fixture bound 1e-10"
    );
}

#[test]
fn all_unique_patterns_provenance_reports_no_reduction() {
    let n_persons = 4usize;
    let n_items = 4usize;
    let n_cat = 3usize;
    let blocks = [vec![0usize, 1], vec![2usize, 3]];
    let general_only: Vec<usize> = Vec::new();
    let rows: [[usize; 4]; 4] = [
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [1, 1, 1, 1],
    ];
    let mut y = Vec::with_capacity(n_persons * n_items);
    for row in rows {
        y.extend_from_slice(&row);
    }
    let prov = crate::bifactor_block_patterns::block_pattern_collapse_provenance(
        &y,
        None,
        n_persons,
        n_items,
        n_cat,
        &blocks,
        &general_only,
    )
    .expect("provenance");
    assert_eq!(prov.n_unique_patterns_per_block, vec![n_persons, n_persons]);
}

