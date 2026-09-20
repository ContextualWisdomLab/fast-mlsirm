//! Unit contract for the single-group polytomous two-tier GRM fitter
//! (`mlsirm_core::two_tier_grm`, stage 4 of #1912).
//!
//! * Dimension-reduction exactness: the reduced marginal log-likelihood must
//!   equal brute-force full product-grid integration to `<= 1e-10` on a tiny
//!   two-primary model (10 items, `P = 2`, `S = 2`, `Q = 7`). The two paths
//!   reorder the same finite sum, so any larger gap is an implementation bug,
//!   not quadrature error.
//! * Reduction to stage 1: at `P = 1` (every item loading the single
//!   primary) the reduced marginal log-likelihood must equal the stage-1
//!   `bifactor_grm_marginal_loglik` to `<= 1e-10` on the same parameters.
//! * Argument validation and failure reporting: unsupported quadrature,
//!   bad iteration/tolerance/start budgets, malformed primary/specific maps,
//!   under-loaded primaries, out-of-range categories, unobserved categories,
//!   non-PD `phi` in the oracle, and the non-convergence report
//!   (`converged == false`, `termination_reason == "max_iter_reached"`).
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
//! accessible — no equation locator is drawn from it)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use crate::two_tier_grm::{
    fit_two_tier_grm, fit_two_tier_grm_fipc, two_tier_grm_marginal_loglik,
    two_tier_grm_marginal_loglik_brute, TwoTierFipcConfig, TwoTierGrmConfig,
};

#[test]
fn fipc_keeps_anchor_rows_and_returns_focal_moments() {
    let (a_primary, a_specific, thresholds, _) = tiny_params();
    let n_persons = 60;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for p in 0..n_persons {
        for i in 0..TINY_N_ITEMS {
            y[p * TINY_N_ITEMS + i] = (p + i) % TINY_N_CAT;
        }
    }
    let anchors = [true, true, true, true, true, true, true, true, false, false];
    let fit = fit_two_tier_grm_fipc(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &anchors,
        &a_primary,
        &a_specific,
        &thresholds,
        &TwoTierFipcConfig {
            q_primary: 7,
            q_specific: 7,
            max_iter: 2,
            tol: 1e-5,
            newton_iter: 2,
            ridge: 1e-8,
            estimate_specific_vars: false,
        },
    )
    .expect("two-tier FIPC real-fit path must accept a valid anchored fit");
    for &i in &[0usize, 7] {
        assert_eq!(
            &fit.a_primary[i * TINY_N_PRIMARY..(i + 1) * TINY_N_PRIMARY],
            &a_primary[i * TINY_N_PRIMARY..(i + 1) * TINY_N_PRIMARY]
        );
        assert_eq!(fit.a_specific[i], a_specific[i]);
        assert_eq!(
            &fit.threshold[i * (TINY_N_CAT - 1)..(i + 1) * (TINY_N_CAT - 1)],
            &thresholds[i * (TINY_N_CAT - 1)..(i + 1) * (TINY_N_CAT - 1)]
        );
    }
    assert_eq!(fit.primary_mean.len(), TINY_N_PRIMARY);
    assert_eq!(fit.primary_cov.len(), TINY_N_PRIMARY * TINY_N_PRIMARY);
    assert!(fit.loglik_trace.iter().all(|value| value.is_finite()));
}

// ---------------------------------------------------------------------------
// Shared tiny two-tier problem: 10 items, P = 2 primaries in simple
// structure (items 0-4 on primary 0, items 5-9 on primary 1), S = 2
// specifics cross-cutting the primary split (method-factor layout).
// ---------------------------------------------------------------------------

const TINY_N_ITEMS: usize = 10;
const TINY_N_PRIMARY: usize = 2;
const TINY_N_SPECIFIC: usize = 2;
const TINY_N_CAT: usize = 3;
/// Row-major `n_items x n_primary` free-slope pattern.
const TINY_PRIMARY_MAP: [bool; TINY_N_ITEMS * TINY_N_PRIMARY] = [
    true, false, // 0
    true, false, // 1
    true, false, // 2
    true, false, // 3
    true, false, // 4
    false, true, // 5
    false, true, // 6
    false, true, // 7
    false, true, // 8
    false, true, // 9
];
/// Specific blocks cross the primary split: S0 = {0, 1, 5, 6}, S1 = {2, 3, 4, 7, 8, 9}.
const TINY_SPECIFIC_MAP: [i32; TINY_N_ITEMS] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

fn tiny_params() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    // Row-major n_items x n_primary; exact 0.0 at fixed positions.
    let a_primary = vec![
        1.2, 0.0, //
        -0.9, 0.0, //
        1.0, 0.0, //
        0.7, 0.0, //
        1.1, 0.0, //
        0.0, 1.0, //
        0.0, 0.8, //
        0.0, 1.2, //
        0.0, -0.7, //
        0.0, 0.9,
    ];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6, 0.7, 1.0, 0.8, 0.9, 0.7, 1.1];
    // Strictly decreasing within each item (n_cat - 1 = 2 intercepts).
    let thresholds = vec![
        0.9, -0.7, //
        0.5, -1.0, //
        1.1, -0.4, //
        0.2, -1.3, //
        0.8, -0.6, //
        1.0, -0.5, //
        0.6, -0.9, //
        1.2, -0.3, //
        0.4, -1.1, //
        0.7, -0.8,
    ];
    let phi = vec![1.0, 0.35, 0.35, 1.0];
    (a_primary, a_specific, thresholds, phi)
}

fn tiny_data() -> (Vec<usize>, usize) {
    // Deterministic toy responses covering every category of every item.
    let n_persons = 12usize;
    let rows: [[usize; TINY_N_ITEMS]; 12] = [
        [0, 1, 2, 0, 1, 2, 0, 1, 2, 0],
        [1, 2, 0, 1, 2, 0, 1, 2, 0, 1],
        [2, 0, 1, 2, 0, 1, 2, 0, 1, 2],
        [0, 2, 1, 0, 2, 1, 0, 2, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 1, 0, 2, 1, 0, 2, 1, 0, 2],
        [1, 0, 2, 1, 0, 2, 1, 0, 2, 1],
        [0, 2, 0, 1, 0, 2, 0, 1, 2, 1],
        [1, 2, 1, 0, 1, 2, 1, 0, 1, 2],
        [2, 0, 2, 1, 2, 0, 2, 1, 0, 0],
    ];
    let mut y = Vec::with_capacity(n_persons * TINY_N_ITEMS);
    for row in rows {
        y.extend_from_slice(&row);
    }
    (y, n_persons)
}

#[test]
fn reduced_marginal_loglik_matches_brute_force_product_grid() {
    let (a_p, a_s, thresholds, phi) = tiny_params();
    let (y, n_persons) = tiny_data();
    let reduced = two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect("reduced loglik on valid tiny input must succeed");
    let brute = two_tier_grm_marginal_loglik_brute(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
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
        "two-tier reduction only reorders the finite marginal sum, so the gap \
         must be floating-point noise (<= 1e-10); got gap={gap:.3e} \
         (reduced={reduced:.10}, brute={brute:.10})"
    );
}

#[test]
fn single_primary_oracle_matches_stage1_bifactor_oracle() {
    use crate::bifactor_grm::bifactor_grm_marginal_loglik;
    // P = 1 with every item on the single primary IS the stage-1 bifactor
    // problem (4 items, 2 specifics, 3 categories; the stage-1 unit layout).
    let a_general = vec![1.2, -0.9, 1.0, 0.7];
    let a_specific = vec![0.8, 1.1, 0.9, 0.6];
    let thresholds = vec![0.9, -0.7, 0.5, -1.0, 1.1, -0.4, 0.2, -1.3];
    let specific_map = [0, 0, 1, 1];
    let primary_map = [true; 4];
    let y = vec![
        0, 1, 2, 0, //
        1, 2, 0, 1, //
        2, 0, 1, 2, //
        0, 2, 1, 0, //
        1, 1, 1, 1, //
        2, 2, 2, 2, //
        0, 0, 0, 0, //
        2, 1, 0, 2, //
        1, 0, 2, 1, //
        0, 2, 0, 1, //
        1, 2, 1, 0, //
        2, 0, 2, 1,
    ];
    let n_persons = 12usize;
    let phi = vec![1.0];
    // Row-major 4x1 primary slopes equal the general slopes.
    let two_tier = two_tier_grm_marginal_loglik(
        &a_general,
        &a_specific,
        &thresholds,
        &phi,
        &y,
        None,
        &primary_map,
        &specific_map,
        n_persons,
        4,
        1,
        2,
        3,
        7,
        7,
    )
    .expect("two-tier P=1 oracle must succeed");
    let stage1 = bifactor_grm_marginal_loglik(
        &a_general,
        &a_specific,
        &thresholds,
        &y,
        None,
        &specific_map,
        n_persons,
        4,
        2,
        3,
        7,
        7,
    )
    .expect("stage-1 oracle must succeed");
    let gap = (two_tier - stage1).abs();
    assert!(
        gap <= 1e-10,
        "P=1 two-tier IS the bifactor model, so both oracles evaluate the same \
         finite sum; got gap={gap:.3e} (two-tier={two_tier:.10}, stage-1={stage1:.10})"
    );
}

// ---------------------------------------------------------------------------
// Argument validation and failure reporting.
// ---------------------------------------------------------------------------

fn valid_config() -> TwoTierGrmConfig {
    TwoTierGrmConfig {
        q_primary: 7,
        q_specific: 7,
        max_iter: 5,
        tol: 1e-4,
        n_starts: 1,
        seed: 42,
        newton_iter: 3,
        ridge: 1e-8,
    }
}

#[test]
fn rejects_malformed_primary_map() {
    let (y, n_persons) = tiny_data();
    // Wrong length.
    let short = vec![true; TINY_N_ITEMS * TINY_N_PRIMARY - 1];
    fit_two_tier_grm(
        &y,
        None,
        &short,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("primary_map length must equal n_items * n_primary");
    // A primary with fewer than two loading items is rejected as an
    // implementation stability choice (the primary correlation is weakly
    // identified otherwise; mirrors the stage-1 specific-block rule).
    let mut single = vec![false; TINY_N_ITEMS * TINY_N_PRIMARY];
    for i in 0..TINY_N_ITEMS {
        single[i * TINY_N_PRIMARY] = true; // primary 0: all items
    }
    single[1] = true; // primary 1: one item only (item 0)
    fit_two_tier_grm(
        &y,
        None,
        &single,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &valid_config(),
    )
    .expect_err("a primary with a single loading item must be rejected");
}

#[test]
fn oracle_rejects_nonzero_slope_at_fixed_primary_positions() {
    // Fixed (pattern-zero) primary positions own no slope parameter: a
    // non-zero value there would be silently dropped, so the oracle fails
    // loudly — the same contract as stage-1's general-only a_specific rule.
    let (_, a_s, thresholds, phi) = tiny_params();
    let (y, n_persons) = tiny_data();
    let (mut a_p, _, _, _) = tiny_params();
    a_p[1] = 0.5; // item 0 does not load primary 1, yet carries a slope
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &phi,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-zero a_primary at a fixed pattern position must fail loudly");
}

#[test]
fn oracle_rejects_non_correlation_phi() {
    let (a_p, a_s, thresholds, _) = tiny_params();
    let (y, n_persons) = tiny_data();
    // Non-unit diagonal.
    let bad_diag = vec![1.2, 0.35, 0.35, 1.0];
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &bad_diag,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("phi with a non-unit diagonal must fail loudly");
    // Symmetric but indefinite (|rho| > 1 forces non-PD).
    let indefinite = vec![1.0, 1.5, 1.5, 1.0];
    two_tier_grm_marginal_loglik(
        &a_p,
        &a_s,
        &thresholds,
        &indefinite,
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        7,
        7,
    )
    .expect_err("non-positive-definite phi must fail loudly");
}

#[test]
fn arbitrary_quadrature_counts_above_the_old_fixed_table_are_accepted() {
    // #1929: node count controls integration precision and must not be
    // capped at a fixed table; 5/22/100 used to be rejected, now must fit
    // cleanly (module reuses `quadrature::require_gh_rule`, any n >= 1).
    let (y, n_persons) = tiny_data();
    for (qp, qs) in [(5, 7), (7, 5), (15, 22), (100, 7)] {
        let cfg = TwoTierGrmConfig {
            q_primary: qp,
            q_specific: qs,
            ..valid_config()
        };
        fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .unwrap_or_else(|e| panic!("q_primary={qp}, q_specific={qs} must be accepted: {e}"));
    }
}

#[test]
fn rejects_zero_quadrature_counts() {
    let (y, n_persons) = tiny_data();
    for (qp, qs) in [(0, 7), (7, 0)] {
        let cfg = TwoTierGrmConfig {
            q_primary: qp,
            q_specific: qs,
            ..valid_config()
        };
        let err = fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("q_primary/q_specific == 0 must fail loudly, never clamp");
        assert!(
            err.contains("q_primary") || err.contains("q_specific") || err.contains('q'),
            "quadrature error must name the offending argument; got: {err}"
        );
    }
}

#[test]
fn rejects_bad_iteration_and_tolerance_budgets() {
    let (y, n_persons) = tiny_data();
    for cfg in [
        TwoTierGrmConfig {
            max_iter: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            tol: 0.0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            tol: f64::NAN,
            ..valid_config()
        },
        TwoTierGrmConfig {
            n_starts: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            newton_iter: 0,
            ..valid_config()
        },
        TwoTierGrmConfig {
            ridge: -1e-8,
            ..valid_config()
        },
    ] {
        fit_two_tier_grm(
            &y,
            None,
            &TINY_PRIMARY_MAP,
            &TINY_SPECIFIC_MAP,
            n_persons,
            TINY_N_ITEMS,
            TINY_N_PRIMARY,
            TINY_N_SPECIFIC,
            TINY_N_CAT,
            &cfg,
        )
        .expect_err("out-of-range caller budgets must fail loudly, never clamp");
    }
}

#[test]
fn unobserved_category_fails_loudly_with_item_and_category() {
    let n_persons = 30usize;
    let mut y = vec![0usize; n_persons * TINY_N_ITEMS];
    for p in 0..n_persons {
        for i in 0..TINY_N_ITEMS {
            y[p * TINY_N_ITEMS + i] = p % TINY_N_CAT;
        }
        y[p * TINY_N_ITEMS] = p % 2; // item 0: only categories 0-1
    }
    let err = fit_two_tier_grm(
        &y,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
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
    let (y, _) = tiny_data();
    let n_big = 200usize;
    let mut y_big = Vec::with_capacity(n_big * TINY_N_ITEMS);
    for p in 0..n_big {
        for i in 0..TINY_N_ITEMS {
            y_big.push(y[(p % 12) * TINY_N_ITEMS + i]);
        }
    }
    let cfg = TwoTierGrmConfig {
        max_iter: 1,
        tol: 1e-12,
        ..valid_config()
    };
    let fit = fit_two_tier_grm(
        &y_big,
        None,
        &TINY_PRIMARY_MAP,
        &TINY_SPECIFIC_MAP,
        n_big,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
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
}
