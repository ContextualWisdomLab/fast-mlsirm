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
//! https://doi.org/10.1007/s11336-010-9178-0 (the posterior-moment test below
//! uses directly inspected pp. 608-609, Appendices A/B)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use crate::two_tier_grm::{
    fit_two_tier_grm, two_tier_grm_marginal_loglik, two_tier_grm_marginal_loglik_brute,
    TwoTierGrmConfig,
};

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
        estimate_primary_correlation: true,
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
fn identity_rejects_identical_primary_support_but_accepts_nested_support() {
    let (y, n_persons) = tiny_data();
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        max_iter: 500,
        ..valid_config()
    };
    let shared = [true; TINY_N_ITEMS * TINY_N_PRIMARY];
    let err = fit_two_tier_grm(
        &y,
        None,
        &shared,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect_err("identical supports admit a continuous rotation");
    assert!(err.contains("identical free-loading item sets"), "{err}");

    let nested: Vec<bool> = (0..TINY_N_ITEMS).flat_map(|i| [true, i < 5]).collect();
    let fit = fit_two_tier_grm(
        &y,
        None,
        &nested,
        &TINY_SPECIFIC_MAP,
        n_persons,
        TINY_N_ITEMS,
        TINY_N_PRIMARY,
        TINY_N_SPECIFIC,
        TINY_N_CAT,
        &cfg,
    )
    .expect("distinct nested supports must fit");
    assert!(fit.converged, "nested-support fit did not converge");
    assert_ne!(fit.termination_reason, "max_iter_reached");
    assert_eq!(fit.phi, vec![1.0, 0.0, 0.0, 1.0]);
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

// Same-node oracle for Cai (2010), pp. 608-609, Appendices A/B. Enumerating
// every specific-node combination independently checks the reduced posterior
// without claiming continuous-integral accuracy or empirical fit recovery.
#[test]
fn latent_moments_match_full_grid_with_missing_blocks() {
    use crate::two_tier_grm::{
        e_step, e_step_with_moments, ItemParams, LatentPosteriorMoments, Validated,
    };
    let v = Validated {
        n_persons: 3,
        n_items: 5,
        n_primary: 2,
        n_specific: 2,
        n_cat: 3,
        m1: 2,
        grid_size: 4,
        free_primaries: vec![vec![0, 1]; 5],
        blocks: vec![vec![0, 1], vec![2, 3]],
        specific_free: vec![4],
        item_block: vec![Some(0), Some(0), Some(1), Some(1), None],
    };
    let pars: Vec<ItemParams> = (0..5)
        .map(|i| ItemParams {
            a_p: vec![0.7 + i as f64 * 0.1, -0.4 + i as f64 * 0.2],
            a_s: if i < 4 {
                Some(0.5 - i as f64 * 0.3)
            } else {
                None
            },
            d: vec![0.8, -0.6],
        })
        .collect();
    // Asymmetric, non-centered nodes/weights exercise first moments.
    let coords = [-0.5, 0.2, -0.5, 1.1, 0.9, 0.2, 0.9, 1.1];
    let wg: [f64; 4] = [0.1, 0.2, 0.3, 0.4];
    let ts: [f64; 2] = [-0.7, 1.2];
    let ws: [f64; 2] = [0.6, 0.4];
    let log_w: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    let log_ws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let y = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0];
    let obs = [
        true, true, true, true, true, false, false, true, true, true, false, false, false, false,
        false,
    ];
    let mut moments = LatentPosteriorMoments {
        mean: vec![123.0; 12],
        second: vec![456.0; 12],
    };
    let result = e_step_with_moments(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
        Some(&mut moments),
        true,
    );
    let old = e_step(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
    );
    assert_eq!(
        result, old,
        "opting into moments must preserve existing outputs"
    );
    let expected_mean = moments.mean.clone();
    let expected_second = moments.second.clone();
    let no_counts = e_step_with_moments(
        &v,
        &y,
        Some(&obs),
        &pars,
        &log_w,
        &log_ws,
        &coords,
        &ts,
        4,
        2,
        Some(&mut moments),
        false,
    );
    assert!(no_counts.1.is_empty());
    assert_eq!(no_counts.0, result.0);
    assert_eq!(no_counts.2, result.2);
    assert_eq!(moments.mean, expected_mean);
    assert_eq!(moments.second, expected_second);

    let mut brute_ll = 0.0;
    for person in 0..3 {
        let mut mass = 0.0;
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        for g in 0..4 {
            for h0 in 0..2 {
                for h1 in 0..2 {
                    let latent = [coords[g * 2], coords[g * 2 + 1], ts[h0], ts[h1]];
                    let mut weight = wg[g] * ws[h0] * ws[h1];
                    for i in 0..5 {
                        if !obs[person * 5 + i] {
                            continue;
                        }
                        let mut base = pars[i].a_p[0] * latent[0] + pars[i].a_p[1] * latent[1];
                        if let Some(s) = v.item_block[i] {
                            base += pars[i].a_s.unwrap() * latent[2 + s];
                        }
                        // Full-grid oracle uses probabilities, not reduced log-posterior algebra.
                        weight *=
                            crate::poly::grm_logprobs(base, &pars[i].d)[y[person * 5 + i]].exp();
                    }
                    mass += weight;
                    for d in 0..4 {
                        first[d] += weight * latent[d];
                        second[d] += weight * latent[d] * latent[d];
                    }
                }
            }
        }
        brute_ll += mass.ln();
        for d in 0..4 {
            assert!(
                (moments.mean[person * 4 + d] - first[d] / mass).abs() < 1e-12,
                "first moment person {person}, dimension {d}"
            );
            assert!(
                (moments.second[person * 4 + d] - second[d] / mass).abs() < 1e-12,
                "second moment person {person}, dimension {d}"
            );
        }
    }
    assert!((result.0 - brute_ll).abs() < 1e-12);
    // Entirely missing persons retain the finite quadrature prior.
    for s in 0..2 {
        assert!((moments.mean[8 + 2 + s] - (ws[0] * ts[0] + ws[1] * ts[1])).abs() < 1e-12);
    }
}

#[test]
fn focal_scores_match_physical_full_grid_and_missing_prior() {
    use crate::two_tier_grm::{gh_rule, score_two_tier_grm_orthogonal};
    let a_p = [0.7, -0.4, 0.8, -0.2, 0.9, 0.1, 1.0, 0.2, 1.1, 0.4];
    let a_s = [0.5, 0.2, -0.1, -0.4, 0.0];
    let threshold = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let primary = [true; 10];
    let specific = [0, 0, 1, 1, -1];
    let mean = [0.4, -0.2, 0.7, -0.6];
    let sd = [1.1, 0.8, 0.9, 1.3];
    let y = [0, 1, 2, 0, 1, 0, 0, 0, 0, 0];
    let observed = [
        true, true, true, true, true, false, false, false, false, false,
    ];
    let bank_before = (a_p, a_s, threshold);
    let score = score_two_tier_grm_orthogonal(
        &a_p,
        &a_s,
        &threshold,
        &mean,
        &sd,
        &y,
        Some(&observed),
        &primary,
        &specific,
        2,
        5,
        2,
        2,
        3,
        7,
        7,
    )
    .unwrap();
    assert_eq!((a_p, a_s, threshold), bank_before);
    let (z, w) = gh_rule(7).unwrap();
    let mut full_loglik = 0.0;
    for person in 0..2 {
        let mut mass = 0.0;
        let mut first = [0.0; 4];
        let mut second = [0.0; 4];
        // Physical latent grid keeps original slopes/intercepts untouched.
        for g0 in 0..7 {
            for g1 in 0..7 {
                for h0 in 0..7 {
                    for h1 in 0..7 {
                        let index = [g0, g1, h0, h1];
                        let mut latent = [0.0; 4];
                        let mut weight = 1.0;
                        for d in 0..4 {
                            latent[d] = mean[d] + sd[d] * z[index[d]];
                            weight *= w[index[d]];
                        }
                        for i in 0..5 {
                            if !observed[person * 5 + i] {
                                continue;
                            }
                            let mut base = a_p[i * 2] * latent[0] + a_p[i * 2 + 1] * latent[1];
                            if specific[i] >= 0 {
                                base += a_s[i] * latent[2 + specific[i] as usize];
                            }
                            weight *= crate::poly::grm_logprobs(base, &threshold[i * 2..i * 2 + 2])
                                [y[person * 5 + i]]
                                .exp();
                        }
                        mass += weight;
                        for d in 0..4 {
                            first[d] += weight * latent[d];
                            second[d] += weight * latent[d] * latent[d];
                        }
                    }
                }
            }
        }
        full_loglik += mass.ln();
        for d in 0..4 {
            assert!((score.mean[person * 4 + d] - first[d] / mass).abs() < 1e-12);
            assert!((score.second[person * 4 + d] - second[d] / mass).abs() < 1e-12);
            let var = second[d] / mass - (first[d] / mass).powi(2);
            assert!((score.sd[person * 4 + d] - var.sqrt()).abs() < 1e-12);
        }
    }
    assert!((score.loglik - full_loglik).abs() < 1e-12);
    for d in 0..4 {
        assert!((score.mean[4 + d] - mean[d]).abs() < 1e-12);
        assert!((score.sd[4 + d] - sd[d]).abs() < 1e-12);
    }
    let call = |mu: &[f64], sigma: &[f64], observed: &[bool], responses: &[usize]| {
        score_two_tier_grm_orthogonal(
            &a_p,
            &a_s,
            &threshold,
            mu,
            sigma,
            responses,
            Some(observed),
            &primary,
            &specific,
            2,
            5,
            2,
            2,
            3,
            7,
            7,
        )
    };
    assert!(call(&mean, &[1.0, 0.0, 1.0, 1.0], &observed, &y).is_err());
    assert!(call(&mean, &[1.0, f64::NAN, 1.0, 1.0], &observed, &y).is_err());
    assert!(call(&[f64::INFINITY, 0.0, 0.0, 0.0], &sd, &observed, &y).is_err());
    assert!(call(&[0.0], &sd, &observed, &y).is_err());
    let mut bad = y;
    bad[0] = 3;
    assert!(call(&mean, &sd, &observed, &bad).is_err());
    // Missing placeholders are ignored when their observation flag is false.
    bad[0] = 0;
    bad[5] = usize::MAX;
    assert!(call(&mean, &sd, &observed, &bad).is_ok());
    assert!(call(&[1e308; 4], &sd, &observed, &y).is_err());
    // The same fixed-bank responses may omit categories for scoring, while
    // the existing fitter still requires category coverage before estimating.
    let cfg = crate::two_tier_grm::TwoTierGrmConfig {
        estimate_primary_correlation: true,
        q_primary: 7,
        q_specific: 7,
        max_iter: 1,
        tol: 1.0,
        n_starts: 1,
        seed: 0,
        newton_iter: 1,
        ridge: 1.0,
    };
    let fitted_validation = crate::two_tier_grm::validate(
        &y,
        Some(&observed),
        &primary,
        &specific,
        2,
        5,
        2,
        2,
        3,
        &cfg,
    );
    assert!(matches!(fitted_validation, Err(message) if message.contains("never observed")));
}

#[test]
fn focal_em_preserves_one_step_moments_and_termination_receipts() {
    use crate::two_tier_grm::{fit_two_tier_grm_focal_orthogonal, score_two_tier_grm_orthogonal};
    let ap = [0.7, -0.4, 0.8, -0.2, 0.9, 0.1, 1.0, 0.2, 1.1, 0.4];
    let asp = [0.5, 0.2, -0.1, -0.4, 0.0];
    let threshold = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let primary = [true; 10];
    let specific = [0, 0, 1, 1, -1];
    let y = [0, 1, 2, 0, 1, 1, 2, 0, 1, 2, 2, 0, 1, 2, 0];
    let mu = [0.4, -0.2, 0.7, -0.6];
    let sd = [1.1, 0.8, 0.9, 1.3];
    let initial = score_two_tier_grm_orthogonal(
        &ap, &asp, &threshold, &mu, &sd, &y, None, &primary, &specific, 3, 5, 2, 2, 3, 7, 7,
    )
    .unwrap();
    let run = |q, cap, tol, observed: Option<&[bool]>| {
        fit_two_tier_grm_focal_orthogonal(
            &ap, &asp, &threshold, &mu, &sd, &y, observed, &primary, &specific, 3, 5, 2, 2, 3, q,
            q, cap, tol,
        )
    };
    let fit = run(7, 1, 1e-14, None).unwrap();
    assert_eq!(fit.n_iter, 1);
    assert_eq!(fit.loglik_trace.len(), 2);
    assert_eq!(fit.loglik_trace[0], initial.loglik);
    assert_eq!(fit.loglik_trace[1], fit.scores.loglik);
    assert_eq!(
        fit.final_loglik_change,
        fit.loglik_trace[1] - fit.loglik_trace[0]
    );
    assert_eq!(fit.initial_mean, mu);
    assert_eq!(fit.initial_sd, sd);
    assert_eq!(fit.q_primary, 7);
    assert_eq!(fit.q_specific, 7);
    assert_eq!(fit.max_iter, 1);
    assert_eq!(fit.tol, 1e-14);
    for d in 0..4 {
        let m = (0..3)
            .map(|person| initial.mean[person * 4 + d])
            .sum::<f64>()
            / 3.0;
        // Independent raw-second-moment identity at moderate mean values.
        let var = (0..3)
            .map(|person| initial.second[person * 4 + d])
            .sum::<f64>()
            / 3.0
            - m * m;
        assert!((fit.latent_mean[d] - m).abs() < 1e-12);
        assert!((fit.latent_sd[d] - var.sqrt()).abs() < 1e-12);
    }
    for q in [2, 7] {
        let result = run(q, 12, 1e-6, None).unwrap();
        assert_eq!(result.loglik_trace.len(), result.n_iter + 1);
        assert_eq!(*result.loglik_trace.last().unwrap(), result.scores.loglik);
        assert_eq!(
            result.converged,
            result.termination_reason == "tolerance_met"
        );
        match result.termination_reason {
            "loglik_decreased" => assert!(result.final_loglik_change < 0.0),
            "tolerance_met" => assert!(
                result.final_loglik_change >= 0.0 && result.final_loglik_change <= result.tol
            ),
            "max_iter_reached" => assert_eq!(result.n_iter, result.max_iter),
            other => panic!("unexpected termination {other}"),
        }
        eprintln!(
            "toy q={q}: {}, updates={}",
            result.termination_reason, result.n_iter
        );
    }
    // Moving coarse GH nodes can decrease the evaluated likelihood. Retain
    // that evaluated state and stop without reporting convergence.
    let decreased = fit_two_tier_grm_focal_orthogonal(
        &ap, &asp, &threshold, &[0.0; 4], &[2.0; 4], &y, None, &primary, &specific, 3, 5, 2, 2, 3,
        2, 2, 30, 1e-12,
    )
    .unwrap();
    assert!(!decreased.converged);
    assert_eq!(decreased.termination_reason, "loglik_decreased");
    assert!(decreased.final_loglik_change < 0.0);
    assert!(decreased.n_iter < decreased.max_iter);
    assert_eq!(
        *decreased.loglik_trace.last().unwrap(),
        decreased.scores.loglik
    );
    // Loose synthetic tolerance tests the convergence branch, not a study setting.
    let stopped = run(7, 1, 1e6, None).unwrap();
    assert!(stopped.converged);
    assert_eq!(stopped.termination_reason, "tolerance_met");
    assert_eq!(stopped.n_iter, 1);
    assert!(run(7, 0, 1e-6, None).is_err());
    assert!(run(7, 1, 0.0, None).is_err());
    assert!(run(7, 1, f64::INFINITY, None).is_err());
    assert!(run(7, 1, 1e-6, Some(&[false; 15])).is_err());
    let mut mask = [true; 15];
    for person in 0..3 {
        mask[person * 5 + 2] = false;
        mask[person * 5 + 3] = false;
    }
    assert!(run(7, 1, 1e-6, Some(&mask)).is_err());
}

/// Known-population diagnostic for the fixed-item Gaussian EM application.
/// Source: Cai (2010), DOI 10.1007/s11336-010-9178-0, pp. 608-609,
/// Appendices A/B (Gaussian latent-density objective and posterior moments).
/// The fixture independently integrates response-pattern probabilities over
/// two physical latent coordinates. Rounded expected counts remove RNG noise.
/// This bank, count, quadrature and error tolerance are test choices; the
/// source does not prescribe them or establish study-model identification.
#[test]
fn focal_gaussian_recovers_declared_distribution() {
    use crate::two_tier_grm::fit_two_tier_grm_focal_orthogonal;
    let ap = [1.2, 1.6, 0.5, 0.8];
    let asp = [0.0, 0.0, 1.4, -1.6];
    let d = [0.8, -0.7, 0.4, -1.0, 1.0, -0.5, 0.6, -0.9];
    let pm = [true; 4];
    let sm = [-1, -1, 0, 0];
    let truth_mean = [0.4, -0.3];
    let truth_sd = [1.2, 0.8];
    let (nodes, weights) = crate::quadrature::gh_rule(31).unwrap();
    let mut probabilities = vec![0.0_f64; 81];
    for (g, &z) in nodes.iter().enumerate() {
        let primary = truth_mean[0] + truth_sd[0] * z;
        for (h, &zs) in nodes.iter().enumerate() {
            let specific = truth_mean[1] + truth_sd[1] * zs;
            let mut item = [[0.0; 3]; 4];
            for i in 0..4 {
                let eta = ap[i] * primary + asp[i] * specific;
                let upper = 1.0 / (1.0 + (-(eta + d[2 * i])).exp());
                let lower = 1.0 / (1.0 + (-(eta + d[2 * i + 1])).exp());
                item[i] = [1.0 - upper, upper - lower, lower];
            }
            for (pattern, probability) in probabilities.iter_mut().enumerate() {
                let mut code = pattern;
                let mut value = weights[g] * weights[h];
                for categories in &item {
                    value *= categories[code % 3];
                    code /= 3;
                }
                *probability += value;
            }
        }
    }
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    let mut y = Vec::new();
    for (pattern, probability) in probabilities.iter().enumerate() {
        let count = (4000.0 * probability).round() as usize;
        let mut code = pattern;
        let mut row = [0; 4];
        for category in &mut row {
            *category = code % 3;
            code /= 3;
        }
        for _ in 0..count {
            y.extend_from_slice(&row);
        }
    }
    let n = y.len() / 4;
    for q in [15, 21] {
        let fit = fit_two_tier_grm_focal_orthogonal(
            &ap, &asp, &d, &[0.0; 2], &[1.0; 2], &y, None, &pm, &sm, n, 4, 1, 1, 3, q, q, 300, 1e-6,
        )
        .unwrap();
        eprintln!(
            "recovery n={n} q={q} reason={} updates={} mean={:?} sd={:?} delta={}",
            fit.termination_reason,
            fit.n_iter,
            fit.latent_mean,
            fit.latent_sd,
            fit.final_loglik_change
        );
        assert!(fit.converged, "{}", fit.termination_reason);
        for dim in 0..2 {
            assert!((fit.latent_mean[dim] - truth_mean[dim]).abs() < 0.04);
            assert!((fit.latent_sd[dim] - truth_sd[dim]).abs() < 0.04);
        }
    }
}

/// Same-node posterior oracle for shared G/W and four specific blocks.
/// Cai (2010), DOI 10.1007/s11336-010-9178-0, pp.589-590 eqs.11-12,15-16 and
/// pp.608-609 Appendices A/B: integrate shared primary dimensions jointly,
/// and use posterior first/second moments for EAP and variance.
/// This synthetic 16-item layout mirrors the study's G+4+W loading pattern;
/// parameters and five-point nodes are test choices, not study estimates or
/// evidence of quadrature adequacy, identification or population recovery.
#[test]
fn six_latent_crossed_primary_scores_match_full_product() {
    use crate::two_tier_grm::score_two_tier_grm_orthogonal;
    let wording = [4_usize, 5, 6, 9, 12, 14, 15];
    let mut ap = vec![0.0; 32];
    let mut asp = vec![0.0; 16];
    let mut threshold = vec![0.0; 48];
    let mut pm = vec![false; 32];
    let sm: Vec<i32> = (0..16).map(|i| (i / 4) as i32).collect();
    for i in 0..16 {
        pm[2 * i] = true;
        ap[2 * i] = 0.65 + 0.15 * (i % 4) as f64;
        if wording.contains(&i) {
            pm[2 * i + 1] = true;
            ap[2 * i + 1] = if i % 2 == 0 { 0.45 } else { -0.5 };
        }
        asp[i] = 0.35 + 0.2 * (i % 3) as f64;
        for (k, value) in [1.0, 0.1, -0.9].iter().enumerate() {
            threshold[3 * i + k] = value + 0.04 * (i % 5) as f64;
        }
    }
    let mu = [0.4, -0.3, 0.2, -0.5, 0.6, -0.2];
    let sd = [1.2, 0.8, 0.9, 1.1, 0.7, 1.3];
    let y: Vec<usize> = (0..48).map(|i| (i + i / 16) % 4).collect();
    let mut mask = vec![true; 48];
    for i in 0..16 {
        mask[16 + i] = !(4..8).contains(&i) && i != 12;
        mask[32 + i] = false;
    }
    let scores = score_two_tier_grm_orthogonal(
        &ap,
        &asp,
        &threshold,
        &mu,
        &sd,
        &y,
        Some(&mask),
        &pm,
        &sm,
        3,
        16,
        2,
        4,
        4,
        5,
        5,
    )
    .unwrap();
    let (nodes, weights) = crate::quadrature::gh_rule(5).unwrap();
    let mut total_loglik = 0.0;
    for person in 0..3 {
        let mut mass = 0.0;
        let mut first = [0.0; 6];
        let mut second = [0.0; 6];
        for code in 0..5_usize.pow(6) {
            let mut remaining = code;
            let mut theta = [0.0; 6];
            let mut probability = 1.0;
            for dim in 0..6 {
                let h = remaining % 5;
                remaining /= 5;
                theta[dim] = mu[dim] + sd[dim] * nodes[h];
                probability *= weights[h];
            }
            for i in 0..16 {
                if !mask[person * 16 + i] {
                    continue;
                }
                let eta = ap[2 * i] * theta[0]
                    + ap[2 * i + 1] * theta[1]
                    + asp[i] * theta[2 + sm[i] as usize];
                let mut cumulative = [1.0, 0.0, 0.0, 0.0, 0.0];
                for k in 0..3 {
                    cumulative[k + 1] = 1.0 / (1.0 + (-(eta + threshold[3 * i + k])).exp());
                }
                let category = y[person * 16 + i];
                probability *= cumulative[category] - cumulative[category + 1];
            }
            mass += probability;
            for dim in 0..6 {
                first[dim] += probability * theta[dim];
                second[dim] += probability * theta[dim] * theta[dim];
            }
        }
        total_loglik += mass.ln();
        for dim in 0..6 {
            let mean = first[dim] / mass;
            let raw_second = second[dim] / mass;
            let posterior_sd = (raw_second - mean * mean).sqrt();
            assert!((scores.mean[person * 6 + dim] - mean).abs() < 1e-10);
            assert!((scores.second[person * 6 + dim] - raw_second).abs() < 1e-10);
            assert!((scores.sd[person * 6 + dim] - posterior_sd).abs() < 1e-10);
            if person == 2 {
                assert!((scores.mean[person * 6 + dim] - mu[dim]).abs() < 1e-12);
                assert!((scores.sd[person * 6 + dim] - sd[dim]).abs() < 1e-12);
            }
        }
    }
    assert!((scores.loglik - total_loglik).abs() < 1e-10);
    // The six-dimensional crossed-primary bank must pass the mean-rank
    // guard; this is a one-update contract check, not population recovery.
    assert!(crate::two_tier_grm::fit_two_tier_grm_focal_orthogonal(
        &ap,
        &asp,
        &threshold,
        &mu,
        &sd,
        &y,
        Some(&mask),
        &pm,
        &sm,
        3,
        16,
        2,
        4,
        4,
        5,
        5,
        1,
        1e-6,
    )
    .is_ok());
    // Reproduce the installed-native fixture with cargo test -- --nocapture.
    eprintln!("SIX_FIXTURE={{\"a_primary\":{:?},\"a_specific\":{:?},\"threshold\":{:?},\"primary_map\":{:?},\"specific_map\":{:?},\"latent_mean\":{:?},\"latent_sd\":{:?},\"responses\":{:?},\"observed\":{:?},\"person_mean\":{:?},\"person_second\":{:?},\"person_sd\":{:?},\"loglik\":{}}}", ap,asp,threshold,pm,sm,mu,sd,y,mask,scores.mean,scores.second,scores.sd,scores.loglik);
}

/// Free latent means require an injective fixed-item predictor map.
/// Derived from Cai (2010), p.589 eq.11, DOI 10.1007/s11336-010-9178-0:
/// A*v=0 makes mu and mu+v observationally equivalent. The shared pivot
/// guard follows the opened LAPACK DGETF2 Purpose/INFO/pivot loop; its
/// caller threshold is an implementation guard, not DGELSY effective rank.
#[test]
fn focal_mean_rank_rejects_nullspaces_and_preserves_declared_scoring() {
    use crate::two_tier_grm::{fit_two_tier_grm_focal_orthogonal, score_two_tier_grm_orthogonal};
    let ap = [0.8, 1.2, 1.0, 0.6];
    let asp = ap;
    let d = [0.8, -0.6, 0.8, -0.6, 0.8, -0.6, 0.8, -0.6];
    let y = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 0, 0, 1, 2, 1, 0];
    let pm = [true; 4];
    let sm = [0; 4];
    let score = |mu: &[f64]| {
        score_two_tier_grm_orthogonal(
            &ap, &asp, &d, mu, &[1.0; 2], &y, None, &pm, &sm, 4, 4, 1, 1, 3, 5, 5,
        )
        .unwrap()
    };
    // Declared-prior scoring remains meaningful; fitting both means does not.
    let a = score(&[0.0, 0.0]);
    let b = score(&[0.5, -0.5]);
    assert_eq!(a.loglik, b.loglik);
    let fit = fit_two_tier_grm_focal_orthogonal(
        &ap, &asp, &d, &[0.0; 2], &[1.0; 2], &y, None, &pm, &sm, 4, 4, 1, 1, 3, 5, 5, 1, 1e6,
    );
    assert!(matches!(fit, Err(message) if message.contains("column rank")));
    // A third column can be a sum of two independent columns; detecting
    // only pairwise duplicates or nonzero columns would miss this nullspace.
    let ap3 = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0];
    let asp3 = [1.0, 1.0, 2.0, 1.0];
    let pm3 = [true; 8];
    let fit3 = |asp: &[f64], mask: Option<&[bool]>| {
        fit_two_tier_grm_focal_orthogonal(
            &ap3, asp, &d, &[0.0; 3], &[1.0; 3], &y, mask, &pm3, &sm, 4, 4, 2, 1, 3, 5, 5, 1, 1e6,
        )
    };
    assert!(matches!(fit3(&asp3,None),Err(message) if message.contains("column rank")));
    let independent = [0.0, 0.0, 1.0, 1.0];
    assert!(fit3(&independent, None).is_ok());
    let mask: Vec<bool> = (0..16).map(|i| i % 4 == 0 || i % 4 == 2).collect();
    assert!(
        matches!(fit3(&independent,Some(&mask)),Err(message) if message.contains("column rank"))
    );
    // Existing square-Gram users keep their caller thresholds; the direct
    // rectangular path avoids squaring the condition number.
    let rank = |mut matrix: Vec<Vec<f64>>, cols, threshold| {
        crate::lltm::gram_full_rank(&mut matrix, cols, threshold)
    };
    assert!(rank(vec![vec![1.0, 0.0], vec![0.0, 1.0]], 2, 1e-9));
    assert!(!rank(vec![vec![1.0, 1.0], vec![1.0, 1.0]], 2, 1e-9));
    assert!(rank(
        vec![vec![0.0, 1.0], vec![1.0, 1.0], vec![2.0, -1.0]],
        2,
        3.0 * f64::EPSILON
    ));
    assert!(rank(
        vec![vec![1.0, 1.0], vec![1.0, 1.0 + 1e-10]],
        2,
        2.0 * f64::EPSILON
    ));
    assert!(rank(
        vec![vec![1.0, 0.0], vec![2.0, 0.0], vec![0.0, 1.0]],
        2,
        3.0 * f64::EPSILON
    ));
    assert!(!rank(vec![vec![1.0, 0.0]], 2, f64::EPSILON));
    assert!(!rank(vec![vec![1.0], vec![0.0, 1.0]], 2, f64::EPSILON));
    assert!(!rank(vec![vec![f64::NAN]], 1, f64::EPSILON));
    assert!(!rank(vec![vec![1.0]], 1, f64::NAN));
    assert!(!rank(vec![vec![0.0]], 1, 0.0));
}
