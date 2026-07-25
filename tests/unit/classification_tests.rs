//! Tests for IRT classification accuracy/consistency (Rudner and Lee).
//!
//! Every assert reads values returned by the crate
//! (`rudner_classification` / `lee_classification` outputs); no assert is
//! computed from test-local re-derivations. Fixture literals come from an
//! independent NumPy transcription of the verified sources (session artifact
//! `cac_fixture.py`) that never imports this crate.
//!
//! Disclosed limitations:
//! - For a single cut, simultaneous == per-cut outputs by construction
//!   (deliberate API divergence from cacIRT); the m = 2 fixtures anchor the
//!   case where they differ.
//! - Rudner values inherit the crate `erfc` accuracy (|err| < 1.2e-7), so
//!   Rudner asserts use 1e-6 tolerances; the targeted mutations shift values
//!   by >= 1e-3. Lee asserts are exact-recursion values pinned at 1e-12.
//! - Uniform-weight == unweighted-mean is an identity; the non-uniform,
//!   non-normalized weight fixtures anchor weight handling.

use super::{lee_classification, rudner_classification};

const RUD_THETA: [f64; 7] = [-1.7, -0.63, -0.4, 0.11, 0.52, 1.28, 1.95];
const RUD_SEM: [f64; 7] = [0.85, 0.42, 0.31, 0.27, 0.33, 0.48, 0.66];
// deliberately unnormalized (sum 8.5): kills a "forgot to normalize" mutation
const RUD_W: [f64; 7] = [0.4, 1.1, 1.7, 2.3, 1.9, 0.8, 0.3];
const RUD_CUTS: [f64; 2] = [-0.4, 0.85];

#[test]
fn rudner_matches_independent_reference() {
    // Kills: swapped tail direction, dropped squared term in consistency,
    // per-cut/simultaneous mixups, unnormalized-weight aggregation.
    let r = rudner_classification(&RUD_THETA, &RUD_SEM, &RUD_W, &RUD_CUTS).unwrap();
    let tol = 1e-6;
    assert!((r.per_cut_accuracy[0] - 0.8506551287597999).abs() < tol);
    assert!((r.per_cut_accuracy[1] - 0.9444953787199609).abs() < tol);
    assert!((r.per_cut_consistency[0] - 0.8242222191464211).abs() < tol);
    assert!((r.per_cut_consistency[1] - 0.9068646398531202).abs() < tol);
    assert!((r.simultaneous_accuracy - 0.7952699812339523).abs() < tol);
    assert!((r.simultaneous_consistency - 0.7315355182946305).abs() < tol);
    // conditional rows (m x n row-major)
    assert!((r.conditional_accuracy[0] - 0.936918801128759).abs() < tol);
    assert!((r.conditional_accuracy[7 + 4] - 0.8413447460685428).abs() < tol);
    assert!((r.conditional_consistency[1] - 0.5865473418589322).abs() < tol);
    assert!((r.conditional_simultaneous_accuracy[4] - 0.8386920265482238).abs() < tol);
    assert!((r.conditional_simultaneous_consistency[6] - 0.9089695170230561).abs() < tol);
}

#[test]
fn rudner_theta_on_cut_classifies_into_upper_category() {
    // theta[2] == -0.4 sits EXACTLY on cut 1. Left-closed categorization
    // puts it in the upper category, so its conditional accuracy is the
    // upper-tail mass 0.5 (theta centered on the cut). A right-closed
    // mutation flips it to the lower mass — also 0.5 here, so the
    // discriminating read is the SIMULTANEOUS accuracy, whose upper-category
    // mass [−0.4, 0.85) = 0.4999723782624206 differs from the lower-category
    // mass under a right-closed mutation (Phi((-0.4+0.4)/0.31) - 0 = 0.5).
    let r = rudner_classification(&RUD_THETA, &RUD_SEM, &RUD_W, &RUD_CUTS).unwrap();
    assert!((r.conditional_accuracy[2] - 0.5).abs() < 1e-6);
    assert!((r.conditional_simultaneous_accuracy[2] - 0.4999723782624206).abs() < 1e-6);
    assert!((r.conditional_simultaneous_consistency[2] - 0.49997237978834136).abs() < 1e-6);
}

#[test]
fn rudner_rejects_degenerate_inputs() {
    let ok_t = [0.0, 1.0];
    let ok_s = [0.5, 0.5];
    let ok_w = [1.0, 1.0];
    assert!(rudner_classification(&[], &[], &[], &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &[0.5], &ok_w, &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &[0.5, 0.0], &ok_w, &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &[0.5, f64::NAN], &ok_w, &[0.0]).is_err());
    assert!(rudner_classification(&[0.0, f64::INFINITY], &ok_s, &ok_w, &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &ok_s, &[1.0, -1.0], &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &ok_s, &[0.0, 0.0], &[0.0]).is_err());
    // finite weights whose SUM overflows to inf would otherwise normalize
    // every weight to 0.0 and silently zero all marginals
    assert!(rudner_classification(&ok_t, &ok_s, &[1e308, 1e308], &[0.0]).is_err());
    assert!(rudner_classification(&ok_t, &ok_s, &ok_w, &[]).is_err());
    assert!(rudner_classification(&ok_t, &ok_s, &ok_w, &[0.3, 0.3]).is_err());
    assert!(rudner_classification(&ok_t, &ok_s, &ok_w, &[0.3, f64::NAN]).is_err());
}

const LEE_P: [f64; 30] = [
    0.08, 0.15, 0.22, 0.31, 0.12, 0.19, //
    0.23, 0.34, 0.41, 0.52, 0.28, 0.37, //
    0.47, 0.55, 0.61, 0.68, 0.51, 0.58, //
    0.66, 0.72, 0.79, 0.83, 0.69, 0.76, //
    0.81, 0.87, 0.90, 0.93, 0.84, 0.88,
];
const LEE_W: [f64; 5] = [0.6, 1.4, 2.0, 1.4, 0.6];
const LEE_CUTS: [f64; 2] = [2.4, 4.0]; // non-integer cut kills floor-vs-ceil

#[test]
fn lee_matches_independent_reference() {
    // Kills: floor(c) boundary mutation (cut 2.4), dropped square in
    // consistency, per-cut/simultaneous mixups, unnormalized weights,
    // Lord-Wingersky misuse (wrong table orientation collapses everything).
    let r = lee_classification(&LEE_P, 5, 6, &LEE_W, &LEE_CUTS).unwrap();
    let tol = 1e-12;
    assert!((r.per_cut_accuracy[0] - 0.8221250493919999).abs() < tol);
    assert!((r.per_cut_accuracy[1] - 0.7672704000213333).abs() < tol);
    assert!((r.per_cut_consistency[0] - 0.7432160432137724).abs() < tol);
    assert!((r.per_cut_consistency[1] - 0.7071425914148844).abs() < tol);
    assert!((r.simultaneous_accuracy - 0.6284128106933332).abs() < tol);
    assert!((r.simultaneous_consistency - 0.5754886794891432).abs() < tol);
    assert!((r.conditional_accuracy[1] - 0.6320063246400001).abs() < tol);
    assert!((r.conditional_consistency[5 + 2] - 0.5012741633381826).abs() < tol);
    assert!((r.conditional_simultaneous_accuracy[2] - 0.2992865452).abs() < tol);
    assert!((r.conditional_simultaneous_consistency[3] - 0.6924377380497955).abs() < tol);
}

#[test]
fn lee_true_score_on_cut_classifies_into_upper_category() {
    // Dyadic P row sums to EXACTLY 3.0 (binary64); cut at 3.0. Left-closed
    // categorization -> upper category -> accuracy = P(X >= 3) = 0.6640625.
    // A right-closed mutation reads the lower mass 1 - 0.6640625 = 0.3359375.
    let p = [0.5, 0.5, 0.5, 0.5, 0.25, 0.75];
    let r = lee_classification(&p, 1, 6, &[1.0], &[3.0]).unwrap();
    assert!((r.per_cut_accuracy[0] - 0.6640625).abs() < 1e-12);
    assert!((r.per_cut_consistency[0] - 0.5538330078125).abs() < 1e-12);
}

#[test]
fn lee_rejects_degenerate_inputs() {
    let p = [0.2, 0.4, 0.6, 0.8];
    assert!(lee_classification(&p, 2, 2, &[1.0, 1.0], &[1.5]).is_ok());
    // probs on/outside the open interval
    assert!(lee_classification(&[0.0, 0.4, 0.6, 0.8], 2, 2, &[1.0, 1.0], &[1.5]).is_err());
    assert!(lee_classification(&[1.0, 0.4, 0.6, 0.8], 2, 2, &[1.0, 1.0], &[1.5]).is_err());
    assert!(lee_classification(&[f64::NAN, 0.4, 0.6, 0.8], 2, 2, &[1.0, 1.0], &[1.5]).is_err());
    // shape mismatch
    assert!(lee_classification(&p, 2, 3, &[1.0, 1.0], &[1.5]).is_err());
    assert!(lee_classification(&p, 0, 2, &[], &[1.5]).is_err());
    // raw cuts outside (0, n_items]
    assert!(lee_classification(&p, 2, 2, &[1.0, 1.0], &[0.0]).is_err());
    assert!(lee_classification(&p, 2, 2, &[1.0, 1.0], &[2.5]).is_err());
    // ceil-collision: 0.2 and 0.9 both ceil to 1
    assert!(lee_classification(&p, 2, 2, &[1.0, 1.0], &[0.2, 0.9]).is_err());
    // weights
    assert!(lee_classification(&p, 2, 2, &[0.0, 0.0], &[1.5]).is_err());
    assert!(lee_classification(&p, 2, 2, &[1e308, 1e308], &[1.5]).is_err());
    assert!(lee_classification(&p, 2, 2, &[1.0], &[1.5]).is_err());
}

/// 500-replication Monte Carlo: on 2PL-simulated tests, longer/more
/// informative tests must yield strictly higher marginal simultaneous
/// accuracy and consistency than short noisy tests, and both statistics must
/// stay in (0, 1]. Reads only crate outputs. Run with `--ignored`.
#[test]
#[ignore]
fn monte_carlo_accuracy_orders_test_quality() {
    // Minimal LCG (Numerical Recipes constants) — no external dependencies.
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut unif = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let n_nodes = 21;
    let nodes: Vec<f64> = (0..n_nodes).map(|i| -3.0 + 6.0 * i as f64 / 20.0).collect();
    // standard normal quadrature-ish weights (unnormalized; crate normalizes)
    let wts: Vec<f64> = nodes.iter().map(|t| (-0.5 * t * t).exp()).collect();
    let mut wins_acc = 0usize;
    let mut wins_con = 0usize;
    let reps = 500;
    for _ in 0..reps {
        let mut probs = |n_items: usize, a_lo: f64, a_hi: f64| -> Vec<f64> {
            let items: Vec<(f64, f64)> = (0..n_items)
                .map(|_| (a_lo + (a_hi - a_lo) * unif(), -1.5 + 3.0 * unif()))
                .collect();
            nodes
                .iter()
                .flat_map(|&t| {
                    items
                        .iter()
                        .map(move |&(a, b)| 1.0 / (1.0 + (-a * (t - b)).exp()))
                })
                .collect()
        };
        let short = probs(8, 0.4, 0.9);
        let long = probs(40, 1.2, 2.2);
        let rs = lee_classification(&short, n_nodes, 8, &wts, &[4.0]).unwrap();
        let rl = lee_classification(&long, n_nodes, 40, &wts, &[20.0]).unwrap();
        for r in [&rs, &rl] {
            assert!(r.simultaneous_accuracy > 0.0 && r.simultaneous_accuracy <= 1.0);
            assert!(r.simultaneous_consistency > 0.0 && r.simultaneous_consistency <= 1.0);
        }
        if rl.simultaneous_accuracy > rs.simultaneous_accuracy {
            wins_acc += 1;
        }
        if rl.simultaneous_consistency > rs.simultaneous_consistency {
            wins_con += 1;
        }
    }
    // The informative long test should dominate in the vast majority of reps.
    assert!(
        wins_acc as f64 / reps as f64 > 0.95,
        "acc wins {wins_acc}/{reps}"
    );
    assert!(
        wins_con as f64 / reps as f64 > 0.95,
        "con wins {wins_con}/{reps}"
    );
}
