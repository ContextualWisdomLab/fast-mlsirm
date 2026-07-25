//! Tests for confirmatory DETECT (Zhang & Stout, 1999; transcribed from CRAN
//! sirt's `scale_score=FALSE` sum-score path).
//!
//! Fixture literals were computed by an INDEPENDENT NumPy transcription of
//! the R semantics (ML covariance dividing by n, group-frequency weights,
//! bias-corrected average of total-score and rest-score conditioning); the
//! generating script never imports this crate.
//!
//! Disclosed limitations of this suite:
//! - Grouping by unique score values is invariant to strictly monotone score
//!   transforms, so NO fixture here can discriminate `scale_score=FALSE`
//!   (implemented) from a z-standardizing mutant whose rounding merges no
//!   groups; the module doc's scope statement pins that contract instead.
//! - Under a single cluster, DETECT == MCOV100 is an identity (delta = +1
//!   everywhere); the single-cluster test therefore anchors DETECT to an
//!   external literal and does not use that equality as a signal.
//! - |RATIO| <= 1 is the triangle inequality (tautological); never asserted
//!   as a correctness signal.

use super::*;

const TOL: f64 = 1e-12;

/// 9 persons x 5 items; item 4 is constant (=1), forcing exact ccov = 0 for
/// its four pairs (exercises R's sign(0) = 0 in ASSI).
fn fixture_x() -> Vec<f64> {
    [
        [1, 0, 0, 1, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 1, 0, 1, 1],
        [1, 0, 1, 0, 1],
    ]
    .iter()
    .flatten()
    .map(|&v| v as f64)
    .collect()
}

/// Asserts read: DetectResult.{detect, assi, ratio, madcov100, mcov100}
/// returned by the crate. Killed by: delta sign swap, dropped bias
/// correction, n-1 covariance divisor, missing/extra x100 scaling on any
/// index, sign(0) = +1, wrong weight normalization.
#[test]
fn detect_fixture_values() {
    let x = fixture_x();
    let r = detect_analysis(&x, 9, 5, &[0, 0, 1, 1, 0]).unwrap();
    assert!((r.detect - (-0.2830687830687831)).abs() < TOL, "{}", r.detect);
    assert!((r.assi - (-0.2)).abs() < TOL, "{}", r.assi);
    assert!((r.ratio - (-0.05439755973563803)).abs() < TOL, "{}", r.ratio);
    assert!((r.madcov100 - 5.203703703703703).abs() < TOL, "{}", r.madcov100);
    assert!((r.mcov100 - (-3.5105820105820107)).abs() < TOL, "{}", r.mcov100);
    assert_eq!(r.n_pairs, 10);
}

/// Asserts read: DetectResult.{pair_i, pair_j, ccov} per pair. Killed by:
/// rest score computed as S - X_i only (forgot X_j), aggregation weight
/// 1/n_groups instead of group frequency, singleton-group mishandling,
/// reusing total-score weights for the rest-score pass, pair-order/layout
/// bugs (all six non-constant ccovs are distinct).
#[test]
fn ccov_per_pair_fixture() {
    let x = fixture_x();
    let r = detect_analysis(&x, 9, 5, &[0, 0, 1, 1, 0]).unwrap();
    let expected: [(usize, usize, f64); 10] = [
        (0, 1, -0.10317460317460317),
        (0, 2, 0.025132275132275124),
        (0, 3, -0.11904761904761904),
        (0, 4, 0.0),
        (1, 2, -0.12698412698412698),
        (1, 3, 0.05952380952380952),
        (1, 4, 0.0),
        (2, 3, -0.08650793650793652),
        (2, 4, 0.0),
        (3, 4, 0.0),
    ];
    assert_eq!(r.ccov.len(), 10);
    for (k, &(i, j, c)) in expected.iter().enumerate() {
        assert_eq!(r.pair_i[k], i);
        assert_eq!(r.pair_j[k], j);
        assert!((r.ccov[k] - c).abs() < TOL, "pair ({i},{j}): {}", r.ccov[k]);
    }
}

/// Interleaved (non-contiguous) clusters over the SAME responses. Asserts
/// read: crate indices under cluster [0,1,0,1,0]. Killed by: contiguous-
/// block cluster assumptions, label-position bugs, delta mapping errors
/// (DETECT here is +5.2037..., not the negation of the primary fixture).
#[test]
fn delta_structure_interleaved() {
    let x = fixture_x();
    let r = detect_analysis(&x, 9, 5, &[0, 1, 0, 1, 0]).unwrap();
    assert!((r.detect - 5.203703703703703).abs() < TOL, "{}", r.detect);
    assert!((r.assi - 0.6).abs() < TOL, "{}", r.assi);
    assert!((r.ratio - 1.0).abs() < TOL, "{}", r.ratio);
    assert!((r.madcov100 - 5.203703703703703).abs() < TOL, "{}", r.madcov100);
    assert!((r.mcov100 - (-3.5105820105820107)).abs() < TOL, "{}", r.mcov100);
}

/// All items in one cluster: DETECT anchored to an external literal.
/// (crate-DETECT == crate-MCOV100 here is an identity since delta = +1
/// everywhere — deliberately NOT used as a signal.) Asserts read: crate
/// detect/assi under a shared nonzero label.
#[test]
fn single_cluster_anchor() {
    let x = fixture_x();
    let r = detect_analysis(&x, 9, 5, &[7, 7, 7, 7, 7]).unwrap();
    assert!((r.detect - (-3.5105820105820107)).abs() < TOL, "{}", r.detect);
    assert!((r.assi - (-0.2)).abs() < TOL, "{}", r.assi);
    assert!((r.ratio - (-0.6746314184036604)).abs() < TOL, "{}", r.ratio);
}

/// Hostile labels: extreme i64 values must behave exactly like [0,0,1,1,0]
/// (labels compared for equality, never used as indices). Asserts read:
/// crate indices under i64::MIN/i64::MAX labels. Killed by: any use of the
/// label value as an array index or dense-relabeling arithmetic.
#[test]
fn hostile_cluster_labels() {
    let x = fixture_x();
    let cl = [i64::MIN, i64::MIN, i64::MAX, i64::MAX, i64::MIN];
    let r = detect_analysis(&x, 9, 5, &cl).unwrap();
    assert!((r.detect - (-0.2830687830687831)).abs() < TOL, "{}", r.detect);
    assert!((r.assi - (-0.2)).abs() < TOL, "{}", r.assi);
}

/// Guard rejections. Asserts read: Err values returned by the crate.
#[test]
fn guard_rejections() {
    let x = fixture_x();
    // cluster length mismatch
    assert!(detect_analysis(&x, 9, 5, &[0, 0, 1, 1]).is_err());
    // non-binary value
    let mut bad = x.clone();
    bad[3] = 0.5;
    assert!(detect_analysis(&bad, 9, 5, &[0, 0, 1, 1, 0]).is_err());
    // NaN (missing data not supported)
    let mut nan = x.clone();
    nan[7] = f64::NAN;
    assert!(detect_analysis(&nan, 9, 5, &[0, 0, 1, 1, 0]).is_err());
    // too few items / persons
    assert!(detect_analysis(&[0.0, 1.0], 2, 1, &[0]).is_err());
    assert!(detect_analysis(&[0.0, 1.0], 1, 2, &[0, 1]).is_err());
    // length mismatch
    assert!(detect_analysis(&x[..40], 9, 5, &[0, 0, 1, 1, 0]).is_err());
    // all-zero conditional covariances (two constant items): RATIO is 0/0 in
    // R; the crate must return Err, not NaN.
    let ones = vec![1.0; 8];
    assert!(detect_analysis(&ones, 4, 2, &[0, 1]).is_err());
}

/// Monte Carlo behavioral check (>= 500 replications; run with --ignored).
/// 2D simple structure (theta correlation 0.3): DETECT with the correct
/// partition should be clearly positive on average; unidimensional data with
/// the same partition should yield DETECT near zero. Asserts read: mean of
/// crate detect values across replications.
#[test]
#[ignore]
fn monte_carlo_detect_recovery() {
    // Minimal deterministic LCG (no rand dependency in dev-deps for core).
    struct Lcg(u64);
    impl Lcg {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        /// Box-Muller standard normal.
        fn next_norm(&mut self) -> f64 {
            let (u1, u2) = (self.next_f64().max(1e-12), self.next_f64());
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        }
    }
    let reps = 500;
    let (n, items_per_dim) = (400, 6);
    let n_items = 2 * items_per_dim;
    let cluster: Vec<i64> = (0..n_items).map(|i| (i / items_per_dim) as i64).collect();
    let mut rng = Lcg(20240601);
    let (mut sum_multi, mut sum_uni) = (0.0, 0.0);
    for _ in 0..reps {
        // Multidimensional: theta1, theta2 with corr 0.3.
        let mut x_multi = vec![0.0; n * n_items];
        let mut x_uni = vec![0.0; n * n_items];
        for p in 0..n {
            let z1 = rng.next_norm();
            let z2 = 0.3 * z1 + (1.0f64 - 0.09).sqrt() * rng.next_norm();
            let t_uni = rng.next_norm();
            for i in 0..n_items {
                let b = -1.0 + 2.0 * (i % items_per_dim) as f64 / (items_per_dim - 1) as f64;
                let th = if i < items_per_dim { z1 } else { z2 };
                let pr_m = 1.0 / (1.0 + (-(1.2 * (th - b))).exp());
                let pr_u = 1.0 / (1.0 + (-(1.2 * (t_uni - b))).exp());
                let u = rng.next_f64();
                x_multi[p * n_items + i] = if u < pr_m { 1.0 } else { 0.0 };
                // reuse the same uniform for the unidimensional draw is NOT
                // independent; draw a fresh one.
                let u2 = rng.next_f64();
                x_uni[p * n_items + i] = if u2 < pr_u { 1.0 } else { 0.0 };
            }
        }
        sum_multi += detect_analysis(&x_multi, n, n_items, &cluster).unwrap().detect;
        sum_uni += detect_analysis(&x_uni, n, n_items, &cluster).unwrap().detect;
    }
    let (mean_multi, mean_uni) = (sum_multi / reps as f64, sum_uni / reps as f64);
    assert!(
        mean_multi > 0.5,
        "2D simple structure should give clearly positive DETECT, got {mean_multi}"
    );
    assert!(
        mean_uni.abs() < 0.2,
        "unidimensional data should give DETECT near zero, got {mean_uni}"
    );
    assert!(
        mean_multi > mean_uni + 0.3,
        "multidimensional DETECT should exceed unidimensional: {mean_multi} vs {mean_uni}"
    );
}
