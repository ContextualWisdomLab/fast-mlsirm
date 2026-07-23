//! Tests for Horn's parallel analysis (`crate::parallel`).
//!
//! Fixture literals were computed by an independent NumPy replication
//! (session `files/pa_fixture.py`) that mirrors this crate's LCG/Box-Muller
//! stream with explicit `np.uint64` arithmetic and uses `np.corrcoef` +
//! `np.linalg.eigvalsh` (LAPACK) for the linear algebra, so every pinned
//! value cross-checks the Jacobi eigensolver, the correlation code, the
//! benchmark reduction, and the adjustment/retention pipeline against an
//! implementation that shares no code with this crate. Tolerance 1e-9
//! (Jacobi converges to 1e-12 off-diagonal; LAPACK differs in the last bits).
//!
//! Mutation kills verified by hand (each mutant compiled and observed to
//! FAIL before being reverted):
//! - M1 uncentered correlation (skip mean subtraction): fixture A data has
//!   nonzero column means by construction -> `fixture_a_mean_benchmark` fails.
//! - M2 quantile without interpolation (index `ceil(h)`): fixture B's
//!   `h = 49 * 0.95 = 46.55` is fractional -> `fixture_b_centile_95` fails.
//! - M3 bias without the `- 1` (paran's CFA formula): fixture A bias/adjusted
//!   literals and retained count shift -> `fixture_a_mean_benchmark` fails.
//! - M4 retention as `count(adj > 1)` instead of first-failure scan:
//!   `retention_scan_stops_at_first_failure` uses [1.2, 0.9, 1.4] -> fails.
//! - M5 eigenvalues not re-sorted descending after Jacobi:
//!   `eigen_order_anchor`'s matrix diagonalizes with an ascending diagonal ->
//!   fails without the explicit sort.
//!
//! Known identity limits: the trace identity (sum of eigenvalues = p) holds
//! for any trace-preserving mutant and is asserted only as a sanity check,
//! never as the sole discriminating assert; the discriminating anchors are
//! the externally computed fixture literals above.

use super::{parallel_analysis, retained_count, symmetric_eigenvalues_desc};

const TOL: f64 = 1e-9;

const PA_N: usize = 40;
const PA_P: usize = 6;
#[rustfmt::skip]
const PA_DATA: [f64; 240] = [
    0.333503, -2.453654, 3.848587, 0.369779, -1.916606, 4.949936,
    1.026322, -2.084842, 3.11915, -0.8241, -3.145648, 4.694622,
    1.552238, -2.359914, 2.981319, -0.127321, -3.118339, 3.789723,
    1.09053, -1.706376, 3.630015, -1.237264, -1.718574, 3.712301,
    -0.011874, -2.792449, 2.230646, 0.954288, -0.676455, 4.202065,
    2.046904, -1.840269, 4.448778, -0.1642, -2.330755, 4.361154,
    -0.070695, -3.141409, 3.780624, 0.945734, -0.868845, 6.67772,
    0.325687, -2.210104, 2.360288, 1.214687, -2.085856, 3.844716,
    1.987676, 0.250326, 4.229779, 1.201447, -1.766095, 4.466545,
    2.792376, -0.794664, 3.711836, 1.464382, -1.295142, 4.959527,
    0.333378, -2.316583, 2.953186, 1.215496, -1.016352, 5.35509,
    1.020831, -2.288995, 3.424951, -0.095608, -1.134892, 5.041666,
    -0.258048, -2.688289, 3.104094, 0.41638, -1.675181, 5.081826,
    0.533767, -2.231017, 3.483801, 0.294858, -2.387986, 3.88554,
    0.603806, -2.543523, 3.252245, -0.091563, -1.735494, 4.115728,
    0.172629, -2.915245, 3.586819, 1.242912, -1.56492, 4.770616,
    2.226923, 0.563981, 4.284004, 1.589635, -0.429457, 5.496989,
    1.250357, -2.280015, 3.148378, 1.992453, -0.799593, 5.95754,
    1.20547, -2.280869, 4.473257, 0.455678, -2.19263, 4.121817,
    0.916669, -1.318089, 3.419921, 0.945613, -1.384316, 6.121018,
    1.196008, -1.730592, 3.418524, 1.383073, -1.061234, 5.365921,
    -0.257571, -1.96941, 1.970674, 0.093193, -1.333776, 3.897921,
    0.61962, -1.820835, 4.064294, 2.006661, -0.789629, 4.325192,
    2.312542, -0.812288, 3.325681, 0.5785, -1.732155, 4.437357,
    1.41419, -1.973323, 3.083683, 1.075274, -1.480239, 5.052933,
    1.505653, -1.056415, 3.252816, 1.429761, -0.88199, 5.940036,
    1.066391, -1.652664, 4.712282, 1.062132, -0.848488, 4.366314,
    1.464433, -2.137135, 3.227048, 1.539832, -1.064145, 5.23423,
    -0.135288, -1.620645, 3.866206, 0.174687, -0.699639, 5.210777,
    0.278362, -2.071635, 3.14087, 1.25105, -0.85097, 5.61079,
    -0.032029, -3.293086, 3.63403, 0.508087, -0.659782, 4.454752,
    1.522665, -1.992241, 3.660703, 0.375077, -1.749287, 5.780725,
    0.471204, -2.859986, 2.957715, 1.59788, 0.069784, 4.820623,
    1.235528, -1.710864, 3.226485, 0.989522, -0.743298, 4.939406,
    0.587199, -3.078375, 3.47583, -0.519378, -2.137547, 3.895758,
    -0.399063, -3.111449, 2.544858, 0.649683, -1.086784, 5.043018,
    -0.013262, -4.922791, 2.067882, 1.336552, -1.08281, 4.974112,
    1.240054, -1.573638, 3.159136, 1.606172, -0.508916, 5.737783,
    1.17282, -1.835477, 3.961551, 2.061825, -0.716887, 5.814711,
    2.13695, -0.577062, 4.669558, 1.289963, 0.226037, 5.254548,
];

const PA_A_EV: [f64; 6] = [
    2.3959774381235563,
    1.895617877600974,
    0.6090518440099216,
    0.5342611004737847,
    0.3505459023772192,
    0.21454583741454453,
];
const PA_A_RND: [f64; 6] = [
    1.5202762241108199,
    1.2751860889451079,
    1.0687715522961592,
    0.8847511020131834,
    0.7182772464541606,
    0.5327377861805684,
];
const PA_A_BIAS: [f64; 6] = [
    0.5202762241108199,
    0.2751860889451079,
    0.06877155229615917,
    -0.11524889798681659,
    -0.2817227535458394,
    -0.46726221381943156,
];
const PA_A_ADJ: [f64; 6] = [
    1.8757012140127365,
    1.6204317886558661,
    0.5402802917137625,
    0.6495099984606013,
    0.6322686559230586,
    0.6818080512339761,
];

const PA_B_RND: [f64; 6] = [
    1.726611194216871,
    1.4248947316409448,
    1.202107881131543,
    0.9872604909299066,
    0.8367765031192741,
    0.6608230139628977,
];
const PA_B_ADJ: [f64; 6] = [
    1.6693662439066854,
    1.4707231459600292,
    0.40694396287837853,
    0.5470006095438781,
    0.5137693992579451,
    0.5537228234516469,
];

const PA_D_RND: [f64; 2] = [1.265490626983912, 0.734509373016088];
const PA_D_ADJ: [f64; 2] = [1.734509373016088, 0.265490626983912];

fn assert_close(got: &[f64], want: &[f64], label: &str) {
    assert_eq!(got.len(), want.len(), "{label} length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < TOL, "{label}[{i}]: got {g}, want {w}");
    }
}

/// Kills M1 (uncentered correlation: nonzero column means shift every
/// eigenvalue) and M3 (bias without -1: bias/adjusted/retained all shift).
/// Every assert reads crate outputs from `parallel_analysis`.
#[test]
fn fixture_a_mean_benchmark() {
    let res = parallel_analysis(&PA_DATA, PA_N, PA_P, 50, 0, 42).expect("fixture A");
    assert_close(&res.eigenvalues, &PA_A_EV, "eigenvalues");
    assert_close(&res.random_eigenvalues, &PA_A_RND, "random_eigenvalues");
    assert_close(&res.bias, &PA_A_BIAS, "bias");
    assert_close(&res.adjusted_eigenvalues, &PA_A_ADJ, "adjusted");
    assert_eq!(res.retained, 2);
    // Sanity only (identity — holds for trace-preserving mutants too):
    let trace: f64 = res.eigenvalues.iter().sum();
    assert!((trace - PA_P as f64).abs() < 1e-8, "trace identity");
}

/// Kills M2 (quantile without interpolation): with 50 replicates the type-7
/// index h = 49 * 0.95 = 46.55 is fractional, so a ceil-indexed mutant
/// changes `random_eigenvalues`. Asserts read crate outputs.
#[test]
fn fixture_b_centile_95() {
    let res = parallel_analysis(&PA_DATA, PA_N, PA_P, 50, 95, 42).expect("fixture B");
    assert_close(&res.eigenvalues, &PA_A_EV, "eigenvalues (same data)");
    assert_close(&res.random_eigenvalues, &PA_B_RND, "random_eigenvalues");
    assert_close(&res.adjusted_eigenvalues, &PA_B_ADJ, "adjusted");
    assert_eq!(res.retained, 2);
    // The 95th-centile benchmark must dominate the mean benchmark (Glorfeld's
    // point); read from crate outputs of both runs.
    let mean_run = parallel_analysis(&PA_DATA, PA_N, PA_P, 50, 0, 42).expect("mean run");
    for (b, a) in res
        .random_eigenvalues
        .iter()
        .zip(&mean_run.random_eigenvalues)
    {
        assert!(b > a, "95th centile must exceed mean per position");
    }
}

/// Duplicate columns: observed correlation eigenvalues are analytically
/// {2, 0}; the rest of the pipeline is pinned deterministically.
#[test]
fn fixture_d_duplicate_columns() {
    let mut data = Vec::with_capacity(24);
    for i in 1..=12 {
        data.push(i as f64);
        data.push(i as f64);
    }
    let res = parallel_analysis(&data, 12, 2, 20, 0, 7).expect("fixture D");
    assert!((res.eigenvalues[0] - 2.0).abs() < TOL, "lambda_1 = 2");
    assert!(res.eigenvalues[1].abs() < TOL, "lambda_2 = 0");
    assert_close(&res.random_eigenvalues, &PA_D_RND, "random_eigenvalues");
    assert_close(&res.adjusted_eigenvalues, &PA_D_ADJ, "adjusted");
    assert_eq!(res.retained, 1);
}

/// Kills M4 (count(adj > 1) instead of the first-failure scan, paran.R lines
/// 250-267): the resurgent third value 1.4 must NOT be retained. Reads the
/// crate's `retained_count` directly.
#[test]
fn retention_scan_stops_at_first_failure() {
    assert_eq!(retained_count(&[1.2, 0.9, 1.4]), 1);
    assert_eq!(retained_count(&[0.9, 1.4]), 0);
    assert_eq!(retained_count(&[1.2, 1.1]), 2);
    // Boundary: adj == 1 exactly is NOT retained (paran: AdjEv <= 1 breaks).
    assert_eq!(retained_count(&[1.0, 2.0]), 0);
}

/// Kills M5 (missing descending sort after Jacobi): this matrix's Jacobi
/// diagonal converges in ascending-ish order; eigenvalues are analytically
/// {1.9, 1.0, 0.1} (2x2 block [[1, .9], [.9, 1]] plus isolated 1). Reads the
/// crate eigensolver output.
#[test]
fn eigen_order_anchor() {
    #[rustfmt::skip]
    let m = [
        1.0, 0.0, 0.9,
        0.0, 1.0, 0.0,
        0.9, 0.0, 1.0,
    ];
    let ev = symmetric_eigenvalues_desc(&m, 3).expect("jacobi");
    assert_close(&ev, &[1.9, 1.0, 0.1], "eigenvalues descending");
}

#[test]
fn rejections() {
    let ok = [1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 4.5, 5.5, 7.0, 6.5, 8.0];
    assert!(parallel_analysis(&ok, 6, 2, 10, 0, 1).is_ok());
    // Too few persons / items.
    assert!(parallel_analysis(&ok[..4], 2, 2, 10, 0, 1).is_err());
    assert!(parallel_analysis(&ok[..6], 6, 1, 10, 0, 1).is_err());
    // Zero iterations, out-of-range centile.
    assert!(parallel_analysis(&ok, 6, 2, 0, 0, 1).is_err());
    assert!(parallel_analysis(&ok, 6, 2, 10, 100, 1).is_err());
    // Non-finite cell.
    let mut nan = ok;
    nan[3] = f64::NAN;
    assert!(parallel_analysis(&nan, 6, 2, 10, 0, 1).is_err());
    // Constant column (zero variance).
    let constant = [1.0, 2.0, 1.0, 3.0, 1.0, 5.0, 1.0, 4.5, 1.0, 7.0, 1.0, 8.0];
    let err = parallel_analysis(&constant, 6, 2, 10, 0, 1).unwrap_err();
    assert!(err.contains("zero variance"), "got: {err}");
    // Length mismatch.
    assert!(parallel_analysis(&ok, 5, 2, 10, 0, 1).is_err());
}

/// Monte-Carlo consistency check (>= 500 reps, ignored by default): for a
/// strong 2-factor design, parallel analysis should retain exactly 2
/// components in at least 90% of replications (threshold deliberately below
/// the empirical rate to be binomial-safe; empirically the rate is ~100%).
/// Run with: cargo test -p mlsirm-core mc_two_factor_retention -- --ignored
#[test]
#[ignore]
fn mc_two_factor_retention() {
    let n = 200;
    let p = 10;
    let mut hits = 0usize;
    let reps = 500usize;
    for rep in 0..reps {
        // Generate 2-factor data with the same LCG idiom, offset seed stream.
        let mut state = 1_000_003_u64.wrapping_mul(rep as u64 + 1) | 1;
        let uniform = |st: &mut u64| {
            *st = st
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (*st >> 11) as f64 / (1u64 << 53) as f64
        };
        let normal = |st: &mut u64| {
            let u1 = uniform(st).max(1e-12);
            let u2 = uniform(st);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };
        let mut data = vec![0.0_f64; n * p];
        for row in 0..n {
            let f1 = normal(&mut state);
            let f2 = normal(&mut state);
            for j in 0..p {
                let loading_f = if j < 5 { f1 } else { f2 };
                let loading = if j % 5 < 3 { 0.7 } else { 0.6 };
                data[row * p + j] = loading * loading_f + 0.5 * normal(&mut state);
            }
        }
        let res =
            parallel_analysis(&data, n, p, 100, 0, 42 + rep as u64).expect("MC parallel analysis");
        if res.retained == 2 {
            hits += 1;
        }
    }
    let rate = hits as f64 / reps as f64;
    assert!(
        rate >= 0.90,
        "2-factor retention rate {rate} below 0.90 ({hits}/{reps})"
    );
}

/// Impl-review regressions: finite-but-huge data must Err (not silently
/// return a bogus identity correlation), and n_iterations * n_items must
/// use checked arithmetic. Both asserts read the crate Result.
/// Mutations killed: removing the is_finite guards in correlation_matrix
/// (first assert flips to Ok), removing checked_mul (second panics in
/// release wrap-around).
#[test]
fn overflow_inputs_rejected() {
    let big = 1e308_f64;
    let data = [big, big, big, -big, -big, big, -big, -big];
    assert!(parallel_analysis(&data, 4, 2, 5, 0, 7).is_err());

    let ok = [1.0, 2.0, 2.0, 1.0, 3.0, 5.0, 4.0, 4.5];
    assert!(parallel_analysis(&ok, 4, 2, usize::MAX, 0, 7).is_err());
}
