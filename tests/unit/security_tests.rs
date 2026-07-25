// Wollack-style omega answer-copying tests.
//
// Oracle: independent Python computation from the adversarial spec review
// (fixture and 17-digit values pinned there). Every assert reads values
// returned by the crate (OmegaResult fields); no test-local recomputation of
// the statistic. Disclosed limitation: p_value uses the crate's Numerical
// Recipes erfc (|error| < 1.2e-7), so p anchors carry 5e-7 absolute
// tolerance while omega/E/V carry 1e-12.

use super::*;

fn fixture() -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    let probs: Vec<f64> = vec![
        0.05, 0.10, 0.70, 0.10, 0.05, //
        0.20, 0.20, 0.20, 0.20, 0.20, //
        0.60, 0.15, 0.10, 0.10, 0.05, //
        0.12, 0.38, 0.25, 0.15, 0.10, //
        0.30, 0.25, 0.20, 0.15, 0.10, //
        0.08, 0.12, 0.16, 0.24, 0.40, //
        0.45, 0.05, 0.25, 0.15, 0.10, //
        0.11, 0.22, 0.33, 0.22, 0.12, //
        0.18, 0.32, 0.22, 0.18, 0.10, //
        0.07, 0.14, 0.21, 0.28, 0.30, //
    ];
    let source = vec![2, 3, 0, 1, 4, 4, 2, 1, 0, 3];
    let copier = vec![2, 1, 0, 1, 3, 4, 0, 1, 2, 3];
    (copier, source, probs)
}

/// Asserts read: OmegaResult.{observed_matches, expected_matches, variance,
/// omega, p_value} returned by the crate on the pinned fixture. Killed by:
/// dividing by V instead of sqrt(V) (omega -> 1.4278889537661232), indexing
/// probabilities by the copier's responses instead of the source's
/// (E -> 3.6, omega -> 1.6947698777903752), two-sided p
/// (p -> 0.05001304885862598), counting only key-correct matches
/// (h -> 4), continuity correction -0.5 (omega -> 1.5955675373451383),
/// dropped upper-tail direction, wrong row indexing.
#[test]
fn omega_pinned_oracle() {
    let (copier, source, probs) = fixture();
    let r = wollack_omega(&copier, &source, &probs, 5).unwrap();
    let tol = 1e-12;
    assert_eq!(r.observed_matches, 6);
    assert!(
        (r.expected_matches - 3.3100000000000001).abs() < tol,
        "E {}",
        r.expected_matches
    );
    assert!(
        (r.variance - 1.8839000000000001).abs() < tol,
        "V {}",
        r.variance
    );
    assert!(
        (r.omega - 1.9598523632230238).abs() < tol,
        "omega {}",
        r.omega
    );
    assert!(
        (r.p_value - 0.02500652442931299).abs() < 5e-7,
        "p {}",
        r.p_value
    );
}

/// Structural anchor: appending one extra item whose source-option
/// probability is p changes E by exactly p and V by exactly p(1-p), read
/// back from crate outputs on the extended fixture. Killed by: accumulation
/// bugs, wrong row stride (n_options mis-multiplication), off-by-one in the
/// item loop.
#[test]
fn omega_incremental_item_consistency() {
    let (mut copier, mut source, mut probs) = fixture();
    let base = wollack_omega(&copier, &source, &probs, 5).unwrap();
    // New item: distribution [0.5, 0.2, 0.1, 0.1, 0.1], source picks 0,
    // copier picks 1 (no match).
    probs.extend_from_slice(&[0.5, 0.2, 0.1, 0.1, 0.1]);
    source.push(0);
    copier.push(1);
    let ext = wollack_omega(&copier, &source, &probs, 5).unwrap();
    let tol = 1e-12;
    assert_eq!(ext.observed_matches, base.observed_matches);
    assert!(
        (ext.expected_matches - (base.expected_matches + 0.5)).abs() < tol,
        "E ext {}",
        ext.expected_matches
    );
    assert!(
        (ext.variance - (base.variance + 0.25)).abs() < tol,
        "V ext {}",
        ext.variance
    );
    // Statistic must move DOWN: E grew, h did not.
    assert!(
        ext.omega < base.omega,
        "omega {} !< {}",
        ext.omega,
        base.omega
    );
}

/// Error paths. Asserts read: Err strings returned by the crate. Killed by:
/// dropped validation (out-of-range options, row-sum check, length
/// mismatches, zero variance, empty input).
#[test]
fn omega_error_paths() {
    let (copier, source, probs) = fixture();
    assert!(wollack_omega(&[], &[], &[], 5).is_err());
    assert!(wollack_omega(&copier, &source[..9], &probs, 5).is_err());
    assert!(wollack_omega(&copier, &source, &probs[..49], 5).is_err());
    assert!(wollack_omega(&copier, &source, &probs, 0).is_err());
    let mut bad = copier.clone();
    bad[3] = 5;
    assert!(wollack_omega(&bad, &source, &probs, 5).is_err());
    let mut bad_src = source.clone();
    bad_src[0] = 9;
    assert!(wollack_omega(&copier, &bad_src, &probs, 5).is_err());
    let mut bad_p = probs.clone();
    bad_p[0] = 0.5; // row 0 sums to 1.45
    assert!(wollack_omega(&copier, &source, &bad_p, 5).is_err());
    let mut neg_p = probs.clone();
    neg_p[0] = -0.05;
    neg_p[1] = 0.20;
    assert!(wollack_omega(&copier, &source, &neg_p, 5).is_err());
    let mut nan_p = probs.clone();
    nan_p[2] = f64::NAN;
    assert!(wollack_omega(&copier, &source, &nan_p, 5).is_err());
    // Degenerate: all source-option probabilities exactly 1 -> V = 0.
    let det_probs = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0];
    let r = wollack_omega(&[0, 0], &[0, 0], &det_probs, 4);
    assert!(r.is_err(), "expected zero-variance error");
}

/// Deterministic LCG for the Monte Carlo harness (test-local; matches the
/// crate's private per-module LCG pattern).
struct TestLcg(u64);
impl TestLcg {
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// Monte Carlo null calibration and power (500 reps). Under the null the
/// copier answers independently from its own row distributions, so
/// P(reject at alpha=0.05) should be near 0.05 (normal approximation:
/// allow < 0.10); under 40% copying the test should reject often.
/// Asserts read: OmegaResult.p_value from the crate per replication.
#[test]
#[ignore]
fn monte_carlo_omega_size_and_power() {
    let n_items = 40;
    let n_options = 4;
    let reps = 500;
    let mut rng = TestLcg(20260726);
    // Fixed item distributions (moderately informative).
    let mut probs = Vec::with_capacity(n_items * n_options);
    for i in 0..n_items {
        let a = 0.4 + 0.4 * ((i % 5) as f64) / 4.0; // dominant-option prob
        let rest = (1.0 - a) / (n_options as f64 - 1.0);
        for k in 0..n_options {
            probs.push(if k == i % n_options { a } else { rest });
        }
    }
    let draw = |rng: &mut TestLcg, row: &[f64]| -> usize {
        let u = rng.next_f64();
        let mut c = 0.0;
        for (k, &p) in row.iter().enumerate() {
            c += p;
            if u < c {
                return k;
            }
        }
        row.len() - 1
    };
    let mut null_rej = 0usize;
    let mut alt_rej = 0usize;
    for _ in 0..reps {
        let mut source = Vec::with_capacity(n_items);
        let mut copier_null = Vec::with_capacity(n_items);
        let mut copier_alt = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let row = &probs[i * n_options..(i + 1) * n_options];
            let s = draw(&mut rng, row);
            let c = draw(&mut rng, row);
            source.push(s);
            copier_null.push(c);
            // Alternative: copy the source on 40% of items.
            copier_alt.push(if rng.next_f64() < 0.4 { s } else { c });
        }
        let rn = wollack_omega(&copier_null, &source, &probs, n_options).unwrap();
        if rn.p_value < 0.05 {
            null_rej += 1;
        }
        let ra = wollack_omega(&copier_alt, &source, &probs, n_options).unwrap();
        if ra.p_value < 0.05 {
            alt_rej += 1;
        }
    }
    let size = null_rej as f64 / reps as f64;
    let power = alt_rej as f64 / reps as f64;
    assert!(size < 0.10, "empirical size {}", size);
    assert!(power > 0.8, "empirical power {}", power);
}
