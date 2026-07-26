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

// ---------------------------------------------------------------------------
// K-index tests. Oracle: independent Python computation (math.comb exact
// binomial) from the adversarial spec review; fixture 10 persons x 10 items,
// copier = row 2, source = row 7.
// ---------------------------------------------------------------------------

fn k_fixture() -> Vec<f64> {
    let rows: [[u8; 10]; 10] = [
        [1, 1, 0, 1, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 1, 1, 0, 1, 1],
        [0, 1, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 1, 1, 0, 0, 1],
        [1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
    ];
    rows.iter().flatten().map(|&x| x as f64).collect()
}

/// Asserts read: KIndexResult.{wc, ws, m, subgroup, emp_agg, p, k_index}
/// returned by the crate on the pinned fixture. Killed by: p from the
/// copier pair only (K -> 0.66304000000000007), lower tail
/// (K -> 0.43768493827160498), off-by-one upper tail P(Bin >= m+1)
/// (K -> 0.56231506172839507), matching correct (1,1) pairs
/// (K -> 0.86831275720164613), subgroup keyed on the source's ws
/// (K -> 0.96921999999999997) — all differ from the true K by > 1e-6.
#[test]
fn k_index_pinned_oracle() {
    let x = k_fixture();
    let r = k_index(&x, 10, 10, 2, 7).unwrap();
    assert_eq!(r.wc, 6);
    assert_eq!(r.ws, 5);
    assert_eq!(r.m, 2);
    assert_eq!(r.subgroup, vec![1, 2, 5]);
    assert_eq!(r.emp_agg, vec![3, 2, 3]);
    assert!((r.p - 0.53333333333333333).abs() < 1e-12, "p {}", r.p);
    assert!(
        (r.k_index - 0.85139489711934158).abs() < 1e-12,
        "K {}",
        r.k_index
    );
}

/// Subgroup membership: the copier is always in its own subgroup, and the
/// source IS included when its number-incorrect equals the copier's
/// (CopyDetect convention, paper-source-excluded convention NOT applied).
/// Asserts read: KIndexResult.subgroup from the crate.
#[test]
fn k_index_subgroup_includes_copier_and_matching_source() {
    let x = k_fixture();
    let base = k_index(&x, 10, 10, 2, 7).unwrap();
    assert!(base.subgroup.contains(&2), "copier not in subgroup");
    // Row 7 (ws = 5) is not in the wc = 6 subgroup here.
    assert!(!base.subgroup.contains(&7));
    // Make row 0 the source: it has 5 incorrect. Copier row 7 has wc = 5,
    // so the source (row 0) shares the number-incorrect score and must be
    // included in the subgroup.
    let r = k_index(&x, 10, 10, 7, 0).unwrap();
    assert_eq!(r.wc, 5);
    assert!(r.subgroup.contains(&7), "copier not in subgroup");
    assert!(
        r.subgroup.contains(&0),
        "source with matching wc must be included (CopyDetect convention)"
    );
}

/// Degenerate branches read back from crate outputs: m == 0 -> K = 1;
/// p == 1 -> K = 1. Killed by inverted tails or wrong m handling.
#[test]
fn k_index_degenerate_branches() {
    // Copier all correct except item 9; source incorrect only at items 0-1;
    // no shared incorrect item -> m = 0 -> K = 1.
    let mut x = vec![1.0; 3 * 10];
    x[9] = 0.0; // person 0 (copier) misses item 9
    x[10] = 0.0; // person 1 (source) misses items 0, 1
    x[11] = 0.0;
    let r = k_index(&x, 3, 10, 0, 1).unwrap();
    assert_eq!(r.m, 0);
    assert_eq!(r.k_index, 1.0);
    // p == 1: every subgroup member incorrect exactly on the source's
    // incorrect items. Two persons, both incorrect on items 0..2 only.
    let mut y = vec![1.0; 2 * 6];
    for i in 0..3 {
        y[i] = 0.0;
        y[6 + i] = 0.0;
    }
    let r1 = k_index(&y, 2, 6, 0, 1).unwrap();
    assert_eq!(r1.ws, 3);
    assert_eq!(r1.m, 3);
    assert!((r1.p - 1.0).abs() < 1e-15, "p {}", r1.p);
    assert_eq!(r1.k_index, 1.0);
}

/// Error paths. Asserts read: Err values returned by the crate.
#[test]
fn k_index_error_paths() {
    let x = k_fixture();
    assert!(k_index(&x, 1, 10, 0, 0).is_err()); // n_persons < 2
    assert!(k_index(&x, 10, 0, 2, 7).is_err()); // n_items = 0
    assert!(k_index(&x[..99], 10, 10, 2, 7).is_err()); // bad length
    assert!(k_index(&x, 10, 10, 10, 7).is_err()); // copier out of range
    assert!(k_index(&x, 10, 10, 2, 10).is_err()); // source out of range
    assert!(k_index(&x, 10, 10, 3, 3).is_err()); // copier == source
    let mut bad = x.clone();
    bad[5] = 0.5;
    assert!(k_index(&bad, 10, 10, 2, 7).is_err()); // non-binary
    let mut nan = x.clone();
    nan[5] = f64::NAN;
    assert!(k_index(&nan, 10, 10, 2, 7).is_err()); // NaN
                                                   // ws == 0: source (row 4 modified) all correct.
    let mut allc = x.clone();
    for i in 0..10 {
        allc[4 * 10 + i] = 1.0;
    }
    assert!(k_index(&allc, 10, 10, 2, 4).is_err());
}

/// Monte Carlo null calibration and power (500 reps). Under the null all
/// examinees answer independently with per-item difficulty; K is discrete
/// and conservative, so the size at alpha = 0.05 must stay below 0.10.
/// Under copying (copier copies the source's full answer on 90% of items)
/// small K values should appear much more often. Asserts read:
/// KIndexResult.k_index from the crate per replication.
#[test]
#[ignore]
fn monte_carlo_k_index_size_and_power() {
    let n_persons = 200;
    let n_items = 40;
    let reps = 500;
    let mut rng = TestLcg(20260727);
    let mut null_rej = 0usize;
    let mut alt_rej = 0usize;
    let mut used = 0usize;
    for _ in 0..reps {
        // Item difficulties: P(correct) in [0.3, 0.8].
        let diffs: Vec<f64> = (0..n_items)
            .map(|i| 0.3 + 0.5 * (i as f64) / (n_items as f64 - 1.0))
            .collect();
        let mut x = vec![0.0f64; n_persons * n_items];
        for r in 0..n_persons {
            for i in 0..n_items {
                x[r * n_items + i] = if rng.next_f64() < diffs[i] { 1.0 } else { 0.0 };
            }
        }
        // Ensure source (row 1) has at least one incorrect answer.
        if (0..n_items).all(|i| x[n_items + i] == 1.0) {
            x[n_items] = 0.0;
        }
        let rn = k_index(&x, n_persons, n_items, 0, 1).unwrap();
        if rn.k_index < 0.05 {
            null_rej += 1;
        }
        // Alternative: copier copies the source's answer (correct or
        // incorrect) on 90% of items. Copying whole responses keeps the
        // copier's number-incorrect near the source's typical score, so the
        // subgroup stays populated (copying only incorrect answers inflates
        // wc into a sparse subgroup where the self-inclusion bias makes K
        // conservative -- a documented property of the CopyDetect k()).
        let mut y = x.clone();
        for i in 0..n_items {
            if rng.next_f64() < 0.9 {
                y[i] = y[n_items + i];
            }
        }
        let ra = k_index(&y, n_persons, n_items, 0, 1).unwrap();
        if ra.k_index < 0.05 {
            alt_rej += 1;
        }
        used += 1;
    }
    let size = null_rej as f64 / used as f64;
    let power = alt_rej as f64 / used as f64;
    assert!(size < 0.10, "empirical size {}", size);
    assert!(power > 0.5, "empirical power {}", power);
}

/// Regression for the impl-review finding: the linear-space recurrence
/// underflowed at extreme p / large n ((1-p)^1000 == 0), returning K = 0.
/// Exact reference computed with Python fractions.Fraction on the exact
/// binary value of 0.99: P(Bin(1000, 0.99) >= 990) = 0.58304080330109709.
/// Asserts read: binom_sf_ge (the crate's binomial kernel used by k_index).
/// Killed by: any complement/linear-space form that underflows (returns 0),
/// and by tail flips (lower tail = 0.475...).
#[test]
fn k_index_binomial_tail_extreme_p_regression() {
    let k = binom_sf_ge(1000, 0.99, 990);
    assert!((k - 0.58304080330109709).abs() < 1e-12, "tail {}", k);
    // Symmetric moderate case stays exact too.
    let k2 = binom_sf_ge(5, 0.53333333333333333, 2);
    assert!((k2 - 0.85139489711934158).abs() < 1e-12, "tail {}", k2);
}
