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
    /// Standard normal via Box-Muller (test-local; used by the MC harness).
    fn normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

// ---------------------------------------------------------------------------
// GBT (generalized binomial test) tail kernel
// ---------------------------------------------------------------------------

/// Fixture from files/gbt_spec.md: 10 items; copier [1,0,1,1,0,1,0,0,1,1]
/// vs source [1,0,0,1,0,1,1,0,1,0] gives matches below with obs = 7.
fn gbt_fixture() -> (Vec<f64>, Vec<f64>) {
    let matches = vec![1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let probs = vec![0.62, 0.55, 0.48, 0.71, 0.52, 0.66, 0.43, 0.58, 0.73, 0.49];
    (matches, probs)
}

/// Pinned oracle: exact reference computed independently with Python
/// fractions.Fraction on the exact binary float values, implementing the
/// aberrance compute_GBT recursion verbatim (verified again by the
/// adversarial spec review's own script). Asserts read: gbt()'s
/// observed_matches, match_dist and p_value (crate outputs).
/// Killed by: M1 exclusive tail (0.1277605867075642), M2 lower tail
/// (0.8722394132924358), M3 duplicate item 0 i.e. probs [p0]+p
/// (0.47925477021800067), M4 tail from obs-1 (0.5754779926114737),
/// M5 swapped convolution mixing q*f[j-1]+p*f[j] (tail 0.06957009419363154
/// and pmf reversed — the pinned pmf entries kill it too).
#[test]
fn gbt_pinned_oracle() {
    let (m, p) = gbt_fixture();
    let r = gbt(&m, &p).unwrap();
    assert_eq!(r.observed_matches, 7);
    assert!(
        (r.p_value - 0.32225898631286054).abs() < 1e-12,
        "p {}",
        r.p_value
    );
    assert_eq!(r.match_dist.len(), 11);
    let pmf = [
        0.00013873169507258882,
        0.0020878412745965764,
        0.013861663456144322,
        0.05348185776781805,
        0.1328495797358767,
        0.2221023334590181,
        0.25321900629861316,
        0.19449839960529633,
        0.09637293070413779,
        0.027829568425056288,
        0.0035580875783701245,
    ];
    for (k, &want) in pmf.iter().enumerate() {
        assert!(
            (r.match_dist[k] - want).abs() < 1e-12,
            "pmf[{}] = {} want {}",
            k,
            r.match_dist[k],
            want
        );
    }
    let total: f64 = r.match_dist.iter().sum();
    assert!((total - 1.0).abs() < 1e-12, "pmf total {}", total);
}

/// Structural invariants with exactly known values (all asserts read crate
/// outputs). (a) all 10 items match with probs 0.5: p = P(M >= 10) =
/// 2^-10 = 1/1024 exactly — kills tail off-by-one in either direction
/// (exclusive tail would be 0, tail from obs-1 would add mass).
/// (b) obs = 0: inclusive upper tail is 1 — kills exclusive-tail mutants
/// structurally. (c) deterministic probs [1,1,0]: M is a.s. 2, so
/// matches [1,1,0] (obs 2) gives p = 1 and matches [1,1,1] (obs 3) gives
/// p = 0 — kills mishandling of boundary probabilities.
#[test]
fn gbt_structural_invariants() {
    let r = gbt(&[1.0; 10], &[0.5; 10]).unwrap();
    assert_eq!(r.observed_matches, 10);
    assert!((r.p_value - 1.0 / 1024.0).abs() < 1e-15, "p {}", r.p_value);

    let r0 = gbt(&[0.0; 4], &[0.3, 0.9, 0.5, 0.1]).unwrap();
    assert_eq!(r0.observed_matches, 0);
    assert!((r0.p_value - 1.0).abs() < 1e-15, "p {}", r0.p_value);

    let rd = gbt(&[1.0, 1.0, 0.0], &[1.0, 1.0, 0.0]).unwrap();
    assert_eq!(rd.observed_matches, 2);
    assert!((rd.p_value - 1.0).abs() < 1e-15, "p {}", rd.p_value);
    assert!((rd.match_dist[2] - 1.0).abs() < 1e-15);

    let ri = gbt(&[1.0, 1.0, 1.0], &[1.0, 1.0, 0.0]).unwrap();
    assert_eq!(ri.observed_matches, 3);
    assert!(ri.p_value.abs() < 1e-15, "p {}", ri.p_value);
}

/// Error paths: every rejected input returns Err (asserts read gbt()'s
/// Result). Killed by removing any validation branch.
#[test]
fn gbt_error_paths() {
    assert!(gbt(&[], &[]).is_err());
    assert!(gbt(&[1.0, 0.0], &[0.5]).is_err());
    assert!(gbt(&[1.0, 0.5], &[0.5, 0.5]).is_err());
    assert!(gbt(&[1.0, 0.0], &[0.5, 1.5]).is_err());
    assert!(gbt(&[1.0, 0.0], &[0.5, -0.1]).is_err());
    assert!(gbt(&[1.0, 0.0], &[0.5, f64::NAN]).is_err());
    assert!(gbt(&[1.0, 0.0], &[0.5, f64::INFINITY]).is_err());
}

/// Monte Carlo size/power, 500 reps (ignored: heavy). Null: two independent
/// Rasch-like examinees; per-item match probability computed from the SAME
/// model that generates responses, so the GBT p-value is exact and the
/// rejection rate at alpha = .05 must be at most about nominal (discrete
/// conservative test). Alternative: copier copies the source's response on
/// a fixed 40% of items while match_probs stay model-implied. Asserts read:
/// gbt()'s p_value across replications.
#[test]
#[ignore]
fn monte_carlo_gbt_size_and_power() {
    let n_items = 40usize;
    let reps = 500usize;
    let alpha = 0.05f64;
    let mut rng = TestLcg(20260726);
    let mut null_rej = 0usize;
    let mut alt_rej = 0usize;
    for _ in 0..reps {
        // Item easiness in [-1.5, 1.5]; two independent abilities N(0,1).
        let mut b = vec![0.0f64; n_items];
        for bi in b.iter_mut() {
            *bi = -1.5 + 3.0 * rng.next_f64();
        }
        let t1 = rng.normal();
        let t2 = rng.normal();
        let p1: Vec<f64> = b
            .iter()
            .map(|&bi| 1.0 / (1.0 + (-(t1 - bi)).exp()))
            .collect();
        let p2: Vec<f64> = b
            .iter()
            .map(|&bi| 1.0 / (1.0 + (-(t2 - bi)).exp()))
            .collect();
        let x1: Vec<f64> = p1
            .iter()
            .map(|&p| if rng.next_f64() < p { 1.0 } else { 0.0 })
            .collect();
        let x2: Vec<f64> = p2
            .iter()
            .map(|&p| if rng.next_f64() < p { 1.0 } else { 0.0 })
            .collect();
        // Symmetric CopyDetect-style match probabilities (caller recipe).
        let mp: Vec<f64> = p1
            .iter()
            .zip(&p2)
            .map(|(&a, &c)| a * c + (1.0 - a) * (1.0 - c))
            .collect();
        let matches: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .map(|(&a, &c)| if a == c { 1.0 } else { 0.0 })
            .collect();
        if gbt(&matches, &mp).unwrap().p_value < alpha {
            null_rej += 1;
        }
        // Alternative: copy source's responses on the first 40% of items.
        let n_copy = (n_items as f64 * 0.4) as usize;
        let mut x1c = x1.clone();
        for i in 0..n_copy {
            x1c[i] = x2[i];
        }
        let matches_c: Vec<f64> = x1c
            .iter()
            .zip(&x2)
            .map(|(&a, &c)| if a == c { 1.0 } else { 0.0 })
            .collect();
        if gbt(&matches_c, &mp).unwrap().p_value < alpha {
            alt_rej += 1;
        }
    }
    let size = null_rej as f64 / reps as f64;
    let power = alt_rej as f64 / reps as f64;
    assert!(size < 0.08, "empirical size {}", size);
    assert!(power > 0.5, "empirical power {}", power);
}
