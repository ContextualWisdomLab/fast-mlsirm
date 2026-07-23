//! Tests for Mokken scale analysis (`mlsirm_core::mokken`).
//!
//! Every assert reads values returned by the crate (`MokkenH` fields or the
//! `aisp` label vector). Each test names the crate value it reads and a
//! mutant it kills.

use super::{aisp, coef_h, normal_upper_quantile};

/// Reads: `normal_upper_quantile` directly against published anchors
/// Phi^-1(0.95) = 1.6448536..., Phi^-1(0.999) = 3.0902323... .
/// Kills: sign/branch flips in the Acklam approximation (one such flip was
/// caught by `aisp_z_gate_blocks_insignificant_pair` during development).
#[test]
fn normal_quantile_matches_published_anchors() {
    assert!((normal_upper_quantile(0.05) - 1.6448536269514722).abs() < 1e-8);
    assert!((normal_upper_quantile(0.001) - 3.090232306167813).abs() < 1e-8);
    assert!((normal_upper_quantile(0.5)).abs() < 1e-8);
}

/// Deterministic xorshift for simulation without external deps.
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.max(1))
    }
    fn next_f64(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Standard normal via Box-Muller.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Simulate Rasch data: P(X=1) = logistic(theta - b).
fn simulate_rasch(rng: &mut Rng, n: usize, bs: &[f64], theta_scale: f64) -> Vec<i64> {
    let j = bs.len();
    let mut x = vec![0i64; n * j];
    for p in 0..n {
        let th = rng.next_normal() * theta_scale;
        for (i, &b) in bs.iter().enumerate() {
            let pr = 1.0 / (1.0 + (-(th - b)).exp());
            x[p * j + i] = if rng.next_f64() < pr { 1 } else { 0 };
        }
    }
    x
}

/// Brute-force oracle: two-pass covariance and sorted-column covariance,
/// computed with a DIFFERENT code path (per-pair, f64 accumulation in a
/// different order) than the crate's matrix construction.
fn oracle_pair(x: &[i64], n: usize, j: usize, a: usize, b: usize) -> (f64, f64) {
    let col = |it: usize| -> Vec<f64> { (0..n).map(|p| x[p * j + it] as f64).collect() };
    let (ca, cb) = (col(a), col(b));
    let (ma, mb) = (
        ca.iter().sum::<f64>() / n as f64,
        cb.iter().sum::<f64>() / n as f64,
    );
    let cov = (0..n).map(|p| (ca[p] - ma) * (cb[p] - mb)).sum::<f64>() / (n as f64 - 1.0);
    let mut sa = ca.clone();
    let mut sb = cb.clone();
    sa.sort_by(|u, v| u.partial_cmp(v).unwrap());
    sb.sort_by(|u, v| u.partial_cmp(v).unwrap());
    let cmx = (0..n).map(|p| (sa[p] - ma) * (sb[p] - mb)).sum::<f64>() / (n as f64 - 1.0);
    (cov, cmx)
}

/// Reads: `MokkenH::hij`, `hi`, `h` on random polytomous data, compared to a
/// brute-force oracle computed by an independent path.
/// Kills: any algebra mutant in `pairwise`/`h_subset` (wrong denominator,
/// mean subtraction, sorted-column pairing, aggregation order).
#[test]
fn coefficients_match_brute_force_oracle() {
    let mut rng = Rng::new(42);
    let (n, j) = (60, 4);
    // polytomous 0..=3 with item-varying marginals
    let mut x = vec![0i64; n * j];
    for p in 0..n {
        let th = rng.next_normal();
        for i in 0..j {
            let mut score = 0i64;
            for k in 0..3 {
                let cut = -1.0 + i as f64 * 0.4 + k as f64 * 0.8;
                if th + rng.next_normal() * 0.7 > cut {
                    score += 1;
                }
            }
            x[p * j + i] = score;
        }
    }
    let res = coef_h(&x, n, j).expect("fit");
    let mut num_tot = 0.0;
    let mut den_tot = 0.0;
    for a in 0..j {
        let mut num_i = 0.0;
        let mut den_i = 0.0;
        for b in 0..j {
            if a == b {
                assert!(res.hij[a * j + b].is_nan());
                continue;
            }
            let (cov, cmx) = oracle_pair(&x, n, j, a, b);
            assert!(
                (res.hij[a * j + b] - cov / cmx).abs() < 1e-12,
                "Hij[{a},{b}] crate {} oracle {}",
                res.hij[a * j + b],
                cov / cmx
            );
            num_i += cov;
            den_i += cmx;
            if b > a {
                num_tot += cov;
                den_tot += cmx;
            }
        }
        assert!((res.hi[a] - num_i / den_i).abs() < 1e-12, "Hi[{a}]");
    }
    assert!((res.h - num_tot / den_tot).abs() < 1e-12, "H");
}

/// Reads: `MokkenH::h` and `hij` on a perfect Guttman scalogram.
/// Kills: covmax mutants — any error in the sorted-column max covariance
/// breaks the exact H = 1 identity (for a nested dichotomous scalogram every
/// observed pair is already comonotone, so S_ij = Smax_ij).
#[test]
fn perfect_guttman_scalogram_has_h_one() {
    // 5 persons x 3 items, nested pattern
    let x = vec![
        0, 0, 0, //
        1, 0, 0, //
        1, 1, 0, //
        1, 1, 1, //
        1, 1, 1, //
    ];
    let res = coef_h(&x, 5, 3).expect("fit");
    assert!((res.h - 1.0).abs() < 1e-12, "H = {}", res.h);
    for a in 0..3 {
        for b in 0..3 {
            if a != b {
                assert!((res.hij[a * 3 + b] - 1.0).abs() < 1e-12);
            }
        }
    }
}

/// Reads: `MokkenH::zij`, `z` on a hand-computed 2-item fixture
/// (X = [0,0,0,1,1,1], Y = [0,0,1,0,1,1], N = 6); the exact hand derivation
/// is in the test body.
/// Kills: sqrt(N-1) and variance-product mutants in the Z formula.
#[test]
fn z_statistic_matches_hand_computation() {
    let x = vec![
        0, 0, //
        0, 0, //
        0, 1, //
        1, 0, //
        1, 1, //
        1, 1, //
    ];
    let res = coef_h(&x, 6, 2).expect("fit");
    // hand: means .5/.5; centered cross products:
    // (-.5)(-.5)*2 + (-.5)(.5) + (.5)(-.5) + (.5)(.5)*2 = .5 - .5 + .5 = 0.5
    // S_xy = 0.5/5 = 0.1 ; s_xx = s_yy = (6*.25)/5 = 0.3
    // Smax: sorted-sorted = comonotone = 1.5/5 = 0.3 -> Hij = 1/3
    // Zij = 0.1*sqrt(5)/sqrt(0.09) = 0.1*2.23606.../0.3 = 0.745355...
    let expect_z = 0.1 * 5f64.sqrt() / 0.3;
    assert!((res.hij[1] - 1.0 / 3.0).abs() < 1e-12, "Hij = {}", res.hij[1]);
    assert!((res.zij[1] - expect_z).abs() < 1e-12, "Zij = {}", res.zij[1]);
    // total Z for 2 items equals Zij
    assert!((res.z - expect_z).abs() < 1e-12);
}

/// Reads: `aisp` labels on the same fixture: Hij = 1/3 > c = 0.3 but
/// Zij ~ 0.745 < z_crit(0.05) = 1.645, so NO scale may form.
/// Kills: deleting the start-pair Z significance gate (mutant seeds a scale
/// because Hij exceeds c).
#[test]
fn aisp_z_gate_blocks_insignificant_pair() {
    let x = vec![
        0, 0, //
        0, 0, //
        0, 1, //
        1, 0, //
        1, 1, //
        1, 1, //
    ];
    let labels = aisp(&x, 6, 2, 0.3, 0.05).expect("aisp");
    assert_eq!(labels, vec![0, 0], "Z-gate must block the scale");
}

/// Reads: `aisp` labels plus `MokkenH::hij`/`hi` on a hand-constructed 80x3
/// contingency design (profile counts: 19x(1,1,1), 11x(1,1,0), 10x(1,0,0),
/// 10x(0,1,1), 11x(0,0,1), 19x(0,0,0); all marginals 0.5). By construction
/// H01 = 0.5 (start pair), H12 = 0.45, H02 = -0.05, and candidate item 2
/// passes every other gate at c = 0.15: Hi(2) = 0.2 >= c, Zi(2) ~ 2.51 >
/// z_crit, augmented H = 0.3 >= c. Only the negative-Hij (Criterion 1)
/// exclusion keeps it out.
/// Kills: removing the `hij >= 0` candidate filter in the add loop (the
/// mutant then admits item 2, flipping labels to [1,1,1]).
#[test]
fn aisp_excludes_candidate_with_negative_hij() {
    let profiles: [([i64; 3], usize); 6] = [
        ([1, 1, 1], 19),
        ([1, 1, 0], 11),
        ([1, 0, 0], 10),
        ([0, 1, 1], 10),
        ([0, 0, 1], 11),
        ([0, 0, 0], 19),
    ];
    let mut x = Vec::with_capacity(80 * 3);
    for (row, count) in profiles {
        for _ in 0..count {
            x.extend_from_slice(&row);
        }
    }
    let res = coef_h(&x, 80, 3).expect("fit");
    // verify the construction via crate values: item 2 negative with item 0
    // yet passes the Hi gate at c = 0.15
    assert!(res.hij[2 * 3] < 0.0, "Hij(2,0) = {}", res.hij[2 * 3]);
    assert!((res.hij[2 * 3] - (-0.05)).abs() < 1e-12);
    assert!((res.hi[2] - 0.2).abs() < 1e-12, "Hi(2) = {}", res.hi[2]);
    assert!((res.hij[1] - 0.5).abs() < 1e-12, "Hij(0,1) = {}", res.hij[1]);
    let labels = aisp(&x, 80, 3, 0.15, 0.05).expect("aisp");
    assert_eq!(labels, vec![1, 1, 0], "Criterion 1 must exclude item 2");
}

/// Reads: `aisp` labels on a two-cluster simulation (two independent Rasch
/// dimensions). AISP at c = 0.3 must recover the two clusters exactly.
/// Kills: selection-logic mutants (wrong argmax, wrong exclusion of previous
/// scales, missing multi-scale restart).
#[test]
fn aisp_recovers_two_clusters() {
    let mut rng = Rng::new(2013);
    let n = 1500;
    let bs = [-0.8, -0.3, 0.3, 0.8];
    // cluster A: items 0..4 driven by theta1; cluster B: items 4..8 by theta2
    let j = 8;
    let mut x = vec![0i64; n * j];
    for p in 0..n {
        let t1 = rng.next_normal() * 1.6;
        let t2 = rng.next_normal() * 1.6;
        for (i, &b) in bs.iter().enumerate() {
            let pr1 = 1.0 / (1.0 + (-(t1 - b)).exp());
            let pr2 = 1.0 / (1.0 + (-(t2 - b)).exp());
            x[p * j + i] = if rng.next_f64() < pr1 { 1 } else { 0 };
            x[p * j + 4 + i] = if rng.next_f64() < pr2 { 1 } else { 0 };
        }
    }
    let labels = aisp(&x, n, j, 0.3, 0.05).expect("aisp");
    let first = labels[0];
    let second = labels[4];
    assert!(first > 0 && second > 0 && first != second, "labels = {labels:?}");
    assert!(labels[..4].iter().all(|&l| l == first), "{labels:?}");
    assert!(labels[4..].iter().all(|&l| l == second), "{labels:?}");
}

/// Reads: `MokkenH` fields for score-translation invariance: adding a
/// constant to every score of an item must not change any coefficient
/// (covariances are translation-invariant).
/// Kills: accidental use of raw (uncentered) moments.
#[test]
fn coefficients_invariant_to_score_translation() {
    let mut rng = Rng::new(99);
    let n = 200;
    let x = simulate_rasch(&mut rng, n, &[-0.5, 0.0, 0.5], 1.3);
    let mut shifted = x.clone();
    for p in 0..n {
        shifted[p * 3 + 1] += 3; // item 1 scored 3..4 instead of 0..1
    }
    let a = coef_h(&x, n, 3).expect("fit");
    let b = coef_h(&shifted, n, 3).expect("fit");
    assert!((a.h - b.h).abs() < 1e-12);
    for i in 0..3 {
        assert!((a.hi[i] - b.hi[i]).abs() < 1e-12);
        assert!((a.zi[i] - b.zi[i]).abs() < 1e-12);
    }
}

/// Reads: error `Result`s from both entry points.
/// Kills: deletion of the validation guards.
#[test]
fn rejects_bad_inputs() {
    let ok = vec![0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1];
    assert!(coef_h(&ok, 2, 2).is_err(), "n_persons < 3");
    assert!(coef_h(&ok[..4], 4, 2).is_err(), "length mismatch");
    assert!(coef_h(&[0, -1, 1, 0, 1, 1], 3, 2).is_err(), "negative score");
    assert!(coef_h(&[1, 0, 1, 1, 1, 0], 3, 2).is_err(), "zero variance item 0");
    assert!(coef_h(&ok, 6, 1).is_err(), "single item");
    assert!(aisp(&ok, 6, 2, 1.2, 0.05).is_err(), "c out of range");
    assert!(aisp(&ok, 6, 2, 0.3, 0.0).is_err(), "alpha out of range");
}

/// Reads: `aisp` labels on an exact-tie design: X0 == X3 and X1 == X2
/// (identical columns, Hij = 1 for both pairs) with the two blocks exactly
/// uncorrelated (balanced half-split vs alternating pattern gives sample
/// cov = 0). mokken's eps tie-break (search.normal.R: penalty row*1e-10 on
/// the LARGER member index) must pick pair {1,2} first, so labels are
/// [2, 1, 1, 2].
/// Kills: reverting to first-encountered lexicographic tie-breaking, which
/// would start with pair {0,3} and yield [1, 2, 2, 1].
#[test]
fn aisp_tie_break_matches_mokken_eps_rule() {
    let n = 40;
    let j = 4;
    let mut x = vec![0i64; n * j];
    for p in 0..n {
        let a = if p < 20 { 1 } else { 0 }; // half-split
        let b = (p % 2) as i64; // alternating; sample cov(a, b) = 0 exactly
        x[p * j] = a;
        x[p * j + 1] = b;
        x[p * j + 2] = b;
        x[p * j + 3] = a;
    }
    let labels = aisp(&x, n, j, 0.3, 0.05).expect("aisp");
    assert_eq!(labels, vec![2, 1, 1, 2], "eps tie-break must favor pair {{1,2}}");
}

/// Reads: `aisp` labels; independent items (no common trait) must all remain
/// unscalable at c = 0.3. Smoke check of overall gating (not attributed to a
/// single mutant; the Z-gate kill lives in `aisp_z_gate_blocks_insignificant_pair`).
#[test]
fn aisp_leaves_independent_items_unscaled() {
    let mut rng = Rng::new(5);
    let n = 500;
    let j = 5;
    let mut x = vec![0i64; n * j];
    for v in x.iter_mut() {
        *v = if rng.next_f64() < 0.5 { 1 } else { 0 };
    }
    let labels = aisp(&x, n, j, 0.3, 0.05).expect("aisp");
    assert_eq!(labels, vec![0; j], "labels = {labels:?}");
}

/// Monte Carlo: >= 500 replications of a unidimensional Rasch scale
/// (normal and skew-positive traits). Reads crate `h` and `aisp` labels.
/// Asserts distributional behavior: mean H within a plausible band and
/// one-scale full recovery in >= 95% of replications.
/// Limitations stated: this cannot pin exact constants; the algebra anchors
/// live in `coefficients_match_brute_force_oracle` and
/// `z_statistic_matches_hand_computation`.
#[test]
#[ignore]
fn monte_carlo_unidimensional_recovery() {
    let bs = [-1.0, -0.5, 0.0, 0.5, 1.0];
    let n = 500;
    for (label, skew) in [("normal", false), ("skew", true)] {
        let mut full = 0usize;
        let mut h_sum = 0.0;
        let reps = 500;
        for rep in 0..reps {
            let mut rng = Rng::new(1000 + rep as u64);
            let j = bs.len();
            let mut x = vec![0i64; n * j];
            for p in 0..n {
                let mut th = rng.next_normal();
                if skew {
                    // half-normal shifted: skewed positive trait
                    th = th.abs() * 1.2 - 0.9;
                }
                th *= 1.5;
                for (i, &b) in bs.iter().enumerate() {
                    let pr = 1.0 / (1.0 + (-(th - b)).exp());
                    x[p * j + i] = if rng.next_f64() < pr { 1 } else { 0 };
                }
            }
            let res = coef_h(&x, n, bs.len()).expect("fit");
            h_sum += res.h;
            let labels = aisp(&x, n, bs.len(), 0.3, 0.05).expect("aisp");
            if labels.iter().all(|&l| l == 1) {
                full += 1;
            }
        }
        let mean_h = h_sum / reps as f64;
        assert!(
            mean_h > 0.35 && mean_h < 0.75,
            "{label}: mean H = {mean_h}"
        );
        assert!(
            full as f64 / reps as f64 >= 0.95,
            "{label}: full-recovery rate = {}",
            full as f64 / reps as f64
        );
    }
}
