//! Tests for kernel-smoothing nonparametric IRT (`mlsirm_core::ksirt`).
//!
//! Every assert reads values returned by the crate (`KsirtResult` fields).
//! Each test names the crate value it reads and a mutant it kills.
//!
//! Known unkillable mutations (documented per the test discipline): all
//! three supported kernels are symmetric, so flipping the sign of the
//! Nadaraya-Watson argument `(grid - theta)/h` is an identity; likewise a
//! pure multiplicative kernel constant cancels in the NW normalization.
//! No anchor can exist for either — this is a property of the model, not a
//! test gap.

use super::{ksirt, KsirtKernel};

/// qnorm anchors (R `qnorm`, 16 digits) used as independent fixture
/// constants so expected values never route through the crate's quantile.
const QN_1_5: f64 = -0.8416212335729143; // Phi^-1(0.2)
const QN_2_5: f64 = -0.2533471031357997; // Phi^-1(0.4)
const QN_3_5: f64 = 0.2533471031357997; // Phi^-1(0.6)
const QN_4_5: f64 = 0.8416212335729143; // Phi^-1(0.8)

/// n=4, k=1 binary fixture: subjects 1 and 3 score 1, subjects 0 and 2
/// score 0. Totals [0,1,0,1] -> first-occurrence ranks [1,3,2,4] ->
/// theta [Phi^-1(.2), Phi^-1(.6), Phi^-1(.4), Phi^-1(.8)].
fn fixture_theta() -> [f64; 4] {
    [QN_1_5, QN_3_5, QN_2_5, QN_4_5]
}

fn fixture_x() -> Vec<Vec<f64>> {
    vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]]
}

/// Reads: `items[0].occ` (both option rows, all grid points).
/// Expected values recomputed here from hard-coded qnorm literals and the
/// Gaussian NW formula (never from crate outputs), so the assert compares
/// the crate curve to an independent hand derivation.
/// Kills: subject/theta index misalignment (swapping two subjects' theta),
/// wrong weight denominator, gaussian<->quadratic kernel swap, and rank
/// mis-ordering (theta enters the expected values).
#[test]
fn hand_fixture_occ_exact() {
    let res = ksirt(&fixture_x(), KsirtKernel::Gaussian, 3, Some(&[0.5])).unwrap();
    let theta = fixture_theta();
    let grid = [QN_1_5, 0.0, QN_4_5];
    for (s, &g) in grid.iter().enumerate() {
        let kw: Vec<f64> = theta
            .iter()
            .map(|&t| (-0.5 * ((g - t) / 0.5).powi(2)).exp())
            .collect();
        let denom: f64 = kw.iter().sum();
        let exp_p1 = (kw[1] + kw[3]) / denom; // subjects with score 1
        assert!(
            (res.items[0].occ[1][s] - exp_p1).abs() < 1e-9,
            "occ[1][{s}] = {} expected {exp_p1}",
            res.items[0].occ[1][s]
        );
        assert!((res.items[0].occ[0][s] - (1.0 - exp_p1)).abs() < 1e-9);
    }
    // asymmetry anchor: the curve is not flat, low grid point is low
    assert!(res.items[0].occ[1][0] < 0.10);
    assert!(res.items[0].occ[1][2] > 0.90);
}

/// Reads: `result.theta`.
/// Kills: `ties.method="first"` violations (tied totals must rank in
/// subject order), and the n+1 -> n denominator mutation (rank 4 of 4
/// would hit Phi^-1(1.0) = infinity instead of Phi^-1(0.8)).
#[test]
fn theta_rank_ties_first() {
    // totals: [2, 5, 2, 7] -> ties between subjects 0 and 2 broken by
    // original order: ranks [1, 3, 2, 4]
    let x = vec![
        vec![1.0, 1.0, 0.0],
        vec![2.0, 2.0, 1.0],
        vec![0.0, 1.0, 1.0],
        vec![3.0, 3.0, 1.0],
    ];
    let res = ksirt(&x, KsirtKernel::Gaussian, 5, None).unwrap();
    let expected = [QN_1_5, QN_3_5, QN_2_5, QN_4_5];
    for i in 0..4 {
        assert!(
            (res.theta[i] - expected[i]).abs() < 1e-8,
            "theta[{i}] = {} expected {}",
            res.theta[i],
            expected[i]
        );
    }
}

/// Reads: `result.grid`.
/// Kills: wrong endpoints (1/(n+1) vs 1/n), wrong point count, and
/// non-uniform spacing mutations.
#[test]
fn grid_endpoints_and_spacing() {
    let x = fixture_x();
    let res = ksirt(&x, KsirtKernel::Gaussian, 5, None).unwrap();
    assert_eq!(res.grid.len(), 5);
    assert!((res.grid[0] - QN_1_5).abs() < 1e-8);
    assert!((res.grid[4] - QN_4_5).abs() < 1e-8);
    let step = (QN_4_5 - QN_1_5) / 4.0;
    for s in 1..5 {
        // 1e-8: the crate grid endpoints come from the Acklam quantile
        // (|error| < 1.15e-9), which propagates into the spacing.
        assert!((res.grid[s] - res.grid[s - 1] - step).abs() < 1e-8);
    }
}

/// Reads: `result.bandwidth`.
/// Kills: mutations of the Silverman constant (1.06) or exponent (-1/5),
/// verified against 1.06 * 100^(-0.2) = 0.4219936007867... computed by hand.
#[test]
fn silverman_bandwidth_value() {
    let mut x = Vec::new();
    for i in 0..100 {
        x.push(vec![(i % 2) as f64, ((i / 2) % 2) as f64]);
    }
    let res = ksirt(&x, KsirtKernel::Gaussian, 11, None).unwrap();
    let expected = 1.06 * 100f64.powf(-0.2);
    assert!((expected - 0.4219936007867).abs() < 1e-9); // pin the hand value
    assert_eq!(res.bandwidth.len(), 2);
    for &h in &res.bandwidth {
        assert!((h - expected).abs() < 1e-12);
    }
}

/// Reads: `items[*].occ` column sums.
/// Weak near-identity on its own (documented); paired with the exact-value
/// tests above. Still kills: per-option denominators (normalizing each
/// option row separately would break cross-option coherence when combined
/// with `hand_fixture_occ_exact`), and dropped options (a missing row makes
/// sums fall short of 1).
#[test]
fn occ_rows_sum_to_one() {
    let x = vec![
        vec![0.0, 2.0],
        vec![1.0, 0.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 2.0],
        vec![0.0, 0.0],
    ];
    let res = ksirt(&x, KsirtKernel::Gaussian, 7, None).unwrap();
    for item in &res.items {
        assert_eq!(item.occ.len(), 3);
        for s in 0..7 {
            let sum: f64 = item.occ.iter().map(|row| row[s]).sum();
            assert!((sum - 1.0).abs() < 1e-12, "column {s} sums to {sum}");
        }
    }
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
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

/// Simulate binary 2PL responses: P(X=1) = logistic(a*(theta - b)).
fn simulate_2pl(rng: &mut Rng, n: usize, a: &[f64], b: &[f64], skew: bool) -> Vec<Vec<f64>> {
    let k = a.len();
    let mut x = Vec::with_capacity(n);
    for _ in 0..n {
        let z = rng.next_normal();
        // shifted lognormal with mean approx 0 for the skewed condition
        let th = if skew { (0.5 * z).exp() - 1.1331 } else { z };
        let mut row = Vec::with_capacity(k);
        for j in 0..k {
            let p = 1.0 / (1.0 + (-a[j] * (th - b[j])).exp());
            row.push(if rng.next_f64() < p { 1.0 } else { 0.0 });
        }
        x.push(row);
    }
    x
}

/// Reads: `result.expected_total`.
/// Kills: theta/rank inversion (descending ranks would flip the curve) and
/// weight-matrix transposition (grid and subject axes swapped destroys the
/// monotone trend).
#[test]
fn expected_score_monotone_recovery() {
    let mut rng = Rng::new(20260213);
    let a = [1.2, 0.8, 1.5, 1.0, 0.9, 1.3, 1.1, 0.7, 1.4, 1.0];
    let b = [-1.0, -0.5, 0.0, 0.5, 1.0, -0.8, 0.3, 0.8, -0.2, 0.1];
    let x = simulate_2pl(&mut rng, 300, &a, &b, false);
    let res = ksirt(&x, KsirtKernel::Gaussian, 51, None).unwrap();
    let q = res.expected_total.len();
    // top decile of grid clearly above bottom decile
    assert!(
        res.expected_total[q - 3] > res.expected_total[2] + 2.0,
        "expected total not increasing: low {} high {}",
        res.expected_total[2],
        res.expected_total[q - 3]
    );
    // and grid itself is ascending (guards a reversed-grid mutant)
    assert!(res.grid[q - 1] > res.grid[0]);
}

/// Reads: `items[0].occ` under a tiny-bandwidth uniform kernel.
/// n=2: theta = [Phi^-1(1/3), Phi^-1(2/3)], grid (q=3) endpoints coincide
/// with the two thetas. With h=0.01 only the co-located subject is in
/// support at each endpoint, and no subject is in support at the middle.
/// Kills: removal of the zero-denominator fallback (0/0 -> NaN would fail
/// the middle-point zero assertions) and support-window widening.
/// Documented limit: the boundary mutation `|u| <= 1` -> `|u| < 1` is NOT
/// killed here (u=0 at the co-located point passes both); no fixture can
/// place a subject exactly on the support edge with irrational thetas.
#[test]
fn uniform_kernel_support() {
    let x = vec![vec![0.0], vec![1.0]];
    let res = ksirt(&x, KsirtKernel::Uniform, 3, Some(&[0.01])).unwrap();
    // endpoint 0: only subject 0 (score 0) in support
    assert!((res.items[0].occ[0][0] - 1.0).abs() < 1e-12);
    assert!(res.items[0].occ[1][0].abs() < 1e-12);
    // middle: nobody in support -> all-zero fallback, finite
    assert!(res.items[0].occ[0][1].abs() < 1e-12);
    assert!(res.items[0].occ[1][1].abs() < 1e-12);
    assert!(res.items[0].occ[0][1].is_finite());
    // endpoint 2: only subject 1 (score 1)
    assert!((res.items[0].occ[1][2] - 1.0).abs() < 1e-12);
}

/// Reads: `items[0].occ` under the quadratic kernel, h=1.5.
/// Expected values recomputed from qnorm literals with (1-u^2) truncation;
/// subject 3 falls outside the support at the low endpoint (|u| > 1), so a
/// gaussian<->quadratic swap or a dropped support check changes the value.
/// Kills: kernel dispatch mutations and support-truncation removal.
#[test]
fn quadratic_kernel_exact() {
    let res = ksirt(&fixture_x(), KsirtKernel::Quadratic, 3, Some(&[1.5])).unwrap();
    let theta = fixture_theta();
    let g = QN_1_5; // low endpoint
    let kw: Vec<f64> = theta
        .iter()
        .map(|&t| {
            let u = (g - t) / 1.5;
            if u.abs() <= 1.0 {
                1.0 - u * u
            } else {
                0.0
            }
        })
        .collect();
    assert!(kw[3] == 0.0, "fixture must exercise the out-of-support branch");
    let denom: f64 = kw.iter().sum();
    let expected = (kw[1] + kw[3]) / denom;
    assert!(
        (res.items[0].occ[1][0] - expected).abs() < 1e-9,
        "occ[1][0] = {} expected {expected}",
        res.items[0].occ[1][0]
    );
}

/// Reads: `items[*].expected` and `result.expected_total`.
/// Kills: option-score/probability pairing mutations (expected uses the
/// sorted option scores against their own occ rows) and per-item summation
/// errors in the total.
#[test]
fn expected_score_matches_occ_combination() {
    let x = vec![
        vec![0.0, 2.0],
        vec![1.0, 0.0],
        vec![2.0, 1.0],
        vec![1.0, 2.0],
        vec![2.0, 2.0],
        vec![0.0, 0.0],
    ];
    let res = ksirt(&x, KsirtKernel::Gaussian, 7, None).unwrap();
    for s in 0..7 {
        let mut total = 0.0;
        for item in &res.items {
            let mut e = 0.0;
            for (l, &opt) in item.options.iter().enumerate() {
                e += opt * item.occ[l][s];
            }
            assert!(
                (item.expected[s] - e).abs() < 1e-12,
                "expected[{s}] mismatch"
            );
            total += e;
        }
        assert!((res.expected_total[s] - total).abs() < 1e-12);
    }
}

/// Reads: the `Result` error branch for each documented rejection.
/// Kills: removal of any input validation guard.
#[test]
fn input_rejection() {
    let ok = fixture_x();
    assert!(ksirt(&ok[..1], KsirtKernel::Gaussian, 3, None).is_err()); // n < 2
    assert!(ksirt(&[vec![], vec![]], KsirtKernel::Gaussian, 3, None).is_err()); // k = 0
    assert!(
        ksirt(&[vec![1.0], vec![1.0, 2.0]], KsirtKernel::Gaussian, 3, None).is_err(),
        "ragged"
    );
    assert!(
        ksirt(&[vec![f64::NAN], vec![1.0]], KsirtKernel::Gaussian, 3, None).is_err(),
        "NaN"
    );
    assert!(ksirt(&ok, KsirtKernel::Gaussian, 1, None).is_err()); // q < 2
    assert!(ksirt(&ok, KsirtKernel::Gaussian, 3, Some(&[0.5, 0.5])).is_err()); // len
    assert!(ksirt(&ok, KsirtKernel::Gaussian, 3, Some(&[0.0])).is_err()); // h <= 0
    assert!(ksirt(&ok, KsirtKernel::Gaussian, 3, Some(&[f64::NAN])).is_err());
}

/// Monte Carlo ICC recovery, 500 replications, normal and skewed abilities.
/// Reads: `items[*].occ` (the score-1 row) against the true 2PL ICC on the
/// central grid, averaged over items and replications. Because theta is
/// rank-based on the normal metric, the skewed condition's oracle is the
/// ICC composed with the monotone generating map t(g) = exp(g/2) - 1.1331
/// (rank invariance: the estimate depends on abilities only through their
/// ranks, so it recovers P(X=1 | z = g) with z the normal driver).
/// Kills: gross formula errors (wrong theta metric, broken smoothing) that
/// the deterministic fixtures could miss at scale.
#[test]
#[ignore = "500-replication Monte Carlo; run explicitly"]
fn mc_2pl_recovery_500() {
    let a: Vec<f64> = (0..20).map(|j| 0.7 + 0.05 * j as f64).collect();
    let b: Vec<f64> = (0..20).map(|j| -1.5 + 0.15 * j as f64).collect();
    for &skew in &[false, true] {
        let mut sum_rmse = 0.0;
        let mut reps = 0usize;
        for rep in 0..500u64 {
            let mut rng = Rng::new(7_000_003 * (rep + 1) + skew as u64);
            let x = simulate_2pl(&mut rng, 500, &a, &b, skew);
            let res = ksirt(&x, KsirtKernel::Gaussian, 51, None).unwrap();
            let mut se_sum = 0.0;
            let mut cnt = 0usize;
            for (j, item) in res.items.iter().enumerate() {
                // score-1 row (options sorted ascending: [0,1])
                let row = &item.occ[item.options.len() - 1];
                for (s, &g) in res.grid.iter().enumerate() {
                    if g.abs() <= 1.5 {
                        // effective ability at normal-metric grid point g
                        let t = if skew { (0.5 * g).exp() - 1.1331 } else { g };
                        let p = 1.0 / (1.0 + (-a[j] * (t - b[j])).exp());
                        se_sum += (row[s] - p).powi(2);
                        cnt += 1;
                    }
                }
            }
            sum_rmse += (se_sum / cnt as f64).sqrt();
            reps += 1;
        }
        let avg_rmse = sum_rmse / reps as f64;
        assert!(
            avg_rmse < 0.06,
            "avg ICC RMSE {avg_rmse} (skew={skew}) exceeds 0.06"
        );
    }
}
