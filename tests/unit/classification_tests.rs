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

// ===================== Livingston & Lewis (1995) =====================
//
// Every assert below reads fields of the `LivingstonLewisResult` returned by
// `livingston_lewis` (crate outputs). Fixture literals come from an
// independent Python replication of spec ll_spec.md rev 2 (session artifact
// ll_fixture.py: scipy adaptive quadrature vs the crate substituted
// Gauss-Legendre — independent integration methods), tolerance 1e-7 for
// integral-derived fields and 1e-9 for pre-integration arithmetic.
//
// Mutation-kill map (spot-checked by actually applying each mutation):
// - M1 swap upper/lower binomial tail in accuracy integrands ->
//   p_tp/p_ff literals FAIL.
// - M2 use unrounded ETL in the passing threshold -> k shifts, literals FAIL
//   (fixture A has ETL = 91.123, materially non-integer).
// - M4 threshold off-by-one (k-1 -> k) -> literals FAIL (fixture A has
//   round(N c) = 55 != floor = 54, locking round-ties-even).
// - M5 sample variance ddof n-1 -> n -> ETL literal FAILs.
// - Failsafe drop (keep invalid 4P) -> fixture A/B (2P path) FAIL.
//
// Disclosed limitation: p_ij == p_ji is BY CONSTRUCTION under the
// single-threshold contract, so no mutation distinguishable through the
// p_ji field alone exists; the discriminating anchor for threshold handling
// is the fixture-A k = 55 vs floor 54 literal set.

use super::{livingston_lewis, LivingstonLewisResult};

/// LCG mirrored bit-for-bit in ll_fixture.py.
struct Lcg {
    s: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            s: seed
                .wrapping_mul(2862933555777941757)
                .wrapping_add(3037000493),
        }
    }
    fn unif(&mut self) -> f64 {
        self.s = self
            .s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.s >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn gen_scores(n: usize, n_items: usize, seed: u64, lo: f64, hi: f64, bates_k: usize) -> Vec<f64> {
    let mut r = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let mut acc = 0.0;
            for _ in 0..bates_k {
                acc += r.unif();
            }
            let p = lo + (hi - lo) * acc / bates_k as f64;
            let mut x = 0u32;
            for _ in 0..n_items {
                if r.unif() < p {
                    x += 1;
                }
            }
            x as f64
        })
        .collect()
}

fn gen_scores_beta(
    n: usize,
    n_items: usize,
    seed: u64,
    a: f64,
    b: f64,
    lo: f64,
    hi: f64,
) -> Vec<f64> {
    let mut r = Lcg::new(seed);
    (0..n)
        .map(|_| {
            let p = loop {
                let x = r.unif().powf(1.0 / a);
                let y = r.unif().powf(1.0 / b);
                if x + y <= 1.0 {
                    break lo + (hi - lo) * (x / (x + y));
                }
            };
            let mut s = 0u32;
            for _ in 0..n_items {
                if r.unif() < p {
                    s += 1;
                }
            }
            s as f64
        })
        .collect()
}

fn fixture_a() -> LivingstonLewisResult {
    let scores = gen_scores(250, 60, 42, 0.15, 0.95, 4);
    livingston_lewis(&scores, 0.85, 0.0, 60.0, 36.0).unwrap()
}

#[test]
fn ll_matches_independent_reference_two_parameter() {
    // Kills M1/M2/M4/M5 and the failsafe drop (see map above); fixture A has
    // round(91 * 0.6) = 55 while floor = 54, anchoring round-ties-even.
    let r = fixture_a();
    assert!((r.effective_test_length - 91.12301435951845).abs() < 1e-9);
    assert_eq!(r.etl_rounded, 91);
    assert!(r.used_two_parameter);
    assert_eq!(r.lower, 0.0);
    assert_eq!(r.upper, 1.0);
    assert!((r.alpha - 8.526094673559584).abs() < 1e-9);
    assert!((r.beta - 7.63609667022833).abs() < 1e-9);
    let tol = 1e-7;
    assert!((r.p_tp - 0.23854946889141398).abs() < tol);
    assert!((r.p_fp - 0.06401868217743813).abs() < tol);
    assert!((r.p_tf - 0.6521887072379015).abs() < tol);
    assert!((r.p_ff - 0.04524314169324793).abs() < tol);
    assert!((r.accuracy - 0.8907381761293154).abs() < tol);
    assert!((r.sensitivity - 0.8405767451096093).abs() < tol);
    assert!((r.specificity - 0.9106143232762532).abs() < tol);
    assert!((r.p_ii - 0.6209475077858381).abs() < tol);
    assert!((r.p_ij - 0.07648434114531032).abs() < tol);
    assert!((r.p_jj - 0.22608380992354124).abs() < tol);
    assert!((r.consistency - 0.8470313177093793).abs() < tol);
    assert!((r.chance_consistency - 0.5779586699447437).abs() < tol);
    assert!((r.kappa - 0.6375504686458243).abs() < tol);
}

#[test]
fn ll_extreme_cut_distinguishes_rounded_etl_in_k() {
    // Same data as fixture A but cut = 59: round(N*c) = round(91*59/60) = 89
    // while round(etl*c) = round(89.604) = 90, so this fixture kills the
    // mutation that computes k from the unrounded ETL (which fixtures A/B/P
    // cannot distinguish). Reference literals from ll_fixture.py (scipy quad).
    let scores = gen_scores(250, 60, 42, 0.15, 0.95, 4);
    let r = livingston_lewis(&scores, 0.85, 0.0, 60.0, 59.0).unwrap();
    assert_eq!(r.etl_rounded, 91);
    let tol = 1e-8;
    assert!((r.p_fp - 2.152128586520244e-06).abs() < tol);
    assert!((r.p_tf - 0.9999978476968782).abs() < 1e-7);
    assert!((r.kappa - 0.04370046162326584).abs() < 1e-7);
}

#[test]
fn ll_failsafe_engages_on_skewed_data() {
    // Ceiling-skewed data: the 4P moment fit lands out of bounds and the 2P
    // failsafe must engage. Kills a dropped-failsafe mutation.
    let scores = gen_scores(150, 40, 7, 0.70, 1.0, 2);
    let r = livingston_lewis(&scores, 0.80, 0.0, 40.0, 30.0).unwrap();
    assert!(r.used_two_parameter);
    assert!((r.effective_test_length - 90.29478586833179).abs() < 1e-9);
    assert!((r.alpha - 19.251648456163373).abs() < 1e-9);
    assert!((r.beta - 3.522383629203395).abs() < 1e-9);
    let tol = 1e-7;
    assert!((r.p_tp - 0.8432448010046247).abs() < tol);
    assert!((r.p_fp - 0.025111134448713338).abs() < tol);
    assert!((r.p_tf - 0.0853397836347724).abs() < tol);
    assert!((r.p_ff - 0.046304280911892266).abs() < tol);
    assert!((r.accuracy - 0.9285845846393972).abs() < tol);
    assert!((r.consistency - 0.89849902437627).abs() < tol);
    assert!((r.kappa - 0.5560427413146216).abs() < tol);
}

#[test]
fn ll_four_parameter_path() {
    // Beta(2, 1.5) true scores on [0.2, 0.95]: valid 4P solution with
    // non-trivial location parameters. Kills mutations in the 4P branch
    // (l/u formulas, g3-sign branch, spread term) that the 2P fixtures
    // cannot see.
    let scores = gen_scores_beta(300, 50, 1, 2.0, 1.5, 0.2, 0.95);
    let r = livingston_lewis(&scores, 0.9, 0.0, 50.0, 30.0).unwrap();
    assert!(!r.used_two_parameter);
    assert!((r.effective_test_length - 61.449520085331116).abs() < 1e-9);
    assert_eq!(r.etl_rounded, 61);
    assert!((r.lower - 0.24143017903895592).abs() < 1e-9);
    assert!((r.upper - 0.9393717155534436).abs() < 1e-9);
    assert!((r.alpha - 1.7565737858531474).abs() < 1e-9);
    assert!((r.beta - 1.3137395685852833).abs() < 1e-9);
    let tol = 1e-7;
    assert!((r.p_tp - 0.5543443184884189).abs() < tol);
    assert!((r.p_fp - 0.044475496859695596).abs() < tol);
    assert!((r.p_tf - 0.3583220858730486).abs() < tol);
    assert!((r.p_ff - 0.04285809877883742).abs() < tol);
    assert!((r.accuracy - 0.9126664043614675).abs() < tol);
    assert!((r.sensitivity - 0.928235222196601).abs() < tol);
    assert!((r.specificity - 0.8895835060430215).abs() < tol);
    assert!((r.p_ii - 0.33976752439044877).abs() < tol);
    assert!((r.p_ij - 0.06141266026143706).abs() < tol);
    assert!((r.p_jj - 0.537407155086677).abs() < tol);
    assert!((r.consistency - 0.8771746794771258).abs() < tol);
    assert!((r.chance_consistency - 0.5195307118108707).abs() < tol);
    assert!((r.kappa - 0.7443638468843696).abs() < tol);
}

#[test]
fn ll_structural_invariants_read_crate_fields() {
    // All operands are crate outputs; these anchors tie the derived fields
    // (accuracy, consistency, kappa) to the cell fields so a mutation that
    // desynchronizes them (e.g. computing kappa from raw unnormalized cells)
    // fails here even if it preserved the individual pinned literals.
    let r = fixture_a();
    assert!((r.p_tp + r.p_fp + r.p_tf + r.p_ff - 1.0).abs() < 1e-9);
    assert!((r.p_ii + r.p_ij + r.p_ji + r.p_jj - 1.0).abs() < 1e-12);
    assert_eq!(r.p_ij, r.p_ji); // by construction (disclosed above)
    assert!((r.accuracy - (r.p_tp + r.p_tf)).abs() < 1e-15);
    assert!((r.consistency - (r.p_ii + r.p_jj)).abs() < 1e-15);
    let pc = (r.p_ii + r.p_ij) * (r.p_ii + r.p_ji) + (r.p_ij + r.p_jj) * (r.p_ji + r.p_jj);
    assert!((r.chance_consistency - pc).abs() < 1e-15);
    assert!((r.kappa - (r.consistency - pc) / (1.0 - pc)).abs() < 1e-12);
}

#[test]
fn ll_rejects_malformed_input() {
    let ok = gen_scores(30, 20, 3, 0.2, 0.9, 3);
    assert!(livingston_lewis(&ok[..5], 0.8, 0.0, 20.0, 10.0).is_err());
    assert!(livingston_lewis(&ok, 0.8, 20.0, 0.0, 10.0).is_err());
    assert!(livingston_lewis(&ok, 0.8, 0.0, 20.0, 0.0).is_err());
    assert!(livingston_lewis(&ok, 0.8, 0.0, 20.0, 20.0).is_err());
    assert!(livingston_lewis(&ok, 0.0, 0.0, 20.0, 10.0).is_err());
    assert!(livingston_lewis(&ok, 1.0, 0.0, 20.0, 10.0).is_err());
    assert!(livingston_lewis(&ok, f64::NAN, 0.0, 20.0, 10.0).is_err());
    let mut with_nan = ok.clone();
    with_nan[0] = f64::NAN;
    assert!(livingston_lewis(&with_nan, 0.8, 0.0, 20.0, 10.0).is_err());
    let mut out_of_range = ok.clone();
    out_of_range[0] = 25.0;
    assert!(livingston_lewis(&out_of_range, 0.8, 0.0, 20.0, 10.0).is_err());
    let constant = vec![7.0; 30];
    assert!(livingston_lewis(&constant, 0.8, 0.0, 20.0, 10.0).is_err());
}

#[test]
#[ignore = "500-rep Monte Carlo; run with --ignored"]
fn ll_mc_consistency_recovers_empirical_agreement() {
    // Value recovery: crate `consistency` (model-based, from ONE
    // administration + reliability) vs the empirical agreement rate of two
    // independent simulated administrations. Both operands per rep: one is
    // the crate output, the other simulation truth.
    let n_items = 60usize;
    let n = 400usize;
    let reps = 500u64;
    let mut diff_sum = 0.0;
    for rep in 0..reps {
        let mut r = Lcg::new(9000 + rep);
        let mut x1 = Vec::with_capacity(n);
        let mut agree = 0usize;
        let cut = 33.0;
        let mut s1 = Vec::with_capacity(n);
        let mut s2 = Vec::with_capacity(n);
        for _ in 0..n {
            let mut acc = 0.0;
            for _ in 0..4 {
                acc += r.unif();
            }
            let p = 0.2 + 0.7 * acc / 4.0;
            let mut a = 0u32;
            let mut b = 0u32;
            for _ in 0..n_items {
                if r.unif() < p {
                    a += 1;
                }
                if r.unif() < p {
                    b += 1;
                }
            }
            if (a as f64 >= cut) == (b as f64 >= cut) {
                agree += 1;
            }
            x1.push(a as f64);
            s1.push(a as f64);
            s2.push(b as f64);
        }
        // Parallel-forms reliability = corr(X1, X2) for this rep.
        let m1 = s1.iter().sum::<f64>() / n as f64;
        let m2 = s2.iter().sum::<f64>() / n as f64;
        let (mut c12, mut v1, mut v2) = (0.0, 0.0, 0.0);
        for i in 0..n {
            c12 += (s1[i] - m1) * (s2[i] - m2);
            v1 += (s1[i] - m1) * (s1[i] - m1);
            v2 += (s2[i] - m2) * (s2[i] - m2);
        }
        let rel = c12 / (v1.sqrt() * v2.sqrt());
        let est = livingston_lewis(&x1, rel, 0.0, n_items as f64, cut).unwrap();
        diff_sum += est.consistency - agree as f64 / n as f64;
    }
    let bias = diff_sum / reps as f64;
    assert!(bias.abs() < 0.02, "mean consistency bias {bias}");
}

#[test]
fn ll_conditional_ratios_nan_when_margin_vanishes() {
    // Cut far below the fitted 4P support: the true-fail margin is zero, so
    // specificity must be NaN while the aggregate cells stay finite and sum
    // to 1. With cut = 0.1 on fixture A data, k = round(91 * c) = 0 forces
    // fail_prob == 0, giving consistency == chance == 1 and kappa NaN.
    // Limitation: at exactly-zero denominators IEEE 0/0 is already NaN, so
    // a dropped `ratio` guard is NOT killed here; the guard's value is for
    // tiny nonzero denominators (documented, no discriminating fixture).
    let scores = gen_scores_beta(300, 50, 1, 2.0, 1.5, 0.2, 0.95);
    let r = livingston_lewis(&scores, 0.9, 0.0, 50.0, 5.0).unwrap();
    assert!(r.specificity.is_nan());
    assert!(r.sensitivity.is_finite());
    assert!((r.p_tp + r.p_fp + r.p_tf + r.p_ff - 1.0).abs() < 1e-9);

    let scores_a = gen_scores(250, 60, 42, 0.15, 0.95, 4);
    let r = livingston_lewis(&scores_a, 0.85, 0.0, 60.0, 0.1).unwrap();
    assert!(r.kappa.is_nan());
    assert!((r.consistency - 1.0).abs() < 1e-9);
    assert!((r.accuracy - 1.0).abs() < 1e-9);
}
