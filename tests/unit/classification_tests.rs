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

// ===================== Hanson-Brennan (1991) =====================
//
// Every assert below reads fields of the `HansonBrennanResult` returned by
// `hanson_brennan` / `hanson_brennan_from_params` (crate outputs). Fixture
// literals come from an independent exact-Fraction stdlib oracle (session
// artifact hanson_brennan_oracle.py; never imports this crate): params
// fixtures A/C/E and the fixture-B moments/k/shapes/consistency are exact
// rationals (tolerance 1e-12); fixture-B accuracy cells and all fixture-D
// values are endpoint-aware float quadrature (tolerances 1e-9 / 1e-7 per
// spec hanson_brennan_spec.md rev 2).
//
// Mutation-kill map (each mutation spot-checked by actually applying it):
// - MU1 drop the k-correction term in the fail CDF -> fixture A (k = 1/2)
//   pinned cells FAIL.
// - MU2 drop the `+ k i(i-1) m_{i-1}` correction in the moment recursion ->
//   fixture B m2..m4 / alpha / beta literals FAIL (B has k = 3.5997).
// - MU3 drop the factor 2 in Lord's k denominator -> fixture B k literal
//   FAILs.
// - MU4 integrate F instead of 1 - F in p_tp -> fixture A p_tp/accuracy
//   literals FAIL.
// - MU5 use (1 - F) instead of (1 - F)^2 in p_jj -> fixture A p_jj /
//   consistency literals FAIL.
//
// Disclosed limitations:
// - p_ij == p_ji BY CONSTRUCTION (single threshold), as for LL.
// - The pmf normalization (sum of raw two-term cells = 1) holds
//   analytically for every k, so a "renormalize" mutation is unobservable
//   through these fields; the discriminating anchors are the raw negative-
//   cell fixtures (D has k < 0, B has k > 3).

use super::{hanson_brennan, hanson_brennan_from_params};

const HB_B_SCORES: [f64; 12] = [3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 6.0, 7.0, 7.0, 8.0, 9.0];
const HB_D_SCORES: [f64; 22] = [
    7.0, 1.0, 11.0, 2.0, 9.0, 10.0, 6.0, 6.0, 12.0, 6.0, 10.0, 8.0, 10.0, 8.0, 2.0, 2.0, 5.0, 8.0,
    12.0, 11.0, 2.0, 1.0,
];

#[test]
fn hb_params_matches_exact_oracle_fixture_a() {
    // Kills MU1/MU4/MU5 (see map above); K=8, k=1/2, beta(2,3) on [0,1],
    // cut=5. All pins are exact rationals from the oracle.
    let r = hanson_brennan_from_params(8, 0.5, 0.0, 1.0, 2.0, 3.0, 5).unwrap();
    let tol = 1e-12;
    assert!((r.p_ii - 0.626795675944283).abs() < tol);
    assert!((r.p_ij - 0.1058883067397).abs() < tol);
    assert!((r.p_jj - 0.161427710576317).abs() < tol);
    assert!((r.consistency - 0.7882233865206).abs() < tol);
    assert!((r.chance_consistency - 0.60828367159536).abs() < tol);
    assert!((r.kappa - 0.459362303476315).abs() < tol);
    assert!((r.p_tp - 0.1271510992216).abs() < tol);
    assert!((r.p_fp - 0.140164918094417).abs() < tol);
    assert!((r.p_ff - 0.0244602289033998).abs() < tol);
    assert!((r.p_tf - 0.708223753780583).abs() < tol);
    assert!((r.accuracy - 0.835374853002183).abs() < tol);
    assert!((r.sensitivity - 0.838664899213647).abs() < tol);
    assert!((r.specificity - 0.834786905175617).abs() < tol);
    assert!(!r.used_two_parameter);
    assert!(r.true_score_moments.iter().all(|m| m.is_nan()));
}

#[test]
fn hb_params_matches_exact_oracle_fixture_c_shifted_support() {
    // Non-[0,1] support (l=0.1, u=0.9) anchors the p = l + (u-l)t mapping
    // and the truecut clamp; K=6, k=1/4, beta(2,2), cut=4.
    let r = hanson_brennan_from_params(6, 0.25, 0.1, 0.9, 2.0, 2.0, 4).unwrap();
    let tol = 1e-12;
    assert!((r.p_ii - 0.462276103171718).abs() < tol);
    assert!((r.p_ij - 0.151822587304473).abs() < tol);
    assert!((r.p_jj - 0.234078722219337).abs() < tol);
    assert!((r.consistency - 0.696354825391055).abs() < tol);
    assert!((r.chance_consistency - 0.526037022336763).abs() < tol);
    assert!((r.kappa - 0.35934832693896).abs() < tol);
    assert!((r.p_tp - 0.170177671527219).abs() < tol);
    assert!((r.p_fp - 0.215723637996591).abs() < tol);
    assert!((r.p_ff - 0.035406819213522).abs() < tol);
    assert!((r.p_tf - 0.578691871262669).abs() < tol);
    assert!((r.accuracy - 0.748869542789887).abs() < tol);
    assert!((r.sensitivity - 0.827774852636268).abs() < tol);
    assert!((r.specificity - 0.728449865993).abs() < tol);
}

#[test]
fn hb_params_boundary_cuts_fixture_e() {
    // K=4, k=1/3, beta(2,2). cut=1 vs cut=K swap the consistency blocks
    // (identical consistency/kappa) while the accuracy cells flip; cut=K
    // yields an empty pass region -> p_tp = p_ff = 0 and NaN sensitivity.
    let tol = 1e-12;
    let r1 = hanson_brennan_from_params(4, 1.0 / 3.0, 0.0, 1.0, 2.0, 2.0, 1).unwrap();
    assert!((r1.p_ii - 0.046969696969697).abs() < tol);
    assert!((r1.p_ij - 0.0768398268398268).abs() < tol);
    assert!((r1.p_jj - 0.799350649350649).abs() < tol);
    assert!((r1.consistency - 0.846320346320346).abs() < tol);
    assert!((r1.kappa - 0.291673000912131).abs() < tol);
    assert!((r1.p_tp - 0.794594029017857).abs() < tol);
    assert!((r1.p_fp - 0.081596447172619).abs() < tol);
    assert!((r1.p_ff - 0.0491559709821429).abs() < tol);
    assert!((r1.p_tf - 0.074653552827381).abs() < tol);
    assert!((r1.accuracy - 0.869247581845238).abs() < tol);
    assert!((r1.sensitivity - 0.941741071428571).abs() < tol);
    assert!((r1.specificity - 0.477782738095238).abs() < tol);

    let rk = hanson_brennan_from_params(4, 1.0 / 3.0, 0.0, 1.0, 2.0, 2.0, 4).unwrap();
    assert!((rk.p_ii - 0.799350649350649).abs() < tol);
    assert!((rk.p_jj - 0.046969696969697).abs() < tol);
    assert!((rk.consistency - r1.consistency).abs() < tol);
    assert!((rk.kappa - r1.kappa).abs() < tol);
    assert_eq!(rk.p_tp, 0.0);
    assert_eq!(rk.p_ff, 0.0);
    assert!((rk.p_fp - 0.123809523809524).abs() < tol);
    assert!((rk.p_tf - 0.876190476190476).abs() < tol);
    assert!((rk.accuracy - 0.876190476190476).abs() < tol);
    assert!(rk.sensitivity.is_nan());
    assert!((rk.specificity - 0.876190476190476).abs() < tol);
}

#[test]
fn hb_data_two_parameter_matches_exact_oracle_fixture_b() {
    // Kills MU2 (moment-recursion correction; k = 3.5997 here) and MU3
    // (Lord's k denominator factor 2). k, moments, and shapes are exact
    // rationals (1e-12); integral cells are cross-quadrature pins (1e-9).
    let r = hanson_brennan(&HB_B_SCORES, 10, 0.8, 6, true).unwrap();
    let tol = 1e-12;
    assert!((r.lords_k - 3.59971809506204).abs() < tol);
    assert!((r.true_score_moments[0] - 7.0 / 12.0).abs() < tol);
    assert!((r.true_score_moments[1] - 0.36213863465272).abs() < tol);
    assert!((r.true_score_moments[2] - 0.237540177051756).abs() < tol);
    assert!((r.true_score_moments[3] - 0.163642642262692).abs() < tol);
    assert!(r.used_two_parameter);
    assert_eq!(r.lower, 0.0);
    assert_eq!(r.upper, 1.0);
    assert!((r.alpha - 5.90234141484747).abs() < 1e-11);
    assert!((r.beta - 4.21595815346248).abs() < 1e-11);
    let tol9 = 1e-9;
    assert!((r.p_ii - 0.298000987219999).abs() < tol9);
    assert!((r.p_ij - 0.118776784992038).abs() < tol9);
    assert!((r.p_jj - 0.464445442795926).abs() < tol9);
    assert!((r.consistency - 0.762446430015925).abs() < tol9);
    assert!((r.chance_consistency - 0.513851878395983).abs() < tol9);
    assert!((r.kappa - 0.511355573687539).abs() < tol9);
    assert!((r.p_tp - 0.428148532096005).abs() < tol9);
    assert!((r.p_fp - 0.155073695691995).abs() < tol9);
    assert!((r.p_ff - 0.0441750264685139).abs() < tol9);
    assert!((r.p_tf - 0.372602745743486).abs() < tol9);
    assert!((r.accuracy - 0.800751277839491).abs() < tol9);
    assert!((r.sensitivity - 0.906472955524873).abs() < tol9);
    assert!((r.specificity - 0.706119728843426).abs() < tol9);
}

#[test]
fn hb_data_four_parameter_matches_oracle_fixture_d() {
    // Genuine 4P fit with NEGATIVE Lord's k (raw negative two-term cells
    // integrated unclamped) and both beta shapes < 1 (integrable endpoint
    // singularities exercising the substituted quadrature). Oracle used
    // singularity-substituted Simpson; tolerance 1e-7 per spec.
    let r = hanson_brennan(&HB_D_SCORES, 12, 0.85, 7, false).unwrap();
    assert!((r.lords_k + 0.428757070304408).abs() < 1e-12);
    assert!((r.true_score_moments[0] - 0.564393939393939).abs() < 1e-12);
    assert!((r.true_score_moments[1] - 0.396290868897843).abs() < 1e-12);
    assert!((r.true_score_moments[2] - 0.305461229963853).abs() < 1e-12);
    assert!((r.true_score_moments[3] - 0.245696195187782).abs() < 1e-12);
    assert!(!r.used_two_parameter);
    assert!((r.lower - 0.132649605850816).abs() < 1e-11);
    assert!((r.upper - 0.888674369964715).abs() < 1e-11);
    assert!((r.alpha - 0.45726578470813).abs() < 1e-11);
    assert!((r.beta - 0.343449430670119).abs() < 1e-11);
    let tol = 1e-7;
    assert!((r.p_tp - 0.496827945781606).abs() < tol);
    assert!((r.p_fp - 0.0584972285971357).abs() < tol);
    assert!((r.p_ff - 0.0307477664292897).abs() < tol);
    assert!((r.p_tf - 0.41392705919197).abs() < tol);
    assert!((r.accuracy - 0.910755004973576).abs() < tol);
    assert!((r.sensitivity - 0.941718760516029).abs() < tol);
    assert!((r.specificity - 0.876176500427409).abs() < tol);
    assert!((r.p_ii - 0.383607601024351).abs() < tol);
    assert!((r.p_ij - 0.0610672245969103).abs() < tol);
    assert!((r.p_jj - 0.494257949781828).abs() < tol);
    assert!((r.consistency - 0.877865550806179).abs() < tol);
    assert!((r.chance_consistency - 0.506121749840076).abs() < tol);
    assert!((r.kappa - 0.752703324849248).abs() < tol);
}

#[test]
fn hb_structural_invariants_read_crate_fields() {
    // Ties derived fields to cell fields through crate outputs only, so a
    // mutation desynchronizing them fails even if pinned literals survive.
    let r = hanson_brennan(&HB_D_SCORES, 12, 0.85, 7, false).unwrap();
    assert!((r.p_tp + r.p_fp + r.p_tf + r.p_ff - 1.0).abs() < 1e-7);
    assert!((r.p_ii + r.p_ij + r.p_ji + r.p_jj - 1.0).abs() < 1e-12);
    assert_eq!(r.p_ij, r.p_ji); // by construction (disclosed above)
    assert!((r.accuracy - (r.p_tp + r.p_tf)).abs() < 1e-15);
    assert!((r.consistency - (r.p_ii + r.p_jj)).abs() < 1e-15);
    let pc = (r.p_ii + r.p_ij) * (r.p_ii + r.p_ji) + (r.p_ij + r.p_jj) * (r.p_ji + r.p_jj);
    assert!((r.chance_consistency - pc).abs() < 1e-15);
    assert!((r.kappa - (r.consistency - pc) / (1.0 - pc)).abs() < 1e-12);
}

#[test]
fn hb_rejects_malformed_input() {
    let ok: Vec<f64> = HB_D_SCORES.to_vec();
    // data path
    assert!(hanson_brennan(&ok[..5], 12, 0.85, 7, false).is_err());
    assert!(hanson_brennan(&ok, 3, 0.85, 2, false).is_err());
    assert!(hanson_brennan(&ok, 12, 0.85, 0, false).is_err());
    assert!(hanson_brennan(&ok, 12, 0.85, 13, false).is_err());
    assert!(hanson_brennan(&ok, 12, 0.0, 7, false).is_err());
    assert!(hanson_brennan(&ok, 12, 1.0, 7, false).is_err());
    assert!(hanson_brennan(&ok, 12, f64::NAN, 7, false).is_err());
    let mut with_nan = ok.clone();
    with_nan[0] = f64::NAN;
    assert!(hanson_brennan(&with_nan, 12, 0.85, 7, false).is_err());
    let mut frac = ok.clone();
    frac[0] = 6.5;
    assert!(hanson_brennan(&frac, 12, 0.85, 7, false).is_err());
    let mut oob = ok.clone();
    oob[0] = 13.0;
    assert!(hanson_brennan(&oob, 12, 0.85, 7, false).is_err());
    let constant = vec![7.0; 22];
    assert!(hanson_brennan(&constant, 12, 0.85, 7, false).is_err());
    // params path
    assert!(hanson_brennan_from_params(1, 0.0, 0.0, 1.0, 2.0, 2.0, 1).is_err());
    assert!(hanson_brennan_from_params(8, f64::NAN, 0.0, 1.0, 2.0, 2.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.5, 0.4, 2.0, 2.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, -0.1, 1.0, 2.0, 2.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.0, 1.1, 2.0, 2.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.0, 1.0, 0.0, 2.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.0, 1.0, 2.0, -1.0, 4).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.0, 1.0, 2.0, 2.0, 0).is_err());
    assert!(hanson_brennan_from_params(8, 0.0, 0.0, 1.0, 2.0, 2.0, 9).is_err());
}

#[test]
#[ignore = "500-rep Monte Carlo; run with --ignored"]
fn hb_mc_consistency_recovers_empirical_agreement() {
    // Value recovery: crate `consistency` (from ONE administration +
    // parallel-forms reliability) vs the empirical agreement rate of two
    // independent simulated binomial administrations (true k = 0). One
    // operand per rep is the crate output, the other simulation truth.
    let n_items = 40usize;
    let n = 400usize;
    let reps = 500u64;
    let mut diff_sum = 0.0;
    let mut used = 0u64;
    for rep in 0..reps {
        let mut r = Lcg::new(77_000 + rep);
        let cut = 22usize;
        let mut s1 = Vec::with_capacity(n);
        let mut s2 = Vec::with_capacity(n);
        let mut agree = 0usize;
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
            if (a as f64 >= cut as f64) == (b as f64 >= cut as f64) {
                agree += 1;
            }
            s1.push(a as f64);
            s2.push(b as f64);
        }
        let m1 = s1.iter().sum::<f64>() / n as f64;
        let m2 = s2.iter().sum::<f64>() / n as f64;
        let (mut c12, mut v1, mut v2) = (0.0, 0.0, 0.0);
        for i in 0..n {
            c12 += (s1[i] - m1) * (s2[i] - m2);
            v1 += (s1[i] - m1) * (s1[i] - m1);
            v2 += (s2[i] - m2) * (s2[i] - m2);
        }
        let rel = c12 / (v1.sqrt() * v2.sqrt());
        if let Ok(est) = hanson_brennan(&s1, n_items, rel, cut, false) {
            diff_sum += est.consistency - agree as f64 / n as f64;
            used += 1;
        }
    }
    assert!(used > 450, "too many degenerate reps: {used}");
    let bias = diff_sum / used as f64;
    assert!(bias.abs() < 0.01, "consistency bias {bias}");
}

use super::subkoviak_agreement;

// ---- Subkoviak (1976) single-administration coefficient of agreement ----
//
// All exact pins below are from the executed exact-Fraction oracle
// (subkoviak_oracle.py) against the READ ERIC ED120229 source. Every
// assert reads crate outputs (fields of SubkoviakResult).

#[test]
fn subkoviak_table1_alpha_supplied_exact() {
    // Fixture A: paper Table 1 (n = 5, cut C = 4, alpha = .58 supplied
    // because the footnote a21 = .58 is irreproducible from the printed
    // S^2 = 2.61 -- disclosed in the core doc comment).
    let x = [0.0, 4.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0];
    let r = subkoviak_agreement(&x, 5, &[4.0], Some(0.58)).unwrap();
    // Kills MU5 (swapped Eq. 16 weights: p_hat[0] would be .2668). Note:
    // category-label permutation is unobservable through this API (see the
    // core doc comment) -- no test can anchor category ordering.
    assert!((r.alpha - 0.58).abs() < 1e-15, "alpha {}", r.alpha);
    assert!((r.p_hat[0] - 0.1932).abs() < 1e-12, "p_hat0 {}", r.p_hat[0]);
    // Kills MU1 (mastery > vs >=) and MU2 (missing squaring: P(i) = 1).
    assert!(
        (r.per_person[0] - 0.988290295814609).abs() < 1e-12,
        "P(0) {}",
        r.per_person[0]
    );
    assert!(
        (r.per_person[1] - 0.506648836339306).abs() < 1e-12,
        "P(1) {}",
        r.per_person[1]
    );
    assert!(
        (r.agreement - 0.754404497506925).abs() < 1e-12,
        "Pc {}",
        r.agreement
    );
    // Kills MU3 (mean-of-squares instead of square-of-means chance term).
    assert!(
        (r.chance_agreement - 0.659131077528193).abs() < 1e-12,
        "Pchance {}",
        r.chance_agreement
    );
    assert!(
        (r.kappa - 0.279501631559306).abs() < 1e-12,
        "kappa {}",
        r.kappa
    );
}

#[test]
fn subkoviak_derived_kr21_alpha() {
    // Fixture B: same scores, alpha = None -> KR-21 with population
    // variance 2.61 -> alpha = 19/29. Kills MU4 (ddof = 1 gives alpha
    // = .714655..., executed kill).
    let x = [0.0, 4.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0];
    let r = subkoviak_agreement(&x, 5, &[4.0], None).unwrap();
    assert!((r.alpha - 19.0 / 29.0).abs() < 1e-12, "alpha {}", r.alpha);
    assert!(
        (r.agreement - 0.763498543160431).abs() < 1e-12,
        "Pc {}",
        r.agreement
    );
    assert!(
        (r.kappa - 0.343090844315065).abs() < 1e-12,
        "kappa {}",
        r.kappa
    );
}

#[test]
fn subkoviak_multi_cut_ml() {
    // Fixture C: asymmetric 3-category split (h = 3), n = 6,
    // cuts = [2, 5], alpha = 1 (ML p_hat = X/n, Eq. 15). Kills MU1's
    // interior-boundary variant (upper bound <= C vs < C).
    let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 3.0];
    let r = subkoviak_agreement(&x, 6, &[2.0, 5.0], Some(1.0)).unwrap();
    assert!(
        (r.per_person[0] - 0.611776411438135).abs() < 1e-12,
        "P(0) {}",
        r.per_person[0]
    );
    assert!(
        (r.agreement - 0.691963008928055).abs() < 1e-12,
        "Pc {}",
        r.agreement
    );
    assert!(
        (r.chance_agreement - 0.344265263966199).abs() < 1e-12,
        "Pchance {}",
        r.chance_agreement
    );
    assert!(
        (r.kappa - 0.53024146176074).abs() < 1e-12,
        "kappa {}",
        r.kappa
    );
}

#[test]
fn subkoviak_alpha_zero_kappa_zero() {
    // Fixture D: alpha = 0 collapses every p_hat to M/n, so P(i) is
    // identical across persons and Pc equals Pchance exactly -> kappa 0.
    // Over-collapse anchor; reads crate kappa/per_person/p_hat.
    let x = [0.0, 4.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0];
    let r = subkoviak_agreement(&x, 5, &[4.0], Some(0.0)).unwrap();
    assert!(r.kappa.abs() < 1e-12, "kappa {}", r.kappa);
    for (i, p) in r.p_hat.iter().enumerate() {
        assert!((p - 0.46).abs() < 1e-15, "p_hat[{i}] {p}");
    }
    for (i, pi) in r.per_person.iter().enumerate() {
        assert!((pi - r.per_person[0]).abs() < 1e-15, "P({i}) {pi} != P(0)");
    }
}

#[test]
fn subkoviak_error_contract() {
    let ok = [0.0, 4.0, 2.0, 0.0, 2.0, 2.0, 1.0, 3.0, 4.0, 5.0];
    // n_items too small.
    assert!(subkoviak_agreement(&[0.0, 1.0], 1, &[1.0], Some(0.5)).is_err());
    // Fewer than 2 persons.
    assert!(subkoviak_agreement(&[3.0], 5, &[4.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&[], 5, &[4.0], Some(0.5)).is_err());
    // Non-integer, out-of-range, and non-finite scores.
    assert!(subkoviak_agreement(&[0.5, 1.0], 5, &[4.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&[6.0, 1.0], 5, &[4.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&[-1.0, 1.0], 5, &[4.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&[f64::NAN, 1.0], 5, &[4.0], Some(0.5)).is_err());
    // Cuts: empty, out of 1..=n, non-integer, not strictly increasing.
    assert!(subkoviak_agreement(&ok, 5, &[], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[0.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[6.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[2.5], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[3.0, 3.0], Some(0.5)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[4.0, 2.0], Some(0.5)).is_err());
    // Alpha out of [0, 1] or non-finite.
    assert!(subkoviak_agreement(&ok, 5, &[4.0], Some(-0.1)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[4.0], Some(1.1)).is_err());
    assert!(subkoviak_agreement(&ok, 5, &[4.0], Some(f64::NAN)).is_err());
    // KR-21 derivation with zero variance.
    assert!(subkoviak_agreement(&[3.0, 3.0, 3.0], 5, &[4.0], None).is_err());
}

#[test]
#[ignore = "500-rep Monte Carlo; run with --ignored"]
fn subkoviak_mc_500_recovers_two_administration_agreement() {
    // Value recovery of the agreement formula itself (Eqs. 7-8/5),
    // decoupled from p-hat estimation bias: each person's true domain
    // proportion is set to exactly X/n so the ML (alpha = 1) p-hat equals
    // truth, then crate Pc is compared with the empirical agreement rate
    // of two independent simulated binomial administrations at that truth.
    // (The estimator variants are genuinely biased under this design --
    // measured ML alpha=1 bias ~ +.057 and KR-21 regression bias ~ -.045,
    // consistent with the paper's own report that the ML estimate
    // overstates agreement -- so an estimator-in-the-loop MC cannot pin
    // the formula.) One operand per rep is the crate output, the other is
    // simulation truth.
    let n_items = 30usize;
    let reps = 500u64;
    let cut = 17.0f64;
    let scores: Vec<f64> = (10..=24).map(|x| x as f64).collect();
    let est = subkoviak_agreement(&scores, n_items, &[cut], Some(1.0)).unwrap();
    let mut agree = 0u64;
    let mut total = 0u64;
    let mut r = Lcg::new(91_000);
    for _ in 0..reps {
        for x in &scores {
            let p = x / n_items as f64;
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
            total += 1;
        }
    }
    let bias = est.agreement - agree as f64 / total as f64;
    assert!(bias.abs() < 0.02, "agreement bias {bias}");
}

// ---------------------------------------------------------------------------
// Livingston (1972) criterion-referenced reliability k^2 and correlation.
// Oracle: exact-Fraction values in files/livingston_oracle.py (session dir),
// derived from ERIC ED069624 (READ). Every assert reads crate outputs.
// ---------------------------------------------------------------------------

use super::{livingston_correlation, livingston_k2};

// Fixture A: X = [2,4,4,6], C = 2, rho^2 = 1/2.
// Exact: mean 4, var 2 (ddof=0), msd 6, k^2 = 5/6, SB(n=2) = 10/11.
// Kills MU1 (drop offset^2 in numerator -> 1/6), MU2 (drop in denominator
// -> 5/2), MU3 (unsquared offset -> 3/4), MU4 (SB applied to rho^2 not k^2
// -> 2/3), MU5 (ddof=1 variance -> 4/5).
#[test]
fn liv_fixture_a_exact() {
    let r = livingston_k2(&[2.0, 4.0, 4.0, 6.0], 2.0, 0.5, &[1.0, 2.0]).unwrap();
    assert_eq!(r.mean, 4.0);
    assert_eq!(r.var, 2.0);
    assert_eq!(r.msd, 6.0);
    assert!((r.k2[0] - 5.0 / 6.0).abs() < 1e-15, "k2 {}", r.k2[0]);
    assert!((r.k2[1] - 10.0 / 11.0).abs() < 1e-15, "SB(2) {}", r.k2[1]);
}

// Equality anchor (source pp. 3-4): k^2 = rho^2 iff mean = cut, and
// k^2 >= rho^2 otherwise. Reads crate k2 values at two cuts.
#[test]
fn liv_equality_iff_mean_eq_cut() {
    let at_mean = livingston_k2(&[2.0, 4.0, 4.0, 6.0], 4.0, 0.5, &[1.0]).unwrap();
    assert!((at_mean.k2[0] - 0.5).abs() < 1e-15, "{}", at_mean.k2[0]);
    let off = livingston_k2(&[2.0, 4.0, 4.0, 6.0], 2.0, 0.5, &[1.0]).unwrap();
    assert!(off.k2[0] > 0.5);
}

// Zero-variance property (source p. 4): constant scores with mean != cut
// give k^2 = 1 exactly; the exact degenerate case mean == cut gives NaN.
// A tiny offset (1e-7) must still give 1, not NaN (spec-review change 2:
// no absolute D^2 tolerance may widen the NaN case).
#[test]
fn liv_zero_variance_property() {
    let r = livingston_k2(&[3.0, 3.0, 3.0, 3.0], 1.0, 0.0, &[1.0]).unwrap();
    assert_eq!(r.k2[0], 1.0);
    let tiny = livingston_k2(&[3.0, 3.0, 3.0, 3.0], 3.0 - 1e-7, 0.0, &[1.0]).unwrap();
    assert_eq!(tiny.k2[0], 1.0);
    let degen = livingston_k2(&[3.0, 3.0, 3.0, 3.0], 3.0, 0.5, &[1.0, 2.0]).unwrap();
    assert!(degen.k2[0].is_nan() && degen.k2[1].is_nan());
}

// Fixture B: sign anchor. X=[1,2,3], Y=[3,2,1], cuts 0: norm rho = -1 but
// k(X,Y) = +5/7 (formula-derived adversarial pin; the source figures show
// analogous sign disagreements). Kills MU6 (drop the mean-offset cross
// product in D(X,Y) -> -1/7, wrong sign).
#[test]
fn liv_correlation_sign_flip() {
    let k = livingston_correlation(&[1.0, 2.0, 3.0], &[3.0, 2.0, 1.0], 0.0, 0.0).unwrap();
    assert!((k - 5.0 / 7.0).abs() < 1e-15, "k {k}");
    assert!(k > 0.0);
}

// Fixture E: asymmetric offsets (mx-cx = 2 != my-cy = 3), spec-review
// change 5. Exact num = 22/3, den^2 = 490/9, k = 22/(7 sqrt(10)).
// Kills MU7 (swapped cuts -> num 16/3) and MU8 (single-side offset squared
// -> num 16/3); both shift k to 16/(7 sqrt(10)).
#[test]
fn liv_correlation_asymmetric_offsets() {
    let k = livingston_correlation(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0], 0.0, 1.0).unwrap();
    let expect = 22.0 / (7.0 * 10.0f64.sqrt());
    assert!((k - expect).abs() < 1e-15, "k {k} expect {expect}");
}

// Correlation zero-denominator: constant X exactly at its cut -> D^2(X) = 0
// -> NaN (spec-review change 5, direct pin).
#[test]
fn liv_correlation_degenerate_nan() {
    let k = livingston_correlation(&[2.0, 2.0, 2.0], &[2.0, 4.0, 6.0], 2.0, 1.0).unwrap();
    assert!(k.is_nan());
}

// Error contract.
#[test]
fn liv_error_contract() {
    assert!(livingston_k2(&[1.0], 0.0, 0.5, &[1.0]).is_err());
    assert!(livingston_k2(&[1.0, f64::NAN], 0.0, 0.5, &[1.0]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], f64::INFINITY, 0.5, &[1.0]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], 0.0, 1.5, &[1.0]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], 0.0, -0.1, &[1.0]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], 0.0, 0.5, &[]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], 0.0, 0.5, &[0.0]).is_err());
    assert!(livingston_k2(&[1.0, 2.0], 0.0, 0.5, &[-1.0]).is_err());
    assert!(livingston_correlation(&[1.0], &[1.0], 0.0, 0.0).is_err());
    assert!(livingston_correlation(&[1.0, 2.0], &[1.0], 0.0, 0.0).is_err());
    assert!(livingston_correlation(&[1.0, f64::NAN], &[1.0, 2.0], 0.0, 0.0).is_err());
    assert!(livingston_correlation(&[1.0, 2.0], &[1.0, 2.0], f64::NAN, 0.0).is_err());
}

// MC-500: X = T + E with T ~ N(0.6, 0.04), E ~ N(0, 0.01) per person,
// n = 200 persons. Population k^2 = (rho^2 s^2 + (mu-C)^2)/(s^2 + (mu-C)^2)
// with s^2 = 0.05, rho^2 = 0.8, mu = 0.6, C = 0.5. This recovers the
// Livingston transform and population-moment path when fed the TRUE rho^2;
// it does not test reliability estimation (disclosed in spec).
#[test]
#[ignore]
fn liv_mc_500() {
    let s2 = 0.05f64;
    let rho2 = 0.04 / s2;
    let (mu, cut) = (0.6f64, 0.5f64);
    let pop = (rho2 * s2 + (mu - cut).powi(2)) / (s2 + (mu - cut).powi(2));
    let mut r = Lcg::new(20260726);
    let normal = |r: &mut Lcg| {
        let (u1, u2) = (r.unif().max(1e-12), r.unif());
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    let mut sum = 0.0;
    let reps = 500;
    for _ in 0..reps {
        let scores: Vec<f64> = (0..200)
            .map(|_| mu + 0.2 * normal(&mut r) + 0.1 * normal(&mut r))
            .collect();
        let est = livingston_k2(&scores, cut, rho2, &[1.0]).unwrap();
        sum += est.k2[0];
    }
    let bias = sum / reps as f64 - pop;
    assert!(bias.abs() < 0.01, "bias {bias} pop {pop}");
}

// Regression pins for impl-review findings (files/livingston_impl_review.md):
// finding 1 (MEDIUM): constant scores at a non-representable decimal cut
// (0.1) must still hit the exact-degenerate NaN path even though the summed
// mean rounds away from 0.1 (element-wise check); finding 2 (LOW): a huge
// finite cut whose squared offset overflows returns the formula limit 1 for
// k^2 and an Err (not NaN-by-overflow) for the correlation.
#[test]
fn liv_degenerate_nonrepresentable_decimal() {
    let r = livingston_k2(&[0.1, 0.1, 0.1], 0.1, 0.5, &[1.0]).unwrap();
    assert!(r.k2[0].is_nan(), "k2 {}", r.k2[0]);
    let k = livingston_correlation(&[0.1, 0.1, 0.1], &[1.0, 2.0, 3.0], 0.1, 0.0).unwrap();
    assert!(k.is_nan(), "k {k}");
}

#[test]
fn liv_huge_cut_overflow() {
    let r = livingston_k2(&[1.0, 2.0, 3.0], 1e308, 0.5, &[1.0, 2.0]).unwrap();
    assert_eq!(r.k2[0], 1.0);
    assert_eq!(r.k2[1], 1.0); // SB(1) = 1 for all n
    assert!(livingston_correlation(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], 1e308, 0.0).is_err());
}

// ---------------------------------------------------------------------------
// Woodruff & Sawyer (1988) pass-fail reliability from parallel half-tests
// (ERIC ED292877). Every assert reads crate outputs (woodruff_sawyer_sb /
// woodruff_sawyer_normal fields); expected values from the exact-Fraction /
// mpmath session oracle. Mutation notes name the crate value each assert
// reads and the mutation it kills.
// ---------------------------------------------------------------------------
use super::{woodruff_sawyer_normal, woodruff_sawyer_sb};

/// Fixture A: counts (n00,n01,n10,n11) = (2,1,3,10). Exact rationals:
/// pi01_s = 1/8, p = 3/4, q = 1/4, phi_half = 1/3, theta_half = 3/4,
/// phi* = 1/2, theta* = 13/16, pi*00 = 5/32, pi*01 = 3/32, pi*11 = 21/32.
/// Kills MU1 (drop smoothing entirely: raw p = 11/16 -> phi* = 39/47),
/// MU2 (denominator 2pq instead of 2pq - pi01_s: phi* = 2/3),
/// MU3 (no Spearman-Brown step-up: phi* = phi_half = 1/3),
/// MU4 (theta* missing p^2 + q^2: 3/16 instead of 13/16).
/// Reads: r.phi_half, r.theta_half, r.phi, r.theta, r.pi00, r.pi01, r.pi11,
/// r.pass_rate — all crate outputs.
#[test]
fn ws_sb_fixture_a_exact() {
    let r = woodruff_sawyer_sb(&[2.0, 1.0, 3.0, 10.0]).unwrap();
    assert!((r.pass_rate - 0.75).abs() < 1e-15);
    assert!((r.phi_half - 1.0 / 3.0).abs() < 1e-15);
    assert!((r.theta_half - 0.75).abs() < 1e-15);
    assert!((r.phi - 0.5).abs() < 1e-15);
    assert!((r.theta - 13.0 / 16.0).abs() < 1e-15);
    assert!((r.pi00 - 5.0 / 32.0).abs() < 1e-15);
    assert!((r.pi01 - 3.0 / 32.0).abs() < 1e-15);
    assert!((r.pi11 - 21.0 / 32.0).abs() < 1e-15);
    // Structural invariant (DERIVED, oracle-verified): cells sum to 1 with
    // the off-diagonal counted twice.
    assert!((r.pi00 + 2.0 * r.pi01 + r.pi11 - 1.0).abs() < 1e-15);
}

/// Table 4 regression pins, Total group and English subgroup. The integer
/// counts are ROUNDED normalized weights from the printed 3-digit table
/// (sums 10010 / 9990) — regression inputs, not exact representations.
/// Tolerance 0.0105 is a PAPER-TABLE tolerance (3-digit printing plus
/// 4-digit computations, source p. 18), not algorithm accuracy.
/// Reads r.phi / r.theta from both methods; kills gross formula swaps that
/// survive the synthetic fixtures (e.g. exchanging theta*/phi* outputs).
#[test]
fn ws_table4_paper_pins() {
    // Total group: printed proportions .078/.0265/.0265/.870 (already
    // symmetrized in the paper's Table 3 discussion), SB phi* = .84,
    // theta* = .97.
    let sb = woodruff_sawyer_sb(&[780.0, 265.0, 265.0, 8700.0]).unwrap();
    assert!((sb.phi - 0.84).abs() < 0.0105, "sb.phi = {}", sb.phi);
    assert!((sb.theta - 0.97).abs() < 0.0105, "sb.theta = {}", sb.theta);
    assert!((sb.phi_half - 0.72).abs() < 0.0105);
    assert!((sb.theta_half - 0.948).abs() < 0.0105);
    assert!((sb.pi00 - 0.089).abs() < 0.0105);
    assert!((sb.pi01 - 0.015).abs() < 0.0105);
    assert!((sb.pi11 - 0.881).abs() < 0.0105);
    // Accredited first-time row: .030/.019/.019/.931 -> phi* = .74,
    // theta* = .98.
    let sb2 = woodruff_sawyer_sb(&[300.0, 190.0, 190.0, 9310.0]).unwrap();
    assert!((sb2.phi - 0.74).abs() < 0.0105, "sb2.phi = {}", sb2.phi);
    assert!(
        (sb2.theta - 0.98).abs() < 0.0105,
        "sb2.theta = {}",
        sb2.theta
    );
    assert!((sb2.pi00 - 0.037).abs() < 0.0105);
    assert!((sb2.pi01 - 0.012).abs() < 0.0105);
    assert!((sb2.pi11 - 0.938).abs() < 0.0105);
}

/// Normal-method exact orthant anchor (NOT from the paper): mean 0, sd 1,
/// cut 0, r_half = 1/3 -> r_SB = 1/2, q = 1/2, and Sheppard's orthant
/// formula gives pi*00 = 1/4 + asin(1/2)/(2 pi) = 1/3 exactly, so
/// theta* = 2/3 and phi* = 1/3. Tolerance 1e-6 per the bvn_upper and erfc
/// accuracy contracts (the crate's rational erfc has ~1.5e-8 error even at
/// z = 0, so pass_rate is pinned at 1e-6, not exactly). Kills MU5 (using r unstepped: pi*00 = 0.30409) and MU6
/// (upper-tail q = 1 - Phi(Kq): breaks pi01/pi11 asymmetrically for
/// nonzero cuts — see ws_normal_float_pins). Reads r.pi00, r.theta, r.phi,
/// r.pass_rate, r.phi_half (NaN contract).
#[test]
fn ws_normal_orthant_exact() {
    let r = woodruff_sawyer_normal(0.0, 1.0, 0.0, 1.0 / 3.0).unwrap();
    assert!((r.pass_rate - 0.5).abs() < 1e-6);
    assert!((r.pi00 - 1.0 / 3.0).abs() < 1e-6);
    assert!((r.theta - 2.0 / 3.0).abs() < 1e-6);
    assert!((r.phi - 1.0 / 3.0).abs() < 1e-6);
    assert!(r.phi_half.is_nan());
    assert!(r.theta_half.is_nan());
}

/// Normal-method float pins (mpmath oracle, tol 1e-6): mean 100, sd 15,
/// cut 85, r_half = 0.6 -> r_SB = 0.75, q = Phi(-1) = 0.15865525393145705.
/// Asymmetric cut (q != 1/2) so MU6 (upper-tail q) shifts q to 0.84134 and
/// every downstream value. Reads all probability fields plus phi/theta.
#[test]
fn ws_normal_float_pins() {
    let r = woodruff_sawyer_normal(100.0, 15.0, 85.0, 0.6).unwrap();
    assert!((r.pass_rate - 0.84134474606854295).abs() < 1e-6);
    assert!((r.pi00 - 0.090456950720012368).abs() < 1e-6);
    assert!((r.pi01 - 0.068198303211444683).abs() < 1e-6);
    assert!((r.pi11 - 0.77314644285709827).abs() < 1e-6);
    assert!((r.theta - 0.86360339357711063).abs() < 1e-6);
    assert!((r.phi - 0.48908915212993364).abs() < 1e-6);
}

/// Error contract. Each arm reads the Err/Ok discriminant returned by the
/// crate. Also pins that a negative half-test phi passes through (not an
/// error) and that the SB phi = -1 singularity errs.
#[test]
fn ws_error_contract() {
    // SB method
    assert!(woodruff_sawyer_sb(&[1.0, 2.0, 3.0]).is_err());
    assert!(woodruff_sawyer_sb(&[1.0, -1.0, 2.0, 3.0]).is_err());
    assert!(woodruff_sawyer_sb(&[1.0, f64::NAN, 2.0, 3.0]).is_err());
    assert!(woodruff_sawyer_sb(&[0.0, 0.0, 0.0, 0.0]).is_err());
    // margin q = 0 (no off-diagonal, all pass)
    assert!(woodruff_sawyer_sb(&[0.0, 0.0, 0.0, 5.0]).is_err());
    // phi = -1 singularity: pi01_s = 1/2, p = q = 1/2, 2pq = pi01_s
    assert!(woodruff_sawyer_sb(&[0.0, 1.0, 1.0, 0.0]).is_err());
    // overflow-to-inf total is rejected, not silently normalized
    assert!(woodruff_sawyer_sb(&[f64::MAX, f64::MAX, 1.0, 1.0]).is_err());
    // negative phi passes through: (5,4,4,5) -> pi01_s = 2/9 > pq = 25/121?
    // Use (1,4,4,1): pi01_s = 2/5, p = 1/2, phi_half = 1 - (2/5)/(1/4) < 0.
    let neg = woodruff_sawyer_sb(&[1.0, 4.0, 4.0, 1.0]).unwrap();
    assert!(neg.phi_half < 0.0 && neg.phi < 0.0);
    // Normal method
    assert!(woodruff_sawyer_normal(f64::NAN, 1.0, 0.0, 0.5).is_err());
    assert!(woodruff_sawyer_normal(0.0, 0.0, 0.0, 0.5).is_err());
    assert!(woodruff_sawyer_normal(0.0, -1.0, 0.0, 0.5).is_err());
    assert!(woodruff_sawyer_normal(0.0, 1.0, f64::INFINITY, 0.5).is_err());
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, 1.5).is_err());
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, f64::NAN).is_err());
    // r_half = 1 -> r_SB = 1 (not strictly inside (-1, 1))
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, 1.0).is_err());
    // r_half < -1/3 -> r_SB < -1
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, -0.5).is_err());
    // r_half = -1/3 -> r_SB = -1 exactly
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, -1.0 / 3.0).is_err());
    // quadrature limit: r_half close enough to 1 that sqrt(1-r_SB^2) < 1e-4
    assert!(woodruff_sawyer_normal(0.0, 1.0, 0.0, 0.9999999999).is_err());
    // tiny sd -> Kq overflows or q rounds to 0/1
    assert!(woodruff_sawyer_normal(0.0, 1e-300, 1e300, 0.5).is_err());
    // cut far outside range -> q rounds to 0
    assert!(woodruff_sawyer_normal(0.0, 1.0, -50.0, 0.5).is_err());
}

/// MC-500: 500 random SB tables; structural invariants read from crate
/// outputs each rep: cells sum to 1 (off-diagonal twice), theta* <= 1 and
/// theta* = p^2 + q^2 + 2 p q phi* (identity vs the crate's own pass_rate
/// and phi outputs; note phi* < -1 and hence theta* < 0 are REACHABLE for
/// worse-than-chance tables since SB step-up of phi in (-1, 0) diverges
/// downward — so no lower bound is asserted), phi* =
/// 2 phi_half/(1 + phi_half) (kills any divergence between the eq. 5
/// single-expression form and the step-up), pass_rate in (0,1).
#[test]
#[ignore]
fn ws_mc_500() {
    let mut rng = Lcg::new(0x5EED_2026);
    let mut done = 0usize;
    while done < 500 {
        let n00 = (rng.unif() * 50.0).floor() + 1.0;
        let n01 = (rng.unif() * 20.0).floor();
        let n10 = (rng.unif() * 20.0).floor();
        let n11 = (rng.unif() * 80.0).floor() + 1.0;
        let r = match woodruff_sawyer_sb(&[n00, n01, n10, n11]) {
            Ok(r) => r,
            Err(_) => continue,
        };
        assert!((r.pi00 + 2.0 * r.pi01 + r.pi11 - 1.0).abs() < 1e-12);
        assert!(r.theta <= 1.0 + 1e-12);
        let p = r.pass_rate;
        let q = 1.0 - p;
        assert!((r.theta - (p * p + q * q + 2.0 * p * q * r.phi)).abs() < 1e-12);
        assert!(r.pass_rate > 0.0 && r.pass_rate < 1.0);
        let sb_identity = 2.0 * r.phi_half / (1.0 + r.phi_half);
        assert!((r.phi - sb_identity).abs() < 1e-12);
        done += 1;
    }
}
