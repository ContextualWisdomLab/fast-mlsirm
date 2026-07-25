//! Sympson-Hetter exposure control tests.
//!
//! Mutation-kill audit (executed kills recorded in the PR evidence):
//! every assert below reads crate outputs (`SympsonHetterResult` fields or
//! the returned `Err`); no assert recomputes the algorithm locally.
//!
//! - M1 gate flip (`u <= k` -> `u >= k` for k < 1): killed by
//!   `sh_controls_max_exposure` (a flipped gate rejects with probability k,
//!   so calibration cannot pull max exposure to the target).
//! - M2 update denominator (`r/P(S)` -> `r/P(A)`): killed by
//!   `sh_controls_max_exposure` (the wrong denominator weakens the filter and
//!   the reported max exposure stays above r_max + tol).
//! - M3 rejected item NOT blocked (usable[s] stays true on reject): killed by
//!   non-termination (EXECUTED: the mutant diverges — the rejected top item is
//!   re-encountered, s_count inflates across cycles, k -> 0, and encounter
//!   counts explode; the suite that normally finishes in ~1.4 s did not finish
//!   in 7+ minutes and was stopped). The discriminating anchor is the
//!   calibration loop itself, not an assert; documented rather than hidden.
//! - M4 swapped S/A bookkeeping: killed by `sh_counting_identities`
//!   (P(A) <= P(S) fails, and sum of exposure != test_length).
//! - Denominator mutants (counts divided by n_items or test_length instead
//!   of n_simulees): killed by `sh_counting_identities` (sum P(A) == L is an
//!   exact counting identity, not an MC approximation).
//!
//! Known limitation (documented, not hidden): with `r_max = 1` the gate is
//! skipped entirely, so gate-only mutants are invisible to
//! `sh_rmax_one_is_unconstrained`; the discriminating anchor for the gate is
//! `sh_controls_max_exposure`.

use crate::exposure::{
    a_stratified, eap_interim, kl_information, kl_select, p3pl, sympson_hetter, AStratifiedConfig,
    Lcg, SympsonHetterConfig,
};

fn pool30() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Deterministic, asymmetric 30-item 2PL pool: a in [0.6, 2.0],
    // b spread over [-2.4, 2.4] with an uneven pattern so max-info CAT
    // concentrates selections and exposure control has real work to do.
    let n = 30;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f64;
        a.push(0.6 + 1.4 * ((x * 0.37).sin().abs()));
        b.push(-2.4 + 4.8 * (x / (n as f64 - 1.0)) + 0.3 * (x * 0.71).sin());
    }
    (a, b, vec![0.0; n])
}

fn base_cfg() -> SympsonHetterConfig {
    SympsonHetterConfig {
        r_max: 0.25,
        test_length: 5,
        n_simulees: 1500,
        max_iter: 12,
        tol: 0.02,
        seed: 42,
        q_theta: 31,
    }
}

#[test]
fn sh_controls_max_exposure() {
    let (a, b, c) = pool30();
    // Uncontrolled baseline: r_max = 1 leaves k = 1 everywhere.
    let free = sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            r_max: 1.0,
            max_iter: 1,
            ..base_cfg()
        },
    )
    .expect("uncontrolled run");
    // The pool is deliberately peaked: without control some item must be
    // administered far above the 0.25 target (reads crate exposure output).
    assert!(
        free.max_exposure > 0.35,
        "baseline max exposure {} unexpectedly low; pool no longer discriminates the gate",
        free.max_exposure
    );

    let ctl = sympson_hetter(&a, &b, &c, &base_cfg()).expect("controlled run");
    assert!(
        ctl.converged,
        "calibration did not converge: history {:?}",
        ctl.history_max_exposure
    );
    assert!(
        ctl.max_exposure <= 0.25 + 0.02 + 1e-12,
        "max exposure {} above target",
        ctl.max_exposure
    );
    // Control must actually reduce the crate-reported max exposure.
    assert!(ctl.max_exposure < free.max_exposure);
    // Some item must have been throttled (k < 1) for the reduction to be
    // attributable to the gate rather than to chance.
    assert!(ctl.k.iter().any(|&v| v < 1.0));
}

#[test]
fn sh_counting_identities() {
    let (a, b, c) = pool30();
    let r = sympson_hetter(&a, &b, &c, &base_cfg()).expect("run");
    let n_items = a.len();
    assert_eq!(r.k.len(), n_items);
    assert_eq!(r.exposure.len(), n_items);
    assert_eq!(r.selection.len(), n_items);
    // Exact counting identity: every simulee gets exactly L items, so the
    // exposure rates (crate outputs) sum to L up to f64 summation error.
    let sum_expo: f64 = r.exposure.iter().sum();
    assert!(
        (sum_expo - 5.0).abs() < 1e-9,
        "sum of exposure rates {} != test_length",
        sum_expo
    );
    // Administration requires selection, per item.
    for i in 0..n_items {
        assert!(
            r.exposure[i] <= r.selection[i] + 1e-12,
            "item {}: P(A) {} > P(S) {}",
            i,
            r.exposure[i],
            r.selection[i]
        );
    }
    // k stays in (0, 1].
    for (i, &v) in r.k.iter().enumerate() {
        assert!(v > 0.0 && v <= 1.0, "k[{}] = {} out of (0, 1]", i, v);
    }
    // max_exposure is the max of the reported exposure vector and the last
    // history entry (internal consistency of the returned struct).
    let max_from_vec = r.exposure.iter().cloned().fold(0.0_f64, f64::max);
    assert_eq!(r.max_exposure, max_from_vec);
    assert_eq!(r.max_exposure, *r.history_max_exposure.last().unwrap());
    assert_eq!(r.history_max_exposure.len(), r.n_iter);
}

#[test]
fn sh_rmax_one_is_unconstrained() {
    let (a, b, c) = pool30();
    let r = sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            r_max: 1.0,
            ..base_cfg()
        },
    )
    .expect("run");
    // r_max = 1 is trivially satisfied: one cycle, converged, all k = 1
    // (kills mutants that update k unconditionally), and selection ==
    // exposure exactly (every selected item is administered; both are crate
    // outputs).
    assert!(r.converged);
    assert_eq!(r.n_iter, 1);
    assert!(r.k.iter().all(|&v| v == 1.0));
    assert_eq!(r.exposure, r.selection);
}

#[test]
fn sh_deterministic_under_seed() {
    let (a, b, c) = pool30();
    let r1 = sympson_hetter(&a, &b, &c, &base_cfg()).expect("run 1");
    let r2 = sympson_hetter(&a, &b, &c, &base_cfg()).expect("run 2");
    assert_eq!(r1.k, r2.k);
    assert_eq!(r1.exposure, r2.exposure);
    assert_eq!(r1.history_max_exposure, r2.history_max_exposure);
    // A different seed must actually change the simulation (kills mutants
    // that ignore the seed).
    let r3 = sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            seed: 43,
            ..base_cfg()
        },
    )
    .expect("run 3");
    assert_ne!(r1.exposure, r3.exposure);
}

#[test]
fn sh_input_validation() {
    let (a, b, c) = pool30();
    let cfg = base_cfg();
    // Infeasible target: r_max below L / n_items (counting identity).
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            r_max: 0.1,
            test_length: 5,
            ..cfg.clone()
        }
    )
    .is_err());
    // Mismatched lengths, empty pool, bad domains.
    assert!(sympson_hetter(&a[..29], &b, &c, &cfg).is_err());
    assert!(sympson_hetter(&[], &[], &[], &cfg).is_err());
    let mut bad_a = a.clone();
    bad_a[0] = f64::NAN;
    assert!(sympson_hetter(&bad_a, &b, &c, &cfg).is_err());
    let mut bad_c = c.clone();
    bad_c[3] = 1.0;
    assert!(sympson_hetter(&a, &b, &bad_c, &cfg).is_err());
    for r_max in [0.0, -0.2, 1.5, f64::NAN] {
        assert!(sympson_hetter(
            &a,
            &b,
            &c,
            &SympsonHetterConfig {
                r_max,
                ..cfg.clone()
            }
        )
        .is_err());
    }
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            test_length: 0,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            test_length: 31,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            n_simulees: 0,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            max_iter: 0,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            q_theta: 2,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            tol: f64::NAN,
            ..cfg
        }
    )
    .is_err());
}

// Exact-boundary policy pin: `r_max == test_length / n_items` passes the
// necessary feasibility bound, but the no-forced-administration policy means
// the stochastic gate can exhaust the pool mid-test; the crate then returns
// the documented pool-exhausted error rather than forcing an item. This
// regression reads the crate `Err` (round-2 impl-review reproducer) and
// fails if the policy silently changes to forced administration or the
// validation starts rejecting the exact boundary.
#[test]
fn sh_exact_feasibility_boundary_policy() {
    let a = vec![1.0, 1.1, 1.2, 1.3];
    let b = vec![-1.0, -0.3, 0.3, 1.0];
    let c = vec![0.0; 4];
    let cfg = SympsonHetterConfig {
        r_max: 0.5, // == test_length / n_items exactly
        test_length: 2,
        n_simulees: 200,
        max_iter: 5,
        tol: 0.02,
        seed: 7,
        q_theta: 3,
    };
    let err = sympson_hetter(&a, &b, &c, &cfg).unwrap_err();
    assert!(
        err.contains("item pool exhausted"),
        "expected the documented pool-exhausted policy error, got: {}",
        err
    );
    // Just below the bound is rejected up front by validation instead.
    let below = SympsonHetterConfig {
        r_max: 0.5 - 1e-12,
        ..cfg
    };
    let err2 = sympson_hetter(&a, &b, &c, &below).unwrap_err();
    assert!(err2.contains("infeasible"), "got: {}", err2);
}

// >= 500-replication Monte Carlo: across seeds the calibrated max exposure
// stays at the target (within MC noise) and the counting identity holds in
// every replication. Run with `cargo test -- --ignored`.
#[test]
#[ignore]
fn sh_monte_carlo_500() {
    let (a, b, c) = pool30();
    let mut worst = 0.0_f64;
    let mut sum_max = 0.0;
    let reps = 500;
    for rep in 0..reps {
        let r = sympson_hetter(
            &a,
            &b,
            &c,
            &SympsonHetterConfig {
                n_simulees: 1500,
                seed: 1000 + rep as u64,
                ..base_cfg()
            },
        )
        .expect("run");
        let sum_expo: f64 = r.exposure.iter().sum();
        assert!((sum_expo - 5.0).abs() < 1e-9, "rep {}: identity broke", rep);
        sum_max += r.max_exposure;
        worst = worst.max(r.max_exposure);
    }
    let mean_max = sum_max / reps as f64;
    assert!(
        mean_max <= 0.25 + 0.025,
        "mean calibrated max exposure {} drifted above target",
        mean_max
    );
    // Individual replications may sit above target (convergence is not
    // guaranteed; van der Linden, 2003) but not wildly so at these sizes.
    assert!(worst <= 0.25 + 0.08, "worst-case max exposure {}", worst);
}

// ===================== a-stratified multistage CAT =====================
//
// Mutation-kill audit (executed kills recorded in the PR evidence): every
// assert reads crate outputs (`AStratifiedResult` fields or the returned
// `Err`); no assert recomputes the algorithm locally.
//
// - M1 selection rule (b-matching -> max information within the stratum):
//   EXECUTED, killed by `as_bmatching_not_maxinfo` (the fixture makes the
//   two rules pick different items and reads the crate exposure vector).
// - M2 stratification direction (ascending -> descending a): EXECUTED,
//   killed by `as_stratum_assignment_and_remainder` (returned stratum
//   vector flips) and `as_bmatching_not_maxinfo`.
// - M3 remainder placement (first strata -> last strata): EXECUTED, killed
//   by `as_stratum_assignment_and_remainder` (7 items, 3 strata pins
//   [3,2,2]) and `as_per_stratum_counting_identity` (stage lengths shift).
// - M4 stage confinement removed (b-matching over the whole pool): EXECUTED,
//   killed by `as_bmatching_not_maxinfo` (stage 1 grabs the perfect-match
//   high-a item) and `as_per_stratum_counting_identity`.
// - M5 tie-break direction (`i < j` -> `i > j` on equal b-distance):
//   EXECUTED, killed by `as_tiebreak_lowest_index` (exact-tie fixture
//   b = [-0.5, 0.5, ...] at theta_hat = 0; the mutant administers item 1
//   instead of item 0 and the returned exposure vector flips).
//
// Known limitation (documented, not hidden): the per-stratum counting
// identity holds under a stage-order reversal that swaps equal-length
// stages; the discriminating anchor for stage ORDER is
// `as_stratum_assignment_and_remainder` plus `as_bmatching_not_maxinfo`,
// whose fixtures make the first administered item stratum-0-only.

fn as_cfg() -> AStratifiedConfig {
    AStratifiedConfig {
        n_strata: 4,
        test_length: 10,
        n_simulees: 300,
        seed: 42,
        q_theta: 31,
    }
}

#[test]
fn as_per_stratum_counting_identity() {
    let (a, b, c) = pool30();
    let r = a_stratified(&a, &b, &c, &as_cfg()).expect("run");
    assert_eq!(r.exposure.len(), 30);
    assert_eq!(r.stratum.len(), 30);
    // 10 items over 4 stages, first-remainder policy: [3, 3, 2, 2].
    assert_eq!(r.stage_lengths, vec![3, 3, 2, 2]);
    // Exact per-stratum counting identity: every simulee takes exactly
    // stage_lengths[k] items from stratum k (all values read from the
    // crate result).
    for k in 0..4 {
        let sum_k: f64 = r
            .exposure
            .iter()
            .zip(&r.stratum)
            .filter(|(_, &s)| s == k)
            .map(|(&e, _)| e)
            .sum();
        assert!(
            (sum_k - r.stage_lengths[k] as f64).abs() < 1e-9,
            "stratum {}: exposure sum {} != stage length {}",
            k,
            sum_k,
            r.stage_lengths[k]
        );
    }
    let total: f64 = r.exposure.iter().sum();
    assert!((total - 10.0).abs() < 1e-9, "global sum {} != L", total);
    let max_from_vec = r.exposure.iter().cloned().fold(0.0_f64, f64::max);
    assert_eq!(r.max_exposure, max_from_vec);
    // theta stats are finite and rmse is nonnegative by construction of the
    // crate computation (reads crate outputs; guards NaN regressions).
    assert!(r.theta_rmse.is_finite() && r.theta_rmse >= 0.0);
    assert!(r.theta_bias.is_finite());
}

#[test]
fn as_stratum_assignment_and_remainder() {
    // Scrambled a with one exact tie (indices 1 and 5 share a = 0.9):
    // ascending stable order by (a, index) is
    // idx 3 (0.5), 1 (0.9), 5 (0.9), 0 (1.0), 6 (1.4), 2 (1.8), 4 (2.2)
    // and the 7 = 3+2+2 first-remainder partition gives strata
    // {3,1,5} -> 0, {0,6} -> 1, {2,4} -> 2.
    let a = vec![1.0, 0.9, 1.8, 0.5, 2.2, 0.9, 1.4];
    let b = vec![0.0; 7];
    let c = vec![0.0; 7];
    let cfg = AStratifiedConfig {
        n_strata: 3,
        test_length: 3,
        n_simulees: 5,
        seed: 1,
        q_theta: 11,
    };
    let r = a_stratified(&a, &b, &c, &cfg).expect("run");
    assert_eq!(r.stratum, vec![1, 0, 2, 0, 2, 0, 1]);
    assert_eq!(r.stage_lengths, vec![1, 1, 1]);
}

#[test]
fn as_bmatching_not_maxinfo() {
    // Stage 1 draws only from the low-a stratum {i0, i1}. At the initial
    // theta_hat = 0: b-matching picks i0 (|b| = 1 < 2), while maximum
    // information picks i1 (info 0.064 > 0.059) — the rules disagree, so a
    // max-info mutant flips the crate exposure vector. A no-stratification
    // mutant instead picks i2 (b = 0, perfect match) in stage 1.
    let a = vec![0.5, 0.6, 2.0, 2.5];
    let b = vec![-1.0, 2.0, 0.0, 5.0];
    let c = vec![0.0; 4];
    let cfg = AStratifiedConfig {
        n_strata: 2,
        test_length: 2,
        n_simulees: 50,
        seed: 9,
        q_theta: 21,
    };
    let r = a_stratified(&a, &b, &c, &cfg).expect("run");
    // Stage 1: i0 for every simulee (deterministic: selection depends only
    // on the initial theta_hat). Stage 2: i2 beats i3 at any theta in
    // [-4, 4]. All asserts read the crate exposure/stratum outputs.
    assert_eq!(r.stratum, vec![0, 0, 1, 1]);
    assert_eq!(r.exposure, vec![1.0, 0.0, 1.0, 0.0]);
}

#[test]
fn as_deterministic_under_seed() {
    let (a, b, c) = pool30();
    let r1 = a_stratified(&a, &b, &c, &as_cfg()).expect("run 1");
    let r2 = a_stratified(&a, &b, &c, &as_cfg()).expect("run 2");
    assert_eq!(r1.exposure, r2.exposure);
    assert_eq!(r1.theta_rmse, r2.theta_rmse);
    let r3 = a_stratified(
        &a,
        &b,
        &c,
        &AStratifiedConfig {
            seed: 43,
            ..as_cfg()
        },
    )
    .expect("run 3");
    // A different seed must actually change the simulation.
    assert_ne!(r1.exposure, r3.exposure);
}

#[test]
fn as_input_validation() {
    let (a, b, c) = pool30();
    let cfg = as_cfg();
    assert!(a_stratified(&a[..29], &b, &c, &cfg).is_err());
    assert!(a_stratified(&[], &[], &[], &cfg).is_err());
    let mut bad_a = a.clone();
    bad_a[0] = -0.5;
    assert!(a_stratified(&bad_a, &b, &c, &cfg).is_err());
    let mut bad_c = c.clone();
    bad_c[3] = 1.0;
    assert!(a_stratified(&a, &b, &bad_c, &cfg).is_err());
    for (n_strata, test_length) in [(0, 10), (11, 10), (4, 0), (4, 31)] {
        assert!(a_stratified(
            &a,
            &b,
            &c,
            &AStratifiedConfig {
                n_strata,
                test_length,
                ..cfg.clone()
            }
        )
        .is_err());
    }
    assert!(a_stratified(
        &a,
        &b,
        &c,
        &AStratifiedConfig {
            n_simulees: 0,
            ..cfg.clone()
        }
    )
    .is_err());
    assert!(a_stratified(&a, &b, &c, &AStratifiedConfig { q_theta: 2, ..cfg }).is_err());
}

#[test]
fn as_tiebreak_lowest_index() {
    // Exact b-distance tie at the deterministic initial theta_hat = 0:
    // |(-0.5) - 0| == |0.5 - 0| == 0.5 (exact in f64). All a equal, so the
    // single stratum lists items in original-index order; the crate must
    // administer item 0, not item 1. Reads the returned exposure vector.
    // Kills M5 (tie-break `i < j` -> `i > j`), which returns [0, 1, 0, 0].
    let a = [1.0, 1.0, 1.0, 1.0];
    let b = [-0.5, 0.5, 3.0, 4.0];
    let c = [0.0; 4];
    let r = a_stratified(
        &a,
        &b,
        &c,
        &AStratifiedConfig {
            n_strata: 1,
            test_length: 1,
            n_simulees: 1,
            seed: 7,
            q_theta: 21,
        },
    )
    .expect("tie fixture");
    assert_eq!(r.exposure, vec![1.0, 0.0, 0.0, 0.0]);
}

// >= 500-replication Monte Carlo: the design's headline property — exposure
// is more BALANCED than unconstrained max-info CAT — holds on average, the
// counting identity holds in every replication, and theta recovery stays
// reasonable. With the repository's deterministic initial theta_hat = 0 the
// first administered item is the same for every simulee under BOTH designs
// (its exposure is exactly 1), so max exposure cannot discriminate; the
// imbalance metric is the summed squared deviation of the exposure rates
// from the uniform rate L/n (a chi-square-type summary of crate outputs,
// in the spirit of the exposure chi-square used by Barrada et al., 2006).
// Run with `cargo test -- --ignored`.
#[test]
#[ignore]
fn as_monte_carlo_500() {
    let (a, b, c) = pool30();
    let uniform = 10.0 / 30.0;
    let imbalance = |e: &[f64]| -> f64 { e.iter().map(|&x| (x - uniform) * (x - uniform)).sum() };
    // Unconstrained max-info baseline via sympson_hetter with r_max = 1
    // (documented exact reduction; reads crate output).
    let free = sympson_hetter(
        &a,
        &b,
        &c,
        &SympsonHetterConfig {
            r_max: 1.0,
            test_length: 10,
            n_simulees: 1500,
            max_iter: 1,
            tol: 0.02,
            seed: 5,
            q_theta: 31,
        },
    )
    .expect("baseline");
    let free_imbalance = imbalance(&free.exposure);
    let reps = 500;
    let mut sum_imb = 0.0;
    let mut sum_rmse = 0.0;
    for rep in 0..reps {
        let r = a_stratified(
            &a,
            &b,
            &c,
            &AStratifiedConfig {
                n_simulees: 300,
                seed: 1000 + rep as u64,
                ..as_cfg()
            },
        )
        .expect("run");
        let total: f64 = r.exposure.iter().sum();
        assert!((total - 10.0).abs() < 1e-9, "rep {}: identity broke", rep);
        sum_imb += imbalance(&r.exposure);
        sum_rmse += r.theta_rmse;
    }
    let mean_imb = sum_imb / reps as f64;
    let mean_rmse = sum_rmse / reps as f64;
    assert!(
        mean_imb < free_imbalance,
        "a-stratified mean exposure imbalance {} not below max-info baseline {}",
        mean_imb,
        free_imbalance
    );
    assert!(
        mean_rmse < 0.7,
        "mean theta RMSE {} unexpectedly poor",
        mean_rmse
    );
}

// ===================== Chang-Ying (1996) KL information selection =====================
//
// Mutation-kill audit (executed kills recorded in the PR evidence). Every
// assert reads crate outputs (`kl_information` vectors, `KlSelectResult`
// fields, or returned `Err`); no assert recomputes the integral locally.
//
// - KL-M1 direction flip (`p0/p` -> `p/p0` inside both logs): killed by
//   `kl_pinned_oracles` (reverse-KL areas differ from the pinned constants).
// - KL-M2 drop the Q-term of the pointwise KL: killed by `kl_pinned_oracles`
//   and by `kl_fisher_small_delta_anchor` (the Taylor coefficient changes,
//   so the ratio gate fails).
// - KL-M3 `delta = r / n` instead of `r / sqrt(n)`: killed by
//   `kl_select_full_vector_oracle`, which asserts BOTH the returned `delta`
//   AND the full index vector against independent constants at n = 4
//   (an argmax-only test would NOT reliably catch this; that limitation is
//   disclosed here per the spec review).
// - KL-M4 argmax -> argmin: killed by `kl_select_full_vector_oracle` (the
//   pool has a unique maximum, item 1, and a unique minimum, item 4, and no
//   masking removes the minimum).
//
// Oracle provenance: every pinned constant was computed independently TWICE
// before implementation (adversarial spec review: 80-digit Decimal Simpson;
// implementer: 60-digit Decimal Simpson, 4000 panels); the two computations
// agree to 15 significant digits. Values are UNNORMALIZED areas (see the
// contract in the exposure.rs section comment).

/// Pinned KL-index areas, plus the 3PL mirrored-b asymmetry pair: with c > 0
/// the index is NOT invariant under b -> 2*theta0 - b, so a mirror-blind
/// implementation cannot pass. All asserts read kl_information outputs.
#[test]
fn kl_pinned_oracles() {
    let v = kl_information(&[1.2], &[0.3], &[0.0], 0.5, 1.0).unwrap();
    assert!(
        (v[0] - 0.114454883565329).abs() < 1e-9,
        "2PL area = {}",
        v[0]
    );
    let v = kl_information(&[1.0], &[0.0], &[0.2], 0.0, 0.5).unwrap();
    assert!(
        (v[0] - 0.00687308819864807).abs() < 1e-9,
        "3PL area = {}",
        v[0]
    );
    // Mirrored-b pair (asymmetry anchor).
    let hi = kl_information(&[1.0], &[0.7], &[0.2], 0.0, 0.5).unwrap()[0];
    let lo = kl_information(&[1.0], &[-0.7], &[0.2], 0.0, 0.5).unwrap()[0];
    assert!((hi - 0.00524190785695519).abs() < 1e-9, "b=+0.7 = {hi}");
    assert!((lo - 0.00666896390226957).abs() < 1e-9, "b=-0.7 = {lo}");
    assert!(lo > hi, "3PL c > 0 must break the b-mirror symmetry");
    // Extreme high-discrimination oracle (a = 20, delta = 3): tail
    // probabilities reach ~1e-26, so any hard probability clamp (e.g. at
    // 1e-12) truncates the log tails and underestimates the area (the
    // pre-fix clamped implementation returned 59.7296..., ~30% low).
    // Oracle recomputed independently: Decimal 60-digit Simpson converges
    // to 85.923363619982739465... at 2048-16384 panels. Kills any
    // reintroduced probability clamping or saturating ICC evaluation.
    let v = kl_information(&[20.0], &[0.0], &[0.0], 0.0, 3.0).unwrap();
    assert!(
        (v[0] - 85.9233636199827395).abs() < 1e-9,
        "extreme a=20 area = {}",
        v[0]
    );
    // Saturated finite inputs must stay finite (0 ln 0 = 0 convention):
    // a = 1e308 drives z to inf so ln Q0 = ln Q = -inf; an unguarded
    // 0 * (-inf - -inf) product returns NaN (round-2 review finding).
    let v = kl_information(&[1e308], &[0.0], &[0.0], 2.0, 1.0).unwrap();
    assert!(
        v[0].is_finite() && v[0] >= 0.0,
        "saturated area must be finite nonneg, got {}",
        v[0]
    );
}

/// Small-delta Fisher anchor: area / (I(theta0) * delta^3 / 3) -> 1. The
/// numerator reads the crate area; the denominator is the INDEPENDENT
/// closed-form limit (2PL Fisher information a^2 p q), not a recomputation
/// of the integral, so the anchor discriminates.
#[test]
fn kl_fisher_small_delta_anchor() {
    let (a, b, t0) = (1.2f64, 0.3f64, 0.5f64);
    let p = 1.0 / (1.0 + (-a * (t0 - b)).exp());
    let fisher = a * a * p * (1.0 - p);
    let delta = 1e-3;
    let area = kl_information(&[a], &[b], &[0.0], t0, delta).unwrap()[0];
    let ratio = area / (fisher * delta * delta * delta / 3.0);
    assert!((ratio - 1.0).abs() < 1e-4, "ratio = {ratio}");
}

/// Full-vector selection oracle at n = 4 (delta = 3/2): asserts the returned
/// delta, all five index values against independent constants, the argmax
/// (item 1, unique max), and masking (with item 1 administered the argmax
/// moves to item 3). Item 4 is the unique minimum and stays available, so
/// argmax -> argmin (KL-M4) and delta = r/n (KL-M3: delta would be 0.75 and
/// every index value changes) both fail here. Asserts read KlSelectResult.
#[test]
fn kl_select_full_vector_oracle() {
    let a = [0.8, 1.5, 1.0, 2.0, 0.6];
    let b = [-1.0, 0.4, 0.0, 1.2, 0.3];
    let c = [0.0, 0.0, 0.15, 0.1, 0.0];
    let expected = [
        0.139656556289429,
        0.562390595050374,
        0.194256509461839,
        0.349021047637362,
        0.0992541089721506,
    ];
    let r = kl_select(&a, &b, &c, &[false; 5], 0.25, 4, 3.0).unwrap();
    assert!((r.delta - 1.5).abs() < 1e-15, "delta = {}", r.delta);
    for i in 0..5 {
        assert!(
            (r.index[i] - expected[i]).abs() < 1e-9,
            "index[{i}] = {}",
            r.index[i]
        );
    }
    assert_eq!(r.selected, 1);
    // Masking: administered items keep their index but cannot be selected.
    let mask = [false, true, false, false, false];
    let r2 = kl_select(&a, &b, &c, &mask, 0.25, 4, 3.0).unwrap();
    assert_eq!(r2.selected, 3);
    assert!((r2.index[1] - expected[1]).abs() < 1e-9);
}

/// K(theta0 || theta0) = 0 and the area is strictly positive for delta > 0
/// (Gibbs / non-negativity). Reads crate outputs.
#[test]
fn kl_nonnegativity_and_zero_at_center() {
    let area = kl_information(&[1.0], &[0.0], &[0.0], 0.0, 2.0).unwrap()[0];
    assert!(area > 0.0);
    let tiny = kl_information(&[1.0], &[0.0], &[0.0], 0.0, 1e-9).unwrap()[0];
    // Floating rounding can leave the near-zero area at +/- a few ulps.
    assert!(tiny.abs() < 1e-15, "tiny = {tiny}");
}

/// Every rejection is a crate Err.
#[test]
fn kl_error_paths() {
    assert!(kl_information(&[], &[], &[], 0.0, 1.0).is_err()); // empty
    assert!(kl_information(&[1.0], &[0.0, 1.0], &[0.0], 0.0, 1.0).is_err()); // mismatch
    assert!(kl_information(&[1.0], &[0.0], &[0.0], f64::NAN, 1.0).is_err()); // theta0
    assert!(kl_information(&[1.0], &[0.0], &[0.0], 0.0, 0.0).is_err()); // delta <= 0
    assert!(kl_information(&[1.0], &[0.0], &[0.0], 0.0, f64::INFINITY).is_err());
    assert!(kl_information(&[-1.0], &[0.0], &[0.0], 0.0, 1.0).is_err()); // a <= 0
    assert!(kl_information(&[1.0], &[f64::NAN], &[0.0], 0.0, 1.0).is_err()); // non-finite
    assert!(kl_information(&[1.0], &[0.0], &[1.0], 0.0, 1.0).is_err()); // c >= 1
    assert!(kl_information(&[1.0], &[0.0], &[-0.1], 0.0, 1.0).is_err()); // c < 0
    let (a, b, c) = ([1.0], [0.0], [0.0]);
    assert!(kl_select(&a, &b, &c, &[false, false], 0.0, 1, 3.0).is_err()); // mask len
    assert!(kl_select(&a, &b, &c, &[false], 0.0, 0, 3.0).is_err()); // n = 0
    assert!(kl_select(&a, &b, &c, &[false], 0.0, 1, 0.0).is_err()); // r <= 0
    assert!(kl_select(&a, &b, &c, &[true], 0.0, 1, 3.0).is_err()); // exhausted
}

/// MC-500 (#[ignore]): paired-design CAT comparison, KL selection vs random
/// selection. 40-item deterministic 2PL pool, 500 simulees, test length 15,
/// COMMON random numbers: both arms share each simulee's true theta and the
/// same response-uniform stream indexed by administration position, so the
/// arms differ ONLY in the selection rule. First item is fixed (argmin |b|)
/// in both arms because delta = r/sqrt(n) needs n >= 1. Gates (adversarial
/// spec review ruling 5): RMSE_KL < 0.45 sanity ceiling and paired
/// superiority RMSE_KL + 0.01 < RMSE_random under the fixed seed
/// (seed-locked diagnostic, margin piloted before pinning). The KL arm's
/// selections read kl_select crate outputs each step.
#[test]
#[ignore]
fn kl_mc500_paired_cat_recovery() {
    let n_items = 40usize;
    let mut a = vec![0.0; n_items];
    let mut b = vec![0.0; n_items];
    let c = vec![0.0; n_items];
    for i in 0..n_items {
        a[i] = 0.5 + 1.5 * ((i % 8) as f64) / 7.0;
        b[i] = -2.0 + 4.0 * (i as f64) / ((n_items - 1) as f64);
    }
    let first = (0..n_items)
        .min_by(|&i, &j| b[i].abs().partial_cmp(&b[j].abs()).unwrap())
        .unwrap();
    let grid: Vec<f64> = (0..81).map(|q| -4.0 + 8.0 * (q as f64) / 80.0).collect();
    let log_prior: Vec<f64> = grid.iter().map(|t| -0.5 * t * t).collect();
    let test_len = 15usize;
    let n_sim = 500usize;
    let mut rng = Lcg(0xC4A6_1996);
    let mut scratch: Vec<f64> = Vec::with_capacity(grid.len());
    let (mut sse_kl, mut sse_rand) = (0.0f64, 0.0f64);
    for _ in 0..n_sim {
        let theta_true = rng.normal();
        let u: Vec<f64> = (0..test_len).map(|_| rng.next_f64()).collect();
        let pick: Vec<f64> = (0..test_len).map(|_| rng.next_f64()).collect();
        for arm in 0..2 {
            let mut used = vec![false; n_items];
            let mut responses: Vec<(usize, f64)> = Vec::with_capacity(test_len);
            let mut theta_hat = 0.0f64;
            for k in 0..test_len {
                let item = if k == 0 {
                    first
                } else if arm == 0 {
                    kl_select(&a, &b, &c, &used, theta_hat, k, 3.0)
                        .unwrap()
                        .selected
                } else {
                    let avail: Vec<usize> = (0..n_items).filter(|&i| !used[i]).collect();
                    avail[((pick[k] * avail.len() as f64) as usize).min(avail.len() - 1)]
                };
                used[item] = true;
                let p = p3pl(theta_true, a[item], b[item], c[item]);
                let y = if u[k] < p { 1.0 } else { 0.0 };
                responses.push((item, y));
                theta_hat = eap_interim(&a, &b, &c, &responses, &grid, &log_prior, &mut scratch);
            }
            let e = theta_hat - theta_true;
            if arm == 0 {
                sse_kl += e * e;
            } else {
                sse_rand += e * e;
            }
        }
    }
    let rmse_kl = (sse_kl / n_sim as f64).sqrt();
    let rmse_rand = (sse_rand / n_sim as f64).sqrt();
    assert!(rmse_kl < 0.45, "rmse_kl = {rmse_kl}");
    assert!(
        rmse_kl + 0.01 < rmse_rand,
        "paired superiority failed: kl = {rmse_kl}, random = {rmse_rand}"
    );
}
