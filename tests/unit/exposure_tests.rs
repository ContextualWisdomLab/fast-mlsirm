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
    a_stratified, ccat_select, eap_interim, kl_information, kl_select, owen_cat, owen_update, p3pl,
    sympson_hetter, AStratifiedConfig, Lcg, SympsonHetterConfig,
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

// ===================== Owen (1975) approximate Bayesian sequential CAT =====================
//
// Mutation-kill audit (kills executed and recorded in the PR evidence):
// every assert reads owen_update / owen_cat crate outputs. Pinned oracles
// come from the adversarial spec review's high-precision numerical
// integration of the EXACT posterior (agreement with the closed forms at
// ~1e-13); test tolerance 5e-7 absorbs the crate's erfc approximation error
// (|err| < 1.2e-7) while still killing O(0.01)+ formula mutations.
//
// - OWEN-M1 incorrect-update sign flip (mu + instead of -): killed by
//   owen_pinned_update_oracles (incorrect mean oracle) and owen_mirror_symmetry.
// - OWEN-M2 sig2/s -> sig2/s2 in the mean shift: killed by both pinned mean
//   oracles.
// - OWEN-M3 drop the D term in the variance update (K*(K+D) -> K*K): killed
//   by the pinned variance oracles.
// - OWEN-M4 selection argmin -> argmax: killed by owen_cat_trajectory_oracle
//   (administered order changes).

/// Pinned posterior-moment oracles at (a=1.5, b=0.3, mu=0.2, sig2=1.2),
/// independently verified by numerical integration of the exact posterior
/// (spec review oracle table). All asserts read owen_update outputs.
#[test]
fn owen_pinned_update_oracles() {
    let (mu, sig2) = owen_update(1.5, 0.3, 0.0, true, 0.2, 1.2).unwrap();
    assert!(
        (mu - 0.993708628794091).abs() < 5e-7,
        "c=0 correct mu = {mu}"
    );
    assert!(
        (sig2 - 0.627945890895211).abs() < 5e-7,
        "c=0 correct sig2 = {sig2}"
    );
    let (mu, sig2) = owen_update(1.5, 0.3, 0.0, false, 0.2, 1.2).unwrap();
    assert!(
        (mu - -0.500813523713129).abs() < 5e-7,
        "c=0 incorrect mu = {mu}"
    );
    assert!(
        (sig2 - 0.657719958655775).abs() < 5e-7,
        "c=0 incorrect sig2 = {sig2}"
    );
    let (mu, sig2) = owen_update(1.5, 0.3, 0.2, true, 0.2, 1.2).unwrap();
    assert!(
        (mu - 0.717701908086462).abs() < 5e-7,
        "c=0.2 correct mu = {mu}"
    );
    assert!(
        (sig2 - 0.969762981710486).abs() < 5e-7,
        "c=0.2 correct sig2 = {sig2}"
    );
}

/// Structural invariants that discriminate directional mutations: with c = 0
/// and mu = b the correct/incorrect updates are exact mirrors (equal-magnitude
/// opposite mean shifts, identical variances); both variances shrink; and as
/// c -> 1 a correct response carries no information so mu' -> mu. All asserts
/// read owen_update outputs.
#[test]
fn owen_mirror_symmetry_and_shrinkage() {
    let (mu_c, s2_c) = owen_update(1.3, 0.4, 0.0, true, 0.4, 0.9).unwrap();
    let (mu_i, s2_i) = owen_update(1.3, 0.4, 0.0, false, 0.4, 0.9).unwrap();
    // Tolerance 1e-6, not 1e-12: the crate's rational erfc approximation
    // (|err| < 1.2e-7) perturbs Phi(0) away from exactly 0.5, so the exact
    // mirror identity holds only up to approximation error. Still kills a
    // sign-flip mutation (shift magnitude ~0.5).
    assert!(
        ((mu_c - 0.4) + (mu_i - 0.4)).abs() < 1e-6,
        "mirror mean shifts: {mu_c} vs {mu_i}"
    );
    assert!((s2_c - s2_i).abs() < 1e-6, "mirror variances");
    assert!(s2_c < 0.9 && s2_i < 0.9, "variance must shrink at c = 0");
    // Guessing damps the correct-response information: mean shift and
    // variance reduction both smaller than at c = 0.
    let (mu_g, s2_g) = owen_update(1.3, 0.4, 0.9, true, 0.4, 0.9).unwrap();
    assert!(mu_g - 0.4 > 0.0 && mu_g - 0.4 < mu_c - 0.4, "c damps shift");
    assert!(s2_g > s2_c, "c damps variance reduction");
}

/// Pinned 4-step CAT trajectory on a 6-item 3PNO pool: administration order
/// and both moment traces (tolerance 1e-5: four compounded erfc-approximation
/// steps). Kills selection mutations (argmin -> argmax changes the order) and
/// any update mutation (traces change). All asserts read owen_cat outputs.
#[test]
fn owen_cat_trajectory_oracle() {
    let a = [0.9, 1.4, 1.1, 2.0, 0.7, 1.6];
    let b = [-1.2, 0.5, 0.0, 1.0, -0.4, 0.2];
    let c = [0.0, 0.1, 0.0, 0.15, 0.0, 0.0];
    let resp = [1u8, 0, 1, 0, 1, 1];
    let r = owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 4, None).unwrap();
    assert_eq!(r.administered, vec![2, 1, 5, 3]);
    let mu_exp = [
        0.5903867604819626,
        0.07540289320219584,
        0.48423427681776365,
        0.33188033525266003,
    ];
    let s2_exp = [
        0.6514434730476137,
        0.4123387131860715,
        0.27135396476526175,
        0.20724352639310067,
    ];
    for k in 0..4 {
        assert!(
            (r.mu_trace[k] - mu_exp[k]).abs() < 1e-5,
            "mu_trace[{k}] = {}",
            r.mu_trace[k]
        );
        assert!(
            (r.sig2_trace[k] - s2_exp[k]).abs() < 1e-5,
            "sig2_trace[{k}] = {}",
            r.sig2_trace[k]
        );
    }
    assert!((r.mu - mu_exp[3]).abs() < 1e-5);
    assert!((r.sig2 - s2_exp[3]).abs() < 1e-5);
    // Variance-threshold stopping (Owen's rule): a loose threshold stops
    // after the first item even though test_length allows more.
    let r = owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 4, Some(0.7)).unwrap();
    assert_eq!(r.administered.len(), 1);
    assert!(r.sig2 <= 0.7, "stop threshold respected, sig2 = {}", r.sig2);
}

/// Error paths: every case reads the returned Err.
#[test]
fn owen_error_paths() {
    assert!(owen_update(0.0, 0.0, 0.0, true, 0.0, 1.0).is_err());
    assert!(owen_update(1.0, 0.0, 1.0, true, 0.0, 1.0).is_err());
    assert!(owen_update(1.0, 0.0, -0.1, true, 0.0, 1.0).is_err());
    assert!(owen_update(1.0, f64::NAN, 0.0, true, 0.0, 1.0).is_err());
    assert!(owen_update(1.0, 0.0, 0.0, true, 0.0, 0.0).is_err());
    assert!(owen_update(1.0, 0.0, 0.0, true, f64::INFINITY, 1.0).is_err());
    let a = [1.0, 1.2];
    let b = [0.0, 0.5];
    let c = [0.0, 0.0];
    let resp = [1u8, 0];
    assert!(owen_cat(&[], &[], &[], &[], 0.0, 1.0, 1, None).is_err());
    assert!(owen_cat(&a, &b[..1], &c, &resp, 0.0, 1.0, 1, None).is_err());
    assert!(owen_cat(&a, &b, &c, &[2u8, 0], 0.0, 1.0, 1, None).is_err());
    assert!(owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 0, None).is_err());
    assert!(owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 3, None).is_err());
    assert!(owen_cat(&a, &b, &c, &resp, 0.0, 0.0, 1, None).is_err());
    assert!(owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 1, Some(0.0)).is_err());
    assert!(owen_cat(&a, &b, &c, &resp, 0.0, 1.0, 1, Some(f64::NAN)).is_err());
}

/// MC-500 (#[ignore]): Owen CAT ability recovery on a 40-item 2PL pool with
/// generated 3PNO responses, paired against a random-order arm using COMMON
/// random numbers (shared theta_true and per-item response uniforms), the
/// same design as the KL MC test. Gates read only crate outputs (final mu
/// per replicate).
#[test]
#[ignore]
fn owen_mc500_paired_recovery() {
    let n_items = 40usize;
    let mut a = Vec::with_capacity(n_items);
    let mut b = Vec::with_capacity(n_items);
    for i in 0..n_items {
        a.push(0.5 + 1.5 * ((i % 8) as f64) / 7.0);
        b.push(-2.0 + 4.0 * (i as f64) / ((n_items - 1) as f64));
    }
    let c = vec![0.0; n_items];
    let test_len = 15usize;
    let n_rep = 500usize;
    let mut rng = Lcg(0x0EE1_1975);
    let (mut sse_owen, mut sse_rand) = (0.0f64, 0.0f64);
    for rep in 0..n_rep {
        let theta = rng.normal();
        // Common random numbers: one uniform per item, shared across arms.
        let mut resp = vec![0u8; n_items];
        for i in 0..n_items {
            let p = c[i] + (1.0 - c[i]) * crate::exposure::norm_cdf(a[i] * (theta - b[i]));
            resp[i] = u8::from(rng.next_f64() < p);
        }
        let r = owen_cat(&a, &b, &c, &resp, 0.0, 1.0, test_len, None).unwrap();
        let e = r.mu - theta;
        sse_owen += e * e;
        // Random arm: deterministic pseudo-random order seeded by rep, same
        // responses, same Owen updates.
        let mut order: Vec<usize> = (0..n_items).collect();
        let mut sh = Lcg(0x5EED_0000 ^ rep as u64);
        for i in (1..n_items).rev() {
            let j = (sh.next_f64() * ((i + 1) as f64)) as usize;
            order.swap(i, j.min(i));
        }
        let (mut mu, mut s2) = (0.0f64, 1.0f64);
        for &item in order.iter().take(test_len) {
            let (m, v) = owen_update(a[item], b[item], c[item], resp[item] == 1, mu, s2).unwrap();
            mu = m;
            s2 = v;
        }
        let e = mu - theta;
        sse_rand += e * e;
    }
    let rmse_owen = (sse_owen / n_rep as f64).sqrt();
    let rmse_rand = (sse_rand / n_rep as f64).sqrt();
    assert!(rmse_owen < 0.45, "Owen RMSE = {rmse_owen}");
    assert!(
        rmse_owen + 0.01 < rmse_rand,
        "adaptive must beat random: {rmse_owen} vs {rmse_rand}"
    );
}

/// Regression for the impl-review tail-cancellation finding: at D = 8 the
/// subtractive form `1 - Phi(D)` cancels catastrophically (it returned
/// sig2' ~ 2.58, an INFLATED variance); the fixed `Phi(-D)` form must match
/// the exact-erfc reference (5.571034034310448, 0.507162441721855) computed
/// with math.erfc. Tolerance 1e-4: the crate erfc's ~1e-7 relative tail
/// error is amplified by L*(L-D). Both asserts read owen_update outputs;
/// the pre-fix mutant fails both by ~0.4 and ~2.07 respectively.
#[test]
fn owen_incorrect_tail_stability() {
    let mu0 = 11.313708498984761; // D = mu0 / sqrt(2) = 8
    let (mu, sig2) = owen_update(1.0, 0.0, 0.0, false, mu0, 1.0).unwrap();
    assert!((mu - 5.571034034310448).abs() < 1e-4, "tail mu = {mu}");
    assert!(
        (sig2 - 0.507162441721855).abs() < 1e-4,
        "tail sig2 = {sig2}"
    );
    assert!(
        sig2 < 1.0,
        "an observed response must not inflate the variance"
    );
}

/// Doc-claim anchor (round-2 review): a deep-tail incorrect response where
/// Phi(-D) underflows must return the documented degenerate-posterior Err,
/// never a silent garbage update. Reads the crate's Err.
#[test]
fn owen_deep_tail_returns_err() {
    assert!(owen_update(1.0, 0.0, 0.0, false, 141.421356237, 1.0).is_err());
}

// ===================== Kingsbury-Zara CCAT content balancing =====================
//
// Mutation-kill audit (kills executed, recorded in the PR evidence): every
// assert reads CcatSelectResult fields or the returned Err. Pinned oracles
// from the adversarial spec review (>= 30 significant digits, exact
// arithmetic); f64 tolerance 1e-12.
//
// - CCAT-M1 gap argmax -> argmin: killed by ccat_pinned_oracle (group flips
//   to 1).
// - CCAT-M2 within-group info argmax -> argmin: killed by ccat_pinned_oracle
//   (item 1 -> item 4).
// - CCAT-M3 zero-coverage priority dropped (always gap rule): killed by
//   ccat_zero_coverage_priority (gap rule would pick group 0, priority
//   demands group 2).
// - CCAT-M4 info formula drops the guessing factor ((P-c)/(1-c))^2: killed
//   by ccat_pinned_oracle info values.

fn ccat_pool() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
    (
        vec![1.0, 1.5, 0.8, 2.0, 1.2, 0.9],
        vec![-0.5, 0.2, 0.0, 0.8, -0.2, 0.4],
        vec![0.0, 0.1, 0.0, 0.2, 0.0, 0.0],
        vec![0, 0, 1, 1, 0, 1],
        vec![0.6, 0.4],
    )
}

/// Pinned oracle (spec review): both groups covered, group 0 has the larger
/// gap (0.1 vs -0.1); items 1 and 4 compete on information and item 1 wins.
/// Info values pinned to the review's exact-arithmetic computation.
#[test]
fn ccat_pinned_oracle() {
    let (a, b, c, g, t) = ccat_pool();
    let adm = [true, false, false, true, false, false];
    let r = ccat_select(&a, &b, &c, &g, &t, &adm, 0.1).unwrap();
    assert_eq!(r.group, 0);
    assert_eq!(r.selected, 1);
    assert!((r.discrepancy[0] - 0.1).abs() < 1e-12);
    assert!((r.discrepancy[1] - -0.1).abs() < 1e-12);
    assert!((r.info[1] - 0.451012779418390197920232968).abs() < 1e-12);
    assert!((r.info[4] - 0.348583393587808097407435360).abs() < 1e-12);
    assert!((r.info[3] - 0.280386779886269687918289656).abs() < 1e-12);
}

/// Discriminating content-balancing oracle (spec review): unconstrained
/// max-info would pick item 1 (group 0, I = 0.4510...), but group 1 has
/// zero coverage after administering item 0, so CCAT must pick group 1 and
/// its most informative item 3 (I = 0.2804 > I_5 = 0.1989 > I_2 = 0.1597).
/// Kills a "global max info" mutant of the whole feature.
#[test]
fn ccat_balancing_overrides_global_max_info() {
    let (a, b, c, g, t) = ccat_pool();
    let adm = [true, false, false, false, false, false];
    let r = ccat_select(&a, &b, &c, &g, &t, &adm, 0.1).unwrap();
    assert_eq!(r.group, 1);
    assert_eq!(r.selected, 3);
    // The globally most informative item is NOT the selection.
    let imax = (0..6)
        .filter(|&i| !adm[i])
        .max_by(|&x, &y| r.info[x].partial_cmp(&r.info[y]).unwrap())
        .unwrap();
    assert_eq!(imax, 1);
    assert_ne!(r.selected, imax);
}

/// Zero-coverage priority beats the raw gap rule (catR nextItem.R branch:
/// `min(empProp) == 0` short-circuits the gap comparison). 13-item pool,
/// 3 groups, 10 administered (1 from group 0, 9 from group 1, 0 from
/// group 2): gaps are [0.7, -0.8, 0.1], so the plain gap rule would choose
/// group 0, but group 2 is uncovered and must win.
#[test]
fn ccat_zero_coverage_priority() {
    let n = 13;
    let a = vec![1.0; n];
    let mut b = vec![0.0; n];
    b[11] = 0.3; // group-2 items differ so item choice is non-trivial
    b[12] = 0.1;
    let c = vec![0.0; n];
    // groups: 0,0,0 | 1 x 9 | 2,2  -> targets [0.8, 0.1, 0.1]
    let mut g = vec![0usize; 3];
    g.extend(std::iter::repeat(1usize).take(8));
    g.extend([2usize, 2]);
    let t = vec![0.8, 0.1, 0.1];
    let mut adm = vec![false; n];
    adm[0] = true;
    adm[2] = true; // two from group 0
    for x in adm.iter_mut().take(11).skip(3) {
        *x = true; // eight from group 1
    }
    // k = 10, k_g = [2, 8, 0] -> f = [0.2, 0.8, 0.0], gaps [0.6, -0.7, 0.1]:
    // the plain gap rule would choose group 0, but group 2 is uncovered.
    let r = ccat_select(&a, &b, &c, &g, &t, &adm, 0.0).unwrap();
    assert_eq!(r.group, 2, "uncovered group must have priority");
    // gap rule alone would have chosen group 0:
    let gap_argmax = (0..3)
        .max_by(|&x, &y| r.discrepancy[x].partial_cmp(&r.discrepancy[y]).unwrap())
        .unwrap();
    assert_eq!(gap_argmax, 0);
    // within group 2, item 12 (b = 0.1 closer to theta0 = 0 than b = 0.3)
    // has higher information at equal a, c:
    assert_eq!(r.selected, 12);
    assert!(r.info[12] > r.info[11]);
}

/// Exhausted groups are skipped: group 1 has the max gap but no
/// unadministered items, so the next-best eligible group is chosen.
#[test]
fn ccat_exhausted_group_skipped() {
    let a = vec![1.0, 1.0, 1.0, 1.0];
    let b = vec![0.0, 0.1, 0.2, 0.3];
    let c = vec![0.0; 4];
    let g = vec![0usize, 0, 1, 1];
    let t = vec![0.2, 0.8];
    // Both group-1 items administered: k = 2, f = [0, 1], gaps [0.2, -0.2],
    // group 1 exhausted -> eligible only group 0... but k_0 = 0 so the
    // zero-priority rule also lands on group 0. Force covered case instead:
    let adm = [true, false, true, true];
    // k = 3, k_0 = 1, k_1 = 2 -> f = [1/3, 2/3], gaps [ -0.1333, 0.1333 ].
    // Group 1 has the max gap but is exhausted; group 0 must be chosen.
    let r = ccat_select(&a, &b, &c, &g, &t, &adm, 0.0).unwrap();
    assert_eq!(r.group, 0);
    assert_eq!(r.selected, 1);
    assert!(r.discrepancy[1] > r.discrepancy[0], "gap favored group 1");
}

/// Error paths: every case reads the returned Err.
#[test]
fn ccat_error_paths() {
    let a = vec![1.0, 1.0];
    let b = vec![0.0, 0.0];
    let c = vec![0.0, 0.0];
    let g = vec![0usize, 1];
    let t = vec![0.5, 0.5];
    let adm = [false, false];
    assert!(ccat_select(&[], &[], &[], &[], &[], &[], 0.0).is_err());
    assert!(ccat_select(&a, &b[..1], &c, &g, &t, &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &c, &[0, 2], &t, &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &c, &g, &[0.5, 0.6], &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &c, &g, &[1.0, 0.0], &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &c, &g, &[], &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &c, &g, &t, &adm, f64::NAN).is_err());
    assert!(ccat_select(&a, &b, &c, &g, &t, &[true, true], 0.0).is_err());
    assert!(ccat_select(&[1.0, -1.0], &b, &c, &g, &t, &adm, 0.0).is_err());
    assert!(ccat_select(&a, &b, &[0.0, 1.0], &g, &t, &adm, 0.0).is_err());
}

/// MC-500 (#[ignore]): 500 random pools/masks; structural invariants of the
/// catR-reproduced rule, all read from CcatSelectResult: the selection is
/// unadministered and inside the chosen group; the chosen group is eligible;
/// if every eligible group is covered, the chosen group attains the maximal
/// eligible discrepancy; if some eligible group is uncovered, the chosen
/// group is the lowest-index uncovered eligible group; the selected item
/// maximizes info within the chosen group (lowest index on ties).
#[test]
#[ignore]
fn ccat_mc500_invariants() {
    let mut rng = Lcg(0x0CCA_1989);
    for rep in 0..500 {
        let n = 5 + (rng.next_f64() * 20.0) as usize;
        let n_groups = 2 + (rng.next_f64() * 3.0) as usize;
        let mut a = Vec::with_capacity(n);
        let mut b = Vec::with_capacity(n);
        let mut c = Vec::with_capacity(n);
        let mut g = Vec::with_capacity(n);
        for i in 0..n {
            a.push(0.5 + 1.5 * rng.next_f64());
            b.push(-2.0 + 4.0 * rng.next_f64());
            c.push(0.25 * rng.next_f64());
            g.push(if i < n_groups {
                i // guarantee every group is non-empty
            } else {
                (rng.next_f64() * n_groups as f64) as usize % n_groups
            });
        }
        let mut t: Vec<f64> = (0..n_groups).map(|_| 0.1 + rng.next_f64()).collect();
        let s: f64 = t.iter().sum();
        for x in t.iter_mut() {
            *x /= s;
        }
        let mut adm: Vec<bool> = (0..n).map(|_| rng.next_f64() < 0.5).collect();
        if adm.iter().all(|&x| x) {
            adm[rep % n] = false;
        }
        let theta0 = -1.0 + 2.0 * rng.next_f64();
        let r = ccat_select(&a, &b, &c, &g, &t, &adm, theta0).unwrap();
        assert!(!adm[r.selected], "rep {rep}: selected already administered");
        assert_eq!(g[r.selected], r.group, "rep {rep}: item outside group");
        let eligible: Vec<usize> = (0..n_groups)
            .filter(|&gg| (0..n).any(|i| g[i] == gg && !adm[i]))
            .collect();
        assert!(eligible.contains(&r.group), "rep {rep}: ineligible group");
        let k_g: Vec<usize> = (0..n_groups)
            .map(|gg| (0..n).filter(|&i| g[i] == gg && adm[i]).count())
            .collect();
        let uncovered: Vec<usize> = eligible
            .iter()
            .cloned()
            .filter(|&gg| k_g[gg] == 0)
            .collect();
        if let Some(&first) = uncovered.first() {
            assert_eq!(r.group, first, "rep {rep}: zero-coverage priority");
        } else {
            let best = eligible
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY, |m, gg| m.max(r.discrepancy[gg]));
            assert!(
                r.discrepancy[r.group] >= best - 1e-15,
                "rep {rep}: gap rule violated"
            );
        }
        for i in 0..n {
            if g[i] == r.group && !adm[i] {
                assert!(
                    r.info[r.selected] >= r.info[i]
                        || (r.info[r.selected] == r.info[i] && r.selected <= i),
                    "rep {rep}: info rule violated"
                );
            }
        }
    }
}

/// Regression (impl-review MAJOR): with c = 0 and an extreme finite theta0
/// the logistic underflows to P = 0 and the naive q/p * r^2 info form is
/// inf * 0 = NaN, which silently corrupted the within-group argmax (NaN
/// item stayed selected because `x > NaN` is false). The limit is 0, so
/// item 1 (P = 0.5, I = 0.25) must win and info[0] must be exactly 0.
#[test]
fn ccat_underflow_info_is_zero_not_nan() {
    let a = vec![1.0, 1.0];
    let b = vec![0.0, -1e308];
    let c = vec![0.0, 0.0];
    let g = vec![0usize, 0];
    let t = vec![1.0];
    let adm = [false, false];
    let r = ccat_select(&a, &b, &c, &g, &t, &adm, -1e308).unwrap();
    assert_eq!(r.selected, 1);
    assert!(r.info.iter().all(|x| x.is_finite()), "info must be finite");
    assert_eq!(r.info[0], 0.0);
    assert!((r.info[1] - 0.25).abs() < 1e-12);
}
