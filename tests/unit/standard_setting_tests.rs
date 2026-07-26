//! Tests for `standard_setting::hofstee` (Hofstee compromise method,
//! ported from psychometricsGP `fn_plot_hofstee.R`; see module header).
//!
//! Oracle provenance: pinned values were computed with the implementer's
//! Python oracle and independently reproduced by the adversarial spec
//! review's own oracle (files/hofstee_spec_review.md); MU3/MU4 fixture
//! values were computed by the reviewer. Every assert below reads crate
//! outputs (`HofsteeResult` fields or the returned `Err` string).
//!
//! Mutation-kill map (executed kills logged in files/hofstee_spec.md):
//! - MU1 floor-binning (`v.floor()` instead of `v.ceil()`): killed by
//!   `main_pinned_oracle` (cut becomes 75.0 fallback) and
//!   `vertical_diagonal` (fail becomes 62.5).
//! - MU2 upward diagonal (Q1=(min_cut, min_fail)): killed by
//!   `main_pinned_oracle` (falls back to (75.0, 82.5, true)).
//! - MU3 fallback `<=` instead of strict `<`: killed by
//!   `fallback_strict_less_fixture` (fail 66.67 instead of 33.34).
//! - MU4 ceil2/floor2 swap: killed by `fallback_directed_rounding_fixture`
//!   (fail 66.66 instead of 66.67).
//! - MU5 `u` used twice for the crossing point: killed by
//!   `main_pinned_oracle` (crossing missed, falls back).

use super::{hofstee, HofsteeResult};

/// numpy default_rng(2026), round(clip(normal(68, 9, 40), 0, 100), 1).
const MAIN_SCORES: [f64; 40] = [
    60.9, 70.2, 50.9, 80.6, 73.7, 65.4, 65.2, 70.7, 65.6, 66.0, 74.5, 72.6, 67.4, 67.2, 69.4, 62.5,
    64.4, 72.9, 66.8, 55.6, 63.7, 73.9, 65.9, 66.7, 73.8, 84.4, 61.6, 80.1, 56.9, 69.6, 57.5, 80.2,
    75.5, 78.2, 60.0, 74.2, 63.3, 63.9, 72.6, 75.9,
];

fn main_fit() -> HofsteeResult {
    hofstee(&MAIN_SCORES, 62.5, 75.0, 0.0, 20.0).unwrap()
}

#[test]
fn main_pinned_oracle() {
    let r = main_fit();
    assert!(!r.failed);
    assert!(
        (r.cut_score - 62.804_878_048_780_488).abs() < 1e-12,
        "cut_score = {}",
        r.cut_score
    );
    assert!(
        (r.fail_rate - 19.512_195_121_951_219).abs() < 1e-12,
        "fail_rate = {}",
        r.fail_rate
    );
}

#[test]
fn main_cum_freq_pinned() {
    let r = main_fit();
    assert_eq!(r.cum_freq_percent.len(), 101);
    // Exact bit-for-bit R arithmetic order (count / n) * 100:
    assert_eq!(r.cum_freq_percent[62], 17.5);
    assert_eq!(r.cum_freq_percent[63], 20.0);
    // 23/40 * 100 in divide-first order prints ...49999999999999, NOT 57.5;
    // this pins mandated change 7 (multiply-first would give exactly 57.5).
    assert_eq!(r.cum_freq_percent[70], 57.499_999_999_999_99);
    assert_eq!(r.cum_freq_percent[100], 100.0);
    assert_eq!(r.cum_freq_percent[0], 0.0);
}

#[test]
fn fallback_high_pinned() {
    let scores = [30.0; 10];
    let r = hofstee(&scores, 62.5, 75.0, 0.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 62.5);
    assert_eq!(r.fail_rate, 100.0);
}

#[test]
fn fallback_low_pinned() {
    let scores = [95.0; 10];
    let r = hofstee(&scores, 62.5, 75.0, 5.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 75.0);
    assert_eq!(r.fail_rate, 0.0);
}

#[test]
fn vertical_diagonal() {
    // min_cut == max_cut (vertical diagonal) is supported; only the
    // zero-length diagonal is rejected.
    let r = hofstee(&MAIN_SCORES, 70.0, 70.0, 0.0, 100.0).unwrap();
    assert!(!r.failed);
    assert_eq!(r.cut_score, 70.0);
    assert!(
        (r.fail_rate - 57.499_999_999_999_993).abs() < 1e-12,
        "fail_rate = {}",
        r.fail_rate
    );
}

#[test]
fn fallback_strict_less_fixture() {
    // Reviewer-computed MU3 kill: 62.5 sits exactly at min_cut, strict '<'
    // counts 1 of 3 -> fr1 = 33.33.. > 20 -> ceil2 = 33.34. A '<=' mutant
    // counts 2 of 3 and reports 66.67.
    let r = hofstee(&[30.0, 62.5, 90.0], 62.5, 75.0, 0.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 62.5);
    assert_eq!(r.fail_rate, 33.34);
}

#[test]
fn fallback_directed_rounding_fixture() {
    // Reviewer-computed MU4 kill: fr1 = 66.66.. -> ceil2 = 66.67; a
    // floor2 mutant reports 66.66.
    let r = hofstee(&[30.0, 30.0, 90.0], 62.5, 75.0, 0.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 62.5);
    assert_eq!(r.fail_rate, 66.67);
}

#[test]
fn degenerate_probes() {
    // Single score at min_cut: no crossing, fr1 = 0 <= 20 -> max_cut branch.
    let r = hofstee(&[62.5], 62.5, 75.0, 0.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 75.0);
    assert_eq!(r.fail_rate, 100.0);

    let r = hofstee(&[50.0; 5], 62.5, 75.0, 0.0, 20.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 62.5);
    assert_eq!(r.fail_rate, 100.0);

    let r = hofstee(&[0.0, 100.0], 0.0, 100.0, 0.0, 100.0).unwrap();
    assert!(!r.failed);
    assert_eq!(r.cut_score, 50.0);
    assert_eq!(r.fail_rate, 50.0);
}

#[test]
fn collinear_overlap_rejected() {
    // Two scores 10 and 90: the ogive is flat at 50% for s in [10, 89].
    // A horizontal diagonal y = 50 over x in [40, 60] overlaps it
    // collinearly -> Err (reduced scope, mandated change 2).
    let err = hofstee(&[10.0, 90.0], 40.0, 60.0, 50.0, 50.0).unwrap_err();
    assert!(err.contains("collinear overlap"), "err = {err}");
}

#[test]
fn parallel_non_overlap_falls_back() {
    // Same flat ogive at 50%, horizontal diagonal at y = 30 (parallel,
    // NOT collinear): no crossing anywhere, so the R fallback runs.
    let r = hofstee(&[10.0, 90.0], 40.0, 60.0, 30.0, 30.0).unwrap();
    assert!(r.failed);
    assert_eq!(r.cut_score, 40.0);
    assert_eq!(r.fail_rate, 50.0);
}

#[test]
fn zero_length_diagonal_rejected() {
    let err = hofstee(&MAIN_SCORES, 70.0, 70.0, 20.0, 20.0).unwrap_err();
    assert!(err.contains("zero-length"), "err = {err}");
}

#[test]
fn error_paths() {
    assert!(hofstee(&[], 62.5, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[f64::NAN], 62.5, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[f64::INFINITY], 62.5, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[-0.1], 62.5, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[100.1], 62.5, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[50.0], f64::NAN, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[50.0], -1.0, 75.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[50.0], 62.5, 101.0, 0.0, 20.0).is_err());
    assert!(hofstee(&[50.0], 75.0, 62.5, 0.0, 20.0).is_err());
    assert!(hofstee(&[50.0], 62.5, 75.0, 20.0, 0.0).is_err());
}

/// Monte-Carlo invariants, 500 reps: never panics; a non-fallback point
/// lies on the diagonal's bounding box and between the bracketing ogive
/// ordinates read back from the crate output.
#[test]
#[ignore]
fn mc_500_invariants() {
    // Local LCG (crate Lcg structs are module-private); MINSTD constants.
    let mut state: u64 = 20_260_207;
    let mut next_f64 = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for _ in 0..500 {
        let n = 5 + (next_f64() * 195.0) as usize;
        let scores: Vec<f64> = (0..n)
            .map(|_| {
                // Box-Muller from two uniforms.
                let u1 = next_f64().max(1e-12);
                let u2 = next_f64();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                (60.0 + 15.0 * z).clamp(0.0, 100.0)
            })
            .collect();
        let a = next_f64() * 100.0;
        let b = next_f64() * 100.0;
        let (min_cut, max_cut) = if a <= b { (a, b) } else { (b, a) };
        let c = next_f64() * 100.0;
        let d = next_f64() * 100.0;
        let (min_fail, max_fail) = if c <= d { (c, d) } else { (d, c) };
        if min_cut == max_cut && min_fail == max_fail {
            continue;
        }
        let r = match hofstee(&scores, min_cut, max_cut, min_fail, max_fail) {
            Ok(r) => r,
            Err(e) => {
                assert!(e.contains("collinear overlap"), "unexpected err: {e}");
                continue;
            }
        };
        assert_eq!(r.cum_freq_percent.len(), 101);
        assert!(
            r.cum_freq_percent
                .windows(2)
                .all(|w| w[1] >= w[0] && w[0].is_finite()),
            "cum curve must be nondecreasing and finite"
        );
        assert_eq!(r.cum_freq_percent[100], 100.0);
        if !r.failed {
            assert!(r.cut_score >= min_cut - 1e-9 && r.cut_score <= max_cut + 1e-9);
            assert!(r.fail_rate >= min_fail - 1e-9 && r.fail_rate <= max_fail + 1e-9);
            let lo = r.cut_score.floor().clamp(0.0, 100.0) as usize;
            let hi = r.cut_score.ceil().clamp(0.0, 100.0) as usize;
            assert!(
                r.fail_rate >= r.cum_freq_percent[lo] - 1e-9
                    && r.fail_rate <= r.cum_freq_percent[hi] + 1e-9,
                "Hofstee point must lie on the ogive segment"
            );
        } else {
            assert!(r.cut_score == min_cut || r.cut_score == max_cut);
        }
    }
}
