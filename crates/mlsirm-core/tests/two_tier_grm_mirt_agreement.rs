//! mirt agreement for the single-group polytomous two-tier GRM
//! (stage 4 of #1912).
//!
//! The committed dataset (`tests/fixtures/two_tier_grm_stage4/dataset.csv`)
//! and the mirt fit (`mirt_fixture.json`) are both produced by
//! `generate_mirt_fixture.R` (R 4.x, mirt 1.46.1,
//! `mirt::bfactor(..., model2 = <two primaries + COV>, itemtype = "graded")`).
//! This test fits the same data with the Rust two-tier EM at the same
//! quadrature density and compares primary/specific slopes (after aligning
//! the per-dimension reflection), boundary intercepts, the primary
//! correlation, and the observed-data log-likelihood.
//!
//! Intercept mapping (verified against the mirt fit, not assumed): mirt's
//! graded `d_k` columns are the SAME additive boundary intercepts the Rust
//! fitter reports as `threshold`, i.e.
//! `P(Y >= k | theta) = logistic(sum_p a_ip * theta_p + a_S * theta_S + d_k)`
//! (no sign flip, no division by `a`). The fixture records this convention
//! string alongside the numbers.
//!
//! Tolerance basis: with `n = 1,500` the MML standard errors of graded
//! slopes are on the order of 0.06-0.10, so two independent EM
//! implementations of the same MLE started from different initial values
//! should agree an order of magnitude below sampling noise. The asserted
//! bands (0.05 per slope, 0.05 per intercept, 0.02 on the primary
//! correlation, 1.0 loglik units at `|ll| ~ 1.9e4`) encode that, with the
//! measured differences reported in the stage-4 PR. Per the task, a
//! disagreement beyond tolerance is REPORTED (the assertion prints both
//! vectors), never tuned away.
//!
//! Measured at the committed fixture (test config `q_primary = q_specific =
//! 15`, `tol = 1e-7`): Rust `ll = -19239.1857` vs mirt `ll = -19239.1800`
//! (gap 0.0057); worst slope gap 0.0065; worst intercept gap 0.0016;
//! primary-correlation gap 0.0007 (Rust 0.2953 vs mirt 0.2961, truth 0.35).
//! The Rust EM converged in 24 sweeps.
//!
//! QUADRATURE (maintainer rule, 2026-09-16): node counts govern numerical
//! precision with a 121-per-dimension floor, chosen by precision convergence
//! (121 vs 241 vs ...). Node counts are required caller arguments with no
//! defaults (Project rule, #1929). The fixture's `QUADPTS = 15` and this
//! test's `q_* = 15` predate the removal of the fixed `SUPPORTED_Q` table
//! (#1929/#1945, which replaced it with an arbitrary-`n` Golub & Welsch,
//! 1969 rule); this mirt-agreement test is pinned to the committed R
//! fixture's own `QUADPTS`, so raising it would require regenerating the
//! fixture. `two_tier_grm_node_agreement.rs` adds the >= 121-node
//! numerical-agreement regression this comment used to defer. The counts
//! below are caller arguments (single knob), not hardcoded model constants.
//!
//! # References (APA 7th ed.)
//!
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. *Psychometrika, 75*(4), 581-612.
//! https://doi.org/10.1007/s11336-010-9178-0 (abstract read; full text not
//! accessible — no equation locator is drawn from it)
//!
//! Chalmers, R. P. (2026). mirt: Multidimensional item response theory
//! (Version 1.46.1) [R package].
//! https://cran.r-project.org/package=mirt (oracle software; the committed
//! fixture pins this exact version, and the `bfactor` help topic's
//! two-tier specification is read)

use mlsirm_core::two_tier_grm::{fit_two_tier_grm, TwoTierGrmConfig};
use std::collections::HashMap;

const N_ITEMS: usize = 10;
const N_PRIMARY: usize = 2;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const QUADPTS: usize = 15;
/// Row-major `n_items x n_primary` free-slope pattern: items 0-4 on
/// primary 0, items 5-9 on primary 1.
const PRIMARY_MAP: [bool; N_ITEMS * N_PRIMARY] = [
    true, false, //
    true, false, //
    true, false, //
    true, false, //
    true, false, //
    false, true, //
    false, true, //
    false, true, //
    false, true, //
    false, true,
];
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

// Bands from the measured fixture agreement (see module docs).
const MAX_SLOPE_GAP: f64 = 0.05;
const MAX_INTERCEPT_GAP: f64 = 0.05;
const MAX_RHO_GAP: f64 = 0.02;
const MAX_LOGLIK_GAP: f64 = 1.0;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/two_tier_grm_stage4")
}

fn parse_number_list(s: &str) -> Vec<f64> {
    s.split(|c: char| c == ',' || c.is_whitespace() || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|_| panic!("fixture number did not parse as f64: {t:?} in {s:?}"))
        })
        .collect()
}

fn fixture_numbers(text: &str, key: &str) -> Vec<f64> {
    let anchor = format!("\"{key}\"");
    let start = text
        .find(&anchor)
        .unwrap_or_else(|| panic!("fixture is missing key {key:?}"));
    let after = &text[start + anchor.len()..];
    let open = after.find('[').expect("fixture array must open with [");
    let mut depth = 0usize;
    let mut end = None;
    for (i, ch) in after[open..].char_indices() {
        if ch == '[' {
            depth += 1;
        } else if ch == ']' {
            depth -= 1;
            if depth == 0 {
                end = Some(open + i);
                break;
            }
        }
    }
    let end = end.expect("fixture array must close");
    parse_number_list(&after[open..=end])
}

fn fixture_scalar(text: &str, key: &str) -> f64 {
    let anchor = format!("\"{key}\"");
    let start = text
        .find(&anchor)
        .unwrap_or_else(|| panic!("fixture is missing key {key:?}"));
    let after = &text[start + anchor.len()..];
    let colon = after.find(':').expect("fixture entry must have a colon");
    let token: String = after[colon + 1..]
        .chars()
        .skip_while(|c| c.is_whitespace())
        .take_while(|c| {
            c.is_ascii_digit() || *c == '.' || *c == '-' || *c == '+' || *c == 'e' || *c == 'E'
        })
        .collect();
    token
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("fixture scalar {key:?} did not parse: {token:?}"))
}

fn load_dataset() -> Vec<usize> {
    let path = fixture_dir().join("dataset.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read mirt fixture dataset {path:?}: {e}"));
    let mut rows: Vec<Vec<usize>> = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue; // header / trailing newline
        }
        rows.push(
            line.split(',')
                .map(|c| {
                    c.trim()
                        .parse::<usize>()
                        .unwrap_or_else(|_| panic!("dataset cell did not parse: {c:?} ({path:?})"))
                })
                .collect(),
        );
    }
    assert!(
        !rows.is_empty() && rows[0].len() == N_ITEMS,
        "dataset must be n_persons x {N_ITEMS}: {path:?}"
    );
    rows.concat()
}

#[test]
fn two_tier_grm_agrees_with_mirt_bfactor_two_tier_graded() {
    let fixture_path = fixture_dir().join("mirt_fixture.json");
    let fixture = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "cannot read mirt fixture {fixture_path:?}: {e} — run \
             Rscript tests/fixtures/two_tier_grm_stage4/generate_mirt_fixture.R first"
        )
    });
    assert!(
        fixture.contains("logistic(sum_p a_ip * theta_p + a_S * theta_S + d_k)")
            || fixture.contains("logistic(sum_p a_ip*theta_p + a_S*theta_S + d_k)"),
        "fixture must record the verified intercept convention; got: {fixture}"
    );
    let mirt_a_p = fixture_numbers(&fixture, "mirt_a_primary");
    let mirt_a_s = fixture_numbers(&fixture, "mirt_a_specific");
    let mirt_d = fixture_numbers(&fixture, "mirt_d");
    let mirt_phi = fixture_numbers(&fixture, "mirt_phi");
    let mirt_ll = fixture_scalar(&fixture, "mirt_loglik");
    assert_eq!(
        mirt_a_p.len(),
        N_ITEMS * N_PRIMARY,
        "mirt primary slopes: {mirt_a_p:?}"
    );
    assert_eq!(
        mirt_a_s.len(),
        N_ITEMS,
        "mirt specific slopes: {mirt_a_s:?}"
    );
    assert_eq!(
        mirt_d.len(),
        N_ITEMS * (N_CAT - 1),
        "mirt intercepts: expected {} values",
        N_ITEMS * (N_CAT - 1)
    );
    assert_eq!(mirt_phi.len(), N_PRIMARY * N_PRIMARY, "mirt phi");
    assert!(mirt_ll.is_finite(), "mirt loglik must be finite");

    let y = load_dataset();
    let n_persons = y.len() / N_ITEMS;
    assert!(n_persons > 100, "fixture dataset must be non-trivial");

    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: true,
        q_primary: QUADPTS,
        q_specific: QUADPTS,
        max_iter: 2000,
        tol: 1e-7,
        n_starts: 1,
        seed: 0x2E54_1EED_2026_0916,
        newton_iter: 10,
        ridge: 1e-8,
    };
    let fit = fit_two_tier_grm(
        &y,
        None,
        &PRIMARY_MAP,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_PRIMARY,
        N_SPECIFIC,
        N_CAT,
        &cfg,
    )
    .expect("two-tier GRM fit on the mirt fixture dataset must succeed");
    assert!(
        fit.converged,
        "fixture fit must converge (termination: {})",
        fit.termination_reason
    );

    // Reflection alignment BEFORE the comparison: each implementation pins
    // its own latent orientation, so compare mirt against the Rust fit up to
    // the per-dimension sign both estimators canonicalize independently.
    // Align at the Rust anchors (comparing each estimator's own argmax can
    // false-flip when the argmax items differ across estimators).
    let mut mirt_a_p = mirt_a_p;
    let mut mirt_a_s = mirt_a_s;
    let mut mirt_phi = mirt_phi;
    let mut flipped = [false; N_PRIMARY];
    for d in 0..N_PRIMARY {
        let rust_anchor = (0..N_ITEMS)
            .filter(|&i| PRIMARY_MAP[i * N_PRIMARY + d])
            .max_by(|&i, &j| {
                fit.a_primary[i * N_PRIMARY + d]
                    .abs()
                    .total_cmp(&fit.a_primary[j * N_PRIMARY + d].abs())
            })
            .unwrap();
        if (fit.a_primary[rust_anchor * N_PRIMARY + d] > 0.0)
            != (mirt_a_p[rust_anchor * N_PRIMARY + d] > 0.0)
        {
            flipped[d] = true;
            for i in 0..N_ITEMS {
                if PRIMARY_MAP[i * N_PRIMARY + d] {
                    mirt_a_p[i * N_PRIMARY + d] = -mirt_a_p[i * N_PRIMARY + d];
                }
            }
        }
    }
    // A flipped primary dimension flips its phi row/column sign (the joint
    // (a_.d, theta_d, Phi[d, .]) sign flip leaves every likelihood invariant).
    for a in 0..N_PRIMARY {
        for b in 0..N_PRIMARY {
            if flipped[a] != flipped[b] {
                mirt_phi[a * N_PRIMARY + b] = -mirt_phi[a * N_PRIMARY + b];
            }
        }
    }
    let mut block_of: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &s) in SPECIFIC_MAP.iter().enumerate() {
        block_of.entry(s as usize).or_default().push(i);
    }
    for items in block_of.values() {
        let rust_anchor = items
            .iter()
            .max_by(|&&i, &&j| fit.a_specific[i].abs().total_cmp(&fit.a_specific[j].abs()))
            .copied()
            .unwrap();
        if (fit.a_specific[rust_anchor] > 0.0) != (mirt_a_s[rust_anchor] > 0.0) {
            for &i in items {
                mirt_a_s[i] = -mirt_a_s[i];
            }
        }
    }

    let mut worst_slope = 0.0f64;
    for i in 0..N_ITEMS {
        for d in 0..N_PRIMARY {
            if PRIMARY_MAP[i * N_PRIMARY + d] {
                worst_slope = worst_slope
                    .max((fit.a_primary[i * N_PRIMARY + d] - mirt_a_p[i * N_PRIMARY + d]).abs());
            }
        }
        worst_slope = worst_slope.max((fit.a_specific[i] - mirt_a_s[i]).abs());
    }
    let mut worst_intercept = 0.0f64;
    for (i, &m) in mirt_d.iter().enumerate() {
        worst_intercept = worst_intercept.max((fit.threshold[i] - m).abs());
    }
    let rho_gap = (fit.phi[1] - mirt_phi[1]).abs();
    let rust_ll = *fit.loglik_trace.last().expect("trace is never empty");
    let ll_gap = (rust_ll - mirt_ll).abs();

    assert!(
        worst_slope <= MAX_SLOPE_GAP,
        "Rust↔mirt slope gap {worst_slope:.4} exceeds {MAX_SLOPE_GAP} (REPORTED, not tuned):\n\
         rust a_P {:?}\n mirt a_P {mirt_a_p:?}\n rust a_S {:?}\n mirt a_S {mirt_a_s:?}",
        fit.a_primary,
        fit.a_specific
    );
    assert!(
        worst_intercept <= MAX_INTERCEPT_GAP,
        "Rust↔mirt intercept gap {worst_intercept:.4} exceeds {MAX_INTERCEPT_GAP} (REPORTED, not tuned):\n\
         rust {:?}\n mirt {mirt_d:?}",
        fit.threshold
    );
    assert!(
        rho_gap <= MAX_RHO_GAP,
        "Rust↔mirt primary-correlation gap {rho_gap:.4} exceeds {MAX_RHO_GAP} (REPORTED, not tuned): \
         rust {} vs mirt {}",
        fit.phi[1],
        mirt_phi[1]
    );
    assert!(
        ll_gap <= MAX_LOGLIK_GAP,
        "Rust↔mirt loglik gap {ll_gap:.4} exceeds {MAX_LOGLIK_GAP} (REPORTED, not tuned): \
         rust {rust_ll:.4} vs mirt {mirt_ll:.4}"
    );
}
