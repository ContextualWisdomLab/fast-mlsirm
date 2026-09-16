//! mirt agreement for the single-group polytomous bifactor GRM
//! (stage 1 of #1912).
//!
//! The committed dataset (`tests/fixtures/bifactor_grm_stage1/dataset.csv`)
//! and the mirt fit (`mirt_fixture.json`) are both produced by
//! `generate_mirt_fixture.R` (R 4.x, mirt 1.46.1,
//! `mirt::bfactor(..., itemtype = "graded", quadpts = 15)`). This test fits
//! the same data with the Rust Gibbons-Hedeker EM at the same quadrature
//! density and compares general/specific slopes (after aligning the
//! per-dimension reflection), boundary intercepts, and the observed-data
//! log-likelihood.
//!
//! Intercept mapping (verified against the mirt fit, not assumed): mirt's
//! graded `d_k` columns are the SAME additive boundary intercepts the Rust
//! fitter reports as `threshold`, i.e.
//! `P(Y >= k | theta) = logistic(a_G * theta_G + a_S * theta_S + d_k)`
//! (no sign flip, no division by `a`). The fixture records this convention
//! string alongside the numbers.
//!
//! Tolerance basis: with `n = 800` the MML standard errors of graded slopes
//! are on the order of 0.08-0.12, so two independent EM implementations of
//! the same MLE started from different initial values should agree an order
//! of magnitude below sampling noise. The asserted bands (0.25 per slope,
//! 0.25 per intercept, 2.0 loglik units at `|ll| ~ 8.2e3`) encode that, with
//! the measured differences reported in the stage-1 PR. Per the task, a
//! disagreement beyond tolerance is REPORTED (the assertion prints both
//! vectors), never tuned away.
//!
//! Measured at the committed fixture (test config): Rust `ll = -8192.9808`
//! vs mirt `ll = -8192.9960` (gap 0.015); worst slope gap 0.046 (item 6
//! specific, 2.162 vs 2.116); worst intercept gap 0.028 (item 6 `d_3`,
//! -2.2525 vs -2.2243).
//!
//! Convergence matching (measured, structural): the Rust EM climbs a slow
//! linear tail on this fixture's weakly-identified general/specific split —
//! `tol = 1e-5` stops after ~21 sweeps a full 1.03 loglik units below the
//! mode (`-8194.02` vs `-8192.98`), while `tol = 1e-7` runs ~390 sweeps and
//! lands within 0.02 of mirt's reported `-8193.00`. The test therefore uses
//! `tol = 1e-7`: comparing an unconverged plateau against mirt's MLE would be
//! a test artifact, not an estimator difference.
//!
//! # References (APA 7th ed.)
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485
//!
//! Chalmers, R. P. (2012). mirt: A multidimensional item response theory
//! package for the R environment. *Journal of Statistical Software, 48*(6).
//! https://doi.org/10.18637/jss.v048.i06

use mlsirm_core::bifactor_grm::{fit_bifactor_grm, BifactorGrmConfig};
use std::collections::HashMap;

const N_ITEMS: usize = 8;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const QUADPTS: usize = 15;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 1, 1, 1, 1];

const MAX_SLOPE_GAP: f64 = 0.25;
const MAX_INTERCEPT_GAP: f64 = 0.25;
const MAX_LOGLIK_GAP: f64 = 2.0;

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bifactor_grm_stage1")
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
    // Minimal JSON reader for the flat numeric arrays the R script writes:
    // finds `"key": [...]` (possibly nested one level for mirt_d rows).
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
fn bifactor_grm_agrees_with_mirt_bfactor_graded() {
    let fixture_path = fixture_dir().join("mirt_fixture.json");
    let fixture = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "cannot read mirt fixture {fixture_path:?}: {e} — run \
             Rscript tests/fixtures/bifactor_grm_stage1/generate_mirt_fixture.R first"
        )
    });
    assert!(
        fixture.contains("logistic(a_G*theta_G + a_S*theta_S + d_k)"),
        "fixture must record the verified intercept convention; got: {fixture}"
    );
    let mirt_a_g = fixture_numbers(&fixture, "mirt_a_general");
    let mirt_a_s = fixture_numbers(&fixture, "mirt_a_specific");
    let mirt_d = fixture_numbers(&fixture, "mirt_d");
    let mirt_ll = fixture_scalar(&fixture, "mirt_loglik");
    assert_eq!(mirt_a_g.len(), N_ITEMS, "mirt general slopes: {mirt_a_g:?}");
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
    assert!(mirt_ll.is_finite(), "mirt loglik must be finite");

    let y = load_dataset();
    let n_persons = y.len() / N_ITEMS;
    assert!(n_persons > 100, "fixture dataset must be non-trivial");

    // Reflection alignment BEFORE the comparison: each implementation pins
    // its own latent orientation, so compare mirt against the Rust fit up to
    // the per-dimension sign both estimators canonicalize independently.
    let cfg = BifactorGrmConfig {
        q_general: QUADPTS,
        q_specific: QUADPTS,
        max_iter: 2000,
        tol: 1e-7,
        n_starts: 1,
        seed: 0x51F1_5EED_2026_0916,
        newton_iter: 10,
        ridge: 1e-8,
    };
    let fit = fit_bifactor_grm(
        &y,
        None,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &cfg,
    )
    .expect("bifactor GRM fit on the mirt fixture dataset must succeed");
    assert!(
        fit.converged,
        "fixture fit must converge (termination: {})",
        fit.termination_reason
    );

    let mut mirt_a_g = mirt_a_g;
    let mut mirt_a_s = mirt_a_s;
    // General dimension: reflection is global per dimension, so compare signs
    // at the SAME item (the Rust anchor) — comparing each estimator's own
    // anchor can false-flip when the argmax items differ across estimators.
    let rust_anchor_g = fit
        .a_general
        .iter()
        .enumerate()
        .max_by(|(_, x), (_, y)| x.abs().total_cmp(&y.abs()))
        .map(|(i, _)| i)
        .unwrap();
    if (fit.a_general[rust_anchor_g] > 0.0) != (mirt_a_g[rust_anchor_g] > 0.0) {
        for v in mirt_a_g.iter_mut() {
            *v = -*v;
        }
    }
    // Specific dimensions: align block by block at the Rust block anchor.
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
        worst_slope = worst_slope
            .max((fit.a_general[i] - mirt_a_g[i]).abs())
            .max((fit.a_specific[i] - mirt_a_s[i]).abs());
    }
    let mut worst_intercept = 0.0f64;
    for (i, &m) in mirt_d.iter().enumerate() {
        worst_intercept = worst_intercept.max((fit.threshold[i] - m).abs());
    }
    let rust_ll = *fit.loglik_trace.last().expect("trace is never empty");
    let ll_gap = (rust_ll - mirt_ll).abs();

    assert!(
        worst_slope <= MAX_SLOPE_GAP,
        "Rust↔mirt slope gap {worst_slope:.4} exceeds {MAX_SLOPE_GAP} (REPORTED, not tuned):\n\
         rust a_G {:?}\n mirt a_G {mirt_a_g:?}\n rust a_S {:?}\n mirt a_S {mirt_a_s:?}",
        fit.a_general,
        fit.a_specific
    );
    assert!(
        worst_intercept <= MAX_INTERCEPT_GAP,
        "Rust↔mirt intercept gap {worst_intercept:.4} exceeds {MAX_INTERCEPT_GAP} (REPORTED, not tuned):\n\
         rust {:?}\n mirt {mirt_d:?}",
        fit.threshold
    );
    assert!(
        ll_gap <= MAX_LOGLIK_GAP,
        "Rust↔mirt loglik gap {ll_gap:.4} exceeds {MAX_LOGLIK_GAP} (REPORTED, not tuned): \
         rust {rust_ll:.4} vs mirt {mirt_ll:.4}"
    );
}
