//! mirt Oakes-SE agreement for the single-group polytomous bifactor GRM
//! (stage 3 of #1912).
//!
//! The committed oracle (`tests/fixtures/bifactor_grm_stage3_oakes/
//! mirt_oakes_fixture.json`) is produced by `generate_mirt_oakes_fixture.R`
//! (R 4.x, mirt 1.46.1, `mirt::bfactor(..., itemtype = "graded",
//! quadpts = 15, SE = TRUE, SE.type = "Oakes")`) on the STAGE-1 fixture
//! dataset, so both sides see identical data. This test evaluates the Rust
//! Oakes assembly at mirt's MLE (the Oakes, 1999, eq. 6 identity holds at
//! every parameter point — no Rust fit is involved) at the same quadrature
//! density and compares standard errors and the full vcov against mirt's.
//!
//! GRID SCOPE (maintainer quadrature rule): matched quadpts = 15 is an
//! implementation cross-check, NOT study settings. The 121+-node
//! study-settings fixture follows the SUPPORTED_Q cap removal (rebase on
//! `fix/1929-quadrature-defaults`).
//!
//! # References (APA 7th ed.)
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188
//!
//! Gibbons, R. D., Bock, R. D., Hedeker, D., Weiss, D. J., Segawa, E., Bhaumik,
//! D. K., Kupfer, D. J., Frank, E., Grochocinski, V. J., & Stover, A. (2007).
//! Full-information item bifactor analysis of graded response data. *Applied
//! Psychological Measurement, 31*(1), 4-19.
//! https://doi.org/10.1177/0146621606289485

use mlsirm_core::bifactor_oakes::{bifactor_oakes_se, BifactorOakesConfig};

const N_ITEMS: usize = 8;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const M1: usize = N_CAT - 1;
const QUADPTS: usize = 15;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 0, 0, 1, 1, 1, 1];

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/bifactor_grm_stage3_oakes")
}

fn stage1_dir() -> std::path::PathBuf {
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
    // Minimal JSON reader for the numeric arrays the R script writes:
    // finds `"key": [...]` and flattens nested matrices row-major (the
    // brackets are treated as separators, matching the R writer's layout).
    // The trailing `:` check keeps a future key that extends this name from
    // misparsing.
    let anchor = format!("\"{key}\"");
    let start = text
        .find(&anchor)
        .unwrap_or_else(|| panic!("fixture is missing key {key:?}"));
    let after = &text[start + anchor.len()..];
    assert!(
        after.starts_with(':'),
        "fixture key {key:?} must be followed by ':'"
    );
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

fn load_dataset() -> Vec<usize> {
    let path = stage1_dir().join("dataset.csv");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read stage-1 dataset {path:?}: {e}"));
    let mut y = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        for cell in line.split(',') {
            y.push(
                cell.trim()
                    .parse::<usize>()
                    .unwrap_or_else(|_| panic!("dataset cell did not parse: {cell:?}")),
            );
        }
    }
    assert_eq!(y.len() % N_ITEMS, 0, "dataset columns must match n_items");
    y
}

#[test]
fn rust_oakes_se_matches_mirt_oakes_fixture() {
    let y = load_dataset();
    let n_persons = y.len() / N_ITEMS;
    let text = std::fs::read_to_string(fixture_dir().join("mirt_oakes_fixture.json"))
        .expect("cannot read mirt Oakes fixture");
    let a_g = fixture_numbers(&text, "mirt_a_general");
    let a_s = fixture_numbers(&text, "mirt_a_specific");
    let d_flat = fixture_numbers(&text, "mirt_d");
    let se_ag = fixture_numbers(&text, "mirt_se_a_general");
    let se_as = fixture_numbers(&text, "mirt_se_a_specific");
    let se_d = fixture_numbers(&text, "mirt_se_d");
    let vcov_mirt = fixture_numbers(&text, "mirt_vcov");
    assert_eq!(a_g.len(), N_ITEMS);
    assert_eq!(a_s.len(), N_ITEMS);
    assert_eq!(d_flat.len(), N_ITEMS * M1);
    assert_eq!(se_ag.len(), N_ITEMS);
    assert_eq!(se_as.len(), N_ITEMS);
    assert_eq!(se_d.len(), N_ITEMS * M1);
    let k = N_ITEMS * (2 + M1);
    assert_eq!(vcov_mirt.len(), k * k);
    // Order check: mirt's vcov is item-major [a_G, a_S, d0, d1, d2] per
    // item (verified in the R generator via flat-index arithmetic), so its
    // diagonal must reproduce the aligned per-item SE arrays.
    for i in 0..N_ITEMS {
        let base = i * (2 + M1);
        let expect = [
            se_ag[i],
            se_as[i],
            se_d[i * M1],
            se_d[i * M1 + 1],
            se_d[i * M1 + 2],
        ];
        for (j, e) in expect.iter().enumerate() {
            let got = vcov_mirt[(base + j) * k + base + j].sqrt();
            assert!(
                (got - e).abs() <= 1e-9,
                "mirt vcov diagonal at item {i} param {j} must match the \
                 aligned SE array (order check): got {got}, expect {e}"
            );
        }
    }
    // Rust Oakes assembly at mirt's MLE, same quadrature density. The
    // Oakes identity holds at every point, so no Rust fit is involved.
    let cfg = BifactorOakesConfig {
        q_general: QUADPTS,
        q_specific: QUADPTS,
        fd_step: 1e-5,
    };
    let res = bifactor_oakes_se(
        &a_g,
        &a_s,
        &d_flat,
        &y,
        None,
        &SPECIFIC_MAP,
        n_persons,
        N_ITEMS,
        N_SPECIFIC,
        N_CAT,
        &cfg,
    )
    .expect("Oakes assembly at mirt's MLE must return Ok");
    assert!(
        res.positive_definite,
        "info at mirt's converged MLE must be PD; reason: {:?}",
        res.non_pd_reason
    );
    let (vcov, se) = (
        res.vcov.expect("PD implies vcov"),
        res.se.expect("PD implies se"),
    );
    // Per-item mirt SEs in Rust free order.
    let mut se_mirt = Vec::with_capacity(k);
    for i in 0..N_ITEMS {
        se_mirt.push(se_ag[i]);
        se_mirt.push(se_as[i]);
        se_mirt.extend_from_slice(&se_d[i * M1..(i + 1) * M1]);
    }
    let mut worst_se = 0.0f64;
    let mut worst_se_j = 0usize;
    for (j, (r, m)) in se.iter().zip(se_mirt.iter()).enumerate() {
        let gap = (r - m).abs() / (1.0 + m.abs());
        if gap > worst_se {
            worst_se = gap;
            worst_se_j = j;
        }
    }
    eprintln!("mirt-SE comparison: worst scaled SE gap = {worst_se:.3e} at param {worst_se_j}");
    let mut worst_v = 0.0f64;
    for (r, m) in vcov.iter().zip(vcov_mirt.iter()) {
        worst_v = worst_v.max((r - m).abs() / (1.0 + m.abs()));
    }
    eprintln!("mirt-vcov comparison: worst scaled vcov gap = {worst_v:.3e}");
    eprintln!("rust SEs: {se:.4?}");
    eprintln!("mirt SEs: {se_mirt:.4?}");
    assert!(
        worst_se <= 0.05,
        "Rust Oakes SEs must match mirt SE.type='Oakes' within 5% (same \
         data, same MLE point, matched quadrature family). This is a \
         REGRESSION GUARD, not a calibration: the band sits at ~6x the \
         measured 8.4e-3 gap at quadpts = 15 so harmless refactors pass \
         while implementation drift trips it. The gap sources are GH-node \
         conventions and mirt's TOL = 5e-4 fit looseness; the 121+-node \
         fixture on rebase recalibrates the band. Worst = {worst_se:.3e} \
         at param {worst_se_j}"
    );
    assert!(
        worst_v <= 0.10,
        "Rust vcov regression guard (~5x the measured 1.9e-2 at \
         quadpts = 15); worst = {worst_v:.3e}"
    );
}
