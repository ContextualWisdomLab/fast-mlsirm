//! mirt Oakes-SE agreement for the confirmatory two-tier GRM (#1992).
//!
//! Oracle: `tests/fixtures/two_tier_grm_oakes/mirt_oakes_fixture.json`
//! produced by `generate_mirt_oakes_fixture.R` (R 4.x, mirt 1.46.1,
//! `mirt::bfactor(..., model2 = <G1/G2 + COV>, itemtype = "graded",
//! quadpts = 15, SE = TRUE, SE.type = "Oakes")`) on the stage-4 dataset
//! (`seed = 20260917`, 1500 persons, 10 items, 4 categories).
//!
//! GRID SCOPE: quadpts = 15 is an implementation cross-check, NOT study
//! settings (>= 121 nodes/dim).
//!
//! # References (APA 7th ed.)
//!
//! Oakes, D. (1999). Direct calculation of the information matrix via the EM
//! algorithm. *Journal of the Royal Statistical Society Series B: Statistical
//! Methodology, 61*(2), 479-482. https://doi.org/10.1111/1467-9868.00188
//!
//! Cai, L., Yang, J. S., & Hansen, M. (2011). Generalized full-information
//! item bifactor analysis. *Psychological Methods, 16*(3), 221-248.
//! https://doi.org/10.1037/a0023350

use mlsirm_core::two_tier_oakes::{two_tier_oakes_se, TwoTierOakesConfig};

const N_ITEMS: usize = 10;
const N_PRIMARY: usize = 2;
const N_SPECIFIC: usize = 2;
const N_CAT: usize = 4;
const M1: usize = N_CAT - 1;
const QUADPTS: usize = 15;
const SPECIFIC_MAP: [i32; N_ITEMS] = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1];

fn fixture_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/two_tier_grm_oakes")
}

fn stage4_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/two_tier_grm_stage4")
}

fn parse_number_list(s: &str) -> Vec<f64> {
    s.split(|c: char| c == ',' || c.is_whitespace() || c == '[' || c == ']')
        .filter(|t| !t.is_empty())
        .map(|t| {
            t.parse::<f64>()
                .unwrap_or_else(|_| panic!("fixture number did not parse as f64: {t:?}"))
        })
        .collect()
}

fn fixture_numbers(text: &str, key: &str) -> Vec<f64> {
    let anchor = format!("\"{key}\"");
    let start = text
        .find(&anchor)
        .unwrap_or_else(|| panic!("fixture is missing key {key:?}"));
    let after = &text[start + anchor.len()..];
    assert!(after.starts_with(':'), "fixture key {key:?} must be followed by ':'");
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

fn fixture_string_list(text: &str, key: &str) -> Vec<String> {
    let anchor = format!("\"{key}\"");
    let start = text.find(&anchor).expect("missing key");
    let after = &text[start + anchor.len()..];
    let open = after.find('[').expect("array");
    let close = after[open..].find(']').expect("close") + open;
    after[open + 1..close]
        .split(',')
        .map(|s| s.trim().trim_matches('"').to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn load_dataset() -> Vec<usize> {
    let path = stage4_dir().join("dataset.csv");
    let text = std::fs::read_to_string(&path).expect("dataset");
    let mut y = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if lineno == 0 || line.trim().is_empty() {
            continue;
        }
        for cell in line.split(',') {
            y.push(cell.trim().parse::<usize>().expect("cell"));
        }
    }
    assert_eq!(y.len() % N_ITEMS, 0);
    y
}

fn primary_map() -> Vec<bool> {
    // Items 1-5 on G1, 6-10 on G2 (1-based in R).
    let mut m = vec![false; N_ITEMS * N_PRIMARY];
    for i in 0..5 {
        m[i * N_PRIMARY] = true;
    }
    for i in 5..10 {
        m[i * N_PRIMARY + 1] = true;
    }
    m
}

/// Map Rust free-label to mirt vcov dimname / SE index.
fn rust_label_to_mirt_name(label: &str, specific: &[i32]) -> Option<String> {
    // mirt flat index uses 7-slot blocks for 2-primary+2-specific graded:
    // a1,a2,a3,a4,d1,d2,d3 — only free slots appear in vcov rownames as
    // "par.flatidx" (see stage-3 bifactor fixture; here flat stride = 7).
    if let Some(rest) = label.strip_prefix("a_primary:") {
        let mut parts = rest.split(':');
        let item: usize = parts.next()?.parse().ok()?;
        let dim: usize = parts.next()?.parse().ok()?;
        let flat = item * 7 + dim + 1; // 1-based flat
        return Some(format!("a{}.{flat}", dim + 1));
    }
    if let Some(rest) = label.strip_prefix("a_specific:") {
        let item: usize = rest.parse().ok()?;
        let s = specific[item] as usize; // 0-based
        let flat = item * 7 + 2 + s + 1;
        return Some(format!("a{}.{flat}", 3 + s));
    }
    if let Some(rest) = label.strip_prefix("d:") {
        let mut parts = rest.split(':');
        let item: usize = parts.next()?.parse().ok()?;
        let k: usize = parts.next()?.parse().ok()?;
        let flat = item * 7 + 4 + k + 1;
        return Some(format!("d{}.{flat}", k + 1));
    }
    if label.starts_with("phi_z:") {
        // mirt 1.46.1 two-tier free cov row (fixture: COV_21.76).
        return Some("COV_21.76".to_string());
    }
    None
}

#[test]
fn rust_two_tier_oakes_se_matches_mirt_fixture() {
    let y = load_dataset();
    let n_persons = y.len() / N_ITEMS;
    let text = std::fs::read_to_string(fixture_dir().join("mirt_oakes_fixture.json"))
        .expect("cannot read mirt Oakes fixture");
    let a_primary = fixture_numbers(&text, "mirt_a_primary");
    let a_specific = fixture_numbers(&text, "mirt_a_specific");
    let d = fixture_numbers(&text, "mirt_d");
    let phi = fixture_numbers(&text, "mirt_phi");
    let mirt_se = fixture_numbers(&text, "mirt_se");
    let mirt_names = fixture_string_list(&text, "mirt_vcov_dimnames");
    assert_eq!(a_primary.len(), N_ITEMS * N_PRIMARY);
    assert_eq!(a_specific.len(), N_ITEMS);
    assert_eq!(d.len(), N_ITEMS * M1);
    assert_eq!(mirt_se.len(), mirt_names.len());

    let pmap = primary_map();
    let smap = SPECIFIC_MAP.to_vec();
    let cfg = TwoTierOakesConfig {
        estimate_primary_correlation: true,
        q_primary: QUADPTS,
        q_specific: QUADPTS,
        fd_step: 1e-5,
    };
    let res = two_tier_oakes_se(
        &a_primary,
        &a_specific,
        &d,
        &phi,
        &y,
        None,
        &pmap,
        &smap,
        n_persons,
        N_ITEMS,
        N_PRIMARY,
        N_SPECIFIC,
        N_CAT,
        &cfg,
    )
    .expect("rust oakes");
    assert!(
        res.positive_definite,
        "expected PD information on mirt MLE; got {:?}",
        res.non_pd_reason
    );
    let se = res.se.expect("se");

    // Align item SEs (skip phi for the primary band — mirt cov naming varies).
    let mut compared = 0usize;
    let mut max_rel = 0.0f64;
    for (lab, rust_se) in res.labels.iter().zip(se.iter()) {
        if lab.starts_with("phi_z:") {
            continue;
        }
        let Some(mname) = rust_label_to_mirt_name(lab, &smap) else {
            panic!("unmapped label {lab}");
        };
        let idx = mirt_names
            .iter()
            .position(|n| n == &mname)
            .unwrap_or_else(|| panic!("mirt name {mname} for {lab} not in fixture"));
        let mirt = mirt_se[idx];
        let rel = (rust_se - mirt).abs() / mirt.abs().max(1e-8);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= 0.05,
            "SE mismatch for {lab}/{mname}: rust={rust_se} mirt={mirt} rel={rel}"
        );
        compared += 1;
    }
    assert!(compared >= 40, "expected >=40 item SEs compared; got {compared}");
    assert!(max_rel <= 0.05, "max relative SE error {max_rel}");
}
