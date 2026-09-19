//! Unit tests for two-tier Oakes SE assembly (#1992).

use super::*;

fn tiny_design() -> (Vec<usize>, Vec<bool>, Vec<i32>, usize, usize) {
    // 4 persons, 4 items, 2 primaries, 1 specific, 2 categories — minimal
    // identifiable pattern for smoke tests (not study settings).
    let n_persons = 4usize;
    let n_items = 4usize;
    // Y row-major
    let y = vec![
        0, 1, 0, 1, // p0
        1, 1, 0, 0, // p1
        0, 0, 1, 1, // p2
        1, 0, 1, 0, // p3
    ];
    // Items 0,1 on P0; 2,3 on P1; all on specific 0
    let mut primary_map = vec![false; n_items * 2];
    primary_map[0 * 2] = true;
    primary_map[1 * 2] = true;
    primary_map[2 * 2 + 1] = true;
    primary_map[3 * 2 + 1] = true;
    let specific_map = vec![0i32, 0, 0, 0];
    (y, primary_map, specific_map, n_persons, n_items)
}

#[test]
fn fd_step_must_be_positive() {
    let (y, pmap, smap, n_persons, n_items) = tiny_design();
    let cfg = TwoTierOakesConfig {
        q_primary: 5,
        q_specific: 5,
        fd_step: 0.0,
    };
    let a_p = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
    let a_s = vec![0.8; 4];
    let thr = vec![0.0; 4]; // n_cat=2 => 1 threshold each
    let phi = vec![1.0, 0.2, 0.2, 1.0];
    let err = two_tier_oakes_se(
        &a_p, &a_s, &thr, &phi, &y, None, &pmap, &smap, n_persons, n_items, 2, 1, 2, &cfg,
    )
    .unwrap_err();
    assert!(err.contains("fd_step"));
}

#[test]
fn returns_finite_information_on_tiny_case() {
    let (y, pmap, smap, n_persons, n_items) = tiny_design();
    // Need every category observed per item — expand categories carefully.
    // With n_cat=2 the design above observes both cats on each item.
    let cfg = TwoTierOakesConfig {
        q_primary: 7,
        q_specific: 7,
        fd_step: 1e-5,
    };
    let a_p = vec![1.2, 0.0, 0.9, 0.0, 0.0, 1.1, 0.0, 1.0];
    let a_s = vec![0.7, 0.8, 0.6, 0.9];
    let thr = vec![0.1, -0.1, 0.2, 0.0];
    let phi = vec![1.0, 0.25, 0.25, 1.0];
    let res = two_tier_oakes_se(
        &a_p, &a_s, &thr, &phi, &y, None, &pmap, &smap, n_persons, n_items, 2, 1, 2, &cfg,
    )
    .expect("oakes should run");
    assert_eq!(res.information.len(), res.labels.len() * res.labels.len());
    assert!(res.information.iter().all(|v| v.is_finite()));
    // Tiny N may be non-PD; either finite SEs or explicit non-PD is OK.
    if res.positive_definite {
        let se = res.se.expect("PD => se");
        assert!(se.iter().all(|s| s.is_finite() && *s > 0.0));
    } else {
        assert!(res.se.is_none());
        assert!(res.non_pd_reason.is_some());
    }
}

/// Air / study-settings memory probe: one streaming E-step at q=241, P=2
/// (full product GH grid; no silent node cap). Prefer fit max_iter=1 over
/// full Oakes (Oakes repeats E-step k+1 times).
///
/// `n_persons` is kept modest (24): streaming peak RSS is dominated by the
/// primary×specific grid scratch/counts, not N (persons are processed
/// sequentially). Larger N only inflates wall-clock on shared Air.
/// Run on Air: cargo test -p mlsirm-core two_tier_q241_rss_probe --release -- --ignored --nocapture
#[test]
#[ignore]
fn two_tier_q241_rss_probe() {
    use crate::two_tier_grm::{fit_two_tier_grm, TwoTierGrmConfig};
    use std::process::Command;
    let n_persons = 24usize;
    let n_items = 8usize;
    let n_primary = 2usize;
    let n_specific = 2usize;
    let n_cat = 4usize;
    let q = 241usize;
    let mut y = Vec::with_capacity(n_persons * n_items);
    for p in 0..n_persons {
        for i in 0..n_items {
            y.push((p + 2 * i) % n_cat);
        }
    }
    // Ensure every category observed per item.
    for i in 0..n_items {
        for k in 0..n_cat {
            y[k * n_items + i] = k;
        }
    }
    let mut primary_map = vec![false; n_items * n_primary];
    for i in 0..4 {
        primary_map[i * n_primary] = true;
    }
    for i in 4..8 {
        primary_map[i * n_primary + 1] = true;
    }
    let specific_map = vec![0i32, 0, 0, 0, 1, 1, 1, 1];
    let cfg = TwoTierGrmConfig {
        q_primary: q,
        q_specific: q,
        max_iter: 1,
        tol: 1e-4,
        n_starts: 1,
        seed: 20260918,
        newton_iter: 5,
        ridge: 1e-8,
        log_progress: false,
    };
    let rss = |pid: u32| -> u64 {
        let out = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok();
        out.and_then(|o| String::from_utf8_lossy(&o.stdout).trim().parse().ok())
            .unwrap_or(0)
    };
    let pid = std::process::id();
    let before = rss(pid);
    let res = fit_two_tier_grm(
        &y,
        None,
        &primary_map,
        &specific_map,
        n_persons,
        n_items,
        n_primary,
        n_specific,
        n_cat,
        &cfg,
    );
    let after = rss(pid);
    eprintln!(
        "two_tier_q241_rss_probe: rss_before_kb={before} rss_after_kb={after} result={:?}",
        res.as_ref().map(|r| (r.n_iter, r.converged, r.final_loglik_change))
    );
    assert!(
        after < 28_000_000,
        "peak RSS {after} KB exceeds ~28 GB budget on 32 GB Air"
    );
    assert!(res.is_ok(), "fit at q=241 failed: {res:?}");
}
