//! P=1 expected-count arithmetic contract, including zero primary mass.
//! Cai (2010, pp. 589–590, Eqs. 15–16; pp. 607–609, Appendix A)
//! defines the shared reduced posterior and expected-category frequencies.
//! Bit equality here is a software reduction contract, not a paper tolerance.
//! Reference: Cai, L. (2010). A two-tier full-information item factor analysis
//! model with applications. Psychometrika, 75(4), 581–612.
//! https://doi.org/10.1007/s11336-010-9178-0

use super::*;
use crate::bifactor_grm as bifactor;

#[test]
fn single_primary_expected_counts_match_bifactor_bits() {
    for mixed in [false, true] {
        for missing in [false, true] {
            for zero_mass in [false, true] {
                check_counts(mixed, missing, zero_mass);
            }
        }
    }
}

fn check_counts(mixed: bool, missing: bool, zero_mass: bool) {
    let n_items = 6;
    let n_persons = 12;
    let n_cat = 4;
    let specific = [0, 0, 1, 1, 1, if mixed { -1 } else { 0 }];
    let primary = [true; 6];
    let y: Vec<usize> = (0..n_persons * n_items)
        .map(|index| (index / n_items + index % n_items) % n_cat)
        .collect();
    let mask: Vec<bool> = (0..y.len()).map(|index| index % 7 != 0).collect();
    let observed = missing.then_some(mask.as_slice());
    let cfg = TwoTierGrmConfig {
        estimate_primary_correlation: false,
        q_primary: 15,
        q_specific: 11,
        max_iter: 1,
        tol: 1e-7,
        n_starts: 1,
        seed: 0,
        newton_iter: 10,
        ridge: 1e-8,
    };
    let bcfg = bifactor::BifactorGrmConfig {
        q_general: cfg.q_primary,
        q_specific: cfg.q_specific,
        max_iter: cfg.max_iter,
        tol: cfg.tol,
        n_starts: cfg.n_starts,
        seed: cfg.seed,
        newton_iter: cfg.newton_iter,
        ridge: cfg.ridge,
        slope_prior: bifactor::SlopePrior::None,
        device: crate::Device::Cpu,
    };
    let v = validate(
        &y, observed, &primary, &specific, n_persons, n_items, 1, 2, n_cat, &cfg,
    )
    .unwrap();
    let bv =
        bifactor::validate(&y, observed, &specific, n_persons, n_items, 2, n_cat, &bcfg).unwrap();
    let (tg, wg) = gh_rule(cfg.q_primary).unwrap();
    let (ts, ws) = gh_rule(cfg.q_specific).unwrap();
    let mut coords = tg.to_vec();
    let mut lw: Vec<f64> = wg.iter().map(|w| w.ln()).collect();
    if zero_mass {
        coords.insert(0, -43.0);
        lw.insert(0, f64::NEG_INFINITY);
    }
    let lws: Vec<f64> = ws.iter().map(|w| w.ln()).collect();
    let params: Vec<ItemParams> = (0..n_items)
        .map(|i| ItemParams {
            a_p: vec![[1.2, -0.9, 1.0, 0.7, 1.3, 0.8][i]],
            a_s: (specific[i] >= 0).then_some(0.6 + i as f64 * 0.1),
            d: vec![1.1 + i as f64 * 0.03, 0.1, -1.2],
        })
        .collect();
    let bparams: Vec<bifactor::ItemParams> = params
        .iter()
        .map(|p| bifactor::ItemParams {
            a_g: p.a_p[0],
            a_s: p.a_s,
            d: p.d.clone(),
        })
        .collect();
    let tables = bifactor::fill_logprob_tables(&bv, &bparams, &coords, ts, coords.len(), ts.len());
    // Exclude the category-cache/predictor boundary before comparing counts.
    for i in 0..n_items {
        let actual = ordinary_item_logprobs(
            &v,
            &y,
            observed,
            &params,
            &coords,
            ts,
            i,
            coords.len(),
            ts.len(),
        );
        let ns = if params[i].a_s.is_some() { ts.len() } else { 1 };
        for g in 0..coords.len() {
            for h in 0..ns {
                for k in 0..n_cat {
                    assert_eq!(
                        actual.get(g, h, k).to_bits(),
                        tables[i][(g * ns + h) * n_cat + k].to_bits(),
                        "category cell i={i}, g={g}, h={h}, k={k}"
                    );
                }
            }
        }
    }
    let expected = bifactor::e_step(
        &bv,
        &y,
        observed,
        &tables,
        &lw,
        &lws,
        coords.len(),
        ts.len(),
        &coords,
        ts,
        crate::Device::Cpu,
    );
    let actual = e_step_contiguous(
        &v,
        &y,
        observed,
        &params,
        &lw,
        &lws,
        &coords,
        ts,
        coords.len(),
        ts.len(),
    );
    assert_eq!(
        actual.0.to_bits(),
        expected.0.to_bits(),
        "marginal likelihood"
    );
    for i in 0..n_items {
        let mut mass = 0.0;
        for (node, row) in actual.1[i].iter().enumerate() {
            for (k, &value) in row.iter().enumerate() {
                assert!(value.is_finite());
                mass += value;
                assert_eq!(value.to_bits(), expected.1[i][node][k].to_bits(), "expected count mixed={mixed}, missing={missing}, zero_mass={zero_mass}, i={i}, node={node}, k={k}");
            }
        }
        let observed_n = (0..n_persons)
            .filter(|&p| observed.is_none_or(|o| o[p * n_items + i]))
            .count();
        assert!((mass - observed_n as f64).abs() < 1e-10);
        if zero_mass {
            let zero_rows = if params[i].a_s.is_some() { ts.len() } else { 1 };
            assert!(actual.1[i]
                .iter()
                .take(zero_rows)
                .flatten()
                .all(|&x| x == 0.0));
        }
    }
}
