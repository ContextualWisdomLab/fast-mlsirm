use super::*;

/// Candidate backtracking consumes only the expected complete-data objective.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn item_line_search_does_not_compute_discarded_gradients() {
    let q = 121;
    let (x, w) = gh_rule(q).unwrap();
    let v = Validated {
        n_persons: 4,
        n_items: 4,
        n_primary: 1,
        n_specific: 1,
        n_cat: 3,
        m1: 2,
        grid_size: q,
        free_primaries: vec![vec![0]; 4],
        blocks: vec![vec![0, 1, 2]],
        specific_free: vec![3],
        item_block: vec![Some(0), Some(0), Some(0), None],
    };
    let y: Vec<usize> = (0..16).map(|i| (i / 4 + i % 4) % 3).collect();
    let params: Vec<ItemParams> = (0..4)
        .map(|i| ItemParams {
            a_p: vec![1.0 + i as f64 * 0.1],
            a_s: if i == 3 { None } else { Some(0.8) },
            d: vec![0.8, -0.8],
        })
        .collect();
    let lw: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let full = focal_cell_cache_legacy::legacy_e_step_fipc_cpu(
        &v,
        &y,
        None,
        &params,
        &lw,
        &[lw.clone()],
        x,
        &[x.to_vec()],
        q,
        q,
    );
    let mut packed = vec![1.1, 0.8, 0.8, -0.8];
    let before = item_neg_ll_grad(&packed, &[0], true, x, x, 1, q, q, &full.1[0], 3).0;
    ITEM_LINE_SEARCH_GRADIENT_ROWS.with(|c| c.set(0));
    packed = m_step_item(packed, &[0], true, x, x, 1, q, q, &full.1[0], 3, 1e-8, 2);
    let rows = ITEM_LINE_SEARCH_GRADIENT_ROWS.with(|c| c.get());
    let after = item_neg_ll_grad(&packed, &[0], true, x, x, 1, q, q, &full.1[0], 3).0;
    assert!(before.is_finite() && after.is_finite() && after < before);
    assert_eq!(
        rows, 0,
        "line-search objective computed {rows} discarded gradient rows"
    );
}

/// Characterize objective projection and unchanged Newton acceptance bitwise.
/// Basis: Cai (2010, pp. 608–609, Appendix A).
/// Reference: Cai, L. (2010). A two-tier full-information item factor analysis
/// model with applications. Psychometrika, 75(4), 581–612.
/// doi:10.1007/s11336-010-9178-0.
#[test]
fn objective_projection_matches_gradient_objective_and_original_newton_bits() {
    let q = 121;
    let (x, _) = gh_rule(q).unwrap();
    let mut coords: Vec<f64> = x
        .iter()
        .flat_map(|t| [0.35 + 1.2 * t, -0.1 + 0.9 * t])
        .collect();
    coords.extend_from_slice(&[-43.0, 37.0]);
    let mut ts: Vec<f64> = x.iter().map(|t| 0.75 * t).collect();
    for phase in 0..2 {
        for (free, specific) in [(vec![0], true), (vec![0, 1], true), (vec![0], false)] {
            let nodes = if specific { (q + 1) * q } else { q + 1 };
            let counts: Vec<Vec<f64>> = (0..nodes)
                .map(|node| {
                    if node % 11 == 0 {
                        vec![0.0; 3]
                    } else {
                        vec![
                            0.3 + (node % 3) as f64 * 0.1,
                            0.5 + (node % 5) as f64 * 0.1,
                            0.7,
                        ]
                    }
                })
                .collect();
            let mut valid = vec![-1.1; free.len()];
            if specific {
                valid.push(-0.8);
            }
            valid.extend_from_slice(&[0.8, -0.8]);
            let mut rejected = valid.clone();
            let n = rejected.len();
            rejected[n - 2] = -0.8;
            rejected[n - 1] = 0.8;
            let mut nonfinite = valid.clone();
            nonfinite[0] = f64::INFINITY;
            for trial in [&valid, &rejected, &nonfinite] {
                let previous = item_neg_ll_grad(
                    trial,
                    &free,
                    specific,
                    &coords,
                    &ts,
                    2,
                    q + 1,
                    q,
                    &counts,
                    3,
                )
                .0;
                let actual = item_neg_ll(
                    trial,
                    &free,
                    specific,
                    &coords,
                    &ts,
                    2,
                    q + 1,
                    q,
                    &counts,
                    3,
                );
                if previous.is_nan() {
                    assert!(actual.is_nan());
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        previous.to_bits(),
                        "objective phase{phase} specific{specific} free{free:?}"
                    );
                }
            }
            let previous = item_line_search_legacy::legacy_m_step_item(
                valid.clone(),
                &free,
                specific,
                &coords,
                &ts,
                2,
                q + 1,
                q,
                &counts,
                3,
                1e-8,
                2,
            );
            let actual = m_step_item(
                valid,
                &free,
                specific,
                &coords,
                &ts,
                2,
                q + 1,
                q,
                &counts,
                3,
                1e-8,
                2,
            );
            assert_eq!(actual.len(), previous.len());
            assert!(
                actual
                    .iter()
                    .zip(&previous)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "Newton phase{phase} specific{specific} free{free:?}"
            );
        }
        coords[10] += 0.25;
        ts[10] += 0.125;
    }
}
