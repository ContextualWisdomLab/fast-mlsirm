//! Multigroup bifactor same-fit M2 and FIFA delta-method contracts.
//! Synthetic banks only. Coordinate order is MaskFit::bifactor_coordinates.
use mlsirm_core::mask_grm::{bifactor_coordinates, Coordinate};
use mlsirm_core::samefit_fifa::infer;
use mlsirm_core::samefit_grm::quadrature;
use mlsirm_core::samefit_grm_joint::fit_indices;

fn bank() -> mlsirm_core::samefit_grm::BifactorPackedArchive {
    // item 0,1 -> specific 0; item 2 -> specific 1; item 3 general-only.
    // item 0 is the common anchor; items 1..3 are free in the focal group.
    let map = [0i32, 0, 1, -1];
    let anchor = [true, false, false, false];
    let mut packed = Vec::new();
    let groups_for = |common: bool| if common { 1 } else { 2 };
    for (item, &common) in anchor.iter().enumerate() {
        for g in 0..groups_for(common) {
            let shift = 0.15 * g as f64;
            packed.push(1.1 + 0.2 * item as f64 + shift);
            if map[item] >= 0 {
                packed.push(0.55 + 0.1 * item as f64 + 0.05 * shift);
            }
            packed.push(0.8 - 0.1 * item as f64 - shift);
            packed.push((0.9 + 0.05 * item as f64).ln());
        }
    }
    packed.extend([0.25, 0.12_f64.ln()]);
    packed.extend([0.18, (-0.07_f64), -0.11, 0.15_f64.ln()]);
    mlsirm_core::samefit_grm::BifactorPackedArchive::new(
        anchor.to_vec(),
        2,
        3,
        map.to_vec(),
        2,
        true,
        true,
        packed,
    )
    .expect("synthetic bifactor archive")
}

#[test]
fn coordinate_order_matches_maskfit_and_rejects_general_only_length() {
    let archive = bank();
    let expected = bifactor_coordinates(
        &archive.anchor(),
        2,
        3,
        &archive.specific_map(),
        2,
        true,
        true,
    )
    .unwrap();
    assert_eq!(archive.coordinates().unwrap(), expected);
    assert!(expected
        .iter()
        .any(|c| matches!(c, Coordinate::SpecificSlope(_, _))));
    assert!(expected
        .iter()
        .any(|c| matches!(c, Coordinate::SpecificMean(1, _))));
    assert!(expected
        .iter()
        .any(|c| matches!(c, Coordinate::SpecificLogSd(1, _))));
    assert!(mlsirm_core::samefit_grm::PackedArchive::new(
        archive.anchor().to_vec(),
        2,
        3,
        archive.packed_values(),
    )
    .is_err());
}

#[test]
fn shared_specific_moments_match_central_differences_and_not_product() {
    let archive = bank();
    let (nodes, weights) = quadrature(7).unwrap();
    let (moments, delta, xi) =
        mlsirm_core::samefit_grm_joint::bifactor_moments_delta(&archive, &nodes, &weights).unwrap();
    let x = archive.packed_values();
    let p = x.len();
    for column in 0..p {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[column] += 1e-5;
        minus[column] -= 1e-5;
        let (mp, _, _) = mlsirm_core::samefit_grm_joint::bifactor_moments_delta(
            &archive.with_packed(plus).unwrap(),
            &nodes,
            &weights,
        )
        .unwrap();
        let (mm, _, _) = mlsirm_core::samefit_grm_joint::bifactor_moments_delta(
            &archive.with_packed(minus).unwrap(),
            &nodes,
            &weights,
        )
        .unwrap();
        for row in 0..moments.len() {
            let finite = (mp[row] - mm[row]) / 2e-5;
            assert!(
                (delta[row * p + column] - finite).abs() < 2e-7,
                "row {row} column {column}: analytic {} finite {finite}",
                delta[row * p + column]
            );
        }
    }
    let product = mlsirm_core::samefit_grm_joint::independent_specific_product_moments(
        &archive, &nodes, &weights,
    )
    .unwrap();
    let shared = moments
        .iter()
        .zip(&product)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        shared > 1e-4,
        "shared-S reduction collapsed to a product: {shared}"
    );
    assert!(xi.iter().any(|v| v.is_finite() && v.abs() > 0.0));
}

#[test]
fn cpu_joint_diagnostics_return_measured_m2_and_comparative_indices() {
    let archive = bank();
    let d = mlsirm_core::samefit_grm::decode_bifactor(&archive).unwrap();
    let mut y = Vec::new();
    let mut groups = Vec::new();
    let mut state = 20261027_u64;
    let mut next_u = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / ((1u64 << 53) as f64)
    };
    let z = archive.n_categories() - 1;
    for group in 0..2 {
        for person in 0..120 {
            groups.push(group);
            let g_draw = ((person as f64 + 0.5) / 120.0 - 0.5) * 3.0;
            let s0_draw = (((person * 7 % 120) as f64 + 0.5) / 120.0 - 0.5) * 2.5;
            let s1_draw = (((person * 13 % 120) as f64 + 0.5) / 120.0 - 0.5) * 2.5;
            let tg = d.general_means[group] + d.general_sds[group] * g_draw;
            let ts = [
                d.specific_means[group * 2] + d.specific_sds[group * 2] * s0_draw,
                d.specific_means[group * 2 + 1] + d.specific_sds[group * 2 + 1] * s1_draw,
            ];
            for item in 0..4 {
                let sp = archive.specific_map()[item];
                let eta = d.general_slope[group * 4 + item] * tg
                    + if sp >= 0 {
                        d.specific_slope[group * 4 + item] * ts[sp as usize]
                    } else {
                        0.0
                    };
                let probs = mlsirm_core::poly::grm_logprobs(
                    eta,
                    &d.intercepts[(group * 4 + item) * z..(group * 4 + item + 1) * z],
                );
                let u = next_u();
                let mut acc = 0.0;
                let mut cat = archive.n_categories() - 1;
                for (k, &lp) in probs.iter().enumerate() {
                    acc += lp.exp();
                    if u <= acc {
                        cat = k;
                        break;
                    }
                }
                y.push(cat);
            }
        }
    }
    let result =
        mlsirm_core::samefit_grm_joint::bifactor_joint_diagnostics(&y, &groups, &archive, 7, 7)
            .expect("bifactor same-fit diagnostics");
    for key in ["m2", "df", "p_value", "rmsea", "cfi", "tli"] {
        assert!(
            result[key].as_f64().is_some_and(f64::is_finite),
            "{key}: {}",
            result[key]
        );
    }
    assert!(result["df"].as_f64().unwrap() > 0.0);
    assert_eq!(result["rmsea_denominator"], "N * df");
    assert_eq!(result["integration"], "shared_g_conditional_specific");
    let indices = fit_indices(
        result["m2"].as_f64().unwrap(),
        result["df"].as_f64().unwrap(),
        groups.len(),
        result["null_m2"].as_f64().unwrap(),
        result["null_df"].as_f64().unwrap(),
    )
    .unwrap();
    assert!((indices["cfi"].as_f64().unwrap() - result["cfi"].as_f64().unwrap()).abs() < 1e-12);
}

#[test]
fn fifa_jacobian_and_acov_delta_match_finite_differences() {
    let archive = bank();
    let transformed = mlsirm_core::samefit_fifa::transform_bifactor(&archive).unwrap();
    let x = archive.packed_values();
    let p = x.len();
    let n = transformed.rows.len();
    assert!(transformed.rows.iter().any(|r| r.name.contains("specific")));
    assert!(transformed.rows.iter().any(|r| r.name.contains("lambda")));
    for column in 0..p {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[column] += 1e-5;
        minus[column] -= 1e-5;
        let tp = mlsirm_core::samefit_fifa::transform_bifactor(&archive.with_packed(plus).unwrap())
            .unwrap();
        let tm =
            mlsirm_core::samefit_fifa::transform_bifactor(&archive.with_packed(minus).unwrap())
                .unwrap();
        for row in 0..n {
            let finite = (tp.rows[row].estimate - tm.rows[row].estimate) / 2e-5;
            assert!(
                (transformed.jacobian[row * p + column] - finite).abs() < 2e-6,
                "row {row} column {column}"
            );
        }
    }
    let mut information = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            information[i * p + j] = if i == j { 4.0 } else { 0.15 };
        }
    }
    let inferred = infer(&transformed, &information).unwrap();
    let direct = mlsirm_core::samefit_fifa::infer_bifactor(&archive, &information).unwrap();
    assert_eq!(inferred.rows.len(), n);
    assert_eq!(direct.rows.len(), n);
    for (idx, row) in inferred.rows.iter().enumerate() {
        assert!((row.se - direct.rows[idx].se).abs() < 1e-15);
        let grad = &transformed.jacobian[idx * p..(idx + 1) * p];
        let is_fixed_ref_moment = grad.iter().all(|&g| g == 0.0);
        if is_fixed_ref_moment {
            assert!(row.name.starts_with("g0."));
            assert_eq!(row.se, 0.0);
        } else {
            assert!(row.se.is_finite() && row.se > 0.0, "{}", row.name);
            assert!(row.ci95_lower < row.estimate && row.estimate < row.ci95_upper);
        }
    }
}

#[test]
fn general_only_reduction_summary_import_and_fail_closed_guards() {
    use sha2::{Digest, Sha256};
    let anchor = vec![true, false];
    let packed = vec![1.2, 0.7, -0.1, 0.9, 0.5, -0.2, 1.1, 0.4, -0.15, 0.2, 0.1];
    let general =
        mlsirm_core::samefit_grm::PackedArchive::new(anchor.clone(), 2, 3, packed.clone()).unwrap();
    let bifactor = mlsirm_core::samefit_grm::BifactorPackedArchive::new(
        anchor.clone(),
        2,
        3,
        vec![-1, -1],
        0,
        false,
        false,
        packed.clone(),
    )
    .unwrap();
    let tg = mlsirm_core::samefit_fifa::transform(&general).unwrap();
    let tb = mlsirm_core::samefit_fifa::transform_bifactor(&bifactor).unwrap();
    for row in &tg.rows {
        let matched = tb.rows.iter().find(|r| r.name == row.name).unwrap();
        assert!(
            (row.estimate - matched.estimate).abs() < 1e-14,
            "{}",
            row.name
        );
    }

    let archive = bank();
    let request = r#"{"y":[0,1,2,0,1,2,0,1],"observed":null,"group_id":[0,1],"n_groups":2,"n_cat":3,"item_ids":["i0","i1","i2","i3"],"q_general":7}"#;
    let hash = format!("{:x}", Sha256::digest(request.as_bytes()));
    let probe_summary = serde_json::json!({
        "anchor": archive.anchor(),
        "specific_map": archive.specific_map(),
        "n_specific": archive.n_specific(),
        "q_general": 7,
        "q_specific": 7,
        "estimate_specific_means": true,
        "estimate_specific_vars": true,
        "coordinate_map": archive.coordinates().unwrap(),
        "packed": archive.packed_values(),
        "loglik_cpu_f64": 0.0,
        "fit_identity": "synthetic-bifactor"
    });
    assert!(mlsirm_core::samefit_grm::import_bifactor_summary(
        request,
        &hash,
        &probe_summary,
        1e-12
    )
    .is_err());
    assert!(mlsirm_core::samefit_grm::BifactorPackedArchive::new(
        vec![true; 4],
        2,
        3,
        vec![0, 0, 2, -1],
        2,
        true,
        true,
        archive.packed_values(),
    )
    .is_err());
}
