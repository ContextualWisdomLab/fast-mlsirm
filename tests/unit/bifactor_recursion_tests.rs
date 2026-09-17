use super::*;
use crate::quadrature::gh_rule;

#[test]
fn test_bifactor_lord_wingersky_matches_direct_enumeration_within_1e12_on_257_grid() {
    // Acceptance criterion from Issue #1912:
    // "Lord-Wingersky 재귀의 GPU 구현: 쌍요인에서 특수요인을 적분해 없애는 2단 적용(영역 내 재귀 -> 영역 간 재귀)을 포함해 주십시오.
    // 수용 기준: 직접 열거 계산과의 최대 차이 1e-12 이하, theta 격자 -8~+8 간격 0.0625(257점)에서 검증."

    // 1. Construct the 257-point grid on [-8.0, +8.0] with step 0.0625
    let n_points = 257usize;
    let theta_general: Vec<f64> = (0..n_points)
        .map(|i| -8.0 + (i as f64) * 0.0625)
        .collect();
    assert_eq!(theta_general.len(), 257);
    assert!((theta_general[0] - (-8.0)).abs() < 1e-15);
    assert!((theta_general[256] - 8.0).abs() < 1e-15);

    // 2. Specific factor quadrature rule (Gauss-Hermite with 21 points)
    let (gh_nodes, gh_weights) = gh_rule(21).expect("gh_rule(21) supported");

    // 3. Test setup with 6 items across 2 specific domains (3 items in domain 0, 3 items in domain 1)
    // 4-category items (n_cat = 4), m1 = 3 thresholds per item
    let n_items = 6usize;
    let n_cat = 4usize;
    let n_domains = 2usize;
    let item_domains = vec![0, 0, 0, 1, 1, 1];

    let a_general = vec![1.2, 0.9, 1.5, 1.1, 0.8, 1.3];
    let a_specific = vec![0.8, 1.0, 0.7, 0.9, 1.1, 0.6];
    // Ordered decreasing thresholds for each item (m1 = 3 per item)
    let thresholds = vec![
        1.5, 0.2, -1.1,
        1.8, 0.4, -0.9,
        1.2, 0.0, -1.4,
        2.0, 0.5, -0.8,
        1.4, 0.1, -1.2,
        1.6, 0.3, -1.0,
    ];

    let params = BifactorItemParams {
        a_general,
        a_specific,
        thresholds,
        item_domains,
        n_cat,
        n_domains,
    };
    assert_eq!(params.n_items(), n_items);

    // 4. Compute 2-stage Lord-Wingersky recursion
    let p_recursion = bifactor_lord_wingersky(
        &params,
        &theta_general,
        &gh_nodes,
        &gh_weights,
    ).expect("bifactor_lord_wingersky must succeed");

    // 5. Compute Direct Enumeration exact ground truth
    let p_direct = direct_enumeration_bifactor(
        &params,
        &theta_general,
        &gh_nodes,
        &gh_weights,
    ).expect("direct_enumeration_bifactor must succeed");

    assert_eq!(p_recursion.len(), p_direct.len());
    let total_max_score = params.total_max_score();
    assert_eq!(total_max_score, 6 * 3); // 18 max score

    let mut max_abs_diff = 0.0_f64;
    let mut max_diff_point = (0usize, 0usize);

    for (g_idx, &th0) in theta_general.iter().enumerate() {
        let row_offset = g_idx * (total_max_score + 1);
        let mut row_sum_rec = 0.0_f64;
        let mut row_sum_dir = 0.0_f64;

        for r in 0..=total_max_score {
            let rec_val = p_recursion[row_offset + r];
            let dir_val = p_direct[row_offset + r];
            row_sum_rec += rec_val;
            row_sum_dir += dir_val;

            let diff = (rec_val - dir_val).abs();
            if diff > max_abs_diff {
                max_abs_diff = diff;
                max_diff_point = (g_idx, r);
            }
        }

        // Row sums must be 1.0 within floating point precision
        assert!(
            (row_sum_rec - 1.0).abs() < 1e-12,
            "Row sum for theta_0={th0} ({row_sum_rec}) must be 1.0"
        );
        assert!(
            (row_sum_dir - 1.0).abs() < 1e-12,
            "Row sum for theta_0={th0} ({row_sum_dir}) must be 1.0"
        );
    }

    println!(
        "Max difference between 2-stage Lord-Wingersky and Direct Enumeration: {:.3e} at node {}, score {}",
        max_abs_diff, max_diff_point.0, max_diff_point.1
    );

    // Strict acceptance criteria: max difference <= 1e-12
    assert!(
        max_abs_diff <= 1e-12,
        "Acceptance criteria failed: max diff {:.3e} > 1e-12",
        max_abs_diff
    );
}

#[test]
fn test_bifactor_lord_wingersky_dichotomous_matches_direct_enumeration() {
    let theta_general = vec![-3.0, -1.5, 0.0, 1.5, 3.0];
    let (gh_nodes, gh_weights) = gh_rule(15).expect("gh_rule(15)");

    let params = BifactorItemParams {
        a_general: vec![1.0, 0.8, 1.2, 0.9],
        a_specific: vec![0.7, 0.9, 0.6, 0.8],
        thresholds: vec![0.5, -0.2, 0.1, -0.4],
        item_domains: vec![0, 0, 1, 1],
        n_cat: 2,
        n_domains: 2,
    };

    let p_rec = bifactor_lord_wingersky(&params, &theta_general, &gh_nodes, &gh_weights).unwrap();
    let p_dir = direct_enumeration_bifactor(&params, &theta_general, &gh_nodes, &gh_weights).unwrap();

    let mut max_diff = 0.0_f64;
    for (a, b) in p_rec.iter().zip(&p_dir) {
        let diff = (a - b).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    assert!(max_diff <= 1e-12, "Dichotomous max diff {max_diff:.3e} must be <= 1e-12");
}
