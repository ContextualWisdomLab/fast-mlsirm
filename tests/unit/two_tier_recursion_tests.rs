use super::*;
use crate::quadrature::gh_rule;

#[test]
fn two_tier_lord_wingersky_matches_direct_enumeration_on_small_grid() {
    let n_items = 4usize;
    let n_cat = 3usize;
    let n_primary = 2usize;
    let n_specific = 2usize;
    let specific_map = vec![0i32, 0, 1, 1];
    let a_primary = vec![
        1.0, 0.0, // item 0: G only
        1.1, 0.2, // item 1: G + W
        0.9, 0.0, // item 2
        0.8, 0.3, // item 3
    ];
    let a_specific = vec![0.7, 0.6, 0.5, 0.4];
    let thresholds = vec![
        0.5, -0.5, 0.4, -0.4, 0.3, -0.3, 0.2, -0.2,
    ];

    let params = TwoTierItemParams {
        a_primary,
        a_specific,
        thresholds,
        specific_map,
        n_primary,
        n_specific,
        n_cat,
    };

    let theta_primary = vec![0.25, -0.1, -0.5, 0.3];
    let (gh_nodes, gh_weights) = gh_rule(15).expect("gh_rule(15)");

    let lw = two_tier_lord_wingersky(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("two_tier_lord_wingersky");
    let direct = direct_enumeration_two_tier(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("direct_enumeration_two_tier");
    assert_eq!(lw.len(), direct.len());

    let total_max = params.total_max_score();
    for person in 0..2 {
        let row = person * (total_max + 1);
        let sum_lw: f64 = lw[row..row + total_max + 1].iter().sum();
        let sum_dir: f64 = direct[row..row + total_max + 1].iter().sum();
        assert!((sum_lw - 1.0).abs() <= 1e-10, "person {person} lw sum={sum_lw}");
        assert!((sum_dir - 1.0).abs() <= 1e-10, "person {person} direct sum={sum_dir}");
        for r in 0..=total_max {
            assert!(
                (lw[row + r] - direct[row + r]).abs() <= 1e-12,
                "person {person} score {r}: lw={} direct={}",
                lw[row + r],
                direct[row + r]
            );
        }
    }

    let expected = two_tier_expected_raw(&params, &theta_primary, &gh_nodes, &gh_weights)
        .expect("two_tier_expected_raw");
    assert_eq!(expected.len(), 2);
    for person in 0..2 {
        let row = person * (total_max + 1);
        let mut mean = 0.0_f64;
        for r in 0..=total_max {
            mean += (r as f64) * lw[row + r];
        }
        assert!((expected[person] - mean).abs() <= 1e-12);
    }
}
