use mlsirm_core::samefit_grm::{quadrature, PackedArchive};
use mlsirm_core::samefit_grm_joint::analytic_moments_delta;
#[test]
fn analytic_common_free_prior_columns_match_central_differences() {
    let anchor = vec![true, false];
    let x = vec![
        1.0, 0.5, 0.0, -0.7, 0.8, 0.0, -0.5, 0.6, 0.0, -0.3, 0.4, 0.0, 0.2, 0.1, -0.2, -0.1,
    ];
    let a = PackedArchive::new(anchor.clone(), 3, 3, x.clone()).unwrap();
    let (n, w) = quadrature(121).unwrap();
    let (m, d) = analytic_moments_delta(&a, &n, &w).unwrap();
    for c in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[c] += 1e-5;
        minus[c] -= 1e-5;
        let (mp, _) = analytic_moments_delta(
            &PackedArchive::new(anchor.clone(), 3, 3, plus).unwrap(),
            &n,
            &w,
        )
        .unwrap();
        let (mm, _) = analytic_moments_delta(
            &PackedArchive::new(anchor.clone(), 3, 3, minus).unwrap(),
            &n,
            &w,
        )
        .unwrap();
        for r in 0..m.len() {
            assert!(
                (d[r * x.len() + c] - (mp[r] - mm[r]) / 2e-5).abs() < 1e-8,
                "row {r} col {c}"
            );
        }
    }
}
