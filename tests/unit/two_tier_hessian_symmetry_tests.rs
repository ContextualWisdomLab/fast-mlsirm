//! Actual Newton caller checks for the original-pair symmetric projection.
//! Basis: Higham (1989, reprint pp. 3–4, §2) gives (A + Aᵀ)/2 and its
//! application to finite-difference Hessians; Cai (2010, p. 608, Appendix A)
//! describes Newton-type M-steps conditional on the E-step results.
//! References: Higham, N. J. (1989). Matrix nearness problems and applications.
//! In M. J. C. Gover & S. Barnett (Eds.), Applications of matrix theory
//! (pp. 1–27). Oxford University Press.
//! Cai, L. (2010). A two-tier full-information item factor analysis model
//! with applications. Psychometrika, 75(4), 581–612.
//! doi:10.1007/s11336-010-9178-0.
use super::*;

fn assert_captured_original_pairs(label: &str, ridge: f64) {
    let rows = NEWTON_HESSIAN_CAPTURE.with(|capture| capture.borrow_mut().take().unwrap());
    assert!(
        !rows.is_empty(),
        "actual {label} Newton caller must reach solve"
    );
    for (actual_label, raw, projected) in rows {
        assert_eq!(actual_label, label);
        for r in 0..raw.len() {
            assert_eq!(projected[r][r].to_bits(), (raw[r][r] + ridge).to_bits());
            for c in 0..r {
                let expected = 0.5 * (raw[r][c] + raw[c][r]);
                assert_eq!(
                    projected[r][c].to_bits(),
                    expected.to_bits(),
                    "{label} lower pair ({r}, {c}) must use the original two values"
                );
                assert_eq!(
                    projected[c][r].to_bits(),
                    expected.to_bits(),
                    "{label} upper pair ({c}, {r}) must use the original two values"
                );
            }
        }
    }
}

#[test]
fn item_newton_passes_original_pair_average_to_solver() {
    let q = 121;
    let (x, _) = gh_rule(q).unwrap();
    let counts: Vec<Vec<f64>> = (0..q * q)
        .map(|node| {
            vec![
                0.3 + (node % 3) as f64 * 0.1,
                0.5 + (node % 5) as f64 * 0.1,
                0.7,
            ]
        })
        .collect();
    let ridge = 1e-8;
    NEWTON_HESSIAN_CAPTURE.with(|capture| *capture.borrow_mut() = Some(Vec::new()));
    let updated = m_step_item(
        vec![1.1, 0.8, 0.8, -0.8],
        &[0],
        true,
        x,
        x,
        1,
        q,
        q,
        &counts,
        3,
        ridge,
        1,
    );
    assert!(updated.iter().all(|x| x.is_finite()));
    assert_captured_original_pairs("item", ridge);
}

#[test]
fn correlation_newton_passes_original_pair_average_to_solver() {
    let ridge = 1e-8;
    let s_bar = [1.0, 0.25, -0.1, 0.25, 1.2, 0.15, -0.1, 0.15, 0.9];
    NEWTON_HESSIAN_CAPTURE.with(|capture| *capture.borrow_mut() = Some(Vec::new()));
    let updated = m_step_phi(vec![0.2, -0.1, 0.3], 3, &s_bar, 180, ridge, 1);
    assert!(updated.iter().all(|x| x.is_finite()));
    assert_captured_original_pairs("correlation", ridge);
}
