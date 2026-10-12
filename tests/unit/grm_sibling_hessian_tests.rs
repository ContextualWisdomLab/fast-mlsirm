//! Original-pair Hessian checks through the existing Newton caller.
//! Basis: Higham (1989, author reprint pp. 3–4, §2).
//! Reference: Higham, N. J. (1989). Matrix nearness problems and applications.
//! In M. J. C. Gover & S. Barnett (Eds.), Applications of matrix theory
//! (pp. 1–27). Oxford University Press.
use super::*;
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Vec<(Vec<Vec<f64>>, Vec<Vec<f64>>)>>> = const { std::cell::RefCell::new(None) };
}
pub(super) fn raw_if_active(matrix: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    CAPTURE.with(|v| v.borrow().as_ref().map(|_| matrix.clone()))
}
pub(super) fn record(raw: Vec<Vec<f64>>, delivered: &Vec<Vec<f64>>) {
    CAPTURE.with(|v| {
        if let Some(rows) = v.borrow_mut().as_mut() {
            rows.push((raw, delivered.clone()));
        }
    });
}
fn start() {
    CAPTURE.with(|v| *v.borrow_mut() = Some(Vec::new()));
}
fn check(ridge: f64) {
    let rows = CAPTURE.with(|v| v.borrow_mut().take().unwrap());
    assert!(
        !rows.is_empty(),
        "actual Newton caller must reach its solver"
    );
    let mut asymmetric_input = false;
    for (raw, delivered) in rows {
        for r in 0..raw.len() {
            assert_eq!(delivered[r][r].to_bits(), (raw[r][r] + ridge).to_bits());
            for c in 0..r {
                asymmetric_input |= raw[r][c].to_bits() != raw[c][r].to_bits();
                let expected = 0.5 * (raw[r][c] + raw[c][r]);
                assert_eq!(
                    delivered[r][c].to_bits(),
                    expected.to_bits(),
                    "lower pair ({r},{c}) must use original entries"
                );
                assert_eq!(
                    delivered[c][r].to_bits(),
                    expected.to_bits(),
                    "upper pair ({c},{r}) must use original entries"
                );
            }
        }
    }
    assert!(
        asymmetric_input,
        "fixture must exercise a genuinely asymmetric FD approximation"
    );
}
fn fixture() -> (Vec<f64>, Vec<Vec<f64>>) {
    let (x, _) = crate::quadrature::require_gh_rule(121, "q").unwrap();
    let counts = (0..x.len())
        .map(|i| vec![0.3 + (i % 3) as f64 * 0.1, 0.5 + (i % 5) as f64 * 0.1, 0.7])
        .collect();
    (x.to_vec(), counts)
}
#[test]
fn sibling_hessian_actual_multidimensional_grm() {
    let (x, counts) = fixture();
    let nodes: Vec<f64> = x.iter().flat_map(|v| [*v, v.sin() + 0.3]).collect();
    start();
    let result = grm_m_step(
        vec![1.1, 0.8, 0.8, -0.8],
        &[0, 1],
        &nodes,
        2,
        &counts,
        3,
        1e-8,
        1,
    );
    assert!(result.iter().all(|v| v.is_finite()));
    check(1e-8);
}
