//! Original-pair projection through the nominal Newton caller.
//! Basis: Higham (1989, author reprint pp. 3–4, Section 2).
//! Reference: Higham, N. J. (1989). Matrix nearness problems and applications.
//! In M. J. C. Gover & S. Barnett (Eds.), Applications of matrix theory
//! (pp. 1–27). Oxford University Press.
use super::*;

type Matrix = Vec<Vec<f64>>;
type Projection = (Matrix, Matrix);
thread_local! {
    static CAPTURE: std::cell::RefCell<Option<Vec<Projection>>> = const { std::cell::RefCell::new(None) };
}

pub(super) fn raw_if_active(matrix: &Matrix) -> Option<Matrix> {
    CAPTURE.with(|value| value.borrow().as_ref().map(|_| matrix.clone()))
}

pub(super) fn record(raw: Matrix, delivered: &Matrix) {
    CAPTURE.with(|value| {
        if let Some(rows) = value.borrow_mut().as_mut() {
            rows.push((raw, delivered.clone()));
        }
    });
}

struct CaptureGuard;
impl CaptureGuard {
    fn start() -> Self {
        CAPTURE.with(|value| {
            let mut capture = value.borrow_mut();
            assert!(capture.is_none(), "capture must not be nested");
            *capture = Some(Vec::new());
        });
        Self
    }

    fn take(&self) -> Vec<Projection> {
        CAPTURE.with(|value| value.borrow_mut().take().unwrap())
    }
}
impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE.with(|value| *value.borrow_mut() = None);
    }
}

fn assert_projection(raw: &Matrix, delivered: &Matrix, ridge: f64) {
    assert_eq!(raw.len(), delivered.len());
    for r in 0..raw.len() {
        assert_eq!(raw[r].len(), raw.len());
        assert_eq!(delivered[r].len(), raw.len());
        assert_eq!(
            delivered[r][r].to_bits(),
            (raw[r][r] + ridge).to_bits(),
            "diagonal ridge must be applied once"
        );
        for c in 0..r {
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

fn d1_fixture() -> (Vec<f64>, Vec<Vec<f64>>) {
    let (nodes, _) = crate::quadrature::require_gh_rule(121, "q").unwrap();
    let counts = (0..nodes.len())
        .map(|nd| {
            vec![
                0.3 + (nd % 3) as f64 * 0.1,
                0.5 + (nd % 5) as f64 * 0.1,
                0.7,
            ]
        })
        .collect();
    (nodes.to_vec(), counts)
}

#[test]
fn nominal_d1_objective_and_gradient_match_scalar_cell_path() {
    use crate::poly::gpcm_node_gradient;
    let (nodes, counts) = d1_fixture();
    let params = vec![1.1, -0.4, 0.2, -0.3];
    let scores = vec![0.0, params[0], params[1]];
    let intercepts = vec![0.0, params[2], params[3]];
    let mut scalar_ll = 0.0;
    let mut scalar_grad = vec![0.0f64; 4];
    for (nd, &theta) in nodes.iter().enumerate() {
        let lp = gpcm_logprobs(theta, &scores, &intercepts);
        scalar_ll += counts[nd].iter().zip(&lp).map(|(r, l)| r * l).sum::<f64>();
        let (g_int, _, g_sc) = gpcm_node_gradient(theta, &scores, &intercepts, &counts[nd]);
        for k in 0..2 {
            scalar_grad[k] += g_sc[k];
            scalar_grad[2 + k] += g_int[k];
        }
    }
    let (objective, gradient) = nm_item_neg_ll_grad(&params, &[0], &nodes, 1, &counts, 3);
    assert_eq!(objective.to_bits(), (-scalar_ll).to_bits());
    for (actual, expected) in gradient.iter().zip(&scalar_grad) {
        assert_eq!(actual.to_bits(), (-expected).to_bits());
    }
    // The positive is nontrivial and distinguishes a sign/parameter-slot error.
    assert!(objective.is_finite() && objective > 0.0);
    assert!(gradient.iter().all(|g| g.is_finite()));
    assert!(gradient.iter().any(|g| g.abs() > 1e-3));
    let mut changed = params.clone();
    changed[0] += 0.1;
    let (changed_objective, _) = nm_item_neg_ll_grad(&changed, &[0], &nodes, 1, &counts, 3);
    assert_ne!(objective.to_bits(), changed_objective.to_bits());
}

fn check_actual_caller(
    params: Vec<f64>,
    dims: &[usize],
    nodes: &[f64],
    n_dims: usize,
    counts: &[Vec<f64>],
    n_cat: usize,
    ridge: f64,
) {
    let (before, _) = nm_item_neg_ll_grad(&params, dims, nodes, n_dims, counts, n_cat);
    let capture = CaptureGuard::start();
    let result = nm_m_step(params.clone(), dims, nodes, n_dims, counts, n_cat, ridge, 1);
    let rows = capture.take();
    assert_eq!(rows.len(), 1, "actual Newton caller must reach its solver");
    let (raw, delivered) = &rows[0];
    let max_asymmetry = (0..raw.len())
        .flat_map(|r| (0..r).map(move |c| (raw[r][c] - raw[c][r]).abs()))
        .fold(0.0f64, f64::max);
    assert!(
        max_asymmetry > 0.0,
        "fixture must have asymmetric FD entries"
    );
    eprintln!("nominal projection D={n_dims} dims={dims:?} ridge={ridge:.17e} max_raw_asymmetry={max_asymmetry:.17e}");
    assert_projection(raw, delivered, ridge);
    let (after, _) = nm_item_neg_ll_grad(&result, dims, nodes, n_dims, counts, n_cat);
    assert!(result.iter().all(|value| value.is_finite()));
    assert_ne!(params, result, "positive must take a nonzero step");
    assert!(
        after < before,
        "same-objective update must improve: {before} -> {after}"
    );
    eprintln!("nominal same-objective before={before:.17e} after={after:.17e}");
}

#[test]
fn nominal_hessian_actual_d1_uses_original_pairs() {
    let (nodes, counts) = d1_fixture();
    check_actual_caller(
        vec![1.1, -0.4, 0.2, -0.3],
        &[0],
        &nodes,
        1,
        &counts,
        3,
        1e-8,
    );
}

#[test]
fn nominal_hessian_actual_sparse_multidimensional_uses_original_pairs() {
    let xn = build_xi_nodes(
        XiRule::Halton {
            n: 121,
            shift_seed: 17,
        },
        3,
    )
    .unwrap();
    let counts: Vec<Vec<f64>> = (0..xn.logw.len())
        .map(|nd| {
            vec![
                0.2 + (nd % 3) as f64 * 0.1,
                0.4,
                0.6 + (nd % 5) as f64 * 0.1,
            ]
        })
        .collect();
    check_actual_caller(
        vec![1.1, -0.4, -0.7, 0.9, 0.2, -0.3],
        &[0, 2],
        &xn.grid,
        3,
        &counts,
        3,
        3e-7,
    );
}

#[test]
fn nominal_projection_oracle_accepts_symmetric_and_original_pair_controls() {
    let raw = vec![vec![2.0, 1.0], vec![5.0, 7.0]];
    let projected = vec![vec![2.125, 3.0], vec![3.0, 7.125]];
    assert_projection(&raw, &projected, 0.125);
    let symmetric = vec![vec![2.0, 3.0], vec![3.0, 7.0]];
    assert_projection(&symmetric, &projected, 0.125);
}

#[test]
#[should_panic(expected = "lower pair (1,0) must use original entries")]
fn nominal_projection_oracle_rejects_sequential_average() {
    let raw = vec![vec![2.0, 1.0], vec![5.0, 7.0]];
    let mut sequential = raw.clone();
    for r in 0..raw.len() {
        for c in 0..raw.len() {
            sequential[r][c] = 0.5 * (sequential[r][c] + sequential[c][r]);
        }
        sequential[r][r] += 0.125;
    }
    assert_projection(&raw, &sequential, 0.125);
}

#[test]
#[should_panic(expected = "lower pair (1,0) must use original entries")]
fn nominal_projection_oracle_rejects_symmetric_but_wrong_average() {
    let raw = vec![vec![2.0, 1.0], vec![5.0, 7.0]];
    let wrong = vec![vec![2.125, 4.0], vec![4.0, 7.125]];
    assert_projection(&raw, &wrong, 0.125);
}
