use super::*;

fn gh21() -> (Vec<f64>, Vec<f64>) {
    // coarse standard-normal grid (nodes, weights) sufficient for CC linking
    let nodes: Vec<f64> = (0..41).map(|i| -4.0 + 0.2 * i as f64).collect();
    let w: Vec<f64> = nodes
        .iter()
        .map(|&t| (-0.5 * t * t).exp() / (2.0 * std::f64::consts::PI).sqrt() * 0.2)
        .collect();
    (nodes, w)
}

fn recover(method: LinkMethod) {
    // old-form items (eta form), generate a new form by a known transform
    let a_old = vec![1.2, 0.8, 1.5, 1.0, 0.9, 1.3, 1.1, 0.7];
    let b_old = vec![-0.5, 0.3, 1.0, -1.2, 0.0, 0.6, -0.8, 0.4];
    let (a0, b0) = (1.3_f64, 0.4_f64); // true theta_old = 1.3*theta_new + 0.4
                                       // a_new = A*a_old ; b_new = b_old + a_old*B  (inverse of the transform)
    let a_new: Vec<f64> = a_old.iter().map(|&a| a0 * a).collect();
    let b_new: Vec<f64> = a_old
        .iter()
        .zip(&b_old)
        .map(|(&a, &b)| b + a * b0)
        .collect();
    let (theta, weight) = gh21();
    let res = irt_link(&a_old, &b_old, &a_new, &b_new, &theta, &weight, method).unwrap();
    assert!(
        (res.slope - a0).abs() < 1e-3 && (res.intercept - b0).abs() < 1e-3,
        "{method:?}: recovered ({}, {}) vs (1.3, 0.4)",
        res.slope,
        res.intercept
    );
    assert!(res.converged, "{method:?}: {res:?}");
    match method {
        LinkMethod::MeanMean | LinkMethod::MeanSigma => {
            assert_eq!(res.termination_reason, "closed_form");
            assert_eq!(res.n_iter, 0);
        }
        LinkMethod::Haebara | LinkMethod::StockingLord => {
            assert_eq!(res.termination_reason, "tolerance_met");
            assert!(res.n_iter < res.max_iter);
            assert!(res.final_objective_span <= res.objective_tolerance);
            assert!(res.final_parameter_span <= res.parameter_tolerance);
        }
    }
}

#[test]
fn mean_sigma_recovers_transform() {
    recover(LinkMethod::MeanSigma);
}

#[test]
fn mean_mean_recovers_transform() {
    recover(LinkMethod::MeanMean);
}

#[test]
fn haebara_recovers_transform() {
    recover(LinkMethod::Haebara);
}

#[test]
fn stocking_lord_recovers_transform() {
    recover(LinkMethod::StockingLord);
}

#[test]
fn rejects_bad_input() {
    let (theta, weight) = gh21();
    assert!(irt_link(
        &[1.0],
        &[0.0],
        &[1.0],
        &[0.0],
        &theta,
        &weight,
        LinkMethod::MeanSigma
    )
    .is_err());

    assert!(irt_link(
        &[1.0, 1.2],
        &[0.0, 0.5],
        &[1.1, 1.3],
        &[0.1, 0.6],
        &[f64::NAN],
        &[1.0],
        LinkMethod::Haebara,
    )
    .is_err());

    assert!(moment(&[0.0, 0.0], &[0.0, 1.0], &[1.0, 1.0], &[0.0, 1.0], false).is_err());
    let shrink = nelder_mead(&|_, _| 1.0, [0.0, 0.0]);
    assert!(shrink.converged);
    assert_eq!(link_termination_reason(false), "max_iter_reached");
}

#[test]
fn characteristic_linking_falls_back_when_difficulty_spread_is_zero() {
    let result = irt_link(
        &[1.0, 2.0],
        &[0.0, 0.0],
        &[1.2, 2.4],
        &[0.0, 0.0],
        &[-1.0, 0.0, 1.0],
        &[0.25, 0.5, 0.25],
        LinkMethod::Haebara,
    )
    .unwrap();
    assert!(result.slope.is_finite() && result.slope > 0.0);
    assert!(result.intercept.is_finite());
}
