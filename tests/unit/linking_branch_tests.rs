use super::*;

#[test]
fn parse_all_methods() {
    for (s, m) in [
        ("mean-mean", LinkMethod::MeanMean),
        ("mm", LinkMethod::MeanMean),
        ("MEAN_SIGMA", LinkMethod::MeanSigma),
        ("ms", LinkMethod::MeanSigma),
        ("Haebara", LinkMethod::Haebara),
        ("hb", LinkMethod::Haebara),
        ("stocking-lord", LinkMethod::StockingLord),
        ("SL", LinkMethod::StockingLord),
    ] {
        assert_eq!(LinkMethod::parse(s), Some(m));
    }
    assert_eq!(LinkMethod::parse("nope"), None);
}

#[test]
fn mean_sigma_rejects_zero_spread() {
    // sd(d_new) = 0 makes the mean/sigma scale coefficient unidentified.
    let a_old = vec![1.0, 1.0, 1.0];
    let b_old = vec![-0.3, 0.1, 0.5];
    let a_new = vec![1.0, 1.0, 1.0];
    let b_new = vec![0.0, 0.0, 0.0]; // all difficulties 0
    let (nodes, w) = (vec![-1.0, 0.0, 1.0], vec![0.25, 0.5, 0.25]);
    assert!(irt_link(
        &a_old,
        &b_old,
        &a_new,
        &b_new,
        &nodes,
        &w,
        LinkMethod::MeanSigma,
    )
    .is_err());
}

#[test]
fn cc_objective_penalizes_nonpositive_slope() {
    let a = vec![1.0, 1.0];
    let b = vec![0.0, 0.0];
    let th = vec![0.0];
    let w = vec![1.0];
    // slope <= 1e-6 and non-finite intercept both return the 1e18 penalty
    assert_eq!(cc_objective(0.0, 0.0, &a, &b, &a, &b, &th, &w, true), 1e18);
    assert_eq!(
        cc_objective(1.0, f64::NAN, &a, &b, &a, &b, &th, &w, false),
        1e18
    );
}

#[test]
fn nelder_mead_minimizes_nonsmooth() {
    // a non-smooth V forces contraction/shrink steps, not just reflection
    let result = nelder_mead(&|a, b| (a - 2.0).abs() + 3.0 * (b + 1.0).abs(), [8.0, 8.0]);
    assert!(
        (result.x[0] - 2.0).abs() < 1e-3 && (result.x[1] + 1.0).abs() < 1e-3,
        "x = {:?}",
        result.x
    );
    assert!(result.objective < 1e-3 && result.n_iter > 1);
    assert!(result.converged, "{result:?}");
    assert!(result.final_objective_span <= result.objective_tolerance);
    assert!(result.final_parameter_span <= result.parameter_tolerance);
}

#[test]
fn irt_link_rejects_bad_slopes_and_grids() {
    let a = vec![1.0, 1.0, 1.0];
    let b = vec![-0.3, 0.1, 0.5];
    let bad = vec![0.0, 1.0, 1.0]; // a slope <= 0
    let (nodes, w) = (vec![-1.0, 0.0, 1.0], vec![0.25, 0.5, 0.25]);
    assert!(irt_link(&a, &b, &bad, &b, &nodes, &w, LinkMethod::MeanMean).is_err());
    // empty / mismatched grid for a characteristic-curve method
    assert!(irt_link(&a, &b, &a, &b, &[], &[], LinkMethod::Haebara).is_err());
    assert!(irt_link(&a, &b, &a, &b, &nodes, &[0.5], LinkMethod::StockingLord).is_err());

    let nan_intercept = vec![-0.3, f64::NAN, 0.5];
    assert!(irt_link(&a, &nan_intercept, &a, &b, &nodes, &w, LinkMethod::MeanMean,).is_err());
    assert!(irt_link(
        &a,
        &b,
        &a,
        &b,
        &nodes,
        &[0.25, f64::NAN, 0.25],
        LinkMethod::StockingLord,
    )
    .is_err());
    assert!(irt_link(
        &a,
        &b,
        &a,
        &b,
        &nodes,
        &[0.25, -0.1, 0.25],
        LinkMethod::Haebara,
    )
    .is_err());
    assert!(irt_link(
        &a,
        &b,
        &a,
        &b,
        &nodes,
        &[0.0, 0.0, 0.0],
        LinkMethod::Haebara,
    )
    .is_err());
}
