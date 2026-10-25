use mlsirm_core::samefit_fifa::transform;
use mlsirm_core::samefit_grm::PackedArchive;
#[test]
fn one_factor_group_scale_and_delta() {
    let a = PackedArchive::new(
        vec![true],
        2,
        3,
        vec![2.0, 1.0, 2f64.ln(), 0.5, 1.5f64.ln()],
    )
    .unwrap();
    let t = transform(&a).unwrap();
    let find = |name: &str| t.rows.iter().find(|r| r.name == name).unwrap();
    let v = std::f64::consts::PI.powi(2) / 3.;
    assert!(
        (find("g1.i0.lambda_logit_latent_response").estimate - 3. / (9. + v).sqrt()).abs() < 1e-12
    );
    assert!(
        (find("g1.i0.threshold_standardized[0]").estimate + 2. / (9. + v).sqrt()).abs() < 1e-12
    );
    assert_eq!(find("g1.i0.intercept[1]").estimate, -1.);
    assert_eq!(find("g1.i0.difficulty[1]").estimate, 0.5);
    // Differentiate every output against every packed coordinate independently.
    let x: Vec<_> = a.packed_bits().iter().map(|&b| f64::from_bits(b)).collect();
    for k in 0..x.len() {
        let mut xp = x.clone();
        let mut xm = x.clone();
        xp[k] += 1e-5;
        xm[k] -= 1e-5;
        let p = transform(&PackedArchive::new(vec![true], 2, 3, xp).unwrap()).unwrap();
        let m = transform(&PackedArchive::new(vec![true], 2, 3, xm).unwrap()).unwrap();
        for r in 0..t.rows.len() {
            assert!(
                ((p.rows[r].estimate - m.rows[r].estimate) / 2e-5 - t.jacobian[r * x.len() + k])
                    .abs()
                    < 2e-8,
                "row {r} col {k}"
            );
        }
    }
}

#[test]
fn rejects_nonfinite_public_transformation_estimate() {
    use mlsirm_core::samefit_fifa::{infer, Row, Transformation};
    let t = Transformation {
        rows: vec![Row {
            name: "invalid".into(),
            estimate: f64::NAN,
        }],
        jacobian: vec![1.],
        n_parameters: 1,
    };
    assert!(infer(&t, &[1.]).is_err());
}
#[test]
fn likelihood_delta_covariance_and_rejections() {
    use mlsirm_core::samefit_fifa::{infer, Row, Transformation};
    let t = Transformation {
        rows: vec![Row {
            name: "sum".into(),
            estimate: 3.,
        }],
        jacobian: vec![1., 2.],
        n_parameters: 2,
    };
    // inverse [[2,1],[1,2]] gives variance 2 exactly, including covariance terms.
    let i = infer(&t, &[2., 1., 1., 2.]).unwrap();
    assert!((i.rows[0].se - 2f64.sqrt()).abs() < 1e-12);
    assert!((i.rows[0].ci95_upper - 3. - 1.959963984540054 * 2f64.sqrt()).abs() < 1e-12);
    assert!(infer(&t, &[1., 2., 2., 1.]).is_err());
    assert!(infer(&t, &[1., 1., 1., 1.]).is_err());
    assert!(infer(&t, &[1., 0., 0., f64::NAN]).is_err());
    assert!(infer(&t, &[2., 1., 0., 2.]).is_err());
}
