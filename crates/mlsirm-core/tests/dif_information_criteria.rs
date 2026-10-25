use mlsirm_core::dif_information_criteria::{compare_dif_information_criteria, DifFitEvidence};
use mlsirm_core::fitstats::information_criteria;

fn evidence(loglik: f64, n_parameters: usize) -> DifFitEvidence {
    DifFitEvidence {
        loglik,
        n_parameters,
        n_persons: 200,
        converged: true,
        likelihood_kind: "unpenalized_marginal_ml".into(),
        comparison_key: "same-data-row-order-estimator-integration".into(),
    }
}

#[test]
fn aic_negative_augmented_minus_compact_selects_augmented() {
    let compact = evidence(-500.0, 10);
    let augmented = evidence(-496.0, 12);
    let result = compare_dif_information_criteria(&compact, &augmented, "AIC").unwrap();
    assert_eq!(result.delta_ic, -4.0);
    assert!(result.prefer_augmented);
    assert_eq!(result.compact_ic, 1020.0);
    assert_eq!(result.augmented_ic, 1016.0);
}

#[test]
fn criteria_reuse_existing_definition_and_retain_compact_on_a_tie() {
    use mlsirm_core::fitstats::information_criteria;
    let compact = evidence(-500.0, 10);
    let augmented = evidence(-496.0, 12);
    let c = information_criteria(-500.0, 10, 200);
    let a = information_criteria(-496.0, 12, 200);
    for (name, expected_c, expected_a) in [
        ("AIC", c.aic, a.aic),
        ("BIC", c.bic, a.bic),
        ("AICc", c.aicc, a.aicc),
        ("SABIC", c.sabic, a.sabic),
        ("CAIC", c.caic, a.caic),
    ] {
        let r = compare_dif_information_criteria(&compact, &augmented, name).unwrap();
        assert_eq!(r.compact_ic, expected_c, "{name}");
        assert_eq!(r.augmented_ic, expected_a, "{name}");
        assert_eq!(r.delta_ic, expected_a - expected_c, "{name}");
        assert_eq!(r.prefer_augmented, expected_a < expected_c, "{name}");
    }
    let tie = compare_dif_information_criteria(&compact, &evidence(-498.0, 12), "AIC").unwrap();
    assert_eq!(tie.delta_ic, 0.0);
    assert!(!tie.prefer_augmented);
}

#[test]
fn dic_is_explicitly_unsupported_not_an_alias_for_delta_ic() {
    let c = evidence(-500.0, 10);
    let a = evidence(-496.0, 12);
    let error = compare_dif_information_criteria(&c, &a, "DIC").unwrap_err();
    assert!(error.contains("posterior"), "{error}");
    for invalid in ["dIC", "delta_ic", "Holm", "HQ", "", "bic"] {
        assert!(
            compare_dif_information_criteria(&c, &a, invalid).is_err(),
            "{invalid}"
        );
    }
}

#[test]
fn invalid_fit_evidence_cannot_select() {
    let c = evidence(-500.0, 10);
    let a = evidence(-496.0, 12);
    let mut invalid = Vec::new();
    let mut x = c.clone();
    x.converged = false;
    invalid.push(x);
    let mut x = c.clone();
    x.loglik = f64::NAN;
    invalid.push(x);
    let mut x = c.clone();
    x.loglik = f64::INFINITY;
    invalid.push(x);
    let mut x = c.clone();
    x.n_persons = 0;
    invalid.push(x);
    let mut x = c.clone();
    x.n_persons = 199;
    invalid.push(x);
    let mut x = c.clone();
    x.comparison_key.clear();
    invalid.push(x);
    let mut x = c.clone();
    x.comparison_key = "different-grid".into();
    invalid.push(x);
    let mut x = c.clone();
    x.likelihood_kind = "map_logposterior".into();
    invalid.push(x);
    let mut x = c.clone();
    x.n_parameters = 12;
    invalid.push(x);
    for bad in invalid {
        assert!(
            compare_dif_information_criteria(&bad, &a, "AIC").is_err(),
            "{bad:?}"
        );
    }
    let mut bad_aug = a.clone();
    bad_aug.converged = false;
    assert!(compare_dif_information_criteria(&c, &bad_aug, "AIC").is_err());
}

#[test]
fn undefined_aicc_and_overflow_do_not_select() {
    let mut c = evidence(-500.0, 10);
    let mut a = evidence(-496.0, 12);
    c.n_persons = 13;
    a.n_persons = 13;
    assert!(compare_dif_information_criteria(&c, &a, "AICc").is_err());
    c.n_persons = 200;
    a.n_persons = 200;
    c.loglik = -f64::MAX;
    assert!(compare_dif_information_criteria(&c, &a, "BIC").is_err());
}

#[test]
fn aicc_integer_domain_rejects_undefined_and_maximum_counts_without_overflow() {
    let mut cases = vec![
        (13, 12),
        (12, 12),
        (11, 12),
        (usize::MAX, usize::MAX),
        (usize::MAX, usize::MAX - 1),
        (usize::MAX - 1, usize::MAX - 2),
    ];
    if usize::BITS > 53 {
        cases.push((
            9_007_199_254_740_994_u64 as usize,
            9_007_199_254_740_993_u64 as usize,
        ));
    }
    for (n, k) in cases {
        assert!(
            information_criteria(-496.0, k, n).aicc.is_nan(),
            "N={n}, K={k}"
        );
        let mut compact = evidence(-500.0, 0);
        let mut augmented = evidence(-496.0, k);
        compact.n_persons = n;
        augmented.n_persons = n;
        let error = compare_dif_information_criteria(&compact, &augmented, "AICc").unwrap_err();
        assert!(error.contains("AICc is undefined"), "{error}");
    }
    assert!(information_criteria(-500.0, usize::MAX, 0).aicc.is_nan());
}

#[test]
fn aicc_uses_exact_positive_integer_denominator_at_small_and_maximum_boundaries() {
    let mut cases = vec![(14, 12), (usize::MAX, usize::MAX - 2)];
    if usize::BITS > 53 {
        cases.push((
            9_007_199_254_740_995_u64 as usize,
            9_007_199_254_740_993_u64 as usize,
        ));
        cases.push((
            9_007_199_254_740_994_u64 as usize,
            9_007_199_254_740_992_u64 as usize,
        ));
    }
    for (n, k) in cases {
        assert_eq!(n.checked_sub(k).and_then(|gap| gap.checked_sub(1)), Some(1));
        let kf = k as f64;
        let expected = 992.0 + 2.0 * kf + 2.0 * kf * (kf + 1.0);
        let ic = information_criteria(-496.0, k, n);
        assert_eq!(ic.aicc, expected, "N={n}, K={k}");
        let mut compact = evidence(-500.0, 0);
        let mut augmented = evidence(-496.0, k);
        compact.n_persons = n;
        augmented.n_persons = n;
        let result = compare_dif_information_criteria(&compact, &augmented, "AICc").unwrap();
        assert_eq!(result.compact_ic, 1000.0);
        assert_eq!(result.augmented_ic, expected);
        assert_eq!(result.delta_ic, expected - 1000.0);
        assert_eq!(result.prefer_augmented, expected < 1000.0);
    }
}

#[test]
fn undefined_aicc_does_not_reject_other_information_criteria() {
    let mut compact = evidence(-500.0, 0);
    let mut augmented = evidence(-496.0, usize::MAX);
    compact.n_persons = usize::MAX;
    augmented.n_persons = usize::MAX;
    for criterion in ["AIC", "BIC", "SABIC", "CAIC"] {
        let result = compare_dif_information_criteria(&compact, &augmented, criterion).unwrap();
        assert!(result.compact_ic.is_finite());
        assert!(result.augmented_ic.is_finite());
        assert!(result.delta_ic.is_finite());
    }
}
