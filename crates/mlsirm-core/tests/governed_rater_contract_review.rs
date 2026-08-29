use mlsirm_core::governed_rater_contracts::{
    CriterionObservation, DomainReference, RaterConfigurationIdentity, RaterInvocation,
    UncertaintyLevel,
};

fn reference(value: &str) -> DomainReference {
    DomainReference::parse("reference", value).expect("valid reference")
}

fn configuration() -> RaterConfigurationIdentity {
    RaterConfigurationIdentity {
        rater_family_ref: reference("rater-family"),
        provider_ref: reference("provider"),
        implementation_revision_ref: reference("implementation-v1"),
        instruction_revision_ref: reference("instruction-v1"),
        response_schema_revision_ref: reference("schema-v1"),
        workflow_mode_ref: reference("blind-independent"),
        modality_channel_ref: reference("text"),
    }
}

fn observation(index: usize) -> CriterionObservation {
    CriterionObservation::observed(
        reference(&format!("criterion-{index}")),
        reference("anchor"),
        vec![reference(&format!("evidence-{index}"))],
        UncertaintyLevel::Medium,
        Vec::new(),
    )
    .expect("bounded observation")
}

#[test]
fn references_must_arrive_in_canonical_identity_form() {
    assert_eq!(reference("canonical-reference").as_str(), "canonical-reference");
    assert!(DomainReference::parse("reference", " leading").is_err());
    assert!(DomainReference::parse("reference", "trailing ").is_err());
    assert!(DomainReference::parse("reference", "a\u{0085}b").is_err());
}

#[test]
fn rust_collection_limits_match_the_published_schema() {
    let evidence = (0..65)
        .map(|index| reference(&format!("evidence-{index}")))
        .collect();
    assert!(CriterionObservation::observed(
        reference("criterion"),
        reference("anchor"),
        evidence,
        UncertaintyLevel::Medium,
        Vec::new(),
    )
    .is_err());

    let review_signals = (0..33)
        .map(|index| reference(&format!("review-{index}")))
        .collect();
    assert!(CriterionObservation::abstained(
        reference("criterion"),
        reference("reason"),
        UncertaintyLevel::Medium,
        review_signals,
    )
    .is_err());

    let observations = (0..129).map(observation).collect();
    assert!(RaterInvocation::new(
        reference("invocation"),
        configuration(),
        reference("task-v1"),
        reference("rubric-v1"),
        reference("response-evidence"),
        observations,
    )
    .is_err());
}
