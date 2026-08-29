use mlsirm_core::governed_rater_contracts::{
    ContractError, CriterionObservation, DomainReference, RaterConfigurationIdentity,
    RaterInvocation, UncertaintyLevel,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct ConformanceFixture {
    reference_cases: Vec<ReferenceCase>,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    value: String,
    valid: bool,
}

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
fn references_follow_the_shared_cross_sdk_conformance_fixture() {
    let fixture: ConformanceFixture = serde_json::from_str(include_str!(
        "../../../contracts/governed-rater-observation-v1.conformance.json"
    ))
    .expect("valid conformance fixture");

    for case in fixture.reference_cases {
        assert_eq!(
            DomainReference::parse("reference", case.value).is_ok(),
            case.valid,
            "reference conformance case: {}",
            case.name
        );
    }
}

#[test]
fn rust_collection_limits_match_the_published_schema() {
    let max_evidence = (0..64)
        .map(|index| reference(&format!("evidence-{index}")))
        .collect();
    assert!(CriterionObservation::observed(
        reference("criterion"),
        reference("anchor"),
        max_evidence,
        UncertaintyLevel::Medium,
        Vec::new(),
    )
    .is_ok());

    let too_many_evidence = (0..65)
        .map(|index| reference(&format!("evidence-{index}")))
        .collect();
    assert_eq!(
        CriterionObservation::observed(
            reference("criterion"),
            reference("anchor"),
            too_many_evidence,
            UncertaintyLevel::Medium,
            Vec::new(),
        ),
        Err(ContractError::TooManyEvidenceReferences)
    );

    let max_review_signals = (0..32)
        .map(|index| reference(&format!("review-{index}")))
        .collect::<Vec<_>>();
    assert!(CriterionObservation::observed(
        reference("criterion"),
        reference("anchor"),
        vec![reference("evidence")],
        UncertaintyLevel::Medium,
        max_review_signals.clone(),
    )
    .is_ok());
    assert!(CriterionObservation::abstained(
        reference("criterion"),
        reference("reason"),
        UncertaintyLevel::Medium,
        max_review_signals,
    )
    .is_ok());

    let too_many_review_signals = (0..33)
        .map(|index| reference(&format!("review-{index}")))
        .collect::<Vec<_>>();
    assert_eq!(
        CriterionObservation::observed(
            reference("criterion"),
            reference("anchor"),
            vec![reference("evidence")],
            UncertaintyLevel::Medium,
            too_many_review_signals.clone(),
        ),
        Err(ContractError::TooManyReviewSignals)
    );
    assert_eq!(
        CriterionObservation::abstained(
            reference("criterion"),
            reference("reason"),
            UncertaintyLevel::Medium,
            too_many_review_signals,
        ),
        Err(ContractError::TooManyReviewSignals)
    );

    let max_observations = (0..128).map(observation).collect();
    assert!(RaterInvocation::new(
        reference("invocation"),
        configuration(),
        reference("task-v1"),
        reference("rubric-v1"),
        reference("response-evidence"),
        max_observations,
    )
    .is_ok());

    let too_many_observations = (0..129).map(observation).collect();
    assert_eq!(
        RaterInvocation::new(
            reference("invocation"),
            configuration(),
            reference("task-v1"),
            reference("rubric-v1"),
            reference("response-evidence"),
            too_many_observations,
        ),
        Err(ContractError::TooManyObservations)
    );
}
