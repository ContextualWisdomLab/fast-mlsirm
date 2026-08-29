use mlsirm_core::governed_rater_contracts::{
    ContractError, CriterionObservation, DomainReference, RaterConfigurationIdentity,
    RaterInvocation, UncertaintyLevel,
};
use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use std::collections::HashSet;
use std::fmt;

#[derive(Deserialize)]
struct ConformanceFixture {
    reference_cases: Vec<ReferenceCase>,
    observation_identity_cases: Vec<ObservationIdentityCase>,
}

#[derive(Deserialize)]
struct ReferenceCase {
    name: String,
    value: String,
    valid: bool,
}

#[derive(Deserialize)]
struct ObservationIdentityCase {
    name: String,
    valid: bool,
    #[serde(default)]
    value: Option<serde_json::Value>,
    #[serde(default)]
    stage: Option<String>,
    #[serde(default)]
    json_text: Option<String>,
}

struct UniqueJsonValue;

impl<'de> Deserialize<'de> for UniqueJsonValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(UniqueJsonVisitor)
    }
}

struct UniqueJsonVisitor;

impl<'de> Visitor<'de> for UniqueJsonVisitor {
    type Value = UniqueJsonValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("JSON with unique object member names at every depth")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_string<E>(self, _value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(UniqueJsonValue)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element::<UniqueJsonValue>()?.is_some() {}
        Ok(UniqueJsonValue)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut seen = HashSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !seen.insert(key.clone()) {
                return Err(de::Error::custom(format!("duplicate object member: {key}")));
            }
            map.next_value::<UniqueJsonValue>()?;
        }
        Ok(UniqueJsonValue)
    }
}

fn has_unique_object_members(raw: &str) -> bool {
    let mut deserializer = serde_json::Deserializer::from_str(raw);
    let result = UniqueJsonValue::deserialize(&mut deserializer);
    result.is_ok() && deserializer.end().is_ok()
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
fn duplicate_member_admission_is_gated_by_the_shared_fixture() {
    let fixture: ConformanceFixture = serde_json::from_str(include_str!(
        "../../../contracts/governed-rater-observation-v1.conformance.json"
    ))
    .expect("valid conformance fixture");

    for case in fixture.observation_identity_cases {
        let raw = match (case.value, case.json_text) {
            (Some(value), None) => serde_json::to_string(&value).expect("serializable fixture value"),
            (None, Some(raw)) => raw,
            _ => panic!("observation identity case must contain exactly one payload form"),
        };
        assert_eq!(
            has_unique_object_members(&raw),
            case.valid,
            "observation identity case: {} ({})",
            case.name,
            case.stage.as_deref().unwrap_or("unique-member admission")
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
