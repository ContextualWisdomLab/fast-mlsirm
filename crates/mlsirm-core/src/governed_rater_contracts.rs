//! Domain-neutral governed-rater published language.
//!
//! This module owns only observation-boundary invariants. It deliberately does
//! not calculate latent scores, cut scores, placements, certifications, or
//! other product decisions.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

/// Published-language version owned by the measurement-calibration context.
pub const GOVERNED_RATER_OBSERVATION_CONTRACT_V1: &str =
    "cwl_governed_rater_observation/v1";

const MAX_REFERENCE_LENGTH: usize = 256;

/// A bounded, non-empty opaque reference crossing a bounded-context boundary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct DomainReference(String);

impl DomainReference {
    /// Parse an opaque reference while rejecting whitespace-only, oversized, or
    /// control-character-bearing values.
    pub fn parse(
        field_name: &'static str,
        value: impl Into<String>,
    ) -> Result<Self, ContractError> {
        let value = value.into();
        let normalized = value.trim();
        if normalized.is_empty() {
            return Err(ContractError::EmptyReference(field_name));
        }
        if normalized.chars().count() > MAX_REFERENCE_LENGTH {
            return Err(ContractError::ReferenceTooLong(field_name));
        }
        if normalized.chars().any(char::is_control) {
            return Err(ContractError::ControlCharacter(field_name));
        }
        Ok(Self(normalized.to_owned()))
    }

    /// Borrow the exact normalized reference text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Validation failures at the governed-rater published-language boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContractError {
    /// A required reference was empty or whitespace-only.
    EmptyReference(&'static str),
    /// A reference exceeded the bounded maximum length.
    ReferenceTooLong(&'static str),
    /// A reference contained a control character.
    ControlCharacter(&'static str),
    /// The same evidence reference appeared more than once.
    DuplicateEvidenceReference,
    /// The same review signal appeared more than once.
    DuplicateReviewSignal,
    /// An observed category did not carry at least one evidence reference.
    ObservedWithoutEvidence,
    /// An invocation did not contain any criterion observations.
    EmptyObservationSet,
    /// An invocation contained more than one observation for a criterion.
    DuplicateCriterionObservation,
}

impl Display for ContractError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyReference(field) => write!(formatter, "{field} must not be empty"),
            Self::ReferenceTooLong(field) => {
                write!(formatter, "{field} exceeds the reference limit")
            }
            Self::ControlCharacter(field) => {
                write!(formatter, "{field} contains a control character")
            }
            Self::DuplicateEvidenceReference => {
                formatter.write_str("evidence references must be unique")
            }
            Self::DuplicateReviewSignal => {
                formatter.write_str("review signals must be unique")
            }
            Self::ObservedWithoutEvidence => {
                formatter.write_str("an observed category requires evidence")
            }
            Self::EmptyObservationSet => {
                formatter.write_str("an invocation requires observations")
            }
            Self::DuplicateCriterionObservation => {
                formatter.write_str("criterion observations must be unique")
            }
        }
    }
}

impl Error for ContractError {}

/// Exact identity of one reusable human, model, or algorithmic rater configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaterConfigurationIdentity {
    /// Stable rater family reference.
    pub rater_family_ref: DomainReference,
    /// Provider or employing authority reference.
    pub provider_ref: DomainReference,
    /// Exact model, human qualification, or algorithm revision reference.
    pub implementation_revision_ref: DomainReference,
    /// Exact prompt or operating-procedure revision reference.
    pub instruction_revision_ref: DomainReference,
    /// Exact response schema revision reference.
    pub response_schema_revision_ref: DomainReference,
    /// Workflow mode used by this configuration.
    pub workflow_mode_ref: DomainReference,
    /// Modality channel observed by this configuration.
    pub modality_channel_ref: DomainReference,
}

/// Ordinal uncertainty class reported by the observation producer.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UncertaintyLevel {
    /// The producer reports low uncertainty.
    Low,
    /// The producer reports medium uncertainty.
    Medium,
    /// The producer reports high uncertainty.
    High,
}

/// Whether a criterion was observed or the rater abstained.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObservationStatus {
    /// The rater selected one category anchor and supplied evidence.
    Observed,
    /// The rater did not select a category and supplied a bounded reason.
    Abstained,
}

/// One criterion-level observation entity produced by a single rater invocation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CriterionObservation {
    criterion_ref: DomainReference,
    category_anchor_ref: Option<DomainReference>,
    evidence_reference_ids: Vec<DomainReference>,
    status: ObservationStatus,
    uncertainty: UncertaintyLevel,
    review_signal_refs: Vec<DomainReference>,
    reason_ref: Option<DomainReference>,
}

impl CriterionObservation {
    /// Create an evidence-bearing observed category.
    pub fn observed(
        criterion_ref: DomainReference,
        category_anchor_ref: DomainReference,
        evidence_reference_ids: Vec<DomainReference>,
        uncertainty: UncertaintyLevel,
        review_signal_refs: Vec<DomainReference>,
    ) -> Result<Self, ContractError> {
        if evidence_reference_ids.is_empty() {
            return Err(ContractError::ObservedWithoutEvidence);
        }
        ensure_unique(
            &evidence_reference_ids,
            ContractError::DuplicateEvidenceReference,
        )?;
        ensure_unique(&review_signal_refs, ContractError::DuplicateReviewSignal)?;
        Ok(Self {
            criterion_ref,
            category_anchor_ref: Some(category_anchor_ref),
            evidence_reference_ids,
            status: ObservationStatus::Observed,
            uncertainty,
            review_signal_refs,
            reason_ref: None,
        })
    }

    /// Create an abstention without manufacturing a category observation.
    pub fn abstained(
        criterion_ref: DomainReference,
        reason_ref: DomainReference,
        uncertainty: UncertaintyLevel,
        review_signal_refs: Vec<DomainReference>,
    ) -> Result<Self, ContractError> {
        ensure_unique(&review_signal_refs, ContractError::DuplicateReviewSignal)?;
        Ok(Self {
            criterion_ref,
            category_anchor_ref: None,
            evidence_reference_ids: Vec::new(),
            status: ObservationStatus::Abstained,
            uncertainty,
            review_signal_refs,
            reason_ref: Some(reason_ref),
        })
    }

    /// Return the criterion identity.
    #[must_use]
    pub fn criterion_ref(&self) -> &DomainReference {
        &self.criterion_ref
    }

    /// Return the selected category anchor, if one was observed.
    #[must_use]
    pub fn category_anchor_ref(&self) -> Option<&DomainReference> {
        self.category_anchor_ref.as_ref()
    }

    /// Return immutable evidence references.
    #[must_use]
    pub fn evidence_reference_ids(&self) -> &[DomainReference] {
        &self.evidence_reference_ids
    }

    /// Return the observation status.
    #[must_use]
    pub fn status(&self) -> ObservationStatus {
        self.status
    }

    /// Return the producer-reported uncertainty class.
    #[must_use]
    pub fn uncertainty(&self) -> UncertaintyLevel {
        self.uncertainty
    }

    /// Return immutable review-signal references.
    #[must_use]
    pub fn review_signal_refs(&self) -> &[DomainReference] {
        &self.review_signal_refs
    }

    /// Return the abstention reason, if the rater abstained.
    #[must_use]
    pub fn reason_ref(&self) -> Option<&DomainReference> {
        self.reason_ref.as_ref()
    }
}

/// Aggregate root for exactly one execution of one rater configuration.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaterInvocation {
    invocation_ref: DomainReference,
    configuration: RaterConfigurationIdentity,
    task_revision_ref: DomainReference,
    rubric_revision_ref: DomainReference,
    response_evidence_ref: DomainReference,
    observations: Vec<CriterionObservation>,
}

impl RaterInvocation {
    /// Create an invocation while enforcing one observation per criterion.
    pub fn new(
        invocation_ref: DomainReference,
        configuration: RaterConfigurationIdentity,
        task_revision_ref: DomainReference,
        rubric_revision_ref: DomainReference,
        response_evidence_ref: DomainReference,
        observations: Vec<CriterionObservation>,
    ) -> Result<Self, ContractError> {
        if observations.is_empty() {
            return Err(ContractError::EmptyObservationSet);
        }
        let mut criteria = HashSet::with_capacity(observations.len());
        for observation in &observations {
            if !criteria.insert(observation.criterion_ref()) {
                return Err(ContractError::DuplicateCriterionObservation);
            }
        }
        Ok(Self {
            invocation_ref,
            configuration,
            task_revision_ref,
            rubric_revision_ref,
            response_evidence_ref,
            observations,
        })
    }

    /// Return the invocation identity.
    #[must_use]
    pub fn invocation_ref(&self) -> &DomainReference {
        &self.invocation_ref
    }

    /// Return the exact reusable rater configuration identity.
    #[must_use]
    pub fn configuration(&self) -> &RaterConfigurationIdentity {
        &self.configuration
    }

    /// Return the task revision observed by this invocation.
    #[must_use]
    pub fn task_revision_ref(&self) -> &DomainReference {
        &self.task_revision_ref
    }

    /// Return the rubric revision applied by this invocation.
    #[must_use]
    pub fn rubric_revision_ref(&self) -> &DomainReference {
        &self.rubric_revision_ref
    }

    /// Return the opaque response-evidence reference.
    #[must_use]
    pub fn response_evidence_ref(&self) -> &DomainReference {
        &self.response_evidence_ref
    }

    /// Return the criterion observations in producer order.
    #[must_use]
    pub fn observations(&self) -> &[CriterionObservation] {
        &self.observations
    }
}

fn ensure_unique(
    values: &[DomainReference],
    error: ContractError,
) -> Result<(), ContractError> {
    let mut seen = HashSet::with_capacity(values.len());
    if values.iter().all(|value| seen.insert(value)) {
        Ok(())
    } else {
        Err(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn reference_validation_covers_every_rejection_branch() {
        assert_eq!(reference("  value  ").as_str(), "value");
        assert_eq!(
            DomainReference::parse("field", "   "),
            Err(ContractError::EmptyReference("field"))
        );
        assert_eq!(
            DomainReference::parse("field", "x".repeat(MAX_REFERENCE_LENGTH + 1)),
            Err(ContractError::ReferenceTooLong("field"))
        );
        assert_eq!(
            DomainReference::parse("field", "bad\nvalue"),
            Err(ContractError::ControlCharacter("field"))
        );
    }

    #[test]
    fn observed_and_abstained_states_are_structurally_distinct() {
        let observed = CriterionObservation::observed(
            reference("criterion-a"),
            reference("anchor-2"),
            vec![reference("evidence-1")],
            UncertaintyLevel::Low,
            vec![reference("review-none")],
        )
        .expect("observed criterion");
        assert_eq!(observed.status(), ObservationStatus::Observed);
        assert_eq!(
            observed
                .category_anchor_ref()
                .map(DomainReference::as_str),
            Some("anchor-2")
        );
        assert_eq!(observed.evidence_reference_ids().len(), 1);
        assert_eq!(observed.uncertainty(), UncertaintyLevel::Low);
        assert_eq!(observed.review_signal_refs().len(), 1);
        assert!(observed.reason_ref().is_none());

        let abstained = CriterionObservation::abstained(
            reference("criterion-b"),
            reference("insufficient-evidence"),
            UncertaintyLevel::High,
            Vec::new(),
        )
        .expect("abstention");
        assert_eq!(abstained.status(), ObservationStatus::Abstained);
        assert!(abstained.category_anchor_ref().is_none());
        assert!(abstained.evidence_reference_ids().is_empty());
        assert_eq!(abstained.uncertainty(), UncertaintyLevel::High);
        assert!(abstained.review_signal_refs().is_empty());
        assert_eq!(
            abstained.reason_ref().map(DomainReference::as_str),
            Some("insufficient-evidence")
        );
    }

    #[test]
    fn criterion_validation_rejects_missing_or_duplicate_boundary_data() {
        assert_eq!(
            CriterionObservation::observed(
                reference("criterion"),
                reference("anchor"),
                Vec::new(),
                UncertaintyLevel::Medium,
                Vec::new(),
            ),
            Err(ContractError::ObservedWithoutEvidence)
        );
        assert_eq!(
            CriterionObservation::observed(
                reference("criterion"),
                reference("anchor"),
                vec![reference("evidence"), reference("evidence")],
                UncertaintyLevel::Medium,
                Vec::new(),
            ),
            Err(ContractError::DuplicateEvidenceReference)
        );
        assert_eq!(
            CriterionObservation::abstained(
                reference("criterion"),
                reference("reason"),
                UncertaintyLevel::Medium,
                vec![reference("signal"), reference("signal")],
            ),
            Err(ContractError::DuplicateReviewSignal)
        );
    }

    #[test]
    fn invocation_is_the_only_consistency_boundary_for_criterion_uniqueness() {
        let observation = CriterionObservation::observed(
            reference("criterion"),
            reference("anchor"),
            vec![reference("evidence")],
            UncertaintyLevel::Medium,
            Vec::new(),
        )
        .expect("observation");
        assert_eq!(
            RaterInvocation::new(
                reference("invocation"),
                configuration(),
                reference("task-v1"),
                reference("rubric-v1"),
                reference("response-evidence"),
                Vec::new(),
            ),
            Err(ContractError::EmptyObservationSet)
        );
        assert_eq!(
            RaterInvocation::new(
                reference("invocation"),
                configuration(),
                reference("task-v1"),
                reference("rubric-v1"),
                reference("response-evidence"),
                vec![observation.clone(), observation],
            ),
            Err(ContractError::DuplicateCriterionObservation)
        );
    }

    #[test]
    fn invocation_exposes_only_observation_contract_data() {
        let observation = CriterionObservation::observed(
            reference("criterion"),
            reference("anchor"),
            vec![reference("evidence")],
            UncertaintyLevel::Medium,
            Vec::new(),
        )
        .expect("observation");
        let invocation = RaterInvocation::new(
            reference("invocation"),
            configuration(),
            reference("task-v1"),
            reference("rubric-v1"),
            reference("response-evidence"),
            vec![observation],
        )
        .expect("invocation");
        assert_eq!(invocation.invocation_ref().as_str(), "invocation");
        assert_eq!(invocation.configuration().provider_ref.as_str(), "provider");
        assert_eq!(invocation.task_revision_ref().as_str(), "task-v1");
        assert_eq!(invocation.rubric_revision_ref().as_str(), "rubric-v1");
        assert_eq!(
            invocation.response_evidence_ref().as_str(),
            "response-evidence"
        );
        assert_eq!(invocation.observations().len(), 1);
        assert_eq!(
            GOVERNED_RATER_OBSERVATION_CONTRACT_V1,
            "cwl_governed_rater_observation/v1"
        );
        assert!(
            ContractError::ObservedWithoutEvidence
                .to_string()
                .contains("requires evidence")
        );
    }
}
