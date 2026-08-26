//! Fail-closed TEPP topic-posterior input contract.
//!
//! This module validates the complete `tepp.topic_context_posterior.v1`
//! evidence boundary. It intentionally does not estimate case-deletion
//! influence: the corresponding continuous-posterior MMMC estimator has not
//! yet passed identification, recovery, and CPU/GPU parity gates.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use uuid::Uuid;

/// Exact TEPP artifact schema accepted by this consumer.
pub const TOPIC_CONTEXT_POSTERIOR_SCHEMA_VERSION: &str = "tepp.topic_context_posterior.v1";
/// Maximum accepted JSON payload size.
pub const TOPIC_CONTEXT_POSTERIOR_BYTE_LIMIT: usize = 16 * 1024 * 1024;
const ENTRY_LIMIT: usize = 1_000_000;
const DIMENSIONS: [&str; 4] = ["business_unit", "process_unit", "team", "person"];
type PosteriorDraws = BTreeMap<Uuid, BTreeSet<u64>>;
type DocumentEventTimes = BTreeMap<Uuid, OffsetDateTime>;

/// Stable fail-closed contract error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopicContextContractError {
    /// The serialized input exceeded the public resource bound.
    LimitExceeded,
    /// The input was foreign, malformed, incomplete, or internally inconsistent.
    InvalidEvidence,
    /// The producer artifact lacks case-deleted refit evidence.
    CaseDeletionRefitEvidenceUnavailable,
}

/// One global topic activity interval.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicActivityInterval {
    /// Stable producer-owned topic UUID.
    pub topic_id: String,
    /// `active`, `dormant`, or `reactivated`.
    pub state_code: String,
    /// Inclusive RFC 3339 start on the declared event clock.
    pub valid_from: String,
    /// Inclusive RFC 3339 end on the declared event clock.
    pub valid_to: String,
}

/// One producer-fitted topic lineage event.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicLineageEvent {
    /// `birth`, `split`, `merge`, or `retirement`.
    pub event_code: String,
    /// Source stable topic UUID.
    pub source_topic_id: String,
    /// Target stable topic UUID for split and merge events.
    pub target_topic_id: Option<String>,
    /// Canonical RFC 3339 event time.
    pub event_time: String,
    /// Opaque evidence resource identity.
    pub evidence_resource_id: String,
    /// Opaque provenance assertion identity.
    pub provenance_assertion_id: String,
    /// Lowercase SHA-256 evidence digest.
    pub evidence_sha256: String,
}

/// One admitted Event Lineage relation between documents.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicDocumentRelation {
    /// Source document UUID.
    pub source_document_id: String,
    /// Target document UUID.
    pub target_document_id: String,
    /// Exact producer relation code.
    pub relation_kind_code: String,
    /// Canonical RFC 3339 relation time.
    pub event_time: String,
    /// Opaque evidence resource identity.
    pub evidence_resource_id: String,
    /// Opaque provenance assertion identity.
    pub provenance_assertion_id: String,
    /// Lowercase SHA-256 evidence digest.
    pub evidence_sha256: String,
}

/// One document posterior plausible value for one draw.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicPostPlausibleValue {
    /// Document UUID.
    pub document_id: String,
    /// Zero-based posterior draw index.
    pub draw_index: u64,
    /// Canonical RFC 3339 document event time.
    pub event_time: String,
    /// Full-rank logistic-normal coordinates.
    pub logistic_normal_coordinates: Vec<f64>,
}

/// One time-valid, provenance-bound organizational membership.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicContextMembership {
    /// Document UUID.
    pub document_id: String,
    /// BU, PU, team, or person dimension code.
    pub dimension_code: String,
    /// Opaque context identity inside the dimension.
    pub context_id: String,
    /// Source-derived positive multiple-membership weight.
    pub weight: f64,
    /// Inclusive RFC 3339 validity start.
    pub valid_from: String,
    /// Inclusive RFC 3339 validity end.
    pub valid_to: String,
    /// Opaque evidence resource identity.
    pub evidence_resource_id: String,
    /// Opaque provenance assertion identity.
    pub provenance_assertion_id: String,
    /// Lowercase SHA-256 evidence digest.
    pub evidence_sha256: String,
}

/// Complete TEPP posterior artifact admitted for a future influence fit.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TopicContextPosteriorArtifact {
    /// Exact wire schema identity.
    pub schema_version: String,
    /// Opaque TEPP run identity.
    pub run_id: String,
    /// Immutable source snapshot identity.
    pub snapshot_id: String,
    /// Source snapshot SHA-256.
    pub source_snapshot_sha256: String,
    /// Historical knowledge cutoff.
    pub knowledge_cutoff: String,
    /// Exact event-clock declaration.
    pub event_clock_code: String,
    /// TEPP model contract identity.
    pub model_contract_version: String,
    /// Opaque posterior draw-set identity.
    pub posterior_draw_set_id: String,
    /// Draw count present for every document.
    pub posterior_draw_count: u64,
    /// Number of stable global topics.
    pub topic_count: u64,
    /// Stable topic UUIDs in coordinate order.
    pub topic_ids: Vec<String>,
    /// Topic activity intervals.
    pub activity_intervals: Vec<TopicActivityInterval>,
    /// Topic birth, split, merge, and retirement events.
    pub lineage_events: Vec<TopicLineageEvent>,
    /// Complete admitted Event Lineage relations.
    pub document_relations: Vec<TopicDocumentRelation>,
    /// Complete document-by-draw posterior coordinates.
    pub plausible_values: Vec<TopicPostPlausibleValue>,
    /// Time-valid BU/PU/team/person multiple memberships.
    pub memberships: Vec<TopicContextMembership>,
    /// Fixed producer interpretation boundary.
    pub inference_status: String,
}

/// Validated input that cannot be constructed without passing the contract.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedTopicContextPosterior(TopicContextPosteriorArtifact);

impl ValidatedTopicContextPosterior {
    /// Borrow the validated producer artifact.
    #[must_use]
    pub fn artifact(&self) -> &TopicContextPosteriorArtifact {
        &self.0
    }
}

fn identifier(value: &str) -> bool {
    !value.is_empty() && value.len() <= 256 && value.trim() == value
}

fn digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn instant(value: &str) -> Option<OffsetDateTime> {
    OffsetDateTime::parse(value, &Rfc3339)
        .ok()
        .filter(|parsed| {
            parsed
                .format(&Rfc3339)
                .is_ok_and(|formatted| formatted == value)
        })
}

fn evidence(resource_id: &str, assertion_id: &str, sha256: &str) -> bool {
    identifier(resource_id) && identifier(assertion_id) && digest(sha256)
}

fn finite_coordinates(values: &[f64], expected: usize) -> bool {
    values.len() == expected && values.iter().all(|value| value.is_finite())
}

impl TopicContextPosteriorArtifact {
    /// Parse and fully validate one bounded TEPP artifact.
    ///
    /// # Errors
    ///
    /// Returns a stable fail-closed error for oversized, malformed, foreign,
    /// incomplete, or internally inconsistent evidence.
    pub fn from_json(
        payload: &str,
    ) -> Result<ValidatedTopicContextPosterior, TopicContextContractError> {
        if payload.len() > TOPIC_CONTEXT_POSTERIOR_BYTE_LIMIT {
            return Err(TopicContextContractError::LimitExceeded);
        }
        let artifact: Self = serde_json::from_str(payload)
            .map_err(|_| TopicContextContractError::InvalidEvidence)?;
        artifact.validate()?;
        Ok(ValidatedTopicContextPosterior(artifact))
    }

    fn validate(&self) -> Result<(), TopicContextContractError> {
        self.validate_with_entry_limit(ENTRY_LIMIT)
    }

    fn validate_with_entry_limit(
        &self,
        entry_limit: usize,
    ) -> Result<(), TopicContextContractError> {
        let cutoff =
            instant(&self.knowledge_cutoff).ok_or(TopicContextContractError::InvalidEvidence)?;
        if self.schema_version != TOPIC_CONTEXT_POSTERIOR_SCHEMA_VERSION
            || !identifier(&self.run_id)
            || !identifier(&self.snapshot_id)
            || !digest(&self.source_snapshot_sha256)
            || self.event_clock_code != "event_time_rfc3339"
            || !identifier(&self.model_contract_version)
            || !identifier(&self.posterior_draw_set_id)
            || self.posterior_draw_count == 0
            || self.topic_count < 2
            || usize::try_from(self.topic_count) != Ok(self.topic_ids.len())
            || self.inference_status != "posterior_topic_coordinates_not_importance"
            || [
                self.activity_intervals.len(),
                self.lineage_events.len(),
                self.document_relations.len(),
                self.plausible_values.len(),
                self.memberships.len(),
            ]
            .into_iter()
            .any(|length| length > entry_limit)
        {
            return Err(TopicContextContractError::InvalidEvidence);
        }
        let topics: BTreeSet<Uuid> = self
            .topic_ids
            .iter()
            .map(|value| {
                Uuid::parse_str(value).map_err(|_| TopicContextContractError::InvalidEvidence)
            })
            .collect::<Result<_, _>>()?;
        if topics.len() != self.topic_ids.len() {
            return Err(TopicContextContractError::InvalidEvidence);
        }
        self.validate_topic_records(cutoff, &topics)?;
        let (draws, event_times) = self.validate_draws(cutoff)?;
        self.validate_relations(cutoff, &draws, &event_times)?;
        self.validate_memberships(&draws, &event_times)?;
        self.validate_provenance_bindings()
    }

    fn validate_provenance_bindings(&self) -> Result<(), TopicContextContractError> {
        let mut bindings = BTreeMap::new();
        let mut bind = |id: &str, evidence: String| {
            if bindings
                .insert(id.to_owned(), evidence.clone())
                .is_some_and(|prior| prior != evidence)
            {
                Err(TopicContextContractError::InvalidEvidence)
            } else {
                Ok(())
            }
        };
        for value in &self.lineage_events {
            bind(
                &value.provenance_assertion_id,
                format!(
                    "topic:{}:{}:{}:{}:{}",
                    value.event_code,
                    value.source_topic_id,
                    value.target_topic_id.as_deref().unwrap_or(""),
                    value.evidence_resource_id,
                    value.evidence_sha256
                ),
            )?;
        }
        for value in &self.document_relations {
            bind(
                &value.provenance_assertion_id,
                format!(
                    "document:{}:{}:{}:{}:{}:{}",
                    value.relation_kind_code,
                    value.source_document_id,
                    value.target_document_id,
                    value.event_time,
                    value.evidence_resource_id,
                    value.evidence_sha256
                ),
            )?;
        }
        for value in &self.memberships {
            bind(
                &value.provenance_assertion_id,
                format!(
                    "membership:{}:{}:{}:{}:{}:{}:{}:{}",
                    value.dimension_code,
                    value.document_id,
                    value.context_id,
                    value.weight,
                    value.valid_from,
                    value.valid_to,
                    value.evidence_resource_id,
                    value.evidence_sha256
                ),
            )?;
        }
        Ok(())
    }

    fn validate_topic_records(
        &self,
        cutoff: OffsetDateTime,
        topics: &BTreeSet<Uuid>,
    ) -> Result<(), TopicContextContractError> {
        let mut intervals: BTreeMap<Uuid, Vec<(OffsetDateTime, OffsetDateTime, &str)>> =
            BTreeMap::new();
        let mut seen = BTreeSet::new();
        for value in &self.activity_intervals {
            let topic = Uuid::parse_str(&value.topic_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let start =
                instant(&value.valid_from).ok_or(TopicContextContractError::InvalidEvidence)?;
            let end = instant(&value.valid_to).ok_or(TopicContextContractError::InvalidEvidence)?;
            if !topics.contains(&topic)
                || !["active", "dormant", "reactivated"].contains(&value.state_code.as_str())
                || start > end
                || end > cutoff
                || !seen.insert((topic, start, end, value.state_code.as_str()))
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
            intervals
                .entry(topic)
                .or_default()
                .push((start, end, &value.state_code));
        }
        if intervals.len() != topics.len() {
            return Err(TopicContextContractError::InvalidEvidence);
        }
        for values in intervals.values_mut() {
            values.sort();
            if values.first().map(|value| value.2) != Some("active")
                || values.windows(2).any(|pair| {
                    pair[1].0 <= pair[0].1
                        || !matches!(
                            (pair[0].2, pair[1].2),
                            ("active" | "reactivated", "dormant") | ("dormant", "reactivated")
                        )
                })
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
        }
        let mut lineage = BTreeSet::new();
        for value in &self.lineage_events {
            let source = Uuid::parse_str(&value.source_topic_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let target = value
                .target_topic_id
                .as_deref()
                .map(Uuid::parse_str)
                .transpose()
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let at =
                instant(&value.event_time).ok_or(TopicContextContractError::InvalidEvidence)?;
            let shape_ok = match value.event_code.as_str() {
                "birth" | "retirement" => target.is_none(),
                "split" | "merge" => target.is_some_and(|id| id != source),
                _ => false,
            };
            if !topics.contains(&source)
                || target.is_some_and(|id| !topics.contains(&id))
                || !shape_ok
                || at > cutoff
                || !evidence(
                    &value.evidence_resource_id,
                    &value.provenance_assertion_id,
                    &value.evidence_sha256,
                )
                || !lineage.insert((
                    value.event_code.as_str(),
                    source,
                    target,
                    at,
                    value.provenance_assertion_id.as_str(),
                ))
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
        }
        Ok(())
    }

    fn validate_draws(
        &self,
        cutoff: OffsetDateTime,
    ) -> Result<(PosteriorDraws, DocumentEventTimes), TopicContextContractError> {
        let coordinate_count = self.topic_ids.len() - 1;
        let mut draws: PosteriorDraws = BTreeMap::new();
        let mut event_times = BTreeMap::new();
        for value in &self.plausible_values {
            let document = Uuid::parse_str(&value.document_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let at =
                instant(&value.event_time).ok_or(TopicContextContractError::InvalidEvidence)?;
            if value.draw_index >= self.posterior_draw_count
                || at > cutoff
                || !finite_coordinates(&value.logistic_normal_coordinates, coordinate_count)
                || !draws.entry(document).or_default().insert(value.draw_index)
                || event_times
                    .insert(document, at)
                    .is_some_and(|prior| prior != at)
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
        }
        if draws.len() < 2
            || draws
                .values()
                .any(|indices| self.posterior_draw_count != indices.len() as u64)
        {
            return Err(TopicContextContractError::InvalidEvidence);
        }
        Ok((draws, event_times))
    }

    fn validate_relations(
        &self,
        cutoff: OffsetDateTime,
        draws: &PosteriorDraws,
        event_times: &DocumentEventTimes,
    ) -> Result<(), TopicContextContractError> {
        let mut seen = BTreeSet::new();
        for value in &self.document_relations {
            let source = Uuid::parse_str(&value.source_document_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let target = Uuid::parse_str(&value.target_document_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let at =
                instant(&value.event_time).ok_or(TopicContextContractError::InvalidEvidence)?;
            if source == target
                || !draws.contains_key(&source)
                || !draws.contains_key(&target)
                || value.relation_kind_code != "event_lineage_precedes"
                || event_times.get(&source) > event_times.get(&target)
                || at > cutoff
                || !evidence(
                    &value.evidence_resource_id,
                    &value.provenance_assertion_id,
                    &value.evidence_sha256,
                )
                || !seen.insert((source, target, at, value.provenance_assertion_id.as_str()))
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
        }
        Ok(())
    }

    fn validate_memberships(
        &self,
        draws: &PosteriorDraws,
        event_times: &DocumentEventTimes,
    ) -> Result<(), TopicContextContractError> {
        let mut present: BTreeMap<Uuid, BTreeSet<&str>> = BTreeMap::new();
        let mut seen = BTreeSet::new();
        for value in &self.memberships {
            let document = Uuid::parse_str(&value.document_id)
                .map_err(|_| TopicContextContractError::InvalidEvidence)?;
            let start =
                instant(&value.valid_from).ok_or(TopicContextContractError::InvalidEvidence)?;
            let end = instant(&value.valid_to).ok_or(TopicContextContractError::InvalidEvidence)?;
            let document_time = event_times
                .get(&document)
                .ok_or(TopicContextContractError::InvalidEvidence)?;
            if !DIMENSIONS.contains(&value.dimension_code.as_str())
                || !identifier(&value.context_id)
                || !value.weight.is_finite()
                || value.weight <= 0.0
                || start > end
                || document_time < &start
                || document_time > &end
                || !evidence(
                    &value.evidence_resource_id,
                    &value.provenance_assertion_id,
                    &value.evidence_sha256,
                )
                || !seen.insert((
                    document,
                    value.dimension_code.as_str(),
                    value.context_id.as_str(),
                    start,
                    end,
                ))
            {
                return Err(TopicContextContractError::InvalidEvidence);
            }
            present
                .entry(document)
                .or_default()
                .insert(&value.dimension_code);
        }
        if present.len() != draws.len()
            || present.values().any(|dimensions| {
                DIMENSIONS
                    .iter()
                    .any(|required| !dimensions.contains(required))
            })
        {
            return Err(TopicContextContractError::InvalidEvidence);
        }
        Ok(())
    }
}

/// Refuse to emit model case-deletion influence from full-data draws alone.
///
/// A draw from `p(theta | D)` does not identify `p(theta | D \\ {i})`.
/// Exact refitting requires a producer-owned case-deleted posterior. An
/// importance-sampling alternative requires the deleted case's likelihood
/// contribution evaluated at every retained joint draw, which v1 does not
/// carry. The function therefore rejects even a completely valid v1 artifact
/// instead of relabeling fixed-posterior weighted-mean leverage as Bayesian
/// case-deletion influence (Bradlow & Zaslavsky, 1997; Jackson et al., 2009).
///
/// # Errors
///
/// Always returns `CaseDeletionRefitEvidenceUnavailable` because this v1
/// artifact cannot identify the deleted-data posterior.
pub fn estimate_topic_context_case_deletion_influence(
    _posterior: &ValidatedTopicContextPosterior,
) -> Result<(), TopicContextContractError> {
    Err(TopicContextContractError::CaseDeletionRefitEvidenceUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> String {
        let membership = |document: &str, dimension: &str| {
            format!(
                r#"{{"document_id":"{document}","dimension_code":"{dimension}","context_id":"context-1","weight":1.0,"valid_from":"2026-07-01T00:00:00Z","valid_to":"2026-08-01T00:00:00Z","evidence_resource_id":"evidence-1","provenance_assertion_id":"{document}-{dimension}","evidence_sha256":"{}"}}"#,
                "a".repeat(64)
            )
        };
        let documents = [
            "018f3f7a-7b7c-7d00-8000-000000000001",
            "018f3f7a-7b7c-7d00-8000-000000000002",
        ];
        let memberships = documents
            .iter()
            .flat_map(|document| {
                DIMENSIONS
                    .iter()
                    .map(move |dimension| membership(document, dimension))
            })
            .collect::<Vec<_>>()
            .join(",");
        format!(
            r#"{{"schema_version":"tepp.topic_context_posterior.v1","run_id":"run-1","snapshot_id":"snapshot-1","source_snapshot_sha256":"{}","knowledge_cutoff":"2026-08-01T00:00:00Z","event_clock_code":"event_time_rfc3339","model_contract_version":"trsl-tm-v1","posterior_draw_set_id":"draw-set-1","posterior_draw_count":2,"topic_count":2,"topic_ids":["018f3f7a-7b7c-7d00-8000-000000000101","018f3f7a-7b7c-7d00-8000-000000000102"],"activity_intervals":[{{"topic_id":"018f3f7a-7b7c-7d00-8000-000000000101","state_code":"active","valid_from":"2026-07-01T00:00:00Z","valid_to":"2026-07-31T00:00:00Z"}},{{"topic_id":"018f3f7a-7b7c-7d00-8000-000000000102","state_code":"active","valid_from":"2026-07-01T00:00:00Z","valid_to":"2026-07-31T00:00:00Z"}}],"lineage_events":[],"document_relations":[{{"source_document_id":"{}","target_document_id":"{}","relation_kind_code":"event_lineage_precedes","event_time":"2026-07-15T00:00:00Z","evidence_resource_id":"relation-evidence","provenance_assertion_id":"relation-assertion","evidence_sha256":"{}"}}],"plausible_values":[{{"document_id":"{}","draw_index":0,"event_time":"2026-07-15T00:00:00Z","logistic_normal_coordinates":[0.1]}},{{"document_id":"{}","draw_index":1,"event_time":"2026-07-15T00:00:00Z","logistic_normal_coordinates":[0.2]}},{{"document_id":"{}","draw_index":0,"event_time":"2026-07-16T00:00:00Z","logistic_normal_coordinates":[0.3]}},{{"document_id":"{}","draw_index":1,"event_time":"2026-07-16T00:00:00Z","logistic_normal_coordinates":[0.4]}}],"memberships":[{}],"inference_status":"posterior_topic_coordinates_not_importance"}}"#,
            "0".repeat(64),
            documents[0],
            documents[1],
            "b".repeat(64),
            documents[0],
            documents[0],
            documents[1],
            documents[1],
            memberships
        )
    }

    fn artifact() -> TopicContextPosteriorArtifact {
        serde_json::from_str(&valid_json()).unwrap()
    }

    fn lineage_event(code: &str, target: Option<&str>) -> TopicLineageEvent {
        TopicLineageEvent {
            event_code: code.into(),
            source_topic_id: "018f3f7a-7b7c-7d00-8000-000000000101".into(),
            target_topic_id: target.map(str::to_owned),
            event_time: "2026-07-10T00:00:00Z".into(),
            evidence_resource_id: format!("{code}-evidence"),
            provenance_assertion_id: format!("{code}-assertion"),
            evidence_sha256: "c".repeat(64),
        }
    }

    #[test]
    fn complete_artifact_is_admitted_but_estimator_stays_unavailable() {
        let posterior = TopicContextPosteriorArtifact::from_json(&valid_json()).unwrap();
        assert_eq!(posterior.artifact().posterior_draw_count, 2);
        assert_eq!(
            estimate_topic_context_case_deletion_influence(&posterior),
            Err(TopicContextContractError::CaseDeletionRefitEvidenceUnavailable)
        );
    }

    #[test]
    fn malformed_foreign_and_incomplete_artifacts_fail_closed() {
        assert_eq!(
            TopicContextPosteriorArtifact::from_json("{}"),
            Err(TopicContextContractError::InvalidEvidence)
        );
        let foreign = valid_json().replace(
            "tepp.topic_context_posterior.v1",
            "tepp.topic_context_posterior.v2",
        );
        assert_eq!(
            TopicContextPosteriorArtifact::from_json(&foreign),
            Err(TopicContextContractError::InvalidEvidence)
        );
        let missing_dimension = valid_json().replace(",{\"document_id\":\"018f3f7a-7b7c-7d00-8000-000000000002\",\"dimension_code\":\"person\"", ",{\"document_id\":\"018f3f7a-7b7c-7d00-8000-000000000099\",\"dimension_code\":\"person\"");
        assert_eq!(
            TopicContextPosteriorArtifact::from_json(&missing_dimension),
            Err(TopicContextContractError::InvalidEvidence)
        );
    }

    #[test]
    fn resource_bound_and_nonfinite_values_fail_closed() {
        let oversized = " ".repeat(TOPIC_CONTEXT_POSTERIOR_BYTE_LIMIT + 1);
        assert_eq!(
            TopicContextPosteriorArtifact::from_json(&oversized),
            Err(TopicContextContractError::LimitExceeded)
        );
        let nonfinite = valid_json().replace("[0.1]", "[1e999]");
        assert_eq!(
            TopicContextPosteriorArtifact::from_json(&nonfinite),
            Err(TopicContextContractError::InvalidEvidence)
        );
    }

    #[test]
    fn every_header_and_topic_boundary_fails_closed() {
        macro_rules! invalid {
            ($change:expr) => {{
                let mut candidate = artifact();
                $change(&mut candidate);
                assert_eq!(
                    candidate.validate(),
                    Err(TopicContextContractError::InvalidEvidence)
                );
            }};
        }
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.knowledge_cutoff = "not-time".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.run_id = "".into());
        assert!(!identifier(&"x".repeat(257)));
        assert!(!evidence("resource-1", "", &"a".repeat(64)));
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.snapshot_id = " x ".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.source_snapshot_sha256 = "g".repeat(64));
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.event_clock_code = "recorded_time".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.model_contract_version = "".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.posterior_draw_set_id = "".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.posterior_draw_count = 0);
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.topic_count = 1);
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.topic_count = 3);
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.inference_status = "importance".into());
        assert_eq!(
            artifact().validate_with_entry_limit(0),
            Err(TopicContextContractError::InvalidEvidence)
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.topic_ids[0] = "not-uuid".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.topic_ids[1] = a.topic_ids[0].clone());
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].topic_id =
                "not-uuid".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].topic_id =
                "018f3f7a-7b7c-7d00-8000-000000000999".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].state_code =
                "unknown".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].valid_from =
                "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].valid_to = "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].valid_from =
                "2026-08-02T00:00:00Z".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].valid_to =
                "2026-08-02T00:00:00Z".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a
            .activity_intervals
            .push(a.activity_intervals[0].clone()));
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.activity_intervals.pop();
        });

        let mut transitions = artifact();
        transitions.activity_intervals[0].valid_to = "2026-07-10T00:00:00Z".into();
        transitions.activity_intervals.push(TopicActivityInterval {
            topic_id: transitions.topic_ids[0].clone(),
            state_code: "dormant".into(),
            valid_from: "2026-07-11T00:00:00Z".into(),
            valid_to: "2026-07-20T00:00:00Z".into(),
        });
        transitions.activity_intervals.push(TopicActivityInterval {
            topic_id: transitions.topic_ids[0].clone(),
            state_code: "reactivated".into(),
            valid_from: "2026-07-21T00:00:00Z".into(),
            valid_to: "2026-07-31T00:00:00Z".into(),
        });
        assert!(transitions.validate().is_ok());
        transitions.activity_intervals[3].valid_from = "2026-07-20T00:00:00Z".into();
        assert_eq!(
            transitions.validate(),
            Err(TopicContextContractError::InvalidEvidence)
        );
        transitions.activity_intervals[3].valid_from = "2026-07-21T00:00:00Z".into();
        transitions.activity_intervals[3].state_code = "dormant".into();
        assert_eq!(
            transitions.validate(),
            Err(TopicContextContractError::InvalidEvidence)
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.activity_intervals[0].state_code =
                "dormant".into()
        );
    }

    #[test]
    fn lineage_draw_relation_and_membership_boundaries_fail_closed() {
        macro_rules! invalid {
            ($change:expr) => {{
                let mut candidate = artifact();
                $change(&mut candidate);
                assert_eq!(
                    candidate.validate(),
                    Err(TopicContextContractError::InvalidEvidence)
                );
            }};
        }
        let target = "018f3f7a-7b7c-7d00-8000-000000000102";
        let mut all_events = artifact();
        all_events.lineage_events = vec![
            lineage_event("birth", None),
            lineage_event("retirement", None),
            lineage_event("split", Some(target)),
            lineage_event("merge", Some(target)),
        ];
        assert!(all_events.validate().is_ok());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.lineage_events =
            vec![lineage_event("bad", None)]);
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut e = lineage_event("birth", None);
            e.source_topic_id = "bad".into();
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let e = lineage_event("split", Some("bad"));
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut e = lineage_event("birth", None);
            e.source_topic_id = "018f3f7a-7b7c-7d00-8000-000000000999".into();
            a.lineage_events = vec![e];
        });
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.lineage_events = vec![lineage_event(
                "split",
                Some("018f3f7a-7b7c-7d00-8000-000000000999")
            )]
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.lineage_events =
            vec![lineage_event("split", None)]);
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let e = lineage_event("split", Some("018f3f7a-7b7c-7d00-8000-000000000101"));
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut e = lineage_event("birth", None);
            e.event_time = "bad".into();
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut e = lineage_event("birth", None);
            e.event_time = "2026-08-02T00:00:00Z".into();
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut e = lineage_event("birth", None);
            e.evidence_sha256 = "bad".into();
            a.lineage_events = vec![e];
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let e = lineage_event("birth", None);
            a.lineage_events = vec![e.clone(), e];
        });

        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.plausible_values[0].document_id =
                "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.plausible_values[0].event_time = "bad".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.plausible_values[0].draw_index = 2);
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.plausible_values[0].event_time =
                "2026-08-02T00:00:00Z".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.plausible_values[0]
                .logistic_normal_coordinates
                .push(0.2)
        );
        assert!(finite_coordinates(&[], 0));
        invalid!(|a: &mut TopicContextPosteriorArtifact| a
            .plausible_values
            .push(a.plausible_values[0].clone()));
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.plausible_values[1].event_time =
                "2026-07-16T00:00:00Z".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.plausible_values.truncate(2));
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.plausible_values.pop();
        });

        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].source_document_id =
                "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].target_document_id =
                "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].event_time =
                "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].target_document_id =
                a.document_relations[0].source_document_id.clone()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].source_document_id =
                "018f3f7a-7b7c-7d00-8000-000000000099".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].target_document_id =
                "018f3f7a-7b7c-7d00-8000-000000000099".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].relation_kind_code =
                "related".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            for value in &mut a.plausible_values {
                if value.document_id == "018f3f7a-7b7c-7d00-8000-000000000001" {
                    value.event_time = "2026-07-17T00:00:00Z".into();
                }
            }
        });
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].event_time =
                "2026-08-02T00:00:00Z".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.document_relations[0].evidence_resource_id =
                "".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a
            .document_relations
            .push(a.document_relations[0].clone()));

        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].document_id = "bad".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].document_id =
                "018f3f7a-7b7c-7d00-8000-000000000099".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].valid_from = "bad".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.memberships[0].valid_to = "bad".into());
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].dimension_code =
                "division".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.memberships[0].context_id = "".into());
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.memberships[0].weight = f64::NAN);
        invalid!(|a: &mut TopicContextPosteriorArtifact| a.memberships[0].weight = 0.0);
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].valid_from =
                "2026-07-20T00:00:00Z".into()
        );
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].valid_to =
                "2026-07-10T00:00:00Z".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.memberships[0].valid_from = "2026-07-20T00:00:00Z".into();
            a.memberships[0].valid_to = "2026-07-10T00:00:00Z".into();
        });
        invalid!(
            |a: &mut TopicContextPosteriorArtifact| a.memberships[0].evidence_sha256 = "bad".into()
        );
        invalid!(|a: &mut TopicContextPosteriorArtifact| a
            .memberships
            .push(a.memberships[0].clone()));
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.memberships.pop();
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.memberships.retain(|membership| {
                membership.document_id != "018f3f7a-7b7c-7d00-8000-000000000002"
            });
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            a.memberships[0].provenance_assertion_id =
                a.document_relations[0].provenance_assertion_id.clone();
            a.memberships[0].evidence_sha256 = "d".repeat(64);
        });
        invalid!(|a: &mut TopicContextPosteriorArtifact| {
            let mut conflicting = a.memberships[0].clone();
            conflicting.context_id = "context-2".into();
            conflicting.evidence_resource_id = "different-resource".into();
            a.memberships.push(conflicting);
        });

        let mut multiple = artifact();
        multiple.memberships[0].weight = 0.5;
        let mut second = multiple.memberships[0].clone();
        second.context_id = "context-2".into();
        second.provenance_assertion_id = "second-membership".into();
        multiple.memberships.push(second);
        assert!(multiple.validate().is_ok());
        let mut unnormalized = artifact();
        unnormalized.memberships[0].weight = 0.5;
        assert!(unnormalized.validate().is_ok());
    }
}
