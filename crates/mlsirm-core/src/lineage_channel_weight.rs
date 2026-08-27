//! Fail-closed lineage channel-weight evidence boundary.
//!
//! A versioned criterion anchor reports whether a proposed estimation run
//! passed an independent criterion-validity assessment. The core deliberately
//! does not assume which upstream product produced that anchor, and the anchor
//! does not substitute for the pair-level criterion observations needed to
//! estimate weights. Consequently this module validates continuous channel
//! evidence and exact accepted-anchor identity but does not fit or normalize
//! weights. Treating channel covariance, a score floor, or the anchor's
//! accepted flag as response data would recreate an unanchored latent factor
//! and is prohibited.

use std::collections::BTreeSet;

use serde::Deserialize;
use time::{format_description::well_known::Rfc3339, OffsetDateTime};
use uuid::Uuid;

/// Exact request schema admitted by this prerequisite.
pub const LINEAGE_CHANNEL_WEIGHT_EVIDENCE_SCHEMA: &str =
    "fast-mlsirm.lineage_channel_weight_evidence.v1";
/// Maximum serialized evidence size.
pub const LINEAGE_CHANNEL_WEIGHT_EVIDENCE_BYTE_LIMIT: usize = 16 * 1024 * 1024;

/// Stable fail-closed outcomes for the channel-weight boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineageChannelWeightError {
    /// Caller-controlled input exceeded the public resource bound.
    LimitExceeded,
    /// Evidence was malformed, incomplete, foreign, or mixed-provenance.
    InvalidEvidence,
    /// The accepted anchor has no pair-level independent criterion observations.
    IndependentCriterionObservationsUnavailable,
}

/// Product-neutral v1 criterion-validity projection.
#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LineageCriterionAnchorV1 {
    /// Contract version; exactly one.
    pub contract_version: u16,
    /// Exact artifact kind.
    pub anchor_kind_code: String,
    /// Candidate fast-mlsirm estimation-run identity.
    pub estimation_run_id: String,
    /// Immutable source snapshot digest.
    pub source_snapshot_sha256: String,
    /// Exact historical knowledge cutoff.
    pub knowledge_cutoff: String,
    /// Upstream criterion-validity decision.
    pub criterion_validity_status: String,
    /// Number of independently validated pairs.
    pub validated_pair_count: u64,
}

/// One continuous, pre-fusion channel observation for one candidate pair.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LineagePairChannelEvidence {
    /// Opaque pair identity.
    pub pair_id: String,
    /// Opaque reconstruction-group identity.
    pub group_id: String,
    /// Scores in the request's declared channel order.
    pub channel_scores: Vec<f64>,
}

/// Full evidence admitted before an anchored estimator can be released.
#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LineageChannelWeightEvidence {
    /// Exact fast-mlsirm request schema.
    pub schema_version: String,
    /// Candidate estimation-run identity.
    pub estimation_run_id: String,
    /// Immutable source snapshot digest.
    pub source_snapshot_sha256: String,
    /// Exact historical knowledge cutoff.
    pub knowledge_cutoff: String,
    /// Stable active-channel order.
    pub channel_codes: Vec<String>,
    /// Complete continuous pair-by-channel matrix.
    pub pair_evidence: Vec<LineagePairChannelEvidence>,
    /// Product-neutral accepted criterion anchor.
    ///
    /// `tepp_anchor` is accepted only as a serialized compatibility alias for
    /// payloads produced before the core contract was restored to its
    /// domain-neutral boundary.
    #[serde(alias = "tepp_anchor")]
    pub criterion_anchor: LineageCriterionAnchorV1,
}

/// Validated evidence that still cannot identify estimation responses.
#[derive(Clone, Debug, PartialEq)]
pub struct ValidatedLineageChannelWeightEvidence(LineageChannelWeightEvidence);

impl ValidatedLineageChannelWeightEvidence {
    /// Borrow the admitted source evidence.
    pub fn evidence(&self) -> &LineageChannelWeightEvidence {
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

fn canonical_uuid(value: &str) -> bool {
    Uuid::parse_str(value)
        .ok()
        .is_some_and(|parsed| parsed.hyphenated().to_string() == value)
}

fn canonical_time(value: &str) -> bool {
    OffsetDateTime::parse(value, &Rfc3339)
        .ok()
        .and_then(|parsed| parsed.format(&Rfc3339).ok())
        .is_some_and(|formatted| formatted == value)
}

impl LineageChannelWeightEvidence {
    /// Parse and validate one bounded evidence envelope.
    pub fn from_json(
        payload: &str,
    ) -> Result<ValidatedLineageChannelWeightEvidence, LineageChannelWeightError> {
        if payload.len() > LINEAGE_CHANNEL_WEIGHT_EVIDENCE_BYTE_LIMIT {
            return Err(LineageChannelWeightError::LimitExceeded);
        }
        let evidence: Self = serde_json::from_str(payload)
            .map_err(|_| LineageChannelWeightError::InvalidEvidence)?;
        evidence.validate()?;
        Ok(ValidatedLineageChannelWeightEvidence(evidence))
    }

    fn validate(&self) -> Result<(), LineageChannelWeightError> {
        let anchor = &self.criterion_anchor;
        let channels: BTreeSet<&str> = self.channel_codes.iter().map(String::as_str).collect();
        let pair_ids: BTreeSet<&str> = self
            .pair_evidence
            .iter()
            .map(|pair| pair.pair_id.as_str())
            .collect();
        let header_valid = self.schema_version == LINEAGE_CHANNEL_WEIGHT_EVIDENCE_SCHEMA
            && canonical_uuid(&self.estimation_run_id)
            && digest(&self.source_snapshot_sha256)
            && canonical_time(&self.knowledge_cutoff)
            && !self.channel_codes.is_empty()
            && channels.len() == self.channel_codes.len()
            && self.channel_codes.iter().all(|code| identifier(code))
            && !self.pair_evidence.is_empty()
            && pair_ids.len() == self.pair_evidence.len();
        let pairs_valid = self.pair_evidence.iter().all(|pair| {
            canonical_uuid(&pair.pair_id)
                && identifier(&pair.group_id)
                && pair.channel_scores.len() == self.channel_codes.len()
                && pair
                    .channel_scores
                    .iter()
                    .all(|score| score.is_finite() && (0.0..=1.0).contains(score))
        });
        let anchor_valid = anchor.contract_version == 1
            && anchor.anchor_kind_code == "lineage_pair_criterion"
            && anchor.estimation_run_id == self.estimation_run_id
            && anchor.source_snapshot_sha256 == self.source_snapshot_sha256
            && anchor.knowledge_cutoff == self.knowledge_cutoff
            && anchor.criterion_validity_status == "accepted"
            && anchor.validated_pair_count > 0
            && usize::try_from(anchor.validated_pair_count) == Ok(self.pair_evidence.len());
        if header_valid && pairs_valid && anchor_valid {
            Ok(())
        } else {
            Err(LineageChannelWeightError::InvalidEvidence)
        }
    }
}

/// Refuse to infer channel weights without independent pair-level outcomes.
pub fn estimate_lineage_channel_weights(
    _evidence: &ValidatedLineageChannelWeightEvidence,
) -> Result<(), LineageChannelWeightError> {
    Err(LineageChannelWeightError::IndependentCriterionObservationsUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_json() -> String {
        format!(
            r#"{{"schema_version":"{LINEAGE_CHANNEL_WEIGHT_EVIDENCE_SCHEMA}","estimation_run_id":"018f47e7-7b5b-7cc0-98c6-15fdf9e3d9b1","source_snapshot_sha256":"{}","knowledge_cutoff":"2026-08-25T00:00:00Z","channel_codes":["temporal","text"],"pair_evidence":[{{"pair_id":"018f47e7-7b5b-7cc0-98c6-015fdf9e3d91","group_id":"group-a","channel_scores":[0.2,0.8]}},{{"pair_id":"018f47e7-7b5b-7cc0-98c6-015fdf9e3d92","group_id":"group-b","channel_scores":[0.7,0.3]}}],"criterion_anchor":{{"contract_version":1,"anchor_kind_code":"lineage_pair_criterion","estimation_run_id":"018f47e7-7b5b-7cc0-98c6-15fdf9e3d9b1","source_snapshot_sha256":"{}","knowledge_cutoff":"2026-08-25T00:00:00Z","criterion_validity_status":"accepted","validated_pair_count":2}}}}"#,
            "a".repeat(64),
            "a".repeat(64)
        )
    }

    fn evidence() -> LineageChannelWeightEvidence {
        serde_json::from_str(&valid_json()).expect("fixture parses")
    }

    #[test]
    fn accepted_projection_is_admitted_but_estimation_stays_unavailable() {
        let admitted = LineageChannelWeightEvidence::from_json(&valid_json()).expect("admitted");
        assert_eq!(admitted.evidence().pair_evidence.len(), 2);
        assert_eq!(
            estimate_lineage_channel_weights(&admitted),
            Err(LineageChannelWeightError::IndependentCriterionObservationsUnavailable)
        );
    }

    #[test]
    fn malformed_and_oversized_payloads_fail_closed() {
        assert_eq!(
            LineageChannelWeightEvidence::from_json("{}"),
            Err(LineageChannelWeightError::InvalidEvidence)
        );
        assert_eq!(
            LineageChannelWeightEvidence::from_json(
                &" ".repeat(LINEAGE_CHANNEL_WEIGHT_EVIDENCE_BYTE_LIMIT + 1)
            ),
            Err(LineageChannelWeightError::LimitExceeded)
        );
    }

    #[test]
    fn every_identity_matrix_and_anchor_boundary_fails_closed() {
        macro_rules! invalid {
            ($change:expr) => {{
                let mut candidate = evidence();
                $change(&mut candidate);
                assert_eq!(
                    candidate.validate(),
                    Err(LineageChannelWeightError::InvalidEvidence)
                );
            }};
        }
        invalid!(|v: &mut LineageChannelWeightEvidence| v.schema_version = "v2".into());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.estimation_run_id = "bad".into());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.source_snapshot_sha256 = "A".repeat(64));
        invalid!(|v: &mut LineageChannelWeightEvidence| v.knowledge_cutoff = "bad".into());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.channel_codes.clear());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.channel_codes[0] = " x ".into());
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.channel_codes[1] = v.channel_codes[0].clone()
        );
        invalid!(|v: &mut LineageChannelWeightEvidence| v.pair_evidence.clear());
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.pair_evidence[1].pair_id =
                v.pair_evidence[0].pair_id.clone()
        );
        invalid!(|v: &mut LineageChannelWeightEvidence| v.pair_evidence[0].pair_id = "bad".into());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.pair_evidence[0].group_id.clear());
        invalid!(|v: &mut LineageChannelWeightEvidence| v.pair_evidence[0].channel_scores.pop());
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.pair_evidence[0].channel_scores[0] = f64::NAN
        );
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.pair_evidence[0].channel_scores[0] = -0.1
        );
        invalid!(|v: &mut LineageChannelWeightEvidence| v.criterion_anchor.contract_version = 2);
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.criterion_anchor.anchor_kind_code =
                "internal".into()
        );
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.criterion_anchor.estimation_run_id =
                "018f47e7-7b5b-7cc0-98c6-15fdf9e3d99".into()
        );
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.criterion_anchor.source_snapshot_sha256 =
                "b".repeat(64)
        );
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.criterion_anchor.knowledge_cutoff =
                "2026-08-25T00:00:01Z".into()
        );
        invalid!(
            |v: &mut LineageChannelWeightEvidence| v.criterion_anchor.criterion_validity_status =
                "rejected".into()
        );
        invalid!(|v: &mut LineageChannelWeightEvidence| v.criterion_anchor.validated_pair_count = 0);
        invalid!(|v: &mut LineageChannelWeightEvidence| v.criterion_anchor.validated_pair_count = 1);
        assert!(!identifier(&"x".repeat(257)));
        assert!(digest(&"0".repeat(64)));
        assert!(!digest("short"));
        assert!(!canonical_time("2026-08-25T00:00:00+00:00"));
    }
}
