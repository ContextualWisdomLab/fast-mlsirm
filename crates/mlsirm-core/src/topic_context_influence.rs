//! Fail-closed boundary for TEPP posterior topic-context influence.
//!
//! The current crossed binary MAP estimator cannot consume posterior
//! logistic-normal plausible values or compute the ADR-0210 case-deletion
//! diagnostic.  This boundary validates the producer identity and then refuses
//! estimation so callers cannot threshold plausible values into binary data or
//! mislabel an existing estimator as the required estimand.

/// Exact TEPP posterior schema required by the future estimator.
pub const TEPP_TOPIC_CONTEXT_POSTERIOR_SCHEMA: &str = "tepp.topic_context_posterior.v1";
/// Exact fast-mlsirm result schema reserved for the future estimator.
pub const TOPIC_CONTEXT_INFLUENCE_SCHEMA: &str = "fast_mlsirm.topic_context_influence.v1";

/// Provenance needed before a posterior-aware influence fit may start.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TopicContextInfluenceRequest {
    /// TEPP producer schema identity.
    pub tepp_schema_version: String,
    /// Canonical TEPP posterior artifact SHA-256.
    pub tepp_artifact_sha256: String,
    /// Opaque TEPP run identity.
    pub tepp_run_id: String,
    /// Immutable source snapshot identity.
    pub snapshot_id: String,
    /// Historical knowledge cutoff.
    pub knowledge_cutoff: String,
    /// Number of posterior draws represented by the artifact.
    pub posterior_draw_count: usize,
}

fn canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

/// Validate the TEPP producer identity and fail until the exact estimator ships.
///
/// # Errors
///
/// Returns a contract error for a foreign, unbound, or draw-free artifact.
/// A valid request returns `topic_context_influence_estimator_unavailable`;
/// it is never routed to the binary crossed-person estimator.
pub fn fit_topic_context_influence(
    request: &TopicContextInfluenceRequest,
) -> Result<(), &'static str> {
    if request.tepp_schema_version != TEPP_TOPIC_CONTEXT_POSTERIOR_SCHEMA
        || !canonical_sha256(&request.tepp_artifact_sha256)
        || request.tepp_run_id.trim().is_empty()
        || request.snapshot_id.trim().is_empty()
        || request.knowledge_cutoff.trim().is_empty()
        || request.posterior_draw_count == 0
    {
        return Err("invalid_tepp_topic_context_posterior");
    }
    Err("topic_context_influence_estimator_unavailable")
}

#[cfg(test)]
mod tests {
    use super::{
        fit_topic_context_influence, TopicContextInfluenceRequest,
        TEPP_TOPIC_CONTEXT_POSTERIOR_SCHEMA,
    };

    fn request() -> TopicContextInfluenceRequest {
        TopicContextInfluenceRequest {
            tepp_schema_version: TEPP_TOPIC_CONTEXT_POSTERIOR_SCHEMA.into(),
            tepp_artifact_sha256: "a".repeat(64),
            tepp_run_id: "run-1".into(),
            snapshot_id: "snapshot-1".into(),
            knowledge_cutoff: "2026-08-01T00:00:00Z".into(),
            posterior_draw_count: 8,
        }
    }

    #[test]
    fn valid_posterior_does_not_fall_through_to_binary_estimator() {
        assert_eq!(
            fit_topic_context_influence(&request()),
            Err("topic_context_influence_estimator_unavailable")
        );
    }

    #[test]
    fn rejects_foreign_or_draw_free_input() {
        let mut invalid = request();
        invalid.posterior_draw_count = 0;
        assert_eq!(
            fit_topic_context_influence(&invalid),
            Err("invalid_tepp_topic_context_posterior")
        );
        invalid = request();
        invalid.tepp_schema_version = "binary-threshold.v1".into();
        assert_eq!(
            fit_topic_context_influence(&invalid),
            Err("invalid_tepp_topic_context_posterior")
        );
    }
}
