//! TEPP posterior topic-context influence arithmetic and fail-closed fitting boundary.
//!
//! The current crossed binary MAP estimator cannot consume posterior
//! logistic-normal plausible values or compute the ADR-0210 case-deletion
//! diagnostic's complete case-deletion refits. This module owns the exact
//! observed-information quadratic form, validates producer identity, and then
//! refuses fitting so callers cannot threshold plausible values into binary
//! data or mislabel an existing estimator as the required estimand.

/// Exact TEPP posterior schema required by the future estimator.
pub const TEPP_TOPIC_CONTEXT_POSTERIOR_SCHEMA: &str = "tepp.topic_context_posterior.v1";
/// Exact fast-mlsirm result schema reserved for the future estimator.
pub const TOPIC_CONTEXT_INFLUENCE_SCHEMA: &str = "fast_mlsirm.topic_context_influence.v1";

/// Posterior-draw case-deletion influence under one observed-information block.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseDeletionInfluence {
    /// The exact diagnostic for every TEPP plausible-value draw.
    pub per_draw: Vec<f64>,
    /// Monte Carlo posterior expectation, the arithmetic mean across draws.
    pub posterior_mean: f64,
}

/// Evaluate `(ψ[-d] - ψ)' I(ψ) (ψ[-d] - ψ)` for each draw.
///
/// Arrays are draw-major. `information` contains one row-major `p × p`
/// observed-information matrix per draw. Equal Monte Carlo mass is inherited
/// from TEPP's plausible-value sample; no application weight is introduced.
pub fn case_deletion_influence_cpu(
    full: &[f64],
    deleted: &[f64],
    information: &[f64],
    draws: usize,
    parameters: usize,
) -> Result<CaseDeletionInfluence, &'static str> {
    let vector_len = draws
        .checked_mul(parameters)
        .ok_or("topic_context_influence_shape_overflow")?;
    let matrix_len = vector_len
        .checked_mul(parameters)
        .ok_or("topic_context_influence_shape_overflow")?;
    if draws == 0
        || parameters == 0
        || full.len() != vector_len
        || deleted.len() != vector_len
        || information.len() != matrix_len
        || full
            .iter()
            .chain(deleted)
            .chain(information)
            .any(|x| !x.is_finite())
    {
        return Err("invalid_topic_context_influence_inputs");
    }

    let mut per_draw = Vec::with_capacity(draws);
    for draw in 0..draws {
        let vector_start = draw * parameters;
        let matrix_start = draw * parameters * parameters;
        let delta = (0..parameters)
            .map(|index| deleted[vector_start + index] - full[vector_start + index])
            .collect::<Vec<_>>();
        let mut diagnostic = 0.0;
        for row in 0..parameters {
            for column in 0..parameters {
                let forward = information[matrix_start + row * parameters + column];
                let reverse = information[matrix_start + column * parameters + row];
                if forward != reverse {
                    return Err("observed_information_not_symmetric");
                }
                diagnostic += delta[row] * forward * delta[column];
            }
        }
        if !diagnostic.is_finite() || diagnostic < 0.0 {
            return Err("invalid_observed_information_quadratic_form");
        }
        per_draw.push(diagnostic);
    }
    let posterior_mean = per_draw.iter().sum::<f64>() / draws as f64;
    Ok(CaseDeletionInfluence {
        per_draw,
        posterior_mean,
    })
}

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
        case_deletion_influence_cpu, fit_topic_context_influence, TopicContextInfluenceRequest,
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

    #[test]
    fn case_deletion_primitive_recovers_injected_influence() {
        let result = case_deletion_influence_cpu(
            &[1.0, 2.0, 3.0, 4.0],
            &[2.0, 4.0, 4.0, 6.0],
            &[2.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0],
            2,
            2,
        )
        .unwrap();
        assert_eq!(result.per_draw, vec![14.0, 17.0]);
        assert_eq!(result.posterior_mean, 15.5);
    }

    #[test]
    fn case_deletion_primitive_rejects_non_information_input() {
        assert_eq!(
            case_deletion_influence_cpu(&[0.0, 0.0], &[1.0, 1.0], &[1.0, 2.0, 0.0, 1.0], 1, 2),
            Err("observed_information_not_symmetric")
        );
        assert_eq!(
            case_deletion_influence_cpu(&[0.0], &[1.0], &[-1.0], 1, 1),
            Err("invalid_observed_information_quadratic_form")
        );
    }
}
