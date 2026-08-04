"""Replay extensions for task-revision-aware calibration artifacts."""

from __future__ import annotations

from typing import Any


def install(validation_module: Any, calibration_module: Any) -> None:
    """Teach the existing fail-closed replay layer about revision fields."""
    if getattr(validation_module, "_task_revision_replay_installed", False):
        return

    def _replay_rating_record(value: Any, *, path: str):
        """Reconstruct one revision-aware rating before assembly or fitting."""
        if type(value) is not calibration_module.ScoringFacetsRatingRecord:
            raise validation_module.assessment_error(
                "invalid_facets_rating_record",
                path,
                "records must contain exact ScoringFacetsRatingRecord values",
            )
        rebuilt = calibration_module.ScoringFacetsRatingRecord(
            assessment_fingerprint=value.assessment_fingerprint,
            rubric_fingerprint=value.rubric_fingerprint,
            construct_id=value.construct_id,
            request_fingerprint=value.request_fingerprint,
            result_fingerprint=value.result_fingerprint,
            observation_fingerprint=value.observation_fingerprint,
            respondent_id=value.respondent_id,
            response_id=value.response_id,
            response_content_fingerprint=value.response_content_fingerprint,
            task_id=value.task_id,
            occasion_id=value.occasion_id,
            criterion_id=value.criterion_id,
            engine_id=value.engine_id,
            engine_family_id=value.engine_family_id,
            engine_fingerprint=value.engine_fingerprint,
            status=value.status,
            score_category=value.score_category,
            allowed_scores=value.allowed_scores,
            schema_version=value.schema_version,
            task_revision_fingerprint=value.task_revision_fingerprint,
            task_family_id=value.task_family_id,
            _rating_token=calibration_module._RATING_TOKEN,
        )
        if not validation_module._same_value(
            value._content_dict(),
            rebuilt._content_dict(),
        ):
            raise validation_module.assessment_error(
                "facets_rating_replay_mismatch",
                path,
                "rating record no longer matches its normalized factory contract",
            )
        return rebuilt

    def _design_fields(
        design: Any,
        *,
        rating_fingerprints: tuple[str, ...],
    ) -> tuple[Any, ...]:
        """Return complete revision-aware design identity without handles."""
        return (
            design.schema_version,
            design.assessment_fingerprint,
            design.rubric_fingerprint,
            design.construct_id,
            design.occasion_id,
            design.criterion_id,
            design.category_values,
            design.respondent_ids,
            design.task_revision_fingerprints,
            design.task_ids,
            design.task_family_ids,
            design.response_ids,
            design.response_respondent_ids,
            design.response_task_ids,
            design.response_task_revision_fingerprints,
            design.response_content_fingerprints,
            design.rater_engine_ids,
            design.rater_engine_family_ids,
            design.rater_engine_fingerprints,
            rating_fingerprints,
            design.respondent_task_connected,
            design.task_rater_connected,
            design.connected,
        )

    validation_module._replay_rating_record = _replay_rating_record
    validation_module._design_fields = _design_fields
    validation_module._task_revision_replay_installed = True


__all__ = ["install"]
