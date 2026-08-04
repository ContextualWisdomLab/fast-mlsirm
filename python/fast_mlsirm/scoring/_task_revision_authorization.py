"""Task-revision propagation through the engine-authorization boundary."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from fast_mlsirm.rubric.models import RubricSpecification

from ._validation import assessment_error
from .assessment import AssessmentSpec
from .execution import ObservationGranularity
from ._task_revision_contract import ScoringRequest


def install(authorization_module: Any) -> None:
    """Replace the authorization request factory with its schema-1.1 form."""
    if getattr(authorization_module, "_task_revision_authorization_installed", False):
        return

    def build_scoring_request(
        *,
        request_id: str,
        assessment: AssessmentSpec,
        rubric: RubricSpecification,
        granularity: ObservationGranularity | str,
        respondent_id: str,
        response_id: str,
        task_id: str,
        task_revision_fingerprint: str,
        task_family_id: str,
        occasion_id: str,
        criterion_ids: Iterable[str] = (),
        response_content_fingerprint: str,
        response_character_count: int,
        response_unit_count: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> ScoringRequest:
        """Build a revision-bound request with authoritative engine policy."""
        if not isinstance(assessment, AssessmentSpec):
            raise assessment_error(
                "invalid_assessment_spec",
                "$.assessment",
                "assessment must be an AssessmentSpec",
            )
        return authorization_module._base_build_scoring_request(
            request_id=request_id,
            assessment=assessment,
            rubric=rubric,
            granularity=granularity,
            respondent_id=respondent_id,
            response_id=response_id,
            task_id=task_id,
            task_revision_fingerprint=task_revision_fingerprint,
            task_family_id=task_family_id,
            occasion_id=occasion_id,
            criterion_ids=criterion_ids,
            response_content_fingerprint=response_content_fingerprint,
            response_character_count=response_character_count,
            response_unit_count=response_unit_count,
            metadata=authorization_module._authorization_metadata(
                assessment,
                metadata,
            ),
        )

    authorization_module.build_scoring_request = build_scoring_request
    authorization_module._task_revision_authorization_installed = True


__all__ = ["install"]
