"""Revision-indexed contract extensions for scoring-facets handoffs."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any

import numpy as np

from . import calibration as _base
from ._validation import assessment_error, descriptive_identifier, fingerprint
from .execution import ObservationStatus

_OriginalRatingRecord = _base.ScoringFacetsRatingRecord
_OriginalDesign = _base.ScoringFacetsDesign


@dataclass(frozen=True, init=False)
class ScoringFacetsRatingRecord(_OriginalRatingRecord):
    """One governed criterion rating bound to an exact task revision."""

    task_family_id: str
    task_revision_fingerprint: str = field(init=False)

    def __init__(
        self,
        assessment_fingerprint: str,
        rubric_fingerprint: str,
        construct_id: str,
        request_fingerprint: str,
        result_fingerprint: str,
        observation_fingerprint: str,
        respondent_id: str,
        response_id: str,
        response_content_fingerprint: str,
        task_id: str,
        occasion_id: str,
        criterion_id: str,
        engine_id: str,
        engine_family_id: str,
        engine_fingerprint: str,
        status: ObservationStatus,
        score_category: int | None,
        allowed_scores: tuple[int, ...],
        schema_version: str = _base.ASSESSMENT_SCHEMA_VERSION,
        _rating_token: object | None = None,
        task_family_id: str | None = None,
        task_revision_fingerprint: str | None = None,
    ) -> None:
        """Build through the original sealed rating invariants, then add revision."""
        _OriginalRatingRecord.__init__(
            self,
            assessment_fingerprint=assessment_fingerprint,
            rubric_fingerprint=rubric_fingerprint,
            construct_id=construct_id,
            request_fingerprint=request_fingerprint,
            result_fingerprint=result_fingerprint,
            observation_fingerprint=observation_fingerprint,
            respondent_id=respondent_id,
            response_id=response_id,
            response_content_fingerprint=response_content_fingerprint,
            task_id=task_id,
            occasion_id=occasion_id,
            criterion_id=criterion_id,
            engine_id=engine_id,
            engine_family_id=engine_family_id,
            engine_fingerprint=engine_fingerprint,
            status=status,
            score_category=score_category,
            allowed_scores=allowed_scores,
            schema_version=schema_version,
            _rating_token=_rating_token,
        )
        family = descriptive_identifier(
            "legacy_task_family" if task_family_id is None else task_family_id,
            "task_family_id",
        )
        revision = task_revision_fingerprint
        if revision is None:
            revision = hashlib.sha256(
                f"{self.task_id}_revision".encode("utf-8")
            ).hexdigest()
        object.__setattr__(self, "task_family_id", family)
        object.__setattr__(
            self,
            "task_revision_fingerprint",
            fingerprint(revision, "task_revision_fingerprint"),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return original rating content plus exact task revision provenance."""
        content = super()._content_dict()
        content["task_family_id"] = self.task_family_id
        content["task_revision_fingerprint"] = self.task_revision_fingerprint
        return content


@dataclass(frozen=True, init=False)
class ScoringFacetsDesign(_OriginalDesign):
    """One respondent-by-task-revision-by-rater Rust estimator design."""

    task_revision_fingerprints: tuple[str, ...]
    task_family_ids: tuple[str, ...]
    response_task_revision_fingerprints: tuple[str, ...]

    def __init__(
        self,
        assessment_fingerprint: str,
        rubric_fingerprint: str,
        construct_id: str,
        occasion_id: str,
        criterion_id: str,
        category_values: tuple[int, ...],
        respondent_ids: tuple[str, ...],
        task_ids: tuple[str, ...],
        response_ids: tuple[str, ...],
        response_respondent_ids: tuple[str, ...],
        response_task_ids: tuple[str, ...],
        response_content_fingerprints: tuple[str, ...],
        rater_engine_ids: tuple[str, ...],
        rater_engine_family_ids: tuple[str, ...],
        rater_engine_fingerprints: tuple[str, ...],
        rating_records: tuple[ScoringFacetsRatingRecord, ...],
        respondent_task_connected: bool,
        task_rater_connected: bool,
        connected: bool,
        schema_version: str = _base.ASSESSMENT_SCHEMA_VERSION,
        _design_token: object | None = None,
        task_revision_fingerprints: tuple[str, ...] = (),
        task_family_ids: tuple[str, ...] = (),
        response_task_revision_fingerprints: tuple[str, ...] = (),
    ) -> None:
        """Build the sealed base design and attach aligned revision identities."""
        _OriginalDesign.__init__(
            self,
            assessment_fingerprint=assessment_fingerprint,
            rubric_fingerprint=rubric_fingerprint,
            construct_id=construct_id,
            occasion_id=occasion_id,
            criterion_id=criterion_id,
            category_values=category_values,
            respondent_ids=respondent_ids,
            task_ids=task_ids,
            response_ids=response_ids,
            response_respondent_ids=response_respondent_ids,
            response_task_ids=response_task_ids,
            response_content_fingerprints=response_content_fingerprints,
            rater_engine_ids=rater_engine_ids,
            rater_engine_family_ids=rater_engine_family_ids,
            rater_engine_fingerprints=rater_engine_fingerprints,
            rating_records=rating_records,
            respondent_task_connected=respondent_task_connected,
            task_rater_connected=task_rater_connected,
            connected=connected,
            schema_version=schema_version,
            _design_token=_design_token,
        )
        axis_count = len(task_revision_fingerprints)
        if axis_count != len(task_ids) or axis_count != len(task_family_ids):
            raise assessment_error(
                "invalid_task_revision_axis",
                "$.task_revision_fingerprints",
                "revision, logical task, and task-family axes must align",
            )
        if len(response_task_revision_fingerprints) != len(response_ids):
            raise assessment_error(
                "invalid_response_revision_axis",
                "$.response_task_revision_fingerprints",
                "response and task-revision audit axes must align",
            )
        object.__setattr__(
            self,
            "task_revision_fingerprints",
            tuple(task_revision_fingerprints),
        )
        object.__setattr__(self, "task_family_ids", tuple(task_family_ids))
        object.__setattr__(
            self,
            "response_task_revision_fingerprints",
            tuple(response_task_revision_fingerprints),
        )

    def _content_dict(self) -> dict[str, Any]:
        """Return base sparse content plus aligned exact revision metadata."""
        content = super()._content_dict()
        content["task_revision_fingerprints"] = list(
            self.task_revision_fingerprints
        )
        content["task_family_ids"] = list(self.task_family_ids)
        content["task_revisions"] = [
            {
                "task_revision_fingerprint": revision,
                "task_id": task_id,
                "task_family_id": family_id,
            }
            for revision, task_id, family_id in zip(
                self.task_revision_fingerprints,
                self.task_ids,
                self.task_family_ids,
                strict=True,
            )
        ]
        content["response_task_revision_fingerprints"] = list(
            self.response_task_revision_fingerprints
        )
        for entry, revision in zip(
            content["respondent_task_responses"],
            self.response_task_revision_fingerprints,
            strict=True,
        ):
            entry["task_revision_fingerprint"] = revision
        return content

    def _indexes(self) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        """Return deterministic respondent, task-revision, and rater maps."""
        return (
            {value: index for index, value in enumerate(self.respondent_ids)},
            {
                value: index
                for index, value in enumerate(self.task_revision_fingerprints)
            },
            {
                value: index
                for index, value in enumerate(self.rater_engine_fingerprints)
            },
        )

    def _dense_array(self, *, zero_based: bool) -> np.ndarray:
        """Materialize a fresh respondent-by-revision-by-rater tensor."""
        output = np.full(
            (
                len(self.respondent_ids),
                len(self.task_revision_fingerprints),
                len(self.rater_engine_fingerprints),
            ),
            np.nan,
            dtype=np.float64,
        )
        respondent_index, task_index, rater_index = self._indexes()
        category_index = {
            value: index for index, value in enumerate(self.category_values)
        }
        for record in self.rating_records:
            if record.status is not ObservationStatus.SCORED:
                continue
            score = record.score_category
            if score is None:  # pragma: no cover - sealed record invariant
                raise RuntimeError("scored rating category is unavailable")
            output[
                respondent_index[record.respondent_id],
                task_index[record.task_revision_fingerprint],
                rater_index[record.engine_fingerprint],
            ] = category_index[score] if zero_based else score
        return output

    def response_states(
        self,
    ) -> tuple[tuple[tuple[ObservationStatus | None, ...], ...], ...]:
        """Return assigned states and ``None`` for unassigned revision cells."""
        states: list[list[list[ObservationStatus | None]]] = [
            [
                [None for _rater in self.rater_engine_fingerprints]
                for _revision in self.task_revision_fingerprints
            ]
            for _respondent in self.respondent_ids
        ]
        respondent_index, task_index, rater_index = self._indexes()
        for record in self.rating_records:
            states[respondent_index[record.respondent_id]][
                task_index[record.task_revision_fingerprint]
            ][rater_index[record.engine_fingerprint]] = record.status
        return tuple(
            tuple(tuple(rater_states) for rater_states in task_states)
            for task_states in states
        )


__all__ = ["ScoringFacetsDesign", "ScoringFacetsRatingRecord"]
