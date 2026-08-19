"""Prompt-, model-, and rubric-stratified essay validation evidence.

This module adds provenance and review-routing semantics only. Numerical agreement,
degradation, and subgroup metrics remain delegated to the existing Rust-backed
validation report builder.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import InitVar, dataclass
from itertools import chain
from typing import Any

import numpy as np

from fast_mlsirm.rubric import RubricSpecification

from .._contract_safety import artifact_digest
from .._validation import (
    ASSESSMENT_SCHEMA_VERSION,
    CanonicalContract,
    assessment_error,
)
from ..assessment import AssessmentSpec
from ..execution import EngineDescriptor, EngineKind
from .contracts import EssayPrompt
from .validation_reporting import (
    EssayValidationEvidenceReport as _BaseEssayValidationEvidenceReport,
)
from .validation_reporting import (
    _REPORT_TOKEN,
    build_essay_validation_evidence_report as _build_base_validation_report,
)

_STRATUM_TOKEN = object()


@dataclass(frozen=True)
class EssayValidationStratum(CanonicalContract):
    """Factory-sealed prompt/model/rubric scope for one validation slice."""

    prompt_fingerprint: str
    prompt_id: str
    genre_id: str
    language_id: str
    model_family_id: str
    rubric_fingerprint: str
    rubric_version: str
    schema_version: str = ASSESSMENT_SCHEMA_VERSION
    _stratum_token: InitVar[object | None] = None

    def __post_init__(self, _stratum_token: object | None) -> None:
        """Reject direct construction; factories copy only sealed contract values."""
        if _stratum_token is not _STRATUM_TOKEN:
            raise assessment_error(
                "unverified_essay_validation_stratum",
                "$",
                "use build_essay_validation_stratum",
            )

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical stratum content without derived identities."""
        return {
            "schema_version": self.schema_version,
            "prompt_fingerprint": self.prompt_fingerprint,
            "prompt_id": self.prompt_id,
            "genre_id": self.genre_id,
            "language_id": self.language_id,
            "model_family_id": self.model_family_id,
            "rubric_fingerprint": self.rubric_fingerprint,
            "rubric_version": self.rubric_version,
        }

    @property
    def stratum_fingerprint(self) -> str:
        """Return SHA-256 over the exact validation stratum."""
        return artifact_digest(self)

    @property
    def stratum_handle(self) -> str:
        """Return a descriptive 128-bit public stratum handle."""
        return f"essay_validation_stratum_{self.stratum_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical stratum content and deterministic identities."""
        return {
            **self._content_dict(),
            "stratum_handle": self.stratum_handle,
            "stratum_fingerprint": self.stratum_fingerprint,
        }


def build_essay_validation_stratum(
    *,
    prompt: EssayPrompt,
    rubric: RubricSpecification,
    automated_engine: EngineDescriptor,
) -> EssayValidationStratum:
    """Build one immutable prompt/model/rubric validation scope."""
    if type(prompt) is not EssayPrompt:
        raise assessment_error(
            "invalid_essay_validation_prompt",
            "$.prompt",
            "prompt must be an exact EssayPrompt",
        )
    if type(rubric) is not RubricSpecification:
        raise assessment_error(
            "invalid_essay_validation_rubric",
            "$.rubric",
            "rubric must be an exact RubricSpecification",
        )
    if type(automated_engine) is not EngineDescriptor or (
        automated_engine.engine_kind is not EngineKind.AUTOMATED
    ):
        raise assessment_error(
            "invalid_essay_validation_automated_engine",
            "$.automated_engine",
            "automated_engine must be an exact automated EngineDescriptor",
        )
    if prompt.task_family_id not in rubric.task_families:
        raise assessment_error(
            "essay_validation_prompt_task_family_mismatch",
            "$.prompt.task_family_id",
            "prompt task family is not declared by the rubric",
        )
    return EssayValidationStratum(
        prompt_fingerprint=prompt.prompt_fingerprint,
        prompt_id=prompt.prompt_id,
        genre_id=prompt.genre_id,
        language_id=prompt.language_id,
        model_family_id=automated_engine.engine_family_id,
        rubric_fingerprint=rubric.fingerprint,
        rubric_version=rubric.rubric_version,
        _stratum_token=_STRATUM_TOKEN,
    )


@dataclass(frozen=True)
class EssayValidationEvidenceReport(_BaseEssayValidationEvidenceReport):
    """Validation evidence with an explicit or explicitly pooled stratum scope."""

    validation_stratum: EssayValidationStratum | None = None

    def __post_init__(self, _report_token: object | None) -> None:
        """Preserve the base report factory seal."""
        super().__post_init__(_report_token)

    def _content_dict(self) -> dict[str, Any]:
        """Return base evidence plus explicit stratification provenance."""
        return {
            **super()._content_dict(),
            "validation_stratum": (
                None
                if self.validation_stratum is None
                else self.validation_stratum.to_dict()
            ),
        }


def _bind_stratum(
    report: _BaseEssayValidationEvidenceReport,
    validation_stratum: EssayValidationStratum | None,
) -> EssayValidationEvidenceReport:
    """Re-seal a base report with provenance only; no metric arithmetic is repeated."""
    return EssayValidationEvidenceReport(
        report_id=report.report_id,
        assessment_spec=report.assessment_spec,
        construct_id=report.construct_id,
        rubric_fingerprint=report.rubric_fingerprint,
        criterion_id=report.criterion_id,
        automated_engine=report.automated_engine,
        reference_engine=report.reference_engine,
        validation_dataset_fingerprint=report.validation_dataset_fingerprint,
        category_count=report.category_count,
        paired_observation_count=report.paired_observation_count,
        metrics=report.metrics,
        review_trigger_ids=report.review_trigger_ids,
        metadata=report.metadata,
        schema_version=report.schema_version,
        validation_stratum=validation_stratum,
        _report_token=_REPORT_TOKEN,
    )


def build_essay_validation_evidence_report(
    *,
    report_id: str,
    assessment: AssessmentSpec,
    construct_id: str,
    rubric_fingerprint: str,
    criterion_id: str,
    automated_engine: EngineDescriptor,
    reference_engine: EngineDescriptor,
    validation_dataset_fingerprint: str,
    automated_labels: np.ndarray,
    reference_labels: np.ndarray,
    category_count: int,
    human_human_labels: tuple[np.ndarray, np.ndarray] | None = None,
    subgroup_labels: np.ndarray | None = None,
    validation_stratum: EssayValidationStratum | None = None,
    additional_review_trigger_ids: Iterable[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> EssayValidationEvidenceReport:
    """Build Rust-backed validation evidence and bind its declared stratum."""
    if (
        validation_stratum is not None
        and type(validation_stratum) is not EssayValidationStratum
    ):
        raise assessment_error(
            "invalid_essay_validation_stratum",
            "$.validation_stratum",
            "validation_stratum must be an EssayValidationStratum or None",
        )
    if (
        validation_stratum is not None
        and type(automated_engine) is EngineDescriptor
        and validation_stratum.model_family_id != automated_engine.engine_family_id
    ):
        raise assessment_error(
            "essay_validation_stratum_engine_mismatch",
            "$.validation_stratum.model_family_id",
            "validation stratum does not match the automated engine family",
        )

    review_triggers: Iterable[str] = additional_review_trigger_ids
    if validation_stratum is None:
        review_triggers = chain(
            additional_review_trigger_ids,
            ("validation_stratification_missing",),
        )

    report = _build_base_validation_report(
        report_id=report_id,
        assessment=assessment,
        construct_id=construct_id,
        rubric_fingerprint=rubric_fingerprint,
        criterion_id=criterion_id,
        automated_engine=automated_engine,
        reference_engine=reference_engine,
        validation_dataset_fingerprint=validation_dataset_fingerprint,
        automated_labels=automated_labels,
        reference_labels=reference_labels,
        category_count=category_count,
        human_human_labels=human_human_labels,
        subgroup_labels=subgroup_labels,
        additional_review_trigger_ids=review_triggers,
        metadata=metadata,
    )
    if validation_stratum is not None:
        if validation_stratum.rubric_fingerprint != report.rubric_fingerprint:
            raise assessment_error(
                "essay_validation_stratum_rubric_mismatch",
                "$.validation_stratum.rubric_fingerprint",
                "validation stratum does not match the report rubric",
            )
    return _bind_stratum(report, validation_stratum)


__all__ = [
    "EssayValidationEvidenceReport",
    "EssayValidationStratum",
    "build_essay_validation_evidence_report",
    "build_essay_validation_stratum",
]
