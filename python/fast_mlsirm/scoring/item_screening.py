"""Governed structural and semantic screening evidence for item-bank candidates.

This module is deliberately non-numerical. It binds exact item/rubric/blueprint/
generation provenance to explicit screening dimensions and keeps accepted
limitations distinct from failures. Psychometric calibration, DIF, information,
linking, drift, and uncertainty remain owned by their existing Rust-backed
surfaces.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass
from enum import Enum
from typing import Any

from ._contract_safety import (
    artifact_digest,
    bounded_values,
    descriptive_identifier,
    enum_value,
    freeze_metadata,
    sorted_fingerprints,
)
from ._validation import CanonicalContract, assessment_error, fingerprint, thaw_json_value

_FINDING_TOKEN = object()
_RESULT_TOKEN = object()


class ScreeningDimension(str, Enum):
    """Required governed screening dimensions for one generated item candidate."""

    ANSWERABILITY = "answerability"
    AMBIGUITY_RISK = "ambiguity_risk"
    EVIDENCE_ENTAILMENT = "evidence_entailment"
    DISTRACTOR_QUALITY = "distractor_quality"
    SEMANTIC_REDUNDANCY = "semantic_redundancy"
    LEAKAGE_RISK = "leakage_risk"
    FAIRNESS_RISK = "fairness_risk"
    ADVERSARIAL_CONTENT = "adversarial_content"
    PERTURBATION_DIRECTION = "perturbation_direction"
    COST_RUNTIME_SUITABILITY = "cost_runtime_suitability"


REQUIRED_SCREENING_DIMENSIONS = tuple(ScreeningDimension)


class ScreeningStatus(str, Enum):
    """Outcome of one screening dimension or the aggregate screening result."""

    PASS = "pass"
    ACCEPTED_WITH_LIMITATION = "accepted_with_limitation"
    FAIL = "fail"


def _optional_identifier(value: Any, name: str) -> str | None:
    """Validate one optional descriptive identifier."""
    return None if value is None else descriptive_identifier(value, name)


@dataclass(frozen=True)
class ItemScreeningFinding(CanonicalContract):
    """One evidence-bound screening judgment for one governed dimension."""

    dimension: ScreeningDimension
    status: ScreeningStatus
    reason_code: str
    evidence_fingerprints: tuple[str, ...]
    limitation_code: str | None
    metadata: Mapping[str, Any]
    _token: InitVar[object | None] = None

    def __post_init__(self, _token: object | None) -> None:
        """Seal the factory boundary and require explicit evidence/provenance."""
        if _token is not _FINDING_TOKEN:
            raise assessment_error(
                "unverified_screening_finding",
                "$",
                "use build_item_screening_finding",
            )
        object.__setattr__(
            self,
            "dimension",
            enum_value(self.dimension, ScreeningDimension, "dimension"),
        )
        object.__setattr__(
            self,
            "status",
            enum_value(self.status, ScreeningStatus, "status"),
        )
        object.__setattr__(
            self,
            "reason_code",
            descriptive_identifier(self.reason_code, "reason_code"),
        )
        evidence = sorted_fingerprints(
            self.evidence_fingerprints,
            "evidence_fingerprints",
            minimum=0,
            maximum=64,
        )
        if not evidence:
            raise assessment_error(
                "screening_evidence_required",
                "$.evidence_fingerprints",
                "screening findings require at least one evidence fingerprint",
            )
        object.__setattr__(self, "evidence_fingerprints", evidence)
        object.__setattr__(
            self,
            "limitation_code",
            _optional_identifier(self.limitation_code, "limitation_code"),
        )
        if (
            self.status is ScreeningStatus.ACCEPTED_WITH_LIMITATION
            and self.limitation_code is None
        ):
            raise assessment_error(
                "limitation_code_required",
                "$.limitation_code",
                "accepted limitations require an explicit limitation code",
            )
        if (
            self.status is not ScreeningStatus.ACCEPTED_WITH_LIMITATION
            and self.limitation_code is not None
        ):
            raise assessment_error(
                "unexpected_limitation_code",
                "$.limitation_code",
                "limitation code is valid only for an accepted limitation",
            )
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical finding content excluding the derived fingerprint."""
        return {
            "dimension": self.dimension.value,
            "status": self.status.value,
            "reason_code": self.reason_code,
            "evidence_fingerprints": list(self.evidence_fingerprints),
            "limitation_code": self.limitation_code,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def finding_fingerprint(self) -> str:
        """Return the exact immutable screening-finding digest."""
        return artifact_digest(self)

    def to_dict(self) -> dict[str, Any]:
        """Return canonical finding content plus its exact digest."""
        return {
            **self._content_dict(),
            "finding_fingerprint": self.finding_fingerprint,
        }


@dataclass(frozen=True)
class CandidateScreeningResult(CanonicalContract):
    """Complete immutable screening evidence for one exact generated candidate."""

    result_id: str
    item_content_fingerprint: str
    rubric_fingerprint: str
    blueprint_fingerprint: str
    generation_contract_fingerprint: str
    screening_policy_fingerprint: str
    findings: tuple[ItemScreeningFinding, ...]
    metadata: Mapping[str, Any]
    _token: InitVar[object | None] = None

    def __post_init__(self, _token: object | None) -> None:
        """Require exact provenance and every screening dimension exactly once."""
        if _token is not _RESULT_TOKEN:
            raise assessment_error(
                "unverified_candidate_screening_result",
                "$",
                "use build_candidate_screening_result",
            )
        object.__setattr__(
            self,
            "result_id",
            descriptive_identifier(self.result_id, "result_id"),
        )
        for name in (
            "item_content_fingerprint",
            "rubric_fingerprint",
            "blueprint_fingerprint",
            "generation_contract_fingerprint",
            "screening_policy_fingerprint",
        ):
            object.__setattr__(self, name, fingerprint(getattr(self, name), name))

        raw_findings = bounded_values(
            self.findings,
            "findings",
            minimum=1,
            maximum=len(REQUIRED_SCREENING_DIMENSIONS) + 1,
        )
        normalized: list[ItemScreeningFinding] = []
        seen: set[ScreeningDimension] = set()
        for index, finding in enumerate(raw_findings):
            if not isinstance(finding, ItemScreeningFinding):
                raise assessment_error(
                    "invalid_screening_finding",
                    f"$.findings[{index}]",
                    "findings must contain verified ItemScreeningFinding values",
                )
            if finding.dimension in seen:
                raise assessment_error(
                    "duplicate_screening_dimension",
                    f"$.findings[{index}].dimension",
                    "each screening dimension must appear exactly once",
                )
            seen.add(finding.dimension)
            normalized.append(finding)
        required = set(REQUIRED_SCREENING_DIMENSIONS)
        if seen != required:
            raise assessment_error(
                "screening_dimensions_incomplete",
                "$.findings",
                "screening requires every governed dimension exactly once",
            )
        normalized.sort(key=lambda finding: finding.dimension.value)
        canonical_order = {dimension: index for index, dimension in enumerate(REQUIRED_SCREENING_DIMENSIONS)}
        normalized.sort(key=lambda finding: canonical_order[finding.dimension])
        object.__setattr__(self, "findings", tuple(normalized))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def screening_status(self) -> ScreeningStatus:
        """Derive aggregate status without caller-controlled promotion."""
        if any(finding.status is ScreeningStatus.FAIL for finding in self.findings):
            return ScreeningStatus.FAIL
        if any(
            finding.status is ScreeningStatus.ACCEPTED_WITH_LIMITATION
            for finding in self.findings
        ):
            return ScreeningStatus.ACCEPTED_WITH_LIMITATION
        return ScreeningStatus.PASS

    @property
    def eligible_for_pilot(self) -> bool:
        """Return whether this screening result permits entry into governed piloting."""
        return self.screening_status is not ScreeningStatus.FAIL

    def _content_dict(self) -> dict[str, Any]:
        """Return canonical result content excluding derived identities."""
        return {
            "result_id": self.result_id,
            "item_content_fingerprint": self.item_content_fingerprint,
            "rubric_fingerprint": self.rubric_fingerprint,
            "blueprint_fingerprint": self.blueprint_fingerprint,
            "generation_contract_fingerprint": self.generation_contract_fingerprint,
            "screening_policy_fingerprint": self.screening_policy_fingerprint,
            "findings": [finding.to_dict() for finding in self.findings],
            "screening_status": self.screening_status.value,
            "eligible_for_pilot": self.eligible_for_pilot,
            "metadata": thaw_json_value(self.metadata),
        }

    @property
    def result_fingerprint(self) -> str:
        """Return the exact immutable candidate-screening digest."""
        return artifact_digest(self)

    @property
    def result_handle(self) -> str:
        """Return a compact public handle derived from the full result digest."""
        return f"candidate_screening_result_{self.result_fingerprint[:32]}"

    def to_dict(self) -> dict[str, Any]:
        """Return canonical result content plus deterministic public identities."""
        return {
            **self._content_dict(),
            "result_handle": self.result_handle,
            "result_fingerprint": self.result_fingerprint,
        }


def build_item_screening_finding(**values: Any) -> ItemScreeningFinding:
    """Build one validated immutable screening finding."""
    normalized = dict(values)
    normalized.setdefault("limitation_code", None)
    return ItemScreeningFinding(**normalized, _token=_FINDING_TOKEN)


def build_candidate_screening_result(**values: Any) -> CandidateScreeningResult:
    """Build one validated complete candidate-screening result."""
    return CandidateScreeningResult(**values, _token=_RESULT_TOKEN)


__all__ = [
    "CandidateScreeningResult",
    "ItemScreeningFinding",
    "REQUIRED_SCREENING_DIMENSIONS",
    "ScreeningDimension",
    "ScreeningStatus",
    "build_candidate_screening_result",
    "build_item_screening_finding",
]
