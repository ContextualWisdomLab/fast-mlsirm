"""Fail-first governed item-bank screening contracts for issue #609."""

from __future__ import annotations

import pytest

from fast_mlsirm.scoring.item_screening import (
    REQUIRED_SCREENING_DIMENSIONS,
    CandidateScreeningResult,
    ScreeningDimension,
    ScreeningStatus,
    build_candidate_screening_result,
    build_item_screening_finding,
)
from fast_mlsirm.scoring._validation import AssessmentSpecError


def _digest(char: str) -> str:
    return char * 64


def _finding(dimension: ScreeningDimension, *, status: ScreeningStatus = ScreeningStatus.PASS):
    values = {
        "dimension": dimension,
        "status": status,
        "reason_code": f"{dimension.value}_review",
        "evidence_fingerprints": (_digest("a"),),
        "metadata": {},
    }
    if status is ScreeningStatus.ACCEPTED_WITH_LIMITATION:
        values["limitation_code"] = "accepted_bounded_limitation"
    return build_item_screening_finding(**values)


def _all_findings(*, override: dict[ScreeningDimension, ScreeningStatus] | None = None):
    override = override or {}
    return tuple(
        _finding(dimension, status=override.get(dimension, ScreeningStatus.PASS))
        for dimension in reversed(REQUIRED_SCREENING_DIMENSIONS)
    )


def _result(**overrides):
    values = {
        "result_id": "candidate_screening_result",
        "item_content_fingerprint": _digest("b"),
        "rubric_fingerprint": _digest("c"),
        "blueprint_fingerprint": _digest("d"),
        "generation_contract_fingerprint": _digest("e"),
        "screening_policy_fingerprint": _digest("f"),
        "findings": _all_findings(),
        "metadata": {"screening_batch": "offline_fixture"},
    }
    values.update(overrides)
    return build_candidate_screening_result(**values)


def test_complete_passing_screening_is_pilot_eligible_and_canonical() -> None:
    """A complete passing screen is immutable, ordered, and pilot-eligible."""
    result = _result()

    assert isinstance(result, CandidateScreeningResult)
    assert result.screening_status is ScreeningStatus.PASS
    assert result.eligible_for_pilot is True
    assert tuple(f.dimension for f in result.findings) == REQUIRED_SCREENING_DIMENSIONS
    assert result.result_handle.startswith("candidate_screening_result_")
    assert len(result.result_fingerprint) == 64
    assert result.to_dict()["screening_status"] == "pass"


def test_accepted_limitation_stays_explicit_without_becoming_failure() -> None:
    """Accepted limitations remain visible while allowing the screened state."""
    result = _result(
        findings=_all_findings(
            override={ScreeningDimension.COST_RUNTIME_SUITABILITY: ScreeningStatus.ACCEPTED_WITH_LIMITATION}
        )
    )

    assert result.screening_status is ScreeningStatus.ACCEPTED_WITH_LIMITATION
    assert result.eligible_for_pilot is True
    finding = next(
        item for item in result.findings if item.dimension is ScreeningDimension.COST_RUNTIME_SUITABILITY
    )
    assert finding.limitation_code == "accepted_bounded_limitation"


def test_failed_dimension_blocks_pilot_eligibility() -> None:
    """One failed screening dimension must fail the aggregate result closed."""
    result = _result(
        findings=_all_findings(
            override={ScreeningDimension.ANSWERABILITY: ScreeningStatus.FAIL}
        )
    )

    assert result.screening_status is ScreeningStatus.FAIL
    assert result.eligible_for_pilot is False


def test_screening_requires_every_governed_dimension_exactly_once() -> None:
    """Missing or duplicate dimensions cannot be promoted to screened evidence."""
    findings = _all_findings()
    with pytest.raises(AssessmentSpecError, match="screening_dimensions_incomplete"):
        _result(findings=findings[:-1])

    with pytest.raises(AssessmentSpecError, match="duplicate_screening_dimension"):
        _result(findings=findings + (findings[0],))


def test_each_finding_requires_evidence_and_limitation_provenance() -> None:
    """Findings need evidence, and accepted limitations need an explicit code."""
    with pytest.raises(AssessmentSpecError, match="screening_evidence_required"):
        build_item_screening_finding(
            dimension=ScreeningDimension.AMBIGUITY_RISK,
            status=ScreeningStatus.PASS,
            reason_code="ambiguity_review",
            evidence_fingerprints=(),
            metadata={},
        )

    with pytest.raises(AssessmentSpecError, match="limitation_code_required"):
        build_item_screening_finding(
            dimension=ScreeningDimension.AMBIGUITY_RISK,
            status=ScreeningStatus.ACCEPTED_WITH_LIMITATION,
            reason_code="ambiguity_review",
            evidence_fingerprints=(_digest("a"),),
            metadata={},
        )


def test_limitation_code_is_rejected_for_non_limitation_status() -> None:
    """Limitation provenance cannot be attached to a passing or failing finding."""
    with pytest.raises(AssessmentSpecError, match="unexpected_limitation_code"):
        build_item_screening_finding(
            dimension=ScreeningDimension.AMBIGUITY_RISK,
            status=ScreeningStatus.PASS,
            reason_code="ambiguity_review",
            evidence_fingerprints=(_digest("a"),),
            limitation_code="accepted_bounded_limitation",
            metadata={},
        )


def test_result_binds_exact_item_and_governance_provenance() -> None:
    """Changing any exact upstream fingerprint changes the screening identity."""
    baseline = _result()
    changed = _result(item_content_fingerprint=_digest("1"))

    assert baseline.result_fingerprint != changed.result_fingerprint
    assert baseline.to_dict()["screening_policy_fingerprint"] == _digest("f")


def test_sensitive_raw_content_is_rejected_from_screening_metadata() -> None:
    """Screening metadata inherits the package's raw-content rejection boundary."""
    with pytest.raises(AssessmentSpecError, match="sensitive_metadata_field"):
        _result(metadata={"source_text": "must not persist"})
