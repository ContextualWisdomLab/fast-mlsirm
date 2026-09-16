"""Public fail-closed audit policy and verified pilot-admission entrypoints."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .audit import (
    CandidateAuditReport,
    PilotAdmissionError,
    audit_generated_item_candidate as _audit_generated_item_candidate,
    build_pilot_candidate_record as _build_pilot_candidate_record,
)
from .candidates import GeneratedItemCandidate
from .verified_pilot import PilotCandidateRecord
from .verified_pilot import _from_verified_core

if TYPE_CHECKING:
    from .semantic_screening import CandidateScreeningResult

AUDIT_POLICY_ID = "generated_item_audit"
AUDIT_POLICY_VERSION = "2.0.0"


def audit_generated_item_candidate(
    candidate: GeneratedItemCandidate,
) -> CandidateAuditReport:
    """Run the exact audit policy implemented by this package version."""
    return _audit_generated_item_candidate(
        candidate,
        audit_policy_id=AUDIT_POLICY_ID,
        audit_policy_version=AUDIT_POLICY_VERSION,
    )


def build_pilot_candidate_record(
    candidate: GeneratedItemCandidate,
    audit_report: CandidateAuditReport,
    *,
    screening_result: CandidateScreeningResult | None = None,
    pilot_study_id: str,
    query_testlet_id: str,
    generator_family_id: str,
    judge_policy_id: str,
    occasion_id: str,
) -> PilotCandidateRecord:
    """Admit a candidate only after replaying and matching the current policy."""
    if not isinstance(candidate, GeneratedItemCandidate):
        raise TypeError("candidate must be a GeneratedItemCandidate")
    if not isinstance(audit_report, CandidateAuditReport):
        raise TypeError("audit_report must be a CandidateAuditReport")

    candidate_fingerprint = candidate.candidate_fingerprint
    if audit_report.candidate_fingerprint != candidate_fingerprint:
        raise PilotAdmissionError(
            "candidate_report_mismatch",
            "$.candidate_fingerprint",
            "audit report does not bind the exact candidate",
        )
    if (
        audit_report.audit_policy_id != AUDIT_POLICY_ID
        or audit_report.audit_policy_version != AUDIT_POLICY_VERSION
    ):
        raise PilotAdmissionError(
            "unsupported_audit_policy",
            "$.audit_report.audit_policy_version",
            "audit report policy is not implemented by this package version",
        )

    expected_report = audit_generated_item_candidate(candidate)
    if audit_report.audit_report_fingerprint != expected_report.audit_report_fingerprint:
        raise PilotAdmissionError(
            "audit_report_unverified",
            "$.audit_report.audit_report_fingerprint",
            "audit report does not match a replay of the current policy",
        )
    if not audit_report.is_pilot_eligible:
        raise PilotAdmissionError(
            "audit_not_clear",
            "$.audit_report",
            "candidate has unresolved blocking or review-required findings",
        )

    if screening_result is None:
        raise PilotAdmissionError(
            "screening_required",
            "$.screening_result",
            "candidate requires a verified semantic screening result",
        )
    from .semantic_screening import CandidateScreeningResult

    if type(screening_result) is not CandidateScreeningResult:
        raise TypeError("screening_result must be a CandidateScreeningResult")
    try:
        screening_content, screening_result_fingerprint = screening_result._verify_seal()
        screening_eligible = screening_result._pilot_eligible_from_content(
            screening_content
        )
    except ValueError:
        raise PilotAdmissionError(
            "screening_result_unverified",
            "$.screening_result",
            "screening result does not match its creation-time identity",
        ) from None
    if screening_content["candidate_fingerprint"] != candidate_fingerprint:
        raise PilotAdmissionError(
            "screening_candidate_mismatch",
            "$.screening_result.candidate_fingerprint",
            "screening result does not bind the exact candidate",
        )
    if (
        screening_content["audit_report_fingerprint"]
        != audit_report.audit_report_fingerprint
    ):
        raise PilotAdmissionError(
            "screening_report_mismatch",
            "$.screening_result.audit_report_fingerprint",
            "screening result does not bind the exact audit report",
        )
    if not screening_eligible:
        raise PilotAdmissionError(
            "screening_not_clear",
            "$.screening_result",
            "candidate has unresolved semantic screening decisions",
        )

    core_record = _build_pilot_candidate_record(
        candidate,
        audit_report,
        screening_result_fingerprint=screening_result_fingerprint,
        pilot_study_id=pilot_study_id,
        query_testlet_id=query_testlet_id,
        generator_family_id=generator_family_id,
        judge_policy_id=judge_policy_id,
        occasion_id=occasion_id,
    )
    return _from_verified_core(core_record)
