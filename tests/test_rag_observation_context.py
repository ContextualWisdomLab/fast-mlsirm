"""Fail-first contracts for reference-free RAG observation provenance."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.rag import (
    CandidateVisibility,
    EvidenceRegime,
    RagObservationContext,
    build_rag_observation_context,
)

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("scoring_execution_fixtures.py"))
)
automated_engine = _FIXTURES["automated_engine"]
criterion_request = _FIXTURES["criterion_request"]


def _context(**overrides):
    request = overrides.pop("request", criterion_request())
    engine = overrides.pop("judge_engine", automated_engine())
    values = {
        "request": request,
        "judge_engine": engine,
        "system_run_id": "rag_system_run_001",
        "system_configuration_fingerprint": "c" * 64,
        "retrieval_run_fingerprint": "d" * 64,
        "evidence_regime": EvidenceRegime.RETRIEVED_CONTEXT,
        "query_testlet_id": "rag_query_testlet_001",
        "candidate_visibility": CandidateVisibility.BLIND,
    }
    values.update(overrides)
    return build_rag_observation_context(**values)


def test_rag_context_reuses_scoring_and_engine_identities_without_raw_text() -> None:
    """RAG provenance binds existing scoring contracts instead of duplicating them."""
    request = criterion_request()
    engine = automated_engine()

    context = _context(request=request, judge_engine=engine)

    assert context.request_fingerprint == request.request_fingerprint
    assert context.rag_query_id == request.task_id
    assert context.query_revision_fingerprint == request.task_revision_fingerprint
    assert context.response_content_fingerprint == request.response_content_fingerprint
    assert context.occasion_id == request.occasion_id
    assert context.judge_engine_fingerprint == engine.engine_fingerprint
    assert context.judge_prompt_fingerprint == engine.prompt_template_fingerprint
    assert context.evidence_regime is EvidenceRegime.RETRIEVED_CONTEXT
    assert context.candidate_visibility is CandidateVisibility.BLIND
    assert len(context.context_fingerprint) == 64

    serialized = context.to_dict()
    assert serialized["context_fingerprint"] == context.context_fingerprint
    forbidden = {"question_text", "context_text", "response_text", "answer_text"}
    assert forbidden.isdisjoint(serialized)


def test_system_configuration_and_stochastic_run_remain_distinct_identities() -> None:
    """Repeated runs of one configuration must not collapse into one respondent."""
    first = _context(system_run_id="rag_system_run_001")
    second = _context(system_run_id="rag_system_run_002")

    assert first.system_configuration_fingerprint == second.system_configuration_fingerprint
    assert first.system_run_id != second.system_run_id
    assert first.context_fingerprint != second.context_fingerprint


def test_evidence_regime_and_candidate_visibility_are_identity_bearing() -> None:
    """Evidence authority and candidate exposure must survive into exact provenance."""
    blind = _context()
    visible = _context(candidate_visibility=CandidateVisibility.CANDIDATE_VISIBLE)
    authoritative = _context(evidence_regime=EvidenceRegime.AUTHORITATIVE_CORPUS)

    assert visible.context_fingerprint != blind.context_fingerprint
    assert authoritative.context_fingerprint != blind.context_fingerprint


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"system_run_id": "bad"}, "invalid_system_run_id"),
        (
            {"system_configuration_fingerprint": "not-a-digest"},
            "invalid_system_configuration_fingerprint",
        ),
        ({"retrieval_run_fingerprint": "not-a-digest"}, "invalid_retrieval_run_fingerprint"),
        ({"evidence_regime": "unknown"}, "invalid_evidence_regime"),
        ({"candidate_visibility": "unknown"}, "invalid_candidate_visibility"),
    ],
)
def test_rag_context_fails_closed_on_malformed_provenance(overrides, code) -> None:
    """Malformed adapter provenance must fail before it reaches calibration."""
    with pytest.raises(AssessmentSpecError) as captured:
        _context(**overrides)
    assert captured.value.code == code


def test_rag_context_is_factory_sealed() -> None:
    """Callers cannot relabel an unverified set of scoring identities."""
    request = criterion_request()
    engine = automated_engine()
    with pytest.raises(AssessmentSpecError, match="unverified_rag_observation_context"):
        RagObservationContext(
            request_fingerprint=request.request_fingerprint,
            rag_query_id=request.task_id,
            query_revision_fingerprint=request.task_revision_fingerprint,
            system_run_id="rag_system_run_001",
            system_configuration_fingerprint="c" * 64,
            retrieval_run_fingerprint="d" * 64,
            response_content_fingerprint=request.response_content_fingerprint,
            evidence_regime=EvidenceRegime.RETRIEVED_CONTEXT,
            query_testlet_id="rag_query_testlet_001",
            judge_engine_fingerprint=engine.engine_fingerprint,
            judge_prompt_fingerprint=engine.prompt_template_fingerprint,
            occasion_id=request.occasion_id,
            candidate_visibility=CandidateVisibility.BLIND,
        )
