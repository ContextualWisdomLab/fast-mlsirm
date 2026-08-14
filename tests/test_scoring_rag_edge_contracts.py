"""Adversarial branch coverage for governed RAG request and anchor contracts."""

from __future__ import annotations

from pathlib import Path
import runpy

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.rag import (
    RAGPerturbationAnchor,
    RAGPerturbationKind,
    build_rag_perturbation_anchor,
)
import fast_mlsirm.scoring.rag as rag


_BASE = runpy.run_path(
    str(Path(__file__).with_name("test_scoring_rag_perturbation_relationships.py"))
)
_request = _BASE["_request"]
_anchor = _BASE["_anchor"]


def _error(action):
    """Return one structured RAG contract error."""
    with pytest.raises(AssessmentSpecError) as captured:
        action()
    return captured.value


def test_anchor_and_canonical_request_replay_reject_identity_mismatches():
    """Anchor construction validates both the typed request and its query revision."""
    fingerprint = "a" * 64
    assert _error(
        lambda: RAGPerturbationAnchor(
            anchor_id="same_request_anchor",
            baseline_request_fingerprint=fingerprint,
            perturbed_request_fingerprint=fingerprint,
            perturbation_specification_fingerprint=fingerprint,
            perturbation_run_fingerprint=fingerprint,
            perturbation_kind=RAGPerturbationKind.UNSUPPORTED_CLAIM,
            _anchor_token=rag._RAG_PERTURBATION_ANCHOR_TOKEN,
        )
    ).code == "identical_rag_perturbation_requests"
    assert _error(lambda: rag._canonical_rag_request(object(), "request")).code == "invalid_request"

    request = _request()
    metadata = dict(request.to_dict()["metadata"])
    metadata["rag_query_revision_fingerprint"] = "b" * 64
    object.__setattr__(request, "metadata", metadata)
    assert _error(lambda: rag._canonical_rag_request(request, "request")).code == "invalid_request"


def test_relationship_validation_covers_request_id_response_id_and_artifact_guards():
    """Each governed perturbation axis rejects an independently changed identity."""
    baseline = _request()
    assert _error(
        lambda: _anchor(
            kind="unsupported_claim",
            baseline=baseline,
            perturbed=_request(
                request_id="different_system_request",
                system_configuration_id="other_stack",
                system_configuration_fingerprint="c" * 64,
                query_id="other_query",
                response_id="generated_response_002",
                response_content_fingerprint="b" * 64,
            ),
        )
    ).code == "unrelated_rag_perturbation_requests"
    assert _error(
        lambda: _anchor(
            kind="unsupported_claim",
            baseline=baseline,
            perturbed=baseline,
        )
    ).code == "identical_rag_perturbation_requests"
    assert _error(
        lambda: _anchor(
            kind="unsupported_claim",
            baseline=baseline,
            perturbed=_request(
                request_id=baseline.request_id,
                response_id="generated_response_002",
                response_content_fingerprint="b" * 64,
            ),
        )
    ).code == "invalid_rag_perturbation_relationship"
    assert _error(
        lambda: _anchor(
            kind="unsupported_claim",
            baseline=baseline,
            perturbed=_request(
                request_id="different_request",
                response_content_fingerprint="b" * 64,
            ),
        )
    ).code == "invalid_rag_perturbation_relationship"
    assert _error(
        lambda: _anchor(
            kind="irrelevant_context",
            baseline=baseline,
            perturbed=_request(
                request_id="different_retrieval_request",
                retrieval_run_fingerprint="b" * 64,
                response_id="generated_response_002",
            ),
        )
    ).code == "invalid_rag_perturbation_relationship"


def test_relationship_validation_exposes_unhandled_kind_and_metadata_boundaries(monkeypatch):
    """RAG metadata remains allowlisted and unknown relationship kinds fail closed."""
    baseline = _request()
    perturbed = _request(request_id="unknown_kind_request")
    with pytest.raises(AssertionError, match="unhandled RAG perturbation kind"):
        rag._validate_perturbation_relationship(baseline, perturbed, object())

    assert _error(lambda: _request(metadata=[])).code == "invalid_rag_metadata"
    monkeypatch.setattr(rag, "freeze_metadata", lambda _value: object())
    assert _error(lambda: _request(metadata={})).code == "invalid_rag_metadata"

    monkeypatch.setattr(rag, "freeze_metadata", lambda value: value)
    monkeypatch.setattr(rag, "thaw_json_value", lambda _value: [])
    assert _error(lambda: _request(metadata={})).code == "invalid_rag_metadata"

    monkeypatch.undo()
    assert _request(metadata={}).to_dict()["metadata"]["rag_evidence_regime"]
