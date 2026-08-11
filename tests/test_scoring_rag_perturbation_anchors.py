"""Fail-first contracts for governed reference-free RAG perturbation anchors."""

from __future__ import annotations

import hashlib
import inspect

import pytest

from fast_mlsirm.scoring import AssessmentSpecError
from fast_mlsirm.scoring.rag import (
    RAGPerturbationDirection,
    RAGPerturbationKind,
    build_rag_perturbation_anchor,
)

BASELINE_FP = hashlib.sha256(b"rag-anchor-baseline-request").hexdigest()
PERTURBED_FP = hashlib.sha256(b"rag-anchor-perturbed-request").hexdigest()
SECOND_FP = hashlib.sha256(b"rag-anchor-second-perturbed-request").hexdigest()


@pytest.mark.parametrize(
    ("kind", "construct", "direction"),
    (
        ("unsupported_claim", "grounded_generation", "decrease"),
        ("explicit_contradiction", "grounded_generation", "decrease"),
        ("irrelevant_context", "retrieval_relevance", "decrease"),
        ("required_evidence_removal", "coverage_or_completeness_proxy", "decrease"),
        ("citation_target_swap", "citation_attribution", "decrease"),
        ("semantic_query_paraphrase", "robustness", "invariant"),
        ("style_only_rewrite", "robustness", "invariant"),
        ("unanswerable_query", "answerability_and_abstention", "increase"),
    ),
)
def test_perturbation_kind_has_one_construct_specific_expected_direction(
    kind: str,
    construct: str,
    direction: str,
) -> None:
    """Known perturbations preserve their preregistered construct direction."""
    anchor = build_rag_perturbation_anchor(
        anchor_id=f"{kind}_anchor",
        baseline_request_fingerprint=BASELINE_FP,
        perturbed_request_fingerprint=PERTURBED_FP,
        perturbation_kind=kind,
    )

    assert anchor.perturbation_kind is RAGPerturbationKind(kind)
    assert anchor.expected_construct == construct
    assert anchor.expected_direction is RAGPerturbationDirection(direction)


def test_perturbation_enums_are_finite_and_explicit() -> None:
    """The public contract must not accept arbitrary perturbation semantics."""
    assert {member.value for member in RAGPerturbationKind} == {
        "unsupported_claim",
        "explicit_contradiction",
        "irrelevant_context",
        "required_evidence_removal",
        "citation_target_swap",
        "semantic_query_paraphrase",
        "style_only_rewrite",
        "unanswerable_query",
    }
    assert {member.value for member in RAGPerturbationDirection} == {
        "decrease",
        "invariant",
        "increase",
    }


def test_anchor_is_content_addressed_and_identity_bearing() -> None:
    """The exact baseline/perturbation pair participates in anchor identity."""
    first = build_rag_perturbation_anchor(
        anchor_id="unsupported_claim_anchor",
        baseline_request_fingerprint=BASELINE_FP,
        perturbed_request_fingerprint=PERTURBED_FP,
        perturbation_kind=RAGPerturbationKind.UNSUPPORTED_CLAIM,
    )
    replay = build_rag_perturbation_anchor(
        anchor_id="unsupported_claim_anchor",
        baseline_request_fingerprint=BASELINE_FP,
        perturbed_request_fingerprint=PERTURBED_FP,
        perturbation_kind="unsupported_claim",
    )
    second = build_rag_perturbation_anchor(
        anchor_id="unsupported_claim_anchor",
        baseline_request_fingerprint=BASELINE_FP,
        perturbed_request_fingerprint=SECOND_FP,
        perturbation_kind="unsupported_claim",
    )

    assert first.anchor_fingerprint == replay.anchor_fingerprint
    assert first.anchor_fingerprint != second.anchor_fingerprint
    payload = first.to_dict()
    assert payload["baseline_request_fingerprint"] == BASELINE_FP
    assert payload["perturbed_request_fingerprint"] == PERTURBED_FP
    assert payload["perturbation_kind"] == "unsupported_claim"
    assert payload["expected_construct"] == "grounded_generation"
    assert payload["expected_direction"] == "decrease"


def test_anchor_requires_distinct_baseline_and_perturbed_requests() -> None:
    """A no-op request pair cannot masquerade as perturbation evidence."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="no_op_anchor",
            baseline_request_fingerprint=BASELINE_FP,
            perturbed_request_fingerprint=BASELINE_FP,
            perturbation_kind="style_only_rewrite",
        )
    assert caught.value.code == "identical_rag_perturbation_requests"


def test_anchor_rejects_unknown_semantics_and_malformed_fingerprints() -> None:
    """Unknown perturbations and non-content-addressed inputs fail closed."""
    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="unknown_anchor",
            baseline_request_fingerprint=BASELINE_FP,
            perturbed_request_fingerprint=PERTURBED_FP,
            perturbation_kind="improve_everything",
        )
    assert caught.value.code == "invalid_rag_perturbation_kind"

    with pytest.raises(AssessmentSpecError) as caught:
        build_rag_perturbation_anchor(
            anchor_id="bad_fingerprint_anchor",
            baseline_request_fingerprint="not-a-sha256",
            perturbed_request_fingerprint=PERTURBED_FP,
            perturbation_kind="unsupported_claim",
        )
    assert caught.value.code == "invalid_baseline_request_fingerprint"


def test_anchor_contract_cannot_store_raw_query_context_or_answer_text() -> None:
    """Perturbation evidence is source-free identity metadata, not a content store."""
    parameters = inspect.signature(build_rag_perturbation_anchor).parameters
    assert not set(parameters).intersection(
        {
            "query_text",
            "question_text",
            "context_text",
            "retrieved_text",
            "answer_text",
            "response_text",
            "source_text",
        }
    )
