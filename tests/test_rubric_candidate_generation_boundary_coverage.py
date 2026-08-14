"""Direct boundary coverage for rubric candidate and generation contracts."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import runpy

import pytest

from fast_mlsirm.rubric import EvidenceMode, ResponseFormat
import fast_mlsirm.rubric.candidates as candidates
import fast_mlsirm.rubric.generation as generation
import fast_mlsirm.rubric.models as rubric_models

_FIXTURES = runpy.run_path(
    str(Path(__file__).with_name("test_rubric_candidate_audit.py"))
)
_request = _FIXTURES["_request"]
_payload = _FIXTURES["_payload"]
_candidate = _FIXTURES["_candidate"]


def _assert_candidate_error(code: str, callback) -> None:
    """Assert one redacted candidate-validation code."""
    with pytest.raises(candidates.CandidateValidationError) as captured:
        callback()
    assert captured.value.code == code


def test_candidate_parser_depth_fingerprint_and_proof_boundaries() -> None:
    """Escaped JSON, nested nodes, fingerprints, and parser seals fail closed."""
    candidates._validate_raw_json_depth(r'["\\"]')
    nested: object = []
    for _ in range(candidates.MAX_JSON_DEPTH + 1):
        nested = [nested]
    _assert_candidate_error(
        "json_too_deep",
        lambda: candidates._validate_json_depth(nested),
    )
    _assert_candidate_error(
        "invalid_fingerprint",
        lambda: candidates._fingerprint("not_a_digest", "$.fingerprint"),
    )
    with pytest.raises(ValueError, match="proof is invalid"):
        candidates._CandidateValidationProof(object(), "a" * 64)
    with pytest.raises(RuntimeError, match="invalid fields"):
        candidates.GeneratedItemCandidate._from_validated()

    candidate = _candidate()
    proof = candidate._validation_proof
    assert proof is not None
    object.__setattr__(proof, "seal", object())
    with pytest.raises(ValueError, match="proof is invalid"):
        candidate.candidate_fingerprint


def test_candidate_parser_alignment_attribution_and_answer_key_boundaries() -> None:
    """Candidate subdocuments reject duplicate scores, cardinality, and key errors."""
    request = _request()
    duplicate_alignment = _payload(request)
    duplicate_alignment["rubric_alignment"][1]["score"] = 0
    _assert_candidate_error(
        "duplicate_score",
        lambda: candidates.parse_generated_item_candidate(
            json.dumps(duplicate_alignment), request
        ),
    )

    incomplete_alignment = _payload(request)
    incomplete_alignment["rubric_alignment"][2]["score"] = 9
    _assert_candidate_error(
        "score_coverage",
        lambda: candidates.parse_generated_item_candidate(
            json.dumps(incomplete_alignment), request
        ),
    )

    second_source = generation.SourceDocument(
        "alternate_source",
        "The policy requires every claim to cite evidence.",
    )
    object.__setattr__(request, "sources", (request.sources[0], second_source))
    _assert_candidate_error(
        "source_cardinality",
        lambda: candidates._parse_attributions(
            [
                {"source_id": "policy_source", "evidence_span": "requires every claim"},
                {"source_id": "alternate_source", "evidence_span": "requires every claim"},
            ],
            request,
        ),
    )

    ordinal_request = _request(ResponseFormat.ORDINAL_RATING)
    ordinal_payload = _payload(ordinal_request)
    ordinal_payload["options"] = [{"option_id": "option_alpha", "text": "extra"}]
    _assert_candidate_error(
        "options_not_allowed",
        lambda: candidates.parse_generated_item_candidate(
            json.dumps(ordinal_payload), ordinal_request
        ),
    )

    pairwise_request = _request(ResponseFormat.PAIRWISE_COMPARISON)
    pairwise_payload = _payload(pairwise_request)
    pairwise_payload["answer_key"]["outcome"] = "unknown_outcome"
    _assert_candidate_error(
        "invalid_answer_key",
        lambda: candidates.parse_generated_item_candidate(
            json.dumps(pairwise_payload), pairwise_request
        ),
    )

    class ChangingOption:
        """Option whose identity changes to exercise the final membership guard."""

        def __init__(self) -> None:
            self._values = iter(("left_option", "missing_option"))

        @property
        def option_id(self) -> str:
            """Return one staged option identity."""
            return next(self._values)

    _assert_candidate_error(
        "invalid_answer_key",
        lambda: candidates._parse_answer_key(
            ResponseFormat.PAIRWISE_COMPARISON,
            (ChangingOption(), ChangingOption()),
            {
                "outcome": "left_option",
                "preferred_option_id": "missing_option",
                "rationale": "The first option is selected.",
            },
            (0, 1, 2),
        ),
    )


def test_candidate_parser_requires_a_generation_request() -> None:
    """The parser refuses to infer a contract from an arbitrary object."""
    with pytest.raises(TypeError, match="GenerationRequest"):
        candidates.parse_generated_item_candidate("{}", object())


def test_bounded_rubric_collection_accepts_an_empty_optional_iterator() -> None:
    """Optional rubric collections may exhaust cleanly at their minimum bound."""
    assert rubric_models._bounded_values(iter(()), "items", minimum=0, maximum=1) == ()
    assert rubric_models._bounded_values(iter(()), "items", minimum=0, maximum=-1) == ()


def _assert_generation_error(callback, message: str) -> None:
    """Assert one redacted generation-contract ValueError message."""
    with pytest.raises(ValueError, match=message):
        callback()


def test_generation_contract_parser_and_identity_boundaries() -> None:
    """Generation contract parsing rejects malformed JSON and forged identities."""
    _assert_generation_error(
        lambda: generation._digest("not_a_digest", "contract_fingerprint"),
        "64-character lower hexadecimal",
    )
    generation._validate_contract_depth('"escaped\\\"quote"')
    _assert_generation_error(
        lambda: generation._validate_contract_depth("[" * 129 + "]" * 129),
        "maximum JSON nesting depth",
    )
    _assert_generation_error(lambda: generation._contract_object(""), "non-empty")
    _assert_generation_error(lambda: generation._contract_object("{"), "valid JSON")
    _assert_generation_error(lambda: generation._contract_object("[]"), "JSON object")
    _assert_generation_error(
        lambda: generation._contract_string({}, "contract_id"),
        "non-empty contract_id",
    )

    request = _request()
    contract = request.contract
    forged_id = dict(contract)
    forged_id["contract_id"] = "alternate_contract"
    _assert_generation_error(
        lambda: generation._validate_contract_identity(forged_id, request.contract_id),
        "contract_id must match",
    )
    forged_handle = dict(contract)
    forged_handle["contract_handle"] = "generation_contract_" + "0" * 32
    _assert_generation_error(
        lambda: generation._validate_contract_identity(forged_handle, request.contract_id),
        "contract_handle must match",
    )
    _assert_generation_error(
        lambda: generation._validate_source_cardinality(EvidenceMode.MULTI_SOURCE, 1),
        "source cardinality",
    )
    generation._validate_source_cardinality(EvidenceMode.UNANSWERABLE, 1)
    assert request.sources[0].to_provider_dict()["trust_boundary"] == "untrusted_source_data"
    assert request.to_provider_dict()["trust_boundary"] == "rubric_and_sources_are_untrusted_data"


def test_generation_request_and_provider_boundaries_fail_closed() -> None:
    """Request construction, fixture providers, and execution outputs remain typed."""
    request = _request()
    _assert_generation_error(
        lambda: replace(request, blueprint=object()),
        "blueprint must be an ItemBlueprint",
    )
    malformed_contract = request.contract
    malformed_contract["blueprint"] = []
    _assert_generation_error(
        lambda: replace(request, contract_json=json.dumps(malformed_contract)),
        "blueprint object",
    )
    with pytest.raises(ValueError, match="response_text must be a string"):
        generation.StaticFixtureProvider("provider_alpha", "model_alpha", object())
    provider = generation.StaticFixtureProvider(
        "provider_alpha",
        "model_alpha",
        "{}",
    )
    with pytest.raises(TypeError, match="GenerationRequest"):
        provider.generate(object())


def _execution_values(request, candidate) -> dict[str, object]:
    """Return one valid execution provenance packet for focused mutations."""
    return {
        "execution_id": "generation_execution_wrong",
        "request_id": request.request_id,
        "contract_id": request.contract_id,
        "provider_id": "provider_alpha",
        "model_id": "model_alpha",
        "candidate": candidate,
        "raw_response_digest": "a" * 64,
        "request_fingerprint": request.request_fingerprint,
        "contract_fingerprint": request.contract_fingerprint,
    }


def test_generation_execution_replays_candidate_and_identity_provenance() -> None:
    """Execution records cannot rebind candidate, request, contract, or identity."""
    request = _request()
    candidate = _candidate(request)
    valid = _execution_values(request, candidate)
    with pytest.raises(ValueError, match="candidate must be"):
        generation.GenerationExecution(**{**valid, "candidate": object()})
    with pytest.raises(ValueError, match="candidate request_id"):
        generation.GenerationExecution(**{**valid, "request_id": "other_request"})
    with pytest.raises(ValueError, match="candidate contract_id"):
        generation.GenerationExecution(**{**valid, "contract_id": "other_contract"})
    with pytest.raises(ValueError, match="candidate request_fingerprint"):
        generation.GenerationExecution(**{**valid, "request_fingerprint": "b" * 64})
    with pytest.raises(ValueError, match="candidate contract_fingerprint"):
        generation.GenerationExecution(**{**valid, "contract_fingerprint": "c" * 64})
    with pytest.raises(ValueError, match="execution_id must match"):
        generation.GenerationExecution(**valid)


def test_execute_generation_rejects_non_text_provider_output() -> None:
    """Provider adapters returning non-text payloads are rejected without leakage."""
    request = _request()

    class NonTextProvider:
        """Minimal runtime-protocol provider returning an invalid payload."""

        provider_id = "provider_alpha"
        model_id = "model_alpha"

        def generate(self, _request):
            """Return a value outside the provider JSON contract."""
            return object()

    with pytest.raises(generation.GenerationProviderError) as captured:
        generation.execute_generation(NonTextProvider(), request)
    assert captured.value.code == "invalid_provider_output"
