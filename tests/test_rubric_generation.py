"""Behavioral contract for governed rubric item generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fast_mlsirm.rubric import (
    BlueprintPlan,
    CandidateValidationError,
    DifficultyBand,
    EvidenceMode,
    GeneratedItemCandidate,
    GenerationProviderError,
    ItemGenerationProvider,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    SourceDocument,
    StaticFixtureProvider,
    build_generation_request,
    compile_item_blueprints,
    execute_generation,
    parse_generated_item_candidate,
)


def _rubric(response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING):
    """Return a reusable three-level groundedness rubric."""
    return RubricSpecification(
        rubric_id="faithfulness_rubric",
        construct_id="evidence_grounding",
        construct_definition="Degree to which substantive claims are supported.",
        response_format=response_format,
        levels=(
            RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
            RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
            RubricLevel(2, "full_support", "Full support.", ("complete support",)),
        ),
        task_families=("claim_verification",),
        evidence_requirements=("Quote the supporting source span.",),
        prohibited_patterns=("Do not invent support.",),
        locale="en-US",
        rubric_version="1.2.3",
    )


def _blueprint(
    response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING,
    evidence_mode: EvidenceMode = EvidenceMode.SINGLE_SOURCE,
):
    """Compile one blueprint for the requested response and evidence modes."""
    rubric = _rubric(response_format)
    blueprint = compile_item_blueprints(
        rubric,
        BlueprintPlan(
            difficulty_bands=(DifficultyBand.MEDIUM,),
            evidence_modes=(evidence_mode,),
            items_per_cell=1,
            seed=7,
        ),
    )[0]
    return rubric, blueprint


def _source(
    source_id: str = "policy_source",
    content: str = "The policy requires every substantive claim to cite evidence.",
):
    """Return one valid source document."""
    return SourceDocument(source_id, content, "text/plain", "en-US")


def _sources_for(mode: EvidenceMode):
    """Return a valid source tuple for one evidence mode."""
    if mode is EvidenceMode.CLOSED_BOOK:
        return ()
    if mode in {EvidenceMode.MULTI_SOURCE, EvidenceMode.ADVERSARIAL_CONTEXT}:
        return (
            _source(),
            _source("secondary_source", "A second source supplies corroboration."),
        )
    return (_source(),)


def _request(
    response_format: ResponseFormat = ResponseFormat.ORDINAL_RATING,
    evidence_mode: EvidenceMode = EvidenceMode.SINGLE_SOURCE,
):
    """Return a valid content-addressed generation request."""
    rubric, blueprint = _blueprint(response_format, evidence_mode)
    return build_generation_request(rubric, blueprint, _sources_for(evidence_mode))


def _answer_key(response_format: ResponseFormat) -> dict:
    """Return one valid typed answer key for the declared response format."""
    if response_format is ResponseFormat.CONSTRUCTED_RESPONSE:
        return {
            "reference_response": "A supported answer cites the supplied policy.",
            "accepted_variants": ["Cite the supplied policy."],
            "rationale": "The response must use supplied evidence.",
        }
    if response_format is ResponseFormat.SELECTED_RESPONSE:
        return {
            "option_ids": ["option_alpha"],
            "rationale": "Only option alpha is source-supported.",
        }
    if response_format is ResponseFormat.BINARY_JUDGMENT:
        return {
            "value": True,
            "rationale": "The cited statement occurs in the source.",
        }
    if response_format is ResponseFormat.ORDINAL_RATING:
        return {
            "score": 2,
            "rationale": "All substantive claims are supported.",
        }
    return {
        "outcome": "left_option",
        "preferred_option_id": "response_alpha",
        "rationale": "Response A is better grounded.",
    }


def _options(response_format: ResponseFormat) -> list[dict[str, str]]:
    """Return valid options for selected and pairwise response formats."""
    if response_format is ResponseFormat.SELECTED_RESPONSE:
        return [
            {"option_id": "option_alpha", "text": "Supported"},
            {"option_id": "option_beta", "text": "Unsupported"},
        ]
    if response_format is ResponseFormat.PAIRWISE_COMPARISON:
        return [
            {"option_id": "response_alpha", "text": "Response A"},
            {"option_id": "response_beta", "text": "Response B"},
        ]
    return []


def _candidate_payload(request, *, closed_book: bool = False) -> dict:
    """Return a contract-ordered candidate payload for one request."""
    response_format = request.blueprint.response_format
    contract = request.contract
    blueprint = contract["blueprint"]
    rubric = contract["rubric"]
    attributions = []
    if not closed_book:
        attributions = [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ]
        if request.blueprint.evidence_mode is EvidenceMode.MULTI_SOURCE:
            attributions.append(
                {
                    "source_id": "secondary_source",
                    "evidence_span": "supplies corroboration",
                }
            )
    return {
        "blueprint_id": request.blueprint.blueprint_id,
        "blueprint_handle": blueprint["blueprint_handle"],
        "blueprint_fingerprint": request.blueprint.blueprint_fingerprint,
        "rubric_id": request.blueprint.rubric_id,
        "rubric_version": request.blueprint.rubric_version,
        "rubric_fingerprint": rubric["fingerprint"],
        "item_id": "generated_item_001",
        "stem": "Judge whether the response is supported by the source.",
        "stimulus": ["The response states that claims require evidence."],
        "response_format": response_format.value,
        "options": _options(response_format),
        "answer_key": _answer_key(response_format),
        "scoring_guide": [
            {"score": 0, "evidence": "No claims supported.", "rationale": "No support."},
            {"score": 1, "evidence": "Some claims supported.", "rationale": "Partial support."},
            {"score": 2, "evidence": "All claims supported.", "rationale": "Full support."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported claim"]},
            {"score": 1, "observable_indicators": ["mixed support"]},
            {"score": 2, "observable_indicators": ["complete support"]},
        ],
        "source_attributions": attributions,
        "safety_notes": [],
    }


def _candidate_json(request, *, closed_book: bool = False) -> str:
    """Serialize one valid candidate payload."""
    return json.dumps(
        _candidate_payload(request, closed_book=closed_book),
        ensure_ascii=False,
    )


def test_source_document_is_content_addressed_and_redacted():
    """Source audit metadata contains a digest and count but no source text."""
    first = _source()
    second = _source()
    assert first == second
    assert first.content_digest == second.content_digest
    assert len(first.content_digest) == 64
    metadata = first.to_metadata_dict()
    assert metadata["content_digest"] == first.content_digest
    assert metadata["character_count"] == len(first.content)
    assert "content" not in metadata
    assert first.content not in json.dumps(metadata)
    assert first.to_provider_dict()["content"] == first.content


def test_request_is_deterministic_content_sensitive_and_fully_addressed():
    """Request provenance binds exact contract and source content without disclosure."""
    first = _request()
    second = _request()
    assert first == second
    assert first.request_id == second.request_id
    assert first.request_fingerprint == second.request_fingerprint
    assert len(first.request_fingerprint) == 64
    assert first.request_handle.endswith(first.request_fingerprint[:32])
    assert first.contract_fingerprint == first.contract["contract_fingerprint"]
    metadata = first.to_metadata_dict()
    assert metadata["request_fingerprint"] == first.request_fingerprint
    assert metadata["contract_fingerprint"] == first.contract_fingerprint
    assert _source().content not in json.dumps(metadata)
    assert '"content":' not in json.dumps(metadata)
    assert first.to_provider_dict()["sources"][0]["content"] == _source().content

    rubric, blueprint = _blueprint()
    changed = build_generation_request(
        rubric,
        blueprint,
        (_source(content="The changed policy requires two citations."),),
    )
    assert changed.request_id != first.request_id
    assert changed.request_fingerprint != first.request_fingerprint
    assert changed.sources[0].content_digest != first.sources[0].content_digest


@pytest.mark.parametrize("mode", tuple(EvidenceMode))
def test_every_evidence_mode_accepts_its_declared_source_cardinality(mode):
    """Each evidence mode compiles into a valid request with its source policy."""
    rubric, blueprint = _blueprint(evidence_mode=mode)
    request = build_generation_request(rubric, blueprint, _sources_for(mode))
    assert request.blueprint.evidence_mode is mode


def test_fixture_provider_executes_once_and_returns_redacted_provenance():
    """The offline provider crosses the protocol boundary exactly once."""
    request = _request()
    provider = StaticFixtureProvider(
        provider_id="fixture_provider",
        model_id="fixture_model",
        response_text=_candidate_json(request),
    )
    assert isinstance(provider, ItemGenerationProvider)
    execution = execute_generation(provider, request)
    assert provider.call_count == 1
    assert execution.provider_id == "fixture_provider"
    assert execution.model_id == "fixture_model"
    assert execution.request_id == request.request_id
    assert execution.contract_id == request.contract_id
    assert isinstance(execution.candidate, GeneratedItemCandidate)
    assert execution.candidate.request_fingerprint == request.request_fingerprint
    assert len(execution.raw_response_digest) == 64
    assert len(execution.execution_fingerprint) == 64
    assert execution.execution_handle.endswith(execution.execution_fingerprint[:32])
    serialized = execution.to_dict()
    assert "response" not in serialized
    assert _source().content not in json.dumps(serialized)


@pytest.mark.parametrize("response_format", tuple(ResponseFormat))
def test_every_response_format_has_one_valid_candidate(response_format):
    """All response formats cross the same strict candidate boundary."""
    request = _request(response_format)
    raw = _candidate_json(request)
    first = parse_generated_item_candidate(raw, request)
    second = parse_generated_item_candidate(raw, request)
    assert first == second
    assert first.response_format is response_format
    assert first.candidate_fingerprint == second.candidate_fingerprint
    assert first.to_dict()["candidate_fingerprint"] == first.candidate_fingerprint
    assert first.blueprint_fingerprint == request.blueprint.blueprint_fingerprint
    assert first.rubric_fingerprint == request.blueprint.rubric_fingerprint
    assert first.contract_fingerprint == request.contract_fingerprint
    assert [entry.score for entry in first.scoring_guide] == [0, 1, 2]
    assert [entry.score for entry in first.rubric_alignment] == [0, 1, 2]


def test_closed_book_candidate_has_no_source_attribution():
    """Closed-book authoring accepts a candidate with no source provenance claim."""
    request = _request(
        ResponseFormat.CONSTRUCTED_RESPONSE,
        EvidenceMode.CLOSED_BOOK,
    )
    candidate = parse_generated_item_candidate(
        _candidate_json(request, closed_book=True),
        request,
    )
    assert candidate.source_attributions == ()


def test_duplicate_keys_and_nonfinite_numbers_are_rejected_without_values():
    """Unsafe JSON interoperability cases fail with stable redacted codes."""
    request = _request()
    payload = _candidate_payload(request)
    duplicate = json.dumps(payload)[:-1] + ',"item_id":"other_item"}'
    with pytest.raises(CandidateValidationError) as duplicate_error:
        parse_generated_item_candidate(duplicate, request)
    assert duplicate_error.value.code == "duplicate_json_key"
    assert "other_item" not in str(duplicate_error.value)

    nonfinite = json.dumps(payload).replace('"score": 2', '"score": NaN', 1)
    with pytest.raises(CandidateValidationError) as nonfinite_error:
        parse_generated_item_candidate(nonfinite, request)
    assert nonfinite_error.value.code == "nonfinite_json_number"
    assert "NaN" not in str(nonfinite_error.value)


def test_provider_failures_and_non_string_output_are_redacted():
    """Provider diagnostics and invalid values cannot escape the trust boundary."""
    request = _request()

    class FailingProvider:
        provider_id = "failing_provider"
        model_id = "failing_model"

        def generate(self, request):
            raise RuntimeError("secret provider diagnostic and source text")

    with pytest.raises(GenerationProviderError) as provider_error:
        execute_generation(FailingProvider(), request)
    assert provider_error.value.code == "provider_failure"
    assert "secret" not in str(provider_error.value)
    assert _source().content not in str(provider_error.value)

    class NonStringProvider:
        provider_id = "invalid_provider"
        model_id = "invalid_model"

        def generate(self, request):
            return {"not": "JSON text"}

    with pytest.raises(GenerationProviderError, match="JSON text"):
        execute_generation(NonStringProvider(), request)


def test_generation_guide_exists():
    """The buyer-facing generation trust-boundary guide is retained."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "docs" / "rubric_generation_validation.md").is_file()
