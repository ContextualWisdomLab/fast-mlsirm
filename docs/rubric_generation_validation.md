# Governed Item Generation and Candidate Validation

`fast_mlsirm.rubric` carries an approved rubric blueprint across an untrusted provider boundary without treating raw provider JSON as an assessment item. The generation layer binds the exact rubric contract to a bounded source packet, invokes one provider through a minimal protocol, and constructs a candidate only after provenance, structure, and source-grounding checks pass.

## Product boundary

```text
RubricSpecification + ItemBlueprint + SourceDocument[]
                         |
                         v
                GenerationRequest
                         |
                         v
              ItemGenerationProvider
                         |
                         v
                 untrusted JSON
                         |
                         v
              GeneratedItemCandidate
              + GenerationExecution
```

This layer validates authoring artifacts and audit provenance. It does **not** estimate item difficulty, discrimination, information, fit, DIF, evaluator severity, or latent positions. Those numerical operations remain in the Rust-backed psychometric layer. Structural acceptance is not evidence of content validity or operational scoreability.

The core package also does not provide hosted-model SDKs, credential discovery, URL retrieval, billing, or asynchronous transport. Optional adapters may implement those concerns, but they must return one JSON text through `ItemGenerationProvider` and cannot bypass the core parser.

## Offline example

```python
import json

from fast_mlsirm.rubric import (
    BlueprintPlan,
    DifficultyBand,
    EvidenceMode,
    ResponseFormat,
    RubricLevel,
    RubricSpecification,
    SourceDocument,
    StaticFixtureProvider,
    build_generation_request,
    compile_item_blueprints,
    execute_generation,
)

rubric = RubricSpecification(
    rubric_id="faithfulness_rubric",
    construct_id="evidence_grounding",
    construct_definition=(
        "Degree to which substantive claims are supported by supplied evidence."
    ),
    response_format=ResponseFormat.ORDINAL_RATING,
    levels=(
        RubricLevel(0, "unsupported", "No support.", ("unsupported claim",)),
        RubricLevel(1, "partial_support", "Partial support.", ("mixed support",)),
        RubricLevel(2, "full_support", "Full support.", ("complete support",)),
    ),
    task_families=("claim_verification",),
    evidence_requirements=("Quote the supporting source span.",),
    prohibited_patterns=("Do not invent source support.",),
    locale="en-US",
    rubric_version="1.0.0",
)

blueprint = compile_item_blueprints(
    rubric,
    BlueprintPlan(
        difficulty_bands=(DifficultyBand.MEDIUM,),
        evidence_modes=(EvidenceMode.SINGLE_SOURCE,),
        items_per_cell=1,
        seed=20260803,
    ),
)[0]
source = SourceDocument(
    source_id="policy_source",
    content="The policy requires every substantive claim to cite evidence.",
    locale="en-US",
)
request = build_generation_request(rubric, blueprint, (source,))

provenance = {
    "blueprint_id": blueprint.blueprint_id,
    "blueprint_handle": request.contract["blueprint"]["blueprint_handle"],
    "blueprint_fingerprint": blueprint.blueprint_fingerprint,
    "rubric_id": blueprint.rubric_id,
    "rubric_version": blueprint.rubric_version,
    "rubric_fingerprint": blueprint.rubric_fingerprint,
}
fixture_response = json.dumps(
    {
        **provenance,
        "item_id": "generated_item_001",
        "stem": "Judge whether the response is supported by the source.",
        "stimulus": ["The response says that claims require evidence."],
        "response_format": "ordinal_rating",
        "options": [],
        "answer_key": {
            "score": 2,
            "rationale": "Every substantive claim is supported.",
        },
        "scoring_guide": [
            {"score": 0, "evidence": "No support.", "rationale": "Unsupported."},
            {"score": 1, "evidence": "Some support.", "rationale": "Partial."},
            {"score": 2, "evidence": "Full support.", "rationale": "Complete."},
        ],
        "rubric_alignment": [
            {"score": 0, "observable_indicators": ["unsupported claim"]},
            {"score": 1, "observable_indicators": ["mixed support"]},
            {"score": 2, "observable_indicators": ["complete support"]},
        ],
        "source_attributions": [
            {
                "source_id": "policy_source",
                "evidence_span": "requires every substantive claim to cite evidence",
            }
        ],
        "safety_notes": [],
    },
    ensure_ascii=False,
)
provider = StaticFixtureProvider(
    provider_id="fixture_provider",
    model_id="fixture_model",
    response_text=fixture_response,
)
execution = execute_generation(provider, request)
print(execution.candidate.candidate_fingerprint)
```

`StaticFixtureProvider` is an offline integration fixture. It proves request, parser, and provenance behavior; it is not evidence of model or item quality.

## Request and source contract

A `SourceDocument` preserves exact text for provider input and verbatim evidence-span checks. Its redacted metadata contains only source identity, SHA-256 digest, character count, media type, locale, and schema version. Source content appears only in the explicit provider payload and is omitted from request metadata and execution results.

| Input | Limit |
|---|---:|
| One source | 262,144 characters |
| Sources per request | 32 |
| Aggregate source packet | 1,048,576 characters |
| Raw provider JSON | 262,144 characters |
| JSON nesting depth | 32 |
| JSON values | 20,000 |
| Candidate text field | 8,192 characters |
| General candidate collection | 32 values |

Evidence-mode source cardinality is checked before invocation: `closed_book` requires zero sources, `single_source` exactly one, `multi_source` and `adversarial_context` at least two, and `unanswerable` at least one.

## Immutable replay protection

Every candidate must echo the exact constants from its request:

- `blueprint_id`;
- `blueprint_handle`;
- `blueprint_fingerprint`;
- `rubric_id`;
- `rubric_version`; and
- `rubric_fingerprint`.

A mismatch returns the redacted `provenance_mismatch` code before candidate construction. These fields are included in `candidate_fingerprint`, so a candidate cannot be moved to another rubric revision or blueprint without changing its durable audit identity.

## JSON safety

The parser is stricter than Python's default decoder:

- duplicate object member names are rejected at every nesting level;
- `NaN` and infinities are rejected;
- size, depth, and total-node budgets are enforced;
- the top-level value must be an object;
- every required field must exist;
- undeclared fields are rejected;
- score arrays must cover every score exactly once in declared order; and
- errors include a stable code and path but never the rejected value.

Instruction-like text embedded in rubric, source, stem, or stimulus fields remains inert data. It is never evaluated or dispatched as code.

## Response-format contracts

| Response format | Options | Answer key |
|---|---|---|
| `constructed_response` | none | `reference_response`, bounded `accepted_variants`, `rationale` |
| `selected_response` | at least two unique options | one or more declared `option_ids`, `rationale` |
| `binary_judgment` | none | Boolean `value`, `rationale` |
| `ordinal_rating` | none | allowed rubric `score`, `rationale` |
| `pairwise_comparison` | exactly two unique options | `outcome`, nullable `preferred_option_id`, `rationale` |

For pairwise items, `left_option` must identify the first option, `right_option` the second, and `tie` requires a null preferred option. The parser enforces these relations even when a provider bypasses its advertised JSON Schema.

## Source attribution contract

For source-backed modes, at least one attribution is required. Every source id must occur in the request, every `(source_id, evidence_span)` pair must be unique, and each evidence span must occur verbatim in the referenced source. Closed-book candidates cannot claim source attribution. Exact substring validation does not establish semantic entailment; that remains a later screening task.

## Provider failure boundary

A provider adapter exposes stable `provider_id`, `model_id`, and one synchronous `generate(request) -> str` method. The executor invokes it once. Provider exceptions become `GenerationProviderError` without the original diagnostic because provider messages may contain credentials, source text, or generated content. Non-string output is rejected before parsing.

## Deterministic provenance

- `SourceDocument.content_digest` hashes exact UTF-8 source content.
- `GenerationRequest.request_id` binds the contract, blueprint, seed, and source metadata.
- `GeneratedItemCandidate.candidate_fingerprint` hashes normalized candidate content, including immutable provenance.
- `GenerationExecution.execution_id` binds the request, provider/model ids, candidate fingerprint, and raw-response digest.

No current time, process id, global random state, source text, or raw provider response is embedded in those identifiers.

## Next quality gates

A structurally valid candidate still requires ambiguity and answerability screening, source-support review, distractor analysis, leakage and memorization checks, fairness and language review, human or artificial-crowd pilot responses, Rust-backed calibration and fit diagnostics, DIF/local-dependence/exposure controls, and governed item-bank acceptance.

## References

Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). Internet Engineering Task Force. https://doi.org/10.17487/RFC8259

Bhutton, H., Andrews, H., Wright, A., & Hutton, B. (2022). *JSON Schema: A media type for describing JSON documents* (Draft 2020-12). JSON Schema. https://json-schema.org/draft/2020-12/json-schema-core
