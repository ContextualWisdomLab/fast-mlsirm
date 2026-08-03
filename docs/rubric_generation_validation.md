# Governed Item Generation and Candidate Validation

`fast_mlsirm.rubric` can carry a compiled rubric blueprint across an untrusted model boundary without treating raw model JSON as an assessment item. The generation layer binds the exact rubric contract to a bounded source packet, invokes a provider through a minimal protocol, and accepts output only after strict structural and source-grounding validation.

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
                   raw JSON
                         |
                         v
              GeneratedItemCandidate
              + GenerationExecution
```

This layer validates authoring artifacts and provenance. It does not estimate item difficulty, discrimination, information, fit, DIF, evaluator severity, or latent positions. Those numerical operations remain in the Rust-backed psychometric layer.

The core package also does not provide hosted-model SDKs, credential discovery, retries, billing, rate limiting, URL retrieval, or asynchronous transport. Those concerns belong in optional provider adapters or isolated services that implement `ItemGenerationProvider`.

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
        RubricLevel(
            0,
            "unsupported",
            "Substantive claims are unsupported.",
            ("unsupported substantive claim",),
        ),
        RubricLevel(
            1,
            "partial_support",
            "Some substantive claims are supported.",
            ("mixed support",),
        ),
        RubricLevel(
            2,
            "full_support",
            "Every substantive claim is supported.",
            ("complete source support",),
        ),
    ),
    task_families=("claim_verification",),
    evidence_requirements=("Quote the supporting source span.",),
    prohibited_patterns=("Do not invent source support.",),
    locale="en-US",
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
    media_type="text/plain",
    locale="en-US",
)

request = build_generation_request(rubric, blueprint, (source,))

fixture_response = json.dumps(
    {
        "item_id": "generated_item_001",
        "stem": "Judge whether the response is supported by the source.",
        "stimulus": ["The response says that claims require evidence."],
        "response_format": "ordinal_rating",
        "options": [],
        "answer_key": 2,
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
                "evidence_span": (
                    "requires every substantive claim to cite evidence"
                ),
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

print(request.request_id)
print(execution.execution_id)
print(execution.candidate.candidate_fingerprint)
```

`StaticFixtureProvider` is a deterministic integration fixture. It proves request, parsing, and provenance behavior; it is not evidence of item quality and must not be represented as a production generator.

## Source packet contract

A `SourceDocument` preserves exact source text for provider input and verbatim evidence-span checks. Its audit representation contains only:

- source id;
- SHA-256 content digest;
- character count;
- media type;
- locale; and
- schema version.

The source content itself is omitted from request metadata and execution results. A provider payload includes it only under an explicit `untrusted_source_data` boundary.

### Limits

| Input | Limit |
|---|---:|
| One source | 262,144 characters |
| Sources per request | 32 |
| Aggregate source packet | 1,048,576 characters |
| Raw provider JSON | 262,144 characters |
| Candidate text field | 8,192 characters |
| General candidate collection | 32 values |
| Options | 16 values |

### Evidence-mode cardinality

| Evidence mode | Source requirement |
|---|---:|
| `closed_book` | exactly 0 |
| `single_source` | exactly 1 |
| `multi_source` | at least 2 |
| `adversarial_context` | at least 2 |
| `unanswerable` | at least 1 |

Invalid source cardinality is rejected before the provider can be invoked.

## Provider protocol

A provider adapter needs only stable provenance and one synchronous method:

```python
from typing import Protocol

class ItemGenerationProvider(Protocol):
    provider_id: str
    model_id: str

    def generate(self, request):
        ...
```

`provider_id` and `model_id` use two-or-more-token lower `snake_case`. The executor calls the provider once. Provider exceptions are converted into `GenerationProviderError` without including the provider's original message, because that message may contain source or generated content.

Production adapters should live outside the core authoring package and should implement their own transport-level timeout, retry, authentication, quota, and observability policies. They must still return one JSON text to the core boundary.

## JSON safety

Candidate parsing is stricter than Python's default JSON behavior:

- output above the byte-budget proxy is rejected before decoding;
- duplicate object member names are rejected at every nesting level;
- `NaN`, positive infinity, and negative infinity are rejected;
- the top-level value must be an object;
- every required field must exist;
- undeclared fields are rejected; and
- errors contain a reason code and field path but never the rejected value.

RFC 8259 describes duplicate member names as an interoperability risk because receiver behavior is unpredictable when names are not unique. The parser therefore never accepts last-value-wins behavior.

## Rubric integrity

The candidate must contain exactly one `scoring_guide` entry and one `rubric_alignment` entry for every score in the compiled blueprint. Duplicate scores, omitted scores, and additional scores are rejected. Entries are normalized into ascending score order before the candidate fingerprint is computed.

This structural check does not establish validity. It ensures only that every rubric category has an explicit observable authoring claim available for later screening and calibration.

## Source attribution integrity

For source-backed evidence modes:

1. at least one source attribution is required;
2. every source id must occur in the request;
3. duplicate `(source_id, evidence_span)` pairs are rejected; and
4. every evidence span must occur verbatim in the referenced source.

A closed-book candidate must not claim source attribution. Exact substring validation intentionally does not infer semantic entailment; semantic support is a later screening and evaluator task.

## Response-format rules

| Response format | Structural rule |
|---|---|
| `constructed_response` | no options; text or null answer key |
| `selected_response` | at least two unique options; answer key names one option |
| `binary_judgment` | exactly two unique options; answer key names one option |
| `ordinal_rating` | no options; answer key is an allowed rubric score or null |
| `pairwise_comparison` | exactly two unique options; answer key names one option or `tie` |

These are authoring contracts rather than scoring algorithms.

## Deterministic provenance

- `SourceDocument.content_digest` hashes exact UTF-8 source content.
- `GenerationRequest.request_id` binds the contract, blueprint, generation seed, and source metadata.
- `GeneratedItemCandidate.candidate_fingerprint` hashes canonical normalized candidate content.
- `GenerationExecution.execution_id` binds the request, provider/model ids, candidate fingerprint, and raw-response digest.

No current time, process id, global random state, class name, source text, or raw response is embedded in these identifiers.

## Expected validation failures

```python
from fast_mlsirm.rubric import CandidateValidationError

try:
    execute_generation(provider, request)
except CandidateValidationError as error:
    print(error.code)  # e.g. duplicate_json_key
    print(error.path)  # e.g. $
```

The error does not echo the duplicate name or its value. Stable codes include:

- `invalid_json`;
- `duplicate_json_key`;
- `nonfinite_json_number`;
- `missing_field`;
- `unknown_field`;
- `response_format_mismatch`;
- `duplicate_score`;
- `score_coverage`;
- `unknown_source`;
- `evidence_span_not_found`; and
- `invalid_answer_key`.

## Next quality gates

A structurally valid candidate still requires:

1. ambiguity and answerability screening;
2. source-support and distractor-quality screening;
3. leakage and memorization checks;
4. fairness and language review;
5. artificial-crowd and/or human pilot responses;
6. Rust-backed calibration and fit diagnostics;
7. DIF, local-dependence, exposure, and drift controls; and
8. governed item-bank acceptance.

A hosted provider adapter and semantic screening should remain separable modules so the core package can be embedded in an MSA or used locally without a network dependency.

## References

Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). Internet Engineering Task Force. https://doi.org/10.17487/RFC8259

Bhutton, H., Andrews, H., Wright, A., & Hutton, B. (2022). *JSON Schema: A media type for describing JSON documents* (Draft 2020-12). JSON Schema. https://json-schema.org/draft/2020-12/json-schema-core

Bhutton, H., Andrews, H., Wright, A., & Hutton, B. (2022). *JSON Schema validation: A vocabulary for structural validation of JSON* (Draft 2020-12). JSON Schema. https://json-schema.org/draft/2020-12/json-schema-validation
