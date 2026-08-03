# Governed Item Generation and Candidate Validation Design

## Problem and outcome

The rubric compiler closes the authoring-design gap but stops at a canonical contract. Enterprise buyers still need a governed trust boundary between that contract and an untrusted model response. Without it, provider output can contain duplicate JSON keys, undeclared fields, omitted score levels, fabricated source identifiers, or evidence spans that do not exist in supplied material while still looking superficially valid.

This slice delivers:

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
                 raw JSON response
                         |
                         v
              GeneratedItemCandidate
              + GenerationExecution
```

The package remains provider-neutral and offline-testable. It defines the portable request, provider protocol, deterministic fixture adapter, strict parser, and provenance result. Hosted SDKs, credentials, retries, billing, and network transport stay in optional integration packages or services.

## Standards and trust model

- RFC 8259 states that object member names should be unique because behavior with duplicates is unpredictable across implementations. Candidate parsing therefore rejects duplicate keys rather than accepting Python's last-value behavior.
- JSON Schema Draft 2020-12 supplies the structural vocabulary represented by the existing generation contract. The runtime parser enforces the contract directly without adding a runtime dependency.
- Source documents and provider output are untrusted data, never executable instructions. Errors expose field paths and stable reason codes, not source or model-generated text.
- Psychometric calibration, scoring, item information, DIF, and fit remain Rust-backed. Python validates authoring artifacts and provenance only.

## Public architecture

### `fast_mlsirm.rubric.generation`

#### `SourceDocument`

An immutable source packet member:

- `source_id`: two-or-more-token lower `snake_case`;
- `content`: non-empty text bounded to 262,144 characters;
- `media_type`: allowlisted text media type;
- `locale`: BCP 47-style tag;
- `schema_version`: `1.0`.

`content_digest` is SHA-256 over UTF-8 source content. `to_metadata_dict()` includes the digest and character count but excludes content. `to_provider_dict()` includes content for the isolated provider call.

#### `GenerationRequest`

Created only through `build_generation_request(rubric, blueprint, sources)` after exact rubric/blueprint compatibility validation. It contains:

- canonical generation contract and contract id;
- immutable source tuple;
- source metadata with content digests;
- deterministic `generation_request_*` id;
- blueprint generation seed;
- schema version.

The request id hashes the canonical contract id, blueprint id, source ids/digests/media types/locales, and seed. It never hashes or serializes raw source content into audit logs. The provider payload contains sources under an explicit untrusted-data boundary.

Evidence-mode source cardinality is fail-closed:

| Evidence mode | Required sources |
|---|---:|
| `closed_book` | exactly 0 |
| `single_source` | exactly 1 |
| `multi_source` | at least 2 |
| `adversarial_context` | at least 2 |
| `unanswerable` | at least 1 |

A request carries at most 32 sources and 1,048,576 total characters.

#### `ItemGenerationProvider`

A `@runtime_checkable` protocol:

```python
class ItemGenerationProvider(Protocol):
    provider_id: str
    model_id: str

    def generate(self, request: GenerationRequest) -> str: ...
```

Identifiers use the repository naming contract. The core executor checks protocol conformance and metadata before invocation. The provider receives exactly one immutable request and returns one JSON text.

#### `StaticFixtureProvider`

A deterministic injected adapter for unit tests, demos, and offline acceptance only. It receives explicit `provider_id`, `model_id`, and response text at construction and returns that exact text. Its docstring and guide state that it is not a quality generator and must not be used as production evidence.

### `fast_mlsirm.rubric.candidates`

#### Safe JSON decoding

`parse_generated_item_candidate(raw_json, request)`:

1. validates types and rejects output above 262,144 characters before decoding;
2. rejects NaN, Infinity, and negative Infinity through `parse_constant`;
3. rejects duplicate object names through `object_pairs_hook`;
4. requires a top-level object;
5. rejects missing and unknown top-level fields;
6. validates every nested object, collection, scalar, identifier, and size bound;
7. enforces the blueprint response format and exact rubric score coverage;
8. validates every source attribution against the request source packet;
9. returns an immutable content-addressed candidate.

Validation exceptions use `CandidateValidationError(code, path, message)`. `str(error)` includes only the stable code, field path, and generic message. It never includes the rejected value.

#### Candidate models

- `GeneratedOption(option_id, text)`
- `ScoreGuideEntry(score, evidence, rationale)`
- `RubricAlignmentEntry(score, observable_indicators)`
- `SourceAttribution(source_id, evidence_span)`
- `GeneratedItemCandidate(...)`

The candidate id is recomputed from canonical normalized content and must use two-or-more-token lower `snake_case`. The package additionally exposes a SHA-256 `candidate_fingerprint` independent of the model-supplied id.

Scoring-guide and rubric-alignment entries must each contain exactly one entry for every score in the blueprint, with no duplicate or missing score. Alignment indicators must be non-empty.

Source attribution rules:

- each `source_id` must occur in the request;
- duplicate `(source_id, evidence_span)` pairs are rejected;
- every evidence span must be a non-empty exact substring of the referenced source;
- source-backed modes require at least one attribution;
- `closed_book` requires none.

Response-format structural rules:

- `constructed_response`: `options` must be empty;
- `selected_response`: at least two unique options and `answer_key` must identify an option;
- `binary_judgment`: exactly two unique options and `answer_key` must identify one;
- `ordinal_rating`: `options` must be empty and `answer_key` must be an allowed rubric score or `null`;
- `pairwise_comparison`: exactly two unique options and `answer_key` must identify one option or equal `tie`.

These are authoring-artifact invariants, not psychometric scoring calculations.

### `execute_generation`

`execute_generation(provider, request)` validates provider metadata, calls the provider exactly once, parses the result, and returns:

```text
GenerationExecution(
    execution_id,
    request_id,
    contract_id,
    provider_id,
    model_id,
    candidate,
    raw_response_digest,
    schema_version,
)
```

The execution id hashes ids and digests only. Raw source and response text are not retained in the result. Provider exceptions are wrapped in `GenerationProviderError` without leaking provider messages that might contain sensitive material.

## Determinism, privacy, and security

- No timestamps enter content-addressed identifiers.
- No source content or raw provider response is stored in request metadata or execution results.
- Source and response digests support audit comparison without disclosing text.
- All public strings and collections are bounded before copies or nested validation.
- Duplicate source ids, option ids, score entries, alignment entries, and attributions fail closed.
- Provider ids and model ids require explicit stable identifiers; class names are never used as provenance.
- The executor performs one synchronous call. Retry, timeout, rate limit, and async orchestration belong to provider integrations.
- No URL fetching, filesystem access, shell execution, dynamic imports, or credential discovery is added.

## Errors

- Caller schema violations: `ValueError` with field names only.
- Wrong boundary object: `TypeError`.
- Candidate content: `CandidateValidationError` with stable `code` and `path`.
- Provider failure/non-string response: `GenerationProviderError` with generic text.

No exception includes source content, candidate values, or raw provider exception text.

## Testing strategy

RED tests specify:

- source normalization, digest determinism, metadata redaction, type/size/id/media/locale guards;
- evidence-mode cardinality and aggregate-source budget;
- request id determinism and content-change sensitivity;
- protocol conformance and exactly-once fixture execution;
- duplicate JSON key, non-finite number, top-level type, unknown/missing field, and size rejection;
- every nested candidate field and collection bound;
- exact score/alignment coverage, ordering normalization, and duplicate rejection;
- evidence-source existence and verbatim-span validation;
- all response-format structural branches;
- candidate/execution fingerprint determinism and raw-text redaction;
- provider failure redaction;
- public API and documentation links.

Added code must retain 100% statement and branch coverage and 100% docstring coverage.

## Release effect

This slice remains unreleased while stacked on the rubric compiler. It adds a changelog fragment but does not change package version. A release becomes appropriate after semantic screening and at least one optional real provider adapter complete a rubric-to-candidate-to-calibration demonstration on an exact artifact.

## References

Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). Internet Engineering Task Force. https://doi.org/10.17487/RFC8259

Bhutton, H., Andrews, H., Wright, A., & Hutton, B. (2022). *JSON Schema: A media type for describing JSON documents* (Draft 2020-12). JSON Schema. https://json-schema.org/draft/2020-12/json-schema-core

Bhutton, H., Andrews, H., Wright, A., & Hutton, B. (2022). *JSON Schema validation: A vocabulary for structural validation of JSON* (Draft 2020-12). JSON Schema. https://json-schema.org/draft/2020-12/json-schema-validation

Closes the design scope of issue #406.