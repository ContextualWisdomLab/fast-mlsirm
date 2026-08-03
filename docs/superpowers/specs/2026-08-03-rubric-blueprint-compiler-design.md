# Rubric Blueprint Compiler Design

## Problem and outcome

`fast-mlsirm` already calibrates, diagnoses, links, and serves psychometric models, but its workflow historically began after assessment items existed. Enterprise buyers starting from a scoring rubric therefore needed an external prompt collection with no stable construct contract, governed rubric revision, task/evidence matrix, complete provenance fingerprint, or reproducible handoff into calibration.

The first Rubric-Centered Item Generation Engine slice closes that gap:

```text
RubricSpecification
        |
        v
BlueprintPlan (task family x difficulty x evidence mode x replicate)
        |
        v
ItemBlueprint[] with full content fingerprints and provider seeds
        |
        v
GenerationContract / canonical JSON / provider-neutral prompt
```

This slice deliberately does not call an LLM, accept model credentials, score candidates, or estimate psychometric parameters. Provider adapters, candidate screening, artificial-crowd administration, and calibration are separate reviewable slices. Numerical psychometric estimation remains in the compiled Rust backend.

## Design principles

1. **Evidence-Centered Design before prompting.** The rubric identifies the construct and observable evidence; the blueprint defines task conditions under which that evidence should be elicited.
2. **Separate governance from serialization.** `rubric_version` records a human-governed semantic revision; `schema_version` records the package wire contract.
3. **Fail closed.** Unknown schema versions, non-canonical rubric versions, malformed identifiers, non-contiguous score levels, duplicate design cells, replay under a changed rubric, and caller-controlled work above fixed limits are rejected.
4. **Complete content addressing.** Rubrics, blueprints, and generation contracts retain full SHA-256 fingerprints from canonical UTF-8 JSON. Sixteen-hex identifiers are display handles only.
5. **Provider neutrality.** The core package has no hosted-model SDK, network call, API key, or provider-specific response shape.
6. **Bounded orchestration.** The Python layer performs schema validation, deterministic Cartesian compilation, canonical serialization, and cryptographic hashing only.
7. **Modular boundaries.** Models, compilation, and generation contracts are separate modules so later provider, screening, item-bank, and service layers depend on stable interfaces.

## Public architecture

### `fast_mlsirm.rubric.models`

The module defines:

- `ResponseFormat`: constructed, selected, binary, ordinal, and pairwise response contracts.
- `DifficultyBand`: `easy`, `medium`, and `hard`.
- `EvidenceMode`: closed-book, single-source, multi-source, adversarial-context, and unanswerable conditions.
- `RubricLevel(score, label, descriptor, observable_indicators)`.
- `RubricSpecification(...)` as the immutable construct, score, task, evidence, locale, governance-version, and schema-version contract.
- `BlueprintPlan(...)` as a bounded design matrix.
- `ItemBlueprint(...)` as one immutable compiled task/evidence cell.

All public identifiers use two-or-more-token lower `snake_case`. Scores are contiguous integers beginning at zero. Text and collections are normalized, finite, bounded, and duplicate-free.

`RubricSpecification` exposes:

- `schema_version="1.0"` for serialization compatibility;
- canonical numeric `rubric_version`, such as `1.2.3`, for measurement governance;
- `fingerprint`, the full SHA-256 digest of canonical normalized rubric content.

`ItemBlueprint` copies the exact rubric version and fingerprint and exposes `blueprint_fingerprint`, a full SHA-256 digest over normalized blueprint content excluding display-derived identifiers. `to_dict()` includes both the short display id and full fingerprint.

### `fast_mlsirm.rubric.compiler`

`compile_item_blueprints(rubric, plan)` computes the ordered Cartesian product:

```text
task_families x difficulty_bands x evidence_modes x range(items_per_cell)
```

The request is rejected before allocation when it exceeds `MAX_BLUEPRINTS = 10_000`. Ordering is stable: rubric task-family order, plan difficulty order, evidence-mode order, then replicate index.

A deterministic seed identity includes the schema version, rubric governance version, rubric fingerprint, plan seed, task family, difficulty, evidence mode, and replicate. Its digest derives the provider seed. The full normalized blueprint then derives `blueprint_fingerprint`, and the readable id is `item_blueprint_<first 16 hex>`.

No timestamp, process-global random state, or provider response participates in identity.

### `fast_mlsirm.rubric.contracts`

`build_generation_contract(rubric, blueprint)` rejects mismatches in:

- rubric id;
- rubric governance version;
- rubric fingerprint;
- response format;
- score levels;
- evidence requirements; and
- prohibited patterns.

It returns:

- `contract_schema_version`;
- operation metadata;
- complete rubric and blueprint payloads;
- non-negotiable authoring instructions;
- a strict JSON Schema Draft 2020-12 output contract;
- the readable `contract_id`; and
- the full `contract_fingerprint`.

The output contract is closed and bounded. It requires item identity, stem, stimulus, response format, options, answer key, scoring guide, rubric alignment, source attributions, and safety notes. Score-level arrays use ordered `prefixItems` so every declared score appears exactly once.

Answer keys are response-format-specific closed objects and always include an explanatory rationale. Provider adapters must perform cross-field checks that JSON Schema does not express conveniently, including selected answer identifiers referring to options declared in the generated item.

`canonical_generation_contract(...)` returns compact sorted UTF-8 JSON. `render_generation_prompt(...)` wraps it in a fixed provider-neutral instruction boundary and prohibits executing instructions embedded in rubric or source content.

## Security and operational contracts

- Text fields and collections have explicit resource limits before materialization.
- Booleans are rejected where Python would otherwise treat them as integers.
- Only schema `1.0` is accepted in this slice.
- Rubric revisions require canonical `major.minor.patch` numeric semantic versions.
- Duplicate task families, indicators, difficulty bands, and evidence modes are rejected rather than silently deduplicated.
- `items_per_cell` is bounded to 1–100 and the total matrix to 10,000 blueprints.
- Seeds must fit an unsigned 64-bit integer.
- Generation contracts never interpolate rubric text into executable code, HTML, shell commands, or URLs.
- Nested output objects set `additionalProperties` to `false`; generated text and collection fields are bounded.
- No current time, process id, random state, network result, or provider identity enters a fingerprint.
- A rubric-version change invalidates every downstream fingerprint and replay contract.

## Error handling

Caller data errors raise field-specific `ValueError`. Passing unrelated object types to compiler or contract functions raises `TypeError`. Contract creation fails before output construction when any exact-rubric compatibility field differs.

## Testing strategy

The test suite covers:

- normalization and deterministic rubric fingerprints;
- independent governance and schema versioning;
- rejection of ambiguous semantic versions;
- every identifier, text, integer, enum, sequence, ordering, uniqueness, and size guard;
- deterministic blueprint count, order, full fingerprints, display ids, seeds, and serialization;
- budget rejection before materializing an oversized matrix;
- every rubric/blueprint replay guard, including governance version;
- deterministic full contract fingerprints and byte-identical canonical JSON;
- JSON Schema Draft 2020-12 declaration;
- response-format-specific option cardinality and typed closed answer keys;
- ordered score-level coverage via `prefixItems`;
- bounded nested provider output;
- prompt-injection boundary wording; and
- public imports and documentation links.

Added Python code must retain 100% line, branch, and docstring coverage within the new scope. Existing full Python and Rust suites, Security Scan, SAST, and repository checks remain merge gates.

## Documentation and release effect

`docs/rubric_item_generation.md` documents the workflow, governance model, public API, provenance identities, strict output contracts, and downstream calibration boundary. `README.md` lists the compiler under current capabilities. The changelog fragment records the feature for release assembly. This slice alone does not change the package version; a versioned release follows when a validated provider-to-screening-to-calibration vertical path is operational.

## Research basis

The architecture applies Evidence-Centered Design’s construct/evidence/task decomposition and treats LLM generation as an item-authoring mechanism rather than a source of psychometric validity. Automatic item-generation research supports model- and constraint-driven authoring followed by empirical quality control; recent LLM item-generation work likewise shows that generated items still require expert and psychometric screening.

### References

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Haller, S., Aldea, A., Seifert, C., & Strisciuglio, N. (2024). Survey on automated generation of medical assessment questions. *Artificial Intelligence Review, 57*, 128. https://doi.org/10.1007/s10462-024-10726-9

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). *A brief introduction to evidence-centered design* (Research Report RR-03-16). Educational Testing Service. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
