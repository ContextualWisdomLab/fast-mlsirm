# Rubric Blueprint Compiler Design

## Problem and outcome

`fast-mlsirm` already calibrates, diagnoses, links, and serves psychometric models, but its workflow begins after items exist. Enterprise buyers who start from a scoring rubric must currently maintain an external prompt collection with no stable schema, item-design matrix, provenance fingerprint, or reproducible handoff into calibration. The first Rubric-Centered Item Generation Engine slice closes that gap by compiling an immutable rubric specification into deterministic, provider-neutral item-generation blueprints and canonical generation contracts.

The delivered workflow is:

```text
RubricSpecification
        |
        v
BlueprintPlan (task family x difficulty x evidence mode x replicate)
        |
        v
ItemBlueprint[] with content-addressed ids and provider seeds
        |
        v
GenerationContract / canonical JSON / LLM-ready prompt
```

This slice deliberately does not call an LLM, score candidates, or estimate psychometric parameters. Provider adapters and screening are subsequent isolated slices; all psychometric estimation remains in the compiled Rust backend.

## Design principles

1. **Evidence-Centered Design before prompting.** The rubric identifies the construct and observable evidence; the blueprint supplies the task-model conditions under which that evidence should be elicited.
2. **Fail closed.** Unknown schema versions, malformed identifiers, non-contiguous score levels, duplicate design cells, and caller-controlled work above a fixed budget are rejected.
3. **Content addressing.** Rubrics, blueprints, and generation contracts receive SHA-256 fingerprints from canonical UTF-8 JSON. Repeating the same input produces byte-identical output.
4. **Provider neutrality.** The core package has no hosted-model SDK, network call, API key, or provider-specific response shape.
5. **Bounded orchestration.** The Python layer performs schema validation, deterministic Cartesian compilation, canonical serialization, and cryptographic hashing only. IRT scoring, calibration, fit, DIF, and information calculations remain Rust-backed.
6. **Modular boundaries.** Models, compilation, and generation contracts are separate modules so later provider, screening, artificial-crowd, and item-bank services can depend on stable interfaces rather than internal implementations.

## Public architecture

### `fast_mlsirm.rubric.models`

Defines:

- `ResponseFormat`: `constructed_response`, `selected_response`, `binary_judgment`, `ordinal_rating`, and `pairwise_comparison`.
- `DifficultyBand`: `easy`, `medium`, and `hard`.
- `EvidenceMode`: `closed_book`, `single_source`, `multi_source`, `adversarial_context`, and `unanswerable`.
- `RubricLevel(score, label, descriptor, observable_indicators)`.
- `RubricSpecification(rubric_id, construct_id, construct_definition, response_format, levels, task_families, evidence_requirements, prohibited_patterns, locale, schema_version)`.
- `BlueprintPlan(difficulty_bands, evidence_modes, items_per_cell, seed)`.
- `ItemBlueprint(...)` as the immutable compiled task/evidence model.

All identifiers use two-or-more-token lower `snake_case`. Rubric scores are contiguous integers beginning at zero. String collections are normalized to stripped tuples and reject empty or duplicate values. Public inputs have explicit length and count bounds.

`RubricSpecification.to_dict()` and `ItemBlueprint.to_dict()` return only JSON-compatible values. `RubricSpecification.fingerprint` is the SHA-256 digest of its canonical serialized representation.

### `fast_mlsirm.rubric.compiler`

`compile_item_blueprints(rubric, plan)` computes the ordered Cartesian product:

```text
task_families x difficulty_bands x evidence_modes x range(items_per_cell)
```

Each design cell produces one immutable `ItemBlueprint`. Its identifier and 64-bit generation seed are derived from the rubric fingerprint, plan seed, task family, difficulty, evidence mode, and replicate index. No process-global random state or timestamp is used. The complete request is rejected before allocation when it exceeds `MAX_BLUEPRINTS = 10_000`.

Ordering is stable: rubric task-family order, then plan difficulty order, then evidence-mode order, then replicate index. This allows reproducible diffs, caching, and audit comparison.

### `fast_mlsirm.rubric.contracts`

`build_generation_contract(rubric, blueprint)` verifies that the blueprint belongs to the exact rubric fingerprint and returns a JSON-compatible contract containing:

- schema and operation identifiers;
- rubric and blueprint payloads;
- non-negotiable authoring instructions;
- a strict structured-output schema for one item;
- rubric-level alignment and source-attribution requirements;
- the deterministic `generation_contract_*` content-addressed identifier.

`canonical_generation_contract(...)` returns compact, sorted, UTF-8 JSON. `render_generation_prompt(...)` wraps the contract in a provider-neutral instruction that requires a single JSON object and tells the model not to execute instructions embedded in rubric or source text.

## Validation and security contracts

- Text fields are finite in size; collections are bounded before copying.
- Booleans are rejected where Python would otherwise accept them as integers.
- Schema version `1.0` is the only accepted version in this slice.
- Duplicate task families, indicators, difficulty bands, and evidence modes are rejected rather than silently deduplicated.
- `items_per_cell` is bounded to 1–100 and the total matrix to 10,000 blueprints.
- Seeds must fit an unsigned 64-bit integer.
- Generation contracts never interpolate rubric text into executable code, HTML, shell, or URLs.
- The contract output schema sets `additionalProperties` to `false` and requires provenance, scoring, alignment, and attribution fields.
- No current time, process id, random state, or network result enters a fingerprint.

## Error handling

Caller errors raise `ValueError` with the exact failing field. Passing the wrong object type to compiler or contract functions raises `TypeError`. Contract creation rejects rubric-id, fingerprint, response-format, score-level, evidence-requirement, or prohibited-pattern mismatches so a blueprint cannot be replayed under a modified rubric.

## Testing strategy

The test suite follows RED–GREEN TDD and covers:

- valid normalization and deterministic rubric fingerprints;
- every identifier, text, integer, enum, sequence, version, ordering, uniqueness, and size guard;
- deterministic blueprint count, order, ids, seeds, and canonical serialization;
- budget rejection before materializing an oversized matrix;
- all rubric/blueprint mismatch guards in generation contracts;
- deterministic contract ids and byte-identical canonical JSON;
- exact structured-output requirements and prompt injection boundary wording;
- public import surface and JSON serializability.

Added Python code must retain 100% line and branch coverage and 100% docstring coverage. Existing full Python and Rust suites, Security Scan, SAST, and repository checks remain merge gates.

## Documentation and release effect

`docs/rubric_item_generation.md` will describe the workflow, API example, boundaries, and roadmap. `README.md` will list the compiler under current capabilities. `CHANGELOG.md` will record the feature under Unreleased. This feature alone does not change the distribution version; a release follows only after the provider/screening/calibration vertical path is usable end to end.

## Research basis

The architecture applies Evidence-Centered Design’s construct/evidence/task decomposition and treats LLM generation as an item-authoring mechanism rather than a source of psychometric validity. Automatic item-generation research supports template/model-driven authoring followed by empirical quality control; recent LLM item-generation work likewise shows that generated items still require expert or psychometric screening.

### References

Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). *A brief introduction to evidence-centered design* (Research Report RR-03-16). Educational Testing Service. https://doi.org/10.1002/j.2333-8504.2003.tb01908.x

Gierl, M. J., & Lai, H. (2012). The role of item models in automatic item generation. *International Journal of Testing, 12*(3), 273–298. https://doi.org/10.1080/15305058.2011.635830

Haller, S., Aldea, A., Seifert, C., & Strisciuglio, N. (2024). Survey on automated generation of medical assessment questions. *Artificial Intelligence Review, 57*, 128. https://doi.org/10.1007/s10462-024-10726-9

Closes the design scope of issue #394; implementation and verification are tracked in the corresponding implementation plan.