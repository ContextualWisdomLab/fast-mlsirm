# Auditable Rubric-to-Item Blueprint Compiler Design

## Decision

Build the first vertical slice of the Rubric-Centered Item Generation Engine as a
provider-neutral, offline compiler. The compiler turns a validated rubric
specification into an exhaustive and bounded matrix of immutable item
blueprints, then emits content-addressed generation contracts for downstream
model adapters.

This slice does **not** call an LLM, score items, fit an IRT model, or introduce a
provider SDK. Psychometric estimation remains in the Rust core. The purpose of
this slice is to make the authoring input to that core reproducible, auditable,
and portable.

## Problem

`fast-mlsirm` already calibrates and diagnoses items, but buyers still need an
external prompt workflow to translate a natural-language rubric into candidate
item specifications. Those workflows tend to be provider-specific, mutable,
and difficult to reproduce. A buyer cannot reliably answer:

- which rubric version produced an item;
- which task, evidence, locale, and difficulty constraints were applied;
- whether two generation runs used semantically identical contracts;
- whether a later calibration result can be traced back to its authoring input.

The compiler closes that gap before any hosted-model integration is added.

## Evidence-centered design basis

The architecture follows evidence-centered assessment design by separating:

1. **Claim/construct specification** — what capability or quality is measured.
2. **Evidence specification** — what observable indicators support each score.
3. **Task specification** — what task and evidence conditions elicit those
   observations.
4. **Assembly/operational specification** — how the task matrix is compiled and
   bounded for use.

Primary methodological grounding:

- Mislevy, R. J., Almond, R. G., & Lukas, J. F. (2003). *A brief
  introduction to evidence-centered design* (ETS Research Report RR-03-16).
  https://doi.org/10.1002/j.2333-8504.2003.tb01908.x
- Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2003). On the
  structure of educational assessments. *Measurement: Interdisciplinary
  Research and Perspectives, 1*(1), 3-62.
- Mislevy, R. J., Steinberg, L. S., & Almond, R. G. (2002). Design and
  analysis in task-based language assessment. *Language Testing, 19*(4).
  https://doi.org/10.1191/0265532202lt241oa

The serialization contract follows the deterministic, sorted-key, UTF-8,
no-insignificant-whitespace principles of RFC 8785. The first slice deliberately
restricts canonical payload values to strings, integers, booleans, nulls,
arrays, and string-keyed objects; floats are rejected. Under that restricted
value set, the standard-library serializer can produce a stable JCS-compatible
representation without implementing ECMAScript floating-point formatting.
Structured-output schemas identify JSON Schema Draft 2020-12.

## Approaches considered

### 1. Provider-first prompt templates

Create prompt strings for one hosted model and add other providers later.

Rejected because prompt text would become the de facto schema, provider details
would leak into the core package, and identical rubric intent would not have a
stable cross-provider identity.

### 2. Pydantic or another schema dependency

Use a third-party validation package for compact model definitions.

Rejected for this slice because the package currently has one runtime dependency
(`numpy`), the core compiler needs no external validation engine, and a new
runtime dependency would enlarge procurement and supply-chain review surface.

### 3. Immutable standard-library schema plus deterministic compiler

Use frozen dataclasses, explicit validators, a bounded Cartesian compiler,
canonical JSON, SHA-256 fingerprints, and JSON Schema output contracts.

Selected because it is offline, dependency-free, independently testable, and
cleanly separates authoring from later provider and calibration adapters.

## Public architecture

```text
RubricSpecification
    -> strict schema validation
    -> compile_item_blueprints(max_blueprints=...)
    -> tuple[CompiledBlueprint, ...]
    -> build_generation_contract(...)
    -> GenerationContract
```

### `rubric_schema.py`

Defines immutable value objects:

- `RubricLevel`
- `RubricCriterion`
- `TaskFamily`
- `EvidenceMode`
- `DifficultyBand`
- `RubricSpecification`

The models validate two-or-more-token `snake_case` identifiers, non-empty text,
unique identifiers, ordered contiguous score levels, ordered contiguous
difficulty bands, locale syntax, evidence-source bounds, and explicit response
formats.

### `canonical_contract.py`

Provides:

- conversion of supported dataclasses to canonical JSON primitives;
- stable UTF-8 canonical JSON;
- `sha256:<hex>` fingerprints;
- immutable `CompiledBlueprint` and `GenerationContract` envelopes;
- a Draft 2020-12 structured-output schema for generated item records.

The canonicalizer rejects floats, bytes, sets, non-string object keys, NaN, and
other values outside the documented contract instead of silently producing
language-specific identities.

### `blueprint_compiler.py`

Compiles the deterministic product:

```text
criterion x task_family x evidence_mode x difficulty_band
```

Axis order is canonical rather than caller-order dependent:

- criteria by `criterion_id`;
- task families by `task_family_id`;
- evidence modes by `evidence_mode_id`;
- difficulty bands by `order_index`, then `difficulty_band_id`.

The compiler calculates the complete product size before materialization and
fails closed when it exceeds the caller-provided `max_blueprints`. It never
silently truncates a matrix.

Each payload receives:

- a full SHA-256 fingerprint over the canonical payload;
- an identifier `item_blueprint_<digest-prefix>`;
- the source rubric fingerprint;
- all task, evidence, locale, response, difficulty, score-level, and prohibited
  pattern constraints required by a downstream adapter.

### Provider-neutral generation contract

A generation contract is a content-addressed canonical JSON document containing:

- the complete blueprint payload and its fingerprints;
- explicit generation rules that prohibit invented evidence and schema drift;
- a JSON Schema Draft 2020-12 output schema;
- required traceability fields for rubric alignment and evidence references.

No model name, API endpoint, token parameter, temperature, or provider SDK type
appears in the core contract.

## Identifier and data rules

- Domain identifiers use at least two lower-case `snake_case` tokens.
- Schema versions use semantic version strings and are checked against supported
  constants.
- Ordinal scores are integers, strictly ordered, unique, and contiguous.
- Difficulty `order_index` values start at zero and are contiguous.
- Every rubric level contains at least one observable indicator.
- Every rubric contains at least one criterion, task family, evidence mode, and
  difficulty band.
- Duplicate axis identifiers are rejected.
- Evidence source minima and maxima are non-negative and internally consistent.
- Canonical payloads contain no floating-point numbers.

## Error handling

Validation errors are deterministic `ValueError` exceptions that name the
invalid field. Product overflow is reported before any blueprint is created.
Canonicalization rejects unsupported types rather than coercing them. Contract
building verifies that the supplied blueprint fingerprint still matches its
payload before emitting a generation contract.

## Testing strategy

The implementation is test-driven. A test-only commit establishes RED before
production files are added. Tests cover:

- every identifier, version, text, uniqueness, ordering, and bounds guard;
- immutable schema objects;
- canonical serialization and type rejection;
- order-independent rubric and blueprint identities where order is semantic-free;
- deterministic Cartesian output and fail-closed size limits;
- fingerprint sensitivity to every material constraint;
- output-schema strictness and generation-contract traceability;
- public exports, README examples, CHANGELOG, branch coverage, and docstrings.

All new Python branches and public docstrings remain at 100%. Existing Rust
suites continue unchanged because this slice adds no psychometric arithmetic.

## Security and procurement boundaries

- No network access.
- No dynamic code execution.
- No provider credentials.
- No arbitrary template evaluation.
- No new runtime dependency.
- Canonical documents are content-addressed with SHA-256.
- Caller-controlled products are bounded before allocation.
- Generated-item output schemas use `additionalProperties: false`.

## Deferred slices

The following are explicitly out of scope and remain separate PRs:

1. provider protocol and offline fixture provider;
2. semantic and structural screening;
3. artificial-crowd execution;
4. Rust-backed calibration orchestration;
5. living item-bank drift, DIF, exposure, and regeneration;
6. buyer-facing authoring/calibration UI and Figma workflow.

This boundary preserves standalone package use while leaving stable interfaces
for MSA integration and future submodule consumption.
