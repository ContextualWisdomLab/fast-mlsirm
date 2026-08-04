# Automated-Scoring Assessment Contracts Design

## Objective

Add the first executable slice of the provider-neutral automated-scoring core to the existing `fast-mlsirm` distribution. The slice defines one immutable assessment contract that binds declared constructs, exact `fast_mlsirm.rubric.RubricSpecification` revisions, and the five operational policy families required by later essay-scoring and enterprise-issue verticals.

The implementation performs validation, canonical serialization, and provenance only. It adds no scoring model, likelihood, gradient, uncertainty calculation, provider SDK, database, service, or authentication layer.

## Product boundary

```text
RubricSpecification[]
        +
ConstructSpec[]
        +
PolicyDocument[engine, calibration, validation, adjudication, monitoring]
        ↓
factory-sealed AssessmentSpec
```

`fast_mlsirm.rubric` remains the only source of truth for rubric levels, response formats, semantic versions, and content fingerprints. The scoring package stores exact rubric bindings instead of copying score-level definitions.

## Components

### `scoring.errors`

`ScoringContractError` carries a descriptive two-or-more-token lower `snake_case` error code, a JSON-style path, and a bounded message. Errors never include response text, source content, or raw provider output.

### `scoring._json`

A private bounded JSON normalizer accepts only null, booleans, integers, finite floats, strings, arrays, and mappings. It enforces depth, node, collection, string, and total encoded-size limits; requires descriptive metadata keys; and rejects known response/source-text fields.

### `scoring.contracts`

- `ConstructSpec`: declared construct identity, label, and definition.
- `PolicyKind`: `engine_policy`, `calibration_policy`, `validation_policy`, `adjudication_policy`, and `monitoring_policy`.
- `PolicyDocument`: factory-sealed policy identity/version/kind plus bounded canonical settings.
- `RubricBinding`: factory-issued exact rubric ID, semantic version, SHA-256 fingerprint, construct, and response format.
- `AssessmentSpec`: factory-sealed, canonically ordered constructs, rubric bindings, policy documents, and bounded metadata.

Every governed value exposes a full 64-character SHA-256 fingerprint and a descriptive 128-bit public handle. Hashes are content identity and audit evidence, not authorization credentials.

## Invariants

1. Public identifiers contain at least two lower `snake_case` tokens.
2. Assessment and policy versions are canonical `major.minor.patch` values.
3. Each rubric construct is declared exactly once in the assessment.
4. Rubric IDs are unique and bind to their exact full fingerprints.
5. Exactly one policy document exists for every required `PolicyKind`.
6. Equivalent caller ordering produces byte-identical canonical content.
7. Any construct, rubric, policy, metadata, or version change changes the appropriate fingerprint.
8. Direct construction cannot forge governed policy, rubric-binding, or assessment objects.
9. Metadata cannot carry raw response or source-content fields.
10. No psychometric arithmetic is introduced in Python.

## Error handling

Factories fail closed before large allocations. Validation errors expose stable code/path metadata without embedding rejected text. Internal factory-sealed objects recheck canonical JSON and cross-contract ordering so corrupted serialized values cannot be relabelled as governed artifacts.

## Testing

The module requires 100% statement and branch coverage. Tests cover deterministic ordering, fingerprint mutation, factory seals, rubric/construct references, complete policy-family enforcement, malformed and oversized iterables, JSON resource bounds, non-finite values, sensitive-field rejection, malformed internal serialization, and explicit package exports.

## Documentation and release governance

The feature is documented for standalone and later MSA use. An authoritative changelog fragment is rendered into `CHANGELOG.md` on the same branch. A version bump is deferred until the shared scoring core has at least the observation and engine protocol slices needed for a coherent public release.
