# ADR-0003: Content-addressed measurement contracts

Status: **Accepted**  
Date: 2026-08-09

## Context

Psychometric interpretation depends on exact construct, rubric, task, response, rater/engine, calibration and software revisions. Human-readable IDs alone do not prove content identity, while mutable objects can make later audits falsely appear to reproduce the original analysis.

## Decision

Reusable measurement artifacts use versioned, canonical and content-addressed contracts wherever semantic identity affects interpretation.

Key rules:

1. Schema/wire version and semantic/domain revision are separate fields.
2. Authoritative content identity uses deterministic canonical serialization and a full cryptographic digest where the contract exposes fingerprints.
3. Short public handles may aid display but never replace full fingerprint comparison at a trust boundary.
4. Aggregate artifacts replay/verify package-owned child objects when crossing a trust boundary rather than trusting parent caches or display IDs.
5. Logical task ID and exact task revision remain separate.
6. Stable respondent/system identity and response artifact/revision remain separate.
7. Rater/engine descriptor identity and individual rater identity remain separately representable where many-facet interpretation requires them.
8. Published/approved artifacts are immutable; corrections create superseding revisions.
9. Caller/provider text is bounded and sensitive content is not embedded in error identifiers or audit hashes beyond the explicit canonical input.

## Canonical fingerprint preimage contract

The v1 fingerprint preimage contract is named `fast-mlsirm-cjson-v1` and is
part of every exposed fingerprint record together with the digest algorithm
`sha-256`. Implementations MUST reject an unknown canonicalization version or
digest algorithm rather than silently selecting a local default.

For `fast-mlsirm-cjson-v1`, the preimage is the UTF-8 encoding of compact JSON
with no trailing newline, using these rules:

1. Object keys are unique strings and are ordered by Unicode scalar-value order;
   arrays retain their declared order.
2. A contract's declared field set is authoritative. Omitted fields and explicit
   `null` are different values; no serializer may add, omit, or coerce fields.
3. Strings are preserved as supplied after valid UTF-8 validation. No implicit
   Unicode normalization is performed, so composed and decomposed spellings
   have different identities unless the contract normalizes them before
   construction.
4. Integers are signed 64-bit values. Floating-point values are finite; negative
   zero is serialized as `0.0`. Non-finite values are rejected. Numbers use the
   shortest round-trippable JSON decimal representation of the package
   canonicalizer.
5. Whitespace is omitted, escaping follows the UTF-8 JSON serializer, and the
   SHA-256 digest is computed over the exact resulting bytes.

The package-owned implementation and any Rust/other-language implementation
MUST pass the same normative vectors. For example, after the v1 validation and
negative-zero normalization, this value:

```json
{"a":null,"items":[2,1.5],"n":0.0,"z":"café"}
```

has UTF-8 SHA-256
`b2384fee029c793d3e661b5a741b318155a401482a7495d057bb696a2711c9c5`.
The vector binds the version, encoding, key ordering, null retention, array
ordering, Unicode preservation, numeric formatting and digest calculation; a
cross-language implementation is not interoperable until it reproduces it.

## Primary contract family

```text
RubricSpecification
        |
        v
AssessmentSpec / policy references
        |
        +--> task + exact task revision
        |
        v
ScoringRequest
        |
        v
ScoreObservation / ScoringResult
        |
        v
Calibration design/report
```

Rubric-centered item generation extends the same chain:

```text
RubricSpecification -> ItemBlueprint -> GenerationContract
        -> Candidate -> Screening/Calibration -> Item-bank revision
```

## Invariants

- A one-byte semantic change that participates in canonical content changes the fingerprint.
- Reusing an identifier with changed canonical content fails closed at governed boundaries.
- Source/evidence spans are verified against the exact source revision when source evidence is part of the contract.
- Unknown major schema versions do not silently downgrade.
- Serialization is deterministic across input ordering where ordering is not semantically meaningful and preserves declared ordering where it is meaningful.

## Consequences

This increases object/schema discipline and migration work but provides replay protection, reproducible research, audit evidence, safe caching and reliable cross-service composition.

## Alternatives considered

- **Database integer IDs as identity.** Rejected because they identify rows, not semantic content, and are not portable across independent deployments.
- **Mutable named revisions.** Rejected because an old result could silently point at new content.
- **Hash everything including uncontrolled raw text in every report.** Rejected; only contract-relevant content is included, with privacy/data-minimization boundaries.

## Reversal conditions

A replacement must provide equal or stronger deterministic replay, compatibility, privacy and cross-deployment identity guarantees.
