# ADR-0003 — Canonical Contracts and Immutable Provenance

- **Status:** Accepted
- **Date:** 2026-08-09
- **Decision owner:** `fast-mlsirm`
- **Implementation status:** active for canonical reusable assessment/rubric/scoring contracts; feature-specific lifecycle coverage varies

## Context

Rubric generation, automated scoring, enterprise/RAG adapters and downstream assessment products all need to refer to the same measurement definition. Parallel schemas or mutable identifiers permit replay mistakes, provenance confusion and incompatible scoring semantics.

## Decision

`fast-mlsirm` owns canonical reusable Assessment/Rubric/Scoring contracts. Domain adapters compile into these contracts rather than defining competing rubric, observation, result or engine schemas.

Material artifacts use immutable/versioned content identity. Human-friendly handles may be shorter, but exact scientific/audit identity uses the full fingerprint or another collision-resistant canonical identity. Provider-generated content is untrusted until it is rebound to the exact request, rubric, blueprint, source/evidence and engine provenance.

## Invariants

- Assessment/rubric/scoring lineage remains versioned and explicit.
- Rubric/item-generation contracts extend canonical lineage; they do not replace it.
- Operational artifacts are immutable. Revision creates a new version/fingerprint.
- Caller metadata cannot overwrite package-managed provenance fields.
- Cross-layer schema/parser parity is tested.
- Stable errors do not echo rejected provider/source content.
- Raw content is retained only where the explicit contract requires it; digest/opaque provenance is preferred for audit artifacts.

## Untrusted-generation boundary

Generated-item inputs/outputs use bounded closed schemas. At minimum the boundary rejects duplicate keys, invalid/non-finite JSON numbers where disallowed, unknown/missing fields, oversized payloads, response-format-incompatible answer keys, stale contract/blueprint/rubric identities, invalid source references and unverified evidence spans.

Structural validity is not psychometric validity. Answerability, ambiguity, construct alignment, distractor quality, evidence entailment, redundancy, leakage, bias/DIF risk and calibration remain later gates.

## Alternatives considered

1. Domain-specific duplicate scoring schemas — rejected due drift and migration complexity.
2. Opaque random IDs without content binding — insufficient for replay/audit.
3. Canonical content-addressed contracts with adapters — accepted.

## Consequences

Adapters become slightly more verbose because they must map into shared contracts. In return, model calibration, reports, audit evidence and downstream services can share one interpretation and one version lineage.

## Failure / degraded behavior

Unknown schema versions, stale fingerprints, forged provenance, or cross-request artifacts fail closed. Compatibility adapters may translate only explicitly supported versions; they cannot silently reinterpret historical results under a newer contract.

## Security / privacy

Canonical provenance narrows confused-deputy/replay attacks and permits evidence without retaining ambient raw content. Content hashing is not anonymization; privacy controls must still address re-identification and authorized access.

## Compatibility / migration

Schema changes require a version/deprecation or explicit migration boundary. Migration verifies the original canonical content and fingerprint before creating a new version. Historical observations/results are not implicitly re-signed under a newer contract.

## Verification

- canonical serialization and fingerprint tests;
- forged/replayed/cross-request artifact rejection;
- JSON Schema/parser parity;
- adversarial provider-output tests;
- package-root export and round-trip tests;
- changelog/migration evidence for wire changes.

## Sources

Bray, T. (2017). *The JavaScript Object Notation (JSON) data interchange format* (RFC 8259). RFC Editor. https://doi.org/10.17487/RFC8259

JSON Schema. (2022). *JSON Schema Draft 2020-12*.

American Educational Research Association, American Psychological Association, & National Council on Measurement in Education. (2014). *Standards for educational and psychological testing*. American Educational Research Association.

## Supersession criteria

Supersede if a separately versioned shared-contract package becomes necessary for multiple independent producers, provided it preserves canonical lineage and does not introduce duplicate semantic owners.
