# ADR-0004: Treat generated items as untrusted candidates

- Status: **Accepted**
- Date: 2026-08-09
- Owner: rubric/item-generation contract layer

## Context

LLM or external item generators can return malformed JSON, duplicate keys, non-finite values, stale/replayed identities, invented source references, invalid answer keys, or content that is structurally valid but psychometrically unusable. Treating a provider response as an assessment item would collapse transport success, structural validity, content validity, and calibration into one unsafe step.

## Decision

The generation path is a sequence of explicit trust transitions:

```text
RubricSpecification
 -> ItemBlueprint
 -> GenerationContract / GenerationRequest
 -> untrusted provider JSON
 -> structural + provenance + source validation
 -> GeneratedItemCandidate
 -> semantic/content screening
 -> pilot responses
 -> Rust-backed psychometric calibration
 -> governed item-bank decision
```

The core parser is deterministic and stricter than a default JSON decoder. Candidate-blind generation is the benchmark/default evaluation mode; candidate-aware criterion discovery, when intentionally supported, must use separated discovery/scoring folds so the same candidate cannot define its own final evaluation criteria.

## Invariants

1. Raw provider output is never an accepted operational item.
2. Duplicate JSON members, non-finite literals, excessive depth/size, unknown/missing fields, and malformed typed answer keys are rejected.
3. Provider output must replay exact rubric/blueprint/request provenance.
4. Source-backed candidates may cite only supplied sources and bounded verified spans.
5. Structural source-span presence is not claimed as semantic entailment.
6. Provider exception text does not become durable audit output because it may contain source content or credentials.
7. Candidate semantic screening and psychometric calibration remain separate evidence stages.
8. A future item bank uses immutable revisions and state transitions rather than editing accepted items in place.

## Alternatives considered

- **Trust the provider's JSON schema response:** rejected because a provider can violate its advertised schema and schema compliance does not prove source or psychometric validity.
- **Use one LLM call to generate and approve an item:** rejected because generation and validation errors become inseparable.
- **Store provider output as the canonical item and patch later:** rejected because provenance and score semantics can drift silently.

## Failure and recovery

Validation failure returns stable bounded codes/paths without the rejected value. A candidate that fails semantic or psychometric screening is quarantined/rejected; it is not repaired in place as if the same fingerprint remained valid. Regeneration produces a new candidate identity.

## Compatibility and rollback

Generation schemas are versioned. Downstream item-bank or hosted-product implementations may persist these artifacts but cannot bypass the core validation invariants. A rollback can stop accepting a generator/model version while retaining prior immutable execution evidence.

## Verification

Required evidence includes hostile JSON, direct-construction bypass, provenance replay, source cardinality, evidence-span, typed answer-key, score-order, resource-bound, exception-redaction, determinism, and later semantic/psychometric screening tests.

## Consequences

This adds multiple gates before an item can become operational, but it creates an auditable path from measurement intent to calibrated item evidence instead of treating generation fluency as validity.
