# ADR-0009 — Governed Rubric and Item-Bank Lifecycle

- **Status:** Accepted
- **Date:** 2026-08-09

## Context

A rubric-to-item generator that stops after LLM text generation is not a
measurement product. Generated criteria/items can be structurally malformed,
semantically ambiguous, redundant, misaligned, leaked, biased or psychometrically
uninformative. Dynamic rubrics can also leak evaluation candidates or change the
meaning of scores across versions.

## Decision

Treat dynamic rubric/item generation as online item-bank construction with
separate generation, validation, calibration, approval, release and retirement
phases.

Canonical lifecycle:

```text
draft
→ audited / structurally screened
→ semantically screened
→ pilot
→ calibrated
→ approved
→ active
→ suspended or quarantined
→ retired
```

Operational versions are immutable. Revisions create new version/fingerprint
identities and require anchor/linking evidence for longitudinal comparison.

## Benchmark and discovery modes

- **benchmark mode:** criteria are candidate-blind and based on task contract and
  an explicit evidence regime;
- **diagnostic/discovery mode:** candidate-aware discovery is permitted only with
  cross-fitting or otherwise separate discovery/scoring data;
- **training mode:** evolving reward/rubric banks are isolated from frozen
  benchmark banks.

## Invariants

- Generation provider output is untrusted until schema/provenance validation.
- Structural validation is distinct from semantic screening and psychometric
  calibration.
- Atomic Boolean/discrete criteria are preferred when they improve judge
  reliability, but response type follows the construct/evidence need rather than a
  universal rule.
- Scoring-critical failures are policy gates, not merely low-weight psychometric
  items that can be averaged away.
- Candidate/rubric/evidence/source/engine versions remain auditable.
- Item exposure, DIF/drift, linking and calibration history persist across
  operational releases.
- A rubric criterion's psychometric discrimination is not the same as business or
  safety criticality.

## Evidence regimes

Every factual criterion states what universe supports it, for example:
`prompt_only`, `retrieved_context`, `pooled_corpus`, `authoritative_corpus`, or
`human_anchor`. Groundedness against retrieved context is not renamed world
correctness when the evidence universe cannot establish truth.

## Alternatives considered

1. One-shot `generate_rubric()` returning free text — rejected as the canonical
   architecture.
2. Generate a 1–5 holistic rubric and average judge scores — insufficient for
   item/rater calibration and lifecycle governance.
3. Rubric→blueprint→candidate→screening→pilot→calibration→bank→revision — accepted.

## Consequences

The feature requires more orchestration and versioning, but creates a defensible
buyer workflow and allows existing CAT/ATA, DIF, linking, fit, many-facet,
bifactor/testlet and recovery capabilities to operate on governed items instead
of arbitrary prompt text.

## Failure / degraded behavior

If evidence is insufficient, a candidate is rejected/abstained/quarantined rather
than converted into a low score. If version linking is not identified, cross-
version score comparison is blocked. If semantic screening is unavailable, the
item remains pre-calibration and cannot be called operational solely because JSON
validation passed.

## Security and privacy

Provider calls use bounded protocols and controlled egress. Source/evidence text
is minimized in audit artifacts; fingerprints and span identities do not by
themselves anonymize sensitive content. Benchmark-bank contamination is treated
as an integrity risk.

## Verification

- schema/parser/replay adversarial tests;
- candidate-blind vs cross-fitted leakage experiments;
- answerability/alignment/ambiguity/redundancy screening fixtures;
- artificial-crowd and human pilot recovery;
- item fit, information, DIF/local dependence and rater diagnostics;
- fixed-item linking/version drift evidence;
- lifecycle transition and immutable-release tests.

## Sources

American Educational Research Association, American Psychological Association,
& National Council on Measurement in Education. (2014). *Standards for
educational and psychological testing*.

Hashemi, H., Eisner, J., Rosset, C., Van Durme, B., & Kedzie, C. (2024).
LLM-Rubric and related multidimensional rubric-evaluation research provide a
recent AI-evaluation context; method-specific dynamic-rubric papers and evidence
are maintained in the corresponding doctoring records.

## Supersession criteria

Supersede if a separate reusable item-bank component is created with a clearer
bounded context, provided the measurement contracts, immutable linking history
and calibration evidence remain interoperable with fast-mlsirm.
