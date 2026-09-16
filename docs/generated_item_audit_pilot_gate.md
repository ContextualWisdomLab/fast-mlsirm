# Generated-item audit and pilot-admission gate

## Purpose

The provider-output parser establishes structural, provenance, answer-key, and
verbatim source-span integrity. It does not establish that a generated item is
unambiguous, criterion-atomic, safe to pilot, psychometrically valid, fair, or
operationally deployable. This module adds the next fail-closed governance
boundary without introducing provider SDKs or Python psychometric arithmetic.

```text
generation contract
  -> strict provider-output parser
  -> immutable GeneratedItemCandidate
  -> deterministic CandidateAuditReport
  -> CandidateScreeningResult
  -> replay-verified pilot admission
  -> factory-sealed PilotCandidateRecord
```

## Lifecycle contract

Generated content may occupy only these states in this slice:

1. `draft`: the candidate has at least one blocking or review-required audit
   finding;
2. `audited`: deterministic screening has no unresolved blocking or
   review-required finding; and
3. `pilot`: an unchanged audited candidate is admitted only after a complete,
   pilot-eligible `CandidateScreeningResult` is bound to the exact
   candidate/audit fingerprints.

An audit report cannot assign `pilot` directly. `build_pilot_candidate_record`
rejects a stale report, a changed candidate, any report that remains in
`draft`, or a report labelled with an audit policy not implemented by the
installed package. Before admission, it reruns the named package policy over
the exact candidate and requires the complete report fingerprint to match.
This prevents callers from constructing a clean-looking report that hides real
findings. It then requires a factory-created screening result with exactly one
decision for every governed semantic dimension, rejects review-required or
blocking screening decisions, and binds that result fingerprint into the
pilot record. The public `PilotCandidateRecord` also rejects ordinary direct
construction, so supported callers must pass through the replay-verified
factory. The pilot record requires descriptive nonnumeric identifiers for the
pilot study, query/testlet, generator family, judge policy, occasion, item,
blueprint, and rubric, plus the exact screening-result identity.

The screening-bound pilot record is a distinct serialized contract:
`schema_version="2.0"` and the public admission/audit policy is
`AUDIT_POLICY_VERSION="2.0.0"`. The shared rubric/blueprint schema remains at
its own version; pilot admission does not silently re-version unrelated rubric
contracts. Legacy pilot-record payloads advertising `schema_version="1.0"`
lack the mandatory screening-result binding and are rejected explicitly rather
than being interpreted as current records. This package does not expose a
pilot-record deserializer/migration API, so callers that retain legacy payloads
must preserve them as historical evidence and create a new current admission
through the verified screening-and-audit path when a current pilot record is
required.

The factory seal is an API-governance boundary, not a cryptographic capability
inside a hostile Python process. Downstream services must verify the complete
candidate, audit-report, and pilot-record fingerprints instead of trusting an
in-memory object solely because it has the expected class name.

## Deterministic findings

The current policy emits bounded, redacted findings for:

- explicit instruction-override and prompt-injection markers;
- duplicate normalized option text and aggregate answer patterns;
- duplicated score evidence or cross-level rubric indicators;
- potentially non-atomic observable indicators;
- normalized duplicate source-attribution spans;
- deterministic stem ambiguity markers and multiple questions;
- declared safety notes; and
- an adversarial finding count above the report budget.

Finding messages never copy candidate or source text. Audit output is
content-addressed with a complete SHA-256 fingerprint and a descriptive
128-bit public handle. Identical candidates and policy versions produce
identical reports. The public policy identity and version are fixed constants
for the installed package so policy labels cannot drift independently of the
implemented logic.

## Scientific and product boundary

These lexical, structural, and semantic-screening records are governance
signals, not psychometric validity evidence. A pilot-eligible screening result
does **not** claim that the item is calibrated, scoreable, unbiased, or
suitable for high-stakes use. Human review, pilot response collection,
Rust-backed measurement, DIF/fairness analysis, recovery studies, and validity
evidence remain mandatory downstream gates.

The next issue #407 slices should add immutable audit-resolution evidence,
provider-neutral evaluator adapters and offline fixtures for semantic duplicate
and answerability protocols, and a deterministic conversion from admitted
candidates into the existing respondent/item/rater observation contracts used
by facet, MIRT, testlet, DIF, and G-theory APIs.
