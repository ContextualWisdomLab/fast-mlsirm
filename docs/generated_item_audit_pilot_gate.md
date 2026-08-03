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
  -> replay-verified pilot admission
  -> immutable PilotCandidateRecord
```

## Lifecycle contract

Generated content may occupy only these states in this slice:

1. `draft`: the candidate has at least one blocking or review-required audit
   finding;
2. `audited`: deterministic screening has no unresolved blocking or
   review-required finding; and
3. `pilot`: an unchanged audited candidate is admitted through an exact
   candidate/audit fingerprint binding.

An audit report cannot assign `pilot` directly. `build_pilot_candidate_record`
rejects a stale report, a changed candidate, any report that remains in
`draft`, or a report labelled with an audit policy not implemented by the
installed package. Before admission, it reruns the named package policy over
the exact candidate and requires the complete report fingerprint to match.
This prevents callers from constructing a clean-looking report that hides real
findings. The pilot record requires descriptive nonnumeric identifiers for the
pilot study, query/testlet, generator family, judge policy, occasion, item,
blueprint, and rubric.

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

These lexical and structural checks are governance signals, not semantic
validity evidence. A clean report does **not** claim that the item is
answerable, construct-relevant, unbiased, calibrated, scoreable, or suitable
for high-stakes use. Human review, pilot response collection, Rust-backed
measurement, DIF/fairness analysis, recovery studies, and validity evidence
remain mandatory downstream gates.

The next issue #407 slices should add immutable audit-resolution evidence,
semantic duplicate and answerability protocols with offline fixtures, and a
deterministic conversion from admitted candidates into the existing
respondent/item/rater observation contracts used by facet, MIRT, testlet,
DIF, and G-theory APIs.
