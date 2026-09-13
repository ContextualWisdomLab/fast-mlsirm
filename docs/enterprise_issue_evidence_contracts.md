# Enterprise issue evidence contracts

`fast_mlsirm.scoring.enterprise_issue` provides the first provider-neutral domain
boundary for issue #404. The module stores content identities, exact source-span
offsets, epistemic roles, stakeholder perspectives, candidate-intervention
provenance, deterministic explicit values, semantic issue proposals,
criterion-level request provenance, and governed criterion observations without
retaining raw enterprise text.

## Contract boundary

The domain deliberately separates five assertion kinds:

- directly stated facts;
- supported inferences;
- counterevidence;
- unresolved ambiguities; and
- stakeholder value judgments.

`EvidenceSpanRecord.to_evidence_reference()` compiles each span into the existing
shared `EvidenceReference` contract. Direct facts and supported inferences map to
supporting evidence, counterevidence maps to counter evidence, and ambiguities or
value judgments map to contextual evidence. This mapping does not convert an
inference into a fact or a preference into a materiality estimate.

Canonical records contain no source text, complaint text, lead notes, customer
names, or proposed-action text. Callers retain those values in an authorized
source system and pass SHA-256 content fingerprints plus offsets. Sensitive
metadata fields already prohibited by the shared scoring contract are rejected
here as well.

## Deterministic explicit-value parser

`DeterministicExplicitValueParser` adds a narrow auditable boundary for values
that are already explicit in authorized source text. Its first grammar recognizes:

- Gregorian calendar dates in extended `YYYY-MM-DD` form;
- deadlines marked by `due`, `deadline`, `by`, or `no later than`;
- exact nonnegative decimal amounts preceded by a caller-allowlisted uppercase
  three-letter currency code;
- positive recurrence counts per day, week, month, quarter, or year; and
- customer or account identifiers introduced by an explicit identifier label.

The parser verifies transient text against the exact
`EnterpriseSourceRecord.source_content_fingerprint` and Python string character
count before extraction. Match offsets are Python Unicode-code-point indices,
which are appropriate for replaying slices of the same Python `str`; they are not
UTF-8 byte offsets or user-perceived grapheme-cluster positions. Every persisted
span also carries SHA-256 over the exact UTF-8 bytes of the matched slice.

Deadline matches supersede the calendar-date match embedded inside the same
marked deadline. Any other accepted overlap fails closed rather than multiplying
one occurrence into several evidence records. Validation is whole-source rather
than per-span: any date-shaped candidate that is not a real Gregorian date, or
any labeled customer/account identifier that is empty, malformed, or oversized,
rejects the complete parse. Output order and parser revision identity are
deterministic and independent of caller currency-code ordering.

Money normalization constructs `Decimal` directly from accepted text after
removing validated grouping commas; it never passes through binary floating
point. Calendar dates are validated with `date.fromisoformat()` only after the
strict extended-date grammar has matched. Currency membership remains a caller
governance responsibility: an uppercase three-letter token is accepted only when
it occurs in the parser's explicit allowlist, and the parser does not claim that a
caller-supplied list is complete or current.

Clear-text customer and account identifiers are replaced with SHA-256 before a
public record is constructed. Raw source text and clear-text identifiers are not
retained in `ExplicitValueRecord`, its evidence projection, or serialized output.
`ExplicitValueRecord.to_evidence_span()` marks an exact occurrence as a directly
stated fact. This means only that the matched text was present in the verified
source revision; it does not establish that the value is true, current, material,
probable, decision-relevant, or causally related to an outcome.

The parser is deliberately not a semantic issue extractor. It performs no
sentiment analysis, inference, scoring, calibration, ranking, utility arithmetic,
causal estimation, or queue routing. Custom parser output is bounded,
canonicalized, and rebound to the exact verified source revision and span bytes
before it can cross the public API. Arbitrary provider exceptions are redacted.
Explicit-value caller metadata is restricted to the declared offset unit.

## Semantic issue extraction boundary

`extract_enterprise_atomic_issues()` is the provider-neutral trust boundary for
semantic issue proposals. It accepts exact source records, transient source text,
and an `EnterpriseAtomicIssueExtractor`, then returns only fresh canonical
`AtomicIssueRecord` values. The package imports no provider SDK and provides no
default production semantic model.

Before a provider runs, the boundary consumes source records with a fixed cap,
reconstructs exact source contracts, rejects duplicate source identities, requires
an exact source-text dictionary, and replays every Python character count and
SHA-256 fingerprint over valid UTF-8. Providers receive deterministic source
record order and a read-only transient text mapping. Every provider exception,
including package-domain exceptions, is replaced with a fixed redacted boundary
error.

Provider output must be an exact bounded tuple of exact `AtomicIssueRecord`
values. Each issue, evidence span, and counterevidence record is reconstructed as
a fresh canonical instance. Every source ID and source-record fingerprint must
name the same verified packet revision, and every span must replay its exact
code-point offsets and SHA-256 fingerprint over the corresponding UTF-8 slice.
Nested subclasses, malformed or mutated records, overlapping spans, duplicate
issues, duplicate logical issue IDs, and duplicate family/content revisions fail
closed. Returned issues use deterministic content order and retain no raw text.

`StaticEnterpriseIssueExtractor` is an offline fixture and integration adapter. It
returns only caller-declared issue records and performs no NLP, sentiment
analysis, inference, generation, ranking, or automatic issue discovery. Adding
sentiment-only text cannot create an issue in this path. Acceptance proves only
that a provider proposed one replayable canonical structure, not that the issue
is true, complete, material, probable, fair, construct-valid, or operationally
useful. Human validation and held-out provider evaluation remain prerequisites
for product claims.

## Replay and provenance

An `AtomicIssueRecord` binds one issue-content revision to declared source-record
fingerprints. Every supporting or contextual span must name one of those exact
source revisions. Counterevidence is wrapped separately and must name the same
issue-content fingerprint. Input order is canonicalized before content identities
are computed.

`StakeholderPerspective` requires an explicit stakeholder-value-judgment span.
`CandidateIntervention` records a candidate action revision and affected
stakeholder identities but makes no claim that the action causes an outcome.

`enterprise_issue_evidence_references()` returns the exact issue,
counterevidence, and stakeholder-perspective spans as shared
`EvidenceReference` values. It rejects duplicate references rather than silently
multiplying the apparent evidence weight. Complete provenance improves replay
and auditability but is not itself evidence that a claim is accurate or strong.

## Criterion-level scoring request

`build_enterprise_issue_scoring_request()` compiles one accepted
`AtomicIssueRecord` into the authoritative shared `ScoringRequest` contract:

- `AtomicIssueRecord.issue_id` becomes the shared respondent identity;
- `AtomicIssueRecord.issue_content_fingerprint` becomes the exact response
  revision fingerprint;
- the caller supplies exact response character and unit counts without passing
  raw text;
- the request is fixed to criterion-level granularity;
- assessment, rubric, task revision, task family, occasion, criterion, and engine
  authorization provenance remain governed by the shared scoring stack; and
- issue, source, evidence-span, counterevidence, perspective, intervention, and
  epistemic-role fingerprints are written into package-managed immutable
  metadata.

Stakeholder perspectives and candidate interventions must name the exact issue
content revision. Perspective evidence must also reference a source revision
already declared by the issue. Caller metadata cannot overwrite managed
enterprise provenance fields.

This compiler performs deterministic validation and marshaling only. Existing
`ScoringEngine`, `ScoreObservation`, and `ScoringResult` contracts remain the only
execution and output boundaries.

## Criterion-level enterprise observations

`build_enterprise_issue_score_observation()` compiles one engine outcome into the
existing shared `ScoreObservation` contract. It accepts only a request produced by
the enterprise request compiler and only evidence references already declared by
that exact request revision. The adapter canonicalizes evidence ordering and
writes exact issue and selected-evidence fingerprints plus role counts into
package-managed confidence metadata.

Every non-abstained enterprise observation must retain supporting evidence. When
the issue declares counterevidence, the observation must also retain at least one
counterevidence reference rather than silently omitting contradictory evidence.
An engine that cannot satisfy those conditions must abstain with a stable reason
code. Abstention may retain no evidence and is not converted into a low score.
Caller confidence metadata cannot overwrite managed enterprise provenance.

The adapter does not calculate a rating or confidence value. Score categories,
criterion coverage, terminal-state semantics, engine identity, assessment and
rubric replay, and result completeness remain governed by the shared scoring
contracts. Evidence presence is an auditability condition, not proof that a score
is accurate, reliable, fair, valid, calibrated, or decision-ready.

## Interpretation limits

These contracts improve traceability and replay resistance; they do not establish
construct validity, evidence truth, causal identification, model fairness, or
high-stakes deployment readiness. The extraction, request, and observation
adapters perform no latent measurement, calibration, comparative ranking,
expected utility, value-of-information, intervention-effect, or queue-routing
arithmetic.

The first provider-neutral decision-support arithmetic boundary is exposed
separately as `fast_mlsirm.decision_support`. It accepts only explicit
caller-supplied state probabilities, action utilities, intervention costs, and
an optional joint state/signal distribution. Its Rust core computes expected
net intervention value, EVPI, and EVSI; it does not infer organizational
utilities, causal effects, or queue policy from issue text or scores. The issue
adapters remain responsible for binding any future decision result to their
content-addressed provenance.

Successful compilation confirms only schema and provenance consistency. It does
not determine evidence sufficiency, recommend an intervention, or complete human
review. A persisted score remains one fallible judge observation that requires
subsequent calibration and validation.

Human review remains necessary wherever source meaning, organizational values,
legal rights, or material consequences are in dispute. A candidate intervention
is a caller-supplied hypothesis, not evidence of an identified causal effect.

ISO/IEC 42001:2023 remains published as Edition 1. ISO 8601-1:2019 and ISO
4217:2015 are the published standards used to describe accepted date and
currency-code forms; later amendments, maintenance updates, or replacement
editions must be evaluated before changing parser grammar or allowlists. NIST
reports that AI RMF 1.0 is under revision as of August 2026, so this module cites
the current published framework without assuming that its terminology or
profiles are frozen.

## References

Autio, C., Schwartz, R., Dunietz, J., Jain, S., Stanley, M., Tabassi, E., Hall,
P., & Roberts, K. (2024). *Artificial intelligence risk management framework:
Generative artificial intelligence profile* (NIST AI 600-1). National Institute
of Standards and Technology. https://doi.org/10.6028/NIST.AI.600-1

International Organization for Standardization. (2015). *Codes for the
representation of currencies* (ISO Standard No. 4217:2015).
https://www.iso.org/standard/64758.html

International Organization for Standardization. (2019). *Date and time—
Representations for information interchange—Part 1: Basic rules* (ISO Standard
No. 8601-1:2019). https://www.iso.org/standard/70907.html

International Organization for Standardization. (2023). *Information
technology—Artificial intelligence—Management system* (ISO/IEC Standard No.
42001:2023). https://www.iso.org/standard/81230.html

National Institute of Standards and Technology. (2023). *Artificial intelligence
risk management framework (AI RMF 1.0)* (NIST AI 100-1).
https://doi.org/10.6028/NIST.AI.100-1

Python Software Foundation. (2025). *Python 3.13 standard library: `datetime`,
`hashlib`, `types`, and `typing`*. https://docs.python.org/3.13/

The Unicode Consortium. (2025). *Unicode text segmentation* (Unicode Standard
Annex No. 29, Revision 47). https://www.unicode.org/reports/tr29/
