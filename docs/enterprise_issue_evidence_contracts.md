# Enterprise issue evidence contracts

`fast_mlsirm.scoring.enterprise_issue` provides the first provider-neutral domain
boundary for issue #404. The module stores content identities, exact source-span
offsets, epistemic roles, stakeholder perspectives, candidate-intervention
provenance, and criterion-level request provenance without retaining raw
enterprise text.

## Contract boundary

The initial slice deliberately separates five assertion kinds:

- directly stated facts;
- supported inferences;
- counterevidence;
- unresolved ambiguities;
- stakeholder value judgments.

`EvidenceSpanRecord.to_evidence_reference()` compiles each span into the existing
shared `EvidenceReference` contract. Direct facts and supported inferences map to
supporting evidence, counterevidence maps to counter evidence, and ambiguities or
value judgments map to contextual evidence. This mapping does not convert an
inference into a fact or a preference into a materiality estimate.

The canonical records contain no source text, complaint text, lead notes,
customer names, or proposed-action text. Callers retain those values in an
authorized source system and pass SHA-256 content fingerprints plus offsets.
Sensitive metadata fields already prohibited by the shared scoring contract are
rejected here as well.

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
already declared by the issue. Caller metadata cannot overwrite the managed
enterprise provenance fields.

This compiler performs deterministic validation and marshaling only. Existing
`ScoringEngine`, `ScoreObservation`, and `ScoringResult` contracts remain the only
execution and output boundaries. Engines must cite the returned shared evidence
references on non-abstained observations under the existing scoring policy.

## Interpretation limits

These contracts improve traceability and replay resistance; they do not establish
construct validity, evidence truth, causal identification, model fairness, or
high-stakes deployment readiness. The request compiler performs no sentiment
analysis, latent measurement, calibration, comparative ranking, expected utility,
value-of-information, intervention-effect, or queue-routing arithmetic.

Successful compilation confirms only schema and provenance consistency. It does
not produce a score, determine evidence sufficiency, recommend an intervention,
or complete human review.

Human review remains necessary wherever source meaning, organizational values,
legal rights, or material consequences are in dispute. A candidate intervention
is a caller-supplied hypothesis, not evidence of an identified causal effect.

ISO/IEC 42001:2023 remains published as Edition 1. NIST reports that AI RMF 1.0
is under revision as of August 2026, so this module cites the current published
framework without assuming that its terminology or profiles are frozen.

## References

Autio, C., Schwartz, R., Dunietz, J., Jain, S., Stanley, M., Tabassi, E., Hall,
P., & Roberts, K. (2024). *Artificial intelligence risk management framework:
Generative artificial intelligence profile* (NIST AI 600-1). National Institute
of Standards and Technology. https://doi.org/10.6028/NIST.AI.600-1

International Organization for Standardization. (2023). *Information
technology—Artificial intelligence—Management system* (ISO/IEC Standard No.
42001:2023). https://www.iso.org/standard/81230.html

National Institute of Standards and Technology. (2023). *Artificial intelligence
risk management framework (AI RMF 1.0)* (NIST AI 100-1).
https://doi.org/10.6028/NIST.AI.100-1
