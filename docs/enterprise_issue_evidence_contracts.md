# Enterprise issue evidence contracts

`fast_mlsirm.scoring.enterprise_issue` provides the first provider-neutral domain
boundary for issue #404. The module stores content identities, exact source-span
offsets, epistemic roles, stakeholder perspectives, and candidate-intervention
provenance without retaining raw enterprise text.

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
Scoring, calibration, comparative ranking, expected utility, value of information,
and queue routing remain outside this slice.

## Interpretation limits

These contracts improve traceability and replay resistance; they do not establish
construct validity, evidence truth, causal identification, model fairness, or
high-stakes deployment readiness. Human review remains necessary wherever source
meaning, organizational values, legal rights, or material consequences are in
dispute.

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
