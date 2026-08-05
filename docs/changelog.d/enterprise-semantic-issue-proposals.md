# Evidence-grounded enterprise semantic issue proposals

## Added

- Added a provider-neutral semantic issue proposal boundary that verifies exact
  `EnterpriseSourceRecord` content fingerprints and Python Unicode-code-point
  counts before any provider callback.
- Added a runtime-checkable provider protocol and deterministic disconnected
  fixture provider whose untrusted primitive output is reconstructed into the
  existing `AtomicIssueRecord`, `EvidenceSpanRecord`, `CounterevidenceRecord`,
  and `StakeholderPerspective` contracts.
- Added strict separation of direct facts, supported inferences,
  counterevidence, unresolved ambiguities, and stakeholder value judgments;
  stakeholder preferences alone cannot create a factual atomic issue.
- Added package-derived source-span, issue-content, counterevidence, perspective,
  and provider-revision provenance with deterministic order, bounded output,
  duplicate and overlap rejection, and exact UTF-8 span fingerprints.
- Added privacy and security controls that retain no raw source or transient issue
  text, reject source/prompt/credential/secret-like metadata, redact provider
  failures, and treat enterprise text solely as data without executing tools,
  network calls, or provider instructions.
- Added realistic mixed-source, protocol, source-replay, Unicode-offset, privacy,
  ordering, metamorphic, resource-bound, hostile-provider, and fail-closed tests,
  plus APA 7th standards and research traceability.
