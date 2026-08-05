# Enterprise semantic issue proposal doctoring record

## Purpose

This record documents the trust boundary introduced for issue #536, a bounded
slice of the issue #404 enterprise intelligence vertical. The boundary accepts
untrusted semantic issue proposals from a provider-neutral protocol, replays
exact authorized source revisions, and reconstructs only the existing enterprise
evidence contracts.

The implementation is not a sentiment analyzer, score generator, calibrator,
ranking engine, causal estimator, utility model, or autonomous action system.
It does not call a model or provider. Live model adapters remain optional service
components and must pass the same deterministic package boundary.

## Traceability decisions

### Exact source replay

Every `EnterpriseSourceRecord` is paired with transient source text. Before a
provider is invoked, the package verifies:

- exact descriptive source identity;
- Python Unicode-code-point character count;
- SHA-256 over the source text's exact UTF-8 bytes;
- exact agreement between declared source IDs and transient text keys.

This establishes reproducible input identity and prevents a provider result from
being rebound to a different source revision. It does not establish that the
source content is true.

### Provider-neutral proposals

`EnterpriseSemanticIssueProvider` carries a complete provider revision
fingerprint and returns primitive mappings. The package does not trust provider
span IDs, source fingerprints, issue fingerprints, counterevidence identifiers,
or perspective identifiers. It derives each persisted identity from verified
source text and normalized proposal fields.

This separation responds to published evidence that LLM structured output can be
inconsistent and that information-extraction labels can exhibit spurious
associations. A provider may generate candidates; deterministic package code owns
schema, source, offset, privacy, and epistemic validation.

### Epistemic role separation

Every assertion is mapped to one existing `EnterpriseAssertionKind`:

- direct fact;
- supported inference;
- counterevidence;
- unresolved ambiguity;
- stakeholder value judgment.

Counterevidence is wrapped as `CounterevidenceRecord`. Stakeholder value
judgments are emitted separately as `StakeholderPerspective`. An issue supported
only by a preference cannot cross the atomic-issue boundary. These controls
preserve the difference between what a source states, what a provider infers,
what conflicts with an issue, what remains unresolved, and what reflects an
organizational value.

### Privacy and prompt-injection boundary

Raw source text and transient issue statements are not serialized. Span content
is represented by UTF-8 SHA-256 and code-point offsets. Metadata rejects source,
prompt, authorization, credential, token, password, API-key, and secret-like
keys. Provider exceptions are redacted.

Source text is data only. The core executes no shell command, tool call, network
request, provider instruction, filesystem mutation, or model-generated code.
Prompt-injection defenses for a live LLM adapter belong in that adapter and do
not weaken this core boundary.

### Risk-management alignment

ISO/IEC 42001:2023 identifies traceability, transparency, reliability, and
continual risk management as AI-management concerns. NIST AI 600-1 applies the AI
RMF lifecycle to generative-AI risks. This implementation contributes bounded
technical controls for input identity, provenance, output validation, privacy,
and failure reporting. It does not claim formal conformity or certification.

## Verification contract

Merge requires one unchanged head to prove:

- realistic mixed-source and all-five-role compilation;
- source replay before provider execution;
- deterministic order and content identities;
- exact Unicode code-point offsets and UTF-8 span digests;
- absence of raw source, issue, prompt, customer, and credential values in public
  artifacts and errors;
- duplicate, overlap, source-drift, malformed-shape, stakeholder-role, bound,
  hostile-callback, and provider-exception rejection;
- runtime protocol substitution and disconnected fixture operation;
- 100% statement and branch coverage for the new production module;
- complete public docstrings;
- full Python, Rust/PyO3, package, GPU-no-skip, fuzz, Security Scan, and SAST
  success;
- changelog render parity and no unresolved review threads.

## Limitations and next work

The provider boundary does not validate semantic truth, issue completeness,
construct validity, fairness, judge reliability, temporal trajectory, stakeholder
utility, or intervention effects. Those claims require human-anchored data,
criterion-level multi-judge observations, Rust-backed measurement, recovery
simulation, leave-domain/customer/period/judge-family validation, and explicit
decision-policy evaluation.

The next issue #404 slice should compile accepted atomic issues into connected
criterion-level judge observations using the existing shared scoring contracts.
Live LLM testing, when introduced, must use `NVIDIA_NIM_API_KEY` through an
optional adapter or `contextual-orchestrator` without modifying the independent
review-agent credential chain.

## References

Autio, C., Schwartz, R., Dunietz, J., Jain, S., Stanley, M., Tabassi, E., Hall,
P., & Roberts, K. (2024). *Artificial intelligence risk management framework:
Generative artificial intelligence profile* (NIST AI 600-1). National Institute
of Standards and Technology. https://doi.org/10.6028/NIST.AI.600-1

International Organization for Standardization. (2023). *Information
technology—Artificial intelligence—Management system* (ISO/IEC Standard No.
42001:2023). https://www.iso.org/standard/42001.html

Li, Y., Ramprasad, R., & Zhang, C. (2024). A simple but effective approach to
improve structured language model output for information extraction. In
*Findings of the Association for Computational Linguistics: EMNLP 2024* (pp.
5081–5097). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2024.findings-emnlp.295

Zhang, W., Lu, W., Wang, J., Wang, Y., Chen, L., Jiang, H., Liu, J., & Ruan, T.
(2024). Unexpected phenomenon: LLMs' spurious associations in information
extraction. In *Findings of the Association for Computational Linguistics: ACL
2024* (pp. 9176–9190). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2024.findings-acl.545
