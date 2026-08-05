# Enterprise semantic issue proposal boundary design

## Goal

Convert untrusted provider proposals over authorized enterprise text into the
existing content-addressed enterprise evidence contracts without retaining raw
text, creating a provider-specific core dependency, or performing scoring,
calibration, ranking, utility, causal, or sentiment arithmetic.

This is the next bounded delivery slice for issue #536 and advances the larger
issue #404 enterprise intelligence vertical after the deterministic explicit-value
parser.

## Context and buyer-visible gap

The package can now preserve enterprise source records, exact evidence spans,
atomic issues, counterevidence, stakeholder perspectives, candidate
interventions, governed scoring requests, and deterministic explicit values. It
still lacks the trust boundary needed to integrate a semantic extractor. Without
that boundary, an LLM or smaller information-extraction model would need to
construct package-owned records directly, making source replay, span identity,
epistemic role separation, privacy, provider revision provenance, and malformed
output handling inconsistent across integrations.

The product needs a provider-neutral protocol whose output is treated as
untrusted data. Canonical package code—not the provider—must derive every
persisted fingerprint and reconstruct every accepted record.

## Design decision

Add `semantic_proposals.py` under
`fast_mlsirm.scoring.enterprise_issue`. The module exposes:

- runtime-checkable `EnterpriseSemanticIssueProvider`;
- deterministic `OfflineSemanticIssueFixtureProvider` for tests and disconnected
  use;
- `extract_enterprise_semantic_issues(...)` as the stable validation and
  compilation boundary.

The stable function returns a two-tuple:

```python
(
    tuple[AtomicIssueRecord, ...],
    tuple[StakeholderPerspective, ...],
)
```

It deliberately introduces no parallel issue, observation, result, report,
scoring, or decision schema.

## Source input boundary

Callers provide:

- a bounded iterable of exact `EnterpriseSourceRecord` values;
- a mapping from descriptive `source_id` to transient source text;
- one provider implementing the protocol.

Before provider execution, the package:

1. requires exact `EnterpriseSourceRecord` instances;
2. rejects duplicate source IDs and duplicate source-record fingerprints;
3. requires an exact text entry for every source and no unknown text keys;
4. requires every text value to be `str`;
5. verifies `len(source_text)` against `source_character_count` using Python
   Unicode-code-point indexing;
6. verifies SHA-256 over UTF-8 source bytes against
   `source_content_fingerprint`;
7. sorts source inputs by source-record fingerprint.

Source replay failure occurs before any provider callback. Raw text remains
transient and never enters a public record, error message, or metadata value.

## Provider protocol

The protocol exposes:

```python
provider_revision_fingerprint: str

def propose(
    self,
    *,
    sources: tuple[tuple[EnterpriseSourceRecord, str], ...],
) -> Iterable[Mapping[str, Any]]:
    ...
```

The revision fingerprint identifies the complete external extraction revision,
for example model, prompt, parser, schema, and adapter configuration. Core code
validates it as a complete SHA-256 digest but does not interpret provider
metadata or call a network.

The offline fixture provider stores a deeply frozen primitive proposal sequence,
returns a fresh primitive representation on every call, and is useful as an
independent standalone adapter and an exact test oracle. It performs no semantic
inference.

## Primitive proposal schema

Each proposal is an exact mapping with:

```text
issue_id
issue_family_id
issue_statement
assertions
metadata
```

`issue_statement` is transient source-independent issue wording. The package
normalizes no prose and stores only SHA-256 over its exact UTF-8 bytes as
`issue_content_fingerprint`.

Each assertion is an exact mapping with:

```text
source_id
start_offset
end_offset
assertion_kind
stakeholder_id
metadata
```

`stakeholder_id` must be null unless the assertion kind is
`stakeholder_value_judgment`; it is required for that kind. Provider output does
not supply span IDs, span fingerprints, source-record fingerprints,
counterevidence IDs, perspective IDs, or record fingerprints. The package
derives all of them.

Proposal and assertion collections are bounded before full materialization.
Mappings require exact key sets. Metadata is limited by existing deeply immutable
metadata controls and rejects keys that imply raw text, prompt content,
authorization values, credentials, tokens, API keys, passwords, or secrets.

## Canonical compilation

For every assertion, the package:

1. resolves the verified source by `source_id`;
2. validates bounded non-Boolean integer offsets;
3. requires a nonempty in-range span;
4. derives `span_content_fingerprint` from the exact UTF-8 source slice;
5. validates `EnterpriseAssertionKind`;
6. derives a descriptive content-addressed span ID;
7. reconstructs an exact `EvidenceSpanRecord`.

Assertions are sorted by source-record fingerprint, offsets, kind, and span
digest. Duplicate source occurrences and unresolved overlapping spans within one
issue fail closed.

The compiler then:

- places direct facts, supported inferences, and unresolved ambiguities in
  `AtomicIssueRecord.evidence_spans`;
- wraps counterevidence in exact `CounterevidenceRecord` values;
- converts stakeholder value judgments to separate `StakeholderPerspective`
  values;
- rejects an issue supported only by stakeholder value judgments;
- derives issue source-record fingerprints from accepted non-value-judgment and
  counterevidence spans;
- stores provider revision, assertion fingerprints, and perspective fingerprints
  only as package-managed audit metadata;
- reconstructs `AtomicIssueRecord` directly through its public contract.

Issue records and perspectives are returned in deterministic fingerprint order.
Duplicate issue IDs or duplicate issue-content fingerprints fail closed.

## Security, privacy, and failure behavior

The boundary treats provider output and source text as untrusted.

- Provider exceptions other than package-owned `AssessmentSpecError` are replaced
  with a stable `semantic_issue_provider_failure` error.
- Provider iterables that are not bounded, mappings with extra/missing fields,
  malformed identifiers, invalid assertion roles, unknown source IDs, bad
  offsets, duplicate or overlapping spans, or oversized output fail with stable
  non-reflective errors.
- Provider output is never accepted as a package-owned record and cannot use a
  subclass or mutation to bypass validation.
- Source content is never interpreted as instructions by the core; the module
  executes no shell, tool, model, network, or filesystem operation.
- Raw source text, transient issue statements, customer tokens, credentials,
  prompts, and authorization values must be absent from serialized output and
  public errors.

## MSA and standalone boundary

The module depends only on existing scoring contract helpers and the enterprise
contracts. It has no provider SDK, HTTP, database, queue, UI, contextual-
orchestrator, or deployment dependency. A service may wrap the protocol and call
an external model, while the standalone package can use the offline fixture or a
local adapter. Both paths receive identical canonical validation.

`contextual-orchestrator` belongs in an optional service adapter when a live LLM
provider is introduced. The core remains provider-neutral, and live tests must
use `NVIDIA_NIM_API_KEY` without changing the independent review-agent key
system.

## Testing

The focused suite must cover:

- all five epistemic assertion kinds in realistic mixed report, lead-note, and
  complaint fixtures;
- exact issue, span, source, counterevidence, perspective, and provider revision
  provenance;
- deterministic output across proposal, assertion, source, and mapping order;
- exact Unicode code-point offsets and UTF-8 span hashing;
- no persisted raw source text, issue statement, identifier token, prompt, or
  credential value;
- source replay rejection before provider invocation;
- issue-statement identity stability under neutral/sentiment wording changes
  outside selected evidence;
- sentence reordering with updated offsets preserving issue-content identity but
  changing visible source revision provenance;
- invalid types, Boolean offsets, unknown sources, empty spans, malformed roles,
  stakeholder-role mismatches, overlapping/duplicate spans, duplicate issues,
  excessive records, generator overflow, hostile callbacks, and metadata secret
  keys;
- runtime protocol and offline fixture behavior;
- 100% statement and branch coverage for the new module and complete public
  docstrings.

## Documentation and release boundary

Extend the enterprise evidence guide and add an authoritative changelog fragment.
Add a doctoring record that maps the design to ISO/IEC 42001:2023, NIST AI
600-1, and recent information-extraction findings. Render tracked
`CHANGELOG.md` and preserve all historical release sections.

This slice does not justify a version bump or release by itself. Exact-head
Python, Rust/PyO3, package, GPU-no-skip, fuzz, security, SAST, coverage,
docstring, changelog, review, and unresolved-thread gates remain mandatory.

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
(2024). Unexpected phenomenon: LLMs’ spurious associations in information
extraction. In *Findings of the Association for Computational Linguistics: ACL
2024* (pp. 9176–9190). Association for Computational Linguistics.
https://doi.org/10.18653/v1/2024.findings-acl.545
