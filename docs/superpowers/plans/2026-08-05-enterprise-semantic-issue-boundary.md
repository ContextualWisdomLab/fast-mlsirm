# Enterprise semantic-issue provider boundary

## Objective

Advance issue #404 with the smallest provider-neutral semantic extraction slice
that can be reviewed independently after the deterministic explicit-value parser
and governed criterion-observation adapter. The slice lets offline fixtures,
human analysts, and future external engines propose the existing
`AtomicIssueRecord` contract without introducing a second issue, evidence,
observation, scoring, or engine schema.

## Architectural boundary

The implementation belongs under `fast_mlsirm.scoring.enterprise_issue` and
reuses:

- `EnterpriseSourceRecord` as the exact source-revision contract;
- `EvidenceSpanRecord` and `EnterpriseAssertionKind` for every semantic assertion;
- `CounterevidenceRecord` for counterevidence separation;
- `AtomicIssueRecord` as the only accepted semantic issue output; and
- the shared scoring request, observation, result, and engine contracts for later
  criterion-level scoring.

The package must not import a provider SDK. Python may validate, replay,
canonicalize, redact, and marshal this boundary. This slice adds no likelihood,
gradient, Hessian, optimization, scoring, ranking, or utility arithmetic.

## Public API

Implement:

- runtime-checkable `EnterpriseAtomicIssueExtractor`;
- `extract_enterprise_atomic_issues()` as the fail-closed public entry point;
- bounded `MAX_ENTERPRISE_ATOMIC_ISSUES`; and
- `StaticEnterpriseIssueExtractor` as an offline fixture and integration adapter,
  not a semantic language model or production default.

No competing extraction-request or semantic-assertion record is permitted where
the existing source, span, counterevidence, and atomic-issue contracts suffice.

## Trust-boundary requirements

The public entry point must:

1. consume source records through bounded iteration;
2. require exact `EnterpriseSourceRecord` values, unique source IDs, unique source
   record fingerprints, one schema version, and deterministic ordering;
3. require an exact dictionary whose keys equal the declared source IDs;
4. replay transient source text against its Python character count, UTF-8
   encodability, and SHA-256 content fingerprint;
5. invoke only an object satisfying the provider-neutral protocol;
6. redact every provider exception, including structured domain exceptions;
7. accept only a bounded exact tuple of exact `AtomicIssueRecord` values;
8. reconstruct fresh canonical issue, evidence-span, and counterevidence objects;
9. bind every referenced source fingerprint and source ID to the same supplied
   source revision;
10. verify every code-point span offset and SHA-256 fingerprint over the exact
    UTF-8 slice;
11. preserve all five assertion kinds and wrapped counterevidence;
12. reject duplicate issue fingerprints, duplicate issue IDs, duplicate
    family/revision pairs, overlapping or duplicated nested spans, malformed
    nested records, and oversized outputs;
13. return deterministic content order; and
14. retain no raw source text or clear-text semantic issue text in public records,
    exceptions, metadata, logs, or serialized output.

Use stable structured error codes and JSON paths. Never reflect untrusted values
in error messages.

## Scientific and product limits

An accepted record proves only that an extractor proposed one canonical
issue/evidence structure whose spans replay against exact source revisions. It
does not prove that the issue is true, complete, material, probable, causally
related to an outcome, fair, construct-valid, or suitable for intervention
automation. An inference is not converted into a fact, and counterevidence is not
assumed weaker than supporting evidence.

The fixture extractor must not be described as a semantic model. Human validation
and held-out provider evaluation remain prerequisites for product claims.

## Validation

Tests must provide complete statement and branch coverage and demonstrate:

- deterministic output under source-record and issue reordering;
- exact source replay, Unicode code-point offsets, and UTF-8 span fingerprints;
- survival of all five assertion kinds without epistemic collapse;
- fresh canonical reconstruction and rejection of mutated or subclassed nested
  records;
- provider exception redaction and privacy preservation;
- missing, extra, duplicated, mismatched, oversized, or unexpectedly prolific
  source/output collections failing before unbounded consumption;
- source ID/fingerprint pair consistency;
- changed text, offsets, or span bytes invalidating replay;
- duplicate and overlapping nested evidence rejection;
- sentiment-only source text creating no issue in the deterministic fixture path;
  and
- stable package exports and serialized shapes.

Run Ruff, focused and repository tests, branch coverage, changelog parity,
packaging, Security Scan, SAST, and exact-head acceptance gates. Every public
object requires a complete docstring.

## Documentation and changelog

Update `docs/enterprise_issue_evidence_contracts.md` with provider trust,
source/span replay, fixture limitations, exception redaction, and conservative
interpretation limits. Add an authoritative changelog fragment and render
`CHANGELOG.md`. Do not bump a version or publish a release for this isolated
issue #404 slice.

## Review discipline

Keep the pull request draft until its exact current head has no unresolved valid
human, CodeRabbit, security, Dependabot, or automated feedback and all required
gates pass. Do not leave temporary workflows, triggers, credentials, raw source
fixtures, or generated artifacts in the final tree.
