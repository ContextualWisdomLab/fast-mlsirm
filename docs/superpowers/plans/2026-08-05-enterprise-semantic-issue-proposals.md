# Enterprise semantic issue proposal boundary implementation plan

> **Execution rule:** Follow test-driven development. Keep the pull request draft
> until every exact-head verification gate succeeds.

## Goal

Implement issue #536 by compiling untrusted semantic issue proposals into the
existing enterprise source, evidence, atomic-issue, counterevidence, and
stakeholder-perspective contracts. Retain no raw source text or issue statement,
introduce no provider SDK, and perform no scoring, calibration, ranking, utility,
causal, sentiment, or queue arithmetic.

## Architecture constraints

- Add one focused module:
  `python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py`.
- Modify only the enterprise package exports, focused tests, enterprise evidence
  documentation, doctoring, changelog fragment, and rendered `CHANGELOG.md`.
- Do not add a parallel issue, observation, result, report, engine, assessment,
  rubric, calibration, or decision schema.
- Use only existing public contract constructors to create accepted records.
- Treat provider output as primitive untrusted data and reconstruct every record.
- Do not call a model or require `NVIDIA_NIM_API_KEY` in deterministic CI.
- All public symbols require complete docstrings.
- The new module requires 100% statement and branch coverage.

---

## Task 1: Define RED semantic proposal contracts

**Files**

- Create: `tests/test_scoring_enterprise_semantic_proposals.py`
- Create: `tests/test_scoring_enterprise_semantic_provider_boundary.py`

### Step 1: Add realistic source and proposal fixtures

Use three source families representing a report, sales lead note, and customer
complaint. Build exact `EnterpriseSourceRecord` values from UTF-8 text SHA-256 and
Python Unicode-code-point length.

### Step 2: Add desired public API tests

Import and exercise:

- `EnterpriseSemanticIssueProvider`
- `OfflineSemanticIssueFixtureProvider`
- `extract_enterprise_semantic_issues`
- bounded constants for issue proposals, assertions, issue-statement characters,
  and proposal metadata.

Expected initial failure: import errors because the module and exports do not
exist.

### Step 3: Add epistemic compilation tests

Prove direct fact, supported inference, unresolved ambiguity, counterevidence,
and stakeholder value judgment compile into the existing records with exact
roles and deterministic fingerprints.

### Step 4: Add privacy and metamorphic tests

Prove serialized output and errors exclude source text, transient issue text,
customer tokens, prompt content, credentials, and provider exception text. Prove
proposal/source/assertion order invariance, source-revision visibility, and issue
content identity stability where the transient issue statement is unchanged.

### Step 5: Add fail-closed provider tests

Cover source replay before callback, exact key sets, collection bounds, generator
overrun, duplicate/overlapping spans, duplicate issues, unknown sources, invalid
roles, stakeholder mismatches, metadata secret keys, provider exceptions, and
package-owned error preservation.

### Step 6: Observe RED

Run:

```bash
pytest -q \
  tests/test_scoring_enterprise_semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_provider_boundary.py
```

Record the expected missing-module/public-symbol failure on the exact test-only
head.

---

## Task 2: Implement source replay and provider protocol

**Files**

- Create: `python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py`

### Step 1: Add bounded constants and primitive helpers

Add descriptive constants for maximum proposal count, assertions per proposal,
issue-statement characters, and provider metadata collection sizes. Reuse
`bounded_values`, `freeze_metadata`, `thaw_json_value`, `descriptive_identifier`,
`fingerprint`, `enum_value`, and `assessment_error`.

### Step 2: Validate sources before provider execution

Require exact source records, unique IDs/fingerprints, exact mapping keys, string
text, code-point counts, and UTF-8 SHA-256 replay. Sort verified source pairs by
source-record fingerprint.

### Step 3: Add runtime protocol

Define a runtime-checkable protocol with `provider_revision_fingerprint` and
keyword-only `propose(sources=...)`. Validate the provider before callback.

### Step 4: Add offline fixture provider

Deep-freeze primitive proposals at construction, validate the provider revision,
and return a fresh primitive representation. Do not create package records in the
fixture provider.

### Step 5: Run the source/protocol subset

Expected: source and protocol tests pass; compilation tests remain red.

---

## Task 3: Implement canonical proposal compilation

**Files**

- Modify: `python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py`

### Step 1: Validate exact proposal and assertion shapes

Require exact mapping keys and bounded iterable materialization. Validate issue
and family IDs, nonempty bounded issue statement, assertion mapping shape,
metadata keys, assertion kind, stakeholder-role relationship, and bounded
non-Boolean offsets.

### Step 2: Reconstruct evidence spans

Resolve exact sources, derive UTF-8 span fingerprints and descriptive span IDs,
and construct exact `EvidenceSpanRecord` values. Reject duplicates and unresolved
overlaps after deterministic sorting.

### Step 3: Reconstruct issue components

- ordinary assertions → `evidence_spans`;
- counter assertions → `CounterevidenceRecord`;
- value judgments → `StakeholderPerspective`.

Derive issue-content fingerprint from the exact transient issue statement. Reject
issues supported only by stakeholder judgments.

### Step 4: Reconstruct atomic issues

Build `AtomicIssueRecord` through the existing constructor. Add only managed
provider revision and assertion/perspective audit fingerprints to metadata. Return
issue and perspective tuples in deterministic fingerprint order and reject
duplicate issue IDs/content revisions.

### Step 5: Harden callback failures

Preserve package-owned `AssessmentSpecError`; redact every other callback
exception as `semantic_issue_provider_failure`. Never reflect source/provider
values in public messages.

### Step 6: Run focused GREEN

Run both focused test modules. Fix production code rather than weakening tests.

---

## Task 4: Export and document the boundary

**Files**

- Modify: `python/fast_mlsirm/scoring/enterprise_issue/__init__.py`
- Modify: `docs/enterprise_issue_evidence_contracts.md`
- Create: `docs/doctoring/enterprise-semantic-issue-proposals.md`
- Create: `docs/changelog.d/enterprise-semantic-issue-proposals.md`
- Modify: `CHANGELOG.md`

### Step 1: Export public symbols

Keep the current explicit import and pinned `__all__` pattern.

### Step 2: Document product and security boundaries

Explain exact source replay, code-point offsets, transient issue text, provider
revision provenance, role separation, whole-batch fail-closed behavior, and the
absence of scoring/causal/utility claims.

### Step 3: Add APA 7 doctoring

Map implementation decisions to ISO/IEC 42001:2023, NIST AI 600-1, and the two
2024 information-extraction papers recorded in the design.

### Step 4: Render authoritative changelog

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

Preserve every historical release section and comparison link.

---

## Task 5: Verify and merge

### Step 1: Focused quality gates

```bash
ruff format --check \
  python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_provider_boundary.py
ruff check \
  python/fast_mlsirm/scoring/enterprise_issue/semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_provider_boundary.py
pytest -q \
  tests/test_scoring_enterprise_semantic_proposals.py \
  tests/test_scoring_enterprise_semantic_provider_boundary.py \
  --cov=fast_mlsirm.scoring.enterprise_issue.semantic_proposals \
  --cov-branch --cov-fail-under=100
python scripts/check_docstring_coverage.py
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

### Step 2: Full repository gates

Run full Python, Rust workspace, PyO3, package, explicit GPU-no-skip, fuzz,
Security Scan, and SAST gates on one unchanged head.

### Step 3: Review loop

Inspect every current-head review and unresolved thread. Implement only verified
findings, rerun all affected tests and exact-head checks, and request final
CodeRabbit/OpenCode/Noema review.

### Step 4: Merge

Remove draft/blocking labels only after all exact-head checks, independent review,
coverage/docstrings, changelog parity, and unresolved-thread gates succeed. Use
policy-compliant squash/auto-merge. Confirm the open PR count afterward and
continue issue #404 with criterion-level observation orchestration.
