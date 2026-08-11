# Governed Item-Bank Lifecycle Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an immutable, content-addressed, fail-closed lifecycle contract that bridges an already verified `PilotCandidateRecord` into calibrated, approved, active, suspended, reactivated, and retired item-bank states without adding a database or a new numerical estimator.

**Architecture:** Existing `fast_mlsirm.rubric` audit and pilot records remain the only pre-pilot source of truth. A new `item_bank` module owns post-admission lifecycle records, evidence references, transition rules, and exact provenance replay. Numerical evidence remains an opaque fingerprinted reference to existing/future Rust-backed calibration, fit, DIF, information, linking, exposure, and drift outputs.

**Tech Stack:** Python 3.10+, frozen dataclasses, enums, canonical SHA-256 helpers already present in `fast_mlsirm.rubric.models`, pytest, existing rubric candidate/audit fixtures.

## Global Constraints

- Work only in `ContextualWisdomLab/fast-mlsirm` and preserve standalone installation.
- Reuse `PilotCandidateRecord`; do not create another candidate, rubric, assessment, scoring, or pilot schema.
- Add no ORM, database, provider SDK, network call, numerical estimator, likelihood, optimizer, workflow, dependency, version bump, or hosted-product concern.
- Every identifier uses two-or-more-token lower `snake_case`; exact content identities use complete lower-hex SHA-256 fingerprints.
- Operational versions are immutable; transitions create new records rather than mutating earlier records.
- Psychometric discrimination and policy criticality remain separate.
- Raw source, response, prompt, provider output, rejected content, and secrets are not lifecycle fields or error text.
- Added public code requires complete docstrings and 100% statement/branch coverage.

---

### Task 1: Pin the Public Lifecycle and Transition Contract

**Files:**
- Create: `tests/test_rubric_item_bank_lifecycle.py`
- Create: `docs/doctoring/governed_item_bank_lifecycle.md`

**Interfaces:**
- Consumes: `PilotCandidateRecord`, `CandidateLifecycleState.PILOT`, canonical validators from `fast_mlsirm.rubric.models`.
- Produces: test-locked names `ItemBankLifecycleState`, `ItemBankEvidenceKind`, `ItemBankEvidenceReference`, `ItemBankLifecycleRecord`, `PolicyCriticality`, `ItemBankLifecycleError`, `build_item_bank_pilot_record`, and `transition_item_bank_record`.

- [ ] **Step 1: Write failing public-contract tests**

Cover deterministic pilot admission, exact provenance, immutable fingerprints, direct-construction rejection, required evidence by transition, cumulative evidence, order-invariant identities, criticality preservation, no skipped/backward/no-op transitions, terminal retirement, active-use scope, stale-record mutation detection, exact-type validation, redacted errors, and JSON-compatible serialization.

- [ ] **Step 2: Run focused tests and observe RED**

Run: `pytest -q tests/test_rubric_item_bank_lifecycle.py`

Expected: collection failure because `fast_mlsirm.rubric.item_bank` does not yet exist.

- [ ] **Step 3: Commit the RED contract and doctoring**

Commit only tests, this plan, and method/interpretation doctoring. Keep the PR Draft.

### Task 2: Implement Sealed Evidence and Lifecycle Records

**Files:**
- Create: `python/fast_mlsirm/rubric/item_bank.py`
- Modify: `python/fast_mlsirm/rubric/__init__.py`
- Test: `tests/test_rubric_item_bank_lifecycle.py`

**Interfaces:**
- `ItemBankEvidenceReference(evidence_kind, evidence_id, evidence_fingerprint)` validates one source-text-free evidence identity.
- `build_item_bank_pilot_record(pilot_record, *, item_version, policy_criticality)` creates the only initial `PILOTING` record.
- `transition_item_bank_record(current_record, target_state, *, evidence_references, transition_reason_id, approved_use_ids=())` creates a new record linked to the exact previous record fingerprint.

- [ ] **Step 1: Add the enum and redacted error surface**

States: `piloting`, `calibrated`, `approved`, `active`, `suspended`, `retired`.

Evidence kinds: `calibration`, `item_fit`, `dif`, `item_information`, `linking`, `exposure`, `drift`, `approval`, `suspension`, `retirement`.

Criticality: `ordinary`, `required`, `conjunctive_gate`.

- [ ] **Step 2: Implement immutable evidence references**

Require exact enum values, descriptive IDs, and complete SHA-256 fingerprints. Serialize no raw evidence.

- [ ] **Step 3: Implement factory-sealed lifecycle records**

Store the creation-time record fingerprint, derive a 128-bit public handle, and verify the stored fingerprint against freshly canonicalized content before every transition so `object.__setattr__` mutation fails closed.

- [ ] **Step 4: Implement initial pilot record creation**

Require exact `PilotCandidateRecord`, `CandidateLifecycleState.PILOT`, complete pilot/candidate/audit provenance, semantic item version, and no lifecycle evidence yet.

- [ ] **Step 5: Implement transition graph and evidence gates**

Allowed transitions:

```text
piloting -> calibrated -> approved -> active
active -> suspended -> active
active -> retired
suspended -> retired
```

Required newly supplied evidence:

- `calibrated`: calibration + item-fit + DIF + item-information;
- `approved`: approval;
- `active` from approved: no additional kind, but at least one approved use ID;
- reactivation from suspended: approval + drift;
- `suspended`: suspension plus at least one DIF or drift reference;
- `retired`: retirement.

Reject skipping, reversal, no-op, duplicate/conflicting evidence identity, evidence removal, policy-criticality mutation, provenance mutation, and retirement exit.

- [ ] **Step 6: Run focused GREEN and coverage**

Run:

```bash
pytest -q tests/test_rubric_item_bank_lifecycle.py
coverage run --branch -m pytest -q tests/test_rubric_item_bank_lifecycle.py
coverage report -m --include='python/fast_mlsirm/rubric/item_bank.py'
```

Expected: all tests pass; new module reaches 100% statement and branch coverage.

- [ ] **Step 7: Commit implementation**

Commit the module, exports, and tests without changing dependencies or unrelated files.

### Task 3: Release Documentation and Repository Validation

**Files:**
- Create: `docs/changelog.d/609-governed-item-bank-lifecycle.md`
- Modify: `CHANGELOG.md` using `scripts/render_changelog_fragments.py`
- Modify only if required after canonical PR integration: requirements/architecture traceability in the single canonical documentation PR.

**Interfaces:**
- Produces a documented public contract while leaving physical persistence and hosted workflow ownership downstream.

- [ ] **Step 1: Add authoritative changelog fragment**

Describe lifecycle contracts, evidence gates, immutable transitions, downstream persistence boundary, and explicit absence of new psychometric estimates.

- [ ] **Step 2: Render and verify changelog**

Run:

```bash
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

- [ ] **Step 3: Run complete relevant validation**

Run focused tests, full Python suite, Rust workspace and PyO3 crate tests, package/reinstall/release acceptance, explicit GPU no-skip, fuzz, Security Scan, and SAST on one unchanged exact head.

- [ ] **Step 4: Current-head review and merge gate**

Keep Draft until exact-head automated review has no valid unresolved finding, repository approval/branch protection is satisfied, and every required check passes. No release/version bump belongs in this bounded contract slice.

## Self-Review

- The plan reuses the existing rubric/audit/pilot hierarchy and introduces only post-pilot lifecycle records.
- Numerical evidence is referenced, not recomputed in Python.
- No physical DB or hosted workflow is introduced.
- Every transition, evidence gate, mutation boundary, terminal state, and interpretation limitation has a deterministic test target.
- The canonical architecture documentation remains owned by PR #604; this feature PR links to it after that writer is available rather than opening a second architecture authority.
