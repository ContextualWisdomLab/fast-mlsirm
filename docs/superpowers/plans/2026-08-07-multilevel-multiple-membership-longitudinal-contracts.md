# Multilevel, Multiple-Membership, and Longitudinal Contracts Plan

> **Execution:** Use Superpowers test-driven development and verification-before-completion. Keep the pull request Draft until all contracts, release records, and exact-head gates are GREEN.

**Goal:** Add reusable, immutable contracts for nested, cross-classified, weighted multiple-membership, and longitudinal measurement without implementing psychometric arithmetic outside Rust.

**Architecture:** `fast_mlsirm.multilevel` owns bounded validation, child replay, canonical serialization, content identity, and sparse design marshalling. Future Rust PRs own likelihood, integration, optimization, uncertainty, multithreading/GPU, and true-parameter recovery.

## Global constraints

- Every public class, function, and enum has a complete docstring.
- Object identifiers use two-or-more-token lower `snake_case`.
- Full SHA-256 identities back descriptive 128-bit public handles.
- Direct aggregate construction is factory-sealed.
- Errors are stable, structured, path-specific, and non-reflective.
- Caller-controlled iterables and enum callbacks are safely bounded/redacted.
- No raw source, response, prompt, or provider text is stored.
- No Python likelihood, gradient, integration, optimizer, uncertainty, simulation, or recovery implementation.
- `CHANGELOG.md` is rendered from the authoritative fragment only after focused GREEN.

---

## Task 1 — RED base membership and temporal contracts

**Files**

- `tests/test_multilevel_contracts.py`

Required behavior:

- one-hot nesting and weighted multiple membership;
- exact weights and permutation-invariant identity;
- invalid totals/numerics, duplicate cells, revision rebinding, direct construction, unbounded iterables, and source-text leakage fail closed;
- irregular occasion order, exact integer fields, duplicate IDs/sequences/times, AR/growth boundaries, lagged-response independence, and factory sealing.

## Task 2 — RED integrity and hostile-input contracts

**Files**

- `tests/test_multilevel_contract_integrity.py`
- `tests/test_multilevel_exact_type_contracts.py`

Required behavior:

- replay every package-owned child before aggregation;
- reject post-construction field mutation against sealed fingerprints;
- redact ordinary iterator and enum callback exceptions;
- preserve process-control exceptions;
- reject subclassed/Boolean numeric coercion where exact types are required;
- canonicalize negative zero where identity would otherwise drift.

## Task 3 — RED cross-classification contracts

**Files**

- `tests/test_multilevel_cross_classification_contracts.py`

Required behavior:

- `context_dimension_id` is an explicit required factory field;
- context identity is `(context_dimension_id, context_id)`;
- duplicate cells are observation × dimension × context;
- weights normalize independently per observation × dimension;
- one observation can have weighted membership in one dimension and one-hot membership in another;
- every observation carries every declared dimension in schema 1.0;
- revision digests bind exact observation, dimension, context, and weight;
- deterministic serialization exposes dimension IDs, context keys, per-dimension counts, and exact per-dimension weights.

## Task 4 — Implement contract safety and public namespace

**Files**

- `python/fast_mlsirm/multilevel/_validation.py`
- `python/fast_mlsirm/multilevel/contracts.py`
- `python/fast_mlsirm/multilevel/__init__.py`

Implementation:

- `MultilevelContractError` and bounded canonical validation;
- explicit contextual dimension and level contracts;
- exact per-dimension membership normalization;
- content-addressed factory-sealed membership/design/occasion/state/longitudinal artifacts;
- child replay and revision conflict checks;
- deterministic sorting and dimension-scoped summaries;
- strict discrete occasion-step AR(1) contract and separate lagged-response Boolean.

Require focused 100% added-production statement/branch coverage.

## Task 5 — Align research, architecture, and release records

**Files**

- `docs/multilevel_multiple_membership_longitudinal_rfc.md`
- `docs/doctoring/multilevel_longitudinal_measurement.md`
- `docs/changelog.d/565-multilevel-longitudinal-contracts.md`
- this spec and plan
- update `ARCHITECTURE.md`, `AGENTS.md`, or `CLAUDE.md` only if an existing architectural statement would otherwise become false
- render `CHANGELOG.md` after focused GREEN

Document:

- atomistic-fallacy and multiple-membership multiple-classification boundaries;
- dimension-specific weights and revision provenance;
- MSA reuse and Rust ownership;
- identification, fairness, causal, and rollback limits;
- `autoregressive_coefficient` as discrete occasion-step AR(1);
- irregular millisecond offsets as provenance only until a separate continuous-time Rust contract;
- APA 7 references.

## Task 6 — Focused verification

Run repository-prescribed equivalents of:

```bash
python -m compileall -q python/fast_mlsirm/multilevel tests
pytest -q \
  tests/test_multilevel_contracts.py \
  tests/test_multilevel_cross_classification_contracts.py \
  tests/test_multilevel_contract_integrity.py \
  tests/test_multilevel_exact_type_contracts.py
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

Fix genuine contract/coverage failures only. Do not restore an inferred/default dimension, delete replay tests, or weaken missing-dimension enforcement.

## Task 7 — Complete exact-head verification and merge discipline

- Complete Python statement/branch coverage and public docstring gates.
- Complete Rust/PyO3 workspace/all-target tests, fmt, and clippy.
- Complete wheel reinstall, package acceptance, explicit GPU no-skip, fuzz, Security Scan, and SAST.
- Request current-head automated review and qualifying independent non-author approval.
- Require zero unresolved actionable threads.
- Remove blocker labels, mark Ready, and enable auto-merge only after every gate succeeds on one unchanged head.
- Close issue #565 only after protected merge and accepted-main verification.

## Deferred estimator boundary

No recovery simulation specification or placeholder estimator is needed in this contract PR. The next Rust estimator PR must begin with objective/gradient parity, identification failures, and scale-aligned true-parameter recovery for nested, crossed, weighted multiple-membership, multiple-classification, balanced/unbalanced longitudinal, missing-data, and discrete-step AR conditions. A later PR must define continuous-time/interval-adjusted transitions explicitly before elapsed gaps enter the likelihood.
