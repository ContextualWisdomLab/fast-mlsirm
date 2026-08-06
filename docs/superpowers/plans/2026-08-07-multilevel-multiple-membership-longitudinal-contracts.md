# Multilevel, Multiple-Membership, and Longitudinal Contracts Plan

> **Execution:** Use Superpowers test-driven development. Keep the pull request
> Draft until all intentional RED contracts have a coherent GREEN
> implementation and exact-head repository gates pass.

**Goal:** Add reusable, immutable contracts for sparse contextual membership and
longitudinal measurement without introducing psychometric arithmetic outside
Rust.

**Architecture:** Create a standalone `fast_mlsirm.multilevel` package. Python
owns bounded validation, canonical serialization, content identity, and sparse
design marshalling. Estimation remains explicitly unimplemented in this PR and
is reserved for the Rust core described in issue #565.

## Global constraints

- All public classes, functions, enums, and tests have complete docstrings.
- Object identifiers use two-or-more-token lower `snake_case`.
- Public handles retain at least 128 bits of SHA-256 identity.
- Direct construction of aggregate designs is factory-sealed.
- Errors use stable two-or-more-token lower `snake_case` codes and non-reflective
  JSON paths/messages.
- Iterables and serialized artifacts are resource bounded.
- No raw source, response, prompt, or provider text is stored.
- No likelihood, gradient, integration, optimization, uncertainty, or recovery
  arithmetic is implemented in Python.
- `CHANGELOG.md` is rendered from the authoritative fragment only after GREEN.

---

## Task 1 — Write the RED contextual-membership contracts

**Files**

- Create: `tests/test_multilevel_contracts.py`

**Required RED behavior**

- one-hot nesting is valid;
- two- and three-context weighted membership is valid when weights sum to one;
- input permutation produces identical content and fingerprints;
- duplicate observation–context cells fail;
- empty observation groups fail;
- Boolean, zero, negative, NaN, infinity, and over-one weights fail;
- observation-level weight totals outside tolerance fail;
- one context or observation identity cannot be rebound to conflicting revision
  provenance;
- direct aggregate construction fails;
- serialized output is deterministic and source-text-free;
- bounded infinite iterators fail without unbounded materialization.

Run:

```bash
pytest -q tests/test_multilevel_contracts.py -k membership
```

Expected initial result: import or missing-symbol failure.

## Task 2 — Write the RED temporal contracts

**Files**

- Modify: `tests/test_multilevel_contracts.py`

**Required RED behavior**

- repeated occasions are grouped and ordered per respondent;
- irregular time intervals are retained exactly;
- input permutation does not change identity;
- duplicate occasion IDs, sequence indices, or time offsets fail;
- non-increasing temporal order fails;
- Boolean/fractional indices and offsets fail;
- malformed revision fingerprints fail without reflection;
- direct aggregate construction fails;
- a random-intercept/slope state forbids an AR coefficient;
- stationary AR(1) requires a finite coefficient strictly between -1 and 1;
- lagged-response dependence remains an independent Boolean.

Run:

```bash
pytest -q tests/test_multilevel_contracts.py -k temporal
```

Expected initial result: import or missing-symbol failure.

## Task 3 — Implement shared contract safety

**Files**

- Create: `python/fast_mlsirm/multilevel/_validation.py`
- Create: `python/fast_mlsirm/multilevel/contracts.py`
- Create: `python/fast_mlsirm/multilevel/__init__.py`

**Implementation**

- stable `MultilevelContractError`;
- descriptive identifier, SHA-256, integer, real-weight, Boolean, and bounded
  iterable normalization;
- canonical JSON with UTF-8, signed-integer, finite-float, negative-zero, depth,
  node, collection, and text bounds;
- content-addressed factory-sealed dataclasses;
- deterministic sorting and duplicate detection;
- public `to_dict()`, full fingerprint, and 128-bit handle properties.

Run the focused file and require GREEN with 100% added production statement and
branch coverage.

## Task 4 — Implement membership factories

**Files**

- Modify: `python/fast_mlsirm/multilevel/contracts.py`
- Modify: `tests/test_multilevel_contracts.py`

**Public API**

```python
build_context_membership(...)
build_context_membership_design(...)
```

Group weights by `observation_id`, enforce an absolute sum-to-one tolerance, and
canonicalize by observation/context/revision identity. Preserve exact caller
weights; do not silently renormalize materially invalid assignments.

## Task 5 — Implement temporal factories

**Files**

- Modify: `python/fast_mlsirm/multilevel/contracts.py`
- Modify: `tests/test_multilevel_contracts.py`

**Public API**

```python
build_temporal_occasion(...)
build_longitudinal_state_spec(...)
build_longitudinal_design(...)
```

Group occasions by respondent and validate exact sequence/time ordering.
Serialization must expose respondent sequence summaries without duplicating raw
records or losing revision provenance.

## Task 6 — Add recovery-study specifications without an estimator

**Files**

- Create: `python/fast_mlsirm/multilevel/recovery.py`
- Create: `tests/test_multilevel_recovery_contracts.py`

Add immutable simulation specifications for one-hot nesting, weighted membership,
random slopes, AR(1), lagged response, missingness, worker counts, and future GPU
parity. Add result-schema contracts for bias, MAE, RMSE, coverage, convergence,
and failure counts. Do not simulate or estimate parameters in Python.

## Task 7 — Documentation, architecture, and changelog

**Files**

- Create: `docs/multilevel_multiple_membership_longitudinal_rfc.md`
- Create: `docs/doctoring/multilevel_longitudinal_measurement.md`
- Create: `docs/changelog.d/565-multilevel-longitudinal-contracts.md`
- Modify only where necessary: `ARCHITECTURE.md`, `AGENTS.md`, `CLAUDE.md`
- Render after GREEN: `CHANGELOG.md`

Include the atomistic-fallacy boundary, MSA reuse, Rust numerical ownership,
GPU profiling gate, identification limits, rollback, and APA 7 references.

## Task 8 — Complete exact-head verification

Run repository-prescribed equivalents of:

```bash
python -m compileall -q python/fast_mlsirm/multilevel tests
pytest -q tests/test_multilevel_contracts.py tests/test_multilevel_recovery_contracts.py
pytest -q
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace --all-targets
python scripts/check_docstring_coverage.py
python scripts/render_changelog_fragments.py --update CHANGELOG.md
python scripts/render_changelog_fragments.py --check CHANGELOG.md
```

Also require exact-head PyO3, wheel reinstall, package acceptance, GPU no-skip,
fuzz, Security Scan, SAST, zero unresolved review threads, and repository-policy
review evidence. Mark Ready and enable auto-merge only after all gates pass.

## Deferred estimator boundary

Do not add a placeholder Python estimator. The next PR begins with Rust RED
objective/gradient and true-parameter recovery tests for the sparse multilevel
predictor. This contract PR must remain useful even when imported without any
future estimator backend.
