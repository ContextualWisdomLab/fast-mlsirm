# Fail-Closed Vuong Selection Summary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans when
> extending this implementation.

**Goal:** Add an auditable public summary of the existing Rust-backed Vuong
selection statistic without silently applying it before the mathematically
required model-relation and formal-distinguishability stages.

**Architecture:** Keep likelihood means, variance, BIC correction, z
standardization, and normal p-value calculation in the compiled Rust kernel
exposed through `fitstats.vuong_nonnested`. The Python module owns bounded input
marshalling, audit metadata, relation classification, redacted typed boundary
errors, and fail-closed procedure routing. It does not implement cluster
aggregation, bootstrap resampling, or the weighted-chi-square
distinguishability statistic in Python.

**Tech Stack:** Python 3.10+, existing Rust/PyO3 Vuong kernel, pytest.

## Global Constraints

- Preserve the existing low-level `vuong_nonnested` numerical API.
- Keep all selection-statistic arithmetic in Rust.
- Treat `omega_tol` as a numerical floor only, never as Vuong's formal
  distinguishability test.
- Default mathematical relation to `unknown`.
- Report no preferred model for any relation until typed formal first-stage
  evidence exists.
- Bound casewise materialization and audit-label size.
- Reject control characters, booleans, and fractional parameter counts.
- Never inspect or expose human-readable compiled exception wording.
- Preserve cluster-dependence limitations rather than adding ad hoc Python
  pseudo-replication fixes.
- Maintain complete public docstrings and branch-focused tests.
- Document equations and limitations from primary methodological sources.

---

### Task 1: Red safety-contract tests

**Files:**
- Test: `tests/test_model_comparison_safety_contract.py`

- [x] Prove omitted relation metadata defaults to `unknown`.
- [x] Prove strictly non-nested comparisons still require formal
  distinguishability.
- [x] Bound oversized and non-terminating casewise iterables.
- [x] Bound model labels and reject control-character injection.
- [x] Reject boolean and fractional parameter counts.
- [x] Prove arbitrary compiled error wording maps to one stable redacted
  boundary.

### Task 2: Fail-closed selection-statistic API

**Files:**
- Modify: `python/fast_mlsirm/model_comparison.py`

**Interfaces:**
- Produces: `ModelRelation`, `ComparisonStatus`, `ModelComparisonResult`,
  `VuongKernelError`, and `compare_nonnested_models`.

- [x] Delegate every statistical quantity to the compiled Rust kernel.
- [x] Preserve raw mean, omega, z, and p-value fields on successful calls.
- [x] Keep interpreted z/p unavailable and `preferred_model=None`.
- [x] Return `requires_distinguishability_test` for strictly non-nested and
  overlapping relations.
- [x] Return relation-appropriate likelihood-ratio requirements for nested and
  boundary-nested models.
- [x] Convert any compiled rejection to `kernel_error` without message parsing,
  subtype guessing, or Python moment calculations.
- [x] Add bounded, printable model-label validation.
- [x] Add bounded casewise iterable consumption.

### Task 3: Scientific and product documentation

**Files:**
- Modify: `docs/non_nested_model_comparison.md`

- [x] Record the casewise difference, omega, z, and BIC-corrected z equations.
- [x] Explain the two-stage distinguishability-then-selection contract.
- [x] State that the current API produces no winner.
- [x] Explain required procedures for nested and boundary cases.
- [x] Explain the redacted `kernel_error` compatibility boundary.
- [x] Explicitly exclude ad hoc Python clustering and bootstrap calculations.
- [x] Cite Vuong (1989), Schneider et al. (2020), and Merkle et al. (2016).

### Task 4: Formal distinguishability follow-up

**Required future Rust inputs:**

- [ ] Common casewise score-vector contract across fit families.
- [ ] Observed and expected information matrices with stable parameter order.
- [ ] Nestedness, overlap, and boundary metadata.
- [ ] Cluster identifiers or clusterwise contributions where iid sampling is
  not justified.
- [ ] Weighted-chi-square distinguishability kernel and recovery simulations.
- [ ] Typed evidence object that can unlock model-preference statuses only after
  successful first-stage verification.
- [ ] Optional compiled structured error codes that can safely refine the
  current generic `kernel_error` state.

### Task 5: Same-head verification and merge

- [ ] Run focused Python tests with the compiled core.
- [ ] Run the complete Python and Rust suites.
- [ ] Confirm coverage and docstring gates.
- [ ] Confirm no unresolved review findings.
- [ ] Mark ready and enable auto-merge only after exact-head checks succeed.
