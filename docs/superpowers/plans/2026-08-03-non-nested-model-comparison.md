# Non-Nested Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a safe public API for comparing nested, strictly non-nested, overlapping, boundary-nested, and unknown IRT model pairs from casewise or clusterwise log-likelihood contributions.

**Architecture:** Add a focused Python orchestration module that validates and aggregates casewise likelihoods, delegates the Vuong statistic to the existing Rust-backed `fitstats.vuong_nonnested` kernel, and adds model-relation metadata, AIC/BIC corrections, cluster bootstrap confidence intervals, and explicit indeterminate outcomes. Keep model fitting separate from model comparison.

**Tech Stack:** Python 3.10+, NumPy, existing Rust/PyO3 Vuong kernel, pytest.

## Global Constraints

- Preserve the existing `vuong_nonnested` API.
- Do not claim a formal distinguishability p-value until the weighted chi-square test is implemented in Rust.
- Treat zero-variance log-likelihood differences as observationally indistinguishable.
- Aggregate by supplied cluster IDs before testing to avoid pseudo-replication.
- Use deterministic bootstrap sampling from a caller-provided seed.
- Maintain 100% docstring and branch coverage requirements.

---

### Task 1: Model comparison API

**Files:**
- Create: `python/fast_mlsirm/model_comparison.py`
- Test: `tests/test_model_comparison.py`

**Interfaces:**
- Produces: `ModelRelation`, `ModelComparisonResult`, `compare_nonnested_models`.

- [ ] Write tests for validation, cluster aggregation, corrections, preference, indistinguishability, and deterministic bootstrap.
- [ ] Implement dataclasses and enums.
- [ ] Implement safe likelihood validation and stable cluster compaction.
- [ ] Delegate the Vuong z statistic to the existing Rust-backed kernel.
- [ ] Add AIC and BIC corrections and bootstrap confidence intervals.
- [ ] Run focused tests.

### Task 2: Documentation and release note

**Files:**
- Create: `docs/non_nested_model_comparison.md`

- [ ] Document the supported model relations and decision sequence.
- [ ] State the limitation that the formal Vuong distinguishability weighted-chi-square test remains separate future Rust work.
- [ ] Document query/system/judge-family clustering examples.
