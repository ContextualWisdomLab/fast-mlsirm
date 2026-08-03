# Decision-Safe Non-Nested Model Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development or superpowers:executing-plans when
> extending this implementation.

**Goal:** Add an auditable public API that interprets the existing Rust-backed
Vuong statistic without silently applying the strictly non-nested normal
reference distribution to nested, boundary-nested, overlapping, or unknown
model relationships.

**Architecture:** Keep every statistical calculation in the existing compiled
Rust kernel exposed through `fitstats.vuong_nonnested`. The Python module is a
thin relation/label validation and interpretation layer. It intentionally omits
cluster aggregation, bootstrap resampling, and AIC arithmetic rather than
performing those numerical operations in Python or claiming unsupported
asymptotics.

**Tech Stack:** Python 3.10+, existing Rust/PyO3 Vuong kernel, pytest.

## Global Constraints

- Preserve the existing low-level `vuong_nonnested` API.
- Keep likelihood means, variance, BIC correction, z standardization, and
  normal p-value calculation in Rust.
- Do not label a numerical omega floor as Vuong's formal weighted-chi-square
  distinguishability test.
- Suppress preference inference for nested, boundary-nested, overlapping, and
  unknown relationships.
- Maintain 100% docstring and branch coverage requirements.
- Document each equation and limitation from primary methodological sources.

---

### Task 1: Red tests for unsafe interpretation

**Files:**
- Test: `tests/test_model_comparison.py`

- [x] Prove all statistics are delegated to `vuong_nonnested`.
- [x] Cover positive, negative, non-significant, and directionless outcomes.
- [x] Cover zero/tiny variance and non-finite inferential outputs.
- [x] Cover nested, boundary-nested, overlapping, and unknown relations.
- [x] Cover metadata validation and preservation of low-level input guards.
- [x] Add direct parity against the Rust-backed low-level wrapper.

### Task 2: Fail-closed orchestration API

**Files:**
- Create: `python/fast_mlsirm/model_comparison.py`

**Interfaces:**
- Produces: `ModelRelation`, `ComparisonStatus`, `ModelComparisonResult`, and
  `compare_nonnested_models`.

- [x] Add auditable model labels and relation metadata.
- [x] Delegate the complete statistic to the compiled Rust kernel.
- [x] Return preferences only for significant strictly non-nested results.
- [x] Suppress invalid normal inference for all other model relationships.
- [x] Preserve raw mean and omega values for auditability.
- [x] Add complete public docstrings and primary-source references.

### Task 3: Scientific documentation

**Files:**
- Create: `docs/non_nested_model_comparison.md`

- [x] Record the casewise difference, omega, z, and BIC-corrected z equations.
- [x] Explain the distinction between an omega tolerance and the formal
  distinguishability hypothesis test.
- [x] Explain the required procedures for nested and boundary cases.
- [x] Explicitly exclude ad hoc Python clustering and bootstrap calculations.
- [x] Cite Vuong (1989), Schneider et al. (2020), and Merkle et al. (2016).

### Task 4: Same-head verification and merge

- [ ] Run the focused Python tests with the compiled core.
- [ ] Run the complete Python and Rust suites.
- [ ] Confirm 100% coverage and docstring gates.
- [ ] Resolve all review findings.
- [ ] Merge only after every required check succeeds on the unchanged head.
