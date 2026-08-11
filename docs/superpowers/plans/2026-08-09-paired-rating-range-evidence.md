# Paired Rating Range Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Rust-owned, Python-exposed paired categorical range-use evidence that conservatively identifies narrower observed automated-score support on the same held-out cases without claiming a population range-restriction parameter.

**Architecture:** Extend the existing agreement/validation numerical layer in `mlsirm-core`, expose the exact Rust result through the existing PyO3 `_core` module and `fast_mlsirm.validation`, and leave essay-report schema integration to a later issue-#397 slice. No new likelihood is introduced.

**Tech Stack:** Rust 1.97+, PyO3, NumPy, Python 3.12+, pytest.

## Global Constraints

- All numerical calculations belong to Rust; Python validates and marshals only.
- The diagnostic uses paired automated/reference labels from the same cases.
- No universal acceptance threshold or scorer pass/fail is added.
- `narrower_observed_support` is sample evidence, not a population rater trait.
- A degenerate reference distribution returns unavailable relative ratios rather than NaN/Inf.
- No raw essay/prompt text or provider-controlled diagnostic content is retained.
- Public functions/classes require complete documentation; added production branches require 100% statement/branch coverage.
- Preserve all existing agreement-validation semantics and ABI fields.

---

### Task 1: Add RED Python public-contract tests

**Files:**
- Create: `tests/test_paired_rating_range_evidence.py`

**Interfaces:**
- Consumes future `fast_mlsirm.validation.RatingRangeEvidence` and `paired_rating_range_evidence`.
- Produces exact expected output and validation behavior for later tasks.

- [ ] **Step 1: Write a hand-calculated compressed-range case.**

Use paired labels where reference uses `0..4` and automated uses only `1..3`; assert endpoints, spans, distinct categories, empirical SDs/ratios, both endpoint gaps, `narrower_observed_support=True`, and `central_tendency_signal=True`.

- [ ] **Step 2: Add full-range and one-sided cases.**

Prove an identical full-range scorer is not signaled and upper-tail-only truncation produces a positive upper gap without the stricter central-tendency flag.

- [ ] **Step 3: Add same-span/fewer-internal-category case.**

Prove the combined signal remains false because span itself was not narrowed.

- [ ] **Step 4: Add degenerate-reference behavior.**

Prove zero reference span/SD maps to `None` relative ratios, with finite remaining fields.

- [ ] **Step 5: Add fail-closed wrapper validation.**

Reject 2-D arrays, unequal lengths, one observation, Boolean/fractional/negative/out-of-range labels, Boolean/noninteger/out-of-range `category_count`, and resource-amplifying vectors according to current package caps.

- [ ] **Step 6: Run the focused file and verify RED** because the public API does not exist.

### Task 2: Add Rust range-evidence kernel

**Files:**
- Modify: `crates/mlsirm-core/src/agreement.rs`
- Test: Rust unit tests colocated with the agreement module or its existing integration test target.

**Interfaces:**
- Produces `PairedRatingRangeEvidence` and `paired_rating_range_evidence()`.

- [ ] **Step 1: Validate pairing, sample size, category count, bounds, and overflow-safe counters.**
- [ ] **Step 2: Compute min/max, distinct counts, spans, empirical SD, optional ratios, signed endpoint gaps, and conservative Boolean signals in one bounded pass plus fixed-size category bookkeeping.**
- [ ] **Step 3: Add Rust oracle tests matching the Python RED fixtures exactly.**
- [ ] **Step 4: Add degenerate and malformed-input branch tests.**
- [ ] **Step 5: Run focused Rust tests to GREEN.**

### Task 3: Expose the Rust result through PyO3 and Python

**Files:**
- Modify: `crates/fast-mlsirm-py/src/lib.rs`
- Modify: `python/fast_mlsirm/validation.py`
- Modify: `python/fast_mlsirm/__init__.py` only if current public-export policy requires root exposure.
- Test: `tests/test_paired_rating_range_evidence.py`

**Interfaces:**
- Produces immutable `RatingRangeEvidence` and `paired_rating_range_evidence()`.

- [ ] **Step 1: Add a dedicated PyO3 function returning all fields without Python recomputation.**
- [ ] **Step 2: Add exact Python type/shape/category validation consistent with existing validation APIs.**
- [ ] **Step 3: Build immutable typed result from the raw core payload.**
- [ ] **Step 4: Add delegation tests comparing every public field to `_core` raw output.**
- [ ] **Step 5: Prove source-array mutation after the call does not change the result.**
- [ ] **Step 6: Run focused Python and Rust/PyO3 tests to GREEN.**

### Task 4: Add scientific and release documentation

**Files:**
- Create: `docs/doctoring/paired-rating-range-evidence.md`
- Create: `docs/changelog.d/397-paired-rating-range-evidence.md`
- Modify: `CHANGELOG.md` only through the authoritative renderer.

**Interfaces:**
- Produces APA 7 equation-to-source traceability and explicit interpretation limits.

- [ ] **Step 1: Document paired-sample rationale, formulas, degeneracy, no-threshold policy, and distinction from severity.**
- [ ] **Step 2: Cite Jiao et al. (2026), Uto and Ueno (2020), and Wu et al. (2026) in APA 7 form.**
- [ ] **Step 3: State that rMFRM with rater-specific thresholds is the later inferential range-restriction path.**
- [ ] **Step 4: Render/check the authoritative changelog block.**

### Task 5: Exact-head verification and handoff

- [ ] **Step 1: Run focused range-evidence tests.**
- [ ] **Step 2: Run full Python test/coverage/docstring gates.**
- [ ] **Step 3: Run Rust workspace and PyO3 tests.**
- [ ] **Step 4: Run package/release acceptance.**
- [ ] **Step 5: Run explicit GPU no-skip, fuzz, Security Scan, and SAST.**
- [ ] **Step 6: Request exact-head independent review.**
- [ ] **Step 7: Keep Draft until protected merge gates are all green; do not close issue #397 because report wiring, generalized range restriction, fairness, drift, and shortcut diagnostics remain.**

## Self-review

- The design does not duplicate the rubric, scoring, essay, or facets contract hierarchy.
- The new statistic cannot be mistaken for rater severity or a generalized MFRM parameter by name or output.
- The paired design avoids a gross case-mix comparison between unrelated rater samples, but documentation still prohibits population inference from the descriptive signal.
- No new provider SDK, generated feedback, latent-space model, or Python numerical fallback enters this slice.
