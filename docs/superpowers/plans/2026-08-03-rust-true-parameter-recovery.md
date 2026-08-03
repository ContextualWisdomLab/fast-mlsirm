# Rust True-Parameter Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a literature-traceable, Rust-only MLS2PLM true-parameter recovery experiment and make every previously ignored or hardware-skipped CI path execute.

**Architecture:** Keep the existing simple-structure MLS2PLM implementation unchanged. Add a Rust integration experiment that generates a Kang-and-Jeon simulation cell, calls the compiled marginal estimator, and computes identification-safe recovery metrics in Rust. Extend GitHub Actions with an ignored-statistics job and a Mesa/Lavapipe software-Vulkan job; remove the duplicate NumPy recovery experiment.

**Tech Stack:** Rust 2021, `mlsirm-core`, wgpu/Vulkan, GitHub Actions, PyO3/maturin, pytest.

## Global Constraints

- Every model equation and simulation choice must be traced to primary psychometric literature.
- The local simple-structure MLS2PLM equation and sign convention must not change.
- Response generation, estimation, alignment/invariant metrics, and recovery statistics must execute in Rust.
- CPU execution must exercise the multithreaded Rust core; GPU execution must use the explicit wgpu path.
- Python and Rust production coverage thresholds remain 100%.
- Public Python docstring coverage remains 100%.
- Ignored Rust tests and hardware-conditional Python tests must execute in CI.

---

### Task 1: Paper and equation traceability

**Files:**
- Create: `docs/papers/true-parameter-recovery-study.md`

**Interfaces:**
- Consumes: the model contract in `AGENTS.md` and primary sources listed there.
- Produces: a stable equation-to-experiment contract used by the Rust test and PR review.

- [x] **Step 1: Record the verified general and simple-structure equations**
- [x] **Step 2: Record the exact deterministic paper simulation cell**
- [x] **Step 3: Define identification-safe recovery metrics and thresholds**
- [x] **Step 4: Document CPU, GPU, skip, coverage, and docstring evidence**
- [x] **Step 5: Commit the research contract**

### Task 2: Rust-only recovery experiment

**Files:**
- Create: `crates/mlsirm-core/tests/literature_true_parameter_recovery.rs`
- Delete: `tests/test_true_parameter_recovery.py`

**Interfaces:**
- Consumes: `mlsirm_core::marginal::fit_marginal`, `MarginalConfig`, `PopulationSpec`, `Device`, `ModelConfig`, `ModelType`, and `PenaltyConfig`.
- Produces: deterministic ignored tests named `kang_jeon_2025_minimum_cell_recovers_true_parameters` and `gpu_recovery_matches_cpu_on_paper_design`.

- [ ] **Step 1: Add the deterministic Rust simulation and a failing recovery test**

The test must generate `P=500`, `D=2`, `I_d=8`, `K=2`, `rho=.30`, `gamma=1.5`, `a in [.5,2.5]`, and deterministically permuted `b in [0,5]`.

- [ ] **Step 2: Run the targeted test and capture the expected red result**

Run:

```bash
cargo test --release -p mlsirm-core --test literature_true_parameter_recovery \
  kang_jeon_2025_minimum_cell_recovers_true_parameters -- --ignored --nocapture
```

Expected: the initial recovery threshold or missing helper fails before the final implementation is accepted.

- [ ] **Step 3: Implement all metric helpers in Rust**

Implement deterministic PRNG/normal generation, Pearson correlation, RMSE,
interaction-adjusted easiness, pairwise item-map distances, and likelihood-trace
checks in the integration test. No NumPy or Python arithmetic is permitted.

- [ ] **Step 4: Add the explicit CPU/GPU parity recovery test**

The GPU test must call `fit_marginal(..., Device::Gpu)` and compare the result to
`Device::Cpu` with documented f32 tolerances.

- [ ] **Step 5: Delete the NumPy-only recovery test**

Remove `tests/test_true_parameter_recovery.py`; the public `recovery_report`
unit coverage remains in the existing diagnostics tests.

- [ ] **Step 6: Run the targeted tests to green**

```bash
cargo test --release -p mlsirm-core --test literature_true_parameter_recovery \
  -- --ignored --test-threads=1 --nocapture
cargo test --workspace
cargo test --manifest-path crates/fast-mlsirm-py/Cargo.toml
pytest
```

- [ ] **Step 7: Commit the Rust recovery experiment**

```bash
git add crates/mlsirm-core/tests/literature_true_parameter_recovery.rs \
  tests/test_true_parameter_recovery.py
git commit -m "test: add Rust literature recovery experiment"
```

### Task 3: Execute every ignored Rust test

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: all `#[ignore]` tests in the workspace and excluded PyO3 crate.
- Produces: required `rust-statistical` CI evidence.

- [ ] **Step 1: Add a release-mode ignored-test job**

Run both commands serially with a 120-minute job timeout:

```bash
cargo test --release --workspace -- --ignored --test-threads=1 --nocapture
cargo test --release --manifest-path crates/fast-mlsirm-py/Cargo.toml \
  -- --ignored --test-threads=1 --nocapture
```

- [ ] **Step 2: Verify the job runs the new recovery experiment**
- [ ] **Step 3: Commit the ignored-test gate**

### Task 4: Execute the real GPU path without skips

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Mesa Lavapipe, wgpu Vulkan, and `tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely`.
- Produces: required `gpu-software` evidence with zero skipped tests.

- [ ] **Step 1: Install Mesa Vulkan packages on Ubuntu**
- [ ] **Step 2: Discover and export the Lavapipe ICD JSON path**
- [ ] **Step 3: Build the PyO3 extension and confirm `vulkaninfo` sees an adapter**
- [ ] **Step 4: Run the explicit Python GPU parity test with JUnit XML**
- [ ] **Step 5: Parse JUnit XML and fail if `skipped != 0`**
- [ ] **Step 6: Run the Rust GPU recovery parity test**
- [ ] **Step 7: Commit the accelerator gate**

### Task 5: Full same-head verification and merge

**Files:**
- Modify only files required by review findings.

**Interfaces:**
- Consumes: GitHub review threads and all same-head checks.
- Produces: a mergeable PR closing issue #389.

- [ ] **Step 1: Open a draft PR linked to #389**
- [ ] **Step 2: Inspect CodeRabbit/OpenCode review findings**
- [ ] **Step 3: Fix every valid finding and resolve review threads**
- [ ] **Step 4: Confirm CI, Security Scan, SAST, coverage, and docstring evidence**
- [ ] **Step 5: Mark ready and merge only when the head SHA is unchanged**
