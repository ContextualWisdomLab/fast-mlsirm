# Rust True-Parameter Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a literature-traceable, Rust-only MLS2PLM true-parameter recovery experiment and make every previously ignored or hardware-skipped CI path execute.

**Architecture:** Keep the existing simple-structure MLS2PLM implementation unchanged. Add a Rust integration experiment that generates a bounded Kang-and-Jeon simulation condition, calls the compiled marginal estimator, and computes identification-safe recovery metrics in Rust. Extend GitHub Actions with separate ignored-test, CPU-recovery, and Mesa/Lavapipe software-Vulkan jobs; remove the duplicate NumPy recovery experiment.

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
- [x] **Step 2: Record the exact deterministic paper simulation condition**
- [x] **Step 3: Define identification-safe recovery metrics and thresholds**
- [x] **Step 4: Document CPU, GPU, skip, coverage, and docstring evidence**
- [x] **Step 5: Commit the research contract**

### Task 2: Rust-only recovery experiment

**Files:**
- Create: `crates/mlsirm-core/tests/literature_true_parameter_recovery.rs`
- Delete: `tests/test_true_parameter_recovery.py`

**Interfaces:**
- Consumes: `mlsirm_core::marginal::fit_marginal`, `MarginalConfig`, `PopulationSpec`, `Device`, `ModelConfig`, `ModelType`, and `PenaltyConfig`.
- Produces: deterministic ignored tests named `kang_jeon_2025_minimum_cell_recovers_true_parameters` and `gpu_recovery_matches_cpu_on_paper_design`. In the first name, “minimum” refers to the paper's lowest dimensionality and item-count levels; the sample size is the paper's `P=500` condition.

- [x] **Step 1: Add the deterministic Rust simulation and recovery test**

The test generates `P=500`, `D=2`, `I_d=8`, `K=2`, `rho=.30`, `gamma=1.5`, `a in [.5,2.5]`, and deterministically permuted `b in [0,5]`.

- [x] **Step 2: Establish a failing build/test before the corrected implementation**

The first same-head CI attempt failed in the new equation fixture before the explicit floating-point types were corrected, preserving a red-to-green implementation record.

- [x] **Step 3: Implement all metric helpers in Rust**

The integration test implements deterministic PRNG/normal generation, Pearson correlation, RMSE, interaction-adjusted easiness, pairwise person-map and item-map distances, and likelihood-trace checks. No NumPy or Python arithmetic is used.

- [x] **Step 4: Add the explicit CPU/GPU parity recovery test**

The GPU test calls `fit_marginal(..., Device::Gpu)` and compares item parameters, person trait EAPs, person interaction-position EAPs, item positions, the distance weight, and final likelihood to `Device::Cpu` with documented f32 tolerances.

- [x] **Step 5: Delete the NumPy-only recovery test**

`tests/test_true_parameter_recovery.py` is removed; the public `recovery_report` unit coverage remains in the existing diagnostics tests.

- [x] **Step 6: Run ordinary targeted suites to green**

The ordinary Python, Rust workspace, PyO3, package, fuzz, and GPU suites pass on the PR head. The long-running CPU recovery and complete ignored-test sweeps remain merge gates.

- [x] **Step 7: Commit the Rust recovery experiment**

### Task 3: Execute every ignored Rust test

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: all `#[ignore]` tests in the workspace and excluded PyO3 crate.
- Produces: required `rust-ignored`, `rust-recovery`, and `gpu-software` evidence whose union executes every ignored path.

- [x] **Step 1: Add a release-mode ignored-test sweep**

The general sweep executes all ignored workspace tests except the two recovery tests, which are executed explicitly in the dedicated CPU and GPU jobs. The excluded PyO3 crate's ignored tests run separately.

- [x] **Step 2: Add a dedicated CPU recovery job**

The job first exercises the explicit four-worker objective parity test, then runs:

```bash
cargo test --release -p mlsirm-core \
  --test literature_true_parameter_recovery \
  kang_jeon_2025_minimum_cell_recovers_true_parameters \
  -- --ignored --exact --nocapture
```

- [ ] **Step 3: Confirm both jobs pass on the final unchanged head**

### Task 4: Execute the real GPU path without skips

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: Mesa Lavapipe, wgpu Vulkan, and `tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely`.
- Produces: required `gpu-software` evidence with zero skipped tests.

- [x] **Step 1: Install Mesa Vulkan packages on Ubuntu**
- [x] **Step 2: Discover and export the Lavapipe ICD JSON path**
- [x] **Step 3: Build the PyO3 extension and confirm `vulkaninfo` sees an adapter**
- [x] **Step 4: Run the explicit Python GPU parity test with JUnit XML**
- [x] **Step 5: Parse JUnit XML and fail if `skipped != 0`**
- [x] **Step 6: Run the Rust GPU recovery parity test**
- [ ] **Step 7: Confirm the GPU job passes on the final unchanged head**

### Task 5: Full same-head verification and merge

**Files:**
- Modify only files required by review findings.

**Interfaces:**
- Consumes: GitHub review threads and all same-head checks.
- Produces: a mergeable PR closing issue #389.

- [x] **Step 1: Open a draft PR linked to #389**
- [x] **Step 2: Mark the PR ready after ordinary suites pass**
- [ ] **Step 3: Inspect CodeRabbit/OpenCode review findings**
- [ ] **Step 4: Fix every valid finding and resolve review threads**
- [ ] **Step 5: Confirm CI, Security Scan, SAST, coverage, and docstring evidence**
- [ ] **Step 6: Merge only when the head SHA is unchanged**
