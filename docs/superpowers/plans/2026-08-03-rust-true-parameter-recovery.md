# Rust True-Parameter Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a literature-traceable, Rust-only MLS2PLM true-parameter recovery experiment and make every supported ignored or hardware-skipped evidence path execute under an exact, auditable partition.

**Architecture:** Keep the existing simple-structure MLS2PLM implementation unchanged. Add a Rust integration experiment that generates a bounded Kang-and-Jeon simulation condition, calls the compiled marginal estimator, and computes identification-safe recovery metrics in Rust. Separate bounded pull-request sentinels from exhaustive scheduled studies, execute CPU and Mesa/Lavapipe GPU evidence in dedicated jobs, and quarantine only the fully qualified historical higher-order DINA duplicate whose finite-sample assertion is superseded by a reviewed replacement.

**Tech Stack:** Rust 2021, `mlsirm-core`, wgpu/Vulkan, GitHub Actions, PyO3/maturin, pytest.

## Global Constraints

- Every model equation and simulation choice must be traced to primary psychometric literature.
- The local simple-structure MLS2PLM equation and sign convention must not change.
- Response generation, estimation, alignment/invariant metrics, and recovery statistics must execute in Rust.
- CPU execution must exercise the multithreaded Rust core; GPU execution must use the explicit wgpu path.
- Python and Rust production coverage thresholds remain 100%.
- Public Python docstring coverage remains 100%.
- Every active ignored Rust test and hardware-conditional Python test must execute in CI evidence.
- A superseded historical duplicate may be quarantined only by one fully qualified exact name, with a reviewed replacement executed on the same workflow head and contract tests preventing broad or stale skips.

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

The ordinary Python, Rust workspace, PyO3, package, fuzz, and GPU suites passed before publication. Exact-head hosted verification remains the merge gate.

- [x] **Step 7: Commit the Rust recovery experiment**

### Task 3: Partition exhaustive ignored studies without ambiguous skips

**Files:**
- Create: `.github/workflows/statistical-studies.yml`
- Create: `scripts/run_ignored_rust_shard.py`
- Create: `tests/test_ignored_rust_shard.py`
- Create: `tests/test_historical_ho_mc_source_contract.py`
- Create: `tests/test_statistical_studies_workflow.py`

**Interfaces:**
- Consumes: the exact inventory of `#[ignore]` tests in the workspace and excluded PyO3 crate.
- Produces: non-overlapping `rust-ignored`, `rust-pyo3-ignored`, `rust-recovery`, `higher-order-recovery`, and `gpu-recovery` evidence, plus one exact quarantine for `cdm::tests::mc_ho_recovery_500`.

- [x] **Step 1: Add a source-read-only exact-name shard runner**

The runner inventories Cargo's fully qualified ignored tests, rejects unmatched or duplicate declarations, rejects empty selected shards, and invokes each selected test with `--ignored --exact`.

- [x] **Step 2: Add dedicated CPU and GPU recovery jobs**

The dedicated jobs execute the bounded Kang-and-Jeon recovery, higher-order DINA finite-Monte-Carlo replacement, four-worker CPU parity, Python GPU parity with zero skipped tests, and Rust CPU/GPU recovery parity.

- [x] **Step 3: Formalize the historical duplicate quarantine**

The historical function remains detectable as provenance, but exactly one fully qualified exclusion removes it from active evidence. Contract tests require the explicit MCSE-based replacement to appear once in the general exclusions and once in its dedicated command, forbid final-component skips, and forbid source-mutating workflows.

- [ ] **Step 4: Confirm every statistical-study job passes on the final unchanged head**

### Task 4: Execute the real GPU path without skips

**Files:**
- Create: `.github/workflows/statistical-studies.yml`

**Interfaces:**
- Consumes: Mesa Lavapipe, wgpu Vulkan, and `tests/test_marginal_parity.py::test_marginal_gpu_agrees_with_cpu_loosely`.
- Produces: scheduled/manual/tag `gpu-recovery` evidence with zero skipped tests.

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
- [x] **Step 2: Inspect CodeRabbit and human review state**
- [x] **Step 3: Resolve the contradictory historical-source and workflow contracts**
- [x] **Step 4: Mark the PR ready after the scientific evidence contract is coherent**
- [ ] **Step 5: Confirm CI, Security Scan, SAST, ClusterFuzzLite, coverage, and docstring evidence**
- [ ] **Step 6: Merge only when the current head SHA is unchanged and policy is satisfied**
